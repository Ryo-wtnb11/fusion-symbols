//! The Cartan engine's query context ([`GeneralFamily`]): it loads every
//! exact value from, and publishes it to, the CGC, F and R tiers of
//! [`crate::cache`], and asks the rounding layer for the views and their
//! gates. It computes no mathematics and rounds nothing itself.
//!
//! A CGC view miss claims the view, then gets or generates the exact
//! channel below it. Waits go only downward, view ▷ exact channel ▷ seeds,
//! norm tables ▷ module, and an exact generation never claims, publishes or
//! waits on a view: its leader keeps the view it gated and hands it to the
//! view miss that called it. An exact channel is published, and its disk row
//! queued, only after that view passed its gates.

use std::sync::Arc;

use threadteam::Exec;

use super::cgc::exact::{ExactCgc, HighestColumn, NormTable, Pending};
use super::cgc::{CgcChannel, Ladders, ProductScratch, ProductSpace};
#[cfg(test)]
use super::cgc::{ExactChannel, AT_CONTRACTION, BEFORE_FOURTH, BUILDS, REQUEST_MODULES};
#[cfg(test)]
use super::cgc::{AFTER_EXACT, BEFORE_LOAD, CORRUPT, CORRUPT_BLOCK, FAIL_PHASE};
use super::roots::CartanData;
use crate::cache::Label;
use crate::category::{FBlock, RBlock};
use crate::exact::radical::RadicalBlock;
use crate::frcore::{f_exact, r_exact, FInputs, RInputs};
use crate::lie::{LieError, RootSystem};
use crate::rounding::blocks::{check_f_block, check_r_block, round_block};
use crate::rounding::cgc::{
    check_rounded, round_channel, round_column, view_slots, ChannelEstimates,
};

/// The exact channel over `dims` and its rounded view from the descent's
/// `pending` columns, in the one stage after the descent: each item, one
/// column, lets the exact layer write the column's integers and pairs
/// ([`super::cgc::exact::Writer::column`]) and then the rounding layer round
/// that column into its range of the view ([`round_column`]). Neither layer
/// calls the other; a token without a team runs every column inline, and
/// both outputs are the same at every width.
pub(crate) fn write_and_round(
    mut pending: Pending,
    dims: [usize; 3],
    norms: [&NormTable; 3],
    exec: Option<&mut Exec<'_>>,
) -> (ExactCgc, CgcChannel) {
    use crate::stage::{run_items, split_parts, take_part, StageBuf};
    let units = pending.units();
    let estimates = ChannelEstimates::for_channel(pending.vertices(), norms, &units);
    let mut view = view_slots(pending.len());
    let signs = {
        let writer = pending.writer();
        let parts = split_parts(&mut view, writer.ends().iter().copied());
        let mut buf = StageBuf::default();
        let _width = run_items(exec, &units, &mut buf, |k| {
            let column = writer.column(k);
            round_column(
                &column.coords(),
                norms,
                estimates.as_ref(),
                take_part(&parts, k),
            );
            column.into_signs()
        });
        #[cfg(test)]
        if _width.get() > 1 {
            COLUMN_SECTIONS.with(|s| s.set(s.get() + 1));
        }
        buf.drain().collect()
    };
    let mult = pending.mult();
    (
        pending.into_channel(dims, signs),
        CgcChannel {
            mult,
            entries: view.into_boxed_slice(),
        },
    )
}

#[cfg(test)]
thread_local! {
    /// Column stages of [`write_and_round`] on this thread that ran a section
    /// of width two or more.
    pub(crate) static COLUMN_SECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// An exact channel with its three modules.
#[cfg(test)]
type BuiltChannel = (ExactChannel, Arc<Ladders>, Arc<Ladders>, Arc<Ladders>);

#[cfg(test)]
fn count_build(kind: usize) {
    BUILDS.with(|b| {
        let mut v = b.get();
        v[kind] += 1;
        b.set(v);
    });
}

/// The modules held across the channel loads of one F or R request, by label
/// (at most six, so a short vector with equality lookup). A module the CGC
/// tier does not retain is then built once per request rather than once per
/// missed channel; a single channel request is the same with one channel.
/// Holding is strong: a weak handle would not keep a rejected module alive.
#[derive(Default)]
pub(crate) struct RequestModules<'l> {
    held: Vec<(Label<'l>, Arc<Ladders>)>,
}

impl<'l> RequestModules<'l> {
    /// The module of `label`, from this request or else loaded and held.
    fn load(&mut self, fam: &GeneralFamily, label: Label<'l>) -> Result<Arc<Ladders>, LieError> {
        if let Some((_, m)) = self.held.iter().find(|(l, _)| l.words == label.words) {
            return Ok(m.clone());
        }
        let m = fam.module(label)?;
        self.held.push((label, m.clone()));
        #[cfg(test)]
        self.count();
        Ok(m)
    }

    /// Releases every module whose label is not in `later`, the labels of
    /// the channels this request may still load.
    pub(crate) fn retain(&mut self, later: &[Label<'_>]) {
        self.held
            .retain(|(l, _)| later.iter().any(|x| x.words == l.words));
        #[cfg(test)]
        self.count();
    }

    /// Records the held labels before F's fourth channel.
    #[cfg(test)]
    pub(crate) fn observe_before_fourth(&self) {
        BEFORE_FOURTH
            .with(|b| *b.borrow_mut() = self.held.iter().map(|(l, _)| l.words.to_vec()).collect());
    }

    #[cfg(test)]
    fn count(&self) {
        REQUEST_MODULES.with(|c| {
            let [_, max] = c.get();
            c.set([self.held.len(), max.max(self.held.len())]);
        });
    }
}

/// Records the modules held as an F or R contraction begins.
#[cfg(test)]
fn observe_contraction() {
    AT_CONTRACTION.with(|c| c.set(Some(REQUEST_MODULES.with(std::cell::Cell::get)[0])));
}

#[cfg(test)]
impl Drop for RequestModules<'_> {
    fn drop(&mut self) {
        REQUEST_MODULES.with(|c| c.set([0, c.get()[1]]));
    }
}

/// The Cartan engine over one root system: exact CGC (§4), `κ_a` (§5.3) and
/// the F and R contractions of §6, each loaded from and published to the CGC,
/// F and R tiers of [`crate::cache`] only after its gates pass.
#[derive(Clone)]
pub(crate) struct GeneralFamily {
    data: Arc<CartanData>,
}

impl GeneralFamily {
    #[cfg(test)]
    pub(crate) fn new(root: RootSystem) -> Self {
        Self::with_data(CartanData::new(root))
    }
    pub(crate) fn with_data(data: Arc<CartanData>) -> Self {
        Self { data }
    }

    pub(crate) fn data(&self) -> &Arc<CartanData> {
        &self.data
    }

    fn root(&self) -> RootSystem {
        self.data.root
    }

    /// The module of `lambda`, shared through the CGC tier.
    pub(crate) fn module(&self, lambda: Label<'_>) -> Result<Arc<Ladders>, LieError> {
        #[cfg(test)]
        if let Some(hook) = BEFORE_LOAD.with(std::cell::Cell::get) {
            hook();
        }
        crate::cache::cgc_module(self.root(), lambda, || {
            #[cfg(test)]
            count_build(0);
            Ladders::new(&self.data, lambda.words).map(Arc::new)
        })
    }

    /// `a ⊗ b` by Brauer–Klimyk, label → `N^c_{ab}`.
    #[cfg(test)]
    pub(crate) fn product(&self, a: &[i64], b: &[i64]) -> Result<super::roots::Product, LieError> {
        self.data.shared_product(a, b)
    }

    /// `N^c_{ab}`: the triangle rule for `A_1`, Brauer–Klimyk otherwise.
    pub(crate) fn mult(&self, a: Label<'_>, b: Label<'_>, c: Label<'_>) -> Result<usize, LieError> {
        if self.root() == RootSystem::A(1) {
            let dj = |x: Label<'_>| u32::try_from(x.words[0]).map_err(|_| LieError::LabelOverflow);
            return Ok(usize::from(crate::lie::su2::triangle_ok(
                dj(a)?,
                dj(b)?,
                dj(c)?,
            )));
        }
        Ok(self.data.label_multiplicity(a, b, c.words, None)? as usize)
    }

    /// The rational channel with its three modules, gated by the exact
    /// nullity check, uncached.
    #[cfg(test)]
    pub(super) fn channel(
        &self,
        a: &[i64],
        b: &[i64],
        c: &[i64],
    ) -> Result<BuiltChannel, LieError> {
        let [a, b, c] = [a, b, c].map(Label::new);
        let mut modules = RequestModules::default();
        let ma = modules.load(self, a)?;
        let mb = modules.load(self, b)?;
        let mc = modules.load(self, c)?;
        let ch = ExactChannel::new(&self.data, &ma, &mb, &mc, self.mult(a, b, c)?)?;
        Ok((ch, ma, mb, mc))
    }

    /// The Weyl dimension of `l` as an index bound.
    pub(crate) fn dimension(&self, l: Label<'_>) -> Result<usize, LieError> {
        usize::try_from(&self.data.weyl_dimension(l.words)?).map_err(|_| LieError::LabelOverflow)
    }

    /// The completed channel `a ⊗ b → c`: its multiplicity and its rounded,
    /// gated entries in §4.7 order, a view of the exact channel shared through
    /// the CGC tier, so a hit needs no multiplicity lookup.
    pub(crate) fn cgc_entries(
        &self,
        a: Label<'_>,
        b: Label<'_>,
        c: Label<'_>,
        exec: Option<&mut Exec<'_>>,
    ) -> Result<Arc<CgcChannel>, LieError> {
        self.cgc_entries_in(a, b, c, &mut RequestModules::default(), exec)
    }

    /// [`Self::cgc_entries`] as one channel of a request: a miss takes its
    /// modules from, and adds them to, `modules`; a hit leaves it untouched.
    /// Only a miss uses `exec`, for the descent, the rounding and the
    /// isometry gate; `None` is the serial token, so a hit never builds one.
    ///
    /// A view miss reuses the view an exact generation it led produced, and
    /// otherwise rounds the resident or disk-loaded exact channel with its
    /// norm tables, building no module when they are available. An empty
    /// channel needs no norms.
    pub(crate) fn cgc_entries_in<'l>(
        &self,
        a: Label<'l>,
        b: Label<'l>,
        c: Label<'l>,
        modules: &mut RequestModules<'l>,
        mut exec: Option<&mut Exec<'_>>,
    ) -> Result<Arc<CgcChannel>, LieError> {
        #[cfg(test)]
        if let Some(hook) = BEFORE_LOAD.with(std::cell::Cell::get) {
            hook();
        }
        crate::cache::cgc_channel(self.root(), [a, b, c], || {
            let mut fresh = None;
            let exact = self.exact_channel_in(a, b, c, modules, exec.as_deref_mut(), &mut fresh)?;
            #[cfg(test)]
            if let Some(hook) = AFTER_EXACT.with(std::cell::Cell::take) {
                hook();
            }
            if let Some(view) = fresh {
                return Ok(view);
            }
            if exact.is_empty() {
                return Ok(Arc::new(CgcChannel {
                    mult: exact.mult(),
                    entries: Box::new([]),
                }));
            }
            let norms = [
                self.norm_table_in(a, modules)?,
                self.norm_table_in(b, modules)?,
                self.norm_table_in(c, modules)?,
            ];
            Ok(Arc::new(round_channel(
                &exact,
                norms.each_ref().map(|n| &**n),
                exec,
            )))
        })
    }

    /// The exact channel `a ⊗ b → c`, shared through the CGC tier and the
    /// disk tier. A generation also rounds and gates its view, and is
    /// retained only once that view passed.
    #[cfg(test)]
    pub(crate) fn exact_channel(
        &self,
        a: Label<'_>,
        b: Label<'_>,
        c: Label<'_>,
        exec: Option<&mut Exec<'_>>,
    ) -> Result<Arc<ExactCgc>, LieError> {
        self.exact_channel_in(a, b, c, &mut RequestModules::default(), exec, &mut None)
    }

    /// [`Self::exact_channel`] within a request; `fresh` receives the gated
    /// view when this call generated the channel. On a miss the block
    /// `d_a d_b d_c N` must be addressable before anything is built.
    fn exact_channel_in<'l>(
        &self,
        a: Label<'l>,
        b: Label<'l>,
        c: Label<'l>,
        modules: &mut RequestModules<'l>,
        exec: Option<&mut Exec<'_>>,
        fresh: &mut Option<Arc<CgcChannel>>,
    ) -> Result<Arc<ExactCgc>, LieError> {
        crate::cache::cgc_exact_channel(self.root(), [a, b, c], || {
            let n = self.mult(a, b, c)?;
            let dims = [self.dimension(a)?, self.dimension(b)?, self.dimension(c)?];
            crate::lie::checked_size([dims[0], dims[1], dims[2], n])?;
            crate::cache::through(
                crate::cache::DISK_CHANNEL,
                self.root(),
                [a, b, c],
                || Ok(crate::cache::Shape::Channel { dims, mult: n }),
                || {
                    let (exact, view) = self.generate_channel(a, b, c, n, dims, modules, exec)?;
                    *fresh = Some(view);
                    Ok(exact)
                },
            )
        })
    }

    /// The miss body of [`Self::exact_channel_in`] once `N` and the block
    /// size are known: an exact empty certificate, or the descent, its view
    /// and the view's gates. A channel with `N = 0` is empty, after an exact
    /// certificate: no coupling pair has the highest weight of `c`, or the
    /// highest-weight nullity is zero (§7 gate 2).
    #[allow(clippy::too_many_arguments)]
    fn generate_channel<'l>(
        &self,
        a: Label<'l>,
        b: Label<'l>,
        c: Label<'l>,
        n: usize,
        dims: [usize; 3],
        modules: &mut RequestModules<'l>,
        mut exec: Option<&mut Exec<'_>>,
    ) -> Result<(Arc<ExactCgc>, Arc<CgcChannel>), LieError> {
        #[cfg(test)]
        count_build(1);
        let empty = || {
            Ok((
                Arc::new(ExactCgc::empty(dims)),
                Arc::new(CgcChannel {
                    mult: 0,
                    entries: Box::new([]),
                }),
            ))
        };
        // No coupling pair has the weight `λ_c`, so `P` is empty and a zero
        // multiplicity needs no nullspace. For `A_1` the weight `c` occurs in
        // `a ⊗ b` exactly when `c ≤ a + b` with `a + b − c` even; other types
        // read the weights off the two modules, which the channel then reuses.
        let a1 = self.root() == RootSystem::A(1);
        if n == 0 && a1 && {
            let s = a.words[0] + b.words[0] - c.words[0];
            s < 0 || s % 2 != 0
        } {
            return empty();
        }
        let ma = modules.load(self, a)?;
        let mb = modules.load(self, b)?;
        if n == 0
            && !a1
            && ProductSpace::new(&ma, &mb, c.words, &mut ProductScratch::new(&ma, &mb)).len() == 0
        {
            return empty();
        }
        let mc = modules.load(self, c)?;
        // §4.6: the A_1 vertex sign (−1)^{j_a + j_b − j_c}, once per channel,
        // on the exact coordinates.
        let negate = a1 && (a.words[0] + b.words[0] - c.words[0]) / 2 % 2 != 0;
        let pending =
            super::cgc::exact_columns(&self.data, &ma, &mb, &mc, n, negate, exec.as_deref_mut())?;
        // No exact vector outlives the descent; the norms are resolved before
        // any team section, and the gates read the view and the modules only.
        #[allow(unused_mut)]
        let (exact, mut view) = if pending.is_empty() {
            let mult = pending.mult();
            (
                pending.integralize(dims, None),
                CgcChannel {
                    mult,
                    entries: Box::new([]),
                },
            )
        } else {
            let norms = [
                self.norm_table_in(a, modules)?,
                self.norm_table_in(b, modules)?,
                self.norm_table_in(c, modules)?,
            ];
            write_and_round(
                pending,
                dims,
                norms.each_ref().map(|n| &**n),
                exec.as_deref_mut(),
            )
        };
        #[cfg(test)]
        if let Some(corrupt) = CORRUPT.with(std::cell::Cell::get) {
            corrupt(&mut view.entries[..]);
        }
        check_rounded(&self.data, &ma, &mb, &mc, n, &view.entries, exec)?;
        Ok((Arc::new(exact), Arc::new(view)))
    }

    /// The highest-weight column of `a ⊗ b → c` (engine index 0 of `c`) for a
    /// root system of rank two or more, shared through the CGC tier and the
    /// disk tier. Rank-one F and R never read seeds, so `A_1` is refused.
    pub(crate) fn cgc_seeds_in<'l>(
        &self,
        a: Label<'l>,
        b: Label<'l>,
        c: Label<'l>,
        modules: &mut RequestModules<'l>,
    ) -> Result<Arc<HighestColumn>, LieError> {
        if self.root() == RootSystem::A(1) {
            return Err(LieError::Unsupported {
                requires: "a root system of rank two or more",
            });
        }
        crate::cache::cgc_seeds(self.root(), [a, b, c], || {
            let n = self.mult(a, b, c)?;
            let dims = [self.dimension(a)?, self.dimension(b)?, self.dimension(c)?];
            crate::lie::checked_size([dims[0], dims[1], dims[2], n])?;
            crate::cache::through(
                crate::cache::DISK_SEEDS,
                self.root(),
                [a, b, c],
                || {
                    Ok(crate::cache::Shape::Seeds {
                        dims: [dims[0], dims[1]],
                        mult: n,
                    })
                },
                || {
                    #[cfg(test)]
                    count_build(2);
                    let ma = modules.load(self, a)?;
                    let mb = modules.load(self, b)?;
                    if n == 0
                        && ProductSpace::new(&ma, &mb, c.words, &mut ProductScratch::new(&ma, &mb))
                            .len()
                            == 0
                    {
                        return Ok(Arc::new(HighestColumn::empty()));
                    }
                    let mc = modules.load(self, c)?;
                    super::cgc::highest_column(&self.data, &ma, &mb, &mc, n).map(Arc::new)
                },
            )
        })
    }

    /// The norm table of `label`, shared through the CGC tier and the disk
    /// tier; a miss builds it from the module, which `modules` holds.
    pub(crate) fn norm_table_in<'l>(
        &self,
        label: Label<'l>,
        modules: &mut RequestModules<'l>,
    ) -> Result<Arc<NormTable>, LieError> {
        crate::cache::cgc_norms(self.root(), label, || {
            crate::cache::through(
                crate::cache::DISK_NORMS,
                self.root(),
                [label],
                || {
                    Ok(crate::cache::Shape::Norms {
                        dim: self.dimension(label)?,
                    })
                },
                || {
                    #[cfg(test)]
                    count_build(3);
                    let module = modules.load(self, label)?;
                    NormTable::new(&module).map(Arc::new)
                },
            )
        })
    }

    /// `κ_a` (§5.3) of `a` with dual label `dual`: `(−1)^{⟨λ, 2ρ∨⟩}` for a
    /// self-dual irrep and for every `A_r` irrep, otherwise
    /// `(−1)^{⟨λ, 2ρ∨⟩} σ_a σ_ā`, computed once per dual pair and shared
    /// through the CGC tier.
    pub(crate) fn frobenius_schur_phase(
        &self,
        a: Label<'_>,
        dual: Label<'_>,
    ) -> Result<i8, LieError> {
        let closed = self.data.two_rho_vee_sign(a.words)?;
        if dual.words == a.words || matches!(self.root(), RootSystem::A(_)) {
            return Ok(closed);
        }
        #[cfg(test)]
        if let Some(fail) = FAIL_PHASE.with(std::cell::Cell::get) {
            return Err(fail());
        }
        // κ_ā = κ_a (§5.3), so one generation answers both labels.
        let key = if a.words <= dual.words { a } else { dual };
        crate::cache::cgc_phase(self.root(), key, || {
            crate::cache::through(
                crate::cache::DISK_PHASE,
                self.root(),
                [key],
                || Ok(crate::cache::Shape::Phase),
                || {
                    let sigma_a = self.module(a)?.principal_sign(a.words)?;
                    let sigma_dual = self.module(dual)?.principal_sign(dual.words)?;
                    Ok(closed * sigma_a * sigma_dual)
                },
            )
        })
    }

    /// `F^{abc}_d[e, f]` (§6): the rounded view of the exact block, gated by
    /// gate 7. A view miss reuses the view an exact generation it led
    /// produced, and otherwise rounds the resident or disk-loaded exact block.
    pub(crate) fn f_symbol(
        &self,
        labels: [Label<'_>; 6],
        exec: Option<&mut Exec<'_>>,
    ) -> Result<FBlock<f64>, LieError> {
        crate::cache::f_block(self.root(), labels, || {
            let mut fresh = None;
            let exact = self.f_exact_in(labels, exec, &mut fresh)?;
            match fresh {
                Some(view) => Ok(view),
                None => f_view(&exact),
            }
        })
    }

    /// The exact F block of the six labels, shared through the F tier and the
    /// disk tier.
    pub(crate) fn f_exact(&self, labels: [Label<'_>; 6]) -> Result<Arc<RadicalBlock<4>>, LieError> {
        self.f_exact_in(labels, None, &mut None)
    }

    /// [`Self::f_exact`]; `fresh` receives the gated view when this call
    /// generated the block. A generation is retained only once its view
    /// passed gate 7, and never claims, publishes or waits on a view.
    fn f_exact_in(
        &self,
        labels: [Label<'_>; 6],
        exec: Option<&mut Exec<'_>>,
        fresh: &mut Option<FBlock<f64>>,
    ) -> Result<Arc<RadicalBlock<4>>, LieError> {
        crate::cache::f_exact(self.root(), labels, || {
            let [a, b, c, d, e, f] = labels;
            crate::cache::through(
                crate::cache::DISK_F,
                self.root(),
                labels,
                || {
                    Ok(crate::cache::Shape::F([
                        self.mult(a, b, e)?,
                        self.mult(e, c, d)?,
                        self.mult(b, c, f)?,
                        self.mult(a, f, d)?,
                    ]))
                },
                || {
                    let exact = Arc::new(self.generate_f(labels, exec)?);
                    *fresh = Some(f_view_generated(&exact)?);
                    Ok(exact)
                },
            )
        })
    }

    /// The exact contraction behind [`Self::f_exact`] (§6, §9), loading its
    /// inputs in the order that keeps at most five of the six labels'
    /// modules live and none during the contraction.
    fn generate_f(
        &self,
        labels: [Label<'_>; 6],
        mut exec: Option<&mut Exec<'_>>,
    ) -> Result<RadicalBlock<4>, LieError> {
        let [a, b, c, d, e, f] = labels;
        let dims = [
            self.mult(a, b, e)?,
            self.mult(e, c, d)?,
            self.mult(b, c, f)?,
            self.mult(a, f, d)?,
        ];
        if dims.contains(&0) {
            return RadicalBlock::zeros(dims).ok_or(LieError::LabelOverflow);
        }
        crate::lie::checked_size(dims)?;
        let mut modules = RequestModules::default();
        #[cfg(test)]
        before_load();
        let ca = self.exact_channel_in(a, b, e, &mut modules, exec.as_deref_mut(), &mut None)?;
        let [na, nb] = [a, b].map(|l| self.norm_table_in(l, &mut modules));
        let (na, nb) = (na?, nb?);
        #[cfg(test)]
        before_load();
        let sb = self.cgc_seeds_in(e, c, d, &mut modules)?;
        let (nc, nd) = (
            self.norm_table_in(c, &mut modules)?,
            self.norm_table_in(d, &mut modules)?,
        );
        modules.retain(&[b, c, f, a, d]);
        #[cfg(test)]
        before_load();
        let cc = self.exact_channel_in(b, c, f, &mut modules, exec, &mut None)?;
        modules.retain(&[a, f, d]);
        #[cfg(test)]
        modules.observe_before_fourth();
        #[cfg(test)]
        before_load();
        let sd = self.cgc_seeds_in(a, f, d, &mut modules)?;
        drop(modules);
        #[cfg(test)]
        observe_contraction();
        f_exact(&FInputs {
            a: &ca,
            cb: &cc,
            b: &sb,
            d: &sd,
            norms: [&na, &nb, &nc, &nd],
            dims,
        })
    }

    /// `R^{ab}_c` (§6): the rounded view of the exact block, gated by gate 6.
    pub(crate) fn r_symbol(&self, labels: [Label<'_>; 3]) -> Result<RBlock<f64>, LieError> {
        crate::cache::r_block(self.root(), labels, || {
            let mut fresh = None;
            let exact = self.r_exact_in(labels, &mut fresh)?;
            match fresh {
                Some(view) => Ok(view),
                None => r_view(&exact),
            }
        })
    }

    /// The exact R block of the three labels, shared through the R tier and
    /// the disk tier.
    pub(crate) fn r_exact(&self, labels: [Label<'_>; 3]) -> Result<Arc<RadicalBlock<2>>, LieError> {
        self.r_exact_in(labels, &mut None)
    }

    /// The exact R block of the three labels, shared through the R tier and
    /// the disk tier; as [`Self::f_exact_in`], retained only once its view
    /// passed gate 6.
    fn r_exact_in(
        &self,
        labels: [Label<'_>; 3],
        fresh: &mut Option<RBlock<f64>>,
    ) -> Result<Arc<RadicalBlock<2>>, LieError> {
        crate::cache::r_exact(self.root(), labels, || {
            let [a, b, c] = labels;
            crate::cache::through(
                crate::cache::DISK_R,
                self.root(),
                labels,
                || Ok(crate::cache::Shape::R(self.mult(a, b, c)?)),
                || {
                    let exact = Arc::new(self.generate_r(labels)?);
                    *fresh = Some(r_view_generated(&exact)?);
                    Ok(exact)
                },
            )
        })
    }

    /// The exact contraction behind [`Self::r_exact_in`]: two seed columns
    /// and three norm tables, no descent.
    fn generate_r(&self, [a, b, c]: [Label<'_>; 3]) -> Result<RadicalBlock<2>, LieError> {
        let n = self.mult(a, b, c)?;
        if n == 0 {
            return RadicalBlock::zeros([0, 0]).ok_or(LieError::LabelOverflow);
        }
        crate::lie::checked_size([n, n])?;
        let mut modules = RequestModules::default();
        #[cfg(test)]
        before_load();
        let ab = self.cgc_seeds_in(a, b, c, &mut modules)?;
        let na = self.norm_table_in(a, &mut modules)?;
        let nb = self.norm_table_in(b, &mut modules)?;
        let nc = self.norm_table_in(c, &mut modules)?;
        #[cfg(test)]
        before_load();
        let ba = self.cgc_seeds_in(b, a, c, &mut modules)?;
        drop(modules);
        #[cfg(test)]
        observe_contraction();
        r_exact(&RInputs {
            ab: &ab,
            ba: &ba,
            norms: [&na, &nb, &nc],
            n,
        })
    }
}

/// The rounded F view of `exact`, gated by gate 7.
fn f_view(exact: &RadicalBlock<4>) -> Result<FBlock<f64>, LieError> {
    let view = FBlock::from_vec(exact.dims(), round_block(exact));
    check_f_block(&view)?;
    Ok(view)
}

/// [`f_view`] of a freshly generated block, where a test may corrupt the
/// rounding before the gate; a disk-loaded block is never corrupted.
fn f_view_generated(exact: &RadicalBlock<4>) -> Result<FBlock<f64>, LieError> {
    #[cfg(test)]
    if let Some(fault) = CORRUPT_BLOCK.with(std::cell::Cell::get) {
        let view = FBlock::from_vec(
            exact.dims(),
            round_block(exact).into_iter().map(fault).collect(),
        );
        check_f_block(&view)?;
        return Ok(view);
    }
    f_view(exact)
}

/// The rounded R view of `exact`, gated by gate 6.
fn r_view(exact: &RadicalBlock<2>) -> Result<RBlock<f64>, LieError> {
    let view = RBlock::from_vec(exact.dims()[0], round_block(exact));
    check_r_block(&view)?;
    Ok(view)
}

/// [`r_view`] of a freshly generated block (see [`f_view_generated`]).
fn r_view_generated(exact: &RadicalBlock<2>) -> Result<RBlock<f64>, LieError> {
    #[cfg(test)]
    if let Some(fault) = CORRUPT_BLOCK.with(std::cell::Cell::get) {
        let view = RBlock::from_vec(
            exact.dims()[0],
            round_block(exact).into_iter().map(fault).collect(),
        );
        check_r_block(&view)?;
        return Ok(view);
    }
    r_view(exact)
}

/// The test hook run before each channel or seed load of an F or R
/// generation.
#[cfg(test)]
fn before_load() {
    if let Some(hook) = BEFORE_LOAD.with(std::cell::Cell::get) {
        hook();
    }
}
