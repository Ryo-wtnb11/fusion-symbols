//! The exceptional Lie engine (`docs/gauge_exceptional.md`): generic
//! Cartan-matrix machinery and its binding to `LieGroup`. Crate-private.

pub(crate) mod cgc;
pub(crate) mod module;
pub(crate) mod roots;

use std::sync::{Arc, Mutex, MutexGuard};

use num_bigint::BigInt;

use self::cgc::GeneralFamily;
use self::roots::CartanData;
use crate::category::{FBlock, RBlock};
use crate::frcore::{f_block_raw, r_block_raw, Family};
use crate::lie::{checked_dimension, checked_size, weyl_dim_f64, LieError, RootSystem};

/// The exceptional root systems whose coefficients this engine serves: all
/// five. E6 is the only one with complex irreps, whose `κ_a` is generated
/// (§6.3).
pub(crate) fn serves(root: RootSystem) -> bool {
    matches!(
        root,
        RootSystem::G2 | RootSystem::F4 | RootSystem::E6 | RootSystem::E7 | RootSystem::E8
    )
}

/// `docs/gauge_exceptional.md` §8. Bump the trailing `epoch=N` (and the
/// literal in `tests/exceptional_fingerprint.rs`) only when that document is
/// corrected under the four-step rule of `docs/gauge.md`, "Status".
pub(crate) fn fingerprint() -> &'static [u8] {
    b"fusion-symbols:lie-exceptional:ref=gauge-exceptional-spec:input=bourbaki-cartan:order=depth-rootlex-asc:basis=greedy-rank-gs-own-positive:kron=a-fast:mult=rcef-pivot-gs:descent=intertwiner-own-basis:sign=own-candidate-and-pivot-positive:round=once-correct:fs=gauge-derived:fr=frcore:twist=one:scalar=f64:epoch=1"
}

/// Memo entries (modules + products + channels + phases) retained after a query.
/// A pentagon through the G2 vertex `27 ⊗ 27 → 27` fits; at 256 it ran twice as long.
const MEMO_ENTRIES: usize = 512;

/// Estimated retained memo bytes (see [`GeneralFamily::memo_bytes`]).
/// Every exceptional group value and composite factor owns its own memo, so this is per factor. 256 MiB is the
/// default per-group B/C/D catalog budget, and it holds the whole working set
/// of a pentagon through the G2 `N = 2` vertex `27 ⊗ 27 → 27` (about 150 MiB
/// estimated); at 64 MiB that pentagon reset repeatedly and ran 3.5× longer.
/// One entry for an irrep of a few hundred dimensions can be tens of MiB, and
/// one query may exceed the cap transiently: it is enforced between queries.
const MEMO_BYTES: usize = 256 << 20;

/// One exceptional simple factor: its root data and the memo of exact
/// modules, products, channels and complex-irrep phases, owned by the group value and bounded by
/// [`MEMO_ENTRIES`] and [`MEMO_BYTES`]. Values are exact, so the memo state
/// never changes a bit. Fusion and N use only the root data and never take
/// the memo lock.
pub(crate) struct Engine {
    root: RootSystem,
    data: CartanData,
    family: Mutex<GeneralFamily>,
}

impl Engine {
    pub(crate) fn new(root: RootSystem) -> Self {
        Self {
            root,
            data: CartanData::new(root),
            family: Mutex::new(GeneralFamily::new(root)),
        }
    }

    fn family(&self) -> MutexGuard<'_, GeneralFamily> {
        // A panic mid-query can leave a memo without its partner entries;
        // starting over is always valid.
        self.family.lock().unwrap_or_else(|poisoned| {
            self.family.clear_poison();
            let mut fam = poisoned.into_inner();
            *fam = GeneralFamily::new(self.root);
            fam
        })
    }

    fn query<T>(&self, query: impl FnOnce(&mut GeneralFamily) -> T) -> T {
        let mut fam = self.family();
        let result = query(&mut fam);
        fam.trim(MEMO_ENTRIES, MEMO_BYTES);
        #[cfg(test)]
        assert!(fam.memo_len() <= MEMO_ENTRIES && fam.memo_bytes() <= MEMO_BYTES);
        result
    }

    /// `(entries, estimated bytes)` currently retained.
    #[cfg(test)]
    pub(crate) fn memo_stats(&self) -> (usize, usize) {
        let fam = self.family.lock().unwrap();
        (fam.memo_len(), fam.memo_bytes())
    }

    /// Drop every memoized module, product, channel and phase.
    pub(crate) fn reset(&self) {
        // Poison recovery already starts over; only the healthy case needs it.
        *self.family() = GeneralFamily::new(self.root);
    }

    /// A validated label: nonnegative, of this rank, with an exact dimension.
    /// A complex irrep (and so its dual) must have dimension at most
    /// `u32::MAX`: its `κ_a` is generated in the module basis (§6.3), whose
    /// magnetic index is `u32`, and `frobenius_schur_phase` is infallible.
    pub(crate) fn irrep(&self, dynkin: &[i64]) -> Result<Irrep, LieError> {
        let dim = self.data.weyl_dimension(dynkin)?;
        let dual = self.data.dual(dynkin)?;
        if dual != dynkin && u32::try_from(&dim).is_err() {
            return Err(LieError::LabelOverflow);
        }
        let sign = self.data.two_rho_vee_sign(dynkin)?;
        let dim_f64 = weyl_dim_f64(u128::try_from(&dim).ok(), || dim.clone());
        Ok(Irrep(Arc::new(Data {
            dynkin: dynkin.to_vec(),
            dual,
            dim,
            dim_f64,
            sign,
        })))
    }

    /// `a ⊗ b` in ascending Dynkin-label order, with multiplicities.
    pub(crate) fn product(&self, a: &Irrep, b: &Irrep) -> Result<Arc<[(Irrep, u32)]>, LieError> {
        self.data
            .tensor_product(&a.0.dynkin, &b.0.dynkin)?
            .into_iter()
            .map(|(c, n)| Ok((self.irrep(&c)?, n)))
            .collect()
    }

    pub(crate) fn nsymbol(&self, a: &Irrep, b: &Irrep, c: &Irrep) -> Result<u32, LieError> {
        Ok(self
            .data
            .tensor_product(&a.0.dynkin, &b.0.dynkin)?
            .get(&c.0.dynkin)
            .copied()
            .unwrap_or(0))
    }

    /// `κ_a` (§6.3): the stored closed form for a self-dual irrep, otherwise
    /// the gauge-derived `σ_a σ_ā`, generated on first use under the memo
    /// lock and then memoized for `a` and `ā`.
    pub(crate) fn frobenius_schur_phase(&self, a: &Irrep) -> Result<i8, LieError> {
        if a.is_self_dual() {
            return Ok(a.0.sign);
        }
        self.query(|fam| fam.frobenius_schur_phase(&a.0.dynkin))
    }

    pub(crate) fn f_symbol(
        &self,
        [a, b, c, d, e, f]: [&Irrep; 6],
    ) -> Result<FBlock<f64>, LieError> {
        let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|x| &x.0.dynkin);
        self.query(|fam| f_block_raw(fam, a, b, c, d, e, f))
    }

    pub(crate) fn r_symbol(
        &self,
        a: &Irrep,
        b: &Irrep,
        c: &Irrep,
    ) -> Result<RBlock<f64>, LieError> {
        self.query(|fam| r_block_raw(fam, &a.0.dynkin, &b.0.dynkin, &c.0.dynkin))
    }

    /// The CGC as dims `[d_a, d_b, d_c, N]` and a dense row-major buffer
    /// (empty for an absent channel).
    pub(crate) fn cgc(
        &self,
        a: &Irrep,
        b: &Irrep,
        c: &Irrep,
    ) -> Result<([usize; 4], Vec<f64>), LieError> {
        self.query(|fam| {
            let [da, db, dc] = [a, b, c].map(|x| checked_dimension(&x.0.dim));
            let dims = [
                da?,
                db?,
                dc?,
                fam.mult(&a.0.dynkin, &b.0.dynkin, &c.0.dynkin)?,
            ];
            if dims[3] == 0 {
                return Ok((dims, Vec::new()));
            }
            let mut data = vec![0.0; checked_size(dims)?];
            for e in fam.cgc_entries(&a.0.dynkin, &b.0.dynkin, &c.0.dynkin)? {
                let [m1, m2, m3, mu] = [e.m1, e.m2, e.m3, e.mu].map(|i| i as usize);
                data[((m1 * dims[1] + m2) * dims[2] + m3) * dims[3] + mu] = e.value;
            }
            Ok((dims, data))
        })
    }
}

/// An exceptional irrep label with its exact dimension, dual label and the
/// closed-form sign `(−1)^{⟨λ, 2ρ∨⟩}`. Clones share one allocation; identity
/// is the Dynkin label (the group is compared by `LieIrrep`).
#[derive(Clone, Debug)]
pub(crate) struct Irrep(Arc<Data>);

#[derive(Debug)]
struct Data {
    dynkin: Vec<i64>,
    dual: Vec<i64>,
    dim: BigInt,
    dim_f64: f64,
    /// `⟨−w₀λ, 2ρ∨⟩ = ⟨λ, 2ρ∨⟩`, so the dual carries the same sign.
    sign: i8,
}

impl Irrep {
    pub(crate) fn dynkin(&self) -> &[i64] {
        &self.0.dynkin
    }

    pub(crate) fn dim(&self) -> &BigInt {
        &self.0.dim
    }

    pub(crate) fn dim_f64(&self) -> f64 {
        self.0.dim_f64
    }

    pub(crate) fn is_self_dual(&self) -> bool {
        self.0.dynkin == self.0.dual
    }

    pub(crate) fn dual(&self) -> Self {
        if self.is_self_dual() {
            return self.clone();
        }
        Self(Arc::new(Data {
            dynkin: self.0.dual.clone(),
            dual: self.0.dynkin.clone(),
            dim: self.0.dim.clone(),
            dim_f64: self.0.dim_f64,
            sign: self.0.sign,
        }))
    }

    /// Every weight with its multiplicity, in Dynkin coordinates.
    pub(crate) fn weight_multiplicities(
        &self,
        root: RootSystem,
    ) -> Result<std::collections::BTreeMap<Vec<i64>, u64>, LieError> {
        CartanData::new(root).weight_multiplicities(&self.0.dynkin)
    }
}

impl PartialEq for Irrep {
    fn eq(&self, other: &Self) -> bool {
        self.0.dynkin == other.0.dynkin
    }
}
impl Eq for Irrep {}
impl PartialOrd for Irrep {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Irrep {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.dynkin.cmp(&other.0.dynkin)
    }
}
impl std::hash::Hash for Irrep {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.dynkin.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::{f_unitarity_residual, FusionCategory};
    use crate::lie::{GroupId, LieGroup};
    use std::time::Instant;

    /// A hot completed channel survives a product working set larger than the
    /// public entry limit; eviction and reset never change coefficient bits.
    #[test]
    fn memo_pressure_retains_completed_channels() {
        let engine = Engine::new(RootSystem::G2);
        let a = engine.irrep(&[1, 0]).unwrap();
        let b = engine.irrep(&[0, 1]).unwrap();
        let one = engine.irrep(&[0, 0]).unwrap();
        let bits = || {
            engine
                .cgc(&a, &a, &b)
                .unwrap()
                .1
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        let expected = bits();
        let builds = engine.family().builds;
        assert_eq!(bits(), expected);
        assert_eq!(
            engine.family().builds,
            builds,
            "warm calls must not regenerate"
        );
        for _ in 0..3 {
            for n in 1..=MEMO_ENTRIES + 16 {
                let label = engine.irrep(&[n as i64, 0]).unwrap();
                // Absent R channels fill products without building large modules.
                assert!(engine
                    .r_symbol(&one, &label, &one)
                    .unwrap()
                    .data()
                    .is_empty());
                let (entries, bytes) = engine.memo_stats();
                assert!(entries <= MEMO_ENTRIES && bytes <= MEMO_BYTES);
            }
            assert_eq!(bits(), expected);
            assert_eq!(
                engine.family().builds,
                builds,
                "pressure must not rebuild a retained channel"
            );
        }
        engine.family().trim(0, 0);
        assert_eq!(engine.memo_stats(), (0, 0));
        assert_eq!(bits(), expected);
        assert_eq!(engine.family().builds, [2 * builds[0], 2 * builds[1]]);
        engine.reset();
        assert_eq!(engine.memo_stats(), (0, 0));
        assert_eq!(bits(), expected);
        let symbols = || {
            let f = engine.f_symbol([&a, &a, &a, &a, &b, &b]).unwrap();
            let r = engine.r_symbol(&a, &a, &b).unwrap();
            f.data()
                .iter()
                .chain(r.data())
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        let expected = symbols();
        engine.family().trim(0, 0);
        assert_eq!(symbols(), expected);
    }

    #[test]
    #[ignore = "release timing and generation-count report"]
    fn e8_memo_workload_report() {
        let g = LieGroup::new(GroupId::e8()).unwrap();
        let a = g.irrep(&[0, 0, 0, 0, 0, 0, 0, 1]).unwrap();
        let t = Instant::now();
        let residual = f_unitarity_residual(&g, &a, &a, &a, &a).unwrap();
        assert!(residual < 1e-12);
        let fam = g.general.as_ref().unwrap().family.lock().unwrap();
        eprintln!(
            "E8: {:?}, residual {residual:e}, module/channel builds {:?}, retained ({}, {})",
            t.elapsed(),
            fam.builds,
            fam.memo_len(),
            fam.memo_bytes()
        );
    }

    #[test]
    #[ignore = "release cold/warm timing report"]
    fn small_memo_workload_report() {
        for (id, label) in [
            (GroupId::g2(), vec![1, 0]),
            (GroupId::f4(), vec![0, 0, 0, 1]),
            (GroupId::e6(), vec![1, 0, 0, 0, 0, 0]),
        ] {
            for _ in 0..5 {
                let g = LieGroup::new(id.clone()).unwrap();
                let a = g.irrep(&label).unwrap();
                let query = || {
                    if label.len() == 6 {
                        assert_eq!(g.frobenius_schur_phase(&a), 1.0);
                    } else {
                        std::hint::black_box(g.r_symbol(&a, &a, &g.unit()).unwrap());
                    }
                };
                let t = Instant::now();
                query();
                let cold = t.elapsed();
                let builds = g.general.as_ref().unwrap().family.lock().unwrap().builds;
                let t = Instant::now();
                for _ in 0..128 {
                    query();
                }
                let warm = t.elapsed() / 128;
                assert_eq!(
                    g.general.as_ref().unwrap().family.lock().unwrap().builds,
                    builds
                );
                eprintln!("{label:?}: cold {cold:?}, warm {warm:?}, builds {builds:?}");
            }
        }
    }
}
