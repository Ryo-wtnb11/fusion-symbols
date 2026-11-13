//! Exact Clebsch–Gordan coefficients of `a ⊗ b → c`
//! (`docs/gauge_cartan.md` §4), the F and R contractions of §6 and the
//! Frobenius–Schur phase `κ_a` of §5.3, for every simple root system.
//!
//! Everything is computed in the rational coordinates `b^a_{m_a} ⊗ b^b_{m_b}`
//! of §3.3, whose metric is `diag(n^a_{m_a} n^b_{m_b})`. Each lowered vector
//! is `√N_μ Φ_μ(b^c_{m_c})`, so a CGC entry is its rational coordinate `x`
//! times `√(n^a n^b / (N_μ n^c))`: one signed square root, rounded once.
//!
//! Before a rounded channel is retained or returned it passes two
//! verification gates (§7, gates 4 and 5), which never change a value: an
//! isometry check over every expected column within each output weight block,
//! and a first-descent ladder check for every applicable simple generator.
//! Completed F and R blocks pass gates 7 and 6 before they are retained or
//! returned.
//!
//! Every retained module, channel and phase lives in the process-wide CGC
//! tier of [`crate::cache`]; [`GeneralFamily`] is a query context over one
//! root system and owns no state of its own.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use threadteam::Exec;

use super::module::{q_heap, qs_bytes, IrrepModule, Q};
use super::roots::CartanData;
use crate::cache::Label;
use crate::category::{FBlock, RBlock};
#[cfg(test)]
use crate::exact::SignedSqrtRational;
use crate::frcore::{check_f_block, check_r_block, f_block_raw, r_block_raw, GateError, MEntry};
use crate::lie::{GenerationError, LieError, RootSystem};

/// A sparse operator in the rational basis: `op[m]` lists `(m', x)` with
/// `X b_m = Σ x b_{m'}`.
type Op = Vec<Vec<(usize, Q)>>;

/// An internal inconsistency between the construction and its gates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum CgcError {
    /// `dim H` differs from the Brauer–Klimyk multiplicity (§7, gate 2).
    NullityMismatch {
        a: Vec<i64>,
        b: Vec<i64>,
        c: Vec<i64>,
        nullity: usize,
        multiplicity: usize,
    },
    /// `⟨low| L^h |high⟩ = 0`, which §5.3 proves impossible (§7, gate 3).
    ZeroPrincipalLadder { lambda: Vec<i64> },
}

impl std::fmt::Display for CgcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NullityMismatch {
                a,
                b,
                c,
                nullity,
                multiplicity,
            } => write!(
                f,
                "{a:?} ⊗ {b:?} → {c:?}: highest-weight space has dimension {nullity}, \
                 Brauer–Klimyk multiplicity is {multiplicity}"
            ),
            Self::ZeroPrincipalLadder { lambda } => {
                write!(f, "{lambda:?}: principal lowering of v_λ vanishes")
            }
        }
    }
}

impl std::error::Error for CgcError {}

impl From<CgcError> for LieError {
    fn from(error: CgcError) -> Self {
        Self::Generation(GenerationError(std::sync::Arc::new(error)))
    }
}

impl From<GateError> for LieError {
    fn from(error: GateError) -> Self {
        Self::Generation(GenerationError(std::sync::Arc::new(error)))
    }
}

/// `L(λ)` with its ladder operators flattened to engine indices: the weight
/// spaces of the module in depth order (§3.2.1), so index 0 is the highest
/// weight and the last weight space is the lowest. `public` maps them to
/// magnetic indices.
pub(crate) struct Ladders {
    pub(super) module: IrrepModule,
    /// `n_m = ⟨b_m, b_m⟩` per magnetic index.
    pub(super) norm: Vec<Q>,
    /// Weight space of each magnetic index.
    pub(super) space: Vec<usize>,
    /// `f_j` per `j`.
    pub(super) lower: Vec<Op>,
    /// `e_j` per `j`, the contravariant transpose `E[b][a] = F[a][b] n_a / n_b`.
    pub(super) raise: Vec<Op>,
    /// The public magnetic index of each engine index, a permutation; the
    /// identity for the Gram–Schmidt basis.
    pub(super) public: Vec<u32>,
    /// The position of each engine index within its weight space, in public
    /// order: the column of a right state within its product-space row.
    pub(super) rank: Vec<u32>,
}

impl Ladders {
    /// `L(λ)` in the basis of its root system (§3): Gelfand–Tsetlin for
    /// `A_r`, the contravariant Gram–Schmidt basis otherwise.
    pub(super) fn new(data: &CartanData, lambda: &[i64]) -> Result<Self, LieError> {
        if matches!(data.root, RootSystem::A(_)) {
            let (module, public) = super::gt::module(data, lambda)?;
            return Ok(Self::from_module(module, public));
        }
        let module = IrrepModule::new(data, lambda)?;
        // The module dimension is at most `u32::MAX` (§7 gate 1).
        let public = (0..module.dim as u32).collect();
        Ok(Self::from_module(module, public))
    }

    /// Flattens `module`'s ladders; `public` is the magnetic index of each
    /// engine index.
    ///
    /// Every rational is compacted here, the one point where a module and its
    /// ladders become cache-owned, so that [`Ladders::bytes`] bounds them.
    pub(super) fn from_module(mut module: IrrepModule, public: Vec<u32>) -> Self {
        module.compact();
        let rank = module.weights[0].mu.len();
        let mut norm = Vec::with_capacity(module.dim);
        let mut space = Vec::with_capacity(module.dim);
        for (w, s) in module.weights.iter().enumerate() {
            norm.extend(s.norms.iter().cloned());
            space.extend(std::iter::repeat_n(w, s.norms.len()));
        }
        // Ranks are read from `public`, so the row formula of `PairRows`
        // does not rely on how a basis orders a weight space internally.
        let mut by_rank = vec![0u32; module.dim];
        let mut members = Vec::new();
        for s in &module.weights {
            members.clear();
            members.extend(s.offset..s.offset + s.norms.len());
            members.sort_unstable_by_key(|&m| public[m]);
            for (r, &m) in members.iter().enumerate() {
                // A weight multiplicity is at most the dimension, a `u32`.
                by_rank[m] = r as u32;
            }
        }
        let mut lower = vec![vec![Vec::new(); module.dim]; rank];
        let mut raise = vec![vec![Vec::new(); module.dim]; rank];
        for s in &module.weights {
            for (j, slot) in s.lower.iter().enumerate() {
                let Some((src, f)) = slot else { continue };
                let src = &module.weights[*src];
                for (a, row) in f.iter().enumerate() {
                    for (b, x) in row.iter().enumerate() {
                        if x.is_zero() {
                            continue;
                        }
                        let (ma, mb) = (s.offset + a, src.offset + b);
                        lower[j][mb].push((ma, x.clone()));
                        raise[j][ma].push((mb, x * &s.norms[a] / &src.norms[b]));
                    }
                }
            }
        }
        for x in raise.iter_mut().flatten().flatten().map(|(_, x)| x) {
            x.compact();
        }
        Self {
            module,
            norm,
            space,
            lower,
            raise,
            public,
            rank: by_rank,
        }
    }

    pub(super) fn dim(&self) -> usize {
        self.module.dim
    }

    /// Estimated retained bytes, including the module and both ladder sets.
    pub(crate) fn bytes(&self) -> usize {
        let ops = |ops: &Vec<Op>| -> usize {
            ops.iter()
                .map(|op| {
                    op.capacity() * std::mem::size_of::<Vec<(usize, Q)>>()
                        + op.iter()
                            .map(|row| {
                                row.capacity() * std::mem::size_of::<(usize, Q)>()
                                    + row.iter().map(|(_, x)| q_heap(x)).sum::<usize>()
                            })
                            .sum::<usize>()
                })
                .sum()
        };
        self.module.bytes()
            + qs_bytes(&self.norm)
            + self.space.capacity() * std::mem::size_of::<usize>()
            + ops(&self.lower)
            + ops(&self.raise)
            + (self.public.capacity() + self.rank.capacity()) * std::mem::size_of::<u32>()
    }

    /// The weight of engine index `m`.
    fn weight(&self, m: usize) -> &[i64] {
        &self.module.weights[self.space[m]].mu
    }

    /// The engine index of each public magnetic index.
    fn engine(&self) -> Vec<usize> {
        let mut out = vec![0; self.dim()];
        for (m, &p) in self.public.iter().enumerate() {
            out[p as usize] = m;
        }
        out
    }

    /// `⟨u_m, f_j u_src⟩` rounded once, for an `(m, x)` entry of `lower[j][src]`.
    fn rounded_lower(&self, src: usize, m: usize, x: &Q) -> f64 {
        super::rational::rounded_radical(x, &self.norm[m], &Q::from(1), &self.norm[src])
    }

    /// `sign ⟨low| L^h |high⟩` with `L = Σ_i f_i` (§5.3). Positive rescalings
    /// keep the sign, so the rational basis `b` gives it exactly.
    fn principal_sign(&self, lambda: &[i64]) -> Result<i8, LieError> {
        let mut v = vec![Q::from(0); self.dim()];
        v[0] = Q::from(1);
        // The lowest weight is the last weight space, at depth h = ⟨λ, 2ρ∨⟩.
        let h: i64 = self.module.weights.last().map_or(0, |s| s.k.iter().sum());
        for _ in 0..h {
            let mut next = vec![Q::from(0); self.dim()];
            for op in &self.lower {
                for (m, x) in v.iter().enumerate() {
                    if x.is_zero() {
                        continue;
                    }
                    for (m2, y) in &op[m] {
                        next[*m2].add_mul(x, y);
                    }
                }
            }
            v = next;
        }
        match v.last() {
            Some(x) if x.signum().is_gt() => Ok(1),
            Some(x) if x.signum().is_lt() => Ok(-1),
            _ => Err(CgcError::ZeroPrincipalLadder {
                lambda: lambda.to_vec(),
            }
            .into()),
        }
    }
}

/// The coupling pairs `(m_a, m_b)` of one weight at engine indices, ordered
/// by their public indices with the left factor slow ("a-slow", §4.1).
#[cfg_attr(test, derive(Clone))]
pub(super) struct ProductSpace {
    pub(super) pairs: Vec<(usize, usize)>,
}

impl ProductSpace {
    fn new(a: &Ladders, b: &Ladders, nu: &[i64]) -> Self {
        let mut pairs = Vec::new();
        // Scan the smaller weight set; the order is restored below either way.
        let (inner, outer, swapped) = if a.module.weights.len() < b.module.weights.len() {
            (b, a, true)
        } else {
            (a, b, false)
        };
        for wb in &outer.module.weights {
            let mu_a: Vec<i64> = nu.iter().zip(&wb.mu).map(|(x, y)| x - y).collect();
            let Some(wa) = inner.module.weight_index(&mu_a) else {
                continue;
            };
            let wa = &inner.module.weights[wa];
            for q in wb.offset..wb.offset + wb.norms.len() {
                pairs.extend((wa.offset..wa.offset + wa.norms.len()).map(|p| {
                    if swapped {
                        (q, p)
                    } else {
                        (p, q)
                    }
                }));
            }
        }
        pairs.sort_unstable_by_key(|&(p, q)| (a.public[p], b.public[q]));
        Self { pairs }
    }

    fn len(&self) -> usize {
        self.pairs.len()
    }
}

/// Row starts of the left states, reused for every target product space of
/// one channel: `d_a` slots, of which a target writes only its own rows.
///
/// For a target weight `w` and left state `p`, the right states of `P_w` are
/// the whole weight space of `w - wt(p)`, one contiguous run of the a-slow
/// order sorted by public index. So `(p, q)` sits at `start[p] + rank_b[q]`.
struct RowScratch(Vec<usize>);

impl RowScratch {
    fn new(a: &Ladders) -> Self {
        Self(vec![0; a.dim()])
    }

    /// The row index of `to`, writing one slot per left state of `to`.
    /// Slots of other left states keep stale values; [`PairRows::at`]
    /// rejects them by pair equality, so no clearing is needed.
    fn rows<'s>(&'s mut self, to: &'s ProductSpace, b: &'s Ladders) -> PairRows<'s> {
        let mut last = None;
        for (t, &(p, _)) in to.pairs.iter().enumerate() {
            if last != Some(p) {
                self.0[p] = t;
                last = Some(p);
                #[cfg(test)]
                ROW_WRITES.with(|w| w.set(w.get() + 1));
            }
        }
        PairRows {
            start: &self.0,
            pairs: &to.pairs,
            rank: &b.rank,
        }
    }
}

/// [`ProductSpace`] positions by row arithmetic, for one bound target.
struct PairRows<'s> {
    start: &'s [usize],
    pairs: &'s [(usize, usize)],
    rank: &'s [u32],
}

impl PairRows<'_> {
    fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Row of `pq`. Every image of a weight vector under a ladder operator
    /// has the target weight, so the pair is always there; an impossible
    /// image fails loudly instead of landing on a stale row.
    fn at(&self, pq: (usize, usize)) -> usize {
        let t = self.start[pq.0] + self.rank[pq.1] as usize;
        assert!(
            self.pairs.get(t) == Some(&pq),
            "ladder image {pq:?} is not a coupling pair of the target weight (§4.1)"
        );
        t
    }
}

/// `Δ(x) = x ⊗ 1 + 1 ⊗ x` applied to `v` on `from`, landing on `to`.
fn coproduct(op_a: &Op, op_b: &Op, from: &ProductSpace, to: &PairRows, v: &[Q]) -> Vec<Q> {
    let mut out = vec![Q::from(0); to.len()];
    for (&(p, q), x) in from.pairs.iter().zip(v) {
        if x.is_zero() {
            continue;
        }
        for (p2, y) in &op_a[p] {
            out[to.at((*p2, q))].add_mul(x, y);
        }
        for (q2, y) in &op_b[q] {
            out[to.at((p, *q2))].add_mul(x, y);
        }
    }
    out
}

/// A completed coupled state: `N_μ`, its engine index `m_c`, its product
/// space and `√N_μ Φ_μ(b^c_{m_c})` per `μ`.
type Emit<'e> = dyn FnMut(&[Q], usize, &ProductSpace, &[Vec<Q>]) -> Result<(), LieError> + 'e;

/// The highest-weight states of `a ⊗ b → c` (§4.2–§4.3): `P`, `√N_μ Φ_μ(v_λ)`
/// per `μ` and `N_μ`.
struct Seeds {
    p: ProductSpace,
    seeds: Vec<Vec<Q>>,
    norms: Vec<Q>,
}

/// §4.2–§4.3 for both descents: the pivots, and the seeds unless the
/// multiplicity is zero.
fn highest(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    c: &Ladders,
    multiplicity: usize,
    scratch: &mut RowScratch,
) -> Result<(Vec<usize>, Option<Seeds>), LieError> {
    let top = &c.module.weights[0].mu;
    let p = ProductSpace::new(a, b, top);

    // §4.2: H is the nullspace of the stacked Δ(e_i) on span(P).
    let rows = raising_rows(data, a, b, &p, top, scratch);
    let (pivots, echelon) = echelon_nullspace(rows, p.len());
    if echelon.len() != multiplicity {
        return Err(CgcError::NullityMismatch {
            a: a.module.weights[0].mu.clone(),
            b: b.module.weights[0].mu.clone(),
            c: top.clone(),
            nullity: echelon.len(),
            multiplicity,
        }
        .into());
    }
    if multiplicity == 0 {
        // Gate 2 proved the channel absent; there is nothing to lower.
        return Ok((pivots, None));
    }

    // §4.3: Gram–Schmidt of E_1, …, E_N in pivot order, metric n^a n^b.
    let metric: Vec<Q> = p
        .pairs
        .iter()
        .map(|&(x, y)| &a.norm[x] * &b.norm[y])
        .collect();
    let dot = |x: &[Q], y: &[Q]| -> Q {
        x.iter()
            .zip(y)
            .zip(&metric)
            .filter(|((u, v), _)| !u.is_zero() && !v.is_zero())
            .map(|((u, v), g)| u * v * g)
            .fold(Q::from(0), |acc, x| acc + x)
    };
    let mut seeds: Vec<Vec<Q>> = Vec::with_capacity(multiplicity);
    let mut norms: Vec<Q> = Vec::with_capacity(multiplicity);
    for e in echelon {
        let mut v = e.clone();
        for (s, n) in seeds.iter().zip(&norms) {
            let scale = dot(&e, s) / n;
            if scale.is_zero() {
                continue;
            }
            for (x, y) in v.iter_mut().zip(s) {
                x.sub_mul(&scale, y);
            }
        }
        norms.push(dot(&v, &v));
        seeds.push(v);
    }
    Ok((pivots, Some(Seeds { p, seeds, norms })))
}

/// The source weight space of the kept candidate `(j, ·)` of `space`.
fn source(space: &super::module::WeightSpace, j: usize) -> usize {
    space.lower[j]
        .as_ref()
        .expect("a kept candidate's source weight space exists (§4.1)")
        .0
}

/// The kept candidates reading each state of `c`, and each weight space.
fn readers(c: &Ladders) -> (Vec<usize>, Vec<usize>) {
    let weights = &c.module.weights;
    let mut readers = vec![0usize; c.dim()];
    let mut space_readers = vec![0usize; weights.len()];
    for space in &weights[1..] {
        for &(j, q) in &space.kept {
            let src = source(space, j);
            readers[weights[src].offset + q] += 1;
            space_readers[src] += 1;
        }
    }
    (readers, space_readers)
}

/// How often each kept candidate of a weight space enters its recipe.
fn recipe_uses(space: &super::module::WeightSpace) -> Vec<usize> {
    let mut uses = vec![0usize; space.kept.len()];
    for coeffs in &space.recipe {
        for (t, r) in coeffs.iter().enumerate() {
            if !r.is_zero() {
                uses[t] += 1;
            }
        }
    }
    uses
}

/// The candidate a recipe row `b_k = c_t` takes as it is, when only that row
/// uses it: the general sum would rebuild the same exact vector.
fn moved_candidate(coeffs: &[Q], uses: &[usize]) -> Option<usize> {
    let mut terms = coeffs.iter().enumerate().filter(|(_, r)| !r.is_zero());
    match (terms.next(), terms.next()) {
        (Some((t, r)), None) if r.is_one() && uses[t] == 1 => Some(t),
        _ => None,
    }
}

/// `Φ(b_k) = Σ_t recipe[k][t] Δ(f_{j_t}) Φ(b_{q_t})` for one `μ`, from the
/// lowered candidates `cand(t)` of the weight space, over `len` pairs.
fn recipe_row<'v>(coeffs: &[Q], cand: impl Fn(usize) -> &'v [Q], len: usize) -> Vec<Q> {
    let mut v = vec![Q::from(0); len];
    for (t, r) in coeffs.iter().enumerate() {
        if r.is_zero() {
            continue;
        }
        let unit = r.is_one();
        for (x, y) in v.iter_mut().zip(cand(t)) {
            if y.is_zero() {
                continue;
            }
            if unit {
                *x += y;
            } else {
                x.add_mul(r, y);
            }
        }
    }
    v
}

/// The rational CGC of `a ⊗ b → c` (§4.2–§4.4), streamed: every coupled
/// state is passed to `emit` once, in engine order, as soon as it completes.
///
/// A state at depth `ℓ` is lowered only from states at depth `ℓ − 1`, and a
/// recipe may mix several of them, so a rounded column cannot replace an
/// exact parent. Each exact vector is therefore kept exactly until its last
/// kept candidate has been lowered, and a weight's product space until its
/// last reader: the live exact set is at most two depth layers plus one
/// weight's candidates (`docs/gauge_cartan.md` §9). Returns the pivots of §4.3.
fn descend(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    c: &Ladders,
    multiplicity: usize,
    emit: &mut Emit<'_>,
) -> Result<Vec<usize>, LieError> {
    let mut scratch = RowScratch::new(a);
    let (pivots, seeded) = highest(data, a, b, c, multiplicity, &mut scratch)?;
    let Some(Seeds {
        p,
        seeds,
        norms: seed_norms,
    }) = seeded
    else {
        return Ok(pivots);
    };

    let weights = &c.module.weights;
    let (mut readers, mut space_readers) = readers(c);
    let mut phi: Vec<Vec<Vec<Q>>> = vec![Vec::new(); c.dim()];
    let mut spaces: Vec<Option<ProductSpace>> = (0..weights.len()).map(|_| None).collect();
    // Emits a completed state and keeps it only while a reader remains.
    let mut complete = |phi: &mut Vec<Vec<Vec<Q>>>,
                        readers: &[usize],
                        m: usize,
                        to: &ProductSpace,
                        v: Vec<Vec<Q>>| {
        emit(&seed_norms, m, to, &v)?;
        if readers[m] > 0 {
            phi[m] = v;
        } else {
            track(&v, false);
        }
        Ok::<_, LieError>(())
    };

    // The highest weight space is the single state `m_c = 0`.
    track(&seeds, true);
    complete(&mut phi, &readers, 0, &p, seeds)?;
    if space_readers[0] > 0 {
        spaces[0] = Some(p);
    }

    // §4.4: replay c's basis recipe through Δ(f_j).
    for (w, space) in weights.iter().enumerate().skip(1) {
        let to = ProductSpace::new(a, b, &space.mu);
        let rows = scratch.rows(&to, b);
        let mut lowered: Vec<Vec<Vec<Q>>> = Vec::with_capacity(space.kept.len());
        for &(j, q) in &space.kept {
            let src = source(space, j);
            let m = weights[src].offset + q;
            let from = spaces[src]
                .as_ref()
                .expect("a product space outlives its last reader");
            let cand: Vec<Vec<Q>> = phi[m]
                .iter()
                .map(|v| coproduct(&a.lower[j], &b.lower[j], from, &rows, v))
                .collect();
            track(&cand, true);
            lowered.push(cand);
            readers[m] -= 1;
            if readers[m] == 0 {
                track(&phi[m], false);
                phi[m] = Vec::new();
            }
            space_readers[src] -= 1;
            if space_readers[src] == 0 {
                spaces[src] = None;
            }
        }
        let uses = recipe_uses(space);
        for (k, coeffs) in space.recipe.iter().enumerate() {
            let m = space.offset + k;
            if let Some(t) = moved_candidate(coeffs, &uses) {
                complete(&mut phi, &readers, m, &to, std::mem::take(&mut lowered[t]))?;
                continue;
            }
            let per_mu: Vec<Vec<Q>> = (0..multiplicity)
                .map(|mu| recipe_row(coeffs, |t| &lowered[t][mu], to.len()))
                .collect();
            track(&per_mu, true);
            complete(&mut phi, &readers, m, &to, per_mu)?;
        }
        for cand in &lowered {
            track(cand, false);
        }
        if space_readers[w] > 0 {
            spaces[w] = Some(to);
        }
    }
    Ok(pivots)
}

/// Stores a completed column in its slot, which each column fills once: the
/// rounded payload is moved, never copied.
fn place(slot: &mut Vec<MEntry>, column: Vec<MEntry>) {
    debug_assert!(slot.is_empty(), "a column is written once");
    #[cfg(test)]
    COLUMNS.with(|c| {
        let [writes, entries] = c.get();
        c.set([writes + 1, entries + column.len()]);
    });
    *slot = column;
}

/// The rounded column `(μ, m_c)` of the state `v = √N_μ Φ_μ(b^c_{m_c})` over
/// `space`, with `scale = N_μ n^c_{m_c}`: one entry per exactly nonzero
/// coordinate, a-slow.
fn round_column(
    a: &Ladders,
    b: &Ladders,
    mc: u32,
    mu: usize,
    scale: &Q,
    space: &ProductSpace,
    v: &[Q],
) -> Result<Vec<MEntry>, LieError> {
    let mut column = Vec::with_capacity(v.iter().filter(|x| !x.is_zero()).count());
    let mu = magnetic_index(mu)?;
    for (&(pa, pb), x) in space.pairs.iter().zip(v) {
        if x.is_zero() {
            continue;
        }
        column.push(MEntry {
            m1: a.public[pa],
            m2: b.public[pb],
            m3: mc,
            mu,
            value: super::rational::rounded_radical(x, &a.norm[pa], &b.norm[pb], scale),
        });
    }
    Ok(column)
}

/// Per-participant row starts of [`descend_layers`]' phase A, bound to the
/// last target weight it lowered into.
struct BoundRows {
    rows: RowScratch,
    bound: Option<usize>,
}

/// The rounded columns of [`descend`], one depth layer at a time, on a team
/// token (`docs/gauge_cartan.md` §9).
///
/// A layer runs as two stages of [`crate::stage::run_stage`]: phase A lowers
/// every kept candidate × `μ` of the layer, phase B combines every recipe row
/// × `μ` and rounds its column. Every item computes exactly the operations
/// [`descend`] does for it, so the entries are bit-identical at every width.
/// The live exact set is at most two layers plus one layer's candidates; the
/// leader frees a parent after phase A, when all its readers have run.
fn descend_layers(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    c: &Ladders,
    n: usize,
    exec: &mut Exec<'_>,
) -> Result<Vec<Vec<MEntry>>, LieError> {
    use crate::stage::{run_stage, StageBuf, SystemClock};
    let dc = c.dim();
    let mut columns: Vec<Vec<MEntry>> = vec![Vec::new(); n * dc];
    let mut scratch = RowScratch::new(a);
    let Some(Seeds {
        p,
        seeds,
        norms: seed_norms,
    }) = highest(data, a, b, c, n, &mut scratch)?.1
    else {
        return Ok(columns);
    };
    // The leader's phase-A rows reuse the raising rows' scratch.
    let mut leader = Some(BoundRows {
        rows: scratch,
        bound: None,
    });
    let weights = &c.module.weights;
    let (mut readers, mut space_readers) = readers(c);
    let mut phi: Vec<Vec<Vec<Q>>> = vec![Vec::new(); c.dim()];
    let mut spaces: Vec<Option<ProductSpace>> = (0..weights.len()).map(|_| None).collect();
    let scale = |mu: usize, m: usize| &seed_norms[mu] * &c.norm[m];

    track(&seeds, true);
    for (mu, v) in seeds.iter().enumerate() {
        place(
            &mut columns[mu * dc + c.public[0] as usize],
            round_column(a, b, c.public[0], mu, &scale(mu, 0), &p, v)?,
        );
    }
    if readers[0] > 0 {
        phi[0] = seeds;
    } else {
        track(&seeds, false);
    }
    if space_readers[0] > 0 {
        spaces[0] = Some(p);
    }

    // Per-layer tables and stage buffers, refilled per layer, so a layer
    // whose items all run on the leader allocates only what `descend` does.
    let mut tos: Vec<ProductSpace> = Vec::new();
    let mut lowered: Vec<Vec<Vec<Q>>> = Vec::new();
    let mut uses: Vec<Vec<usize>> = Vec::new();
    let mut lowering = StageBuf::default();
    let mut combining = StageBuf::default();
    let mut cands: Vec<(usize, usize)> = Vec::new();
    let mut rows: Vec<(usize, usize)> = Vec::new();
    let mut base: Vec<usize> = Vec::new();
    let mut units: Vec<u64> = Vec::new();
    let depth = |w: usize| weights[w].k.iter().sum::<i64>();
    let mut lo = 1;
    while lo < weights.len() {
        let hi = lo
            + weights[lo..]
                .iter()
                .take_while(|s| s.k.iter().sum::<i64>() == depth(lo))
                .count();
        let layer = &weights[lo..hi];
        tos.extend(layer.iter().map(|s| ProductSpace::new(a, b, &s.mu)));

        // Phase A: item `i` lowers candidate `i / n` for `μ = i % n`.
        cands.clear();
        cands.extend(
            layer
                .iter()
                .enumerate()
                .flat_map(|(w, s)| (0..s.kept.len()).map(move |t| (w, t))),
        );
        units.clear();
        units.extend(
            cands
                .iter()
                .flat_map(|&(w, _)| std::iter::repeat_n(tos[w].len() as u64, n)),
        );
        // Layer-relative targets: the leader's binding is stale.
        if let Some(l) = &mut leader {
            l.bound = None;
        }
        run_stage(
            exec,
            &units,
            &SystemClock,
            &mut leader,
            &mut lowering,
            || BoundRows {
                rows: RowScratch::new(a),
                bound: None,
            },
            |s, i| {
                let (w, t) = cands[i / n];
                let space = &layer[w];
                let (j, q) = space.kept[t];
                let src = source(space, j);
                let from = spaces[src]
                    .as_ref()
                    .expect("a product space outlives its last reader");
                let rows = if s.bound == Some(w) {
                    PairRows {
                        start: &s.rows.0,
                        pairs: &tos[w].pairs,
                        rank: &b.rank,
                    }
                } else {
                    s.bound = Some(w);
                    s.rows.rows(&tos[w], b)
                };
                let v = &phi[weights[src].offset + q][i % n];
                coproduct(&a.lower[j], &b.lower[j], from, &rows, v)
            },
        );
        // Every candidate of the layer is live once the stage returns,
        // before any parent is released.
        track(lowering.results(), true);
        let mut flat = lowering.drain();
        for _ in &cands {
            lowered.push(flat.by_ref().take(n).collect());
        }
        for &(w, t) in &cands {
            let (j, q) = layer[w].kept[t];
            let src = source(&layer[w], j);
            let m = weights[src].offset + q;
            readers[m] -= 1;
            if readers[m] == 0 {
                track(&phi[m], false);
                phi[m] = Vec::new();
            }
            space_readers[src] -= 1;
            if space_readers[src] == 0 {
                spaces[src] = None;
            }
        }

        // Phase B: item `i` is recipe row `i / n` for `μ = i % n`.
        drop(flat);
        base.clear();
        rows.clear();
        let mut start = 0;
        for (w, s) in layer.iter().enumerate() {
            base.push(start);
            start += s.kept.len();
            uses.push(recipe_uses(s));
            rows.extend((0..s.recipe.len()).map(|k| (w, k)));
        }
        units.clear();
        units.extend(
            rows.iter()
                .flat_map(|&(w, _)| std::iter::repeat_n(tos[w].len() as u64, n)),
        );
        let _ran = run_stage(
            exec,
            &units,
            &SystemClock,
            &mut None,
            &mut combining,
            || (),
            |(), i| {
                let (w, k) = rows[i / n];
                let mu = i % n;
                let space = &layer[w];
                let m = space.offset + k;
                let coeffs = &space.recipe[k];
                let cand = |t: usize| -> &[Q] { &lowered[base[w] + t][mu] };
                let (v, kept) = match moved_candidate(coeffs, &uses[w]) {
                    Some(t) => (None, cand(t)),
                    None => {
                        let v = recipe_row(coeffs, cand, tos[w].len());
                        (Some(v), &[][..])
                    }
                };
                let column = round_column(
                    a,
                    b,
                    c.public[m],
                    mu,
                    &scale(mu, m),
                    &tos[w],
                    v.as_deref().unwrap_or(kept),
                )?;
                // A state no candidate reads dies with its share.
                let transient = v.as_ref().filter(|_| readers[m] == 0).map_or(0, Vec::len);
                Ok::<_, LieError>((column, v.filter(|_| readers[m] > 0), transient))
            },
        );
        #[cfg(test)]
        let mut transient = Vec::with_capacity(units.len());
        let mut done = combining.drain();
        for &(w, k) in &rows {
            let m = layer[w].offset + k;
            let mut states = Vec::with_capacity(if readers[m] > 0 { n } else { 0 });
            for mu in 0..n {
                let (column, state, _t) = done.next().expect("one result per item")?;
                #[cfg(test)]
                transient.push(_t);
                place(&mut columns[mu * dc + c.public[m] as usize], column);
                if readers[m] == 0 {
                    continue;
                }
                states.push(match state {
                    Some(v) => {
                        track(std::slice::from_ref(&v), true);
                        v
                    }
                    None => {
                        let t = moved_candidate(&layer[w].recipe[k], &uses[w])
                            .expect("a row without a state moves its candidate");
                        std::mem::take(&mut lowered[base[w] + t][mu])
                    }
                });
            }
            if readers[m] > 0 {
                phi[m] = states;
            }
        }
        #[cfg(test)]
        {
            // Every participant holds at most one dropped state at a time.
            transient.sort_unstable_by(|x, y| y.cmp(x));
            let peak: usize = transient.iter().take(_ran.width.get()).sum();
            LIVE_EXACT.with(|cell| {
                let [now, max] = cell.get();
                cell.set([now, max.max(now + peak)]);
            });
        }
        for cand in &lowered {
            track(cand, false);
        }
        lowered.clear();
        uses.clear();
        for (w, to) in (lo..hi).zip(tos.drain(..)) {
            if space_readers[w] > 0 {
                spaces[w] = Some(to);
            }
        }
        lo = hi;
    }
    Ok(columns)
}

/// Test accounting of the live exact slots of [`descend`] (current, maximum).
#[cfg_attr(not(test), allow(unused_variables))]
fn track<'v>(v: impl IntoIterator<Item = &'v Vec<Q>>, live: bool) {
    #[cfg(test)]
    LIVE_EXACT.with(|cell| {
        let n: usize = v.into_iter().map(Vec::len).sum();
        let [now, max] = cell.get();
        let now = if live { now + n } else { now - n };
        cell.set([now, max.max(now)]);
    });
}

/// The entries of `a ⊗ b → c` with multiplicity `n`, rounded once as each
/// coupled state completes, before any gate: per column `(μ, m_c)`, then
/// concatenated in §4.7 order, `μ`, then public `m_c`, then `(m_a, m_b)`
/// a-slow (§4.1). The entry `x·√(n^a n^b/(N_μ n^c))` of an exactly nonzero
/// coordinate `x` is kept even when it rounds to zero.
///
/// This is the descent's split point: a token that carries a team of two or
/// more descends layer by layer ([`descend_layers`]); any other token, and
/// `None`, runs [`descend`] on the calling thread.
pub(super) fn rounded_entries(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    c: &Ladders,
    n: usize,
    exec: Option<&mut Exec<'_>>,
) -> Result<Vec<MEntry>, LieError> {
    let columns = match exec {
        Some(exec) if exec.carries_team() && !exec.is_serial() => {
            descend_layers(data, a, b, c, n, exec)?
        }
        _ => {
            let dc = c.dim();
            let mut columns: Vec<Vec<MEntry>> = vec![Vec::new(); n * dc];
            descend(data, a, b, c, n, &mut |seed_norms, m, space, per_mu| {
                let mc = c.public[m];
                for (mu, v) in per_mu.iter().enumerate() {
                    let scale = &seed_norms[mu] * &c.norm[m];
                    place(
                        &mut columns[mu * dc + mc as usize],
                        round_column(a, b, mc, mu, &scale, space, v)?,
                    );
                }
                Ok(())
            })?;
            columns
        }
    };
    // The flattening copy briefly doubles the rounded payload (24 B per
    // entry, counted in §9); reordering in place would avoid it.
    let mut entries = Vec::with_capacity(columns.iter().map(Vec::len).sum());
    for column in columns {
        entries.extend(column);
    }
    Ok(entries)
}

/// The exact channel of [`descend`] retained whole, for the exact oracles.
#[cfg(test)]
pub(super) struct ExactChannel {
    pub(super) mult: usize,
    /// The pivot rows `p_1 < … < p_N` of §4.3, as indices into `P`.
    pub(super) pivots: Vec<usize>,
    /// `N_μ = ⟨B_μ, B_μ⟩`.
    pub(super) seed_norms: Vec<Q>,
    /// The product space of each weight space of `c`; `spaces[0]` is `P`.
    pub(super) spaces: Vec<ProductSpace>,
    /// `phi[m_c][μ]`: coordinates of `√N_μ Φ_μ(b^c_{m_c})` over its space.
    pub(super) phi: Vec<Vec<Vec<Q>>>,
}

#[cfg(test)]
impl ExactChannel {
    pub(super) fn new(
        data: &CartanData,
        a: &Ladders,
        b: &Ladders,
        c: &Ladders,
        multiplicity: usize,
    ) -> Result<Self, LieError> {
        let mut seed_norms = Vec::new();
        let mut spaces: Vec<Option<ProductSpace>> =
            (0..c.module.weights.len()).map(|_| None).collect();
        let mut phi = vec![Vec::new(); c.dim()];
        let pivots = descend(
            data,
            a,
            b,
            c,
            multiplicity,
            &mut |norms, m, space, per_mu| {
                seed_norms = norms.to_vec();
                spaces[c.space[m]].get_or_insert_with(|| space.clone());
                phi[m] = per_mu.to_vec();
                Ok(())
            },
        )?;
        Ok(Self {
            mult: multiplicity,
            pivots,
            seed_norms,
            spaces: spaces.into_iter().flatten().collect(),
            phi,
        })
    }

    /// The exactly nonzero entries `(m_a, m_b, m_c, μ, C)` at public magnetic
    /// indices, in the §4.7 order of [`rounded_entries`], with the exact
    /// `C = x·√(n^a n^b/(N_μ n^c))`.
    pub(super) fn exact_entries(
        &self,
        a: &Ladders,
        b: &Ladders,
        c: &Ladders,
    ) -> Vec<(usize, usize, usize, usize, SignedSqrtRational)> {
        let mut out = Vec::new();
        for mu in 0..self.mult {
            for mc in c.engine() {
                let space = &self.spaces[c.space[mc]];
                let scale = &self.seed_norms[mu] * &c.norm[mc];
                for (&(pa, pb), x) in space.pairs.iter().zip(&self.phi[mc][mu]) {
                    if x.is_zero() {
                        continue;
                    }
                    out.push((
                        a.public[pa] as usize,
                        b.public[pb] as usize,
                        c.public[mc] as usize,
                        mu,
                        super::rational::radical(x, &a.norm[pa], &b.norm[pb], &scale),
                    ));
                }
            }
        }
        out
    }
}

/// A rounded entry in one output weight block: `((m_a, m_b), column, value)`.
type BlockEntry = ((u32, u32), usize, f64);

/// Verification tolerance of the isometry and ladder gates (not value-fixing).
const TOL_ROUNDED: f64 = 1e-9;

/// The largest `|x|` over `residuals`, with a nonfinite residual a failure of
/// `gate` (a float `max` would hide NaN).
fn worst(gate: &'static str, residuals: impl Iterator<Item = f64>) -> Result<(), GateError> {
    let mut worst = 0.0f64;
    for r in residuals {
        if !r.is_finite() {
            return Err(GateError::NonFinite { gate });
        }
        worst = worst.max(r.abs());
    }
    if worst > TOL_ROUNDED {
        return Err(GateError::Residual {
            gate,
            residual: worst,
        });
    }
    Ok(())
}

/// Element counts the isometry gate requests for its three buffers of one
/// weight block: the gathered block, its column order and its Gram matrix.
type GateCounts = [usize; 3];

/// Element sizes of the [`GateCounts`] buffers.
const GATE_SIZES: [usize; 3] = [
    std::mem::size_of::<BlockEntry>(),
    std::mem::size_of::<(usize, usize, usize)>(),
    std::mem::size_of::<f64>(),
];

/// Requested bytes of buffers with element counts `c`. Allocator slack is
/// outside the account, as in the cache resource contract.
fn gate_bytes(c: GateCounts) -> usize {
    c.iter()
        .zip(GATE_SIZES)
        .fold(0usize, |t, (&n, s)| t.saturating_add(n.saturating_mul(s)))
}

/// The buffers a block needing `need` replaces when `held` are requested and
/// the participant's charge is `high`, already raised to `gate_bytes(need)`.
///
/// While the componentwise maxima fit the charge, only a buffer that must
/// grow is replaced. Otherwise every buffer of another size is replaced
/// exactly, so the held counts become `need`. Either way the held bytes stay
/// within `high`, also mid-replacement, since every replaced buffer is
/// dropped before any is allocated.
fn gate_replaced(held: GateCounts, need: GateCounts, high: usize) -> [bool; 3] {
    let kept = std::array::from_fn(|c| held[c].max(need[c]));
    let fits = gate_bytes(kept) <= high;
    std::array::from_fn(|c| need[c] > held[c] || (!fits && held[c] != need[c]))
}

/// One participant's isometry-gate buffers, with their requested element
/// counts and its high-water charge: the largest `gate_bytes` of a block it
/// has run. Its live scratch never exceeds that charge, so participants with
/// disjoint blocks hold at most the sum of their largest blocks.
#[derive(Default)]
struct GateScratch {
    block: Vec<BlockEntry>,
    order: Vec<(usize, usize, usize)>,
    gram: Vec<f64>,
    held: GateCounts,
    high: usize,
}

impl GateScratch {
    /// Makes room for a block needing `need`. Accounting uses the counts
    /// requested here, never `capacity()`; every fill stays within them, so
    /// no fill requests memory.
    fn reserve(&mut self, need: GateCounts) {
        self.high = self.high.max(gate_bytes(need));
        let replace = gate_replaced(self.held, need, self.high);
        // Release first: an assignment would hold the old and new buffer.
        if replace[0] {
            self.block = Vec::new();
        }
        if replace[1] {
            self.order = Vec::new();
        }
        if replace[2] {
            self.gram = Vec::new();
        }
        if replace[0] {
            self.block = Vec::with_capacity(need[0]);
        }
        if replace[1] {
            self.order = Vec::with_capacity(need[1]);
        }
        if replace[2] {
            self.gram = Vec::with_capacity(need[2]);
        }
        for c in 0..3 {
            if replace[c] {
                self.held[c] = need[c];
            }
        }
    }
}

/// The isometry block of one output weight: `k` columns from column run
/// `offset`, holding `e` entries.
#[derive(Clone, Copy)]
struct GateBlock {
    offset: usize,
    k: usize,
    e: usize,
}

impl GateBlock {
    fn need(self) -> GateCounts {
        [self.e, self.k, self.k * self.k]
    }
}

/// `max |MᵀM − I| ≤ 1e-9` over one weight block whose column runs are
/// `cols`, with its buffers in `s`.
fn isometry_block(
    s: &mut GateScratch,
    entries: &[MEntry],
    cols: &[(usize, usize)],
    w: GateBlock,
) -> Result<(), GateError> {
    let k = w.k;
    s.reserve(w.need());
    let GateScratch {
        block, order, gram, ..
    } = s;
    // Columns are gathered in their order in `entries`, so the unstable
    // sort below sees the same sequence as a scan of the whole list: a
    // duplicated (row, column) key then sums in the same order.
    order.clear();
    order.extend(
        cols.iter()
            .enumerate()
            .filter(|(_, &(s, t))| t > s)
            .map(|(col, &(s, t))| (s, t, col)),
    );
    order.sort_unstable();
    block.clear();
    for &(s, t, col) in order.iter() {
        block.extend(entries[s..t].iter().map(|e| ((e.m1, e.m2), col, e.value)));
    }
    gram.clear();
    gram.resize(k * k, 0.0);
    // Index a slice, not the scratch's `Vec`: indexing through `&mut Vec`
    // made the accumulation below measurably slower.
    let gram = &mut gram[..];
    block.sort_unstable_by_key(|&(row, col, _)| (row, col));
    for run in block.chunk_by(|x, y| x.0 == y.0) {
        for (i, &(_, ci, vi)) in run.iter().enumerate() {
            for &(_, cj, vj) in &run[i..] {
                gram[ci * k + cj] += vi * vj;
            }
        }
    }
    let gram = &*gram;
    worst(
        "isometry",
        (0..k).flat_map(|i| (i..k).map(move |j| gram[i * k + j] - f64::from(u8::from(i == j)))),
    )
}

/// The isometry gate over every weight block, in weight order. On a team
/// token the blocks are one stage of [`crate::stage::run_bounded_stage`]:
/// live scratch stays within the serial peak plus the CGC tier's budget, the
/// allowance [`crate::cache::CoefficientCacheBudgets`] documents for scratch
/// beyond the serial peak, and the first failing block in weight order is
/// the result at every width.
fn isometry_gate(
    c: &Ladders,
    n: usize,
    entries: &[MEntry],
    runs: &[(usize, usize)],
    exec: Option<&mut Exec<'_>>,
) -> Result<(), GateError> {
    let blocks = c.module.weights.iter().map(|space| {
        let k = space.norms.len() * n;
        let cols = &runs[space.offset * n..space.offset * n + k];
        let e = cols.iter().map(|&(s, t)| t - s).sum();
        GateBlock {
            offset: space.offset * n,
            k,
            e,
        }
    });
    let cols = |w: GateBlock| &runs[w.offset..w.offset + w.k];
    let exec = match exec {
        Some(exec) if exec.carries_team() && !exec.is_serial() => exec,
        _ => {
            let mut s = GateScratch::default();
            for w in blocks {
                isometry_block(&mut s, entries, cols(w), w)?;
            }
            return Ok(());
        }
    };
    use crate::stage::{memory_width, run_bounded_stage, StageBuf, SystemClock};
    let blocks: Vec<GateBlock> = blocks.collect();
    let sigma: Vec<usize> = blocks.iter().map(|&w| gate_bytes(w.need())).collect();
    // Work per block: the gather and its sort, then the Gram matrix.
    let units: Vec<u64> = blocks
        .iter()
        .map(|w| (w.e as u64).saturating_add((w.k as u64).saturating_mul(w.k as u64)))
        .collect();
    let allowance = gate_allowance();
    let ceiling = sigma
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(allowance);
    let memory = |rest: &[usize]| memory_width(&sigma, rest, ceiling);
    let mut buf = StageBuf::default();
    run_bounded_stage(
        exec,
        &units,
        &SystemClock,
        &memory,
        &mut None,
        &mut buf,
        GateScratch::default,
        |s, w| isometry_block(s, entries, cols(blocks[w]), blocks[w]),
    );
    buf.drain().collect()
}

/// The allowance for isometry-gate scratch beyond the serial peak.
fn gate_allowance() -> usize {
    #[cfg(test)]
    if let Some(a) = GATE_ALLOWANCE.with(std::cell::Cell::get) {
        return a;
    }
    crate::cache::cache_budgets().limit(crate::cache::CoefficientCacheTier::Cgc)
}

/// The rounded gates on one channel `a ⊗ b → c` of multiplicity `n`, with
/// `entries` at public magnetic indices:
///
/// - every entry in range, finite and on its weight, `wt(m_a) + wt(m_b) = wt(m_c)`;
/// - isometry, `max |MᵀM − I| ≤ 1e-9` with `M[(m_a, m_b), (m_c, μ)]` over every
///   expected column `(m_c, μ)`, including absent ones. Validated support makes
///   columns of different output weights disjoint, so the weight blocks are
///   the whole condition;
/// - ladder consistency at the highest weight: for each `μ` and each `j` with
///   `λ_c − α_j` a weight, `Δ(f_j) C[·, ·, top, μ] = Σ_m ⟨u_m, f_j u_top⟩
///   C[·, ·, m, μ]` within `1e-9`, with correctly rounded ladder entries.
///
/// Each column `(m_c, μ)` must be one contiguous run of `entries`, as the
/// §4.7 order makes it; a split column is a support failure. The gates then
/// read the entries in place through those runs: one weight block of
/// `BlockEntry` at a time, so no copy of the whole channel is made.
pub(super) fn check_rounded(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    c: &Ladders,
    n: usize,
    entries: &[MEntry],
    exec: Option<&mut Exec<'_>>,
) -> Result<(), GateError> {
    let (ea, eb, ec) = (a.engine(), b.engine(), c.engine());
    // `runs[m_c·n + μ]` (engine `m_c`) is the column's `entries[start..end]`;
    // `end == 0` marks a column not yet seen.
    let mut runs = vec![(0usize, 0usize); c.dim() * n];
    let mut prev = usize::MAX;
    for (i, e) in entries.iter().enumerate() {
        let (m1, m2, m3, mu) = (e.m1 as usize, e.m2 as usize, e.m3 as usize, e.mu as usize);
        if m1 >= a.dim() || m2 >= b.dim() || m3 >= c.dim() || mu >= n {
            return Err(GateError::Support);
        }
        if !e.value.is_finite() {
            return Err(GateError::NonFinite { gate: "isometry" });
        }
        let (pa, pb, pc) = (ea[m1], eb[m2], ec[m3]);
        let on_weight = a
            .weight(pa)
            .iter()
            .zip(b.weight(pb))
            .zip(c.weight(pc))
            .all(|((x, y), z)| x.checked_add(*y) == Some(*z));
        if !on_weight {
            return Err(GateError::Support);
        }
        let col = pc * n + mu;
        if col != prev {
            if runs[col].1 != 0 {
                return Err(GateError::Support);
            }
            runs[col].0 = i;
            prev = col;
        }
        runs[col].1 = i + 1;
    }
    let column = |m: usize, mu: usize| {
        let (s, t) = runs[m * n + mu];
        &entries[s..t]
    };

    isometry_gate(c, n, entries, &runs, exec)?;

    // First descent: f_j u_top lies in the weight space λ_c − α_j, which
    // exists exactly when (λ_c)_j > 0 (an unbroken α_j-string).
    let top = &c.module.weights[0].mu;
    let mut descents: Vec<(usize, Vec<(usize, f64)>)> = Vec::new();
    for j in 0..top.len() {
        if top[j] <= 0 {
            continue;
        }
        let nu = (0..top.len())
            .map(|i| top[i].checked_sub(data.c(j, i)))
            .collect::<Option<Vec<i64>>>();
        let Some(w) = nu.and_then(|nu| c.module.weight_index(&nu)) else {
            return Err(GateError::Support);
        };
        let space = &c.module.weights[w];
        let ladder = (space.offset..space.offset + space.norms.len())
            .map(|m| {
                let x = c.lower[j][0].iter().find(|(t, _)| *t == m);
                (m, x.map_or(0.0, |(_, x)| c.rounded_lower(0, m, x)))
            })
            .collect();
        descents.push((j, ladder));
    }
    for (j, ladder) in &descents {
        for mu in 0..n {
            let mut diff: HashMap<(usize, usize), f64> = HashMap::new();
            for e in column(0, mu) {
                let (pa, pb, v) = (ea[e.m1 as usize], eb[e.m2 as usize], e.value);
                for (pa2, x) in &a.lower[*j][pa] {
                    *diff.entry((*pa2, pb)).or_default() += a.rounded_lower(pa, *pa2, x) * v;
                }
                for (pb2, x) in &b.lower[*j][pb] {
                    *diff.entry((pa, *pb2)).or_default() += b.rounded_lower(pb, *pb2, x) * v;
                }
            }
            for &(m, l) in ladder {
                for e in column(m, mu) {
                    *diff
                        .entry((ea[e.m1 as usize], eb[e.m2 as usize]))
                        .or_default() -= l * e.value;
                }
            }
            worst("ladder", diff.into_values())?;
        }
    }
    Ok(())
}

/// Stacked raising coproducts on `P`, retaining only nonzero coordinates.
fn raising_rows(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    p: &ProductSpace,
    top: &[i64],
    scratch: &mut RowScratch,
) -> Vec<BTreeMap<usize, Q>> {
    let mut rows = Vec::new();
    for i in 0..top.len() {
        let nu: Vec<i64> = (0..top.len()).map(|k| top[k] + data.c(i, k)).collect();
        let target = ProductSpace::new(a, b, &nu);
        if target.pairs.is_empty() {
            continue;
        }
        let target = scratch.rows(&target, b);
        let base = rows.len();
        rows.resize_with(base + target.len(), BTreeMap::new);
        for (col, &(pa, pb)) in p.pairs.iter().enumerate() {
            for (p2, y) in &a.raise[i][pa] {
                *rows[base + target.at((*p2, pb))]
                    .entry(col)
                    .or_insert_with(|| Q::from(0)) += y;
            }
            for (q2, y) in &b.raise[i][pb] {
                *rows[base + target.at((pa, *q2))]
                    .entry(col)
                    .or_insert_with(|| Q::from(0)) += y;
            }
        }
    }
    for row in &mut rows {
        row.retain(|_, x| !x.is_zero());
    }
    rows
}

/// The echelon basis of `ker M` (§4.3): `E_k[p_l] = δ_kl` and `E_k[t] = 0`
/// for `t < p_k`, returned with its pivots `p_1 < … < p_N`.
///
/// `M` is reduced with its columns taken from last to first. Each free column
/// `f` then gives the kernel vector with `1` at `f`, `0` at every other free
/// column, and nonzeros only at pivot columns processed before `f`, which lie
/// after `f`. These vectors are exactly the echelon basis, and the free columns
/// are its pivots, so no second elimination is needed. Sparse updates are exact
/// row operations on the same matrix; uniqueness of RREF (with reversed columns)
/// fixes the same free columns and rational vectors, independent of storage.
fn echelon_nullspace(mut m: Vec<BTreeMap<usize, Q>>, cols: usize) -> (Vec<usize>, Vec<Vec<Q>>) {
    let mut pivot_of_row: Vec<usize> = Vec::new();
    let mut is_pivot = vec![false; cols];
    for col in (0..cols).rev() {
        let r0 = pivot_of_row.len();
        let Some(r) = (r0..m.len()).find(|&r| m[r].contains_key(&col)) else {
            continue;
        };
        m.swap(r0, r);
        let inv = Q::from(1) / &m[r0][&col];
        for x in m[r0].values_mut() {
            *x *= &inv;
        }
        let pivot = m[r0].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r == r0 {
                continue;
            }
            let Some(scale) = row.remove(&col) else {
                continue;
            };
            for (&j, y) in &pivot {
                if j == col {
                    continue;
                }
                let x = row.entry(j).or_insert_with(|| Q::from(0));
                x.sub_mul(&scale, y);
                if x.is_zero() {
                    row.remove(&j);
                }
            }
        }
        pivot_of_row.push(col);
        is_pivot[col] = true;
    }
    let free: Vec<usize> = (0..cols).filter(|&f| !is_pivot[f]).collect();
    let basis = free
        .iter()
        .map(|&f| {
            let mut v = vec![Q::from(0); cols];
            v[f] = Q::from(1);
            for (row, &pc) in m.iter().zip(&pivot_of_row) {
                if let Some(x) = row.get(&f) {
                    v[pc] = -x.clone();
                }
            }
            v
        })
        .collect();
    (free, basis)
}

#[cfg(test)]
type Fault<T> = std::cell::Cell<Option<T>>;

#[cfg(test)]
thread_local! {
    /// Modules and channels this thread generated (not loaded from the cache).
    pub(crate) static BUILDS: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
    /// Columns placed by [`rounded_entries`] on this thread: `[writes, entries]`.
    pub(super) static COLUMNS: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
    /// Live exact slots of [`descend`] on this thread: `[current, maximum]`.
    pub(super) static LIVE_EXACT: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
    /// Replaces the isometry gate's allowance beyond the serial peak.
    pub(super) static GATE_ALLOWANCE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// Modules held by the live [`RequestModules`] of this thread:
    /// `[current, maximum]`.
    pub(crate) static REQUEST_MODULES: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
    /// Modules held by this thread when an F or R contraction last began.
    pub(crate) static AT_CONTRACTION: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// Labels a request held when F's fourth channel was last loaded.
    pub(crate) static BEFORE_FOURTH: std::cell::RefCell<Vec<Vec<i64>>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Row-start slots written by [`RowScratch::rows`].
    pub(super) static ROW_WRITES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Fault injection between rounding and the gates.
    pub(super) static CORRUPT: Fault<fn(&mut Vec<MEntry>)> = const { std::cell::Cell::new(None) };
    /// Called on entry to every module and channel load, before its cache lookup.
    pub(crate) static BEFORE_LOAD: Fault<fn()> = const { std::cell::Cell::new(None) };
    /// Fault injection on every completed F and R entry before its gate.
    pub(super) static CORRUPT_BLOCK: Fault<fn(f64) -> f64> = const { std::cell::Cell::new(None) };
    /// Failure injected where a complex irrep's phase would be generated.
    pub(crate) static FAIL_PHASE: Fault<fn() -> LieError> = const { std::cell::Cell::new(None) };
}

/// A completed, gated CGC channel: its outer multiplicity and its exactly
/// nonzero rounded entries in §4.7 order, ascending `(μ, m_c, m_a, m_b)`.
#[derive(Debug, PartialEq)]
pub(crate) struct CgcChannel {
    pub(crate) mult: usize,
    pub(crate) entries: Box<[MEntry]>,
}

impl std::ops::Deref for CgcChannel {
    type Target = [MEntry];
    fn deref(&self) -> &[MEntry] {
        &self.entries
    }
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
pub(crate) fn observe_contraction() {
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
        Ok(self.data.multiplicities(a, b, None)?.multiplicity(c.words) as usize)
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

    /// The completed channel `a ⊗ b → c`: its multiplicity and its rounded,
    /// gated entries in §4.7 order, shared through the CGC tier, so a hit
    /// needs no multiplicity lookup. On a miss the block `d_a d_b d_c N` must
    /// be addressable before anything is built. A channel with `N = 0` is
    /// empty, after an exact certificate: no coupling pair has the highest
    /// weight of `c`, or the highest-weight nullity is zero (§7 gate 2).
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
    /// Only a miss uses `exec`, for the descent and the isometry gate;
    /// `None` is the serial token, so a hit never builds one.
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
            #[cfg(test)]
            count_build(1);
            let n = self.mult(a, b, c)?;
            let dimension = |l: Label<'_>| -> Result<usize, LieError> {
                usize::try_from(&self.data.weyl_dimension(l.words)?)
                    .map_err(|_| LieError::LabelOverflow)
            };
            crate::lie::checked_size([dimension(a)?, dimension(b)?, dimension(c)?, n])?;
            let empty = || {
                Ok(Arc::new(CgcChannel {
                    mult: 0,
                    entries: Box::new([]),
                }))
            };
            // No coupling pair has the weight `λ_c`, so `P` is empty and a
            // zero multiplicity needs no nullspace. For `A_1` the weight `c`
            // occurs in `a ⊗ b` exactly when `c ≤ a + b` with `a + b − c`
            // even; other types read the weights off the two modules, which
            // the channel then reuses.
            let a1 = self.root() == RootSystem::A(1);
            if n == 0 && a1 && {
                let s = a.words[0] + b.words[0] - c.words[0];
                s < 0 || s % 2 != 0
            } {
                return empty();
            }
            let ma = modules.load(self, a)?;
            let mb = modules.load(self, b)?;
            if n == 0 && !a1 && ProductSpace::new(&ma, &mb, c.words).len() == 0 {
                return empty();
            }
            let mc = modules.load(self, c)?;
            // No exact vector outlives the descent; the gates read the
            // rounded entries and the modules only.
            #[allow(unused_mut)]
            let mut entries = rounded_entries(&self.data, &ma, &mb, &mc, n, exec.as_deref_mut())?;
            if self.root() == RootSystem::A(1)
                && (a.words[0] + b.words[0] - c.words[0]) / 2 % 2 != 0
            {
                // §4.6: the A_1 vertex sign (−1)^{j_a + j_b − j_c}, once per
                // channel. Rounding is odd, so negating the rounded value is
                // negating the exact one.
                for e in &mut entries {
                    e.value = -e.value;
                }
            }
            #[cfg(test)]
            if let Some(corrupt) = CORRUPT.with(std::cell::Cell::get) {
                corrupt(&mut entries);
            }
            check_rounded(&self.data, &ma, &mb, &mc, n, &entries, exec)?;
            Ok(Arc::new(CgcChannel {
                mult: n,
                entries: entries.into_boxed_slice(),
            }))
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
            let sigma_a = self.module(a)?.principal_sign(a.words)?;
            let sigma_dual = self.module(dual)?.principal_sign(dual.words)?;
            Ok(closed * sigma_a * sigma_dual)
        })
    }

    /// `F^{abc}_d[e, f]` (§6), gated by gate 7 before it is retained.
    pub(crate) fn f_symbol(
        &self,
        labels: [Label<'_>; 6],
        exec: Option<&mut Exec<'_>>,
    ) -> Result<FBlock<f64>, LieError> {
        crate::cache::f_block(self.root(), labels, || {
            let block = f_block_raw(self, labels, exec)?;
            #[cfg(test)]
            let block = match CORRUPT_BLOCK.with(std::cell::Cell::get) {
                Some(fault) => {
                    use crate::category::SymbolBlock;
                    FBlock::from_vec(
                        block.dims(),
                        block.as_slice().iter().map(|&x| fault(x)).collect(),
                    )
                }
                None => block,
            };
            check_f_block(&block)?;
            Ok(block)
        })
    }

    /// `R^{ab}_c` (§6), gated by gate 6 before it is retained.
    pub(crate) fn r_symbol(
        &self,
        labels: [Label<'_>; 3],
        exec: Option<&mut Exec<'_>>,
    ) -> Result<RBlock<f64>, LieError> {
        crate::cache::r_block(self.root(), labels, || {
            let block = r_block_raw(self, labels, exec)?;
            #[cfg(test)]
            let block = match CORRUPT_BLOCK.with(std::cell::Cell::get) {
                Some(fault) => {
                    use crate::category::SymbolBlock;
                    RBlock::from_vec(
                        block.dims()[0],
                        block.as_slice().iter().map(|&x| fault(x)).collect(),
                    )
                }
                None => block,
            };
            check_r_block(&block)?;
            Ok(block)
        })
    }
}

fn magnetic_index(value: usize) -> Result<u32, LieError> {
    u32::try_from(value).map_err(|_| LieError::LabelOverflow)
}

#[cfg(test)]
pub(super) mod tests;
