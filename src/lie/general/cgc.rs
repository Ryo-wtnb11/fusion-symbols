//! Exact Clebsch–Gordan coefficients of `a ⊗ b → c`
//! (`docs/gauge_cartan.md` §4): the exact layer of the Cartan CGC.
//!
//! Everything is computed in the rational coordinates `b^a_{m_a} ⊗ b^b_{m_b}`
//! of §3.3, whose metric is `diag(n^a_{m_a} n^b_{m_b})`. Each lowered vector
//! is `√N_μ Φ_μ(b^c_{m_c})`, so a CGC entry is its rational coordinate `x`
//! times `√(n^a n^b / (N_μ n^c))`: one signed square root.
//!
//! The descent emits exact columns only; [`exact::Pending`] integralizes
//! them into the stored authority of the channel. The rounded channel
//! ([`CgcChannel`]) is a view derived from it by [`crate::rounding::cgc`],
//! which also runs the gates defined on rounded data (§7, gates 4 and 5).
//! Nothing here touches a float. The query context that loads and publishes
//! both is [`super::family::GeneralFamily`].

use std::collections::BTreeMap;

use threadteam::Exec;

#[cfg(test)]
use self::exact::ExactCgc;
use self::exact::{Column, HighestColumn, Pending};
use super::module::{q_heap, qs_bytes, IrrepModule, Q};
use super::roots::CartanData;
#[cfg(test)]
use crate::exact::SignedSqrtRational;
use crate::frcore::{GateError, MEntry};
use crate::lie::{GenerationError, LieError, RootSystem};

pub(crate) mod exact;

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
    /// `N_μ = ⟨B_μ, B_μ⟩ ≤ 0`, which a positive definite metric excludes
    /// (§4.3).
    NonPositiveNorm {
        a: Vec<i64>,
        b: Vec<i64>,
        c: Vec<i64>,
        mu: usize,
    },
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
            Self::NonPositiveNorm { a, b, c, mu } => {
                write!(f, "{a:?} ⊗ {b:?} → {c:?}: N_{mu} is not positive")
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
    pub(crate) module: IrrepModule,
    /// `n_m = ⟨b_m, b_m⟩` per magnetic index.
    pub(crate) norm: Vec<Q>,
    /// Weight space of each magnetic index.
    pub(super) space: Vec<usize>,
    /// `f_j` per `j`.
    pub(crate) lower: Vec<Op>,
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

    pub(crate) fn dim(&self) -> usize {
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
    pub(crate) fn weight(&self, m: usize) -> &[i64] {
        &self.module.weights[self.space[m]].mu
    }

    /// The engine index of each public magnetic index.
    pub(crate) fn engine(&self) -> Vec<usize> {
        let mut out = vec![0; self.dim()];
        for (m, &p) in self.public.iter().enumerate() {
            out[p as usize] = m;
        }
        out
    }

    /// `sign ⟨low| L^h |high⟩` with `L = Σ_i f_i` (§5.3). Positive rescalings
    /// keep the sign, so the rational basis `b` gives it exactly.
    pub(super) fn principal_sign(&self, lambda: &[i64]) -> Result<i8, LieError> {
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
    /// The fibre of `nu`, with one allocation when it is non-empty and none
    /// otherwise: the scan's key and matches live in `scratch`.
    pub(super) fn new(a: &Ladders, b: &Ladders, nu: &[i64], scratch: &mut ProductScratch) -> Self {
        // Scan the smaller weight set; each of its weights meets at most one
        // weight of the other factor.
        let swapped = a.module.weights.len() < b.module.weights.len();
        let (inner, outer) = if swapped { (b, a) } else { (a, b) };
        let ProductScratch { key, matches } = scratch;
        matches.clear();
        for (wo, wb) in outer.module.weights.iter().enumerate() {
            key.clear();
            key.extend(nu.iter().zip(&wb.mu).map(|(x, y)| x - y));
            if let Some(wi) = inner.module.weight_index(key) {
                matches.push(if swapped { (wo, wi) } else { (wi, wo) });
            }
        }
        // Weight spaces are disjoint, increasing offset intervals and each
        // left space has one partner, so emitting in left-offset order gives
        // lexicographic engine pairs.
        matches.sort_unstable_by_key(|&(wa, _)| a.module.weights[wa].offset);
        let span = |m: &Ladders, w: usize| {
            let s = &m.module.weights[w];
            s.offset..s.offset + s.norms.len()
        };
        let len = matches
            .iter()
            .map(|&(wa, wb)| span(a, wa).len() * span(b, wb).len())
            .sum();
        let mut pairs = Vec::with_capacity(len);
        for &(wa, wb) in matches.iter() {
            for p in span(a, wa) {
                pairs.extend(span(b, wb).map(|q| (p, q)));
            }
        }
        // Engine order is the public a-slow order when public is the
        // identity (Gram–Schmidt); any other basis is checked and sorted.
        let key = |&(p, q): &(usize, usize)| (a.public[p], b.public[q]);
        if !pairs.is_sorted_by_key(key) {
            #[cfg(test)]
            FIBRE_SORTS.with(|n| n.set(n.get() + 1));
            pairs.sort_unstable_by_key(key);
        }
        Self { pairs }
    }

    pub(super) fn len(&self) -> usize {
        self.pairs.len()
    }
}

/// The scan buffers of [`ProductSpace::new`] for one channel, sized once at
/// their maxima so no construction grows them: the rank-`r` lookup key, and
/// the matched weight spaces `(w_a, w_b)`, at most one per weight of the
/// smaller factor for a fixed target weight.
pub(super) struct ProductScratch {
    key: Vec<i64>,
    matches: Vec<(usize, usize)>,
}

impl ProductScratch {
    pub(super) fn new(a: &Ladders, b: &Ladders) -> Self {
        let rank = a.module.weights[0].mu.len();
        let smaller = a.module.weights.len().min(b.module.weights.len());
        Self {
            key: Vec::with_capacity(rank),
            matches: Vec::with_capacity(smaller),
        }
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
    spaces: &mut ProductScratch,
) -> Result<(Vec<usize>, Option<Seeds>), LieError> {
    let top = &c.module.weights[0].mu;
    let p = ProductSpace::new(a, b, top, spaces);

    // §4.2: H is the nullspace of the stacked Δ(e_i) on span(P).
    let rows = raising_rows(data, a, b, &p, top, scratch, spaces);
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
        let norm = dot(&v, &v);
        // Checked here, before a later Gram–Schmidt step divides by it.
        if !norm.signum().is_gt() {
            return Err(CgcError::NonPositiveNorm {
                a: a.module.weights[0].mu.clone(),
                b: b.module.weights[0].mu.clone(),
                c: top.clone(),
                mu: norms.len(),
            }
            .into());
        }
        norms.push(norm);
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
    let mut product = ProductScratch::new(a, b);
    let (pivots, seeded) = highest(data, a, b, c, multiplicity, &mut scratch, &mut product)?;
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
        let to = ProductSpace::new(a, b, &space.mu, &mut product);
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
/// exact payload is moved, never copied.
fn place(slot: &mut Column, column: Column) {
    debug_assert!(slot.is_empty(), "a column is written once");
    #[cfg(test)]
    COLUMNS.with(|c| {
        let [writes, entries] = c.get();
        c.set([writes + 1, entries + column.len()]);
    });
    *slot = column;
}

/// The exact column `(μ, m_c)` of the state `v = √N_μ Φ_μ(b^c_{m_c})` over
/// `space`: one entry per exactly nonzero coordinate, a-slow.
/// Its scale is reduced here, in the item that emits it.
fn exact_column(a: &Ladders, b: &Ladders, space: &ProductSpace, v: &[Q]) -> Column {
    let mut column = Vec::with_capacity(v.iter().filter(|x| !x.is_zero()).count());
    for (&(pa, pb), x) in space.pairs.iter().zip(v) {
        if !x.is_zero() {
            column.push((a.public[pa], b.public[pb], x.clone()));
        }
    }
    Column::new(column)
}

/// Per-participant row starts of [`descend_layers`]' phase A, bound to the
/// last target weight it lowered into.
struct BoundRows {
    rows: RowScratch,
    bound: Option<usize>,
}

/// The exact columns of [`descend`] and the seed norms `N_μ`, one depth
/// layer at a time, on a team token (`docs/gauge_cartan.md` §9).
///
/// A layer runs as three stages of [`crate::stage::run_stage`]: the first
/// builds the product space of every weight of the layer, each participant
/// on its own [`ProductScratch`]; phase A lowers every kept candidate × `μ`
/// of the layer; phase B combines every recipe row × `μ` and emits its
/// column. Every item computes exactly the operations
/// [`descend`] does for it, so the columns are identical at every width.
/// The live exact set is at most two layers plus one layer's candidates; the
/// leader frees a parent after phase A, when all its readers have run.
fn descend_layers(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    c: &Ladders,
    n: usize,
    exec: &mut Exec<'_>,
) -> Result<(Vec<Column>, Vec<Q>), LieError> {
    use crate::stage::{run_stage, StageBuf, SystemClock};
    let dc = c.dim();
    let mut columns: Vec<Column> = vec![Column::default(); n * dc];
    let mut scratch = RowScratch::new(a);
    let mut product = ProductScratch::new(a, b);
    let Some(Seeds {
        p,
        seeds,
        norms: seed_norms,
    }) = highest(data, a, b, c, n, &mut scratch, &mut product)?.1
    else {
        return Ok((columns, Vec::new()));
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

    track(&seeds, true);
    for (mu, v) in seeds.iter().enumerate() {
        place(
            &mut columns[mu * dc + c.public[0] as usize],
            exact_column(a, b, &p, v),
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
    // The leader's scan scratch, carried across the layers' construction
    // stages; every construction scans the smaller factor once.
    let mut product = Some(product);
    let scan = a.module.weights.len().min(b.module.weights.len()) as u64;
    let mut building = StageBuf::default();
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
        // The layer's product spaces: one item per weight, each scanning the
        // smaller factor. Drained in weight order before phase A reads `tos`.
        units.clear();
        units.resize(layer.len(), scan);
        let _built = run_stage(
            exec,
            &units,
            &SystemClock,
            &mut product,
            &mut building,
            || ProductScratch::new(a, b),
            |s, w| ProductSpace::new(a, b, &layer[w].mu, s),
        );
        #[cfg(test)]
        if _built.width.get() > 1 {
            CONSTRUCTION_SECTIONS.with(|n| n.set(n.get() + 1));
        }
        tos.extend(building.drain());

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
                let column = exact_column(a, b, &tos[w], v.as_deref().unwrap_or(kept));
                // A state no candidate reads dies with its share.
                let transient = v.as_ref().filter(|_| readers[m] == 0).map_or(0, Vec::len);
                (column, v.filter(|_| readers[m] > 0), transient)
            },
        );
        #[cfg(test)]
        let mut transient = Vec::with_capacity(units.len());
        let mut done = combining.drain();
        for &(w, k) in &rows {
            let m = layer[w].offset + k;
            let mut states = Vec::with_capacity(if readers[m] > 0 { n } else { 0 });
            for mu in 0..n {
                let (column, state, _t) = done.next().expect("one result per item");
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
    Ok((columns, seed_norms))
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

/// The emitted columns of `a ⊗ b → c` of multiplicity `n` (§4.2–§4.4,
/// §4.7), each with its scale, planned for integralization per `μ`: every
/// exactly nonzero coordinate, emitted per column `(μ, m_c)` as its coupled
/// state completes, every `X` to be negated when `negate` (the `A_1` vertex
/// sign, §4.6). The integers are written by one per-column stage after the
/// descent ([`Pending`]).
///
/// This is the descent's split point: a token that carries a team of two or
/// more descends layer by layer ([`descend_layers`]); any other token, and
/// `None`, runs [`descend`] on the calling thread. Both emit the same
/// columns.
pub(super) fn exact_columns(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    c: &Ladders,
    n: usize,
    negate: bool,
    exec: Option<&mut Exec<'_>>,
) -> Result<Pending, LieError> {
    let (columns, norms) = match exec.filter(|exec| exec.carries_team() && !exec.is_serial()) {
        Some(exec) => descend_layers(data, a, b, c, n, exec)?,
        None => {
            let dc = c.dim();
            let mut columns: Vec<Column> = vec![Column::default(); n * dc];
            let mut norms = Vec::new();
            descend(data, a, b, c, n, &mut |seed_norms, m, space, per_mu| {
                if norms.is_empty() {
                    norms.extend_from_slice(seed_norms);
                }
                let mc = c.public[m] as usize;
                for (mu, v) in per_mu.iter().enumerate() {
                    place(&mut columns[mu * dc + mc], exact_column(a, b, space, v));
                }
                Ok(())
            })?;
            (columns, norms)
        }
    };
    Pending::new(columns, c.dim(), &norms, negate)
}

/// The exact channel of [`exact_columns`], every column integralized in one
/// stage on the same token.
#[cfg(test)]
pub(super) fn exact_channel(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    c: &Ladders,
    n: usize,
    negate: bool,
    mut exec: Option<&mut Exec<'_>>,
) -> Result<ExactCgc, LieError> {
    let pending = exact_columns(data, a, b, c, n, negate, exec.as_deref_mut())?;
    Ok(pending.integralize([a.dim(), b.dim(), c.dim()], exec))
}

/// The highest-weight column of `a ⊗ b → c` (engine index 0 of `c`): the
/// exact certificates and seeds of §4.2–§4.3, integralized per `μ`, with no
/// descent.
pub(super) fn highest_column(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    c: &Ladders,
    n: usize,
) -> Result<HighestColumn, LieError> {
    let mut scratch = RowScratch::new(a);
    let mut product = ProductScratch::new(a, b);
    let Some(Seeds { p, seeds, norms }) = highest(data, a, b, c, n, &mut scratch, &mut product)?.1
    else {
        return Ok(HighestColumn::empty());
    };
    let columns = seeds.iter().map(|v| exact_column(a, b, &p, v)).collect();
    HighestColumn::from_columns(columns, &norms)
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
    /// indices, in the §4.7 order of [`exact_channel`], with the exact
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

/// Stacked raising coproducts on `P`, retaining only nonzero coordinates.
fn raising_rows(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    p: &ProductSpace,
    top: &[i64],
    scratch: &mut RowScratch,
    spaces: &mut ProductScratch,
) -> Vec<BTreeMap<usize, Q>> {
    let mut rows = Vec::new();
    for i in 0..top.len() {
        let nu: Vec<i64> = (0..top.len()).map(|k| top[k] + data.c(i, k)).collect();
        let target = ProductSpace::new(a, b, &nu, spaces);
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
    /// Generations on this thread (not loaded from the cache):
    /// `[modules, exact channels, seeds, norm tables]`.
    pub(crate) static BUILDS: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
    /// Exact columns emitted by the descents of this thread: `[writes, entries]`.
    pub(super) static COLUMNS: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
    /// Construction stages of [`descend_layers`] on this thread that ran a
    /// section of width two or more.
    pub(super) static CONSTRUCTION_SECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Live exact slots of [`descend`] on this thread: `[current, maximum]`.
    pub(super) static LIVE_EXACT: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
    /// Modules held by the live [`super::family::RequestModules`] of this thread:
    /// `[current, maximum]`.
    pub(crate) static REQUEST_MODULES: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
    /// Modules held by this thread when an F or R contraction last began.
    pub(crate) static AT_CONTRACTION: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// Labels a request held when F's fourth channel was last loaded.
    pub(crate) static BEFORE_FOURTH: std::cell::RefCell<Vec<Vec<i64>>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Row-start slots written by [`RowScratch::rows`].
    pub(super) static ROW_WRITES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Product spaces whose emitted order needed [`ProductSpace::new`]'s sort.
    pub(super) static FIBRE_SORTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Run once by the next view miss of this thread, after its exact
    /// channel was published and before the view is.
    pub(super) static AFTER_EXACT: Fault<fn()> = const { std::cell::Cell::new(None) };
    /// Fault injection between rounding and the gates.
    pub(super) static CORRUPT: Fault<fn(&mut [MEntry])> = const { std::cell::Cell::new(None) };
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

#[cfg(test)]
pub(super) mod tests;

#[cfg(all(test, feature = "disk-cache"))]
mod disk_tests;
