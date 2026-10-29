//! Exact Clebsch–Gordan coefficients of `a ⊗ b → c`
//! (`docs/gauge_exceptional.md` §5), the [`Family`] binding of the exceptional
//! engine to `frcore` (§7) and the Frobenius–Schur phase `κ_a` (§6.3).
//!
//! Everything is computed in the rational coordinates `b^a_{m_a} ⊗ b^b_{m_b}`
//! of §4.5, whose metric is `diag(n^a_{m_a} n^b_{m_b})`. Each lowered vector
//! is `√N_μ Φ_μ(b^c_{m_c})`, so a CGC entry is its rational coordinate `x`
//! times `√(n^a n^b / (N_μ n^c))`: one signed square root, rounded once.

use std::collections::{BTreeMap, HashMap};

use super::module::{labels_bytes, q_heap, qs_bytes, IrrepModule, Q};
use super::roots::CartanData;
use crate::exact::SignedSqrtRational;
use crate::frcore::{Family, MEntry};
#[cfg(test)]
use crate::lie::RootSystem;
use crate::lie::{GenerationError, LieError};

/// A sparse operator in the rational basis: `op[m]` lists `(m', x)` with
/// `X b_m = Σ x b_{m'}`.
type Op = Vec<Vec<(usize, Q)>>;

/// An internal inconsistency between the construction and its gates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum CgcError {
    /// `dim H` differs from the Brauer–Klimyk multiplicity (§5.2 gate).
    NullityMismatch {
        a: Vec<i64>,
        b: Vec<i64>,
        c: Vec<i64>,
        nullity: usize,
        multiplicity: usize,
    },
    /// `⟨low| L^h |high⟩ = 0`, which §6.3 proves impossible.
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

/// `L(λ)` with its ladder operators flattened to magnetic indices.
pub(super) struct Ladders {
    pub(super) module: IrrepModule,
    /// `n_m = ⟨b_m, b_m⟩` per magnetic index.
    pub(super) norm: Vec<Q>,
    /// Weight space of each magnetic index.
    pub(super) space: Vec<usize>,
    /// `f_j` per `j`.
    pub(super) lower: Vec<Op>,
    /// `e_j` per `j`, the contravariant transpose `E[b][a] = F[a][b] n_a / n_b`.
    pub(super) raise: Vec<Op>,
}

impl Ladders {
    pub(super) fn new(data: &CartanData, lambda: &[i64]) -> Result<Self, LieError> {
        let module = IrrepModule::new(data, lambda)?;
        let rank = lambda.len();
        let mut norm = Vec::with_capacity(module.dim);
        let mut space = Vec::with_capacity(module.dim);
        for (w, s) in module.weights.iter().enumerate() {
            norm.extend(s.norms.iter().cloned());
            space.extend(std::iter::repeat_n(w, s.norms.len()));
        }
        let mut lower = vec![vec![Vec::new(); module.dim]; rank];
        let mut raise = vec![vec![Vec::new(); module.dim]; rank];
        for s in &module.weights {
            for (j, slot) in s.lower.iter().enumerate() {
                let Some((src, f)) = slot else { continue };
                let src = &module.weights[*src];
                for (a, row) in f.iter().enumerate() {
                    for (b, x) in row.iter().enumerate() {
                        if x.numerator_ref() == &0 {
                            continue;
                        }
                        let (ma, mb) = (s.offset + a, src.offset + b);
                        lower[j][mb].push((ma, x.clone()));
                        raise[j][ma].push((mb, x * &s.norms[a] / &src.norms[b]));
                    }
                }
            }
        }
        Ok(Self {
            module,
            norm,
            space,
            lower,
            raise,
        })
    }

    pub(super) fn dim(&self) -> usize {
        self.module.dim
    }

    /// Estimated retained bytes, including the module and both ladder sets.
    fn bytes(&self) -> usize {
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
    }

    /// `sign ⟨low| L^h |high⟩` with `L = Σ_i f_i` (§6.3). Positive rescalings
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
                    if x.numerator_ref() == &0 {
                        continue;
                    }
                    for (m2, y) in &op[m] {
                        next[*m2] += x * y;
                    }
                }
            }
            v = next;
        }
        match v.last() {
            Some(x) if x > &0 => Ok(1),
            Some(x) if x < &0 => Ok(-1),
            _ => Err(CgcError::ZeroPrincipalLadder {
                lambda: lambda.to_vec(),
            }
            .into()),
        }
    }
}

/// The product states `(m_a, m_b)` of one weight, in composite order
/// `m_a + d_a·m_b` (§5.1).
pub(super) struct ProductSpace {
    pub(super) pairs: Vec<(usize, usize)>,
    index: HashMap<(usize, usize), usize>,
}

impl ProductSpace {
    fn new(a: &Ladders, b: &Ladders, nu: &[i64]) -> Self {
        let mut pairs = Vec::new();
        // Scan the smaller weight set, then restore §5.1 composite order if swapped.
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
        if swapped {
            pairs.sort_unstable_by_key(|&(p, q)| (q, p));
        }
        let index = pairs.iter().enumerate().map(|(t, &pq)| (pq, t)).collect();
        Self { pairs, index }
    }

    fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Row of `pq`; every image of a weight vector under a ladder operator has
    /// the target weight, so the lookup always succeeds.
    fn at(&self, pq: (usize, usize)) -> usize {
        self.index[&pq]
    }
}

/// `Δ(x) = x ⊗ 1 + 1 ⊗ x` applied to `v` on `from`, landing on `to`.
fn coproduct(op_a: &Op, op_b: &Op, from: &ProductSpace, to: &ProductSpace, v: &[Q]) -> Vec<Q> {
    let mut out = vec![Q::from(0); to.len()];
    for (&(p, q), x) in from.pairs.iter().zip(v) {
        if x.numerator_ref() == &0 {
            continue;
        }
        for (p2, y) in &op_a[p] {
            out[to.at((*p2, q))] += x * y;
        }
        for (q2, y) in &op_b[q] {
            out[to.at((p, *q2))] += x * y;
        }
    }
    out
}

/// The CGC of one channel in rational form.
pub(super) struct Channel {
    pub(super) mult: usize,
    /// The pivot rows `p_1 < … < p_N` of §5.3, as indices into `P`.
    #[cfg(test)]
    pub(super) pivots: Vec<usize>,
    /// `N_μ = ⟨B_μ, B_μ⟩`.
    pub(super) seed_norms: Vec<Q>,
    /// The product space of each weight space of `c`; `spaces[0]` is `P`.
    pub(super) spaces: Vec<ProductSpace>,
    /// `phi[m_c][μ]`: coordinates of `√N_μ Φ_μ(b^c_{m_c})` over its space.
    pub(super) phi: Vec<Vec<Vec<Q>>>,
}

impl Channel {
    pub(super) fn new(
        data: &CartanData,
        a: &Ladders,
        b: &Ladders,
        c: &Ladders,
        multiplicity: usize,
    ) -> Result<Self, LieError> {
        let top = &c.module.weights[0].mu;
        let p = ProductSpace::new(a, b, top);

        // §5.2: H is the nullspace of the stacked Δ(e_i) on span(P).
        let rows = raising_rows(data, a, b, &p, top);
        let (_pivots, echelon) = echelon_nullspace(rows, p.len());
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

        // §5.3: Gram–Schmidt of E_1, …, E_N in pivot order, metric n^a n^b.
        let metric: Vec<Q> = p
            .pairs
            .iter()
            .map(|&(x, y)| &a.norm[x] * &b.norm[y])
            .collect();
        let dot = |x: &[Q], y: &[Q]| -> Q {
            x.iter()
                .zip(y)
                .zip(&metric)
                .filter(|((u, v), _)| u.numerator_ref() != &0 && v.numerator_ref() != &0)
                .map(|((u, v), g)| u * v * g)
                .fold(Q::from(0), |acc, x| acc + x)
        };
        let mut seeds: Vec<Vec<Q>> = Vec::with_capacity(multiplicity);
        let mut seed_norms: Vec<Q> = Vec::with_capacity(multiplicity);
        for e in echelon {
            let mut v = e.clone();
            for (s, n) in seeds.iter().zip(&seed_norms) {
                let scale = dot(&e, s) / n;
                if scale.numerator_ref() == &0 {
                    continue;
                }
                for (x, y) in v.iter_mut().zip(s) {
                    *x -= &scale * y;
                }
            }
            seed_norms.push(dot(&v, &v));
            seeds.push(v);
        }

        // §5.4: replay c's own basis recipe through Δ(f_j).
        let mut spaces = vec![p];
        let mut phi: Vec<Vec<Vec<Q>>> = Vec::with_capacity(c.dim());
        phi.push(seeds);
        for space in &c.module.weights[1..] {
            let to = ProductSpace::new(a, b, &space.mu);
            let lowered: Vec<Vec<Vec<Q>>> = space
                .kept
                .iter()
                .map(|&(j, q)| {
                    let (src, _) = space.lower[j]
                        .as_ref()
                        .expect("a kept candidate's source weight space exists (§4.1)");
                    let from = &spaces[*src];
                    let m = c.module.weights[*src].offset + q;
                    phi[m]
                        .iter()
                        .map(|v| coproduct(&a.lower[j], &b.lower[j], from, &to, v))
                        .collect()
                })
                .collect();
            for (k, coeffs) in space.gs.iter().enumerate() {
                let per_mu = (0..multiplicity)
                    .map(|mu| {
                        // §4.3: own-candidate coefficient is 1; later ones are 0.
                        let mut v = lowered[k][mu].clone();
                        for (r, cand) in coeffs[..k].iter().zip(&lowered) {
                            if r.numerator_ref() == &0 {
                                continue;
                            }
                            for (x, y) in v.iter_mut().zip(&cand[mu]) {
                                if y.numerator_ref() != &0 {
                                    *x += r * y;
                                }
                            }
                        }
                        v
                    })
                    .collect();
                phi.push(per_mu);
            }
            spaces.push(to);
        }
        Ok(Self {
            mult: multiplicity,
            #[cfg(test)]
            pivots: _pivots,
            seed_norms,
            spaces,
            phi,
        })
    }

    /// The exactly nonzero entries `(m_a, m_b, m_c, μ, C)` in the §5.6 order:
    /// `μ`, then `m_c`, then `composite(m_a, m_b)`, all ascending.
    pub(super) fn exact_entries(
        &self,
        a: &Ladders,
        b: &Ladders,
        c: &Ladders,
    ) -> Vec<(usize, usize, usize, usize, SignedSqrtRational)> {
        let mut out = Vec::new();
        for mu in 0..self.mult {
            for (mc, per_mu) in self.phi.iter().enumerate() {
                let space = &self.spaces[c.space[mc]];
                let scale = &self.seed_norms[mu] * &c.norm[mc];
                for (&(pa, pb), x) in space.pairs.iter().zip(&per_mu[mu]) {
                    if x.numerator_ref() == &0 {
                        continue;
                    }
                    out.push((
                        pa,
                        pb,
                        mc,
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
) -> Vec<BTreeMap<usize, Q>> {
    let mut rows = Vec::new();
    for i in 0..top.len() {
        let nu: Vec<i64> = (0..top.len()).map(|k| top[k] + data.c(i, k)).collect();
        let target = ProductSpace::new(a, b, &nu);
        if target.pairs.is_empty() {
            continue;
        }
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
        row.retain(|_, x| x.numerator_ref() != &0);
    }
    rows
}

/// The echelon basis of `ker M` (§5.3): `E_k[p_l] = δ_kl` and `E_k[t] = 0`
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
                *x -= &scale * y;
                if x.numerator_ref() == &0 {
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

struct MemoEntry<T> {
    value: T,
    bytes: usize,
    used: u64,
}

fn evict_oldest<K: Clone + Eq + std::hash::Hash, V>(
    entries: &mut HashMap<K, MemoEntry<V>>,
) -> Option<usize> {
    let key = entries.iter().min_by_key(|(_, e)| e.used)?.0.clone();
    Some(entries.remove(&key)?.bytes)
}

/// The exceptional engine as a [`Family`]: exact CGC (§5) and `κ_a` (§6.3),
/// with F and R from the generic `frcore` contractions (§7).
///
/// Modules, channels and complex-irrep phases are memoized
/// for the lifetime of the value only; `bytes` is the estimated retained size
/// of these three kinds.
pub(crate) struct GeneralFamily {
    data: std::sync::Arc<CartanData>,
    modules: HashMap<Vec<i64>, MemoEntry<Ladders>>,
    channels: HashMap<[Vec<i64>; 3], MemoEntry<Vec<MEntry>>>,
    /// One `κ` per non-self-dual pair, keyed by its smaller label.
    phases: HashMap<Vec<i64>, MemoEntry<i8>>,
    bytes: usize,
    clock: u64,
    #[cfg(test)]
    pub(super) builds: [usize; 2],
}

impl GeneralFamily {
    #[cfg(test)]
    pub(crate) fn new(root: RootSystem) -> Self {
        Self::with_data(CartanData::new(root))
    }
    pub(crate) fn with_data(data: std::sync::Arc<CartanData>) -> Self {
        Self {
            data,
            modules: HashMap::new(),
            channels: HashMap::new(),
            phases: HashMap::new(),
            bytes: 0,
            clock: 0,
            #[cfg(test)]
            builds: [0; 2],
        }
    }

    #[cfg(test)]
    pub(super) fn data(&self) -> &CartanData {
        &self.data
    }

    /// Retained memo entries; each phase pair counts as two label entries.
    pub(super) fn memo_len(&self) -> usize {
        self.modules.len() + self.channels.len() + 2 * self.phases.len()
    }

    /// Estimated retained bytes of the memo (capacity-based, saturating).
    pub(super) fn memo_bytes(&self) -> usize {
        self.bytes
    }

    /// Only the owner calls this, after a query's module borrows have ended.
    /// Completed channels and paired phases are useful without their modules.
    pub(super) fn trim(&mut self, max_entries: usize, max_bytes: usize) {
        while self.memo_len() > max_entries || self.memo_bytes() > max_bytes {
            let bytes = evict_oldest(&mut self.modules)
                .or_else(|| evict_oldest(&mut self.channels))
                .or_else(|| evict_oldest(&mut self.phases))
                .expect("an over-budget memo has entries");
            self.bytes -= bytes;
        }
    }

    pub(super) fn module(&mut self, lambda: &[i64]) -> Result<&Ladders, LieError> {
        self.clock = self.clock.saturating_add(1);
        if !self.modules.contains_key(lambda) {
            let built = Ladders::new(&self.data, lambda)?;
            #[cfg(test)]
            {
                self.builds[0] += 1;
            }
            let bytes =
                labels_bytes(lambda) + built.bytes() + std::mem::size_of::<MemoEntry<Ladders>>();
            self.bytes = self.bytes.saturating_add(bytes);
            self.modules.insert(
                lambda.to_vec(),
                MemoEntry {
                    value: built,
                    bytes,
                    used: self.clock,
                },
            );
        }
        let hit = self.modules.get_mut(lambda).unwrap();
        hit.used = self.clock;
        Ok(&hit.value)
    }

    /// `a ⊗ b` by Brauer–Klimyk, label → `N^c_{ab}`.
    #[cfg(test)]
    pub(crate) fn product(&self, a: &[i64], b: &[i64]) -> Result<super::roots::Product, LieError> {
        self.data.shared_product(a, b)
    }

    /// The rational channel with its three modules.
    pub(super) fn channel(
        &mut self,
        a: &[i64],
        b: &[i64],
        c: &[i64],
    ) -> Result<(Channel, &Ladders, &Ladders, &Ladders), LieError> {
        let n = self.data.multiplicities(a, b, None)?.multiplicity(c) as usize;
        for l in [a, b, c] {
            self.module(l)?;
        }
        let (ma, mb, mc) = (
            &self.modules[a].value,
            &self.modules[b].value,
            &self.modules[c].value,
        );
        Ok((Channel::new(&self.data, ma, mb, mc, n)?, ma, mb, mc))
    }

    /// `κ_a` of §6.3: the closed form `(−1)^{⟨λ, 2ρ∨⟩}` for a self-dual irrep,
    /// and `(−1)^{⟨λ, 2ρ∨⟩} σ_a σ_ā` otherwise.
    pub(crate) fn frobenius_schur_phase(&mut self, a: &[i64]) -> Result<i8, LieError> {
        let dual = self.data.dual(a)?;
        if dual == a {
            return self.data.two_rho_vee_sign(a);
        }
        self.clock = self.clock.saturating_add(1);
        let key = a.min(&dual);
        if let Some(hit) = self.phases.get_mut(key) {
            hit.used = self.clock;
            return Ok(hit.value);
        }
        let closed = self.data.two_rho_vee_sign(a)?;
        let sigma_a = self.module(a)?.principal_sign(a)?;
        let sigma_dual = self.module(&dual)?.principal_sign(&dual)?;
        // κ_ā = κ_a (§6.3), so one generation answers both labels.
        let kappa = closed * sigma_a * sigma_dual;
        let bytes = labels_bytes(key) + std::mem::size_of::<MemoEntry<i8>>();
        self.bytes = self.bytes.saturating_add(bytes);
        self.phases.insert(
            key.to_vec(),
            MemoEntry {
                value: kappa,
                bytes,
                used: self.clock,
            },
        );
        Ok(kappa)
    }
}

fn magnetic_index(value: usize) -> Result<u32, LieError> {
    u32::try_from(value).map_err(|_| LieError::LabelOverflow)
}

impl Family for GeneralFamily {
    type Irrep = Vec<i64>;
    type Error = LieError;

    fn overflow() -> LieError {
        LieError::LabelOverflow
    }

    fn mult(&mut self, a: &Vec<i64>, b: &Vec<i64>, c: &Vec<i64>) -> Result<usize, LieError> {
        Ok(self.data.multiplicities(a, b, None)?.multiplicity(c) as usize)
    }

    fn cgc_entries(
        &mut self,
        a: &Vec<i64>,
        b: &Vec<i64>,
        c: &Vec<i64>,
    ) -> Result<Vec<MEntry>, LieError> {
        let key = [a.clone(), b.clone(), c.clone()];
        self.clock = self.clock.saturating_add(1);
        if let Some(hit) = self.channels.get_mut(&key) {
            hit.used = self.clock;
            return Ok(hit.value.clone());
        }
        #[cfg(test)]
        {
            self.builds[1] += 1;
        }
        let (channel, ma, mb, mc) = self.channel(a, b, c)?;
        let entries = channel
            .exact_entries(ma, mb, mc)
            .into_iter()
            .map(|(m1, m2, m3, mu, x)| {
                Ok(MEntry {
                    m1: magnetic_index(m1)?,
                    m2: magnetic_index(m2)?,
                    m3: magnetic_index(m3)?,
                    mu: magnetic_index(mu)?,
                    value: x.to_f64(),
                })
            })
            .collect::<Result<Vec<_>, LieError>>()?;
        let key_bytes: usize = key.iter().map(|l| labels_bytes(l)).sum();
        let retained = entries.clone();
        let bytes = key_bytes
            + retained.capacity() * std::mem::size_of::<MEntry>()
            + std::mem::size_of::<MemoEntry<Vec<MEntry>>>();
        self.bytes = self.bytes.saturating_add(bytes);
        self.channels.insert(
            key,
            MemoEntry {
                value: retained,
                bytes,
                used: self.clock,
            },
        );
        Ok(entries)
    }
}

#[cfg(test)]
mod tests;
