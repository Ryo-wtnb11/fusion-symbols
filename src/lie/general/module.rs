//! The irreducible module `L(λ)` in the Gram–Schmidt basis of the
//! contravariant form (`docs/gauge_cartan.md` §3.2), built exactly from the
//! Cartan matrix alone, and the depth order shared with the Gelfand–Tsetlin
//! basis of §3.1.
//!
//! The module is kept in the unnormalized orthogonal basis `b_k` of §3.2.3: the
//! norms `n_k = ⟨b_k, b_k⟩`, a rational lowering recipe over the kept
//! candidates (here the Gram–Schmidt coefficients), and the rational lowering
//! matrices `F` of §3.3. Every ladder entry of the orthonormal basis
//! `u_k = b_k / √n_k` is then the single signed square root
//! `F[a, b] · √(n_a / n_b)`, returned as a
//! [`SignedSqrtRational`](crate::exact::SignedSqrtRational); no two square
//! roots are ever added.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::roots::CartanData;
use crate::lie::{GenerationError, LieError};

pub(super) use super::rational::Q;

/// Estimated bytes of a `Vec<i64>` with its header.
pub(super) fn labels_bytes(v: &[i64]) -> usize {
    std::mem::size_of::<Vec<i64>>() + std::mem::size_of_val(v)
}

/// Estimated heap bytes of one rational: its numerator and denominator digits.
pub(super) fn q_heap(q: &Q) -> usize {
    (q.numerator_ref().limbs().len() + q.denominator_ref().limbs().len())
        * std::mem::size_of::<malachite_nz::platform::Limb>()
}

/// Estimated bytes of a vector of rationals, capacity-based.
pub(super) fn qs_bytes(v: &Vec<Q>) -> usize {
    v.capacity() * std::mem::size_of::<Q>() + v.iter().map(q_heap).sum::<usize>()
}

/// Estimated bytes of a rational matrix stored as rows.
pub(super) fn qrows_bytes(m: &Vec<Vec<Q>>) -> usize {
    m.capacity() * std::mem::size_of::<Vec<Q>>() + m.iter().map(qs_bytes).sum::<usize>()
}

/// `L(λ)` in the §3.2 basis.
pub(super) struct IrrepModule {
    /// Weight spaces in the §3.2.1 order.
    pub(super) weights: Vec<WeightSpace>,
    index: HashMap<Vec<i64>, usize>,
    pub(super) dim: usize,
}

pub(super) struct WeightSpace {
    /// The weight `μ` in Dynkin coordinates.
    pub(super) mu: Vec<i64>,
    /// `λ − μ = Σ_i k_i α_i` (§3.2.1).
    pub(super) k: Vec<i64>,
    /// Magnetic index of `b_0`; this space holds `offset .. offset + norms.len()`.
    pub(super) offset: usize,
    /// `n_k = ⟨b_k, b_k⟩ > 0`.
    pub(super) norms: Vec<Q>,
    /// The kept candidates `c_t = f_j b_q^{μ+α_j}` as `(j, q)`, in §3.2.3 order.
    pub(super) kept: Vec<(usize, usize)>,
    /// The lowering recipe `b_k = Σ_t recipe[k][t] c_t` over the kept
    /// candidates. The Gram–Schmidt basis of §3.2.3 has a unit lower-triangular
    /// recipe; another basis may have any invertible one.
    pub(super) recipe: Vec<Vec<Q>>,
    /// Per `j`: the index of `μ + α_j` and `F` with `f_j b_b = Σ_a F[a][b] b_a`.
    pub(super) lower: Vec<Option<(usize, Vec<Vec<Q>>)>>,
}

/// An internal inconsistency between the construction and its gates (§3.2.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ModuleError {
    /// The greedy rank of a weight space differs from its Freudenthal multiplicity.
    RankMismatch {
        weight: Vec<i64>,
        rank: usize,
        multiplicity: u64,
    },
    /// The built dimension differs from the Weyl dimension.
    DimensionMismatch { built: usize, weyl: usize },
    /// A basis vector has nonpositive norm, so the form is not positive definite.
    IndefiniteForm { weight: Vec<i64> },
    /// A closed-form ladder coefficient is undefined (zero denominator) or
    /// leaves its weight.
    InvalidLadder { weight: Vec<i64> },
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RankMismatch {
                weight,
                rank,
                multiplicity,
            } => write!(
                f,
                "weight {weight:?}: selected rank {rank} differs from multiplicity {multiplicity}"
            ),
            Self::DimensionMismatch { built, weyl } => {
                write!(
                    f,
                    "built dimension {built} differs from Weyl dimension {weyl}"
                )
            }
            Self::IndefiniteForm { weight } => {
                write!(f, "weight {weight:?}: contravariant form is not positive")
            }
            Self::InvalidLadder { weight } => {
                write!(
                    f,
                    "weight {weight:?}: closed-form ladder coefficient is invalid"
                )
            }
        }
    }
}

impl std::error::Error for ModuleError {}

impl From<ModuleError> for LieError {
    fn from(error: ModuleError) -> Self {
        Self::Generation(GenerationError(std::sync::Arc::new(error)))
    }
}

impl IrrepModule {
    /// Estimated retained bytes: weight spaces with their norms, Gram–Schmidt
    /// coefficients and lowering matrices, and the weight index.
    pub(super) fn bytes(&self) -> usize {
        let spaces: usize = self
            .weights
            .iter()
            .map(|s| {
                let lower: usize = s.lower.capacity()
                    * std::mem::size_of::<Option<(usize, Vec<Vec<Q>>)>>()
                    + s.lower
                        .iter()
                        .flatten()
                        .map(|(_, f)| qrows_bytes(f))
                        .sum::<usize>();
                labels_bytes(&s.mu)
                    + labels_bytes(&s.k)
                    + qs_bytes(&s.norms)
                    + s.kept.capacity() * std::mem::size_of::<(usize, usize)>()
                    + qrows_bytes(&s.recipe)
                    + lower
            })
            .sum();
        let index: usize = self
            .index
            .keys()
            .map(|k| labels_bytes(k) + std::mem::size_of::<usize>())
            .sum();
        self.weights.capacity() * std::mem::size_of::<WeightSpace>() + spaces + index
    }

    /// Builds `L(λ)` for a dominant `λ` in the Gram–Schmidt basis of §3.2.
    /// Dimensions beyond `u32`, the magnetic-index type handed to `frcore`,
    /// are [`LieError::LabelOverflow`].
    pub(super) fn new(data: &CartanData, lambda: &[i64]) -> Result<Self, LieError> {
        let mut order = depth_order(data, lambda)?;
        let rank = lambda.len();
        let spaces = std::mem::take(&mut order.spaces);
        let mut weights: Vec<WeightSpace> = Vec::with_capacity(spaces.len());
        let mut offset = 0;
        for (k, mu) in spaces {
            let space = if weights.is_empty() {
                WeightSpace {
                    lower: vec![None; rank],
                    mu,
                    k,
                    offset,
                    norms: vec![Q::from(1)],
                    kept: Vec::new(),
                    recipe: vec![Vec::new()],
                }
            } else {
                let multiplicity = order.mult.get(&mu).copied().unwrap_or(0);
                build_space(data, &weights, &order.index, mu, k, offset, multiplicity)?
            };
            offset += space.norms.len();
            weights.push(space);
        }
        Self::from_spaces(weights, order)
    }

    /// Assembles built weight spaces in the §3.2.1 order of `order` (whose
    /// `spaces` may already have been consumed), with the final dimension
    /// gate against the Weyl dimension.
    pub(super) fn from_spaces(
        weights: Vec<WeightSpace>,
        order: DepthOrder,
    ) -> Result<Self, LieError> {
        let dim = weights.iter().map(|s| s.norms.len()).sum();
        if dim != order.weyl {
            return Err(ModuleError::DimensionMismatch {
                built: dim,
                weyl: order.weyl,
            }
            .into());
        }
        Ok(Self {
            weights,
            index: order.index,
            dim,
        })
    }

    /// A module whose spaces are known to have total dimension `dim`.
    pub(super) fn assembled(
        weights: Vec<WeightSpace>,
        index: HashMap<Vec<i64>, usize>,
        dim: usize,
    ) -> Self {
        debug_assert_eq!(weights.iter().map(|s| s.norms.len()).sum::<usize>(), dim);
        Self {
            weights,
            index,
            dim,
        }
    }

    pub(super) fn weight_index(&self, mu: &[i64]) -> Option<usize> {
        self.index.get(mu).copied()
    }

    /// `⟨u_a, f_j u_b⟩` with `u_a` the `a`-th basis vector of weight space `w`
    /// and `u_b` the `b`-th of `μ_w + α_j` (§3.3).
    #[cfg(test)]
    pub(super) fn f_entry(
        &self,
        w: usize,
        j: usize,
        a: usize,
        b: usize,
    ) -> crate::exact::SignedSqrtRational {
        use crate::exact::SignedSqrtRational;
        let space = &self.weights[w];
        let Some((src, f)) = &space.lower[j] else {
            return SignedSqrtRational::zero();
        };
        super::rational::radical(
            &f[a][b],
            &space.norms[a],
            &Q::from(1),
            &self.weights[*src].norms[b],
        )
    }
}

/// The weights of `L(λ)` in the §3.2.1 order, with their multiplicities and the
/// Weyl dimension: the layout every basis of the module shares.
pub(super) struct DepthOrder {
    /// `(k, μ)` with `λ − μ = Σ_i k_i α_i`, in §3.2.1 order.
    pub(super) spaces: Vec<(Vec<i64>, Vec<i64>)>,
    /// Position of each weight in `spaces`.
    pub(super) index: HashMap<Vec<i64>, usize>,
    /// Weight multiplicities.
    pub(super) mult: BTreeMap<Vec<i64>, u64>,
    /// The Weyl dimension, at most `u32::MAX`.
    pub(super) weyl: usize,
}

/// The Weyl dimension of `L(λ)`, which must fit the `u32` magnetic index
/// handed to `frcore`; checked before any weight or pattern is enumerated.
pub(super) fn checked_weyl(data: &CartanData, lambda: &[i64]) -> Result<usize, LieError> {
    Ok(u32::try_from(&data.weyl_dimension(lambda)?).map_err(|_| LieError::LabelOverflow)? as usize)
}

/// The §3.2.1 weight order of `L(λ)`. Dimensions beyond `u32` are
/// [`LieError::LabelOverflow`].
pub(super) fn depth_order(data: &CartanData, lambda: &[i64]) -> Result<DepthOrder, LieError> {
    let weyl = checked_weyl(data, lambda)?;
    Ok(DepthOrder::new(
        data,
        lambda,
        data.weight_multiplicities(lambda)?,
        weyl,
    ))
}

impl DepthOrder {
    /// The §3.2.1 order of the weights `mult` of `L(λ)`, of dimension `weyl`.
    pub(super) fn new(
        data: &CartanData,
        lambda: &[i64],
        mult: BTreeMap<Vec<i64>, u64>,
        weyl: usize,
    ) -> Self {
        let rank = lambda.len();
        // Every weight lies on an unbroken root string through a higher one,
        // so lowering by simple roots from λ reaches them all.
        let mut spaces = vec![(vec![0i64; rank], lambda.to_vec())];
        let mut seen: HashSet<Vec<i64>> = HashSet::from([lambda.to_vec()]);
        let mut next = 0;
        while next < spaces.len() {
            let (k, mu) = spaces[next].clone();
            next += 1;
            for i in 0..rank {
                // Every coordinate is bounded by the weights already enumerated
                // with checked arithmetic, so this cannot overflow.
                let nu: Vec<i64> = (0..rank).map(|j| mu[j] - data.c(i, j)).collect();
                if mult.contains_key(&nu) && seen.insert(nu.clone()) {
                    let mut k = k.clone();
                    k[i] += 1;
                    spaces.push((k, nu));
                }
            }
        }
        spaces.sort_by_cached_key(|(k, _)| (k.iter().sum::<i64>(), k.clone()));
        let index = spaces
            .iter()
            .enumerate()
            .map(|(w, (_, mu))| (mu.clone(), w))
            .collect();
        Self {
            spaces,
            index,
            mult,
            weyl,
        }
    }
}

/// One weight space at depth `≥ 1` by §3.2.2–§3.2.3, with the form of §3.2.4.
fn build_space(
    data: &CartanData,
    weights: &[WeightSpace],
    index: &HashMap<Vec<i64>, usize>,
    mu: Vec<i64>,
    k: Vec<i64>,
    offset: usize,
    multiplicity: u64,
) -> Result<WeightSpace, LieError> {
    let rank = mu.len();
    let raise =
        |nu: &[i64], j: usize| -> Vec<i64> { (0..rank).map(|i| nu[i] + data.c(j, i)).collect() };
    // §3.2.2: candidates (j, source weight, q); `start[j]` is where j's block begins.
    let mut cands: Vec<(usize, usize, usize)> = Vec::new();
    let mut start = vec![None; rank];
    for (j, slot) in start.iter_mut().enumerate() {
        if let Some(&src) = index.get(&raise(&mu, j)) {
            *slot = Some(cands.len());
            cands.extend((0..weights[src].norms.len()).map(|q| (j, src, q)));
        }
    }

    // §3.2.4: ⟨f_j w, f_k w′⟩ = ⟨e_k w, e_j w′⟩ + δ_jk (μ+α_j)_j ⟨w, w′⟩, with
    // ⟨e_k b_q, e_j b_q′⟩ = Σ_p F_k[q][p] F_j[q′][p] n_q n_q′ / n_p by contravariance.
    let c = cands.len();
    let form = |s: usize, t: usize| -> Q {
        #[cfg(test)]
        tests::FORM_EVALS.with(|n| n.set(n.get() + 1));
        let (j, v1, q) = cands[s];
        let (kk, v2, q2) = cands[t];
        let (w1, w2) = (&weights[v1], &weights[v2]);
        let mut g = Q::from(0);
        if let (Some((tau, f1)), Some((_, f2))) = (&w1.lower[kk], &w2.lower[j]) {
            let n_tau = &weights[*tau].norms;
            for (p, n_p) in n_tau.iter().enumerate() {
                if f1[q][p].numerator_ref() != &0 && f2[q2][p].numerator_ref() != &0 {
                    g += &f1[q][p] * &f2[q2][p] / n_p;
                }
            }
            if g.numerator_ref() != &0 {
                g *= &w1.norms[q] * &w2.norms[q2];
            }
        }
        if j == kk && q == q2 {
            g += Q::from(w1.mu[j]) * &w1.norms[q];
        }
        g
    };

    // Greedy Gram–Schmidt over the lazily evaluated Gram matrix: a candidate
    // needs only its diagonal and its column of the kept rows `inner`; the
    // form row to later candidates is evaluated only once it is kept, so a
    // rejected candidate costs one evaluation and no C × C matrix exists.
    // `inner[a][t] = ⟨b_a, c_t⟩` serves projections here and lowering as F / n_a.
    // Rejected candidates never occur in a residual, so GS needs only kept coordinates.
    let mut gs: Vec<Vec<Q>> = Vec::with_capacity(usize::try_from(multiplicity).unwrap_or(0));
    let mut inner: Vec<Vec<Q>> = Vec::new();
    let mut norms: Vec<Q> = Vec::new();
    let mut kept_idx: Vec<usize> = Vec::new();
    for t in 0..c {
        let mut n = form(t, t);
        let scales: Vec<Q> = inner
            .iter()
            .zip(&norms)
            .map(|(row, n_l)| {
                let proj = &row[t];
                if proj.numerator_ref() == &0 {
                    return Q::from(0);
                }
                let scale = proj / n_l;
                n -= &scale * proj;
                scale
            })
            .collect();
        if n < 0 {
            return Err(ModuleError::IndefiniteForm { weight: mu }.into());
        }
        if n.numerator_ref() == &0 {
            continue;
        }
        let mut coeffs = vec![Q::from(0); gs.len() + 1];
        coeffs[gs.len()] = Q::from(1);
        for (scale, r) in scales.iter().zip(&gs) {
            if scale.numerator_ref() != &0 {
                for (x, y) in coeffs.iter_mut().zip(r) {
                    if y.numerator_ref() != &0 {
                        *x -= scale * y;
                    }
                }
            }
        }
        // All earlier candidates lie in the preceding span, orthogonal to b_a.
        #[cfg(test)]
        tests::RETAINED_ROW_SLOTS.with(|n| n.set(n.get() + c));
        let mut row = vec![Q::from(0); c];
        row[t] = n.clone();
        for col in t + 1..c {
            row[col] = form(t, col);
            for (scale, prev) in scales.iter().zip(&inner) {
                if scale.numerator_ref() != &0 && prev[col].numerator_ref() != &0 {
                    row[col] -= scale * &prev[col];
                }
            }
        }
        gs.push(coeffs);
        inner.push(row);
        norms.push(n);
        kept_idx.push(t);
    }
    if u64::try_from(norms.len()).ok() != Some(multiplicity) {
        return Err(ModuleError::RankMismatch {
            weight: mu,
            rank: norms.len(),
            multiplicity,
        }
        .into());
    }

    // §3.3: F[a][b] = ⟨b_a, c_(j,b)⟩ / n_a.
    let lower = (0..rank)
        .map(|j| {
            let s0 = start[j]?;
            let src = cands[s0].1;
            let f = inner
                .iter()
                .zip(&norms)
                .map(|(row, n_a)| {
                    row[s0..s0 + weights[src].norms.len()]
                        .iter()
                        .map(|dot| dot / n_a)
                        .collect()
                })
                .collect();
            Some((src, f))
        })
        .collect();
    for row in &mut gs {
        row.reserve_exact(norms.len() - row.len());
        row.resize(norms.len(), Q::from(0));
    }
    let kept = kept_idx.iter().map(|&t| (cands[t].0, cands[t].2)).collect();
    Ok(WeightSpace {
        mu,
        k,
        offset,
        norms,
        kept,
        recipe: gs,
        lower,
    })
}

#[cfg(test)]
pub(super) mod tests;
