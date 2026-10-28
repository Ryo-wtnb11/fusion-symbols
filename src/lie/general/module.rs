//! The irreducible module `L(λ)` in the orthonormal basis of
//! `docs/gauge_exceptional.md` §3–§4, built exactly from the contravariant
//! form and the Cartan matrix alone.
//!
//! The module is kept in the unnormalized orthogonal basis `b_k` of §4.3: the
//! norms `n_k = ⟨b_k, b_k⟩`, the Gram–Schmidt coefficients over the kept
//! candidates, and the rational lowering matrices `F` of §4.5. Every ladder
//! entry of the orthonormal basis `u_k = b_k / √n_k` is then the single signed
//! square root `F[a, b] · √(n_a / n_b)`, returned as a [`SignedSqrtRational`](crate::exact::SignedSqrtRational);
//! no two square roots are ever added.

use std::collections::{HashMap, HashSet};

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{Signed, Zero};

use super::roots::CartanData;
use crate::lie::{GenerationError, LieError};

pub(super) type Q = Ratio<BigInt>;

/// Estimated bytes of a `Vec<i64>` with its header.
pub(super) fn labels_bytes(v: &[i64]) -> usize {
    std::mem::size_of::<Vec<i64>>() + std::mem::size_of_val(v)
}

/// Estimated heap bytes of one rational: its numerator and denominator digits.
pub(super) fn q_heap(q: &Q) -> usize {
    let words = |x: &BigInt| x.bits().div_ceil(64) as usize;
    8 * (words(q.numer()) + words(q.denom()))
}

/// Estimated bytes of a vector of rationals, capacity-based.
pub(super) fn qs_bytes(v: &Vec<Q>) -> usize {
    v.capacity() * std::mem::size_of::<Q>() + v.iter().map(q_heap).sum::<usize>()
}

/// Estimated bytes of a rational matrix stored as rows.
pub(super) fn qrows_bytes(m: &Vec<Vec<Q>>) -> usize {
    m.capacity() * std::mem::size_of::<Vec<Q>>() + m.iter().map(qs_bytes).sum::<usize>()
}

/// `L(λ)` in the §4 basis.
pub(super) struct IrrepModule {
    /// Weight spaces in the §3 order.
    pub(super) weights: Vec<WeightSpace>,
    index: HashMap<Vec<i64>, usize>,
    pub(super) dim: usize,
}

pub(super) struct WeightSpace {
    /// The weight `μ` in Dynkin coordinates.
    pub(super) mu: Vec<i64>,
    /// `λ − μ = Σ_i k_i α_i` (§3).
    pub(super) k: Vec<i64>,
    /// Magnetic index of `b_0`; this space holds `offset .. offset + norms.len()`.
    pub(super) offset: usize,
    /// `n_k = ⟨b_k, b_k⟩ > 0`.
    pub(super) norms: Vec<Q>,
    /// The kept candidates `c_t = f_j b_q^{μ+α_j}` as `(j, q)`, in §4.2 order.
    pub(super) kept: Vec<(usize, usize)>,
    /// `b_k = Σ_{t ≤ k} gs[k][t] c_t`, with `gs[k][k] = 1` (§4.3).
    pub(super) gs: Vec<Vec<Q>>,
    /// Per `j`: the index of `μ + α_j` and `F` with `f_j b_b = Σ_a F[a][b] b_a`.
    pub(super) lower: Vec<Option<(usize, Vec<Vec<Q>>)>>,
}

/// An internal inconsistency between the construction and its gates (§4.2).
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
    /// A Gram–Schmidt residual has negative norm, so the form is not positive definite.
    IndefiniteForm { weight: Vec<i64> },
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
                    + qrows_bytes(&s.gs)
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

    /// Builds `L(λ)` for a dominant `λ`. Dimensions beyond `u32`, the
    /// magnetic-index type handed to `frcore`, are [`LieError::LabelOverflow`].
    pub(super) fn new(data: &CartanData, lambda: &[i64]) -> Result<Self, LieError> {
        let weyl = u32::try_from(&data.weyl_dimension(lambda)?)
            .map_err(|_| LieError::LabelOverflow)? as usize;
        let mult = data.weight_multiplicities(lambda)?;
        let rank = lambda.len();

        // Every weight lies on an unbroken root string through a higher one,
        // so lowering by simple roots from λ reaches them all.
        let mut order = vec![(vec![0i64; rank], lambda.to_vec())];
        let mut seen: HashSet<Vec<i64>> = HashSet::from([lambda.to_vec()]);
        let mut next = 0;
        while next < order.len() {
            let (k, mu) = order[next].clone();
            next += 1;
            for i in 0..rank {
                // Every coordinate is bounded by the weights Freudenthal already
                // enumerated with checked arithmetic, so this cannot overflow.
                let nu: Vec<i64> = (0..rank).map(|j| mu[j] - data.c(i, j)).collect();
                if mult.contains_key(&nu) && seen.insert(nu.clone()) {
                    let mut k = k.clone();
                    k[i] += 1;
                    order.push((k, nu));
                }
            }
        }
        order.sort_by_cached_key(|(k, _)| (k.iter().sum::<i64>(), k.clone()));
        let index: HashMap<Vec<i64>, usize> = order
            .iter()
            .enumerate()
            .map(|(w, (_, mu))| (mu.clone(), w))
            .collect();

        let mut weights: Vec<WeightSpace> = Vec::with_capacity(order.len());
        let mut offset = 0;
        for (k, mu) in order {
            let space = if weights.is_empty() {
                WeightSpace {
                    lower: vec![None; rank],
                    mu,
                    k,
                    offset,
                    norms: vec![Q::from_integer(1.into())],
                    kept: Vec::new(),
                    gs: vec![Vec::new()],
                }
            } else {
                let multiplicity = mult.get(&mu).copied().unwrap_or(0);
                build_space(data, &weights, &index, mu, k, offset, multiplicity)?
            };
            offset += space.norms.len();
            weights.push(space);
        }
        if offset != weyl {
            return Err(ModuleError::DimensionMismatch {
                built: offset,
                weyl,
            }
            .into());
        }
        Ok(Self {
            weights,
            index,
            dim: offset,
        })
    }

    pub(super) fn weight_index(&self, mu: &[i64]) -> Option<usize> {
        self.index.get(mu).copied()
    }

    /// `⟨u_a, f_j u_b⟩` with `u_a` the `a`-th basis vector of weight space `w`
    /// and `u_b` the `b`-th of `μ_w + α_j` (§4.5).
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
        let ratio = &space.norms[a] / &self.weights[*src].norms[b];
        SignedSqrtRational::from_prefactor_radical(f[a][b].clone(), ratio)
    }
}

/// One weight space at depth `≥ 1` by §4.1–§4.3, with the form of §4.4.
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
    // §4.1: candidates (j, source weight, q); `start[j]` is where j's block begins.
    let mut cands: Vec<(usize, usize, usize)> = Vec::new();
    let mut start = vec![None; rank];
    for (j, slot) in start.iter_mut().enumerate() {
        if let Some(&src) = index.get(&raise(&mu, j)) {
            *slot = Some(cands.len());
            cands.extend((0..weights[src].norms.len()).map(|q| (j, src, q)));
        }
    }

    // §4.4: ⟨f_j w, f_k w′⟩ = ⟨e_k w, e_j w′⟩ + δ_jk (μ+α_j)_j ⟨w, w′⟩, with
    // ⟨e_k b_q, e_j b_q′⟩ = Σ_p F_k[q][p] F_j[q′][p] n_q n_q′ / n_p by contravariance.
    let c = cands.len();
    let mut gram = vec![vec![Q::zero(); c]; c];
    for s in 0..c {
        let (j, v1, q) = cands[s];
        for t in s..c {
            let (kk, v2, q2) = cands[t];
            let (w1, w2) = (&weights[v1], &weights[v2]);
            let mut g = Q::zero();
            if let (Some((tau, f1)), Some((_, f2))) = (&w1.lower[kk], &w2.lower[j]) {
                let n_tau = &weights[*tau].norms;
                for (p, n_p) in n_tau.iter().enumerate() {
                    if !f1[q][p].is_zero() && !f2[q2][p].is_zero() {
                        g += &f1[q][p] * &f2[q2][p] / n_p;
                    }
                }
                if !g.is_zero() {
                    g *= &w1.norms[q] * &w2.norms[q2];
                }
            }
            if j == kk && q == q2 {
                g += Q::from_integer(w1.mu[j].into()) * &w1.norms[q];
            }
            gram[t][s] = g.clone();
            gram[s][t] = g;
        }
    }

    // Cache ⟨b_a, c_t⟩ once: GS uses it for projections and lowering uses it / n_a.
    // Rejected candidates never occur in a residual, so GS needs only kept coordinates.
    let mut gs: Vec<Vec<Q>> = Vec::with_capacity(usize::try_from(multiplicity).unwrap_or(0));
    let mut inner: Vec<Vec<Q>> = Vec::new();
    let mut norms: Vec<Q> = Vec::new();
    let mut kept_idx: Vec<usize> = Vec::new();
    for t in 0..c {
        let mut n = gram[t][t].clone();
        let scales: Vec<Q> = inner
            .iter()
            .zip(&norms)
            .map(|(row, n_l)| {
                let proj = &row[t];
                if proj.is_zero() {
                    return Q::zero();
                }
                let scale = proj / n_l;
                n -= &scale * proj;
                scale
            })
            .collect();
        if n.is_negative() {
            return Err(ModuleError::IndefiniteForm { weight: mu }.into());
        }
        if n.is_zero() {
            continue;
        }
        let mut coeffs = vec![Q::zero(); gs.len() + 1];
        coeffs[gs.len()] = Q::from_integer(1.into());
        for (scale, r) in scales.iter().zip(&gs) {
            if !scale.is_zero() {
                for (x, y) in coeffs.iter_mut().zip(r) {
                    if !y.is_zero() {
                        *x -= scale * y;
                    }
                }
            }
        }
        // All earlier candidates lie in the preceding span, orthogonal to b_a.
        let mut row = vec![Q::zero(); c];
        row[t] = n.clone();
        for col in t + 1..c {
            row[col] = gram[t][col].clone();
            for (scale, prev) in scales.iter().zip(&inner) {
                if !scale.is_zero() && !prev[col].is_zero() {
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

    // §4.5: F[a][b] = ⟨b_a, c_(j,b)⟩ / n_a.
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
        row.resize(norms.len(), Q::zero());
    }
    let kept = kept_idx.iter().map(|&t| (cands[t].0, cands[t].2)).collect();
    Ok(WeightSpace {
        mu,
        k,
        offset,
        norms,
        kept,
        gs,
        lower,
    })
}

#[cfg(test)]
mod tests;
