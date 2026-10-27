//! Root-system combinatorics read from a Cartan matrix alone.
//!
//! One code path serves every finite-type Cartan matrix in the convention of
//! `src/lie/group.rs::cartan`, `cartan(i, j) = ⟨α_i, α_j∨⟩`
//! (`docs/gauge_exceptional.md` §2.1). Weights are in Dynkin coordinates, so
//! subtracting a simple root is `(μ − α_i)_j = μ_j − cartan(i, j)`; roots are
//! in simple-root coefficient form.
//!
//! - Positive roots by closure of the simple roots under simple reflections.
//! - The symmetrizer `d_j = (α_j, α_j)/2` with long roots of length² 2, and the
//!   form `(α_i, α_j) = d_j cartan(i, j)` (§2.2). Internally the form is carried
//!   at the integer scale `L = 1 / min_j d_j`, so every quantity stays integral.
//! - `2ρ` in simple roots and `2ρ∨` in simple coroots (§6.2); `ρ` itself is
//!   `(1, …, 1)` in Dynkin coordinates.
//! - The Weyl dimension (§9.1), exact.
//! - Dominant-weight multiplicities by Freudenthal's recursion, exact (§4.2
//!   gate, §9.2).
//! - Tensor-product multiplicities by Brauer–Klimyk (§5.2 gate, §9.3), with
//!   each shifted weight reflected into the dominant chamber; the Weyl group is
//!   never enumerated.
//! - The dual label `−w₀λ` (§6.1) and the closed-form sign
//!   `(−1)^{⟨λ, 2ρ∨⟩}`, which is `κ_a` for a self-dual irrep (§6.2, §6.3).
//!
//! Arithmetic is checked; anything outside the integer range is
//! [`LieError::LabelOverflow`].

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Zero};

use crate::lie::group::cartan;
use crate::lie::{LieError, RootSystem};

/// The root data of one simple root system, derived from its Cartan matrix.
pub(crate) struct CartanData {
    rank: usize,
    /// Row-major, `cartan[i * rank + j] = ⟨α_i, α_j∨⟩`.
    cartan: Vec<i64>,
    /// `L · d_j`, coprime integers; `L = max(sym)` since the longest root has `d = 1`.
    sym: Vec<i64>,
    positive: Vec<PositiveRoot>,
    two_rho_vee: Vec<i64>,
}

struct PositiveRoot {
    coeffs: Vec<i64>,
    dynkin: Vec<i64>,
    /// `form[i] = coeffs[i] · sym[i]`, so `L (μ, α) = Σ_i form[i] μ_i`.
    form: Vec<i64>,
}

impl CartanData {
    pub(crate) fn new(root: RootSystem) -> Self {
        let r = root.rank();
        let matrix = (0..r)
            .flat_map(|i| (0..r).map(move |j| cartan(root, i, j)))
            .collect();
        Self::from_cartan(r, matrix)
    }

    fn from_cartan(rank: usize, cartan: Vec<i64>) -> Self {
        let coeffs = positive_root_coeffs(rank, &cartan);
        // The coroots of a root system are the roots of the transposed matrix.
        let transposed: Vec<i64> = (0..rank * rank)
            .map(|n| cartan[(n % rank) * rank + n / rank])
            .collect();
        let two_rho_vee = coefficient_sum(rank, &positive_root_coeffs(rank, &transposed));
        let sym = symmetrizer(rank, &cartan);
        let positive = coeffs
            .into_iter()
            .map(|c| PositiveRoot {
                dynkin: root_dynkin(rank, &cartan, &c),
                form: c.iter().zip(&sym).map(|(c, s)| c * s).collect(),
                coeffs: c,
            })
            .collect();
        Self {
            rank,
            cartan,
            sym,
            positive,
            two_rho_vee,
        }
    }

    pub(super) fn c(&self, i: usize, j: usize) -> i64 {
        self.cartan[i * self.rank + j]
    }

    fn check_label(&self, lambda: &[i64]) -> Result<(), LieError> {
        if lambda.len() != self.rank || lambda.iter().any(|&x| x < 0) {
            return Err(LieError::InvalidLabel);
        }
        Ok(())
    }

    /// Positive roots in simple-root coefficient form, by height then lexicographically.
    #[cfg(test)]
    pub(crate) fn positive_roots(&self) -> impl Iterator<Item = &[i64]> {
        self.positive.iter().map(|a| a.coeffs.as_slice())
    }

    /// `d_j = (α_j, α_j) / 2`, normalized so long roots have `d = 1` (§2.2).
    #[cfg(test)]
    pub(crate) fn symmetrizer(&self) -> Vec<Ratio<i64>> {
        let long = self.long();
        self.sym.iter().map(|&s| Ratio::new(s, long)).collect()
    }

    /// The invariant form on simple roots, `(α_i, α_j) = d_j cartan(i, j)` (§2.2).
    #[cfg(test)]
    pub(crate) fn form(&self, i: usize, j: usize) -> Ratio<i64> {
        Ratio::new(self.sym[j] * self.c(i, j), self.long())
    }

    #[cfg(test)]
    fn long(&self) -> i64 {
        self.sym.iter().copied().max().unwrap_or(1)
    }

    /// `2ρ = Σ_{α>0} α` in simple roots.
    #[cfg(test)]
    pub(crate) fn two_rho(&self) -> Vec<i64> {
        let coeffs: Vec<Vec<i64>> = self.positive.iter().map(|a| a.coeffs.clone()).collect();
        coefficient_sum(self.rank, &coeffs)
    }

    /// `2ρ∨ = Σ_{α>0} α∨` in simple coroots (§6.2), so `⟨λ, 2ρ∨⟩ = Σ_i c_i λ_i`.
    #[cfg(test)]
    pub(crate) fn two_rho_vee(&self) -> &[i64] {
        &self.two_rho_vee
    }

    /// `dim L(λ) = ∏_{α>0} (λ+ρ, α) / (ρ, α)` (§9.1), in checked `u128` and,
    /// only when that overflows, in big rationals.
    pub(crate) fn weyl_dimension(&self, lambda: &[i64]) -> Result<BigInt, LieError> {
        self.check_label(lambda)?;
        let factors = self
            .positive
            .iter()
            .map(|alpha| {
                let mut num = 0u128;
                let mut den = 0u128;
                for (&f, &l) in alpha.form.iter().zip(lambda) {
                    let f = u128::try_from(f).map_err(|_| LieError::LabelOverflow)?;
                    let shifted = u128::try_from(l).map_err(|_| LieError::LabelOverflow)? + 1;
                    num = shifted
                        .checked_mul(f)
                        .and_then(|t| num.checked_add(t))
                        .ok_or(LieError::LabelOverflow)?;
                    den += f;
                }
                Ok((num, den))
            })
            .collect::<Result<Vec<_>, LieError>>()?;
        if let Some(d) = crate::exact::checked_integral_quotient(factors.iter().copied()) {
            return Ok(BigInt::from(d));
        }
        Ok(factors
            .into_iter()
            .fold(Ratio::<BigInt>::one(), |acc, (n, d)| {
                acc * Ratio::new(BigInt::from(n), BigInt::from(d))
            })
            .to_integer())
    }

    /// `s_i w`, in place: `w_j ← w_j − w_i cartan(i, j)`.
    fn reflect(&self, w: &mut [i64], i: usize) -> Result<(), LieError> {
        let a = w[i];
        for (j, x) in w.iter_mut().enumerate() {
            *x = a
                .checked_mul(self.c(i, j))
                .and_then(|t| x.checked_sub(t))
                .ok_or(LieError::LabelOverflow)?;
        }
        Ok(())
    }

    /// The dominant Weyl conjugate of `w` and the parity of the number of simple
    /// reflections used. Each reflection at a negative coordinate shortens the
    /// Weyl element still to apply, so this takes at most `|Φ+|` steps.
    fn to_dominant(&self, mut w: Vec<i64>) -> Result<(Vec<i64>, bool), LieError> {
        let mut odd = false;
        while let Some(i) = w.iter().position(|&x| x < 0) {
            self.reflect(&mut w, i)?;
            odd = !odd;
        }
        Ok((w, odd))
    }

    /// The Weyl orbit of a dominant weight: lower by `s_i` wherever `w_i > 0`.
    fn orbit(&self, dominant: &[i64]) -> Result<Vec<Vec<i64>>, LieError> {
        let mut seen: HashSet<Vec<i64>> = HashSet::from([dominant.to_vec()]);
        let mut out = vec![dominant.to_vec()];
        let mut next = 0;
        while next < out.len() {
            let w = out[next].clone();
            next += 1;
            for i in (0..self.rank).filter(|&i| w[i] > 0) {
                let mut v = w.clone();
                self.reflect(&mut v, i)?;
                if seen.insert(v.clone()) {
                    out.push(v);
                }
            }
        }
        Ok(out)
    }

    /// The dual label `−w₀λ` (§6.1): `w₀λ` is the antidominant conjugate of
    /// `λ`, so `−w₀λ` is the dominant conjugate of `−λ`.
    pub(crate) fn dual(&self, lambda: &[i64]) -> Result<Vec<i64>, LieError> {
        self.check_label(lambda)?;
        let negated = lambda.iter().map(|&x| -x).collect();
        Ok(self.to_dominant(negated)?.0)
    }

    /// `(−1)^{⟨λ, 2ρ∨⟩}` (§6.2). This is `κ_a` only for a self-dual `λ`; for a
    /// complex E6 irrep `κ_a` is the gauge-derived `σ_a σ_ā` (§6.3).
    pub(crate) fn two_rho_vee_sign(&self, lambda: &[i64]) -> Result<i8, LieError> {
        self.check_label(lambda)?;
        let odd = lambda
            .iter()
            .zip(&self.two_rho_vee)
            .filter(|&(l, c)| l % 2 == 1 && c % 2 == 1)
            .count()
            % 2
            == 1;
        Ok(if odd { -1 } else { 1 })
    }

    /// The dominant weights of `L(λ)` with their multiplicities, by depth
    /// `Σ k_i` of `λ − μ = Σ k_i α_i`, then by `k` (Freudenthal, Humphreys §22.3):
    ///
    /// `m(μ) Σ_i k_i d_i (λ_i + μ_i + 2) = 2 Σ_{α>0} Σ_{j≥1} m(μ + jα) (μ + jα, α)`,
    ///
    /// whose left factor is `(λ+ρ, λ+ρ) − (μ+ρ, μ+ρ)`. Both sides are carried
    /// at scale `L`, so the recursion is integral.
    pub(crate) fn dominant_multiplicities(
        &self,
        lambda: &[i64],
    ) -> Result<Vec<(Vec<i64>, u64)>, LieError> {
        self.check_label(lambda)?;
        for &x in lambda {
            // λ + 2ρ enters the Casimir difference.
            x.checked_add(2).ok_or(LieError::LabelOverflow)?;
        }
        // Dominant weights below λ are reached by subtracting positive roots
        // through dominant weights only (Stembridge 1998, Cor. 2.7).
        let mut found: HashSet<Vec<i64>> = HashSet::from([lambda.to_vec()]);
        let mut doms = vec![(vec![0i64; self.rank], lambda.to_vec())];
        let mut next = 0;
        while next < doms.len() {
            let (k, mu) = doms[next].clone();
            next += 1;
            for alpha in &self.positive {
                let nu = sub(&mu, &alpha.dynkin)?;
                if nu.iter().all(|&x| x >= 0) && found.insert(nu.clone()) {
                    doms.push((add(&k, &alpha.coeffs)?, nu));
                }
            }
        }
        doms.sort_by_cached_key(|(k, _)| (k.iter().sum::<i64>(), k.clone()));

        let mut mult: HashMap<Vec<i64>, u64> = HashMap::with_capacity(doms.len());
        let mut out = Vec::with_capacity(doms.len());
        for (k, mu) in doms {
            let m = if k.iter().all(|&x| x == 0) {
                1
            } else {
                self.freudenthal_step(lambda, &k, &mu, &mult)?
            };
            mult.insert(mu.clone(), m);
            out.push((mu, m));
        }
        Ok(out)
    }

    fn freudenthal_step(
        &self,
        lambda: &[i64],
        k: &[i64],
        mu: &[i64],
        mult: &HashMap<Vec<i64>, u64>,
    ) -> Result<u64, LieError> {
        let mut denom = 0i128;
        for i in 0..self.rank {
            let term = i128::from(lambda[i]) + i128::from(mu[i]) + 2;
            denom = term
                .checked_mul(i128::from(k[i]) * i128::from(self.sym[i]))
                .and_then(|t| denom.checked_add(t))
                .ok_or(LieError::LabelOverflow)?;
        }
        let mut num = 0i128;
        for alpha in &self.positive {
            let mut shifted = mu.to_vec();
            loop {
                shifted = add(&shifted, &alpha.dynkin)?;
                // μ + jα is higher than μ, so its dominant conjugate is already known.
                let Some(&m) = mult.get(&self.to_dominant(shifted.clone())?.0) else {
                    break;
                };
                let pairing = dot(&alpha.form, &shifted)?;
                num = pairing
                    .checked_mul(2 * i128::from(m))
                    .and_then(|t| num.checked_add(t))
                    .ok_or(LieError::LabelOverflow)?;
            }
        }
        debug_assert!(denom > 0 && num % denom == 0, "Freudenthal divides exactly");
        u64::try_from(num / denom).map_err(|_| LieError::LabelOverflow)
    }

    /// Every weight of `L(λ)` with its multiplicity: the Weyl orbits of the
    /// dominant weights.
    pub(crate) fn weight_multiplicities(
        &self,
        lambda: &[i64],
    ) -> Result<BTreeMap<Vec<i64>, u64>, LieError> {
        let mut out = BTreeMap::new();
        for (mu, m) in self.dominant_multiplicities(lambda)? {
            for w in self.orbit(&mu)? {
                out.insert(w, m);
            }
        }
        Ok(out)
    }

    /// `N^c_{ab}` for every `c`, by Brauer–Klimyk: for each weight `ν` of the
    /// smaller factor, reflect `a + ν + ρ` into the dominant chamber; a
    /// coordinate `0` there means a fixed reflection (contributes nothing),
    /// otherwise add `det(w) m(ν)` at `w(a + ν + ρ) − ρ`.
    pub(crate) fn tensor_product(
        &self,
        a: &[i64],
        b: &[i64],
    ) -> Result<BTreeMap<Vec<i64>, u32>, LieError> {
        self.check_label(a)?;
        self.check_label(b)?;
        for (&x, &y) in a.iter().zip(b) {
            // The largest shifted weight is a + b + ρ.
            x.checked_add(y)
                .and_then(|s| s.checked_add(1))
                .ok_or(LieError::LabelOverflow)?;
        }
        let (a, b) = if self.weyl_dimension(a)? < self.weyl_dimension(b)? {
            (b, a)
        } else {
            (a, b)
        };
        let a_rho: Vec<i64> = a.iter().map(|&x| x + 1).collect();
        let mut acc: BTreeMap<Vec<i64>, i64> = BTreeMap::new();
        for (nu, m) in self.weight_multiplicities(b)? {
            let (dom, odd) = self.to_dominant(add(&a_rho, &nu)?)?;
            if dom.contains(&0) {
                continue;
            }
            let m = i64::try_from(m).map_err(|_| LieError::LabelOverflow)?;
            let c = dom.iter().map(|&x| x - 1).collect();
            let entry = acc.entry(c).or_insert(0);
            *entry = entry
                .checked_add(if odd { -m } else { m })
                .ok_or(LieError::LabelOverflow)?;
        }
        let mut out = BTreeMap::new();
        for (c, n) in acc {
            debug_assert!(n >= 0, "Brauer–Klimyk multiplicity is nonnegative");
            if n != 0 {
                out.insert(c, u32::try_from(n).map_err(|_| LieError::LabelOverflow)?);
            }
        }
        Ok(out)
    }
}

/// Simple roots closed under `s_i α = α − ⟨α, α_i∨⟩ α_i`, keeping positive
/// results: every positive non-simple root has some `⟨α, α_i∨⟩ > 0`, and
/// `s_i α` is then a positive root of smaller height.
fn positive_root_coeffs(rank: usize, cartan: &[i64]) -> Vec<Vec<i64>> {
    let simple = (0..rank).map(|i| {
        let mut c = vec![0; rank];
        c[i] = 1;
        c
    });
    let mut queue: VecDeque<Vec<i64>> = simple.collect();
    let mut seen: HashSet<Vec<i64>> = queue.iter().cloned().collect();
    while let Some(c) = queue.pop_front() {
        let a = root_dynkin(rank, cartan, &c);
        for i in 0..rank {
            let mut s = c.clone();
            s[i] -= a[i];
            if s.iter().all(|&x| x >= 0) && seen.insert(s.clone()) {
                queue.push_back(s);
            }
        }
    }
    let mut roots: Vec<Vec<i64>> = seen.into_iter().collect();
    roots.sort_by_cached_key(|c| (c.iter().sum::<i64>(), c.clone()));
    roots
}

/// `⟨α, α_j∨⟩ = Σ_i c_i cartan(i, j)`.
fn root_dynkin(rank: usize, cartan: &[i64], c: &[i64]) -> Vec<i64> {
    (0..rank)
        .map(|j| (0..rank).map(|i| c[i] * cartan[i * rank + j]).sum())
        .collect()
}

fn coefficient_sum(rank: usize, roots: &[Vec<i64>]) -> Vec<i64> {
    (0..rank)
        .map(|i| roots.iter().map(|c| c[i]).sum())
        .collect()
}

/// `L d_j` as coprime integers, from `d_i cartan(j, i) = d_j cartan(i, j)`
/// along the edges of each connected component.
fn symmetrizer(rank: usize, cartan: &[i64]) -> Vec<i64> {
    // Zero marks a node not yet reached; every reached d is positive.
    let mut d = vec![Ratio::<i64>::zero(); rank];
    for start in 0..rank {
        if !d[start].is_zero() {
            continue;
        }
        d[start] = Ratio::one();
        let mut stack = vec![start];
        while let Some(i) = stack.pop() {
            for j in 0..rank {
                let (cij, cji) = (cartan[i * rank + j], cartan[j * rank + i]);
                if j != i && cij != 0 && d[j].is_zero() {
                    d[j] = d[i] * Ratio::new(cji, cij);
                    stack.push(j);
                }
            }
        }
    }
    // Finite type: every ratio d_i / d_j is 1, 2 or 3, so d / min d is integral.
    let min = d.iter().copied().min().unwrap_or_else(Ratio::one);
    d.iter().map(|x| (x / min).to_integer()).collect()
}

fn add(a: &[i64], b: &[i64]) -> Result<Vec<i64>, LieError> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x.checked_add(y).ok_or(LieError::LabelOverflow))
        .collect()
}

fn sub(a: &[i64], b: &[i64]) -> Result<Vec<i64>, LieError> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x.checked_sub(y).ok_or(LieError::LabelOverflow))
        .collect()
}

fn dot(a: &[i64], b: &[i64]) -> Result<i128, LieError> {
    a.iter().zip(b).try_fold(0i128, |s, (&x, &y)| {
        i128::from(x)
            .checked_mul(i128::from(y))
            .and_then(|t| s.checked_add(t))
            .ok_or(LieError::LabelOverflow)
    })
}

#[cfg(test)]
mod tests;
