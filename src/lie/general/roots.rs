//! Exact finite root-system mathematics with structural classical coordinate branches.
//!
//! One producer serves each exact operation, using the Cartan convention of
//! `src/lie/group.rs::cartan`, `cartan(i, j) = ⟨α_i, α_j∨⟩`
//! (`docs/gauge_cartan.md` §2.1). Weights are in Dynkin coordinates, so
//! subtracting a simple root is `(μ − α_i)_j = μ_j − cartan(i, j)`; roots are
//! in simple-root coefficient form.
//!
//! - Classical positive roots by their closed lists; exceptional roots by simple-reflection closure.
//! - The symmetrizer `d_j = (α_j, α_j)/2` with long roots of length² 2, and the
//!   form `(α_i, α_j) = d_j cartan(i, j)` (§2.2). Internally the form is carried
//!   at the integer scale `L = 1 / min_j d_j`, so every quantity stays integral.
//! - `2ρ` in simple roots and `2ρ∨` in simple coroots (§5.2); `ρ` itself is
//!   `(1, …, 1)` in Dynkin coordinates.
//! - The Weyl dimension (§8 oracle 1), exact.
//! - Dominant-weight multiplicities by Freudenthal's recursion, exact (§3.2.3
//!   gate).
//! - Tensor-product multiplicities by Brauer–Klimyk (§4.2 gate), with
//!   each shifted weight reflected into the dominant chamber; the Weyl group is
//!   never enumerated.
//! - The dual label `−w₀λ` (§5.1) and the closed-form sign
//!   `(−1)^{⟨λ, 2ρ∨⟩}`, which is `κ_a` for a self-dual irrep (§5.2, §5.3).
//!
//! Arithmetic is checked; anything outside the integer range is
//! [`LieError::LabelOverflow`].

use malachite_base::num::arithmetic::traits::{DivRound, UnsignedAbs};
use malachite_base::rounding_modes::RoundingMode::Exact;
use malachite_nz::natural::Natural;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
pub(crate) mod classical;
use classical::Series;

use malachite_nz::integer::Integer;
use malachite_q::Rational;

use crate::lie::group::cartan;
use crate::lie::gt::gt_enumerate;
use crate::lie::{LieError, RootSystem};

/// The root data of one simple root system, derived from its Cartan matrix.
#[derive(Debug)]
pub(crate) struct CartanData {
    pub(crate) root: RootSystem,
    rank: usize,
    /// Row-major, `cartan[i * rank + j] = ⟨α_i, α_j∨⟩`.
    cartan: Vec<i64>,
    /// `L · d_j`, coprime integers; `L = max(sym)` since the longest root has `d = 1`.
    sym: Vec<i64>,
    positive: Vec<PositiveRoot>,
    two_rho_vee: Vec<i64>,
    classical_rho: Vec<i64>,
    /// Product of the scaled Weyl denominators, in the same coordinates as the factors.
    weyl_denominator: Natural,
}

#[derive(Debug)]
struct PositiveRoot {
    coeffs: Vec<i64>,
    dynkin: Vec<i64>,
    epsilon: Vec<i64>,
    sparse_epsilon: Vec<(usize, i64)>,
    /// `form[i] = coeffs[i] · sym[i]`, so `L (μ, α) = Σ_i form[i] μ_i`.
    form: Vec<i64>,
}

impl CartanData {
    pub(crate) fn new(root: RootSystem) -> Arc<Self> {
        #[cfg(test)]
        EVALUATIONS.with(|n| {
            let (r, d) = n.get();
            n.set((r + 1, d));
        });
        let r = root.rank();
        let matrix = (0..r)
            .flat_map(|i| (0..r).map(move |j| cartan(root, i, j)))
            .collect();
        Arc::new(Self::from_cartan(root, r, matrix))
    }

    fn from_cartan(root: RootSystem, rank: usize, cartan: Vec<i64>) -> Self {
        let series = match root {
            RootSystem::B(_) => Some(Series::B),
            RootSystem::C(_) => Some(Series::C),
            RootSystem::D(_) => Some(Series::D),
            _ => None,
        };
        let mut coeffs = if let Some(series) = series {
            // Classical roots have at most two nonzero epsilon coordinates.
            classical::positive_roots(series, rank)
                .iter()
                .map(|w| {
                    classical::simple_root_coeffs(
                        series,
                        &w.iter().map(|x| 2 * x).collect::<Vec<_>>(),
                    )
                    .expect("root coefficients fit")
                    .expect("positive roots have nonnegative integral coefficients")
                })
                .collect()
        } else if matches!(root, RootSystem::A(_)) {
            // A positive roots are contiguous strings of simple roots.
            (0..rank)
                .flat_map(|i| {
                    (i + 1..=rank).map(move |j| {
                        (0..rank)
                            .map(|k| i64::from(i <= k && k < j))
                            .collect::<Vec<_>>()
                    })
                })
                .collect()
        } else {
            positive_root_coeffs(rank, &cartan)
        };
        coeffs.sort_by_cached_key(|c| (c.iter().sum::<i64>(), c.clone()));
        let sym = symmetrizer(rank, &cartan);
        let sparse = cartan
            .chunks(rank.max(1))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(_, c)| **c != 0)
                    .map(|(j, &c)| (j, c))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let positive: Vec<_> = coeffs
            .into_iter()
            .map(|c| {
                let mut dynkin = vec![0; rank];
                for (&ci, row) in c.iter().zip(&sparse) {
                    for &(j, cij) in row {
                        dynkin[j] += ci * cij;
                    }
                }
                let epsilon: Vec<i64> = series.map_or_else(Vec::new, |series| {
                    classical::dynkin_to_two_partition(series, &dynkin)
                        .expect("root coordinates fit")
                        .into_iter()
                        .map(|x| x / 2)
                        .collect()
                });
                PositiveRoot {
                    form: c.iter().zip(&sym).map(|(c, s)| c * s).collect(),
                    coeffs: c,
                    dynkin,
                    sparse_epsilon: epsilon
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, x)| *x != 0)
                        .collect(),
                    epsilon,
                }
            })
            .collect();
        // alpha^vee has coefficient 2*c_i*sym_i / (L*(alpha,alpha)) on alpha_i^vee.
        let mut two_rho_vee = vec![0; rank];
        for alpha in &positive {
            let norm: i64 = alpha
                .form
                .iter()
                .zip(&alpha.dynkin)
                .map(|(a, b)| a * b)
                .sum();
            for (sum, &f) in two_rho_vee.iter_mut().zip(&alpha.form) {
                debug_assert_eq!(2 * f % norm, 0);
                *sum += 2 * f / norm;
            }
        }
        let classical_rho = series.map_or_else(Vec::new, |s| classical::two_rho(s, rank));
        let weyl_denominator = positive
            .iter()
            .map(|alpha| {
                Natural::from(if series.is_some() {
                    alpha
                        .sparse_epsilon
                        .iter()
                        .map(|&(i, x)| i128::from(classical_rho[i]) * i128::from(x))
                        .sum::<i128>() as u128
                } else {
                    alpha.form.iter().map(|&x| x as u128).sum()
                })
            })
            .product();
        Self {
            root,
            rank,
            cartan,
            sym,
            positive,
            two_rho_vee,
            classical_rho,
            weyl_denominator,
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
    pub(crate) fn symmetrizer(&self) -> Vec<Rational> {
        let long = self.long();
        self.sym
            .iter()
            .map(|&s| Rational::from_signeds(s, long))
            .collect()
    }

    /// The invariant form on simple roots, `(α_i, α_j) = d_j cartan(i, j)` (§2.2).
    #[cfg(test)]
    pub(crate) fn form(&self, i: usize, j: usize) -> Rational {
        Rational::from_signeds(self.sym[j] * self.c(i, j), self.long())
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

    /// `2ρ∨ = Σ_{α>0} α∨` in simple coroots (§5.2), so `⟨λ, 2ρ∨⟩ = Σ_i c_i λ_i`.
    #[cfg(test)]
    pub(crate) fn two_rho_vee(&self) -> &[i64] {
        &self.two_rho_vee
    }

    /// `dim L(λ) = ∏_{α>0} (λ+ρ, α) / (ρ, α)` (§8 oracle 1), in checked `u128` and,
    /// only when that overflows, one big-natural product and exact division.
    pub(crate) fn weyl_dimension(&self, lambda: &[i64]) -> Result<Integer, LieError> {
        self.check_label(lambda)?;
        self.weyl_dimension_in(lambda, &self.classical(lambda)?)
    }

    fn weyl_dimension_in(&self, lambda: &[i64], w: &[i64]) -> Result<Integer, LieError> {
        #[cfg(test)]
        EVALUATIONS.with(|n| {
            let (r, d) = n.get();
            n.set((r, d + 1));
        });
        let factors = match self.root {
            RootSystem::A(_) => {
                // A roots are epsilon_i-epsilon_j; one suffix pass makes each factor O(1).
                (2..=w.len())
                    .flat_map(|j| (1..j).map(move |i| (i, j)))
                    .map(|(i, j)| {
                        let d = (j - i) as u128;
                        (d + u128::from(w[i - 1].abs_diff(w[j - 1])), d)
                    })
                    .collect()
            }
            RootSystem::B(_) | RootSystem::C(_) | RootSystem::D(_) => {
                let rho = &self.classical_rho;
                self.positive
                    .iter()
                    .map(|root| {
                        // Each classical root has at most two nonzero epsilon coordinates.
                        let pairing = |v: &[i64]| {
                            root.sparse_epsilon
                                .iter()
                                .map(|&(i, x)| i128::from(v[i]) * i128::from(x))
                                .sum::<i128>()
                        };
                        let den = pairing(rho);
                        let num = pairing(w).checked_add(den).ok_or(LieError::LabelOverflow)?;
                        Ok((num.unsigned_abs(), den.unsigned_abs()))
                    })
                    .collect::<Result<Vec<_>, LieError>>()?
            }
            _ => self
                .positive
                .iter()
                .map(|alpha| {
                    let mut num = 0u128;
                    let mut den = 0u128;
                    for (&f, &l) in alpha.form.iter().zip(lambda) {
                        let f = u128::try_from(f).map_err(|_| LieError::LabelOverflow)?;
                        num = (l as u128 + 1)
                            .checked_mul(f)
                            .and_then(|t| num.checked_add(t))
                            .ok_or(LieError::LabelOverflow)?;
                        den += f;
                    }
                    Ok((num, den))
                })
                .collect::<Result<Vec<_>, LieError>>()?,
        };
        // Every factor is at least one; 128 factors >= 2 prove u128 overflow.
        let overflows = factors.len() >= 128
            && factors
                .iter()
                .filter(|&&(n, d)| n / 2 >= d)
                .take(128)
                .count()
                == 128;
        if !overflows {
            if let Some(d) = crate::exact::checked_integral_quotient(factors.iter().copied()) {
                return Ok(Integer::from(d));
            }
        }
        let numerator: Natural = factors.into_iter().map(|(n, _)| Natural::from(n)).product();
        // The Weyl formula is integral for every validated dominant label.
        Ok(Integer::from(
            numerator.div_round(&self.weyl_denominator, Exact).0,
        ))
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
        if let Some(series) = self.series() {
            return classical::try_weyl_orbit(series, &self.classical(dominant)?)?
                .iter()
                .map(|w| self.dynkin(w))
                .collect();
        }
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

    /// The dual label `−w₀λ` (§5.1): `w₀λ` is the antidominant conjugate of
    /// `λ`, so `−w₀λ` is the dominant conjugate of `−λ`.
    #[cfg(test)]
    pub(crate) fn dual(&self, lambda: &[i64]) -> Result<Vec<i64>, LieError> {
        self.check_label(lambda)?;
        Ok(self.opposition(lambda))
    }

    pub(crate) fn check_dual_domain(&self, lambda: &[i64]) -> Result<(), LieError> {
        self.check_label(lambda)?;
        if !matches!(
            self.root,
            RootSystem::G2 | RootSystem::F4 | RootSystem::E6 | RootSystem::E7 | RootSystem::E8
        ) {
            return Ok(());
        }
        // Exceptional rank is at most eight: retain the checked public guard without warm allocations.
        let mut buffer = [0i64; 8];
        let w = &mut buffer[..self.rank];
        for (x, &l) in w.iter_mut().zip(lambda) {
            *x = -l;
        }
        while let Some(i) = w.iter().position(|&x| x < 0) {
            self.reflect(w, i)?;
        }
        Ok(())
    }

    fn opposition(&self, lambda: &[i64]) -> Vec<i64> {
        let mut d = lambda.to_vec();
        match self.root {
            RootSystem::A(_) => d.reverse(),
            RootSystem::D(r) if r % 2 == 1 => d.swap(r - 2, r - 1),
            RootSystem::E6 => {
                d.swap(0, 5);
                d.swap(2, 4);
            }
            _ => {}
        }
        d
    }

    /// `(−1)^{⟨λ, 2ρ∨⟩}` (§5.2). This is `κ_a` for a self-dual `λ` and for
    /// every `A_r` irrep; for a complex `D_r` (`r` odd) or E6 irrep `κ_a` is
    /// the gauge-derived `(−1)^{⟨λ, 2ρ∨⟩} σ_a σ_ā` (§5.3).
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
        self.chamber_multiplicities(lambda)?
            .into_iter()
            .map(|(w, m)| {
                Ok((
                    if self.series().is_some() {
                        self.dynkin(&w)?
                    } else {
                        w
                    },
                    m,
                ))
            })
            .collect()
    }

    fn chamber_multiplicities(&self, lambda: &[i64]) -> Result<Vec<(Vec<i64>, u64)>, LieError> {
        self.check_label(lambda)?;
        for &x in lambda {
            // λ + 2ρ enters the Casimir difference.
            x.checked_add(2).ok_or(LieError::LabelOverflow)?;
        }
        let mut doms = if let Some(series) = self.series() {
            // Dominance is a bounded monotone box in doubled epsilon coordinates.
            let top = self.classical(lambda)?;
            classical::enumerate_dominant_below(series, &top)?
                .into_iter()
                .map(|w| {
                    let k = classical::simple_root_coeffs(series, &sub(&top, &w)?)?
                        .expect("enumeration retains only the root lattice");
                    Ok((k, w))
                })
                .collect::<Result<Vec<_>, LieError>>()?
        } else {
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
            doms
        };
        doms.sort_by_cached_key(|(k, _)| (k.iter().sum::<i64>(), k.clone()));

        let top = if self.series().is_some() {
            self.classical(lambda)?
        } else {
            lambda.to_vec()
        };
        let mut mult: HashMap<Vec<i64>, u64> = HashMap::with_capacity(doms.len());
        let mut out = Vec::with_capacity(doms.len());
        for (k, mu) in doms {
            let m = if k.iter().all(|&x| x == 0) {
                1
            } else {
                self.freudenthal_step(&top, &k, &mu, &mult)?
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
            let term = if self.series().is_some() {
                (i128::from(lambda[i]) - i128::from(mu[i])).checked_mul(
                    i128::from(lambda[i])
                        + i128::from(mu[i])
                        + 2 * i128::from(self.classical_rho[i]),
                )
            } else {
                (i128::from(lambda[i]) + i128::from(mu[i]) + 2)
                    .checked_mul(i128::from(k[i]) * i128::from(self.sym[i]))
            }
            .ok_or(LieError::LabelOverflow)?;
            denom = denom.checked_add(term).ok_or(LieError::LabelOverflow)?;
        }
        let mut num = 0i128;
        for alpha in &self.positive {
            let mut shifted = mu.to_vec();
            loop {
                let dom = if let Some(series) = self.series() {
                    for (x, &v) in shifted.iter_mut().zip(&alpha.epsilon) {
                        *x = add_weight(*x, 2 * v)?;
                        x.checked_abs().ok_or(LieError::LabelOverflow)?;
                    }
                    classical::dominant_conjugate(series, &shifted).0
                } else {
                    shifted = add(&shifted, &alpha.dynkin)?;
                    self.to_dominant(shifted.clone())?.0
                };
                // μ + jα is higher than μ, so its dominant conjugate is already known.
                let Some(&m) = mult.get(&dom) else {
                    break;
                };
                let pairing = if self.series().is_some() {
                    dot(&alpha.epsilon, &shifted)?
                        .checked_mul(2)
                        .ok_or(LieError::LabelOverflow)?
                } else {
                    dot(&alpha.form, &shifted)?
                };
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
        if matches!(self.root, RootSystem::A(_)) {
            self.check_label(lambda)?;
            let mut out = BTreeMap::new();
            for p in self.gt_patterns(lambda)? {
                let entry = out.entry(self.gt_weight(&p)?).or_insert(0u64);
                *entry = entry.checked_add(1).ok_or(LieError::LabelOverflow)?;
            }
            return Ok(out);
        }
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
    #[cfg(test)]
    pub(crate) fn tensor_product(
        &self,
        a: &[i64],
        b: &[i64],
    ) -> Result<BTreeMap<Vec<i64>, u32>, LieError> {
        self.tensor_product_with_dimensions(a, b, None)
    }

    fn tensor_product_with_dimensions(
        &self,
        a: &[i64],
        b: &[i64],
        dims: Option<(&Natural, &Natural)>,
    ) -> Result<BTreeMap<Vec<i64>, u32>, LieError> {
        self.check_label(a)?;
        self.check_label(b)?;
        #[cfg(test)]
        {
            PRODUCT_BUILDS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            THREAD_PRODUCT_BUILDS.with(|n| n.set(n.get() + 1));
        }
        if matches!(self.root, RootSystem::A(_)) {
            return self.gt_product(a, b, dims);
        }
        if self.series().is_some() {
            let (ac, bc) = (self.classical(a)?, self.classical(b)?);
            let rho = &self.classical_rho;
            // This preflight precedes unit shortcuts and the dimension-based expansion choice.
            for ((&x, &y), &r) in ac.iter().zip(&bc).zip(rho) {
                x.checked_add(r)
                    .and_then(|s| s.checked_add(y))
                    .ok_or(LieError::LabelOverflow)?;
            }
        } else {
            for (&x, &y) in a.iter().zip(b) {
                x.checked_add(y)
                    .and_then(|s| s.checked_add(1))
                    .ok_or(LieError::LabelOverflow)?;
            }
        }
        if a.iter().all(|&x| x == 0) {
            return Ok(BTreeMap::from([(b.to_vec(), 1)]));
        }
        if b.iter().all(|&x| x == 0) {
            return Ok(BTreeMap::from([(a.to_vec(), 1)]));
        }
        let order = self.dimension_order(a, b, dims)?;
        let swap = if self.series().is_some() {
            order.is_le()
        } else {
            order.is_lt()
        };
        let (a, b) = if swap { (b, a) } else { (a, b) };
        let mut acc: BTreeMap<Vec<i64>, i64> = BTreeMap::new();
        // One BK sum; classical signed permutations and exceptional reflections supply its chamber map.
        let a_rho = if self.series().is_some() {
            add(&self.classical(a)?, &self.classical_rho)?
        } else {
            a.iter().map(|&x| x + 1).collect()
        };
        for (mu, m) in self.chamber_multiplicities(b)? {
            let orbit = if let Some(series) = self.series() {
                classical::try_weyl_orbit(series, &mu)?
            } else {
                self.orbit(&mu)?
            };
            for nu in orbit {
                let channel = if let Some(series) = self.series() {
                    let shifted = add(&a_rho, &nu)?;
                    for &x in &shifted {
                        x.checked_abs().ok_or(LieError::LabelOverflow)?;
                    }
                    let (dom, odd, wall) = classical::dominant_conjugate(series, &shifted);
                    if wall {
                        None
                    } else {
                        Some((self.dynkin(&sub(&dom, &self.classical_rho)?)?, odd))
                    }
                } else {
                    let (dom, odd) = self.to_dominant(add(&a_rho, &nu)?)?;
                    if dom.contains(&0) {
                        None
                    } else {
                        Some((dom.iter().map(|&x| x - 1).collect(), odd))
                    }
                };
                if let Some((c, odd)) = channel {
                    let m = i64::try_from(m).map_err(|_| LieError::LabelOverflow)?;
                    let entry = acc.entry(c).or_insert(0);
                    *entry = entry
                        .checked_add(if odd { -m } else { m })
                        .ok_or(LieError::LabelOverflow)?;
                }
            }
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
    fn series(&self) -> Option<Series> {
        match self.root {
            RootSystem::B(_) => Some(Series::B),
            RootSystem::C(_) => Some(Series::C),
            RootSystem::D(_) => Some(Series::D),
            _ => None,
        }
    }
    pub(crate) fn classical(&self, lambda: &[i64]) -> Result<Vec<i64>, LieError> {
        if let Some(series) = self.series() {
            return classical::dynkin_to_two_partition(series, lambda);
        }
        if matches!(self.root, RootSystem::A(_)) {
            let mut w = vec![0; lambda.len() + 1];
            for i in (0..lambda.len()).rev() {
                w[i] = lambda[i]
                    .checked_add(w[i + 1])
                    .ok_or(LieError::LabelOverflow)?;
            }
            return Ok(w);
        }
        Ok(vec![])
    }
    pub(crate) fn dynkin(&self, w: &[i64]) -> Result<Vec<i64>, LieError> {
        if let Some(series) = self.series() {
            for &x in w {
                x.checked_abs().ok_or(LieError::LabelOverflow)?;
            }
            return Ok(classical::two_partition_to_dynkin(series, w));
        }
        w.windows(2)
            .map(|x| x[0].checked_sub(x[1]).ok_or(LieError::LabelOverflow))
            .collect()
    }
    /// Whether the phase `κ` of a complex irrep is generated (`D_r` with `r`
    /// odd, and `E6`) rather than a closed form (`docs/gauge_cartan.md` §5.3).
    fn generates_phase(&self) -> bool {
        matches!(self.root, RootSystem::D(r) if r % 2 == 1) || self.root == RootSystem::E6
    }
    pub(crate) fn irrep(self: &Arc<Self>, lambda: &[i64]) -> Result<RootIrrep, LieError> {
        self.check_label(lambda)?;
        let label = Coordinates {
            dynkin: lambda.into(),
            classical: self.classical(lambda)?.into(),
        };
        let dual = self.opposition(lambda);
        let dual = if dual == lambda {
            None
        } else {
            Some(Coordinates {
                classical: self.classical(&dual)?.into(),
                dynkin: dual.into(),
            })
        };
        let dim = self
            .weyl_dimension_in(lambda, &label.classical)?
            .unsigned_abs();
        let dim_f64 = crate::lie::weyl_dim_f64(u128::try_from(&dim).ok(), || dim.clone());
        Ok(RootIrrep {
            data: Arc::new(IrrepData {
                root: self.root,
                roots: Arc::downgrade(self),
                public_coefficient_label_ok: self.check_dual_domain(lambda).is_ok()
                    && (dual.is_none() || !self.generates_phase() || u32::try_from(&dim).is_ok()),
                label,
                dual,
                dim,
                dim_f64,
                principal_sign: self.two_rho_vee_sign(lambda)?,
            }),
            conjugate: false,
        })
    }
    #[cfg(test)]
    pub(crate) fn shared_product(
        self: &Arc<Self>,
        a: &[i64],
        b: &[i64],
    ) -> Result<Product, LieError> {
        crate::cache::product(
            self.root,
            a,
            b,
            || self.compute_multiplicities(a, b, None),
            |channels| self.load_product(channels, None),
        )
    }
    pub(crate) fn product(
        self: &Arc<Self>,
        a: &RootIrrep,
        b: &RootIrrep,
    ) -> Result<Product, LieError> {
        crate::cache::product(
            self.root,
            a.dynkin(),
            b.dynkin(),
            || self.compute_multiplicities(a.dynkin(), b.dynkin(), Some((a, b))),
            |channels| self.load_product(channels, Some((a, b))),
        )
    }
    fn load_product(
        self: &Arc<Self>,
        channels: Multiplicities,
        inputs: Option<(&RootIrrep, &RootIrrep)>,
    ) -> Result<Product, LieError> {
        #[cfg(test)]
        LABEL_PROJECTIONS.with(|n| n.set(n.get() + 1));
        let mut channels = channels
            .0
            .iter()
            .map(|(c, n)| {
                let label = match inputs {
                    Some((a, _)) if a.dynkin() == c.as_ref() => a.clone(),
                    Some((_, b)) if b.dynkin() == c.as_ref() => b.clone(),
                    _ => self.irrep(c)?,
                };
                Ok((label, *n))
            })
            .collect::<Result<Vec<_>, LieError>>()?;
        channels.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        Ok(Product(channels.into()))
    }
    pub(super) fn multiplicities(
        &self,
        a: &[i64],
        b: &[i64],
        inputs: Option<(&RootIrrep, &RootIrrep)>,
    ) -> Result<Multiplicities, LieError> {
        crate::cache::product_multiplicities(self.root, a, b, || {
            self.compute_multiplicities(a, b, inputs)
        })
    }
    fn compute_multiplicities(
        &self,
        a: &[i64],
        b: &[i64],
        inputs: Option<(&RootIrrep, &RootIrrep)>,
    ) -> Result<Multiplicities, LieError> {
        Ok(Multiplicities(
            self.tensor_product_with_dimensions(a, b, inputs.map(|(a, b)| (a.dim(), b.dim())))?
                .into_iter()
                .map(|(c, n)| (c.into_boxed_slice(), n))
                .collect(),
        ))
    }

    pub(crate) fn multiplicity(
        &self,
        a: &RootIrrep,
        b: &RootIrrep,
        c: &RootIrrep,
    ) -> Result<u32, LieError> {
        if c.root() != self.root {
            return Ok(0);
        }
        Ok(self
            .multiplicities(a.dynkin(), b.dynkin(), Some((a, b)))?
            .multiplicity(c.dynkin()))
    }
    /// The Dynkin weight of an `A_r` pattern: `μ_j = w_{j+1} − w_{j+2}` with
    /// `w_l` the sum of row `l` minus the sum of row `l − 1`.
    pub(crate) fn gt_weight(&self, p: &crate::lie::GtPattern) -> Result<Vec<i64>, LieError> {
        let mut previous = 0;
        let mut w = Vec::with_capacity(self.rank + 1);
        for l in 1..=self.rank + 1 {
            let sum = (1..=l).try_fold(0i64, |sum, k| {
                sum.checked_add(p.get(k, l)).ok_or(LieError::LabelOverflow)
            })?;
            w.push(sum.checked_sub(previous).ok_or(LieError::LabelOverflow)?);
            previous = sum;
        }
        self.dynkin(&w)
    }
    pub(crate) fn gt_patterns(
        &self,
        lambda: &[i64],
    ) -> Result<Vec<crate::lie::GtPattern>, LieError> {
        Ok(gt_enumerate(&self.classical(lambda)?)
            .into_iter()
            .map(|data| crate::lie::GtPattern {
                n: self.rank + 1,
                data: data.into_boxed_slice(),
            })
            .collect())
    }
    fn dimension_order(
        &self,
        a: &[i64],
        b: &[i64],
        dims: Option<(&Natural, &Natural)>,
    ) -> Result<std::cmp::Ordering, LieError> {
        Ok(match dims {
            Some((a, b)) => a.cmp(b),
            None => self.weyl_dimension(a)?.cmp(&self.weyl_dimension(b)?),
        })
    }
    fn gt_product(
        &self,
        a: &[i64],
        b: &[i64],
        dims: Option<(&Natural, &Natural)>,
    ) -> Result<BTreeMap<Vec<i64>, u32>, LieError> {
        // Multiplicity-free U(N) > U(N-1) branching makes integer GT pattern counting exact.
        let (aw, bw) = (self.classical(a)?, self.classical(b)?);
        for (&x, &y) in aw.iter().zip(&bw) {
            add_weight(x, y)?;
        }
        let (a, bw) = if self.dimension_order(a, b, dims)?.is_le() {
            (a, bw)
        } else {
            (b, aw)
        };
        let n = self.rank + 1;
        let mut out = BTreeMap::new();
        for m in self.gt_patterns(a)? {
            let mut t = bw.clone();
            let mut bad = false;
            'scan: for k in 1..=n {
                for l in (k..=n).rev() {
                    let mut bkl = m.get(k, l);
                    if l > k {
                        bkl -= m.get(k, l - 1);
                    }
                    t[l - 1] = t[l - 1].checked_add(bkl).ok_or(LieError::LabelOverflow)?;
                    if l > 1 && t[l - 2] < t[l - 1] {
                        bad = true;
                        break 'scan;
                    }
                }
            }
            if !bad {
                let c = self.dynkin(&t)?;
                let count = out.entry(c).or_insert(0u32);
                *count = count.checked_add(1).ok_or(LieError::LabelOverflow)?;
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

#[cfg(test)]
fn coefficient_sum(rank: usize, roots: &[Vec<i64>]) -> Vec<i64> {
    (0..rank)
        .map(|i| roots.iter().map(|c| c[i]).sum())
        .collect()
}

/// `L d_j` as coprime integers, from `d_i cartan(j, i) = d_j cartan(i, j)`
/// along the edges of each connected component.
fn symmetrizer(rank: usize, cartan: &[i64]) -> Vec<i64> {
    // Zero marks a node not yet reached; every reached d is positive.
    let mut d = vec![Rational::from(0); rank];
    for start in 0..rank {
        if d[start].numerator_ref() != &0 {
            continue;
        }
        d[start] = Rational::from(1);
        let mut stack = vec![start];
        while let Some(i) = stack.pop() {
            for j in 0..rank {
                let (cij, cji) = (cartan[i * rank + j], cartan[j * rank + i]);
                if j != i && cij != 0 && d[j].numerator_ref() == &0 {
                    d[j] = &d[i] * Rational::from_signeds(cji, cij);
                    stack.push(j);
                }
            }
        }
    }
    // Finite type: every ratio d_i / d_j is 1, 2 or 3, so d / min d is integral.
    let min = d.iter().min().cloned().unwrap_or_else(|| Rational::from(1));
    d.iter()
        .map(|x| {
            i64::try_from(
                &Integer::try_from(x / &min).expect("finite-type symmetrizer is integral"),
            )
            .expect("finite-type symmetrizer fits i64")
        })
        .collect()
}

pub(crate) fn add_weight(a: i64, b: i64) -> Result<i64, LieError> {
    a.checked_add(b).ok_or(LieError::LabelOverflow)
}

fn add(a: &[i64], b: &[i64]) -> Result<Vec<i64>, LieError> {
    a.iter().zip(b).map(|(&x, &y)| add_weight(x, y)).collect()
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

#[derive(Debug)]
struct Coordinates {
    dynkin: Box<[i64]>,
    classical: Box<[i64]>,
}
#[derive(Debug)]
struct IrrepData {
    root: RootSystem,
    roots: std::sync::Weak<CartanData>,
    public_coefficient_label_ok: bool,
    label: Coordinates,
    dual: Option<Coordinates>,
    dim: Natural,
    dim_f64: f64,
    principal_sign: i8,
}
/// One immutable exact payload; conjugation selects the other coordinate view.
#[derive(Clone, Debug)]
pub(crate) struct RootIrrep {
    data: Arc<IrrepData>,
    conjugate: bool,
}
impl RootIrrep {
    fn coordinates(&self) -> &Coordinates {
        if self.conjugate {
            self.data
                .dual
                .as_ref()
                .expect("only complex labels select the dual")
        } else {
            &self.data.label
        }
    }
    pub(crate) fn root(&self) -> RootSystem {
        self.data.root
    }
    pub(crate) fn dynkin(&self) -> &[i64] {
        &self.coordinates().dynkin
    }
    pub(crate) fn dim(&self) -> &Natural {
        &self.data.dim
    }
    pub(crate) fn dim_f64(&self) -> f64 {
        self.data.dim_f64
    }
    pub(crate) fn sign(&self) -> i8 {
        self.data.principal_sign
    }
    pub(crate) fn is_self_dual(&self) -> bool {
        self.data.dual.is_none()
    }
    pub(crate) fn dual(&self) -> Self {
        Self {
            data: self.data.clone(),
            conjugate: self.data.dual.is_some() && !self.conjugate,
        }
    }
    pub(crate) fn weight_multiplicities(
        &self,
        roots: &CartanData,
    ) -> Result<BTreeMap<Vec<i64>, u64>, LieError> {
        roots.weight_multiplicities(self.dynkin())
    }
    pub(crate) fn roots(&self) -> Arc<CartanData> {
        // Standalone labels can outlive their group; cached labels never own its root tables.
        self.data
            .roots
            .upgrade()
            .unwrap_or_else(|| CartanData::new(self.root()))
    }
    pub(crate) fn public_coefficient_label_ok(&self) -> bool {
        self.data.public_coefficient_label_ok
    }
    fn order_coordinates(&self) -> &[i64] {
        let c = self.coordinates();
        if c.classical.is_empty() {
            &c.dynkin
        } else {
            &c.classical
        }
    }
    pub(crate) fn heap_bytes(&self) -> usize {
        use malachite_base::num::logic::traits::SignificantBits;
        let coords = |c: &Coordinates| {
            std::mem::size_of_val(c.dynkin.as_ref())
                .saturating_add(std::mem::size_of_val(c.classical.as_ref()))
        };
        // Naturals at most one limb are inline; larger values own their limb buffer.
        let bits = self.dim().significant_bits() as usize;
        let limbs = if bits > usize::BITS as usize {
            bits.div_ceil(usize::BITS as usize) * std::mem::size_of::<usize>()
        } else {
            0
        };
        crate::cache::ARC_COUNTS_BYTES
            .saturating_add(std::mem::size_of::<IrrepData>())
            // A Weak keeps only the allocation shell alive after the group drops its tables.
            .saturating_add(crate::cache::ARC_COUNTS_BYTES + std::mem::size_of::<CartanData>())
            .saturating_add(coords(&self.data.label))
            .saturating_add(self.data.dual.as_ref().map_or(0, coords))
            .saturating_add(limbs)
    }
}
impl PartialEq for RootIrrep {
    fn eq(&self, other: &Self) -> bool {
        self.root() == other.root() && self.dynkin() == other.dynkin()
    }
}
impl Eq for RootIrrep {}
impl PartialOrd for RootIrrep {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RootIrrep {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.root()
            .cmp(&other.root())
            .then_with(|| self.order_coordinates().cmp(other.order_coordinates()))
    }
}
impl std::hash::Hash for RootIrrep {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.root().hash(h);
        self.dynkin().hash(h);
    }
}
impl crate::cache::CacheKeyCharge for RootIrrep {
    fn key_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(self.heap_bytes())
    }
}

/// Exact Dynkin-ordered channels, without public label payloads or channel dimensions.
#[derive(Clone, Debug)]
pub(crate) struct Multiplicities(pub(crate) Arc<[(Box<[i64]>, u32)]>);
impl Multiplicities {
    pub(super) fn multiplicity(&self, c: &[i64]) -> u32 {
        self.0
            .binary_search_by(|(x, _)| x.as_ref().cmp(c))
            .map_or(0, |i| self.0[i].1)
    }
}
impl crate::cache::CacheCharge for Multiplicities {
    fn value_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + crate::cache::ARC_COUNTS_BYTES
            + std::mem::size_of_val(self.0.as_ref())
            + self
                .0
                .iter()
                .map(|(c, _)| std::mem::size_of_val(c.as_ref()))
                .sum::<usize>()
    }
}
/// Both projections are immutable and charged on publication in the same bounded tier.
#[derive(Clone)]
pub(crate) enum ProductValue {
    Multiplicities(Multiplicities),
    Labels(Product),
}
impl crate::cache::CacheCharge for ProductValue {
    fn value_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match self {
                Self::Multiplicities(x) => x.value_bytes(),
                Self::Labels(x) => x.value_bytes(),
            }
    }
}

/// The sole retained simple-factor decomposition, in legacy label order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Product(pub(crate) Arc<[(RootIrrep, u32)]>);
impl std::ops::Deref for Product {
    type Target = [(RootIrrep, u32)];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Product {
    #[cfg(test)]
    pub(crate) fn multiplicity(&self, c: &RootIrrep) -> u32 {
        if self.first().is_none_or(|(x, _)| x.root() != c.root()) {
            return 0;
        }
        let coordinates = c.order_coordinates();
        self.binary_search_by(|(x, _)| x.order_coordinates().cmp(coordinates))
            .map_or(0, |i| self[i].1)
    }
    #[cfg(test)]
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl crate::cache::CacheCharge for Product {
    fn value_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(crate::cache::ARC_COUNTS_BYTES)
            .saturating_add(std::mem::size_of_val(self.0.as_ref()))
            .saturating_add(
                self.iter()
                    .map(|(c, _)| c.heap_bytes())
                    .fold(0usize, usize::saturating_add),
            )
    }
}
#[cfg(test)]
pub(super) static PRODUCT_BUILDS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn shared_product(a: &RootIrrep, b: &RootIrrep) -> Result<Product, LieError> {
    // Resolve context only on a miss: standalone labels may outlive its group owner.
    let roots = std::cell::OnceCell::new();
    let data = || {
        roots.get_or_init(|| {
            a.data
                .roots
                .upgrade()
                .or_else(|| b.data.roots.upgrade())
                .unwrap_or_else(|| CartanData::new(a.root()))
        })
    };
    crate::cache::product(
        a.root(),
        a.dynkin(),
        b.dynkin(),
        || data().compute_multiplicities(a.dynkin(), b.dynkin(), Some((a, b))),
        |channels| data().load_product(channels, Some((a, b))),
    )
}

#[cfg(test)]
std::thread_local! {
    pub(crate) static EVALUATIONS:std::cell::Cell<(usize,usize)>=const{std::cell::Cell::new((0,0))};
    static LABEL_PROJECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Exact tensor products this thread computed.
    pub(crate) static THREAD_PRODUCT_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
