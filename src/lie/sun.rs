//! SU(N) irreps and their Clebsch–Gordan / recoupling coefficients, built by
//! the Gelfand–Tsetlin (GT) construction.
//!
//! An [`Irrep`](crate::lie::sun::Irrep) is an SU(N) highest weight; from it this
//! module gives the Weyl [`dimension`](crate::lie::sun::Irrep::dim), the
//! [`dual`](crate::lie::sun::Irrep::dual), the GT basis
//! ([`patterns`](crate::lie::sun::Irrep::patterns)), Littlewood–Richardson products (fusion
//! multiplicities), and — the point of the module — the Clebsch–Gordan
//! coefficients [`cgc`](crate::lie::sun::cgc) and the recoupling
//! [`f_symbol`](crate::lie::sun::f_symbol) / [`r_symbol`](crate::lie::sun::r_symbol),
//! with outer-multiplicity indices. Values are exact rationals (labels, dimensions,
//! GT ladder matrices) up to the CGC nullspace solve, which is
//! verification-gated floating point.
//!
//! The GT construction applies to SU(N) because the unitary chain
//! `U(N) ⊃ U(N-1) ⊃ … ⊃ U(1)` is multiplicity-free (the intermediate U(1) charge
//! separates copies the SU chain alone would repeat): GT patterns label basis
//! states of an SU(N) irrep uniquely, so the ladder operators have exact
//! closed-form matrix elements. See [`docs/references.md`] for the algorithm
//! sources and port provenance.
//!
//! [`docs/references.md`]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/references.md
//!
//! # Conventions
//!
//! ## Label normalization invariant
//!
//! An [`Irrep`](crate::lie::sun::Irrep) stores the SU(N) highest weight as a *normalized* weight
//! `λ = (λ₁ ≥ λ₂ ≥ … ≥ λ_N)` with `λ_N = 0` and all `λ_i ≥ 0`, of length `N`
//! (`= rank`). This matches `sunirrep.jl`'s `weight(s)` (`_dynkin_to_weight`
//! produces a nonincreasing tuple with last component 0). Weight input is
//! shift-invariant: any representative is accepted and normalized by
//! subtracting `λ_N`. The Dynkin labels `aᵢ = λᵢ − λᵢ₊₁` (all `≥ 0`) are
//! derivable via [`Irrep::dynkin`](crate::lie::sun::Irrep::dynkin).
//!
//! # Published object and global form
//!
//! The object is the set of finite-dimensional linear representations of a
//! **connected compact group** with root system `A_{N-1}`. Which group is named
//! by a [`GroupId`](crate::lie::GroupId), and the only thing the choice
//! changes is *which highest weights are admissible* — never a coefficient
//! value.
//!
//! Public labels are constructed by `LieGroup::irrep`, which validates
//! `GroupId::admits` before calling this private, form-free generator.
//! `PSU(3)` admits the adjoint `[1,1]` and rejects the fundamental `[1,0]`.
//!
//! The `N`-ality rule lives once, in
//! [`GroupId::admits`](crate::lie::GroupId::admits); the public LieGroup calls
//! it before entering this module. There is exactly one GT/CGC/F/R engine, and an [`Irrep`](crate::lie::sun::Irrep)
//! stores no global form: two forms that share a weight share its basis
//! ordering, phases, normalization, gauge and cache entries.
//!
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;

use num_bigint::BigInt;
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

use super::GtPattern;
use crate::exact::SignedSqrtRational;

pub(super) mod cgc;
mod fr;
mod linalg;

pub use cgc::{cgc, Cgc};
pub use fr::{f_symbol, r_symbol, FBlock};

/// Error for a malformed SU(N) irrep label. The public constructors never
/// panic; they return this instead.
///
/// Not `Eq`: the generation gates ([`SunError::NotOrthonormal`] and friends)
/// carry an `f64` residual for diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum SunError {
    /// Integer arithmetic or narrowing exceeded its range.
    LabelOverflow,
    /// A rank-0 label (empty weight / empty Dynkin for `N = 1` is a single
    /// zero, so a genuinely empty slice is rejected here).
    EmptyLabel,
    /// A weight that is not nonincreasing (`λᵢ < λᵢ₊₁` for some `i`), i.e. its
    /// implied Dynkin label would be negative.
    NotNonincreasing {
        /// The offending (unnormalized) weight.
        weight: Vec<i64>,
    },
    /// A Dynkin label with a negative component.
    NegativeDynkin {
        /// The offending Dynkin label.
        dynkin: Vec<i64>,
    },
    /// A [`directproduct`] of two irreps of different rank (distinct SU(N)
    /// groups have no common product; this is an ill-posed input, not a
    /// zero-channel fusion).
    RankMismatch {
        /// Rank `N` of the first irrep.
        a: usize,
        /// Rank `N` of the second irrep.
        b: usize,
    },
    /// The highest-weight nullspace dimension did not equal the exact fusion
    /// multiplicity `N^{s3}_{s1 s2}`. Ported gate from `clebschgordan.jl`'s
    /// `@assert N123 == directproduct(s1, s2)[s3]`; a mismatch means the
    /// numerical rank cut disagrees with the exact combinatorics.
    NullspaceDimMismatch {
        /// Expected multiplicity from [`directproduct`].
        expected: usize,
        /// Nullspace dimension found by the SVD rank cut.
        found: usize,
    },
    /// Assembled CGC columns are not orthonormal to tolerance (generation
    /// gate). The value is the worst `|<α|β> - δ_{αβ}|` over multiplicity
    /// columns.
    NotOrthonormal {
        /// Worst orthonormality residual.
        residual: f64,
    },
    /// The ladder-descent consistency spot check exceeded tolerance
    /// (generation gate): a lowered highest-weight relation was not reproduced
    /// by the descended coefficients.
    LadderInconsistent {
        /// Worst ladder residual.
        residual: f64,
    },
    /// A dense factorization routed through `t4a-faer` failed. Carries
    /// the backend's message; surfaced rather than panicked because the
    /// floating-point stages are verification-gated, not proven-unreachable.
    Linalg(String),
}

impl fmt::Display for SunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SunError::LabelOverflow => write!(
                f,
                "a label, weight, dimension or multiplicity exceeds the integer range"
            ),
            SunError::EmptyLabel => write!(f, "SU(N) irrep label must be non-empty"),
            SunError::NotNonincreasing { weight } => {
                write!(f, "SU(N) weight is not nonincreasing: {weight:?}")
            }
            SunError::NegativeDynkin { dynkin } => {
                write!(f, "SU(N) Dynkin label has a negative component: {dynkin:?}")
            }
            SunError::RankMismatch { a, b } => {
                write!(
                    f,
                    "directproduct of SU({a}) and SU({b}) irreps (rank mismatch)"
                )
            }
            SunError::NullspaceDimMismatch { expected, found } => write!(
                f,
                "CGC nullspace dimension {found} != fusion multiplicity {expected}"
            ),
            SunError::NotOrthonormal { residual } => {
                write!(f, "CGC columns not orthonormal (residual {residual:e})")
            }
            SunError::LadderInconsistent { residual } => {
                write!(f, "CGC ladder-descent inconsistent (residual {residual:e})")
            }
            SunError::Linalg(msg) => write!(f, "dense factorization failed: {msg}"),
        }
    }
}

impl std::error::Error for SunError {}

/// An irreducible representation of SU(N), labelled by its normalized highest
/// weight (see module docs for the invariant).
///
/// Build one from its `N-1` Dynkin labels with [`Irrep::from_dynkin`] — the
/// label *length* fixes `N`, so `&[1, 0]` is SU(3). [`Irrep::rank`] returns `N`
/// (not `N-1`).
///
///
/// `Ord`/`Hash` are on the normalized weight, so two `Irrep`s are equal iff
/// they denote the same irrep; the order is deterministic (used as a map key).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Irrep {
    /// Normalized highest weight, length `N`, nonincreasing, last entry `0`.
    weight: Arc<[i64]>,
}

/// Order-normalized product pair, or one label for a LieGroup dual lookup.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SunProductKey {
    Product(Irrep, Irrep),
    Dual(Irrep),
}

impl SunProductKey {
    pub(crate) fn new(a: &Irrep, b: &Irrep) -> Self {
        let (left, right) = if a <= b { (a, b) } else { (b, a) };
        Self::Product(left.clone(), right.clone())
    }
}

impl crate::cache::CacheKeyCharge for SunProductKey {
    fn key_bytes(&self) -> usize {
        match self {
            Self::Product(a, b) => a.key_bytes().saturating_add(b.key_bytes()),
            Self::Dual(a) => std::mem::size_of::<Self>().saturating_add(a.weight_bytes()),
        }
    }
}

/// Shared, sorted channels of one SU(N) tensor-product decomposition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SunProduct(Arc<[(Irrep, u32)]>);

impl SunProduct {
    pub(super) fn channel(&self, index: usize) -> &Irrep {
        &self.0[index].0
    }

    pub(crate) fn from_map(product: BTreeMap<Irrep, u32>) -> Self {
        Self(product.into_iter().collect())
    }

    /// Iterates over channels in irrep order with their multiplicities.
    pub fn iter(&self) -> impl Iterator<Item = (&Irrep, u32)> {
        self.0
            .iter()
            .map(|(irrep, multiplicity)| (irrep, *multiplicity))
    }

    /// Returns the channel multiplicity, or zero when `irrep` is absent.
    pub fn multiplicity(&self, irrep: &Irrep) -> u32 {
        self.0
            .binary_search_by(|(candidate, _)| candidate.cmp(irrep))
            .map(|index| self.0[index].1)
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    #[cfg(test)]
    pub(crate) fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

#[cfg(test)]
std::thread_local! {
    static PUBLIC_DIRECTPRODUCT_RECONSTRUCTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_public_directproduct_reconstructions() {
    PUBLIC_DIRECTPRODUCT_RECONSTRUCTIONS.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn public_directproduct_reconstructions() -> usize {
    PUBLIC_DIRECTPRODUCT_RECONSTRUCTIONS.with(std::cell::Cell::get)
}

impl crate::cache::CacheCharge for SunProduct {
    fn value_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(crate::cache::ARC_COUNTS_BYTES)
            .saturating_add(std::mem::size_of_val(self.0.as_ref()))
            .saturating_add(
                self.0
                    .iter()
                    .map(|(irrep, _)| irrep.weight_bytes())
                    .fold(0usize, usize::saturating_add),
            )
    }
}

impl crate::cache::CacheKeyCharge for Irrep {
    fn key_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(self.weight_bytes())
    }
}

impl Irrep {
    /// Shared weight allocation: counts plus payload.
    fn weight_bytes(&self) -> usize {
        crate::cache::ARC_COUNTS_BYTES.saturating_add(std::mem::size_of_val(self.weight.as_ref()))
    }

    /// Construct from an `N`-component highest weight (any shift representative).
    ///
    /// Normalizes by subtracting the last component. Rejects an empty slice
    /// ([`SunError::EmptyLabel`]) or a non-nonincreasing weight
    /// ([`SunError::NotNonincreasing`]).
    pub fn from_weight(weight: &[i64]) -> Result<Self, SunError> {
        if weight.is_empty() {
            return Err(SunError::EmptyLabel);
        }
        for w in weight.windows(2) {
            if w[0] < w[1] {
                return Err(SunError::NotNonincreasing {
                    weight: weight.to_vec(),
                });
            }
        }
        let last = weight[weight.len() - 1];
        weight[0].checked_sub(last).ok_or(SunError::LabelOverflow)?;
        // Keep the exact-length iterator so Arc fills its allocation directly.
        let norm: Arc<[i64]> = weight
            .iter()
            .map(|x| {
                x.checked_sub(last).expect(
                    "nonincreasing weights bound every difference by the checked first-minus-last",
                )
            })
            .collect();
        Ok(Irrep { weight: norm })
    }

    /// Construct from the `N-1` Dynkin labels `aᵢ = λᵢ − λᵢ₊₁` (all `≥ 0`).
    ///
    /// `N = dynkin.len() + 1`. Mirrors `_dynkin_to_weight`: the weight is the
    /// suffix sums with a trailing `0`. An empty slice yields the SU(1) trivial
    /// irrep. Rejects a negative component ([`SunError::NegativeDynkin`]).
    pub fn from_dynkin(dynkin: &[i64]) -> Result<Self, SunError> {
        if dynkin.iter().any(|&a| a < 0) {
            return Err(SunError::NegativeDynkin {
                dynkin: dynkin.to_vec(),
            });
        }
        // Suffix sums, emitted front to back so the exact-length iterator fills
        // the shared allocation directly.
        let mut suffix = weight_sum(dynkin)?;
        let weight = dynkin
            .iter()
            .map(|&a| {
                let w = suffix;
                suffix -= a;
                w
            })
            .chain([0])
            .collect();
        Ok(Irrep { weight })
    }

    /// The trivial (vacuum) SU(N) irrep — the all-zero weight of length `N`.
    /// `N` must be `≥ 1` ([`SunError::EmptyLabel`] otherwise).
    #[cfg(test)]
    pub fn trivial(n: usize) -> Result<Self, SunError> {
        if n == 0 {
            return Err(SunError::EmptyLabel);
        }
        Ok(Irrep {
            weight: vec![0i64; n].into(),
        })
    }

    /// The rank `N` of the SU(N) group.
    pub fn rank(&self) -> usize {
        self.weight.len()
    }

    /// The normalized highest weight (length `N`, nonincreasing, last `0`).
    pub(super) fn weight(&self) -> &[i64] {
        &self.weight
    }

    /// The `N-1` Dynkin labels `aᵢ = λᵢ − λᵢ₊₁`.
    pub fn dynkin(&self) -> Vec<i64> {
        self.weight.windows(2).map(|w| w[0] - w[1]).collect()
    }

    /// The Weyl dimension, exact.
    ///
    /// Ported from `sector.jl:dim`:
    /// `∏_{k₂=2..N, k₁=1..k₂-1} (k₂-k₁ + λ_{k₁} - λ_{k₂}) / (k₂-k₁)`.
    /// The product is an integer; we accumulate as a `Ratio<BigInt>` and return
    /// the (guaranteed-integral) value as a `BigInt` (no `u*` cap on `N`).
    pub fn dim(&self) -> BigInt {
        let w = &self.weight;
        let n = w.len();
        let mut acc = Ratio::<BigInt>::one();
        for k2 in 2..=n {
            for k1 in 1..k2 {
                let d = BigInt::from(k2 - k1);
                let numer = &d + BigInt::from(w[k1 - 1]) - BigInt::from(w[k2 - 1]);
                acc *= Ratio::new(numer, d);
            }
        }
        acc.to_integer()
    }

    /// The dual (conjugate) irrep. Ported from `sector.jl:dual`: reverse the
    /// Dynkin labels. Reversing a nonnegative Dynkin label stays valid, so the
    /// reconstruction cannot fail.
    pub fn dual(&self) -> Irrep {
        let mut d = self.dynkin();
        d.reverse();
        Irrep::from_dynkin(&d)
            .expect("reversing nonnegative Dynkin labels preserves their representable sum")
    }

    /// The Frobenius–Schur phase $\kappa_a = \operatorname{sign} F^{a\bar a a}_a[1, 1]$
    /// as `±1.0`, for **every** irrep, self-dual or not.
    ///
    /// This is TensorKitSectors 0.3.9 `src/sectors.jl:frobenius_schur_phase`
    /// (`frobenius_schur_phase_from_Fsymbol`, the line-bending phase used by
    /// `Asymbol`); SUNRepresentations 0.4.0 does not override it. On a
    /// self-dual irrep it is the gauge-invariant indicator
    /// ([`frobenius_schur`](Self::frobenius_schur)); on a complex irrep it is
    /// fixed by the CGC gauge (`docs/gauge.md`), and in that gauge it is
    /// **not** always `+1` — e.g. the SU(4) fundamental has `-1`.
    ///
    /// Closed form: $\kappa_a = (-1)^{\langle\lambda, 2\rho\rangle}$, with
    /// $\langle\lambda, 2\rho\rangle = \sum_i a_i\, i(N - i)$ over the Dynkin
    /// labels. Why it is exact in this gauge: with `e = f = 1` the F
    /// contraction reduces to $F \cdot d_a = \sqrt{d_a}\,C^{a\bar a}_1[\ell_a, h_{\bar a}]
    /// \cdot \sqrt{d_a}\,C^{\bar a a}_1[h_{\bar a}, \ell_a]$ ($\ell$/$h$ = lowest/highest
    /// GT pattern). The gauge (§2, §4) makes the first coupling pair of a
    /// singlet, $(\ell_{x}, h_{\bar x})$, positive, so the first factor is
    /// `+1` and $C^{\bar a a}_1[\ell_{\bar a}, h_a] > 0$. With
    /// $L = \sum_l J^-_l$, whose GT matrix elements are non-negative
    /// (`gtpatterns.jl:creation`, transposed), the singlet obeys
    /// $(L^h \otimes 1)\psi = (-1)^h (1 \otimes L^h)\psi$; its
    /// $(\ell_{\bar a}, \ell_a)$ component relates
    /// $C^{\bar a a}_1[h_{\bar a}, \ell_a]$ to $C^{\bar a a}_1[\ell_{\bar a}, h_a]$
    /// by $(-1)^h$ times two positive matrix elements $\langle\ell|L^h|h\rangle$,
    /// where $h = \langle\lambda + \bar\lambda, \rho\rangle = \langle\lambda, 2\rho\rangle$
    /// is the number of simple lowering steps from highest to lowest weight.
    /// `tests/frobenius_schur.rs` checks the closed form against the F-symbol
    /// sweep and SUNRepresentations' own values. No CGC is generated to
    /// answer it.
    pub fn frobenius_schur_phase(&self) -> f64 {
        let n = self.rank();
        // Only the parity of Σ aᵢ·i(N−i) matters; summing parities avoids overflow.
        let odd = self
            .weight
            .windows(2)
            .enumerate()
            .filter(|&(i, w)| {
                (w[0] - w[1]) % 2 != 0
                    && !(i + 1).is_multiple_of(2)
                    && !(n - i - 1).is_multiple_of(2)
            })
            .count()
            % 2
            == 1;
        if odd {
            -1.0
        } else {
            1.0
        }
    }

    /// All GT patterns of this irrep, in the reference basis order.
    ///
    /// Ported from `gtpatterns.jl:GTPatternIterator` /
    /// `basis(s) = GTPatternIterator{N}(weight(s))`, reproduced index-for-index.
    /// The order is load-bearing — the CGC gauge is a deterministic function of
    /// it — so it is pinned by `tests/sun_oracle.rs`. See the private
    /// `gt_enumerate` for the recursion.
    pub fn patterns(&self) -> Vec<GtPattern> {
        let n = self.rank();
        gt_enumerate(&self.weight)
            .into_iter()
            .map(|data| GtPattern {
                n,
                data: data.into_boxed_slice(),
            })
            .collect()
    }

    /// The GT creation (raising) matrices, one per level `l = 1..N-1`.
    ///
    /// Result index `l-1` holds the sparse entries of the raising operator that
    /// increments `m[k, l]`. Entry rows/cols are 0-based indices into
    /// [`Irrep::patterns`]. Ported from `gtpatterns.jl:creation`; the entry is
    /// `signedroot(coef)` of the exact GT rational `coef`, carried as a
    /// [`SignedSqrtRational`] (its `signed_square()` equals `coef`).
    #[cfg(test)]
    fn creation(&self) -> Vec<Vec<LadderEntry>> {
        self.try_creation()
            .expect("test fixture has representable GT ladder weights")
    }
    fn try_creation(&self) -> Result<Vec<Vec<LadderEntry>>, SunError> {
        let n = self.rank();
        let pats = self.patterns();
        // Basis index of each pattern (the reference `table`).
        let table: HashMap<&GtPattern, usize> =
            pats.iter().enumerate().map(|(i, m)| (m, i)).collect();
        let mut result: Vec<Vec<LadderEntry>> = vec![Vec::new(); n.saturating_sub(1)];

        for (i, m) in pats.iter().enumerate() {
            if n < 2 {
                break;
            }
            for l in 1..=(n - 1) {
                for k in 1..=l {
                    // coef = -1 * ∏_{k'} [ (m[k',l+1] - m[k,l] + k - k')
                    //                     · (m[k',l-1] - m[k,l] + k - k' - 1) if k'≤l-1 ]
                    //             / [ (m[k',l] - m[k,l] + k - k')
                    //                 (m[k',l] - m[k,l] + k - k' - 1) if k'≤l, k'≠k ]
                    let mkl = m.get(k, l);
                    let mut coef = Ratio::<BigInt>::from(BigInt::from(-1));
                    let mut skip = false;
                    for kp in 1..=(l + 1) {
                        let base = ladder_base(mkl, kp, k)?;
                        let f1 = ladder_factor(m.get(kp, l + 1), base, 0)?;
                        coef *= BigInt::from(f1);
                        if kp < l {
                            // reference: k' ≤ l-1 (so m[k', l-1] is in range)
                            let f2 = ladder_factor(m.get(kp, l - 1), base, 1)?;
                            coef *= BigInt::from(f2);
                        }
                        if coef.numer().is_zero() {
                            skip = true; // coef == 0: no entry
                            break;
                        }
                        if kp <= l && kp != k {
                            let g = ladder_factor(m.get(kp, l), base, 0)?;
                            // Julia's Rational hits denominator 0 here and the
                            // reference skips; we detect the zero divisor and
                            // skip too (num_rational would panic on ÷0).
                            let den = ladder_denominator(g)?;
                            if den == 0 {
                                skip = true;
                                break;
                            }
                            coef /= BigInt::from(den);
                        }
                    }
                    if skip || coef.numer().is_zero() {
                        continue;
                    }
                    // m' = m with m[k,l] raised by 1. Look up its basis index.
                    //
                    // Why-not (the one exception to this crate's no-panic
                    // contract): a missing m' is proven unreachable, so we panic
                    // in every build rather than silently drop the entry.
                    // Invariant proof: raising m[k,l] zeroes a numerator factor
                    // (`m[k,l+1] - m[k,l]` or the lower-neighbour term) exactly
                    // when a GT betweenness constraint would break, so coef ≠ 0
                    // implies m' is a valid basis member. The panic is thus dead
                    // code by proof; a silent release drop would instead turn a
                    // future proof-breaking regression into a missing ladder
                    // entry — a silent-wrong-answer defect. Both reference
                    // implementations abort here (Julia `table[m']` throws
                    // KeyError in all builds; QSpace aborts on invariant
                    // violations).
                    let mut mp = m.clone();
                    mp.set(k, l, add_weight(mkl, 1)?);
                    let &j = table.get(&mp).expect(
                        "GT invariant violated: coef != 0 implies the raised \
                         pattern is a valid basis member",
                    );
                    result[l - 1].push(LadderEntry {
                        row: j,
                        col: i,
                        value: signedroot(&coef),
                    });
                }
            }
        }
        Ok(result)
    }

    /// The GT annihilation (lowering) matrices: the transpose of
    /// [`Irrep::try_creation`] (`annihilation(s) = [op' for op in creation(s)]`,
    /// `gtpatterns.jl`). Entries are real, so the transpose just swaps
    /// `row`/`col`.
    #[cfg(test)]
    fn annihilation(&self) -> Vec<Vec<LadderEntry>> {
        self.try_annihilation()
            .expect("test fixture has representable GT ladder weights")
    }
    fn try_annihilation(&self) -> Result<Vec<Vec<LadderEntry>>, SunError> {
        Ok(self
            .try_creation()?
            .into_iter()
            .map(|mat| {
                mat.into_iter()
                    .map(|e| LadderEntry {
                        row: e.col,
                        col: e.row,
                        value: e.value,
                    })
                    .collect()
            })
            .collect())
    }
}

/// A single nonzero entry of a GT ladder matrix: `value` at `(row, col)`, with
/// `row`/`col` 0-based indices into [`Irrep::patterns`].
#[derive(Clone, Debug, PartialEq, Eq)]
struct LadderEntry {
    /// Row index (0-based basis index of the raised pattern).
    pub row: usize,
    /// Column index (0-based basis index of the source pattern).
    pub col: usize,
    /// The exact matrix element `sign * sqrt(radicand)`.
    pub value: SignedSqrtRational,
}

/// Enumerate the flat data of every GT pattern with the given top row, in the
/// reference `GTPatternIterator` order.
///
/// The reference builds, for `N ≥ 2`, the product over the possible second rows
/// `I[i+1]:I[i]` (`reverse`d so the *last* sub-row entry varies fastest), and
/// for each recurses into `GTPatternIterator{N-1}`, with the recursion as the
/// inner (faster) loop. Concatenating `(toprow, subpattern)` reproduces both
/// the flat storage layout and the iteration order.
fn gt_enumerate(toprow: &[i64]) -> Vec<Vec<i64>> {
    let n = toprow.len();
    if n == 1 {
        return vec![vec![toprow[0]]];
    }
    let mut out = Vec::new();
    for subrow in subrow_order(toprow) {
        for sub in gt_enumerate(&subrow) {
            let mut data = toprow.to_vec();
            data.extend_from_slice(&sub);
            out.push(data);
        }
    }
    out
}

/// All admissible second rows for `toprow`, in the reference product order:
/// `sub[j] ∈ [toprow[j+1], toprow[j]]` (GT betweenness), with the *last* entry
/// `sub[N-2]` varying fastest and `sub[0]` slowest.
fn subrow_order(toprow: &[i64]) -> Vec<Vec<i64>> {
    let m = toprow.len() - 1;
    let mut out = Vec::new();
    let mut cur = Vec::with_capacity(m);
    subrow_rec(0, m, toprow, &mut cur, &mut out);
    out
}

fn subrow_rec(j: usize, m: usize, toprow: &[i64], cur: &mut Vec<i64>, out: &mut Vec<Vec<i64>>) {
    if j == m {
        out.push(cur.clone());
        return;
    }
    // Ascending, matching Julia's `I[j+1]:I[j]` UnitRange iteration.
    for val in toprow[j + 1]..=toprow[j] {
        cur.push(val);
        subrow_rec(j + 1, m, toprow, cur, out);
        cur.pop();
    }
}

fn ladder_base(weight: i64, kp: usize, k: usize) -> Result<i64, SunError> {
    i64::try_from(kp)
        .ok()
        .and_then(|kp| weight.checked_add(kp))
        .and_then(|base| base.checked_sub(i64::try_from(k).ok()?))
        .ok_or(SunError::LabelOverflow)
}
fn ladder_factor(weight: i64, base: i64, offset: i64) -> Result<i64, SunError> {
    weight
        .checked_sub(base)
        .and_then(|v| v.checked_sub(offset))
        .ok_or(SunError::LabelOverflow)
}
fn ladder_denominator(g: i64) -> Result<i64, SunError> {
    g.checked_sub(1)
        .and_then(|h| g.checked_mul(h))
        .ok_or(SunError::LabelOverflow)
}
fn add_weight(a: i64, b: i64) -> Result<i64, SunError> {
    a.checked_add(b).ok_or(SunError::LabelOverflow)
}
fn weight_sum(weights: &[i64]) -> Result<i64, SunError> {
    weights.iter().try_fold(0i64, |sum, &w| add_weight(sum, w))
}
fn add_multiplicity(a: u32, b: u32) -> Result<u32, SunError> {
    a.checked_add(b).ok_or(SunError::LabelOverflow)
}
fn magnetic_index(value: usize) -> Result<u32, SunError> {
    u32::try_from(value).map_err(|_| SunError::LabelOverflow)
}

/// Littlewood–Richardson product decomposition: the fusion multiplicities of
/// `a ⊗ b`, keyed by the resulting irrep.
///
/// Ported from `gtpatterns.jl:directproduct`. Requires `rank(a) == rank(b)`
/// (both label the same SU(N)); a rank mismatch is an ill-posed input across
/// distinct groups and returns [`SunError::RankMismatch`] rather than a
/// zero-channel map. The reference
/// iterates the smaller-dimensional basis; we replicate the `dim` swap (the
/// result is independent of it, but the port stays faithful).
///
/// # Returns
///
/// Every irrep `c` with `N^c_ab > 0`, mapped to that multiplicity. Channels
/// with `N^c_ab = 0` are **absent**, not present with value `0`. Iteration
/// order is the deterministic `Ord` order of [`Irrep`] (the normalized weight),
/// which is stable but carries no physical meaning. Use
/// [`shared_directproduct`] instead when you decompose the same pair
/// repeatedly and want the cached, cheaply cloneable form.
///
#[cfg(test)]
pub fn directproduct(a: &Irrep, b: &Irrep) -> Result<BTreeMap<Irrep, u32>, SunError> {
    let product = shared_directproduct(a, b)?;
    #[cfg(test)]
    PUBLIC_DIRECTPRODUCT_RECONSTRUCTIONS.with(|count| count.set(count.get() + 1));
    Ok(product
        .iter()
        .map(|(irrep, multiplicity)| (irrep.clone(), multiplicity))
        .collect())
}

/// The cached, read-only tensor-product decomposition of `a ⊗ b`.
///
/// Same content as [`directproduct`] — every `c` with `N^c_ab > 0` and its
/// multiplicity — but returned as a cheaply cloneable [`SunProduct`] backed by
/// the process-global product cache, so repeated queries on the same pair do
/// not rebuild the map. Query it with [`SunProduct::multiplicity`] or iterate
/// with [`SunProduct::iter`]. Errors identically ([`SunError::RankMismatch`]
/// on mismatched `N`).
pub fn shared_directproduct(a: &Irrep, b: &Irrep) -> Result<SunProduct, SunError> {
    if a.rank() != b.rank() {
        return Err(SunError::RankMismatch {
            a: a.rank(),
            b: b.rank(),
        });
    }
    let key = SunProductKey::new(a, b);
    let cache = crate::cache::cache_sun_product();
    if let Some(hit) = cache.get(&key) {
        return Ok(hit);
    }
    Ok(cache.insert(key, directproduct_uncached(a, b)?))
}

fn directproduct_uncached(a: &Irrep, b: &Irrep) -> Result<SunProduct, SunError> {
    // Check before GT enumeration, including when one huge label is paired with a small one.
    for (&x, &y) in a.weight().iter().zip(b.weight()) {
        add_weight(x, y)?;
    }
    let (a, b) = if a.dim() <= b.dim() { (a, b) } else { (b, a) };
    let n = a.rank();
    let mut result: BTreeMap<Irrep, u32> = BTreeMap::new();
    for m in a.patterns() {
        // t starts as b's weight; each GT row of `a` shifts one component.
        let mut t: Vec<i64> = b.weight.to_vec();
        let mut bad = false;
        'scan: for k in 1..=n {
            for l in (k..=n).rev() {
                let mut bkl = m.get(k, l);
                if l > k {
                    // checkbounds(m, k, l-1): valid when l-1 ≥ k.
                    bkl -= m.get(k, l - 1);
                }
                t[l - 1] = add_weight(t[l - 1], bkl)?;
                if l > 1 && t[l - 2] < t[l - 1] {
                    bad = true;
                    break 'scan;
                }
            }
        }
        if !bad {
            // t is a valid (nonincreasing) weight; from_weight normalizes it.
            let s = Irrep::from_weight(&t)?;
            let count = result.entry(s).or_insert(0);
            *count = add_multiplicity(*count, 1)?;
        }
    }
    Ok(SunProduct::from_map(result))
}

/// `sign(coef) * sqrt(|coef|)` as an exact [`SignedSqrtRational`], matching
/// `RationalRoots.signedroot`. `signed_square()` of the result equals `coef`.
fn signedroot(coef: &Ratio<BigInt>) -> SignedSqrtRational {
    if coef.is_zero() {
        return SignedSqrtRational::zero();
    }
    let s = if coef.is_negative() {
        Ratio::from(BigInt::from(-1))
    } else {
        Ratio::from(BigInt::from(1))
    };
    SignedSqrtRational::from_prefactor_radical(s, coef.abs())
}

/// Opaque authority fingerprint of the generated SU(N) provider — **the version
/// of the SU(N) gauge specification**.
///
/// The bytes identify the *convention set*, generation pipeline, and
/// verification/tolerance policy under which every SU(N) Clebsch–Gordan
/// coefficient (and the F/R symbols contracted from it) is produced. Their sole
/// use is equality comparison: a consumer may persist the bytes next to data
/// derived from these coefficients and later compare them to decide whether that
/// derived data was produced under the same convention.
///
/// # This is a spec version, not a build label
///
/// `docs/gauge.md` is a **frozen normative specification**: it is the authority,
/// and this crate is an implementation of it. So these bytes do *not* mean
/// "whatever the current code outputs" — they name the version of that
/// document. A refactor that moves a coefficient value is a deviation from the
/// spec (a bug, and `tests/gauge_golden.rs` fails on it), **not** a new gauge
/// deserving a new fingerprint.
///
/// The value therefore changes only when the specification itself is corrected:
/// an exceptional, deliberate, loud event that requires, together, the
/// `docs/gauge.md` edit stating the defect it corrects, the `epoch` bump here
/// and in `tests/sun_fingerprint.rs`, a CHANGELOG breaking-change entry, and
/// regenerated golden values (`docs/gauge.md`, "Status"). For a consumer that
/// means the strong promise: **same fingerprint, same coefficients** — within
/// the tolerance class the binding contract below disclaims — across any amount
/// of internal churn.
///
/// # Contract (binding)
///
/// > Equal fingerprints identify the same convention, generation pipeline, and
/// > tolerance policy. They do not imply byte-identical values or independently
/// > prove numerical agreement.
///
/// This is deliberately weaker than the base SU(2) fingerprint
/// ([`crate::category::FusionCategory::authority_fingerprint`]), whose exact big-rational surface lets
/// equal bytes mean equal values. The generated SU(N) family is a *two-layer*
/// contract (`docs/gauge.md` "value agreement within the oracle tolerance, not
/// cross-process bit-identity"): the gauge/sign/structure is a deterministic
/// function of the subspace, but the final dense linear-algebra stages run in
/// `f64` and the backend's parallel reductions are not bit-reproducible across
/// processes, so two builds may differ by a few ULPs. **Numerical agreement is
/// established by the generation-time verification gates** (`docs/gauge.md` §9:
/// multiplicity, orthonormality, ladder consistency — typed `SunError`, never
/// silent) **and the independent oracle suites** (`docs/gauge.md` §11: the SU(2)
/// embedding and the signed SUNRepresentations.jl v0.4.0 values), **never by
/// this fingerprint.**
///
/// # Consumer contract
///
/// - **Opaque.** Compare by equality only; never parse the tags or split on
///   `:` / `=`. The internal shape is not a stable interface.
/// - **Stable across patch and minor releases.** The value is not derived from
///   the crate version, source, docs, a pointer, or any process-local state.
/// - **Changes exactly with a specification correction.** The trailing `epoch` is
///   bumped by hand — and only — when `docs/gauge.md` is corrected in a way that
///   alters a returned coefficient value, its normalization, or the canonical
///   convention it is expressed in (the four-step rule of `docs/gauge.md`,
///   "Status"). The compatibility-policy test (`tests/sun_fingerprint.rs`) pins
///   the exact bytes, so any such change is a mutation-visible review event.
/// - **Epoch is per-family and independent.** The SU(N) `epoch` moves
///   independently of the SU(2) and B/C/D epochs; an SU(N) gauge change never
///   invalidates SU(2)-derived or B/C/D-derived consumer state (and vice versa).
///   The base SU(2) surface is untouched by this fingerprint.
///
/// # Tags and the conventions they pin (each cites `docs/gauge.md`)
///
/// Every tag names a rule the gauge document already pins; nothing here invents
/// a convention. The backend identity is deliberately excluded — per-backend ULP
/// differences are inside the tolerance class this fingerprint's contract
/// disclaims, and a discrete gauge flip across backends is a defect, not a
/// tolerance event (`docs/gauge.md` §10).
///
/// - `ref=sunrep-0.4` — the port reference: SUNRepresentations.jl v0.4.0
///   (`docs/gauge.md`, header).
/// - `basis=gt-order` — the Gelfand–Tsetlin pattern basis order that indexes
///   `m1, m2, m3`, and the highest-weight-system coupling-pair enumeration that
///   follows from it (`docs/gauge.md` §1, §2).
/// - `gauge=qrpos-cref` — the gauge canonicalization
///   `gaugefix! = first ∘ qrpos! ∘ cref!`: the column-pivoted reduced echelon
///   pivot rule and the positive-diagonal QR sign fix (`docs/gauge.md` §4, 4a/4b).
/// - `descent=ladder-lstsq` — the lower-weight descent: reverse-lexicographic
///   weight order and the QR least-squares lowering solve (`docs/gauge.md` §5).
/// - `tol=sunrep-tol-tier` — the value-fixing tolerance tier (the reference
///   SUNRepresentations `TOL_*` constants: `TOL_NULLSPACE` rank cut, `TOL_GAUGE`
///   pivot, `TOL_PURGE`; `docs/gauge.md` §3, §4a, §6). The `TOL_ORTHO`/
///   `TOL_LADDER` gates are excluded: they cannot move a returned value, so
///   tightening them is not a breaking release (`docs/gauge.md` §9).
/// - `epoch=1` — the per-family manual epoch (see above).
///
/// # Stability
///
/// **Unstable: shape may change.** Cargo features cannot express instability
/// tiers, so this rustdoc label is the marker.
#[cfg(feature = "lie-generated")]
pub fn fingerprint() -> &'static [u8] {
    // Spec version. Bump the trailing `epoch=N` (and the literal in
    // tests/sun_fingerprint.rs) only when docs/gauge.md is corrected, under the
    // four-step rule in its "Status" section. A refactor never bumps it.
    b"racah:sun-gt:ref=sunrep-0.4:basis=gt-order:gauge=qrpos-cref:descent=ladder-lstsq:tol=sunrep-tol-tier:epoch=1"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn irr(dynkin: &[i64]) -> Irrep {
        Irrep::from_dynkin(dynkin).unwrap()
    }

    // ---- labels / normalization ----

    #[test]
    fn weight_normalizes_by_shift() {
        // (3,1,0) and (5,3,2) denote the same SU(3) irrep (Dynkin (2,1)).
        let a = Irrep::from_weight(&[3, 1, 0]).unwrap();
        let b = Irrep::from_weight(&[5, 3, 2]).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.weight(), &[3, 1, 0]);
        assert_eq!(a.dynkin(), vec![2, 1]);
    }

    #[test]
    fn dynkin_round_trip() {
        let s = irr(&[2, 0, 1]); // SU(4)
        assert_eq!(s.rank(), 4);
        let round = Irrep::from_dynkin(&s.dynkin()).unwrap();
        assert_eq!(s, round);
    }

    #[test]
    fn malformed_labels_are_typed_errors_not_panics() {
        assert_eq!(Irrep::from_weight(&[]), Err(SunError::EmptyLabel));
        assert!(matches!(
            Irrep::from_weight(&[1, 2, 0]),
            Err(SunError::NotNonincreasing { .. })
        ));
        assert!(matches!(
            Irrep::from_dynkin(&[1, -1]),
            Err(SunError::NegativeDynkin { .. })
        ));
        assert_eq!(Irrep::trivial(0), Err(SunError::EmptyLabel));
    }

    // ---- dim / dual ----

    #[test]
    fn weyl_dim_known_su3() {
        assert_eq!(irr(&[0, 0]).dim(), BigInt::from(1)); // trivial
        assert_eq!(irr(&[1, 0]).dim(), BigInt::from(3)); // fundamental
        assert_eq!(irr(&[0, 1]).dim(), BigInt::from(3)); // antifundamental
        assert_eq!(irr(&[1, 1]).dim(), BigInt::from(8)); // adjoint
        assert_eq!(irr(&[2, 0]).dim(), BigInt::from(6));
        assert_eq!(irr(&[3, 0]).dim(), BigInt::from(10));
    }

    #[test]
    fn weyl_dim_large_su2_labels_stay_exact() {
        assert_eq!(irr(&[i64::MAX]).dim(), BigInt::from(1_u64) << 63);
        assert_eq!(irr(&[i64::MAX - 1]).dim(), BigInt::from(i64::MAX));
    }

    #[test]
    fn dual_is_reverse_dynkin_and_involutive() {
        let s = irr(&[1, 0]); // SU(3) fundamental
        assert_eq!(s.dual().dynkin(), vec![0, 1]); // antifundamental
        for d in [vec![1, 0], vec![2, 1, 0], vec![1, 2, 0, 3]] {
            let x = irr(&d);
            assert_eq!(x.dual().dual(), x);
            assert_eq!(x.dim(), x.dual().dim()); // dual preserves dimension
        }
    }

    // ---- GT patterns ----

    #[test]
    fn patterns_count_equals_dim() {
        for d in [
            vec![1, 0],
            vec![1, 1],
            vec![2, 1],
            vec![1, 0, 1],
            vec![1, 1, 0, 1],
        ] {
            let s = irr(&d);
            assert_eq!(BigInt::from(s.patterns().len()), s.dim());
        }
    }

    #[test]
    fn su3_fundamental_pattern_order() {
        // Reference basis(SU3Irrep(1,0,0)) data order (verified against Julia):
        // (1,0,0, 0,0, 0), (1,0,0, 1,0, 0), (1,0,0, 1,0, 1).
        let s = Irrep::from_weight(&[1, 0, 0]).unwrap();
        let got: Vec<Vec<i64>> = s.patterns().iter().map(|p| p.data().to_vec()).collect();
        assert_eq!(
            got,
            vec![
                vec![1, 0, 0, 0, 0, 0],
                vec![1, 0, 0, 1, 0, 0],
                vec![1, 0, 0, 1, 0, 1],
            ]
        );
    }

    #[test]
    fn pattern_get_matches_reference_layout() {
        let s = Irrep::from_weight(&[1, 0, 0]).unwrap();
        let p = &s.patterns()[2]; // data (1,0,0, 1,0, 1)
                                  // Top row l=3: m[1,3]=1, m[2,3]=0, m[3,3]=0.
        assert_eq!(p.get(1, 3), 1);
        assert_eq!(p.get(2, 3), 0);
        assert_eq!(p.get(3, 3), 0);
        // Row l=2: m[1,2]=1, m[2,2]=0. Row l=1: m[1,1]=1.
        assert_eq!(p.get(1, 2), 1);
        assert_eq!(p.get(2, 2), 0);
        assert_eq!(p.get(1, 1), 1);
    }

    // ---- directproduct ----

    #[test]
    fn su3_product_known() {
        // 3 ⊗ 3̄ = 8 ⊕ 1
        let dp = directproduct(&irr(&[1, 0]), &irr(&[0, 1])).unwrap();
        let mut got: Vec<(Vec<i64>, u32)> = dp.iter().map(|(k, &v)| (k.dynkin(), v)).collect();
        got.sort();
        assert_eq!(got, vec![(vec![0, 0], 1), (vec![1, 1], 1)]);

        // 3 ⊗ 3 = 6 ⊕ 3̄
        let dp = directproduct(&irr(&[1, 0]), &irr(&[1, 0])).unwrap();
        let mut got: Vec<(Vec<i64>, u32)> = dp.iter().map(|(k, &v)| (k.dynkin(), v)).collect();
        got.sort();
        assert_eq!(got, vec![(vec![0, 1], 1), (vec![2, 0], 1)]);
    }

    #[test]
    fn directproduct_rank_mismatch_is_typed_error() {
        // Distinct SU(N) groups: ill-posed input must be a typed error, never a
        // (well-formed-looking) empty decomposition.
        let su3 = irr(&[1, 0]);
        let su4 = irr(&[1, 0, 0]);
        assert_eq!(
            directproduct(&su3, &su4),
            Err(SunError::RankMismatch { a: 3, b: 4 })
        );
    }

    #[test]
    fn directproduct_dim_sum_rule() {
        // dim(a)·dim(b) == Σ_c N^c_ab dim(c)
        for (da, db) in [
            (vec![1, 1], vec![1, 1]),             // SU(3) 8⊗8
            (vec![2, 1], vec![1, 2]),             // SU(3)
            (vec![1, 0, 1], vec![1, 1, 0]),       // SU(4)
            (vec![1, 1, 0, 1], vec![0, 1, 1, 0]), // SU(5)
        ] {
            assert_dim_sum_rule(&irr(&da), &irr(&db));
        }
    }

    #[test]
    fn directproduct_commutes_and_dual_twist() {
        assert_commute_and_dual_twist(&irr(&[2, 1]), &irr(&[1, 1]));
    }

    #[test]
    #[ignore = "release-mode product access collector; not a CI timing gate"]
    fn sun_product_private_collector() {
        use std::hint::black_box;
        use std::time::Instant;

        let a = irr(&[1, 1]);
        let b = irr(&[1, 1]);
        let channel = irr(&[1, 1]);
        let product = shared_directproduct(&a, &b).unwrap();
        let repetitions = 1_000_000u32;

        let started = Instant::now();
        for _ in 0..repetitions {
            black_box(product.multiplicity(black_box(&channel)));
        }
        let shared_multiplicity = started.elapsed();

        let started = Instant::now();
        for _ in 0..repetitions {
            black_box(
                product
                    .iter()
                    .map(|(_, multiplicity)| multiplicity)
                    .sum::<u32>(),
            );
        }
        let shared_channels = started.elapsed();

        let started = Instant::now();
        for _ in 0..repetitions {
            black_box(
                shared_directproduct(black_box(&a), black_box(&b))
                    .unwrap()
                    .multiplicity(black_box(&channel)),
            );
        }
        let cached_shared_multiplicity = started.elapsed();

        let started = Instant::now();
        for _ in 0..repetitions {
            black_box(
                shared_directproduct(black_box(&a), black_box(&b))
                    .unwrap()
                    .iter()
                    .map(|(_, multiplicity)| multiplicity)
                    .sum::<u32>(),
            );
        }
        let cached_shared_channels = started.elapsed();

        let started = Instant::now();
        for _ in 0..repetitions {
            let map = directproduct(black_box(&a), black_box(&b)).unwrap();
            black_box(map.get(black_box(&channel)).copied().unwrap_or(0));
        }
        let public_multiplicity = started.elapsed();

        let started = Instant::now();
        for _ in 0..repetitions {
            black_box(
                directproduct(black_box(&a), black_box(&b))
                    .unwrap()
                    .into_keys()
                    .collect::<Vec<_>>(),
            );
        }
        let public_channels = started.elapsed();

        let ns = |elapsed: std::time::Duration| elapsed.as_nanos() as f64 / repetitions as f64;
        eprintln!(
            "SUN_PRODUCT_PRIVATE_ACCESS case=su3_8x8 channels={} repetitions={} shared_value_multiplicity_ns={:.3} shared_value_channels_ns={:.3} cached_shared_multiplicity_ns={:.3} cached_shared_channels_ns={:.3} public_multiplicity_ns={:.3} public_channels_ns={:.3}",
            product.0.len(),
            repetitions,
            ns(shared_multiplicity),
            ns(shared_channels),
            ns(cached_shared_multiplicity),
            ns(cached_shared_channels),
            ns(public_multiplicity),
            ns(public_channels),
        );
    }

    fn assert_dim_sum_rule(a: &Irrep, b: &Irrep) {
        // dim(a)·dim(b) == Σ_c N^c_ab dim(c)
        let lhs = a.dim() * b.dim();
        let rhs: BigInt = directproduct(a, b)
            .unwrap()
            .iter()
            .map(|(c, &m)| c.dim() * BigInt::from(m))
            .sum();
        assert_eq!(
            lhs,
            rhs,
            "sum rule failed for {:?} ⊗ {:?}",
            a.dynkin(),
            b.dynkin()
        );
    }

    fn assert_commute_and_dual_twist(a: &Irrep, b: &Irrep) {
        // Commutativity (bosonic fusion) and N^c_ab == N^{c̄}_{ā b̄}.
        assert_eq!(directproduct(a, b), directproduct(b, a));
        let dp = directproduct(a, b).unwrap();
        let dpd = directproduct(&a.dual(), &b.dual()).unwrap();
        let twisted: BTreeMap<Irrep, u32> = dp.iter().map(|(c, &m)| (c.dual(), m)).collect();
        assert_eq!(dpd, twisted);
    }

    #[test]
    fn randomized_property_sweep() {
        // Seeded pair sweep so the in-crate properties are themselves
        // randomized (acceptance item 5): 50 random pairs per N in 2..=5.
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x5150_4E37_0DEC_0DE5);
        // Bound the Dynkin range per rank so dims (hence directproduct basis
        // size) stay modest for higher N — the sweep is a property check, not a
        // large-irrep stress test.
        let max_dynkin = |n: usize| -> i64 {
            match n {
                2 => 4,
                3 => 3,
                4 => 2,
                _ => 1,
            }
        };
        let rand_irrep = |rng: &mut rand_chacha::ChaCha8Rng, n: usize| -> Irrep {
            let hi = max_dynkin(n);
            let dynkin: Vec<i64> = (0..n - 1).map(|_| rng.gen_range(0..=hi)).collect();
            Irrep::from_dynkin(&dynkin).unwrap()
        };
        for n in 2..=5usize {
            for _ in 0..50 {
                let a = rand_irrep(&mut rng, n);
                let b = rand_irrep(&mut rng, n);
                assert_dim_sum_rule(&a, &b);
                assert_commute_and_dual_twist(&a, &b);
                assert_eq!(a.dual().dual(), a); // dual involution
                let _ = a.creation(); // panics if a raised pattern is missing from the basis
            }
        }
    }

    // ---- ladder matrices ----

    #[test]
    fn su3_adjoint_creation_matches_reference() {
        // creation(SU3Irrep(2,1,0)) nonzero entries, signed_square (from Julia):
        // l=1: (i,j) with sq: (2,1)=1, (5,4)=2, (6,5)=2, (8,7)=1
        // l=2: (4,1)=1, (3,2)=3/2, (5,2)=1/2, (7,3)=3/2, (7,5)=1/2, (8,6)=1
        // (i,j) are 1-based; our row/col are 0-based.
        let s = Irrep::from_weight(&[2, 1, 0]).unwrap();
        let cr = s.creation();
        let key = |mat: &Vec<LadderEntry>| -> Vec<(usize, usize, (i64, i64))> {
            let mut v: Vec<_> = mat
                .iter()
                .map(|e| {
                    let sq = e.value.signed_square();
                    (
                        e.row + 1,
                        e.col + 1,
                        (
                            sq.numer().try_into().unwrap(),
                            sq.denom().try_into().unwrap(),
                        ),
                    )
                })
                .collect();
            v.sort();
            v
        };
        assert_eq!(
            key(&cr[0]),
            vec![
                (2, 1, (1, 1)),
                (5, 4, (2, 1)),
                (6, 5, (2, 1)),
                (8, 7, (1, 1)),
            ]
        );
        assert_eq!(
            key(&cr[1]),
            vec![
                (3, 2, (3, 2)),
                (4, 1, (1, 1)),
                (5, 2, (1, 2)),
                (7, 3, (3, 2)),
                (7, 5, (1, 2)),
                (8, 6, (1, 1)),
            ]
        );
    }

    #[test]
    fn annihilation_is_transpose_of_creation() {
        let s = Irrep::from_weight(&[2, 1, 0]).unwrap();
        let cr = s.creation();
        let an = s.annihilation();
        for (cm, am) in cr.iter().zip(an.iter()) {
            let mut a: Vec<_> = am.iter().map(|e| (e.row, e.col, e.value.clone())).collect();
            let mut ct: Vec<_> = cm.iter().map(|e| (e.col, e.row, e.value.clone())).collect();
            a.sort_by_key(|x| (x.0, x.1));
            ct.sort_by_key(|x| (x.0, x.1));
            assert_eq!(a, ct);
        }
    }

    // ---- SU(2) embedding cross-check (item 4) ----

    #[test]
    fn su2_dim_dual_fusion_match_closed_form() {
        for dj in 0..=6i64 {
            let j = irr(&[dj]); // SU(2), Dynkin (2j)
            assert_eq!(j.dim(), BigInt::from(dj + 1)); // 2j+1
            assert_eq!(j.dual(), j); // SU(2) is self-dual
        }
        // Fusion range |j1-j2|..j1+j2, each multiplicity 1 (doubled labels).
        for dj1 in 0..=4i64 {
            for dj2 in 0..=4i64 {
                let dp = directproduct(&irr(&[dj1]), &irr(&[dj2])).unwrap();
                let mut got: Vec<i64> = dp
                    .iter()
                    .map(|(c, &m)| {
                        assert_eq!(m, 1);
                        c.dynkin()[0]
                    })
                    .collect();
                got.sort();
                let want: Vec<i64> = ((dj1 - dj2).abs()..=(dj1 + dj2)).step_by(2).collect();
                assert_eq!(got, want, "SU(2) fusion {dj1}⊗{dj2}");
            }
        }
    }

    #[test]
    fn su2_creation_matches_closed_form() {
        // For SU(2) irrep 2j = dj, GT basis is x = m[1,1] = 0..dj (ascending),
        // and raising x→x+1 has matrix element sqrt((dj-x)(x+1)):
        // <j,m+1|J+|j,m> with m = x - j. signed_square = (dj-x)(x+1) > 0.
        for dj in 1..=6i64 {
            let s = irr(&[dj]);
            let cr = s.creation();
            assert_eq!(cr.len(), 1);
            let mut got: Vec<(usize, usize, BigInt)> = cr[0]
                .iter()
                .map(|e| {
                    assert_eq!(e.value.sign(), 1);
                    (e.row, e.col, e.value.signed_square().to_integer())
                })
                .collect();
            got.sort();
            let want: Vec<(usize, usize, BigInt)> = (0..dj as usize)
                .map(|x| (x + 1, x, BigInt::from((dj - x as i64) * (x as i64 + 1))))
                .collect();
            assert_eq!(got, want, "SU(2) creation dj={dj}");
        }
    }
}

#[cfg(test)]
#[path = "sun/sun_oracle_tests.rs"]
mod sun_oracle_tests;

#[cfg(test)]
#[path = "sun/sun_cgc_gauge_tests.rs"]
mod sun_cgc_gauge_tests;

#[cfg(all(test, feature = "lie-generated"))]
#[path = "sun/sun_fr_su2_tests.rs"]
mod sun_fr_su2_tests;

#[cfg(test)]
#[path = "sun/su2_embedding_tests.rs"]
mod su2_embedding_tests;

#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    #[test]
    fn weight_and_multiplicity_bounds() {
        assert_eq!(add_weight(i64::MAX, 1), Err(SunError::LabelOverflow));
        assert_eq!(add_weight(i64::MIN, -1), Err(SunError::LabelOverflow));
        assert_eq!(weight_sum(&[i64::MAX, 1]), Err(SunError::LabelOverflow));
        assert_eq!(
            Irrep::from_dynkin(&[i64::MAX, 1]),
            Err(SunError::LabelOverflow)
        );
        assert_eq!(
            Irrep::from_weight(&[i64::MAX, -1]),
            Err(SunError::LabelOverflow)
        );
        assert_eq!(add_multiplicity(u32::MAX, 1), Err(SunError::LabelOverflow));
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            magnetic_index(u32::MAX as usize + 1),
            Err(SunError::LabelOverflow)
        );
        let x = Irrep::from_dynkin(&[i64::MAX, 0]).unwrap();
        assert_eq!(ladder_base(i64::MAX, 1, 1), Err(SunError::LabelOverflow));
        assert_eq!(ladder_base(i64::MIN, 0, 1), Err(SunError::LabelOverflow));
        assert_eq!(ladder_base(3, 2, 1), Ok(4));
        #[cfg(target_pointer_width = "64")]
        assert_eq!(ladder_base(0, usize::MAX, 1), Err(SunError::LabelOverflow));
        assert_eq!(ladder_factor(i64::MAX, -1, 0), Err(SunError::LabelOverflow));
        assert_eq!(ladder_factor(i64::MIN, 0, 1), Err(SunError::LabelOverflow));
        assert_eq!(ladder_denominator(i64::MIN), Err(SunError::LabelOverflow));
        assert_eq!(ladder_denominator(i64::MAX), Err(SunError::LabelOverflow));
        assert_eq!(ladder_denominator(0), Ok(0));
        assert_eq!(ladder_denominator(1), Ok(0));
        assert_eq!(ladder_denominator(-2), Ok(6));
        assert_eq!(x.dual().dual(), x);
    }
}
