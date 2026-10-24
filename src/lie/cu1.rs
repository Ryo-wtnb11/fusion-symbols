//! Closed-form CU(1) coefficients.

use crate::category::{BraidingStyle, EnumerationError, FusionCategory, FusionStyle};

/// An irrep of `CU(1)`, TensorKitSectors' `CU1Irrep(j, s)`, stored as
/// `twice_charge = 2j` and `s`.
///
/// Valid labels (`cu1irrep.jl` constructor): `(0, 0)` = `0+`, `(0, 1)` =
/// `0-`, and `(2j, 2)` for `j > 0`. Fields are private so only those exist.
/// `Ord` is TensorKitSectors' `isless`: by charge, then `0+` before `0-`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CU1Irrep {
    twice_charge: u32,
    s: u8,
}

/// An invalid label or fusion label overflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CU1Error {
    /// `(twice_charge, s)` is not a `CU1Irrep`: `s` must be `0` or `1` for
    /// `twice_charge == 0` and `2` otherwise (`cu1irrep.jl` constructor).
    InvalidLabel {
        /// The requested twice charge.
        twice_charge: u32,
        /// The requested `s`.
        s: u8,
    },
    /// A fusion channel `2(j + k)` of [`CU1::fusion`] above `u32::MAX`.
    TwiceChargeOutOfRange {
        /// The exact twice charge.
        twice_charge: u64,
    },
}

impl std::fmt::Display for CU1Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CU1Error::InvalidLabel { twice_charge, s } => write!(
                f,
                "not a CU(1) irrep: twice_charge {twice_charge}, s {s} (s ∈ {{0, 1}} for charge 0, s = 2 otherwise)"
            ),
            CU1Error::TwiceChargeOutOfRange { twice_charge } => write!(
                f,
                "CU(1) twice charge {twice_charge} exceeds u32::MAX"
            ),
        }
    }
}

impl std::error::Error for CU1Error {}

impl CU1Irrep {
    /// The unit `0+`, TensorKitSectors `CU1Irrep(0, 0)`.
    pub const VACUUM: Self = Self {
        twice_charge: 0,
        s: 0,
    };
    /// The pseudoscalar `0-`, TensorKitSectors `CU1Irrep(0, 1)`.
    pub const PSEUDOSCALAR: Self = Self {
        twice_charge: 0,
        s: 1,
    };

    /// TensorKitSectors `CU1Irrep(j, s)` with `j = twice_charge / 2`.
    ///
    /// # Errors
    ///
    /// [`CU1Error::InvalidLabel`] for an `s` the charge does not admit.
    pub fn new(twice_charge: u32, s: u8) -> Result<Self, CU1Error> {
        let valid = if twice_charge == 0 { s <= 1 } else { s == 2 };
        if valid {
            Ok(Self { twice_charge, s })
        } else {
            Err(CU1Error::InvalidLabel { twice_charge, s })
        }
    }

    /// TensorKitSectors `CU1Irrep(j)` with its default `s`: `0+` for
    /// `twice_charge == 0`, the two-dimensional irrep otherwise. Infallible:
    /// every `u32` is a label with that `s`.
    pub const fn from_twice_charge(twice_charge: u32) -> Self {
        Self {
            twice_charge,
            s: if twice_charge == 0 { 0 } else { 2 },
        }
    }

    fn from_exact(twice_charge: u64) -> Result<Self, CU1Error> {
        match u32::try_from(twice_charge) {
            Ok(q) => Ok(Self::from_twice_charge(q)),
            Err(_) => Err(CU1Error::TwiceChargeOutOfRange { twice_charge }),
        }
    }

    /// `2j`.
    pub const fn twice_charge(self) -> u32 {
        self.twice_charge
    }

    /// TensorKitSectors' `s`: `0` for `0+`, `1` for `0-`, `2` for `j > 0`.
    pub const fn s(self) -> u8 {
        self.s
    }
}

/// Allocation-free iterator over the fusion channels of two `CU(1)` irreps
/// (see [`CU1::fusion`]).
#[derive(Clone, Debug)]
pub struct CU1Fusion(std::iter::Take<std::array::IntoIter<CU1Irrep, 3>>);

impl Iterator for CU1Fusion {
    type Item = CU1Irrep;

    fn next(&mut self) -> Option<CU1Irrep> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl DoubleEndedIterator for CU1Fusion {
    fn next_back(&mut self) -> Option<CU1Irrep> {
        self.0.next_back()
    }
}

impl ExactSizeIterator for CU1Fusion {}

/// Closed-form CU(1) = U(1) ⋊ C = O(2).
///
/// Every value follows TensorKitSectors.jl 0.3.9 (git `67d342a1`)
/// `src/irreps/cu1irrep.jl` (`CU1Irrep`: constructor, `dual`, `⊗`, `dim`,
/// `Nsymbol`, `Fsymbol`, `Rsymbol`, `frobenius_schur_phase`), with the generic
/// `src/sectors.jl:twist_from_Rsymbol` for the twist, which `CU1Irrep` does
/// not override. Never "simplify" a sign or branch of [`CU1::f_symbol`]
/// without rereading `cu1irrep.jl:Fsymbol`: the gauge is in its branch order.
///
/// The irreps are `0+` (the unit), `0-` (the pseudoscalar, charge
/// conjugation acting by `-1`) and, for each half-integer `j > 0`, a
/// two-dimensional irrep. Fusion is multiplicity free; every irrep is
/// self-dual; the braiding is bosonic (twist `1`) with
/// `R^{ab}_{0-} = -1` for charged `a`, `b`; the Frobenius–Schur phase is `1`.
///
/// Scalars are `f64`, as TensorKitSectors' `fusionscalartype` and
/// `braidingscalartype` (`Float64`). F and R are the recouplings of
/// TensorKitSectors' `CU1Irrep` fusion tensors (`tests/cu1_values.rs`
/// contracts them densely).
/// One deliberate signed-zero difference: TensorKitSectors' `Rsymbol` returns
/// `-0.0` for an inadmissible `R^{ab}_{0-}` with charged `a` (it negates the
/// zero `N`); here every inadmissible symbol is `+0.0`, as in the other
/// families of this crate.
///
/// Label domain: `twice_charge = 2j` is any `u32`. A consumer with a narrower
/// label encoding validates its own bound.
///
/// Every validated CU(1) label belongs to this category.
///
/// Enumeration (frozen): `0+, 0−`, then the two-dimensional irreps by
/// increasing `twice_charge`; `(twice_charge, 2)` has index
/// `twice_charge + 1`. The count is infinite; indices past
/// `u32::MAX + 1` are [`EnumerationError::Overflow`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CU1;

impl FusionCategory for CU1 {
    type Object = CU1Irrep;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = CU1Fusion;
    type Error = CU1Error;
    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Simple
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> CU1Irrep {
        CU1Irrep::VACUUM
    }
    fn contains(&self, a: &CU1Irrep) -> bool {
        CU1Irrep::new(a.twice_charge, a.s).is_ok()
    }
    fn dual(&self, a: &CU1Irrep) -> CU1Irrep {
        *a
    }
    fn dim(&self, a: &CU1Irrep) -> f64 {
        if a.twice_charge == 0 {
            1.0
        } else {
            2.0
        }
    }
    fn frobenius_schur_phase(&self, _a: &CU1Irrep) -> f64 {
        1.0
    }
    fn frobenius_schur_indicator(&self, _a: &CU1Irrep) -> f64 {
        1.0
    }
    fn twist(&self, _a: &CU1Irrep) -> f64 {
        1.0
    }
    /// Fusion channels of `a ⊗ b`, each with multiplicity one, in
    /// TensorKitSectors' `CU1IrrepProdIterator` order: `0± ⊗ 0± = 0+` or `0-`
    /// (s added mod 2); `0± ⊗ j = j`; `j ⊗ j = 0+ ⊕ 0- ⊕ 2j`; otherwise
    /// `|j - k| ⊕ (j + k)`.
    ///
    /// # Errors
    ///
    /// [`CU1Error::TwiceChargeOutOfRange`] when `2(j + k)` exceeds
    /// `u32::MAX`.
    fn fusion(&self, a: &CU1Irrep, b: &CU1Irrep) -> Result<CU1Fusion, CU1Error> {
        let (a, b) = (*a, *b);
        let mut channels = [CU1Irrep::VACUUM; 3];
        let len = if a.twice_charge == 0 && b.twice_charge == 0 {
            channels[0] = CU1Irrep {
                twice_charge: 0,
                s: a.s ^ b.s,
            };
            1
        } else if a.twice_charge == 0 {
            channels[0] = b;
            1
        } else if b.twice_charge == 0 {
            channels[0] = a;
            1
        } else {
            let sum = CU1Irrep::from_exact(u64::from(a.twice_charge) + u64::from(b.twice_charge))?;
            if a == b {
                channels = [CU1Irrep::VACUUM, CU1Irrep::PSEUDOSCALAR, sum];
                3
            } else {
                // `|j - k| > 0` and below `j + k`, so it is a charged label.
                channels[0] = CU1Irrep {
                    twice_charge: a.twice_charge.abs_diff(b.twice_charge),
                    s: 2,
                };
                channels[1] = sum;
                2
            }
        };
        Ok(CU1Fusion(channels.into_iter().take(len)))
    }
    /// `false`: the two-dimensional irreps make some admissible F differ from
    /// `1` (`F^{jjj}_j[0+, 0+] = 1/2`).
    fn has_trivial_associator_gauge(&self) -> bool {
        false
    }
    /// `true`: `0+` is self-dual, `0+ ⊗ a = a ⊗ 0+ = a` (multiplicity free),
    /// and [`CU1::f_symbol`] returns exactly `1` for every admissible F with
    /// `a`, `b` or `c` equal to `0+` (`cu1irrep.jl:Fsymbol`'s first branch).
    fn has_canonical_unit(&self) -> bool {
        true
    }
    /// Fusion multiplicity `N^{ab}_c ∈ {0, 1}` (`cu1irrep.jl:Nsymbol`), exact for
    /// every label triple: the charge sum is compared in `u64`, so a fusion that
    /// [`CU1::fusion`] cannot represent still has `N = 0` here.
    fn nsymbol(&self, a: &CU1Irrep, b: &CU1Irrep, c: &CU1Irrep) -> u32 {
        let admissible = match c.s {
            // `(a.s == b.s == 2) | (a.s == b.s)` is `a.s == b.s`.
            0 => a.twice_charge == b.twice_charge && a.s == b.s,
            1 => a.twice_charge == b.twice_charge && ((a.s == 2 && b.s == 2) || a.s != b.s),
            _ => {
                u64::from(c.twice_charge) == u64::from(a.twice_charge) + u64::from(b.twice_charge)
                    || c.twice_charge == a.twice_charge.abs_diff(b.twice_charge)
            }
        };
        u32::from(admissible)
    }

    /// F-symbol `F^{abc}_d[e, f]`; exact `0.0` when a vertex is inadmissible.
    ///
    /// `cu1irrep.jl:Fsymbol`, branch for branch: `1` when `a`, `b` or `c` is
    /// `0+` or two of them are `0-`; `±1` when one is `0-`; then entries `±1`,
    /// `±1/2`, `±√2/2` or `0` for charged `a`, `b`, `c`.
    fn f_symbol(
        &self,
        a: &CU1Irrep,
        b: &CU1Irrep,
        c: &CU1Irrep,
        d: &CU1Irrep,
        e: &CU1Irrep,
        f: &CU1Irrep,
    ) -> Result<f64, CU1Error> {
        if [(a, b, e), (e, c, d), (b, c, f), (a, f, d)]
            .iter()
            .any(|&(a, b, c)| self.nsymbol(a, b, c) == 0)
        {
            return Ok(0.0);
        }
        let (a, b, c, d, e, f) = (*a, *b, *c, *d, *e, *f);
        let (op, om) = (CU1Irrep::VACUUM, CU1Irrep::PSEUDOSCALAR);
        let q = |x: CU1Irrep| i64::from(x.twice_charge);
        if a == op || b == op || c == op {
            return Ok(1.0);
        }
        // `(a == b == om) || (a == c == om) || (b == c == om)`.
        if [a, b, c].iter().filter(|&&x| x == om).count() >= 2 {
            return Ok(1.0);
        }
        let sign = |negative: bool| if negative { -1.0 } else { 1.0 };
        if a == om {
            return Ok(sign(q(d) != 0 && q(d) == q(c) - q(b)));
        }
        if b == om {
            return Ok(sign(q(d) == (q(a) - q(c)).abs()));
        }
        if c == om {
            return Ok(sign(q(d) == q(a) - q(b)));
        }
        // From here on a, b, c are charged. `T(sqrt(2) / 2)` in TensorKitSectors.
        let s = 2.0_f64.sqrt() / 2.0;
        if a == b && b == c {
            return Ok(if d == a {
                if q(e) == 0 {
                    if q(f) == 0 {
                        if f.s == 1 {
                            -0.5
                        } else {
                            0.5
                        }
                    } else if e.s == 1 {
                        -s
                    } else {
                        s
                    }
                } else if q(f) == 0 {
                    s
                } else {
                    0.0
                }
            } else {
                1.0
            });
        }
        if a == b {
            return Ok(if d == c {
                if q(f) == q(b) + q(c) && e.s == 1 {
                    -s
                } else {
                    s
                }
            } else {
                1.0
            });
        }
        if b == c {
            return Ok(if d == a {
                if q(e) != q(a) + q(b) && f.s == 1 {
                    -s
                } else {
                    s
                }
            } else {
                1.0
            });
        }
        if a == c {
            return Ok(if d == b {
                if q(e) == q(f) {
                    0.0
                } else {
                    1.0
                }
            } else {
                sign(d.s == 1)
            });
        }
        Ok(sign(d == om && q(b) == q(a) + q(c)))
    }

    /// R-symbol `R^{ab}_c`: `-1.0` when admissible with `c = 0-` and `a`
    /// charged, `1.0` otherwise when admissible, exact `0.0` when not
    /// (`cu1irrep.jl:Rsymbol`).
    fn r_symbol(&self, a: &CU1Irrep, b: &CU1Irrep, c: &CU1Irrep) -> Result<f64, CU1Error> {
        if self.nsymbol(a, b, c) == 0 {
            return Ok(0.0);
        }
        Ok(if c.s == 1 && a.twice_charge > 0 {
            -1.0
        } else {
            1.0
        })
    }

    /// Opaque authority fingerprint of the `CU(1)` provider. Same contract as
    /// [`crate::category::FusionCategory::authority_fingerprint`]: compare by equality only; the epoch
    /// is bumped exactly when a returned value or convention changes.
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        Ok(None)
    }
    fn object_at(&self, index: u64) -> Result<CU1Irrep, EnumerationError> {
        match index {
            0 => Ok(CU1Irrep::VACUUM),
            1 => Ok(CU1Irrep::PSEUDOSCALAR),
            _ => u32::try_from(index - 1)
                .map(CU1Irrep::from_twice_charge)
                .map_err(|_| EnumerationError::Overflow),
        }
    }
    fn object_index(&self, a: &CU1Irrep) -> Result<u64, EnumerationError> {
        assert!(self.contains(a), "foreign CU(1) object");
        Ok(match a.twice_charge {
            0 => u64::from(a.s),
            q => u64::from(q) + 1,
        })
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"fusion-symbols:cu1:labels=u32-twice-charge-full:n=tks-cu1irrep-exact:f=tks-cu1irrep:r=tks-cu1irrep:dim=tks-cu1irrep:fs=tks-cu1irrep-one:twist=tks-rsymbol-sum-one:scalar=f64:epoch=1"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn j(twice_charge: u32) -> CU1Irrep {
        CU1Irrep::from_twice_charge(twice_charge)
    }

    const OP: CU1Irrep = CU1Irrep::VACUUM;
    const OM: CU1Irrep = CU1Irrep::PSEUDOSCALAR;

    fn labels() -> Vec<CU1Irrep> {
        [OP, OM].into_iter().chain((1..=5).map(j)).collect()
    }

    #[test]
    fn invalid_labels_are_typed_errors() {
        for s in [2, 3, u8::MAX] {
            assert_eq!(
                CU1Irrep::new(0, s),
                Err(CU1Error::InvalidLabel { twice_charge: 0, s })
            );
        }
        for s in [0, 1, 3] {
            assert_eq!(
                CU1Irrep::new(3, s),
                Err(CU1Error::InvalidLabel { twice_charge: 3, s })
            );
        }
        assert_eq!(
            CU1Irrep::new(u32::MAX, 0),
            Err(CU1Error::InvalidLabel {
                twice_charge: u32::MAX,
                s: 0
            })
        );
        assert_eq!(CU1Irrep::new(0, 0), Ok(OP));
        assert_eq!(CU1Irrep::new(0, 1), Ok(OM));
        assert_eq!(CU1Irrep::from_twice_charge(0), OP);
        // The whole u32 range is labels.
        assert_eq!(CU1Irrep::new(u32::MAX, 2), Ok(j(u32::MAX)));
        assert_eq!((j(u32::MAX).twice_charge(), j(u32::MAX).s()), (u32::MAX, 2));
    }

    #[test]
    fn order_is_tensorkitsectors_isless() {
        let mut shuffled = vec![j(2), OM, j(1), OP, j(5)];
        shuffled.sort();
        assert_eq!(shuffled, vec![OP, OM, j(1), j(2), j(5)]);
    }

    #[test]
    fn fusion_overflow_is_a_typed_error_and_nsymbol_stays_exact() {
        let max = j(u32::MAX);
        let half = j(u32::MAX / 2);
        let over = j(u32::MAX / 2 + 1);
        // The largest representable channels are labels.
        let channels: Vec<_> = CU1.fusion(&half, &over).unwrap().collect();
        assert_eq!(channels, vec![j(1), max]);
        let channels: Vec<_> = CU1.fusion(&j(1), &j(u32::MAX - 1)).unwrap().collect();
        assert_eq!(channels, vec![j(u32::MAX - 2), max]);
        // One past u32::MAX is the typed error.
        for (a, b, sum) in [
            (j(1), max, u64::from(u32::MAX) + 1),
            (over, over, u64::from(u32::MAX) + 1),
            (max, max, 2 * u64::from(u32::MAX)),
        ] {
            assert_eq!(
                CU1.fusion(&a, &b).map(|_| ()),
                Err(CU1Error::TwiceChargeOutOfRange { twice_charge: sum })
            );
        }
        // The difference channels exist although the sum does not.
        assert_eq!(CU1.nsymbol(&max, &j(1), &j(u32::MAX - 1)), 1);
        assert_eq!(CU1.nsymbol(&max, &max, &OP), 1);
        assert_eq!(CU1.nsymbol(&max, &max, &OM), 1);
        assert_eq!(CU1.nsymbol(&max, &max, &max), 0);
        assert_eq!(CU1.r_symbol(&max, &max, &OM).unwrap(), -1.0);
    }

    #[test]
    fn nsymbol_agrees_with_fusion() {
        for a in labels() {
            for b in labels() {
                let channels: Vec<_> = CU1.fusion(&a, &b).unwrap().collect();
                assert_eq!(CU1.fusion(&a, &b).unwrap().len(), channels.len());
                for c in labels() {
                    assert_eq!(CU1.nsymbol(&a, &b, &c) == 1, channels.contains(&c));
                }
                let dims: f64 = channels.iter().map(|c| CU1.dim(c)).sum();
                assert_eq!(dims, CU1.dim(&a) * CU1.dim(&b));
            }
        }
    }

    #[test]
    fn derived_quantities_follow_the_generic_tensorkitsectors_formulas() {
        // What: twist is `twist_from_Rsymbol` and the FS phase
        // `sign(F^{a ā a}_a[0+, 0+])` on this provider's own F and R, and the
        // F-moves are orthogonal (real unitary) on every (a, b, c, d) block.
        let all = labels();
        // Every channel of three labels of `all[..5]` (2j ≤ 3).
        let wide: Vec<_> = [OP, OM].into_iter().chain((1..=9).map(j)).collect();
        for &a in &all {
            let twist: f64 = CU1
                .fusion(&a, &a)
                .unwrap()
                .map(|b| CU1.dim(&b) / CU1.dim(&a) * CU1.r_symbol(&a, &a, &b).unwrap())
                .sum();
            assert_eq!(CU1.twist(&a), twist);
            assert_eq!(
                CU1.frobenius_schur_phase(&a),
                CU1.f_symbol(&a, &CU1.dual(&a), &a, &a, &OP, &OP)
                    .unwrap()
                    .signum()
            );
        }
        for &a in &all[..5] {
            for &b in &all[..5] {
                for &c in &all[..5] {
                    for &d in &wide {
                        let es: Vec<_> = wide
                            .iter()
                            .copied()
                            .filter(|&e| CU1.nsymbol(&a, &b, &e) * CU1.nsymbol(&e, &c, &d) == 1)
                            .collect();
                        let fs: Vec<_> = wide
                            .iter()
                            .copied()
                            .filter(|&f| CU1.nsymbol(&b, &c, &f) * CU1.nsymbol(&a, &f, &d) == 1)
                            .collect();
                        assert_eq!(es.len(), fs.len());
                        for &e1 in &es {
                            for &e2 in &es {
                                let dot: f64 = fs
                                    .iter()
                                    .map(|&f| {
                                        CU1.f_symbol(&a, &b, &c, &d, &e1, &f).unwrap()
                                            * CU1.f_symbol(&a, &b, &c, &d, &e2, &f).unwrap()
                                    })
                                    .sum();
                                let want = if e1 == e2 { 1.0 } else { 0.0 };
                                assert!((dot - want).abs() < 1e-15, "{a:?}{b:?}{c:?}{d:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fingerprint_is_pinned_and_distinct() {
        assert_eq!(
            CU1.authority_fingerprint(),
            &b"fusion-symbols:cu1:labels=u32-twice-charge-full:n=tks-cu1irrep-exact:f=tks-cu1irrep:r=tks-cu1irrep:dim=tks-cu1irrep:fs=tks-cu1irrep-one:twist=tks-rsymbol-sum-one:scalar=f64:epoch=1"[..]
        );
        let fibonacci = crate::table::TableCategory::fibonacci();
        for other in [
            super::super::u1::U1_FINGERPRINT,
            crate::finite::LEGACY_PARITY_FINGERPRINT,
            fibonacci.authority_fingerprint(),
        ] {
            assert_ne!(CU1.authority_fingerprint(), other);
        }
    }
}
