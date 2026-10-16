//! `CU(1) = U(1) ⋊ C` (U(1) with charge conjugation, also `O(2)`) in closed
//! form.
//!
//! Every value follows TensorKitSectors.jl 0.3.9 (git `67d342a1`)
//! `src/irreps/cu1irrep.jl` (`CU1Irrep`: constructor, `dual`, `⊗`, `dim`,
//! `Nsymbol`, `Fsymbol`, `Rsymbol`, `frobenius_schur_phase`), with the generic
//! `src/sectors.jl:twist_from_Rsymbol` for the twist, which `CU1Irrep` does
//! not override. Never "simplify" a sign or branch of [`cu1_f_symbol`]
//! without rereading `cu1irrep.jl:Fsymbol`: the gauge is in its branch order.
//!
//! The irreps are `0+` (the unit), `0-` (the pseudoscalar, charge
//! conjugation acting by `-1`) and, for each half-integer `j > 0`, a
//! two-dimensional irrep. Fusion is multiplicity free; every irrep is
//! self-dual; the braiding is bosonic (twist `1`) with
//! `R^{ab}_{0-} = -1` for charged `a`, `b`; the Frobenius–Schur phase is `1`.
//!
//! Scalars are `f64`, as TensorKitSectors' `fusionscalartype` and
//! `braidingscalartype` (`Float64`). The values reproduce bit for bit those of
//! the provider this was ported from, symtn `symtn-sectors/src/cu1.rs`, and
//! those measured from TensorKitSectors (`tests/cu1_values.rs`).
//!
//! Label domain: `twice_charge = 2j` in `0 ..= CU1_MAX_TWICE_CHARGE`
//! (`u32::MAX - 1`), symtn's bound (see [`CU1_MAX_TWICE_CHARGE`]).

/// Largest `twice_charge` of a [`CU1Irrep`].
///
/// Why `u32::MAX - 1`, not `u32::MAX`: it is the bound of symtn's CU(1)
/// sector codec, the consumer this module replaces (symtn #2016), which
/// encodes `j > 0` as sector id `2j + 1` and keeps that id within 32 bits.
/// Using the same bound makes the label sets of the two crates identical, so
/// every label here is representable there and no fusion result exists on
/// only one side.
pub const CU1_MAX_TWICE_CHARGE: u32 = u32::MAX - 1;

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

/// An ill-posed or inadmissible `CU(1)` request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CU1Error {
    /// The vertex `left ⊗ right → coupled` has `N = 0`. For an F-symbol this
    /// is the first failing vertex in the order `(a, b, e)`, `(e, c, d)`,
    /// `(b, c, f)`, `(a, f, d)`.
    NotAdmissible {
        /// First label of the vertex.
        left: CU1Irrep,
        /// Second label of the vertex.
        right: CU1Irrep,
        /// Coupled label of the vertex.
        coupled: CU1Irrep,
    },
    /// `(twice_charge, s)` is not a `CU1Irrep`: `s` must be `0` or `1` for
    /// `twice_charge == 0` and `2` otherwise (`cu1irrep.jl` constructor).
    InvalidLabel {
        /// The requested twice charge.
        twice_charge: u32,
        /// The requested `s`.
        s: u8,
    },
    /// A twice charge above [`CU1_MAX_TWICE_CHARGE`], either requested
    /// directly or produced by [`CU1Irrep::fusion`].
    TwiceChargeOutOfRange {
        /// The exact twice charge.
        twice_charge: u64,
    },
}

impl std::fmt::Display for CU1Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CU1Error::NotAdmissible {
                left,
                right,
                coupled,
            } => write!(
                f,
                "inadmissible CU(1) vertex: {left:?} ⊗ {right:?} does not contain {coupled:?}"
            ),
            CU1Error::InvalidLabel { twice_charge, s } => write!(
                f,
                "not a CU(1) irrep: twice_charge {twice_charge}, s {s} (s ∈ {{0, 1}} for charge 0, s = 2 otherwise)"
            ),
            CU1Error::TwiceChargeOutOfRange { twice_charge } => write!(
                f,
                "CU(1) twice charge {twice_charge} exceeds {CU1_MAX_TWICE_CHARGE}"
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
    /// [`CU1Error::InvalidLabel`] for an `s` the charge does not admit,
    /// [`CU1Error::TwiceChargeOutOfRange`] above [`CU1_MAX_TWICE_CHARGE`].
    pub fn new(twice_charge: u32, s: u8) -> Result<Self, CU1Error> {
        if twice_charge > CU1_MAX_TWICE_CHARGE {
            return Err(CU1Error::TwiceChargeOutOfRange {
                twice_charge: u64::from(twice_charge),
            });
        }
        let valid = if twice_charge == 0 { s <= 1 } else { s == 2 };
        if valid {
            Ok(Self { twice_charge, s })
        } else {
            Err(CU1Error::InvalidLabel { twice_charge, s })
        }
    }

    /// TensorKitSectors `CU1Irrep(j)` with its default `s`: `0+` for
    /// `twice_charge == 0`, the two-dimensional irrep otherwise.
    ///
    /// # Errors
    ///
    /// [`CU1Error::TwiceChargeOutOfRange`] above [`CU1_MAX_TWICE_CHARGE`].
    pub fn from_twice_charge(twice_charge: u32) -> Result<Self, CU1Error> {
        Self::new(twice_charge, if twice_charge == 0 { 0 } else { 2 })
    }

    fn from_exact(twice_charge: u64) -> Result<Self, CU1Error> {
        match u32::try_from(twice_charge) {
            Ok(q) if q <= CU1_MAX_TWICE_CHARGE => Self::from_twice_charge(q),
            _ => Err(CU1Error::TwiceChargeOutOfRange { twice_charge }),
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

    /// Dimension: `1` for `j = 0`, `2` otherwise (`cu1irrep.jl:dim`).
    pub fn dim(self) -> f64 {
        if self.twice_charge == 0 {
            1.0
        } else {
            2.0
        }
    }

    /// The dual irrep: every irrep is self-dual (`cu1irrep.jl:dual`).
    pub fn dual(self) -> Self {
        self
    }

    /// Frobenius–Schur phase, `1` (`cu1irrep.jl:frobenius_schur_phase`).
    pub fn frobenius_schur(self) -> f64 {
        1.0
    }

    /// Twist, `1`.
    ///
    /// No `CU1Irrep` override; the generic `sectors.jl:twist_from_Rsymbol`,
    /// `θ_a = Σ_{b ∈ a⊗a} d_b/d_a · R^{aa}_b`, is `1` for `0±` and
    /// `(1 - 1 + 2)/2 = 1` for `j > 0` (`b = 0+, 0-, 2j`).
    pub fn twist(self) -> f64 {
        1.0
    }

    /// Fusion channels of `self ⊗ other`, each with multiplicity one, in
    /// TensorKitSectors' `CU1IrrepProdIterator` order: `0± ⊗ 0± = 0+` or `0-`
    /// (s added mod 2); `0± ⊗ j = j`; `j ⊗ j = 0+ ⊕ 0- ⊕ 2j`; otherwise
    /// `|j - k| ⊕ (j + k)`.
    ///
    /// # Errors
    ///
    /// [`CU1Error::TwiceChargeOutOfRange`] when `2(j + k)` exceeds
    /// [`CU1_MAX_TWICE_CHARGE`].
    pub fn fusion(self, other: Self) -> Result<CU1Fusion, CU1Error> {
        let (a, b) = (self, other);
        let mut channels = [Self::VACUUM; 3];
        let len = if a.twice_charge == 0 && b.twice_charge == 0 {
            channels[0] = Self {
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
            let sum = Self::from_exact(u64::from(a.twice_charge) + u64::from(b.twice_charge))?;
            if a == b {
                channels = [Self::VACUUM, Self::PSEUDOSCALAR, sum];
                3
            } else {
                // `|j - k| > 0` and below `j + k`, so it is a charged label.
                channels[0] = Self {
                    twice_charge: a.twice_charge.abs_diff(b.twice_charge),
                    s: 2,
                };
                channels[1] = sum;
                2
            }
        };
        Ok(CU1Fusion(channels.into_iter().take(len)))
    }
}

/// Allocation-free iterator over the fusion channels of two `CU(1)` irreps
/// (see [`CU1Irrep::fusion`]).
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

/// Fusion multiplicity `N^{ab}_c ∈ {0, 1}` (`cu1irrep.jl:Nsymbol`), exact for
/// every label triple: the charge sum is compared in `u64`, so a fusion that
/// [`CU1Irrep::fusion`] cannot represent still has `N = 0` here.
pub fn cu1_nsymbol(a: CU1Irrep, b: CU1Irrep, c: CU1Irrep) -> u32 {
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

fn check_vertex(left: CU1Irrep, right: CU1Irrep, coupled: CU1Irrep) -> Result<(), CU1Error> {
    if cu1_nsymbol(left, right, coupled) == 1 {
        Ok(())
    } else {
        Err(CU1Error::NotAdmissible {
            left,
            right,
            coupled,
        })
    }
}

/// F-symbol `F^{abc}_d[e, f]`; exact `0.0` when a vertex is inadmissible.
///
/// `cu1irrep.jl:Fsymbol`, branch for branch: `1` when `a`, `b` or `c` is
/// `0+` or two of them are `0-`; `±1` when one is `0-`; then entries `±1`,
/// `±1/2`, `±√2/2` or `0` for charged `a`, `b`, `c`.
pub fn cu1_f_symbol(
    a: CU1Irrep,
    b: CU1Irrep,
    c: CU1Irrep,
    d: CU1Irrep,
    e: CU1Irrep,
    f: CU1Irrep,
) -> f64 {
    cu1_f_symbol_checked(a, b, c, d, e, f).unwrap_or(0.0)
}

/// Checked [`cu1_f_symbol`]: [`CU1Error::NotAdmissible`] for the first
/// failing vertex instead of a zero.
pub fn cu1_f_symbol_checked(
    a: CU1Irrep,
    b: CU1Irrep,
    c: CU1Irrep,
    d: CU1Irrep,
    e: CU1Irrep,
    f: CU1Irrep,
) -> Result<f64, CU1Error> {
    check_vertex(a, b, e)?;
    check_vertex(e, c, d)?;
    check_vertex(b, c, f)?;
    check_vertex(a, f, d)?;
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
pub fn cu1_r_symbol(a: CU1Irrep, b: CU1Irrep, c: CU1Irrep) -> f64 {
    cu1_r_symbol_checked(a, b, c).unwrap_or(0.0)
}

/// Checked [`cu1_r_symbol`]: [`CU1Error::NotAdmissible`] instead of a zero.
pub fn cu1_r_symbol_checked(a: CU1Irrep, b: CU1Irrep, c: CU1Irrep) -> Result<f64, CU1Error> {
    check_vertex(a, b, c)?;
    Ok(if c.s == 1 && a.twice_charge > 0 {
        -1.0
    } else {
        1.0
    })
}

/// Opaque authority fingerprint of the `CU(1)` provider. Same contract as
/// [`crate::su2_authority_fingerprint`]: compare by equality only; the epoch
/// is bumped exactly when a returned value or convention changes.
pub fn cu1_authority_fingerprint() -> &'static [u8] {
    b"fusion-symbols:cu1:labels=u32-twice-charge-max-u32-minus-1:n=tks-cu1irrep-exact:f=tks-cu1irrep:r=tks-cu1irrep:dim=tks-cu1irrep:fs=tks-cu1irrep-one:twist=tks-rsymbol-sum-one:scalar=f64:epoch=1"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn j(twice_charge: u32) -> CU1Irrep {
        CU1Irrep::from_twice_charge(twice_charge).expect("label")
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
        let out = Err(CU1Error::TwiceChargeOutOfRange {
            twice_charge: u64::from(u32::MAX),
        });
        assert_eq!(CU1Irrep::new(u32::MAX, 2), out);
        assert_eq!(CU1Irrep::from_twice_charge(u32::MAX), out);
        assert_eq!(CU1Irrep::new(0, 0), Ok(OP));
        assert_eq!(CU1Irrep::new(0, 1), Ok(OM));
        assert_eq!(CU1Irrep::from_twice_charge(0), Ok(OP));
        let max = j(CU1_MAX_TWICE_CHARGE);
        assert_eq!((max.twice_charge(), max.s()), (CU1_MAX_TWICE_CHARGE, 2));
    }

    #[test]
    fn order_is_tensorkitsectors_isless() {
        let mut shuffled = vec![j(2), OM, j(1), OP, j(5)];
        shuffled.sort();
        assert_eq!(shuffled, vec![OP, OM, j(1), j(2), j(5)]);
    }

    #[test]
    fn fusion_overflow_is_a_typed_error_and_nsymbol_stays_exact() {
        let half = j(CU1_MAX_TWICE_CHARGE / 2);
        let over = j(CU1_MAX_TWICE_CHARGE / 2 + 1);
        let channels: Vec<_> = half.fusion(half).unwrap().collect();
        assert_eq!(channels, vec![OP, OM, j(CU1_MAX_TWICE_CHARGE)]);
        assert_eq!(
            half.fusion(over).map(|_| ()),
            Err(CU1Error::TwiceChargeOutOfRange {
                twice_charge: u64::from(CU1_MAX_TWICE_CHARGE) + 1
            })
        );
        let max = j(CU1_MAX_TWICE_CHARGE);
        assert_eq!(
            max.fusion(max).map(|_| ()),
            Err(CU1Error::TwiceChargeOutOfRange {
                twice_charge: 2 * u64::from(CU1_MAX_TWICE_CHARGE)
            })
        );
        // The difference channel exists although the sum does not.
        assert_eq!(cu1_nsymbol(max, j(1), j(CU1_MAX_TWICE_CHARGE - 1)), 1);
        assert_eq!(cu1_nsymbol(max, max, OP), 1);
        assert_eq!(cu1_nsymbol(max, max, OM), 1);
        assert_eq!(cu1_nsymbol(max, max, j(CU1_MAX_TWICE_CHARGE)), 0);
        assert_eq!(cu1_r_symbol(max, max, OM), -1.0);
    }

    #[test]
    fn nsymbol_agrees_with_fusion() {
        for a in labels() {
            for b in labels() {
                let channels: Vec<_> = a.fusion(b).unwrap().collect();
                assert_eq!(a.fusion(b).unwrap().len(), channels.len());
                for c in labels() {
                    assert_eq!(cu1_nsymbol(a, b, c) == 1, channels.contains(&c));
                }
                let dims: f64 = channels.iter().map(|c| c.dim()).sum();
                assert_eq!(dims, a.dim() * b.dim());
            }
        }
    }

    #[test]
    fn checked_symbols_name_the_first_failing_vertex() {
        assert_eq!(
            cu1_r_symbol_checked(j(1), j(1), j(1)),
            Err(CU1Error::NotAdmissible {
                left: j(1),
                right: j(1),
                coupled: j(1)
            })
        );
        // (a, b, e) = (1/2, 1/2, 0-) holds; (e, c, d) = (0-, 1/2, 1) does not.
        assert_eq!(
            cu1_f_symbol_checked(j(1), j(1), j(1), j(2), OM, j(2)),
            Err(CU1Error::NotAdmissible {
                left: OM,
                right: j(1),
                coupled: j(2)
            })
        );
        for a in labels() {
            for b in labels() {
                for c in labels() {
                    let checked = cu1_r_symbol_checked(a, b, c);
                    assert_eq!(checked.is_ok(), cu1_nsymbol(a, b, c) == 1);
                    assert_eq!(checked.unwrap_or(0.0), cu1_r_symbol(a, b, c));
                }
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
            let twist: f64 = a
                .fusion(a)
                .unwrap()
                .map(|b| b.dim() / a.dim() * cu1_r_symbol(a, a, b))
                .sum();
            assert_eq!(a.twist(), twist);
            assert_eq!(
                a.frobenius_schur(),
                cu1_f_symbol(a, a.dual(), a, a, OP, OP).signum()
            );
        }
        for &a in &all[..5] {
            for &b in &all[..5] {
                for &c in &all[..5] {
                    for &d in &wide {
                        let es: Vec<_> = wide
                            .iter()
                            .copied()
                            .filter(|&e| cu1_nsymbol(a, b, e) * cu1_nsymbol(e, c, d) == 1)
                            .collect();
                        let fs: Vec<_> = wide
                            .iter()
                            .copied()
                            .filter(|&f| cu1_nsymbol(b, c, f) * cu1_nsymbol(a, f, d) == 1)
                            .collect();
                        assert_eq!(es.len(), fs.len());
                        for &e1 in &es {
                            for &e2 in &es {
                                let dot: f64 = fs
                                    .iter()
                                    .map(|&f| {
                                        cu1_f_symbol(a, b, c, d, e1, f)
                                            * cu1_f_symbol(a, b, c, d, e2, f)
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
            cu1_authority_fingerprint(),
            &b"fusion-symbols:cu1:labels=u32-twice-charge-max-u32-minus-1:n=tks-cu1irrep-exact:f=tks-cu1irrep:r=tks-cu1irrep:dim=tks-cu1irrep:fs=tks-cu1irrep-one:twist=tks-rsymbol-sum-one:scalar=f64:epoch=1"[..]
        );
        for other in [
            crate::abelian::u1_authority_fingerprint(),
            crate::abelian::z2_authority_fingerprint(),
            crate::fibonacci::fibonacci_authority_fingerprint(),
        ] {
            assert_ne!(cu1_authority_fingerprint(), other);
        }
    }
}
