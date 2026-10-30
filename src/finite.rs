//! Finite groups: closed-form cyclic categories [`ZN`] (including Z2).
//! Non-abelian finite groups are not provided.
//!
//! Labels are validated at construction. Symbols are exact `f64` zeros and
//! ones, with unique fusion; inadmissible vertices give zero, not errors.

use crate::category::{
    BraidingStyle, EnumerationError, FusionCategory, FusionStyle, GroupRepresentations, RBlock,
};
use std::{convert::Infallible, num::NonZeroU32};

/// An invalid cyclic modulus or noncanonical charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZNError {
    /// A charge outside the canonical range `0..modulus`.
    NonCanonicalCharge {
        /// Requested charge (never reduced).
        charge: u32,
        /// Requested modulus.
        modulus: u32,
    },
    /// A `Z_N` modulus of zero.
    ZeroModulus,
}

impl std::fmt::Display for ZNError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZNError::NonCanonicalCharge { charge, modulus } => {
                write!(f, "charge {charge} is outside 0..{modulus}")
            }
            ZNError::ZeroModulus => write!(f, "Z_N modulus must be nonzero"),
        }
    }
}

impl std::error::Error for ZNError {}

/// An irrep of the cyclic group `Z_N`: a charge `n mod N`, with the modulus
/// `N ≥ 1` carried by the label.
///
/// TensorKitSectors `ZNIrrep{N}` / `LargeZNIrrep{N}` fix `N` in the type; here
/// it is a runtime value. Category methods panic on a foreign modulus.
/// `Ord` compares the modulus, then the charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ZNIrrep {
    modulus: NonZeroU32,
    charge: u32,
}

impl ZNIrrep {
    /// The irrep of charge `charge mod modulus` (any integer, reduced as
    /// TensorKitSectors `ZNIrrep{N}(n) = mod(n, N)`).
    ///
    /// # Errors
    ///
    /// [`ZNError::ZeroModulus`] for `modulus == 0`.
    pub fn new(charge: i64, modulus: u32) -> Result<Self, ZNError> {
        let modulus = NonZeroU32::new(modulus).ok_or(ZNError::ZeroModulus)?;
        // rem_euclid of an i64 by a positive u32 lies in 0..modulus.
        let charge = charge.rem_euclid(i64::from(modulus.get())) as u32;
        Ok(Self { modulus, charge })
    }

    /// Construct a canonical charge without reduction.
    ///
    /// # Errors
    /// Returns `ZeroModulus` for zero modulus, otherwise `NonCanonicalCharge`
    /// when `charge >= modulus`.
    pub fn from_canonical(charge: u32, modulus: u32) -> Result<Self, ZNError> {
        let modulus = NonZeroU32::new(modulus).ok_or(ZNError::ZeroModulus)?;
        if charge >= modulus.get() {
            return Err(ZNError::NonCanonicalCharge {
                charge,
                modulus: modulus.get(),
            });
        }
        Ok(Self { modulus, charge })
    }

    /// The charge, in `0..modulus`.
    pub const fn charge(self) -> u32 {
        self.charge
    }

    /// The modulus `N`.
    pub const fn modulus(self) -> u32 {
        self.modulus.get()
    }

    fn with_charge(self, charge: u64) -> Self {
        Self {
            modulus: self.modulus,
            // Below the u32 modulus.
            charge: (charge % u64::from(self.modulus.get())) as u32,
        }
    }
}

/// Cyclic bosonic category with validated, modulus-tagged objects.
///
/// Enumeration (frozen): the `N` charges `0, 1, …, N − 1`; the index of a
/// charge is the charge itself.
///
/// # Panics
/// Every object-taking method except `contains` panics if any object's modulus differs from
/// this category's modulus, even when another vertex has `N = 0`.
///
/// # Examples
///
/// ```
/// use fusion_symbols::{category::FusionCategory, finite::{ZNError, ZNIrrep, ZN}};
///
/// let z3 = ZN::new(3)?;
/// let two = ZNIrrep::new(2, 3)?;
/// assert_eq!(z3.fusion(&two, &two)?.next(), Some(ZNIrrep::new(1, 3)?)); // 2 + 2 = 1 mod 3
/// assert_eq!(z3.dual(&two), ZNIrrep::new(1, 3)?);
/// assert_eq!(ZN::new(0), Err(ZNError::ZeroModulus));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ZN {
    modulus: NonZeroU32,
}

impl ZN {
    /// Construct a cyclic category; zero modulus returns `ZeroModulus`.
    pub fn new(modulus: u32) -> Result<Self, ZNError> {
        Ok(Self {
            modulus: NonZeroU32::new(modulus).ok_or(ZNError::ZeroModulus)?,
        })
    }

    /// The nonzero modulus defining this category's mathematical identity.
    pub const fn modulus(&self) -> u32 {
        self.modulus.get()
    }

    fn validate(&self, objects: &[&ZNIrrep]) {
        for a in objects {
            assert_eq!(a.modulus, self.modulus, "foreign cyclic object");
        }
    }
}

impl FusionCategory for ZN {
    type Object = ZNIrrep;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = std::iter::Once<ZNIrrep>;
    type Error = Infallible;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> ZNIrrep {
        ZNIrrep {
            modulus: self.modulus,
            charge: 0,
        }
    }
    fn contains(&self, a: &ZNIrrep) -> bool {
        a.modulus == self.modulus && a.charge < self.modulus()
    }
    fn dual(&self, a: &ZNIrrep) -> ZNIrrep {
        self.validate(&[a]);
        a.with_charge(u64::from(self.modulus()) - u64::from(a.charge))
    }
    fn dim(&self, a: &ZNIrrep) -> f64 {
        self.validate(&[a]);
        1.0
    }
    fn frobenius_schur_phase(&self, a: &ZNIrrep) -> f64 {
        self.validate(&[a]);
        1.0
    }
    fn frobenius_schur_indicator(&self, a: &ZNIrrep) -> f64 {
        f64::from(u8::from(*a == self.dual(a)))
    }
    fn twist(&self, a: &ZNIrrep) -> f64 {
        self.validate(&[a]);
        1.0
    }
    fn fusion<'a>(&'a self, a: &ZNIrrep, b: &ZNIrrep) -> Result<Self::Fusion<'a>, Self::Error> {
        self.validate(&[a, b]);
        Ok(std::iter::once(
            a.with_charge(u64::from(a.charge) + u64::from(b.charge)),
        ))
    }
    fn nsymbol(&self, a: &ZNIrrep, b: &ZNIrrep, c: &ZNIrrep) -> u32 {
        self.validate(&[a, b, c]);
        u32::from(
            (u64::from(a.charge) + u64::from(b.charge)) % u64::from(self.modulus())
                == u64::from(c.charge),
        )
    }
    fn f_symbol(
        &self,
        a: &ZNIrrep,
        b: &ZNIrrep,
        c: &ZNIrrep,
        d: &ZNIrrep,
        e: &ZNIrrep,
        f: &ZNIrrep,
    ) -> Result<f64, Self::Error> {
        self.validate(&[a, b, c, d, e, f]);
        Ok(f64::from(
            self.nsymbol(a, b, e)
                * self.nsymbol(e, c, d)
                * self.nsymbol(b, c, f)
                * self.nsymbol(a, f, d),
        ))
    }
    fn r_symbol(&self, a: &ZNIrrep, b: &ZNIrrep, c: &ZNIrrep) -> Result<f64, Self::Error> {
        Ok(f64::from(self.nsymbol(a, b, c)))
    }
    // TensorKitSectors `irreps.jl` (AbelianIrrep): A and B are N, since every
    // admissible F and every dimension is one.
    fn a_symbol(&self, a: &ZNIrrep, b: &ZNIrrep, c: &ZNIrrep) -> Result<RBlock<f64>, Self::Error> {
        Ok(if self.nsymbol(a, b, c) == 0 {
            RBlock::zeros(0)
        } else {
            RBlock::one(1.0)
        })
    }
    fn b_symbol(&self, a: &ZNIrrep, b: &ZNIrrep, c: &ZNIrrep) -> Result<RBlock<f64>, Self::Error> {
        Ok(if self.nsymbol(a, b, c) == 0 {
            RBlock::zeros(0)
        } else {
            RBlock::one(1.0)
        })
    }
    // F is the product of the four vertex N, so exactly `1` when admissible;
    // the unit is the self-dual identity charge.
    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        Ok(Some(u64::from(self.modulus())))
    }
    fn object_at(&self, index: u64) -> Result<ZNIrrep, EnumerationError> {
        match u32::try_from(index) {
            Ok(charge) if charge < self.modulus() => Ok(ZNIrrep {
                modulus: self.modulus,
                charge,
            }),
            _ => Err(EnumerationError::OutOfRange),
        }
    }
    fn object_index(&self, a: &ZNIrrep) -> Result<u64, EnumerationError> {
        self.validate(&[a]);
        Ok(u64::from(a.charge))
    }
}

impl GroupRepresentations for ZN {
    type Cgc = f64;
    fn cgc(&self, a: &ZNIrrep, b: &ZNIrrep, c: &ZNIrrep) -> Result<f64, Infallible> {
        Ok(f64::from(self.nsymbol(a, b, c)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fermion::{FermionParity, FermionParityIrrep};
    use crate::lie::{U1Error, U1Irrep, U1};

    fn zn(charge: i64, modulus: u32) -> ZNIrrep {
        ZNIrrep::new(charge, modulus).unwrap()
    }
    fn u1(charge: i32) -> U1Irrep {
        U1Irrep::new(charge).unwrap()
    }

    #[test]
    fn constructors_and_integer_extremes() {
        assert_eq!(ZN::new(0), Err(ZNError::ZeroModulus));
        assert_eq!(ZNIrrep::new(3, 0), Err(ZNError::ZeroModulus));
        assert_eq!(ZNIrrep::from_canonical(0, 0), Err(ZNError::ZeroModulus));
        assert_eq!(
            ZNIrrep::from_canonical(3, 3),
            Err(ZNError::NonCanonicalCharge {
                charge: 3,
                modulus: 3
            })
        );
        assert_eq!(
            ZNIrrep::from_canonical(u32::MAX, 3),
            Err(ZNError::NonCanonicalCharge {
                charge: u32::MAX,
                modulus: 3
            })
        );
        assert_eq!(ZNIrrep::from_canonical(2, 3), Ok(zn(2, 3)));
        assert_eq!(zn(-1, 3).charge(), 2);
        assert_eq!(zn(i64::MIN, 7).charge(), i64::MIN.rem_euclid(7) as u32);
        assert_eq!(
            zn(i64::MAX, u32::MAX).charge(),
            (i64::MAX % i64::from(u32::MAX)) as u32
        );
        let cat = ZN::new(u32::MAX).unwrap();
        let edge = zn(i64::from(u32::MAX) - 1, u32::MAX);
        assert_eq!(
            cat.fusion(&edge, &edge).unwrap().next(),
            Some(zn(i64::from(u32::MAX) - 2, u32::MAX))
        );
        assert_eq!(cat.dual(&edge), zn(1, u32::MAX));
        assert_eq!(ZN::new(5).unwrap().dual(&zn(0, 5)), zn(0, 5));
        assert_eq!(zn(5, 1), ZN::new(1).unwrap().unit());
        assert_eq!(
            U1Irrep::new(i32::MIN),
            Err(U1Error::LabelOverflow {
                charge: i64::from(i32::MIN)
            })
        );
        for q in [i32::MIN + 1, i32::MAX] {
            assert_eq!(U1.dual(&U1.dual(&u1(q))), u1(q));
            assert_eq!(U1.dual(&u1(q)).charge(), -q);
        }
    }

    #[test]
    fn overflow_only_fails_fusion() {
        let (max, min) = (u1(i32::MAX), u1(i32::MIN + 1));
        assert_eq!(
            U1.fusion(&max, &u1(1)).unwrap_err(),
            U1Error::LabelOverflow {
                charge: i64::from(i32::MAX) + 1
            }
        );
        assert_eq!(
            U1.fusion(&min, &u1(-1)).unwrap_err(),
            U1Error::LabelOverflow {
                charge: i64::from(i32::MIN)
            }
        );
        assert_eq!(U1.fusion(&max, &min).unwrap().next(), Some(U1Irrep::UNIT));
        assert_eq!(U1.fusion(&max, &U1Irrep::UNIT).unwrap().next(), Some(max));
        for c in [min, U1Irrep::UNIT, max] {
            assert_eq!(U1.nsymbol(&max, &max, &c), 0);
            assert_eq!(U1.r_symbol(&max, &max, &c), Ok(0.0));
            assert_eq!(U1.f_symbol(&max, &max, &c, &c, &c, &c), Ok(0.0));
        }
        assert_eq!(U1.nsymbol(&max, &min, &U1Irrep::UNIT), 1);
    }

    #[test]
    fn inadmissible_vertices_are_zero_values() {
        let cat = ZN::new(2).unwrap();
        let (e, o) = (zn(0, 2), zn(1, 2));
        assert_eq!(cat.r_symbol(&o, &o, &o), Ok(0.0));
        assert_eq!(cat.f_symbol(&o, &o, &o, &e, &e, &e), Ok(0.0));
        let cat = ZN::new(3).unwrap();
        let z = |q| zn(q, 3);
        assert_eq!(
            cat.f_symbol(&z(1), &z(1), &z(1), &z(0), &z(2), &z(1)),
            Ok(0.0)
        );
        assert_eq!(
            cat.f_symbol(&z(1), &z(1), &z(1), &z(0), &z(2), &z(2)),
            Ok(1.0)
        );
        assert_eq!(
            U1.f_symbol(&u1(1), &u1(1), &u1(-1), &u1(1), &u1(2), &u1(0)),
            Ok(1.0)
        );
        assert_eq!(
            U1.f_symbol(&u1(1), &u1(1), &u1(-1), &u1(1), &u1(2), &u1(1)),
            Ok(0.0)
        );
        for a in [false, true] {
            for b in [false, true] {
                for c in [false, true] {
                    let (x, y, z) = (
                        zn(i64::from(a), 2),
                        zn(i64::from(b), 2),
                        zn(i64::from(c), 2),
                    );
                    assert_eq!(
                        ZN::new(2).unwrap().r_symbol(&x, &y, &z),
                        Ok(f64::from(ZN::new(2).unwrap().nsymbol(&x, &y, &z)))
                    );
                    let (x, y, z) = (
                        FermionParityIrrep::new(a),
                        FermionParityIrrep::new(b),
                        FermionParityIrrep::new(c),
                    );
                    let n = f64::from(FermionParity.nsymbol(&x, &y, &z));
                    assert_eq!(
                        FermionParity.r_symbol(&x, &y, &z),
                        Ok(if a && b && n != 0.0 { -n } else { n })
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "foreign cyclic object")]
    fn foreign_fusion_panics() {
        ZN::new(3).unwrap().fusion(&zn(1, 3), &zn(1, 4)).unwrap();
    }

    #[test]
    #[should_panic(expected = "foreign cyclic object")]
    fn foreign_r_input_panics() {
        ZN::new(3)
            .unwrap()
            .r_symbol(&zn(1, 3), &zn(1, 4), &zn(2, 3))
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "foreign cyclic object")]
    fn foreign_r_output_panics() {
        ZN::new(3)
            .unwrap()
            .r_symbol(&zn(1, 3), &zn(1, 3), &zn(2, 4))
            .unwrap();
    }

    #[test]
    #[should_panic(expected = "foreign cyclic object")]
    fn foreign_n_panics() {
        ZN::new(3).unwrap().nsymbol(&zn(1, 3), &zn(1, 4), &zn(2, 3));
    }

    #[test]
    #[should_panic(expected = "foreign cyclic object")]
    fn foreign_f_panics_even_after_inadmissible_vertex() {
        let cat = ZN::new(3).unwrap();
        let u = cat.unit();
        cat.f_symbol(&u, &u, &u, &u, &zn(1, 3), &zn(0, 4)).unwrap();
    }
    /// Checked against TensorKitSectors 0.3.9 (git `67d342a1`, Julia 1.11.6):
    /// duals, `dim`, `frobenius_schur_phase` and `twist` of every Z2, Z3, Z4
    /// and FermionParity label; every `Rsymbol` that differs from `Nsymbol`
    /// (only fZ2 `R(1, 1, 0) = -1`); `Fsymbol` = the product of its four
    /// `Nsymbol`s (`irreps.jl`) and its nonzero count over all label tuples;
    /// and U(1) duals, fusion with charge 1, and two F/R values.
    #[test]
    fn matches_measured_tensorkitsectors_values() {
        // (N, duals by charge, nonzero F count); dim, FS phase, twist are 1.
        let zn_tks: [(u32, &[u32], usize); 3] =
            [(2, &[0, 1], 8), (3, &[0, 2, 1], 27), (4, &[0, 3, 2, 1], 64)];
        for (n, duals, nonzero_f) in zn_tks {
            let cat = ZN::new(n).unwrap();
            let labels: Vec<ZNIrrep> = (0..i64::from(n)).map(|q| zn(q, n)).collect();
            for (a, &dual) in labels.iter().zip(duals) {
                assert_eq!(cat.dual(a).charge(), dual);
                assert_eq!(
                    (cat.dim(a), cat.frobenius_schur_phase(a), cat.twist(a)),
                    (1.0, 1.0, 1.0)
                );
            }
            let mut count = 0;
            for &a in &labels {
                for &b in &labels {
                    for &c in &labels {
                        assert_eq!(
                            cat.r_symbol(&a, &b, &c).unwrap(),
                            f64::from(cat.nsymbol(&a, &b, &c))
                        );
                        for &d in &labels {
                            for &e in &labels {
                                for &f in &labels {
                                    // `irreps.jl:Fsymbol` for abelian irreps.
                                    let n = cat.nsymbol(&a, &b, &e)
                                        * cat.nsymbol(&e, &c, &d)
                                        * cat.nsymbol(&b, &c, &f)
                                        * cat.nsymbol(&a, &f, &d);
                                    assert_eq!(
                                        cat.f_symbol(&a, &b, &c, &d, &e, &f).unwrap(),
                                        f64::from(n)
                                    );
                                    count += usize::from(n != 0);
                                }
                            }
                        }
                    }
                }
            }
            assert_eq!(count, nonzero_f, "Z{n}");
        }
        let (e, o) = (FermionParityIrrep::EVEN, FermionParityIrrep::ODD);
        assert_eq!((FermionParity.dual(&e), FermionParity.dual(&o)), (e, o));
        assert_eq!(
            (FermionParity.twist(&e), FermionParity.twist(&o)),
            (1.0, -1.0)
        );
        assert_eq!(
            (
                FermionParity.frobenius_schur_phase(&e),
                FermionParity.frobenius_schur_phase(&o)
            ),
            (1.0, 1.0)
        );
        let mut count = 0;
        for a in [e, o] {
            for b in [e, o] {
                for c in [e, o] {
                    let n = f64::from(FermionParity.nsymbol(&a, &b, &c));
                    let tks = if (a, b, c) == (o, o, e) { -1.0 } else { n };
                    assert_eq!(FermionParity.r_symbol(&a, &b, &c).unwrap(), tks);
                    for d in [e, o] {
                        for x in [e, o] {
                            for y in [e, o] {
                                count += usize::from(
                                    FermionParity.f_symbol(&a, &b, &c, &d, &x, &y).unwrap() != 0.0,
                                );
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(count, 8);
        for q in -2..=2 {
            assert_eq!(U1.dual(&u1(q)), u1(-q));
            assert_eq!(
                U1.fusion(&u1(q), &u1(1)).map(|mut it| it.next().unwrap()),
                Ok(u1(q + 1))
            );
            assert_eq!(
                (
                    U1.dim(&u1(q)),
                    U1.frobenius_schur_phase(&u1(q)),
                    U1.twist(&u1(q))
                ),
                (1.0, 1.0, 1.0)
            );
        }
        assert_eq!(
            U1.f_symbol(&u1(1), &u1(1), &u1(-1), &u1(1), &u1(2), &u1(0))
                .unwrap(),
            1.0
        );
        assert_eq!(
            U1.f_symbol(&u1(1), &u1(1), &u1(-1), &u1(1), &u1(2), &u1(1))
                .unwrap(),
            0.0
        );
        assert_eq!(U1.r_symbol(&u1(1), &u1(2), &u1(3)).unwrap(), 1.0);
        assert_eq!(U1.r_symbol(&u1(1), &u1(2), &u1(2)).unwrap(), 0.0);
    }
}
