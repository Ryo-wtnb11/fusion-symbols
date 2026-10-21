//! Closed-form integer-charge U(1).
//!
//! Labels exclude `i32::MIN` so every dual is representable. Symbols are
//! exact `f64` zeros and ones; inadmissible vertices give zero, not errors.

use crate::category::{BraidingStyle, FusionCategory, FusionStyle};

/// An unrepresentable U(1) charge, requested directly or produced by fusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum U1Error {
    /// A U(1) charge outside `i32::MIN + 1 ..= i32::MAX`, either requested
    /// directly or produced by [`FusionCategory::fusion`].
    U1ChargeOutOfRange {
        /// The exact charge.
        charge: i64,
    },
}

impl std::fmt::Display for U1Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            U1Error::U1ChargeOutOfRange { charge } => write!(
                f,
                "U(1) charge {charge} is outside i32::MIN + 1 ..= i32::MAX"
            ),
        }
    }
}

impl std::error::Error for U1Error {}

/// An irrep of `U(1)`: an integer charge in `i32::MIN + 1 ..= i32::MAX`.
///
/// `i32::MIN` is excluded so that every label has a representable dual.
/// TensorKitSectors' `U1Irrep` stores a `HalfInt`; half-integer charges are
/// not representable here. `Ord` is numeric charge order, not
/// TensorKitSectors' `isless` (which orders by `|q|` first).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U1Irrep {
    charge: i32,
}

impl U1Irrep {
    /// The unit, charge `0`.
    pub const UNIT: Self = Self { charge: 0 };

    /// The irrep of the given charge.
    ///
    /// # Errors
    ///
    /// [`U1Error::U1ChargeOutOfRange`] for `i32::MIN`.
    pub fn new(charge: i32) -> Result<Self, U1Error> {
        Self::from_exact(i64::from(charge))
    }

    fn from_exact(charge: i64) -> Result<Self, U1Error> {
        match i32::try_from(charge) {
            Ok(charge) if charge != i32::MIN => Ok(Self { charge }),
            _ => Err(U1Error::U1ChargeOutOfRange { charge }),
        }
    }

    /// The charge.
    pub const fn charge(self) -> i32 {
        self.charge
    }
}

pub(crate) const U1_FINGERPRINT: &[u8] = b"fusion-symbols:abelian-u1:labels=i32-integer-no-min:n=exact:f=tks-irreps:r=tks-irreps:dim=one:fs=tks-irreps-one:twist=one:scalar=f64:epoch=1";

/// Integer-charge U(1), with validated labels and checked fusion arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct U1;

impl FusionCategory for U1 {
    type Object = U1Irrep;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = std::iter::Once<U1Irrep>;
    type Error = U1Error;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> U1Irrep {
        U1Irrep::UNIT
    }
    fn contains(&self, a: &U1Irrep) -> bool {
        a.charge != i32::MIN
    }
    fn dual(&self, a: &U1Irrep) -> U1Irrep {
        U1Irrep { charge: -a.charge }
    }
    fn dim(&self, _a: &U1Irrep) -> f64 {
        1.0
    }
    fn frobenius_schur_phase(&self, _a: &U1Irrep) -> f64 {
        1.0
    }
    fn frobenius_schur_indicator(&self, a: &U1Irrep) -> f64 {
        f64::from(u8::from(*a == self.dual(a)))
    }
    fn twist(&self, _a: &U1Irrep) -> f64 {
        1.0
    }
    fn fusion<'a>(&'a self, a: &U1Irrep, b: &U1Irrep) -> Result<Self::Fusion<'a>, Self::Error> {
        Ok(std::iter::once(U1Irrep::from_exact(
            i64::from(a.charge) + i64::from(b.charge),
        )?))
    }
    fn nsymbol(&self, a: &U1Irrep, b: &U1Irrep, c: &U1Irrep) -> u32 {
        u32::from(i64::from(a.charge) + i64::from(b.charge) == i64::from(c.charge))
    }
    fn f_symbol(
        &self,
        a: &U1Irrep,
        b: &U1Irrep,
        c: &U1Irrep,
        d: &U1Irrep,
        e: &U1Irrep,
        f: &U1Irrep,
    ) -> Result<f64, Self::Error> {
        Ok(f64::from(
            self.nsymbol(a, b, e)
                * self.nsymbol(e, c, d)
                * self.nsymbol(b, c, f)
                * self.nsymbol(a, f, d),
        ))
    }
    fn r_symbol(&self, a: &U1Irrep, b: &U1Irrep, c: &U1Irrep) -> Result<f64, Self::Error> {
        Ok(f64::from(self.nsymbol(a, b, c)))
    }
    // F is the product of the four vertex N, so exactly `1` when admissible;
    // the unit is the self-dual identity charge.
    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }
    fn has_canonical_unit(&self) -> bool {
        true
    }
    fn authority_fingerprint(&self) -> &[u8] {
        U1_FINGERPRINT
    }
}
