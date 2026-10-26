//! Closed-form integer-charge U(1).
//!
//! Labels exclude `i32::MIN` so every dual is representable. Symbols are
//! exact `f64` zeros and ones; inadmissible vertices give zero, not errors.

use crate::category::{
    BraidingStyle, EnumerationError, FusionCategory, FusionStyle, GroupRepresentations, RBlock,
};

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

    pub(super) fn from_exact(charge: i64) -> Result<Self, U1Error> {
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
///
/// Enumeration (frozen): charges `0, 1, −1, 2, −2, …`; charge `q > 0` has
/// index `2q − 1` and charge `q ≤ 0` index `−2q`. The count is infinite;
/// indices past the label range (`|q| > i32::MAX`) are
/// [`EnumerationError::Overflow`]. This is TensorKitSectors' `U1Irrep` order
/// restricted to its integer charges.
///
/// Every irrep is one-dimensional, so [`GroupRepresentations::cgc`] is the
/// single coefficient `⟨a; b | c⟩`: exactly `1.0` when `c = a + b` and `0.0`
/// otherwise, including when `a + b` leaves `i32` (as [`FusionCategory::nsymbol`]).
///
/// ```
/// use fusion_symbols::{category::GroupRepresentations, lie::{U1, U1Irrep}};
/// let q = |c| U1Irrep::new(c).unwrap();
/// assert_eq!(U1.cgc(&q(2), &q(-5), &q(-3)), Ok(1.0));
/// assert_eq!(U1.cgc(&q(2), &q(-5), &q(3)), Ok(0.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct U1;

impl U1 {
    /// The connected compact group classified by this category.
    /// U(1) always has a valid rank and character lattice.
    pub fn group_id(&self) -> super::GroupId {
        super::GroupId::u(1).expect("U(1) has a valid rank and character lattice")
    }
}

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
    // TensorKitSectors `irreps.jl` (AbelianIrrep): A and B are N, since every
    // admissible F and every dimension is one.
    fn a_symbol(&self, a: &U1Irrep, b: &U1Irrep, c: &U1Irrep) -> Result<RBlock<f64>, Self::Error> {
        Ok(if self.nsymbol(a, b, c) == 0 {
            RBlock::zeros(0)
        } else {
            RBlock::one(1.0)
        })
    }
    fn b_symbol(&self, a: &U1Irrep, b: &U1Irrep, c: &U1Irrep) -> Result<RBlock<f64>, Self::Error> {
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
        Ok(None)
    }
    fn object_at(&self, index: u64) -> Result<U1Irrep, EnumerationError> {
        let half = i64::try_from(index / 2)
            .ok()
            .filter(|&h| h <= i64::from(i32::MAX))
            .ok_or(EnumerationError::Overflow)?;
        let charge = if index % 2 == 1 { half + 1 } else { -half };
        U1Irrep::from_exact(charge).map_err(|_| EnumerationError::Overflow)
    }
    fn object_index(&self, a: &U1Irrep) -> Result<u64, EnumerationError> {
        assert!(self.contains(a), "foreign U(1) object");
        let q = i64::from(a.charge);
        // |q| ≤ i32::MAX, so both forms fit.
        Ok((if q > 0 { 2 * q - 1 } else { -2 * q }) as u64)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        U1_FINGERPRINT
    }
}

// The `1x1x1` CGC tensor with its unit multiplicity axis is one scalar; an
// absent channel is the zero `1x1x1x0` block, represented by `0.0`.
impl GroupRepresentations for U1 {
    type Cgc = f64;
    fn cgc(&self, a: &U1Irrep, b: &U1Irrep, c: &U1Irrep) -> Result<f64, U1Error> {
        Ok(f64::from(self.nsymbol(a, b, c)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(charge: i32) -> U1Irrep {
        U1Irrep::new(charge).unwrap()
    }

    #[test]
    fn cgc_is_one_on_charge_conservation_and_zero_otherwise() {
        for (a, b, c, expected) in [
            (0, 0, 0, 1.0),
            (3, -1, 2, 1.0),
            (-4, -7, -11, 1.0),
            (5, -5, 0, 1.0),
            (3, -1, 1, 0.0),
            (0, 0, 1, 0.0),
            (2, 2, -4, 0.0),
            // Extremes: sums at and beyond the i32 label range.
            (i32::MAX, 0, i32::MAX, 1.0),
            (i32::MAX - 1, 1, i32::MAX, 1.0),
            (i32::MIN + 1, i32::MAX, 0, 1.0),
            (i32::MIN + 1, -1, i32::MIN + 1, 0.0),
            (i32::MAX, 1, i32::MIN + 1, 0.0),
            (i32::MAX, i32::MAX, -2, 0.0),
            (i32::MIN + 1, i32::MIN + 1, 2, 0.0),
        ] {
            assert_eq!(U1.cgc(&q(a), &q(b), &q(c)), Ok(expected), "{a} {b} {c}");
        }
    }

    #[test]
    fn cgc_agrees_with_nsymbol() {
        let charges = [i32::MIN + 1, -3, -1, 0, 1, 2, 3, i32::MAX];
        for &a in &charges {
            for &b in &charges {
                for &c in &charges {
                    let (a, b, c) = (q(a), q(b), q(c));
                    assert_eq!(U1.cgc(&a, &b, &c), Ok(f64::from(U1.nsymbol(&a, &b, &c))));
                }
            }
        }
    }
}
