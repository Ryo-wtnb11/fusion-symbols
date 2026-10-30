//! Fermion parity with unique fusion and the odd–odd braiding sign.

use crate::category::{BraidingStyle, EnumerationError, FusionCategory, FusionStyle};
use std::convert::Infallible;

/// Fermion-parity category; all validated parity labels belong to it.
///
/// Enumeration (frozen): `EVEN, ODD`, indices `0, 1`.
///
/// # Examples
///
/// ```
/// use fusion_symbols::{category::FusionCategory, fermion::{FermionParity, FermionParityIrrep}};
///
/// let (even, odd) = (FermionParityIrrep::EVEN, FermionParityIrrep::ODD);
/// assert_eq!(FermionParity.fusion(&odd, &odd)?.next(), Some(even));
/// assert_eq!(FermionParity.r_symbol(&odd, &odd, &even)?, -1.0); // exchange sign
/// # Ok::<(), std::convert::Infallible>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FermionParity;

/// Validated parity object, ordered with the unit first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FermionParityIrrep {
    odd: bool,
}

impl FermionParityIrrep {
    /// Even parity, the unit.
    pub const EVEN: Self = Self { odd: false };
    /// Odd parity.
    pub const ODD: Self = Self { odd: true };
    /// Construct either validated parity.
    pub const fn new(odd: bool) -> Self {
        Self { odd }
    }
    /// Whether the object has odd parity.
    pub const fn is_odd(self) -> bool {
        self.odd
    }
}

// N(a, b, c): parity conservation.
fn n(a: &FermionParityIrrep, b: &FermionParityIrrep, c: &FermionParityIrrep) -> u32 {
    u32::from((a.odd ^ b.odd) == c.odd)
}

impl FusionCategory for FermionParity {
    type Object = FermionParityIrrep;
    type Scalar = f64;
    type FSymbol = f64;
    type RSymbol = f64;
    type Fusion<'a> = std::iter::Once<FermionParityIrrep>;
    type Error = Infallible;

    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Fermionic
    }
    fn unit(&self) -> FermionParityIrrep {
        FermionParityIrrep::EVEN
    }
    // Both parity labels belong to the single fermion-parity category.
    fn contains(&self, _a: &FermionParityIrrep) -> bool {
        true
    }
    fn dual(&self, a: &FermionParityIrrep) -> FermionParityIrrep {
        *a
    }
    fn dim(&self, _a: &FermionParityIrrep) -> f64 {
        1.0
    }
    fn frobenius_schur_phase(&self, _a: &FermionParityIrrep) -> Result<f64, Self::Error> {
        Ok(1.0)
    }
    fn frobenius_schur_indicator(&self, a: &FermionParityIrrep) -> f64 {
        f64::from(u8::from(*a == self.dual(a)))
    }
    fn twist(&self, a: &FermionParityIrrep) -> f64 {
        if a.odd {
            -1.0
        } else {
            1.0
        }
    }
    fn fusion<'a>(
        &'a self,
        a: &FermionParityIrrep,
        b: &FermionParityIrrep,
    ) -> Result<Self::Fusion<'a>, Self::Error> {
        Ok(std::iter::once(FermionParityIrrep::new(a.odd ^ b.odd)))
    }
    fn nsymbol(
        &self,
        a: &FermionParityIrrep,
        b: &FermionParityIrrep,
        c: &FermionParityIrrep,
    ) -> Result<u32, Self::Error> {
        Ok(n(a, b, c))
    }
    fn f_symbol(
        &self,
        a: &FermionParityIrrep,
        b: &FermionParityIrrep,
        c: &FermionParityIrrep,
        d: &FermionParityIrrep,
        e: &FermionParityIrrep,
        f: &FermionParityIrrep,
    ) -> Result<f64, Self::Error> {
        Ok(f64::from(n(a, b, e) * n(e, c, d) * n(b, c, f) * n(a, f, d)))
    }
    fn r_symbol(
        &self,
        a: &FermionParityIrrep,
        b: &FermionParityIrrep,
        c: &FermionParityIrrep,
    ) -> Result<f64, Self::Error> {
        let n = f64::from(n(a, b, c));
        Ok(if a.odd && b.odd && n != 0.0 { -n } else { n })
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
        Ok(Some(2))
    }
    fn object_at(&self, index: u64) -> Result<FermionParityIrrep, EnumerationError> {
        match index {
            0 => Ok(FermionParityIrrep::EVEN),
            1 => Ok(FermionParityIrrep::ODD),
            _ => Err(EnumerationError::OutOfRange),
        }
    }
    fn object_index(&self, a: &FermionParityIrrep) -> Result<u64, EnumerationError> {
        Ok(u64::from(a.odd))
    }
}
