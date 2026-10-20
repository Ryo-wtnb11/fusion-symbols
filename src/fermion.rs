//! Fermion parity with unique fusion and the odd–odd braiding sign.

use crate::category::{BraidingStyle, FusionCategory, FusionStyle};
use std::convert::Infallible;

/// Fermion-parity category; all validated parity labels belong to it.
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

const FINGERPRINT: &[u8] = b"fusion-symbols:abelian-fz2:n=tks-fermions:f=tks-fermions:r=tks-fermions-odd-odd-minus:dim=one:fs=tks-fsymbol-sign:twist=tks-fermions:scalar=f64:epoch=1";

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
    fn dual(&self, a: &FermionParityIrrep) -> FermionParityIrrep {
        *a
    }
    fn dim(&self, _a: &FermionParityIrrep) -> f64 {
        1.0
    }
    fn frobenius_schur_phase(&self, _a: &FermionParityIrrep) -> f64 {
        1.0
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
    ) -> u32 {
        u32::from((a.odd ^ b.odd) == c.odd)
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
        Ok(f64::from(
            self.nsymbol(a, b, e)
                * self.nsymbol(e, c, d)
                * self.nsymbol(b, c, f)
                * self.nsymbol(a, f, d),
        ))
    }
    fn r_symbol(
        &self,
        a: &FermionParityIrrep,
        b: &FermionParityIrrep,
        c: &FermionParityIrrep,
    ) -> Result<f64, Self::Error> {
        let n = f64::from(self.nsymbol(a, b, c));
        Ok(if a.odd && b.odd && n != 0.0 { -n } else { n })
    }
    fn authority_fingerprint(&self) -> &[u8] {
        FINGERPRINT
    }
}
