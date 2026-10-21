//! Common mathematical contract for validated fusion-category objects.

use num_complex::Complex64;
use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, Mul, Neg, Sub},
};

/// Fusion multiplicities: one outcome, multiplicity-free, or general.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FusionStyle {
    /// Exactly one outcome, of multiplicity one.
    Unique,
    /// Outcomes have multiplicity at most one.
    Simple,
    /// Outcomes may have higher multiplicity.
    Generic,
}

/// Availability and symmetry of braiding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BraidingStyle {
    /// Braiding is absent: R returns an error and twist panics.
    None,
    /// Symmetric braiding with trivial twists.
    Bosonic,
    /// Symmetric braiding with possibly negative twists.
    Fermionic,
    /// General braiding and twists.
    Anyonic,
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex64 {}
}

/// Closed scalar set (`f64`, `Complex64`) with arithmetic preserving the type.
pub trait Scalar:
    sealed::Sealed
    + Copy
    + Debug
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + Send
    + Sync
    + 'static
{
    /// Additive identity, including inadmissible scalar symbols.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;
    /// Complex conjugation (identity for real scalars).
    fn conj(self) -> Self;
    /// Absolute value, used to compare residuals.
    fn abs(self) -> f64;
    /// Embed a real number.
    fn from_f64(value: f64) -> Self;
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn conj(self) -> Self {
        self
    }
    fn abs(self) -> f64 {
        self.abs()
    }
    fn from_f64(value: f64) -> Self {
        value
    }
}
impl Scalar for Complex64 {
    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
    fn one() -> Self {
        Self::new(1.0, 0.0)
    }
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    fn abs(self) -> f64 {
        self.norm()
    }
    fn from_f64(value: f64) -> Self {
        Self::new(value, 0.0)
    }
}

/// Lossless common scalar for a pair of category scalars.
pub trait Promote<R: Scalar>: Scalar {
    /// Real only when both inputs are real.
    type Output: Scalar;
    /// Embed the left scalar.
    fn left(self) -> <Self as Promote<R>>::Output;
    /// Embed the right scalar.
    fn right(r: R) -> <Self as Promote<R>>::Output;
}
impl Promote<f64> for f64 {
    type Output = f64;
    fn left(self) -> f64 {
        self
    }
    fn right(r: f64) -> f64 {
        r
    }
}
impl Promote<Complex64> for f64 {
    type Output = Complex64;
    fn left(self) -> Complex64 {
        self.into()
    }
    fn right(r: Complex64) -> Complex64 {
        r
    }
}
impl Promote<f64> for Complex64 {
    type Output = Complex64;
    fn left(self) -> Complex64 {
        self
    }
    fn right(r: f64) -> Complex64 {
        r.into()
    }
}
impl Promote<Complex64> for Complex64 {
    type Output = Complex64;
    fn left(self) -> Complex64 {
        self
    }
    fn right(r: Complex64) -> Complex64 {
        r
    }
}

/// Row-major symbol storage. Logical shape comes from `FusionCategory::nsymbol`,
/// never from `dims`: bare scalars have all-one storage axes even for zero symbols.
pub trait SymbolBlock<const RANK: usize> {
    /// Stored scalar type.
    type Scalar: Scalar;
    /// Storage shape, not the logical vertex multiplicities.
    fn dims(&self) -> [usize; RANK];
    /// Read an entry.
    ///
    /// # Panics
    /// Panics if any index is outside the storage shape.
    fn get(&self, index: [usize; RANK]) -> Self::Scalar;
    /// Flat row-major storage (F axes μ, ν, κ, λ; R axes μ, ν).
    fn as_slice(&self) -> &[Self::Scalar];
}

impl<const RANK: usize> SymbolBlock<RANK> for f64 {
    type Scalar = Self;
    fn dims(&self) -> [usize; RANK] {
        [1; RANK]
    }
    fn get(&self, index: [usize; RANK]) -> Self {
        assert!(index.iter().all(|&i| i == 0), "symbol index out of range");
        *self
    }
    fn as_slice(&self) -> &[Self] {
        std::slice::from_ref(self)
    }
}
impl<const RANK: usize> SymbolBlock<RANK> for Complex64 {
    type Scalar = Self;
    fn dims(&self) -> [usize; RANK] {
        [1; RANK]
    }
    fn get(&self, index: [usize; RANK]) -> Self {
        assert!(index.iter().all(|&i| i == 0), "symbol index out of range");
        *self
    }
    fn as_slice(&self) -> &[Self] {
        std::slice::from_ref(self)
    }
}

/// One mathematical category, with objects validated at construction.
///
/// Every object-taking method panics on a foreign object, before testing
/// capability or admissibility. `N = 0` is a value, never an error. Errors
/// mean genuine computation failure or absent capability. Category equality
/// and hashing use full mathematical identity, independently of the fingerprint.
///
/// # Panics
/// All object-taking methods panic for objects from a different category value.
/// Capability checks and zero-channel handling must not bypass this validation.
pub trait FusionCategory: Clone + Eq + Hash + Debug {
    /// Validated simple-object label, in deterministic order.
    type Object: Clone + Eq + Ord + Hash + Debug;
    /// Real or complex symbol scalar.
    type Scalar: Scalar;
    /// F storage; logical shape is the four vertex multiplicities from N.
    type FSymbol: SymbolBlock<4, Scalar = Self::Scalar>;
    /// R storage; logical shape is `[N(a,b,c), N(b,a,c)]`.
    type RSymbol: SymbolBlock<2, Scalar = Self::Scalar>;
    /// Distinct fusion outcomes, in deterministic order.
    type Fusion<'a>: Iterator<Item = Self::Object>
    where
        Self: 'a;
    /// Genuine failure or absent capability; `Infallible` when neither occurs.
    type Error: std::error::Error;

    /// Multiplicity class of fusion.
    fn fusion_style(&self) -> FusionStyle;
    /// Braiding capability and symmetry class.
    fn braiding_style(&self) -> BraidingStyle;
    /// The single simple unit.
    fn unit(&self) -> Self::Object;
    /// Dual of a validated object; foreign objects panic.
    fn dual(&self, a: &Self::Object) -> Self::Object;
    /// Quantum dimension; foreign objects panic.
    fn dim(&self, a: &Self::Object) -> f64;
    /// Phase of `F(a, dual(a), a, a, unit, unit)`; foreign objects panic.
    fn frobenius_schur_phase(&self, a: &Self::Object) -> Self::Scalar;
    /// Phase if self-dual, otherwise zero (TensorKitSectors 0.3.9,
    /// `src/sectors.jl::frobenius_schur_indicator`); foreign objects panic.
    fn frobenius_schur_indicator(&self, a: &Self::Object) -> Self::Scalar;
    /// Ribbon twist; panics on foreign objects or `BraidingStyle::None`.
    fn twist(&self, a: &Self::Object) -> Self::Scalar;
    /// Deterministic outcomes; checked arithmetic overflow is an error.
    /// Foreign objects panic before computation.
    fn fusion<'a>(
        &'a self,
        a: &Self::Object,
        b: &Self::Object,
    ) -> Result<Self::Fusion<'a>, Self::Error>;
    /// Exact vertex multiplicity, including zero; foreign objects panic.
    fn nsymbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> u32;
    /// `F^{abc}_d[e,f]`; inadmissible vertices return zero storage. Logical
    /// axes are `[N(a,b,e), N(e,c,d), N(b,c,f), N(a,f,d)]`.
    /// Foreign objects panic; errors indicate genuine failure.
    #[allow(clippy::too_many_arguments)]
    fn f_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        d: &Self::Object,
        e: &Self::Object,
        f: &Self::Object,
    ) -> Result<Self::FSymbol, Self::Error>;
    /// `R^{ab}_c`; zero for `N = 0`, error if braiding is absent.
    /// Foreign objects panic before capability checks.
    fn r_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<Self::RSymbol, Self::Error>;
    /// Opaque convention bytes; compare by equality, never parse.
    fn authority_fingerprint(&self) -> &[u8];
}

/// Group categories that additionally provide Clebsch–Gordan data.
pub trait GroupRepresentations: FusionCategory {
    /// Family-specific CGC storage.
    type Cgc;
    /// CGC for validated objects; foreign objects panic, absent channels
    /// return zeros, and errors indicate genuine failure or absent capability.
    fn cgc(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<Self::Cgc, Self::Error>;
}

mod residual;
pub use residual::{f_unitarity_residual, hexagon_residual, pentagon_residual};

mod blocks;
pub(crate) use blocks::Storage;
pub use blocks::{FBlock, RBlock};
