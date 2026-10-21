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

pub(crate) mod sealed {
    pub trait Sealed {}
    /// Proof, constructible only inside this crate, that every object passed
    /// alongside it already passed the caller's validation.
    #[derive(Clone, Copy)]
    pub struct Validated(pub(crate) ());
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
/// Every object-taking method except [`Self::contains`] panics on a foreign
/// object, before testing capability or admissibility. `N = 0` is a value, never an error. Errors
/// mean genuine computation failure or absent capability. Category equality
/// and hashing use full mathematical identity, independently of the fingerprint.
///
/// # Panics
/// All object-taking methods except [`Self::contains`] panic for objects from a
/// different category value.
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
    /// Whether `a` is an object of this category value; never panics.
    ///
    /// This is the non-panicking pre-check for every other object-taking
    /// method: `true` exactly when those methods accept `a` without a
    /// foreign-object panic. It inspects only the label's identity and
    /// range, never computing a dimension or symbol, so it costs O(1) for
    /// a leaf category and one leaf check per leaf for a product.
    ///
    /// The default returns `true`, which asserts that no value of
    /// `Self::Object` can be foreign to any value of `Self`: it is sound only
    /// when the object type carries no category parameter. A category whose
    /// values differ by a parameter its objects record (a modulus, a group, a
    /// table) must override it.
    fn contains(&self, a: &Self::Object) -> bool {
        let _ = a;
        true
    }
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
    /// Checked [`Self::nsymbol`]: an unrepresentable multiplicity is an
    /// error instead of a panic. Foreign objects still panic; use
    /// [`Self::contains`] first to avoid that.
    ///
    /// The default returns `Ok(self.nsymbol(a, b, c))`, which is exact for a
    /// category whose `nsymbol` cannot overflow. A category that composes
    /// multiplicities, such as [`crate::product::Product`], overrides it.
    /// It costs the same as `nsymbol`.
    fn try_nsymbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<u32, Self::Error> {
        Ok(self.nsymbol(a, b, c))
    }
    /// Certificate that the associator is trivial in this category's gauge:
    /// every admissible `F^{abc}_d[e,f]` (all four vertices nonzero) is the
    /// `1 × 1 × 1 × 1` block holding exactly `1`. Inadmissible F stay zero.
    ///
    /// `true` lets a consumer replace every admissible F by `1` without
    /// evaluating it. `false` claims nothing: it is the conservative default
    /// for a category that does not override this method, and the value of
    /// every category with an object of dimension other than `1` (where
    /// `|F^{a ā a}_a[𝟙, 𝟙]| = 1 / d_a ≠ 1`).
    ///
    /// A gauge property, not a gauge-invariant one: a gauge-equivalent
    /// category may differ. Costs O(1) per leaf category (a product takes the
    /// conjunction of its factors) and evaluates no symbol or dimension.
    fn has_trivial_associator_gauge(&self) -> bool {
        false
    }
    /// Certificate that the unit is canonical in this category's gauge:
    ///
    /// - the unit `𝟙` is self-dual;
    /// - `𝟙 ⊗ a` and `a ⊗ 𝟙` are `a` alone, with multiplicity one;
    /// - the unitors are the identity on that unique vertex (this crate
    ///   exposes no separate unitor data, so this is the convention the
    ///   remaining condition is stated in);
    /// - every admissible `F^{abc}_d[e,f]` with `a`, `b` or `c` equal to `𝟙`
    ///   is the identity on the multiplicity spaces: the two vertices through
    ///   `𝟙` have multiplicity one, the two others `n`, and the `n × n`
    ///   block between them is the identity matrix.
    ///
    /// F with the coupled object `d = 𝟙`, or with `𝟙` only as an inner line
    /// `e` or `f` (such as `F^{a ā a}_a[𝟙, 𝟙]`, which carries dimension and
    /// Frobenius–Schur data), is not constrained.
    ///
    /// The values are those of the documented gauge; a family that computes
    /// F in floating point returns them up to its documented accuracy.
    /// `false` claims nothing and is the conservative default for a category
    /// that does not override this method. Costs O(1) per leaf category (a
    /// product takes the conjunction of its factors) and evaluates no symbol
    /// or dimension.
    fn has_canonical_unit(&self) -> bool {
        false
    }
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

    // Crate-internal `__fusion_symbols_validated_*` entry points, not part of
    // the public API. A product calls them after validating every leaf object
    // (and, for F and R, establishing nonzero representable shapes); a nested
    // product overrides them to skip re-validation and shape recomputation,
    // which would otherwise repeat at every nesting level. The token is
    // unnameable outside the crate, so they can be neither called nor
    // overridden there; the prefix keeps them from colliding with downstream
    // extension-trait methods.
    #[doc(hidden)]
    fn __fusion_symbols_validated_dual(
        &self,
        a: &Self::Object,
        _: sealed::Validated,
    ) -> Self::Object {
        self.dual(a)
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_frobenius_schur_phase(
        &self,
        a: &Self::Object,
        _: sealed::Validated,
    ) -> Self::Scalar {
        self.frobenius_schur_phase(a)
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_frobenius_schur_indicator(
        &self,
        a: &Self::Object,
        _: sealed::Validated,
    ) -> Self::Scalar {
        self.frobenius_schur_indicator(a)
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_twist(
        &self,
        a: &Self::Object,
        _: sealed::Validated,
    ) -> Self::Scalar {
        self.twist(a)
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_fusion<'a>(
        &'a self,
        a: &Self::Object,
        b: &Self::Object,
        _: sealed::Validated,
    ) -> Result<Self::Fusion<'a>, Self::Error> {
        self.fusion(a, b)
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_nsymbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        _: sealed::Validated,
    ) -> u32 {
        self.nsymbol(a, b, c)
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_try_nsymbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        _: sealed::Validated,
    ) -> Result<u32, Self::Error> {
        self.try_nsymbol(a, b, c)
    }
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn __fusion_symbols_validated_f_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        d: &Self::Object,
        e: &Self::Object,
        f: &Self::Object,
        _: sealed::Validated,
    ) -> Result<Self::FSymbol, Self::Error> {
        // A leaf block is combined by its dimensions, so a block disagreeing
        // with N would silently misplace entries; reject it here, once per leaf.
        let block = self.f_symbol(a, b, c, d, e, f)?;
        let n = [
            self.nsymbol(a, b, e),
            self.nsymbol(e, c, d),
            self.nsymbol(b, c, f),
            self.nsymbol(a, f, d),
        ];
        assert!(
            block.dims() == n.map(|n| n as usize),
            "F block shape disagrees with its N symbols"
        );
        Ok(block)
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_r_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        _: sealed::Validated,
    ) -> Result<Self::RSymbol, Self::Error> {
        let block = self.r_symbol(a, b, c)?;
        let n = [self.nsymbol(a, b, c), self.nsymbol(b, a, c)];
        assert!(
            block.dims() == n.map(|n| n as usize),
            "R block shape disagrees with its N symbols"
        );
        Ok(block)
    }
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
