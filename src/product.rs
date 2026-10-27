//! Ordered Deligne products of tuple or runtime-length homogeneous factors.
//!
//! Objects are tuples or vectors; explicit nesting is preserved. Fusion outcomes
//! vary the first factor fastest. Each F/R/A/B multiplicity axis varies the last factor
//! fastest, with row-major block storage. Scalar promotion and multiplication
//! fold left to right. Reassociation agrees up to floating-point tolerance where
//! intermediate queries are representable. No cross-factor fermionic sign or
//! coefficient cache is added. Products of group representations provide lazy
//! Clebsch–Gordan blocks, with the last factor fastest on every axis.
//!
//! Nonzero F/R/A/B queries check each factor's storage axes against checked N
//! symbols and its exact row-major storage length before composition. Invalid
//! blocks return [`ProductError::MalformedSymbolData`]; nested errors retain
//! their factor path. Zero-channel queries skip factor symbols.
//!
//! Enumeration (frozen) is the bounded Manhattan order of the immediate
//! factors' enumeration indices `x`: by increasing `Σ x_j`, then ascending
//! lexicographically, skipping tuples past a finite factor's count. This is
//! TensorKitSectors' `to_manhattan_index − 1` over the immediate factors, so an
//! explicitly nested product enumerates differently from the flat one. The
//! count is the product of the factor counts, infinite if any factor is (even
//! when another factor's count overflows `u64`). A factor whose count is above
//! `u64::MAX` (reported as [`EnumerationError::Overflow`]) still bounds no
//! addressable tuple, so every product index up to `u64::MAX` stays available.
//! A product is enumerable only when every factor is
//! ([`EnumerationError::NotEnumerable`] takes precedence over other factor
//! errors). Derived dimensions ([`FusionCategory::sqrt_dim`] and its
//! inverses) multiply the factor values left to right.
//!
//! Factor fusion iterators must be cloneable. The higher-ranked GAT bound on
//! these implementations currently restricts factors to owned category types;
//! arbitrary borrowed providers are not promised.
use crate::category::{
    rank_checked, sealed::ExactCount, sealed::Ranked, sealed::SymbolWithAxes, sealed::Validated,
    unrank, BraidingStyle, CgcBlock, Count, EnumerationError, FBlock, FusionCategory, FusionStyle,
    GroupRepresentations, Promote, RBlock, Scalar, Storage, SymbolBlock,
};
use num_bigint::BigInt;
use std::{
    convert::Infallible,
    fmt,
    hash::{Hash, Hasher},
    iter::Fuse,
    sync::{Arc, OnceLock},
};

/// A product retaining the ordered values and nesting of its factors.
///
/// Tuples support 2..=12 factors, including heterogeneous categories and scalar
/// promotion. `Product<Vec<C>>` supports any number of homogeneous factors;
/// its objects are `Vec<C::Object>` and its scalar is `C::Scalar`. The empty
/// vector is the trivial category, with unit `[]` and scalar symbols equal to 1.
#[derive(Clone)]
pub struct Product<C> {
    factors: C,
    fingerprint: OnceLock<Arc<[u8]>>,
}
impl<C> Product<C> {
    /// Construct a tuple product (2..=12 factors) or a homogeneous vector product (any length).
    pub fn new(factors: C) -> Self
    where
        Self: FusionCategory,
    {
        Self {
            factors,
            fingerprint: OnceLock::new(),
        }
    }
    /// The ordered immediate factors.
    pub fn factors(&self) -> &C {
        &self.factors
    }
    /// Recover the ordered immediate factors.
    pub fn into_factors(self) -> C {
        self.factors
    }
}
impl<C: PartialEq> PartialEq for Product<C> {
    fn eq(&self, other: &Self) -> bool {
        self.factors == other.factors
    }
}
impl<C: Eq> Eq for Product<C> {}
impl<C: Hash> Hash for Product<C> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.factors.hash(state);
    }
}
impl<C: fmt::Debug> fmt::Debug for Product<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Product").field(&self.factors).finish()
    }
}

/// Construct a binary product, preserving any explicit nesting.
pub fn product<A, B>(a: A, b: B) -> Product<(A, B)>
where
    Product<(A, B)>: FusionCategory,
{
    Product::new((a, b))
}

/// Compose 3..=12 factors, evaluating each expression once. A trailing comma is allowed.
///
/// ```
/// use fusion_symbols::{lie::U1, fermion::FermionParity, product};
/// let cat = product![U1, FermionParity, U1,];
/// assert_eq!(cat.into_factors(), (U1, FermionParity, U1));
/// ```
/// ```compile_fail
/// use fusion_symbols::{lie::U1, product};
/// let _ = product![U1];
/// ```
/// ```compile_fail
/// use fusion_symbols::{lie::U1, product};
/// let _ = product![U1,U1,U1,U1,U1,U1,U1,U1,U1,U1,U1,U1,U1];
/// ```
#[macro_export]
macro_rules! product {
    ($a0:expr, $a1:expr, $a2:expr $(,)?) => {
        $crate::product::Product::new(($a0, $a1, $a2))
    };
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr $(,)?) => {
        $crate::product::Product::new(($a0, $a1, $a2, $a3))
    };
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr $(,)?) => {
        $crate::product::Product::new(($a0, $a1, $a2, $a3, $a4))
    };
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr $(,)?) => {
        $crate::product::Product::new(($a0, $a1, $a2, $a3, $a4, $a5))
    };
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr $(,)?) => {
        $crate::product::Product::new(($a0, $a1, $a2, $a3, $a4, $a5, $a6))
    };
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr $(,)?) => {
        $crate::product::Product::new(($a0, $a1, $a2, $a3, $a4, $a5, $a6, $a7))
    };
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr $(,)?) => {
        $crate::product::Product::new(($a0, $a1, $a2, $a3, $a4, $a5, $a6, $a7, $a8))
    };
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr, $a9:expr $(,)?) => {
        $crate::product::Product::new(($a0, $a1, $a2, $a3, $a4, $a5, $a6, $a7, $a8, $a9))
    };
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr, $a9:expr, $a10:expr $(,)?) => {
        $crate::product::Product::new(($a0, $a1, $a2, $a3, $a4, $a5, $a6, $a7, $a8, $a9, $a10))
    };
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr, $a9:expr, $a10:expr, $a11:expr $(,)?) => {
        $crate::product::Product::new((
            $a0, $a1, $a2, $a3, $a4, $a5, $a6, $a7, $a8, $a9, $a10, $a11,
        ))
    };
    ($a:expr $(,)?) => {
        compile_error!("product! requires 3..=12 factors; use product(a, b) for two")
    };
    ($a0:expr, $a1:expr, $a2:expr, $a3:expr, $a4:expr, $a5:expr, $a6:expr, $a7:expr, $a8:expr, $a9:expr, $a10:expr, $a11:expr, $a12:expr $(, $rest:expr)* $(,)?) => {
        compile_error!("product! supports at most 12 immediate factors")
    };
    ($($other:tt)*) => {
        compile_error!("product! requires 3..=12 factors")
    };
}

/// Factor failures retain their concrete error types and immediate positions.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProductError<
    E0,
    E1 = Infallible,
    E2 = Infallible,
    E3 = Infallible,
    E4 = Infallible,
    E5 = Infallible,
    E6 = Infallible,
    E7 = Infallible,
    E8 = Infallible,
    E9 = Infallible,
    E10 = Infallible,
    E11 = Infallible,
> {
    /// Failure from a runtime-length product's immediate factor.
    Factor {
        /// Zero-based immediate factor index.
        index: usize,
        /// The factor's concrete error.
        source: E0,
    },
    /// Failure from immediate factor 0.
    Factor0(E0),
    /// Failure from immediate factor 1.
    Factor1(E1),
    /// Failure from immediate factor 2.
    Factor2(E2),
    /// Failure from immediate factor 3.
    Factor3(E3),
    /// Failure from immediate factor 4.
    Factor4(E4),
    /// Failure from immediate factor 5.
    Factor5(E5),
    /// Failure from immediate factor 6.
    Factor6(E6),
    /// Failure from immediate factor 7.
    Factor7(E7),
    /// Failure from immediate factor 8.
    Factor8(E8),
    /// Failure from immediate factor 9.
    Factor9(E9),
    /// Failure from immediate factor 10.
    Factor10(E10),
    /// Failure from immediate factor 11.
    Factor11(E11),
    /// A factor returned a block inconsistent with its checked multiplicities.
    /// Nested products retain their path through factor errors.
    MalformedSymbolData {
        /// Zero-based immediate factor index.
        factor: usize,
        /// Expected storage axes from checked N symbols (four for F, two for R/A/B).
        expected: Vec<usize>,
        /// Storage axes reported by the returned block.
        actual: Vec<usize>,
        /// Actual number of scalars in the block's row-major slice.
        storage_len: usize,
    },
    /// Multiplicity, shape, element count or byte layout is not representable.
    Overflow,
    /// The indicated immediate factor has no braiding.
    NoBraiding {
        /// Zero-based immediate factor index.
        factor: usize,
    },
}
impl<
        E0: std::error::Error,
        E1: std::error::Error,
        E2: std::error::Error,
        E3: std::error::Error,
        E4: std::error::Error,
        E5: std::error::Error,
        E6: std::error::Error,
        E7: std::error::Error,
        E8: std::error::Error,
        E9: std::error::Error,
        E10: std::error::Error,
        E11: std::error::Error,
    > fmt::Display for ProductError<E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Factor { index, source } => write!(f, "product factor {index}: {source}"),
            Self::Factor0(e) => write!(f, "product factor 0: {e}"),
            Self::Factor1(e) => write!(f, "product factor 1: {e}"),
            Self::Factor2(e) => write!(f, "product factor 2: {e}"),
            Self::Factor3(e) => write!(f, "product factor 3: {e}"),
            Self::Factor4(e) => write!(f, "product factor 4: {e}"),
            Self::Factor5(e) => write!(f, "product factor 5: {e}"),
            Self::Factor6(e) => write!(f, "product factor 6: {e}"),
            Self::Factor7(e) => write!(f, "product factor 7: {e}"),
            Self::Factor8(e) => write!(f, "product factor 8: {e}"),
            Self::Factor9(e) => write!(f, "product factor 9: {e}"),
            Self::Factor10(e) => write!(f, "product factor 10: {e}"),
            Self::Factor11(e) => write!(f, "product factor 11: {e}"),
            Self::MalformedSymbolData { factor, expected, actual, storage_len } => write!(f,
                "product factor {factor}: malformed symbol data (expected axes {expected:?}, actual axes {actual:?}, storage length {storage_len})"),
            Self::Overflow => f.write_str("product shape or multiplicity overflow"),
            Self::NoBraiding { factor } => write!(f, "product factor {factor} has no braiding"),
        }
    }
}
impl<
        E0: std::error::Error,
        E1: std::error::Error,
        E2: std::error::Error,
        E3: std::error::Error,
        E4: std::error::Error,
        E5: std::error::Error,
        E6: std::error::Error,
        E7: std::error::Error,
        E8: std::error::Error,
        E9: std::error::Error,
        E10: std::error::Error,
        E11: std::error::Error,
    > std::error::Error for ProductError<E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11>
{
}

impl<E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11>
    ProductError<E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10, E11>
{
    fn checked_block<B: SymbolBlock<R>, const R: usize>(
        factor: usize,
        (block, n): SymbolWithAxes<B, R>,
    ) -> Result<B, Self> {
        let expected = shape::<B::Scalar, R>(n).ok_or(Self::Overflow)?;
        let count = if expected.contains(&0) {
            0
        } else {
            expected.iter().product()
        };
        let actual = block.dims();
        let storage_len = block.as_slice().len();
        if actual != expected || storage_len != count {
            return Err(Self::MalformedSymbolData {
                factor,
                expected: expected.to_vec(),
                actual: actual.to_vec(),
                storage_len,
            });
        }
        Ok(block)
    }
}

/// Cartesian fusion iterator; resets clone initialized iterators, never query factors again.
#[derive(Clone, Debug)]
pub struct ProductFusion<I, O> {
    initial: I,
    active: I,
    current: Option<O>,
}

// Read all multiplicities before multiplying: a zero factor fixes the product
// at zero, so it beats both a nonzero prefix overflow and any factor failure.
// Otherwise the first factor failure wins, then overflow.
fn checked_multiplicity<E>(
    values: impl IntoIterator<Item = Result<u32, E>>,
    overflow: E,
) -> Result<u32, E> {
    let mut product = Some(1u32);
    let mut error = None;
    for n in values {
        match n {
            Ok(0) => product = Some(0),
            Ok(n) => product = product.and_then(|p| p.checked_mul(n)),
            Err(e) => {
                error.get_or_insert(e);
            }
        }
    }
    match (error, product) {
        (_, Some(0)) => Ok(0),
        (Some(e), _) => Err(e),
        (None, Some(n)) => Ok(n),
        (None, None) => Err(overflow),
    }
}
fn shape<S, const R: usize>(n: [u32; R]) -> Option<[usize; R]> {
    let mut dims = [0; R];
    for (dim, n) in dims.iter_mut().zip(n) {
        *dim = usize::try_from(n).ok()?;
    }
    let count = if dims.contains(&0) {
        0
    } else {
        dims.iter().try_fold(1usize, |p, &n| p.checked_mul(n))?
    };
    std::alloc::Layout::array::<S>(count).ok()?;
    Some(dims)
}

// Every factor result, or one error: a non-enumerable factor makes the whole
// product non-enumerable, whatever the other factors report.
fn all_ok<T>(
    results: impl IntoIterator<Item = Result<T, EnumerationError>>,
) -> Result<Vec<T>, EnumerationError> {
    let results: Vec<_> = results.into_iter().collect();
    if results
        .iter()
        .any(|r| matches!(r, Err(EnumerationError::NotEnumerable)))
    {
        return Err(EnumerationError::NotEnumerable);
    }
    results.into_iter().collect()
}

fn storage<B: SymbolBlock<R>, const R: usize>(block: B) -> ([usize; R], Storage<B::Scalar>) {
    let dims = block.dims();
    (
        dims,
        if dims == [1; R] {
            Storage::One(block.get([0; R]))
        } else {
            Storage::Shared(block.as_slice().into())
        },
    )
}

// Each output axis decodes i = left_digit * right_size + right_digit.
fn kron<A: Scalar, B: SymbolBlock<R>, S: Scalar, const R: usize>(
    left_dims: [usize; R],
    left: Storage<A>,
    right: B,
    multiply: impl Fn(A, B::Scalar) -> S,
) -> ([usize; R], Storage<S>) {
    let right_dims = right.dims();
    let dims = std::array::from_fn(|i| left_dims[i] * right_dims[i]);
    let entry = |mut flat: usize| {
        let mut l = 0;
        let mut stride = 1;
        let mut r = [0; R];
        for axis in (0..R).rev() {
            let digit = flat % dims[axis];
            flat /= dims[axis];
            l += digit / right_dims[axis] * stride;
            stride *= left_dims[axis];
            r[axis] = digit % right_dims[axis];
        }
        multiply(left.as_slice()[l], right.get(r))
    };
    let data = if dims == [1; R] {
        Storage::One(entry(0))
    } else {
        Storage::Shared((0..dims.iter().product()).map(entry).collect::<Arc<[_]>>())
    };
    (dims, data)
}

// Zero axes stay zero even if a preceding partial product would overflow.
fn cgc_dims(factors: impl IntoIterator<IntoIter: Clone, Item = [usize; 4]>) -> Option<[usize; 4]> {
    let factors = factors.into_iter();
    let mut dims = [0; 4];
    for axis in 0..4 {
        if factors.clone().all(|d| d[axis] != 0) {
            dims[axis] = factors
                .clone()
                .try_fold(1usize, |n, d| n.checked_mul(d[axis]))?;
        }
    }
    Some(dims)
}

fn fingerprint<'a>(
    factors: impl IntoIterator<Item = &'a [u8], IntoIter: ExactSizeIterator>,
) -> Arc<[u8]> {
    let factors = factors.into_iter();
    let mut bytes =
        b"fusion-symbols:product:epoch=1:axes=last-fastest:fusion=first-fastest;".to_vec();
    bytes.extend_from_slice(&(factors.len() as u32).to_le_bytes());
    for fp in factors {
        bytes.extend_from_slice(&(fp.len() as u64).to_le_bytes());
        bytes.extend_from_slice(fp);
    }
    bytes.into()
}

fn product_unrank(factors: &[ExactCount], index: u64) -> Result<Vec<BigInt>, EnumerationError> {
    // The zero-coordinate product has exactly one object. The general unrank
    // routine requires at least one coordinate.
    if factors.is_empty() {
        return if index == 0 {
            Ok(Vec::new())
        } else {
            Err(EnumerationError::OutOfRange)
        };
    }
    let bounds: Vec<_> = factors.iter().map(|c| c.total.bound()).collect();
    unrank(&bounds, &BigInt::from(index)).ok_or(EnumerationError::OutOfRange)
}

fn product_rank(factors: Vec<Ranked>) -> Result<Ranked, EnumerationError> {
    let (counts, x): (Vec<_>, Vec<_>) = factors
        .into_iter()
        .map(|Ranked { count, index }| (count, index))
        .unzip();
    let index =
        rank_checked(&counts.iter().collect::<Vec<_>>(), &x).ok_or(EnumerationError::Overflow)?;
    Ok(Ranked {
        count: Count::product(&counts),
        index,
    })
}

fn cgc_indices(
    mut remaining: [usize; 4],
    mut index: [usize; 4],
) -> impl FnMut([usize; 4]) -> [usize; 4] {
    assert!(
        index.iter().zip(remaining).all(|(&i, n)| i < n),
        "CGC index out of range"
    );
    move |dims| {
        std::array::from_fn(|axis| {
            remaining[axis] /= dims[axis];
            let i = index[axis] / remaining[axis];
            index[axis] %= remaining[axis];
            i
        })
    }
}

// Start at the first factor, rather than multiplying it by an identity: this
// preserves signed zeros and the tuple path's floating-point fold exactly.
fn scalar_product<S: Scalar>(values: impl Iterator<Item = S>) -> S {
    values.reduce(|a, b| a * b).unwrap_or_else(S::one)
}

// A carry resets an initialized factor, never calls its category again.
fn advance<I: Iterator + Clone>(initial: &I, active: &mut I) -> (bool, I::Item) {
    if let Some(value) = active.next() {
        (true, value)
    } else {
        let (reset, value) = restarted(initial, 0);
        *active = reset;
        (false, value)
    }
}

fn restarted<I: Iterator + Clone>(initial: &I, skip: usize) -> (I, I::Item) {
    let mut active = initial.clone();
    let value = active
        .nth(skip)
        .expect("initialized fusion iterator length changed");
    (active, value)
}

// Every product query and CGC validates here, so the diagnostic cannot drift.
fn validate<C: FusionCategory>(cat: &C, objects: &[&C::Object]) {
    for a in objects {
        assert!(cat.contains(a), "foreign product object");
    }
}

macro_rules! product_queries {
    ($this:ident; $braidings:expr) => {
            fn dual(&$this, a: &Self::Object) -> Self::Object {
                validate($this, &[a]);
                $this.__fusion_symbols_validated_dual(a, Validated(()))
            }
            fn object_count(&$this) -> Result<Option<u64>, EnumerationError> {
                match $this.__fusion_symbols_exact_object_count(Validated(()))?.total {
                    Count::Finite(n) => u64::try_from(n).map(Some).map_err(|_| EnumerationError::Overflow),
                    Count::AboveU64 => Err(EnumerationError::Overflow),
                    Count::Infinite => Ok(None),
                }
            }
            fn object_at(&$this, index: u64) -> Result<Self::Object, EnumerationError> {
                let t = Validated(());
                let count = $this.__fusion_symbols_exact_object_count(t)?;
                $this.__fusion_symbols_counted_object_at(index, &count, t)
            }
            fn object_index(&$this, a: &Self::Object) -> Result<u64, EnumerationError> {
                validate($this, &[a]);
                let Ranked { index, .. } = $this.__fusion_symbols_validated_object_index(a, Validated(()))?;
                u64::try_from(index).map_err(|_| EnumerationError::Overflow)
            }
            fn frobenius_schur_phase(&$this, a: &Self::Object) -> Self::Scalar {
                validate($this, &[a]);
                $this.__fusion_symbols_validated_frobenius_schur_phase(a, Validated(()))
            }
            fn frobenius_schur_indicator(&$this, a: &Self::Object) -> Self::Scalar {
                validate($this, &[a]);
                $this.__fusion_symbols_validated_frobenius_schur_indicator(a, Validated(()))
            }
            fn twist(&$this, a: &Self::Object) -> Self::Scalar {
                validate($this, &[a]);
                if let Some(factor) = ($braidings).into_iter().position(|s| s == BraidingStyle::None) {
                    panic!("product factor {factor} has no braiding");
                }
                $this.__fusion_symbols_validated_twist(a, Validated(()))
            }
            fn fusion<'a>(&'a $this, a: &Self::Object, b: &Self::Object) -> Result<Self::Fusion<'a>, Self::Error> {
                validate($this, &[a, b]);
                $this.__fusion_symbols_validated_fusion(a, b, Validated(()))
            }
            /// Panics only if a nonzero product multiplicity exceeds `u32`, or an object is foreign;
            /// [`FusionCategory::try_nsymbol`] reports the former as an error.
            fn nsymbol(&$this, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> u32 {
                validate($this, &[a, b, c]);
                $this.__fusion_symbols_validated_nsymbol(a, b, c, Validated(()))
            }
            /// A zero factor multiplicity beats both factor failures and overflow.
            /// Otherwise the first factor failure is returned at its immediate tuple
            /// position or vector index, then [`ProductError::Overflow`] if the
            /// product multiplicity exceeds `u32`. Foreign objects panic.
            fn try_nsymbol(&$this, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Result<u32, Self::Error> {
                validate($this, &[a, b, c]);
                $this.__fusion_symbols_validated_try_nsymbol(a, b, c, Validated(()))
            }
            fn f_symbol(&$this, a: &Self::Object, b: &Self::Object, c: &Self::Object, d: &Self::Object, e: &Self::Object, f: &Self::Object) -> Result<Self::FSymbol, Self::Error> {
                validate($this, &[a, b, c, d, e, f]);
                let t = Validated(());
                let n = |x, y, z| $this.__fusion_symbols_validated_try_nsymbol(x, y, z, t);
                let dims = shape::<Self::Scalar, 4>([n(a, b, e)?, n(e, c, d)?, n(b, c, f)?, n(a, f, d)?]).ok_or(ProductError::Overflow)?;
                if dims.contains(&0) { return Ok(FBlock::from_storage(dims, Storage::Empty)); }
                let (block, _) = $this.__fusion_symbols_validated_f_symbol(a, b, c, d, e, f, t)?;
                debug_assert_eq!(block.dims(), dims);
                Ok(block)
            }
            fn r_symbol(&$this, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Result<Self::RSymbol, Self::Error> {
                validate($this, &[a, b, c]);
                if let Some(factor) = ($braidings).into_iter().position(|s| s == BraidingStyle::None) {
                    return Err(ProductError::NoBraiding { factor });
                }
                let t = Validated(());
                let n = |x, y, z| $this.__fusion_symbols_validated_try_nsymbol(x, y, z, t);
                let dims = shape::<Self::Scalar, 2>([n(a, b, c)?, n(b, a, c)?]).ok_or(ProductError::Overflow)?;
                if dims.contains(&0) { return Ok(RBlock::from_storage(dims[0], Storage::Empty)); }
                let (block, _) = $this.__fusion_symbols_validated_r_symbol(a, b, c, t)?;
                debug_assert_eq!(block.dims(), dims);
                Ok(block)
            }
            // Tuple and vector products use factor Kronecker products as for F/R
            // (TensorKitSectors `product.jl::Asymbol`); braiding is not needed.
            fn a_symbol(&$this, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Result<RBlock<Self::Scalar>, Self::Error> {
                validate($this, &[a, b, c]);
                let t = Validated(());
                let n = |x, y, z| $this.__fusion_symbols_validated_try_nsymbol(x, y, z, t);
                let dims = shape::<Self::Scalar, 2>([n(a, b, c)?, n(&$this.__fusion_symbols_validated_dual(a, t), c, b)?]).ok_or(ProductError::Overflow)?;
                if dims.contains(&0) { return Ok(RBlock::from_storage(dims[0], Storage::Empty)); }
                let (block, _) = $this.__fusion_symbols_validated_a_symbol(a, b, c, t)?;
                debug_assert_eq!(block.dims(), dims);
                Ok(block)
            }
            // Same composition as `product.jl::Bsymbol`, for tuple and vector factors.
            fn b_symbol(&$this, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Result<RBlock<Self::Scalar>, Self::Error> {
                validate($this, &[a, b, c]);
                let t = Validated(());
                let n = |x, y, z| $this.__fusion_symbols_validated_try_nsymbol(x, y, z, t);
                let dims = shape::<Self::Scalar, 2>([n(a, b, c)?, n(c, &$this.__fusion_symbols_validated_dual(b, t), a)?]).ok_or(ProductError::Overflow)?;
                if dims.contains(&0) { return Ok(RBlock::from_storage(dims[0], Storage::Empty)); }
                let (block, _) = $this.__fusion_symbols_validated_b_symbol(a, b, c, t)?;
                debug_assert_eq!(block.dims(), dims);
                Ok(block)
            }
    };
}

macro_rules! impl_product {
    ($C0:ident; $Last:ident; $(($C:ident, $i:tt, $Err:ident, $Prev:ty, $S:ident)),+) => {
        impl<$C0, $($C, $S),+> FusionCategory for Product<($C0, $($C,)+)>
        where
            $C0: FusionCategory,
            for<'a> $C0::Fusion<'a>: Clone,
            $( $C: FusionCategory, for<'a> $C::Fusion<'a>: Clone,
               $Prev: Promote<$C::Scalar, Output = $S>, $S: Scalar, )+
        {
            type Object = ($C0::Object, $($C::Object,)+);
            type Scalar = $Last;
            type FSymbol = FBlock<Self::Scalar>;
            type RSymbol = RBlock<Self::Scalar>;
            type Error = ProductError<$C0::Error, $($C::Error),+>;
            type Fusion<'a> = ProductFusion<(Fuse<$C0::Fusion<'a>>, $(Fuse<$C::Fusion<'a>>,)+), Self::Object> where Self: 'a;

            product_queries!(self; [self.factors.0.braiding_style(), $(self.factors.$i.braiding_style()),+]);

            fn fusion_style(&self) -> FusionStyle {
                use FusionStyle::*;
                let styles = [self.factors.0.fusion_style(), $(self.factors.$i.fusion_style()),+];
                if styles.contains(&Generic) { Generic } else if styles.contains(&Simple) { Simple } else { Unique }
            }
            fn braiding_style(&self) -> BraidingStyle {
                use BraidingStyle::*;
                let styles = [self.factors.0.braiding_style(), $(self.factors.$i.braiding_style()),+];
                if styles.contains(&None) { None } else if styles.contains(&Anyonic) { Anyonic }
                else if styles.contains(&Fermionic) { Fermionic } else { Bosonic }
            }
            fn unit(&self) -> Self::Object { (self.factors.0.unit(), $(self.factors.$i.unit(),)+) }
            // Each Deligne factor acts on its own tensor factor, so a product F is
            // the Kronecker product of factor F (`__fusion_symbols_validated_f_symbol`):
            // all ones, or identities on unit legs, exactly when every factor's are.
            fn has_trivial_associator_gauge(&self) -> bool {
                self.factors.0.has_trivial_associator_gauge() $(&& self.factors.$i.has_trivial_associator_gauge())+
            }
            fn has_canonical_unit(&self) -> bool {
                self.factors.0.has_canonical_unit() $(&& self.factors.$i.has_canonical_unit())+
            }
            // One public `contains` per immediate factor; a nested product
            // recurses once into its own factors, so each leaf is checked once.
            fn contains(&self, a: &Self::Object) -> bool {
                self.factors.0.contains(&a.0) $(&& self.factors.$i.contains(&a.$i))+
            }
            // Public queries validate every leaf object once through `contains`, then
            // use the `__fusion_symbols_validated_*` paths below. Not through `dim`:
            // a leaf dimension may be an exact Weyl-formula evaluation, while
            // membership is O(1) by the `contains` contract.
            fn dim(&self, a: &Self::Object) -> f64 {
                let value = self.factors.0.dim(&a.0); $(let value = value * self.factors.$i.dim(&a.$i);)+ value
            }
            fn sqrt_dim(&self, a: &Self::Object) -> f64 {
                let value = self.factors.0.sqrt_dim(&a.0); $(let value = value * self.factors.$i.sqrt_dim(&a.$i);)+ value
            }
            fn inv_sqrt_dim(&self, a: &Self::Object) -> f64 {
                let value = self.factors.0.inv_sqrt_dim(&a.0); $(let value = value * self.factors.$i.inv_sqrt_dim(&a.$i);)+ value
            }
            fn inv_dim(&self, a: &Self::Object) -> f64 {
                let value = self.factors.0.inv_dim(&a.0); $(let value = value * self.factors.$i.inv_dim(&a.$i);)+ value
            }

            fn __fusion_symbols_counted_object_at(&self, index: u64, count: &ExactCount, t: Validated) -> Result<Self::Object, EnumerationError> {
                let factors = &count.factors;
                // Exact for every `index ≤ u64::MAX`, also with `AboveU64` factors (`Count::bound`).
                let x = product_unrank(factors, index)?;
                // A coordinate never exceeds the index of its tuple.
                let x = |j: usize| u64::try_from(&x[j]).map_err(|_| EnumerationError::Overflow);
                Ok((self.factors.0.__fusion_symbols_counted_object_at(x(0)?, &factors[0], t)?,
                    $(self.factors.$i.__fusion_symbols_counted_object_at(x($i)?, &factors[$i], t)?,)+))
            }

            fn __fusion_symbols_validated_sqrt_dim(&self, a: &Self::Object, t: Validated) -> f64 {
                let value = self.factors.0.__fusion_symbols_validated_sqrt_dim(&a.0, t); $(let value = value * self.factors.$i.__fusion_symbols_validated_sqrt_dim(&a.$i, t);)+ value
            }
            fn __fusion_symbols_validated_inv_sqrt_dim(&self, a: &Self::Object, t: Validated) -> f64 {
                let value = self.factors.0.__fusion_symbols_validated_inv_sqrt_dim(&a.0, t); $(let value = value * self.factors.$i.__fusion_symbols_validated_inv_sqrt_dim(&a.$i, t);)+ value
            }
            fn __fusion_symbols_validated_inv_dim(&self, a: &Self::Object, t: Validated) -> f64 {
                let value = self.factors.0.__fusion_symbols_validated_inv_dim(&a.0, t); $(let value = value * self.factors.$i.__fusion_symbols_validated_inv_dim(&a.$i, t);)+ value
            }
            fn __fusion_symbols_exact_object_count(&self, t: Validated) -> Result<ExactCount, EnumerationError> {
                let counts = all_ok([self.factors.0.__fusion_symbols_exact_object_count(t), $(self.factors.$i.__fusion_symbols_exact_object_count(t)),+])?;
                Ok(ExactCount { total: Count::product(counts.iter().map(|c| &c.total)), factors: counts })
            }
            fn __fusion_symbols_validated_object_index(&self, a: &Self::Object, t: Validated) -> Result<Ranked, EnumerationError> {
                let factors = all_ok([self.factors.0.__fusion_symbols_validated_object_index(&a.0, t), $(self.factors.$i.__fusion_symbols_validated_object_index(&a.$i, t)),+])?;
                product_rank(factors)
            }

            fn __fusion_symbols_validated_dual(&self, a: &Self::Object, t: Validated) -> Self::Object {
                (self.factors.0.__fusion_symbols_validated_dual(&a.0, t), $(self.factors.$i.__fusion_symbols_validated_dual(&a.$i, t),)+)
            }
            fn __fusion_symbols_validated_frobenius_schur_phase(&self, a: &Self::Object, t: Validated) -> Self::Scalar {
                let value = self.factors.0.__fusion_symbols_validated_frobenius_schur_phase(&a.0, t);
                $(let value = <$Prev as Promote<$C::Scalar>>::left(value) * <$Prev as Promote<$C::Scalar>>::right(self.factors.$i.__fusion_symbols_validated_frobenius_schur_phase(&a.$i, t));)+ value
            }
            fn __fusion_symbols_validated_frobenius_schur_indicator(&self, a: &Self::Object, t: Validated) -> Self::Scalar {
                let value = self.factors.0.__fusion_symbols_validated_frobenius_schur_indicator(&a.0, t);
                $(let value = <$Prev as Promote<$C::Scalar>>::left(value) * <$Prev as Promote<$C::Scalar>>::right(self.factors.$i.__fusion_symbols_validated_frobenius_schur_indicator(&a.$i, t));)+ value
            }
            // The caller has checked that every factor braids.
            fn __fusion_symbols_validated_twist(&self, a: &Self::Object, t: Validated) -> Self::Scalar {
                let value = self.factors.0.__fusion_symbols_validated_twist(&a.0, t);
                $(let value = <$Prev as Promote<$C::Scalar>>::left(value) * <$Prev as Promote<$C::Scalar>>::right(self.factors.$i.__fusion_symbols_validated_twist(&a.$i, t));)+ value
            }
            fn __fusion_symbols_validated_fusion<'a>(&'a self, a: &Self::Object, b: &Self::Object, t: Validated) -> Result<Self::Fusion<'a>, Self::Error> {
                let initial = (self.factors.0.__fusion_symbols_validated_fusion(&a.0, &b.0, t).map_err(ProductError::Factor0)?.fuse(),
                    $(self.factors.$i.__fusion_symbols_validated_fusion(&a.$i, &b.$i, t).map_err(ProductError::$Err)?.fuse(),)+);
                let mut active = initial.clone();
                let current = (|| Some((active.0.next()?, $(active.$i.next()?,)+)))();
                Ok(ProductFusion { initial, active, current })
            }
            fn __fusion_symbols_validated_nsymbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object, t: Validated) -> u32 {
                checked_multiplicity([self.factors.0.__fusion_symbols_validated_nsymbol(&a.0, &b.0, &c.0, t), $(self.factors.$i.__fusion_symbols_validated_nsymbol(&a.$i, &b.$i, &c.$i, t)),+].map(Ok), ())
                    .expect("nonzero product multiplicity exceeds u32")
            }
            fn __fusion_symbols_validated_try_nsymbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object, t: Validated) -> Result<u32, Self::Error> {
                checked_multiplicity([
                    self.factors.0.__fusion_symbols_validated_try_nsymbol(&a.0, &b.0, &c.0, t).map_err(ProductError::Factor0),
                    $(self.factors.$i.__fusion_symbols_validated_try_nsymbol(&a.$i, &b.$i, &c.$i, t).map_err(ProductError::$Err)),+
                ], ProductError::Overflow)
            }
            // The caller has established nonzero, representable shapes for every
            // factor (an enclosing product's shape bounds this one's). Check each
            // returned block before storage/kron; pass checked axes up the nesting
            // without re-querying N or copying coefficient data.
            fn __fusion_symbols_validated_f_symbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object, d: &Self::Object, e: &Self::Object, f: &Self::Object, t: Validated) -> Result<SymbolWithAxes<Self::FSymbol, 4>, Self::Error> {
                let (dims, data) = storage(Self::Error::checked_block(0, self.factors.0.__fusion_symbols_validated_f_symbol(&a.0,&b.0,&c.0,&d.0,&e.0,&f.0,t).map_err(ProductError::Factor0)?)?);
                $(let (dims, data) = kron(dims, data, Self::Error::checked_block($i, self.factors.$i.__fusion_symbols_validated_f_symbol(&a.$i,&b.$i,&c.$i,&d.$i,&e.$i,&f.$i,t).map_err(ProductError::$Err)?)?, |l, r| <$Prev as Promote<$C::Scalar>>::left(l) * <$Prev as Promote<$C::Scalar>>::right(r));)+
                let [n0, n1, n2, n3] = dims.map(|n| u32::try_from(n).map_err(|_| ProductError::Overflow));
                Ok((FBlock::from_storage(dims, data), [n0?, n1?, n2?, n3?]))
            }
            // As for F; braiding of every factor has also been checked.
            fn __fusion_symbols_validated_r_symbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object, t: Validated) -> Result<SymbolWithAxes<Self::RSymbol, 2>, Self::Error> {
                let (dims, data) = storage(Self::Error::checked_block(0, self.factors.0.__fusion_symbols_validated_r_symbol(&a.0,&b.0,&c.0,t).map_err(ProductError::Factor0)?)?);
                $(let (dims, data) = kron(dims, data, Self::Error::checked_block($i, self.factors.$i.__fusion_symbols_validated_r_symbol(&a.$i,&b.$i,&c.$i,t).map_err(ProductError::$Err)?)?, |l, r| <$Prev as Promote<$C::Scalar>>::left(l) * <$Prev as Promote<$C::Scalar>>::right(r));)+
                let [n0, n1] = dims.map(|n| u32::try_from(n).map_err(|_| ProductError::Overflow));
                Ok((RBlock::from_storage(dims[0], data), [n0?, n1?]))
            }
            fn __fusion_symbols_validated_a_symbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object, t: Validated) -> Result<SymbolWithAxes<RBlock<Self::Scalar>, 2>, Self::Error> {
                let (dims, data) = storage(Self::Error::checked_block(0, self.factors.0.__fusion_symbols_validated_a_symbol(&a.0,&b.0,&c.0,t).map_err(ProductError::Factor0)?)?);
                $(let (dims, data) = kron(dims, data, Self::Error::checked_block($i, self.factors.$i.__fusion_symbols_validated_a_symbol(&a.$i,&b.$i,&c.$i,t).map_err(ProductError::$Err)?)?, |l, r| <$Prev as Promote<$C::Scalar>>::left(l) * <$Prev as Promote<$C::Scalar>>::right(r));)+
                let [n0, n1] = dims.map(|n| u32::try_from(n).map_err(|_| ProductError::Overflow));
                Ok((RBlock::from_storage(dims[0], data), [n0?, n1?]))
            }
            fn __fusion_symbols_validated_b_symbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object, t: Validated) -> Result<SymbolWithAxes<RBlock<Self::Scalar>, 2>, Self::Error> {
                let (dims, data) = storage(Self::Error::checked_block(0, self.factors.0.__fusion_symbols_validated_b_symbol(&a.0,&b.0,&c.0,t).map_err(ProductError::Factor0)?)?);
                $(let (dims, data) = kron(dims, data, Self::Error::checked_block($i, self.factors.$i.__fusion_symbols_validated_b_symbol(&a.$i,&b.$i,&c.$i,t).map_err(ProductError::$Err)?)?, |l, r| <$Prev as Promote<$C::Scalar>>::left(l) * <$Prev as Promote<$C::Scalar>>::right(r));)+
                let [n0, n1] = dims.map(|n| u32::try_from(n).map_err(|_| ProductError::Overflow));
                Ok((RBlock::from_storage(dims[0], data), [n0?, n1?]))
            }
            fn authority_fingerprint(&self) -> &[u8] {
                self.fingerprint.get_or_init(|| {
                    fingerprint([self.factors.0.authority_fingerprint(), $(self.factors.$i.authority_fingerprint()),+])
                })
            }
        }
        impl<$C0, $($C, $S),+> GroupRepresentations for Product<($C0, $($C,)+)>
        where
            $C0: GroupRepresentations,
            for<'a> $C0::Fusion<'a>: Clone,
            $( $C: GroupRepresentations, for<'a> $C::Fusion<'a>: Clone,
               $Prev: Promote<$C::Scalar, Output = $S>, $S: Scalar, )+
        {
            type Cgc = ($C0::Cgc, $($C::Cgc,)+);
            fn cgc(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Result<Self::Cgc, Self::Error> {
                validate(self, &[a, b, c]);
                let blocks = (self.factors.0.cgc(&a.0, &b.0, &c.0).map_err(ProductError::Factor0)?,
                    $(self.factors.$i.cgc(&a.$i, &b.$i, &c.$i).map_err(ProductError::$Err)?,)+);
                cgc_dims([blocks.0.dims(), $(blocks.$i.dims()),+]).ok_or(ProductError::Overflow)?;
                Ok(blocks)
            }
        }
        impl<$C0: CgcBlock, $($C: CgcBlock),+> CgcBlock for ($C0, $($C,)+) {
            fn dims(&self) -> [usize; 4] {
                cgc_dims([self.0.dims(), $(self.$i.dims()),+]).expect("CGC axis overflow")
            }
            fn at(&self, index: [usize; 4]) -> f64 {
                let mut entry = cgc_indices(self.dims(), index);
                let value = self.0.at(entry(self.0.dims()));
                $(let value = value * self.$i.at(entry(self.$i.dims()));)+
                value
            }
        }
        impl<I0, O0, $($C, $S),+> Iterator for ProductFusion<(I0, $($C,)+), (O0, $($S,)+)>
        where I0: Iterator<Item = O0> + Clone, O0: Clone,
              $($C: Iterator<Item = $S> + Clone, $S: Clone,)+
        {
            type Item = (O0, $($S,)+);
            fn next(&mut self) -> Option<Self::Item> {
                let mut current = self.current.take()?;
                let result = current.clone();
                let (advanced, value) = advance(&self.initial.0, &mut self.active.0);
                current.0 = value;
                if advanced { self.current = Some(current); return Some(result); }
                $(let (advanced, value) = advance(&self.initial.$i, &mut self.active.$i);
                current.$i = value;
                if advanced { self.current = Some(current); return Some(result); })+
                Some(result)
            }
        }
    };
}
impl_product!(C0; S1; (C1, 1, Factor1, C0::Scalar, S1));
impl_product!(C0; S2; (C1, 1, Factor1, C0::Scalar, S1), (C2, 2, Factor2, S1, S2));
impl_product!(C0; S3; (C1, 1, Factor1, C0::Scalar, S1), (C2, 2, Factor2, S1, S2), (C3, 3, Factor3, S2, S3));
impl_product!(C0; S4; (C1, 1, Factor1, C0::Scalar, S1), (C2, 2, Factor2, S1, S2), (C3, 3, Factor3, S2, S3), (C4, 4, Factor4, S3, S4));
impl_product!(C0; S5; (C1, 1, Factor1, C0::Scalar, S1), (C2, 2, Factor2, S1, S2), (C3, 3, Factor3, S2, S3), (C4, 4, Factor4, S3, S4), (C5, 5, Factor5, S4, S5));
impl_product!(C0; S6; (C1, 1, Factor1, C0::Scalar, S1), (C2, 2, Factor2, S1, S2), (C3, 3, Factor3, S2, S3), (C4, 4, Factor4, S3, S4), (C5, 5, Factor5, S4, S5), (C6, 6, Factor6, S5, S6));
impl_product!(C0; S7; (C1, 1, Factor1, C0::Scalar, S1), (C2, 2, Factor2, S1, S2), (C3, 3, Factor3, S2, S3), (C4, 4, Factor4, S3, S4), (C5, 5, Factor5, S4, S5), (C6, 6, Factor6, S5, S6), (C7, 7, Factor7, S6, S7));
impl_product!(C0; S8; (C1, 1, Factor1, C0::Scalar, S1), (C2, 2, Factor2, S1, S2), (C3, 3, Factor3, S2, S3), (C4, 4, Factor4, S3, S4), (C5, 5, Factor5, S4, S5), (C6, 6, Factor6, S5, S6), (C7, 7, Factor7, S6, S7), (C8, 8, Factor8, S7, S8));
impl_product!(C0; S9; (C1, 1, Factor1, C0::Scalar, S1), (C2, 2, Factor2, S1, S2), (C3, 3, Factor3, S2, S3), (C4, 4, Factor4, S3, S4), (C5, 5, Factor5, S4, S5), (C6, 6, Factor6, S5, S6), (C7, 7, Factor7, S6, S7), (C8, 8, Factor8, S7, S8), (C9, 9, Factor9, S8, S9));
impl_product!(C0; S10; (C1, 1, Factor1, C0::Scalar, S1), (C2, 2, Factor2, S1, S2), (C3, 3, Factor3, S2, S3), (C4, 4, Factor4, S3, S4), (C5, 5, Factor5, S4, S5), (C6, 6, Factor6, S5, S6), (C7, 7, Factor7, S6, S7), (C8, 8, Factor8, S7, S8), (C9, 9, Factor9, S8, S9), (C10, 10, Factor10, S9, S10));
impl_product!(C0; S11; (C1, 1, Factor1, C0::Scalar, S1), (C2, 2, Factor2, S1, S2), (C3, 3, Factor3, S2, S3), (C4, 4, Factor4, S3, S4), (C5, 5, Factor5, S4, S5), (C6, 6, Factor6, S5, S6), (C7, 7, Factor7, S6, S7), (C8, 8, Factor8, S7, S8), (C9, 9, Factor9, S8, S9), (C10, 10, Factor10, S9, S10), (C11, 11, Factor11, S10, S11));

impl<C: FusionCategory> FusionCategory for Product<Vec<C>>
where
    for<'a> C::Fusion<'a>: Clone,
{
    type Object = Vec<C::Object>;
    type Scalar = C::Scalar;
    type FSymbol = FBlock<Self::Scalar>;
    type RSymbol = RBlock<Self::Scalar>;
    type Error = ProductError<C::Error>;
    type Fusion<'a>
        = ProductFusion<Vec<(C::Fusion<'a>, usize)>, (usize, usize, Vec<C::Object>)>
    where
        Self: 'a;

    product_queries!(self; self.factors.iter().map(FusionCategory::braiding_style));

    fn fusion_style(&self) -> FusionStyle {
        use FusionStyle::*;
        self.factors
            .iter()
            .map(FusionCategory::fusion_style)
            .fold(Unique, |a, b| {
                if a == Generic || b == Generic {
                    Generic
                } else if a == Simple || b == Simple {
                    Simple
                } else {
                    Unique
                }
            })
    }
    fn braiding_style(&self) -> BraidingStyle {
        use BraidingStyle::*;
        self.factors
            .iter()
            .map(FusionCategory::braiding_style)
            .fold(Bosonic, |a, b| {
                if a == None || b == None {
                    None
                } else if a == Anyonic || b == Anyonic {
                    Anyonic
                } else if a == Fermionic || b == Fermionic {
                    Fermionic
                } else {
                    Bosonic
                }
            })
    }
    fn unit(&self) -> Self::Object {
        self.factors.iter().map(FusionCategory::unit).collect()
    }
    fn contains(&self, a: &Self::Object) -> bool {
        a.len() == self.factors.len() && self.factors.iter().zip(a).all(|(c, a)| c.contains(a))
    }
    fn has_trivial_associator_gauge(&self) -> bool {
        self.factors
            .iter()
            .all(FusionCategory::has_trivial_associator_gauge)
    }
    fn has_canonical_unit(&self) -> bool {
        self.factors.iter().all(FusionCategory::has_canonical_unit)
    }
    fn dim(&self, a: &Self::Object) -> f64 {
        assert_eq!(a.len(), self.factors.len(), "foreign product object");
        scalar_product(self.factors.iter().zip(a).map(|(c, a)| c.dim(a)))
    }
    fn sqrt_dim(&self, a: &Self::Object) -> f64 {
        assert_eq!(a.len(), self.factors.len(), "foreign product object");
        scalar_product(self.factors.iter().zip(a).map(|(c, a)| c.sqrt_dim(a)))
    }
    fn inv_sqrt_dim(&self, a: &Self::Object) -> f64 {
        assert_eq!(a.len(), self.factors.len(), "foreign product object");
        scalar_product(self.factors.iter().zip(a).map(|(c, a)| c.inv_sqrt_dim(a)))
    }
    fn inv_dim(&self, a: &Self::Object) -> f64 {
        assert_eq!(a.len(), self.factors.len(), "foreign product object");
        scalar_product(self.factors.iter().zip(a).map(|(c, a)| c.inv_dim(a)))
    }
    fn __fusion_symbols_validated_sqrt_dim(&self, a: &Self::Object, t: Validated) -> f64 {
        scalar_product(
            self.factors
                .iter()
                .zip(a)
                .map(|(c, a)| c.__fusion_symbols_validated_sqrt_dim(a, t)),
        )
    }
    fn __fusion_symbols_validated_inv_sqrt_dim(&self, a: &Self::Object, t: Validated) -> f64 {
        scalar_product(
            self.factors
                .iter()
                .zip(a)
                .map(|(c, a)| c.__fusion_symbols_validated_inv_sqrt_dim(a, t)),
        )
    }
    fn __fusion_symbols_validated_inv_dim(&self, a: &Self::Object, t: Validated) -> f64 {
        scalar_product(
            self.factors
                .iter()
                .zip(a)
                .map(|(c, a)| c.__fusion_symbols_validated_inv_dim(a, t)),
        )
    }
    fn __fusion_symbols_validated_frobenius_schur_phase(
        &self,
        a: &Self::Object,
        t: Validated,
    ) -> Self::Scalar {
        scalar_product(
            self.factors
                .iter()
                .zip(a)
                .map(|(c, a)| c.__fusion_symbols_validated_frobenius_schur_phase(a, t)),
        )
    }
    fn __fusion_symbols_validated_frobenius_schur_indicator(
        &self,
        a: &Self::Object,
        t: Validated,
    ) -> Self::Scalar {
        scalar_product(
            self.factors
                .iter()
                .zip(a)
                .map(|(c, a)| c.__fusion_symbols_validated_frobenius_schur_indicator(a, t)),
        )
    }
    fn __fusion_symbols_validated_twist(&self, a: &Self::Object, t: Validated) -> Self::Scalar {
        scalar_product(
            self.factors
                .iter()
                .zip(a)
                .map(|(c, a)| c.__fusion_symbols_validated_twist(a, t)),
        )
    }
    fn __fusion_symbols_validated_dual(&self, a: &Self::Object, t: Validated) -> Self::Object {
        self.factors
            .iter()
            .zip(a)
            .map(|(c, a)| c.__fusion_symbols_validated_dual(a, t))
            .collect()
    }
    fn __fusion_symbols_exact_object_count(
        &self,
        t: Validated,
    ) -> Result<ExactCount, EnumerationError> {
        let counts = all_ok(
            self.factors
                .iter()
                .map(|c| c.__fusion_symbols_exact_object_count(t)),
        )?;
        Ok(ExactCount {
            total: Count::product(counts.iter().map(|c| &c.total)),
            factors: counts,
        })
    }
    fn __fusion_symbols_counted_object_at(
        &self,
        index: u64,
        count: &ExactCount,
        t: Validated,
    ) -> Result<Self::Object, EnumerationError> {
        let x = product_unrank(&count.factors, index)?;
        self.factors
            .iter()
            .zip(&count.factors)
            .zip(x)
            .map(|((c, count), i)| {
                c.__fusion_symbols_counted_object_at(
                    u64::try_from(i).map_err(|_| EnumerationError::Overflow)?,
                    count,
                    t,
                )
            })
            .collect()
    }
    fn __fusion_symbols_validated_object_index(
        &self,
        a: &Self::Object,
        t: Validated,
    ) -> Result<Ranked, EnumerationError> {
        product_rank(all_ok(
            self.factors
                .iter()
                .zip(a)
                .map(|(c, a)| c.__fusion_symbols_validated_object_index(a, t)),
        )?)
    }
    fn __fusion_symbols_validated_nsymbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        t: Validated,
    ) -> u32 {
        checked_multiplicity(
            self.factors
                .iter()
                .enumerate()
                .map(|(i, f)| Ok(f.__fusion_symbols_validated_nsymbol(&a[i], &b[i], &c[i], t))),
            (),
        )
        .expect("nonzero product multiplicity exceeds u32")
    }
    fn __fusion_symbols_validated_try_nsymbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        t: Validated,
    ) -> Result<u32, Self::Error> {
        checked_multiplicity(
            self.factors.iter().enumerate().map(|(index, f)| {
                f.__fusion_symbols_validated_try_nsymbol(&a[index], &b[index], &c[index], t)
                    .map_err(|source| ProductError::Factor { index, source })
            }),
            ProductError::Overflow,
        )
    }
    fn __fusion_symbols_validated_fusion<'a>(
        &'a self,
        a: &Self::Object,
        b: &Self::Object,
        t: Validated,
    ) -> Result<Self::Fusion<'a>, Self::Error> {
        let initial: Vec<_> = self
            .factors
            .iter()
            .enumerate()
            .map(|(index, f)| {
                let iter = f
                    .__fusion_symbols_validated_fusion(&a[index], &b[index], t)
                    .map_err(|source| ProductError::Factor { index, source })?;
                let (lo, hi) = iter.size_hint();
                let len = if hi == Some(lo) {
                    lo
                } else {
                    iter.clone()
                        .try_fold(0usize, |n, _| n.checked_add(1))
                        .ok_or(ProductError::Overflow)?
                };
                Ok((iter, len))
            })
            .collect::<Result<_, Self::Error>>()?;
        let len = if initial.iter().any(|(_, n)| *n == 0) {
            0
        } else {
            initial
                .iter()
                .try_fold(1usize, |p, (_, n)| p.checked_mul(*n))
                .ok_or(ProductError::Overflow)?
        };
        let mut active = initial.clone();
        let current = if len == 0 {
            Vec::new()
        } else {
            active
                .iter_mut()
                .map(|(iter, _)| {
                    iter.next()
                        .expect("initialized fusion iterator length changed")
                })
                .collect()
        };
        Ok(ProductFusion {
            initial,
            active,
            current: Some((0, len, current)),
        })
    }
    fn __fusion_symbols_validated_f_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        d: &Self::Object,
        e: &Self::Object,
        f: &Self::Object,
        t: Validated,
    ) -> Result<SymbolWithAxes<Self::FSymbol, 4>, Self::Error> {
        let (dims, data) =
            compose_blocks(self.factors.iter().enumerate().map(|(index, factor)| {
                let block = factor
                    .__fusion_symbols_validated_f_symbol(
                        &a[index], &b[index], &c[index], &d[index], &e[index], &f[index], t,
                    )
                    .map_err(|source| ProductError::Factor { index, source })?;
                ProductError::checked_block(index, block)
            }))?;
        let n = dims.map(|n| u32::try_from(n).map_err(|_| ProductError::Overflow));
        let [n0, n1, n2, n3] = n;
        Ok((FBlock::from_storage(dims, data), [n0?, n1?, n2?, n3?]))
    }
    fn __fusion_symbols_validated_r_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        t: Validated,
    ) -> Result<SymbolWithAxes<Self::RSymbol, 2>, Self::Error> {
        let (dims, data) =
            compose_blocks(self.factors.iter().enumerate().map(|(index, factor)| {
                let block = factor
                    .__fusion_symbols_validated_r_symbol(&a[index], &b[index], &c[index], t)
                    .map_err(|source| ProductError::Factor { index, source })?;
                ProductError::checked_block(index, block)
            }))?;
        let n = dims.map(|n| u32::try_from(n).map_err(|_| ProductError::Overflow));
        let [n0, n1] = n;
        Ok((RBlock::from_storage(dims[0], data), [n0?, n1?]))
    }
    fn __fusion_symbols_validated_a_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        t: Validated,
    ) -> Result<SymbolWithAxes<RBlock<Self::Scalar>, 2>, Self::Error> {
        let (dims, data) =
            compose_blocks(self.factors.iter().enumerate().map(|(index, factor)| {
                let block = factor
                    .__fusion_symbols_validated_a_symbol(&a[index], &b[index], &c[index], t)
                    .map_err(|source| ProductError::Factor { index, source })?;
                ProductError::checked_block(index, block)
            }))?;
        let n = dims.map(|n| u32::try_from(n).map_err(|_| ProductError::Overflow));
        let [n0, n1] = n;
        Ok((RBlock::from_storage(dims[0], data), [n0?, n1?]))
    }
    fn __fusion_symbols_validated_b_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        t: Validated,
    ) -> Result<SymbolWithAxes<RBlock<Self::Scalar>, 2>, Self::Error> {
        let (dims, data) =
            compose_blocks(self.factors.iter().enumerate().map(|(index, factor)| {
                let block = factor
                    .__fusion_symbols_validated_b_symbol(&a[index], &b[index], &c[index], t)
                    .map_err(|source| ProductError::Factor { index, source })?;
                ProductError::checked_block(index, block)
            }))?;
        let n = dims.map(|n| u32::try_from(n).map_err(|_| ProductError::Overflow));
        let [n0, n1] = n;
        Ok((RBlock::from_storage(dims[0], data), [n0?, n1?]))
    }
    fn authority_fingerprint(&self) -> &[u8] {
        self.fingerprint.get_or_init(|| {
            fingerprint(
                self.factors
                    .iter()
                    .map(FusionCategory::authority_fingerprint),
            )
        })
    }
}

fn compose_blocks<B: SymbolBlock<R>, E, const R: usize>(
    blocks: impl IntoIterator<Item = Result<B, E>>,
) -> Result<([usize; R], Storage<B::Scalar>), E> {
    let mut blocks = blocks.into_iter();
    let Some(first) = blocks.next() else {
        return Ok(([1; R], Storage::One(B::Scalar::one())));
    };
    let mut result = storage(first?);
    for block in blocks {
        result = kron(result.0, result.1, block?, |l, r| l * r);
    }
    Ok(result)
}

impl<C: GroupRepresentations> GroupRepresentations for Product<Vec<C>>
where
    for<'a> C::Fusion<'a>: Clone,
{
    type Cgc = Vec<C::Cgc>;
    fn cgc(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<Self::Cgc, Self::Error> {
        validate(self, &[a, b, c]);
        let blocks: Vec<_> = self
            .factors
            .iter()
            .enumerate()
            .map(|(index, f)| {
                f.cgc(&a[index], &b[index], &c[index])
                    .map_err(|source| ProductError::Factor { index, source })
            })
            .collect::<Result<_, _>>()?;
        cgc_dims(blocks.iter().map(CgcBlock::dims)).ok_or(ProductError::Overflow)?;
        Ok(blocks)
    }
}
impl<B: CgcBlock> CgcBlock for Vec<B> {
    fn dims(&self) -> [usize; 4] {
        cgc_dims(self.iter().map(CgcBlock::dims)).expect("CGC axis overflow")
    }
    fn at(&self, index: [usize; 4]) -> f64 {
        let mut entry = cgc_indices(self.dims(), index);
        scalar_product(self.iter().map(|b| b.at(entry(b.dims()))))
    }
}

// Runtime specialization: current stores [front, back) and the front items;
// active holds the incremental front cursor. Each initial factor carries its radix.
impl<I: Iterator + Clone> ProductFusion<Vec<(I, usize)>, (usize, usize, Vec<I::Item>)> {
    fn entry(&self, mut index: usize) -> Vec<I::Item> {
        self.initial
            .iter()
            .map(|(iter, radix)| {
                let digit = index % radix;
                index /= radix;
                restarted(iter, digit).1
            })
            .collect()
    }
}
impl<I: Iterator + Clone> Iterator for ProductFusion<Vec<(I, usize)>, (usize, usize, Vec<I::Item>)>
where
    I::Item: Clone,
{
    type Item = Vec<I::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        let (front, back, current) = self.current.as_mut()?;
        if front == back {
            return None;
        }
        let result = current.clone();
        *front += 1;
        if front != back {
            for (((initial, _), (active, _)), item) in
                self.initial.iter().zip(&mut self.active).zip(current)
            {
                let (advanced, value) = advance(initial, active);
                *item = value;
                if advanced {
                    break;
                }
            }
        }
        Some(result)
    }
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let (front, back, current) = self.current.as_mut()?;
        if n >= *back - *front {
            *front = *back;
            return None;
        }
        if n != 0 {
            *front += n;
            let mut index = *front;
            for (((initial, radix), (active, _)), item) in
                self.initial.iter().zip(&mut self.active).zip(current)
            {
                let digit = index % radix;
                index /= radix;
                (*active, *item) = restarted(initial, digit);
            }
        }
        self.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (front, back, _) = self
            .current
            .as_ref()
            .expect("initialized runtime fusion range");
        (back - front, Some(back - front))
    }
}
impl<I: DoubleEndedIterator + Clone> DoubleEndedIterator
    for ProductFusion<Vec<(I, usize)>, (usize, usize, Vec<I::Item>)>
where
    I::Item: Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.nth_back(0)
    }
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let (front, back, _) = self.current.as_mut()?;
        if n >= *back - *front {
            *front = *back;
            return None;
        }
        *back -= n + 1;
        let index = *back;
        Some(self.entry(index))
    }
}
impl<I: ExactSizeIterator + Clone> ExactSizeIterator
    for ProductFusion<Vec<(I, usize)>, (usize, usize, Vec<I::Item>)>
where
    I::Item: Clone,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fermion::FermionParity;

    #[test]
    fn runtime_fusion_forward_is_incremental() {
        use std::{cell::Cell, ops::Range, rc::Rc};

        #[derive(Clone)]
        struct Counting {
            range: Range<usize>,
            calls: Rc<Cell<usize>>,
        }
        impl Iterator for Counting {
            type Item = usize;
            fn next(&mut self) -> Option<usize> {
                self.calls.set(self.calls.get() + 1);
                self.range.next()
            }
        }
        let calls = Rc::new(Cell::new(0));
        let radices = [8, 9, 10];
        let initial: Vec<_> = radices
            .iter()
            .map(|&n| {
                (
                    Counting {
                        range: 0..n,
                        calls: calls.clone(),
                    },
                    n,
                )
            })
            .collect();
        let len: usize = radices.iter().product();
        let mut active = initial.clone();
        let current = active
            .iter_mut()
            .map(|(iter, _)| iter.next().unwrap())
            .collect();
        let iter = ProductFusion {
            initial,
            active,
            current: Some((0, len, current)),
        };
        assert_eq!(iter.count(), len);
        assert!(
            calls.get() <= len * radices.len() + radices.iter().sum::<usize>(),
            "{} factor next calls for {len} outcomes",
            calls.get()
        );
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum Fault {
        None,
        Axes,
        EqualLengthAxes,
        Short,
        Long,
        HugeAxes,
    }

    // Synthetic storage, not a mathematical category. A/B corruption is injected
    // here because their public RBlock constructors cannot make short storage.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Blocks {
        n: u32,
        fault: Fault,
    }
    struct Block<const R: usize> {
        dims: [usize; R],
        data: Vec<f64>,
    }
    impl<const R: usize> SymbolBlock<R> for Block<R> {
        type Scalar = f64;
        fn dims(&self) -> [usize; R] {
            self.dims
        }
        fn as_slice(&self) -> &[f64] {
            &self.data
        }
        fn get(&self, index: [usize; R]) -> f64 {
            self.data[index
                .into_iter()
                .zip(self.dims)
                .fold(0, |p, (i, n)| p * n + i)]
        }
    }
    impl Blocks {
        fn block<const R: usize>(&self) -> Block<R> {
            assert_ne!(self.n, 0, "zero channels must skip factor symbols");
            let mut dims = [self.n as usize; R];
            let mut len = (self.n as usize).pow(R as u32);
            match self.fault {
                Fault::None => {}
                Fault::Axes => {
                    dims = [1; R];
                    len = 1;
                }
                Fault::EqualLengthAxes => {
                    dims[0] = 1;
                    dims[1] = 4;
                }
                Fault::Short => len = 1,
                Fault::Long => len += 1,
                Fault::HugeAxes => {
                    dims = [usize::MAX; R];
                    len = 1;
                }
            }
            Block {
                dims,
                data: [-0.0, 1.25, -2.5, 3.0]
                    .into_iter()
                    .cycle()
                    .take(len)
                    .collect(),
            }
        }
    }
    impl FusionCategory for Blocks {
        type Object = ();
        type Scalar = f64;
        type FSymbol = Block<4>;
        type RSymbol = Block<2>;
        type Fusion<'a> = std::iter::Once<()>;
        type Error = Infallible;
        fn fusion_style(&self) -> FusionStyle {
            FusionStyle::Generic
        }
        fn braiding_style(&self) -> BraidingStyle {
            BraidingStyle::Bosonic
        }
        fn unit(&self) {}
        fn dual(&self, _: &()) {}
        fn dim(&self, _: &()) -> f64 {
            1.0
        }
        fn frobenius_schur_phase(&self, _: &()) -> f64 {
            1.0
        }
        fn frobenius_schur_indicator(&self, _: &()) -> f64 {
            1.0
        }
        fn twist(&self, _: &()) -> f64 {
            1.0
        }
        fn fusion<'a>(&'a self, _: &(), _: &()) -> Result<Self::Fusion<'a>, Infallible> {
            Ok(std::iter::once(()))
        }
        fn nsymbol(&self, _: &(), _: &(), _: &()) -> u32 {
            self.n
        }
        fn f_symbol(
            &self,
            _: &(),
            _: &(),
            _: &(),
            _: &(),
            _: &(),
            _: &(),
        ) -> Result<Self::FSymbol, Infallible> {
            Ok(self.block())
        }
        fn r_symbol(&self, _: &(), _: &(), _: &()) -> Result<Self::RSymbol, Infallible> {
            Ok(self.block())
        }
        fn a_symbol(&self, _: &(), _: &(), _: &()) -> Result<RBlock<f64>, Infallible> {
            let block = self.block::<2>();
            Ok(RBlock::from_storage(
                block.dims[0],
                Storage::Shared(block.data.into()),
            ))
        }
        fn b_symbol(&self, a: &(), b: &(), c: &()) -> Result<RBlock<f64>, Infallible> {
            self.a_symbol(a, b, c)
        }
        fn authority_fingerprint(&self) -> &[u8] {
            b"malformed-blocks-test"
        }
    }

    fn query<C: FusionCategory<Scalar = f64>>(
        cat: &C,
        symbol: char,
    ) -> Result<(Vec<usize>, Vec<u64>), C::Error> {
        fn data<const R: usize>(
            block: impl SymbolBlock<R, Scalar = f64>,
        ) -> (Vec<usize>, Vec<u64>) {
            (
                block.dims().to_vec(),
                block.as_slice().iter().map(|x| x.to_bits()).collect(),
            )
        }
        let u = cat.unit();
        match symbol {
            'F' => cat.f_symbol(&u, &u, &u, &u, &u, &u).map(data),
            'R' => cat.r_symbol(&u, &u, &u).map(data),
            'A' => cat.a_symbol(&u, &u, &u).map(data),
            'B' => cat.b_symbol(&u, &u, &u).map(data),
            _ => unreachable!(),
        }
    }

    fn malformed_blocks(symbol: char) {
        let rank = if symbol == 'F' { 4 } else { 2 };
        let count = if symbol == 'F' { 16 } else { 4 };
        let good = Blocks {
            n: 2,
            fault: Fault::None,
        };
        let values: Vec<_> = [-0.0_f64, 1.25, -2.5, 3.0]
            .into_iter()
            .cycle()
            .take(count)
            .map(f64::to_bits)
            .collect();
        assert_eq!(
            query(&product(FermionParity, good.clone()), symbol).unwrap(),
            (vec![2; rank], values.clone())
        );
        assert_eq!(
            query(
                &product(product(good, FermionParity), FermionParity),
                symbol
            )
            .unwrap(),
            (vec![2; rank], values.clone())
        );
        assert_eq!(
            query(
                &Product::new(vec![Blocks {
                    n: 2,
                    fault: Fault::None
                }]),
                symbol
            )
            .unwrap(),
            (vec![2; rank], values)
        );
        for fault in [
            Fault::Axes,
            Fault::EqualLengthAxes,
            Fault::Short,
            Fault::Long,
            Fault::HugeAxes,
        ] {
            let (actual, storage_len) = match fault {
                Fault::Axes => (vec![1; rank], 1),
                Fault::EqualLengthAxes => (
                    match symbol {
                        'F' => vec![1, 4, 2, 2],
                        'R' => vec![1, 4],
                        _ => vec![1, 1],
                    },
                    count,
                ),
                Fault::Short => (vec![2; rank], 1),
                Fault::Long => (vec![2; rank], count + 1),
                Fault::HugeAxes => (vec![usize::MAX; rank], 1),
                Fault::None => unreachable!(),
            };
            let bad = Blocks { n: 2, fault };
            let expected = |factor| ProductError::<Infallible, Infallible>::MalformedSymbolData {
                factor,
                expected: vec![2; rank],
                actual: actual.clone(),
                storage_len,
            };
            assert_eq!(
                query(&product(bad.clone(), FermionParity), symbol),
                Err(expected(0))
            );
            assert_eq!(
                query(&product(FermionParity, bad.clone()), symbol),
                Err(expected(1))
            );
            assert_eq!(
                query(
                    &product(product(bad.clone(), FermionParity), FermionParity),
                    symbol
                ),
                Err(ProductError::Factor0(expected(0)))
            );
            assert_eq!(
                query(
                    &product(FermionParity, product(FermionParity, bad.clone())),
                    symbol
                ),
                Err(ProductError::Factor1(expected(1)))
            );
            assert_eq!(
                query(&Product::new(vec![bad.clone()]), symbol),
                Err(expected(0))
            );
            assert_eq!(
                query(
                    &Product::new(vec![
                        Blocks {
                            n: 1,
                            fault: Fault::None
                        },
                        bad.clone()
                    ]),
                    symbol
                ),
                Err(expected(1))
            );
            let zero = Blocks { n: 0, fault };
            assert_eq!(
                query(&product(bad.clone(), zero.clone()), symbol).unwrap(),
                (vec![0; rank], vec![])
            );
            assert_eq!(
                query(&product(product(zero, bad), FermionParity), symbol).unwrap(),
                (vec![0; rank], vec![])
            );
        }
        let overflow = Blocks {
            n: u32::MAX,
            fault: Fault::None,
        };
        assert!(matches!(
            query(&product(overflow, FermionParity), symbol),
            Err(ProductError::Overflow)
        ));
    }

    #[test]
    fn malformed_f_blocks() {
        malformed_blocks('F');
    }
    #[test]
    fn malformed_r_blocks() {
        malformed_blocks('R');
    }
    #[test]
    fn malformed_a_blocks() {
        malformed_blocks('A');
    }
    #[test]
    fn malformed_b_blocks() {
        malformed_blocks('B');
    }
}
