//! Ordered Deligne products of tuple or runtime-length homogeneous factors.
//!
//! Objects are tuples or vectors; explicit nesting is preserved. Fusion outcomes
//! vary the first factor fastest. Each F/R/A/B multiplicity axis varies the last factor
//! fastest, with row-major block storage.
//!
//! When every factor supplies exact values (Lie groups, tori, U(1), CU(1),
//! Z_N and fermion parity, nested in any way), F/R/A/B are the exact
//! Kronecker product of the factors' exact values, rounded once: unit-valued
//! factors (tori, U(1), Z_N, parity, SU(2) R, CU(1) R) contribute an exact
//! `0`/`±1`, and two or more exact non-unit leaves share one cached composite
//! view in the Product tier, keyed by the ordered leaves, so tuple, `Vec` and
//! nested products of the same leaves agree bit for bit. A product with one
//! non-unit leaf returns that leaf's own value. Otherwise (a table category
//! or a category defined outside the crate) F/R/A/B fold the leaves' public
//! values left to right through any nesting, with scalar promotion, so nested
//! products of the same leaves agree bit for bit there too. No
//! cross-factor fermionic sign is added.
//!
//! A product of group representations returns one sparse
//! [`crate::lie::LieCgc`], with the last factor fastest on every axis
//! (multiplicity included). When every factor supplies exact channels (Lie
//! groups, tori, U(1), Z_N, nested in any way), each entry is the exact
//! product of the factors' exact entries, rounded once; two or more Lie
//! leaves share one cached composite channel in the Product tier, keyed by
//! the ordered leaves, and one Lie leaf returns its own block. A product with
//! a factor defined outside the crate builds an owned block of the
//! left-to-right `f64` products of its factors' entries.
//!
//! Nonzero F/R/A/B queries check each leaf's storage axes against checked N
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
//! errors). When every factor's dimensions are exact integers, a product's
//! dimension and its derived values ([`FusionCategory::sqrt_dim`] and its
//! inverses) are correctly rounded once from the exact product of the factor
//! dimensions; with a numeric (table) factor they multiply the factor values
//! left to right.
//!
//! Factor fusion iterators must be cloneable. The higher-ranked GAT bound on
//! these implementations currently restricts factors to owned category types;
//! arbitrary borrowed providers are not promised.
//!
//! An outcome's multiplicity is the product of its factor multiplicities. A
//! nonempty product's largest multiplicity, the product of its factors'
//! largest ones, is checked against `u32` when the iterator is built: past it,
//! `fusion` returns [`ProductError::Overflow`] after any factor error. Every
//! category of this crate states its largest multiplicity without iterating;
//! for a `FusionStyle::Generic` factor defined outside the crate, building a
//! nonempty product iterates that factor's channels once more to find it.
use crate::category::exact_dim;
use crate::category::sealed::{
    CgcExact, CgcLeaves, CgcOp, DimAcc, ExactAnswer, ExactBuf, ExactOp, FloatSink, InlineVec,
    LeafBuf, LegRows, Query, SymbolKind, Unit, View,
};
use crate::category::{
    rank_checked, sealed::Embed, sealed::ExactCount, sealed::Ranked, sealed::Validated, unrank,
    BraidingStyle, CgcBlock, Count, EnumerationError, FBlock, FusionCategory, FusionStyle,
    GroupRepresentations, Promote, RBlock, Scalar, Storage, SymbolBlock,
};
use crate::exact::radical::RadicalBlock;
use crate::frcore::MEntry;
use crate::lie::LieCgc;
use crate::rounding::blocks::{check_f_block, check_r_block, round_block};
use crate::rounding::cgc::{compose_channel, composite_len};
use crate::rounding::DimValue;
use malachite_nz::integer::Integer;
use num_complex::Complex64;
use std::{
    any::{Any, TypeId},
    convert::Infallible,
    fmt,
    hash::{Hash, Hasher},
    iter::Fuse,
    sync::Arc,
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
}
impl<C> Product<C> {
    /// Construct a tuple product (2..=12 factors) or a homogeneous vector product (any length).
    ///
    /// A factor that is itself a product stays one immediate factor.
    ///
    /// ```
    /// use fusion_symbols::{fermion::FermionParity, lie::U1, product::Product};
    /// let cat = Product::new((U1, FermionParity, U1));
    /// assert_eq!(cat.factors(), &(U1, FermionParity, U1));
    /// let chain = Product::new(vec![U1; 3]);
    /// assert_eq!(chain.factors().len(), 3);
    /// ```
    ///
    /// Objects are tuples of factor objects, and fusion acts factorwise; an
    /// outcome's multiplicity is the product of its factor multiplicities:
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{U1Irrep, U1}, product::Product};
    /// let cat = Product::new((U1, U1));
    /// let q = |a, b| Ok::<_, fusion_symbols::lie::U1Error>((U1Irrep::new(a)?, U1Irrep::new(b)?));
    /// let x = q(1, -2)?;
    /// assert_eq!(cat.fusion(&x, &x)?.collect::<Vec<_>>(), [(q(2, -4)?, 1)]);
    /// assert_eq!(cat.nsymbol(&x, &x, &q(2, 0)?)?, 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// Tuples outside 2..=12 factors are not categories:
    ///
    /// ```compile_fail
    /// use fusion_symbols::{lie::U1, product::Product};
    /// let _ = Product::new((U1,));
    /// ```
    /// ```compile_fail
    /// use fusion_symbols::{lie::U1, product::Product};
    /// let _ = Product::new((U1, U1, U1, U1, U1, U1, U1, U1, U1, U1, U1, U1, U1));
    /// ```
    pub fn new(factors: C) -> Self
    where
        Self: FusionCategory,
    {
        Self { factors }
    }
    /// The ordered immediate factors.
    pub fn factors(&self) -> &C {
        &self.factors
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
    /// The rounding of an exactly composed block failed the named check
    /// (`"F"`: a finite entry; `"R"`: unitarity).
    Gate {
        /// The symbol whose check failed.
        gate: &'static str,
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
            Self::Gate { gate } => write!(f, "composed product {gate} symbol failed its rounding gate"),
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
    // A leaf block of the float route that disagreed with its leg N, as the
    // error of the immediate factor that holds it.
    fn malformed(sink: &mut FloatSink, factor: usize) -> Result<(), Self> {
        match sink.0.malformed.take() {
            None => Ok(()),
            Some(Malformed {
                expected,
                actual,
                storage_len,
            }) => Err(Self::MalformedSymbolData {
                factor,
                expected,
                actual,
                storage_len,
            }),
        }
    }
}

/// Fusion outcomes of a [`Product`] with factors `C`: every combination of
/// factor outcomes, each with the product of the factor multiplicities.
pub struct ProductFusion<'a, C: sealed::Factors + 'a>(Cartesian<C::Iters<'a>, C::Current<'a>>);

use sealed::Cartesian;

pub(crate) mod sealed {
    /// Cartesian fusion iterator; resets clone initialized iterators, never query factors again.
    /// Each outcome's multiplicity is the product of its factor multiplicities.
    #[derive(Clone, Debug)]
    pub struct Cartesian<I, O> {
        pub(crate) initial: I,
        pub(crate) active: I,
        pub(crate) current: Option<O>,
        // Exact largest multiplicity of the outcomes, checked against `u32` when
        // the iterator was built; 0 for an empty product.
        pub(crate) max: u32,
    }
    /// The iterator state of a product's fusion, per factor container.
    pub trait Factors {
        type Iters<'a>
        where
            Self: 'a;
        type Current<'a>
        where
            Self: 'a;
    }
}

impl<'a, C: sealed::Factors + 'a> ProductFusion<'a, C> {
    pub(crate) fn max_multiplicity(&self) -> u32 {
        self.0.max
    }
    pub(crate) fn into_cartesian(self) -> Cartesian<C::Iters<'a>, C::Current<'a>> {
        self.0
    }
}
impl<'a, C: sealed::Factors + 'a> Clone for ProductFusion<'a, C>
where
    Cartesian<C::Iters<'a>, C::Current<'a>>: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<'a, C: sealed::Factors + 'a> core::fmt::Debug for ProductFusion<'a, C>
where
    Cartesian<C::Iters<'a>, C::Current<'a>>: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("ProductFusion").field(&self.0).finish()
    }
}
impl<'a, C: sealed::Factors + 'a> Iterator for ProductFusion<'a, C>
where
    Cartesian<C::Iters<'a>, C::Current<'a>>: Iterator,
{
    type Item = <Cartesian<C::Iters<'a>, C::Current<'a>> as Iterator>::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}
impl<'a, C: sealed::Factors + 'a> DoubleEndedIterator for ProductFusion<'a, C>
where
    Cartesian<C::Iters<'a>, C::Current<'a>>: DoubleEndedIterator,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth_back(n)
    }
}
impl<'a, C: sealed::Factors + 'a> ExactSizeIterator for ProductFusion<'a, C> where
    Cartesian<C::Iters<'a>, C::Current<'a>>: ExactSizeIterator
{
}

impl<I, O> Cartesian<I, O> {
    pub(crate) fn max_multiplicity(&self) -> u32 {
        self.max
    }
}

// The largest multiplicity of a factor that cannot state it: one pass over a
// clone, made only for a nonempty product (the exact bound is needed by every
// enclosing product, not only for this one's overflow).
fn max_by_pass<I: Iterator<Item = (O, u32)> + Clone, O>(iter: &I) -> u32 {
    iter.clone().map(|(_, n)| n).max().unwrap_or(0)
}
const BOUNDED: &str = "bounded by the constructor's product of maxima";

// Read all multiplicities before multiplying: a zero factor fixes the product
// at zero, so it beats both a nonzero prefix overflow and any factor failure.
// Otherwise the first factor failure wins, then overflow. Factor results are
// pushed one at a time, so a tuple never materializes an array of results.
struct Multiplicity<E> {
    product: Option<u32>,
    error: Option<E>,
}
impl<E> Multiplicity<E> {
    fn new() -> Self {
        Self {
            product: Some(1),
            error: None,
        }
    }
    #[inline]
    fn push(&mut self, n: Result<u32, E>) {
        match n {
            Ok(0) => self.product = Some(0),
            Ok(n) => self.product = self.product.and_then(|p| p.checked_mul(n)),
            Err(e) => {
                self.error.get_or_insert(e);
            }
        }
    }
    // The overflow error is built only when it is returned.
    fn finish(self, overflow: impl FnOnce() -> E) -> Result<u32, E> {
        match (self.error, self.product) {
            (_, Some(0)) => Ok(0),
            (Some(e), _) => Err(e),
            (None, Some(n)) => Ok(n),
            (None, None) => Err(overflow()),
        }
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

// A left-to-right Kronecker product: one scalar while every block so far has
// one element, then a buffer of the final length whose row-major prefix holds
// the product so far. Every step multiplies exactly as a fresh per-element fold
// `left(l) * right(r)` would, so values do not depend on where the buffer starts.
enum Partial<S, const R: usize> {
    One(S),
    Buf([usize; R], Arc<[S]>),
}

// Zero-filled because in-place expansion needs initialized storage; the
// alternatives are uninit memory with `unsafe`, or an out-of-place fold that
// multiplies again at every level.
fn zeros<S: Scalar>(len: usize) -> Arc<[S]> {
    std::iter::repeat_n(S::zero(), len).collect()
}

fn owned<S>(buf: &mut Arc<[S]>) -> &mut [S] {
    Arc::get_mut(buf).expect("a partial product buffer is never shared")
}

impl<S: Scalar, const R: usize> Partial<S, R> {
    // Not `One(S::one())`: (1 + 0i) z can flip the sign of a zero part of z.
    fn new<B: SymbolBlock<R, Scalar = S>>(block: &B, len: usize) -> Self {
        match block.as_slice() {
            [x] => Self::One(*x),
            data => {
                let mut buf = zeros(len);
                owned(&mut buf)[..data.len()].copy_from_slice(data);
                Self::Buf(block.dims(), buf)
            }
        }
    }

    // `embed(l)` then `multiply(·, r)` is `Embed::left(l) * Embed::right(r)`.
    // `Scalar` is sealed, so a step either keeps the scalar type and its buffer
    // (embed is the identity) or promotes f64 to Complex64 once: the real prefix
    // then leaves its transient f64 buffer for a complex one. The TypeId test
    // folds at compile time.
    fn times<B: SymbolBlock<R>, T: Scalar>(
        self,
        block: &B,
        len: usize,
        embed: impl Fn(S) -> T,
        multiply: impl Fn(T, B::Scalar) -> T,
    ) -> Partial<T, R> {
        let right = block.as_slice();
        let (left, mut buf) = match (self, right) {
            (Self::One(l), &[r]) => return Partial::One(multiply(embed(l), r)),
            (Self::One(l), _) => {
                let mut buf = zeros(len);
                owned(&mut buf)[0] = embed(l);
                ([1; R], buf)
            }
            (Self::Buf(dims, buf), _) => {
                let mut slot = Some(buf);
                let same = (&mut slot as &mut dyn Any)
                    .downcast_mut::<Option<Arc<[T]>>>()
                    .and_then(Option::take);
                let buf = same.unwrap_or_else(|| {
                    let mut promoted = zeros(len);
                    let filled = dims.iter().product();
                    let out = owned(&mut promoted);
                    for (o, &l) in out.iter_mut().zip(slot.iter().flat_map(|b| &b[..filled])) {
                        *o = embed(l);
                    }
                    promoted
                });
                (dims, buf)
            }
        };
        let dims = kron_in_place(owned(&mut buf), left, block.dims(), right, multiply);
        Partial::Buf(dims, buf)
    }

    fn finish(self) -> ([usize; R], Storage<S>) {
        match self {
            Self::One(x) => ([1; R], Storage::One(x)),
            Self::Buf(dims, buf) => (dims, Storage::Shared(buf)),
        }
    }
}

// `out` starts with the row-major `left` block and ends holding its row-major
// Kronecker product with `right` (each axis index is `l * right + r`). Written
// back to front: output `i` reads left offset `l(i) <= i`, and only positions
// after `i` have been overwritten. The (l, r) digits per axis count down
// instead of decoding every `i` with div/mod.
fn kron_in_place<S: Copy, B: Copy, const R: usize>(
    out: &mut [S],
    left: [usize; R],
    right: [usize; R],
    data: &[B],
    multiply: impl Fn(S, B) -> S,
) -> [usize; R] {
    let (mut l_stride, mut r_stride) = ([1; R], [1; R]);
    for a in (0..R.saturating_sub(1)).rev() {
        l_stride[a] = l_stride[a + 1] * left[a + 1];
        r_stride[a] = r_stride[a + 1] * right[a + 1];
    }
    let (mut l_digit, mut r_digit) = (left.map(|n| n - 1), right.map(|n| n - 1));
    let (mut l, mut r) = (
        left.iter().product::<usize>() - 1,
        right.iter().product::<usize>() - 1,
    );
    let dims = std::array::from_fn(|a| left[a] * right[a]);
    for i in (0..dims.iter().product()).rev() {
        out[i] = multiply(out[l], data[r]);
        for a in (0..R).rev() {
            if r_digit[a] > 0 {
                r_digit[a] -= 1;
                r -= r_stride[a];
                break;
            }
            r_digit[a] = right[a] - 1;
            r += r_digit[a] * r_stride[a];
            if l_digit[a] > 0 {
                l_digit[a] -= 1;
                l -= l_stride[a];
                break;
            }
            l_digit[a] = left[a] - 1;
            l += l_digit[a] * l_stride[a];
        }
    }
    dims
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

fn product_unrank(factors: &[ExactCount], index: u64) -> Result<Vec<Integer>, EnumerationError> {
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
    unrank(&bounds, &Integer::from(index)).ok_or(EnumerationError::OutOfRange)
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

/// A product's CGC block: when every factor answers the exact protocol, the
/// scalar of a unit product, the one Radical leaf's own block, or the
/// composite channel of two or more leaves, composed exactly and rounded
/// once ([`compose_channel`]) and shared through the Product tier; else
/// `float`. A warm composite is one tier hit and an `Arc` clone.
fn product_cgc<C: GroupRepresentations>(
    cat: &C,
    q: [&C::Object; 3],
    overflow: impl Fn() -> C::Error,
    float: impl FnOnce() -> Result<LieCgc, C::Error>,
) -> Result<LieCgc, C::Error> {
    let t = Validated(());
    let mut leaves = CgcLeaves::new();
    if !cat.__fusion_symbols_exact_cgc(q, CgcOp::Describe(&mut leaves), t)? {
        return float();
    }
    let dims = leaves.dims.ok_or_else(&overflow)?;
    if leaves.absent {
        return Ok(LieCgc::scalar([dims[0], dims[1], dims[2], 0]));
    }
    match leaves.leaves.0.len() {
        0 => Ok(LieCgc::scalar([1; 4])),
        1 => {
            let mut view = None;
            cat.__fusion_symbols_exact_cgc(q, CgcOp::View(&mut view), t)?;
            Ok(view.expect("the Radical leaf answers its view"))
        }
        n => {
            let channel = crate::cache::composed_cgc(&leaves.leaves, || {
                let mut exact = CgcExact(Vec::with_capacity(n));
                cat.__fusion_symbols_exact_cgc(q, CgcOp::Fetch(&mut exact), t)?;
                compose_channel(&exact.0)
                    .map(std::sync::Arc::new)
                    .ok_or_else(&overflow)
            })?;
            Ok(LieCgc::sparse(dims, channel))
        }
    }
}

/// A factor block's dimensions and present entries
/// ([`CgcBlock::__fusion_symbols_stored`]): a present entry is kept even
/// when its rounded value is `±0.0`.
// ponytail: a block outside `LieCgc` is read by a dense scan of
// `d_a d_b d_c N` reads and reports every one, only on the float path; its
// own sparse override would remove both costs if such blocks grow large.
fn stored(block: &impl CgcBlock) -> ([usize; 4], Vec<MEntry>) {
    let mut out = Vec::new();
    block.__fusion_symbols_stored(&mut |at, value| {
        // Indices below a combined axis checked to fit `u32`.
        let [m1, m2, m3, mu] = at.map(|i| i as u32);
        out.push(MEntry {
            m1,
            m2,
            m3,
            mu,
            value,
        });
    });
    (block.dims(), out)
}

/// The float block of a product with a factor outside the exact protocol:
/// each entry the left-to-right `f64` product of the factors' present entries, last
/// factor fastest on every axis; `None` when an axis overflows (or, for a
/// nonempty block, exceeds `u32`) or the entries are not allocatable
/// ([`composite_len`]).
fn float_cgc(factors: Vec<([usize; 4], Vec<MEntry>)>) -> Option<LieCgc> {
    let dims = cgc_dims(factors.iter().map(|f| f.0))?;
    if dims[3] == 0 {
        return Some(LieCgc::scalar(dims));
    }
    if dims.iter().any(|&d| d > u32::MAX as usize) {
        return None;
    }
    let mut entries = Vec::with_capacity(composite_len(factors.iter().map(|f| f.1.len()))?);
    let mut pick = vec![0usize; factors.len()];
    'odometer: while factors.iter().all(|f| !f.1.is_empty()) {
        let mut at = [0usize; 4];
        let mut value = None;
        for ((d, list), &i) in factors.iter().zip(&pick) {
            let e = list[i];
            for (axis, m) in [e.m1, e.m2, e.m3, e.mu].into_iter().enumerate() {
                at[axis] = at[axis] * d[axis] + m as usize;
            }
            value = Some(value.map_or(e.value, |v: f64| v * e.value));
        }
        let [m1, m2, m3, mu] = at.map(|x| x as u32);
        let value = value.unwrap_or(1.0);
        entries.push(MEntry {
            m1,
            m2,
            m3,
            mu,
            value,
        });
        for (i, (_, list)) in pick.iter_mut().zip(&factors).rev() {
            *i += 1;
            if *i < list.len() {
                continue 'odometer;
            }
            *i = 0;
        }
        break;
    }
    entries.sort_unstable_by_key(|e| (e.mu, e.m3, e.m1, e.m2));
    let channel = crate::lie::general::cgc::CgcChannel {
        mult: dims[3],
        entries: entries.into_boxed_slice(),
    };
    Some(LieCgc::sparse(
        [dims[0], dims[1], dims[2]],
        std::sync::Arc::new(channel),
    ))
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

// The N of each leg of a symbol block, in leg order: the axes it is checked against.
fn f_legs<C: FusionCategory>(
    cat: &C,
    [a, b, c, d, e, f]: [&C::Object; 6],
    t: Validated,
) -> [Result<u32, C::Error>; 4] {
    let n = |x, y, z| cat.__fusion_symbols_validated_nsymbol(x, y, z, t);
    [n(a, b, e), n(e, c, d), n(b, c, f), n(a, f, d)]
}
fn r_legs<C: FusionCategory>(
    cat: &C,
    [a, b, c]: [&C::Object; 3],
    t: Validated,
) -> [Result<u32, C::Error>; 2] {
    let n = |x, y, z| cat.__fusion_symbols_validated_nsymbol(x, y, z, t);
    [n(a, b, c), n(b, a, c)]
}
fn a_legs<C: FusionCategory>(
    cat: &C,
    [a, b, c]: [&C::Object; 3],
    t: Validated,
) -> [Result<u32, C::Error>; 2] {
    let dual = cat.__fusion_symbols_validated_dual(a, t);
    [
        cat.__fusion_symbols_validated_nsymbol(a, b, c, t),
        cat.__fusion_symbols_validated_nsymbol(&dual, c, b, t),
    ]
}
fn b_legs<C: FusionCategory>(
    cat: &C,
    [a, b, c]: [&C::Object; 3],
    t: Validated,
) -> [Result<u32, C::Error>; 2] {
    let dual = cat.__fusion_symbols_validated_dual(b, t);
    [
        cat.__fusion_symbols_validated_nsymbol(a, b, c, t),
        cat.__fusion_symbols_validated_nsymbol(c, &dual, a, t),
    ]
}
// A block shape from its legs' N: the first failing leg's error, then overflow.
fn leg_shape<S, E, const R: usize>(
    legs: [Result<u32, E>; R],
    overflow: impl FnOnce() -> E,
) -> Result<[usize; R], E> {
    let mut n = [0; R];
    for (n, leg) in n.iter_mut().zip(legs) {
        *n = leg?;
    }
    shape::<S, R>(n).ok_or_else(overflow)
}

// The leg N of any query, padded with ones to four legs.
fn query_legs<C: FusionCategory>(
    cat: &C,
    q: Query<'_, C::Object>,
    t: Validated,
) -> [Result<u32, C::Error>; 4] {
    let pad = |[x, y]: [Result<u32, C::Error>; 2]| [x, y, Ok(1), Ok(1)];
    match q {
        Query::F(x) => f_legs(cat, x, t),
        Query::R(x) => pad(r_legs(cat, x, t)),
        Query::A(x) => pad(a_legs(cat, x, t)),
        Query::B(x) => pad(b_legs(cat, x, t)),
    }
}

// The float route of a leaf (any category but a product): its leg N, recorded
// depth-first for its block check.
pub(crate) fn leaf_legs<C: FusionCategory>(
    cat: &C,
    q: Query<'_, C::Object>,
    rows: &mut LegRows,
    t: Validated,
) -> [Result<u32, C::Error>; 4] {
    let legs = query_legs(cat, q, t);
    rows.push(legs.each_ref().map(|n| *n.as_ref().unwrap_or(&0)));
    legs
}

// The leaf's public block, checked against its recorded row and folded.
pub(crate) fn leaf_block<C: FusionCategory>(
    cat: &C,
    q: Query<'_, C::Object>,
    rows: &mut LegRows,
    sink: &mut FloatSink,
) -> Result<(), C::Error> {
    let [n0, n1, n2, n3] = rows.next();
    let state = &mut sink.0;
    match q {
        Query::F([a, b, c, d, e, f]) => {
            state.push_f(&cat.f_symbol(a, b, c, d, e, f)?, [n0, n1, n2, n3])
        }
        Query::R([a, b, c]) => state.push_r(&cat.r_symbol(a, b, c)?, [n0, n1]),
        Query::A([a, b, c]) => state.push_r(&cat.a_symbol(a, b, c)?, [n0, n1]),
        Query::B([a, b, c]) => state.push_r(&cat.b_symbol(a, b, c)?, [n0, n1]),
    }
    Ok(())
}

fn fold_legs<E, F>(
    legs: &mut [Multiplicity<E>; 4],
    row: [Result<u32, F>; 4],
    wrap: impl Fn(F) -> E,
) {
    for (leg, n) in legs.iter_mut().zip(row) {
        leg.push(n.map_err(&wrap));
    }
}

pub(crate) struct Malformed {
    expected: Vec<usize>,
    actual: Vec<usize>,
    storage_len: usize,
}

// `Scalar` is sealed to `f64` and `Complex64`, so these tests fold at compile
// time.
fn is_real<S: Scalar>() -> bool {
    TypeId::of::<S>() == TypeId::of::<f64>()
}
fn real<S: Scalar>(x: S) -> f64 {
    cast(x).expect("a real block")
}
// `Embed::right` and `Embed::left` of a real into the complex scalar.
fn widen<S: Scalar>(x: S) -> Complex64 {
    cast::<f64, S>(x).map_or_else(|x| cast(x).expect("Scalar is sealed"), Complex64::from)
}

// The left-to-right Kronecker product of every leaf block of one query, with
// the scalar promoted to complex at the first complex leaf, as `Promote` does.
enum FloatFold<const R: usize> {
    Empty,
    Real(Partial<f64, R>),
    Complex(Partial<Complex64, R>),
}

impl<const R: usize> FloatFold<R> {
    fn push<B: SymbolBlock<R>>(self, block: &B, len: usize) -> Self {
        match self {
            Self::Empty => match cast(Partial::new(block, len)) {
                Ok(real) => Self::Real(real),
                Err(p) => {
                    Self::Complex(cast(p).unwrap_or_else(|_| unreachable!("Scalar is sealed")))
                }
            },
            Self::Real(p) if is_real::<B::Scalar>() => {
                Self::Real(p.times(block, len, |l| l, |l, r| l * real(r)))
            }
            Self::Real(p) => {
                Self::Complex(p.times(block, len, Complex64::from, |l, r| l * widen(r)))
            }
            Self::Complex(p) => Self::Complex(p.times(block, len, |l| l, |l, r| l * widen(r))),
        }
    }

    // Leaf scalars promote to the product's scalar, so `T` is the fold's.
    fn finish<T: Scalar>(self) -> ([usize; R], Storage<T>) {
        let (dims, data) = match self {
            Self::Empty => return ([1; R], Storage::One(T::one())),
            Self::Real(p) => cast(p.finish()).map_err(|_| ()),
            Self::Complex(p) => cast(p.finish()).map_err(|_| ()),
        }
        .expect("the fold has the product's scalar");
        (dims, data)
    }
}

pub(crate) struct FloatState {
    len: usize,
    f: FloatFold<4>,
    r: FloatFold<2>,
    malformed: Option<Malformed>,
}

impl FloatState {
    fn push_f<B: SymbolBlock<4>>(&mut self, block: &B, n: [u32; 4]) {
        if self.check(block, n) {
            self.f = std::mem::replace(&mut self.f, FloatFold::Empty).push(block, self.len);
        }
    }
    fn push_r<B: SymbolBlock<2>>(&mut self, block: &B, n: [u32; 2]) {
        if self.check(block, n) {
            self.r = std::mem::replace(&mut self.r, FloatFold::Empty).push(block, self.len);
        }
    }
    // Every leaf N is positive here, and the leaf block length divides the
    // product's, so the expected shape is representable.
    fn check<B: SymbolBlock<R>, const R: usize>(&mut self, block: &B, n: [u32; R]) -> bool {
        let expected = n.map(|n| n as usize);
        let (actual, storage_len) = (block.dims(), block.as_slice().len());
        if actual == expected && storage_len == expected.iter().product::<usize>() {
            return true;
        }
        self.malformed = Some(Malformed {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
            storage_len,
        });
        false
    }
}

// The float route, taken when a factor has no exact value: every leaf of the
// query, depth-first through nested products, gives its leg N once (the
// product legs fold them as `nsymbol` does, so a zero beats a failure and
// overflow), then its public block once, checked against its N and folded
// into one buffer of the final length. Each leaf is asked once per query,
// whatever the nesting; the value is the flat product's.
fn float_route<P: FusionCategory, O, const R: usize>(
    p: &P,
    q: Query<'_, P::Object>,
    t: Validated,
    overflow: impl Fn() -> P::Error,
    fold: impl FnOnce(FloatState) -> FloatFold<R>,
    make: impl FnOnce([usize; R], Storage<P::Scalar>) -> O,
) -> Result<O, P::Error> {
    let mut rows = LegRows::default();
    let legs = p.__fusion_symbols_float_legs(q, &mut rows, t);
    let dims = leg_shape::<P::Scalar, _, 4>(legs, overflow)?;
    let dims: [usize; R] = std::array::from_fn(|i| dims[i]);
    if dims.contains(&0) {
        return Ok(make(dims, Storage::Empty));
    }
    let len = dims.iter().product();
    let state = FloatState {
        len,
        f: FloatFold::Empty,
        r: FloatFold::Empty,
        malformed: None,
    };
    let mut sink = FloatSink(state);
    p.__fusion_symbols_float_blocks(q, &mut rows, &mut sink, t)?;
    let (dims, data) = fold(sink.0).finish();
    Ok(make(dims, data))
}

// One product level's exact hook, folded factor by factor: the class of a
// kind (unit only if every factor is; unavailable if any factor is; else the
// summed Radical leaves), the aggregate unit of `Describe` with every leaf
// appended depth-first to the caller's one buffer, or `Fetch`/`View`
// forwarded to the Radical factors only.
struct Fold {
    leaves: usize,
    unit: Unit,
    unavailable: bool,
}

impl Fold {
    fn new() -> Self {
        Self {
            leaves: 0,
            unit: Unit::PlusOne,
            unavailable: false,
        }
    }

    fn finish<O>(self, op: &ExactOp<'_, '_, O>) -> ExactAnswer {
        match op {
            _ if self.unavailable => ExactAnswer::Unavailable,
            ExactOp::Class(_) if self.leaves == 0 => ExactAnswer::UnitClass,
            ExactOp::Class(_) => ExactAnswer::RadicalClass {
                leaves: self.leaves,
            },
            ExactOp::Describe(..) => ExactAnswer::Unit(self.unit),
            ExactOp::Fetch(..) | ExactOp::View(..) => ExactAnswer::Done,
        }
    }

    fn step<'q, O, C: FusionCategory, E>(
        &mut self,
        op: &mut ExactOp<'q, '_, O>,
        cat: &C,
        proj: impl Fn(&'q O) -> &'q C::Object,
        t: Validated,
        wrap: impl Fn(C::Error) -> E,
    ) -> Result<(), E>
    where
        C::Object: 'q,
    {
        let hook = |op| cat.__fusion_symbols_exact_symbol(op, t).map_err(&wrap);
        let radical = |kind| {
            hook(ExactOp::Class(kind)).map(|c| matches!(c, ExactAnswer::RadicalClass { .. }))
        };
        if self.unavailable {
            return Ok(());
        }
        match op {
            ExactOp::Class(kind) => match hook(ExactOp::Class(*kind))? {
                ExactAnswer::UnitClass => {}
                ExactAnswer::RadicalClass { leaves } => self.leaves += leaves,
                _ => self.unavailable = true,
            },
            ExactOp::Describe(q, buf) => match hook(ExactOp::Describe(q.map(proj), buf))? {
                ExactAnswer::Unit(u) => self.unit = self.unit.times(u),
                _ => self.unavailable = true,
            },
            ExactOp::Fetch(q, buf) => {
                if radical(q.kind())? {
                    hook(ExactOp::Fetch(q.map(proj), buf))?;
                }
            }
            ExactOp::View(q, out) => {
                if radical(q.kind())? {
                    hook(ExactOp::View(q.map(proj), out))?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    /// Runs between a composite's leaf fetch and its publication.
    static AFTER_FETCH: std::cell::Cell<Option<fn()>> = const { std::cell::Cell::new(None) };
    /// Makes the next composite rounding non-finite, for the gate tests.
    static CORRUPT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// The outer boundary's exact route (design §e.2): `None` takes the float
// path, which is the route of an unavailable factor and of a zero aggregate
// unit. No leaf: the unit itself. One Radical leaf: its own view. More: the
// composite view, whose miss first runs `precheck` (today's multiplicities,
// overflow and zero channels, in today's order), then composes the leaves'
// exact blocks and rounds once. The unit and any promotion are applied by
// the caller, after publication.
fn exact_route<P: FusionCategory>(
    p: &P,
    q: Query<'_, P::Object>,
    t: Validated,
    precheck: impl FnOnce() -> Result<Option<View>, P::Error>,
    overflow: impl FnOnce() -> P::Error,
    gate: impl FnOnce(&'static str) -> P::Error,
) -> Result<Option<(Unit, View)>, P::Error> {
    let kind = q.kind();
    let leaves = match p.__fusion_symbols_exact_symbol(ExactOp::Class(kind), t)? {
        ExactAnswer::UnitClass => 0,
        ExactAnswer::RadicalClass { leaves } => leaves,
        _ => return Ok(None),
    };
    let mut buf = LeafBuf(InlineVec::with_capacity(leaves));
    let unit = match p.__fusion_symbols_exact_symbol(ExactOp::Describe(q, &mut buf), t)? {
        ExactAnswer::Unit(u) if u != Unit::Zero => u,
        _ => return Ok(None),
    };
    let view = match leaves {
        0 if kind == SymbolKind::F => View::F(FBlock::one(1.0)),
        0 => View::R(RBlock::one(1.0)),
        1 => {
            let mut out = None;
            p.__fusion_symbols_exact_symbol(ExactOp::View(q, &mut out), t)?;
            out.expect("a one-leaf query views its leaf")
        }
        _ => crate::cache::composed(kind, &buf, || {
            if let Some(zero) = precheck()? {
                return Ok(zero);
            }
            let mut exact = ExactBuf::with_capacity(leaves);
            p.__fusion_symbols_exact_symbol(ExactOp::Fetch(q, &mut exact), t)?;
            #[cfg(test)]
            if let Some(f) = AFTER_FETCH.with(std::cell::Cell::get) {
                f();
            }
            compose(kind, &exact).ok_or_else(overflow)?.map_err(gate)
        })?,
    };
    Ok(Some((unit, view)))
}

// The leaves' Kronecker product in leaf order, rounded once and gated (7 for
// F, 6 for R; A and B as their leaf views). `None` on a dimension overflow.
fn compose(kind: SymbolKind, exact: &ExactBuf) -> Option<Result<View, &'static str>> {
    fn kron<const R: usize>(
        blocks: &InlineVec<Arc<RadicalBlock<R>>, 4>,
    ) -> Option<RadicalBlock<R>> {
        let mut blocks = blocks.iter();
        let first = RadicalBlock::clone(blocks.next()?);
        blocks.try_fold(first, |acc, b| acc.kron(b))
    }
    Some(if kind == SymbolKind::F {
        let x = kron(&exact.f)?;
        let data = round_block(&x);
        #[cfg(test)]
        let data = corrupted(data);
        let view = FBlock::from_vec(x.dims(), data);
        check_f_block(&view)
            .map(|()| View::F(view))
            .map_err(|_| "F")
    } else {
        let x = kron(&exact.r2)?;
        let data = round_block(&x);
        #[cfg(test)]
        let data = corrupted(data);
        let view = RBlock::from_vec(x.dims()[0], data);
        match kind {
            SymbolKind::R => check_r_block(&view)
                .map(|()| View::R(view))
                .map_err(|_| "R"),
            _ => Ok(View::R(view)),
        }
    })
}

#[cfg(test)]
fn corrupted(mut x: Vec<f64>) -> Vec<f64> {
    if CORRUPT.with(|c| c.replace(false)) {
        x[0] = f64::NAN;
    }
    x
}

// `Some(x)` when `S` is `T`: a view passes through without a copy.
fn cast<S: 'static, T: 'static>(x: T) -> Result<S, T> {
    let mut slot = Some(x);
    match (&mut slot as &mut dyn Any)
        .downcast_mut::<Option<S>>()
        .and_then(Option::take)
    {
        Some(same) => Ok(same),
        None => Err(slot.expect("a failed downcast leaves the value")),
    }
}

// The unit applied to the stored real entries, embedded as `(x, +0.0)`.
fn remap<T: Copy, S: Scalar>(data: &[T], f: impl Fn(T) -> S) -> Storage<S> {
    match data {
        [] => Storage::Empty,
        &[x] => Storage::One(f(x)),
        _ => Storage::Shared(data.iter().map(|&x| f(x)).collect()),
    }
}

fn f_out<S: Scalar>(unit: Unit, view: View) -> FBlock<S> {
    let View::F(view) = view else {
        unreachable!("an F query views an F block")
    };
    match cast::<FBlock<S>, _>(view) {
        Ok(same) if unit == Unit::PlusOne => same,
        Ok(same) => FBlock::from_storage(same.dims(), remap(same.as_slice(), |x| unit.apply(x))),
        Err(view) => FBlock::from_storage(
            view.dims(),
            remap(view.as_slice(), |x| unit.apply(S::from_f64(x))),
        ),
    }
}

fn r_out<S: Scalar>(unit: Unit, view: View) -> RBlock<S> {
    let View::R(view) = view else {
        unreachable!("an R, A or B query views a square block")
    };
    match cast::<RBlock<S>, _>(view) {
        Ok(same) if unit == Unit::PlusOne => same,
        Ok(same) => RBlock::from_storage(same.dims()[0], remap(same.as_slice(), |x| unit.apply(x))),
        Err(view) => RBlock::from_storage(
            view.dims()[0],
            remap(view.as_slice(), |x| unit.apply(S::from_f64(x))),
        ),
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
                    Count::Finite(n) => u64::try_from(&n).map(Some).map_err(|_| EnumerationError::Overflow),
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
                u64::try_from(&index).map_err(|_| EnumerationError::Overflow)
            }
            /// The first failing factor's error, at its immediate tuple position or
            /// vector index; otherwise the left-to-right product of factor phases.
            fn frobenius_schur_phase(&$this, a: &Self::Object) -> Result<Self::Scalar, Self::Error> {
                validate($this, &[a]);
                $this.__fusion_symbols_validated_frobenius_schur_phase(a, Validated(()))
            }
            fn frobenius_schur_indicator(&$this, a: &Self::Object) -> Self::Scalar {
                validate($this, &[a]);
                $this.__fusion_symbols_validated_frobenius_schur_indicator(a, Validated(()))
            }
            fn twist(&$this, a: &Self::Object) -> Result<Self::Scalar, Self::Error> {
                validate($this, &[a]);
                if let Some(factor) = ($braidings).into_iter().position(|s| s == BraidingStyle::None) {
                    return Err(ProductError::NoBraiding { factor });
                }
                $this.__fusion_symbols_validated_twist(a, Validated(()))
            }
            fn fusion<'a>(&'a $this, a: &Self::Object, b: &Self::Object) -> Result<Self::Fusion<'a>, Self::Error> {
                validate($this, &[a, b]);
                $this.__fusion_symbols_validated_fusion(a, b, Validated(()))
            }
            /// A zero factor multiplicity beats both factor failures and overflow.
            /// Otherwise the first factor failure is returned at its immediate tuple
            /// position or vector index, then [`ProductError::Overflow`] if the
            /// product multiplicity exceeds `u32`. Foreign objects panic.
            fn nsymbol(&$this, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Result<u32, Self::Error> {
                validate($this, &[a, b, c]);
                $this.__fusion_symbols_validated_nsymbol(a, b, c, Validated(()))
            }
            fn f_symbol(&$this, a: &Self::Object, b: &Self::Object, c: &Self::Object, d: &Self::Object, e: &Self::Object, f: &Self::Object) -> Result<Self::FSymbol, Self::Error> {
                validate($this, &[a, b, c, d, e, f]);
                let t = Validated(());
                let dims = || leg_shape::<Self::Scalar, _, 4>(f_legs($this, [a, b, c, d, e, f], t), || ProductError::Overflow);
                let zero = |dims: [usize; 4]| dims.contains(&0).then(|| View::F(FBlock::from_storage(dims, Storage::Empty)));
                if let Some((u, view)) = exact_route($this, Query::F([a, b, c, d, e, f]), t, || dims().map(zero), || ProductError::Overflow, |gate| ProductError::Gate { gate })? {
                    return Ok(f_out(u, view));
                }
                float_route($this, Query::F([a, b, c, d, e, f]), t, || ProductError::Overflow, |s| s.f, FBlock::from_storage)
            }
            fn r_symbol(&$this, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Result<Self::RSymbol, Self::Error> {
                validate($this, &[a, b, c]);
                if let Some(factor) = ($braidings).into_iter().position(|s| s == BraidingStyle::None) {
                    return Err(ProductError::NoBraiding { factor });
                }
                let t = Validated(());
                let dims = || leg_shape::<Self::Scalar, _, 2>(r_legs($this, [a, b, c], t), || ProductError::Overflow);
                let zero = |dims: [usize; 2]| dims.contains(&0).then(|| View::R(RBlock::from_storage(dims[0], Storage::Empty)));
                if let Some((u, view)) = exact_route($this, Query::R([a, b, c]), t, || dims().map(zero), || ProductError::Overflow, |gate| ProductError::Gate { gate })? {
                    return Ok(r_out(u, view));
                }
                float_route($this, Query::R([a, b, c]), t, || ProductError::Overflow, |s| s.r, rblock)
            }
            // Tuple and vector products use factor Kronecker products as for F/R
            // (TensorKitSectors `product.jl::Asymbol`); braiding is not needed.
            fn a_symbol(&$this, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Result<RBlock<Self::Scalar>, Self::Error> {
                validate($this, &[a, b, c]);
                let t = Validated(());
                let dims = || leg_shape::<Self::Scalar, _, 2>(a_legs($this, [a, b, c], t), || ProductError::Overflow);
                let zero = |dims: [usize; 2]| dims.contains(&0).then(|| View::R(RBlock::from_storage(dims[0], Storage::Empty)));
                if let Some((u, view)) = exact_route($this, Query::A([a, b, c]), t, || dims().map(zero), || ProductError::Overflow, |gate| ProductError::Gate { gate })? {
                    return Ok(r_out(u, view));
                }
                float_route($this, Query::A([a, b, c]), t, || ProductError::Overflow, |s| s.r, rblock)
            }
            // Same composition as `product.jl::Bsymbol`, for tuple and vector factors.
            fn b_symbol(&$this, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Result<RBlock<Self::Scalar>, Self::Error> {
                validate($this, &[a, b, c]);
                let t = Validated(());
                let dims = || leg_shape::<Self::Scalar, _, 2>(b_legs($this, [a, b, c], t), || ProductError::Overflow);
                let zero = |dims: [usize; 2]| dims.contains(&0).then(|| View::R(RBlock::from_storage(dims[0], Storage::Empty)));
                if let Some((u, view)) = exact_route($this, Query::B([a, b, c]), t, || dims().map(zero), || ProductError::Overflow, |gate| ProductError::Gate { gate })? {
                    return Ok(r_out(u, view));
                }
                float_route($this, Query::B([a, b, c]), t, || ProductError::Overflow, |s| s.r, rblock)
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
            type Fusion<'a> = ProductFusion<'a, ($C0, $($C,)+)> where Self: 'a;

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
            // the Kronecker product of factor F:
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
                exact_dim(self, a, DimValue::Dim).unwrap_or_else(|| {
                    let value = self.factors.0.dim(&a.0); $(let value = value * self.factors.$i.dim(&a.$i);)+ value
                })
            }
            fn sqrt_dim(&self, a: &Self::Object) -> f64 {
                exact_dim(self, a, DimValue::Sqrt).unwrap_or_else(|| {
                    let value = self.factors.0.sqrt_dim(&a.0); $(let value = value * self.factors.$i.sqrt_dim(&a.$i);)+ value
                })
            }
            fn inv_sqrt_dim(&self, a: &Self::Object) -> f64 {
                exact_dim(self, a, DimValue::InvSqrt).unwrap_or_else(|| {
                    let value = self.factors.0.inv_sqrt_dim(&a.0); $(let value = value * self.factors.$i.inv_sqrt_dim(&a.$i);)+ value
                })
            }
            fn inv_dim(&self, a: &Self::Object) -> f64 {
                exact_dim(self, a, DimValue::Inv).unwrap_or_else(|| {
                    let value = self.factors.0.inv_dim(&a.0); $(let value = value * self.factors.$i.inv_dim(&a.$i);)+ value
                })
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
            fn __fusion_symbols_validated_frobenius_schur_phase(&self, a: &Self::Object, t: Validated) -> Result<Self::Scalar, Self::Error> {
                let value = self.factors.0.__fusion_symbols_validated_frobenius_schur_phase(&a.0, t).map_err(ProductError::Factor0)?;
                $(let value = <$Prev as Embed<$C::Scalar>>::left(value) * <$Prev as Embed<$C::Scalar>>::right(self.factors.$i.__fusion_symbols_validated_frobenius_schur_phase(&a.$i, t).map_err(ProductError::$Err)?);)+
                Ok(value)
            }
            fn __fusion_symbols_validated_frobenius_schur_indicator(&self, a: &Self::Object, t: Validated) -> Self::Scalar {
                let value = self.factors.0.__fusion_symbols_validated_frobenius_schur_indicator(&a.0, t);
                $(let value = <$Prev as Embed<$C::Scalar>>::left(value) * <$Prev as Embed<$C::Scalar>>::right(self.factors.$i.__fusion_symbols_validated_frobenius_schur_indicator(&a.$i, t));)+ value
            }
            // The caller has checked that every factor braids.
            fn __fusion_symbols_validated_twist(&self, a: &Self::Object, t: Validated) -> Result<Self::Scalar, Self::Error> {
                let value = self.factors.0.__fusion_symbols_validated_twist(&a.0, t).map_err(ProductError::Factor0)?;
                $(let value = <$Prev as Embed<$C::Scalar>>::left(value) * <$Prev as Embed<$C::Scalar>>::right(self.factors.$i.__fusion_symbols_validated_twist(&a.$i, t).map_err(ProductError::$Err)?);)+
                Ok(value)
            }
            fn __fusion_symbols_validated_fusion<'a>(&'a self, a: &Self::Object, b: &Self::Object, t: Validated) -> Result<Self::Fusion<'a>, Self::Error> {
                let raw = (self.factors.0.__fusion_symbols_validated_fusion(&a.0, &b.0, t).map_err(ProductError::Factor0)?,
                    $(self.factors.$i.__fusion_symbols_validated_fusion(&a.$i, &b.$i, t).map_err(ProductError::$Err)?,)+);
                let mut active = (raw.0.clone().fuse(), $(raw.$i.clone().fuse(),)+);
                let current = (|| Some((active.0.next()?, $(active.$i.next()?,)+)))();
                // Emptiness first: an empty product consults no factor hook.
                let max = if current.is_none() { 0 } else {
                    let maxima = (self.factors.0.__fusion_symbols_max_multiplicity(&raw.0, t).unwrap_or_else(|| max_by_pass(&raw.0)),
                        $(self.factors.$i.__fusion_symbols_max_multiplicity(&raw.$i, t).unwrap_or_else(|| max_by_pass(&raw.$i)),)+);
                    Some(maxima.0)$(.and_then(|m| m.checked_mul(maxima.$i)))+.ok_or(ProductError::Overflow)?
                };
                let initial = (raw.0.fuse(), $(raw.$i.fuse(),)+);
                Ok(ProductFusion(Cartesian { initial, active, current, max }))
            }
            fn __fusion_symbols_max_multiplicity<'a>(&'a self, fusion: &Self::Fusion<'a>, _: Validated) -> Option<u32> {
                Some(fusion.max_multiplicity())
            }
            fn __fusion_symbols_validated_nsymbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object, t: Validated) -> Result<u32, Self::Error> {
                let mut n = Multiplicity::new();
                n.push(self.factors.0.__fusion_symbols_validated_nsymbol(&a.0, &b.0, &c.0, t).map_err(ProductError::Factor0));
                $(n.push(self.factors.$i.__fusion_symbols_validated_nsymbol(&a.$i, &b.$i, &c.$i, t).map_err(ProductError::$Err));)+
                n.finish(|| ProductError::Overflow)
            }
            fn __fusion_symbols_float_legs(&self, q: Query<'_, Self::Object>, rows: &mut LegRows, t: Validated) -> [Result<u32, Self::Error>; 4] {
                let mut legs = std::array::from_fn(|_| Multiplicity::new());
                fold_legs(&mut legs, self.factors.0.__fusion_symbols_float_legs(q.map(|x| &x.0), rows, t), ProductError::Factor0);
                $(fold_legs(&mut legs, self.factors.$i.__fusion_symbols_float_legs(q.map(|x| &x.$i), rows, t), ProductError::$Err);)+
                legs.map(|leg| leg.finish(|| ProductError::Overflow))
            }
            fn __fusion_symbols_float_blocks(&self, q: Query<'_, Self::Object>, rows: &mut LegRows, sink: &mut FloatSink, t: Validated) -> Result<(), Self::Error> {
                self.factors.0.__fusion_symbols_float_blocks(q.map(|x| &x.0), rows, sink, t).map_err(ProductError::Factor0)?;
                Self::Error::malformed(sink, 0)?;
                $(self.factors.$i.__fusion_symbols_float_blocks(q.map(|x| &x.$i), rows, sink, t).map_err(ProductError::$Err)?;
                Self::Error::malformed(sink, $i)?;)+
                Ok(())
            }
            fn __fusion_symbols_exact_symbol(&self, mut op: ExactOp<'_, '_, Self::Object>, t: Validated) -> Result<ExactAnswer, Self::Error> {
                let mut fold = Fold::new();
                fold.step(&mut op, &self.factors.0, |x| &x.0, t, ProductError::Factor0)?;
                $(fold.step(&mut op, &self.factors.$i, |x| &x.$i, t, ProductError::$Err)?;)+
                Ok(fold.finish(&op))
            }
            fn __fusion_symbols_exact_dim(&self, a: &Self::Object, d: &mut DimAcc, t: Validated) -> bool {
                self.factors.0.__fusion_symbols_exact_dim(&a.0, d, t)
                    $(&& self.factors.$i.__fusion_symbols_exact_dim(&a.$i, d, t))+
            }
        }
        impl<$C0, $($C, $S),+> GroupRepresentations for Product<($C0, $($C,)+)>
        where
            $C0: GroupRepresentations,
            for<'a> $C0::Fusion<'a>: Clone,
            $( $C: GroupRepresentations, for<'a> $C::Fusion<'a>: Clone,
               $Prev: Promote<$C::Scalar, Output = $S>, $S: Scalar, )+
        {
            type Cgc = LieCgc;
            fn cgc(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Result<LieCgc, Self::Error> {
                validate(self, &[a, b, c]);
                product_cgc(self, [a, b, c], || ProductError::Overflow, || {
                    float_cgc(vec![stored(&self.factors.0.cgc(&a.0, &b.0, &c.0).map_err(ProductError::Factor0)?),
                        $(stored(&self.factors.$i.cgc(&a.$i, &b.$i, &c.$i).map_err(ProductError::$Err)?),)+])
                        .ok_or(ProductError::Overflow)
                })
            }
            fn __fusion_symbols_exact_cgc<'q>(&self, [a, b, c]: [&'q Self::Object; 3], mut op: CgcOp<'q, '_>, t: Validated) -> Result<bool, Self::Error> {
                Ok(self.factors.0.__fusion_symbols_exact_cgc([&a.0, &b.0, &c.0], op.reborrow(), t).map_err(ProductError::Factor0)?
                    $(&& self.factors.$i.__fusion_symbols_exact_cgc([&a.$i, &b.$i, &c.$i], op.reborrow(), t).map_err(ProductError::$Err)?)+)
            }
        }
        impl<$C0: FusionCategory, $($C: FusionCategory),+> sealed::Factors for ($C0, $($C,)+) {
            type Iters<'a> = (Fuse<$C0::Fusion<'a>>, $(Fuse<$C::Fusion<'a>>,)+) where Self: 'a;
            type Current<'a> = (($C0::Object, u32), $(($C::Object, u32),)+) where Self: 'a;
        }
        impl<I0, O0, $($C, $S),+> Iterator for Cartesian<(I0, $($C,)+), ((O0, u32), $(($S, u32),)+)>
        where I0: Iterator<Item = (O0, u32)> + Clone, O0: Clone,
              $($C: Iterator<Item = ($S, u32)> + Clone, $S: Clone,)+
        {
            type Item = ((O0, $($S,)+), u32);
            fn next(&mut self) -> Option<Self::Item> {
                let mut current = self.current.take()?;
                let n = current.0.1 $(.checked_mul(current.$i.1).expect(BOUNDED))+;
                let result = ((current.0.0.clone(), $(current.$i.0.clone(),)+), n);
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

impl<C: FusionCategory> sealed::Factors for Vec<C> {
    type Iters<'a>
        = Vec<(C::Fusion<'a>, usize)>
    where
        Self: 'a;
    type Current<'a>
        = (usize, usize, Vec<(C::Object, u32)>)
    where
        Self: 'a;
}

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
        = ProductFusion<'a, Vec<C>>
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
        exact_dim(self, a, DimValue::Dim)
            .unwrap_or_else(|| scalar_product(self.factors.iter().zip(a).map(|(c, a)| c.dim(a))))
    }
    fn sqrt_dim(&self, a: &Self::Object) -> f64 {
        exact_dim(self, a, DimValue::Sqrt).unwrap_or_else(|| {
            scalar_product(self.factors.iter().zip(a).map(|(c, a)| c.sqrt_dim(a)))
        })
    }
    fn inv_sqrt_dim(&self, a: &Self::Object) -> f64 {
        exact_dim(self, a, DimValue::InvSqrt).unwrap_or_else(|| {
            scalar_product(self.factors.iter().zip(a).map(|(c, a)| c.inv_sqrt_dim(a)))
        })
    }
    fn inv_dim(&self, a: &Self::Object) -> f64 {
        exact_dim(self, a, DimValue::Inv).unwrap_or_else(|| {
            scalar_product(self.factors.iter().zip(a).map(|(c, a)| c.inv_dim(a)))
        })
    }
    fn __fusion_symbols_validated_frobenius_schur_phase(
        &self,
        a: &Self::Object,
        t: Validated,
    ) -> Result<Self::Scalar, Self::Error> {
        let mut value = None;
        for (index, (c, a)) in self.factors.iter().zip(a).enumerate() {
            let x = c
                .__fusion_symbols_validated_frobenius_schur_phase(a, t)
                .map_err(|source| ProductError::Factor { index, source })?;
            value = Some(value.map_or(x, |v| v * x));
        }
        Ok(value.unwrap_or_else(Self::Scalar::one))
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
    fn __fusion_symbols_validated_twist(
        &self,
        a: &Self::Object,
        t: Validated,
    ) -> Result<Self::Scalar, Self::Error> {
        let mut value = None;
        for (index, (c, a)) in self.factors.iter().zip(a).enumerate() {
            let x = c
                .__fusion_symbols_validated_twist(a, t)
                .map_err(|source| ProductError::Factor { index, source })?;
            value = Some(value.map_or(x, |v| v * x));
        }
        Ok(value.unwrap_or_else(Self::Scalar::one))
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
                    u64::try_from(&i).map_err(|_| EnumerationError::Overflow)?,
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
    ) -> Result<u32, Self::Error> {
        let mut n = Multiplicity::new();
        for (index, f) in self.factors.iter().enumerate() {
            n.push(
                f.__fusion_symbols_validated_nsymbol(&a[index], &b[index], &c[index], t)
                    .map_err(|source| ProductError::Factor { index, source }),
            );
        }
        n.finish(|| ProductError::Overflow)
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
        let max = if len == 0 {
            0
        } else {
            // Not short-circuiting: every unknown maximum is resolved before an
            // overflow is reported, as for tuple products.
            let mut max = Some(1u32);
            for (f, (iter, _)) in self.factors.iter().zip(&initial) {
                let m = f
                    .__fusion_symbols_max_multiplicity(iter, t)
                    .unwrap_or_else(|| max_by_pass(iter));
                max = max.and_then(|p| p.checked_mul(m));
            }
            max.ok_or(ProductError::Overflow)?
        };
        Ok(ProductFusion(Cartesian {
            initial,
            active,
            current: Some((0, len, current)),
            max,
        }))
    }
    fn __fusion_symbols_max_multiplicity<'a>(
        &'a self,
        fusion: &Self::Fusion<'a>,
        _: Validated,
    ) -> Option<u32> {
        Some(fusion.max_multiplicity())
    }
    fn __fusion_symbols_float_legs(
        &self,
        q: Query<'_, Self::Object>,
        rows: &mut LegRows,
        t: Validated,
    ) -> [Result<u32, Self::Error>; 4] {
        let mut legs = std::array::from_fn(|_| Multiplicity::new());
        for (index, factor) in self.factors.iter().enumerate() {
            let row = factor.__fusion_symbols_float_legs(q.map(|x| &x[index]), rows, t);
            fold_legs(&mut legs, row, |source| ProductError::Factor {
                index,
                source,
            });
        }
        legs.map(|leg| leg.finish(|| ProductError::Overflow))
    }
    // An empty product contributes its unit block, in its own scalar.
    fn __fusion_symbols_float_blocks(
        &self,
        q: Query<'_, Self::Object>,
        rows: &mut LegRows,
        sink: &mut FloatSink,
        t: Validated,
    ) -> Result<(), Self::Error> {
        if self.factors.is_empty() {
            let one = C::Scalar::one();
            match q {
                Query::F(_) => sink.0.push_f(&FBlock::one(one), [1; 4]),
                _ => sink.0.push_r(&RBlock::one(one), [1; 2]),
            }
        }
        for (index, factor) in self.factors.iter().enumerate() {
            factor
                .__fusion_symbols_float_blocks(q.map(|x| &x[index]), rows, sink, t)
                .map_err(|source| ProductError::Factor { index, source })?;
            ProductError::malformed(sink, index)?;
        }
        Ok(())
    }
    fn __fusion_symbols_exact_symbol(
        &self,
        mut op: ExactOp<'_, '_, Self::Object>,
        t: Validated,
    ) -> Result<ExactAnswer, Self::Error> {
        let mut fold = Fold::new();
        for (index, factor) in self.factors.iter().enumerate() {
            fold.step(
                &mut op,
                factor,
                |x| &x[index],
                t,
                |source| ProductError::Factor { index, source },
            )?;
        }
        Ok(fold.finish(&op))
    }
    // Dimension queries reach this hook without `validate`, also from an
    // enclosing product, so the length check lives here rather than in `dim`.
    fn __fusion_symbols_exact_dim(&self, a: &Self::Object, d: &mut DimAcc, t: Validated) -> bool {
        assert_eq!(a.len(), self.factors.len(), "foreign product object");
        self.factors
            .iter()
            .zip(a)
            .all(|(c, a)| c.__fusion_symbols_exact_dim(a, d, t))
    }
}

fn rblock<S: Scalar>([n, _]: [usize; 2], data: Storage<S>) -> RBlock<S> {
    RBlock::from_storage(n, data)
}

impl<C: GroupRepresentations> GroupRepresentations for Product<Vec<C>>
where
    for<'a> C::Fusion<'a>: Clone,
{
    type Cgc = LieCgc;
    fn cgc(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<LieCgc, Self::Error> {
        validate(self, &[a, b, c]);
        product_cgc(
            self,
            [a, b, c],
            || ProductError::Overflow,
            || {
                let blocks = self
                    .factors
                    .iter()
                    .enumerate()
                    .map(|(index, f)| {
                        f.cgc(&a[index], &b[index], &c[index])
                            .map(|block| stored(&block))
                            .map_err(|source| ProductError::Factor { index, source })
                    })
                    .collect::<Result<_, _>>()?;
                float_cgc(blocks).ok_or(ProductError::Overflow)
            },
        )
    }
    fn __fusion_symbols_exact_cgc<'q>(
        &self,
        [a, b, c]: [&'q Self::Object; 3],
        mut op: CgcOp<'q, '_>,
        t: Validated,
    ) -> Result<bool, Self::Error> {
        for (index, f) in self.factors.iter().enumerate() {
            let q = [&a[index], &b[index], &c[index]];
            if !f
                .__fusion_symbols_exact_cgc(q, op.reborrow(), t)
                .map_err(|source| ProductError::Factor { index, source })?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

// Runtime specialization: current stores [front, back) and the front items;
// active holds the incremental front cursor. Each initial factor carries its radix.
impl<I, O> Cartesian<Vec<(I, usize)>, (usize, usize, Vec<(O, u32)>)>
where
    I: Iterator<Item = (O, u32)> + Clone,
{
    fn entry(&self, mut index: usize) -> (Vec<O>, u32) {
        let mut n = 1u32;
        let objects = self
            .initial
            .iter()
            .map(|(iter, radix)| {
                let digit = index % radix;
                index /= radix;
                let (object, m) = restarted(iter, digit).1;
                n = n.checked_mul(m).expect(BOUNDED);
                object
            })
            .collect();
        (objects, n)
    }
}
fn outcome<O: Clone>(items: &[(O, u32)]) -> (Vec<O>, u32) {
    let n = items
        .iter()
        .fold(1u32, |p, (_, m)| p.checked_mul(*m).expect(BOUNDED));
    (items.iter().map(|(o, _)| o.clone()).collect(), n)
}
impl<I, O> Iterator for Cartesian<Vec<(I, usize)>, (usize, usize, Vec<(O, u32)>)>
where
    I: Iterator<Item = (O, u32)> + Clone,
    O: Clone,
{
    type Item = (Vec<O>, u32);
    fn next(&mut self) -> Option<Self::Item> {
        let (front, back, current) = self.current.as_mut()?;
        if front == back {
            return None;
        }
        let result = outcome(current);
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
impl<I, O> DoubleEndedIterator for Cartesian<Vec<(I, usize)>, (usize, usize, Vec<(O, u32)>)>
where
    I: DoubleEndedIterator<Item = (O, u32)> + Clone,
    O: Clone,
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
impl<I, O> ExactSizeIterator for Cartesian<Vec<(I, usize)>, (usize, usize, Vec<(O, u32)>)>
where
    I: ExactSizeIterator<Item = (O, u32)> + Clone,
    O: Clone,
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
            type Item = (usize, u32);
            fn next(&mut self) -> Option<(usize, u32)> {
                self.calls.set(self.calls.get() + 1);
                self.range.next().map(|c| (c, 1))
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
        let iter = Cartesian {
            initial,
            active,
            current: Some((0, len, current)),
            max: 1,
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
        type Fusion<'a> = std::iter::Once<((), u32)>;
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
        fn frobenius_schur_phase(&self, _: &()) -> Result<f64, Infallible> {
            Ok(1.0)
        }
        fn frobenius_schur_indicator(&self, _: &()) -> f64 {
            1.0
        }
        fn twist(&self, _: &()) -> Result<f64, Infallible> {
            Ok(1.0)
        }
        fn fusion<'a>(&'a self, _: &(), _: &()) -> Result<Self::Fusion<'a>, Infallible> {
            Ok(std::iter::once(((), self.n)))
        }
        fn nsymbol(&self, _: &(), _: &(), _: &()) -> Result<u32, Infallible> {
            Ok(self.n)
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
            query(&Product::new((FermionParity, good.clone())), symbol).unwrap(),
            (vec![2; rank], values.clone())
        );
        assert_eq!(
            query(
                &Product::new((Product::new((good, FermionParity)), FermionParity)),
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
                query(&Product::new((bad.clone(), FermionParity)), symbol),
                Err(expected(0))
            );
            assert_eq!(
                query(&Product::new((FermionParity, bad.clone())), symbol),
                Err(expected(1))
            );
            assert_eq!(
                query(
                    &Product::new((Product::new((bad.clone(), FermionParity)), FermionParity)),
                    symbol
                ),
                Err(ProductError::Factor0(expected(0)))
            );
            assert_eq!(
                query(
                    &Product::new((FermionParity, Product::new((FermionParity, bad.clone())))),
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
                query(&Product::new((bad.clone(), zero.clone())), symbol).unwrap(),
                (vec![0; rank], vec![])
            );
            assert_eq!(
                query(
                    &Product::new((Product::new((zero, bad)), FermionParity)),
                    symbol
                )
                .unwrap(),
                (vec![0; rank], vec![])
            );
        }
        let overflow = Blocks {
            n: u32::MAX,
            fault: Fault::None,
        };
        assert!(matches!(
            query(&Product::new((overflow, FermionParity)), symbol),
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

    // The in-place countdown must equal a Kronecker product that decodes every
    // output index independently (`axis = l * right + r`), for any shapes.
    fn kron_matches_naive<const R: usize>(rng: &mut impl rand::RngExt) {
        for trial in 0..200 {
            let left: [usize; R] = std::array::from_fn(|_| {
                if trial % 5 == 0 {
                    1
                } else {
                    rng.random_range(1..=3)
                }
            });
            let right: [usize; R] = std::array::from_fn(|_| rng.random_range(1..=3));
            let (l_len, r_len) = (
                left.iter().product::<usize>(),
                right.iter().product::<usize>(),
            );
            let l: Vec<f64> = (0..l_len).map(|_| rng.random_range(-2.0..2.0)).collect();
            let r: Vec<f64> = (0..r_len).map(|_| rng.random_range(-2.0..2.0)).collect();
            let dims: [usize; R] = std::array::from_fn(|a| left[a] * right[a]);
            let offset =
                |idx: [usize; R], dims: [usize; R]| (0..R).fold(0, |o, a| o * dims[a] + idx[a]);
            let naive: Vec<u64> = (0..l_len * r_len)
                .map(|mut k| {
                    let mut idx = [0; R];
                    for a in (0..R).rev() {
                        idx[a] = k % dims[a];
                        k /= dims[a];
                    }
                    let x = l[offset(std::array::from_fn(|a| idx[a] / right[a]), left)];
                    let y = r[offset(std::array::from_fn(|a| idx[a] % right[a]), right)];
                    (x * y).to_bits()
                })
                .collect();
            let mut out = vec![f64::NAN; l_len * r_len + 1];
            out[..l_len].copy_from_slice(&l);
            assert_eq!(kron_in_place(&mut out, left, right, &r, |x, y| x * y), dims);
            let got: Vec<u64> = out[..l_len * r_len].iter().map(|x| x.to_bits()).collect();
            assert_eq!(got, naive, "{left:?} x {right:?}");
            assert!(
                out[l_len * r_len].is_nan(),
                "writes stay inside the product"
            );
        }
    }

    #[test]
    fn in_place_kron_matches_naive_kron() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(179);
        kron_matches_naive::<2>(&mut rng);
        kron_matches_naive::<4>(&mut rng);
    }

    // A real prefix of several coefficients is multiplied in f64, then each
    // entry is promoted once; a complex-by-complex fold of the embedded prefix
    // would differ in the signs of zero imaginary parts.
    #[test]
    fn real_prefix_promotes_once_into_multi_element_complex_block() {
        use num_complex::Complex64;
        use rand::{RngExt, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1790);
        let mut real =
            |n: usize| -> Vec<f64> { (0..n).map(|_| rng.random_range(-2.0..2.0)).collect() };
        let (a, b) = (real(4), real(2));
        let c: Vec<Complex64> = real(4)
            .chunks(2)
            .map(|z| Complex64::new(z[0], z[1]))
            .collect();
        struct Dense<S, const R: usize>([usize; R], Vec<S>);
        impl<S: Scalar, const R: usize> SymbolBlock<R> for Dense<S, R> {
            type Scalar = S;
            fn dims(&self) -> [usize; R] {
                self.0
            }
            fn get(&self, _: [usize; R]) -> S {
                unreachable!("composition reads storage slices")
            }
            fn as_slice(&self) -> &[S] {
                &self.1
            }
        }
        // [2, 2] then [1, 2] real blocks, then a [2, 1] complex block: [4, 4].
        let a_block = Dense([2, 2], a.clone());
        let b_block = Dense([1, 2], b.clone());
        let c_block = Dense([2, 1], c.clone());
        let (dims, data) = Partial::new(&a_block, 16)
            .times(&b_block, 16, |l| l, |l, r| l * r)
            .times(&c_block, 16, Embed::<Complex64>::left, |l, r| l * r)
            .finish();
        assert_eq!(dims, [4, 4]);
        let naive: Vec<(u64, u64)> = (0..16)
            .map(|k| {
                let (i, j) = (k / 4, k % 4);
                // Row i = (a row, c row) digits; column j = (a col, b col) digits.
                let (ia, ic) = (i / 2, i % 2);
                let (ja, jb) = (j / 2, j % 2);
                let prefix = a[ia * 2 + ja] * b[jb];
                let z = Complex64::from(prefix) * c[ic];
                (z.re.to_bits(), z.im.to_bits())
            })
            .collect();
        let got: Vec<(u64, u64)> = data
            .as_slice()
            .iter()
            .map(|z| (z.re.to_bits(), z.im.to_bits()))
            .collect();
        assert_eq!(got, naive);
    }
}

/// The exact route of the outer boundary (design §e): search counts, shared
/// composite entries, unit copies and the publication barriers.
#[cfg(test)]
mod exact_route_tests {
    use super::*;
    use crate::exact::SignedSqrtRational;
    use crate::lie::{GroupId, LieGroup, LieIrrep, U1};
    use malachite_base::num::arithmetic::traits::Pow;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FETCHES: AtomicUsize = AtomicUsize::new(0);
    fn count_fetch() {
        FETCHES.fetch_add(1, Ordering::SeqCst);
    }

    struct Hook;
    impl Hook {
        fn set(f: fn()) -> Self {
            AFTER_FETCH.with(|h| h.set(Some(f)));
            Self
        }
    }
    impl Drop for Hook {
        fn drop(&mut self) {
            AFTER_FETCH.with(|h| h.set(None));
        }
    }

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn searches<T>(f: impl FnOnce() -> T) -> (usize, T) {
        crate::cache::SEARCHES.with(|n| n.set(0));
        let out = f();
        (crate::cache::SEARCHES.with(std::cell::Cell::get), out)
    }

    fn resident<P: FusionCategory>(p: &P, q: Query<'_, P::Object>) -> bool
    where
        P::Error: std::fmt::Debug,
    {
        let mut buf = LeafBuf(InlineVec::with_capacity(0));
        p.__fusion_symbols_exact_symbol(ExactOp::Describe(q, &mut buf), Validated(()))
            .unwrap();
        crate::cache::composed_is_resident(q.kind(), &buf)
    }

    fn su2() -> (LieGroup, [LieIrrep; 3]) {
        let g = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
        let x = [1, 0, 2].map(|j| g.irrep(&[j]).unwrap());
        (g, x)
    }

    /// A warm query searches once: the composite view of two or more leaves,
    /// or the one leaf's own view; a unit-class R searches nothing.
    #[test]
    fn warm_queries_search_once() {
        let _guard = lock();
        let (g, [h, z, one]) = su2();
        let p = Product::new((g.clone(), g.clone()));
        let [hh, zz, oo] = [&h, &z, &one].map(|x| (x.clone(), x.clone()));
        let f = || p.f_symbol(&hh, &hh, &hh, &hh, &zz, &oo).unwrap();
        let a = || p.a_symbol(&hh, &hh, &oo).unwrap();
        let r = || p.r_symbol(&hh, &hh, &oo).unwrap();
        let (cold, _) = (f(), (a(), r()));
        assert_eq!(searches(f), (1, cold));
        assert_eq!(searches(a).0, 1);
        assert_eq!(searches(r).0, 0);
        let one_leaf = Product::new((g, U1));
        let u = U1.unit();
        let [hu, zu, ou] = [&h, &z, &one].map(|x| (x.clone(), u));
        let f = || one_leaf.f_symbol(&hu, &hu, &hu, &hu, &zu, &ou).unwrap();
        f();
        assert_eq!(searches(f).0, 1);
    }

    /// Four, five and nine leaves (past the inline buffer) round the exact
    /// product of the leaves once, with one warm search.
    /// A composite CGC is one Product-tier entry charged at least by its
    /// entries, served to warm calls by one search, and released by a trim.
    #[test]
    fn composite_cgc_is_charged_shared_and_evictable() {
        use crate::cache::CacheCharge;
        let _guard = lock();
        let (g, [h, _, one]) = su2();
        let p = Product::new((g.clone(), g));
        let [a, b, c] = [(&h, &one), (&h, &h), (&one, &h)].map(|(x, y)| (x.clone(), y.clone()));
        let resident = || {
            let mut leaves = CgcLeaves::new();
            p.__fusion_symbols_exact_cgc([&a, &b, &c], CgcOp::Describe(&mut leaves), Validated(()))
                .unwrap();
            crate::cache::composed_is_resident(SymbolKind::Cgc, &leaves.leaves)
        };
        let cold = p.cgc(&a, &b, &c).unwrap();
        assert!(resident());
        let (n, warm) = searches(|| p.cgc(&a, &b, &c).unwrap());
        assert_eq!((n, &warm), (1, &cold));
        let mut exact = CgcExact(Vec::new());
        p.__fusion_symbols_exact_cgc([&a, &b, &c], CgcOp::Fetch(&mut exact), Validated(()))
            .unwrap();
        let channel = compose_channel(&exact.0).unwrap();
        let nnz = cold.to_dense().iter().filter(|v| **v != 0.0).count();
        assert_eq!(channel.entries.len(), nnz);
        let charge =
            crate::lie::general::roots::ProductValue::CompositeCgc(std::sync::Arc::new(channel))
                .value_bytes();
        assert!(charge >= nnz * std::mem::size_of::<MEntry>());
        crate::cache::trim_to(crate::cache::CoefficientCacheTier::Product, 0);
        assert!(!resident());
        assert_eq!(p.cgc(&a, &b, &c).unwrap(), cold);
    }

    #[test]
    fn many_leaves_round_once() {
        let _guard = lock();
        let (g, [h, z, one]) = su2();
        let leaf = crate::lie::su2::f_symbol_exact(1, 1, 1, 1, 0, 2);
        for k in [4, 5, 9] {
            let p = Product::new(vec![g.clone(); k]);
            let [hh, zz, oo] = [&h, &z, &one].map(|x| vec![x.clone(); k]);
            let f = || p.f_symbol(&hh, &hh, &hh, &hh, &zz, &oo).unwrap();
            let cold = f();
            let sign = if leaf.sign() < 0 && k % 2 == 1 { -1 } else { 1 };
            let exact = SignedSqrtRational::from_radicand(sign, leaf.radicand().pow(k as u64));
            assert_eq!(
                cold.as_slice()[0].to_bits(),
                exact.to_f64().to_bits(),
                "{k}"
            );
            assert_eq!(searches(f), (1, cold), "{k}");
        }
    }

    /// Flat, nested and `Vec` products of one leaf set share one composite
    /// entry: concurrent first requests through all of them build it once.
    #[test]
    fn flat_and_nested_requests_build_one_composite() {
        let _guard = lock();
        let (g, [h, z, one]) = su2();
        let flat = Product::new(vec![g.clone(); 3]);
        let left = Product::new((Product::new((g.clone(), g.clone())), g.clone()));
        let right = Product::new((g.clone(), Product::new(vec![g.clone(); 2])));
        crate::cache::reset();
        FETCHES.store(0, Ordering::SeqCst);
        let start = std::sync::Barrier::new(6);
        let values = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..6)
                .map(|i| {
                    let (flat, left, right, start) = (&flat, &left, &right, &start);
                    let (h, z, one) = (&h, &z, &one);
                    scope.spawn(move || {
                        let _hook = Hook::set(count_fetch);
                        start.wait();
                        let f = match i % 3 {
                            0 => {
                                let [a, e, f] = [h, z, one].map(|x| vec![x.clone(); 3]);
                                flat.f_symbol(&a, &a, &a, &a, &e, &f).unwrap()
                            }
                            1 => {
                                let [a, e, f] =
                                    [h, z, one].map(|x| ((x.clone(), x.clone()), x.clone()));
                                left.f_symbol(&a, &a, &a, &a, &e, &f).unwrap()
                            }
                            _ => {
                                let [a, e, f] =
                                    [h, z, one].map(|x| (x.clone(), vec![x.clone(); 2]));
                                right.f_symbol(&a, &a, &a, &a, &e, &f).unwrap()
                            }
                        };
                        f.as_slice()[0].to_bits()
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|t| t.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(FETCHES.load(Ordering::SeqCst), 1);
        assert!(values.iter().all(|&v| v == values[0]));
        let [a, e, f] = [&h, &z, &one].map(|x| vec![x.clone(); 3]);
        assert!(resident(&flat, Query::F([&a, &a, &a, &a, &e, &f])));
        crate::cache::reset();
    }

    /// The unit applies after publication: an odd⊗odd fermion parity pair
    /// (aggregate `+1`) passes the shared view through without a copy; a
    /// single odd⊗odd parity (`−1`) negates a copy.
    #[cfg(feature = "lie-generated")]
    #[test]
    fn units_copy_only_when_not_plus_one() {
        let _guard = lock();
        use crate::fermion::{FermionParity, FermionParityIrrep as P};
        let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        let o = su3.irrep(&[1, 1]).unwrap();
        let inner = Product::new((su3.clone(), su3.clone()));
        let oo = (o.clone(), o.clone());
        let view = inner.r_symbol(&oo, &oo, &oo).unwrap();
        let two = Product::new((FermionParity, Product::new((FermionParity, inner.clone()))));
        let x = (P::ODD, (P::ODD, oo.clone()));
        let c = (P::EVEN, (P::EVEN, oo.clone()));
        let shared = two.r_symbol(&x, &x, &c).unwrap();
        assert_eq!(shared.as_slice().as_ptr(), view.as_slice().as_ptr());
        let one = Product::new((FermionParity, inner));
        let (x, c) = ((P::ODD, oo.clone()), (P::EVEN, oo));
        let negated = one.r_symbol(&x, &x, &c).unwrap();
        assert_ne!(negated.as_slice().as_ptr(), view.as_slice().as_ptr());
        let expect: Vec<_> = view.as_slice().iter().map(|v| (-v).to_bits()).collect();
        let got: Vec<_> = negated.as_slice().iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, expect);
    }

    /// A reset between the leaves' exact fetch and the composite publication
    /// retains neither; a gate rejection publishes nothing; a leaf exact entry
    /// evicted mid-build still gives the same composite.
    #[cfg(feature = "lie-generated")]
    #[test]
    fn composite_publication_barriers() {
        use crate::lie::RootSystem;
        let _guard = lock();
        let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        let o = su3.irrep(&[1, 1]).unwrap();
        let p = Product::new((su3.clone(), su3.clone()));
        let x = (o.clone(), o.clone());
        let fq = Query::F([&x; 6]);
        let leaf = || crate::cache::f_exact_is_resident(RootSystem::A(2), [&[1, 1][..]; 6]);
        let f = || p.f_symbol(&x, &x, &x, &x, &x, &x).unwrap();
        crate::cache::reset();
        let good = f();
        assert!(resident(&p, fq) && leaf());

        crate::cache::reset();
        let reset = {
            let _hook = Hook::set(crate::cache::reset);
            f()
        };
        assert_eq!(reset.as_slice(), good.as_slice());
        assert!(!resident(&p, fq) && !leaf(), "a reset retains neither");

        crate::cache::reset();
        let evicted = {
            let _hook = Hook::set(|| {
                crate::cache::trim_to(crate::cache::CoefficientCacheTier::FSymbol, 0);
            });
            f()
        };
        assert_eq!(evicted.as_slice(), good.as_slice());
        assert!(resident(&p, fq), "an evicted leaf still publishes");

        let rq = Query::R([&x; 3]);
        crate::cache::reset();
        CORRUPT.with(|c| c.set(true));
        assert_eq!(
            p.r_symbol(&x, &x, &x).err(),
            Some(ProductError::Gate { gate: "R" })
        );
        assert!(!resident(&p, rq), "a gate rejection publishes nothing");
        p.r_symbol(&x, &x, &x).unwrap();
        assert!(resident(&p, rq));
        crate::cache::reset();
    }
}
