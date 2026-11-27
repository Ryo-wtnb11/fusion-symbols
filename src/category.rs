//! Common mathematical contract for validated fusion-category objects.

use num_complex::Complex64;
use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, Mul, Neg, Sub},
};

/// Fusion multiplicities: one outcome, multiplicity-free, or general.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// A returned block and the independently checked N axes to validate it against.
    pub type SymbolWithAxes<B, const R: usize> = (B, [u32; R]);
    /// Object count, unbounded by `u64` so that a product nested in another
    /// still bounds its coordinates, with a product's per-factor counts so
    /// that unranking visits each leaf once.
    pub struct ExactCount {
        pub(crate) total: super::enumeration::Count,
        pub(crate) factors: Vec<ExactCount>,
    }
    /// An object's exact enumeration index with its category's count, so that
    /// a nested product ranks in one pass over its leaves.
    pub struct Ranked {
        pub(crate) count: super::enumeration::Count,
        pub(crate) index: malachite_nz::integer::Integer,
    }
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex64 {}

    /// The kind of a categorical symbol query.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum SymbolKind {
        /// `F^{abc}_d[e, f]`.
        F,
        /// `R^{ab}_c`.
        R,
        /// `A^{ab}_c`.
        A,
        /// `B^{ab}_c`.
        B,
    }

    impl SymbolKind {
        /// Labels per query: six for F, three otherwise.
        pub(crate) fn labels(self) -> usize {
            match self {
                Self::F => 6,
                _ => 3,
            }
        }
    }

    /// One symbol query over validated objects.
    pub enum Query<'q, O> {
        /// `F^{abc}_d[e, f]` as `[a, b, c, d, e, f]`.
        F([&'q O; 6]),
        /// `R^{ab}_c`.
        R([&'q O; 3]),
        /// `A^{ab}_c`.
        A([&'q O; 3]),
        /// `B^{ab}_c`.
        B([&'q O; 3]),
    }

    impl<O> Clone for Query<'_, O> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<O> Copy for Query<'_, O> {}

    impl<'q, O> Query<'q, O> {
        pub(crate) fn kind(&self) -> SymbolKind {
            match self {
                Self::F(_) => SymbolKind::F,
                Self::R(_) => SymbolKind::R,
                Self::A(_) => SymbolKind::A,
                Self::B(_) => SymbolKind::B,
            }
        }

        /// The same query on a factor's (or a composite's) objects.
        pub(crate) fn map<P>(self, f: impl Fn(&'q O) -> &'q P) -> Query<'q, P> {
            match self {
                Self::F(x) => Query::F(x.map(f)),
                Self::R(x) => Query::R(x.map(f)),
                Self::A(x) => Query::A(x.map(f)),
                Self::B(x) => Query::B(x.map(f)),
            }
        }
    }

    /// An operation of the exact product protocol (§e of the design).
    pub enum ExactOp<'q, 'b, O> {
        /// Label-free class of a symbol kind; no cache.
        Class(SymbolKind),
        /// Append this query's Radical leaves and return its aggregate unit;
        /// no cache.
        Describe(Query<'q, O>, &'b mut LeafBuf<'q>),
        /// Append each own Radical leaf's exact block, in leaf order.
        Fetch(Query<'q, O>, &'b mut ExactBuf),
        /// The one Radical leaf's rounded view (a query with one leaf).
        View(Query<'q, O>, &'b mut Option<View>),
    }

    impl<'q, 'b, O> ExactOp<'q, 'b, O> {
        /// The same operation on a composite's product objects.
        pub(crate) fn map<P>(self, f: impl Fn(&'q O) -> &'q P) -> ExactOp<'q, 'b, P> {
            match self {
                Self::Class(k) => ExactOp::Class(k),
                Self::Describe(q, b) => ExactOp::Describe(q.map(f), b),
                Self::Fetch(q, b) => ExactOp::Fetch(q.map(f), b),
                Self::View(q, b) => ExactOp::View(q.map(f), b),
            }
        }
    }

    /// The answer to an [`ExactOp`].
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ExactAnswer {
        /// No exact values: the caller takes the float path.
        Unavailable,
        /// Every admissible query of the kind is a single unit entry.
        UnitClass,
        /// Exact values with this many Radical leaves.
        RadicalClass { leaves: usize },
        /// The aggregate unit of a described query.
        Unit(Unit),
        /// A fetched or viewed query.
        Done,
    }

    /// A unit value: multiplying a correctly rounded value by it is exact.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Unit {
        Zero,
        PlusOne,
        MinusOne,
    }

    impl Unit {
        pub(crate) fn times(self, other: Self) -> Self {
            match (self, other) {
                (Self::Zero, _) | (_, Self::Zero) => Self::Zero,
                (a, b) if a == b => Self::PlusOne,
                _ => Self::MinusOne,
            }
        }

        /// `self · x`, exact: `−1` negates both parts of a complex value.
        pub(crate) fn apply<S: super::Scalar>(self, x: S) -> S {
            match self {
                Self::Zero => S::zero(),
                Self::PlusOne => x,
                Self::MinusOne => -x,
            }
        }

        /// The unit of a unit-class block: empty is an absent channel.
        pub(crate) fn of<S: super::Scalar>(entries: &[S]) -> Option<Self> {
            match entries {
                [] => Some(Self::Zero),
                &[x] if x == S::zero() => Some(Self::Zero),
                &[x] if x == S::one() => Some(Self::PlusOne),
                &[x] if x == -S::one() => Some(Self::MinusOne),
                _ => None,
            }
        }
    }

    /// The rounded view of one Radical leaf or of a composite: real `f64`.
    #[derive(Clone, Debug)]
    pub enum View {
        F(super::FBlock<f64>),
        /// R, A or B.
        R(super::RBlock<f64>),
    }

    /// The exact blocks of a query's Radical leaves, in leaf order.
    pub struct ExactBuf {
        pub(crate) f: InlineVec<std::sync::Arc<crate::exact::radical::RadicalBlock<4>>, 4>,
        pub(crate) r2: InlineVec<std::sync::Arc<crate::exact::radical::RadicalBlock<2>>, 4>,
    }

    impl ExactBuf {
        pub(crate) fn with_capacity(n: usize) -> Self {
            Self {
                f: InlineVec::with_capacity(n),
                r2: InlineVec::with_capacity(n),
            }
        }
    }

    /// Family tags of a Radical leaf: its owned key's `kind`.
    pub(crate) const LEAF_SIMPLE: u8 = 0;
    pub(crate) const LEAF_A1: u8 = 1;
    pub(crate) const LEAF_CU1: u8 = 2;

    /// One label of a Radical leaf, read in place with its fingerprint.
    #[derive(Clone, Copy, Debug)]
    pub enum LeafLabel<'q> {
        /// A simple factor's rank-length words and their stored fingerprint.
        Words(&'q [i64], u64),
        /// A rank-one label's single word and `label_hash(&[w])`.
        Inline(i64, u64),
    }

    impl LeafLabel<'_> {
        /// The rank-one label `w`.
        pub(crate) fn inline(w: i64) -> Self {
            Self::Inline(w, crate::cache::label_hash(&[w]))
        }

        pub(crate) fn words(&self) -> &[i64] {
            match self {
                Self::Words(w, _) => w,
                Self::Inline(w, _) => std::slice::from_ref(w),
            }
        }

        pub(crate) fn hash(&self) -> u64 {
            match *self {
                Self::Words(_, h) | Self::Inline(_, h) => h,
            }
        }
    }

    /// A Radical leaf of a product query, borrowed: its family tag, its
    /// presented root (`A(1)` for rank-one families without one) and the
    /// query's labels in that factor, of which the symbol kind's
    /// [`SymbolKind::labels`] are read.
    #[derive(Clone, Copy, Debug)]
    pub struct LeafRef<'q> {
        pub(crate) family: u8,
        pub(crate) root: crate::lie::RootSystem,
        pub(crate) labels: [LeafLabel<'q>; 6],
    }

    impl<'q> LeafRef<'q> {
        /// The leaf of `labels` (three for R/A/B; the unread rest padded).
        pub(crate) fn new(
            family: u8,
            root: crate::lie::RootSystem,
            labels: impl IntoIterator<Item = LeafLabel<'q>>,
        ) -> Self {
            let mut out = [LeafLabel::Inline(0, 0); 6];
            for (o, l) in out.iter_mut().zip(labels) {
                *o = l;
            }
            Self {
                family,
                root,
                labels: out,
            }
        }
    }

    /// Up to `N` values inline, the rest in one spill vector.
    pub struct InlineVec<T, const N: usize> {
        len: usize,
        inline: [Option<T>; N],
        spill: Vec<T>,
    }

    impl<T, const N: usize> InlineVec<T, N> {
        /// Room for `n` values: the spill is reserved once, only past `N`.
        pub(crate) fn with_capacity(n: usize) -> Self {
            Self {
                len: 0,
                inline: std::array::from_fn(|_| None),
                spill: Vec::with_capacity(n.saturating_sub(N)),
            }
        }

        pub(crate) fn push(&mut self, x: T) {
            match self.inline.get_mut(self.len) {
                Some(slot) => *slot = Some(x),
                None => self.spill.push(x),
            }
            self.len += 1;
        }

        pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
            self.inline.iter().flatten().chain(&self.spill)
        }
    }

    /// The Radical leaves of one product query, depth-first.
    pub struct LeafBuf<'q>(pub(crate) InlineVec<LeafRef<'q>, 4>);
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
/// and hashing use full mathematical identity.
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
    /// Distinct fusion outcomes, each with its multiplicity `N(a, b, c) ≥ 1`,
    /// in deterministic order.
    type Fusion<'a>: Iterator<Item = (Self::Object, u32)>
    where
        Self: 'a;
    /// Genuine failure or absent capability; `Infallible` when neither occurs.
    type Error: std::error::Error;

    /// Multiplicity class of fusion.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::{FusionCategory, FusionStyle}, lie::{GroupId, LieGroup, U1}};
    ///
    /// assert_eq!(U1.fusion_style(), FusionStyle::Unique);
    /// assert_eq!(LieGroup::new(GroupId::su(2)?)?.fusion_style(), FusionStyle::Simple);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn fusion_style(&self) -> FusionStyle;
    /// Braiding capability and symmetry class.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::{BraidingStyle, FusionCategory}, fermion::FermionParity,
    ///     lie::{GroupId, LieGroup}, table::TableCategory};
    ///
    /// assert_eq!(LieGroup::new(GroupId::su(2)?)?.braiding_style(), BraidingStyle::Bosonic);
    /// assert_eq!(FermionParity.braiding_style(), BraidingStyle::Fermionic);
    /// assert_eq!(TableCategory::fibonacci().braiding_style(), BraidingStyle::Anyonic);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn braiding_style(&self) -> BraidingStyle;
    /// The single simple unit.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, HalfInt, LieGroup}};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// let unit = su2.unit();
    /// assert_eq!(unit.spin(), Some(HalfInt::new(0, 1)?));
    /// assert_eq!(su2.fusion(&unit, &half)?.collect::<Vec<_>>(), [(half, 1)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, HalfInt, LieGroup}};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// let so3 = LieGroup::new(GroupId::so(3)?)?;
    /// assert!(su2.contains(&half));
    /// assert!(!so3.contains(&su2.irrep(&[2])?)); // another group's object
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn contains(&self, a: &Self::Object) -> bool {
        let _ = a;
        true
    }
    /// Dual of a validated object; foreign objects panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{U1Irrep, U1}};
    ///
    /// assert_eq!(U1.dual(&U1Irrep::new(3)?), U1Irrep::new(-3)?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn dual(&self, a: &Self::Object) -> Self::Object;
    /// Quantum dimension; foreign objects panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, HalfInt, LieGroup}};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// assert_eq!(su2.dim(&half), 2.0);
    /// assert_eq!(su2.dim(&su2.irrep_spin(HalfInt::new(2, 1)?)?), 5.0); // 2j + 1
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn dim(&self, a: &Self::Object) -> f64;
    /// Phase of `F(a, dual(a), a, a, unit, unit)`; foreign objects panic.
    ///
    /// # Errors
    /// Genuine failure while obtaining the phase, such as a generated Lie
    /// family's module generation for a complex irrep; never a substitute
    /// value.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, HalfInt, LieGroup}};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// assert_eq!(su2.frobenius_schur_phase(&half)?, -1.0); // pseudoreal
    /// assert_eq!(su2.frobenius_schur_phase(&su2.irrep(&[2])?)?, 1.0); // real
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn frobenius_schur_phase(&self, a: &Self::Object) -> Result<Self::Scalar, Self::Error>;
    /// Phase if self-dual, otherwise zero (TensorKitSectors 0.3.9,
    /// `src/sectors.jl::frobenius_schur_indicator`); foreign objects panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{U1Irrep, U1}};
    ///
    /// assert_eq!(U1.frobenius_schur_indicator(&U1Irrep::new(1)?), 0.0); // not self-dual
    /// assert_eq!(U1.frobenius_schur_indicator(&U1Irrep::UNIT), 1.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn frobenius_schur_indicator(&self, a: &Self::Object) -> Self::Scalar;
    /// Ribbon twist; panics on foreign objects or `BraidingStyle::None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, fermion::{FermionParity, FermionParityIrrep}};
    ///
    /// assert_eq!(FermionParity.twist(&FermionParityIrrep::ODD), -1.0);
    /// assert_eq!(FermionParity.twist(&FermionParityIrrep::EVEN), 1.0);
    /// ```
    fn twist(&self, a: &Self::Object) -> Self::Scalar;
    /// Deterministic outcomes `c` of `a ⊗ b`, each once with its multiplicity
    /// `N(a, b, c) ≥ 1`, which equals [`Self::nsymbol`]. Reading every channel
    /// with its multiplicity this way costs one decomposition lookup per
    /// generated simple factor (one lookup for a simple Lie pair), with no
    /// per-channel N lookup. Checked arithmetic overflow is an error. Foreign
    /// objects panic before computation.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, HalfInt, LieGroup}};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// let one = su2.irrep_spin(HalfInt::new(1, 1)?)?;
    /// let out: Vec<_> = su2.fusion(&half, &one)?.map(|(c, n)| (c.spin(), n)).collect();
    /// // ½ ⊗ 1 = ½ ⊕ 3/2
    /// assert_eq!(out, [(Some(HalfInt::new(1, 2)?), 1), (Some(HalfInt::new(3, 2)?), 1)]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn fusion<'a>(
        &'a self,
        a: &Self::Object,
        b: &Self::Object,
    ) -> Result<Self::Fusion<'a>, Self::Error>;
    /// Exact vertex multiplicity, including zero; foreign objects panic.
    ///
    /// An absent channel is `Ok(0)`, never an error.
    ///
    /// # Errors
    /// Genuine failure only: an unrepresentable multiplicity (a
    /// [`crate::product::Product`] of factor multiplicities, or a generated
    /// Lie multiplicity) or a factor's failure. Foreign objects still panic;
    /// use [`Self::contains`] first to avoid that.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, HalfInt, LieGroup}};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// let one = su2.irrep_spin(HalfInt::new(1, 1)?)?;
    /// assert_eq!(su2.nsymbol(&half, &half, &one)?, 1);
    /// assert_eq!(su2.nsymbol(&half, &half, &half)?, 0); // not an error
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn nsymbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<u32, Self::Error>;
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
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup, U1}};
    ///
    /// assert!(U1.has_trivial_associator_gauge());
    /// assert!(!LieGroup::new(GroupId::su(2)?)?.has_trivial_associator_gauge());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup}};
    ///
    /// assert!(LieGroup::new(GroupId::su(2)?)?.has_canonical_unit());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn has_canonical_unit(&self) -> bool {
        false
    }
    /// `F^{abc}_d[e,f]`; inadmissible vertices return zero storage. Logical
    /// axes are `[N(a,b,e), N(e,c,d), N(b,c,f), N(a,f,d)]`.
    /// Foreign objects panic; errors indicate genuine failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::category::{FusionCategory, SymbolBlock};
    /// use fusion_symbols::lie::{GroupId, HalfInt, LieGroup};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// let unit = su2.unit();
    /// // F^{½½½}_½[0, 0] = sqrt(1 · 1) · {½ ½ 0; ½ ½ 0} = -1/2
    /// let f = su2.f_symbol(&half, &half, &half, &half, &unit, &unit)?;
    /// assert_eq!((f.dims(), f.get([0, 0, 0, 0])), ([1, 1, 1, 1], -0.5));
    /// // ½ ⊗ ½ cannot give ½: the first axis has length zero.
    /// let f = su2.f_symbol(&half, &half, &half, &half, &half, &unit)?;
    /// assert_eq!(f.dims()[0], 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::category::{FusionCategory, SymbolBlock};
    /// use fusion_symbols::lie::{GroupId, HalfInt, LieGroup};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// let r = |c| su2.r_symbol(&half, &half, &c).map(|r| r.get([0, 0]));
    /// assert_eq!(r(su2.unit())?, -1.0); // antisymmetric singlet
    /// assert_eq!(r(su2.irrep(&[2])?)?, 1.0); // symmetric triplet
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn r_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<Self::RSymbol, Self::Error>;
    /// `A^{ab}_c`, which turns a splitting vertex `c → a ⊗ b` into a fusion
    /// vertex `ā ⊗ c → b` (TensorKitSectors 0.3.9,
    /// `src/sectors.jl::Asymbol_from_Fsymbol`):
    ///
    /// ```text
    /// A^{ab}_c[μ, ν] = √d_a √d_b / √d_c · conj(κ_a · F^{ā a b}_b[𝟙, c][0, 0, μ, ν])
    /// ```
    ///
    /// with `κ_a` the [Frobenius–Schur phase](Self::frobenius_schur_phase)
    /// (not the indicator). Logical axes are `[N(a,b,c), N(ā,c,b)]`, row
    /// `μ`, column `ν`; the two vertices through `𝟙` have multiplicity one.
    /// `N(a,b,c) = 0` gives the empty `0 × 0` block. Braiding is not needed.
    ///
    /// A and B are returned as [`RBlock`] regardless of the category's
    /// [`Self::RSymbol`] type, so the default works for every category; a
    /// `1 × 1` block is stored inline without allocation. The default
    /// evaluates that F; a category may override it with the same values.
    /// With warm Lie caches and inline dimension arithmetic, only the returned
    /// coefficient buffer allocates (nothing for an inline block). Products also
    /// allocate factor blocks and their per-query Kronecker composition buffers.
    /// Errors are those of N, the phase and F.
    ///
    /// # Panics
    /// Panics on foreign objects. The default also panics if the F block's
    /// size disagrees with `N(a,b,c)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::category::{FusionCategory, SymbolBlock};
    /// use fusion_symbols::lie::{GroupId, HalfInt, LieGroup};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// let a = su2.a_symbol(&half, &half, &su2.unit())?;
    /// assert_eq!(a.dims(), [1, 1]);
    /// assert!((a.get([0, 0]).abs() - 1.0).abs() < 1e-15); // unitary
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn a_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<RBlock<Self::Scalar>, Self::Error> {
        let n = self.nsymbol(a, b, c)? as usize;
        if n == 0 {
            return Ok(rank2_block(n, std::iter::empty()));
        }
        let f = self.f_symbol(&self.dual(a), a, b, b, &self.unit(), c)?;
        let kappa = self.frobenius_schur_phase(a)?;
        let scale = vertex_scale(self, a, b, c);
        Ok(rank2_block(
            n,
            f.as_slice().iter().map(|&x| scale * (kappa * x).conj()),
        ))
    }
    /// `B^{ab}_c`, which turns a splitting vertex `c → a ⊗ b` into a fusion
    /// vertex `c ⊗ b̄ → a` (TensorKitSectors 0.3.9,
    /// `src/sectors.jl::Bsymbol_from_Fsymbol`):
    ///
    /// ```text
    /// B^{ab}_c[μ, ν] = √d_a √d_b / √d_c · F^{a b b̄}_a[c, 𝟙][μ, ν, 0, 0]
    /// ```
    ///
    /// Logical axes are `[N(a,b,c), N(c,b̄,a)]`, row `μ`, column `ν`.
    /// `N(a,b,c) = 0` gives the empty `0 × 0` block. Braiding is not needed.
    ///
    /// Returned as [`RBlock`] regardless of [`Self::RSymbol`], as for
    /// [`Self::a_symbol`]. The default evaluates that F; a category may
    /// override it with the same values. Errors are those of N and F.
    ///
    /// # Panics
    /// As for [`Self::a_symbol`].
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::category::{FusionCategory, SymbolBlock};
    /// use fusion_symbols::lie::{GroupId, HalfInt, LieGroup};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// let b = su2.b_symbol(&half, &half, &su2.irrep(&[2])?)?;
    /// assert_eq!(b.dims(), [1, 1]);
    /// assert!((b.get([0, 0]).abs() - 1.0).abs() < 1e-15); // unitary
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn b_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<RBlock<Self::Scalar>, Self::Error> {
        let n = self.nsymbol(a, b, c)? as usize;
        if n == 0 {
            return Ok(rank2_block(n, std::iter::empty()));
        }
        let f = self.f_symbol(a, b, &self.dual(b), a, c, &self.unit())?;
        let scale = vertex_scale(self, a, b, c);
        Ok(rank2_block(n, f.as_slice().iter().map(|&x| scale * x)))
    }
    /// `√d_a`; foreign objects panic.
    ///
    /// The default is `self.dim(a).sqrt()`. A product multiplies its factors'
    /// values left to right instead of taking the root of the product.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup}};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// assert_eq!(su2.sqrt_dim(&su2.irrep(&[3])?), 2.0); // dim 4
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn sqrt_dim(&self, a: &Self::Object) -> f64 {
        self.dim(a).sqrt()
    }
    /// `1 / √d_a`; foreign objects panic.
    ///
    /// The default is `self.sqrt_dim(a).recip()`; a product multiplies its
    /// factors' values left to right.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup}};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// assert_eq!(su2.inv_sqrt_dim(&su2.irrep(&[3])?), 0.5); // dim 4
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn inv_sqrt_dim(&self, a: &Self::Object) -> f64 {
        self.sqrt_dim(a).recip()
    }
    /// `1 / d_a`; foreign objects panic.
    ///
    /// The default is `self.dim(a).recip()`; a product multiplies its
    /// factors' values left to right.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, HalfInt, LieGroup}};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// assert_eq!(su2.inv_dim(&half), 0.5);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn inv_dim(&self, a: &Self::Object) -> f64 {
        self.dim(a).recip()
    }
    /// Exact number of objects, `Ok(None)` for a mathematically infinite
    /// category.
    ///
    /// Enumeration is a zero-based bijection between objects and indices whose
    /// order is a frozen part of each category's documented contract. It is
    /// independent of the object type's `Ord` and of fusion-iterator order.
    /// Equal category values enumerate identically.
    ///
    /// An infinite category returns `Ok(None)` even though its labels have a
    /// finite machine range; [`Self::object_at`] reports labels past that
    /// range as [`EnumerationError::Overflow`].
    ///
    /// # Errors
    /// [`EnumerationError::NotEnumerable`] (the default) when this category
    /// value does not provide enumeration; [`EnumerationError::Overflow`] for a
    /// finite count above `u64::MAX`, while individual indices stay addressable.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, finite::ZN, lie::U1};
    ///
    /// assert_eq!(ZN::new(3)?.object_count()?, Some(3));
    /// assert_eq!(U1.object_count()?, None); // infinite
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        Err(EnumerationError::NotEnumerable)
    }
    /// The object at a zero-based enumeration index; the inverse of
    /// [`Self::object_index`].
    ///
    /// # Errors
    /// [`EnumerationError::NotEnumerable`] (the default);
    /// [`EnumerationError::OutOfRange`] at or past a finite count;
    /// [`EnumerationError::Overflow`] when the object's label is not
    /// representable.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, HalfInt, LieGroup}};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// assert_eq!(su2.object_at(1)?, half); // SU(2) index = 2j
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn object_at(&self, index: u64) -> Result<Self::Object, EnumerationError> {
        let _ = index;
        Err(EnumerationError::NotEnumerable)
    }
    /// The zero-based enumeration index of an object; the inverse of
    /// [`Self::object_at`].
    ///
    /// # Errors
    /// [`EnumerationError::NotEnumerable`] (the default);
    /// [`EnumerationError::Overflow`] when the index exceeds `u64::MAX`.
    ///
    /// # Panics
    /// Panics on a foreign object, before testing enumerability.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{U1Irrep, U1}};
    ///
    /// // U(1) order 0, 1, -1, 2, -2, …
    /// assert_eq!(U1.object_index(&U1Irrep::new(-2)?)?, 4);
    /// assert_eq!(U1.object_at(4)?, U1Irrep::new(-2)?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn object_index(&self, a: &Self::Object) -> Result<u64, EnumerationError> {
        assert!(self.contains(a), "foreign object");
        Err(EnumerationError::NotEnumerable)
    }

    // Crate-internal `__fusion_symbols_validated_*` entry points, not part of
    // the public API. A product calls them after validating every leaf object
    // with `contains` (and, for F/R/A/B, establishing nonzero representable
    // shapes); a nested product overrides them to skip re-validation and shape
    // recomputation, which would otherwise repeat at every nesting level. Symbol
    // hooks return the block with its checked N axes; Product checks storage
    // before indexing. The token is unnameable outside the crate, so they can be
    // neither called nor overridden there; the prefix keeps them from colliding
    // with downstream extension-trait methods.
    #[doc(hidden)]
    fn __fusion_symbols_validated_sqrt_dim(&self, a: &Self::Object, _: sealed::Validated) -> f64 {
        self.sqrt_dim(a)
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_inv_sqrt_dim(
        &self,
        a: &Self::Object,
        _: sealed::Validated,
    ) -> f64 {
        self.inv_sqrt_dim(a)
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_inv_dim(&self, a: &Self::Object, _: sealed::Validated) -> f64 {
        self.inv_dim(a)
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_object_index(
        &self,
        a: &Self::Object,
        t: sealed::Validated,
    ) -> Result<sealed::Ranked, EnumerationError> {
        let index = self.object_index(a)?.into();
        let count = self.__fusion_symbols_exact_object_count(t)?.total;
        Ok(sealed::Ranked { count, index })
    }
    // Not object-taking: the token only keeps it crate-internal. A product
    // overrides it so that an enclosing product sees counts above `u64::MAX`.
    #[doc(hidden)]
    fn __fusion_symbols_exact_object_count(
        &self,
        _: sealed::Validated,
    ) -> Result<sealed::ExactCount, EnumerationError> {
        // A public count above `u64::MAX` is finite but unknown; see `Count`.
        let total = match self.object_count() {
            Ok(Some(n)) => enumeration::Count::Finite(n.into()),
            Ok(None) => enumeration::Count::Infinite,
            Err(EnumerationError::Overflow) => enumeration::Count::AboveU64,
            Err(e) => return Err(e),
        };
        Ok(sealed::ExactCount {
            total,
            factors: Vec::new(),
        })
    }
    // [`Self::object_at`] given this category's own exact count.
    #[doc(hidden)]
    fn __fusion_symbols_counted_object_at(
        &self,
        index: u64,
        count: &sealed::ExactCount,
        _: sealed::Validated,
    ) -> Result<Self::Object, EnumerationError> {
        let _ = count;
        self.object_at(index)
    }
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
    ) -> Result<Self::Scalar, Self::Error> {
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
    // The exact largest multiplicity of a nonempty `fusion`, or `None` when
    // this category cannot state it without a pass (a Generic category outside
    // this crate). An empty iterator may report an upper bound: a product
    // checks emptiness before reading it.
    #[doc(hidden)]
    fn __fusion_symbols_max_multiplicity<'a>(
        &'a self,
        fusion: &Self::Fusion<'a>,
        _: sealed::Validated,
    ) -> Option<u32> {
        let _ = fusion;
        match self.fusion_style() {
            FusionStyle::Generic => None,
            _ => Some(1),
        }
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_nsymbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        _: sealed::Validated,
    ) -> Result<u32, Self::Error> {
        self.nsymbol(a, b, c)
    }
    // The block with the axes its product checks it against. `ones` is given
    // when every product axis is 1: each leaf N is then 1 (a product of positive
    // multiplicities), so it is not evaluated again.
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
        ones: Option<[u32; 4]>,
        _: sealed::Validated,
    ) -> Result<sealed::SymbolWithAxes<Self::FSymbol, 4>, Self::Error> {
        let block = self.f_symbol(a, b, c, d, e, f)?;
        let n = match ones {
            Some(n) => n,
            None => [
                self.nsymbol(a, b, e)?,
                self.nsymbol(e, c, d)?,
                self.nsymbol(b, c, f)?,
                self.nsymbol(a, f, d)?,
            ],
        };
        Ok((block, n))
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_r_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        ones: Option<[u32; 2]>,
        _: sealed::Validated,
    ) -> Result<sealed::SymbolWithAxes<Self::RSymbol, 2>, Self::Error> {
        let block = self.r_symbol(a, b, c)?;
        let n = match ones {
            Some(n) => n,
            None => [self.nsymbol(a, b, c)?, self.nsymbol(b, a, c)?],
        };
        Ok((block, n))
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_a_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        ones: Option<[u32; 2]>,
        _: sealed::Validated,
    ) -> Result<sealed::SymbolWithAxes<RBlock<Self::Scalar>, 2>, Self::Error> {
        let block = self.a_symbol(a, b, c)?;
        let n = match ones {
            Some(n) => n,
            None => [self.nsymbol(a, b, c)?, self.nsymbol(&self.dual(a), c, b)?],
        };
        Ok((block, n))
    }
    #[doc(hidden)]
    fn __fusion_symbols_validated_b_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        ones: Option<[u32; 2]>,
        _: sealed::Validated,
    ) -> Result<sealed::SymbolWithAxes<RBlock<Self::Scalar>, 2>, Self::Error> {
        let block = self.b_symbol(a, b, c)?;
        let n = match ones {
            Some(n) => n,
            None => [self.nsymbol(a, b, c)?, self.nsymbol(c, &self.dual(b), a)?],
        };
        Ok((block, n))
    }
    // The exact product protocol: a product's F/R/A/B are the rounding of
    // the exact composition of its leaves' exact values when every factor
    // answers. The default is unavailable, which keeps the float path.
    #[doc(hidden)]
    fn __fusion_symbols_exact_symbol(
        &self,
        op: sealed::ExactOp<'_, '_, Self::Object>,
        _: sealed::Validated,
    ) -> Result<sealed::ExactAnswer, Self::Error> {
        let _ = op;
        Ok(sealed::ExactAnswer::Unavailable)
    }
}

/// The hook of a unit-class category: `Class` is unit for every kind, and
/// `Describe` reads the single closed-form entry, with no search.
pub(crate) fn unit_symbol<C: FusionCategory>(
    cat: &C,
    op: sealed::ExactOp<'_, '_, C::Object>,
) -> Result<sealed::ExactAnswer, C::Error> {
    use sealed::{ExactAnswer, ExactOp, Query};
    let q = match op {
        ExactOp::Class(_) => return Ok(ExactAnswer::UnitClass),
        ExactOp::Describe(q, _) => q,
        ExactOp::Fetch(..) | ExactOp::View(..) => return Ok(ExactAnswer::Unavailable),
    };
    let unit = match q {
        Query::F([a, b, c, d, e, f]) => {
            sealed::Unit::of(cat.f_symbol(a, b, c, d, e, f)?.as_slice())
        }
        Query::R([a, b, c]) => sealed::Unit::of(cat.r_symbol(a, b, c)?.as_slice()),
        Query::A([a, b, c]) => sealed::Unit::of(cat.a_symbol(a, b, c)?.as_slice()),
        Query::B([a, b, c]) => sealed::Unit::of(cat.b_symbol(a, b, c)?.as_slice()),
    };
    Ok(unit.map_or(ExactAnswer::Unavailable, ExactAnswer::Unit))
}

// `√d_a √d_b / √d_c`, the normalization shared by A and B.
fn vertex_scale<C: FusionCategory>(
    cat: &C,
    a: &C::Object,
    b: &C::Object,
    c: &C::Object,
) -> C::Scalar {
    C::Scalar::from_f64(cat.sqrt_dim(a) * cat.sqrt_dim(b) / cat.sqrt_dim(c))
}

// The square `n × n` A or B block from row-major data (`N(a,b,c)` equals
// `N(ā,c,b)` and `N(c,b̄,a)` by Frobenius reciprocity).
pub(crate) fn rank2_block<S: Scalar>(
    n: usize,
    data: impl ExactSizeIterator<Item = S>,
) -> RBlock<S> {
    assert_eq!(
        data.len(),
        n * n,
        "F block size disagrees with its N symbols"
    );
    RBlock::from_row_major(n, data)
}

/// Clebsch–Gordan block with axes `[dim(a), dim(b), dim(c), N(a,b,c)]`;
/// an absent channel has multiplicity axis 0.
///
/// Tuples of 2..=12 blocks and vectors of any length represent lazy Kronecker
/// products: the last factor varies fastest on every axis. Their `dims` may panic
/// if a combined axis overflows `usize`; blocks returned by `Product::cgc` are checked.
pub trait CgcBlock {
    /// Axis lengths, with multiplicity last.
    fn dims(&self) -> [usize; 4];
    /// Read one coefficient.
    ///
    /// # Panics
    /// Panics if any index is outside its axis.
    fn get(&self, index: [usize; 4]) -> f64;
    /// Materialize the block as a fresh row-major vector over
    /// `[ma, mb, mc, mu]`, multiplicity fastest: the entry at
    /// `((ma * db + mb) * dc + mc) * n + mu` equals `get([ma, mb, mc, mu])`.
    ///
    /// Every call allocates new owned storage of `da * db * dc * n` entries
    /// and reads each through [`Self::get`]; it generates no coefficient and
    /// registers no cache entry. A shape with a zero axis returns an empty
    /// vector without reading. Prefer [`Self::get`] in loops that touch few
    /// entries.
    ///
    /// # Panics
    /// Panics if the element count or its byte size is not representable
    /// (`usize` / `isize::MAX` bytes), before allocating.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::category::{CgcBlock, FusionCategory, GroupRepresentations};
    /// use fusion_symbols::lie::{GroupId, HalfInt, LieGroup};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// let c = su2.cgc(&half, &half, &su2.unit())?;
    /// let dense = c.to_dense(); // [m₁, m₂, m₃, μ] = [2, 2, 1, 1]
    /// assert_eq!(dense.len(), 4);
    /// assert_eq!(dense[2], c.get([1, 0, 0, 0])); // ((1·2 + 0)·1 + 0)·1 + 0
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn to_dense(&self) -> Vec<f64> {
        let dims = self.dims();
        if dims.contains(&0) {
            return Vec::new();
        }
        let count = dims
            .into_iter()
            .try_fold(1usize, usize::checked_mul)
            .expect("dense CGC size overflow");
        std::alloc::Layout::array::<f64>(count).expect("dense CGC byte size overflow");
        let mut data = Vec::with_capacity(count);
        let [da, db, dc, n] = dims;
        for ma in 0..da {
            for mb in 0..db {
                for mc in 0..dc {
                    for mu in 0..n {
                        data.push(self.get([ma, mb, mc, mu]));
                    }
                }
            }
        }
        data
    }
}

/// A one-dimensional channel: nonzero means present, zero means absent.
/// Abelian providers return exactly `1.0` or `0.0`.
impl CgcBlock for f64 {
    fn dims(&self) -> [usize; 4] {
        [1, 1, 1, usize::from(*self != 0.0)]
    }
    fn get(&self, index: [usize; 4]) -> f64 {
        assert!(index == [0; 4] && *self != 0.0, "CGC index out of range");
        *self
    }
}

/// Group categories that additionally provide Clebsch–Gordan data.
pub trait GroupRepresentations: FusionCategory {
    /// Family-specific CGC storage implementing the common block contract.
    type Cgc: CgcBlock;
    /// CGC for validated objects; foreign objects panic, absent channels
    /// return blocks with multiplicity axis zero, and errors indicate genuine failure or absent capability.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::category::{CgcBlock, FusionCategory, GroupRepresentations};
    /// use fusion_symbols::lie::{GroupId, HalfInt, LieGroup};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let half = su2.irrep_spin(HalfInt::new(1, 2)?)?;
    /// let c = su2.cgc(&half, &half, &su2.unit())?; // axes [m₁, m₂, m₃, μ], m ascending
    /// let s = 0.5f64.sqrt(); // ⟨½ ±½; ½ ∓½ | 0 0⟩ = ±1/√2
    /// assert!((c.get([1, 0, 0, 0]) - s).abs() < 1e-15 && (c.get([0, 1, 0, 0]) + s).abs() < 1e-15);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn cgc(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<Self::Cgc, Self::Error>;
}

mod enumeration;
pub use enumeration::EnumerationError;
pub(crate) use enumeration::{rank, rank_checked, unrank, Count};
mod residual;
pub use residual::{f_unitarity_residual, hexagon_residual, pentagon_residual};

mod blocks;
pub(crate) use blocks::Storage;
pub use blocks::{FBlock, RBlock};
