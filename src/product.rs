//! Ordered Deligne products of two through twelve immediate factors.
//!
//! Objects are tuples; explicit nesting is preserved. Fusion outcomes vary the
//! first factor fastest. Each F/R multiplicity axis varies the last factor
//! fastest, with row-major block storage. Scalar promotion and multiplication
//! fold left to right. Reassociation agrees up to floating-point tolerance where
//! intermediate queries are representable. No cross-factor fermionic sign,
//! coefficient cache, or product Clebsch–Gordan capability is added.
//!
//! Factor fusion iterators must be cloneable. The higher-ranked GAT bound on
//! these implementations currently restricts factors to owned category types;
//! arbitrary borrowed providers are not promised.
use crate::category::{
    sealed::Validated, BraidingStyle, FBlock, FusionCategory, FusionStyle, Promote, RBlock, Scalar,
    Storage, SymbolBlock,
};
use std::{
    convert::Infallible,
    fmt,
    hash::{Hash, Hasher},
    iter::Fuse,
    sync::{Arc, OnceLock},
};

/// A product retaining the ordered values and nesting of its factors.
#[derive(Clone)]
pub struct Product<C> {
    factors: C,
    fingerprint: OnceLock<Arc<[u8]>>,
}
impl<C> Product<C> {
    /// Construct a product of 2..=12 immediate factors.
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

/// Cartesian fusion iterator; resets clone initialized iterators, never query factors again.
#[derive(Clone, Debug)]
pub struct ProductFusion<I, O> {
    initial: I,
    active: I,
    current: Option<O>,
}

// Read all multiplicities before multiplying: a zero beats any nonzero prefix overflow.
fn multiplicity(values: impl IntoIterator<Item = u32>) -> Option<u32> {
    let mut product = Some(1u32);
    for n in values {
        product = if n == 0 || product == Some(0) {
            Some(0)
        } else {
            product.and_then(|p| p.checked_mul(n))
        };
    }
    product
}
fn shape<S, const R: usize>(factors: &[[u32; R]]) -> Option<[usize; R]> {
    let mut dims = [0; R];
    for axis in 0..R {
        dims[axis] = usize::try_from(multiplicity(factors.iter().map(|d| d[axis]))?).ok()?;
    }
    let count = if dims.contains(&0) {
        0
    } else {
        dims.iter().try_fold(1usize, |p, &n| p.checked_mul(n))?
    };
    std::alloc::Layout::array::<S>(count).ok()?;
    Some(dims)
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
fn kron<A: Scalar + Promote<B::Scalar>, B: SymbolBlock<R>, const R: usize>(
    left_dims: [usize; R],
    left: Storage<A>,
    right: B,
) -> ([usize; R], Storage<<A as Promote<B::Scalar>>::Output>) {
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
        A::left(left.as_slice()[l]) * A::right(right.get(r))
    };
    let data = if dims == [1; R] {
        Storage::One(entry(0))
    } else {
        Storage::Shared((0..dims.iter().product()).map(entry).collect::<Arc<[_]>>())
    };
    (dims, data)
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
            // Public queries validate every leaf object once through `dim`, which
            // never re-validates, then use the `__fusion_symbols_validated_*` paths below.
            fn dual(&self, a: &Self::Object) -> Self::Object {
                self.dim(a);
                self.__fusion_symbols_validated_dual(a, Validated(()))
            }
            fn dim(&self, a: &Self::Object) -> f64 {
                let value = self.factors.0.dim(&a.0); $(let value = value * self.factors.$i.dim(&a.$i);)+ value
            }
            fn frobenius_schur_phase(&self, a: &Self::Object) -> Self::Scalar {
                self.dim(a);
                self.__fusion_symbols_validated_frobenius_schur_phase(a, Validated(()))
            }
            fn frobenius_schur_indicator(&self, a: &Self::Object) -> Self::Scalar {
                self.dim(a);
                self.__fusion_symbols_validated_frobenius_schur_indicator(a, Validated(()))
            }
            fn twist(&self, a: &Self::Object) -> Self::Scalar {
                self.dim(a);
                assert!(self.factors.0.braiding_style() != BraidingStyle::None, "product factor 0 has no braiding");
                $(assert!(self.factors.$i.braiding_style() != BraidingStyle::None, "product factor {} has no braiding", $i);)+
                self.__fusion_symbols_validated_twist(a, Validated(()))
            }
            fn fusion<'a>(&'a self, a: &Self::Object, b: &Self::Object) -> Result<Self::Fusion<'a>, Self::Error> {
                for a in [a, b] { self.dim(a); }
                self.__fusion_symbols_validated_fusion(a, b, Validated(()))
            }
            /// Panics only if a nonzero product multiplicity exceeds `u32`, or an object is foreign.
            fn nsymbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> u32 {
                for a in [a, b, c] { self.dim(a); }
                self.__fusion_symbols_validated_nsymbol(a, b, c, Validated(()))
            }
            fn f_symbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object, d: &Self::Object, e: &Self::Object, f: &Self::Object) -> Result<Self::FSymbol, Self::Error> {
                for a in [a, b, c, d, e, f] { self.dim(a); }
                let t = Validated(());
                let shapes = [
                    [self.factors.0.__fusion_symbols_validated_nsymbol(&a.0,&b.0,&e.0,t),self.factors.0.__fusion_symbols_validated_nsymbol(&e.0,&c.0,&d.0,t),self.factors.0.__fusion_symbols_validated_nsymbol(&b.0,&c.0,&f.0,t),self.factors.0.__fusion_symbols_validated_nsymbol(&a.0,&f.0,&d.0,t)],
                    $([self.factors.$i.__fusion_symbols_validated_nsymbol(&a.$i,&b.$i,&e.$i,t),self.factors.$i.__fusion_symbols_validated_nsymbol(&e.$i,&c.$i,&d.$i,t),self.factors.$i.__fusion_symbols_validated_nsymbol(&b.$i,&c.$i,&f.$i,t),self.factors.$i.__fusion_symbols_validated_nsymbol(&a.$i,&f.$i,&d.$i,t)],)+
                ];
                let dims = shape::<Self::Scalar, 4>(&shapes).ok_or(ProductError::Overflow)?;
                if dims.contains(&0) { return Ok(FBlock { dims, data: Storage::Empty }); }
                let block = self.__fusion_symbols_validated_f_symbol(a, b, c, d, e, f, t)?;
                debug_assert_eq!(block.dims(), dims);
                Ok(block)
            }
            fn r_symbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object) -> Result<Self::RSymbol, Self::Error> {
                for a in [a, b, c] { self.dim(a); }
                if self.factors.0.braiding_style() == BraidingStyle::None { return Err(ProductError::NoBraiding { factor: 0 }); }
                $(if self.factors.$i.braiding_style() == BraidingStyle::None { return Err(ProductError::NoBraiding { factor: $i }); })+
                let t = Validated(());
                let shapes = [[self.factors.0.__fusion_symbols_validated_nsymbol(&a.0,&b.0,&c.0,t), self.factors.0.__fusion_symbols_validated_nsymbol(&b.0,&a.0,&c.0,t)],
                    $([self.factors.$i.__fusion_symbols_validated_nsymbol(&a.$i,&b.$i,&c.$i,t), self.factors.$i.__fusion_symbols_validated_nsymbol(&b.$i,&a.$i,&c.$i,t)],)+];
                let dims = shape::<Self::Scalar, 2>(&shapes).ok_or(ProductError::Overflow)?;
                if dims.contains(&0) { return Ok(RBlock { n: dims[0], data: Storage::Empty }); }
                let block = self.__fusion_symbols_validated_r_symbol(a, b, c, t)?;
                debug_assert_eq!(block.dims(), dims);
                Ok(block)
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
                multiplicity([self.factors.0.__fusion_symbols_validated_nsymbol(&a.0, &b.0, &c.0, t), $(self.factors.$i.__fusion_symbols_validated_nsymbol(&a.$i, &b.$i, &c.$i, t)),+])
                    .expect("nonzero product multiplicity exceeds u32")
            }
            // The caller has established nonzero, representable shapes for every
            // factor (an enclosing product's shape bounds this one's), so the
            // factor blocks are combined without recomputing them.
            fn __fusion_symbols_validated_f_symbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object, d: &Self::Object, e: &Self::Object, f: &Self::Object, t: Validated) -> Result<Self::FSymbol, Self::Error> {
                let (dims, data) = storage(self.factors.0.__fusion_symbols_validated_f_symbol(&a.0,&b.0,&c.0,&d.0,&e.0,&f.0,t).map_err(ProductError::Factor0)?);
                $(let (dims, data) = kron(dims, data, self.factors.$i.__fusion_symbols_validated_f_symbol(&a.$i,&b.$i,&c.$i,&d.$i,&e.$i,&f.$i,t).map_err(ProductError::$Err)?);)+
                Ok(FBlock { dims, data })
            }
            // As for F; braiding of every factor has also been checked.
            fn __fusion_symbols_validated_r_symbol(&self, a: &Self::Object, b: &Self::Object, c: &Self::Object, t: Validated) -> Result<Self::RSymbol, Self::Error> {
                let (dims, data) = storage(self.factors.0.__fusion_symbols_validated_r_symbol(&a.0,&b.0,&c.0,t).map_err(ProductError::Factor0)?);
                $(let (dims, data) = kron(dims, data, self.factors.$i.__fusion_symbols_validated_r_symbol(&a.$i,&b.$i,&c.$i,t).map_err(ProductError::$Err)?);)+
                Ok(RBlock { n: dims[0], data })
            }
            fn authority_fingerprint(&self) -> &[u8] {
                self.fingerprint.get_or_init(|| {
                    let fps = [self.factors.0.authority_fingerprint(), $(self.factors.$i.authority_fingerprint()),+];
                    let mut bytes = b"fusion-symbols:product:epoch=1:axes=last-fastest:fusion=first-fastest;".to_vec();
                    bytes.extend_from_slice(&(fps.len() as u32).to_le_bytes());
                    for fp in fps { bytes.extend_from_slice(&(fp.len() as u64).to_le_bytes()); bytes.extend_from_slice(fp); }
                    bytes.into()
                })
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
                if let Some(value) = self.active.0.next() {
                    current.0 = value; self.current = Some(current); return Some(result);
                }
                self.active.0 = self.initial.0.clone();
                current.0 = self.active.0.next().expect("initialized nonempty fusion iterator");
                $(if let Some(value) = self.active.$i.next() {
                    current.$i = value; self.current = Some(current); return Some(result);
                }
                self.active.$i = self.initial.$i.clone();
                current.$i = self.active.$i.next().expect("initialized nonempty fusion iterator");)+
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
