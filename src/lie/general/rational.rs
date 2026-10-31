//! The exact rational of the Cartan engine: reduced values in machine words
//! when they fit, Malachite otherwise.
//!
//! Every value has one representation: `Small(n, d)` exactly when its reduced
//! pair (`d > 0`, `gcd(|n|, d) = 1`, zero as `0/1`) fits `i64 × u64`, and a
//! boxed Malachite rational otherwise. Every result, including those computed
//! in Malachite, is selected by that rule, so a cancelling result returns to
//! words and derived equality is numeric equality. The word kernels prove each
//! result representable with checked arithmetic; on failure the same operation
//! is recomputed from the intact operands in Malachite. Overflow is never
//! approximated or reported.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::num::NonZeroU64;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::exact::word_gcd as gcd;
use malachite_base::num::arithmetic::traits::{NegAssign, Reciprocal, Square};
use malachite_nz::natural::Natural;
use malachite_q::Rational;

/// An exact rational in its unique representation (see the module docs).
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct Q(Repr);

// Private, so no sibling can build an unreduced word pair or a fitting Big.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Repr {
    Small(i64, NonZeroU64),
    Big(Box<Rational>),
}

const ONE: NonZeroU64 = NonZeroU64::MIN;

#[derive(Clone, Copy)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

impl From<i64> for Q {
    fn from(n: i64) -> Self {
        Self(Repr::Small(n, ONE))
    }
}

impl From<Rational> for Q {
    /// Malachite rationals are reduced, so only the fit decides the form.
    fn from(r: Rational) -> Self {
        match word_pair(&r) {
            Some((n, d)) => Self(Repr::Small(n, d)),
            None => Self(Repr::Big(Box::new(r))),
        }
    }
}

/// The reduced pair of `r` when it fits `i64 × u64`.
fn word_pair(r: &Rational) -> Option<(i64, NonZeroU64)> {
    let d = NonZeroU64::new(u64::try_from(r.denominator_ref()).ok()?)?;
    let n = i128::from(u64::try_from(r.numerator_ref()).ok()?);
    let n = i64::try_from(if *r < 0u32 { -n } else { n }).ok()?;
    Some((n, d))
}

impl Q {
    /// `±mag/den` for a reduced pair (`den > 0`, coprime, zero as `+0/1`)
    /// in its unique form.
    fn from_reduced(neg: bool, mag: u128, den: u128) -> Self {
        let n = if neg {
            // -2^63 ..= -1; the wrapping negation of 2^63 is i64::MIN.
            u64::try_from(mag)
                .ok()
                .filter(|&m| m <= 1 << 63)
                .map(|m| m.wrapping_neg() as i64)
        } else {
            i64::try_from(mag).ok()
        };
        match (n, u64::try_from(den)) {
            (Some(n), Ok(d)) => Self(Repr::Small(n, NonZeroU64::new(d).expect("d > 0"))),
            // The pair is already reduced; the constructor's gcd is 1.
            _ => Self(Repr::Big(Box::new(Rational::from_sign_and_naturals(
                !neg,
                Natural::from(mag),
                Natural::from(den),
            )))),
        }
    }

    pub(super) fn is_zero(&self) -> bool {
        matches!(self.0, Repr::Small(0, _))
    }

    pub(super) fn is_one(&self) -> bool {
        matches!(self.0, Repr::Small(1, d) if d == ONE)
    }

    /// The sign of the value against zero.
    pub(super) fn signum(&self) -> Ordering {
        match &self.0 {
            Repr::Small(n, _) => n.cmp(&0),
            Repr::Big(r) => r.as_ref().partial_cmp(&0u32).expect("total order"),
        }
    }

    /// `(|numerator|, denominator)` of the reduced value; borrowed from a Big
    /// value and one inline limb each for a word value.
    pub(super) fn magnitudes(&self) -> (Cow<'_, Natural>, Cow<'_, Natural>) {
        match &self.0 {
            Repr::Small(n, d) => (
                Cow::Owned(Natural::from(n.unsigned_abs())),
                Cow::Owned(Natural::from(d.get())),
            ),
            Repr::Big(r) => (
                Cow::Borrowed(r.numerator_ref()),
                Cow::Borrowed(r.denominator_ref()),
            ),
        }
    }

    /// Heap bytes owned by this value beyond its inline slot: the boxed
    /// rational and its limbs. Exact for a value [`Q::compact`]ed since its
    /// last mutation (limb capacity equals length; inline limbs are charged
    /// too, conservatively); word values own nothing.
    pub(super) fn heap_bytes(&self) -> usize {
        match &self.0 {
            Repr::Small(..) => 0,
            Repr::Big(r) => {
                let limbs = r.numerator_ref().limb_count() + r.denominator_ref().limb_count();
                std::mem::size_of::<Rational>()
                    + usize::try_from(limbs).expect("limb count fits usize")
                        * std::mem::size_of::<malachite_nz::platform::Limb>()
            }
        }
    }

    /// Re-allocates a Big value's limbs at exactly their length, so that
    /// [`Q::heap_bytes`] bounds what retained storage owns: Malachite keeps
    /// a vector's capacity when a result shrinks.
    pub(super) fn compact(&mut self) {
        if let Repr::Big(r) = &mut self.0 {
            let tight = |x: &mut Natural| {
                let mut limbs = x.to_limbs_asc();
                limbs.shrink_to_fit();
                *x = Natural::from_owned_limbs_asc(limbs);
            };
            r.mutate_numerator_and_denominator(|n, d| {
                tight(n);
                tight(d);
            });
        }
    }

    /// The value as a Malachite rational, for exact interop in tests.
    #[cfg(test)]
    pub(super) fn to_rational(&self) -> Rational {
        self.as_rational().into_owned()
    }

    fn as_rational(&self) -> Cow<'_, Rational> {
        match &self.0 {
            Repr::Small(n, d) => Cow::Owned(match (n.unsigned_abs(), d.get()) {
                (_, 1) => Rational::from(*n),
                (1, d) => {
                    let r = Rational::from(d).reciprocal();
                    if *n < 0 {
                        -r
                    } else {
                        r
                    }
                }
                // Malachite has no unchecked constructor: this reduces the
                // coprime pair again, one word gcd.
                (m, d) => Rational::from_sign_and_unsigneds(*n >= 0, m, d),
            }),
            Repr::Big(r) => Cow::Borrowed(r),
        }
    }

    /// A Big value that may have cancelled into the word range.
    fn renormalize(&mut self) {
        if let Repr::Big(r) = &self.0 {
            if let Some((n, d)) = word_pair(r) {
                self.0 = Repr::Small(n, d);
            }
        }
    }

    /// `self ± x·y`. The same decisions as a product followed by an
    /// assignment, but a product of Big operands is added as a plain
    /// Malachite value instead of a boxed temporary.
    pub(super) fn add_mul(&mut self, x: &Q, y: &Q) {
        self.fused(x, y, Op::Add);
    }

    /// `self -= x·y`; see [`Q::add_mul`].
    pub(super) fn sub_mul(&mut self, x: &Q, y: &Q) {
        self.fused(x, y, Op::Sub);
    }

    fn fused(&mut self, x: &Q, y: &Q, op: Op) {
        if let (Repr::Small(xn, xd), Repr::Small(yn, yd)) = (&x.0, &y.0) {
            let p = small(*xn, xd.get(), *yn, yd.get(), Op::Mul).expect("products never overflow");
            self.assign(&p, op);
            return;
        }
        let p = x.as_rational().as_ref() * y.as_rational().as_ref();
        if let Repr::Big(r) = &mut self.0 {
            match op {
                Op::Add => **r += p,
                _ => **r -= p,
            }
            self.renormalize();
        } else {
            let acc = self.as_rational();
            let v = match op {
                Op::Add => acc.as_ref() + p,
                _ => acc.as_ref() - p,
            };
            *self = Q::from(v);
        }
    }

    /// `a op b` from references; the one arithmetic decision per operation.
    fn binop(a: &Q, b: &Q, op: Op) -> Q {
        if matches!(op, Op::Div) && b.is_zero() {
            panic!("division by zero");
        }
        if let (Repr::Small(an, ad), Repr::Small(bn, bd)) = (&a.0, &b.0) {
            if let Some(q) = small(*an, ad.get(), *bn, bd.get(), op) {
                return q;
            }
        }
        let (x, y) = (a.as_rational(), b.as_rational());
        let (x, y) = (x.as_ref(), y.as_ref());
        Q::from(match op {
            Op::Add => x + y,
            Op::Sub => x - y,
            Op::Mul => x * y,
            Op::Div => x / y,
        })
    }

    /// `self op= rhs`, reusing a Big left operand's storage.
    fn assign(&mut self, rhs: &Q, op: Op) {
        if let Repr::Big(r) = &mut self.0 {
            if matches!(op, Op::Div) && rhs.is_zero() {
                panic!("division by zero");
            }
            let y = rhs.as_rational();
            let r = r.as_mut();
            match op {
                Op::Add => *r += y.as_ref(),
                Op::Sub => *r -= y.as_ref(),
                Op::Mul => *r *= y.as_ref(),
                Op::Div => *r /= y.as_ref(),
            }
            self.renormalize();
        } else {
            *self = Q::binop(self, rhs, op);
        }
    }
}

/// `a/b op c/d` for reduced word pairs, as a reduced pair, or `None` when an
/// intermediate needs more than the 128-bit scratch. Divisions and gcds run
/// on 64-bit magnitudes; only products and the sum are 128-bit.
fn small(a: i64, b: u64, c: i64, d: u64, op: Op) -> Option<Q> {
    let zero = || Some(Q::from(0));
    let (am, cm) = (a.unsigned_abs(), c.unsigned_abs());
    let wide = |x: u64, y: u64| u128::from(x) * u128::from(y);
    match op {
        // Cross-cancel first (Knuth 4.5.1): the products are coprime and
        // below 2^128, so only the narrowing can fail.
        Op::Mul => {
            if a == 0 || c == 0 {
                return zero();
            }
            let (g1, g2) = (gcd(am, d), gcd(cm, b));
            Some(Q::from_reduced(
                (a < 0) != (c < 0),
                wide(am / g1, cm / g2),
                wide(b / g2, d / g1),
            ))
        }
        // The caller rejected c = 0; the sign of c moves to the numerator.
        Op::Div => {
            if a == 0 {
                return zero();
            }
            let (g1, g2) = (gcd(am, cm), gcd(b, d));
            Some(Q::from_reduced(
                (a < 0) != (c < 0),
                wide(am / g1, d / g2),
                wide(b / g2, cm / g1),
            ))
        }
        // Each cross product is below 2^127 but their sum need not be.
        Op::Add | Op::Sub => {
            let g = gcd(b, d);
            let (b1, d1) = (b / g, d / g);
            let t1 = i128::from(a) * i128::from(d1);
            let t2 = i128::from(c) * i128::from(b1);
            let s = match op {
                Op::Add => t1.checked_add(t2)?,
                _ => t1.checked_sub(t2)?,
            };
            if s == 0 {
                return zero();
            }
            // gcd(s, b1 d1 g) = gcd(s, g); `s` may be exactly i128::MIN, so
            // only its unsigned magnitude is divided.
            let sm = s.unsigned_abs();
            let h = if g == 1 {
                1
            } else {
                let r = match u64::try_from(sm) {
                    Ok(x) => x % g,
                    Err(_) => u64::try_from(sm % u128::from(g)).expect("below g"),
                };
                gcd(g, r)
            };
            let sm = match (h, u64::try_from(sm)) {
                (1, _) => sm,
                (_, Ok(x)) => u128::from(x / h),
                _ => sm / u128::from(h),
            };
            Some(Q::from_reduced(s < 0, sm, wide(b1, d / h)))
        }
    }
}

macro_rules! binary {
    ($tr:ident, $f:ident, $assign:ident, $af:ident, $op:expr) => {
        impl $tr<&Q> for &Q {
            type Output = Q;
            fn $f(self, rhs: &Q) -> Q {
                Q::binop(self, rhs, $op)
            }
        }
        impl $tr<Q> for &Q {
            type Output = Q;
            fn $f(self, rhs: Q) -> Q {
                Q::binop(self, &rhs, $op)
            }
        }
        impl $tr<&Q> for Q {
            type Output = Q;
            fn $f(mut self, rhs: &Q) -> Q {
                self.assign(rhs, $op);
                self
            }
        }
        impl $tr<Q> for Q {
            type Output = Q;
            fn $f(mut self, rhs: Q) -> Q {
                self.assign(&rhs, $op);
                self
            }
        }
        impl $assign<&Q> for Q {
            fn $af(&mut self, rhs: &Q) {
                self.assign(rhs, $op);
            }
        }
        impl $assign<Q> for Q {
            fn $af(&mut self, rhs: Q) {
                self.assign(&rhs, $op);
            }
        }
    };
}

binary!(Add, add, AddAssign, add_assign, Op::Add);
binary!(Sub, sub, SubAssign, sub_assign, Op::Sub);
binary!(Mul, mul, MulAssign, mul_assign, Op::Mul);

binary!(Div, div, DivAssign, div_assign, Op::Div);

impl Neg for Q {
    type Output = Q;
    fn neg(self) -> Q {
        match self.0 {
            Repr::Small(n, d) => match n.checked_neg() {
                Some(n) => Q(Repr::Small(n, d)),
                None => Q::from_reduced(false, u128::from(n.unsigned_abs()), u128::from(d.get())),
            },
            Repr::Big(mut r) => {
                r.neg_assign();
                let mut q = Q(Repr::Big(r));
                q.renormalize();
                q
            }
        }
    }
}

/// The exact `s·√(a·b/scale)`, with the reduced radicand `s²·a·b/scale`.
#[cfg(test)]
pub(super) fn radical(s: &Q, a: &Q, b: &Q, scale: &Q) -> crate::exact::SignedSqrtRational {
    let sign = if s.signum() == Ordering::Less { -1 } else { 1 };
    crate::exact::SignedSqrtRational::from_radicand(sign, (s * s * a * b / scale).to_rational())
}

/// `s·√(a·b/scale)` rounded once, by the conversion of
/// [`crate::exact::SignedSqrtRational::to_f64`]: round-only callers never hold
/// the exact value. With `s = ps/qs`, `a = pa/qa`, `b = pb/qb` and
/// `scale = pt/qt` (`a, b, scale > 0`), the radicand is `U/V` with
/// `U = ps²·pa·pb·qt` and `V = qs²·qa·qb·pt`; the conversion depends only on
/// the ratio, so the products are passed unreduced and no gcd is taken.
pub(super) fn rounded_radical(s: &Q, a: &Q, b: &Q, scale: &Q) -> f64 {
    let (ps, qs) = s.magnitudes();
    let (pa, qa) = a.magnitudes();
    let (pb, qb) = b.magnitudes();
    let (pt, qt) = scale.magnitudes();
    let mut u = ps.as_ref().square();
    u *= pa.as_ref();
    u *= pb.as_ref();
    u *= qt.as_ref();
    let mut v = qs.as_ref().square();
    v *= qa.as_ref();
    v *= qb.as_ref();
    v *= pt.as_ref();
    let sign = if s.signum() == Ordering::Less { -1 } else { 1 };
    crate::exact::signed_sqrt_to_f64(sign, &u, &v)
}

#[cfg(test)]
mod tests;
