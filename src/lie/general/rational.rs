//! Exact exceptional arithmetic in the crate's shared rational type.
pub(super) use malachite_q::Rational as Q;

use malachite_base::num::arithmetic::traits::Square;

/// The exact `s·√(a·b/scale)`, with the reduced radicand `s²·a·b/scale`.
#[cfg(test)]
pub(super) fn radical(s: &Q, a: &Q, b: &Q, scale: &Q) -> crate::exact::SignedSqrtRational {
    let sign = if s < &0 { -1 } else { 1 };
    crate::exact::SignedSqrtRational::from_radicand(sign, s * s * a * b / scale)
}

/// `s·√(a·b/scale)` rounded once, by the conversion of
/// [`crate::exact::SignedSqrtRational::to_f64`]: round-only callers never hold
/// the exact value. With `s = ps/qs`, `a = pa/qa`, `b = pb/qb` and
/// `scale = pt/qt` (`a, b, scale > 0`), the radicand is `U/V` with
/// `U = ps²·pa·pb·qt` and `V = qs²·qa·qb·pt`; the conversion depends only on
/// the ratio, so the products are passed unreduced and no gcd is taken.
pub(super) fn rounded_radical(s: &Q, a: &Q, b: &Q, scale: &Q) -> f64 {
    let mut u = s.numerator_ref().square();
    u *= a.numerator_ref();
    u *= b.numerator_ref();
    u *= scale.denominator_ref();
    let mut v = s.denominator_ref().square();
    v *= a.denominator_ref();
    v *= b.denominator_ref();
    v *= scale.numerator_ref();
    crate::exact::signed_sqrt_to_f64(if s < &0 { -1 } else { 1 }, &u, &v)
}
