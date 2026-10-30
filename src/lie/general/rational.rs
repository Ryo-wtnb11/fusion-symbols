//! Exact exceptional arithmetic in the crate's shared rational type.
pub(super) use malachite_q::Rational as Q;

/// The sign and reduced radicand of `s·√(a·b/scale)`.
fn signed_radicand(s: &Q, a: &Q, b: &Q, scale: &Q) -> (i8, Q) {
    (if s < &0 { -1 } else { 1 }, s * s * a * b / scale)
}

#[cfg(test)]
pub(super) fn radical(s: &Q, a: &Q, b: &Q, scale: &Q) -> crate::exact::SignedSqrtRational {
    let (sign, r) = signed_radicand(s, a, b, scale);
    crate::exact::SignedSqrtRational::from_radicand(sign, r)
}

/// `s·√(a·b/scale)` rounded once, by the conversion of
/// [`crate::exact::SignedSqrtRational::to_f64`] applied to the reduced radicand
/// directly: round-only callers never hold the exact value.
pub(super) fn rounded_radical(s: &Q, a: &Q, b: &Q, scale: &Q) -> f64 {
    let (sign, r) = signed_radicand(s, a, b, scale);
    crate::exact::signed_sqrt_to_f64(sign, &r)
}
