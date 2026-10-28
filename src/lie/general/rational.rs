//! Exact exceptional arithmetic in the crate's shared rational type.
pub(super) use malachite_q::Rational as Q;

pub(super) fn radical(s: &Q, a: &Q, b: &Q, scale: &Q) -> crate::exact::SignedSqrtRational {
    let r = s * s * a * b / scale;
    crate::exact::SignedSqrtRational::from_radicand(if s < &0 { -1 } else { 1 }, r)
}
