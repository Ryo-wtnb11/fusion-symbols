//! Exceptional arithmetic and the single exact conversion at each num boundary.
use malachite_nz::integer::Integer;
use num_bigint::BigInt;
use num_rational::Ratio;

pub(super) use malachite_q::Rational as Q;

// ponytail: operands beyond 128 bits use decimal conversion; use limbs if this becomes hot.
pub(super) fn to_num(q: &Q) -> Ratio<BigInt> {
    let n = u128::try_from(q.numerator_ref())
        .map(BigInt::from)
        .unwrap_or_else(|_| q.numerator_ref().to_string().parse().unwrap());
    let d = u128::try_from(q.denominator_ref())
        .map(BigInt::from)
        .unwrap_or_else(|_| q.denominator_ref().to_string().parse().unwrap());
    // Malachite already reduced the fraction and keeps its denominator positive.
    Ratio::new_raw(if q < &0 { -n } else { n }, d)
}

pub(super) fn integer_to_num(n: &Integer) -> BigInt {
    i128::try_from(n)
        .map(BigInt::from)
        .unwrap_or_else(|_| n.to_string().parse().unwrap())
}

pub(super) fn radical(s: &Q, a: &Q, b: &Q, scale: &Q) -> crate::exact::SignedSqrtRational {
    let r = s * s * a * b / scale;
    crate::exact::SignedSqrtRational::from_radicand(if s < &0 { -1 } else { 1 }, to_num(&r))
}

#[cfg(test)]
#[test]
fn boundary_conversion_is_exact() {
    for text in [
        "0",
        "-2/3",
        "340282366920938463463374607431768211457/7",
        "-9/1361129467683753853853498429727072845829",
    ] {
        let q: Q = text.parse().unwrap();
        let expected: Ratio<BigInt> = text.parse().unwrap();
        assert_eq!(to_num(&q), expected);
        assert_eq!(
            radical(
                &q,
                &"5/7".parse().unwrap(),
                &"11/13".parse().unwrap(),
                &"23/29".parse().unwrap()
            ),
            crate::exact::SignedSqrtRational::from_prefactor_radical(
                expected,
                Ratio::new(1595.into(), 2093.into())
            )
        );
    }
    for text in [
        "0",
        "-1",
        "340282366920938463463374607431768211457",
        "-340282366920938463463374607431768211457",
    ] {
        assert_eq!(
            integer_to_num(&text.parse().unwrap()),
            text.parse::<BigInt>().unwrap()
        );
    }
}
