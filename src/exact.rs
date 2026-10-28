//! Exact value type for SU(2) recoupling coefficients.
//!
//! Every Wigner 3j/6j/Clebsch-Gordan value equals `sign * sqrt(radicand)` with
//! `radicand` a nonnegative rational over big integers. That is the closed
//! form of the Racah expressions: a triangle/dimension product under a square
//! root times a rational alternating sum. We keep the value in this exact form
//! all the way through and round to `f64` exactly once, in [`SignedSqrtRational::to_f64`].

use std::sync::Arc;

use num_bigint::{BigInt, BigUint};
use num_rational::Ratio;
use num_traits::{One, Signed, Zero};

/// Signed square root of a nonnegative rational: the value `sign * sqrt(radicand)`.
///
/// `sign` is `-1`, `0`, or `+1`; `radicand` is a reduced nonnegative
/// `Ratio<BigInt>`. The zero value has `sign == 0` and `radicand == 0`.
///
/// Exact equality (`Eq`) compares `sign` and `radicand`, so two values are
/// equal iff they denote the same real number (the representation is canonical
/// because `radicand` is reduced and `sign == 0` iff `radicand == 0`).
/// Clones share the immutable rational payload; arithmetic uses copy-on-write.
/// Zero has no allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedSqrtRational {
    sign: i8,
    radicand: Option<Arc<Ratio<BigInt>>>,
}

impl SignedSqrtRational {
    /// The exact zero value.
    pub fn zero() -> Self {
        SignedSqrtRational {
            sign: 0,
            radicand: None,
        }
    }

    /// Construct `s * sqrt(d)` from a rational prefactor `s` (which carries the
    /// sign) and a nonnegative rational `d` under the root.
    ///
    /// The stored form is `sign = sign(s)`, `radicand = s^2 * d`. `d` must be
    /// nonnegative (all triangle/dimension products are); a negative `d` is
    /// clamped to its absolute value rather than panicking, but callers never
    /// pass one.
    pub(crate) fn from_prefactor_radical(s: Ratio<BigInt>, d: Ratio<BigInt>) -> Self {
        if s.is_zero() || d.is_zero() {
            return Self::zero();
        }
        let sign: i8 = if s.is_negative() { -1 } else { 1 };
        let d = if d.is_negative() { -d } else { d };
        let radicand = (&s * &s) * d;
        SignedSqrtRational {
            sign,
            radicand: Some(Arc::new(radicand)),
        }
    }

    /// Form `s * sqrt(a * b / scale)` with checked u128 products and one gcd;
    /// fall back to BigInt formation on any oversized operand or product.
    #[cfg(all(test, feature = "lie-generated"))]
    pub(crate) fn from_prefactor_radical_ratio(
        s: &Ratio<BigInt>,
        a: &Ratio<BigInt>,
        b: &Ratio<BigInt>,
        scale: &Ratio<BigInt>,
    ) -> Self {
        use num_traits::ToPrimitive;

        let small = || {
            let (mut n, mut d) = (1u128, 1u128);
            for q in [s, s, a, b] {
                n = n.checked_mul(q.numer().magnitude().to_u128()?)?;
                d = d.checked_mul(q.denom().magnitude().to_u128()?)?;
            }
            n = n.checked_mul(scale.denom().magnitude().to_u128()?)?;
            d = d.checked_mul(scale.numer().magnitude().to_u128()?)?;
            if d == 0 {
                return None;
            }
            let g = gcd(n, d);
            // Positive denominator and division by gcd establish Ratio's invariant.
            Some(Ratio::new_raw(
                BigInt::from(div(n, g)),
                BigInt::from(div(d, g)),
            ))
        };
        match small() {
            Some(r) if r.is_zero() => Self::zero(),
            Some(r) => Self {
                sign: if s.is_negative() { -1 } else { 1 },
                radicand: Some(Arc::new(r)),
            },
            None => Self::from_prefactor_radical(s.clone(), a * b / scale),
        }
    }

    /// Accept an already reduced nonnegative radicand at the exceptional boundary.
    #[cfg(feature = "lie-generated")]
    pub(crate) fn from_radicand(sign: i8, r: Ratio<BigInt>) -> Self {
        if r.is_zero() {
            Self::zero()
        } else {
            assert!(!r.is_negative());
            assert!(sign == -1 || sign == 1);
            Self {
                sign,
                radicand: Some(Arc::new(r)),
            }
        }
    }

    /// The sign: `-1`, `0`, or `+1`.
    pub fn sign(&self) -> i8 {
        self.sign
    }

    /// The nonnegative radicand `r` such that the value is `sign * sqrt(r)`.
    pub fn radicand(&self) -> &Ratio<BigInt> {
        static ZERO: Ratio<BigInt> = Ratio::new_raw(BigInt::ZERO, BigInt::ONE);
        self.radicand.as_deref().unwrap_or(&ZERO)
    }

    /// The signed square `sign * radicand`. This is the natural exact
    /// comparison key: it equals the reference crate's `signed_sq`, so two
    /// coefficients agree exactly iff their signed squares are equal.
    pub fn signed_square(&self) -> Ratio<BigInt> {
        match self.sign {
            0 => Ratio::zero(),
            1 => self.radicand().clone(),
            _ => -self.radicand().clone(),
        }
    }

    /// Multiply the value by `sqrt(n)` for a nonnegative integer `n`
    /// (`radicand *= n`, sign unchanged). Used to fold a dimension factor
    /// `sqrt(2 j3 + 1)` into a coefficient exactly.
    pub(crate) fn times_sqrt_int(mut self, n: u64) -> Self {
        if n == 0 || self.sign == 0 {
            return Self::zero();
        }
        *Arc::make_mut(self.radicand.as_mut().expect("nonzero radicand")) *=
            Ratio::from(BigInt::from(n));
        self
    }

    /// Negate the value (flip the sign).
    pub(crate) fn neg_value(mut self) -> Self {
        self.sign = -self.sign;
        self
    }

    /// Correctly-rounded conversion to `f64` — the single rounding point.
    ///
    /// The value is `sign * sqrt(N/D)` with `N, D` nonnegative big integers.
    /// We return the `f64` nearest to that real number (ties to even), i.e.
    /// error <= 0.5 ulp for any result in the normal `f64` range. A result that
    /// lands in the subnormal range would pick up a second rounding in the
    /// `scale_pow2` exponent chunking below; genuine Wigner coefficients are
    /// O(1) and never subnormal, so that path is unreachable in practice.
    ///
    /// Method (integer square root with scaling, then an exact midpoint test):
    /// pick an even scale `2g` so that `Q = floor(sqrt(floor(N*2^(2g)/D)))`
    /// has 54..=56 bits. Then `Q ≈ sqrt(N/D) * 2^g` and the inner division
    /// floor perturbs `Q` by less than `1/(2*sqrt(scaled)) < 2^-54` of a unit,
    /// so `Q` is the exact `floor(sqrt(N/D) * 2^g)`. Let `mant = Q >> (bits-53)`
    /// be the 53-bit significand floor and `exp` its binary exponent. The
    /// correctly rounded significand is `mant` or `mant+1` depending on whether
    /// the true value lies below or above the midpoint `(mant + 1/2) * 2^exp`.
    /// We decide this by the exact big-integer comparison of `N/D` against
    /// `(2*mant+1)^2 * 2^(2*exp-2)` — no floating point enters the decision, so
    /// the only rounding is the final exactly-representable `mant * 2^exp`.
    pub fn to_f64(&self) -> f64 {
        if self.sign == 0 {
            return 0.0;
        }
        let n = self.radicand().numer();
        let d = self.radicand().denom();
        // radicand is nonnegative and reduced; numer/denom are nonnegative.
        let n = n.magnitude(); // BigUint
        let d = d.magnitude();
        let mag = sqrt_ratio_to_f64(n, d);
        if self.sign < 0 {
            -mag
        } else {
            mag
        }
    }
}

/// Nearest `f64` to `sqrt(n/d)` for nonnegative big integers, ties to even.
fn sqrt_ratio_to_f64(n: &BigUint, d: &BigUint) -> f64 {
    if n.is_zero() {
        return 0.0;
    }
    let nb = n.bits() as i64;
    let db = d.bits() as i64;
    // floor(log2(sqrt(n/d))) ≈ (nb - db) / 2; choose g so Q ≈ sqrt(n/d)*2^g
    // lands around 54 bits.
    let mut g: i64 = 54 - (nb - db) / 2;
    let (q, exp) = loop {
        let two_g = 2 * g;
        // scaled = floor(n * 2^(2g) / d)
        let scaled = if two_g >= 0 {
            (n << (two_g as u64)) / d
        } else {
            n / (d << ((-two_g) as u64))
        };
        if scaled.is_zero() {
            // g too small (only possible if n/d rounds below 1 at this scale);
            // push the scale up and retry.
            g += 8;
            continue;
        }
        let q = scaled.sqrt();
        let bits = q.bits() as i64;
        if bits < 54 {
            g += (54 - bits + 1) / 2 + 1;
            continue;
        }
        if bits > 56 {
            g -= (bits - 56 + 1) / 2 + 1;
            continue;
        }
        break (q, g);
    };

    let bits = q.bits() as i64;
    let drop = bits - 53; // >= 1 since bits in 54..=56
    let mant: BigUint = &q >> (drop as u64);
    // value ≈ mant * 2^(drop - exp)
    let value_exp = drop - exp;

    // Exact midpoint decision: compare sqrt(n/d) against (mant + 1/2)*2^value_exp.
    // Squared: n/d  vs  (2*mant+1)^2 * 2^(2*value_exp - 2).
    // Cross-multiply by d and clear the sign of the power of two.
    let two_m1 = (&mant << 1u32) + BigUint::one();
    let mid_sq_int = &two_m1 * &two_m1; // (2*mant+1)^2
    let p = 2 * value_exp - 2;
    let (lhs, rhs) = if p >= 0 {
        (n.clone(), (&mid_sq_int * d) << (p as u64))
    } else {
        (n << ((-p) as u64), &mid_sq_int * d)
    };

    let chosen: BigUint = match lhs.cmp(&rhs) {
        std::cmp::Ordering::Less => mant, // below midpoint -> round down
        std::cmp::Ordering::Greater => mant + BigUint::one(), // above -> round up
        std::cmp::Ordering::Equal => {
            // Exact midpoint: round half to even.
            if (&mant & BigUint::one()).is_zero() {
                mant
            } else {
                mant + BigUint::one()
            }
        }
    };

    // `chosen` has at most 54 bits (53, or 2^53 on carry) so it is exact in f64.
    let mant_f = biguint_to_f64_exact(&chosen);
    scale_pow2(mant_f, value_exp)
}

/// Exact `f64` of a `BigUint` that fits in 54 bits.
fn biguint_to_f64_exact(x: &BigUint) -> f64 {
    // Fits in u64 (<= 2^54); u64->f64 is exact for values < 2^53 and exact for
    // 2^53 and 2^54 as well since they are powers of two with a single bit.
    let digits = x.to_u64_digits();
    match digits.as_slice() {
        [] => 0.0,
        [lo] => *lo as f64,
        // Unreachable for <= 54-bit inputs, but stay total.
        _ => f64::INFINITY,
    }
}

/// Multiply `x` by `2^exp` without libm, staying exact within the normal range.
fn scale_pow2(x: f64, exp: i64) -> f64 {
    // Domain: coefficients are O(1); exp stays well inside f64 range. Chunk the
    // exponent so each factor is an exact power of two.
    let mut r = x;
    let mut e = exp;
    while e > 1023 {
        r *= f64::from_bits(0x7FEu64 << 52); // 2^1023
        e -= 1023;
    }
    while e < -1022 {
        r *= f64::from_bits(1u64 << 52); // 2^-1022 (smallest normal)
        e += 1022;
    }
    r * two_pow_i(e)
}

/// `2^e` for `-1022 <= e <= 1023`, exact.
fn two_pow_i(e: i64) -> f64 {
    let biased = (e + 1023) as u64;
    f64::from_bits(biased << 52)
}

/// `∏ nᵢ / ∏ dᵢ` when it is an integer that fits `u128` together with every
/// intermediate reduced fraction; `None` otherwise (callers fall back to
/// exact big-integer arithmetic).
///
/// The running value stays in lowest terms: each incoming pair is reduced
/// against itself and crosswise against the running state before
/// multiplying, so intermediates are as small as the factor order allows.
#[cfg(feature = "lie-generated")]
pub(crate) fn checked_integral_quotient(
    factors: impl IntoIterator<Item = (u128, u128)>,
) -> Option<u128> {
    let (mut num, mut den) = (1u128, 1u128);
    for (n, d) in factors {
        if d == 0 {
            return None;
        }
        let g = gcd(n, d);
        let (n, d) = (div(n, g), div(d, g));
        let (g_n, g_d) = (gcd(n, den), gcd(num, d));
        num = div(num, g_d).checked_mul(div(n, g_n))?;
        den = div(den, g_n).checked_mul(div(d, g_d))?;
    }
    (den == 1).then_some(num)
}

/// `a / b`, in a hardware `u64` division whenever both operands fit.
#[cfg(feature = "lie-generated")]
fn div(a: u128, b: u128) -> u128 {
    match (u64::try_from(a), u64::try_from(b)) {
        (Ok(a), Ok(b)) => u128::from(a / b),
        _ => a / b,
    }
}

/// Euclid in hardware `u64` whenever both operands fit; otherwise binary gcd,
/// since a `u128` division is a slow library call and shifts are not.
#[cfg(feature = "lie-generated")]
fn gcd(mut a: u128, mut b: u128) -> u128 {
    if let (Ok(mut x), Ok(mut y)) = (u64::try_from(a), u64::try_from(b)) {
        while y != 0 {
            (x, y) = (y, x % y);
        }
        return u128::from(x);
    }
    if a == 0 || b == 0 {
        return a | b;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "lie-generated")]
    #[test]
    fn radical_ratio_matches_bigint_formation() {
        let q = |n: i64, d: i64| Ratio::new(BigInt::from(n), BigInt::from(d));
        // Drift check against the original exact formula, including zero, signs,
        // oversized operands and overflow in numerator/denominator products.
        for s in [
            q(0, 1),
            q(-2, 3),
            Ratio::new(BigInt::from(u128::MAX), BigInt::from(3)),
            Ratio::new(BigInt::one(), BigInt::one() << 80),
            Ratio::new(BigInt::one(), BigInt::one() << 129),
        ] {
            for a in [
                q(5, 7),
                Ratio::from(BigInt::one() << 80),
                Ratio::new(BigInt::one(), BigInt::one() << 80),
            ] {
                for b in [q(11, 13), q(-17, 19)] {
                    for scale in [q(23, 29), Ratio::new(BigInt::one() << 140, 31.into())] {
                        let want =
                            SignedSqrtRational::from_prefactor_radical(s.clone(), &a * &b / &scale);
                        let got =
                            SignedSqrtRational::from_prefactor_radical_ratio(&s, &a, &b, &scale);
                        assert_eq!(got, want);
                        assert_eq!(got.to_f64().to_bits(), want.to_f64().to_bits());
                    }
                }
            }
        }
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn checked_integral_quotient_is_exact_up_to_u128() {
        fn q<const N: usize>(f: [(u128, u128); N]) -> Option<u128> {
            checked_integral_quotient(f)
        }
        assert_eq!(q([]), Some(1));
        // SU(3) octet: 2/1 · 4/2 · 2/1.
        assert_eq!(q([(2, 1), (4, 2), (2, 1)]), Some(8));
        assert_eq!(q([(3, 2)]), None, "non-integral");
        assert_eq!(q([(1, 0)]), None);
        assert_eq!(q([(u128::MAX, 1)]), Some(u128::MAX));
        assert_eq!(q([(u128::MAX, 1), (2, 1)]), None, "result overflows");
        assert_eq!(q([(u128::MAX, 1), (6, 6)]), Some(u128::MAX));
        // Only crosswise reduction against the running state keeps these in range.
        assert_eq!(q([(1 << 127, 1), (2, 4)]), Some(1 << 126));
        assert_eq!(q([(1, 3), (3 << 126, 1)]), Some(1 << 126));
        // Exact result fits, but the reduced intermediate 2¹²⁸ does not.
        assert_eq!(q([(1 << 127, 1), (2, 1), (1, 4)]), None);
        // Both the u64 and the u128 branches of gcd and division.
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(3 << 100, 3 << 91), 3 << 91);
        assert_eq!(gcd(u128::MAX, 1 << 64), 1);
        assert_eq!(gcd(0, 1 << 100), 1 << 100);
        assert_eq!(div(1 << 100, 1 << 36), 1 << 64);
        assert_eq!(div(18, 6), 3);
    }
    use std::cell::RefCell;

    thread_local! {
        static FACT: RefCell<Vec<BigInt>> = RefCell::new(vec![BigInt::one()]);
    }

    /// `n!` as a big integer, memoized per thread in a growing table. Retained
    /// only as an independent reference for [`factorial_table_grows`]; the
    /// production factorial lives in the prime-factorized engine.
    fn factorial(n: u64) -> BigInt {
        FACT.with(|cell| {
            let mut table = cell.borrow_mut();
            while (table.len() as u64) <= n {
                let len = table.len();
                let next = &table[len - 1] * BigInt::from(len as u64);
                table.push(next);
            }
            table[n as usize].clone()
        })
    }

    fn ssr(s: i64, num: i64, den: i64) -> SignedSqrtRational {
        SignedSqrtRational::from_prefactor_radical(
            Ratio::new(BigInt::from(s), BigInt::from(1)),
            Ratio::new(BigInt::from(num), BigInt::from(den)),
        )
    }

    #[test]
    fn zero_is_canonical() {
        let z = SignedSqrtRational::zero();
        assert_eq!(z.sign(), 0);
        assert_eq!(z.to_f64(), 0.0);
        // 0 * sqrt(anything) collapses to the canonical zero.
        let also_zero =
            SignedSqrtRational::from_prefactor_radical(Ratio::zero(), Ratio::from(BigInt::from(5)));
        assert_eq!(z, also_zero);
    }

    #[test]
    fn sign_and_signed_square() {
        // value = -1 * sqrt(3/4): sign -1, radicand 3/4, signed_square -3/4.
        let v = ssr(-1, 3, 4);
        assert_eq!(v.sign(), -1);
        assert_eq!(*v.radicand(), Ratio::new(BigInt::from(3), BigInt::from(4)));
        assert_eq!(
            v.signed_square(),
            Ratio::new(BigInt::from(-3), BigInt::from(4))
        );
    }

    #[test]
    fn prefactor_folds_into_radicand() {
        // value = (2/3) * sqrt(5) has radicand (4/9)*5 = 20/9.
        let v = SignedSqrtRational::from_prefactor_radical(
            Ratio::new(BigInt::from(2), BigInt::from(3)),
            Ratio::from(BigInt::from(5)),
        );
        assert_eq!(v.sign(), 1);
        assert_eq!(*v.radicand(), Ratio::new(BigInt::from(20), BigInt::from(9)));
    }

    #[test]
    fn to_f64_known_values() {
        // sqrt exact perfect squares and simple rationals.
        assert!((ssr(1, 1, 4).to_f64() - 0.5).abs() == 0.0);
        assert!((ssr(1, 9, 1).to_f64() - 3.0).abs() == 0.0);
        assert!((ssr(-1, 1, 6).to_f64() - (-(1.0f64 / 6.0).sqrt())).abs() <= f64::EPSILON);
        // 1/6 exactly: compare to correctly rounded reference.
        let got = ssr(1, 1, 6).to_f64();
        let want = (1.0f64 / 6.0).sqrt();
        assert!((got - want).abs() <= (want * f64::EPSILON));
    }

    #[test]
    fn to_f64_correctly_rounded_against_exact() {
        // For many rationals p/q, our correctly-rounded sqrt must match the
        // nearest f64 computed independently via high-precision-ish check:
        // compare with the two neighbouring doubles of the true value.
        for num in 1u64..60 {
            for den in 1u64..60 {
                let got = SignedSqrtRational::from_prefactor_radical(
                    Ratio::one(),
                    Ratio::new(BigInt::from(num), BigInt::from(den)),
                )
                .to_f64();
                let approx = (num as f64 / den as f64).sqrt();
                // got is correctly rounded; approx is within ~1 ulp. They must
                // be within 1 ulp of each other.
                let ulp = approx.abs() * f64::EPSILON;
                assert!(
                    (got - approx).abs() <= ulp + f64::MIN_POSITIVE,
                    "num={num} den={den} got={got} approx={approx}"
                );
            }
        }
    }

    #[test]
    fn to_f64_big_radicand() {
        // A radicand with hundreds of bits still rounds; sanity vs f64 path.
        let big: BigInt = BigInt::from(2u64).pow(200) * 3;
        let v = SignedSqrtRational::from_prefactor_radical(Ratio::one(), Ratio::from(big.clone()));
        let got = v.to_f64();
        let want = (3.0f64).sqrt() * 2f64.powi(100);
        assert!(
            (got - want).abs() <= want * 4.0 * f64::EPSILON,
            "got={got} want={want}"
        );
    }

    #[test]
    fn factorial_table_grows() {
        assert_eq!(factorial(0), BigInt::one());
        assert_eq!(factorial(5), BigInt::from(120));
        assert_eq!(factorial(10), BigInt::from(3_628_800));
        // reproducible under repeated / out-of-order queries
        assert_eq!(factorial(3), BigInt::from(6));
    }
}
