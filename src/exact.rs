//! Exact value type for SU(2) recoupling coefficients.
//!
//! Every Wigner 3j/6j/Clebsch-Gordan value equals `sign * sqrt(radicand)` with
//! `radicand` a nonnegative rational over big integers. That is the closed
//! form of the Racah expressions: a triangle/dimension product under a square
//! root times a rational alternating sum. We keep the value in this exact form
//! all the way through and round to `f64` exactly once, in [`SignedSqrtRational::to_f64`].

use std::sync::Arc;

#[cfg(test)]
use malachite_base::num::arithmetic::traits::Pow;
use malachite_base::num::{
    arithmetic::traits::FloorSqrt,
    basic::traits::{One, Zero},
    logic::traits::SignificantBits,
};
#[cfg(test)]
use malachite_nz::integer::Integer;
use malachite_nz::natural::Natural;
use malachite_q::Rational;

/// Signed square root of a nonnegative rational: the value `sign * sqrt(radicand)`.
///
/// `sign` is `-1`, `0`, or `+1`; `radicand` is a reduced nonnegative
/// `Rational`. The zero value has `sign == 0` and `radicand == 0`.
///
/// Exact equality (`Eq`) compares `sign` and `radicand`, so two values are
/// equal iff they denote the same real number (the representation is canonical
/// because `radicand` is reduced and `sign == 0` iff `radicand == 0`).
/// Clones share the immutable rational payload; arithmetic uses copy-on-write.
/// Zero has no allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedSqrtRational {
    sign: i8,
    radicand: Option<Arc<Rational>>,
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
    pub(crate) fn from_prefactor_radical(s: Rational, d: Rational) -> Self {
        if (s == 0) || (d == 0) {
            return Self::zero();
        }
        let sign: i8 = if s < 0 { -1 } else { 1 };
        let d = if d < 0 { -d } else { d };
        let radicand = (&s * &s) * d;
        SignedSqrtRational {
            sign,
            radicand: Some(Arc::new(radicand)),
        }
    }

    /// Accept an already reduced nonnegative radicand at the exceptional boundary.
    pub(crate) fn from_radicand(sign: i8, r: Rational) -> Self {
        if r == 0 {
            Self::zero()
        } else {
            assert!(!(r < 0));
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
    pub fn radicand(&self) -> &Rational {
        static ZERO: Rational = Rational::ZERO;
        self.radicand.as_deref().unwrap_or(&ZERO)
    }

    /// The signed square `sign * radicand`. This is the natural exact
    /// comparison key: it equals the reference crate's `signed_sq`, so two
    /// coefficients agree exactly iff their signed squares are equal.
    pub fn signed_square(&self) -> Rational {
        match self.sign {
            0 => Rational::ZERO,
            1 => self.radicand().clone(),
            _ => -self.radicand(),
        }
    }

    /// Multiply the value by `sqrt(n)` for a nonnegative integer `n`
    /// (`radicand *= n`, sign unchanged). Used to fold a dimension factor
    /// `sqrt(2 j3 + 1)` into a coefficient exactly.
    pub(crate) fn times_sqrt_int(mut self, n: u64) -> Self {
        if n == 0 || self.sign == 0 {
            return Self::zero();
        }
        *Arc::make_mut(self.radicand.as_mut().expect("nonzero radicand")) *= Rational::from(n);
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
        let n = self.radicand().numerator_ref();
        let d = self.radicand().denominator_ref();
        // radicand is nonnegative and reduced; numer/denom are nonnegative.
        let mag = sqrt_ratio_to_f64(n, d);
        if self.sign < 0 {
            -mag
        } else {
            mag
        }
    }
}

/// Nearest `f64` to `sqrt(n/d)` for nonnegative big integers, ties to even.
fn sqrt_ratio_to_f64(n: &Natural, d: &Natural) -> f64 {
    if n == &0 {
        return 0.0;
    }
    let nb = n.significant_bits() as i64;
    let db = d.significant_bits() as i64;
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
        if scaled == 0 {
            // g too small (only possible if n/d rounds below 1 at this scale);
            // push the scale up and retry.
            g += 8;
            continue;
        }
        let q = scaled.floor_sqrt();
        let bits = q.significant_bits() as i64;
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

    let bits = q.significant_bits() as i64;
    let drop = bits - 53; // >= 1 since bits in 54..=56
    let mant: Natural = &q >> (drop as u64);
    // value ≈ mant * 2^(drop - exp)
    let value_exp = drop - exp;

    // Exact midpoint decision: compare sqrt(n/d) against (mant + 1/2)*2^value_exp.
    // Squared: n/d  vs  (2*mant+1)^2 * 2^(2*value_exp - 2).
    // Cross-multiply by d and clear the sign of the power of two.
    let two_m1 = (&mant << 1u32) + Natural::ONE;
    let mid_sq_int = &two_m1 * &two_m1; // (2*mant+1)^2
    let p = 2 * value_exp - 2;
    let (lhs, rhs) = if p >= 0 {
        (n.clone(), (&mid_sq_int * d) << (p as u64))
    } else {
        (n << ((-p) as u64), &mid_sq_int * d)
    };

    let chosen: Natural = match lhs.cmp(&rhs) {
        std::cmp::Ordering::Less => mant, // below midpoint -> round down
        std::cmp::Ordering::Greater => mant + Natural::ONE, // above -> round up
        std::cmp::Ordering::Equal => {
            // Exact midpoint: round half to even.
            if (&mant & Natural::ONE) == 0 {
                mant
            } else {
                mant + Natural::ONE
            }
        }
    };

    // `chosen` has at most 54 bits (53, or 2^53 on carry) so it is exact in f64.
    let mant_f = natural_to_f64_exact(&chosen);
    scale_pow2(mant_f, value_exp)
}

/// Exact `f64` of a `Natural` that fits in 54 bits.
fn natural_to_f64_exact(x: &Natural) -> f64 {
    // Fits in u64 (<= 2^54); u64->f64 is exact for values < 2^53 and exact for
    // 2^53 and 2^54 as well since they are powers of two with a single bit.
    u64::try_from(x).map_or(f64::INFINITY, |n| n as f64)
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
fn div(a: u128, b: u128) -> u128 {
    match (u64::try_from(a), u64::try_from(b)) {
        (Ok(a), Ok(b)) => u128::from(a / b),
        _ => a / b,
    }
}

/// Euclid in hardware `u64` whenever both operands fit; otherwise binary gcd,
/// since a `u128` division is a slow library call and shifts are not.
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
        static FACT: RefCell<Vec<Integer>> = RefCell::new(vec![Integer::ONE]);
    }

    /// `n!` as a big integer, memoized per thread in a growing table. Retained
    /// only as an independent reference for [`factorial_table_grows`]; the
    /// production factorial lives in the prime-factorized engine.
    fn factorial(n: u64) -> Integer {
        FACT.with(|cell| {
            let mut table = cell.borrow_mut();
            while (table.len() as u64) <= n {
                let len = table.len();
                let next = &table[len - 1] * Integer::from(len as u64);
                table.push(next);
            }
            table[n as usize].clone()
        })
    }

    fn ssr(s: i64, num: i64, den: i64) -> SignedSqrtRational {
        SignedSqrtRational::from_prefactor_radical(
            Rational::from_integers(Integer::from(s), Integer::from(1)),
            Rational::from_integers(Integer::from(num), Integer::from(den)),
        )
    }

    #[test]
    fn zero_is_canonical() {
        let z = SignedSqrtRational::zero();
        assert_eq!(z.sign(), 0);
        assert_eq!(z.to_f64(), 0.0);
        // 0 * sqrt(anything) collapses to the canonical zero.
        let also_zero =
            SignedSqrtRational::from_prefactor_radical(Rational::ZERO, Rational::from(5));
        assert_eq!(z, also_zero);
    }

    #[test]
    fn sign_and_signed_square() {
        // value = -1 * sqrt(3/4): sign -1, radicand 3/4, signed_square -3/4.
        let v = ssr(-1, 3, 4);
        assert_eq!(v.sign(), -1);
        assert_eq!(
            *v.radicand(),
            Rational::from_integers(Integer::from(3), Integer::from(4))
        );
        assert_eq!(
            v.signed_square(),
            Rational::from_integers(Integer::from(-3), Integer::from(4))
        );
    }

    #[test]
    fn prefactor_folds_into_radicand() {
        // value = (2/3) * sqrt(5) has radicand (4/9)*5 = 20/9.
        let v = SignedSqrtRational::from_prefactor_radical(
            Rational::from_integers(Integer::from(2), Integer::from(3)),
            Rational::from(5),
        );
        assert_eq!(v.sign(), 1);
        assert_eq!(
            *v.radicand(),
            Rational::from_integers(Integer::from(20), Integer::from(9))
        );
    }

    #[test]
    fn rounding_matches_num_baseline() {
        // Literal bits captured from origin/main fe4100d's num-based
        // sqrt_ratio_to_f64 (2026-10-28), before this arithmetic migration.
        let one = Natural::ONE;
        let mid = (&one << 53u32) + &one;
        let mid_odd = (&one << 53u32) + Natural::from(3u32);
        for (n, d, bits) in [
            (Natural::ZERO, one.clone(), 0x0000000000000000),
            (one.clone(), one.clone(), 0x3ff0000000000000),
            (2u32.into(), 3u32.into(), 0x3fea20bd700c2c3e),
            (4u32.into(), 9u32.into(), 0x3fe5555555555555),
            (&one << 1200u32, one.clone(), 0x6570000000000000),
            (
                Natural::from(3u32) << 1200u32,
                7u32.into(),
                0x6564f2ec413cb52b,
            ),
            (one.clone(), &one << 1200u32, 0x1a70000000000000),
            (
                3u32.into(),
                Natural::from(7u32) << 1200u32,
                0x1a64f2ec413cb52b,
            ),
            // Midpoints at 1 + 2^-53 (even down) and 1 + 3*2^-53 (even up),
            // plus radicands immediately below/above the first midpoint.
            (&mid * &mid, &one << 106u32, 0x3ff0000000000000),
            (&mid * &mid - &one, &one << 106u32, 0x3ff0000000000000),
            (&mid * &mid + &one, &one << 106u32, 0x3ff0000000000001),
            (&mid_odd * &mid_odd, &one << 106u32, 0x3ff0000000000002),
            (one.clone(), &one << 2148u32, 0x0000000000000001),
        ] {
            assert_eq!(sqrt_ratio_to_f64(&n, &d).to_bits(), bits, "sqrt({n}/{d})");
        }
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
                    Rational::ONE,
                    Rational::from_integers(Integer::from(num), Integer::from(den)),
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
        let big: Integer = Integer::from(2u64).pow(200) * Integer::from(3);
        let v =
            SignedSqrtRational::from_prefactor_radical(Rational::ONE, Rational::from(big.clone()));
        let got = v.to_f64();
        let want = (3.0f64).sqrt() * 2f64.powi(100);
        assert!(
            (got - want).abs() <= want * 4.0 * f64::EPSILON,
            "got={got} want={want}"
        );
    }

    #[test]
    fn factorial_table_grows() {
        assert_eq!(factorial(0), Integer::ONE);
        assert_eq!(factorial(5), Integer::from(120));
        assert_eq!(factorial(10), Integer::from(3_628_800));
        // reproducible under repeated / out-of-order queries
        assert_eq!(factorial(3), Integer::from(6));
    }
}
