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
    #[cfg(test)]
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
    /// error <= 0.5 ulp over the whole `f64` range: subnormal results keep the
    /// significand bits the fixed subnormal spacing allows and are rounded once,
    /// and values below half the smallest subnormal give a signed zero.
    ///
    /// Method (integer square root with scaling, then an exact midpoint test):
    /// pick an even scale `2g` so that `Q = floor(sqrt(floor(N*2^(2g)/D)))`
    /// has 54..=56 bits. Then `Q ≈ sqrt(N/D) * 2^g` and the inner division
    /// floor perturbs `Q` by less than `1/(2*sqrt(scaled)) < 2^-54` of a unit,
    /// so `Q` is the exact `floor(sqrt(N/D) * 2^g)`. Let `mant = Q >> (bits-53)`
    /// be the 53-bit significand floor (fewer bits below `2^-1022`, where the
    /// spacing is `2^-1074`) and `exp` its binary exponent. The
    /// correctly rounded significand is `mant` or `mant+1` depending on whether
    /// the true value lies below or above the midpoint `(mant + 1/2) * 2^exp`.
    /// We decide this by the exact big-integer comparison of `N/D` against
    /// `(2*mant+1)^2 * 2^(2*exp-2)` — no floating point enters the decision, so
    /// the only rounding is the final exactly-representable `mant * 2^exp`.
    pub fn to_f64(&self) -> f64 {
        let r = self.radicand();
        signed_sqrt_to_f64(self.sign, r.numerator_ref(), r.denominator_ref())
    }
}

/// The value of [`SignedSqrtRational::to_f64`] for `sign * sqrt(n/d)`, with
/// `n, d` nonnegative integers (`d > 0`) that need not be coprime, without
/// materializing the exact value. The result depends only on the ratio
/// `n/d` (see [`sqrt_ratio_to_f64`]). The sign is applied after rounding the
/// magnitude, so a nonzero value that underflows keeps its signed zero.
pub(crate) fn signed_sqrt_to_f64(sign: i8, n: &Natural, d: &Natural) -> f64 {
    if sign == 0 {
        return 0.0;
    }
    let mag = sqrt_ratio_to_f64(n, d);
    if sign < 0 {
        -mag
    } else {
        mag
    }
}

/// Nearest `f64` to `sqrt(n/d)` for nonnegative big integers, ties to even.
///
/// The result is a function of the ratio `n/d` alone, so `(k·n, k·d)` gives the
/// same bits for every `k > 0`. With `x = sqrt(n/d)`, `floor(n·2^(2g)/d) =
/// floor(x²·2^(2g))` and `floor(sqrt(floor(y))) = floor(sqrt(y))`, so `q =
/// floor(x·2^g)`. The scale `g` the loop settles on may depend on the bit
/// lengths of `n` and `d` (several `g` can give 54..=56 bits), but `q` has
/// `L + g` bits with `L = floor(log2 x) + 1`, so `value_exp = bits − 53 − g =
/// L − 53` and `mant = floor(q / 2^(bits−53)) = floor(x·2^(−value_exp))` do
/// not: the 53-bit floor and its exponent are those of `x`. The midpoint test
/// is an exact cross-multiplication, and the final `scale_pow2(mant,
/// value_exp)` therefore sees the same operands for every representation of
/// the ratio. The subnormal truncation to spacing `2^-1074` depends only on
/// `value_exp` and `mant`, so it preserves this.
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
    let mut mant: Natural = &q >> (drop as u64);
    // value ≈ mant * 2^(drop - exp)
    let mut value_exp = drop - exp;
    // Below 2^-1022 the f64 spacing is fixed at 2^-1074, so the significand
    // keeps fewer than 53 bits: floor(floor(y) / 2^s) = floor(y / 2^s) gives
    // floor(x·2^1074). Rounding once here leaves `scale_pow2` an exact scale;
    // a carry to 2^52 lands exactly on the smallest normal.
    if value_exp < MIN_ULP_EXP {
        mant >>= (MIN_ULP_EXP - value_exp) as u64;
        value_exp = MIN_ULP_EXP;
    }

    // Exact midpoint decision: compare sqrt(n/d) against (mant + 1/2)*2^value_exp.
    // Squared: n/d  vs  (2*mant+1)^2 * 2^(2*value_exp - 2).
    // Cross-multiply by d and clear the sign of the power of two.
    let two_m1 = (&mant << 1u32) + Natural::ONE;
    let mid_sq_int = &two_m1 * &two_m1; // (2*mant+1)^2
    let p = 2 * value_exp - 2;
    let rhs = mid_sq_int * d;
    let ord = if p >= 0 {
        n.cmp(&(rhs << (p as u64)))
    } else {
        (n << ((-p) as u64)).cmp(&rhs)
    };

    let chosen: Natural = match ord {
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

    // `chosen` has at most 54 bits (53, or 2^53 on carry) so it is exact in f64,
    // and `chosen · 2^value_exp` is a representable f64 (or overflows).
    let mant_f = natural_to_f64_exact(&chosen);
    scale_pow2(mant_f, value_exp)
}

/// Exponent of the smallest subnormal `f64`, `2^-1074`: the finest spacing of
/// the `f64` grid.
const MIN_ULP_EXP: i64 = -1074;

/// Exact `f64` of a `Natural` that fits in 54 bits.
fn natural_to_f64_exact(x: &Natural) -> f64 {
    // Fits in u64 (<= 2^54); u64->f64 is exact for values < 2^53 and exact for
    // 2^53 and 2^54 as well since they are powers of two with a single bit.
    u64::try_from(x).map_or(f64::INFINITY, |n| n as f64)
}

/// Multiply `x` by `2^exp` without libm. Exact whenever the product is a
/// representable `f64`, which the caller guarantees below the overflow range.
fn scale_pow2(x: f64, exp: i64) -> f64 {
    // Chunk the exponent so each factor is an exact power of two.
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

    /// The shared converter at its rounding boundaries, with expected bits
    /// derived by hand. Each radicand is `n / 2^k`, so `sqrt` is the dyadic
    /// `sqrt(n) / 2^(k/2)` when `n` is a square.
    #[test]
    fn signed_sqrt_to_f64_rounding_boundaries() {
        let two = |k: u64| Integer::from(2u64).pow(k);
        let r = |n: Integer, k: u64| Rational::from_integers(n, two(k));
        let sq = |x: Integer| &x * &x;
        let one = Integer::ONE;
        let e = two(53);
        let cases = [
            // 1 + 2^-53: midpoint of 1 and 1 + 2^-52, even significand below.
            (r(sq(&e + &one), 106), 0x3FF0_0000_0000_0000u64),
            (r(sq(&e + &one) + &one, 106), 0x3FF0_0000_0000_0001),
            (r(sq(&e + &one) - &one, 106), 0x3FF0_0000_0000_0000),
            // 1 + 3·2^-53: midpoint of an odd significand and the even one above.
            (r(sq(&e + Integer::from(3)), 106), 0x3FF0_0000_0000_0002),
            (
                r(sq(&e + Integer::from(3)) + &one, 106),
                0x3FF0_0000_0000_0002,
            ),
            (
                r(sq(&e + Integer::from(3)) - &one, 106),
                0x3FF0_0000_0000_0001,
            ),
            // 2 - 2^-53: midpoint of the all-ones significand below 2 and 2;
            // the tie rounds to even and carries into the exponent.
            (r(sq(two(54) - &one), 106), 0x4000_0000_0000_0000),
            (r(sq(two(54) - &one) + &one, 106), 0x4000_0000_0000_0000),
            (r(sq(two(54) - &one) - &one, 106), 0x3FFF_FFFF_FFFF_FFFF),
            // 2^-1070, an exact subnormal.
            (r(one.clone(), 2140), 1 << 4),
            // 2^-1100 underflows to a zero that keeps the sign.
            (r(one.clone(), 2200), 0),
        ];
        // Unreduced representations `(k·n, k·d)` of each ratio: small and large
        // powers of two, small odd, and large non-powers-of-two.
        let ks: Vec<Natural> = [
            Integer::from(2),
            Integer::from(3),
            two(64),
            two(1000),
            Integer::from(3u32).pow(150),
            two(127) - &one,
            Integer::from(10u32).pow(40) + Integer::from(7),
        ]
        .iter()
        .map(|k| Natural::try_from(k).unwrap())
        .collect();
        for (radicand, bits) in cases {
            let (n, d) = (radicand.numerator_ref(), radicand.denominator_ref());
            for sign in [1i8, -1] {
                let want = if sign < 0 { bits | 1 << 63 } else { bits };
                let got = signed_sqrt_to_f64(sign, n, d);
                assert_eq!(got.to_bits(), want, "{sign} sqrt({radicand})");
                let exact = SignedSqrtRational::from_radicand(sign, radicand.clone());
                assert_eq!(exact.to_f64().to_bits(), want, "{sign} sqrt({radicand})");
                for k in &ks {
                    let got = signed_sqrt_to_f64(sign, &(k * n), &(k * d));
                    assert_eq!(got.to_bits(), want, "{sign} sqrt({k}·{radicand}/{k})");
                }
            }
        }
        assert_eq!(
            signed_sqrt_to_f64(0, &Natural::ZERO, &Natural::ONE).to_bits(),
            0
        );
        for k in &ks {
            assert_eq!(signed_sqrt_to_f64(1, &Natural::ZERO, k).to_bits(), 0);
        }
        assert_eq!(SignedSqrtRational::zero().to_f64().to_bits(), 0);
    }

    /// Unreduced `(k·n, k·d)` rounds like the reduced `n/d` over random ratios
    /// whose bit lengths straddle the converter's 54..=56-bit window and the
    /// subnormal range.
    #[test]
    fn unreduced_ratio_rounds_like_reduced() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x256e);
        let mut nat = |bits: u64| -> Natural {
            let mut x = Natural::ZERO;
            for _ in 0..bits.div_ceil(64) {
                x = (x << 64u32) + Natural::from(rng.gen::<u64>());
            }
            (x >> (bits.div_ceil(64) * 64 - bits)) | (Natural::ONE << (bits - 1))
        };
        for i in 0..4000u64 {
            let (nb, db, kb) = (1 + i % 97 * 7, 1 + (i / 97) % 41 * 13, 1 + i % 61 * 5);
            let n = nat(nb);
            // Every tenth denominator puts the value deep in the subnormal range.
            let d = if i % 10 == 0 {
                nat(db) << 2120u32
            } else {
                nat(db)
            };
            let k = nat(kb);
            let r = Rational::from_naturals(n.clone(), d.clone());
            let want = signed_sqrt_to_f64(-1, r.numerator_ref(), r.denominator_ref());
            let got = signed_sqrt_to_f64(-1, &(&k * &n), &(&k * &d));
            assert_eq!(got.to_bits(), want.to_bits(), "sqrt({n}/{d}), k = {k}");
        }
    }

    /// The nearest `f64` to `sqrt(n/d)` (ties to even), found without the
    /// converter: binary search over the finite IEEE-754 bit patterns for the
    /// largest grid point `v` with `v² ≤ n/d`, then an exact comparison of
    /// `n/d` with the squared midpoint between `v` and the next grid point.
    fn nearest_sqrt_bits_oracle(n: &Natural, d: &Natural) -> u64 {
        use std::cmp::Ordering::{Equal, Greater, Less};
        // Bit pattern `b` of a finite nonnegative f64 as `m · 2^e`.
        let split = |b: u64| -> (u64, i64) {
            let (f, ex) = (b & ((1 << 52) - 1), (b >> 52) as i64);
            if ex == 0 {
                (f, -1074)
            } else {
                (f | 1 << 52, ex - 1075)
            }
        };
        // n/d against a² · 2^k.
        let cmp = |a: u64, k: i64| {
            let rhs = Natural::from(a) * Natural::from(a) * d;
            if k >= 0 {
                n.cmp(&(rhs << (k as u64)))
            } else {
                (n << ((-k) as u64)).cmp(&rhs)
            }
        };
        let (mut lo, mut hi) = (0u64, f64::MAX.to_bits());
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            let (m, e) = split(mid);
            if cmp(m, 2 * e) == Less {
                hi = mid - 1;
            } else {
                lo = mid;
            }
        }
        // Next grid point is `(m + 1) · 2^e`, also across a binade or the
        // subnormal/normal boundary, so the midpoint is `(2m + 1) · 2^(e-1)`.
        let (m, e) = split(lo);
        match cmp(2 * m + 1, 2 * e - 2) {
            Less => lo,
            Greater => lo + 1,
            Equal => lo + (lo & 1),
        }
    }

    /// One rounding over the whole range, against the independent grid oracle:
    /// subnormal midpoints of both parities, the `2^-1075` and `2^-1022`
    /// boundaries, carries into the normal range, deep underflow and random
    /// subnormal and normal ratios (#262).
    #[test]
    fn sqrt_ratio_to_f64_is_nearest_over_whole_range() {
        use rand::{Rng, SeedableRng};
        let p2 = |k: u64| Natural::ONE << k;
        let sq = |x: &Natural| x * x;
        let check = |n: &Natural, d: &Natural| {
            let want = nearest_sqrt_bits_oracle(n, d);
            assert_eq!(sqrt_ratio_to_f64(n, d).to_bits(), want, "sqrt({n}/{d})");
            assert_eq!(
                signed_sqrt_to_f64(-1, n, d).to_bits(),
                want | 1 << 63,
                "-sqrt({n}/{d})"
            );
        };
        // The issue's example, just above 2^-1075: the smallest subnormal.
        let (n, d) = (p2(120) + Natural::ONE, p2(2270));
        assert_eq!(sqrt_ratio_to_f64(&n, &d).to_bits(), 1);
        assert_eq!(signed_sqrt_to_f64(-1, &n, &d).to_bits(), 1 << 63 | 1);
        check(&n, &d);
        let mut cases = vec![(n, d)];
        // Midpoints `(2k+1)·2^-1075` of both parities (k = 0 is 2^-1075 itself,
        // k = 2^52 - 1 rounds up to the smallest normal), radicands just off them.
        let ks = [
            0u64,
            1,
            2,
            3,
            1 << 20,
            (1 << 52) - 2,
            (1 << 52) - 1,
            1 << 52,
        ];
        for k in ks {
            let m = sq(&Natural::from(2 * k + 1));
            for n in [&m - Natural::ONE, m.clone(), &m + Natural::ONE] {
                cases.push((n, p2(2150)));
            }
            // A finer perturbation of the same midpoint.
            let fine = &m << 200u32;
            for n in [&fine - Natural::ONE, &fine + Natural::ONE] {
                cases.push((n, p2(2350)));
            }
        }
        // Grid points `j·2^-1074` around 2^-1022 and the smallest subnormal,
        // exact and perturbed.
        for j in [1u64, 2, (1 << 52) - 1, 1 << 52, (1 << 52) + 1] {
            let m = sq(&Natural::from(j)) << 200u32;
            for n in [&m - Natural::ONE, m.clone(), &m + Natural::ONE] {
                cases.push((n, p2(2348)));
            }
        }
        // Deep underflow: 2^-1076, just below it, 2^-1200, 2^-5000.
        cases.push((Natural::ONE, p2(2152)));
        cases.push((p2(300) - Natural::ONE, p2(2452)));
        cases.push((Natural::ONE, p2(2400)));
        cases.push((Natural::from(3u32), p2(10000)));
        // Overflow: f64::MAX = (2^53 - 1)·2^971 exactly, and the midpoint
        // (2^54 - 1)·2^970 between it and 2^1024, whose tie rounds to infinity.
        let max = Natural::from((1u64 << 53) - 1);
        cases.push((sq(&max) << 1942u32, Natural::ONE));
        let top = sq(&Natural::from((1u64 << 54) - 1)) << 1940u32;
        for n in [&top - Natural::ONE, top.clone(), &top + Natural::ONE] {
            cases.push((n, Natural::ONE));
        }
        // A public CGC below 2^-1022 (`su2::tests::subnormal_cgc_rounds_once`):
        // sqrt(1 / binomial(2050, 1015)).
        let binom = (1..=1015u32).fold(Natural::ONE, |acc, i| {
            acc * Natural::from(1035 + i) / Natural::from(i)
        });
        cases.push((Natural::ONE, binom));
        for (n, d) in &cases {
            check(n, d);
        }
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x262);
        let mut nat = |bits: u64| -> Natural {
            let mut x = Natural::ZERO;
            for _ in 0..bits.div_ceil(64) {
                x = (x << 64u32) + Natural::from(rng.gen::<u64>());
            }
            (x >> (bits.div_ceil(64) * 64 - bits)) | (Natural::ONE << (bits - 1))
        };
        // log2(sqrt(n/d)) ≈ (nb - db - shift) / 2 = -t.
        for i in 0..6000u64 {
            let (nb, db) = (1 + i % 131 * 3, 1 + (i / 131) % 37 * 7);
            let t = if i % 4 == 0 {
                (i * 7919) % 1001 // normal range, 2^-1000..2^0
            } else {
                1018 + (i * 7919) % 63 // 2^-1080..2^-1018
            };
            let shift = (2 * t + nb).saturating_sub(db);
            check(&nat(nb), &(nat(db) << shift));
        }
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
