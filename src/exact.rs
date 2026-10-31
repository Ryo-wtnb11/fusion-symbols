//! Exact value type for SU(2) recoupling coefficients.
//!
//! Every Wigner 3j/6j/Clebsch-Gordan value equals `sign * sqrt(radicand)` with
//! `radicand` a nonnegative rational over big integers. That is the closed
//! form of the Racah expressions: a triangle/dimension product under a square
//! root times a rational alternating sum. We keep the value in this exact form
//! all the way through and round to `f64` exactly once, in [`SignedSqrtRational::to_f64`].

use std::sync::Arc;

#[cfg(test)]
use malachite_base::num::{arithmetic::traits::Pow, basic::traits::One};
use malachite_base::num::{
    arithmetic::traits::{FloorSqrt, XMulYToZZ, XXDivModYToQR},
    basic::traits::Zero,
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
    // Both exact inputs as machine words, when they fit (`d > 0` is the
    // caller's invariant). Only the two exact primitives below dispatch on it;
    // the scale, the significand and the rounding decision are shared.
    let words = u64::try_from(n).ok().zip(u64::try_from(d).ok());
    let nb = n.significant_bits() as i64;
    let db = d.significant_bits() as i64;
    // floor(log2(sqrt(n/d))) ≈ (nb - db) / 2; choose g so Q ≈ sqrt(n/d)*2^g
    // lands around 54 bits.
    let mut g: i64 = 54 - (nb - db) / 2;
    let (q, exp) = loop {
        // q = floor(sqrt(floor(n * 2^(2g) / d))), zero iff the quotient is.
        let q = match words.and_then(|(u, v)| word_scaled_sqrt(u, v, 2 * g)) {
            Some(q) => Natural::from(q),
            None => natural_scaled_sqrt(n, d, 2 * g),
        };
        if q == 0 {
            // g too small (only possible if n/d rounds below 1 at this scale);
            // push the scale up and retry.
            g += 8;
            continue;
        }
        let bits = q.significant_bits() as i64;
        if bits < 54 {
            g += (54 - bits + 1) / 2 + 1;
            continue;
        }
        if bits > 56 {
            g -= (bits - 56 + 1) / 2 + 1;
            continue;
        }
        break (u64::try_from(&q).expect("q has at most 56 bits"), g);
    };

    let drop = i64::from(u64::BITS - q.leading_zeros()) - 53; // >= 1 since bits in 54..=56
    let mut mant = q >> drop;
    // value ≈ mant * 2^(drop - exp)
    let mut value_exp = drop - exp;
    // Below 2^-1022 the f64 spacing is fixed at 2^-1074, so the significand
    // keeps fewer than 53 bits: floor(floor(y) / 2^s) = floor(y / 2^s) gives
    // floor(x·2^1074). Rounding once here leaves `scale_pow2` an exact scale;
    // a carry to 2^52 lands exactly on the smallest normal.
    if value_exp < MIN_ULP_EXP {
        mant = u32::try_from(MIN_ULP_EXP - value_exp)
            .ok()
            .and_then(|s| mant.checked_shr(s))
            .unwrap_or(0);
        value_exp = MIN_ULP_EXP;
    }

    // Exact midpoint decision: compare sqrt(n/d) against (mant + 1/2)*2^value_exp.
    // Squared: n/d  vs  (2*mant+1)^2 * 2^(2*value_exp - 2), cross-multiplied
    // by d with the power of two moved to the side it multiplies.
    let two_m1 = 2 * mant + 1; // below 2^54
    let p = 2 * value_exp - 2;
    let ord = match words.and_then(|(u, v)| word_midpoint_cmp(u, v, two_m1, p)) {
        Some(ord) => ord,
        None => natural_midpoint_cmp(n, d, two_m1, p),
    };

    let chosen = match ord {
        std::cmp::Ordering::Less => mant,
        std::cmp::Ordering::Greater => mant + 1,
        // Exact midpoint: round half to even.
        std::cmp::Ordering::Equal => mant + (mant & 1),
    };

    // `chosen` is at most 2^53 so it is exact in f64, and `chosen · 2^value_exp`
    // is a representable f64 (or overflows).
    scale_pow2(chosen as f64, value_exp)
}

/// Exponent of the smallest subnormal `f64`, `2^-1074`: the finest spacing of
/// the `f64` grid.
const MIN_ULP_EXP: i64 = -1074;

/// `floor(sqrt(floor(n · 2^two_g / d)))` in big integers, for either sign of
/// `two_g`.
fn natural_scaled_sqrt(n: &Natural, d: &Natural, two_g: i64) -> Natural {
    let scaled = if two_g >= 0 {
        (n << (two_g as u64)) / d
    } else {
        n / (d << ((-two_g) as u64))
    };
    scaled.floor_sqrt()
}

/// `n` against `two_m1² · d · 2^p` in big integers, for either sign of `p`.
fn natural_midpoint_cmp(n: &Natural, d: &Natural, two_m1: u64, p: i64) -> std::cmp::Ordering {
    let two_m1 = Natural::from(two_m1);
    let rhs = &two_m1 * &two_m1 * d;
    if p >= 0 {
        n.cmp(&(rhs << (p as u64)))
    } else {
        (n << ((-p) as u64)).cmp(&rhs)
    }
}

// The word primitives below compute the same values as the big-integer ones
// above in two `u128` limbs, `(hi, lo) = hi · 2^128 + lo`, and return `None`
// whenever an operand or intermediate does not fit, before computing anything
// from it; the caller then recomputes the same primitive from the original
// inputs in big integers. For `0 < u, v < 2^64` the driver's first scale has
// `23 <= g <= 85`, never retries, and keeps `p` negative, so `u · 2^(2g)` and
// `u · 2^(-p)` have at most 173 bits and `(2m+1)² · v` at most 172.

/// `floor(sqrt(floor(u · 2^two_g / v)))` when `u · 2^two_g` fits 256 bits and
/// the quotient fits `u128`.
fn word_scaled_sqrt(u: u64, v: u64, two_g: i64) -> Option<u64> {
    let (hi, lo) = shl_wide(u128::from(u), two_g)?;
    let v = u128::from(v);
    // The divider's precondition; it also excludes `v == 0`, which the
    // big-integer division then reports.
    if hi >= v {
        return None;
    }
    let (scaled, _) = u128::xx_div_mod_y_to_qr(hi, lo, v);
    #[cfg(test)]
    tests::count_word_hit(0);
    // The square root of a u128 is below 2^64.
    Some(scaled.isqrt() as u64)
}

/// `u` against `two_m1² · v · 2^p` for negative `p` with `u · 2^(-p)` within
/// 256 bits; `None` for `p >= 0`, which the word domain never reaches.
fn word_midpoint_cmp(u: u64, v: u64, two_m1: u64, p: i64) -> Option<std::cmp::Ordering> {
    if p >= 0 {
        return None;
    }
    let sq = u128::from(two_m1).checked_mul(u128::from(two_m1))?;
    let rhs = u128::x_mul_y_to_zz(sq, u128::from(v));
    let lhs = shl_wide(u128::from(u), -p)?;
    #[cfg(test)]
    tests::count_word_hit(1);
    Some(lhs.cmp(&rhs))
}

/// `x · 2^k` as `(hi, lo)` when `0 <= k < 256` and no set bit leaves the 256
/// bits; checked before any shift, since a `u128` shift discards high bits
/// and panics at 128.
fn shl_wide(x: u128, k: i64) -> Option<(u128, u128)> {
    let k = u32::try_from(k).ok().filter(|&k| k < 256)?;
    if k > 128 + x.leading_zeros() {
        return None;
    }
    Some(match k {
        0 => (0, x),
        1..=127 => (x >> (128 - k), x << k),
        _ => (x << (k - 128), 0),
    })
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

/// [`word_gcd`] whenever both operands fit; otherwise binary gcd, since a
/// `u128` division is a slow library call and shifts are not.
fn gcd(mut a: u128, mut b: u128) -> u128 {
    if let (Ok(x), Ok(y)) = (u64::try_from(a), u64::try_from(b)) {
        return u128::from(word_gcd(x, y));
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

/// Euclid in hardware `u64` division: the crate's one word gcd, shared by
/// the Weyl-dimension quotient and the Cartan rational. The installed
/// binary gcd measured slower on the Weyl-dimension factors.
pub(crate) fn word_gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `a / b`, in a hardware `u64` division whenever both operands fit.
fn div(a: u128, b: u128) -> u128 {
    match (u64::try_from(a), u64::try_from(b)) {
        (Ok(a), Ok(b)) => u128::from(a / b),
        _ => a / b,
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
        // Both width branches of gcd and division, and the shared word gcd.
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(3 << 100, 3 << 91), 3 << 91);
        assert_eq!(gcd(u128::MAX, 1 << 64), 1);
        assert_eq!(gcd(0, 1 << 100), 1 << 100);
        assert_eq!(gcd(0, 7), 7);
        assert_eq!((word_gcd(1, 6), word_gcd(6, 1), word_gcd(0, 6)), (1, 1, 6));
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

    thread_local! {
        /// Successful calls of the two word primitives on this thread:
        /// `[scaled square root, midpoint comparison]`.
        static WORD_HITS: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
    }

    pub(super) fn count_word_hit(i: usize) {
        WORD_HITS.with(|h| {
            let mut v = h.get();
            v[i] += 1;
            h.set(v);
        });
    }

    /// The converter's bits and the word-primitive successes of one call.
    fn convert_counting(n: &Natural, d: &Natural) -> (u64, [usize; 2]) {
        let before = WORD_HITS.with(std::cell::Cell::get);
        let bits = sqrt_ratio_to_f64(n, d).to_bits();
        let after = WORD_HITS.with(std::cell::Cell::get);
        (bits, [after[0] - before[0], after[1] - before[1]])
    }

    /// Both signs against the grid oracle; inside the word domain both word
    /// primitives succeed exactly once, outside it neither runs.
    fn check_word_dispatch(n: &Natural, d: &Natural) {
        let want = nearest_sqrt_bits_oracle(n, d);
        let (bits, hits) = convert_counting(n, d);
        assert_eq!(bits, want, "sqrt({n}/{d})");
        assert_eq!(
            signed_sqrt_to_f64(-1, n, d).to_bits(),
            want | 1 << 63,
            "-sqrt({n}/{d})"
        );
        let fits = |x: &Natural| u64::try_from(x).is_ok();
        let expected = if fits(n) && fits(d) { [1, 1] } else { [0, 0] };
        assert_eq!(hits, expected, "word primitives for sqrt({n}/{d})");
    }

    /// Word inputs just around a rounding midpoint and a significand carry,
    /// with expected bits from the IEEE grid: `1 + 2^-52` lies just below the
    /// squared midpoint `(1 + 2^-53)² = 1 + 2^-52 + 2^-106` and the next
    /// numerator just above it; the next pair straddles the midpoint above the
    /// odd significand `1 + 2^-52`; the last rounds up to 2 with a carry.
    /// `(1, 2^64 - 1)` and its reversal are the scale extremes, with a
    /// negative odd bit-length difference.
    #[test]
    fn word_inputs_round_at_midpoints_and_carry() {
        let p63 = 1u64 << 63;
        for (u, v, bits) in [
            (p63 + 2048, p63, 0x3ff0_0000_0000_0000u64),
            (p63 + 2049, p63, 0x3ff0_0000_0000_0001),
            (p63 + 6144, p63, 0x3ff0_0000_0000_0001),
            (p63 + 6145, p63, 0x3ff0_0000_0000_0002),
            (p63 - 1, 1 << 61, 0x4000_0000_0000_0000),
        ] {
            let (n, d) = (Natural::from(u), Natural::from(v));
            assert_eq!(convert_counting(&n, &d), (bits, [1, 1]), "sqrt({u}/{v})");
            check_word_dispatch(&n, &d);
        }
        for (u, v) in [(1, u64::MAX), (u64::MAX, 1)] {
            check_word_dispatch(&Natural::from(u), &Natural::from(v));
        }
    }

    /// Word and big-integer storage agree with the grid oracle on both sides
    /// of the `2^64` boundary of each input, and over random bit lengths.
    #[test]
    fn word_and_big_inputs_round_alike() {
        use rand::{Rng, SeedableRng};
        let mut values: Vec<Natural> = [1u64, 2, 3, 4, 9, 1 << 32, 0xffff_ffff]
            .map(Natural::from)
            .to_vec();
        for k in [62u64, 63, 64, 65] {
            let p = Natural::ONE << k;
            values.extend([&p - Natural::ONE, p.clone(), p + Natural::ONE]);
        }
        for n in &values {
            for d in &values {
                check_word_dispatch(n, d);
            }
        }
        // Perfect squares in and out of the word domain round exactly.
        for x in [3u64, 0xffff_ffff, 1 << 32] {
            let sq = Natural::from(x) * Natural::from(x);
            check_word_dispatch(&sq, &Natural::ONE);
            check_word_dispatch(&Natural::ONE, &sq);
        }
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x8a);
        let mut nat = |bits: u64| -> Natural {
            let x = Natural::from(rng.gen::<u64>()) << 64u32 | Natural::from(rng.gen::<u64>());
            (x >> (128 - bits)) | (Natural::ONE << (bits - 1))
        };
        for i in 0..4000u64 {
            // Bit lengths 1..=64 inside the word domain, 65..=70 just outside.
            let (nb, db) = (1 + i % 70, 1 + (i * 37 + i / 70) % 70);
            check_word_dispatch(&nat(nb), &nat(db));
        }
    }

    /// The word primitives against their big-integer counterparts at the
    /// shift, divider and product limits, including the rejected cases.
    #[test]
    fn word_primitives_match_big_integers() {
        let wide = |(hi, lo): (u128, u128)| (Natural::from(hi) << 128u32) + Natural::from(lo);
        for x in [0u128, 1, 3, u64::MAX.into(), 1 << 100, u128::MAX] {
            for k in [0i64, 63, 64, 127, 128, 129, 255, 256, -1] {
                let got = shl_wide(x, k);
                let fits = (0..256).contains(&k)
                    && (Natural::from(x) << k as u64).significant_bits() <= 256;
                assert_eq!(got.is_some(), fits, "{x} << {k}");
                if let Some(w) = got {
                    assert_eq!(wide(w), Natural::from(x) << k as u64, "{x} << {k}");
                }
            }
        }
        // The divider precondition `hi < v` at its limit: u·2^128 / v with
        // hi = u is accepted for u = v - 1 and rejected for u = v.
        for v in [1u64, 3, 1 << 63, u64::MAX] {
            let d = Natural::from(v);
            for u in [v - 1, v] {
                let got = word_scaled_sqrt(u, v, 128);
                let want = natural_scaled_sqrt(&Natural::from(u), &d, 128);
                if u == v {
                    assert_eq!(got, None, "hi = v = {v}");
                } else if u > 0 {
                    assert_eq!(got.map(Natural::from), Some(want), "hi = v - 1 = {u}");
                }
            }
            for (u, two_g) in [(1u64, 0i64), (u64::MAX, 46), (u64::MAX, 170), (7, -2)] {
                let got = word_scaled_sqrt(u, v, two_g);
                let want = natural_scaled_sqrt(&Natural::from(u), &d, two_g);
                // Accepted iff the scale is nonnegative and the quotient fits u128.
                let fits = two_g >= 0
                    && ((Natural::from(u) << two_g as u64) / &d).significant_bits() <= 128;
                assert_eq!(
                    got.map(Natural::from),
                    fits.then_some(want),
                    "{u}·2^{two_g}/{v}"
                );
            }
        }
        // The midpoint comparison at the largest square and denominator, on
        // both sides of equality, and the rejected `p >= 0`.
        let m = (1u64 << 54) - 1;
        let rhs = Natural::from(m) * Natural::from(m) * Natural::from(u64::MAX);
        for p in [-110i64, -170, -200] {
            for u in [1u64, u64::MAX, 1 << 40] {
                let got = word_midpoint_cmp(u, u64::MAX, m, p);
                let n = Natural::from(u);
                let want = natural_midpoint_cmp(&n, &Natural::from(u64::MAX), m, p);
                let fits = (&n << (-p) as u64).significant_bits() <= 256;
                assert_eq!(got, fits.then_some(want), "{u}·2^{} vs {rhs}", -p);
            }
        }
        // Equality is reachable at the primitive: 9·2^2 = 3²·4.
        assert_eq!(
            word_midpoint_cmp(9, 4, 3, -2),
            Some(std::cmp::Ordering::Equal)
        );
        assert_eq!(
            natural_midpoint_cmp(&Natural::from(9u32), &Natural::from(4u32), 3, -2),
            std::cmp::Ordering::Equal
        );
        assert_eq!(word_midpoint_cmp(1, 1, 1, 0), None);
        assert_eq!(word_midpoint_cmp(1, 1, 1, 5), None);
    }

    /// A conversion whose inputs fit machine words allocates nothing.
    #[cfg(feature = "lie-generated")]
    #[test]
    fn word_conversion_does_not_allocate() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x8a1);
        let p63 = 1u64 << 63;
        let mut pairs: Vec<(u64, u64)> = vec![
            (p63 + 2048, p63),
            (p63 + 6145, p63),
            (p63 - 1, 1 << 61),
            (1, u64::MAX),
            (u64::MAX, 1),
            (u64::MAX, u64::MAX),
            (1, 1),
        ];
        for i in 0..2000u32 {
            let (nb, db) = (1 + i % 64, 1 + (i * 37) % 64);
            let r = |b: u32, x: u64| (x >> (64 - b)) | 1 << (b - 1);
            pairs.push((r(nb, rng.gen()), r(db, rng.gen())));
        }
        let inputs: Vec<(Natural, Natural)> = pairs
            .iter()
            .map(|&(u, v)| (Natural::from(u), Natural::from(v)))
            .collect();
        let mut out = Vec::with_capacity(inputs.len());
        // Touch the thread-local counters before the measured interval.
        let _ = WORD_HITS.with(std::cell::Cell::get);
        let calls = crate::audit_alloc::thread_allocation_calls;
        let before = calls();
        for (n, d) in &inputs {
            out.push(sqrt_ratio_to_f64(n, d).to_bits());
        }
        assert_eq!(calls() - before, 0);
        for ((n, d), bits) in inputs.iter().zip(out) {
            assert_eq!(bits, nearest_sqrt_bits_oracle(n, d), "sqrt({n}/{d})");
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
