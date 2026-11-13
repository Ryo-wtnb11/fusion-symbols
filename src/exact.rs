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
use malachite_nz::platform::Limb;
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
    with_sign(sign, sqrt_ratio_to_f64(n, d))
}

/// [`signed_sqrt_to_f64`] for `n, d` already held in stack limbs.
pub(crate) fn signed_sqrt_limbs_to_f64(
    sign: i8,
    n: Limbs<INPUT_LIMBS>,
    d: Limbs<INPUT_LIMBS>,
) -> f64 {
    if sign == 0 {
        return 0.0;
    }
    with_sign(sign, sqrt_operands_to_f64(&Operands::of_limbs(n, d)))
}

fn with_sign(sign: i8, mag: f64) -> f64 {
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
    sqrt_operands_to_f64(&Operands::of_naturals(n, d))
}

/// The exact inputs `n, d` (`d > 0`) of one conversion in the narrowest form
/// that holds both. Only the two exact primitives dispatch on it; the scale,
/// the significand and the rounding decision are shared.
enum Operands<'a> {
    Words(u64, u64),
    Limbs(Limbs<INPUT_LIMBS>, Limbs<INPUT_LIMBS>),
    Big(&'a Natural, &'a Natural),
}

impl<'a> Operands<'a> {
    fn of_naturals(n: &'a Natural, d: &'a Natural) -> Self {
        if let (Ok(u), Ok(v)) = (u64::try_from(n), u64::try_from(d)) {
            return Self::Words(u, v);
        }
        match (Limbs::from_natural(n), Limbs::from_natural(d)) {
            (Some(n), Some(d)) => Self::of_limbs(n, d),
            _ => Self::Big(n, d),
        }
    }

    fn of_limbs(n: Limbs<INPUT_LIMBS>, d: Limbs<INPUT_LIMBS>) -> Self {
        match (n.word(), d.word()) {
            (Some(u), Some(v)) => Self::Words(u, v),
            _ => Self::Limbs(n, d),
        }
    }

    fn bits(&self) -> (i64, i64) {
        match self {
            Self::Words(u, v) => (word_bits(*u), word_bits(*v)),
            Self::Limbs(n, d) => (n.bits() as i64, d.bits() as i64),
            Self::Big(n, d) => (n.significant_bits() as i64, d.significant_bits() as i64),
        }
    }

    /// `floor(sqrt(floor(n · 2^two_g / d)))`; a stack primitive that cannot
    /// hold an operand recomputes it from the same values in big integers.
    fn scaled_sqrt(&self, two_g: i64) -> Natural {
        match self {
            Self::Words(u, v) => match word_scaled_sqrt(*u, *v, two_g) {
                Some(q) => {
                    #[cfg(test)]
                    tests::count_fixed_hit(0);
                    Natural::from(q)
                }
                None => natural_scaled_sqrt(&Natural::from(*u), &Natural::from(*v), two_g),
            },
            Self::Limbs(n, d) => match limb_scaled_sqrt(n, d, two_g) {
                Some(q) => {
                    #[cfg(test)]
                    tests::count_fixed_hit(2);
                    Natural::from(q)
                }
                None => natural_scaled_sqrt(&n.to_natural(), &d.to_natural(), two_g),
            },
            Self::Big(n, d) => natural_scaled_sqrt(n, d, two_g),
        }
    }

    /// `n` against `two_m1² · d · 2^p`, with the same fallback.
    fn midpoint_cmp(&self, two_m1: u64, p: i64) -> std::cmp::Ordering {
        match self {
            Self::Words(u, v) => match word_midpoint_cmp(*u, *v, two_m1, p) {
                Some(ord) => {
                    #[cfg(test)]
                    tests::count_fixed_hit(1);
                    ord
                }
                None => natural_midpoint_cmp(&Natural::from(*u), &Natural::from(*v), two_m1, p),
            },
            Self::Limbs(n, d) => match limb_midpoint_cmp(n, d, two_m1, p) {
                Some(ord) => {
                    #[cfg(test)]
                    tests::count_fixed_hit(3);
                    ord
                }
                None => natural_midpoint_cmp(&n.to_natural(), &d.to_natural(), two_m1, p),
            },
            Self::Big(n, d) => natural_midpoint_cmp(n, d, two_m1, p),
        }
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Words(u, _) => *u == 0,
            Self::Limbs(n, _) => n.bits() == 0,
            Self::Big(n, _) => **n == 0,
        }
    }
}

fn word_bits(x: u64) -> i64 {
    i64::from(u64::BITS - x.leading_zeros())
}

/// The conversion of [`sqrt_ratio_to_f64`] for `n, d` in any [`Operands`]
/// form. Every form's primitives compute the same exact values, so the proof
/// there covers all of them and the bits do not depend on the form.
fn sqrt_operands_to_f64(x: &Operands<'_>) -> f64 {
    if x.is_zero() {
        return 0.0;
    }
    let (nb, db) = x.bits();
    // floor(log2(sqrt(n/d))) ≈ (nb - db) / 2; choose g so Q ≈ sqrt(n/d)*2^g
    // lands around 54 bits.
    let mut g: i64 = 54 - (nb - db) / 2;
    let (q, exp) = loop {
        // q = floor(sqrt(floor(n * 2^(2g) / d))), zero iff the quotient is.
        let q = x.scaled_sqrt(2 * g);
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
    let ord = x.midpoint_cmp(two_m1, p);

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

// The limb primitives compute the same values for inputs of up to
// `INPUT_LIMBS` limbs, in stack arrays, and likewise return `None` instead of
// overflowing. The bound is a coverage choice, not a tuning constant: every
// Cartan CGC rounding of cold SU(3) (4,4)⊗(4,4) and (6,6)⊗(6,6), F4 and E7
// channels, and all but 127 of 38.4 M in G2 [2,2]⊗[2,2]→[2,2], has `max(bits
// U, bits V) <= 384`; at 256 bits 2% of the SU(3) (6,6)⊗(6,6) rounding
// allocations would remain, at 128 bits 81%. Wider inputs keep the
// big-integer path. For `n, d < 2^384`
// the first scale never retries (`n·2^(2g)/d` lies in `(2^106, 2^110)`) and
// every intermediate fits `WORK_LIMBS`: in either scaling direction a side of
// a comparison (`n` shifted by `2g` or `-p`, or `m² · d` shifted by `p`) has at
// most `max(bits n + 2, bits d + 110) <= 494` bits. The ratio
// stays within `2^±384`, so no input in this domain rounds to a subnormal or
// overflows.

/// Limbs of the widest stack input, `n, d < 2^384`.
///
/// A conversion in this domain allocates nothing with Malachite's default
/// 64-bit limbs; under its `32_bit_limbs` feature the values are the same but
/// a root above `2^32` held as a `Natural` allocates.
pub(crate) const INPUT_LIMBS: usize = 6;

/// Limbs of every stack intermediate. A side of a comparison has at most
/// `max(bits n + 2, bits d + 110) <= 494` bits in the domain above, whichever
/// way the scale shifts, and every shift checks the resulting width first.
const WORK_LIMBS: usize = 8;

/// A nonnegative integer below `2^(64·N)` in little-endian 64-bit limbs.
/// `len` counts the limbs up to the highest nonzero one; the limbs above it
/// are zero, so the representation is unique.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Limbs<const N: usize> {
    limbs: [u64; N],
    len: usize,
}

impl<const N: usize> Limbs<N> {
    pub(crate) const ONE: Self = Self::from_word(1);

    const fn from_word(w: u64) -> Self {
        let mut limbs = [0; N];
        limbs[0] = w;
        Self {
            limbs,
            len: (w != 0) as usize,
        }
    }

    /// The value of `limbs`, of which those from `len` on are zero.
    fn trimmed(limbs: [u64; N], mut len: usize) -> Self {
        while len > 0 && limbs[len - 1] == 0 {
            len -= 1;
        }
        Self { limbs, len }
    }

    /// `x` when it is below `2^(64·N)`; reads Malachite's limbs in place.
    pub(crate) fn from_natural(x: &Natural) -> Option<Self> {
        if let Ok(w) = u64::try_from(x) {
            return Some(Self::from_word(w));
        }
        let bits = x.significant_bits();
        if bits > 64 * N as u64 {
            return None;
        }
        let mut limbs = [0u64; N];
        for (i, w) in x.limbs().enumerate() {
            let bit = i * Limb::BITS as usize;
            // A Malachite limb is 64 bits wide, or 32 under its `32_bit_limbs`
            // feature, where this conversion is not the identity.
            #[allow(clippy::useless_conversion)]
            let w = u64::from(w);
            limbs[bit / 64] |= w << (bit % 64);
        }
        Some(Self {
            limbs,
            len: bits.div_ceil(64) as usize,
        })
    }

    /// The value as a big integer; only the fallback of a rejected primitive.
    fn to_natural(self) -> Natural {
        self.used()
            .iter()
            .rev()
            .fold(Natural::ZERO, |acc, &w| (acc << 64u32) + Natural::from(w))
    }

    fn used(&self) -> &[u64] {
        &self.limbs[..self.len]
    }

    fn bits(&self) -> u64 {
        self.used().last().map_or(0, |&top| {
            64 * self.len as u64 - u64::from(top.leading_zeros())
        })
    }

    fn word(&self) -> Option<u64> {
        (self.len <= 1).then_some(self.limbs[0])
    }

    /// `(w, s)` with `w` the leading at most 64 bits, `self = w · 2^s + r`,
    /// `r < 2^s`.
    fn leading_word(&self) -> (u64, u64) {
        let s = self.bits().saturating_sub(64);
        let (i, k) = ((s / 64) as usize, (s % 64) as u32);
        let mut w = self.limbs[i] >> k;
        if k > 0 {
            w |= self.limbs[i + 1] << (64 - k);
        }
        (w, s)
    }

    /// `self *= x` when the product fits `N` limbs, `None` (leaving `self`
    /// unspecified) otherwise. A word `x` takes one pass in place.
    pub(crate) fn checked_mul_assign(&mut self, x: &Natural) -> Option<()> {
        let Ok(w) = u64::try_from(x) else {
            *self = self.checked_mul_wide(x)?;
            return Some(());
        };
        let mut carry = 0u64;
        for l in &mut self.limbs[..self.len] {
            let t = u128::from(*l) * u128::from(w) + u128::from(carry);
            *l = t as u64;
            carry = (t >> 64) as u64;
        }
        if w == 0 {
            self.len = 0;
        } else if carry != 0 {
            *self.limbs.get_mut(self.len)? = carry;
            self.len += 1;
        }
        Some(())
    }

    // Out of line, so that the word loop above stays small where it inlines.
    #[inline(never)]
    fn checked_mul_wide(&self, x: &Natural) -> Option<Self> {
        self.checked_mul(&Limbs::<N>::from_natural(x)?)
    }

    /// `self · rhs` when the product fits `M` limbs.
    fn checked_mul<const K: usize, const M: usize>(&self, rhs: &Limbs<K>) -> Option<Limbs<M>> {
        let (a, b) = (self.used(), rhs.used());
        let mut r = [0u64; M];
        if a.is_empty() || b.is_empty() {
            return Some(Limbs { limbs: r, len: 0 });
        }
        // The product has `a.len() + b.len() - 1` limbs or one more.
        if a.len() + b.len() - 1 > M {
            return None;
        }
        for (i, &x) in a.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &y) in b.iter().enumerate() {
                let t = u128::from(x) * u128::from(y) + u128::from(r[i + j]) + u128::from(carry);
                r[i + j] = t as u64;
                carry = (t >> 64) as u64;
            }
            match r.get_mut(i + b.len()) {
                Some(top) => *top = carry,
                None if carry != 0 => return None,
                None => {}
            }
        }
        Some(Limbs::trimmed(r, (a.len() + b.len()).min(M)))
    }

    /// `self · 2^k` when it fits `M` limbs.
    fn checked_shl<const M: usize>(&self, k: u64) -> Option<Limbs<M>> {
        let mut r = [0u64; M];
        let bits = self.bits();
        if bits == 0 {
            return Some(Limbs { limbs: r, len: 0 });
        }
        if bits + k > 64 * M as u64 {
            return None;
        }
        let (s, b) = ((k / 64) as usize, (k % 64) as u32);
        // Every used limb lands below `M` by the bit count above, and so does
        // its high part whenever that is nonzero.
        for (i, &w) in self.used().iter().enumerate() {
            r[i + s] |= w << b;
            if b > 0 && i + s + 1 < M {
                r[i + s + 1] |= w >> (64 - b);
            }
        }
        Some(Limbs {
            limbs: r,
            len: (bits + k).div_ceil(64) as usize,
        })
    }
}

impl<const N: usize> Ord for Limbs<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.len
            .cmp(&other.len)
            .then_with(|| self.used().iter().rev().cmp(other.used().iter().rev()))
    }
}

impl<const N: usize> PartialOrd for Limbs<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// `floor(sqrt(floor(n · 2^two_g / d)))`: the word primitive's root of the
/// leading words, settled by exact comparisons. The leading words carry the
/// ratio to a relative `2^-62`, so that root is within one of the exact one;
/// a candidate further off is rejected rather than searched from.
fn limb_scaled_sqrt(n: &Limbs<INPUT_LIMBS>, d: &Limbs<INPUT_LIMBS>, two_g: i64) -> Option<u64> {
    // Whether `q² · d <= n · 2^two_g`, that is, `q` is at most the root.
    let below = |q: u64| Some(limb_midpoint_cmp(n, d, q, -two_g)? != std::cmp::Ordering::Less);
    let ((nw, ns), (dw, ds)) = (n.leading_word(), d.leading_word());
    let c = word_scaled_sqrt(nw, dw, two_g + ns as i64 - ds as i64)?;
    // `below(0)` holds, so `c >= 1` when `c` is above the root.
    let q = if !below(c)? {
        c - 1
    } else if below(c.checked_add(1)?)? {
        c + 1
    } else {
        return Some(c);
    };
    (below(q)? && !below(q.checked_add(1)?)?).then_some(q)
}

/// `n` against `m² · d · 2^p` for either sign of `p`, when both sides fit
/// `WORK_LIMBS`.
fn limb_midpoint_cmp(
    n: &Limbs<INPUT_LIMBS>,
    d: &Limbs<INPUT_LIMBS>,
    m: u64,
    p: i64,
) -> Option<std::cmp::Ordering> {
    let sq = u128::from(m) * u128::from(m);
    let sq = Limbs::trimmed([sq as u64, (sq >> 64) as u64], 2);
    let rhs: Limbs<WORK_LIMBS> = d.checked_mul(&sq)?;
    let (lhs, rhs) = if p >= 0 {
        (n.checked_shl(0)?, rhs.checked_shl(p.unsigned_abs())?)
    } else {
        (n.checked_shl(p.unsigned_abs())?, rhs)
    };
    Some(lhs.cmp(&rhs))
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
        use rand::{RngExt, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x256e);
        let mut nat = |bits: u64| -> Natural {
            let mut x = Natural::ZERO;
            for _ in 0..bits.div_ceil(64) {
                x = (x << 64u32) + Natural::from(rng.random::<u64>());
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
    /// subnormal and normal ratios.
    #[test]
    fn sqrt_ratio_to_f64_is_nearest_over_whole_range() {
        use rand::{RngExt, SeedableRng};
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
        // Just above 2^-1075: the smallest subnormal.
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
                x = (x << 64u32) + Natural::from(rng.random::<u64>());
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
        /// Successful calls of the stack primitives on this thread: `[word
        /// square root, word comparison, limb square root, limb comparison]`.
        static FIXED_HITS: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
    }

    pub(super) fn count_fixed_hit(i: usize) {
        FIXED_HITS.with(|h| {
            let mut v = h.get();
            v[i] += 1;
            h.set(v);
        });
    }

    /// The converter's bits and the stack-primitive successes of one call.
    fn convert_counting(n: &Natural, d: &Natural) -> (u64, [usize; 4]) {
        let before = FIXED_HITS.with(std::cell::Cell::get);
        let bits = sqrt_ratio_to_f64(n, d).to_bits();
        let after = FIXED_HITS.with(std::cell::Cell::get);
        (bits, std::array::from_fn(|i| after[i] - before[i]))
    }

    /// The stack primitives one conversion of `n, d` must use: both word
    /// primitives once when both fit a word, else both limb primitives once
    /// when both fit [`INPUT_LIMBS`], else none.
    fn expected_hits(n: &Natural, d: &Natural) -> [usize; 4] {
        let fits = |x: &Natural, bits: u64| x.significant_bits() <= bits;
        if fits(n, 64) && fits(d, 64) {
            [1, 1, 0, 0]
        } else if fits(n, 384) && fits(d, 384) {
            [0, 0, 1, 1]
        } else {
            [0; 4]
        }
    }

    /// Both signs against the grid oracle, through the stack primitives that
    /// [`expected_hits`] names.
    fn check_dispatch(n: &Natural, d: &Natural) {
        let want = nearest_sqrt_bits_oracle(n, d);
        let (bits, hits) = convert_counting(n, d);
        assert_eq!(bits, want, "sqrt({n}/{d})");
        assert_eq!(
            signed_sqrt_to_f64(-1, n, d).to_bits(),
            want | 1 << 63,
            "-sqrt({n}/{d})"
        );
        assert_eq!(
            hits,
            expected_hits(n, d),
            "stack primitives for sqrt({n}/{d})"
        );
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
            assert_eq!(
                convert_counting(&n, &d),
                (bits, [1, 1, 0, 0]),
                "sqrt({u}/{v})"
            );
            check_dispatch(&n, &d);
        }
        for (u, v) in [(1, u64::MAX), (u64::MAX, 1)] {
            check_dispatch(&Natural::from(u), &Natural::from(v));
        }
    }

    /// Word and big-integer storage agree with the grid oracle on both sides
    /// of the `2^64` boundary of each input, and over random bit lengths.
    #[test]
    fn word_and_big_inputs_round_alike() {
        use rand::{RngExt, SeedableRng};
        let mut values: Vec<Natural> = [1u64, 2, 3, 4, 9, 1 << 32, 0xffff_ffff]
            .map(Natural::from)
            .to_vec();
        for k in [62u64, 63, 64, 65] {
            let p = Natural::ONE << k;
            values.extend([&p - Natural::ONE, p.clone(), p + Natural::ONE]);
        }
        for n in &values {
            for d in &values {
                check_dispatch(n, d);
            }
        }
        // Perfect squares in and out of the word domain round exactly.
        for x in [3u64, 0xffff_ffff, 1 << 32] {
            let sq = Natural::from(x) * Natural::from(x);
            check_dispatch(&sq, &Natural::ONE);
            check_dispatch(&Natural::ONE, &sq);
        }
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x8a);
        let mut nat = |bits: u64| -> Natural {
            let x =
                Natural::from(rng.random::<u64>()) << 64u32 | Natural::from(rng.random::<u64>());
            (x >> (128 - bits)) | (Natural::ONE << (bits - 1))
        };
        for i in 0..4000u64 {
            // Bit lengths 1..=64 inside the word domain, 65..=70 just outside.
            let (nb, db) = (1 + i % 70, 1 + (i * 37 + i / 70) % 70);
            check_dispatch(&nat(nb), &nat(db));
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
        use rand::{RngExt, SeedableRng};
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
            pairs.push((r(nb, rng.random()), r(db, rng.random())));
        }
        let inputs: Vec<(Natural, Natural)> = pairs
            .iter()
            .map(|&(u, v)| (Natural::from(u), Natural::from(v)))
            .collect();
        let mut out = Vec::with_capacity(inputs.len());
        // Touch the thread-local counters before the measured interval.
        let _ = FIXED_HITS.with(std::cell::Cell::get);
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

    /// A natural of exactly `bits` bits (`bits >= 1`) with random lower bits.
    fn random_natural(rng: &mut rand_chacha::ChaCha8Rng, bits: u64) -> Natural {
        use rand::RngExt;
        let mut x = Natural::ZERO;
        for _ in 0..bits.div_ceil(64) {
            x = (x << 64u32) + Natural::from(rng.random::<u64>());
        }
        (x >> (bits.div_ceil(64) * 64 - bits)) | (Natural::ONE << (bits - 1))
    }

    /// Values at every limb boundary up to one limb past the stack width:
    /// `2^(64k) - 1` (all limbs full, so every carry propagates), `2^(64k)`
    /// and `2^(64k) + 1`.
    fn limb_boundaries() -> Vec<Natural> {
        let mut v = vec![Natural::ONE, Natural::from(3u32)];
        for k in 1..=7u64 {
            let p = Natural::ONE << (64 * k);
            v.extend([&p - Natural::ONE, p.clone(), p + Natural::ONE]);
        }
        v
    }

    /// The limb arithmetic against big integers at limb boundaries and over
    /// random widths, including every rejection: reading a value, products
    /// with full carries, shifts, comparisons with equality on both signs of
    /// the exponent, and the settled square root at the driver's scales.
    #[test]
    fn limb_primitives_match_big_integers() {
        use rand::SeedableRng;
        use std::cmp::Ordering::Equal;
        type In = Limbs<INPUT_LIMBS>;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x8d01);
        let mut values = limb_boundaries();
        for bits in (1..=448u64).step_by(9) {
            values.push(random_natural(&mut rng, bits));
        }
        for x in &values {
            let l = In::from_natural(x);
            assert_eq!(l.is_some(), x.significant_bits() <= 384, "read {x}");
            if let Some(l) = l {
                assert_eq!(l.to_natural(), *x);
                assert_eq!(l.bits(), x.significant_bits());
                let (w, s) = l.leading_word();
                assert_eq!(Natural::from(w), x >> s, "leading word of {x}");
                assert!(s == 0 || w >> 63 == 1, "leading word of {x}");
                for k in [0u64, 1, 63, 64, 65, 127, 128, 200] {
                    let got: Option<Limbs<WORK_LIMBS>> = l.checked_shl(k);
                    let want = x << k;
                    let fits = want.significant_bits() <= 512;
                    assert_eq!(
                        got.map(Limbs::to_natural),
                        fits.then_some(want),
                        "{x} << {k}"
                    );
                }
            }
        }
        let fitting: Vec<(&Natural, In)> = values
            .iter()
            .filter_map(|x| In::from_natural(x).map(|l| (x, l)))
            .collect();
        for (x, lx) in &fitting {
            for (y, ly) in &fitting {
                let want = *x * *y;
                let fits = want.significant_bits() <= 384;
                let got: Option<In> = lx.checked_mul(ly);
                assert_eq!(
                    got.map(Limbs::to_natural),
                    fits.then(|| want.clone()),
                    "{x} · {y}"
                );
                let mut got = *lx;
                let got = got.checked_mul_assign(y).map(|()| got.to_natural());
                assert_eq!(got, fits.then_some(want), "{x} · {y}");
                assert_eq!(lx.cmp(ly), x.cmp(y), "{x} vs {y}");
            }
        }
        // Comparisons on both sides of an exact equality `n = m²·d·2^p`, for
        // both signs of `p`, and on random operands.
        for i in 0..400u64 {
            let m = u64::try_from(&random_natural(&mut rng, 1 + i % 64)).unwrap();
            let p = (i % 41) as i64 - 20;
            let base = random_natural(&mut rng, 1 + i % 270);
            // With `d·2^p` an integer, `n = m²·d·2^p` is the exact equality.
            let d = if p < 0 {
                &base << (-p) as u64
            } else {
                base.clone()
            };
            let n = (Natural::from(m) * Natural::from(m) * &base) << p.max(0) as u64;
            let mut ns = vec![
                &n + Natural::ONE,
                n.clone(),
                random_natural(&mut rng, 1 + i % 384),
            ];
            if n > 1 {
                ns.push(&n - Natural::ONE);
            }
            for n in ns {
                let (Some(ln), Some(ld)) = (In::from_natural(&n), In::from_natural(&d)) else {
                    continue;
                };
                let want = natural_midpoint_cmp(&n, &d, m, p);
                assert_eq!(
                    limb_midpoint_cmp(&ln, &ld, m, p),
                    Some(want),
                    "{n} vs {m}²·{d}·2^{p}"
                );
            }
        }
        // Equality is reachable at the primitive with multi-limb operands.
        let (n, d) = (Natural::from(9u32) << 302u32, Natural::ONE << 300u32);
        let (ln, ld) = (In::from_natural(&n).unwrap(), In::from_natural(&d).unwrap());
        assert_eq!(limb_midpoint_cmp(&ln, &ld, 3, 2), Some(Equal));
        // A side wider than the work limbs is rejected, not truncated.
        let top = In::from_natural(&((Natural::ONE << 384u32) - Natural::ONE)).unwrap();
        assert!(limb_midpoint_cmp(&top, &ld, 1, -128).is_some());
        assert_eq!(limb_midpoint_cmp(&top, &ld, 1, -129), None);
        assert_eq!(limb_midpoint_cmp(&ld, &top, 1, 129), None);
        let n = Natural::from(9u32) << 298u32;
        let ln = In::from_natural(&n).unwrap();
        assert_eq!(limb_midpoint_cmp(&ln, &ld, 3, -2), Some(Equal));
        // The settled root at the driver's first scale and at retried scales.
        for i in 0..600u64 {
            let (nb, db) = (1 + i % 384, 1 + (i * 37 + i / 384) % 384);
            let (n, d) = (random_natural(&mut rng, nb), random_natural(&mut rng, db));
            let (ln, ld) = (In::from_natural(&n).unwrap(), In::from_natural(&d).unwrap());
            let g = 54 - (nb as i64 - db as i64) / 2;
            for two_g in [2 * g, 2 * g + 16, 2 * g - 16] {
                let want = natural_scaled_sqrt(&n, &d, two_g);
                if let Some(q) = limb_scaled_sqrt(&ln, &ld, two_g) {
                    assert_eq!(Natural::from(q), want, "sqrt({n}·2^{two_g}/{d})");
                } else {
                    assert!(two_g != 2 * g, "first scale rejected for sqrt({n}/{d})");
                }
            }
        }
    }

    /// The conversion of `n, d` through the big-integer path, whatever their
    /// width.
    fn big_path_bits(n: &Natural, d: &Natural) -> u64 {
        sqrt_operands_to_f64(&Operands::Big(n, d)).to_bits()
    }

    /// Stack-limb inputs give the bits of the big-integer path and of the grid
    /// oracle, through the limb primitives, at every limb boundary and one
    /// past the stack width, over random widths with both orders of
    /// magnitude, and for unreduced ratios scaled across the width limit.
    #[test]
    fn limb_inputs_round_like_big_integers() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x8d02);
        let mut pairs = Vec::new();
        let bounds = limb_boundaries();
        for n in &bounds {
            for d in &bounds {
                pairs.push((n.clone(), d.clone()));
            }
        }
        for i in 0..3000u64 {
            let (nb, db) = (1 + i % 392, 1 + (i * 53 + i / 392) % 392);
            pairs.push((random_natural(&mut rng, nb), random_natural(&mut rng, db)));
        }
        // Unreduced `(k·n, k·d)` whose scaled widths reach and pass 384 bits.
        for i in 0..300u64 {
            let (n, d) = (
                random_natural(&mut rng, 1 + i % 200),
                random_natural(&mut rng, 1 + i % 150),
            );
            let k = random_natural(&mut rng, 1 + i % 240);
            let want = big_path_bits(&n, &d);
            let (kn, kd) = (&k * &n, &k * &d);
            assert_eq!(
                convert_counting(&kn, &kd),
                (want, expected_hits(&kn, &kd)),
                "k = {k}"
            );
            pairs.push((kn, kd));
        }
        for (n, d) in &pairs {
            assert_eq!(
                sqrt_ratio_to_f64(n, d).to_bits(),
                big_path_bits(n, d),
                "sqrt({n}/{d})"
            );
            check_dispatch(n, d);
        }
    }

    /// Exact midpoints `(m + 1/2)·2^e` held in multi-limb operands round to
    /// the even neighbour, and one unit either side of the squared midpoint
    /// rounds away from it, for both parities of `m`, a carry into the next
    /// binade, both signs, and the largest and smallest ratios of 384-bit
    /// operands. The expected bits are the IEEE encoding of the chosen
    /// neighbour.
    #[test]
    fn limb_inputs_round_at_midpoints() {
        use rand::{RngExt, SeedableRng};
        use std::cmp::Ordering::{Equal, Greater, Less};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x8d03);
        // `j · 2^e` for `2^52 <= j <= 2^53`, normal and exact.
        let grid = |j: u64, e: i64| -> u64 {
            let (j, e) = if j == 1 << 53 {
                (1 << 52, e + 1)
            } else {
                (j, e)
            };
            ((e + 52 + 1023) as u64) << 52 | (j - (1 << 52))
        };
        let ms = [
            1u64 << 52,
            (1 << 52) + 1,
            (1 << 52) + 2,
            (1 << 53) - 2,
            (1 << 53) - 1,
        ];
        let random_ms: Vec<u64> = (0..40)
            .map(|_| (1 << 52) | rng.random::<u64>() >> 12)
            .collect();
        for (i, m) in ms.into_iter().chain(random_ms).enumerate() {
            let i = i as u64;
            let e = (i % 21) as i64 - 10;
            // x² = (2m+1)² · 2^(2e-2) = n/d, with a common factor k of up to
            // 270 bits so that both operands span several limbs.
            let k = random_natural(&mut rng, 1 + (i * 29) % 270);
            let sq = Natural::from(2 * m + 1) * Natural::from(2 * m + 1) * &k;
            let (a, b) = ((2 * e - 2).max(0) as u64, (2 - 2 * e).max(0) as u64);
            let (n, d) = (sq << a, k << b);
            for (n, ord) in [
                (&n - Natural::ONE, Less),
                (n.clone(), Equal),
                (&n + Natural::ONE, Greater),
            ] {
                let j = match ord {
                    Less => m,
                    Equal => m + (m & 1),
                    Greater => m + 1,
                };
                let want = grid(j, e);
                assert_eq!(
                    convert_counting(&n, &d),
                    (want, expected_hits(&n, &d)),
                    "{ord:?} m = {m}"
                );
                assert_eq!(big_path_bits(&n, &d), want);
                let (ln, ld) = (
                    Limbs::from_natural(&n).unwrap(),
                    Limbs::from_natural(&d).unwrap(),
                );
                assert_eq!(
                    signed_sqrt_limbs_to_f64(-1, ln, ld).to_bits(),
                    want | 1 << 63
                );
                assert_eq!(signed_sqrt_limbs_to_f64(1, ln, ld).to_bits(), want);
                assert_eq!(signed_sqrt_limbs_to_f64(0, ln, ld).to_bits(), 0);
            }
        }
        // The extreme ratios of 384-bit operands, 2^±192 to within a unit.
        let top = (Natural::ONE << 384u32) - Natural::ONE;
        for (n, d) in [
            (top.clone(), Natural::ONE),
            (Natural::ONE, top.clone()),
            (top.clone(), top),
        ] {
            check_dispatch(&n, &d);
        }
    }

    /// A conversion whose inputs fit the stack limbs allocates nothing.
    #[cfg(feature = "lie-generated")]
    #[test]
    fn limb_conversion_does_not_allocate() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x8d04);
        let mut inputs: Vec<(Natural, Natural)> = Vec::new();
        for i in 0..2000u64 {
            let (nb, db) = (65 + i % 320, 1 + (i * 37) % 384);
            let (n, d) = (random_natural(&mut rng, nb), random_natural(&mut rng, db));
            inputs.push(if i % 2 == 0 { (n, d) } else { (d, n) });
        }
        let top = (Natural::ONE << 384u32) - Natural::ONE;
        inputs.extend([
            (top.clone(), Natural::ONE),
            (Natural::ONE, top.clone()),
            (top.clone(), top),
        ]);
        let mut out = Vec::with_capacity(2 * inputs.len());
        let _ = FIXED_HITS.with(std::cell::Cell::get);
        let calls = crate::audit_alloc::thread_allocation_calls;
        let before = calls();
        for (n, d) in &inputs {
            out.push(sqrt_ratio_to_f64(n, d).to_bits());
            out.push(signed_sqrt_to_f64(-1, n, d).to_bits());
        }
        assert_eq!(calls() - before, 0);
        for ((n, d), bits) in inputs.iter().zip(out.chunks(2)) {
            let want = big_path_bits(n, d);
            assert_eq!(bits, [want, want | 1 << 63], "sqrt({n}/{d})");
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
