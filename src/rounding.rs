//! The rounding layer: one conversion from an exact signed square root of a
//! rational to the nearest `f64` ([`round_radical`]), a certified accelerator
//! for it on CGC entries ([`estimate`]: a 128-bit estimate returned only when
//! it provably rounds like the exact value, else the exact conversion), and
//! the gates defined on rounded CGC channels ([`cgc`]). It computes no
//! mathematics: every value it rounds is held exactly by the exact layer, and
//! only the rounded views it returns are floating point.

pub(crate) mod blocks;
pub(crate) mod cgc;
mod estimate;

use std::borrow::Cow;

use malachite_base::num::{
    arithmetic::traits::{FloorSqrt, XMulYToZZ, XXDivModYToQR},
    basic::traits::One,
    logic::traits::SignificantBits,
};
use malachite_nz::natural::Natural;

use crate::exact::limbs::Limbs;
use crate::exact::radical::words_natural;
use crate::exact::SignedSqrtRational;

/// One factor of a radicand's numerator or denominator, read in place: a
/// slab entry is passed as its words, so no big integer is formed for it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Factor<'a> {
    Natural(&'a Natural),
    /// Little-endian 64-bit words; high zero words are allowed.
    Words(&'a [u64]),
}

/// `sign · √(Π num / Π den)` (every factor positive, `sign` in `{-1, 0, 1}`),
/// correctly rounded to `f64`: the crate's one conversion from an exact
/// value (`docs/gauge_cartan.md` §4.7, [`crate::exact::SignedSqrtRational::to_f64`]).
///
/// The products are formed unreduced, in stack limbs when both fit
/// [`INPUT_LIMBS`] (allocating nothing), otherwise in big integers. The
/// result depends on the ratio alone ([`sqrt_ratio_to_f64`]), so neither
/// the factorization of the ratio nor the width decides a bit. A square is
/// passed as its root twice, never formed. The sign is applied after the
/// magnitude is rounded, so a nonzero value that underflows keeps a signed
/// zero.
pub(crate) fn round_radical(sign: i8, num: &[Factor<'_>], den: &[Factor<'_>]) -> f64 {
    if sign == 0 {
        return 0.0;
    }
    if let (Some(n), Some(d)) = (stack_product(num), stack_product(den)) {
        return signed_sqrt_limbs_to_f64(sign, n, d);
    }
    signed_sqrt_to_f64(sign, &big_product(num), &big_product(den))
}

/// A value derived from a dimension `d`, a positive integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DimValue {
    Dim,
    Sqrt,
    InvSqrt,
    Inv,
}

/// `d`, `√d`, `1/√d` or `1/d` of the integer `d ≥ 1`, correctly rounded
/// once: [`round_radical`] with `d` passed once or as its root twice. Below
/// `2^53` the integer is an exact `f64`, so the value, its square root and
/// its reciprocal are IEEE 754 correctly rounded single operations, and
/// `1/√d` is [`inv_sqrt_small`].
pub(crate) fn round_dim(d: Factor<'_>, value: DimValue) -> f64 {
    let small = match d {
        Factor::Natural(x) => u64::try_from(x).ok(),
        Factor::Words(w) => match crate::exact::limbs::trimmed_words(w) {
            [] => Some(0),
            &[x] => Some(x),
            _ => None,
        },
    };
    if let Some(x) = small.filter(|&x| x < 1 << 53) {
        return round_small(x as f64, value);
    }
    match value {
        DimValue::Dim => round_radical(1, &[d, d], &[]),
        DimValue::Sqrt => round_radical(1, &[d], &[]),
        DimValue::InvSqrt => round_radical(1, &[], &[d]),
        DimValue::Inv => round_radical(1, &[], &[d, d]),
    }
}

/// [`round_dim`] of an integer `1 ≤ x < 2^53`, held exactly.
#[inline]
fn round_small(x: f64, value: DimValue) -> f64 {
    match value {
        DimValue::Dim => x,
        DimValue::Sqrt => x.sqrt(),
        DimValue::Inv => x.recip(),
        DimValue::InvSqrt => inv_sqrt_small(x),
    }
}

/// The nearest `f64` to `1/√d` for an integer `1 ≤ d < 2^53` (an exact
/// `x`), without big integers: the definition itself, checked exactly.
///
/// Bracketing: `y` is the nearest `f64` exactly when `1/√d` lies strictly
/// between the midpoint below `y` (to `prev(y)`) and the midpoint above it
/// (to `next(y)`). Each midpoint is `M·2^k` with `M = 2A + 1` odd and
/// `M > 1` (`A ≥ 2^52` the significand of the smaller neighbour), and
/// [`above_midpoint`] decides `1/√d > M·2^k` exactly as the integer
/// comparison `M²·d < 2^(−2k)`. Ties are impossible: `M²·d` has the odd
/// factor `M² > 1`, so it is never a power of two. The start
/// `1/(√d)` is within a few ulp; the loop steps one ulp toward `1/√d` until
/// both midpoint tests bracket it, so the result does not rest on that
/// bound, only on the two exact tests.
fn inv_sqrt_small(x: f64) -> f64 {
    if x == 1.0 {
        return 1.0;
    }
    let d = x as u64;
    let mut y = x.sqrt().recip();
    // `y ∈ (2^-27, 1)` is normal, so its successor is the next bit pattern.
    let next = |y: f64| f64::from_bits(y.to_bits() + 1);
    let prev = |y: f64| f64::from_bits(y.to_bits() - 1);
    loop {
        if above_midpoint(y, d) {
            y = next(y);
        } else if !above_midpoint(prev(y), d) {
            y = prev(y);
        } else {
            return y;
        }
    }
}

/// Whether `1/√d > (a + a⁺)/2`, `a⁺` the successor of the normal `a < 1`.
///
/// With `a = A·2^e` (`A` the 53-bit significand), `a⁺ = (A + 1)·2^e` also
/// across a binade, so the midpoint is `M·2^(e−1)` with `M = 2A + 1`, and
/// `1/√d > M·2^(e−1)` exactly when `M²·d < 2^(2−2e)`, which (never equal)
/// is a bit-length comparison of the exact product `M²·d < 2^163`.
fn above_midpoint(a: f64, d: u64) -> bool {
    let bits = a.to_bits();
    let m = 2 * ((bits & ((1 << 52) - 1)) | (1 << 52)) + 1;
    let e = (bits >> 52) as i64 - 1075;
    let sq = u128::from(m) * u128::from(m);
    let lo = (sq as u64 as u128) * u128::from(d);
    let hi = (sq >> 64) * u128::from(d) + (lo >> 64);
    let len = if hi > 0 {
        64 + 128 - i64::from(hi.leading_zeros())
    } else {
        64 - i64::from((lo as u64).leading_zeros())
    };
    len <= 2 - 2 * e
}

/// The exact product of a product object's leaf dimensions, on the stack
/// while it fits [`INPUT_LIMBS`] (allocating nothing), else a big integer.
#[derive(Clone, Debug)]
pub(crate) enum DimProduct {
    /// Below `2^64`, the common case, multiplied without limb loops.
    Word(u64),
    Stack(Limbs<INPUT_LIMBS>),
    Big(Natural),
}

impl DimProduct {
    /// `self *= d`.
    pub(crate) fn mul(&mut self, d: Factor<'_>) {
        if let Self::Word(x) = self {
            let w = match d {
                Factor::Natural(d) => u64::try_from(d).ok(),
                Factor::Words(w) => match crate::exact::limbs::trimmed_words(w) {
                    &[w] => Some(w),
                    _ => None,
                },
            };
            if let Some(y) = w.and_then(|w| x.checked_mul(w)) {
                *x = y;
                return;
            }
            *self = Self::Stack(Limbs::trimmed(
                std::array::from_fn(|i| if i == 0 { *x } else { 0 }),
                1,
            ));
        }
        if let Self::Stack(x) = self {
            let mut y = *x;
            if y.checked_mul_factor(&d).is_some() {
                *x = y;
                return;
            }
            *self = Self::Big(x.to_natural());
        }
        if let Self::Big(x) = self {
            match d {
                Factor::Natural(d) => *x *= d,
                Factor::Words(w) => *x *= words_natural(w),
            }
        }
    }

    /// [`round_dim`] of the product.
    pub(crate) fn round(&self, value: DimValue) -> f64 {
        match self {
            Self::Word(x) => round_dim(Factor::Words(&[*x]), value),
            Self::Stack(x) => round_dim(Factor::Words(x.used()), value),
            Self::Big(x) => round_dim(Factor::Natural(x), value),
        }
    }
}

use crate::category::sealed::WideDim;

impl crate::category::sealed::DimAcc {
    pub(crate) const ONE: Self = Self {
        word: 1,
        wide: None,
    };

    /// `self *= w`: one multiply while the product fits a word.
    #[inline(always)]
    pub(crate) fn mul_word(&mut self, w: u64) {
        if self.wide.is_none() {
            if let Some(y) = self.word.checked_mul(w) {
                self.word = y;
                return;
            }
        }
        self.mul(Factor::Words(&[w]));
    }

    /// `self *= d`, past one word.
    #[inline(never)]
    pub(crate) fn mul(&mut self, d: Factor<'_>) {
        self.wide
            .get_or_insert(WideDim(DimProduct::Word(self.word)))
            .0
            .mul(d);
    }

    /// [`round_dim`] of the product; IEEE single operations below `2^53`.
    #[inline(always)]
    pub(crate) fn round(&self, value: DimValue) -> f64 {
        match &self.wide {
            None if self.word < 1 << 53 => round_small(self.word as f64, value),
            None => round_dim(Factor::Words(&[self.word]), value),
            Some(x) => x.0.round(value),
        }
    }
}

impl SignedSqrtRational {
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
        round_radical(
            self.sign(),
            &[Factor::Natural(r.numerator_ref())],
            &[Factor::Natural(r.denominator_ref())],
        )
    }
}

/// `Π f` in stack limbs, when every partial product fits. Every factor is at
/// least 1, so a partial product overflows only when the whole one does.
fn stack_product(factors: &[Factor<'_>]) -> Option<Limbs<INPUT_LIMBS>> {
    let mut acc = Limbs::<INPUT_LIMBS>::ONE;
    for f in factors {
        acc.checked_mul_factor(f)?;
    }
    Some(acc)
}

/// `Π f` in big integers; a single big-integer factor is borrowed.
fn big_product<'a>(factors: &[Factor<'a>]) -> Cow<'a, Natural> {
    if let [Factor::Natural(x)] = factors {
        return Cow::Borrowed(*x);
    }
    let mut acc = Natural::ONE;
    for f in factors {
        match *f {
            Factor::Natural(x) => acc *= x,
            Factor::Words(w) => acc *= words_natural(w),
        }
    }
    Cow::Owned(acc)
}

/// The value of [`crate::exact::SignedSqrtRational::to_f64`] for `sign * sqrt(n/d)`, with
/// `n, d` nonnegative integers (`d > 0`) that need not be coprime, without
/// materializing the exact value. The result depends only on the ratio
/// `n/d` (see [`sqrt_ratio_to_f64`]). The sign is applied after rounding the
/// magnitude, so a nonzero value that underflows keeps its signed zero.
fn signed_sqrt_to_f64(sign: i8, n: &Natural, d: &Natural) -> f64 {
    if sign == 0 {
        return 0.0;
    }
    with_sign(sign, sqrt_ratio_to_f64(n, d))
}

/// [`signed_sqrt_to_f64`] for `n, d` already held in stack limbs.
fn signed_sqrt_limbs_to_f64(sign: i8, n: Limbs<INPUT_LIMBS>, d: Limbs<INPUT_LIMBS>) -> f64 {
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

// The `Factor` adapter stays with the conversion: the exact layer's
// `Limbs` knows integers, not the rounding layer's factor lists.
impl<const N: usize> Limbs<N> {
    /// `self *= f` for a factor in either form.
    fn checked_mul_factor(&mut self, f: &Factor<'_>) -> Option<()> {
        match *f {
            Factor::Natural(x) => self.checked_mul_assign(x),
            Factor::Words(w) => self.checked_mul_words(w),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use malachite_base::num::arithmetic::traits::Pow;
    use malachite_base::num::basic::traits::Zero;
    use malachite_nz::integer::Integer;
    use malachite_q::Rational;

    /// The rounding boundaries of the converter with expected bits derived
    /// by hand: each radicand is `n / 2^k`, so `sqrt` is the dyadic
    /// `sqrt(n) / 2^(k/2)` when `n` is a square.
    fn boundary_cases() -> Vec<(Rational, u64)> {
        let two = |k: u64| Integer::from(2u64).pow(k);
        let r = |n: Integer, k: u64| Rational::from_integers(n, two(k));
        let sq = |x: Integer| &x * &x;
        let one = Integer::ONE;
        let e = two(53);
        vec![
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
        ]
    }

    /// The boundary cases through the factor assembly of [`round_radical`]:
    /// the radicand whole, split into unreduced factors, its numerator read
    /// from a slab entry, and `X²·n/(X²·d)` with `X` passed twice, for `X`
    /// of one, two and more words, so that the products fall on both sides
    /// of the stack width. Both signs; a negative value that underflows keeps
    /// its signed zero.
    #[test]
    fn factor_assembly_rounds_the_boundary_cases() {
        use crate::exact::radical::SignedInts;
        let xs: Vec<Natural> = [
            Natural::from(3u32),
            (Natural::ONE << 64u32) + Natural::ONE,
            (Natural::ONE << 100u32) + Natural::from(7u32),
            (Natural::ONE << 150u32) + Natural::from(3u32),
        ]
        .into();
        let mut stack_and_big = [false; 2];
        for (radicand, bits) in boundary_cases() {
            let (n, d) = (radicand.numerator_ref(), radicand.denominator_ref());
            for sign in [1i8, -1] {
                let want = if sign < 0 { bits | 1 << 63 } else { bits };
                let whole = round_radical(sign, &[Factor::Natural(n)], &[Factor::Natural(d)]);
                assert_eq!(whole.to_bits(), want, "{sign} sqrt({radicand})");
                let three = Natural::from(3u32);
                let split = round_radical(
                    sign,
                    &[Factor::Natural(n), Factor::Natural(&three)],
                    &[Factor::Natural(&three), Factor::Natural(d)],
                );
                assert_eq!(split.to_bits(), want, "{sign} 3·{radicand}/3");
                let slab = SignedInts::from_naturals(&[(sign < 0, n.clone())]).unwrap();
                let read = round_radical(
                    if slab.is_negative(0) { -1 } else { 1 },
                    &[Factor::Words(slab.magnitude(0))],
                    &[Factor::Natural(d)],
                );
                assert_eq!(read.to_bits(), want, "{sign} slab {radicand}");
                for x in &xs {
                    let xd = x * x * d;
                    let words = crate::exact::radical::natural_words(x);
                    let got = round_radical(
                        sign,
                        &[
                            Factor::Words(&words),
                            Factor::Words(&words),
                            Factor::Natural(n),
                        ],
                        &[Factor::Natural(&xd)],
                    );
                    assert_eq!(got.to_bits(), want, "{sign} {x}²·{radicand}/{x}²");
                    let wide = (x * x * n).significant_bits().max(xd.significant_bits());
                    stack_and_big[usize::from(wide > 384)] = true;
                }
            }
        }
        assert_eq!(stack_and_big, [true, true], "both sides of the stack width");
    }

    /// Assembling factors whose products fit the stack limbs allocates
    /// nothing, also from multi-word slab entries, and gives the bits of the
    /// grid oracle.
    #[cfg(feature = "lie-generated")]
    #[test]
    fn factor_assembly_within_384_bits_does_not_allocate() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x8d05);
        let mut inputs = Vec::new();
        for i in 0..400u64 {
            // `x²·a` against `b·c`: at most 4·72 + 72 bits on either side.
            let w = 40 + i % 33;
            let x = random_natural(&mut rng, w);
            let [a, b, c] = [w, w, w].map(|w| random_natural(&mut rng, w));
            inputs.push((crate::exact::radical::natural_words(&x), a, b, c));
        }
        let calls = crate::audit_alloc::thread_allocation_calls;
        let mut got = Vec::with_capacity(inputs.len());
        let _ = FIXED_HITS.with(std::cell::Cell::get);
        let before = calls();
        for (x, a, b, c) in &inputs {
            got.push(round_radical(
                -1,
                &[Factor::Words(x), Factor::Words(x), Factor::Natural(a)],
                &[Factor::Natural(b), Factor::Natural(c)],
            ));
        }
        assert_eq!(calls() - before, 0);
        for ((x, a, b, c), got) in inputs.iter().zip(got) {
            let x = words_natural(x);
            let want = nearest_sqrt_bits_oracle(&(&x * &x * a), &(b * c));
            assert_eq!(got.to_bits(), want | 1 << 63);
        }
    }

    /// The shared converter at its rounding boundaries, with expected bits
    /// derived by hand. Each radicand is `n / 2^k`, so `sqrt` is the dyadic
    /// `sqrt(n) / 2^(k/2)` when `n` is a square.
    #[test]
    fn signed_sqrt_to_f64_rounding_boundaries() {
        let cases = boundary_cases();
        let one = Integer::ONE;
        let two = |k: u64| Integer::from(2u64).pow(k);
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
}
