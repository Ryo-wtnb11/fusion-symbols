//! A certified accelerator for [`super::round_radical`]: a fixed-point
//! estimate of a product of square roots and the decision whether its
//! nearest `f64` is certainly the exact value's. It never decides a bit on
//! its own authority: it returns a value only when the proved error bound
//! puts the exact value and the estimate in the same rounding interval, and
//! `None` otherwise, when the caller converts exactly.
//!
//! An [`Est`] is `m·2^e` with a normalized mantissa `2^127 ≤ m < 2^128` and
//! an `i64` exponent; let `u = 2^-127`. Every exponent operation is checked,
//! and an overflow is `None`. Every estimate is a truncation from below of
//! the value it stands for, by less than one unit `2^e ≤ u·m·2^e`, so the
//! errors only count:
//!
//! - [`top_words`] / [`top_natural`]: an integer's top 128 bits;
//! - [`root`]: `√(p/q)` as the floor of an exact integer square root, one
//!   truncation;
//! - [`mul`]: the exact 256-bit product of four `u64 × u64` partials,
//!   truncated to 128 bits.
//!
//! A CGC entry `|X|·√ρ_μ·(1/√n^c)·√n^a·√n^b` ([`entry`]) takes nine such
//! truncations: `|X|`, four roots and four multiplies (the column constant
//! `√ρ_μ/√n^c` is one). So `V̂ ≤ V < V̂(1 + u)^9`, and with `V̂ < 2^(e+128)`,
//! `V − V̂ < 2^(e+128)·((1 + u)^9 − 1) < 18.001·2^e`: less than 19 units. An
//! F block entry (`2R + 3 = 11` truncations for `R = 4`) is within 23 units,
//! an R, A or B entry (`R = 2`, seven) within 15. [`decide`] rounds the two
//! ends `m ∓ MARGIN` of an interval holding `V` with the language's correctly
//! rounded `u128 as f64` and returns them when they agree and the range is
//! normal; otherwise the caller converts with `round_radical`.
//!
//! `docs/gauge_cartan.md` §4.7 states the value rule this module
//! accelerates.

use std::borrow::Cow;

use malachite_base::num::arithmetic::traits::FloorSqrt;
use malachite_base::num::logic::traits::SignificantBits;
use malachite_nz::natural::Natural;

use super::Factor;
use crate::exact::radical::words_natural;

/// `m·2^e` with `2^127 ≤ m < 2^128`; zero is never an estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Est {
    m: u128,
    e: i64,
}

/// The lowest normalized mantissa.
const HIGH: u128 = 1 << 127;
const LOW: u128 = u64::MAX as u128;

/// The safety margin of [`decide`], in mantissa units `2^e`: the half-width
/// of the interval whose ends must round alike. It is
/// a margin over the proved entry errors (the module doc: below 19 units for
/// a CGC entry, 23 for F and 15 for R, A and B), the same constant for every
/// value shape, and checked on both sides although every error is one-sided.
pub(crate) const MARGIN: u128 = 1 << 10;

impl Est {
    /// `m·2^e` for a normalized `m`.
    #[cfg(test)]
    pub(crate) fn new(m: u128, e: i64) -> Self {
        assert!(m >= HIGH, "a normalized mantissa");
        Self { m, e }
    }
}

/// `x ≥ 1` given as little-endian words (high zero words allowed), or `None`
/// for zero or an exponent beyond `i64`.
///
/// With `L` the bit length, `m` holds the top 128 bits and `e = L − 128`.
/// For `L ≤ 128` this is exact. Otherwise the discarded part is below `2^e`,
/// and `2^e ≤ u·m·2^e ≤ u·x` since `m ≥ 2^127`, so `x(1 − u) < m·2^e ≤ x`.
pub(crate) fn top_words(words: &[u64]) -> Option<Est> {
    let used = words.len() - words.iter().rev().take_while(|&&w| w == 0).count();
    let (&hi, rest) = words[..used].split_last()?;
    let lz = hi.leading_zeros();
    let (mid, low) = match rest {
        [.., l, m] => (*m, *l),
        [m] => (*m, 0),
        [] => (0, 0),
    };
    // `hi ≠ 0` has `lz` leading zeros, so the shift keeps every bit and sets
    // bit 127; the low word supplies the `lz` bits below.
    let two = (u128::from(hi) << 64) | u128::from(mid);
    let m = if lz == 0 {
        two
    } else {
        (two << lz) | (u128::from(low) >> (64 - lz))
    };
    let e = i64::try_from(used)
        .ok()?
        .checked_mul(64)?
        .checked_sub(i64::from(lz))?
        .checked_sub(128)?;
    Some(Est { m, e })
}

/// [`top_words`] of a `Natural`, whose top 128 bits are read by one shift.
pub(crate) fn top_natural(x: &Natural) -> Option<Est> {
    let bits = x.significant_bits();
    if bits == 0 {
        return None;
    }
    let e = i64::try_from(bits).ok()?.checked_sub(128)?;
    let m = if bits <= 128 {
        u128::try_from(x).ok()? << (128 - bits)
    } else {
        u128::try_from(&(x >> (bits - 128))).ok()?
    };
    Some(Est { m, e })
}

/// `a·b` exactly as `(high, low)` 128-bit halves, from four `u64 × u64`
/// partials; no partial is dropped. The middle sum is below `3·2^64`.
fn wide_mul(a: u128, b: u128) -> (u128, u128) {
    let (a1, a0, b1, b0) = (a >> 64, a & LOW, b >> 64, b & LOW);
    let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
    let mid = (p00 >> 64) + (p01 & LOW) + (p10 & LOW);
    let low = (p00 & LOW) | ((mid & LOW) << 64);
    let high = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (high, low)
}

/// `a·b` truncated to 128 bits, or `None` when the exponent leaves `i64`.
///
/// `P = a.m·b.m` lies in `[2^254, 2^256)`. If `P ≥ 2^255` the mantissa is
/// `⌊P/2^128⌋` and the exponent gains 128, otherwise `⌊P/2^127⌋` and 127: two
/// minimum mantissas `2^127·2^127` take the second branch, two maximum ones the
/// first. Either way `2^127 ≤ m < 2^128` and the discarded part is below the
/// unit, so `P(1 − u) < m·2^shift ≤ P`.
pub(crate) fn mul(a: Est, b: Est) -> Option<Est> {
    let (high, low) = wide_mul(a.m, b.m);
    let e = a.e.checked_add(b.e)?;
    if high >= HIGH {
        Some(Est {
            m: high,
            e: e.checked_add(128)?,
        })
    } else {
        Some(Est {
            m: (high << 1) | (low >> 127),
            e: e.checked_add(127)?,
        })
    }
}

/// `√(p/q)` for `p, q ≥ 1`, truncated: the top 128 bits of
/// `⌊√⌊p·2^(2k)/q⌋⌋ = ⌊√(p/q)·2^k⌋` (`⌊√⌊y⌋⌋ = ⌊√y⌋` for real `y ≥ 0`), whose
/// truncation to 128 bits is again the floor of `√(p/q)` at a power-of-two
/// scale. So the estimate is `≤ √(p/q)` and less than one unit below.
///
/// `2k` is the even number `≥ 255 + L(q) − L(p)` for the bit lengths `L`, so
/// `p·2^(2k)/q > 2^(L(p) − 1 − L(q) + 2k) ≥ 2^254` and the root has at least
/// 128 bits; `p·2^(2k)/q < 2^257` keeps it below 2^129. A negative `2k`
/// shifts `q` instead. `None` when a width leaves `i64`.
pub(crate) fn root(p: &Natural, q: &Natural) -> Option<Est> {
    let lp = i64::try_from(p.significant_bits()).ok()?;
    let lq = i64::try_from(q.significant_bits()).ok()?;
    if lp == 0 || lq == 0 {
        return None;
    }
    let least = 255i64.checked_add(lq)?.checked_sub(lp)?;
    let two_k = least.checked_add(least.rem_euclid(2))?;
    let scaled = if two_k >= 0 {
        (p << u64::try_from(two_k).ok()?) / q
    } else {
        p / (q << u64::try_from(two_k.checked_neg()?).ok()?)
    };
    let r = top_natural(&scaled.floor_sqrt())?;
    Some(Est {
        m: r.m,
        e: r.e.checked_sub(two_k / 2)?,
    })
}

/// [`root`] of factors given in either form.
pub(crate) fn root_of(p: Factor<'_>, q: Factor<'_>) -> Option<Est> {
    root(&natural(p), &natural(q))
}

/// A factor as a big integer, borrowed when it is one.
fn natural(f: Factor<'_>) -> Cow<'_, Natural> {
    match f {
        Factor::Natural(x) => Cow::Borrowed(x),
        Factor::Words(w) => Cow::Owned(words_natural(w)),
    }
}

/// `V̂ = |X|·k·s_a·s_b` for the slab words `x` of `|X|`: one truncation for
/// `|X|` and three multiplies, each less than one unit from below.
pub(crate) fn entry(x: &[u64], k: Est, sa: Est, sb: Est) -> Option<Est> {
    mul(mul(mul(top_words(x)?, k)?, sa)?, sb)
}

/// `2^128`, the rounding of every integer in `[2^128, 2^128 + 2^74)`.
const TWO_128: f64 = f64::from_bits((1023 + 128) << 52);

/// The binades `b` of `V̂ ∈ [2^b, 2^(b+1))` the fast path accepts:
/// `EMIN + 1 ..= EMAX − 1` of `f64`.
const BINADES: std::ops::RangeInclusive<i64> = -1021..=1022;

/// The nearest `f64` of a value `±V` whose estimate `v = m·2^e` satisfies
/// `|V − v| < 19·2^e` (or the F, R, A and B bounds of the module doc, all
/// below [`MARGIN`]), when that is certainly the nearest `f64` of `V`;
/// otherwise `None`, and the caller takes the exact conversion.
///
/// The ends `m ∓ MARGIN` of an interval holding `V/2^e` are rounded by Rust's
/// `u128 as f64`, which rounds to nearest, ties to even. Rounding is
/// monotone, so `V/2^e` rounds between the two; when they are equal it
/// rounds to them. An upper end at or past `2^128` lies in
/// `[2^128, 2^128 + MARGIN)`, within half a unit in the last place (`2^74`)
/// of `2^128`, so its rounding is exactly `2^128` and is taken as that
/// constant: a floor estimate of an exact power of two has `m` just below
/// `2^128` and must not fall back for it.
///
/// The result is that rounding scaled by `2^e`, done exactly by adding `e`
/// to the exponent field. Rounding commutes with the scaling when the result
/// is normal and finite, which [`BINADES`] ensures: the rounded end lies in
/// `[2^126, 2^128]`, so the result's exponent is `b − 1 ..= b + 1` with
/// `b = e + 127`, inside `−1022 ..= 1023`; and `V` itself, below
/// `2^(e+128)(1 + 2^-121) ≤ 2^1023(1 + 2^-121) < (2 − 2^-53)·2^1023` and above
/// `2^(e+127)(1 − 2^-121) > 2^-1022`, is normal and finite. An exact midpoint
/// or a value within `MARGIN − 19` units of one falls back, since the ends
/// round apart; a representable value is returned.
pub(crate) fn decide(neg: bool, v: Est) -> Option<f64> {
    let b = v.e.checked_add(127)?;
    if !BINADES.contains(&b) {
        return None;
    }
    let low = (v.m - MARGIN) as f64;
    let high = v.m.checked_add(MARGIN).map_or(TWO_128, |h| h as f64);
    if low != high {
        return None;
    }
    let bits = low.to_bits();
    let biased = i64::try_from(bits >> 52).ok()?.checked_add(v.e)?;
    let magnitude = f64::from_bits((u64::try_from(biased).ok()? << 52) | (bits & ((1 << 52) - 1)));
    Some(if neg { -magnitude } else { magnitude })
}

#[cfg(test)]
mod tests;
