//! Exact bounded Manhattan ranking shared by products and SU(N) Dynkin labels.
//!
//! Tuples `x` of nonnegative integers, each below an optional bound `L_j`, are
//! ordered by `Σ x_j`, then lexicographically. This is TensorKitSectors 0.3.10
//! `src/auxiliary.jl::to_manhattan_index` minus one (and, for unbounded
//! coordinates, SUNRepresentations 0.4.0 `src/sector.jl::findindex`), with the
//! sequential shell and coordinate searches replaced by binary searches.
//! Arithmetic is exact: inclusion–exclusion cancels large signed terms, so a
//! saturating or wrapping integer would corrupt the count.

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

/// Failure to enumerate objects or represent an enumeration result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EnumerationError {
    /// This category value does not provide object enumeration.
    NotEnumerable,
    /// The index is outside this finite category's enumeration.
    OutOfRange,
    /// The count, index, or object label exceeds its integer representation.
    Overflow,
}

impl std::fmt::Display for EnumerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotEnumerable => "this category does not provide object enumeration",
            Self::OutOfRange => "enumeration index is outside the finite object range",
            Self::Overflow => "enumeration count, index or object label exceeds its integer range",
        })
    }
}

impl std::error::Error for EnumerationError {}

/// A category's object count as a product bound.
pub(crate) enum Count {
    /// Exactly this many objects.
    Finite(BigInt),
    /// Finitely many, more than `u64::MAX`, exact value unknown: what a public
    /// [`crate::category::FusionCategory::object_count`] reports as `Overflow`.
    AboveU64,
    /// Infinitely many.
    Infinite,
}

impl Count {
    /// The product count: infinite if any factor is, else unknown above
    /// `u64::MAX` if any factor is (every count is at least one), else exact.
    pub(crate) fn product<'a>(counts: impl IntoIterator<Item = &'a Count>) -> Count {
        let mut exact = Some(BigInt::one());
        let mut infinite = false;
        for count in counts {
            match count {
                Count::Finite(n) => exact = exact.map(|p| p * n),
                Count::AboveU64 => exact = None,
                Count::Infinite => infinite = true,
            }
        }
        match (infinite, exact) {
            (true, _) => Count::Infinite,
            (false, None) => Count::AboveU64,
            (false, Some(n)) => Count::Finite(n),
        }
    }

    /// The coordinate bound used for ranking. `AboveU64` is ranked as
    /// unbounded, which is exact for every query that can succeed:
    ///
    /// A bound `L ≥ 2^64` restricts only tuples with `x ≥ L`, whose sum is at
    /// least `2^64`. Ranking and unranking evaluate tuple counts only for sums
    /// `t ≤ d`, the shell of the queried tuple, and for `t < L` the bound
    /// excludes nothing, so dropping it changes no evaluated count when
    /// `d < 2^64`. When `d ≥ 2^64`, the unbounded coordinate alone gives at
    /// least `min(d, L) ≥ 2^64` tuples of smaller sum, so the index is at
    /// least `2^64`: [`rank_checked`] reports `Overflow`, and an index
    /// `k ≤ u64::MAX` never lies in such a shell. A count `≥ 2^64` also never
    /// makes `k` out of range.
    pub(crate) fn bound(&self) -> Option<BigInt> {
        match self {
            Count::Finite(n) => Some(n.clone()),
            Count::AboveU64 | Count::Infinite => None,
        }
    }
}

/// [`rank`] for factor counts that may be [`Count::AboveU64`]; `None` exactly
/// when the index is at least `2^64` and an unknown bound could matter (see
/// [`Count::bound`]).
pub(crate) fn rank_checked(counts: &[&Count], x: &[BigInt]) -> Option<BigInt> {
    let d: BigInt = x.iter().sum();
    if d > BigInt::from(u64::MAX) && counts.iter().any(|c| matches!(c, Count::AboveU64)) {
        return None;
    }
    let bounds: Vec<_> = counts.iter().map(|c| c.bound()).collect();
    Some(rank(&bounds, x))
}

/// `C(m, n)`, zero for `m < n` (including negative `m`).
fn binomial(m: &BigInt, n: usize) -> BigInt {
    if *m < BigInt::from(n) {
        return BigInt::zero();
    }
    let mut value = BigInt::one();
    for i in 0..n {
        // `value = C(m, i)`, so `value * (m - i)` is divisible by `i + 1`.
        value = value * (m - i) / (i + 1);
    }
    value
}

/// Number of tuples within `bounds` whose coordinate sum is at most `t`.
pub(crate) fn cumulative(bounds: &[Option<BigInt>], t: &BigInt) -> BigInt {
    if t.is_negative() {
        return BigInt::zero();
    }
    let n = bounds.len();
    let finite: Vec<&BigInt> = bounds.iter().flatten().collect();
    if finite.len() == n {
        // Every tuple is counted once `t` reaches the largest coordinate sum.
        let top: BigInt = finite.iter().map(|&l| l - 1).sum();
        if *t >= top {
            return finite.iter().copied().product();
        }
    }
    // Σ_{S ⊆ finite} (-1)^|S| C(t - Σ_S L + n, n); a subset whose bounds already
    // exceed `t` contributes zero, as does every superset of it.
    fn terms(rest: &[&BigInt], budget: &BigInt, sign: bool, n: usize, acc: &mut BigInt) {
        let Some((&l, rest)) = rest.split_first() else {
            let term = binomial(&(budget + n), n);
            if sign {
                *acc -= term;
            } else {
                *acc += term;
            }
            return;
        };
        terms(rest, budget, sign, n, acc);
        let reduced = budget - l;
        if !reduced.is_negative() {
            terms(rest, &reduced, !sign, n, acc);
        }
    }
    let mut acc = BigInt::zero();
    terms(&finite, t, false, n, &mut acc);
    acc
}

/// Exact tuple count, or `None` when a coordinate is unbounded.
pub(crate) fn total(bounds: &[Option<BigInt>]) -> Option<BigInt> {
    bounds
        .iter()
        .map(Option::as_ref)
        .product::<Option<BigInt>>()
}

/// Zero-based Manhattan index of `x`, which must lie within `bounds`.
pub(crate) fn rank(bounds: &[Option<BigInt>], x: &[BigInt]) -> BigInt {
    debug_assert_eq!(bounds.len(), x.len());
    let mut s: BigInt = x.iter().sum();
    let mut index = cumulative(bounds, &(&s - 1));
    for j in 0..x.len().saturating_sub(1) {
        let suffix = &bounds[j + 1..];
        index += cumulative(suffix, &s) - cumulative(suffix, &(&s - &x[j]));
        s -= &x[j];
    }
    index
}

/// Largest `v` in `lo..=hi` with `pred(v)`, given `pred(lo)` and monotone `pred`.
fn last_true(mut lo: BigInt, mut hi: BigInt, pred: impl Fn(&BigInt) -> bool) -> BigInt {
    while lo < hi {
        let mid = (&lo + &hi + 1) / 2;
        if pred(&mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// The tuple of Manhattan index `k`, or `None` past a finite enumeration.
pub(crate) fn unrank(bounds: &[Option<BigInt>], k: &BigInt) -> Option<Vec<BigInt>> {
    let hi = match total(bounds) {
        Some(count) if *k >= count => return None,
        Some(_) => bounds.iter().flatten().map(|l| l - 1).sum(),
        None => {
            // An unbounded coordinate gives at least `t + 1` tuples of sum `≤ t`.
            let mut hi = BigInt::one();
            while cumulative(bounds, &hi) <= *k {
                hi *= 2;
            }
            hi
        }
    };
    // Shell `d`: the largest `d` with fewer than `k + 1` tuples of smaller sum.
    let d = last_true(BigInt::zero(), hi, |d| cumulative(bounds, &(d - 1)) <= *k);
    let mut rem = k - cumulative(bounds, &(&d - 1));
    let mut s = d;
    let mut x = Vec::with_capacity(bounds.len());
    for j in 0..bounds.len().saturating_sub(1) {
        let suffix = &bounds[j + 1..];
        let full = cumulative(suffix, &s);
        let before = |v: &BigInt| &full - cumulative(suffix, &(&s - v));
        let vmax = match &bounds[j] {
            Some(l) if l - 1u8 < s => l - 1u8,
            _ => s.clone(),
        };
        let v = last_true(BigInt::zero(), vmax, |v| before(v) <= rem);
        rem -= before(&v);
        s -= &v;
        x.push(v);
    }
    x.push(s);
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(bounds: &[Option<u64>]) -> Vec<Option<BigInt>> {
        bounds.iter().map(|l| l.map(BigInt::from)).collect()
    }

    // Brute-force enumeration by sum, then lexicographically.
    fn brute(bounds: &[Option<u64>], cap: u64) -> Vec<Vec<u64>> {
        let lim: Vec<u64> = bounds.iter().map(|l| l.unwrap_or(cap + 1)).collect();
        let mut all = vec![vec![]];
        for &l in &lim {
            all = all
                .into_iter()
                .flat_map(|p: Vec<u64>| {
                    (0..l).map(move |v| {
                        let mut p = p.clone();
                        p.push(v);
                        p
                    })
                })
                .collect();
        }
        all.retain(|x| x.iter().sum::<u64>() <= cap);
        all.sort_by_key(|x| (x.iter().sum::<u64>(), x.clone()));
        all
    }

    #[test]
    fn rank_and_unrank_match_brute_force() {
        for bounds in [
            vec![Some(2), Some(3)],
            vec![Some(2), None],
            vec![None, Some(2), Some(3)],
            vec![None, None, None],
            vec![Some(1), Some(4), None],
            vec![Some(3), Some(3), Some(3), Some(2)],
        ] {
            let big = b(&bounds);
            let cap = 6;
            for (k, x) in brute(&bounds, cap).iter().enumerate() {
                let xb: Vec<BigInt> = x.iter().map(|&v| BigInt::from(v)).collect();
                assert_eq!(rank(&big, &xb), BigInt::from(k), "{bounds:?} {x:?}");
                assert_eq!(unrank(&big, &BigInt::from(k)), Some(xb));
            }
        }
    }
}
