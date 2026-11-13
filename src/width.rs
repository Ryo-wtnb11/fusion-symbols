//! Width of one parallel section over the remaining items of a stage.
//!
//! The model prices width `k` of a section over items with units `u_i`,
//! measured at serial rate `ĉ = elapsed / units`, as
//!
//! ```text
//! T(1) = ĉ·Σu = est
//! T(k) = D(k) + ĉ·LS_k(u)          2 ≤ k ≤ cap, only if pays_at(est, 1, D(k))
//! ```
//!
//! and returns the argmin, ties to the smaller width. `LS_k` is the makespan
//! of greedy LPT list scheduling on `k` workers: an idealized load predictor
//! of the section's atomic-counter queue, not its measured makespan. `D(k)`
//! is the dispatch cost of the scope that will run the section.
//!
//! Every width is priced with unit inflation. Narrowing on a measured
//! per-share slowdown `φ̂(k)` is not done: a stage's calibration sample is
//! usually one item, so a slowdown measured against it amplifies that item's
//! noise, and a width priced out by one bad reading is never measured again.
//! Re-introduce it with section-level timings.
//!
//! All arithmetic is in `u128` nanoseconds with one integer division per
//! term, so every candidate is finite and defined. A nonzero product that
//! overflows prices its width at `u128::MAX`: that width never beats a finite
//! candidate, and the serial candidate is always kept. Saturation does not
//! order the remaining parallel widths, so it can move the choice to a
//! wider one.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::time::Duration;

use threadteam::pays_at;

/// Serial time measured over a nonempty sample of a stage's items.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Sample {
    pub(crate) elapsed_ns: u64,
    pub(crate) units: NonZeroU64,
}

/// `k* = argmin T(k)` over `1..=cap`, ties to the smaller width.
///
/// `remaining` holds the units of the items still to run, sorted descending;
/// `cap` is the widest width the caller may run, `min(B, n, k_mem)`. `cost`
/// is `|k| team.dispatch_cost(k)` of the dispatching scope.
///
/// Cost: `cap - 1` calls of `cost`, and `O(n log k)` for each width that
/// passes the pays filter and whose lower bound
/// `D(k) + ĉ·max(⌈Σu/k⌉, u_max)` beats the best width so far; at most
/// `O(n·cap·log cap)` in all, with one heap of at most `cap` loads.
pub(crate) fn choose_width(
    remaining: &[u64],
    sample: Sample,
    cap: NonZeroUsize,
    cost: impl Fn(NonZeroUsize) -> Duration,
) -> NonZeroUsize {
    plan(remaining, sample, cap, cost).0
}

/// Cheapest section among the feasible widths: min over `2..=cap` of `D(k)`;
/// `None` when `cap < 2` or every such `D(k)` is `Duration::MAX`.
pub(crate) fn cheapest_section(
    cap: NonZeroUsize,
    cost: impl Fn(NonZeroUsize) -> Duration,
) -> Option<Duration> {
    (2..=cap.get())
        .filter_map(NonZeroUsize::new)
        .map(cost)
        .filter(|&d| d != Duration::MAX)
        .min()
}

/// The chosen width and its objective in nanoseconds.
fn plan(
    remaining: &[u64],
    sample: Sample,
    cap: NonZeroUsize,
    cost: impl Fn(NonZeroUsize) -> Duration,
) -> (NonZeroUsize, u128) {
    debug_assert!(remaining.is_sorted_by(|a, b| a >= b));
    // A slice holds fewer than 2^61 items of at most 2^64 units: no overflow.
    let total: u128 = remaining.iter().map(|&u| u128::from(u)).sum();
    let u_max = remaining.first().map_or(0, |&u| u128::from(u));
    let est_ns = scaled(total, sample);
    let est = duration_from_ns(est_ns);

    let mut best = (NonZeroUsize::MIN, est_ns);
    let mut loads = BinaryHeap::new();
    for width in (2..=cap.get()).filter_map(NonZeroUsize::new) {
        let d = cost(width);
        // Also rejects `Duration::MAX`, the unavailable width.
        if !pays_at(est, 1, d) {
            continue;
        }
        let objective = |ls| d.as_nanos().saturating_add(scaled(ls, sample));
        let lower = total.div_ceil(width.get() as u128).max(u_max);
        if objective(lower) >= best.1 {
            continue;
        }
        let t = objective(lpt_makespan(remaining, width, &mut loads));
        if t < best.1 {
            best = (width, t);
        }
    }
    best
}

/// `x · elapsed / units`, or `u128::MAX` when the product overflows.
fn scaled(x: u128, sample: Sample) -> u128 {
    x.checked_mul(u128::from(sample.elapsed_ns))
        .map_or(u128::MAX, |p| p / u128::from(sample.units.get()))
}

/// `ns` as a `Duration`, saturating at `Duration::MAX`.
fn duration_from_ns(ns: u128) -> Duration {
    const NANOS: u128 = 1_000_000_000;
    u64::try_from(ns / NANOS).map_or(Duration::MAX, |secs| {
        Duration::new(secs, (ns % NANOS) as u32)
    })
}

/// Makespan of greedy list scheduling of `items` (sorted descending) on
/// `width` workers, each item to the least-loaded worker.
fn lpt_makespan(items: &[u64], width: NonZeroUsize, loads: &mut BinaryHeap<Reverse<u128>>) -> u128 {
    loads.clear();
    loads.extend(std::iter::repeat_n(Reverse(0), width.get()));
    let mut makespan = 0;
    for &u in items {
        if let Some(mut least) = loads.peek_mut() {
            least.0 += u128::from(u);
            makespan = makespan.max(least.0);
        }
    }
    makespan
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(k: usize) -> NonZeroUsize {
        NonZeroUsize::new(k).unwrap()
    }

    /// One nanosecond per unit.
    fn unit_rate(units: u64) -> Sample {
        Sample {
            elapsed_ns: units,
            units: NonZeroU64::new(units).unwrap(),
        }
    }

    fn ns(n: u64) -> Duration {
        Duration::from_nanos(n)
    }

    /// `D(k)` from a row starting at width 2; widths past the row are
    /// unavailable.
    fn row(costs: &'static [u64]) -> impl Fn(NonZeroUsize) -> Duration {
        move |k| costs.get(k.get() - 2).map_or(Duration::MAX, |&c| ns(c))
    }

    #[test]
    fn saturated_items_choose_by_lpt_makespan_not_even_split() {
        // [10, 6, 6, 6], ĉ = 1, D = 1: LS_2 = 16, LS_3 = 12, LS_4 = 10, so
        // T = 28, 17, 13, 11 and width 4 wins (a ⌈Σu/u_max⌉ = 3 ceiling loses).
        let items = [10, 6, 6, 6];
        let mut loads = BinaryHeap::new();
        let ls: Vec<_> = (1..=4)
            .map(|k| lpt_makespan(&items, w(k), &mut loads))
            .collect();
        assert_eq!(ls, [28, 16, 12, 10]);
        let flat = |_| ns(1);
        assert_eq!(plan(&items, unit_rate(28), w(4), flat), (w(4), 11));

        // Negative control: [10, 1, 1] with D(2) = 1, D(3) = 2. An even split
        // (Σu/k = 6, 4) would pick 3 (T = 7, 6); LS_2 = LS_3 = 10 gives
        // T = 11, 12, so 2.
        assert_eq!(
            plan(&[10, 1, 1], unit_rate(12), w(3), row(&[1, 2])),
            (w(2), 11)
        );
    }

    #[test]
    fn non_monotone_costs_pick_the_argmin() {
        // 12 items of 6, ĉ = 1, est = 72. D(2..5) = [20, 2, 10, 20]:
        // T = 72, 20 + 36, 2 + 24, 10 + 18, 20 + 18 = 72, 56, 26, 28, 38.
        let items = [6; 12];
        let costs = row(&[20, 2, 10, 20]);
        assert_eq!(plan(&items, unit_rate(72), w(5), costs), (w(3), 26));
        // Negative control: D(5) = 1 makes the widest width cheapest:
        // T(5) = 1 + LS_5 = 1 + 18 = 19.
        assert_eq!(
            plan(&items, unit_rate(72), w(5), row(&[20, 2, 10, 1])),
            (w(5), 19)
        );
    }

    #[test]
    fn cap_bounds_the_search_without_clipping() {
        // [10; 8], ĉ = 1, est = 80, D(2..8) = [1, 1, 100, 1, 1, 1, 1].
        // Within cap 4: width 4 fails pays_at(80, 1, 100) (80 < 300), and
        // T(2) = 41, T(3) = 1 + LS_3 = 1 + 30 = 31.
        let items = [10; 8];
        const COSTS: &[u64] = &[1, 1, 100, 1, 1, 1, 1];
        assert_eq!(plan(&items, unit_rate(80), w(4), row(COSTS)), (w(3), 31));
        // Negative control: the full cap reaches T(8) = 1 + 10 = 11.
        assert_eq!(plan(&items, unit_rate(80), w(8), row(COSTS)), (w(8), 11));
    }

    #[test]
    fn widths_that_do_not_pay_stay_serial() {
        // est = 80: D = 27 needs 81 and never pays; neither does an
        // unavailable row.
        let items = [10; 8];
        let s = unit_rate(80);
        assert_eq!(plan(&items, s, w(8), |_| ns(27)), (w(1), 80));
        assert_eq!(plan(&items, s, w(8), |_| Duration::MAX), (w(1), 80));
        assert_eq!(plan(&items, s, w(1), |_| ns(0)), (w(1), 80));
        // Negative control: D = 26 pays (80 ≥ 78), T(8) = 26 + 10 = 36.
        assert_eq!(plan(&items, s, w(8), |_| ns(26)), (w(8), 36));
    }

    #[test]
    fn zero_elapsed_time_selects_width_one() {
        // est = 0: a zero dispatch cost pays, but T(k) = 0 ties T(1) = 0.
        let items = [10; 8];
        let zero = Sample {
            elapsed_ns: 0,
            units: NonZeroU64::new(80).unwrap(),
        };
        assert_eq!(choose_width(&items, zero, w(8), |_| ns(0)), w(1));
        assert_eq!(choose_width(&items, zero, w(8), |_| ns(1)), w(1));
        // Negative control: one nanosecond per unit and D = 0 gives T(8) = 10.
        assert_eq!(choose_width(&items, unit_rate(80), w(8), |_| ns(0)), w(8));
        // An empty stage has nothing to run in parallel.
        assert_eq!(choose_width(&[], unit_rate(1), w(8), |_| ns(0)), w(1));
    }

    #[test]
    fn overflowing_products_fall_back_toward_serial() {
        // Four items of m = 2^64 - 1 units at m ns each: est = 4m² overflows
        // u128 and saturates. Up to width 2 every T(k) holds 2m² and
        // saturates too, so the tie keeps width 1.
        let huge = Sample {
            elapsed_ns: u64::MAX,
            units: NonZeroU64::MIN,
        };
        let items = [u64::MAX; 4];
        let m2 = u128::from(u64::MAX).pow(2);
        assert_eq!(plan(&items, huge, w(2), |_| ns(1)), (w(1), u128::MAX));
        // A width that cannot be priced is not chosen either.
        assert_eq!(
            plan(&items, huge, w(4), |_| Duration::MAX - ns(1)),
            (w(1), u128::MAX)
        );
        // Negative control: width 4 holds one item per worker, so
        // T(4) = 1 + m² stays finite and beats the saturated serial estimate.
        assert_eq!(plan(&items, huge, w(4), |_| ns(1)), (w(4), m2 + 1));
        assert_eq!(duration_from_ns(u128::MAX), Duration::MAX);
        assert_eq!(duration_from_ns(1_500_000_000), Duration::from_millis(1500));
    }

    #[test]
    fn cheapest_section_spans_only_feasible_widths() {
        // Width 1 and widths above the cap are not sections of this stage.
        let costs = |k: NonZeroUsize| match k.get() {
            1 => ns(0),
            2 => ns(9),
            3 => Duration::MAX,
            4 => ns(4),
            _ => ns(0),
        };
        assert_eq!(cheapest_section(w(4), costs), Some(ns(4)));
        assert_eq!(cheapest_section(w(3), costs), Some(ns(9)));
        assert_eq!(cheapest_section(w(1), costs), None);
        assert_eq!(cheapest_section(w(8), |_| Duration::MAX), None);
        // Negative control: the zero row is a valid cheapest section.
        assert_eq!(cheapest_section(w(5), costs), Some(ns(0)));
    }
}
