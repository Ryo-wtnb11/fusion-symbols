//! Isolated process: exact tier counts around trim must exclude unrelated cache traffic.
//! Public base-tier cache trim contract.

mod common;

use crate::common::a1;
use a1::exact_f;
use fusion_symbols::cache::{cache_budgets, reset, tier_stats, trim_to, CoefficientCacheTier};
use fusion_symbols::lie::su2::{wigner_3j, wigner_6j};

#[test]
fn trim_is_tier_local_preserves_budget_and_allows_refill() {
    let _guard = common::serial();
    reset();
    let budget = cache_budgets();
    let _ = wigner_3j(2, 2, 2, 0, 0, 0);
    let _ = wigner_6j(2, 2, 2, 2, 2, 2);
    let _ = exact_f(2, 2, 2, 2, 2, 2);
    for tier in [
        CoefficientCacheTier::Wigner3j,
        CoefficientCacheTier::Wigner6j,
        CoefficientCacheTier::FSymbol,
    ] {
        let selected = tier_stats(tier);
        assert!(selected.entries > 0);
        let report = trim_to(tier, 0);
        assert_eq!(report.removed_entries, selected.entries);
        assert_eq!(tier_stats(tier).entries, 0);
    }
    assert_eq!(cache_budgets(), budget);
    let _ = wigner_6j(2, 2, 2, 2, 2, 2);
    assert_eq!(tier_stats(CoefficientCacheTier::Wigner6j).entries, 1);
    assert_eq!(cache_budgets(), budget);
}

/// A trim target between zero and the retained charge removes some entries,
/// keeps others, and leaves the tier within the target.
#[test]
fn partial_trim_keeps_some_entries_within_the_target() {
    let _guard = common::serial();
    reset();
    for j in [20, 40, 60, 80, 100, 120] {
        let _ = wigner_6j(j, j, j, j, j, j);
    }
    let before = tier_stats(CoefficientCacheTier::Wigner6j);
    assert!(before.entries >= 2);
    let target = before.bytes / 2;
    let report = trim_to(CoefficientCacheTier::Wigner6j, target);
    assert!(report.removed_entries > 0 && report.remaining_entries > 0);
    assert_eq!(
        report.removed_entries + report.remaining_entries,
        before.entries
    );
    assert!(report.remaining_charged_bytes <= target);
    assert_eq!(
        tier_stats(CoefficientCacheTier::Wigner6j).entries,
        report.remaining_entries
    );
}
