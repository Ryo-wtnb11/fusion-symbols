//! Isolated process: exact tier counts around trim must exclude unrelated cache traffic.
//! Public base-tier cache trim contract.

mod common;

use crate::common::a1;
use a1::exact_f;
use fusion_symbols::cache::{cache_budgets, reset, tier_stats, trim_to, CoefficientCacheTier};
use fusion_symbols::lie::su2::{wigner_3j, wigner_6j};

#[test]
fn trim_is_tier_local_preserves_budget_and_allows_refill() {
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
