//! Fresh process: installs a one-shot non-default cache policy.
//! Base-only one-shot coefficient-cache budget contract.

mod common;

use fusion_symbols::cache::{
    cache_budgets, configure_cache_budgets, reset, tier_stats, CacheBudgetError,
    CoefficientCacheBudgets, CoefficientCacheTier,
};
use fusion_symbols::lie::su2::wigner_6j;

#[test]
fn zero_budget_is_retryable_then_compute_without_retention() {
    let invalid =
        CoefficientCacheBudgets::default().with_limit(CoefficientCacheTier::Wigner3j, usize::MAX);
    assert!(matches!(
        configure_cache_budgets(invalid),
        Err(CacheBudgetError::ExceedsMaximum {
            tier: CoefficientCacheTier::Wigner3j,
            ..
        })
    ));

    let budgets = CoefficientCacheBudgets::disabled();
    configure_cache_budgets(budgets).unwrap();
    assert_eq!(cache_budgets(), budgets);
    let value = wigner_6j(2, 2, 2, 2, 2, 2);
    assert!((value.to_f64() - 1.0 / 6.0).abs() < 1e-14);
    let stats = tier_stats(CoefficientCacheTier::Wigner6j);
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.bytes, 0);
    assert_eq!(stats.evictions, 1);
    reset();
    assert_eq!(cache_budgets(), budgets, "reset must preserve the policy");
    assert_eq!(
        configure_cache_budgets(CoefficientCacheBudgets::default()),
        Err(CacheBudgetError::AlreadyInitialized)
    );
}
