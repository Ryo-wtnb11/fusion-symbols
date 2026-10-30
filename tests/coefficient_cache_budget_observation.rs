//! Fresh process: observing statistics must freeze the untouched default policy.
//! Cache policy observation freezes the compiled default in a fresh process.

mod common;

use fusion_symbols::cache::{
    cache_budgets, configure_cache_budgets, tier_stats, CacheBudgetError, CoefficientCacheBudgets,
    CoefficientCacheTier,
};

#[test]
fn observing_cache_stats_freezes_the_default_policy() {
    let default = CoefficientCacheBudgets::default();
    let _ = tier_stats(CoefficientCacheTier::Wigner3j);
    assert_eq!(cache_budgets(), default);
    assert_eq!(
        configure_cache_budgets(CoefficientCacheBudgets::disabled()),
        Err(CacheBudgetError::AlreadyInitialized)
    );
}
