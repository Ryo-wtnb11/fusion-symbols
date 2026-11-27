//! Fresh process: installs a one-shot non-default cache policy.
//! Base-only one-shot coefficient-cache budget contract.

mod common;

use crate::common::a1::exact_f;
use fusion_symbols::cache::{
    cache_budgets, configure_cache_budgets, reset, tier_stats, CacheBudgetError,
    CoefficientCacheBudgets, CoefficientCacheTier,
};

#[test]
fn zero_budget_is_retryable_then_compute_without_retention() {
    let invalid =
        CoefficientCacheBudgets::default().with_limit(CoefficientCacheTier::Wigner6j, usize::MAX);
    assert!(matches!(
        configure_cache_budgets(invalid),
        Err(CacheBudgetError::ExceedsMaximum {
            tier: CoefficientCacheTier::Wigner6j,
            ..
        })
    ));

    let budgets = CoefficientCacheBudgets::disabled();
    configure_cache_budgets(budgets).unwrap();
    assert_eq!(cache_budgets(), budgets);
    // F^{111}_1[1, 1] = 1/2 (TensorKitSectors 0.3.9 `SU2Irrep`).
    let value = exact_f(2, 2, 2, 2, 2, 2);
    assert!((value - 0.5).abs() <= 2.0 * f64::EPSILON);
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
