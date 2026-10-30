//! Fresh process: installs a one-shot disabled generated-cache policy.
//! Generated-family zero-budget coefficient-cache contract.

#![cfg(feature = "lie-generated")]

mod common;

use crate::common::lie as lie_test;
use fusion_symbols::cache::{
    cache_budgets, configure_cache_budgets, tier_stats, CoefficientCacheBudgets,
    CoefficientCacheTier,
};
use fusion_symbols::category::FusionCategory;

#[test]
fn disabled_policy_preserves_product_value_without_retention() {
    for &tier in CoefficientCacheTier::ALL {
        assert_eq!(CoefficientCacheBudgets::disabled().limit(tier), 0);
        assert_eq!(
            CoefficientCacheBudgets::default()
                .with_limit(tier, 7)
                .limit(tier),
            7
        );
    }
    let budgets = CoefficientCacheBudgets::disabled();
    configure_cache_budgets(budgets).unwrap();
    let three = lie_test::irrep(&[1, 0]).unwrap();
    let three_bar = lie_test::irrep(&[0, 1]).unwrap();
    let cat = lie_test::group(&three);
    let product = cat.fusion(&three, &three_bar).unwrap();
    assert_eq!(product.len(), 2);
    assert_eq!(cache_budgets(), budgets);
    let stats = tier_stats(CoefficientCacheTier::Product);
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.bytes, 0);
    assert_eq!(stats.evictions, 2);
    // A zero R budget still returns the 2×2 octet block, retaining nothing;
    // with no retention, R and its two channels each regenerate.
    let octet = lie_test::irrep(&[1, 1]).unwrap();
    assert_eq!(cat.r_symbol(&octet, &octet, &octet).unwrap().dim(), 2);
    let stats = tier_stats(CoefficientCacheTier::RSymbol);
    assert_eq!((stats.entries, stats.bytes, stats.misses), (0, 0, 1));
    assert_eq!(stats.evictions, 1);
    let stats = tier_stats(CoefficientCacheTier::Cgc);
    assert_eq!((stats.entries, stats.bytes), (0, 0));
    assert!(stats.misses >= 2);
    // Exact B/C/D products obey the shared Product policy.
    let bcd =
        fusion_symbols::lie::LieGroup::new(fusion_symbols::lie::GroupId::sp(4).unwrap()).unwrap();
    let v = bcd.irrep(&[1, 0]).unwrap();
    let before = tier_stats(CoefficientCacheTier::Product).evictions;
    assert_eq!(bcd.fusion(&v, &v).unwrap().len(), 3);
    let stats = tier_stats(CoefficientCacheTier::Product);
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.bytes, 0);
    assert_eq!(stats.evictions, before + 2);
    // A zero B/C/D R budget still returns 4⊗4→10, retaining nothing.
    let adjoint = bcd.irrep(&[2, 0]).unwrap();
    assert_eq!(bcd.r_symbol(&v, &v, &adjoint).unwrap().dim(), 1);
    let stats = tier_stats(CoefficientCacheTier::RSymbol);
    assert_eq!((stats.entries, stats.bytes, stats.misses), (0, 0, 2));
    assert_eq!(stats.evictions, 2);
    // Label-owned duals need no cache retention.
    for (cat, a) in [(&cat, &three), (&bcd, &v)] {
        assert_eq!(cat.dual(&cat.dual(a)), *a);
        let stats = common::stats_sum(common::generated_tiers());
        assert_eq!((stats.entries, stats.bytes), (0, 0));
    }
}
