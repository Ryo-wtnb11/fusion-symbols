//! Fresh process: installs a one-shot disabled generated-cache policy.
//! Generated-family zero-budget coefficient-cache contract.

#![cfg(feature = "lie-generated")]

mod common;

use crate::common::lie as lie_test;
use fusion_symbols::cache::{
    cache_budgets, configure_cache_budgets, generated_cache_stats, CoefficientCacheBudgets,
    CoefficientCacheTier,
};
use fusion_symbols::category::FusionCategory;

#[test]
fn disabled_policy_preserves_sun_product_value_without_retention() {
    for tier in [
        CoefficientCacheTier::SunProduct,
        CoefficientCacheTier::SunCgc,
        CoefficientCacheTier::SunF,
        CoefficientCacheTier::BcdCgc,
        CoefficientCacheTier::BcdF,
    ] {
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
    let stats = generated_cache_stats().sun_product;
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.bytes, 0);
    assert_eq!(stats.evictions, 1);
    // Exact B/C/D products share the existing BcdCgc byte policy.
    let bcd =
        fusion_symbols::lie::LieGroup::new(fusion_symbols::lie::GroupId::sp(4).unwrap()).unwrap();
    let v = bcd.irrep(&[1, 0], &[]).unwrap();
    assert_eq!(bcd.fusion(&v, &v).unwrap().len(), 3);
    let stats = generated_cache_stats().bcd_cgc;
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.bytes, 0);
    assert_eq!(stats.evictions, 1);
    // LieGroup dual lookups obey this same zero budget, including reverse keys.
    for (cat, a) in [(&cat, &three), (&bcd, &v)] {
        assert_eq!(cat.dual(&cat.dual(a)), *a);
        let stats = generated_cache_stats().total();
        assert_eq!((stats.entries, stats.bytes), (0, 0));
    }
}
