//! Isolated process: exact generated-tier counts around trim exclude unrelated cache traffic.
//! Isolated process-global generated-tier trim routing contract.
#![cfg(feature = "lie-generated")]

mod common;

use crate::common::lie as lie_test;

use crate::common::bcd as bcd_test;
use bcd_test::Series;
use fusion_symbols::category::FusionCategory;

use fusion_symbols::cache::{self, CoefficientCacheTier};

fn irrep(d: &[i64]) -> fusion_symbols::lie::LieIrrep {
    lie_test::irrep(d).unwrap()
}

#[test]
fn trim_routes_to_each_populated_generated_tier() {
    cache::reset();
    let (a, b, c) = (irrep(&[1, 0]), irrep(&[0, 1]), irrep(&[1, 1]));
    let group = lie_test::group(&a);
    let external_product = group.fusion(&a, &b).unwrap();
    let _ = lie_test::cgc(&a, &b, &c).unwrap();
    let e8 = irrep(&[1, 1]);
    let _ = lie_test::f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
    let cat = bcd_test::group(Series::C, 2).unwrap();
    let triv = bcd_test::trivial(Series::C, 2).unwrap();
    let v = bcd_test::irrep(Series::C, &[0, 1]).unwrap();
    let adj = bcd_test::irrep(Series::C, &[2, 0]).unwrap();
    let _ = bcd_test::f_symbol(&cat, &triv, &v, &v, &adj, &v, &adj).unwrap();
    let _ = lie_test::r_symbol(&e8, &e8, &e8).unwrap();
    let _ = bcd_test::r_symbol(&cat, &v, &v, &adj).unwrap();
    let tiers = [
        CoefficientCacheTier::Product,
        CoefficientCacheTier::SunCgc,
        CoefficientCacheTier::SunF,
        CoefficientCacheTier::SunR,
        CoefficientCacheTier::BcdCgc,
        CoefficientCacheTier::BcdF,
        CoefficientCacheTier::BcdR,
    ];
    for tier in tiers {
        let selected = cache::tier_stats(tier);
        assert!(selected.entries > 0, "fixture must populate {tier}");
        let report = cache::trim_to(tier, 0);
        assert_eq!(report.removed_entries, selected.entries);
        assert_eq!(cache::tier_stats(tier).entries, 0);
        if tier == CoefficientCacheTier::Product {
            assert_eq!(
                external_product.clone().count(),
                2,
                "trim cannot invalidate a caller clone"
            );
        }
    }
}
