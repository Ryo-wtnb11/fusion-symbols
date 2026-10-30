//! Isolated process: exact generated reset/counter assertions exclude unrelated cache traffic.
//! Generated-tier coefficient-cache resource contract.
//!
//! The `lie-generated` analogue of `tests/base_cache_contract.rs`: the Product,
//! CGC, F and R tiers, shared by every root system, are process-global and
//! `cache::reset()`/`cache::tier_stats()` act on shared state, so this is
//! kept in one `#[test]` (splitting into parallel tests would race the shared
//! counters — same reasoning as `tests/sun_cgc_cache.rs`).
//!
//! Forcing eviction on the real global caches is intentionally NOT attempted:
//! the production caps are 64 MiB / 256 MiB per tier. The entry bound /
//! byte-eviction / oversize-eviction machinery is covered deterministically on
//! local caches in `src/cache.rs`; here we prove per-tier hit/miss accounting on
//! deterministic accesses, the aggregate retained-charge cap, the const/cap tie,
//! and reset zeroing of every field of every generated tier.

#![cfg(feature = "lie-generated")]

mod common;

use crate::common::lie as lie_test;

use crate::common::bcd as bcd_test;
use bcd_test::Series;

use fusion_symbols::cache::{
    self, tier_stats, CoefficientCacheTier as Tier, TierStats, GENERATED_CACHE_MAX_BYTES,
};

fn sun_irr(d: &[i64]) -> fusion_symbols::lie::LieIrrep {
    lie_test::irrep(d).unwrap()
}

fn bcd_irr(s: Series, d: &[i64]) -> fusion_symbols::lie::LieIrrep {
    bcd_test::irrep(s, d).unwrap()
}

#[test]
fn generated_cache_resource_contract() {
    // ---- the consts are the documented sums of the tier caps ----
    // (The compile-time assertions in src/cache.rs are the real drift guard;
    // these pin the human-facing values.)
    assert_eq!(
        GENERATED_CACHE_MAX_BYTES,
        (64 << 20) + (128 << 10),
        "64 MiB + 128 KiB = R + Product"
    );
    assert_eq!(
        fusion_symbols::cache::BASE_CACHE_MAX_BYTES,
        (64 + 64 + 64 + 256) << 20,
        "448 MiB = Wigner 3j + 6j + F + CGC"
    );
    let all: usize = Tier::ALL
        .iter()
        .map(|&t| fusion_symbols::cache::cache_budgets().limit(t))
        .sum();
    assert_eq!(
        all,
        fusion_symbols::cache::BASE_CACHE_MAX_BYTES + GENERATED_CACHE_MAX_BYTES
    );

    // ---- per-tier hit/miss accounting on deterministic accesses ----
    cache::reset();
    for &tier in Tier::ALL {
        assert_eq!(
            tier_stats(tier),
            TierStats::default(),
            "reset must zero {tier}"
        );
    }

    // CGC tier: `lie_test::cgc` owns it here. Miss then hit on the same channel.
    let (s1, s2, s3) = (sun_irr(&[1, 0]), sun_irr(&[0, 1]), sun_irr(&[1, 1])); // 3⊗3̄→8
    let first = lie_test::cgc(&s1, &s2, &s3).unwrap();
    assert!(
        tier_stats(Tier::Product).misses >= 1 && tier_stats(Tier::Product).entries >= 1,
        "the CGC multiplicity lookup populates the product tier"
    );
    assert!(
        tier_stats(Tier::Cgc).misses >= 1,
        "first sun cgc call is a miss"
    );
    assert!(
        tier_stats(Tier::Cgc).entries >= 1 && tier_stats(Tier::Cgc).bytes > 0,
        "entry retained"
    );
    let again = lie_test::cgc(&s1, &s2, &s3).unwrap();
    assert_eq!(first, again, "warm hit returns the identical CGC");
    assert!(
        tier_stats(Tier::Cgc).hits >= 1,
        "second sun cgc call is a hit"
    );

    // SU(N) F tier: `lie_test::f_symbol` owns it.
    let e8 = sun_irr(&[1, 1]);
    let _ = lie_test::f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap(); // miss
    let _ = lie_test::f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap(); // hit
    assert!(
        tier_stats(Tier::FSymbol).misses >= 1 && tier_stats(Tier::FSymbol).hits >= 1,
        "sun F miss + hit"
    );

    // R tier: exact miss then hit on the 2×2 octet block.
    let r = lie_test::r_symbol(&e8, &e8, &e8).unwrap();
    let g = tier_stats(Tier::RSymbol);
    assert_eq!((g.misses, g.hits, g.entries), (1, 0, 1));
    assert!(g.bytes > 0);
    assert_eq!(lie_test::r_symbol(&e8, &e8, &e8).unwrap(), r);
    assert_eq!(tier_stats(Tier::RSymbol).hits, 1, "sun R hit");

    // B/C/D blocks share the same F, R and CGC tiers. Sextet from the crate's
    // own doctest (Sp(4), F = 1×1×1×1 identity). Miss then hit.
    let cat = bcd_test::group(Series::C, 2).unwrap();
    let triv = bcd_test::trivial(Series::C, 2).unwrap();
    let v = bcd_irr(Series::C, &[0, 1]);
    let adj = bcd_irr(Series::C, &[2, 0]);
    let _ = bcd_test::f_symbol(&cat, &triv, &v, &v, &adj, &v, &adj).unwrap(); // miss
    assert!(
        tier_stats(Tier::Cgc).entries >= 1 && tier_stats(Tier::FSymbol).entries >= 1,
        "an F block populates the CGC and F tiers"
    );
    let _ = bcd_test::f_symbol(&cat, &triv, &v, &v, &adj, &v, &adj).unwrap(); // hit
    assert!(
        tier_stats(Tier::FSymbol).hits >= 1,
        "second F block hits the BCD F tier"
    );
    let r = bcd_test::r_symbol(&cat, &v, &v, &adj).unwrap();
    let g = tier_stats(Tier::RSymbol);
    assert_eq!((g.misses, g.hits, g.entries), (2, 1, 2));
    assert_eq!(bcd_test::r_symbol(&cat, &v, &v, &adj).unwrap(), r);
    assert_eq!(tier_stats(Tier::RSymbol).hits, 2, "BCD R hit");

    // Product only assembles handles/blocks: warm factor queries must not sweep.
    use fusion_symbols::{
        category::{FusionCategory, GroupRepresentations},
        lie::{GroupId, LieGroup},
        product::Product,
    };
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let a = vec![su3.irrep(&[1, 1]).unwrap(), su2.irrep(&[2]).unwrap()];
    let runtime = Product::new(vec![su3, su2]);
    let query = || {
        runtime.fusion(&a, &a).unwrap().for_each(drop);
        runtime.f_symbol(&a, &a, &a, &a, &a, &a).unwrap();
        runtime.r_symbol(&a, &a, &a).unwrap();
        runtime.a_symbol(&a, &a, &a).unwrap();
        runtime.b_symbol(&a, &a, &a).unwrap();
        runtime.cgc(&a, &a, &a).unwrap();
    };
    query();
    let sun = [Tier::Product, Tier::Cgc, Tier::FSymbol, Tier::RSymbol];
    let cold = sun.map(tier_stats);
    query();
    let warm = sun.map(tier_stats);
    for (cold, warm) in cold.into_iter().zip(warm) {
        assert_eq!(
            cold.misses, warm.misses,
            "warm runtime product re-swept a factor"
        );
        assert_eq!(cold.entries, warm.entries);
        assert!(warm.hits > cold.hits);
    }

    // ---- aggregate retained-charge cap ----
    let total = common::stats_sum(common::generated_tiers());
    // Each tier's conservative charged-entry cap and the aggregate corollary.
    assert!(
        tier_stats(Tier::Product).bytes <= (128 << 10),
        "the product tier is over its 128 KiB retained-charge backstop"
    );
    assert!(
        tier_stats(Tier::Cgc).bytes <= (256 << 20),
        "the CGC tier is over its 256 MiB cap"
    );
    assert!(
        tier_stats(Tier::FSymbol).bytes <= (64 << 20),
        "the F tier is over its 64 MiB cap"
    );
    assert!(
        tier_stats(Tier::RSymbol).bytes <= (64 << 20),
        "the R tier is over its 64 MiB cap"
    );
    assert!(
        total.bytes <= GENERATED_CACHE_MAX_BYTES,
        "aggregate over GENERATED_CACHE_MAX_BYTES: {} > {}",
        total.bytes,
        GENERATED_CACHE_MAX_BYTES
    );
    assert!(total.entries > 0, "the accesses retained entries");

    // ---- reset zeroing: every field of every tier ----
    cache::reset();
    for &tier in Tier::ALL {
        assert_eq!(tier_stats(tier), TierStats::default(), "{tier} not cleared");
    }
}
