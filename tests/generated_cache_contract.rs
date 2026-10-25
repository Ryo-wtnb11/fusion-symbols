//! Isolated process: exact generated reset/counter assertions exclude unrelated cache traffic.
//! Generated-tier coefficient-cache resource contract.
//!
//! The `lie-generated` analogue of `tests/base_cache_contract.rs`: the five generated
//! tiers (SU(N) product, SU(N) CGC, SU(N) F, B/C/D CGC, B/C/D F) are process-global and
//! `cache::reset()`/`generated_cache_stats()` act on shared state, so this is
//! kept in one `#[test]` (splitting into parallel tests would race the shared
//! counters — same reasoning as `tests/sun_cgc_cache.rs`).
//!
//! Forcing eviction on the real global caches is intentionally NOT attempted:
//! the production caps are 64 MiB / 256 MiB per tier. The FIFO bound /
//! byte-eviction / oversize-eviction machinery is covered deterministically on
//! local caches in `src/cache.rs`; here we prove per-tier hit/miss accounting on
//! deterministic accesses, the aggregate retained-charge cap, the const/cap tie, `total()`
//! coherence, that the generated stats are a subset of the aggregate `stats()`,
//! and reset zeroing of every field of every generated tier and of `total()`.

#![cfg(feature = "lie-generated")]

mod common;

use crate::common::lie as lie_test;

use crate::common::bcd as bcd_test;
use bcd_test::Series;

use fusion_symbols::cache::{self, GeneratedCacheStats, GENERATED_CACHE_MAX_BYTES};

fn sun_irr(d: &[i64]) -> fusion_symbols::lie::LieIrrep {
    lie_test::irrep(d).unwrap()
}

fn bcd_irr(s: Series, d: &[i64]) -> fusion_symbols::lie::LieIrrep {
    bcd_test::irrep(s, d).unwrap()
}

#[test]
fn generated_cache_resource_contract() {
    // ---- the const is the documented sum of the five tier caps ----
    // (The compile-time assertion in src/cache.rs is the real drift guard; this
    // pins the human-facing value.)
    assert_eq!(
        GENERATED_CACHE_MAX_BYTES,
        ((256 + 64 + 256 + 64) << 20) + (128 << 10),
        "640 MiB + 128 KiB = SU(N) product + CGC/F + BCD CGC/F"
    );

    // ---- per-tier hit/miss accounting on deterministic accesses ----
    cache::reset();
    let g = cache::generated_cache_stats();
    assert_eq!(
        g,
        GeneratedCacheStats::default(),
        "reset must zero every tier"
    );

    // SU(N) CGC tier: `lie_test::cgc` owns it. Miss then hit on the same channel.
    let (s1, s2, s3) = (sun_irr(&[1, 0]), sun_irr(&[0, 1]), sun_irr(&[1, 1])); // 3⊗3̄→8
    let first = lie_test::cgc(&s1, &s2, &s3).unwrap();
    let g = cache::generated_cache_stats();
    assert!(
        g.sun_product.misses >= 1 && g.sun_product.entries >= 1,
        "CGC multiplicity lookup populates the SU(N) product tier"
    );
    assert!(g.sun_cgc.misses >= 1, "first sun cgc call is a miss");
    assert!(
        g.sun_cgc.entries >= 1 && g.sun_cgc.bytes > 0,
        "entry retained"
    );
    let again = lie_test::cgc(&s1, &s2, &s3).unwrap();
    assert_eq!(first, again, "warm hit returns the identical CGC");
    let g = cache::generated_cache_stats();
    assert!(g.sun_cgc.hits >= 1, "second sun cgc call is a hit");

    // SU(N) F tier: `lie_test::f_symbol` owns it.
    let e8 = sun_irr(&[1, 1]);
    let _ = lie_test::f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap(); // miss
    let _ = lie_test::f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap(); // hit
    let g = cache::generated_cache_stats();
    assert!(g.sun_f.misses >= 1 && g.sun_f.hits >= 1, "sun F miss + hit");

    // B/C/D CGC + F tiers: the public `bcd_test::f_symbol` owns the BCD F tier and
    // its CGC materialization owns the BCD CGC tier. Sextet from the crate's own
    // doctest (Sp(4), F = 1×1×1×1 identity). Miss then hit.
    let cat = bcd_test::group(Series::C, 2).unwrap();
    let triv = bcd_test::trivial(Series::C, 2).unwrap();
    let v = bcd_irr(Series::C, &[0, 1]);
    let adj = bcd_irr(Series::C, &[2, 0]);
    let _ = bcd_test::f_symbol(&cat, &triv, &v, &v, &adj, &v, &adj).unwrap(); // miss
    let g = cache::generated_cache_stats();
    assert!(
        g.bcd_cgc.entries >= 1 && g.bcd_f.entries >= 1,
        "an F block populates both BCD tiers"
    );
    let _ = bcd_test::f_symbol(&cat, &triv, &v, &v, &adj, &v, &adj).unwrap(); // hit
    let g = cache::generated_cache_stats();
    assert!(g.bcd_f.hits >= 1, "second F block hits the BCD F tier");

    // Product only assembles handles/blocks: warm factor queries must not sweep.
    use fusion_symbols::{
        category::{FusionCategory, GroupRepresentations},
        lie::{GroupId, LieGroup},
        product::Product,
    };
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let a = vec![
        su3.irrep(&[1, 1], &[]).unwrap(),
        su2.irrep(&[2], &[]).unwrap(),
    ];
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
    let cold = cache::generated_cache_stats();
    query();
    let warm = cache::generated_cache_stats();
    for (cold, warm) in [
        (cold.sun_product, warm.sun_product),
        (cold.sun_cgc, warm.sun_cgc),
        (cold.sun_f, warm.sun_f),
    ] {
        assert_eq!(
            cold.misses, warm.misses,
            "warm runtime product re-swept a factor"
        );
        assert_eq!(cold.entries, warm.entries);
        assert!(warm.hits > cold.hits);
    }

    // ---- aggregate retained-charge cap and total() coherence ----
    let g = cache::generated_cache_stats();
    let total = g.total();
    // Each tier's conservative charged-entry cap and the aggregate corollary.
    assert!(
        g.sun_product.bytes <= (128 << 10),
        "the product tier is over its 128 KiB retained-charge backstop"
    );
    assert!(
        g.sun_cgc.bytes <= (256 << 20) && g.bcd_cgc.bytes <= (256 << 20),
        "a CGC tier is over its 256 MiB cap"
    );
    assert!(
        g.sun_f.bytes <= (64 << 20) && g.bcd_f.bytes <= (64 << 20),
        "an F tier is over its 64 MiB cap"
    );
    assert!(
        total.bytes <= GENERATED_CACHE_MAX_BYTES,
        "aggregate over GENERATED_CACHE_MAX_BYTES: {} > {}",
        total.bytes,
        GENERATED_CACHE_MAX_BYTES
    );
    // total() is the field-wise sum of the five tiers.
    assert_eq!(
        total.entries,
        g.sun_product.entries
            + g.sun_cgc.entries
            + g.sun_f.entries
            + g.bcd_cgc.entries
            + g.bcd_f.entries
    );
    assert_eq!(
        total.bytes,
        g.sun_product.bytes + g.sun_cgc.bytes + g.sun_f.bytes + g.bcd_cgc.bytes + g.bcd_f.bytes
    );
    assert!(total.entries > 0, "the accesses retained entries");

    // Generated tiers are a subset of the whole-process aggregate `stats()`.
    let agg = cache::stats();
    assert!(total.entries <= agg.entries, "generated ⊆ aggregate");
    assert!(total.bytes <= agg.bytes);

    // ---- reset zeroing: every field of every generated tier and of total() ----
    cache::reset();
    let g = cache::generated_cache_stats();
    let zero = fusion_symbols::cache::TierStats::default();
    assert_eq!(g.sun_product, zero, "sun_product not cleared");
    assert_eq!(g.sun_cgc, zero, "sun_cgc not cleared");
    assert_eq!(g.sun_f, zero, "sun_f not cleared");
    assert_eq!(g.bcd_cgc, zero, "bcd_cgc not cleared");
    assert_eq!(g.bcd_f, zero, "bcd_f not cleared");
    assert_eq!(g.total(), zero, "total not cleared");
}
