//! Isolated process: exact reset/counter assertions require no unrelated cache traffic.
//! Base SU(2) coefficient-cache resource contract.
//!
//! Kept in one `#[test]`: the 3j/6j/F/CGC tiers are process-global and
//! `cache::reset()`/`cache::tier_stats()` act on shared state, so splitting into
//! several parallel tests would race the shared counters (same reasoning as
//! `tests/sun_cgc_cache.rs`).
//!
//! This exercises the *base* tiers on the real global caches. Forcing eviction
//! here is intentionally NOT attempted: the production caps are 1M entries /
//! 64 MiB per tier, and reaching them needs millions of big-rational Wigner
//! computations (measured: 1.2M `wigner_3j` calls collapse to ~35k distinct
//! Regge classes). The eviction counter itself is covered deterministically on
//! local caches in `src/cache.rs`; here we prove the aggregate retained-charge cap, the
//! per-tier statistics, hit/miss accounting, and reset zeroing.

mod common;

use crate::common::a1;
use a1::exact_f;
use fusion_symbols::cache::{self, CoefficientCacheTier, TierStats, BASE_CACHE_MAX_BYTES};
use fusion_symbols::category::GroupRepresentations;
use fusion_symbols::lie::su2::{wigner_3j, wigner_6j};

#[test]
fn base_cache_resource_contract() {
    // ---- hit/miss accounting on a deterministic per-tier pattern ----
    cache::reset();

    // 3j: only `wigner_3j` touches this tier, so its counters are exact.
    let _ = wigner_3j(2, 2, 2, 0, 0, 0); // miss
    let _ = wigner_3j(2, 2, 2, 0, 0, 0); // hit
    let s = cache::tier_stats(CoefficientCacheTier::Wigner3j);
    assert_eq!(
        (s.misses, s.hits),
        (1, 1),
        "one miss then one hit on the 3j tier"
    );

    // 6j: `wigner_6j` (and, below, `exact_f`) can populate this tier, so
    // assert monotone accounting rather than exact deltas.
    let _ = wigner_6j(2, 2, 2, 2, 2, 2); // miss
    let _ = wigner_6j(2, 2, 2, 2, 2, 2); // hit
    let s = cache::tier_stats(CoefficientCacheTier::Wigner6j);
    assert!(s.misses >= 1 && s.hits >= 1, "6j miss+hit");

    // F: only `exact_f` touches this tier here.
    let _ = exact_f(2, 2, 2, 2, 2, 2); // miss
    let _ = exact_f(2, 2, 2, 2, 2, 2); // hit
    let s = cache::tier_stats(CoefficientCacheTier::FSymbol);
    assert!(s.misses >= 1 && s.hits >= 1, "F miss+hit");

    // CGC: an A(1) block is built once, then shared.
    let su2 = a1::group();
    let block = |a, b, c| su2.cgc(&a1::ir(a), &a1::ir(b), &a1::ir(c)).unwrap();
    let first = block(4, 2, 2);
    assert_eq!(block(4, 2, 2), first);
    let s = cache::tier_stats(CoefficientCacheTier::Cgc);
    assert!(s.misses >= 1 && s.hits >= 1, "CGC miss+hit");

    // ---- fill all four tiers with a spread of small labels ----
    cache::reset();
    for a in 0..12u32 {
        for b in 0..12u32 {
            for c in 0..12u32 {
                // 3j and 6j direct fills.
                let _ = wigner_3j(a, b, c, 0, 0, 0);
                let _ = wigner_6j(a, b, c, a, b, c);
                // F sweep and A(1) CGC blocks.
                let _ = exact_f(a, b, c, a, b, c);
                let _ = block(a, b, c);
            }
        }
    }

    // Per-tier conservative retained-charge cap.
    for tier in common::BASE_TIERS {
        let bytes = cache::tier_stats(tier).bytes;
        let cap = cache::cache_budgets().limit(tier);
        assert!(bytes <= cap, "{tier} over cap: {bytes}");
    }

    // Aggregate charged-entry cap — the documented per-tier corollary.
    let total = common::stats_sum(&common::BASE_TIERS);
    assert!(
        total.bytes <= BASE_CACHE_MAX_BYTES,
        "aggregate over BASE_CACHE_MAX_BYTES: {} > {}",
        total.bytes,
        BASE_CACHE_MAX_BYTES
    );
    assert!(total.entries > 0, "the fill retained entries");

    // `ALL` lists every tier once in every build; the base tiers are among them.
    // `ALL` contains each named variant; the loop below proves none repeats.
    for tier in [
        CoefficientCacheTier::Product,
        CoefficientCacheTier::Cgc,
        CoefficientCacheTier::FSymbol,
        CoefficientCacheTier::RSymbol,
        CoefficientCacheTier::Wigner3j,
        CoefficientCacheTier::Wigner6j,
    ] {
        assert!(CoefficientCacheTier::ALL.contains(&tier), "{tier} missing");
    }
    assert!(common::BASE_TIERS
        .iter()
        .all(|t| CoefficientCacheTier::ALL.contains(t)));
    for (i, tier) in CoefficientCacheTier::ALL.iter().enumerate() {
        assert!(
            !CoefficientCacheTier::ALL[..i].contains(tier),
            "{tier} twice"
        );
    }

    // ---- reset zeroing: every field of every tier ----
    cache::reset();
    for &tier in CoefficientCacheTier::ALL {
        assert_eq!(
            cache::tier_stats(tier),
            TierStats::default(),
            "{tier} not cleared"
        );
    }
}
