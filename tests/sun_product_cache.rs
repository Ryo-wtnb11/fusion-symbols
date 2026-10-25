//! Isolated process: cold/reset/exact counter assertions exclude unrelated cache traffic.
//! SU(N) product-cache API and concurrency contract.
//!
//! One sequential test owns the process-global reset policy for this binary.

#![cfg(feature = "lie-generated")]

mod common;

use crate::common::a1;
use std::sync::{Arc, Barrier};

use crate::common::lie as lie_test;
use fusion_symbols::cache::{self, TierStats};
use fusion_symbols::category::FusionCategory;
use fusion_symbols::lie::LieIrrep as Irrep;
use lie_test::group;

fn irr(dynkin: &[i64]) -> Irrep {
    lie_test::irrep(dynkin).unwrap()
}

fn outcomes(a: &Irrep, b: &Irrep) -> Vec<Irrep> {
    group(a).fusion(a, b).unwrap().collect()
}

#[test]
fn sun_product_cache_contract() {
    cache::reset();

    // Failure admission is transactional: distinct ranks never query or
    // publish the cache.
    let su3 = irr(&[1, 0]);
    let su4 = irr(&[1, 0, 0]);
    assert!(std::panic::catch_unwind(|| group(&su3).fusion(&su3, &su4)).is_err());
    assert_eq!(
        cache::generated_cache_stats().sun_product,
        TierStats::default()
    );

    // A cold public call sweeps once. Reversing the factors must hit the same
    // unordered entry and reconstruct the same owned map without re-sweeping.
    let a = irr(&[2, 1]);
    let b = irr(&[1, 2]);
    let cold = outcomes(&a, &b);
    let after_cold = cache::generated_cache_stats().sun_product;
    assert_eq!(after_cold.hits, 0);
    assert_eq!(after_cold.misses, 1);
    assert_eq!(after_cold.entries, 1);
    assert!(after_cold.bytes > 0);

    let warm_reversed = outcomes(&b, &a);
    assert_eq!(warm_reversed, cold);
    let after_warm = cache::generated_cache_stats().sun_product;
    assert_eq!(after_warm.hits, 1);
    assert_eq!(after_warm.misses, 1, "warm public map must not re-sweep");
    assert_eq!(after_warm.entries, 1);

    let cat = group(&a);
    let shared: Vec<_> = cat.fusion(&a, &b).unwrap().collect();
    assert_eq!(shared, cold);
    let channel = cold.first().unwrap();
    assert_eq!(cat.nsymbol(&a, &b, channel), 1);
    assert_eq!(cat.nsymbol(&a, &b, &irr(&[99, 0])), 0);
    assert_eq!(cat.fusion(&b, &a).unwrap().collect::<Vec<_>>(), shared);
    let after_shared = cache::generated_cache_stats().sun_product;
    assert_eq!(after_shared.hits, 5);
    assert_eq!(after_shared.misses, 1);
    assert_eq!(after_shared.entries, 1);

    // Compute-outside-lock permits concurrent duplicate misses, but the write
    // recheck publishes exactly one identical entry.
    cache::reset();
    let a = Arc::new(irr(&[3, 1]));
    let b = Arc::new(irr(&[1, 2]));
    let barrier = Arc::new(Barrier::new(8));
    let handles = (0..8)
        .map(|_| {
            let a = Arc::clone(&a);
            let b = Arc::clone(&b);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                outcomes(&a, &b)
            })
        })
        .collect::<Vec<_>>();
    let products = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    assert!(products.iter().all(|product| product == &products[0]));
    let concurrent = cache::generated_cache_stats().sun_product;
    assert_eq!(concurrent.entries, 1);
    assert!((1..=8).contains(&concurrent.misses));

    let misses = concurrent.misses;
    let hits = concurrent.hits;
    assert_eq!(outcomes(&a, &b), products[0]);
    let warm = cache::generated_cache_stats().sun_product;
    assert_eq!(warm.misses, misses, "warm call must not re-sweep");
    assert_eq!(warm.hits, hits + 1);

    // Cheap one-channel products isolate the production entry bound: all 256
    // retained entries together remain below the 128 KiB charge backstop.
    cache::reset();
    let trivial = lie_test::trivial(3).unwrap();
    for label in 1..=257 {
        let _ = outcomes(&trivial, &irr(&[label, 0]));
    }
    let entry_bound = cache::generated_cache_stats().sun_product;
    assert_eq!(entry_bound.entries, 256);
    assert_eq!(entry_bound.evictions, 1);
    assert!(entry_bound.bytes < (128 << 10), "byte bound fired first");

    // The dedicated allocation-free SU(2) decomposition remains independent
    // and uncached by this generated SU(N) tier.
    cache::reset();
    let channels = a1::group()
        .fusion(&a1::ir(3), &a1::ir(2))
        .unwrap()
        .map(|s| s.dynkin()[0] as u32)
        .collect::<Vec<_>>();
    assert_eq!(channels, [1, 3, 5]);
    assert_eq!(
        cache::generated_cache_stats().sun_product,
        TierStats::default()
    );
}
