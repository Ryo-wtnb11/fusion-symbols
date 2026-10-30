//! Isolated process: cold/reset/exact counter assertions exclude unrelated cache traffic.
//! Unified simple-Lie Product API and concurrency contract.
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
fn product_cache_contract() {
    cache::reset();

    // Failure admission is transactional: distinct ranks never query or
    // publish the cache.
    let su3 = irr(&[1, 0]);
    let su4 = irr(&[1, 0, 0]);
    assert!(std::panic::catch_unwind(|| group(&su3).fusion(&su3, &su4)).is_err());
    assert_eq!(
        cache::tier_stats(cache::CoefficientCacheTier::Product),
        TierStats::default()
    );

    // A cold public call sweeps once. Reversing the factors must hit the same
    // unordered entry and reconstruct the same owned map without re-sweeping.
    let a = irr(&[2, 1]);
    let b = irr(&[1, 2]);
    let cold = outcomes(&a, &b);
    let after_cold = cache::tier_stats(cache::CoefficientCacheTier::Product);
    assert_eq!(after_cold.hits, 0);
    assert_eq!(after_cold.misses, 2);
    assert_eq!(after_cold.entries, 2);
    assert!(after_cold.bytes > 0);

    let warm_reversed = outcomes(&b, &a);
    assert_eq!(warm_reversed, cold);
    let after_warm = cache::tier_stats(cache::CoefficientCacheTier::Product);
    assert_eq!(after_warm.hits, 1);
    assert_eq!(after_warm.misses, 2, "warm public map must not re-sweep");
    assert_eq!(after_warm.entries, 2);

    let cat = group(&a);
    let shared: Vec<_> = cat.fusion(&a, &b).unwrap().collect();
    assert_eq!(shared, cold);
    let channel = cold.first().unwrap();
    assert_eq!(cat.nsymbol(&a, &b, channel).unwrap(), 1);
    assert_eq!(cat.nsymbol(&a, &b, &irr(&[99, 0])).unwrap(), 0);
    assert_eq!(cat.fusion(&b, &a).unwrap().collect::<Vec<_>>(), shared);
    let after_shared = cache::tier_stats(cache::CoefficientCacheTier::Product);
    assert_eq!(after_shared.hits, 5);
    assert_eq!(after_shared.misses, 2);
    assert_eq!(after_shared.entries, 2);

    // Single-flight loaders publish one exact channel list and one label projection.
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
    let concurrent = cache::tier_stats(cache::CoefficientCacheTier::Product);
    assert_eq!(concurrent.entries, 2);
    assert_eq!(concurrent.misses, 2);
    assert_eq!(concurrent.hits, 7);

    let misses = concurrent.misses;
    let hits = concurrent.hits;
    assert_eq!(outcomes(&a, &b), products[0]);
    let warm = cache::tier_stats(cache::CoefficientCacheTier::Product);
    assert_eq!(warm.misses, misses, "warm call must not re-sweep");
    assert_eq!(warm.hits, hits + 1);

    // Rich label payloads can reach the byte cap before the unchanged entry ceiling.
    cache::reset();
    let trivial = lie_test::trivial(3).unwrap();
    for label in 1..=257 {
        let _ = outcomes(&trivial, &irr(&[label, 0]));
    }
    let entry_bound = cache::tier_stats(cache::CoefficientCacheTier::Product);
    assert!(entry_bound.entries <= 256);
    assert!(entry_bound.evictions > 0);
    assert!(entry_bound.bytes <= (128 << 10));

    // Root-system identity is part of the key: these rank-five weights differ in fusion.
    use fusion_symbols::lie::{GroupId, LieGroup};
    cache::reset();
    let su = LieGroup::new(GroupId::su(6).unwrap()).unwrap();
    let so = LieGroup::new(GroupId::so(10).unwrap()).unwrap();
    let spin = LieGroup::new(GroupId::spin(10).unwrap()).unwrap();
    let x = su.irrep(&[1, 0, 0, 0, 0]).unwrap();
    let y = so.irrep(&[1, 0, 0, 0, 0]).unwrap();
    assert_eq!(su.fusion(&x, &x).unwrap().count(), 2);
    assert_eq!(so.fusion(&y, &y).unwrap().count(), 3);
    let z = spin.irrep(&[1, 0, 0, 0, 0]).unwrap();
    assert_eq!(spin.fusion(&z, &z).unwrap().count(), 3);
    let stats = cache::tier_stats(cache::CoefficientCacheTier::Product);
    assert_eq!((stats.entries, stats.misses, stats.hits), (4, 4, 1));
    assert_eq!(
        cache::tier_stats(cache::CoefficientCacheTier::Cgc).entries,
        0
    );
    for (id, label) in [
        (GroupId::g2(), vec![1, 0]),
        (GroupId::f4(), vec![0, 0, 0, 1]),
        (GroupId::e6(), vec![1, 0, 0, 0, 0, 0]),
        (GroupId::e7(), vec![0, 0, 0, 0, 0, 0, 1]),
        (GroupId::e8(), vec![0, 0, 0, 0, 0, 0, 0, 1]),
    ] {
        cache::reset();
        let g = LieGroup::new(id).unwrap();
        let a = g.irrep(&label).unwrap();
        let b = g.dual(&a);
        let one = g.unit();
        assert_eq!(g.nsymbol(&a, &b, &one).unwrap(), 1);
        let cold = cache::tier_stats(cache::CoefficientCacheTier::Product);
        assert_eq!((cold.entries, cold.misses), (1, 1));
        assert!(g.fusion(&b, &a).unwrap().any(|c| c == one));
        assert_eq!(g.nsymbol(&a, &b, &one).unwrap(), 1);
        let warm = cache::tier_stats(cache::CoefficientCacheTier::Product);
        assert_eq!((warm.entries, warm.misses, warm.hits), (2, 2, 2));
        cache::trim_to(cache::CoefficientCacheTier::Product, 0);
        assert_eq!(
            cache::tier_stats(cache::CoefficientCacheTier::Product).bytes,
            0
        );
    }

    // The dedicated allocation-free SU(2) decomposition remains independent
    // and uncached by this generated SU(N) tier.
    cache::reset();
    let channels = a1::group()
        .fusion(&a1::ir(3), &a1::ir(2))
        .unwrap()
        .map(|s| s.dynkin().next().unwrap() as u32)
        .collect::<Vec<_>>();
    assert_eq!(channels, [1, 3, 5]);
    assert_eq!(
        cache::tier_stats(cache::CoefficientCacheTier::Product),
        TierStats::default()
    );
}
