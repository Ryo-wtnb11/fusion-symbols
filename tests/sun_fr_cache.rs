//! Isolated process: cold/reset/exact aggregate counters exclude unrelated cache traffic.
//! Derived-f64 SU(N) F- and R-symbol cache behaviour through the public API: a warm
//! hit returns the identical block, `reset` clears it, `stats` accounts it, and concurrent generation is race-free.
//!
//! The entry bound / byte-eviction / eviction-never-changes-values machinery is
//! covered generically in `src/cache.rs` (and specialised for the `FBlock`
//! tier there); this file exercises the end-to-end public path.

#![cfg(feature = "lie-generated")]

mod common;

use crate::common::lie as lie_test;
use fusion_symbols::lie::LieIrrep as Irrep;
use lie_test::{f_symbol, r_symbol};

use std::sync::{Arc, Barrier};
use std::thread;

use fusion_symbols::cache;

fn irr(d: &[i64]) -> Irrep {
    lie_test::irrep(d).unwrap()
}

/// One test (not two): both phases call the process-global `cache::reset`, so
/// they must not run concurrently with each other.
#[test]
fn f_symbol_cache_hit_reset_stats_and_concurrency() {
    let e8 = irr(&[1, 1]);

    // --- hit / stats / reset ---
    cache::reset();
    let first = f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
    let after_miss = cache::stats();
    assert!(
        after_miss.entries >= 1 && after_miss.misses >= 1,
        "first call should record a miss + entry"
    );

    let again = f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
    assert_eq!(first, again, "warm hit must return the identical block");
    let after_hit = cache::stats();
    assert!(after_hit.hits > after_miss.hits, "second call is a hit");
    assert_eq!(after_hit.entries, after_miss.entries, "a hit adds no entry");

    cache::reset();
    let cleared = cache::stats();
    assert_eq!(
        (cleared.hits, cleared.misses, cleared.entries, cleared.bytes),
        (0, 0, 0, 0),
        "reset clears entries and counters"
    );

    // --- concurrency: many threads race on the same key ---
    let e8 = Arc::new(e8);
    let want = first;
    let start = Arc::new(Barrier::new(8));
    let mut handles = Vec::new();
    for _ in 0..8 {
        let e8 = Arc::clone(&e8);
        let want = want.clone();
        let start = Arc::clone(&start);
        handles.push(thread::spawn(move || {
            start.wait();
            for _ in 0..16 {
                let got = f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
                assert_eq!(got, want, "concurrent F block diverged");
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let stats = cache::generated_cache_stats().sun_f;
    assert_eq!(stats.entries, 1, "one F entry survives the race");
    assert_eq!(
        stats.misses, 1,
        "one successful F computation under contention"
    );
    assert_eq!(stats.hits, 8 * 16 - 1, "all other callers share the result");

    // --- R tier: the same single-flight contract ---
    cache::reset();
    let want = r_symbol(&e8, &e8, &e8).unwrap();
    cache::reset();
    let mut handles = Vec::new();
    for _ in 0..8 {
        let e8 = Arc::clone(&e8);
        let want = want.clone();
        let start = Arc::clone(&start);
        handles.push(thread::spawn(move || {
            start.wait();
            for _ in 0..16 {
                let got = r_symbol(&e8, &e8, &e8).unwrap();
                assert_eq!(got, want, "concurrent R block diverged");
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let stats = cache::generated_cache_stats().sun_r;
    assert_eq!(stats.entries, 1, "one R entry survives the race");
    assert_eq!(
        stats.misses, 1,
        "one successful R computation under contention"
    );
    assert_eq!(stats.hits, 8 * 16 - 1, "all other callers share the result");
}
