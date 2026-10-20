//! Timing for the prime-factorized 3j/6j engine. Not a CI gate.
//!
//! Exact-value evaluation on doubled spins <= 254, the thousands tier, and the
//! repeated-label (cache hit) regime. Run with `cargo bench`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use fusion_symbols::lie::su2::{wigner_3j, wigner_6j};

/// Admissible 6j label sets (doubled spins, all <= 254): equal-spin sextuples
/// at a spread of magnitudes so the timing reflects the k-sum length.
const SIX_J: &[[i32; 6]] = &[
    [2, 4, 4, 6, 4, 2],
    [20, 20, 20, 20, 20, 20],
    [60, 60, 60, 60, 60, 60],
    [120, 120, 120, 120, 120, 120],
    [200, 200, 200, 200, 200, 200],
    [254, 254, 254, 254, 254, 254],
];

/// Admissible 3j label sets `(dj,dj,dj; 0,0,0)` (doubled spins, even).
const THREE_J: &[[i32; 3]] = &[[20, 20, 20], [60, 60, 60], [120, 120, 120], [200, 200, 200]];

fn bench_6j(c: &mut Criterion) {
    let mut g = c.benchmark_group("6j_small");
    g.bench_function("racah", |b| {
        b.iter(|| {
            for &d in SIX_J {
                black_box(wigner_6j(
                    d[0] as u32,
                    d[1] as u32,
                    d[2] as u32,
                    d[3] as u32,
                    d[4] as u32,
                    d[5] as u32,
                ));
            }
        })
    });
    g.finish();
}

fn bench_3j(c: &mut Criterion) {
    let mut g = c.benchmark_group("3j_small");
    g.bench_function("racah", |b| {
        b.iter(|| {
            for &d in THREE_J {
                black_box(wigner_3j(d[0] as u32, d[1] as u32, d[2] as u32, 0, 0, 0));
            }
        })
    });
    g.finish();
}

/// Admissible 6j label sets with doubled spins in the thousands.
const SIX_J_THOUSANDS: &[[u32; 6]] = &[
    [1209, 1320, 1193, 1094, 1215, 1012],
    [1041, 1016, 1297, 1231, 1038, 1233],
    [1214, 1179, 1271, 1360, 1239, 1335],
    [1147, 1078, 1355, 1262, 1279, 1328],
    [1175, 1339, 1050, 1008, 1372, 1071],
    [1024, 1338, 1332, 1258, 1388, 1066],
];

fn bench_6j_thousands(c: &mut Criterion) {
    c.bench_function("6j_thousands_racah", |b| {
        b.iter(|| {
            for &d in SIX_J_THOUSANDS {
                black_box(wigner_6j(d[0], d[1], d[2], d[3], d[4], d[5]));
            }
        })
    });
}

/// The repeated-label regime: the same small label sets recur many
/// thousands of times, as in tensor-network consumption. `racah_warm` primes
/// the cache once so every iteration is a hit (target: ~hash-lookup cost);
/// `racah_cold` clears the cache before each batch so every call recomputes the
/// big-rational sum. The gap between them is the cache's payoff.
fn bench_repeated_labels(c: &mut Criterion) {
    let mut g = c.benchmark_group("repeated_labels");

    let run_racah = || {
        for &d in SIX_J {
            black_box(wigner_6j(
                d[0] as u32,
                d[1] as u32,
                d[2] as u32,
                d[3] as u32,
                d[4] as u32,
                d[5] as u32,
            ));
        }
        for &d in THREE_J {
            black_box(wigner_3j(d[0] as u32, d[1] as u32, d[2] as u32, 0, 0, 0));
        }
    };

    // Warm: prime the cache once, then every measured iteration is a hit.
    fusion_symbols::cache::reset();
    run_racah();
    g.bench_function("racah_warm", |b| b.iter(run_racah));

    // Cold: clear before each batch so every call misses and recomputes.
    g.bench_function("racah_cold", |b| {
        b.iter(|| {
            fusion_symbols::cache::reset();
            run_racah();
        })
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_6j,
    bench_3j,
    bench_6j_thousands,
    bench_repeated_labels
);
criterion_main!(benches);
