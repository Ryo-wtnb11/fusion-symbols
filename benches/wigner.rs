//! Timing for the prime-factorized 3j/6j engine. Not a CI gate.
//!
//! Exact-value evaluation on doubled spins <= 254, the thousands tier, and the
//! repeated-label (cache hit) regime. Run with `cargo bench`.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;

use fusion_symbols::category::{FusionCategory, SymbolBlock};
use fusion_symbols::lie::su2::{clebsch_gordan, wigner_3j, wigner_6j};
use fusion_symbols::lie::{GroupId, LieGroup};

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
    c.bench_function("6j_thousands_cold", |b| {
        b.iter(|| {
            fusion_symbols::cache::reset();
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

/// Cold misses with the shared factorial tables already grown: every coefficient
/// cache is reset before each sample (outside the timer), so the time is one
/// full exact evaluation at each label scale.
fn bench_su2_cold(c: &mut Criterion) {
    // Grow the tables past every row (largest factorial 4401!).
    black_box(wigner_6j(2200, 2200, 2200, 2200, 2200, 2200));
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let mut g = c.benchmark_group("su2_cold");
    for dj in [4u32, 20, 200, 2000] {
        g.bench_function(format!("3j/{dj}"), |b| {
            b.iter_batched(
                fusion_symbols::cache::reset,
                |()| black_box(wigner_3j(dj, dj, dj, 0, 0, 0).to_f64()),
                BatchSize::PerIteration,
            )
        });
    }
    for [j1, m1, j2, m2, j3, m3] in [
        [2, 2, 4, -2, 4, 0],
        [20, 6, 20, -10, 20, -4],
        [200, 60, 200, -100, 200, -40],
        [2000, 600, 2000, -1000, 2000, -400],
    ] {
        g.bench_function(format!("cg/{j1}"), |b| {
            b.iter_batched(
                fusion_symbols::cache::reset,
                |()| {
                    black_box(clebsch_gordan(j1 as u32, m1, j2 as u32, m2, j3 as u32, m3).to_f64())
                },
                BatchSize::PerIteration,
            )
        });
    }
    for d in [
        [2u32, 4, 4, 2, 4, 2],
        [20, 18, 16, 19, 17, 21],
        [200, 180, 160, 190, 170, 210],
        [2000, 1800, 1600, 1900, 1700, 2100],
    ] {
        g.bench_function(format!("6j/{}", d[0]), |b| {
            b.iter_batched(
                fusion_symbols::cache::reset,
                |()| black_box(wigner_6j(d[0], d[1], d[2], d[3], d[4], d[5]).to_f64()),
                BatchSize::PerIteration,
            )
        });
    }
    for d in [
        [2i64, 2, 2, 2, 2, 4],
        [20, 20, 20, 20, 18, 22],
        [200, 200, 200, 200, 180, 220],
        [2000, 2000, 2000, 2000, 1800, 2200],
    ] {
        let x: Vec<_> = d.iter().map(|&dj| su2.irrep(&[dj]).unwrap()).collect();
        g.bench_function(format!("f/{}", d[0]), |b| {
            b.iter_batched(
                fusion_symbols::cache::reset,
                |()| {
                    let f = su2.f_symbol(&x[0], &x[1], &x[2], &x[3], &x[4], &x[5]);
                    black_box(f.unwrap().as_slice()[0])
                },
                BatchSize::PerIteration,
            )
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_su2_cold,
    bench_6j,
    bench_3j,
    bench_6j_thousands,
    bench_repeated_labels
);
criterion_main!(benches);
