//! Timing for cold SU(2) F and CGC through `LieGroup`, on the prime-factorized
//! 6j engine and the Cartan engine. Not a CI gate; warm hits are in
//! `benches/warm_hits.rs`.
//!
//! Every coefficient cache is reset before each sample (outside the timer), so
//! the time is one full exact evaluation at each label scale. Run with
//! `cargo bench --bench wigner`.

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;

use fusion_symbols::category::{CgcBlock, FusionCategory, GroupRepresentations, SymbolBlock};
use fusion_symbols::lie::{GroupId, LieGroup, LieIrrep};

/// Admissible F label sets with doubled spins in the thousands, as
/// `[a, b, c, d, e, f]`.
const F_THOUSANDS: &[[i64; 6]] = &[
    [1209, 1320, 1094, 1215, 1193, 1012],
    [1041, 1016, 1231, 1038, 1297, 1233],
    [1214, 1179, 1360, 1239, 1271, 1335],
    [1147, 1078, 1262, 1279, 1355, 1328],
    [1175, 1339, 1008, 1372, 1050, 1071],
    [1024, 1338, 1258, 1388, 1332, 1066],
];

fn irreps(su2: &LieGroup, d: &[i64]) -> Vec<LieIrrep> {
    d.iter().map(|&dj| su2.irrep(&[dj]).unwrap()).collect()
}

fn f(su2: &LieGroup, x: &[LieIrrep]) -> f64 {
    su2.f_symbol(&x[0], &x[1], &x[2], &x[3], &x[4], &x[5])
        .unwrap()
        .as_slice()[0]
}

fn bench_su2_cold(c: &mut Criterion) {
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    // Grow the shared factorial tables past every row.
    let big = irreps(&su2, &[2200; 6]);
    black_box(f(&su2, &big));
    let mut g = c.benchmark_group("su2_cold");
    for d in [
        [2i64, 2, 2, 2, 2, 4],
        [20, 20, 20, 20, 18, 22],
        [200, 200, 200, 200, 180, 220],
        [2000, 2000, 2000, 2000, 1800, 2200],
    ] {
        let x = irreps(&su2, &d);
        g.bench_function(format!("f/{}", d[0]), |b| {
            b.iter_batched(
                fusion_symbols::cache::reset,
                |()| black_box(f(&su2, &x)),
                BatchSize::PerIteration,
            )
        });
    }
    for d in [[2i64, 4, 4], [20, 20, 20], [200, 200, 200]] {
        let x = irreps(&su2, &d);
        g.bench_function(format!("cgc/{}", d[0]), |b| {
            b.iter_batched(
                fusion_symbols::cache::reset,
                |()| black_box(su2.cgc(&x[0], &x[1], &x[2]).unwrap().dims()),
                BatchSize::PerIteration,
            )
        });
    }
    g.finish();
}

fn bench_f_thousands(c: &mut Criterion) {
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let rows: Vec<_> = F_THOUSANDS.iter().map(|d| irreps(&su2, d)).collect();
    c.bench_function("f_thousands_cold", |b| {
        b.iter(|| {
            fusion_symbols::cache::reset();
            for x in &rows {
                black_box(f(&su2, x));
            }
        })
    });
}

criterion_group!(benches, bench_su2_cold, bench_f_thousands);
criterion_main!(benches);
