//! SU(N) tensor-product cache working-set collector, cold/warm timing and
//! warm cycles over many distinct SU(3) pairs.
//!
//! Not a CI gate. Run with
//! `cargo bench --bench sun_product --features lie-generated`.

#[path = "../tests/common/lie.rs"]
mod lie_test;
use criterion::{criterion_group, criterion_main, Criterion};
use fusion_symbols::cache;
use fusion_symbols::lie::LieIrrep as Irrep;
use lie_test::{cgc, f_symbol, product as directproduct};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

struct CountingAllocator;

static ALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn irr(dynkin: &[i64]) -> Irrep {
    lie_test::irrep(dynkin).unwrap()
}

fn representative_families() -> Vec<(&'static str, Vec<Irrep>)> {
    vec![
        (
            "su3",
            vec![
                irr(&[0, 0]),
                irr(&[1, 0]),
                irr(&[0, 1]),
                irr(&[2, 0]),
                irr(&[0, 2]),
                irr(&[1, 1]),
                irr(&[3, 0]),
                irr(&[0, 3]),
                irr(&[2, 1]),
                irr(&[1, 2]),
                irr(&[2, 2]),
            ],
        ),
        (
            "su4",
            vec![
                irr(&[0, 0, 0]),
                irr(&[1, 0, 0]),
                irr(&[0, 0, 1]),
                irr(&[0, 1, 0]),
                irr(&[2, 0, 0]),
                irr(&[0, 0, 2]),
                irr(&[1, 0, 1]),
                irr(&[1, 1, 0]),
            ],
        ),
    ]
}

fn allocation_snapshot() -> (u64, u64) {
    (
        ALLOCATION_CALLS.load(Ordering::Relaxed),
        ALLOCATED_BYTES.load(Ordering::Relaxed),
    )
}

fn run_racah_generation_workload() {
    let cgc_cases = [
        (irr(&[2]), irr(&[2]), irr(&[2])),
        (irr(&[1, 0]), irr(&[0, 1]), irr(&[1, 1])),
        (irr(&[1, 1]), irr(&[1, 1]), irr(&[1, 1])),
        (irr(&[1, 1]), irr(&[1, 1]), irr(&[2, 2])),
        (irr(&[1, 0, 0]), irr(&[0, 0, 1]), irr(&[1, 0, 1])),
    ];
    for (s1, s2, s3) in cgc_cases {
        black_box(cgc(&s1, &s2, &s3).unwrap());
    }

    let f_cases = [
        [
            irr(&[1]),
            irr(&[1]),
            irr(&[1]),
            irr(&[1]),
            irr(&[0]),
            irr(&[0]),
        ],
        [
            irr(&[1, 0]),
            irr(&[0, 1]),
            irr(&[1, 0]),
            irr(&[1, 0]),
            irr(&[1, 1]),
            irr(&[1, 1]),
        ],
        [
            irr(&[1, 1]),
            irr(&[1, 1]),
            irr(&[1, 1]),
            irr(&[1, 1]),
            irr(&[1, 1]),
            irr(&[1, 1]),
        ],
        [
            irr(&[1, 0, 1]),
            irr(&[1, 0, 1]),
            irr(&[1, 0, 1]),
            irr(&[1, 0, 1]),
            irr(&[1, 0, 1]),
            irr(&[1, 0, 1]),
        ],
    ];
    for s in f_cases {
        black_box(f_symbol(&s[0], &s[1], &s[2], &s[3], &s[4], &s[5]).unwrap());
    }
}

fn collect_working_set() {
    cache::reset();
    let families = representative_families();
    let mut unique_pairs = 0usize;
    for (_, labels) in &families {
        for i in 0..labels.len() {
            for j in i..labels.len() {
                black_box(directproduct(&labels[i], &labels[j]).unwrap());
                unique_pairs += 1;
            }
        }
    }
    let cold = cache::tier_stats(cache::CoefficientCacheTier::Product);

    for (_, labels) in &families {
        for i in 0..labels.len() {
            for j in i..labels.len() {
                black_box(directproduct(&labels[j], &labels[i]).unwrap());
            }
        }
    }
    let warm = cache::tier_stats(cache::CoefficientCacheTier::Product);

    let octet = irr(&[1, 1]);
    let _ = directproduct(&octet, &octet).unwrap();
    let before = allocation_snapshot();
    let public_map = directproduct(&octet, &octet).unwrap();
    let after = allocation_snapshot();
    let public_allocations = after.0 - before.0;
    let public_bytes = after.1 - before.1;

    eprintln!(
        "SUN_PRODUCT_STRUCTURAL families=su3:11,su4:8 unique_pairs={} peak_retained_bytes={} cold_hits={} cold_misses={} warm_hits={} warm_misses={} entries={} evictions={} public_case=su3_8x8 public_channels={} public_reconstruction_allocations={} public_reconstruction_bytes={}",
        unique_pairs,
        cold.bytes,
        cold.hits,
        cold.misses,
        warm.hits,
        warm.misses,
        warm.entries,
        warm.evictions,
        public_map.len(),
        public_allocations,
        public_bytes,
    );
    black_box(public_map);

    cache::reset();
    run_racah_generation_workload();
    let generated = cache::tier_stats(cache::CoefficientCacheTier::Product);
    let lookups = generated.hits + generated.misses;
    let hit_rate = if lookups == 0 {
        0.0
    } else {
        generated.hits as f64 / lookups as f64
    };
    eprintln!(
        "SUN_PRODUCT_RACAH_WORKLOAD cgc_cases=5 f_cases=4 unique_pairs={} peak_retained_bytes={} hits={} misses={} hit_rate={:.6} evictions={}",
        generated.entries,
        generated.bytes,
        generated.hits,
        generated.misses,
        hit_rate,
        generated.evictions,
    );
}

fn bench_product(c: &mut Criterion) {
    collect_working_set();

    let a = irr(&[1, 1]);
    let b = irr(&[1, 1]);
    cache::reset();
    let _ = directproduct(&a, &b).unwrap();

    c.bench_function("sun_product/su3_8x8_warm_public_map", |bench| {
        bench.iter(|| black_box(directproduct(black_box(&a), black_box(&b)).unwrap()))
    });
    c.bench_function("sun_product/su3_8x8_cold_with_reset", |bench| {
        bench.iter(|| {
            cache::reset();
            black_box(directproduct(black_box(&a), black_box(&b)).unwrap())
        })
    });
}

/// One query per iteration, cycling over the first `k` pairs `a <= b` of the
/// SU(3) labels with Dynkin sum at most 6 (406 pairs in all), after one
/// priming pass: `n` asks only N, `mixed` iterates the fusion channels first.
fn bench_cycle(c: &mut Criterion) {
    use fusion_symbols::category::FusionCategory;
    let g = lie_test::group(&irr(&[1, 0]));
    let labels: Vec<Irrep> = (0..=6)
        .flat_map(|p| (0..=6 - p).map(move |q| irr(&[p, q])))
        .collect();
    let mut pairs = Vec::new();
    for (i, a) in labels.iter().enumerate() {
        for b in &labels[i..] {
            let target = g.fusion(a, b).unwrap().next().unwrap().0;
            pairs.push((a.clone(), b.clone(), target));
        }
    }
    let n = |(a, b, t): &Key| g.nsymbol(a, b, t).unwrap();
    let mixed = |key: &Key| g.fusion(&key.0, &key.1).unwrap().count() as u32 + n(key);
    for k in [16, 200, 406] {
        cycle(c, "n", &pairs[..k], &n);
    }
    for k in [200, 406] {
        cycle(c, "mixed", &pairs[..k], &mixed);
    }
}

type Key = (Irrep, Irrep, Irrep);

fn cycle(c: &mut Criterion, name: &str, keys: &[Key], query: &dyn Fn(&Key) -> u32) {
    cache::reset();
    keys.iter().for_each(|key| {
        black_box(query(key));
    });
    let mut i = 0;
    c.bench_function(
        &format!("sun_product/su3_cycle/{name}/{}", keys.len()),
        |bench| {
            bench.iter(|| {
                let key = &keys[i];
                i = (i + 1) % keys.len();
                black_box(query(black_box(key)))
            })
        },
    );
}

criterion_group!(benches, bench_product, bench_cycle);
criterion_main!(benches);
