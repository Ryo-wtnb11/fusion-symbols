//! Timing, live-allocation and RSS reports for the coefficient caches, one JSON
//! line per measurement on stderr. This is a measurement, not a test: it
//! asserts nothing. The cache contracts live in `tests/cache_trim_*.rs` and
//! `tests/coefficient_cache_budget_*.rs`.
//!
//! ```text
//! cargo run --release --example cache_pressure -- <report>
//! ```
//!
//! Reports: `audit` (cold, warm and reset-per-query cost per group),
//! `budget` (default, constrained and disabled policies), `trim` (retained,
//! partial and zero trim targets), `cgc` (Cartan CGC tier under an 8 MiB
//! budget) and `d4` (D4 `350 x 350` working set, query by query). Run one
//! report per process: `budget` installs a one-shot policy, and live and RSS
//! figures are process-wide. Live-allocation figures count only requests that
//! reach Rust's System allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use fusion_symbols::cache::{self, CoefficientCacheBudgets, CoefficientCacheTier, TierStats};
use fusion_symbols::category::{FusionCategory, GroupRepresentations};
use fusion_symbols::lie::su2::{wigner_3j, wigner_6j};
use fusion_symbols::lie::{GroupId, LieGroup};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Tracking;

unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, size) };
        if !new.is_null() {
            if size >= layout.size() {
                grow(size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - size, Ordering::Relaxed);
            }
        }
        new
    }
}

fn grow(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

#[global_allocator]
static ALLOCATOR: Tracking = Tracking;

fn rss_bytes() -> Option<usize> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let kib = std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()?;
    Some(kib.saturating_mul(1024))
}

struct Sample {
    elapsed_ns: u128,
    start_live: usize,
    end_live: usize,
    peak_live: usize,
}

fn timed(work: impl FnOnce()) -> Sample {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
    let start_live = LIVE.load(Ordering::Relaxed);
    let start = Instant::now();
    work();
    Sample {
        elapsed_ns: start.elapsed().as_nanos(),
        start_live,
        end_live: LIVE.load(Ordering::Relaxed),
        peak_live: PEAK.load(Ordering::Relaxed),
    }
}

fn tier_json(tier: CoefficientCacheTier, s: TierStats) -> String {
    format!(
        "\"{tier}\":[{},{},{},{},{}]",
        s.entries, s.bytes, s.hits, s.misses, s.evictions
    )
}

fn emit(label: &str, s: &Sample) {
    let tiers = CoefficientCacheTier::ALL
        .iter()
        .map(|&t| tier_json(t, cache::tier_stats(t)))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "{{\"label\":\"{label}\",\"elapsed_ns\":{},\"tiers[entries,bytes,hits,misses,evictions]\":{{{tiers}}},\"live_start\":{},\"live_end\":{},\"live_peak\":{},\"rss\":{}}}",
        s.elapsed_ns,
        s.start_live,
        s.end_live,
        s.peak_live,
        rss_bytes().map_or_else(|| "null".to_owned(), |n| n.to_string()),
    );
}

fn su2() {
    black_box(wigner_3j(20, 20, 20, 0, 0, 0));
    black_box(wigner_6j(60, 60, 60, 60, 60, 60));
}

fn sun(n: usize, reset_each: bool) {
    let g = LieGroup::new(GroupId::su(n).unwrap()).unwrap();
    let mut fundamental = vec![0; n - 1];
    fundamental[0] = 1;
    let a = g.irrep(&fundamental).unwrap();
    let b = g.dual(&a);
    let mut adjoint = fundamental;
    adjoint[n - 2] += 1;
    let adjoint = g.irrep(&adjoint).unwrap();
    let step = || {
        if reset_each {
            cache::reset();
        }
    };
    step();
    black_box(g.fusion(&a, &b).unwrap().count());
    step();
    black_box(g.cgc(&a, &b, &adjoint).unwrap());
    step();
    black_box(g.f_symbol(&a, &b, &a, &a, &adjoint, &adjoint).unwrap());
}

type Family = fn(bool);

fn audit() {
    let families: [(&str, Family); 3] = [
        ("su2", |_| su2()),
        ("su3", |reset| sun(3, reset)),
        ("su4", |reset| sun(4, reset)),
    ];
    for (label, work) in families {
        cache::reset();
        emit(&format!("{label}:cold"), &timed(|| work(false)));
        emit(&format!("{label}:warm"), &timed(|| work(false)));
        emit(
            &format!("{label}:reset_before_each_query"),
            &timed(|| work(true)),
        );
    }
}

fn mixed_workload() {
    for labels in [[20; 6], [60; 6], [120; 6]] {
        black_box(wigner_6j(
            labels[0], labels[1], labels[2], labels[3], labels[4], labels[5],
        ));
        black_box(wigner_3j(labels[0], labels[1], labels[2], 0, 0, 0));
    }
    for (id, label) in [
        (GroupId::su(3).unwrap(), vec![1, 0]),
        (GroupId::su(3).unwrap(), vec![1, 1]),
        (GroupId::su(4).unwrap(), vec![1, 0, 0]),
        (GroupId::sp(4).unwrap(), vec![1, 0]),
        (GroupId::e6(), vec![1, 0, 0, 0, 0, 0]),
    ] {
        let g = LieGroup::new(id).unwrap();
        let a = g.irrep(&label).unwrap();
        let b = g.dual(&a);
        black_box(g.fusion(&a, &b).unwrap().count());
    }
}

fn budget() {
    let constrained = CoefficientCacheBudgets::default()
        .with_limit(CoefficientCacheTier::Wigner3j, 400)
        .with_limit(CoefficientCacheTier::Wigner6j, 400)
        .with_limit(CoefficientCacheTier::Product, 2048);
    let mode = std::env::args().nth(2).unwrap_or_else(|| "default".into());
    let budgets = match mode.as_str() {
        "default" => CoefficientCacheBudgets::default(),
        "constrained" => constrained,
        "disabled" => CoefficientCacheBudgets::disabled(),
        other => panic!("budget mode {other}: expected default, constrained or disabled"),
    };
    cache::configure_cache_budgets(budgets).unwrap();
    cache::reset();
    emit(&format!("budget:{mode}"), &timed(mixed_workload));
}

fn trim() {
    let mode = std::env::args().nth(2).unwrap_or_else(|| "partial".into());
    let tiers = [
        CoefficientCacheTier::Wigner3j,
        CoefficientCacheTier::Wigner6j,
        CoefficientCacheTier::Product,
    ];
    cache::reset();
    mixed_workload();
    let targets = tiers.map(|t| {
        let bytes = cache::tier_stats(t).bytes;
        match mode.as_str() {
            "retained" => bytes,
            "partial" => bytes / 2,
            "zero" => 0,
            other => panic!("trim mode {other}: expected retained, partial or zero"),
        }
    });
    for (t, target) in tiers.into_iter().zip(targets) {
        eprintln!(
            "trim {t}: target {target} -> {:?}",
            cache::trim_to(t, target)
        );
    }
    emit(&format!("trim:{mode}:refill"), &timed(mixed_workload));
}

fn cartan_query(label: &str, run: impl FnOnce()) {
    let before = cache::tier_stats(CoefficientCacheTier::Cgc);
    let sample = timed(run);
    let after = cache::tier_stats(CoefficientCacheTier::Cgc);
    eprintln!(
        "{{\"query\":\"{label}\",\"elapsed_ns\":{},\"cgc_entries\":{},\"cgc_bytes\":{},\"cgc_bytes_delta\":{},\"evictions_delta\":{},\"live_peak_delta\":{},\"rss\":{}}}",
        sample.elapsed_ns,
        after.entries,
        after.bytes,
        after.bytes as i64 - before.bytes as i64,
        after.evictions - before.evictions,
        sample.peak_live.saturating_sub(sample.start_live),
        rss_bytes().map_or_else(|| "null".to_owned(), |n| n.to_string()),
    );
}

fn cgc() {
    cache::configure_cache_budgets(
        CoefficientCacheBudgets::default().with_limit(CoefficientCacheTier::Cgc, 8 << 20),
    )
    .unwrap();
    cache::reset();
    let group = |id: GroupId| LieGroup::new(id).unwrap();
    let (g2, su4, sp4, f4, e6, d5) = (
        group(GroupId::g2()),
        group(GroupId::su(4).unwrap()),
        group(GroupId::sp(4).unwrap()),
        group(GroupId::f4()),
        group(GroupId::e6()),
        group(GroupId::spin(10).unwrap()),
    );
    let all = |g: &LieGroup, a: &[i64], b: &[i64]| {
        let (a, b) = (g.irrep(a).unwrap(), g.irrep(b).unwrap());
        for c in g.fusion(&a, &b).unwrap() {
            black_box(g.cgc(&a, &b, &c).unwrap());
        }
    };
    let phase = |g: &LieGroup, a: &[i64]| {
        black_box(g.frobenius_schur_phase(&g.irrep(a).unwrap()).unwrap());
    };
    let pass = |name: &str| {
        cartan_query(&format!("{name} G2 27x27 all"), || {
            all(&g2, &[2, 0], &[2, 0])
        });
        cartan_query(&format!("{name} SU4 15x15 all"), || {
            all(&su4, &[1, 0, 1], &[1, 0, 1])
        });
        cartan_query(&format!("{name} Sp4 35x35 all"), || {
            all(&sp4, &[2, 1], &[2, 1])
        });
        cartan_query(&format!("{name} F4 26x26 all"), || {
            all(&f4, &[0, 0, 0, 1], &[0, 0, 0, 1])
        });
        cartan_query(&format!("{name} E6 phase 27"), || {
            phase(&e6, &[1, 0, 0, 0, 0, 0])
        });
        cartan_query(&format!("{name} E6 phase 351"), || {
            phase(&e6, &[0, 0, 0, 0, 1, 0])
        });
        cartan_query(&format!("{name} D5 phase"), || phase(&d5, &[1, 0, 0, 1, 0]));
    };
    pass("cold");
    pass("warm");
    eprintln!(
        "trim to zero: {:?}",
        cache::trim_to(CoefficientCacheTier::Cgc, 0)
    );
    pass("after-trim");
}

fn d4() {
    cache::reset();
    let g = LieGroup::new(GroupId::spin(8).unwrap()).unwrap();
    let a = g.irrep(&[1, 0, 1, 1]).unwrap();
    for c in g.fusion(&a, &a).unwrap() {
        let n = g.nsymbol(&a, &a, &c).unwrap();
        let label = format!(
            "D4 350x350 -> {:?} (dim {}, N {n})",
            c.dynkin().collect::<Vec<_>>(),
            g.dim(&c)
        );
        cartan_query(&label, || {
            black_box(g.cgc(&a, &a, &c).unwrap());
        });
    }
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("audit") => audit(),
        Some("budget") => budget(),
        Some("trim") => trim(),
        Some("cgc") => cgc(),
        Some("d4") => d4(),
        _ => eprintln!("usage: cache_pressure audit | budget [default|constrained|disabled] | trim [retained|partial|zero] | cgc | d4"),
    }
}
