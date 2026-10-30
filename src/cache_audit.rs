//! Ignored local coefficient-cache evidence collector; not a production benchmark.

use std::hint::black_box;
use std::time::{Duration, Instant};

use crate::cache::{self, CoefficientCacheTier};
use crate::category::{FusionCategory, GroupRepresentations};
use crate::lie::su2::{f_symbol as exact_f, wigner_3j, wigner_6j};
use crate::lie::{GroupId, LieGroup, LieIrrep};
use crate::primefactor;

fn group(n: usize) -> LieGroup {
    LieGroup::new(GroupId::su(n).unwrap()).unwrap()
}

fn rss_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()?
            .lines()
            .find_map(|line| line.strip_prefix("VmRSS:"))?
            .split_whitespace()
            .next()?
            .parse::<usize>()
            .ok()
            .map(|kib| kib.saturating_mul(1024))
    }
    #[cfg(target_os = "macos")]
    {
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
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

struct Measurement {
    elapsed: Duration,
    start_live: usize,
    end_live: usize,
    peak_live: usize,
}

fn emit(phase: &str, measurement: Option<Measurement>) {
    let tiers = CoefficientCacheTier::ALL
        .iter()
        .map(|&tier| {
            let s = cache::tier_stats(tier);
            format!(
                "\"{tier}\":[{},{},{},{},{}]",
                s.entries, s.bytes, s.hits, s.misses, s.evictions
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let table = primefactor::table_stats();
    let (live, _) = crate::audit_alloc::snapshot();
    let (elapsed, start_live, end_live, peak_live) = measurement
        .map_or((0, live, live, live), |m| {
            (m.elapsed.as_nanos(), m.start_live, m.end_live, m.peak_live)
        });
    eprintln!(
        concat!(
            "COEFFICIENT_CACHE_AUDIT {{\"kind\":\"phase\",\"phase\":\"{}\",\"elapsed_ns\":{},",
            "\"tiers\":{{{}}},",
            "\"factorial_rows\":{},\"primes\":{},\"table_capacity_bytes\":{},",
            "\"system_requested_live_start_bytes\":{},\"system_requested_live_end_bytes\":{},\"system_requested_live_peak_bytes\":{},\"system_requested_live_transient_lower_bound_bytes\":{},\"rss_bytes\":{} }}"
        ),
        phase,
        elapsed,
        tiers,
        table.factorial_rows, table.primes, table.retained_capacity_bytes,
        start_live,
        end_live,
        peak_live,
        peak_live.saturating_sub(start_live.max(end_live)),
        rss_bytes().map_or_else(|| "null".to_owned(), |n| n.to_string()),
    );
}

fn timed(work: impl FnOnce()) -> Measurement {
    crate::audit_alloc::reset_peak_to_live();
    let start_live = crate::audit_alloc::snapshot().0;
    let start = Instant::now();
    work();
    let elapsed = start.elapsed();
    let (end_live, peak_live) = crate::audit_alloc::snapshot();
    Measurement {
        elapsed,
        start_live,
        end_live,
        peak_live,
    }
}

fn clone_slope<T>(label: &str, make: impl Fn() -> T) {
    let before = crate::audit_alloc::allocation_totals();
    let one = make();
    black_box(&one);
    let one_totals = crate::audit_alloc::allocation_totals();
    let mut eight = Vec::with_capacity(8);
    for _ in 0..8 {
        eight.push(make());
    }
    black_box(&eight);
    let nine_totals = crate::audit_alloc::allocation_totals();
    drop(eight);
    drop(one);
    eprintln!(
        "COEFFICIENT_CACHE_AUDIT {{\"kind\":\"clone\",\"label\":\"{label}\",\"one_successful_alloc_calls\":{},\"one_requested_alloc_bytes\":{},\"nine_successful_alloc_calls\":{},\"nine_requested_alloc_bytes\":{}}}",
        one_totals.0.saturating_sub(before.0),
        one_totals.1.saturating_sub(before.1),
        nine_totals.0.saturating_sub(before.0),
        nine_totals.1.saturating_sub(before.1),
    );
}

fn su2() {
    let _ = wigner_3j(20, 20, 20, 0, 0, 0);
    let _ = wigner_6j(60, 60, 60, 60, 60, 60);
    let _ = exact_f(2, 2, 2, 2, 2, 2);
}

fn su2_no_reuse() {
    cache::reset();
    let _ = wigner_3j(20, 20, 20, 0, 0, 0);
    cache::reset();
    let _ = wigner_6j(60, 60, 60, 60, 60, 60);
    cache::reset();
    let _ = exact_f(2, 2, 2, 2, 2, 2);
}

fn sun(n: usize, reset: bool) {
    let g = group(n);
    let mut fundamental = vec![0; n - 1];
    fundamental[0] = 1;
    let a: LieIrrep = g.irrep(&fundamental).unwrap();
    let b = g.dual(&a);
    let mut adjoint = fundamental.clone();
    adjoint[n - 2] += 1;
    let adjoint = g.irrep(&adjoint).unwrap();
    let step = || {
        if reset {
            cache::reset();
        }
    };
    step();
    let _ = g.fusion(&a, &b).unwrap().count();
    step();
    assert!(g.cgc(&a, &b, &adjoint).is_ok());
    step();
    let _ = g.f_symbol(&a, &b, &a, &a, &adjoint, &adjoint).unwrap();
}

fn su3() {
    sun(3, false);
}

fn su3_no_reuse() {
    sun(3, true);
}

fn su4() {
    sun(4, false);
}

fn su4_no_reuse() {
    sun(4, true);
}

fn run(label: &str, work: fn(), no_reuse: fn()) {
    cache::reset();
    emit(&format!("{label}:reset"), None);
    emit(&format!("{label}:cold"), Some(timed(work)));
    emit(&format!("{label}:warm"), Some(timed(work)));
    emit(
        &format!("{label}:reset_before_each_query"),
        Some(timed(no_reuse)),
    );
}

fn sequential_trace(label: &str, work: &[(&str, fn())]) {
    cache::reset();
    emit(&format!("{label}:reset"), None);
    for (family, run) in work {
        emit(&format!("{label}:after_{family}"), Some(timed(*run)));
    }
}

#[test]
#[ignore = "manual release-only cache measurement; run --release --features lie-generated -- --ignored --nocapture"]
fn coefficient_cache_audit() {
    let _guard = crate::cache::TEST_CACHE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    eprintln!(
        "COEFFICIENT_CACHE_AUDIT {{\"kind\":\"metadata\",\"revision\":\"{}\",\"features\":\"lie-generated\",\"consumer_revision\":\"N/A\",\"allocator\":\"System tracking test wrapper\",\"platform\":\"{}/{}\",\"rustc\":\"recorded by command\"}}",
        option_env!("RACAH_AUDIT_REVISION").unwrap_or("not-embedded; record git rev-parse HEAD with command"),
        std::env::consts::OS,
        std::env::consts::ARCH,
    );
    run("su2", su2, su2_no_reuse);
    run("su3", su3, su3_no_reuse);
    run("su4", su4, su4_no_reuse);
    sequential_trace(
        "sequential_su2_su3_su4",
        &[("su2", su2), ("su3", su3), ("su4", su4)],
    );
    sequential_trace(
        "sequential_su4_su3_su2",
        &[("su4", su4), ("su3", su3), ("su2", su2)],
    );

    cache::reset();
    su2();
    clone_slope("exact_su2_6j", || wigner_6j(60, 60, 60, 60, 60, 60));
    let g = group(3);
    let three = g.irrep(&[1, 0]).unwrap();
    let three_bar = g.irrep(&[0, 1]).unwrap();
    let eight = g.irrep(&[1, 1]).unwrap();
    let _ = g.fusion(&three, &three_bar).unwrap().count();
    clone_slope("product", || g.fusion(&three, &three_bar).unwrap());
    let _ = g.cgc(&three, &three_bar, &eight).unwrap();
    clone_slope("public_cgc", || g.cgc(&three, &three_bar, &eight).unwrap());
    let _ = g.f_symbol(&three, &three_bar, &three, &three, &eight, &eight);
    clone_slope("public_f", || {
        g.f_symbol(&three, &three_bar, &three, &three, &eight, &eight)
            .unwrap()
    });
}
