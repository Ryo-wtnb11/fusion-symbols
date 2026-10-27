//! Ignored release-only coefficient-cache budget pressure collector.

use std::hint::black_box;
use std::time::Instant;

use crate::cache::{self, CoefficientCacheBudgets, CoefficientCacheTier};
use crate::lie::su2::{wigner_3j, wigner_6j};
use crate::lie::sun::{directproduct, Irrep};

fn rss_bytes() -> Option<usize> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()
        .map(|kib| kib.saturating_mul(1024))
}

fn workload() {
    for labels in [
        [20, 20, 20, 20, 20, 20],
        [60, 60, 60, 60, 60, 60],
        [120, 120, 120, 120, 120, 120],
    ] {
        black_box(wigner_6j(
            labels[0], labels[1], labels[2], labels[3], labels[4], labels[5],
        ));
        black_box(wigner_3j(labels[0], labels[1], labels[2], 0, 0, 0));
    }
    for (a, b) in [
        (&[1, 0][..], &[0, 1][..]),
        (&[1, 1], &[1, 1]),
        (&[1, 0, 0], &[0, 0, 1]),
    ] {
        let a = Irrep::from_dynkin(a).unwrap();
        let b = Irrep::from_dynkin(b).unwrap();
        black_box(directproduct(&a, &b).unwrap());
    }
}

fn run(mode: &str, budgets: CoefficientCacheBudgets, require_eviction: bool, partial: bool) {
    let _guard = crate::cache::TEST_CACHE_LOCK.lock().unwrap();
    cache::configure_cache_budgets(budgets).unwrap();
    cache::reset();
    crate::audit_alloc::reset_peak_to_live();
    let start_live = crate::audit_alloc::snapshot().0;
    let start = Instant::now();
    workload();
    let elapsed_ns = start.elapsed().as_nanos();
    let (end_live, peak_live) = crate::audit_alloc::snapshot();
    let [three_j, six_j, derived_f] = [
        CoefficientCacheTier::ThreeJ,
        CoefficientCacheTier::SixJ,
        CoefficientCacheTier::DerivedF,
    ]
    .map(cache::tier_stats);
    let [sun_product, sun_cgc, sun_f, sun_r, bcd_cgc, bcd_f, bcd_r] = [
        CoefficientCacheTier::SunProduct,
        CoefficientCacheTier::SunCgc,
        CoefficientCacheTier::SunF,
        CoefficientCacheTier::SunR,
        CoefficientCacheTier::BcdCgc,
        CoefficientCacheTier::BcdF,
        CoefficientCacheTier::BcdR,
    ]
    .map(cache::tier_stats);
    let evictions: u64 = CoefficientCacheTier::ALL
        .iter()
        .map(|&tier| cache::tier_stats(tier).evictions)
        .sum();
    assert_eq!(evictions > 0, require_eviction);
    let active = [three_j, six_j, sun_product];
    if partial {
        for (stats, tier) in active.into_iter().zip([
            CoefficientCacheTier::ThreeJ,
            CoefficientCacheTier::SixJ,
            CoefficientCacheTier::SunProduct,
        ]) {
            assert!(stats.bytes > 0 && stats.bytes <= budgets.limit(tier));
            assert!(stats.evictions > 0);
        }
    }
    let sample = std::env::var("COEFFICIENT_CACHE_BUDGET_SAMPLE").unwrap_or_else(|_| "0".into());
    let revision = option_env!("RACAH_BUDGET_REVISION").unwrap_or("not-embedded");
    eprintln!(
        "COEFFICIENT_CACHE_BUDGET_PRESSURE {{\"schema\":1,\"revision\":\"{revision}\",\"mode\":\"{mode}\",\"sample\":{sample},\"budgets\":{{\"three_j\":{},\"six_j\":{},\"derived_f\":{},\"sun_product\":{},\"sun_cgc\":{},\"sun_f\":{},\"bcd_cgc\":{},\"bcd_f\":{},\"sun_r\":{},\"bcd_r\":{}}},\"elapsed_ns\":{elapsed_ns},\"base\":[[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}]],\"generated\":[[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}],[{},{},{},{},{}]],\"requested_live_start_bytes\":{start_live},\"requested_live_end_bytes\":{end_live},\"requested_live_peak_bytes\":{peak_live},\"rss_bytes\":{} }}",
        budgets.limit(CoefficientCacheTier::ThreeJ), budgets.limit(CoefficientCacheTier::SixJ), budgets.limit(CoefficientCacheTier::DerivedF), budgets.limit(CoefficientCacheTier::SunProduct), budgets.limit(CoefficientCacheTier::SunCgc), budgets.limit(CoefficientCacheTier::SunF), budgets.limit(CoefficientCacheTier::BcdCgc), budgets.limit(CoefficientCacheTier::BcdF), budgets.limit(CoefficientCacheTier::SunR), budgets.limit(CoefficientCacheTier::BcdR),
        three_j.entries, three_j.bytes, three_j.hits, three_j.misses, three_j.evictions,
        six_j.entries, six_j.bytes, six_j.hits, six_j.misses, six_j.evictions,
        derived_f.entries, derived_f.bytes, derived_f.hits, derived_f.misses, derived_f.evictions,
        sun_product.entries, sun_product.bytes, sun_product.hits, sun_product.misses, sun_product.evictions,
        sun_cgc.entries, sun_cgc.bytes, sun_cgc.hits, sun_cgc.misses, sun_cgc.evictions,
        sun_f.entries, sun_f.bytes, sun_f.hits, sun_f.misses, sun_f.evictions,
        bcd_cgc.entries, bcd_cgc.bytes, bcd_cgc.hits, bcd_cgc.misses, bcd_cgc.evictions,
        bcd_f.entries, bcd_f.bytes, bcd_f.hits, bcd_f.misses, bcd_f.evictions,
        sun_r.entries, sun_r.bytes, sun_r.hits, sun_r.misses, sun_r.evictions,
        bcd_r.entries, bcd_r.bytes, bcd_r.hits, bcd_r.misses, bcd_r.evictions,
        rss_bytes().map_or_else(|| "null".to_owned(), |n| n.to_string()),
    );
}

#[test]
#[ignore = "manual release-only coefficient-cache budget pressure measurement"]
fn coefficient_cache_budget_pressure_default() {
    run("default", CoefficientCacheBudgets::default(), false, false);
}

#[test]
#[ignore = "manual release-only coefficient-cache budget pressure measurement"]
fn coefficient_cache_budget_pressure_constrained() {
    let budgets = CoefficientCacheBudgets::default()
        .with_limit(CoefficientCacheTier::ThreeJ, 400)
        .with_limit(CoefficientCacheTier::SixJ, 400)
        .with_limit(CoefficientCacheTier::SunProduct, 400);
    run("constrained", budgets, true, true);
}

#[test]
#[ignore = "manual release-only coefficient-cache budget pressure measurement"]
fn coefficient_cache_budget_pressure_disabled() {
    run("disabled", CoefficientCacheBudgets::disabled(), true, false);
}
