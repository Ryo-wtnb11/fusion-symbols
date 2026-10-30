//! Ignored release-only coefficient-cache budget pressure collector.

use std::hint::black_box;
use std::time::Instant;

use crate::cache::{self, CoefficientCacheBudgets, CoefficientCacheTier};
use crate::lie::su2::{wigner_3j, wigner_6j};

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
    use crate::category::FusionCategory;
    use crate::lie::{GroupId, LieGroup};
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

fn run(mode: &str, budgets: CoefficientCacheBudgets, require_eviction: bool, partial: bool) {
    let _guard = crate::cache::TEST_CACHE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache::configure_cache_budgets(budgets).unwrap();
    cache::reset();
    crate::audit_alloc::reset_peak_to_live();
    let start_live = crate::audit_alloc::snapshot().0;
    let start = Instant::now();
    workload();
    let elapsed_ns = start.elapsed().as_nanos();
    let (end_live, peak_live) = crate::audit_alloc::snapshot();
    let stats = CoefficientCacheTier::ALL
        .iter()
        .map(|&tier| (tier, cache::tier_stats(tier)))
        .collect::<Vec<_>>();
    let evictions: u64 = stats.iter().map(|(_, s)| s.evictions).sum();
    assert_eq!(evictions > 0, require_eviction);
    if partial {
        for (tier, stats) in &stats {
            if matches!(
                tier,
                CoefficientCacheTier::Wigner3j
                    | CoefficientCacheTier::Wigner6j
                    | CoefficientCacheTier::Product
            ) {
                assert!(stats.bytes > 0 && stats.bytes <= budgets.limit(*tier));
                assert!(stats.evictions > 0);
            }
        }
    }
    let sample = std::env::var("COEFFICIENT_CACHE_BUDGET_SAMPLE").unwrap_or_else(|_| "0".into());
    let revision = option_env!("RACAH_BUDGET_REVISION").unwrap_or("not-embedded");
    let tiers = stats
        .iter()
        .map(|(tier, s)| {
            format!(
                "\"{tier}\":{{\"budget\":{},\"stats\":[{},{},{},{},{}]}}",
                budgets.limit(*tier),
                s.entries,
                s.bytes,
                s.hits,
                s.misses,
                s.evictions
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "COEFFICIENT_CACHE_BUDGET_PRESSURE {{\"schema\":2,\"revision\":\"{revision}\",\"mode\":\"{mode}\",\"sample\":{sample},\"tiers\":{{{tiers}}},\"elapsed_ns\":{elapsed_ns},\"requested_live_start_bytes\":{start_live},\"requested_live_end_bytes\":{end_live},\"requested_live_peak_bytes\":{peak_live},\"rss_bytes\":{} }}",
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
        .with_limit(CoefficientCacheTier::Wigner3j, 400)
        .with_limit(CoefficientCacheTier::Wigner6j, 400)
        .with_limit(CoefficientCacheTier::Product, 2048);
    run("constrained", budgets, true, true);
}

#[test]
#[ignore = "manual release-only coefficient-cache budget pressure measurement"]
fn coefficient_cache_budget_pressure_disabled() {
    run("disabled", CoefficientCacheBudgets::disabled(), true, false);
}

/// One Cartan-engine query under the configured CGC budget: its time, the
/// module and channel builds of this thread, the CGC tier afterwards, the
/// live-allocation high-water during the query (process-wide; run alone) and
/// the RSS sampled at query end.
fn cartan_query(label: &str, run: impl FnOnce()) {
    use crate::lie::general::cgc::BUILDS;
    let builds = || BUILDS.with(std::cell::Cell::get);
    let (before, start) = (builds(), cache::tier_stats(CoefficientCacheTier::Cgc));
    crate::audit_alloc::reset_peak_to_live();
    let start_live = crate::audit_alloc::snapshot().0;
    let t = Instant::now();
    run();
    let elapsed_ns = t.elapsed().as_nanos();
    let (end_live, peak_live) = crate::audit_alloc::snapshot();
    let (after, s) = (builds(), cache::tier_stats(CoefficientCacheTier::Cgc));
    let limit = cache::cache_budgets().limit(CoefficientCacheTier::Cgc);
    assert!(s.bytes <= limit, "{label}: {} > {limit}", s.bytes);
    eprintln!(
        "CGC_PRESSURE {{\"query\":\"{label}\",\"elapsed_ns\":{elapsed_ns},\"module_builds\":{},\"channel_builds\":{},\"entries\":{},\"bytes\":{},\"bytes_delta\":{},\"evictions\":{},\"evictions_delta\":{},\"start_live_bytes\":{start_live},\"peak_live_bytes\":{peak_live},\"peak_live_delta_bytes\":{},\"end_live_bytes\":{end_live},\"query_end_rss_bytes\":{}}}",
        after[0] - before[0],
        after[1] - before[1],
        s.entries,
        s.bytes,
        s.bytes as i64 - start.bytes as i64,
        s.evictions,
        s.evictions - start.evictions,
        peak_live.saturating_sub(start_live),
        rss_bytes().map_or_else(|| "null".to_owned(), |n| n.to_string()),
    );
}

/// Real module, channel and complex-phase loads across root systems.
fn cartan_workload(pass: &str) {
    use crate::category::{FusionCategory, GroupRepresentations};
    use crate::lie::{GroupId, LieGroup};
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
    cartan_query(&format!("{pass} G2 27x27 all"), || {
        all(&g2, &[2, 0], &[2, 0])
    });
    cartan_query(&format!("{pass} SU4 15x15 all"), || {
        all(&su4, &[1, 0, 1], &[1, 0, 1])
    });
    cartan_query(&format!("{pass} Sp4 35x35 all"), || {
        all(&sp4, &[2, 1], &[2, 1])
    });
    cartan_query(&format!("{pass} F4 26x26 all"), || {
        all(&f4, &[0, 0, 0, 1], &[0, 0, 0, 1])
    });
    cartan_query(&format!("{pass} E6 phase 27"), || {
        phase(&e6, &[1, 0, 0, 0, 0, 0])
    });
    cartan_query(&format!("{pass} E6 phase 351"), || {
        phase(&e6, &[0, 0, 0, 0, 1, 0])
    });
    cartan_query(&format!("{pass} D5 phase [1,0,0,1,0]"), || {
        phase(&d5, &[1, 0, 0, 1, 0])
    });
    cartan_query(&format!("{pass} G2 27x27 all again"), || {
        all(&g2, &[2, 0], &[2, 0])
    });
}

/// The CGC tier under an 8 MiB budget: modules, channels and phases compete,
/// evict and rebuild; trim and reset clear every kind.
#[test]
#[ignore = "manual release-only CGC-tier pressure measurement; run alone"]
fn coefficient_cache_budget_pressure_cgc() {
    let _guard = crate::cache::TEST_CACHE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let budgets = CoefficientCacheBudgets::default().with_limit(CoefficientCacheTier::Cgc, 8 << 20);
    cache::configure_cache_budgets(budgets).unwrap();
    cache::reset();
    cartan_workload("cold");
    cartan_workload("warm");
    assert!(cache::tier_stats(CoefficientCacheTier::Cgc).evictions > 0);
    let report = cache::trim_to(CoefficientCacheTier::Cgc, 0);
    eprintln!("CGC_PRESSURE_TRIM {report:?}");
    assert_eq!(cache::tier_stats(CoefficientCacheTier::Cgc).entries, 0);
    cartan_workload("after-trim");
    cache::reset();
    for &tier in CoefficientCacheTier::ALL {
        assert_eq!(cache::tier_stats(tier), crate::cache::TierStats::default());
    }
    eprintln!("CGC_PRESSURE_RESET all tiers zero");
}

/// D4 `350 ⊗ 350` (`[1,0,1,1]`, 28 channels, N ≤ 7) under the default
/// budget, query by query: retained charge stays within the cap while the
/// working set (RSS) is reported separately.
#[test]
#[ignore = "manual release-only D4 working-set study; run alone, bounded"]
fn cgc_d4_350_working_set() {
    use crate::category::{FusionCategory, GroupRepresentations};
    use crate::lie::{GroupId, LieGroup};
    let _guard = crate::cache::TEST_CACHE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
