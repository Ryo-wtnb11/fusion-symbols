pub mod a1;
#[cfg(feature = "lie-generated")]
pub mod bcd;
pub mod counting_alloc;
pub mod lie;

use fusion_symbols::cache::{tier_stats, CoefficientCacheTier as Tier, TierStats};

/// Only global-state measurements/lifecycle checks serialize; value tests stay parallel.
#[allow(dead_code)]
pub fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SERIAL.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// The base SU(2) tiers, whose charge `BASE_CACHE_MAX_BYTES` bounds.
#[allow(dead_code)]
pub const BASE_TIERS: [Tier; 3] = [Tier::ThreeJ, Tier::SixJ, Tier::DerivedF];

/// Every generated tier compiled into this build.
#[allow(dead_code)]
pub fn generated_tiers() -> impl Iterator<Item = &'static Tier> {
    Tier::ALL.iter().filter(|tier| !BASE_TIERS.contains(tier))
}

/// Field-wise sum of independent per-tier snapshots.
#[allow(dead_code)]
pub fn stats_sum<'a>(tiers: impl IntoIterator<Item = &'a Tier>) -> TierStats {
    tiers
        .into_iter()
        .fold(TierStats::default(), |mut sum, &tier| {
            let s = tier_stats(tier);
            sum.entries += s.entries;
            sum.bytes += s.bytes;
            sum.hits += s.hits;
            sum.misses += s.misses;
            sum.evictions += s.evictions;
            sum
        })
}
