pub mod a1;
#[cfg(feature = "lie-generated")]
pub mod bcd;
pub mod counting_alloc;
pub mod lie;

use fusion_symbols::cache::{tier_stats, CoefficientCacheTier as Tier, TierStats};
use fusion_symbols::category::FusionCategory;

/// Every object of a finite category in its frozen enumeration order.
#[allow(dead_code)]
pub fn objects<C: FusionCategory>(cat: &C) -> impl Iterator<Item = C::Object> + '_ {
    let count = cat.object_count().unwrap().expect("finite category");
    (0..count).map(|i| cat.object_at(i).unwrap())
}

/// Only global-state measurements/lifecycle checks serialize; value tests stay parallel.
#[allow(dead_code)]
pub fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SERIAL.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// The tiers a build without `lie-generated` fills, whose charge
/// `BASE_CACHE_MAX_BYTES` bounds.
#[allow(dead_code)]
pub const BASE_TIERS: [Tier; 4] = [Tier::Wigner3j, Tier::Wigner6j, Tier::FSymbol, Tier::Cgc];

/// The tiers only higher-rank groups fill.
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
