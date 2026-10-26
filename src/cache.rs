//! Bounded, thread-safe coefficient caches, and the process-local policy that
//! bounds them.
//!
//! Coefficient generation is the expensive step, so `fusion-symbols` caches: the base
//! SU(2) tiers (3j, 6j, derived-F) keyed by canonical Regge classes, and — with
//! `lie-generated` — five generated tiers (SU(N) product / CGC / F, B/C/D product+CGC / F).
//! The SU(N) product and B/C/D product+CGC tiers also retain `LieGroup` dual lookups.
//! Every tier is process-global, byte-weighted Clock-PRO, and independently bounded.
//!
//! # What a consumer needs from this module
//!
//! - **A ceiling.** [`BASE_CACHE_MAX_BYTES`] (192 MiB) and
//!   [`GENERATED_CACHE_MAX_BYTES`] (640 MiB + 128 KiB) are the documented sums
//!   of the per-tier retained-entry charge caps. They bound *cache-owned entry
//!   charge*, not RSS.
//! - **A smaller ceiling, if you want one.** [`configure_cache_budgets`],
//!   once, before first use. Shrink-only.
//! - **Observation.** [`base_cache_stats`] / [`generated_cache_stats`].
//! - **Release.** [`trim_to`] (one tier, unspecified removal order) and [`reset`] (all
//!   tiers). Both act on process-global state, so **a library must not call
//!   them** — exactly one component in a process owns that policy.
//!
//! ```
//! use fusion_symbols::cache::{base_cache_stats, BASE_CACHE_MAX_BYTES};
//!
//! let _ = fusion_symbols::lie::su2::wigner_6j(2, 2, 2, 2, 2, 2);
//! let stats = base_cache_stats();
//! assert!(stats.total().bytes <= BASE_CACHE_MAX_BYTES);
//! ```
//!
//! The task-oriented version is in the [User Guide].
//!
//! [User Guide]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/user-guide/resources.md
//!
//! # Why a cache at all
//!
//! In the tensor-network consumption pattern the same small labels recur many
//! thousands of times, so a warm hit should cost a hash lookup rather than a
//! fresh big-rational Racah sum. WignerSymbols.jl v2.0.0 caches transparently
//! inside `wigner3j`/`wigner6j` (per-kind `LRU` dicts keyed by canonical Regge
//! labels); this crate follows that model but bounds the cache by policy — Julia's
//! caches are effectively unbounded in entries.
//!
//! # Why no gauge/version key component (in-process)
//!
//! A canonical Regge class names exactly one exact symbol value. The stored
//! [`SignedSqrtRational`] is that exact value — not a gauge- or
//! algorithm-dependent float — so within one process the canonical key is a
//! complete key: no gauge tag and no algorithm-version tag can change which
//! value a class maps to. (Contrast the `lie-generated` coefficient caches, whose
//! floating values *are* gauge- and algorithm-dependent and are versioned.)
//!
//! # Why this cache must never be persisted to disk
//!
//! Persisting these entries across builds would reintroduce exactly the
//! versioning problem the in-process argument avoids: a future change to the
//! exact engine (a different but still-correct series arrangement, a widened
//! type, a bug fix) could alter the stored bytes for a class, and a persisted
//! store would then need an algorithm-version key to stay sound. Keeping the
//! cache process-local sidesteps that entirely — it is rebuilt from the engine
//! every run, so it can never disagree with the engine that filled it.
//!
//! # Base cache resource contract (static partition)
//!
//! The three base SU(2) tiers (3j, 6j, derived-F) are each bounded
//! independently by a per-tier entry and retained-charge cap; the documented
//! aggregate cap [`BASE_CACHE_MAX_BYTES`] is simply their sum. This is a **static
//! partition, not a dynamic shared pool**: a shared budget would couple
//! eviction across tiers whose entries differ wildly in size (big-rational
//! exact symbols vs `f64` scalars) and whose hit patterns are unrelated, for no
//! measured benefit — so it is deliberately rejected here and revisited only
//! with measurements. Because each per-tier charged-entry cap is enforced, the
//! aggregate charged-entry bound
//! holds as a corollary rather than needing global enforcement. The charge
//! covers entries currently owned by the cache. It excludes quick_cache retained capacity and scaffolding, allocator metadata and RSS,
//! transient or external clones, and values returned through public APIs.
//! Per-tier and total statistics are exposed via [`base_cache_stats`]; reset
//! ownership is on [`reset`].

use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, OnceLock};

use crate::exact::SignedSqrtRational;
use crate::lie::su2::{FKey, Regge3j, Regge6j};

/// Default entry cap per kind (3j and 6j each). Matches the reference order of
/// magnitude (WignerSymbols.jl uses `10^6`); the retained-charge cap is the real
/// backstop.
const DEFAULT_MAX_ENTRIES: usize = 1 << 20;

/// Default retained-charge cap per kind. At the ~O(1)-limb sizes typical of
/// small-label TN work an entry charges well under a kilobyte, so 64 MiB holds
/// a large working set while bounding the conservative charge of cache-owned
/// entries, not allocator-live memory or RSS.
const DEFAULT_MAX_BYTES: usize = 64 << 20;
/// Strong and weak reference counts heading every `Arc` allocation; charged
/// with each shared label or product payload so retained bytes stay conservative.
#[cfg(feature = "lie-generated")]
pub(crate) const ARC_COUNTS_BYTES: usize = 2 * std::mem::size_of::<usize>();

/// Process-local retained-charge limits for coefficient-cache tiers.
///
/// [`configure_cache_budgets`] accepts this value once, before the first cache
/// operation or cache-policy observation. Zero keeps computing values but
/// retains none for that tier. [`Default`] is the compiled policy and also the
/// maximum accepted policy; budgets can only shrink it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoefficientCacheBudgets {
    /// Exact Wigner 3j retained-charge cap.
    three_j_bytes: usize,
    /// Exact Wigner 6j retained-charge cap.
    six_j_bytes: usize,
    /// Derived SU(2) F-symbol retained-charge cap.
    derived_f_bytes: usize,
    #[cfg(feature = "lie-generated")]
    /// SU(N) product and dual-lookup retained-charge cap.
    sun_product_bytes: usize,
    #[cfg(feature = "lie-generated")]
    /// SU(N) CGC retained-charge cap.
    sun_cgc_bytes: usize,
    #[cfg(feature = "lie-generated")]
    /// SU(N) F-symbol retained-charge cap.
    sun_f_bytes: usize,
    #[cfg(feature = "lie-generated")]
    /// Shared B/C/D exact-product, CGC and Lie dual-lookup retained-charge cap.
    bcd_cgc_bytes: usize,
    #[cfg(feature = "lie-generated")]
    /// B/C/D F-symbol retained-charge cap.
    bcd_f_bytes: usize,
}

/// One existing coefficient-cache tier.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoefficientCacheTier {
    /// Exact Wigner 3j tier.
    ThreeJ,
    /// Exact Wigner 6j tier.
    SixJ,
    /// Derived SU(2) F-symbol tier.
    DerivedF,
    #[cfg(feature = "lie-generated")]
    /// SU(N) product and dual-lookup tier.
    SunProduct,
    #[cfg(feature = "lie-generated")]
    /// SU(N) CGC tier.
    SunCgc,
    #[cfg(feature = "lie-generated")]
    /// SU(N) F-symbol tier.
    SunF,
    #[cfg(feature = "lie-generated")]
    /// B/C/D exact-product and CGC tier, also retaining Lie dual lookups.
    BcdCgc,
    #[cfg(feature = "lie-generated")]
    /// B/C/D F-symbol tier.
    BcdF,
}

impl std::fmt::Display for CoefficientCacheTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ThreeJ => "three_j",
            Self::SixJ => "six_j",
            Self::DerivedF => "derived_f",
            #[cfg(feature = "lie-generated")]
            Self::SunProduct => "sun_product",
            #[cfg(feature = "lie-generated")]
            Self::SunCgc => "sun_cgc",
            #[cfg(feature = "lie-generated")]
            Self::SunF => "sun_f",
            #[cfg(feature = "lie-generated")]
            Self::BcdCgc => "bcd_cgc",
            #[cfg(feature = "lie-generated")]
            Self::BcdF => "bcd_f",
        })
    }
}

/// Snapshot returned by [`trim_to`] for one coefficient-cache tier.
///
/// The report is taken while the selected tier's publication lock is held: its
/// `removed_*` and `remaining_*` fields describe that trim's linearization
/// point. Concurrent lookups or an in-flight miss that computed outside the
/// lock can change occupancy after [`trim_to`] returns. The charged bytes cover
/// cache-owned entries only; they do not imply release of external `Arc`s,
/// container capacity, allocator metadata, workspaces, or RSS.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheTrimReport {
    /// Tier trimmed by this operation.
    pub tier: CoefficientCacheTier,
    /// Number of entries removed (unspecified order).
    pub removed_entries: usize,
    /// Conservative charged bytes removed with those entries.
    pub removed_charged_bytes: usize,
    /// Entries retained at the trim linearization point.
    pub remaining_entries: usize,
    /// Conservative charged bytes retained at the trim linearization point.
    pub remaining_charged_bytes: usize,
}

impl CoefficientCacheBudgets {
    /// Disable retention in every compiled tier while preserving evaluation.
    pub fn disabled() -> Self {
        let mut budgets = Self::default();
        for tier in [
            CoefficientCacheTier::ThreeJ,
            CoefficientCacheTier::SixJ,
            CoefficientCacheTier::DerivedF,
        ] {
            budgets = budgets.with_limit(tier, 0);
        }
        #[cfg(feature = "lie-generated")]
        for tier in [
            CoefficientCacheTier::SunProduct,
            CoefficientCacheTier::SunCgc,
            CoefficientCacheTier::SunF,
            CoefficientCacheTier::BcdCgc,
            CoefficientCacheTier::BcdF,
        ] {
            budgets = budgets.with_limit(tier, 0);
        }
        budgets
    }

    /// Return this policy with one tier's retained-charge cap replaced.
    pub fn with_limit(mut self, tier: CoefficientCacheTier, bytes: usize) -> Self {
        match tier {
            CoefficientCacheTier::ThreeJ => self.three_j_bytes = bytes,
            CoefficientCacheTier::SixJ => self.six_j_bytes = bytes,
            CoefficientCacheTier::DerivedF => self.derived_f_bytes = bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunProduct => self.sun_product_bytes = bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunCgc => self.sun_cgc_bytes = bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunF => self.sun_f_bytes = bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdCgc => self.bcd_cgc_bytes = bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdF => self.bcd_f_bytes = bytes,
        }
        self
    }

    /// Return one tier's retained-charge cap.
    pub fn limit(&self, tier: CoefficientCacheTier) -> usize {
        match tier {
            CoefficientCacheTier::ThreeJ => self.three_j_bytes,
            CoefficientCacheTier::SixJ => self.six_j_bytes,
            CoefficientCacheTier::DerivedF => self.derived_f_bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunProduct => self.sun_product_bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunCgc => self.sun_cgc_bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunF => self.sun_f_bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdCgc => self.bcd_cgc_bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdF => self.bcd_f_bytes,
        }
    }
}

impl Default for CoefficientCacheBudgets {
    fn default() -> Self {
        Self {
            three_j_bytes: DEFAULT_MAX_BYTES,
            six_j_bytes: DEFAULT_MAX_BYTES,
            derived_f_bytes: DEFAULT_MAX_BYTES,
            #[cfg(feature = "lie-generated")]
            sun_product_bytes: sun_product_cache::SUN_PRODUCT_MAX_BYTES,
            #[cfg(feature = "lie-generated")]
            sun_cgc_bytes: cgc_cache::CGC_MAX_BYTES,
            #[cfg(feature = "lie-generated")]
            sun_f_bytes: sun_f_cache::SUN_F_MAX_BYTES,
            #[cfg(feature = "lie-generated")]
            bcd_cgc_bytes: bcd_cgc_cache::BCD_CGC_MAX_BYTES,
            #[cfg(feature = "lie-generated")]
            bcd_f_bytes: bcd_f_cache::BCD_F_MAX_BYTES,
        }
    }
}

/// Failed coefficient-cache policy initialization.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheBudgetError {
    /// A cache operation or policy observation already fixed this process's policy.
    AlreadyInitialized,
    /// A requested tier limit exceeds its compiled maximum.
    ExceedsMaximum {
        /// Stable tier name for diagnostics.
        tier: CoefficientCacheTier,
        /// Rejected requested cap.
        requested: usize,
        /// Compiled maximum cap.
        maximum: usize,
    },
    /// Summing validated tier limits overflowed `usize`.
    AggregateOverflow,
}

impl std::fmt::Display for CacheBudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyInitialized => {
                f.write_str("coefficient-cache policy is already initialized")
            }
            Self::ExceedsMaximum {
                tier,
                requested,
                maximum,
            } => write!(
                f,
                "{tier} budget {requested} exceeds compiled maximum {maximum}"
            ),
            Self::AggregateOverflow => {
                f.write_str("coefficient-cache budget aggregate overflowed usize")
            }
        }
    }
}
impl std::error::Error for CacheBudgetError {}

static CACHE_BUDGETS: OnceLock<CoefficientCacheBudgets> = OnceLock::new();

fn effective_budgets() -> &'static CoefficientCacheBudgets {
    effective_budgets_in(&CACHE_BUDGETS)
}

fn effective_budgets_in(cell: &OnceLock<CoefficientCacheBudgets>) -> &CoefficientCacheBudgets {
    cell.get_or_init(CoefficientCacheBudgets::default)
}

/// Configure one shrink-only coefficient-cache policy for this process.
///
/// Call this **once, before any coefficient call or policy observation**. The
/// compiled defaults are also the maximum accepted caps, so a policy can only
/// shrink retention, never grow it. A zero cap evaluates normally but retains
/// nothing; [`CoefficientCacheBudgets::disabled`] zeroes every compiled tier.
/// [`reset`] clears entries and counters without changing this policy. There
/// are no presets, environment variables, or runtime reconfiguration.
///
/// # Errors
///
/// - [`CacheBudgetError::AlreadyInitialized`] if the policy was already fixed —
///   by an earlier call, or by first use of any cache, which installs the
///   defaults.
/// - [`CacheBudgetError::ExceedsMaximum`] if any requested cap exceeds the
///   compiled maximum for that tier.
/// - [`CacheBudgetError::AggregateOverflow`] if the requested caps do not sum
///   within `usize`.
///
/// ```
/// use fusion_symbols::cache::{
///     cache_budgets, configure_cache_budgets, CoefficientCacheBudgets,
///     CoefficientCacheTier,
/// };
///
/// let budgets = CoefficientCacheBudgets::default()
///     .with_limit(CoefficientCacheTier::SixJ, 1 << 20); // 1 MiB of 6j
/// configure_cache_budgets(budgets).unwrap();
/// assert_eq!(cache_budgets().limit(CoefficientCacheTier::SixJ), 1 << 20);
///
/// // One-shot: a second call is rejected.
/// assert!(configure_cache_budgets(budgets).is_err());
/// ```
pub fn configure_cache_budgets(budgets: CoefficientCacheBudgets) -> Result<(), CacheBudgetError> {
    validate_budgets(budgets)?;
    configure_budgets_in(&CACHE_BUDGETS, budgets)
}

fn configure_budgets_in(
    cell: &OnceLock<CoefficientCacheBudgets>,
    budgets: CoefficientCacheBudgets,
) -> Result<(), CacheBudgetError> {
    cell.set(budgets)
        .map_err(|_| CacheBudgetError::AlreadyInitialized)
}

/// Return the effective policy, fixing the default if it was not configured.
pub fn cache_budgets() -> CoefficientCacheBudgets {
    *effective_budgets()
}

fn validate_budgets(b: CoefficientCacheBudgets) -> Result<(), CacheBudgetError> {
    let maximum = CoefficientCacheBudgets::default();
    let fields = [
        (
            CoefficientCacheTier::ThreeJ,
            b.three_j_bytes,
            maximum.three_j_bytes,
        ),
        (
            CoefficientCacheTier::SixJ,
            b.six_j_bytes,
            maximum.six_j_bytes,
        ),
        (
            CoefficientCacheTier::DerivedF,
            b.derived_f_bytes,
            maximum.derived_f_bytes,
        ),
        #[cfg(feature = "lie-generated")]
        (
            CoefficientCacheTier::SunProduct,
            b.sun_product_bytes,
            maximum.sun_product_bytes,
        ),
        #[cfg(feature = "lie-generated")]
        (
            CoefficientCacheTier::SunCgc,
            b.sun_cgc_bytes,
            maximum.sun_cgc_bytes,
        ),
        #[cfg(feature = "lie-generated")]
        (
            CoefficientCacheTier::SunF,
            b.sun_f_bytes,
            maximum.sun_f_bytes,
        ),
        #[cfg(feature = "lie-generated")]
        (
            CoefficientCacheTier::BcdCgc,
            b.bcd_cgc_bytes,
            maximum.bcd_cgc_bytes,
        ),
        #[cfg(feature = "lie-generated")]
        (
            CoefficientCacheTier::BcdF,
            b.bcd_f_bytes,
            maximum.bcd_f_bytes,
        ),
    ];
    let mut total = 0usize;
    for (tier, requested, limit) in fields {
        if requested > limit {
            return Err(CacheBudgetError::ExceedsMaximum {
                tier,
                requested,
                maximum: limit,
            });
        }
        total = total
            .checked_add(requested)
            .ok_or(CacheBudgetError::AggregateOverflow)?;
    }
    Ok(())
}

/// Aggregate conservative retained-charge cap for the three base SU(2) tiers
/// (3j, 6j, derived-F): `192 MiB` = `3 × 64 MiB`.
///
/// This is a **documented static partition**, not a shared budget: each tier is
/// bounded independently by its own per-tier charged-entry cap
/// (`DEFAULT_MAX_BYTES`). The aggregate is therefore a provable corollary —
/// `Σ tier bytes ≤ Σ tier caps = BASE_CACHE_MAX_BYTES` — rather than an
/// enforced global limit. A dynamic shared pool (tiers competing for one
/// budget) is deliberately rejected: it would couple eviction across tiers with
/// very different entry sizes (big-rational vs `f64`) and hit patterns for no
/// measured benefit.
///
/// The `const` assertion below ties this constant to the per-tier cap so the
/// two cannot silently drift; all three base tiers (`CACHE_3J`, `CACHE_6J`,
/// `CACHE_F`) are constructed with the same `DEFAULT_MAX_BYTES`.
pub const BASE_CACHE_MAX_BYTES: usize = 192 << 20;

// Compile-time tie: if the per-tier retained-charge cap changes, BASE_CACHE_MAX_BYTES must
// be reconciled in the same edit or the crate stops building. (There is no
// compile-time way to read the tiers' runtime `max_bytes`; anchoring to the
// shared `DEFAULT_MAX_BYTES` they are all built from is the enforceable tie.)
const _: () = assert!(BASE_CACHE_MAX_BYTES == 3 * DEFAULT_MAX_BYTES);

/// Aggregate conservative retained-charge cap for the five generated `lie-generated` tiers
/// (SU(N) product, SU(N) CGC, SU(N) F, B/C/D product+CGC, B/C/D F):
/// `640 MiB + 128 KiB` = `128 KiB + 256 MiB + 64 MiB + 256 MiB + 64 MiB`.
///
/// **Unstable: shape may change.** There is no Cargo-feature way to express an
/// instability tier, so this doc label is the marker.
///
/// # Two-layer aggregate story (why there is no single crate-wide constant)
///
/// Retained coefficient-cache entry charge is documented in two layers, not one
/// number:
///
/// - the base SU(2) tiers are bounded by [`BASE_CACHE_MAX_BYTES`] (a static
///   partition with a const-proved sum — see its docs);
/// - the generated tiers are bounded by this constant.
///
/// The whole-process coefficient-cache charged-entry cap is the **documented sum**
/// `BASE_CACHE_MAX_BYTES + GENERATED_CACHE_MAX_BYTES`. There is deliberately no
/// single cross-feature constant spanning both: this constant only exists under
/// `lie-generated`, so a "one number" whole-crate cap would *change value with the
/// feature flag* and read as if the base cap shrank when `lie-generated` is off —
/// misleading. Two feature-honest layers instead. Like the base cap this is a **static partition, not a shared pool**:
/// `Σ tier charged bytes ≤ Σ tier caps = GENERATED_CACHE_MAX_BYTES` holds as a
/// corollary without global enforcement.
///
/// Per-tier and total statistics are exposed via [`generated_cache_stats`]. The
/// the internal generator catalog is *not* a value cache (generator state, its own byte
/// budget, per-`LieGroup` lifecycle) and is intentionally excluded from
/// this budget and from these stats.
#[cfg(feature = "lie-generated")]
pub const GENERATED_CACHE_MAX_BYTES: usize = (640 << 20) + (128 << 10);

// Compile-time tie: if any generated-tier retained-charge cap changes, this constant must
// be reconciled in the same edit or the crate stops building (the same drift
// guard the base tiers use). Each cap is `pub(super)` in its tier module.
#[cfg(feature = "lie-generated")]
const _: () = assert!(
    GENERATED_CACHE_MAX_BYTES
        == cgc_cache::CGC_MAX_BYTES
            + sun_product_cache::SUN_PRODUCT_MAX_BYTES
            + sun_f_cache::SUN_F_MAX_BYTES
            + bcd_cgc_cache::BCD_CGC_MAX_BYTES
            + bcd_f_cache::BCD_F_MAX_BYTES
);

/// Snapshot of the aggregate cache counters (3j and 6j kinds summed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups served from a retained value or a shared in-flight result.
    pub hits: u64,
    /// Successful computations/publications. Concurrent waiters count as hits;
    /// errors and attempts to publish an already-retained key add no miss.
    pub misses: u64,
    /// Entries currently retained across both kinds.
    pub entries: usize,
    /// Conservatively charged bytes currently retained across both kinds.
    pub bytes: usize,
}

/// Per-tier snapshot of one base SU(2) coefficient cache (3j, 6j, or derived-F).
///
/// The fields are consistent for the tier they describe (entries/bytes read
/// under the tier lock). See [`base_cache_stats`] and [`BaseCacheStats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TierStats {
    /// Entries currently retained in this tier.
    pub entries: usize,
    /// Conservatively charged bytes currently retained in this tier.
    pub bytes: usize,
    /// Lookups served from a retained value or a shared in-flight result.
    pub hits: u64,
    /// Successful computations/publications in this tier. Concurrent callers
    /// waiting for the same computation count as hits; errors are not cached.
    pub misses: u64,
    /// Entries rejected from retention or removed from this tier by eviction
    /// over its lifetime. An entry larger than the retained-charge cap and an
    /// entry in a zero-budget tier both count as rejected evictions.
    pub evictions: u64,
}

/// Per-tier statistics for the three base SU(2) coefficient tiers.
///
/// Covers **only** the 3j, 6j, and derived-F tiers by definition — the base
/// SU(2) provider surface. (This is distinct from the aggregate [`stats`], which
/// under the `lie-generated` feature also sums the generated SU(N)/B/C/D tiers.)
///
/// # Snapshot consistency
///
/// Each per-tier [`TierStats`] is internally consistent (taken under that tier's
/// publication lock). [`total`](BaseCacheStats::total) is a field-wise sum of the three
/// per-tier snapshots, **not** a single global atomic snapshot: a concurrent
/// filler can interleave between the tier reads, so the total is only
/// eventually consistent. This crate does not take a global lock spanning the tiers —
/// that would serialize otherwise-independent lookups for no correctness gain
/// (the individual tier charged-entry caps already hold independently).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BaseCacheStats {
    /// The exact 3j tier.
    pub three_j: TierStats,
    /// The exact 6j tier.
    pub six_j: TierStats,
    /// The derived-f64 F-symbol tier.
    pub derived_f: TierStats,
}

impl BaseCacheStats {
    /// Field-wise sum of the three base tiers. See the type-level snapshot-
    /// consistency note: this is a sum of per-tier snapshots, not an atomic
    /// whole-cache snapshot.
    pub fn total(&self) -> TierStats {
        TierStats {
            entries: self.three_j.entries + self.six_j.entries + self.derived_f.entries,
            bytes: self.three_j.bytes + self.six_j.bytes + self.derived_f.bytes,
            hits: self.three_j.hits + self.six_j.hits + self.derived_f.hits,
            misses: self.three_j.misses + self.six_j.misses + self.derived_f.misses,
            evictions: self.three_j.evictions + self.six_j.evictions + self.derived_f.evictions,
        }
    }
}

/// Per-tier statistics for the five generated `lie-generated` tiers (SU(N) product,
/// SU(N) CGC, SU(N) F, B/C/D product+CGC, B/C/D F).
///
/// **Unstable: shape may change.** The struct is `#[non_exhaustive]` — it is
/// constructed only inside the crate (by [`generated_cache_stats`]); consumers
/// read its fields or call [`total`](GeneratedCacheStats::total).
///
/// Reuses the base [`TierStats`] type (no new vocabulary). This covers **only**
/// the generated SU(N)/B/C/D tiers — the base SU(2) surface is
/// [`base_cache_stats`], and the aggregate [`stats`] sums both. Conservative
/// retained entry charge is capped by [`GENERATED_CACHE_MAX_BYTES`]
/// (`total().bytes ≤ GENERATED_CACHE_MAX_BYTES`).
///
/// # Snapshot consistency
///
/// Each per-tier [`TierStats`] is internally consistent (taken under that tier's
/// publication lock). [`total`](GeneratedCacheStats::total) is a field-wise sum of the
/// five per-tier snapshots, **not** a single global atomic snapshot: a
/// concurrent filler can interleave between the tier reads, so the total is only
/// eventually consistent. This crate does not take a global lock spanning the tiers —
/// that would serialize otherwise-independent lookups for no correctness gain
/// (the individual tier charged-entry caps hold independently). Same contract as
/// [`BaseCacheStats`].
#[cfg(feature = "lie-generated")]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GeneratedCacheStats {
    /// The exact SU(N) tensor-product decomposition tier.
    pub sun_product: TierStats,
    /// The SU(N) CGC tier.
    pub sun_cgc: TierStats,
    /// The derived-f64 SU(N) F-symbol tier.
    pub sun_f: TierStats,
    /// The B/C/D exact-product and CGC value tier.
    pub bcd_cgc: TierStats,
    /// The derived-f64 B/C/D F-symbol tier.
    pub bcd_f: TierStats,
}

#[cfg(feature = "lie-generated")]
impl GeneratedCacheStats {
    /// Field-wise sum of the five generated tiers. See the type-level snapshot-
    /// consistency note: this is a sum of per-tier snapshots, not an atomic
    /// whole-cache snapshot.
    pub fn total(&self) -> TierStats {
        TierStats {
            entries: self.sun_product.entries
                + self.sun_cgc.entries
                + self.sun_f.entries
                + self.bcd_cgc.entries
                + self.bcd_f.entries,
            bytes: self.sun_product.bytes
                + self.sun_cgc.bytes
                + self.sun_f.bytes
                + self.bcd_cgc.bytes
                + self.bcd_f.bytes,
            hits: self.sun_product.hits
                + self.sun_cgc.hits
                + self.sun_f.hits
                + self.bcd_cgc.hits
                + self.bcd_f.hits,
            misses: self.sun_product.misses
                + self.sun_cgc.misses
                + self.sun_f.misses
                + self.bcd_cgc.misses
                + self.bcd_f.misses,
            evictions: self.sun_product.evictions
                + self.sun_cgc.evictions
                + self.sun_f.evictions
                + self.bcd_cgc.evictions
                + self.bcd_f.evictions,
        }
    }
}

/// Conservative retained-byte charge for a stored value, implemented per value
/// type as one component of the cache-owned entry charge.
///
/// The exact tier stores a [`SignedSqrtRational`] whose size is data-dependent
/// (big-integer limbs), so it must measure itself; the derived-f64 tier stores
/// a fixed-size scalar. Keys use the sibling [`CacheKeyCharge`] contract so
/// heap-backed generated labels cannot escape the same charged-entry cap.
pub(crate) trait CacheCharge {
    /// Bytes charged for one stored value (over-counts, never under-counts).
    fn value_bytes(&self) -> usize;
}

/// Conservative retained-byte charge for one stored key.
///
/// An entry owns one key; heap-backed labels report its shell and backing.
pub(crate) trait CacheKeyCharge {
    /// Bytes retained by one key clone (over-counts, never under-counts).
    fn key_bytes(&self) -> usize;
}

macro_rules! fixed_key_charge {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CacheKeyCharge for $ty {
                fn key_bytes(&self) -> usize {
                    std::mem::size_of::<Self>()
                }
            }
        )+
    };
}

fixed_key_charge!(u32, Regge3j, Regge6j, FKey);

#[cfg(feature = "lie-generated")]
impl<I: CacheKeyCharge> CacheKeyCharge for (I, I, I) {
    fn key_bytes(&self) -> usize {
        self.0
            .key_bytes()
            .saturating_add(self.1.key_bytes())
            .saturating_add(self.2.key_bytes())
    }
}

#[cfg(feature = "lie-generated")]
impl<I: CacheKeyCharge> CacheKeyCharge for (I, I, I, I, I, I) {
    fn key_bytes(&self) -> usize {
        self.0
            .key_bytes()
            .saturating_add(self.1.key_bytes())
            .saturating_add(self.2.key_bytes())
            .saturating_add(self.3.key_bytes())
            .saturating_add(self.4.key_bytes())
            .saturating_add(self.5.key_bytes())
    }
}

impl CacheCharge for SignedSqrtRational {
    fn value_bytes(&self) -> usize {
        let r = self.radicand();
        let value_limbs = (r.numer().bits() + r.denom().bits()).div_ceil(8) as usize;
        // Conservatively charge both big integers, the shared rational and
        // the SignedSqrtRational shell (including inline/small-limb cases).
        const BIGINT_OVERHEAD: usize = 32;
        std::mem::size_of::<SignedSqrtRational>()
            + 2 * std::mem::size_of::<usize>()
            + std::mem::size_of_val(r)
            + 2 * BIGINT_OVERHEAD
            + value_limbs
    }
}

impl CacheCharge for f64 {
    fn value_bytes(&self) -> usize {
        std::mem::size_of::<f64>()
    }
}

/// Conservative retained-byte charge for one stored entry keyed by `K`.
///
/// Counts the value, one owned key and the cached byte-charge word.
/// This is the conservative retained
/// charge of entries currently owned by the cache; it excludes container
/// retained capacity/scaffolding, allocator metadata/RSS, transient or external
/// clones, and returned public values.
fn entry_charge<K: CacheKeyCharge, V: CacheCharge>(key: &K, value: &V) -> usize {
    value
        .value_bytes()
        .saturating_add(key.key_bytes())
        .saturating_add(std::mem::size_of::<usize>())
}

#[derive(Clone)]
struct Charged<V> {
    value: V,
    bytes: usize,
}

#[derive(Clone)]
struct ByteWeighter {
    minimum: usize,
}

impl<K, V> quick_cache::Weighter<K, Charged<V>> for ByteWeighter {
    fn weight(&self, _: &K, value: &Charged<V>) -> u64 {
        value.bytes.max(self.minimum) as u64
    }
}

#[derive(Clone)]
struct Evictions(std::sync::Arc<AtomicU64>, std::sync::Arc<AtomicUsize>);

impl<K, V> quick_cache::Lifecycle<K, Charged<V>> for Evictions {
    type RequestState = ();

    fn on_evict(&self, _: &mut (), _: K, value: Charged<V>) {
        self.0.fetch_add(1, Ordering::Relaxed);
        self.1.fetch_sub(value.bytes, Ordering::Relaxed);
    }
}

/// Hit-counter stripes per tier: twice quick_cache's eight `sharded-lock`
/// reader stripes, so round-robin thread indices keep 16 concurrent readers on
/// distinct counter lines while a snapshot sums only 16 lines.
const HIT_STRIPES: usize = 16;

#[repr(align(128))]
#[derive(Default)]
struct PaddedCounter(AtomicU64);

/// This thread's hit stripe, assigned round-robin on first use.
fn hit_stripe() -> usize {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        static STRIPE: usize = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    // Reduced here rather than at assignment so the index bound is visible at
    // the array access and needs no bounds check.
    STRIPE.try_with(|stripe| *stripe).unwrap_or(0) % HIT_STRIPES
}

type Guard<'a, K, V> = quick_cache::sync::PlaceholderGuard<
    'a,
    K,
    Charged<V>,
    ByteWeighter,
    quick_cache::DefaultHashBuilder,
    Evictions,
>;

type Store<K, V> = quick_cache::sync::Cache<
    K,
    Charged<V>,
    ByteWeighter,
    quick_cache::DefaultHashBuilder,
    Evictions,
>;

/// Byte-weighted Clock-PRO cache. Hits take one quick_cache reader stripe and
/// one hit-counter stripe. Publication, explicit removal and snapshots share a
/// miss-path-only lock.
pub(crate) struct CoefficientCache<K, V> {
    inner: Store<K, V>,
    publication: std::sync::Mutex<()>,
    /// Advanced by reset; sibling publications from an older sweep are dropped.
    epoch: AtomicU64,
    hits: [PaddedCounter; HIT_STRIPES],
    misses: AtomicU64,
    evictions: std::sync::Arc<AtomicU64>,
    retained_bytes: std::sync::Arc<AtomicUsize>,
    max_bytes: usize,
    /// Test-only observations of the waiter paths quick_cache keeps private.
    #[cfg(test)]
    probes: Probes,
}

#[cfg(test)]
#[derive(Default)]
struct Probes {
    /// Values a waiter received from another loader's placeholder.
    joined: AtomicU64,
    /// Guards issued, including ones handed off by a failed loader.
    guards: AtomicU64,
}

impl<K: Clone + Eq + Hash + CacheKeyCharge, V: Clone + CacheCharge> CoefficientCache<K, V> {
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        let evictions = std::sync::Arc::new(AtomicU64::new(0));
        let retained_bytes = std::sync::Arc::new(AtomicUsize::new(0));
        assert!(
            max_entries > 0,
            "coefficient tiers need a positive entry cap"
        );
        let inner = quick_cache::sync::Cache::with_options(
            quick_cache::OptionsBuilder::new()
                .shards(1)
                .hot_allocation(1.0)
                .estimated_items_capacity(
                    max_entries.min(max_bytes / std::mem::size_of::<(K, Charged<V>)>().max(1)),
                )
                .weight_capacity(max_bytes as u64)
                .build()
                .expect("both required cache capacities are supplied"),
            ByteWeighter {
                minimum: max_bytes.div_ceil(max_entries),
            },
            quick_cache::DefaultHashBuilder::default(),
            Evictions(evictions.clone(), retained_bytes.clone()),
        );
        Self {
            inner,
            publication: std::sync::Mutex::new(()),
            epoch: AtomicU64::new(0),
            hits: Default::default(),
            misses: AtomicU64::new(0),
            evictions,
            retained_bytes,
            max_bytes,
            #[cfg(test)]
            probes: Probes::default(),
        }
    }

    fn hit(&self, value: Charged<V>) -> V {
        self.hits[hit_stripe()].0.fetch_add(1, Ordering::Relaxed);
        value.value
    }

    pub(crate) fn get<Q: Hash + quick_cache::Equivalent<K> + ?Sized>(&self, key: &Q) -> Option<V> {
        self.inner.get(key).map(|value| self.hit(value))
    }

    pub(crate) fn get_or_compute(&self, key: K, compute: impl FnOnce() -> V) -> V {
        self.get_or_try_compute(
            &key,
            || key.clone(),
            || Ok::<_, std::convert::Infallible>(compute()),
        )
        .unwrap()
    }

    /// Borrow before owning the key. Only the guard holder computes; dropping
    /// the guard on error or panic hands it to one waiter, and errors are never
    /// retained.
    pub(crate) fn get_or_try_compute<Q: Hash + quick_cache::Equivalent<K> + ?Sized, E>(
        &self,
        key: &Q,
        own: impl FnOnce() -> K,
        compute: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        if let Some(value) = self.get(key) {
            return Ok(value);
        }
        let key = own();
        let guard = match self.guard(&key) {
            Ok(guard) => guard,
            Err(value) => return Ok(self.hit(value)),
        };
        let value = compute()?;
        // Hold through publication, never during recursive generation.
        let _publication = self.publication.lock().unwrap();
        Ok(self.fill(guard, &key, value))
    }

    /// Single-flight entry for an owned key: `Err` carries a resident or
    /// waited-for value. A sibling publication replaces a placeholder without
    /// filling it, so a guard (possibly handed off) may cover a resident key;
    /// its winner is then delivered to the waiters instead of recomputed.
    fn guard(&self, key: &K) -> Result<Guard<'_, K, V>, Charged<V>> {
        let guard = match self.inner.get_value_or_guard(key, None) {
            quick_cache::sync::GuardResult::Value(value) => {
                #[cfg(test)]
                self.probes.joined.fetch_add(1, Ordering::Relaxed);
                return Err(value);
            }
            quick_cache::sync::GuardResult::Guard(guard) => guard,
            quick_cache::sync::GuardResult::Timeout => unreachable!("waits without timeout"),
        };
        #[cfg(test)]
        self.probes.guards.fetch_add(1, Ordering::Relaxed);
        match self.inner.peek(key) {
            Some(stored) => {
                let _ = guard.insert(stored.clone());
                Err(stored)
            }
            None => Ok(guard),
        }
    }

    /// Publish through a guard; the caller holds the publication mutex.
    fn fill(&self, guard: Guard<'_, K, V>, key: &K, value: V) -> V {
        if let Some(stored) = self.inner.peek(key) {
            let _ = guard.insert(stored.clone());
            return self.hit(stored);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let bytes = entry_charge(key, &value);
        self.retained_bytes.fetch_add(bytes, Ordering::Relaxed);
        debug_assert_eq!(self.inner.capacity(), self.max_bytes as u64);
        // `Ok` includes oversize rejection, which the eviction callback
        // accounts. `Err` means reset invalidated the placeholder: no callback
        // runs, and a plain insert would defeat that invalidation.
        if guard
            .insert(Charged {
                value: value.clone(),
                bytes,
            })
            .is_err()
        {
            self.retained_bytes.fetch_sub(bytes, Ordering::Relaxed);
        }
        value
    }

    /// Single-flight lookup whose computation also yields a reverse lookup
    /// (such as `dual(dual(x)) = x`), seeded only after the requested guard
    /// completes, so opposing requests never wait on each other.
    #[cfg(feature = "lie-generated")]
    pub(crate) fn get_or_compute_seeding<Q: Hash + quick_cache::Equivalent<K> + ?Sized>(
        &self,
        key: &Q,
        own: impl FnOnce() -> K,
        compute: impl FnOnce() -> (V, (K, V)),
    ) -> V {
        let mut reverse = None;
        let value = self
            .get_or_try_compute(key, own, || {
                let epoch = self.epoch.load(Ordering::Relaxed);
                let (value, seed) = compute();
                reverse = Some((epoch, seed));
                Ok::<_, std::convert::Infallible>(value)
            })
            .unwrap();
        if let Some((epoch, (key, value))) = reverse {
            self.seed(epoch, key, value);
        }
        value
    }

    #[cfg(feature = "lie-generated")]
    fn seed(&self, epoch: u64, key: K, value: V) {
        if let Ok(guard) = self.guard(&key) {
            let _publication = self.publication.lock().unwrap();
            if self.epoch.load(Ordering::Relaxed) == epoch {
                self.fill(guard, &key, value);
            }
        }
    }

    /// Reset epoch for [`Self::insert`]; capture it before generating.
    #[cfg(feature = "lie-generated")]
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Publish additional B/C/D sweep channels without waiting on
    /// placeholders whose loaders may be waiting on the catalog.
    /// Values generated before a later reset are returned but not retained.
    #[cfg(feature = "lie-generated")]
    pub(crate) fn insert(&self, epoch: u64, key: K, value: V) -> V {
        let _publication = self.publication.lock().unwrap();
        if self.epoch.load(Ordering::Relaxed) != epoch {
            return value;
        }
        if let Some(stored) = self.inner.peek(&key) {
            return stored.value;
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let bytes = entry_charge(&key, &value);
        self.retained_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.inner.insert(
            key,
            Charged {
                bytes,
                value: value.clone(),
            },
        );
        value
    }

    /// Recheck a sweep publication without counting another lookup.
    #[cfg(feature = "lie-generated")]
    pub(crate) fn peek<Q: Hash + quick_cache::Equivalent<K> + ?Sized>(&self, key: &Q) -> Option<V> {
        self.inner.peek(key).map(|value| value.value)
    }

    fn reset(&self) {
        let _publication = self.publication.lock().unwrap();
        self.epoch.fetch_add(1, Ordering::Relaxed);
        self.inner.clear();
        self.retained_bytes.store(0, Ordering::Relaxed);
        for stripe in &self.hits {
            stripe.0.store(0, Ordering::Relaxed);
        }
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
    }

    /// Remove in unspecified traversal order until the charge fits. In-flight
    /// placeholders survive trim, and may publish after it returns.
    fn trim_to(&self, target_bytes: usize) -> (usize, usize, usize, usize) {
        let _publication = self.publication.lock().unwrap();
        let entries = self.inner.len();
        let bytes = self.retained_bytes.load(Ordering::Relaxed);
        let remaining = std::cell::Cell::new(bytes);
        self.inner.retain(|_, value| {
            if target_bytes != 0 && remaining.get() <= target_bytes {
                true
            } else {
                remaining.set(remaining.get() - value.bytes);
                false
            }
        });
        let removed = entries - self.inner.len();
        self.retained_bytes
            .store(remaining.get(), Ordering::Relaxed);
        self.evictions.fetch_add(removed as u64, Ordering::Relaxed);
        (
            removed,
            bytes - remaining.get(),
            entries - removed,
            remaining.get(),
        )
    }

    fn tier_stats(&self) -> TierStats {
        let _publication = self.publication.lock().unwrap();
        TierStats {
            entries: self.inner.len(),
            bytes: self.retained_bytes.load(Ordering::Relaxed),
            hits: self
                .hits
                .iter()
                .map(|stripe| stripe.0.load(Ordering::Relaxed))
                .sum(),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    fn snapshot(&self) -> (u64, u64, usize, usize) {
        let stats = self.tier_stats();
        (stats.hits, stats.misses, stats.entries, stats.bytes)
    }
}

/// Ordered generated labels, borrowed without cloning their Arc handles.
#[cfg(feature = "lie-generated")]
#[derive(Hash)]
pub(crate) struct Triple<'a, I>(pub &'a I, pub &'a I, pub &'a I);

#[cfg(feature = "lie-generated")]
impl<I: PartialEq> quick_cache::Equivalent<(I, I, I)> for Triple<'_, I> {
    fn equivalent(&self, key: &(I, I, I)) -> bool {
        self.0 == &key.0 && self.1 == &key.1 && self.2 == &key.2
    }
}

#[cfg(feature = "lie-generated")]
#[derive(Hash)]
pub(crate) struct Sextuple<'a, I>(
    pub &'a I,
    pub &'a I,
    pub &'a I,
    pub &'a I,
    pub &'a I,
    pub &'a I,
);

#[cfg(feature = "lie-generated")]
impl<I: PartialEq> quick_cache::Equivalent<(I, I, I, I, I, I)> for Sextuple<'_, I> {
    fn equivalent(&self, key: &(I, I, I, I, I, I)) -> bool {
        self.0 == &key.0
            && self.1 == &key.1
            && self.2 == &key.2
            && self.3 == &key.3
            && self.4 == &key.4
            && self.5 == &key.5
    }
}

static CACHE_3J: LazyLock<CoefficientCache<Regge3j, SignedSqrtRational>> =
    LazyLock::new(|| CoefficientCache::new(DEFAULT_MAX_ENTRIES, effective_budgets().three_j_bytes));
static CACHE_6J: LazyLock<CoefficientCache<Regge6j, SignedSqrtRational>> =
    LazyLock::new(|| CoefficientCache::new(DEFAULT_MAX_ENTRIES, effective_budgets().six_j_bytes));

pub(crate) fn cache_3j() -> &'static CoefficientCache<Regge3j, SignedSqrtRational> {
    &CACHE_3J
}

pub(crate) fn cache_6j() -> &'static CoefficientCache<Regge6j, SignedSqrtRational> {
    &CACHE_6J
}

/// Derived-f64 F-symbol tier. Stores the rounded `f64` F-symbol so a warm
/// hit returns a `Copy` scalar without re-running the bigint `sqrt` in
/// [`SignedSqrtRational::to_f64`]. It is a *presentation* tier over the exact
/// 6j tier (the value authority), never an independent value source: its `f64`
/// is always derived from the exact value, so the two cannot disagree.
static CACHE_F: LazyLock<CoefficientCache<FKey, f64>> = LazyLock::new(|| {
    CoefficientCache::new(DEFAULT_MAX_ENTRIES, effective_budgets().derived_f_bytes)
});

pub(crate) fn cache_f() -> &'static CoefficientCache<FKey, f64> {
    &CACHE_F
}

/// Bounded, byte-accounted SU(N) CGC cache (`lie-generated`).
///
/// CGC tensors are large and expensive (a full SVD/QR/least-squares pipeline),
/// so unlike the exact 3j/6j tiers this cache is charged by *actual sparse
/// storage bytes* ([`crate::lie::sun::Cgc`] entry vector + labels) and holds
/// `Arc<Cgc>` for cheap hit-path cloning. Keyed by the canonical
/// `(s1, s2, s3)` labels.
///
/// # Why in-memory only (no disk tier)
///
/// The reference persists CGCs to a scratch directory. This crate deliberately
/// does not: a persisted store would need an algorithm/gauge-version key to
/// stay sound, because the coefficient *values* are gauge- and
/// algorithm-dependent (unlike the exact 3j/6j tiers, whose bytes are the
/// canonical exact value). Keeping the cache process-local means it is rebuilt
/// from the generator every run and can never disagree with the generator that
/// filled it -- the same argument the exact tiers make for never persisting.
#[cfg(feature = "lie-generated")]
mod cgc_cache {
    use super::{CacheCharge, CoefficientCache};
    use crate::lie::sun::{Cgc, Irrep};
    use std::sync::{Arc, LazyLock};

    /// Canonical cache key: the three irrep labels.
    ///
    /// **No [`GroupId`](crate::lie::GroupId), deliberately.**
    /// A global form gates *which* highest weights may be requested; it never
    /// changes a coefficient of a weight that survives. Admissibility is
    /// enforced once, at construction, by
    /// [`GroupId::admits`](crate::lie::GroupId::admits), so every `Irrep` that
    /// reaches this tier is already a genuine representation of whichever form
    /// asked for it — and two forms sharing a weight *must* share the entry,
    /// since they share the basis, ordering, phase and gauge authority. Adding
    /// the form to the key would only duplicate identical tensors. An
    /// inadmissible weight cannot reach the cache at all: no `Irrep` for it is
    /// ever constructed.
    pub(crate) type CgcKey = (Irrep, Irrep, Irrep);

    impl CacheCharge for Arc<Cgc> {
        fn value_bytes(&self) -> usize {
            self.storage_bytes()
        }
    }

    /// Entry cap for the CGC tier. The retained-charge cap is the real backstop.
    const CGC_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap for the CGC tier (256 MiB): CGC tensors are far larger
    /// than a scalar exact symbol, so this tier gets its own generous budget.
    ///
    /// `pub(super)` so the parent module can tie [`super::GENERATED_CACHE_MAX_BYTES`]
    /// to it in a compile-time assertion (the same drift guard the base tiers use).
    pub(super) const CGC_MAX_BYTES: usize = 256 << 20;

    pub(crate) static CACHE_CGC: LazyLock<CoefficientCache<CgcKey, Arc<Cgc>>> =
        LazyLock::new(|| {
            CoefficientCache::new(CGC_MAX_ENTRIES, super::effective_budgets().sun_cgc_bytes)
        });
}

#[cfg(feature = "lie-generated")]
pub(crate) fn cache_cgc(
) -> &'static CoefficientCache<cgc_cache::CgcKey, std::sync::Arc<crate::lie::sun::Cgc>> {
    &cgc_cache::CACHE_CGC
}

/// Bounded exact SU(N) tensor-product and dual-lookup tier (`lie-generated`).
///
/// The unordered irrep pair is a complete key after `sun::directproduct`
/// rejects rank mismatch. Values are sorted shared channel slices: this crate's
/// multiplicity/channel consumers avoid rebuilding a public `BTreeMap`, while
/// the public API preserves its owned-map contract by reconstructing one from
/// this exact value. Concurrent misses share one computation through a
/// quick_cache placeholder.
/// `LieGroup` dual keys reuse singleton slices under this same budget and
/// lifecycle, without growing coefficient keys or values.
#[cfg(feature = "lie-generated")]
mod sun_product_cache {
    use super::CoefficientCache;
    use crate::lie::sun::{SunProduct, SunProductKey};
    use std::sync::LazyLock;

    // Original FIFO collectors retained 102 pairs / 38,600 charged bytes and
    // 85 pairs / 54,504 charged bytes. Preserve their approved ceilings below;
    // the new charge counts one key plus metadata, rather than two key copies.
    // This is a retained-charge backstop, not a live-memory ceiling.
    pub(super) const SUN_PRODUCT_MAX_ENTRIES: usize = 256;
    pub(super) const SUN_PRODUCT_MAX_BYTES: usize = 128 << 10;

    pub(crate) static CACHE_SUN_PRODUCT: LazyLock<CoefficientCache<SunProductKey, SunProduct>> =
        LazyLock::new(|| {
            CoefficientCache::new(
                SUN_PRODUCT_MAX_ENTRIES,
                super::effective_budgets().sun_product_bytes,
            )
        });
}

#[cfg(feature = "lie-generated")]
pub(crate) fn cache_sun_product(
) -> &'static CoefficientCache<crate::lie::sun::SunProductKey, crate::lie::sun::SunProduct> {
    &sun_product_cache::CACHE_SUN_PRODUCT
}

/// Bounded, byte-accounted derived-f64 SU(N) F-symbol cache (`lie-generated`).
///
/// An F block is the contraction of four CGC; even with warm CGC that is real
/// work, so the derived `[μ,ν,κ,λ]` block is cached. Keyed by the **plain
/// ordered six-label tuple** `(a,b,c,d,e,f)` — see the Why-comment in
/// `sun::fr::f_symbol` for why no Regge-style canonicalization exists for
/// GT-basis F blocks (the 6j symmetry group that lets the exact SU(2) F tier
/// key on a canonical class has no analogue here).
///
/// R needs no cache: it is a single sparse join of two CGC (no four-way
/// contraction), cheap enough that a cache slot would not pay for itself.
///
/// In-memory only, same argument as the CGC tier: the values are
/// gauge/algorithm-dependent, so a persisted store would need a version key;
/// keeping it process-local means it is always consistent with the generator.
#[cfg(feature = "lie-generated")]
mod sun_f_cache {
    use super::{CacheCharge, CoefficientCache};
    use crate::lie::sun::{FBlock, Irrep};
    use std::sync::LazyLock;

    /// Canonical cache key: the six irrep labels `(a, b, c, d, e, f)`.
    pub(crate) type SunFKey = (Irrep, Irrep, Irrep, Irrep, Irrep, Irrep);

    impl CacheCharge for FBlock {
        fn value_bytes(&self) -> usize {
            std::mem::size_of::<FBlock>()
                + if self.data().len() > 1 {
                    super::ARC_COUNTS_BYTES + std::mem::size_of_val(self.data())
                } else {
                    0
                }
        }
    }

    /// Entry cap; the retained-charge cap is the real backstop.
    const SUN_F_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap (64 MiB): F blocks are tiny (a few multiplicity
    /// indices), so this holds a very large working set. `pub(super)` for the
    /// [`super::GENERATED_CACHE_MAX_BYTES`] drift assertion.
    pub(super) const SUN_F_MAX_BYTES: usize = 64 << 20;

    pub(crate) static CACHE_SUN_F: LazyLock<CoefficientCache<SunFKey, FBlock>> =
        LazyLock::new(|| {
            CoefficientCache::new(SUN_F_MAX_ENTRIES, super::effective_budgets().sun_f_bytes)
        });
}

#[cfg(feature = "lie-generated")]
pub(crate) fn cache_sun_f(
) -> &'static CoefficientCache<sun_f_cache::SunFKey, crate::lie::sun::FBlock> {
    &sun_f_cache::CACHE_SUN_F
}

/// Bounded, byte-accounted derived-f64 B/C/D F-symbol cache (`lie-generated`).
///
/// The B/C/D analogue of [`cache_sun_f`]: an F block is the contraction of four
/// catalog-driven CGC, real work even with warm generators, so the derived
/// `[μ,ν,κ,λ]` block is cached. Same design as the SU(N) tier — the **plain
/// ordered six-label key** `(a,b,c,d,e,f)` (no Regge canonicalization exists for
/// GT/sweep-basis F blocks; see the Why-comment in `sun::fr::f_symbol`), the
/// shared `FBlock` [`CacheCharge`] impl, in-memory only (values are
/// gauge/algorithm-dependent, so a persisted store would need a version key).
///
/// R needs no cache: it is a single sparse join of two CGC, and no measurement
/// shows a benefit from caching it.
#[cfg(feature = "lie-generated")]
mod bcd_f_cache {
    use super::CoefficientCache;
    use crate::frcore::FBlock;
    use crate::lie::bcd::Irrep;
    use std::sync::LazyLock;

    /// Canonical cache key: the six B/C/D irrep labels `(a, b, c, d, e, f)`.
    pub(crate) type BcdFKey = (Irrep, Irrep, Irrep, Irrep, Irrep, Irrep);

    /// Entry cap; the retained-charge cap is the real backstop.
    const BCD_F_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap (64 MiB): F blocks are tiny (a few multiplicity indices).
    /// `pub(super)` for the [`super::GENERATED_CACHE_MAX_BYTES`] drift assertion.
    pub(super) const BCD_F_MAX_BYTES: usize = 64 << 20;

    pub(crate) static CACHE_BCD_F: LazyLock<CoefficientCache<BcdFKey, FBlock>> =
        LazyLock::new(|| {
            CoefficientCache::new(BCD_F_MAX_ENTRIES, super::effective_budgets().bcd_f_bytes)
        });
}

#[cfg(feature = "lie-generated")]
pub(crate) fn cache_bcd_f() -> &'static CoefficientCache<bcd_f_cache::BcdFKey, crate::frcore::FBlock>
{
    &bcd_f_cache::CACHE_BCD_F
}

/// Bounded, byte-accounted B/C/D exact-product, CGC and Lie dual-lookup tier.
///
/// Private key/value variants share the existing `BcdCgc` byte budget, Clock-PRO
/// eviction, statistics, trim and reset controls. Exact products are shared
/// sorted channel slices on an unordered irrep-pair key. Dense CGC isometries
/// use ordered triples; one sweep populates all coupled channels.
/// Keys omit global form: admissibility is checked by `LieGroup::irrep`, while
/// surviving weights share their canonical coefficients across global forms.
///
/// Generator catalogs belong to individual `LieGroup` values and are shared
/// by clones. Resetting one does not invalidate these global cached values.
/// `LieGroup` dual lookups share this tier's budget and lifecycle. Their private
/// keys fit inside the existing shell; values reuse singleton product slices.
#[cfg(feature = "lie-generated")]
pub(crate) mod bcd_cgc_cache {
    use super::{CacheCharge, CacheKeyCharge, CoefficientCache};
    use crate::lie::bcd::{CatalogCgc, Irrep};
    use std::sync::{Arc, LazyLock};

    /// Product pairs, CGC triples, or one dual-lookup label.
    ///
    /// **No [`GroupId`](crate::lie::GroupId), deliberately.**
    /// A global form gates *which* highest weights may be requested; it never
    /// changes a coefficient of a weight that survives. Admissibility is
    /// enforced once, at construction, by
    /// [`GroupId::admits`](crate::lie::GroupId::admits), so every `Irrep` that
    /// reaches this tier is already a genuine representation of whichever form
    /// asked for it — and two forms sharing a weight *must* share the entry,
    /// since they share the basis, ordering, phase and gauge authority. Adding
    /// the form to the key would only duplicate identical tensors. An
    /// inadmissible weight cannot reach the cache at all: no `Irrep` for it is
    /// ever constructed.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub(crate) enum BcdKey {
        Product(Irrep, Irrep),
        Cgc(Irrep, Irrep, Irrep),
        Dual(Irrep),
    }
    #[derive(Hash)]
    pub(crate) enum BcdRef<'a> {
        Product(&'a Irrep, &'a Irrep),
        Cgc(&'a Irrep, &'a Irrep, &'a Irrep),
        Dual(&'a Irrep),
    }
    impl quick_cache::Equivalent<BcdKey> for BcdRef<'_> {
        fn equivalent(&self, key: &BcdKey) -> bool {
            match (self, key) {
                (Self::Product(a, b), BcdKey::Product(x, y)) => *a == x && *b == y,
                (Self::Cgc(a, b, c), BcdKey::Cgc(x, y, z)) => *a == x && *b == y && *c == z,
                (Self::Dual(a), BcdKey::Dual(x)) => *a == x,
                _ => false,
            }
        }
    }
    impl CacheKeyCharge for BcdKey {
        fn key_bytes(&self) -> usize {
            std::mem::size_of::<Self>()
                + match self {
                    Self::Product(a, b) => a.key_bytes() + b.key_bytes(),
                    Self::Cgc(a, b, c) => a.key_bytes() + b.key_bytes() + c.key_bytes(),
                    Self::Dual(a) => a.key_bytes(),
                }
        }
    }
    #[derive(Clone, Debug)]
    pub(crate) enum BcdValue {
        Product(Arc<[(Irrep, u32)]>),
        Cgc(Arc<CatalogCgc>),
    }
    impl CacheCharge for BcdValue {
        fn value_bytes(&self) -> usize {
            std::mem::size_of::<Self>()
                + match self {
                    Self::Product(channels) => {
                        super::ARC_COUNTS_BYTES
                            + std::mem::size_of_val(channels.as_ref())
                            + channels
                                .iter()
                                .map(|(c, _)| c.key_bytes() - std::mem::size_of::<Irrep>())
                                .sum::<usize>()
                    }
                    Self::Cgc(cgc) => cgc.storage_bytes(),
                }
        }
    }

    /// Entry cap; the retained-charge cap is the real backstop.
    const BCD_CGC_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap (256 MiB): dense product isometries are far larger
    /// than an F block, so this tier gets its own generous budget (as the SU(N)
    /// CGC tier).
    /// `pub(super)` for the [`super::GENERATED_CACHE_MAX_BYTES`] drift assertion.
    pub(super) const BCD_CGC_MAX_BYTES: usize = 256 << 20;

    pub(crate) static CACHE_BCD_CGC: LazyLock<CoefficientCache<BcdKey, BcdValue>> =
        LazyLock::new(|| {
            CoefficientCache::new(
                BCD_CGC_MAX_ENTRIES,
                super::effective_budgets().bcd_cgc_bytes,
            )
        });
}

#[cfg(feature = "lie-generated")]
pub(crate) fn cache_bcd_cgc(
) -> &'static CoefficientCache<bcd_cgc_cache::BcdKey, bcd_cgc_cache::BcdValue> {
    &bcd_cgc_cache::CACHE_BCD_CGC
}

/// Clear the 3j, 6j, and derived-f64 F-symbol caches (and, under `lie-generated`, the
/// SU(N) product and SU(N)/B/C/D CGC/F caches) and *all* their counters — entries, bytes,
/// hits, misses, and evictions all return to zero.
///
/// # Reset ownership (process-global, single-owner)
///
/// These tiers are `static`, so `reset()` acts on process-global state. It is a
/// **single-owner** operation: exactly one component in a consuming process
/// (for example an engine `Runtime` that owns the coefficient authority) may own
/// the reset policy. **A library must not call `reset()`** — doing so would
/// clear a cache another component is relying on, since there is one shared
/// coefficient-value authority per process (consumers must not keep a mirror).
pub fn reset() {
    CACHE_3J.reset();
    CACHE_6J.reset();
    CACHE_F.reset();
    #[cfg(feature = "lie-generated")]
    {
        sun_product_cache::CACHE_SUN_PRODUCT.reset();
        cgc_cache::CACHE_CGC.reset();
        sun_f_cache::CACHE_SUN_F.reset();
        bcd_f_cache::CACHE_BCD_F.reset();
        bcd_cgc_cache::CACHE_BCD_CGC.reset();
    }
}

/// Release retained entries from one tier down to a per-tier charged-byte target.
///
/// The selected tier removes entries in unspecified traversal order until its
/// charged bytes are at most `target_charged_bytes`. Entries are
/// indivisible, so the retained charge can be below the target. A zero target
/// clears that tier's entries only. Hits and misses are preserved; each removed
/// entry increments the tier's eviction count. This does not alter the
/// process's one-shot [`CoefficientCacheBudgets`] policy, so a later lookup may
/// refill the tier under the same cap.
///
/// Like [`reset`], trimming is a process-global lifecycle operation with one
/// application owner. A library must not trim a shared coefficient cache behind
/// its caller's back. A trim is also a cache-policy observation: if no policy
/// was configured first, accessing the selected cache fixes the default policy.
///
/// A hit that acquired its read lock before trimming can still return its clone;
/// trimming first makes a later lookup miss. A miss computes outside the lock
/// and can publish after this function returns. See [`CacheTrimReport`] for the
/// report's linearization-point and memory-accounting boundaries.
pub fn trim_to(tier: CoefficientCacheTier, target_charged_bytes: usize) -> CacheTrimReport {
    let (removed_entries, removed_charged_bytes, remaining_entries, remaining_charged_bytes) =
        match tier {
            CoefficientCacheTier::ThreeJ => CACHE_3J.trim_to(target_charged_bytes),
            CoefficientCacheTier::SixJ => CACHE_6J.trim_to(target_charged_bytes),
            CoefficientCacheTier::DerivedF => CACHE_F.trim_to(target_charged_bytes),
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunProduct => {
                sun_product_cache::CACHE_SUN_PRODUCT.trim_to(target_charged_bytes)
            }
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunCgc => cgc_cache::CACHE_CGC.trim_to(target_charged_bytes),
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunF => sun_f_cache::CACHE_SUN_F.trim_to(target_charged_bytes),
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdCgc => {
                bcd_cgc_cache::CACHE_BCD_CGC.trim_to(target_charged_bytes)
            }
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdF => bcd_f_cache::CACHE_BCD_F.trim_to(target_charged_bytes),
        };
    CacheTrimReport {
        tier,
        removed_entries,
        removed_charged_bytes,
        remaining_entries,
        remaining_charged_bytes,
    }
}

/// Aggregate hit/miss/entry/byte statistics across the 3j, 6j, and derived-f64
/// F-symbol caches.
pub fn stats() -> CacheStats {
    let (h3, m3, e3, b3) = CACHE_3J.snapshot();
    let (h6, m6, e6, b6) = CACHE_6J.snapshot();
    let (hf, mf, ef, bf) = CACHE_F.snapshot();
    #[cfg(feature = "lie-generated")]
    let (hc, mc, ec, bc) = {
        let (hp, mp, ep, bp) = sun_product_cache::CACHE_SUN_PRODUCT.snapshot();
        let (h, m, e, b) = cgc_cache::CACHE_CGC.snapshot();
        let (h2, m2, e2, b2) = sun_f_cache::CACHE_SUN_F.snapshot();
        let (h3, m3, e3, b3) = bcd_f_cache::CACHE_BCD_F.snapshot();
        let (h4, m4, e4, b4) = bcd_cgc_cache::CACHE_BCD_CGC.snapshot();
        (
            hp + h + h2 + h3 + h4,
            mp + m + m2 + m3 + m4,
            ep + e + e2 + e3 + e4,
            bp + b + b2 + b3 + b4,
        )
    };
    #[cfg(not(feature = "lie-generated"))]
    let (hc, mc, ec, bc) = (0u64, 0u64, 0usize, 0usize);
    CacheStats {
        hits: h3 + h6 + hf + hc,
        misses: m3 + m6 + mf + mc,
        entries: e3 + e6 + ef + ec,
        bytes: b3 + b6 + bf + bc,
    }
}

/// Per-tier and total statistics for the three base SU(2) coefficient tiers
/// (3j, 6j, derived-F).
///
/// Unlike the aggregate [`stats`] — which also sums the `lie-generated` generated
/// tiers when that feature is on — this reports only the base SU(2) surface,
/// split per tier, and adds the eviction counter. Conservative retained entry charge is capped by
/// [`BASE_CACHE_MAX_BYTES`] (`total().bytes ≤ BASE_CACHE_MAX_BYTES`). See
/// [`BaseCacheStats`] for the snapshot-consistency contract of `total()`.
pub fn base_cache_stats() -> BaseCacheStats {
    BaseCacheStats {
        three_j: CACHE_3J.tier_stats(),
        six_j: CACHE_6J.tier_stats(),
        derived_f: CACHE_F.tier_stats(),
    }
}

/// Per-tier and total statistics for the five generated `lie-generated` tiers
/// (SU(N) product, SU(N) CGC, SU(N) F, B/C/D product+CGC, B/C/D F).
///
/// The generated-family analogue of [`base_cache_stats`]: unlike the aggregate
/// [`stats`] (which sums base *and* generated tiers into one flat
/// [`CacheStats`]), this reports each generated tier separately, adds the
/// eviction counter, and exposes a field-wise [`total`](GeneratedCacheStats::total).
/// Conservative retained entry charge is capped by [`GENERATED_CACHE_MAX_BYTES`]
/// (`total().bytes ≤ GENERATED_CACHE_MAX_BYTES`). See [`GeneratedCacheStats`]
/// for the snapshot-consistency contract of `total()` and the stability caveat.
#[cfg(feature = "lie-generated")]
pub fn generated_cache_stats() -> GeneratedCacheStats {
    GeneratedCacheStats {
        sun_product: sun_product_cache::CACHE_SUN_PRODUCT.tier_stats(),
        sun_cgc: cgc_cache::CACHE_CGC.tier_stats(),
        sun_f: sun_f_cache::CACHE_SUN_F.tier_stats(),
        bcd_cgc: bcd_cgc_cache::CACHE_BCD_CGC.tier_stats(),
        bcd_f: bcd_f_cache::CACHE_BCD_F.tier_stats(),
    }
}

// Serialize unit tests that reset process-global coefficient caches.
#[cfg(test)]
pub(crate) static TEST_CACHE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::Ratio;
    use std::sync::Arc;

    fn val(n: i64) -> SignedSqrtRational {
        SignedSqrtRational::from_prefactor_radical(
            Ratio::from(BigInt::from(1)),
            Ratio::new(BigInt::from(n.unsigned_abs()), BigInt::from(1)),
        )
    }

    #[test]
    fn hit_returns_stored_and_counts() {
        let c: CoefficientCache<u32, SignedSqrtRational> = CoefficientCache::new(16, 1 << 20);
        let mut computed = 0;
        let a = c.get_or_compute(7, || {
            computed += 1;
            val(7)
        });
        let b = c.get_or_compute(7, || {
            computed += 1;
            val(999) // must not run
        });
        assert_eq!(a, b);
        assert_eq!(computed, 1, "second lookup must be a hit");
        let (hits, misses, entries, _) = c.snapshot();
        assert_eq!((hits, misses, entries), (1, 1, 1));
    }

    #[test]
    fn entry_bound_evicts() {
        let c: CoefficientCache<u32, SignedSqrtRational> = CoefficientCache::new(3, 3 << 28);
        for k in 0..5u32 {
            c.get_or_compute(k, || val(k as i64 + 1));
        }
        let (_, _, entries, _) = c.snapshot();
        assert!(entries <= 3, "entry bound violated: {entries}");
        assert_eq!(c.tier_stats().evictions, 2);
    }

    #[test]
    fn entry_cap_uses_weight_and_stats_report_real_charge() {
        let cache = CoefficientCache::new(3, 1000);
        let charge = entry_charge(&0u32, &0.0f64);
        for key in 0..100u32 {
            cache.get_or_compute(key, || f64::from(key));
            let stats = cache.tier_stats();
            assert!(stats.entries <= 3);
            assert_eq!(stats.bytes, stats.entries * charge);
            assert_eq!(
                cache.inner.weight() as usize,
                stats.entries * 1000usize.div_ceil(3)
            );
            assert_eq!(stats.evictions as usize + stats.entries, key as usize + 1);
        }
        // The cap is enforced solely by weights, with no resident traversal.
        assert_eq!(cache.inner.num_shards(), 1);
        let before = cache.tier_stats();
        let report = cache.trim_to(charge);
        assert_eq!(report.1 + report.3, before.bytes);
        assert!(report.3 <= charge);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn generated_cgc_at_full_tier_cap_is_retained() {
        use crate::lie::sun::{cgc, Irrep};
        let a = Irrep::from_dynkin(&[1, 1]).unwrap();
        let key = (a.clone(), a.clone(), a.clone());
        let value = Arc::new(cgc(&a, &a, &a).unwrap());
        let charge = entry_charge(&key, &value);
        for budget in [charge, charge + 1] {
            let cache = CoefficientCache::new(16, budget);
            assert_eq!(cache.inner.num_shards(), 1);
            assert_eq!(cache.inner.capacity() as usize, budget);
            let mut loads = 0;
            let cold = cache.get_or_compute(key.clone(), || {
                loads += 1;
                Arc::new(cgc(&a, &a, &a).unwrap())
            });
            let before = cache.tier_stats();
            assert_eq!(
                (before.entries, before.bytes, before.misses),
                (1, charge, 1)
            );
            let warm = cache.get_or_compute(key.clone(), || {
                loads += 1;
                Arc::new(cgc(&a, &a, &a).unwrap())
            });
            assert!(Arc::ptr_eq(&cold, &warm));
            assert_eq!(loads, 1);
            assert_eq!(cache.tier_stats().misses, before.misses);
            assert_eq!(cache.tier_stats().hits, 1);
            // Trim removes without shrinking capacity, so admission stays full-cap.
            cache.trim_to(0);
            cache.get_or_compute(key.clone(), || Arc::new(cgc(&a, &a, &a).unwrap()));
            assert_eq!(cache.tier_stats().entries, 1);
            assert_eq!(cache.inner.capacity() as usize, budget);
        }
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn concurrent_sweep_publication_counts_only_one_miss() {
        use std::sync::mpsc;
        let cache = Arc::new(CoefficientCache::new(16, 1024));
        let (started_tx, started_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let worker_cache = Arc::clone(&cache);
        let worker = std::thread::spawn(move || {
            worker_cache.get_or_compute(1u32, || {
                started_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
                worker_cache.peek(&1).unwrap()
            })
        });
        started_rx.recv().unwrap();
        assert_eq!(cache.insert(0, 1, 1.0f64), 1.0);
        assert_eq!(cache.insert(0, 1, 2.0f64), 1.0);
        assert_eq!(cache.tier_stats().misses, 1);
        resume_tx.send(()).unwrap();
        assert_eq!(worker.join().unwrap(), 1.0);
        let stats = cache.tier_stats();
        assert_eq!(
            (stats.hits, stats.misses, stats.entries, stats.evictions),
            (1, 1, 1, 0)
        );
    }

    #[test]
    fn byte_bound_evicts() {
        // Tiny byte budget: only a couple of entries fit at once.
        let per = entry_charge(&0u32, &val(1));
        let c: CoefficientCache<u32, SignedSqrtRational> =
            CoefficientCache::new(1_000_000, per * 2 + per / 2);
        for k in 0..20u32 {
            c.get_or_compute(k, || val(k as i64 + 1));
        }
        let (_, _, _, bytes) = c.snapshot();
        assert!(
            bytes <= per * 2 + per / 2,
            "retained charge cap violated: {bytes}"
        );
    }

    #[test]
    fn trim_has_exact_accounting_without_an_eviction_order_contract() {
        let c = CoefficientCache::new(16, 1 << 20);
        let values = [
            val(1),
            SignedSqrtRational::from_prefactor_radical(
                Ratio::from(BigInt::from(1)),
                Ratio::new(BigInt::from(1) << 256, BigInt::from(1)),
            ),
            val(3),
        ];
        for (key, value) in (1u32..=3).zip(&values) {
            c.get_or_compute(key, || value.clone());
        }
        let before = c.tier_stats();
        let target = before.bytes / 2;
        let (removed, removed_bytes, retained, retained_bytes) = c.trim_to(target);
        assert!(removed > 0 && retained_bytes <= target);
        assert_eq!(removed + retained, before.entries);
        assert_eq!(removed_bytes + retained_bytes, before.bytes);
        let actual_bytes: usize = c
            .inner
            .iter()
            .map(|(k, v)| {
                assert_eq!(v.value, values[k as usize - 1]);
                entry_charge(&k, &v.value)
            })
            .sum();
        assert_eq!(retained_bytes, actual_bytes);
        assert_eq!(c.tier_stats().evictions, removed as u64);
        assert_eq!(c.trim_to(retained_bytes), (0, 0, retained, retained_bytes));
        assert_eq!(c.trim_to(0), (retained, retained_bytes, 0, 0));
        assert_eq!(c.tier_stats().evictions, before.entries as u64);
        c.reset();
        assert_eq!(c.tier_stats(), TierStats::default());
    }

    #[test]
    fn in_flight_miss_can_publish_after_trim() {
        use std::sync::mpsc;

        let cache = Arc::new(CoefficientCache::<u32, SignedSqrtRational>::new(
            16,
            1 << 20,
        ));
        let (started_tx, started_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let worker_cache = Arc::clone(&cache);
        let worker = std::thread::spawn(move || {
            worker_cache.get_or_compute(9, || {
                started_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
                val(9)
            })
        });
        started_rx.recv().unwrap();
        assert_eq!(cache.trim_to(0), (0, 0, 0, 0));
        resume_tx.send(()).unwrap();
        assert_eq!(worker.join().unwrap(), val(9));
        assert_eq!(cache.tier_stats().entries, 1);
    }

    #[test]
    fn reset_during_generation_does_not_leave_a_phantom_charge() {
        use std::sync::mpsc;
        let cache = Arc::new(CoefficientCache::new(16, 1024));
        let (started_tx, started_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let worker_cache = Arc::clone(&cache);
        let worker = std::thread::spawn(move || {
            worker_cache.get_or_compute(1u32, || {
                started_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
                1.0f64
            })
        });
        started_rx.recv().unwrap();
        cache.reset();
        resume_tx.send(()).unwrap();
        assert_eq!(worker.join().unwrap(), 1.0);
        let stats = cache.tier_stats();
        assert_eq!((stats.entries, stats.bytes, stats.evictions), (0, 0, 0));
        cache.get_or_compute(1, || 1.0);
        assert_eq!(cache.tier_stats().bytes, entry_charge(&1u32, &1.0f64));
    }

    #[test]
    fn eviction_thrash_never_changes_values() {
        // Budget of one entry, hammered with 200 distinct keys in a cycle:
        // every returned value must still equal its from-scratch computation.
        let c: CoefficientCache<u32, SignedSqrtRational> = CoefficientCache::new(1, 1 << 30);
        for round in 0..3 {
            for k in 0..200u32 {
                let got = c.get_or_compute(k, || val(k as i64 * 3 + 1));
                assert_eq!(got, val(k as i64 * 3 + 1), "round {round} key {k}");
            }
        }
    }

    #[test]
    fn f64_tier_hit_skips_recompute() {
        // The derived-f64 F-symbol tier's contract: a warm hit returns the
        // stored scalar WITHOUT re-running the miss closure -- which is the sole
        // site of the bigint `sqrt` in SignedSqrtRational::to_f64 on the F path.
        // So the public LieGroup::f_symbol hot path avoids bigint isqrt on a hit.
        let c: CoefficientCache<u32, f64> = CoefficientCache::new(16, 1 << 20);
        let mut rounded = 0;
        let a = c.get_or_compute(9, || {
            rounded += 1;
            val(9).to_f64() // stands in for f_symbol_exact(..).to_f64()
        });
        let b = c.get_or_compute(9, || {
            rounded += 1;
            val(999).to_f64() // must not run on the hit
        });
        assert_eq!(a, b);
        assert_eq!(rounded, 1, "a hit must not re-run the rounding closure");
        let (hits, misses, entries, _) = c.snapshot();
        assert_eq!((hits, misses, entries), (1, 1, 1));
    }

    #[test]
    fn f64_tier_charge_is_fixed() {
        // f64 values charge a fixed size (no data-dependent limbs), so the tier
        // is bounded by entry count in practice.
        assert_eq!((1.0f64).value_bytes(), std::mem::size_of::<f64>());
        assert_eq!((-3.5f64).value_bytes(), std::mem::size_of::<f64>());
        assert_eq!(
            entry_charge(&7u32, &1.0f64),
            std::mem::size_of::<f64>() + std::mem::size_of::<u32>() + std::mem::size_of::<usize>(),
            "one owned key and cached byte charge"
        );
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn generated_keys_charge_owned_irrep_weights() {
        use super::bcd_f_cache::BcdFKey;
        use super::cgc_cache::CgcKey;
        use crate::lie::bcd::{Irrep as BcdIrrep, Series};
        use crate::lie::sun::Irrep as SunIrrep;

        let sun = SunIrrep::from_dynkin(&[1; 12]).unwrap();
        let sun_key: CgcKey = (sun.clone(), sun.clone(), sun);
        let sun_irrep_bytes =
            std::mem::size_of::<SunIrrep>() + ARC_COUNTS_BYTES + 13 * std::mem::size_of::<i64>();
        assert_eq!(sun_key.key_bytes(), 3 * sun_irrep_bytes);

        let bcd = BcdIrrep::from_dynkin(Series::C, &[1; 12]).unwrap();
        let bcd_key: BcdFKey = (
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd,
        );
        let bcd_irrep_bytes =
            std::mem::size_of::<BcdIrrep>() + ARC_COUNTS_BYTES + 12 * std::mem::size_of::<i64>();
        assert_eq!(bcd_key.key_bytes(), 6 * bcd_irrep_bytes);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn dual_keys_preserve_existing_entry_sizes() {
        use super::bcd_cgc_cache::{BcdKey, BcdValue};
        use crate::lie::bcd::{Irrep as BcdIrrep, Series};
        use crate::lie::sun::{Irrep as SunIrrep, SunProductKey};
        use std::{mem::size_of, sync::Arc};

        // Original Product/Cgc layouts, before adding dual-key variants.
        assert_eq!(size_of::<SunProductKey>(), 2 * size_of::<SunIrrep>());
        assert_eq!(size_of::<BcdKey>(), 3 * size_of::<BcdIrrep>());
        assert_eq!(size_of::<BcdValue>(), size_of::<Arc<[(BcdIrrep, u32)]>>());

        let sun = SunIrrep::from_dynkin(&[1, 0]).unwrap();
        assert_eq!(
            SunProductKey::Dual(sun).key_bytes(),
            size_of::<SunProductKey>() + ARC_COUNTS_BYTES + 3 * size_of::<i64>()
        );
        let bcd = BcdIrrep::from_dynkin(Series::D, &[0, 2, 0]).unwrap();
        assert_eq!(
            BcdKey::Dual(bcd.clone()).key_bytes(),
            size_of::<BcdKey>() + bcd.key_bytes()
        );
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn sun_product_tier_caps_charge_and_oversize_eviction_are_exact() {
        use super::sun_product_cache::{SUN_PRODUCT_MAX_BYTES, SUN_PRODUCT_MAX_ENTRIES};
        use crate::lie::sun::{Irrep as SunIrrep, SunProduct, SunProductKey};
        use std::collections::BTreeMap;

        assert_eq!(SUN_PRODUCT_MAX_ENTRIES, 256);
        assert_eq!(SUN_PRODUCT_MAX_BYTES, 128 << 10);

        let a = SunIrrep::from_dynkin(&[2, 1]).unwrap();
        let b = SunIrrep::from_dynkin(&[1, 2]).unwrap();
        let key = SunProductKey::new(&a, &b);
        let key_bytes = 2 * std::mem::size_of::<SunIrrep>()
            + 2 * ARC_COUNTS_BYTES
            + (a.rank() + b.rank()) * std::mem::size_of::<i64>();
        assert_eq!(key.key_bytes(), key_bytes);

        let trivial = SunIrrep::trivial(3).unwrap();
        let adjoint = SunIrrep::from_dynkin(&[1, 1]).unwrap();
        let product =
            SunProduct::from_map(BTreeMap::from([(trivial.clone(), 1), (adjoint.clone(), 2)]));
        let value_bytes = std::mem::size_of::<SunProduct>()
            + 3 * ARC_COUNTS_BYTES
            + 2 * std::mem::size_of::<(SunIrrep, u32)>()
            + (trivial.rank() + adjoint.rank()) * std::mem::size_of::<i64>();
        assert_eq!(product.value_bytes(), value_bytes);

        let charge = value_bytes + key_bytes + std::mem::size_of::<usize>();
        assert_eq!(entry_charge(&key, &product), charge);
        let cache: CoefficientCache<SunProductKey, SunProduct> =
            CoefficientCache::new(usize::MAX, charge - 1);
        assert_eq!(
            cache.get_or_compute(key, || product.clone()),
            product,
            "an oversize exact value is still returned"
        );
        assert_eq!(
            cache.tier_stats(),
            TierStats {
                misses: 1,
                evictions: 1,
                ..TierStats::default()
            }
        );
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn concurrent_sun_product_misses_compute_once() {
        use crate::lie::sun::{Irrep as SunIrrep, SunProduct, SunProductKey};
        use std::collections::BTreeMap;
        use std::sync::{Arc, Barrier};

        let a = SunIrrep::trivial(3).unwrap();
        let b = SunIrrep::from_dynkin(&[1, 0]).unwrap();
        let key = SunProductKey::new(&a, &b);
        let channel = b.clone();
        let cache = Arc::new(CoefficientCache::new(16, 1 << 20));
        let start = Arc::new(Barrier::new(8));
        let computations = Arc::new(AtomicU64::new(0));
        let products = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let key = key.clone();
                let channel = channel.clone();
                let start = Arc::clone(&start);
                let computations = Arc::clone(&computations);
                std::thread::spawn(move || {
                    start.wait();
                    cache.get_or_compute(key, || {
                        computations.fetch_add(1, Ordering::Relaxed);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        SunProduct::from_map(BTreeMap::from([(channel, 1)]))
                    })
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();

        assert!(products[1..]
            .iter()
            .all(|product| products[0].ptr_eq(product)));
        assert_eq!(cache.tier_stats().entries, 1);
        assert_eq!(cache.tier_stats().misses, 1);
        assert_eq!(cache.tier_stats().hits, 7);
        assert_eq!(computations.load(Ordering::Relaxed), 1);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn local_trim_releases_one_sun_product_arc_owner() {
        use crate::lie::sun::{Irrep as SunIrrep, SunProduct, SunProductKey};
        use std::collections::BTreeMap;

        let three = SunIrrep::from_dynkin(&[1, 0]).unwrap();
        let three_bar = SunIrrep::from_dynkin(&[0, 1]).unwrap();
        let key = SunProductKey::new(&three, &three_bar);
        let external = SunProduct::from_map(BTreeMap::from([
            (SunIrrep::trivial(3).unwrap(), 1),
            (SunIrrep::from_dynkin(&[1, 1]).unwrap(), 1),
        ]));
        let local = CoefficientCache::new(16, 1 << 20);
        local.insert(0, key, external.clone());
        let before = external.strong_count();
        assert!(
            before >= 2,
            "caller and local cache own the shared channels"
        );
        assert_eq!(local.trim_to(0).0, 1);
        assert_eq!(external.strong_count(), before - 1);
        assert_eq!(external.iter().count(), 2);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn deep_key_bytes_drive_eviction_oversize_and_reset() {
        use super::bcd_f_cache::BcdFKey;
        use super::cgc_cache::CgcKey;
        use crate::lie::bcd::{Irrep as BcdIrrep, Series};
        use crate::lie::sun::Irrep as SunIrrep;

        let sun = |label| SunIrrep::from_dynkin(&[label; 12]).unwrap();
        let first: CgcKey = (sun(1), sun(1), sun(1));
        let second: CgcKey = (sun(2), sun(2), sun(2));
        let one_entry_budget = entry_charge(&first, &1.0f64);
        let cache: CoefficientCache<CgcKey, f64> =
            CoefficientCache::new(usize::MAX, one_entry_budget);
        cache.insert(0, first.clone(), 1.0);
        cache.insert(0, second, 2.0);
        assert_eq!(cache.tier_stats().entries, 1);
        assert_eq!(cache.tier_stats().evictions, 1);
        cache.reset();
        assert_eq!(cache.tier_stats(), TierStats::default());

        let bcd = BcdIrrep::from_dynkin(Series::C, &[1; 12]).unwrap();
        let bcd_key: BcdFKey = (
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd.clone(),
            bcd,
        );
        let shallow_only = std::mem::size_of::<f64>() + 2 * std::mem::size_of::<BcdFKey>();
        assert!(entry_charge(&bcd_key, &1.0f64) > shallow_only);
        let oversize: CoefficientCache<BcdFKey, f64> =
            CoefficientCache::new(usize::MAX, shallow_only);
        oversize.insert(0, bcd_key, 1.0);
        assert_eq!(oversize.tier_stats().entries, 0);
        assert_eq!(oversize.tier_stats().bytes, 0);
        assert_eq!(oversize.tier_stats().evictions, 1);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn cgc_tier_charges_storage_bytes_and_evicts_by_bytes() {
        use super::cgc_cache::CgcKey;
        use crate::lie::sun::{cgc, Cgc, Irrep};
        use std::sync::Arc;
        let irr = |d: &[i64]| Irrep::from_dynkin(d).unwrap();
        // Two real CGCs of different sizes.
        let a = Arc::new(cgc(&irr(&[1, 0]), &irr(&[0, 1]), &irr(&[1, 1])).unwrap()); // 3⊗3̄→8
        let b = Arc::new(cgc(&irr(&[1, 1]), &irr(&[1, 1]), &irr(&[1, 1])).unwrap()); // 8⊗8→8, OM=2

        // The tier charge is exactly the sparse storage bytes (plus key copies).
        assert!(a.value_bytes() >= a.storage_bytes());
        assert_eq!(a.value_bytes(), a.storage_bytes());

        // A local CGC-typed cache with a retained-charge budget that fits only
        // one entry must evict or reject a value when the second is inserted.
        let ka = (irr(&[1, 0]), irr(&[0, 1]), irr(&[1, 1]));
        let kb = (irr(&[1, 1]), irr(&[1, 1]), irr(&[1, 1]));
        let budget = a.value_bytes().max(b.value_bytes()) + ka.key_bytes() + 8;
        let c: CoefficientCache<CgcKey, Arc<Cgc>> = CoefficientCache::new(1_000_000, budget);
        c.insert(0, ka.clone(), a);
        c.insert(0, kb, b);
        let (_, _, entries, bytes) = c.snapshot();
        assert!(entries <= 1, "retained-charge cap kept {entries} entries");
        assert!(
            bytes <= budget,
            "retained charge exceeded cap: {bytes} > {budget}"
        );
        assert!(c.tier_stats().evictions > 0);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn sun_f_tier_charges_block_bytes_and_evicts_by_bytes() {
        use super::sun_f_cache::SunFKey;
        use crate::lie::sun::{f_symbol, FBlock, Irrep};
        let irr = |d: &[i64]| Irrep::from_dynkin(d).unwrap();
        // A real SU(3) F block (8⊗8→8 family: the 2×2×2×2 OM=2 block).
        let e8 = irr(&[1, 1]);
        let a = f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
        // A multiplicity-free (smaller, 1⁴) block: a=1 forces e=3, f=d=6.
        let triv = Irrep::trivial(3).unwrap();
        let three = irr(&[1, 0]);
        let six = irr(&[2, 0]);
        let b = f_symbol(&triv, &three, &three, &six, &three, &six).unwrap();

        // Charge is the data bytes plus the block shell.
        assert_eq!(
            a.value_bytes(),
            ARC_COUNTS_BYTES + std::mem::size_of_val(a.data()) + std::mem::size_of::<FBlock>()
        );
        assert!(a.value_bytes() > b.value_bytes(), "2⁴ block > 1⁴ block");

        // Budget for one entry: insertion must evict or reject a value.
        let ka = (
            e8.clone(),
            e8.clone(),
            e8.clone(),
            e8.clone(),
            e8.clone(),
            e8.clone(),
        );
        let kb = (
            triv.clone(),
            three.clone(),
            three.clone(),
            six.clone(),
            three.clone(),
            six.clone(),
        );
        let budget = a.value_bytes() + ka.key_bytes() + 8;
        let c: CoefficientCache<SunFKey, FBlock> = CoefficientCache::new(1_000_000, budget);
        c.insert(0, ka.clone(), a);
        c.insert(0, kb, b);
        let (_, _, entries, bytes) = c.snapshot();
        assert!(entries <= 1, "retained-charge cap kept {entries} entries");
        assert!(
            bytes <= budget,
            "retained charge exceeded cap: {bytes} > {budget}"
        );
        assert!(c.tier_stats().evictions > 0);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn bcd_products_charge_deep_labels_and_respect_the_shared_budget() {
        use crate::lie::bcd::{Irrep, Series};
        use bcd_cgc_cache::{BcdKey, BcdValue};
        let a = Irrep::from_dynkin(Series::C, &[1, 0]).unwrap();
        let key = BcdKey::Product(a.clone(), a.clone());
        let product = crate::lie::bcd::directproduct(&a, &a)
            .unwrap()
            .into_iter()
            .collect::<Arc<[_]>>();
        let value = BcdValue::Product(Arc::clone(&product));
        let expected = std::mem::size_of::<BcdValue>()
            + ARC_COUNTS_BYTES
            + std::mem::size_of_val(product.as_ref())
            + product
                .iter()
                .map(|(c, _)| ARC_COUNTS_BYTES + std::mem::size_of_val(c.two_partition()))
                .sum::<usize>();
        assert_eq!(value.value_bytes(), expected);
        let charge = entry_charge(&key, &value);
        let local = CoefficientCache::new(16, charge - 1);
        let BcdValue::Product(returned) = local.insert(0, key, value) else {
            panic!("product value")
        };
        assert!(Arc::ptr_eq(&product, &returned));
        assert_eq!(local.tier_stats().entries, 0);
        assert_eq!(local.tier_stats().evictions, 1);
        local.reset();
        assert_eq!(local.tier_stats(), TierStats::default());
    }

    #[test]
    fn reset_clears_entries_and_counters() {
        let c: CoefficientCache<u32, SignedSqrtRational> = CoefficientCache::new(16, 1 << 20);
        c.get_or_compute(1, || val(1));
        c.get_or_compute(1, || val(1));
        c.reset();
        let (hits, misses, entries, bytes) = c.snapshot();
        assert_eq!((hits, misses, entries, bytes), (0, 0, 0, 0));
    }

    #[test]
    fn evictions_counted_on_entry_bound() {
        // Cap 3 entries, insert 5 distinct keys: exactly 2 entries evicted.
        let c: CoefficientCache<u32, SignedSqrtRational> = CoefficientCache::new(3, 3 << 28);
        for k in 0..5u32 {
            c.get_or_compute(k, || val(k as i64 + 1));
        }
        let ts = c.tier_stats();
        assert_eq!(ts.entries, 3);
        assert_eq!(ts.evictions, 2, "5 inserts over a cap of 3 evict exactly 2");
    }

    #[test]
    fn evictions_counted_on_byte_bound() {
        // Byte budget for ~2 entries: filling 20 forces many byte-driven evictions.
        let per = entry_charge(&0u32, &val(1));
        let c: CoefficientCache<u32, SignedSqrtRational> =
            CoefficientCache::new(1_000_000, per * 2 + per / 2);
        for k in 0..20u32 {
            c.get_or_compute(k, || val(k as i64 + 1));
        }
        assert!(
            c.tier_stats().evictions > 0,
            "retained-charge pressure must count evictions"
        );
    }

    #[test]
    fn oversize_entry_counts_as_eviction() {
        // Retained-charge cap smaller than any single entry: reject the entry
        // without displacing an older retained value; count the rejection.
        let c: CoefficientCache<u32, SignedSqrtRational> = CoefficientCache::new(1_000_000, 1);
        c.get_or_compute(7, || val(7));
        let ts = c.tier_stats();
        assert_eq!(ts.entries, 0, "oversize entry is not retained");
        assert_eq!(ts.bytes, 0);
        assert_eq!(ts.evictions, 1, "a rejected oversize entry counts");
    }

    #[test]
    fn reset_zeroes_evictions() {
        let c: CoefficientCache<u32, SignedSqrtRational> = CoefficientCache::new(1, 1 << 30);
        for k in 0..5u32 {
            c.get_or_compute(k, || val(k as i64 + 1));
        }
        assert!(c.tier_stats().evictions > 0, "precondition: some evictions");
        c.reset();
        assert_eq!(
            c.tier_stats(),
            TierStats::default(),
            "reset zeroes every field"
        );
    }

    #[test]
    fn concurrent_mixed_hit_miss_equals_sequential() {
        let c: Arc<CoefficientCache<u32, SignedSqrtRational>> =
            Arc::new(CoefficientCache::new(1 << 20, 1 << 30));
        let keys: Vec<u32> = (0..64).collect();
        // Reference: sequential fill.
        let seq: Vec<SignedSqrtRational> = keys.iter().map(|&k| val(k as i64 + 1)).collect();

        let mut handles = Vec::new();
        for t in 0..8u32 {
            let c = Arc::clone(&c);
            handles.push(std::thread::spawn(move || {
                let mut out = Vec::new();
                // Each thread walks all keys (mix of first-miss and later-hit),
                // offset so threads interleave differently.
                for i in 0..64u32 {
                    let k = (i + t) % 64;
                    out.push((k, c.get_or_compute(k, || val(k as i64 + 1))));
                }
                out
            }));
        }
        for h in handles {
            for (k, got) in h.join().unwrap() {
                assert_eq!(got, seq[k as usize], "thread value diverged at key {k}");
            }
        }
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn borrowed_generated_views_hit_before_owning_keys() {
        use super::bcd_cgc_cache::{BcdKey, BcdRef};
        use crate::lie::bcd::{Irrep as BcdIrrep, Series};
        use crate::lie::sun::{Irrep, SunProductKey, SunProductRef};

        fn hit<K: Clone + Eq + Hash + CacheKeyCharge, Q: Hash + quick_cache::Equivalent<K>>(
            owned: K,
            borrowed: &Q,
        ) {
            let cache = CoefficientCache::new(16, 1 << 20);
            cache.get_or_compute(owned, || 1.0f64);
            assert_eq!(
                cache.get_or_try_compute::<_, ()>(
                    borrowed,
                    || panic!("a warm borrowed hit must not build an owned key"),
                    || panic!("a warm borrowed hit must not compute")
                ),
                Ok(1.0)
            );
        }
        let a = Irrep::trivial(3).unwrap();
        let b = Irrep::from_dynkin(&[1, 0]).unwrap();
        hit((a.clone(), b.clone(), b.clone()), &Triple(&a, &b, &b));
        hit(
            (
                a.clone(),
                b.clone(),
                a.clone(),
                b.clone(),
                a.clone(),
                b.clone(),
            ),
            &Sextuple(&a, &b, &a, &b, &a, &b),
        );
        hit(SunProductKey::new(&a, &b), &SunProductRef::Product(&a, &b));
        hit(SunProductKey::Dual(a.clone()), &SunProductRef::Dual(&a));
        let a = BcdIrrep::trivial(Series::B, 2).unwrap();
        let b = BcdIrrep::from_dynkin(Series::B, &[1, 0]).unwrap();
        hit(
            (
                a.clone(),
                b.clone(),
                a.clone(),
                b.clone(),
                a.clone(),
                b.clone(),
            ),
            &Sextuple(&a, &b, &a, &b, &a, &b),
        );
        hit(
            BcdKey::Product(a.clone(), b.clone()),
            &BcdRef::Product(&a, &b),
        );
        hit(
            BcdKey::Cgc(a.clone(), b.clone(), b.clone()),
            &BcdRef::Cgc(&a, &b, &b),
        );
        hit(BcdKey::Dual(a.clone()), &BcdRef::Dual(&a));
    }

    #[test]
    fn oversize_rejection_preserves_retained_values() {
        let cache = CoefficientCache::new(16, 512);
        let small = val(1);
        cache.get_or_compute(1u32, || small.clone());
        let before = cache.tier_stats();
        let large = SignedSqrtRational::from_prefactor_radical(
            Ratio::from(BigInt::from(1)),
            Ratio::new(BigInt::from(1) << 8192, BigInt::from(1)),
        );
        assert!(entry_charge(&2u32, &large) > 512);
        assert_eq!(cache.get_or_compute(2u32, || large.clone()), large);
        let after = cache.tier_stats();
        assert_eq!((after.entries, after.bytes), (before.entries, before.bytes));
        assert_eq!(after.evictions, before.evictions + 1);
        assert_eq!(cache.get(&1), Some(small));
    }

    #[test]
    fn single_shard_nondivisible_budget_and_trim_remain_hard_ceilings() {
        let cache = CoefficientCache::new(20_000, 100_003);
        assert_eq!(cache.inner.num_shards(), 1);
        assert_eq!(cache.inner.capacity(), 100_003);
        for key in 0..10_000u32 {
            cache.get_or_compute(key, || f64::from(key));
        }
        let before = cache.tier_stats();
        assert!(before.bytes <= 100_003 && before.entries <= 20_000);
        assert!(before.evictions > 0);
        let report = cache.trim_to(12_347);
        assert!(report.3 <= 12_347);
        assert_eq!(report.0 + report.2, before.entries);
        assert_eq!(report.1 + report.3, before.bytes);
        assert_eq!(
            cache.inner.iter().map(|(_, v)| v.bytes).sum::<usize>(),
            report.3
        );
    }

    #[test]
    fn borrowed_f_lookup_distinguishes_phase_even_when_hashes_collide() {
        use crate::lie::su2::canonical_regge_6j;
        use std::hash::BuildHasherDefault;
        #[derive(Default)]
        struct CollisionHasher;
        impl std::hash::Hasher for CollisionHasher {
            fn finish(&self) -> u64 {
                0
            }
            fn write(&mut self, _: &[u8]) {}
        }
        let cache = quick_cache::sync::Cache::with(
            16,
            16,
            quick_cache::UnitWeighter,
            BuildHasherDefault::<CollisionHasher>::default(),
            quick_cache::sync::DefaultLifecycle::default(),
        );
        let regge = canonical_regge_6j(1, 1, 0, 1, 1, 0).unwrap();
        let positive = FKey::new(regge, 1, false);
        let negative = FKey::new(regge, 1, true);
        cache.insert(positive, 0.5f64);
        cache.insert(negative, -0.5f64);
        // &FKey uses the same borrowed Equivalent path as the base F tier.
        assert_eq!(cache.get(&positive), Some(0.5));
        assert_eq!(cache.get(&negative), Some(-0.5));
    }

    #[test]
    fn failed_computation_is_retryable_and_does_not_displace_values() {
        let cache = CoefficientCache::new(16, 1024);
        cache.get_or_compute(1u32, || 1.0f64);
        assert_eq!(
            cache.get_or_try_compute(&2, || 2, || Err::<f64, _>("failed")),
            Err("failed")
        );
        assert_eq!(cache.tier_stats().entries, 1);
        assert_eq!(cache.get_or_compute(2, || 2.0), 2.0);
        let stats = cache.tier_stats();
        assert_eq!((stats.entries, stats.misses, stats.evictions), (2, 2, 0));
        let oversize = CoefficientCache::new(16, 1);
        assert_eq!(oversize.get_or_compute(1u32, || 1.0f64), 1.0);
        assert_eq!(
            (
                oversize.tier_stats().entries,
                oversize.tier_stats().evictions
            ),
            (0, 1)
        );
    }

    #[test]
    fn policy_observation_races_configuration_once() {
        let cell = Arc::new(OnceLock::new());
        let configured = CoefficientCacheBudgets::disabled();
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let observer_cell = Arc::clone(&cell);
        let observer_barrier = Arc::clone(&barrier);
        let observer = std::thread::spawn(move || {
            observer_barrier.wait();
            *effective_budgets_in(&observer_cell)
        });
        barrier.wait();
        let configured_result = configure_budgets_in(&cell, configured);
        let observed = observer.join().unwrap();
        match configured_result {
            Ok(()) => assert_eq!(observed, configured),
            Err(CacheBudgetError::AlreadyInitialized) => {
                assert_eq!(observed, CoefficientCacheBudgets::default())
            }
            Err(error) => panic!("unexpected configuration result: {error}"),
        }
    }

    /// Leader blocked in generation while one waiter joins its placeholder;
    /// `during` runs before the leader resumes. quick_cache keeps its waiter
    /// list private, so parking is observed afterwards through `probe` rather
    /// than awaited: an attempt whose waiter came too late is repeated with a
    /// longer head start.
    fn with_parked_waiter(
        max_bytes: usize,
        leader: impl Fn() -> f64 + Send + Sync + 'static,
        during: impl Fn(&CoefficientCache<u32, f64>),
        probe: impl Fn(&Probes) -> bool,
    ) -> (
        Arc<CoefficientCache<u32, f64>>,
        std::thread::Result<f64>,
        f64,
    ) {
        use std::sync::mpsc;
        let leader = Arc::new(leader);
        for attempt in 0..16 {
            let cache = Arc::new(CoefficientCache::new(16, max_bytes));
            let (started_tx, started_rx) = mpsc::channel();
            let (resume_tx, resume_rx) = mpsc::channel::<()>();
            let leader_thread = {
                let (cache, leader) = (Arc::clone(&cache), Arc::clone(&leader));
                std::thread::spawn(move || {
                    cache.get_or_compute(1, || {
                        started_tx.send(()).unwrap();
                        resume_rx.recv().unwrap();
                        leader()
                    })
                })
            };
            started_rx.recv().unwrap();
            let waiter_computed = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let waiter = {
                let (cache, computed) = (Arc::clone(&cache), Arc::clone(&waiter_computed));
                std::thread::spawn(move || {
                    cache.get_or_compute(1, || {
                        computed.store(true, Ordering::Relaxed);
                        f64::NAN
                    })
                })
            };
            std::thread::sleep(std::time::Duration::from_micros(100 << attempt));
            during(&cache);
            resume_tx.send(()).unwrap();
            let leader_result = leader_thread.join();
            let waited = waiter.join().unwrap();
            if probe(&cache.probes) {
                // A waiter that joined or was handed a resident key never computes.
                assert!(!waiter_computed.load(Ordering::Relaxed));
                return (cache, leader_result, waited);
            }
        }
        panic!("the waiter never parked on the leader's placeholder");
    }

    fn joined(probes: &Probes) -> bool {
        probes.joined.load(Ordering::Relaxed) == 1
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn waiters_receive_a_sibling_published_winner() {
        let (cache, leader, waiter) = with_parked_waiter(
            1024,
            || 2.0,
            |cache| assert_eq!(cache.insert(cache.epoch(), 1, 1.0), 1.0),
            joined,
        );
        assert_eq!((leader.unwrap(), waiter), (1.0, 1.0));
        let stats = cache.tier_stats();
        assert_eq!((stats.entries, stats.misses, stats.hits), (1, 1, 2));
        assert_eq!(stats.bytes, entry_charge(&1u32, &1.0f64));
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn handed_off_guard_delivers_a_resident_winner_without_computing() {
        let (cache, leader, waiter) = with_parked_waiter(
            1024,
            || panic!("generation panicked"),
            |cache| assert_eq!(cache.insert(cache.epoch(), 1, 1.0), 1.0),
            // The leader's guard plus the one it handed to the parked waiter.
            |probes| probes.guards.load(Ordering::Relaxed) == 2,
        );
        assert!(leader.is_err());
        assert_eq!(waiter, 1.0);
        let stats = cache.tier_stats();
        assert_eq!((stats.entries, stats.misses, stats.hits), (1, 1, 1));
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn seeding_lookups_share_one_computation_and_never_wait_on_each_other() {
        use std::sync::Barrier;
        // Phase 0: every thread requests 1. Phase 1: requests of 3 and of its
        // reverse 4 race; each seeds the other's key after its own guard.
        for phase in 0..2u32 {
            let cache = Arc::new(CoefficientCache::new(16, 1024));
            let start = Arc::new(Barrier::new(8));
            let computations = Arc::new(AtomicU64::new(0));
            let threads = (0..8u32)
                .map(|t| {
                    let (cache, start, computations) = (
                        Arc::clone(&cache),
                        Arc::clone(&start),
                        Arc::clone(&computations),
                    );
                    let key = if phase == 1 && t % 2 == 1 {
                        4
                    } else {
                        1 + 2 * phase
                    };
                    let reverse = if phase == 0 { 2 } else { 7 - key };
                    std::thread::spawn(move || {
                        start.wait();
                        let value = cache.get_or_compute_seeding(
                            &key,
                            || key,
                            || {
                                computations.fetch_add(1, Ordering::Relaxed);
                                std::thread::sleep(std::time::Duration::from_millis(5));
                                (f64::from(reverse), (reverse, f64::from(key)))
                            },
                        );
                        assert_eq!(value, f64::from(reverse));
                    })
                })
                .collect::<Vec<_>>();
            for thread in threads {
                thread.join().unwrap();
            }
            assert!(computations.load(Ordering::Relaxed) <= 1 + u64::from(phase));
            assert_eq!(
                cache.get(&(if phase == 0 { 2 } else { 4 })),
                Some(f64::from(1 + 2 * phase))
            );
            assert_eq!(cache.tier_stats().entries, 2);
        }
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn reverse_seeds_are_publications_not_lookups() {
        let cache = CoefficientCache::<u32, f64>::new(16, 1024);
        let lookup = |key: u32, dual: u32| {
            cache.get_or_compute_seeding(&key, || key, || (f64::from(dual), (dual, f64::from(key))))
        };
        let counters = || {
            let stats = cache.tier_stats();
            (stats.entries, stats.misses, stats.hits)
        };
        assert_eq!(lookup(1, 2), 2.0);
        assert_eq!(
            counters(),
            (2, 2, 0),
            "cold request publishes both directions"
        );
        assert_eq!((lookup(1, 2), lookup(2, 1)), (2.0, 1.0));
        assert_eq!(counters(), (2, 2, 2), "both directions then hit");
        assert_eq!(lookup(3, 3), 3.0);
        assert_eq!(
            counters(),
            (3, 3, 2),
            "a self-dual seed finds its own publication"
        );
        assert_eq!(lookup(3, 3), 3.0);
        assert_eq!(counters(), (3, 3, 3));
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn reset_between_seeding_computation_and_publication_retains_nothing() {
        let cache = CoefficientCache::<u32, f64>::new(16, 1024);
        let value = cache.get_or_compute_seeding(
            &1,
            || 1,
            || {
                cache.reset();
                (2.0, (2, 1.0))
            },
        );
        assert_eq!(value, 2.0);
        let stats = cache.tier_stats();
        assert_eq!((stats.entries, stats.bytes), (0, 0));
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn reset_during_a_sweep_drops_its_later_sibling_publications() {
        let cache = CoefficientCache::<u32, f64>::new(16, 1024);
        let epoch = cache.epoch();
        cache.reset();
        assert_eq!(cache.insert(epoch, 2, 2.0), 2.0);
        let stats = cache.tier_stats();
        assert_eq!((stats.entries, stats.bytes, stats.misses), (0, 0, 0));
        assert_eq!(cache.insert(cache.epoch(), 2, 2.0), 2.0);
        assert_eq!(cache.tier_stats().entries, 1);
    }

    #[test]
    fn oversize_value_is_still_delivered_to_waiters() {
        let charge = entry_charge(&1u32, &1.0f64);
        let (cache, leader, waiter) = with_parked_waiter(charge - 1, || 3.0, |_| {}, joined);
        assert_eq!((leader.unwrap(), waiter), (3.0, 3.0));
        let stats = cache.tier_stats();
        assert_eq!(
            (stats.entries, stats.bytes, stats.misses, stats.evictions),
            (0, 0, 1, 1)
        );
    }

    #[test]
    fn contended_failed_leader_hands_off_to_exactly_one_new_leader() {
        use std::sync::Barrier;
        let cache = Arc::new(CoefficientCache::new(16, 1024));
        let start = Arc::new(Barrier::new(8));
        let computations = Arc::new(AtomicU64::new(0));
        let results = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let start = Arc::clone(&start);
                let computations = Arc::clone(&computations);
                std::thread::spawn(move || {
                    start.wait();
                    cache.get_or_try_compute(
                        &3u32,
                        || 3,
                        || {
                            let attempt = computations.fetch_add(1, Ordering::Relaxed);
                            std::thread::sleep(std::time::Duration::from_millis(20));
                            if attempt == 0 {
                                Err("first attempt fails")
                            } else {
                                Ok(3.0f64)
                            }
                        },
                    )
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(computations.load(Ordering::Relaxed), 2);
        assert_eq!(results.iter().filter(|r| r.is_err()).count(), 1);
        assert!(results.iter().flatten().all(|&v| v == 3.0));
        let stats = cache.tier_stats();
        assert_eq!((stats.entries, stats.misses, stats.hits), (1, 1, 6));
    }

    #[test]
    fn panicking_computation_releases_its_guard() {
        let cache = CoefficientCache::<u32, f64>::new(16, 1024);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.get_or_compute(5, || panic!("generation panicked"))
        }));
        assert!(panicked.is_err());
        assert_eq!(cache.get_or_compute(5, || 5.0), 5.0);
        let stats = cache.tier_stats();
        assert_eq!((stats.entries, stats.misses, stats.hits), (1, 1, 0));
    }

    #[test]
    fn concurrent_readers_and_writers_share_one_tier_consistently() {
        use std::sync::Barrier;
        let charge = entry_charge(&0u32, &0.0f64);
        let cache = Arc::new(CoefficientCache::new(64, 64 * charge));
        let start = Arc::new(Barrier::new(9));
        let readers = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    for i in 0..20_000u32 {
                        let key = i % 128;
                        assert_eq!(cache.get_or_compute(key, || f64::from(key)), f64::from(key));
                    }
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        for round in 0..200 {
            match round % 3 {
                0 => {
                    cache.trim_to(8 * charge);
                }
                1 => cache.reset(),
                _ => {
                    let stats = cache.tier_stats();
                    assert!(stats.bytes <= 64 * charge);
                    assert_eq!(stats.bytes, stats.entries * charge);
                }
            }
        }
        for reader in readers {
            reader.join().unwrap();
        }
        let stats = cache.tier_stats();
        assert_eq!(stats.entries, cache.inner.len());
        assert_eq!(
            stats.bytes,
            cache.inner.iter().map(|(_, v)| v.bytes).sum::<usize>()
        );

        // Quiescent striped hit counts are exact across threads.
        cache.reset();
        cache.get_or_compute(1, || 1.0);
        let hitters = (0..16)
            .map(|_| {
                let cache = Arc::clone(&cache);
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        assert_eq!(cache.get(&1), Some(1.0));
                    }
                })
            })
            .collect::<Vec<_>>();
        for hitter in hitters {
            hitter.join().unwrap();
        }
        let stats = cache.tier_stats();
        assert_eq!((stats.hits, stats.misses), (16_000, 1));
    }
}
