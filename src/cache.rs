//! Bounded, thread-safe coefficient caches, and the process-local policy that
//! bounds them.
//!
//! Coefficient generation is the expensive step, so `fusion-symbols` caches: the base
//! SU(2) tiers (3j, 6j, derived-F) keyed by canonical Regge classes, and — with
//! `lie-generated` — seven generated tiers (Product, SU(N) CGC / F / R,
//! B/C/D raw+CGC / F / R). Generated duals are owned by their label payloads.
//! Every tier is process-global, byte-weighted Clock-PRO, and independently bounded.
//!
//! # What a consumer needs from this module
//!
//! - **A ceiling.** [`BASE_CACHE_MAX_BYTES`] (192 MiB) and
//!   [`GENERATED_CACHE_MAX_BYTES`] (768 MiB + 128 KiB) are the documented sums
//!   of the per-tier retained-entry charge caps. They bound *cache-owned entry
//!   charge*, not RSS.
//! - **A smaller ceiling, if you want one.** [`configure_cache_budgets`],
//!   once, before first use. Shrink-only.
//! - **Observation.** [`tier_stats`] for one tier; iterate
//!   [`CoefficientCacheTier::ALL`] for every tier in this build.
//! - **Release.** [`trim_to`] (one tier, unspecified removal order) and [`reset`] (all
//!   tiers). Both act on process-global state, so **a library must not call
//!   them** — exactly one component in a process owns that policy.
//!
//! ```
//! use fusion_symbols::cache::{tier_stats, CoefficientCacheTier, BASE_CACHE_MAX_BYTES};
//!
//! let _ = fusion_symbols::lie::su2::wigner_6j(2, 2, 2, 2, 2, 2);
//! let base = [
//!     CoefficientCacheTier::ThreeJ,
//!     CoefficientCacheTier::SixJ,
//!     CoefficientCacheTier::DerivedF,
//! ];
//! let bytes: usize = base.into_iter().map(|tier| tier_stats(tier).bytes).sum();
//! assert!(bytes <= BASE_CACHE_MAX_BYTES);
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
//! Per-tier statistics are exposed via [`tier_stats`]; reset
//! ownership is on [`reset`].

use malachite_base::num::logic::traits::SignificantBits;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, OnceLock};

use crate::exact::SignedSqrtRational;
use crate::lie::su2::{FKey, Regge3j, Regge6j};
#[cfg(feature = "lie-generated")]
use crate::{
    category::{FBlock, RBlock},
    lie::{bcd, sun},
};
#[cfg(feature = "lie-generated")]
use std::sync::Arc;

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
    /// Exact simple-Lie Product retained-charge cap.
    product_bytes: usize,
    #[cfg(feature = "lie-generated")]
    /// SU(N) CGC retained-charge cap.
    sun_cgc_bytes: usize,
    #[cfg(feature = "lie-generated")]
    /// SU(N) F-symbol retained-charge cap.
    sun_f_bytes: usize,
    #[cfg(feature = "lie-generated")]
    /// SU(N) R-symbol retained-charge cap.
    sun_r_bytes: usize,
    #[cfg(feature = "lie-generated")]
    /// Shared B/C/D raw decomposition and completed CGC retained-charge cap.
    bcd_cgc_bytes: usize,
    #[cfg(feature = "lie-generated")]
    /// B/C/D F-symbol retained-charge cap.
    bcd_f_bytes: usize,
    #[cfg(feature = "lie-generated")]
    /// B/C/D R-symbol retained-charge cap.
    bcd_r_bytes: usize,
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
    /// Exact materialized simple-Lie tensor products (all root systems).
    Product,
    #[cfg(feature = "lie-generated")]
    /// SU(N) CGC tier.
    SunCgc,
    #[cfg(feature = "lie-generated")]
    /// SU(N) F-symbol tier.
    SunF,
    #[cfg(feature = "lie-generated")]
    /// SU(N) R-symbol tier.
    SunR,
    #[cfg(feature = "lie-generated")]
    /// B/C/D raw ordered decompositions and completed CGC channels.
    BcdCgc,
    #[cfg(feature = "lie-generated")]
    /// B/C/D F-symbol tier.
    BcdF,
    #[cfg(feature = "lie-generated")]
    /// B/C/D R-symbol tier.
    BcdR,
}

impl CoefficientCacheTier {
    /// Every tier compiled into this build: the base SU(2) tiers, then (with
    /// `lie-generated`) the generated tiers.
    pub const ALL: &'static [Self] = &[
        Self::ThreeJ,
        Self::SixJ,
        Self::DerivedF,
        #[cfg(feature = "lie-generated")]
        Self::Product,
        #[cfg(feature = "lie-generated")]
        Self::SunCgc,
        #[cfg(feature = "lie-generated")]
        Self::SunF,
        #[cfg(feature = "lie-generated")]
        Self::SunR,
        #[cfg(feature = "lie-generated")]
        Self::BcdCgc,
        #[cfg(feature = "lie-generated")]
        Self::BcdF,
        #[cfg(feature = "lie-generated")]
        Self::BcdR,
    ];
}

impl std::fmt::Display for CoefficientCacheTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ThreeJ => "three_j",
            Self::SixJ => "six_j",
            Self::DerivedF => "derived_f",
            #[cfg(feature = "lie-generated")]
            Self::Product => "product",
            #[cfg(feature = "lie-generated")]
            Self::SunCgc => "sun_cgc",
            #[cfg(feature = "lie-generated")]
            Self::SunF => "sun_f",
            #[cfg(feature = "lie-generated")]
            Self::SunR => "sun_r",
            #[cfg(feature = "lie-generated")]
            Self::BcdCgc => "bcd_cgc",
            #[cfg(feature = "lie-generated")]
            Self::BcdF => "bcd_f",
            #[cfg(feature = "lie-generated")]
            Self::BcdR => "bcd_r",
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
            CoefficientCacheTier::Product,
            CoefficientCacheTier::SunCgc,
            CoefficientCacheTier::SunF,
            CoefficientCacheTier::SunR,
            CoefficientCacheTier::BcdCgc,
            CoefficientCacheTier::BcdF,
            CoefficientCacheTier::BcdR,
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
            CoefficientCacheTier::Product => self.product_bytes = bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunCgc => self.sun_cgc_bytes = bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunF => self.sun_f_bytes = bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunR => self.sun_r_bytes = bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdCgc => self.bcd_cgc_bytes = bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdF => self.bcd_f_bytes = bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdR => self.bcd_r_bytes = bytes,
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
            CoefficientCacheTier::Product => self.product_bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunCgc => self.sun_cgc_bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunF => self.sun_f_bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::SunR => self.sun_r_bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdCgc => self.bcd_cgc_bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdF => self.bcd_f_bytes,
            #[cfg(feature = "lie-generated")]
            CoefficientCacheTier::BcdR => self.bcd_r_bytes,
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
            product_bytes: product_cache::PRODUCT_MAX_BYTES,
            #[cfg(feature = "lie-generated")]
            sun_cgc_bytes: cgc_cache::CGC_MAX_BYTES,
            #[cfg(feature = "lie-generated")]
            sun_f_bytes: sun_f_cache::SUN_F_MAX_BYTES,
            #[cfg(feature = "lie-generated")]
            sun_r_bytes: sun_r_cache::SUN_R_MAX_BYTES,
            #[cfg(feature = "lie-generated")]
            bcd_cgc_bytes: bcd_cgc_cache::BCD_CGC_MAX_BYTES,
            #[cfg(feature = "lie-generated")]
            bcd_f_bytes: bcd_f_cache::BCD_F_MAX_BYTES,
            #[cfg(feature = "lie-generated")]
            bcd_r_bytes: bcd_r_cache::BCD_R_MAX_BYTES,
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
            CoefficientCacheTier::Product,
            b.product_bytes,
            maximum.product_bytes,
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
            CoefficientCacheTier::SunR,
            b.sun_r_bytes,
            maximum.sun_r_bytes,
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
        #[cfg(feature = "lie-generated")]
        (
            CoefficientCacheTier::BcdR,
            b.bcd_r_bytes,
            maximum.bcd_r_bytes,
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

/// Aggregate conservative retained-charge cap for the seven generated `lie-generated` tiers
/// (Product, SU(N) CGC, SU(N) F, SU(N) R, B/C/D raw+CGC, B/C/D F, B/C/D R):
/// `768 MiB + 128 KiB` = `128 KiB + 256 MiB + 64 MiB + 64 MiB + 256 MiB + 64 MiB + 64 MiB`.
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
/// Per-tier statistics are exposed via [`tier_stats`]. The internal generator
/// catalog is *not* a value cache (generator state, its own byte budget,
/// per-`LieGroup` lifecycle) and is intentionally excluded from this budget and
/// from these stats.
#[cfg(feature = "lie-generated")]
pub const GENERATED_CACHE_MAX_BYTES: usize = (768 << 20) + (128 << 10);

// Compile-time tie: if any generated-tier retained-charge cap changes, this constant must
// be reconciled in the same edit or the crate stops building (the same drift
// guard the base tiers use). Each cap is `pub(super)` in its tier module.
#[cfg(feature = "lie-generated")]
const _: () = assert!(
    GENERATED_CACHE_MAX_BYTES
        == cgc_cache::CGC_MAX_BYTES
            + product_cache::PRODUCT_MAX_BYTES
            + sun_f_cache::SUN_F_MAX_BYTES
            + sun_r_cache::SUN_R_MAX_BYTES
            + bcd_cgc_cache::BCD_CGC_MAX_BYTES
            + bcd_f_cache::BCD_F_MAX_BYTES
            + bcd_r_cache::BCD_R_MAX_BYTES
);

/// Snapshot of one coefficient-cache tier, returned by [`tier_stats`].
///
/// Entries, bytes, misses and evictions are read under the tier's publication
/// lock; hits are counted concurrently. A sum over several tiers combines
/// independent snapshots, not one atomic whole-cache snapshot: this crate takes
/// no global lock spanning the tiers, since that would serialize independent
/// lookups while each tier's charged-entry cap already holds on its own.
#[non_exhaustive]
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
        let value_limbs = (r.numerator_ref().significant_bits()
            + r.denominator_ref().significant_bits())
        .div_ceil(8) as usize;
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
struct CoefficientCache<K, V> {
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
    attempts: AtomicU64,
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

    fn get<Q: Hash + quick_cache::Equivalent<K> + ?Sized>(&self, key: &Q) -> Option<V> {
        self.inner.get(key).map(|value| self.hit(value))
    }

    #[inline]
    fn get_or_compute(&self, key: K, compute: impl FnOnce() -> V) -> V {
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
    #[inline]
    fn get_or_try_compute<Q: Hash + quick_cache::Equivalent<K> + ?Sized, E>(
        &self,
        key: &Q,
        own: impl FnOnce() -> K,
        compute: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        self.get_or_try_sweep(key, own, |_| compute())
    }

    /// As [`Self::get_or_try_compute`]; `compute` may also publish other
    /// results of the same generation through the supplied [`Publisher`],
    /// whose reset epoch is captured before generating.
    #[inline]
    fn get_or_try_sweep<Q: Hash + quick_cache::Equivalent<K> + ?Sized, E>(
        &self,
        key: &Q,
        own: impl FnOnce() -> K,
        compute: impl FnOnce(&Publisher<'_, K, V>) -> Result<V, E>,
    ) -> Result<V, E> {
        match self.get(key) {
            Some(value) => Ok(value),
            None => self.miss(own, compute),
        }
    }

    /// Kept out of line so a warm hit neither materializes the miss closures'
    /// captures nor pays for the miss path's frame.
    #[cold]
    #[inline(never)]
    fn miss<E>(
        &self,
        own: impl FnOnce() -> K,
        compute: impl FnOnce(&Publisher<'_, K, V>) -> Result<V, E>,
    ) -> Result<V, E> {
        let key = own();
        let guard = match self.guard(&key) {
            Ok(guard) => guard,
            Err(value) => return Ok(self.hit(value)),
        };
        let value = compute(&self.publisher())?;
        // Hold through publication, never during recursive generation.
        let _publication = self.publication.lock().unwrap();
        Ok(self.fill(guard, &key, value))
    }

    /// Single-flight entry for an owned key: `Err` carries a resident or
    /// waited-for value. A sibling publication replaces a placeholder without
    /// filling it, so a guard (possibly handed off) may cover a resident key;
    /// its winner is then delivered to the waiters instead of recomputed.
    fn guard(&self, key: &K) -> Result<Guard<'_, K, V>, Charged<V>> {
        #[cfg(test)]
        self.probes.attempts.fetch_add(1, Ordering::SeqCst);
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

    fn publisher(&self) -> Publisher<'_, K, V> {
        Publisher {
            cache: self,
            epoch: self.epoch.load(Ordering::Relaxed),
        }
    }
}

/// Type-erased controls, so the public lifecycle functions treat every tier's
/// typed store alike.
trait TierControl: Sync {
    fn reset(&self);
    fn trim_to(&self, target_bytes: usize) -> (usize, usize, usize, usize);
    fn tier_stats(&self) -> TierStats;

    #[cfg(test)]
    fn snapshot(&self) -> (u64, u64, usize, usize) {
        let stats = self.tier_stats();
        (stats.hits, stats.misses, stats.entries, stats.bytes)
    }
}

impl<K, V> TierControl for CoefficientCache<K, V>
where
    K: Clone + Eq + Hash + CacheKeyCharge + Send + Sync,
    V: Clone + CacheCharge + Send + Sync,
{
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
}

/// Non-waiting publication sink for one generation's additional results:
/// they replace placeholders without waiting on loaders that may be waiting on
/// a lock this generation holds. Results generated before a later reset are
/// returned but not retained.
#[cfg_attr(not(feature = "lie-generated"), allow(dead_code))]
struct Publisher<'a, K, V> {
    cache: &'a CoefficientCache<K, V>,
    epoch: u64,
}

#[cfg(feature = "lie-generated")]
impl<K: Clone + Eq + Hash + CacheKeyCharge, V: Clone + CacheCharge> Publisher<'_, K, V> {
    fn publish(&self, key: K, value: V) -> V {
        let cache = self.cache;
        let _publication = cache.publication.lock().unwrap();
        if cache.epoch.load(Ordering::Relaxed) != self.epoch {
            return value;
        }
        if let Some(stored) = cache.inner.peek(&key) {
            return stored.value;
        }
        cache.misses.fetch_add(1, Ordering::Relaxed);
        let bytes = entry_charge(&key, &value);
        cache.retained_bytes.fetch_add(bytes, Ordering::Relaxed);
        cache.inner.insert(
            key,
            Charged {
                bytes,
                value: value.clone(),
            },
        );
        value
    }

    /// Nested work inherits this generation's epoch even if reset precedes
    /// acquisition of its own placeholder. The loader must not recurse.
    fn get_or_try_compute<Q: Hash + quick_cache::Equivalent<K> + ?Sized, E>(
        &self,
        key: &Q,
        own: impl FnOnce() -> K,
        compute: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        let cache = self.cache;
        if let Some(value) = cache.get(key) {
            return Ok(value);
        }
        let key = own();
        let guard = match cache.guard(&key) {
            Ok(guard) => guard,
            Err(value) => return Ok(cache.hit(value)),
        };
        let value = compute()?;
        let _publication = cache.publication.lock().unwrap();
        if cache.epoch.load(Ordering::Relaxed) != self.epoch {
            cache.misses.fetch_add(1, Ordering::Relaxed);
            return Ok(value);
        }
        Ok(cache.fill(guard, &key, value))
    }

    /// Uncounted recheck for a generation that waited on an engine lock.
    fn peek<Q: Hash + quick_cache::Equivalent<K> + ?Sized>(&self, key: &Q) -> Option<V> {
        self.cache.inner.peek(key).map(|value| value.value)
    }
}

/// Ordered generated labels, borrowed without cloning their Arc handles.
#[cfg(feature = "lie-generated")]
#[derive(Hash)]
struct Triple<'a, I>(&'a I, &'a I, &'a I);

#[cfg(feature = "lie-generated")]
impl<I: PartialEq> quick_cache::Equivalent<(I, I, I)> for Triple<'_, I> {
    fn equivalent(&self, key: &(I, I, I)) -> bool {
        self.0 == &key.0 && self.1 == &key.1 && self.2 == &key.2
    }
}

#[cfg(feature = "lie-generated")]
#[derive(Hash)]
struct Sextuple<'a, I>(&'a I, &'a I, &'a I, &'a I, &'a I, &'a I);

#[cfg(feature = "lie-generated")]
impl<I: Clone> Sextuple<'_, I> {
    fn to_key(&self) -> (I, I, I, I, I, I) {
        let Self(a, b, c, d, e, f) = *self;
        (
            a.clone(),
            b.clone(),
            c.clone(),
            d.clone(),
            e.clone(),
            f.clone(),
        )
    }
}

/// A generated F tier keyed by the six labels as given.
#[cfg(feature = "lie-generated")]
type FBlockStore<I> = CoefficientCache<(I, I, I, I, I, I), FBlock<f64>>;

/// One F block through `store`: the six labels as given are the key.
#[cfg(feature = "lie-generated")]
#[inline]
fn f_block<I: Clone + Eq + Hash + CacheKeyCharge, E>(
    store: &FBlockStore<I>,
    labels: Sextuple<'_, I>,
    compute: impl FnOnce() -> Result<FBlock<f64>, E>,
) -> Result<FBlock<f64>, E> {
    store.get_or_try_compute(&labels, || labels.to_key(), compute)
}

/// One R block through `store`: the three labels as given are the key.
#[cfg(feature = "lie-generated")]
#[inline]
fn r_block<I: Clone + Eq + Hash + CacheKeyCharge, E>(
    store: &CoefficientCache<(I, I, I), RBlock<f64>>,
    [a, b, c]: [&I; 3],
    compute: impl FnOnce() -> Result<RBlock<f64>, E>,
) -> Result<RBlock<f64>, E> {
    store.get_or_try_compute(
        &Triple(a, b, c),
        || (a.clone(), b.clone(), c.clone()),
        compute,
    )
}

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

/// Exact 3j value of the canonical class `key`.
#[inline]
pub(crate) fn wigner_3j(
    key: Regge3j,
    compute: impl FnOnce() -> SignedSqrtRational,
) -> SignedSqrtRational {
    CACHE_3J.get_or_compute(key, compute)
}

/// Exact 6j value of the canonical class `key`.
#[inline]
pub(crate) fn wigner_6j(
    key: Regge6j,
    compute: impl FnOnce() -> SignedSqrtRational,
) -> SignedSqrtRational {
    CACHE_6J.get_or_compute(key, compute)
}

/// Derived-f64 F-symbol tier. Stores the rounded `f64` F-symbol so a warm
/// hit returns a `Copy` scalar without re-running the bigint `sqrt` in
/// [`SignedSqrtRational::to_f64`]. It is a *presentation* tier over the exact
/// 6j tier (the value authority), never an independent value source: its `f64`
/// is always derived from the exact value, so the two cannot disagree.
static CACHE_F: LazyLock<CoefficientCache<FKey, f64>> = LazyLock::new(|| {
    CoefficientCache::new(DEFAULT_MAX_ENTRIES, effective_budgets().derived_f_bytes)
});

/// Derived SU(2) F-symbol for `key`.
#[inline]
pub(crate) fn su2_f(key: FKey, compute: impl FnOnce() -> f64) -> f64 {
    CACHE_F.get_or_compute(key, compute)
}

/// Bounded, byte-accounted SU(N) CGC cache (`lie-generated`).
///
/// CGC tensors are large and expensive (a full SVD/QR/least-squares pipeline),
/// so unlike the exact 3j/6j tiers this cache is charged by *actual sparse
/// storage bytes* ([`sun::Cgc`] entry vector + labels) and holds
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

    pub(super) static CACHE_CGC: LazyLock<CoefficientCache<CgcKey, Arc<Cgc>>> =
        LazyLock::new(|| {
            CoefficientCache::new(CGC_MAX_ENTRIES, super::effective_budgets().sun_cgc_bytes)
        });
}

/// SU(N) CGC of the ordered triple.
#[cfg(feature = "lie-generated")]
#[inline]
pub(crate) fn sun_cgc<E>(
    s1: &sun::Irrep,
    s2: &sun::Irrep,
    s3: &sun::Irrep,
    compute: impl FnOnce() -> Result<Arc<sun::Cgc>, E>,
) -> Result<Arc<sun::Cgc>, E> {
    cgc_cache::CACHE_CGC.get_or_try_compute(
        &Triple(s1, s2, s3),
        || (s1.clone(), s2.clone(), s3.clone()),
        compute,
    )
}

/// Bounded exact simple-Lie products. Keys omit global form and distinguish root presentations.
#[cfg(feature = "lie-generated")]
mod product_cache {
    use super::{CacheKeyCharge, CoefficientCache};
    use crate::lie::{general::roots::ProductValue, RootSystem};
    use std::sync::LazyLock;
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub(super) struct ProductKey(pub RootSystem, pub Box<[i64]>, pub Box<[i64]>, pub bool);
    #[derive(Hash)]
    pub(super) struct ProductRef<'a>(pub RootSystem, pub &'a [i64], pub &'a [i64], pub bool);
    impl quick_cache::Equivalent<ProductKey> for ProductRef<'_> {
        fn equivalent(&self, k: &ProductKey) -> bool {
            self.0 == k.0 && self.1 == k.1.as_ref() && self.2 == k.2.as_ref() && self.3 == k.3
        }
    }
    impl CacheKeyCharge for ProductKey {
        fn key_bytes(&self) -> usize {
            std::mem::size_of::<Self>()
                .saturating_add(std::mem::size_of_val(self.1.as_ref()))
                .saturating_add(std::mem::size_of_val(self.2.as_ref()))
        }
    }
    pub(super) const PRODUCT_MAX_ENTRIES: usize = 256;
    pub(super) const PRODUCT_MAX_BYTES: usize = 128 << 10;
    pub(super) static CACHE_PRODUCT: LazyLock<CoefficientCache<ProductKey, ProductValue>> =
        LazyLock::new(|| {
            CoefficientCache::new(
                PRODUCT_MAX_ENTRIES,
                super::effective_budgets().product_bytes,
            )
        });
}

/// One form-free, symmetric exact product store; the loader takes no family or catalog lock.
#[cfg(feature = "lie-generated")]
pub(crate) fn product<E>(
    root: crate::lie::RootSystem,
    a: &[i64],
    b: &[i64],
    compute: impl FnOnce() -> Result<crate::lie::general::roots::Multiplicities, E>,
    labels: impl FnOnce(
        crate::lie::general::roots::Multiplicities,
    ) -> Result<crate::lie::general::roots::Product, E>,
) -> Result<crate::lie::general::roots::Product, E> {
    use crate::lie::general::roots::ProductValue;
    use product_cache::{ProductKey, ProductRef, CACHE_PRODUCT};
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    match CACHE_PRODUCT.get_or_try_sweep(
        &ProductRef(root, a, b, false),
        || ProductKey(root, a.into(), b.into(), false),
        |publisher| {
            // The child inherits the label loader's reset epoch, including before its own guard.
            let raw = publisher.get_or_try_compute(
                &ProductRef(root, a, b, true),
                || ProductKey(root, a.into(), b.into(), true),
                || compute().map(ProductValue::Multiplicities),
            )?;
            match raw {
                ProductValue::Multiplicities(x) => labels(x).map(ProductValue::Labels),
                _ => unreachable!("projection is part of the key"),
            }
        },
    )? {
        ProductValue::Labels(x) => Ok(x),
        _ => unreachable!("projection is part of the key"),
    }
}

#[cfg(feature = "lie-generated")]
pub(crate) fn product_multiplicities<E>(
    root: crate::lie::RootSystem,
    a: &[i64],
    b: &[i64],
    compute: impl FnOnce() -> Result<crate::lie::general::roots::Multiplicities, E>,
) -> Result<crate::lie::general::roots::Multiplicities, E> {
    use crate::lie::general::roots::ProductValue;
    use product_cache::{ProductKey, ProductRef, CACHE_PRODUCT};
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    match CACHE_PRODUCT.get_or_try_compute(
        &ProductRef(root, a, b, true),
        || ProductKey(root, a.into(), b.into(), true),
        || compute().map(ProductValue::Multiplicities),
    )? {
        ProductValue::Multiplicities(x) => Ok(x),
        _ => unreachable!("projection is part of the key"),
    }
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
/// In-memory only, same argument as the CGC tier: the values are
/// gauge/algorithm-dependent, so a persisted store would need a version key;
/// keeping it process-local means it is always consistent with the generator.
#[cfg(feature = "lie-generated")]
mod sun_f_cache {
    use super::{CacheCharge, CoefficientCache};
    use crate::{category::FBlock, lie::sun::Irrep};
    use std::sync::LazyLock;

    /// Canonical cache key: the six irrep labels `(a, b, c, d, e, f)`.
    pub(crate) type SunFKey = (Irrep, Irrep, Irrep, Irrep, Irrep, Irrep);

    impl CacheCharge for FBlock<f64> {
        fn value_bytes(&self) -> usize {
            std::mem::size_of::<FBlock<f64>>()
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

    pub(super) static CACHE_SUN_F: LazyLock<CoefficientCache<SunFKey, FBlock<f64>>> =
        LazyLock::new(|| {
            CoefficientCache::new(SUN_F_MAX_ENTRIES, super::effective_budgets().sun_f_bytes)
        });
}

/// SU(N) F block for the six labels as given.
#[cfg(feature = "lie-generated")]
#[inline]
pub(crate) fn sun_f<E>(
    [a, b, c, d, e, f]: [&sun::Irrep; 6],
    compute: impl FnOnce() -> Result<FBlock<f64>, E>,
) -> Result<FBlock<f64>, E> {
    f_block(
        &sun_f_cache::CACHE_SUN_F,
        Sextuple(a, b, c, d, e, f),
        compute,
    )
}

/// Bounded, byte-accounted derived-f64 SU(N) R-symbol cache (`lie-generated`).
///
/// An R block is one sparse join of two CGC, but even with warm CGC a query
/// rebuilds two hash maps over the `m_c = 0` slices and allocates the result,
/// tens of times the cost of a warm F hit. SUNRepresentations
/// (`sector.jl:Rsymbol`, `RCACHE`) caches R blocks for the same reason. Keyed
/// by the **plain ordered triple** `(a, b, c)`: `R^{ab}_c` and `R^{ba}_c` are
/// different blocks, so no label canonicalization applies. Same entry/byte
/// caps, lifecycle and in-memory-only argument as the F tier.
#[cfg(feature = "lie-generated")]
mod sun_r_cache {
    use super::{CacheCharge, CoefficientCache};
    use crate::{category::RBlock, lie::sun::Irrep};
    use std::sync::LazyLock;

    /// Canonical cache key: the ordered irrep labels `(a, b, c)`.
    pub(crate) type SunRKey = (Irrep, Irrep, Irrep);

    impl CacheCharge for RBlock<f64> {
        fn value_bytes(&self) -> usize {
            std::mem::size_of::<RBlock<f64>>()
                + if self.data().len() > 1 {
                    super::ARC_COUNTS_BYTES + std::mem::size_of_val(self.data())
                } else {
                    0
                }
        }
    }

    /// Entry cap; the retained-charge cap is the real backstop.
    const SUN_R_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap (64 MiB), as for F. `pub(super)` for the
    /// [`super::GENERATED_CACHE_MAX_BYTES`] drift assertion.
    pub(super) const SUN_R_MAX_BYTES: usize = 64 << 20;

    pub(super) static CACHE_SUN_R: LazyLock<CoefficientCache<SunRKey, RBlock<f64>>> =
        LazyLock::new(|| {
            CoefficientCache::new(SUN_R_MAX_ENTRIES, super::effective_budgets().sun_r_bytes)
        });
}

/// SU(N) R block for the three labels as given.
#[cfg(feature = "lie-generated")]
#[inline]
pub(crate) fn sun_r<E>(
    labels: [&sun::Irrep; 3],
    compute: impl FnOnce() -> Result<RBlock<f64>, E>,
) -> Result<RBlock<f64>, E> {
    r_block(&sun_r_cache::CACHE_SUN_R, labels, compute)
}

/// Bounded, byte-accounted derived-f64 B/C/D F-symbol cache (`lie-generated`).
///
/// The B/C/D analogue of [`sun_f`]: an F block is the contraction of four
/// catalog-driven CGC, real work even with warm generators, so the derived
/// `[μ,ν,κ,λ]` block is cached. Same design as the SU(N) tier — the **plain
/// ordered six-label key** `(a,b,c,d,e,f)` (no Regge canonicalization exists for
/// GT/sweep-basis F blocks; see the Why-comment in `sun::fr::f_symbol`), the
/// shared `FBlock` [`CacheCharge`] impl, in-memory only (values are
/// gauge/algorithm-dependent, so a persisted store would need a version key).
#[cfg(feature = "lie-generated")]
mod bcd_f_cache {
    use super::CoefficientCache;
    use crate::category::FBlock;
    use crate::lie::bcd::Irrep;
    use std::sync::LazyLock;

    /// Canonical cache key: the six B/C/D irrep labels `(a, b, c, d, e, f)`.
    pub(crate) type BcdFKey = (Irrep, Irrep, Irrep, Irrep, Irrep, Irrep);

    /// Entry cap; the retained-charge cap is the real backstop.
    const BCD_F_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap (64 MiB): F blocks are tiny (a few multiplicity indices).
    /// `pub(super)` for the [`super::GENERATED_CACHE_MAX_BYTES`] drift assertion.
    pub(super) const BCD_F_MAX_BYTES: usize = 64 << 20;

    pub(super) static CACHE_BCD_F: LazyLock<CoefficientCache<BcdFKey, FBlock<f64>>> =
        LazyLock::new(|| {
            CoefficientCache::new(BCD_F_MAX_ENTRIES, super::effective_budgets().bcd_f_bytes)
        });
}

/// B/C/D F block for the six labels as given.
#[cfg(feature = "lie-generated")]
#[inline]
pub(crate) fn bcd_f<E>(
    [a, b, c, d, e, f]: [&bcd::Irrep; 6],
    compute: impl FnOnce() -> Result<FBlock<f64>, E>,
) -> Result<FBlock<f64>, E> {
    f_block(
        &bcd_f_cache::CACHE_BCD_F,
        Sextuple(a, b, c, d, e, f),
        compute,
    )
}

/// Bounded, byte-accounted derived-f64 B/C/D R-symbol cache (`lie-generated`).
///
/// The B/C/D analogue of [`sun_r`]: same ordered-triple key, shared `RBlock`
/// [`CacheCharge`] impl, caps and lifecycle. Like the F tier, its key omits the
/// global form and is not invalidated by a generator-catalog reset.
#[cfg(feature = "lie-generated")]
mod bcd_r_cache {
    use super::CoefficientCache;
    use crate::category::RBlock;
    use crate::lie::bcd::Irrep;
    use std::sync::LazyLock;

    /// Canonical cache key: the ordered B/C/D irrep labels `(a, b, c)`.
    pub(crate) type BcdRKey = (Irrep, Irrep, Irrep);

    /// Entry cap; the retained-charge cap is the real backstop.
    const BCD_R_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap (64 MiB), as for F.
    /// `pub(super)` for the [`super::GENERATED_CACHE_MAX_BYTES`] drift assertion.
    pub(super) const BCD_R_MAX_BYTES: usize = 64 << 20;

    pub(super) static CACHE_BCD_R: LazyLock<CoefficientCache<BcdRKey, RBlock<f64>>> =
        LazyLock::new(|| {
            CoefficientCache::new(BCD_R_MAX_ENTRIES, super::effective_budgets().bcd_r_bytes)
        });
}

/// B/C/D R block for the three labels as given.
#[cfg(feature = "lie-generated")]
#[inline]
pub(crate) fn bcd_r<E>(
    labels: [&bcd::Irrep; 3],
    compute: impl FnOnce() -> Result<RBlock<f64>, E>,
) -> Result<RBlock<f64>, E> {
    r_block(&bcd_r_cache::CACHE_BCD_R, labels, compute)
}

/// Bounded, byte-accounted B/C/D raw decompositions and completed CGC channels.
#[cfg(feature = "lie-generated")]
mod bcd_cgc_cache {
    use super::{CacheCharge, CacheKeyCharge, CoefficientCache};
    use crate::lie::bcd::{CatalogCgc, Decomposition, Irrep};
    use std::sync::{Arc, LazyLock};

    /// Exact unordered pairs, raw ordered pairs, CGC triples, or dual labels.
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
    pub(super) enum BcdKey {
        Cgc(Irrep, Irrep, Irrep),
        RawProduct(Irrep, Irrep),
    }
    #[derive(Hash)]
    pub(super) enum BcdRef<'a> {
        Cgc(&'a Irrep, &'a Irrep, &'a Irrep),
        RawProduct(&'a Irrep, &'a Irrep),
    }
    impl quick_cache::Equivalent<BcdKey> for BcdRef<'_> {
        fn equivalent(&self, key: &BcdKey) -> bool {
            match (self, key) {
                (Self::Cgc(a, b, c), BcdKey::Cgc(x, y, z)) => *a == x && *b == y && *c == z,
                (Self::RawProduct(a, b), BcdKey::RawProduct(x, y)) => *a == x && *b == y,
                _ => false,
            }
        }
    }
    impl CacheKeyCharge for BcdKey {
        fn key_bytes(&self) -> usize {
            std::mem::size_of::<Self>().saturating_add(match self {
                Self::RawProduct(a, b) => a.key_bytes().saturating_add(b.key_bytes()),
                Self::Cgc(a, b, c) => a
                    .key_bytes()
                    .saturating_add(b.key_bytes())
                    .saturating_add(c.key_bytes()),
            })
        }
    }
    #[derive(Clone, Debug)]
    pub(super) enum BcdValue {
        Cgc(Arc<CatalogCgc>),
        Raw(Arc<Decomposition>),
    }
    impl CacheCharge for BcdValue {
        fn value_bytes(&self) -> usize {
            std::mem::size_of::<Self>().saturating_add(match self {
                Self::Cgc(cgc) => cgc.storage_bytes(),
                Self::Raw(raw) => super::ARC_COUNTS_BYTES.saturating_add(raw.storage_bytes()),
            })
        }
    }

    /// Entry cap; the retained-charge cap is the real backstop.
    const BCD_CGC_MAX_ENTRIES: usize = 1 << 16;
    /// Retained-charge cap (256 MiB): dense product isometries are far larger
    /// than an F block, so this tier gets its own generous budget (as the SU(N)
    /// CGC tier).
    /// `pub(super)` for the [`super::GENERATED_CACHE_MAX_BYTES`] drift assertion.
    pub(super) const BCD_CGC_MAX_BYTES: usize = 256 << 20;

    pub(super) static CACHE_BCD_CGC: LazyLock<CoefficientCache<BcdKey, BcdValue>> =
        LazyLock::new(|| {
            CoefficientCache::new(
                BCD_CGC_MAX_ENTRIES,
                super::effective_budgets().bcd_cgc_bytes,
            )
        });
}

/// One B/C/D channel request and its inherited raw/sibling publication epoch.
#[cfg(feature = "lie-generated")]
pub(crate) struct BcdCgcSweep<'a> {
    publisher: &'a Publisher<'a, bcd_cgc_cache::BcdKey, bcd_cgc_cache::BcdValue>,
    a: &'a bcd::Irrep,
    b: &'a bcd::Irrep,
    c: &'a bcd::Irrep,
}

#[cfg(feature = "lie-generated")]
impl BcdCgcSweep<'_> {
    /// The requested channel if another sweep published it meanwhile; call it
    /// after acquiring the lock that serializes sweeps.
    pub(crate) fn resident(&self) -> Option<Arc<bcd::CatalogCgc>> {
        match self
            .publisher
            .peek(&bcd_cgc_cache::BcdRef::Cgc(self.a, self.b, self.c))?
        {
            bcd_cgc_cache::BcdValue::Cgc(cgc) => Some(cgc),
            _ => unreachable!("CGC key stores a CGC"),
        }
    }

    /// Only the catalog's canonical-input path may supply this loader. The
    /// inherited epoch predates the catalog lock and every recursive child.
    pub(crate) fn raw<E>(
        &self,
        a: &bcd::Irrep,
        b: &bcd::Irrep,
        compute: impl FnOnce() -> Result<Arc<bcd::Decomposition>, E>,
    ) -> Result<Arc<bcd::Decomposition>, E> {
        use bcd_cgc_cache::{BcdKey, BcdRef, BcdValue};
        let value = self.publisher.get_or_try_compute(
            &BcdRef::RawProduct(a, b),
            || BcdKey::RawProduct(a.clone(), b.clone()),
            || compute().map(BcdValue::Raw),
        )?;
        let BcdValue::Raw(raw) = value else {
            unreachable!("raw product key stores a decomposition")
        };
        Ok(raw)
    }

    /// Publish another channel of `a ⊗ b` produced by this sweep.
    pub(crate) fn publish(&self, channel: bcd::CatalogCgc) {
        self.publisher.publish(
            bcd_cgc_cache::BcdKey::Cgc(self.a.clone(), self.b.clone(), channel.s3().clone()),
            bcd_cgc_cache::BcdValue::Cgc(Arc::new(channel)),
        );
    }
}

/// B/C/D CGC of the ordered triple.
#[cfg(feature = "lie-generated")]
pub(crate) fn bcd_cgc<E>(
    a: &bcd::Irrep,
    b: &bcd::Irrep,
    c: &bcd::Irrep,
    compute: impl FnOnce(&BcdCgcSweep<'_>) -> Result<Arc<bcd::CatalogCgc>, E>,
) -> Result<Arc<bcd::CatalogCgc>, E> {
    use bcd_cgc_cache::{BcdKey, BcdRef, BcdValue, CACHE_BCD_CGC};
    let value = CACHE_BCD_CGC.get_or_try_sweep(
        &BcdRef::Cgc(a, b, c),
        || BcdKey::Cgc(a.clone(), b.clone(), c.clone()),
        |publisher| {
            let sweep = BcdCgcSweep { publisher, a, b, c };
            compute(&sweep).map(BcdValue::Cgc)
        },
    )?;
    let BcdValue::Cgc(value) = value else {
        unreachable!("CGC key stores a CGC")
    };
    Ok(value)
}

#[cfg(all(test, feature = "lie-generated"))]
pub(crate) fn bcd_cgc_is_resident(a: &bcd::Irrep, b: &bcd::Irrep, c: &bcd::Irrep) -> bool {
    bcd_cgc_cache::CACHE_BCD_CGC
        .inner
        .peek(&bcd_cgc_cache::BcdRef::Cgc(a, b, c))
        .is_some()
}

#[cfg(all(test, feature = "lie-generated"))]
pub(crate) fn with_bcd_test_sweep<T>(
    a: &bcd::Irrep,
    b: &bcd::Irrep,
    c: &bcd::Irrep,
    run: impl FnOnce(&BcdCgcSweep<'_>) -> T,
) -> T {
    let publisher = bcd_cgc_cache::CACHE_BCD_CGC.publisher();
    run(&BcdCgcSweep {
        publisher: &publisher,
        a,
        b,
        c,
    })
}

fn control(tier: CoefficientCacheTier) -> &'static dyn TierControl {
    match tier {
        CoefficientCacheTier::ThreeJ => &*CACHE_3J,
        CoefficientCacheTier::SixJ => &*CACHE_6J,
        CoefficientCacheTier::DerivedF => &*CACHE_F,
        #[cfg(feature = "lie-generated")]
        CoefficientCacheTier::Product => &*product_cache::CACHE_PRODUCT,
        #[cfg(feature = "lie-generated")]
        CoefficientCacheTier::SunCgc => &*cgc_cache::CACHE_CGC,
        #[cfg(feature = "lie-generated")]
        CoefficientCacheTier::SunF => &*sun_f_cache::CACHE_SUN_F,
        #[cfg(feature = "lie-generated")]
        CoefficientCacheTier::SunR => &*sun_r_cache::CACHE_SUN_R,
        #[cfg(feature = "lie-generated")]
        CoefficientCacheTier::BcdCgc => &*bcd_cgc_cache::CACHE_BCD_CGC,
        #[cfg(feature = "lie-generated")]
        CoefficientCacheTier::BcdF => &*bcd_f_cache::CACHE_BCD_F,
        #[cfg(feature = "lie-generated")]
        CoefficientCacheTier::BcdR => &*bcd_r_cache::CACHE_BCD_R,
    }
}

/// Clear the 3j, 6j, and derived-f64 F-symbol caches (and, under `lie-generated`, the
/// simple-Lie Product and SU(N)/B/C/D CGC/F/R caches) and *all* their counters — entries, bytes,
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
///
/// # Examples
///
/// ```
/// use fusion_symbols::cache::{reset, tier_stats, CoefficientCacheTier, TierStats};
///
/// let _ = fusion_symbols::lie::su2::wigner_6j(2, 2, 2, 2, 2, 2);
/// reset(); // application-owned: libraries must not call it
/// for &tier in CoefficientCacheTier::ALL {
///     assert_eq!(tier_stats(tier), TierStats::default());
/// }
/// ```
pub fn reset() {
    for &tier in CoefficientCacheTier::ALL {
        control(tier).reset();
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
///
/// # Examples
///
/// ```
/// use fusion_symbols::cache::{tier_stats, trim_to, CoefficientCacheTier};
///
/// let _ = fusion_symbols::lie::su2::wigner_6j(2, 2, 2, 2, 2, 2);
/// let report = trim_to(CoefficientCacheTier::SixJ, 0); // application-owned
/// assert_eq!(report.remaining_entries, 0);
/// assert_eq!(tier_stats(CoefficientCacheTier::SixJ).entries, 0);
/// ```
pub fn trim_to(tier: CoefficientCacheTier, target_charged_bytes: usize) -> CacheTrimReport {
    let (removed_entries, removed_charged_bytes, remaining_entries, remaining_charged_bytes) =
        control(tier).trim_to(target_charged_bytes);
    CacheTrimReport {
        tier,
        removed_entries,
        removed_charged_bytes,
        remaining_entries,
        remaining_charged_bytes,
    }
}

/// Statistics for one coefficient-cache tier. See [`TierStats`] for its
/// snapshot-consistency contract.
///
/// # Examples
///
/// ```
/// use fusion_symbols::cache::{tier_stats, CoefficientCacheTier};
///
/// let before = tier_stats(CoefficientCacheTier::SixJ);
/// let _ = fusion_symbols::lie::su2::wigner_6j(2, 2, 2, 2, 2, 2);
/// let _ = fusion_symbols::lie::su2::wigner_6j(2, 2, 2, 2, 2, 2);
/// let after = tier_stats(CoefficientCacheTier::SixJ);
/// assert!(after.hits > before.hits); // the second call is served from the cache
/// assert!(after.bytes <= fusion_symbols::cache::cache_budgets().limit(CoefficientCacheTier::SixJ));
/// ```
pub fn tier_stats(tier: CoefficientCacheTier) -> TierStats {
    control(tier).tier_stats()
}

// Serialize unit tests that reset process-global coefficient caches.
#[cfg(test)]
pub(crate) static TEST_CACHE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(all(test, feature = "lie-generated"))]
pub(crate) fn product_multiplicities_are_resident(
    root: crate::lie::RootSystem,
    a: &[i64],
    b: &[i64],
) -> bool {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    product_cache::CACHE_PRODUCT
        .inner
        .peek(&product_cache::ProductRef(root, a, b, true))
        .is_some()
}

#[cfg(all(test, feature = "lie-generated"))]
pub(crate) fn with_publication_locks<T>(f: impl FnOnce() -> T) -> T {
    let _guards = [
        CACHE_3J.publication.lock().unwrap(),
        CACHE_6J.publication.lock().unwrap(),
        CACHE_F.publication.lock().unwrap(),
        product_cache::CACHE_PRODUCT.publication.lock().unwrap(),
        cgc_cache::CACHE_CGC.publication.lock().unwrap(),
        sun_f_cache::CACHE_SUN_F.publication.lock().unwrap(),
        sun_r_cache::CACHE_SUN_R.publication.lock().unwrap(),
        bcd_cgc_cache::CACHE_BCD_CGC.publication.lock().unwrap(),
        bcd_f_cache::CACHE_BCD_F.publication.lock().unwrap(),
        bcd_r_cache::CACHE_BCD_R.publication.lock().unwrap(),
    ];
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use malachite_nz::integer::Integer;
    use malachite_q::Rational;
    use std::sync::Arc;

    fn val(n: i64) -> SignedSqrtRational {
        SignedSqrtRational::from_prefactor_radical(
            Rational::from(1),
            Rational::from_integers(Integer::from(n.unsigned_abs()), Integer::from(1)),
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
                worker_cache.inner.peek(&1).unwrap().value
            })
        });
        started_rx.recv().unwrap();
        assert_eq!(cache.publisher().publish(1, 1.0f64), 1.0);
        assert_eq!(cache.publisher().publish(1, 2.0f64), 1.0);
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
                Rational::from(1),
                Rational::from_integers(Integer::from(1) << 256, Integer::from(1)),
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
        let sun_irrep_bytes = std::mem::size_of::<SunIrrep>() + sun_key.0 .0.heap_bytes();
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
        let bcd_irrep_bytes = std::mem::size_of::<BcdIrrep>() + bcd_key.0 .0.heap_bytes();
        assert_eq!(bcd_key.key_bytes(), 6 * bcd_irrep_bytes);
    }

    #[cfg(feature = "lie-generated")]
    fn product_fixture(
        root: crate::lie::RootSystem,
        label: &[i64],
    ) -> (
        product_cache::ProductKey,
        crate::lie::general::roots::Product,
    ) {
        let data = crate::lie::general::roots::CartanData::new(root);
        let c = data.irrep(label).unwrap();
        (
            product_cache::ProductKey(root, label.into(), label.into(), false),
            crate::lie::general::roots::Product(Arc::from([(c, 1)])),
        )
    }
    #[cfg(feature = "lie-generated")]
    #[test]
    fn product_tier_caps_charge_and_lifecycle() {
        use crate::lie::{general::roots::Product, RootSystem as R};
        use product_cache::{PRODUCT_MAX_BYTES, PRODUCT_MAX_ENTRIES};
        assert_eq!((PRODUCT_MAX_BYTES, PRODUCT_MAX_ENTRIES), (128 << 10, 256));
        // Complex coordinate pairs and an exact dimension above u128 own deep storage.
        for (root, label) in [
            (R::A(5), vec![1000, 0, 0, 0, 1]),
            (R::D(5), vec![0, 0, 0, 2, 0]),
            (R::C(6), vec![1000; 6]),
            (R::E6, vec![1, 0, 0, 0, 0, 0]),
        ] {
            let (key, value) = product_fixture(root, &label);
            let expected_key =
                std::mem::size_of_val(&key) + 2 * label.len() * std::mem::size_of::<i64>();
            assert_eq!(key.key_bytes(), expected_key);
            let expected_value = std::mem::size_of::<Product>()
                + ARC_COUNTS_BYTES
                + std::mem::size_of::<(crate::lie::general::roots::RootIrrep, u32)>()
                + value[0].0.heap_bytes();
            assert_eq!(value.value_bytes(), expected_value);
            let charge = entry_charge(&key, &value);
            assert_eq!(
                charge,
                expected_key + expected_value + std::mem::size_of::<usize>()
            );
            for budget in [0, charge - 1, charge, charge + 1] {
                let cache = CoefficientCache::new(256, budget);
                let got = cache.get_or_compute(key.clone(), || value.clone());
                assert!(got.ptr_eq(&value));
                let stats = cache.tier_stats();
                assert_eq!(stats.entries, usize::from(budget >= charge));
                assert_eq!(stats.bytes, if budget >= charge { charge } else { 0 });
                let owners = Arc::strong_count(&value.0);
                assert_eq!(cache.trim_to(0).0, usize::from(budget >= charge));
                assert_eq!(
                    Arc::strong_count(&value.0),
                    owners - usize::from(budget >= charge)
                );
                assert_eq!(got, value);
                cache.reset();
                assert_eq!(cache.tier_stats(), TierStats::default());
            }
        }
    }
    #[cfg(feature = "lie-generated")]
    #[test]
    fn product_entry_cap_is_256() {
        let (_, value) = product_fixture(crate::lie::RootSystem::A(1), &[0]);
        let local = CoefficientCache::new(256, 128 << 10);
        for n in 0..257 {
            local.get_or_compute(
                product_cache::ProductKey(
                    crate::lie::RootSystem::A(1),
                    Box::new([n]),
                    Box::new([0]),
                    false,
                ),
                || value.clone(),
            );
        }
        assert_eq!(local.tier_stats().entries, 256);
        assert_eq!(local.tier_stats().evictions, 1);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn product_reset_during_parked_load_preserves_external_value() {
        use std::sync::mpsc;
        let (key, value) = product_fixture(crate::lie::RootSystem::E6, &[1, 0, 0, 0, 0, 0]);
        let cache = Arc::new(CoefficientCache::new(256, 128 << 10));
        let (started, loading) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let worker = {
            let cache = cache.clone();
            let value = value.clone();
            std::thread::spawn(move || {
                cache.get_or_compute(key, || {
                    started.send(()).unwrap();
                    released.recv().unwrap();
                    value
                })
            })
        };
        loading.recv().unwrap();
        cache.reset();
        release.send(()).unwrap();
        let result = worker.join().unwrap();
        assert!(result.ptr_eq(&value));
        assert_eq!(cache.tier_stats().entries, 0);
        assert_eq!(result[0].0.dual().dual(), value[0].0);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn concurrent_product_misses_compute_once() {
        use crate::lie::RootSystem as R;
        use std::sync::{mpsc, Barrier};
        for root in [R::A(5), R::D(5), R::E6] {
            let (key, value) = product_fixture(root, &vec![0; root.rank()]);
            let cache = Arc::new(CoefficientCache::new(256, 128 << 10));
            let count = Arc::new(AtomicU64::new(0));
            let start = Arc::new(Barrier::new(9));
            let (loaded, loading) = mpsc::channel();
            let (release, released) = mpsc::channel();
            let first = {
                let cache = cache.clone();
                let key = key.clone();
                let value = value.clone();
                let count = count.clone();
                std::thread::spawn(move || {
                    cache.get_or_compute(key, || {
                        count.fetch_add(1, Ordering::SeqCst);
                        loaded.send(()).unwrap();
                        released.recv().unwrap();
                        value
                    })
                })
            };
            loading.recv().unwrap();
            let workers = (0..8)
                .map(|_| {
                    let cache = cache.clone();
                    let key = key.clone();
                    let value = value.clone();
                    let start = start.clone();
                    let count = count.clone();
                    std::thread::spawn(move || {
                        start.wait();
                        cache.get_or_compute(key, || {
                            count.fetch_add(1, Ordering::SeqCst);
                            value
                        })
                    })
                })
                .collect::<Vec<_>>();
            start.wait();
            // All readers must join the parked load before publication.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while cache.probes.attempts.load(Ordering::SeqCst) != 9 {
                assert!(std::time::Instant::now() < deadline);
                std::thread::yield_now();
            }
            release.send(()).unwrap();
            assert!(first.join().unwrap().ptr_eq(&value));
            for worker in workers {
                assert!(worker.join().unwrap().ptr_eq(&value));
            }
            assert_eq!(count.load(Ordering::SeqCst), 1);
            assert_eq!(cache.tier_stats().entries, 1);
            assert_eq!(cache.tier_stats().hits, 8);
        }
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
        cache.publisher().publish(first.clone(), 1.0);
        cache.publisher().publish(second, 2.0);
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
        oversize.publisher().publish(bcd_key, 1.0);
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
        c.publisher().publish(ka.clone(), a);
        c.publisher().publish(kb, b);
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
        use crate::{
            category::FBlock,
            lie::sun::{f_symbol, Irrep},
        };
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
            ARC_COUNTS_BYTES + std::mem::size_of_val(a.data()) + std::mem::size_of::<FBlock<f64>>()
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
        let c: CoefficientCache<SunFKey, FBlock<f64>> = CoefficientCache::new(1_000_000, budget);
        c.publisher().publish(ka.clone(), a);
        c.publisher().publish(kb, b);
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
    fn sun_r_tier_charges_block_bytes_and_evicts_by_bytes() {
        use super::sun_r_cache::SunRKey;
        use crate::{
            category::RBlock,
            lie::sun::{r_symbol, Irrep},
        };
        let irr = |d: &[i64]| Irrep::from_dynkin(d).unwrap();
        // The 2×2 octet block shares an Arc; a 1×1 block is inline.
        let e8 = irr(&[1, 1]);
        let a = r_symbol(&e8, &e8, &e8).unwrap();
        let (three, bar) = (irr(&[1, 0]), irr(&[0, 1]));
        let b = r_symbol(&three, &bar, &e8).unwrap();
        assert_eq!(
            a.value_bytes(),
            ARC_COUNTS_BYTES + std::mem::size_of_val(a.data()) + std::mem::size_of::<RBlock<f64>>()
        );
        assert_eq!(b.value_bytes(), std::mem::size_of::<RBlock<f64>>());

        let ka: SunRKey = (e8.clone(), e8.clone(), e8.clone());
        let kb: SunRKey = (three, bar, e8);
        let budget = a.value_bytes() + ka.key_bytes() + 8;
        let c: CoefficientCache<SunRKey, RBlock<f64>> = CoefficientCache::new(1_000_000, budget);
        c.publisher().publish(ka, a);
        c.publisher().publish(kb, b);
        let (_, _, entries, bytes) = c.snapshot();
        assert!(entries <= 1, "retained-charge cap kept {entries} entries");
        assert!(
            bytes <= budget,
            "retained charge exceeded cap: {bytes} > {budget}"
        );
        assert!(c.tier_stats().evictions > 0);
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
    #[ignore = "release measurement: retained raw bytes and actual decomposition calls"]
    fn bcd_raw_retention_measurement() {
        use crate::{
            category::{FusionCategory, GroupRepresentations},
            lie::{GroupId, LieGroup},
        };
        use bcd_cgc_cache::{BcdKey, BcdValue, CACHE_BCD_CGC};
        let _lock = TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for mode in ["public", "f", "r", "f_public", "large"] {
            reset();
            let large = mode == "large";
            let group = LieGroup::new(if large {
                GroupId::sp(4).unwrap()
            } else {
                GroupId::so(5).unwrap()
            })
            .unwrap();
            let a = group.irrep(if large { &[2, 1] } else { &[2, 0] }).unwrap();
            let channels: Vec<_> = group.fusion(&a, &a).unwrap().collect();
            bcd::DECOMPOSITIONS.set(0);
            if mode == "f" || mode == "f_public" {
                group.f_symbol(&a, &a, &a, &a, &a, &a).unwrap();
            }
            if mode == "r" {
                group.r_symbol(&a, &a, &a).unwrap();
            }
            if mode != "f" && mode != "r" {
                for c in channels {
                    group.cgc(&a, &a, &c).unwrap();
                }
            }
            let label = bcd::Irrep::from_dynkin(
                if large {
                    bcd::Series::C
                } else {
                    bcd::Series::B
                },
                if large { &[2, 1] } else { &[2, 0] },
            )
            .unwrap();
            let mut entries = 0;
            let mut bytes = 0;
            let mut queried = 0;
            for (key, value) in CACHE_BCD_CGC.inner.iter() {
                if let BcdValue::Raw(_) = &value.value {
                    entries += 1;
                    bytes += value.bytes;
                    if key == BcdKey::RawProduct(label.clone(), label.clone()) {
                        queried = value.bytes;
                    }
                }
            }
            println!("{mode}: sweeps={} raw_entries={entries} raw_bytes={bytes} queried_raw_bytes={queried}", bcd::DECOMPOSITIONS.get());
            assert!(queried > 0 && bytes <= CACHE_BCD_CGC.tier_stats().bytes);
        }
    }

    #[cfg(feature = "lie-generated")]
    fn raw_fixture() -> (bcd_cgc_cache::BcdKey, bcd_cgc_cache::BcdValue) {
        use crate::{
            category::{FusionCategory, GroupRepresentations},
            lie::{GroupId, LieGroup},
        };
        use bcd_cgc_cache::{BcdKey, BcdRef, CACHE_BCD_CGC};
        let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
        let a = cat.irrep(&[1, 0]).unwrap();
        cat.cgc(&a, &a, &cat.unit()).unwrap();
        let a = &bcd::Irrep::from_dynkin(bcd::Series::C, &[1, 0]).unwrap();
        (
            BcdKey::RawProduct(a.clone(), a.clone()),
            CACHE_BCD_CGC
                .inner
                .peek(&BcdRef::RawProduct(a, a))
                .unwrap()
                .value,
        )
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn raw_charge_eviction_and_external_ownership_are_independent() {
        use bcd_cgc_cache::{BcdKey, BcdValue};
        let _lock = TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (key, raw) = raw_fixture();
        let BcdValue::Raw(blocks) = &raw else {
            unreachable!()
        };
        let charge = entry_charge(&key, &raw);
        assert_eq!(
            charge,
            key.key_bytes()
                + std::mem::size_of::<BcdValue>()
                + ARC_COUNTS_BYTES
                + blocks.storage_bytes()
                + std::mem::size_of::<usize>()
        );
        for budget in [charge, charge - 1, 0] {
            let local = CoefficientCache::new(16, budget);
            let owners = Arc::strong_count(blocks);
            local.get_or_compute(key.clone(), || raw.clone());
            let retained = usize::from(budget == charge);
            assert_eq!(
                (local.tier_stats().entries, local.tier_stats().bytes),
                (retained, charge * retained)
            );
            assert_eq!(Arc::strong_count(blocks), owners + retained);
            assert_eq!(local.trim_to(0), (retained, charge * retained, 0, 0));
            assert_eq!(Arc::strong_count(blocks), owners);
        }
        let BcdKey::RawProduct(a, b) = &key else {
            unreachable!()
        };
        let mut cat = bcd::CanonicalCatalog::new(a.series(), a.rank()).unwrap();
        let one = bcd::Irrep::trivial(a.series(), a.rank()).unwrap();
        let channel = Arc::new(cat.cgc(a, b, &one).unwrap());
        let ckey = BcdKey::Cgc(a.clone(), b.clone(), one);
        let cvalue = BcdValue::Cgc(channel.clone());
        let ccharge = entry_charge(&ckey, &cvalue);
        for remove_raw in [true, false] {
            let local = CoefficientCache::new(16, charge + ccharge);
            let raw_owners = Arc::strong_count(blocks);
            let channel_owners = Arc::strong_count(&channel);
            local.publisher().publish(key.clone(), raw.clone());
            local.publisher().publish(ckey.clone(), cvalue.clone());
            let remove = if remove_raw { &key } else { &ckey };
            let removed = local.inner.remove(remove).unwrap();
            local
                .retained_bytes
                .fetch_sub(removed.1.bytes, Ordering::Relaxed);
            drop(removed);
            assert_eq!(
                local.tier_stats().bytes,
                if remove_raw { ccharge } else { charge }
            );
            assert_eq!(
                Arc::strong_count(blocks),
                raw_owners + usize::from(!remove_raw)
            );
            assert_eq!(
                Arc::strong_count(&channel),
                channel_owners + usize::from(remove_raw)
            );
            local.trim_to(0);
            assert_eq!(channel.copy(0).len(), 16);
            assert_eq!(blocks.blocks().len(), 3);
        }
        // A rejected raw value must not displace an existing useful channel.
        let local = CoefficientCache::new(16, ccharge);
        local.publisher().publish(ckey.clone(), cvalue);
        local.publisher().publish(key, raw);
        assert!(local.get(&ckey).is_some());
        assert_eq!(local.tier_stats().bytes, ccharge);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn nested_raw_lookup_inherits_epoch_before_and_after_child_guard() {
        use bcd_cgc_cache::{BcdKey, BcdValue};
        let _lock = TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (raw_key, raw) = raw_fixture();
        let BcdKey::RawProduct(a, b) = &raw_key else {
            unreachable!()
        };
        let top = BcdKey::Cgc(a.clone(), b.clone(), a.clone());
        let sibling = BcdKey::Cgc(
            a.clone(),
            b.clone(),
            bcd::Irrep::trivial(a.series(), a.rank()).unwrap(),
        );
        for reset_before_guard in [true, false] {
            let local = CoefficientCache::new(16, 1 << 20);
            local
                .get_or_try_sweep(
                    &top,
                    || top.clone(),
                    |parent| {
                        if reset_before_guard {
                            local.reset();
                        }
                        let value = parent.get_or_try_compute(
                            &raw_key,
                            || raw_key.clone(),
                            || {
                                if !reset_before_guard {
                                    local.reset();
                                }
                                Ok::<_, ()>(raw.clone())
                            },
                        )?;
                        assert!(matches!(value, BcdValue::Raw(_)));
                        parent.publish(sibling.clone(), raw.clone());
                        Ok::<_, ()>(raw.clone())
                    },
                )
                .unwrap();
            let stats = local.tier_stats();
            assert_eq!((stats.entries, stats.bytes, stats.misses), (0, 0, 2));
            local
                .publisher()
                .get_or_try_compute(&raw_key, || raw_key.clone(), || Ok::<_, ()>(raw.clone()))
                .unwrap();
            assert_eq!(
                local.tier_stats().entries,
                1,
                "stale child guard must be released"
            );
        }
        // Trimming does not invalidate the inherited epoch.
        let local = CoefficientCache::new(16, 1 << 20);
        let parent = local.publisher();
        parent
            .get_or_try_compute(
                &raw_key,
                || raw_key.clone(),
                || {
                    local.trim_to(0);
                    Ok::<_, ()>(raw.clone())
                },
            )
            .unwrap();
        assert_eq!(local.tier_stats().entries, 1);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn raw_single_flight_errors_and_panics_are_retryable() {
        use std::{
            sync::{mpsc, Barrier},
            time::Duration,
        };
        let _lock = TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        reset();
        let (key, raw) = raw_fixture();
        let local = Arc::new(CoefficientCache::new(16, 1 << 20));
        assert!(local
            .publisher()
            .get_or_try_compute(
                &key,
                || key.clone(),
                || Err::<bcd_cgc_cache::BcdValue, _>(())
            )
            .is_err());
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            local.publisher().get_or_try_compute(
                &key,
                || key.clone(),
                || -> Result<_, ()> { panic!("raw loader panic") },
            )
        }))
        .is_err());
        let start = Arc::new(Barrier::new(2));
        let computations = Arc::new(AtomicUsize::new(0));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let release_rx = Arc::new(std::sync::Mutex::new(release_rx));
        let (done_tx, done_rx) = mpsc::channel();
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let (local, key, raw, start, computations, entered_tx, release_rx, done_tx) = (
                    local.clone(),
                    key.clone(),
                    raw.clone(),
                    start.clone(),
                    computations.clone(),
                    entered_tx.clone(),
                    release_rx.clone(),
                    done_tx.clone(),
                );
                std::thread::spawn(move || {
                    let value = local
                        .publisher()
                        .get_or_try_compute(
                            &key,
                            || {
                                start.wait();
                                key.clone()
                            },
                            || {
                                computations.fetch_add(1, Ordering::Relaxed);
                                entered_tx.send(()).unwrap();
                                release_rx
                                    .lock()
                                    .unwrap()
                                    .recv_timeout(Duration::from_secs(5))
                                    .unwrap();
                                Ok::<_, ()>(raw)
                            },
                        )
                        .unwrap();
                    done_tx.send(value).unwrap();
                })
            })
            .collect();
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        release_tx.send(()).unwrap();
        for _ in 0..2 {
            done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(computations.load(Ordering::Relaxed), 1);
        assert_eq!(local.tier_stats().misses, 1);
        assert_eq!(local.probes.joined.load(Ordering::Relaxed), 1);
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn borrowed_generated_views_hit_before_owning_keys() {
        use super::bcd_cgc_cache::{BcdKey, BcdRef};
        use super::product_cache::{ProductKey, ProductRef};
        use crate::lie::bcd::{Irrep as BcdIrrep, Series};
        use crate::lie::sun::Irrep;

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
        hit(
            ProductKey(
                crate::lie::RootSystem::A(2),
                a.0.dynkin().into(),
                b.0.dynkin().into(),
                false,
            ),
            &ProductRef(
                crate::lie::RootSystem::A(2),
                a.0.dynkin(),
                b.0.dynkin(),
                false,
            ),
        );
        let a = BcdIrrep::from_dynkin(Series::D, &[0, 2, 0]).unwrap();
        let b = a.dual();
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
            BcdKey::Cgc(a.clone(), b.clone(), b.clone()),
            &BcdRef::Cgc(&a, &b, &b),
        );
        hit(
            BcdKey::RawProduct(a.clone(), b.clone()),
            &BcdRef::RawProduct(&a, &b),
        );
    }

    #[test]
    fn oversize_rejection_preserves_retained_values() {
        let cache = CoefficientCache::new(16, 512);
        let small = val(1);
        cache.get_or_compute(1u32, || small.clone());
        let before = cache.tier_stats();
        let large = SignedSqrtRational::from_prefactor_radical(
            Rational::from(1),
            Rational::from_integers(Integer::from(1) << 8192, Integer::from(1)),
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
            |cache| assert_eq!(cache.publisher().publish(1, 1.0), 1.0),
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
            |cache| assert_eq!(cache.publisher().publish(1, 1.0), 1.0),
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
    fn reset_during_a_sweep_drops_its_later_sibling_publications() {
        let cache = CoefficientCache::<u32, f64>::new(16, 1024);
        let requested = cache.get_or_try_sweep(
            &1,
            || 1,
            |sweep| {
                cache.reset();
                assert_eq!(sweep.publish(2, 2.0), 2.0);
                Ok::<_, ()>(1.0)
            },
        );
        assert_eq!(requested, Ok(1.0));
        let stats = cache.tier_stats();
        // The stale requested publication still counts its computation.
        assert_eq!((stats.entries, stats.bytes, stats.misses), (0, 0, 1));
        assert_eq!(cache.publisher().publish(2, 2.0), 2.0);
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
