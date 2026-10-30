//! Bounded, thread-safe coefficient caches, and the process-local policy that
//! bounds them.
//!
//! Coefficient generation is the expensive step, so `fusion-symbols` caches,
//! one tier per data kind: exact Wigner 3j and 6j symbols keyed by canonical
//! Regge classes; F symbols (SU(2) closed-form scalars and Cartan-engine
//! blocks); R blocks; exact simple-Lie tensor products; and the Cartan engine's
//! CGC construction data (irrep modules, completed channels and computed
//! Frobenius–Schur phases). Each tier keys on the presented root system and the
//! ordered labels, has one process-wide limit shared by every group, and is
//! byte-weighted Clock-PRO. Generated duals are owned by their label payloads.
//!
//! # What a consumer needs from this module
//!
//! - **A ceiling.** [`BASE_CACHE_MAX_BYTES`] and, with `lie-generated`,
//!   [`GENERATED_CACHE_MAX_BYTES`] are the documented sums of the per-tier
//!   retained-entry charge caps. They bound *cache-owned entry charge*, not RSS.
//! - **A smaller ceiling, if you want one.** [`configure_cache_budgets`],
//!   once, before first use. Shrink-only.
//! - **Observation.** [`tier_stats`] for one tier; iterate
//!   [`CoefficientCacheTier::ALL`] for every tier.
//! - **Release.** [`trim_to`] (one tier, unspecified removal order) and [`reset`] (all
//!   tiers). Both act on process-global state, so **a library must not call
//!   them** — exactly one component in a process owns that policy.
//!
//! ```
//! use fusion_symbols::cache::{tier_stats, CoefficientCacheTier, BASE_CACHE_MAX_BYTES};
//!
//! let _ = fusion_symbols::lie::su2::wigner_6j(2, 2, 2, 2, 2, 2);
//! let base = [
//!     CoefficientCacheTier::Wigner3j,
//!     CoefficientCacheTier::Wigner6j,
//!     CoefficientCacheTier::FSymbol,
//!     CoefficientCacheTier::Cgc,
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
//! fresh big-rational Racah sum or Cartan-engine channel. WignerSymbols.jl
//! v2.0.0 caches transparently inside `wigner3j`/`wigner6j` (per-kind `LRU`
//! dicts keyed by canonical Regge labels); this crate follows that model but
//! bounds the cache by policy — Julia's caches are effectively unbounded in
//! entries.
//!
//! # Why no gauge/version key component (in-process)
//!
//! A canonical Regge class names exactly one exact symbol value, and a Cartan
//! engine key (presented root system and ordered labels) names exactly one
//! value of the frozen gauge. Every value is a deterministic function of its
//! key, so within one process the key is complete: no gauge tag and no
//! algorithm-version tag can change which value it maps to. Global forms only
//! restrict which labels exist and are not part of any key.
//!
//! # Why this cache must never be persisted to disk
//!
//! Persisting these entries across builds would need an algorithm-version key:
//! a corrected gauge or a widened type could alter the stored bytes for a key.
//! Keeping the cache process-local sidesteps that entirely — it is rebuilt
//! every run, so it can never disagree with the code that filled it.
//!
//! # Resource contract (static partition)
//!
//! Every tier is bounded independently by a per-tier entry and retained-charge
//! cap; the documented aggregate caps are their sums. This is a **static
//! partition, not a dynamic shared pool**: a shared budget would couple
//! eviction across tiers whose entries differ wildly in size (big-rational
//! exact symbols, `f64` scalars, irrep modules) and whose hit patterns are
//! unrelated, for no measured benefit. Because each per-tier charged-entry cap
//! is enforced, the aggregate charged-entry bound holds as a corollary rather
//! than needing global enforcement. The charge covers entries currently owned
//! by the cache. It excludes quick_cache retained capacity and scaffolding,
//! allocator metadata and RSS, transient or external clones, values returned
//! through public APIs, and the working memory of a generation in flight.
//! Per-tier statistics are exposed via [`tier_stats`]; reset ownership is on
//! [`reset`].

use malachite_base::num::logic::traits::SignificantBits;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};

use crate::category::{FBlock, RBlock, SymbolBlock};
use crate::exact::SignedSqrtRational;
use crate::lie::general::cgc::{CgcChannel, Ladders};
use crate::lie::general::roots::{Multiplicities, Product, ProductValue};
use crate::lie::su2::{FKey, Regge3j, Regge6j};
use crate::lie::RootSystem;

/// Default entry cap of the Wigner and F tiers. Matches the reference order of
/// magnitude (WignerSymbols.jl uses `10^6`); the retained-charge cap is the real
/// backstop.
const DEFAULT_MAX_ENTRIES: usize = 1 << 20;

/// Default retained-charge cap of the Wigner, F and R tiers. At the ~O(1)-limb
/// sizes typical of small-label TN work an entry charges well under a
/// kilobyte, so 64 MiB holds a large working set while bounding the
/// conservative charge of cache-owned entries, not allocator-live memory or RSS.
const DEFAULT_MAX_BYTES: usize = 64 << 20;

/// Entry cap of the R tier; the retained-charge cap is the real backstop.
const R_MAX_ENTRIES: usize = 1 << 16;

/// Entry cap of the CGC tier; the retained-charge cap is the real backstop.
const CGC_MAX_ENTRIES: usize = 1 << 16;
/// Retained-charge cap of the CGC tier (256 MiB): irrep modules and completed
/// channels are far larger than a symbol. It holds the whole working set of a
/// pentagon through the G2 `N = 2` vertex `27 ⊗ 27 → 27` (about 150 MiB
/// estimated).
const CGC_MAX_BYTES: usize = 256 << 20;

/// Entry cap of the Product tier.
const PRODUCT_MAX_ENTRIES: usize = 256;
/// Retained-charge cap of the Product tier (128 KiB).
const PRODUCT_MAX_BYTES: usize = 128 << 10;

/// Strong and weak reference counts heading every `Arc` allocation; charged
/// with each shared label, product or channel payload so retained bytes stay
/// conservative.
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
    wigner_3j_bytes: usize,
    /// Exact Wigner 6j retained-charge cap.
    wigner_6j_bytes: usize,
    /// F-symbol retained-charge cap.
    f_symbol_bytes: usize,
    /// R-symbol retained-charge cap.
    r_symbol_bytes: usize,
    /// Exact simple-Lie Product retained-charge cap.
    product_bytes: usize,
    /// Cartan-engine CGC retained-charge cap.
    cgc_bytes: usize,
}

/// One coefficient-cache tier, per data kind.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoefficientCacheTier {
    /// Exact materialized simple-Lie tensor products (all root systems).
    Product,
    /// Cartan-engine CGC construction data: irrep modules, completed channels
    /// and computed Frobenius–Schur phases, for every simple root system
    /// including A(1).
    Cgc,
    /// F symbols: SU(2) closed-form scalars and Cartan-engine blocks.
    FSymbol,
    /// Cartan-engine R blocks. SU(2) R is a closed-form sign and is not cached.
    RSymbol,
    /// Exact Wigner 3j symbols.
    Wigner3j,
    /// Exact Wigner 6j symbols.
    Wigner6j,
}

impl CoefficientCacheTier {
    /// Every tier. Every build compiles all of them; a build without
    /// `lie-generated` never fills [`Self::Product`] or [`Self::RSymbol`].
    pub const ALL: &'static [Self] = &[
        Self::Product,
        Self::Cgc,
        Self::FSymbol,
        Self::RSymbol,
        Self::Wigner3j,
        Self::Wigner6j,
    ];
}

impl std::fmt::Display for CoefficientCacheTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Product => "product",
            Self::Cgc => "cgc",
            Self::FSymbol => "f_symbol",
            Self::RSymbol => "r_symbol",
            Self::Wigner3j => "wigner_3j",
            Self::Wigner6j => "wigner_6j",
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
    /// Disable retention in every tier while preserving evaluation.
    pub fn disabled() -> Self {
        CoefficientCacheTier::ALL
            .iter()
            .fold(Self::default(), |budgets, &tier| {
                budgets.with_limit(tier, 0)
            })
    }

    /// Return this policy with one tier's retained-charge cap replaced.
    pub fn with_limit(mut self, tier: CoefficientCacheTier, bytes: usize) -> Self {
        *self.field(tier) = bytes;
        self
    }

    /// Return one tier's retained-charge cap.
    pub fn limit(&self, tier: CoefficientCacheTier) -> usize {
        let mut copy = *self;
        *copy.field(tier)
    }

    fn field(&mut self, tier: CoefficientCacheTier) -> &mut usize {
        match tier {
            CoefficientCacheTier::Product => &mut self.product_bytes,
            CoefficientCacheTier::Cgc => &mut self.cgc_bytes,
            CoefficientCacheTier::FSymbol => &mut self.f_symbol_bytes,
            CoefficientCacheTier::RSymbol => &mut self.r_symbol_bytes,
            CoefficientCacheTier::Wigner3j => &mut self.wigner_3j_bytes,
            CoefficientCacheTier::Wigner6j => &mut self.wigner_6j_bytes,
        }
    }
}

impl Default for CoefficientCacheBudgets {
    fn default() -> Self {
        Self {
            wigner_3j_bytes: DEFAULT_MAX_BYTES,
            wigner_6j_bytes: DEFAULT_MAX_BYTES,
            f_symbol_bytes: DEFAULT_MAX_BYTES,
            r_symbol_bytes: DEFAULT_MAX_BYTES,
            product_bytes: PRODUCT_MAX_BYTES,
            cgc_bytes: CGC_MAX_BYTES,
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
///     .with_limit(CoefficientCacheTier::Wigner6j, 1 << 20); // 1 MiB of 6j
/// configure_cache_budgets(budgets).unwrap();
/// assert_eq!(cache_budgets().limit(CoefficientCacheTier::Wigner6j), 1 << 20);
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
    let mut total = 0usize;
    for &tier in CoefficientCacheTier::ALL {
        let (requested, limit) = (b.limit(tier), maximum.limit(tier));
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

/// Aggregate conservative retained-charge cap of the tiers a build without
/// `lie-generated` can fill: Wigner 3j, Wigner 6j, F and CGC,
/// `448 MiB` = `64 + 64 + 64 + 256 MiB`.
///
/// This is a **documented static partition**, not a shared budget: each tier is
/// bounded independently by its own per-tier charged-entry cap. The aggregate
/// is therefore a provable corollary — `Σ tier bytes ≤ Σ tier caps` — rather
/// than an enforced global limit.
///
/// The whole-process cap is `BASE_CACHE_MAX_BYTES + GENERATED_CACHE_MAX_BYTES`
/// with `lie-generated`, and `BASE_CACHE_MAX_BYTES` without it, whose groups
/// never fill the Product and R tiers.
pub const BASE_CACHE_MAX_BYTES: usize = 448 << 20;

// Compile-time tie: if a per-tier retained-charge cap changes, the aggregate
// must be reconciled in the same edit or the crate stops building.
const _: () = assert!(BASE_CACHE_MAX_BYTES == 3 * DEFAULT_MAX_BYTES + CGC_MAX_BYTES);

/// Aggregate conservative retained-charge cap of the two tiers only
/// higher-rank (`lie-generated`) groups fill, Product and R:
/// `64 MiB + 128 KiB`. Added to [`BASE_CACHE_MAX_BYTES`] it is the
/// whole-process cap.
///
/// **Unstable: shape may change.** There is no Cargo-feature way to express an
/// instability tier, so this doc label is the marker.
///
/// Like the base cap this is a **static partition, not a shared pool**:
/// `Σ tier charged bytes ≤ Σ tier caps` holds as a corollary without global
/// enforcement.
#[cfg(feature = "lie-generated")]
pub const GENERATED_CACHE_MAX_BYTES: usize = (64 << 20) + (128 << 10);

#[cfg(feature = "lie-generated")]
const _: () = assert!(GENERATED_CACHE_MAX_BYTES == DEFAULT_MAX_BYTES + PRODUCT_MAX_BYTES);

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
    /// The reset epoch, shared by every process-wide tier ([`RESETS`]).
    epoch: &'static AtomicU64,
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

/// The reset epoch of every process-wide tier. [`reset`] makes it odd while
/// it clears the tiers and each tier reset advances it by two, so a generation
/// that overlaps any part of a reset (or starts during one) publishes nothing,
/// in every tier it touches.
static RESETS: AtomicU64 = AtomicU64::new(0);

impl<K: Clone + Eq + Hash + CacheKeyCharge, V: Clone + CacheCharge> CoefficientCache<K, V> {
    /// A process-wide tier, invalidated with the others.
    fn shared(max_entries: usize, max_bytes: usize) -> Self {
        Self::with_epoch(max_entries, max_bytes, &RESETS)
    }

    /// A test-local cache with its own reset epoch.
    #[cfg(test)]
    fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self::with_epoch(
            max_entries,
            max_bytes,
            Box::leak(Box::new(AtomicU64::new(0))),
        )
    }

    fn with_epoch(max_entries: usize, max_bytes: usize, epoch: &'static AtomicU64) -> Self {
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
            epoch,
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
    /// whose reset epoch is captured before its guard.
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
    ///
    /// The generation's epoch is the one running on this thread, else the
    /// current one, read before the guard: a reset at any later point (even
    /// before the guard exists) keeps this value and everything it loads
    /// from being retained.
    #[cold]
    #[inline(never)]
    fn miss<E>(
        &self,
        own: impl FnOnce() -> K,
        compute: impl FnOnce(&Publisher<'_, K, V>) -> Result<V, E>,
    ) -> Result<V, E> {
        let epoch = GENERATION
            .with(std::cell::Cell::get)
            .unwrap_or_else(|| self.epoch.load(Ordering::SeqCst));
        Publisher { cache: self, epoch }.load(own, compute)
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

    #[cfg(test)]
    fn publisher(&self) -> Publisher<'_, K, V> {
        Publisher {
            cache: self,
            epoch: self.epoch.load(Ordering::SeqCst),
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
        self.epoch.fetch_add(2, Ordering::SeqCst);
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

/// The reset epoch of one generation, inherited by its nested loads in every
/// tier: a value generated after a reset began is returned but not retained.
struct Publisher<'a, K, V> {
    cache: &'a CoefficientCache<K, V>,
    epoch: u64,
}

impl<K: Clone + Eq + Hash + CacheKeyCharge, V: Clone + CacheCharge> Publisher<'_, K, V> {
    /// Nested work inherits this generation's epoch even if reset precedes
    /// acquisition of its own placeholder.
    fn get_or_try_compute<Q: Hash + quick_cache::Equivalent<K> + ?Sized, E>(
        &self,
        key: &Q,
        own: impl FnOnce() -> K,
        compute: impl FnOnce() -> Result<V, E>,
    ) -> Result<V, E> {
        match self.cache.get(key) {
            Some(value) => Ok(value),
            None => self.load(own, |_| compute()),
        }
    }

    fn load<E>(
        &self,
        own: impl FnOnce() -> K,
        compute: impl FnOnce(&Self) -> Result<V, E>,
    ) -> Result<V, E> {
        let cache = self.cache;
        let key = own();
        let guard = match cache.guard(&key) {
            Ok(guard) => guard,
            Err(value) => return Ok(cache.hit(value)),
        };
        #[cfg(test)]
        if let Some(hook) = AFTER_GUARD.with(std::cell::Cell::take) {
            hook();
        }
        let value = generation(self.epoch, || compute(self))?;
        // Hold through publication, never during recursive generation.
        let _publication = cache.publication.lock().unwrap();
        if !self.epoch.is_multiple_of(2) || cache.epoch.load(Ordering::SeqCst) != self.epoch {
            cache.misses.fetch_add(1, Ordering::Relaxed);
            return Ok(value);
        }
        Ok(cache.fill(guard, &key, value))
    }
}

thread_local! {
    /// The epoch of the outermost miss running on this thread.
    static GENERATION: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// A test hook run once by the next miss of this thread, between
    /// acquiring its guard and generating.
    pub(crate) static AFTER_GUARD: std::cell::Cell<Option<fn()>> = const { std::cell::Cell::new(None) };
}

/// Runs `compute` as the outermost generation unless one is running on this
/// thread; nested loads in every tier inherit its epoch.
fn generation<T>(epoch: u64, compute: impl FnOnce() -> T) -> T {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            GENERATION.with(|g| g.set(None));
        }
    }
    if GENERATION.with(std::cell::Cell::get).is_some() {
        return compute();
    }
    GENERATION.with(|g| g.set(Some(epoch)));
    let _restore = Restore;
    compute()
}

/// An owned Cartan-engine key: a kind tag, the presented root system and the
/// ordered labels concatenated. Every label of one root system has its rank
/// as length, so the concatenation is unambiguous.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LabelKey {
    kind: u8,
    root: RootSystem,
    labels: Box<[i64]>,
}

impl Hash for LabelKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash_labels(self.kind, self.root, [&self.labels[..]], state);
    }
}

impl CacheKeyCharge for LabelKey {
    fn key_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(std::mem::size_of_val(self.labels.as_ref()))
    }
}

/// The same key, borrowed, so a warm lookup owns no label buffer.
struct LabelRef<'a, const N: usize> {
    kind: u8,
    root: RootSystem,
    parts: [&'a [i64]; N],
}

/// One hash for [`LabelKey`] and [`LabelRef`]: equivalent keys must agree.
fn hash_labels<H: std::hash::Hasher, const N: usize>(
    kind: u8,
    root: RootSystem,
    parts: [&[i64]; N],
    state: &mut H,
) {
    kind.hash(state);
    root.hash(state);
    for x in parts.iter().flat_map(|p| p.iter()) {
        x.hash(state);
    }
}

impl<const N: usize> Hash for LabelRef<'_, N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        hash_labels(self.kind, self.root, self.parts, state);
    }
}

impl<const N: usize> quick_cache::Equivalent<LabelKey> for LabelRef<'_, N> {
    fn equivalent(&self, key: &LabelKey) -> bool {
        self.kind == key.kind
            && self.root == key.root
            && self
                .parts
                .iter()
                .flat_map(|p| p.iter())
                .eq(key.labels.iter())
    }
}

impl<const N: usize> LabelRef<'_, N> {
    fn to_key(&self) -> LabelKey {
        LabelKey {
            kind: self.kind,
            root: self.root,
            labels: self.parts.concat().into_boxed_slice(),
        }
    }
}

impl CacheCharge for FBlock<f64> {
    fn value_bytes(&self) -> usize {
        std::mem::size_of::<FBlock<f64>>()
            + if self.as_slice().len() > 1 {
                ARC_COUNTS_BYTES + std::mem::size_of_val(self.as_slice())
            } else {
                0
            }
    }
}

impl CacheCharge for RBlock<f64> {
    fn value_bytes(&self) -> usize {
        std::mem::size_of::<RBlock<f64>>()
            + if self.as_slice().len() > 1 {
                ARC_COUNTS_BYTES + std::mem::size_of_val(self.as_slice())
            } else {
                0
            }
    }
}

static CACHE_3J: LazyLock<CoefficientCache<Regge3j, SignedSqrtRational>> = LazyLock::new(|| {
    CoefficientCache::shared(DEFAULT_MAX_ENTRIES, effective_budgets().wigner_3j_bytes)
});
static CACHE_6J: LazyLock<CoefficientCache<Regge6j, SignedSqrtRational>> = LazyLock::new(|| {
    CoefficientCache::shared(DEFAULT_MAX_ENTRIES, effective_budgets().wigner_6j_bytes)
});

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

/// An F-tier key: an SU(2) Regge class with its two extra determinants, or a
/// Cartan-engine sextet. A Cartan F block has no Regge-style canonical form
/// (the block is not invariant under label permutations), so its key is the
/// six labels as given.
#[derive(Clone, Debug, PartialEq, Eq)]
enum FSymbolKey {
    Su2(FKey),
    Cartan(LabelKey),
}

impl Hash for FSymbolKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Su2(key) => {
                0u8.hash(state);
                key.hash(state);
            }
            Self::Cartan(key) => {
                1u8.hash(state);
                key.hash(state);
            }
        }
    }
}

impl CacheKeyCharge for FSymbolKey {
    fn key_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(match self {
            Self::Su2(_) => 0,
            Self::Cartan(key) => std::mem::size_of_val(key.labels.as_ref()),
        })
    }
}

/// A borrowed Cartan sextet of [`FSymbolKey`].
struct FCartanRef<'a>(LabelRef<'a, 6>);

impl Hash for FCartanRef<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        1u8.hash(state);
        self.0.hash(state);
    }
}

impl quick_cache::Equivalent<FSymbolKey> for FCartanRef<'_> {
    fn equivalent(&self, key: &FSymbolKey) -> bool {
        matches!(key, FSymbolKey::Cartan(key) if self.0.equivalent(key))
    }
}

#[derive(Clone, Debug)]
enum FValue {
    /// A rounded SU(2) F symbol: a *presentation* of the exact 6j tier's
    /// value, never an independent value source.
    Scalar(f64),
    Block(FBlock<f64>),
}

impl CacheCharge for FValue {
    fn value_bytes(&self) -> usize {
        match self {
            Self::Scalar(_) => std::mem::size_of::<Self>(),
            Self::Block(block) => std::mem::size_of::<Self>().saturating_add(block.value_bytes()),
        }
    }
}

/// The F tier. A warm SU(2) hit returns a `Copy` scalar without re-running the
/// big-integer `sqrt` of [`SignedSqrtRational::to_f64`]; a warm Cartan hit
/// returns the shared block without recontracting four CGC.
static CACHE_F: LazyLock<CoefficientCache<FSymbolKey, FValue>> = LazyLock::new(|| {
    CoefficientCache::shared(DEFAULT_MAX_ENTRIES, effective_budgets().f_symbol_bytes)
});

/// Rounded SU(2) F symbol for `key`.
#[inline]
pub(crate) fn su2_f<E>(key: FKey, compute: impl FnOnce() -> Result<f64, E>) -> Result<f64, E> {
    let key = FSymbolKey::Su2(key);
    match CACHE_F.get_or_try_compute(&key, || key.clone(), || compute().map(FValue::Scalar))? {
        FValue::Scalar(x) => Ok(x),
        FValue::Block(_) => unreachable!("an SU(2) key stores a scalar"),
    }
}

/// Cartan-engine F block of the six labels as given, in presented root `root`.
#[inline]
pub(crate) fn f_block<E>(
    root: RootSystem,
    parts: [&[i64]; 6],
    compute: impl FnOnce() -> Result<FBlock<f64>, E>,
) -> Result<FBlock<f64>, E> {
    let key = FCartanRef(LabelRef {
        kind: 0,
        root,
        parts,
    });
    match CACHE_F.get_or_try_compute(
        &key,
        || FSymbolKey::Cartan(key.0.to_key()),
        || compute().map(FValue::Block),
    )? {
        FValue::Block(block) => Ok(block),
        FValue::Scalar(_) => unreachable!("a Cartan key stores a block"),
    }
}

/// The R tier: Cartan-engine blocks keyed by the ordered triple, since
/// `R^{ab}_c` and `R^{ba}_c` are different blocks. SUNRepresentations
/// (`sector.jl:Rsymbol`, `RCACHE`) caches R blocks for the same reason.
static CACHE_R: LazyLock<CoefficientCache<LabelKey, RBlock<f64>>> =
    LazyLock::new(|| CoefficientCache::shared(R_MAX_ENTRIES, effective_budgets().r_symbol_bytes));

/// Cartan-engine R block of the three labels as given.
#[inline]
pub(crate) fn r_block<E>(
    root: RootSystem,
    parts: [&[i64]; 3],
    compute: impl FnOnce() -> Result<RBlock<f64>, E>,
) -> Result<RBlock<f64>, E> {
    let key = LabelRef {
        kind: 0,
        root,
        parts,
    };
    CACHE_R.get_or_try_compute(&key, || key.to_key(), compute)
}

/// One retained Cartan-engine value. Modules, completed channels and phases
/// share one tier and one limit: they are the construction data of one
/// operation, and a module is useful exactly while channels of its label are
/// being built.
#[derive(Clone)]
enum CgcValue {
    Module(Arc<Ladders>),
    Channel(Arc<CgcChannel>),
    Phase(i8),
}

impl CacheCharge for CgcValue {
    fn value_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(match self {
            Self::Module(m) => ARC_COUNTS_BYTES
                .saturating_add(std::mem::size_of::<Ladders>())
                .saturating_add(m.bytes()),
            Self::Channel(c) => ARC_COUNTS_BYTES
                .saturating_add(std::mem::size_of::<CgcChannel>())
                .saturating_add(std::mem::size_of_val(c.entries.as_ref())),
            Self::Phase(_) => 0,
        })
    }
}

const MODULE: u8 = 0;
const CHANNEL: u8 = 1;
const PHASE: u8 = 2;

static CACHE_CGC: LazyLock<CoefficientCache<LabelKey, CgcValue>> =
    LazyLock::new(|| CoefficientCache::shared(CGC_MAX_ENTRIES, effective_budgets().cgc_bytes));

fn cgc_load<const N: usize, E>(
    kind: u8,
    root: RootSystem,
    parts: [&[i64]; N],
    compute: impl FnOnce() -> Result<CgcValue, E>,
) -> Result<CgcValue, E> {
    let key = LabelRef { kind, root, parts };
    CACHE_CGC.get_or_try_compute(&key, || key.to_key(), compute)
}

/// The irrep module of `label`.
pub(crate) fn cgc_module<E>(
    root: RootSystem,
    label: &[i64],
    compute: impl FnOnce() -> Result<Arc<Ladders>, E>,
) -> Result<Arc<Ladders>, E> {
    match cgc_load(MODULE, root, [label], || compute().map(CgcValue::Module))? {
        CgcValue::Module(m) => Ok(m),
        _ => unreachable!("a module key stores a module"),
    }
}

/// The completed, gated channel of the ordered triple.
pub(crate) fn cgc_channel<E>(
    root: RootSystem,
    parts: [&[i64]; 3],
    compute: impl FnOnce() -> Result<Arc<CgcChannel>, E>,
) -> Result<Arc<CgcChannel>, E> {
    match cgc_load(CHANNEL, root, parts, || compute().map(CgcValue::Channel))? {
        CgcValue::Channel(c) => Ok(c),
        _ => unreachable!("a channel key stores a channel"),
    }
}

/// The computed phase `κ` of a complex pair, keyed by its smaller label.
pub(crate) fn cgc_phase<E>(
    root: RootSystem,
    label: &[i64],
    compute: impl FnOnce() -> Result<i8, E>,
) -> Result<i8, E> {
    match cgc_load(PHASE, root, [label], || compute().map(CgcValue::Phase))? {
        CgcValue::Phase(k) => Ok(k),
        _ => unreachable!("a phase key stores a phase"),
    }
}

/// Bounded exact simple-Lie products. Keys omit global form and distinguish
/// root presentations; the flag selects the label-free projection.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProductKey(RootSystem, Box<[i64]>, Box<[i64]>, bool);
#[derive(Hash)]
struct ProductRef<'a>(RootSystem, &'a [i64], &'a [i64], bool);
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
static CACHE_PRODUCT: LazyLock<CoefficientCache<ProductKey, ProductValue>> = LazyLock::new(|| {
    CoefficientCache::shared(PRODUCT_MAX_ENTRIES, effective_budgets().product_bytes)
});

/// One form-free, symmetric exact product store; the loader takes no other lock.
pub(crate) fn product<E>(
    root: RootSystem,
    a: &[i64],
    b: &[i64],
    compute: impl FnOnce() -> Result<Multiplicities, E>,
    labels: impl FnOnce(Multiplicities) -> Result<Product, E>,
) -> Result<Product, E> {
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

pub(crate) fn product_multiplicities<E>(
    root: RootSystem,
    a: &[i64],
    b: &[i64],
    compute: impl FnOnce() -> Result<Multiplicities, E>,
) -> Result<Multiplicities, E> {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    let key = ProductRef(root, a, b, true);
    let own = || ProductKey(root, a.into(), b.into(), true);
    let compute = || compute().map(ProductValue::Multiplicities);
    let value = CACHE_PRODUCT.get_or_try_compute(&key, own, compute)?;
    match value {
        ProductValue::Multiplicities(x) => Ok(x),
        _ => unreachable!("projection is part of the key"),
    }
}

fn control(tier: CoefficientCacheTier) -> &'static dyn TierControl {
    match tier {
        CoefficientCacheTier::Product => &*CACHE_PRODUCT,
        CoefficientCacheTier::Cgc => &*CACHE_CGC,
        CoefficientCacheTier::FSymbol => &*CACHE_F,
        CoefficientCacheTier::RSymbol => &*CACHE_R,
        CoefficientCacheTier::Wigner3j => &*CACHE_3J,
        CoefficientCacheTier::Wigner6j => &*CACHE_6J,
    }
}

/// Clear every tier and *all* its counters — entries, bytes, hits, misses,
/// and evictions all return to zero. This includes every retained irrep
/// module, channel and phase of the CGC tier.
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
    // Overlapping resets would restore an even epoch while both are active.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    RESETS.fetch_add(1, Ordering::SeqCst);
    for &tier in CoefficientCacheTier::ALL {
        control(tier).reset();
        #[cfg(test)]
        reset_pause(tier);
    }
    RESETS.fetch_add(1, Ordering::SeqCst);
}

/// Resets on threads named `overlap-reset` pause after clearing the CGC tier
/// until [`RESET_RELEASE`] is set; [`RESET_PAUSED`] counts them.
#[cfg(test)]
pub(crate) static RESET_PAUSED: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static RESET_RELEASE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
fn reset_pause(tier: CoefficientCacheTier) {
    if tier == CoefficientCacheTier::Cgc && std::thread::current().name() == Some("overlap-reset") {
        RESET_PAUSED.fetch_add(1, Ordering::SeqCst);
        while !RESET_RELEASE.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
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
/// let report = trim_to(CoefficientCacheTier::Wigner6j, 0); // application-owned
/// assert_eq!(report.remaining_entries, 0);
/// assert_eq!(tier_stats(CoefficientCacheTier::Wigner6j).entries, 0);
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
/// let before = tier_stats(CoefficientCacheTier::Wigner6j);
/// let _ = fusion_symbols::lie::su2::wigner_6j(2, 2, 2, 2, 2, 2);
/// let _ = fusion_symbols::lie::su2::wigner_6j(2, 2, 2, 2, 2, 2);
/// let after = tier_stats(CoefficientCacheTier::Wigner6j);
/// assert!(after.hits > before.hits); // the second call is served from the cache
/// assert!(after.bytes <= fusion_symbols::cache::cache_budgets().limit(CoefficientCacheTier::Wigner6j));
/// ```
pub fn tier_stats(tier: CoefficientCacheTier) -> TierStats {
    control(tier).tier_stats()
}

// Serialize unit tests that reset process-global coefficient caches.
#[cfg(test)]
pub(crate) static TEST_CACHE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn product_multiplicities_are_resident(root: RootSystem, a: &[i64], b: &[i64]) -> bool {
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    CACHE_PRODUCT
        .inner
        .peek(&ProductRef(root, a, b, true))
        .is_some()
}

/// Whether the completed channel of the ordered triple is retained.
#[cfg(test)]
pub(crate) fn cgc_channel_is_resident(root: RootSystem, parts: [&[i64]; 3]) -> bool {
    CACHE_CGC
        .inner
        .peek(&LabelRef {
            kind: CHANNEL,
            root,
            parts,
        })
        .is_some()
}

/// Whether the computed phase keyed by `label` is retained.
#[cfg(all(test, feature = "lie-generated"))]
pub(crate) fn cgc_phase_is_resident(root: RootSystem, label: &[i64]) -> bool {
    CACHE_CGC
        .inner
        .peek(&LabelRef {
            kind: PHASE,
            root,
            parts: [label],
        })
        .is_some()
}

/// Whether the module of `label` is retained.
#[cfg(all(test, feature = "lie-generated"))]
pub(crate) fn cgc_module_is_resident(root: RootSystem, label: &[i64]) -> bool {
    CACHE_CGC
        .inner
        .peek(&LabelRef {
            kind: MODULE,
            root,
            parts: [label],
        })
        .is_some()
}

#[cfg(all(test, feature = "lie-generated"))]
pub(crate) fn with_publication_locks<T>(f: impl FnOnce() -> T) -> T {
    let _guards = [
        CACHE_3J.publication.lock().unwrap(),
        CACHE_6J.publication.lock().unwrap(),
        CACHE_F.publication.lock().unwrap(),
        CACHE_R.publication.lock().unwrap(),
        CACHE_PRODUCT.publication.lock().unwrap(),
        CACHE_CGC.publication.lock().unwrap(),
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

    fn channel_fixture() -> (LabelKey, CgcValue) {
        let fam = crate::lie::general::GeneralFamily::new(RootSystem::A(2));
        let eight: &[i64] = &[1, 1];
        let value = CgcValue::Channel(fam.cgc_entries(eight, eight, eight).unwrap());
        let key = LabelRef {
            kind: CHANNEL,
            root: RootSystem::A(2),
            parts: [eight, eight, eight],
        }
        .to_key();
        (key, value)
    }

    #[test]
    fn channel_at_full_tier_cap_is_retained() {
        let (key, value) = channel_fixture();
        let charge = entry_charge(&key, &value);
        for budget in [charge, charge + 1] {
            let cache = CoefficientCache::new(16, budget);
            assert_eq!(cache.inner.num_shards(), 1);
            assert_eq!(cache.inner.capacity() as usize, budget);
            let mut loads = 0;
            let cold = cache.get_or_compute(key.clone(), || {
                loads += 1;
                value.clone()
            });
            let before = cache.tier_stats();
            assert_eq!(
                (before.entries, before.bytes, before.misses),
                (1, charge, 1)
            );
            let warm = cache.get_or_compute(key.clone(), || {
                loads += 1;
                value.clone()
            });
            let (CgcValue::Channel(cold), CgcValue::Channel(warm)) = (cold, warm) else {
                unreachable!()
            };
            assert!(Arc::ptr_eq(&cold, &warm));
            assert_eq!(loads, 1);
            assert_eq!(cache.tier_stats().misses, before.misses);
            assert_eq!(cache.tier_stats().hits, 1);
            // Trim removes without shrinking capacity, so admission stays full-cap.
            cache.trim_to(0);
            cache.get_or_compute(key.clone(), || value.clone());
            assert_eq!(cache.tier_stats().entries, 1);
            assert_eq!(cache.inner.capacity() as usize, budget);
        }
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

    #[test]
    fn label_keys_charge_owned_labels() {
        let label = [1i64; 12];
        let key = LabelRef {
            kind: CHANNEL,
            root: RootSystem::C(12),
            parts: [&label[..], &label, &label],
        }
        .to_key();
        assert_eq!(
            key.key_bytes(),
            std::mem::size_of::<LabelKey>() + 36 * std::mem::size_of::<i64>()
        );
        let sextet = FSymbolKey::Cartan(
            LabelRef {
                kind: 0,
                root: RootSystem::C(12),
                parts: [&label[..]; 6],
            }
            .to_key(),
        );
        assert_eq!(
            sextet.key_bytes(),
            std::mem::size_of::<FSymbolKey>() + 72 * std::mem::size_of::<i64>()
        );
    }

    fn product_fixture(
        root: crate::lie::RootSystem,
        label: &[i64],
    ) -> (ProductKey, crate::lie::general::roots::Product) {
        let data = crate::lie::general::roots::CartanData::new(root);
        let c = data.irrep(label).unwrap();
        (
            ProductKey(root, label.into(), label.into(), false),
            crate::lie::general::roots::Product(Arc::from([(c, 1)])),
        )
    }
    #[test]
    fn product_tier_caps_charge_and_lifecycle() {
        use crate::lie::{general::roots::Product, RootSystem as R};
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
    #[test]
    fn product_entry_cap_is_256() {
        let (_, value) = product_fixture(crate::lie::RootSystem::A(1), &[0]);
        let local = CoefficientCache::new(256, 128 << 10);
        for n in 0..257 {
            local.get_or_compute(
                ProductKey(
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
    #[test]
    fn deep_key_bytes_drive_eviction_oversize_and_reset() {
        let key = |x: i64| {
            let label = [x; 12];
            LabelRef {
                kind: CHANNEL,
                root: RootSystem::A(12),
                parts: [&label[..], &label, &label],
            }
            .to_key()
        };
        let one_entry_budget = entry_charge(&key(1), &1.0f64);
        let cache: CoefficientCache<LabelKey, f64> =
            CoefficientCache::new(usize::MAX, one_entry_budget);
        cache.get_or_compute(key(1), || 1.0);
        cache.get_or_compute(key(2), || 2.0);
        assert_eq!(cache.tier_stats().entries, 1);
        assert_eq!(cache.tier_stats().evictions, 1);
        cache.reset();
        assert_eq!(cache.tier_stats(), TierStats::default());

        let shallow_only = std::mem::size_of::<f64>() + 2 * std::mem::size_of::<LabelKey>();
        assert!(entry_charge(&key(1), &1.0f64) > shallow_only);
        let oversize: CoefficientCache<LabelKey, f64> =
            CoefficientCache::new(usize::MAX, shallow_only);
        oversize.get_or_compute(key(1), || 1.0);
        assert_eq!(oversize.tier_stats().entries, 0);
        assert_eq!(oversize.tier_stats().bytes, 0);
        assert_eq!(oversize.tier_stats().evictions, 1);
    }

    #[test]
    fn cgc_tier_charges_storage_bytes_and_evicts_by_bytes() {
        let fam = crate::lie::general::GeneralFamily::new(RootSystem::A(2));
        let module = fam.module(&[1, 1]).unwrap();
        let (ka, a) = channel_fixture();
        // The channel charge is its sparse entries plus the Arc, channel and
        // enum shells.
        let CgcValue::Channel(entries) = &a else {
            unreachable!()
        };
        assert_eq!(
            a.value_bytes(),
            std::mem::size_of::<CgcValue>()
                + ARC_COUNTS_BYTES
                + std::mem::size_of::<CgcChannel>()
                + entries.len() * std::mem::size_of::<crate::frcore::MEntry>()
        );
        let b = CgcValue::Module(module.clone());
        assert_eq!(
            b.value_bytes(),
            std::mem::size_of::<CgcValue>()
                + ARC_COUNTS_BYTES
                + std::mem::size_of::<Ladders>()
                + module.bytes()
        );
        let kb = LabelRef {
            kind: MODULE,
            root: RootSystem::A(2),
            parts: [&[1i64, 1][..]],
        }
        .to_key();
        // A budget that fits only one entry must evict or reject one.
        let budget = a.value_bytes().max(b.value_bytes()) + ka.key_bytes() + 8;
        let c: CoefficientCache<LabelKey, CgcValue> = CoefficientCache::new(1_000_000, budget);
        c.get_or_compute(ka, || a.clone());
        c.get_or_compute(kb, || b.clone());
        let (_, _, entries, bytes) = c.snapshot();
        assert!(entries <= 1, "retained-charge cap kept {entries} entries");
        assert!(
            bytes <= budget,
            "retained charge exceeded cap: {bytes} > {budget}"
        );
        assert!(c.tier_stats().evictions > 0);
    }

    #[test]
    fn f_tier_charges_block_bytes_and_evicts_by_bytes() {
        let fam = crate::lie::general::GeneralFamily::new(RootSystem::A(2));
        let (e8, three, six, one): (&[i64], &[i64], &[i64], &[i64]) =
            (&[1, 1], &[1, 0], &[2, 0], &[0, 0]);
        // The 2×2×2×2 block of 8⊗8→8 and a multiplicity-free one.
        let a = FValue::Block(fam.f_symbol([e8; 6]).unwrap());
        let b = FValue::Block(fam.f_symbol([one, three, three, six, three, six]).unwrap());
        let FValue::Block(block) = &a else {
            unreachable!()
        };
        assert_eq!(
            a.value_bytes(),
            std::mem::size_of::<FValue>()
                + ARC_COUNTS_BYTES
                + std::mem::size_of_val(block.as_slice())
                + std::mem::size_of::<FBlock<f64>>()
        );
        assert!(a.value_bytes() > b.value_bytes(), "2⁴ block > 1⁴ block");
        assert_eq!(
            FValue::Scalar(0.5).value_bytes(),
            std::mem::size_of::<FValue>()
        );
        let key = |parts| {
            FSymbolKey::Cartan(
                LabelRef {
                    kind: 0,
                    root: RootSystem::A(2),
                    parts,
                }
                .to_key(),
            )
        };
        let (ka, kb) = (key([e8; 6]), key([one, three, three, six, three, six]));
        let budget = a.value_bytes() + ka.key_bytes() + 8;
        let c: CoefficientCache<FSymbolKey, FValue> = CoefficientCache::new(1_000_000, budget);
        c.get_or_compute(ka, || a.clone());
        c.get_or_compute(kb, || b.clone());
        let (_, _, entries, bytes) = c.snapshot();
        assert!(entries <= 1, "retained-charge cap kept {entries} entries");
        assert!(
            bytes <= budget,
            "retained charge exceeded cap: {bytes} > {budget}"
        );
        assert!(c.tier_stats().evictions > 0);
    }

    #[test]
    fn r_tier_charges_block_bytes_and_evicts_by_bytes() {
        let fam = crate::lie::general::GeneralFamily::new(RootSystem::A(2));
        let (e8, three, bar): (&[i64], &[i64], &[i64]) = (&[1, 1], &[1, 0], &[0, 1]);
        // The 2×2 octet block shares an Arc; a 1×1 block is inline.
        let a = fam.r_symbol([e8, e8, e8]).unwrap();
        let b = fam.r_symbol([three, bar, e8]).unwrap();
        assert_eq!(
            a.value_bytes(),
            ARC_COUNTS_BYTES
                + std::mem::size_of_val(a.as_slice())
                + std::mem::size_of::<RBlock<f64>>()
        );
        assert_eq!(b.value_bytes(), std::mem::size_of::<RBlock<f64>>());
        let key = |parts| {
            LabelRef {
                kind: 0,
                root: RootSystem::A(2),
                parts,
            }
            .to_key()
        };
        let (ka, kb) = (key([e8, e8, e8]), key([three, bar, e8]));
        let budget = a.value_bytes() + ka.key_bytes() + 8;
        let c: CoefficientCache<LabelKey, RBlock<f64>> = CoefficientCache::new(1_000_000, budget);
        c.get_or_compute(ka, || a.clone());
        c.get_or_compute(kb, || b.clone());
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

    #[test]
    fn nested_lookup_inherits_epoch_before_and_after_child_guard() {
        for reset_before_guard in [true, false] {
            let local = CoefficientCache::<u32, f64>::new(16, 1 << 20);
            local
                .get_or_try_sweep(
                    &1,
                    || 1,
                    |parent| {
                        if reset_before_guard {
                            local.reset();
                        }
                        parent.get_or_try_compute(
                            &2,
                            || 2,
                            || {
                                if !reset_before_guard {
                                    local.reset();
                                }
                                Ok::<_, ()>(2.0)
                            },
                        )?;
                        Ok::<_, ()>(1.0)
                    },
                )
                .unwrap();
            let stats = local.tier_stats();
            assert_eq!((stats.entries, stats.bytes, stats.misses), (0, 0, 2));
            local
                .publisher()
                .get_or_try_compute(&2, || 2, || Ok::<_, ()>(2.0))
                .unwrap();
            assert_eq!(
                local.tier_stats().entries,
                1,
                "stale child guard must be released"
            );
        }
        // Trimming does not invalidate the inherited epoch.
        let local = CoefficientCache::<u32, f64>::new(16, 1 << 20);
        let parent = local.publisher();
        parent
            .get_or_try_compute(
                &2,
                || 2,
                || {
                    local.trim_to(0);
                    Ok::<_, ()>(2.0)
                },
            )
            .unwrap();
        assert_eq!(local.tier_stats().entries, 1);
    }

    #[test]
    fn nested_single_flight_errors_and_panics_are_retryable() {
        use std::{
            sync::{mpsc, Barrier},
            time::Duration,
        };
        let (key, raw) = (7u32, 7.0f64);
        let local = Arc::new(CoefficientCache::new(16, 1 << 20));
        assert!(local
            .publisher()
            .get_or_try_compute(&key, || key, || Err::<f64, _>(()))
            .is_err());
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            local.publisher().get_or_try_compute(
                &key,
                || key,
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
                    key,
                    raw,
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
                                key
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

    #[test]
    fn borrowed_generated_views_hit_before_owning_keys() {
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
        let (a, b): (&[i64], &[i64]) = (&[0, 0], &[1, 0]);
        let root = RootSystem::A(2);
        for kind in [MODULE, CHANNEL, PHASE] {
            let three = LabelRef {
                kind,
                root,
                parts: [a, b, b],
            };
            hit(three.to_key(), &three);
        }
        let six = LabelRef {
            kind: 0,
            root,
            parts: [a, b, a, b, a, b],
        };
        hit(FSymbolKey::Cartan(six.to_key()), &FCartanRef(six));
        hit(
            ProductKey(root, a.into(), b.into(), false),
            &ProductRef(root, a, b, false),
        );
        // Labels are compared by content, not by their split into parts.
        let split = LabelRef {
            kind: CHANNEL,
            root,
            parts: [&[0i64, 0, 1][..], &[0, 1], &[0]],
        };
        hit(
            LabelRef {
                kind: CHANNEL,
                root,
                parts: [a, b, b],
            }
            .to_key(),
            &split,
        );
        // Different kinds and root presentations never collide.
        let cache = CoefficientCache::new(16, 1 << 20);
        cache.get_or_compute(
            LabelRef {
                kind: MODULE,
                root,
                parts: [b],
            }
            .to_key(),
            || 1.0f64,
        );
        for (kind, root) in [(PHASE, root), (MODULE, RootSystem::C(2))] {
            assert!(cache
                .get(&LabelRef {
                    kind,
                    root,
                    parts: [b],
                })
                .is_none());
        }
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
