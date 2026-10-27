# Resources and reproducibility

## Coefficient caches

Every coefficient-cache tier is process-global, byte-weighted Clock-PRO and independently bounded
(`src/cache.rs`). Default retained-entry charge caps are:

| Tiers | Aggregate cap |
|---|---|
| SU(2) 3j, 6j, derived F | `BASE_CACHE_MAX_BYTES`: 192 MiB (3 × 64 MiB) |
| SU(N) product / CGC / F / R, B/C/D product+CGC / F / R (`lie-generated`) | `GENERATED_CACHE_MAX_BYTES`: 768 MiB + 128 KiB |

The B/C/D product and CGC entries share `CoefficientCacheTier::BcdCgc`.
Caps exclude container capacity, allocator metadata, transient/external clones,
returned values and RSS.

Configure once, **before any cache operation or policy observation**:

```rust
use fusion_symbols::cache::{configure_cache_budgets, CoefficientCacheBudgets, CoefficientCacheTier};
let budgets = CoefficientCacheBudgets::default()
    .with_limit(CoefficientCacheTier::SixJ, 1 << 20);
configure_cache_budgets(budgets).unwrap();
```

Configuration is one-shot and shrink-only; defaults are the maximum caps.
A zero cap computes normally without retaining entries;
`CoefficientCacheBudgets::disabled()` zeroes all tiers. `cache_budgets()`
observes the effective policy and therefore also freezes configuration.

`tier_stats(tier)` reports one tier's entries, charge, hits, misses and
evictions; `CoefficientCacheTier::ALL` lists every tier in this build. Within a
tier snapshot, entries, charge, misses and evictions are read under the tier's
publication lock; hits are counted concurrently (see [Concurrency](#concurrency)).
A sum over tiers combines independent snapshots.
`trim_to(tier,target)` removes entries in unspecified traversal order until the
retained charge is at most the target; it preserves the configured capacity.
A zero target removes all retained values. In-flight computations can publish
after trimming. Each removed entry increments evictions; hits/misses are preserved.

`reset()` clears all tiers and counters. Concurrent fills can change occupancy
after either call. **The application owns this process-wide policy; libraries
must not call reset or trim.** Neither operation promises to lower RSS.

All tiers use `quick_cache` 0.7 with its foldhash hasher, its `sharded-lock`
reader-sharded lock (see [Concurrency](#concurrency)) and exactly one shard per
tier. Hits use borrowed `Equivalent` queries and a read lock, without
constructing owned keys or copying coefficient buffers.
Exact SU(2) values share their immutable rational payload; generated F and R blocks
are stored directly, sharing their coefficient storage. Concurrent
misses on a key, including `LieGroup` dual lookups, share one successful
computation; waiters count as hits. Errors are not retained. Whole-product
B/C/D sweeps also publish other channels, and a dual lookup also records the
reverse dual.

Each entry's real charge is one owned key (including label backing), its value
and one cached charge word. For a tier with
byte cap `B` and positive entry cap `E`, its quick_cache weight is
`max(real_charge, ceil(B / E))`. The single shard has capacity `B` and
`hot_allocation = 1.0`, so its admission limit is exactly `B`, with no margin:
an entry is eligible for admission iff its real charge is at most `B`.
Zero-budget and oversize entries are returned normally without displacing
retained values; each rejection counts as an eviction. Admission does not exempt
an entry from later capacity eviction. These rules do not depend on CPU count:
each tier is one store, so any entry that fits the whole tier cap is eligible,
however many threads use it.

The minimum weight enforces at most `floor(B / ceil(B / E)) <= E` resident
entries when `B > 0`, using quick_cache's amortized O(1) Clock-PRO eviction
rather than scanning residents on insertion. Nondivisible caps can retain fewer
than `E` entries; the minimum weight can also leave real-byte headroom unused.
Statistics and trim reports count real charge, excluding weight padding;
real bytes are tracked separately through publication and eviction, so snapshots
do not scan residents. Hits count retained or shared in-flight results. Misses count successful computations/publications;
a sweep recheck served by another publication counts one hit and no miss, and
publishing an already-retained key adds no miss. Errors add neither counter.
Eviction victims and trim survivors are unspecified.
Ghost history contains hashes, not owned labels or coefficient payloads, and
belongs to excluded container scaffolding. Reset also invalidates in-flight
placeholders and B/C/D sweeps that started before it: their callers
receive computed values, but neither the requested value nor the sweep's other
channels are retained. Returned values remain valid across trim, eviction and
reset.

### Concurrency

Each tier's single shard is guarded by quick_cache's `sharded-lock` lock
(crossbeam's `ShardedLock`), which has eight reader stripes. A warm hit takes
the read lock of its thread's stripe only, which reduces contention between
readers; it does not make hits independent of each other (see below). Every
change to retained values (publication, sweep publication, trim, reset) takes
the tier's publication mutex and then the write lock, which acquires all eight
stripes, so it waits for every active reader. Placeholder bookkeeping for
in-flight misses takes the write lock without the publication mutex.

Coefficient generation holds no coefficient-cache lock, so it may recurse into
other tiers. B/C/D generation does hold its group's generator-catalog mutex
(see below), which is separate from these locks: its sweeps publish other
channels without waiting for in-flight misses. Concurrent misses on one key
wait for its single computation, and a failed or panicking computation hands
the key to one waiter, which computes again.

Threads map to stripes through crossbeam's process-wide thread registry: a
thread's first use takes one global lock and allocates its registration. The
registry's memory is bounded by the peak number of concurrently registered
threads; exited threads' indices are reused, but the registry does not shrink.
Later hits on that thread do not allocate. Two concurrent threads can share a
stripe, and more than eight concurrent readers always do.

Hits are counted in 16 cache-line-padded stripes per tier, selected by a
per-thread index assigned round-robin on first use. A statistics snapshot sums
relaxed loads of the stripes: it is exact once concurrent readers have
finished, but not linearizable with hits in flight. Misses, evictions,
entries and charged bytes change only under the publication mutex and stay
exact.

A hit returns a clone of the stored value. For values with shared storage
(non-inline F blocks, CGCs, products) that is one `Arc` clone, whose reference
count is written by every concurrent hit on the same key, so throughput on one
shared block (such as the SU(3) octet F block) stays limited however many
threads query it. In hot loops, query once and hold the returned block rather
than repeating the query.

## B/C/D generator catalogs

Each B/C/D `LieGroup` owns a generator catalog; clones share it. The default
budget is 256 MiB. `LieGroup::with_catalog_budget(group,max_bytes)` sets a
retained generator charge budget, separate from coefficient-cache limits.
Construction checks the trivial and defining seeds' combined charge with checked
arithmetic before any rank-sized allocation; an insufficient budget or byte-count
overflow returns `LieError::Generation` (including a zero-byte budget).
Materialization stages a whole canonical-parent chain before committing it;
budget failure leaves committed entries unchanged. Temporary generation memory
is not bounded by the retained-byte budget. See `src/lie/bcd/catalog.rs::ensure`.

For a composite Lie group, the same 256 MiB default (or explicit budget) is
an aggregate over its B/C/D catalogs. Construction preflights all seed charges;
if their sum fits, it divides the remaining bytes equally among catalogs in
factor order, with remainder bytes assigned to earlier factors. Each factor's
cap is its seed charge plus its share. Rank-one and A-series factors consume
no catalog budget. `catalog_stats` sums retained entries, bytes and caps with
checked arithmetic; `reset_catalog` visits every catalog once. A composite
without B/C/D catalogs reports all-zero stats and rejects `with_catalog_budget`.

`catalog_stats()` returns `Result<CatalogStats, LieError>` with the retained
generator sets (`entries`, including the seeded base cases), their charged
`bytes`, and the `budget`; all three are zero without a catalog. `reset_catalog()` re-seeds base representations
and is a no-op for other groups. Both report a poisoned lock as
`LieError::Generation`. Global coefficient caches survive catalog reset;
warm CGC/F/R calls consult those caches before locking the catalog.

## Exactness and fingerprints

Lie combinatorics and discrete conventions are exact. SU(2) closed forms
round once; generated Lie coefficient values are verification-gated `f64`.
Table values are offline binary64 projections; their dimensions are derived
numerically from F. See [design](../overview.md#numerical-and-capability-contracts).
No cross-process bit identity is promised for generated coefficients. Discrete
sign, basis or multiplicity-order changes are not ordinary round-off.

Persist `FusionCategory::authority_fingerprint()` next to derived data and
compare bytes on load; never parse them. The SU(2) fingerprint identifies
values; generated-family fingerprints identify conventions, pipeline and
tolerance policy, with numerical agreement established by gates and oracles.
Generated fingerprint bytes and generated cache statistics/ceiling are marked
unstable in rustdoc.

A value-changing specification correction requires the spec edit, affected
family's epoch bump, CHANGELOG breaking-change entry and regenerated golden
values in one change. Ordinary prose or cache edits require no epoch bump.
Fingerprint `racah:` prefixes remain permanent identifiers. Normative rules:
[SU(2)/SU(N)](../gauge.md#status-frozen-normative-specification),
[B/C/D](../gauge_soN.md#status-frozen-normative-specification).
