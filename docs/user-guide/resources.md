# Resources and reproducibility

## Coefficient caches

Every coefficient-cache tier is process-global, byte-weighted Clock-PRO and independently bounded
(`src/cache.rs`). Default retained-entry charge caps are:

| Tiers | Aggregate cap |
|---|---|
| SU(2) 3j, 6j, derived F | `BASE_CACHE_MAX_BYTES`: 192 MiB (3 × 64 MiB) |
| SU(N) product / CGC / F, B/C/D product+CGC / F (`lie-generated`) | `GENERATED_CACHE_MAX_BYTES`: 640 MiB + 128 KiB |

The B/C/D product and CGC entries share `CoefficientCacheTier::BcdCgc`.
Caps exclude container capacity, allocator metadata, transient/external clones,
returned values and RSS. R uses cached CGC but has no separate cache.

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

`base_cache_stats()` and, with generation enabled, `generated_cache_stats()`
report entries, charge, hits, misses and evictions. Each tier snapshot is
consistent under the tier's publication lock; totals combine independent snapshots.
`trim_to(tier,target)` removes entries in unspecified traversal order until the
retained charge is at most the target; it preserves the configured capacity.
A zero target removes all retained values. In-flight computations can publish
after trimming. Each removed entry increments evictions; hits/misses are preserved.

`reset()` clears all tiers and counters. Concurrent fills can change occupancy
after either call. **The application owns this process-wide policy; libraries
must not call reset or trim.** Neither operation promises to lower RSS.

All tiers use `quick_cache` 0.7 with its default foldhash hasher, parking_lot
locks and exactly one shard per tier. Hits use borrowed `Equivalent` queries and a
read lock, without constructing owned keys or copying coefficient buffers.
Exact SU(2) values share their immutable rational payload; generated F blocks
are stored directly, sharing their existing coefficient storage. Concurrent
misses on a key share one successful computation; waiters count as hits.
Errors are not retained. Whole-product B/C/D sweeps also publish other channels.

Each entry's real charge is one owned key (including label backing), its value
and one cached charge word, rather than FIFO's two key copies. For a tier with
byte cap `B` and positive entry cap `E`, its quick_cache weight is
`max(real_charge, ceil(B / E))`. The single shard has capacity `B` and
`hot_allocation = 1.0`, so its admission limit is exactly `B`, with no margin:
an entry is eligible for retention iff its real charge is at most `B`.
Zero-budget and oversize entries are returned normally without displacing
retained values; each rejection counts as an eviction. Admission does not exempt
an entry from later capacity eviction. These rules do not depend on CPU count.
Sharding was considered and rejected because shard-local admission loses large
generated coefficients that fit the tier cap; reconsider it only with contention
measurements and a solution preserving full-tier admission.

The minimum weight enforces at most `floor(B / ceil(B / E)) <= E` resident
entries when `B > 0`, using quick_cache's amortized O(1) Clock-PRO eviction
rather than scanning residents on insertion. Nondivisible caps can retain fewer
than `E` entries; the minimum weight can also leave real-byte headroom unused.
Statistics and trim reports count real charge, excluding weight padding;
real bytes are tracked separately through publication and eviction, so snapshots
do not scan residents. Hits count retained or shared in-flight results. Misses count successful computations/publications;
a sweep recheck served by another publication counts one hit and no miss, and
publishing an already-retained key adds no miss. Errors add neither counter.
Eviction victims and trim survivors are unspecified, replacing the previous FIFO order.
Ghost history contains hashes, not owned labels or coefficient payloads, and
belongs to excluded container scaffolding. Reset also invalidates in-flight
placeholders: their callers still receive computed values, but those values may
not be retained. Returned values remain valid across trim, eviction and reset.

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
no catalog budget. `catalog_stats` sums retained bytes and caps with checked
arithmetic; `reset_catalog` visits every catalog once. A composite without
B/C/D catalogs reports `(0,0)` and rejects `with_catalog_budget`.

`catalog_stats()` returns `Result<(usize,usize),LieError>` for `(bytes,budget)`,
or `(0,0)` without a catalog. `reset_catalog()` re-seeds base representations
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
