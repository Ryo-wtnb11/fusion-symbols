# Resources and reproducibility

## Coefficient caches

Every coefficient-cache tier is process-global and independently bounded by
bytes. Eviction uses Clock-PRO, a scan-resistant approximation of
least-recently-used replacement. Default retained-entry charge caps are:

| Tiers | Aggregate cap |
|---|---|
| SU(2) 3j, 6j, derived F | `BASE_CACHE_MAX_BYTES`: 192 MiB (3 × 64 MiB) |
| Product, SU(N) CGC / F / R, B/C/D CGC / F / R (`lie-generated`) | `GENERATED_CACHE_MAX_BYTES`: 768 MiB + 128 KiB |

B/C/D raw ordered-product decompositions and completed CGC channels share the 256 MiB `CoefficientCacheTier::BcdCgc` cap. A raw
entry owns every discovered CGC copy, projected dense generator and Cartan/weight
table in discovery order, before alignment. Its charge includes allocation
capacities, nested matrix/vector shells, deep labels, the key, value and shared
ownership bookkeeping. Scratch and product adjacency are not retained.
Completed channels own separate coefficient buffers and retain no raw entry.
Either can be evicted independently; externally held values remain valid.

All materialized simple-Lie tensor products use `CoefficientCacheTier::Product`
(Display `"product"`), with a 128 KiB retained-charge cap and 256-entry ceiling.
Its borrowed key is the root-system presentation, an unordered pair of exact
Dynkin labels, and the stored projection (multiplicities or fusion labels). Global forms sharing a weight share the entry; A₅/D₅ and B₂/C₂
remain distinct. SU(N), B/C/D, exceptional N/fusion and exceptional coefficient
multiplicity checks share one exact channel producer. N reads a dimension-free
channel list; fusion builds a separate immutable label projection from that list.
Each projection occupies one entry in the same tier and is fully charged at
publication. Fusion payloads include exact dimensions and both conjugate
coordinate views. Neither projection grows after publication.
Rank-one A₁/B₁/C₁ fusion stays an allocation-free arithmetic range, without retention.
Product eviction/reset/zero budgets can require exact recomputation; they never
invalidate a caller's shared handle. `reset_catalog()` does not clear Product.

Canonical recursion and requested channels reuse a raw decomposition only for
the same ordered pair of canonical input frames. One sweep serves that pair
while its raw entry remains resident. Eviction, reset, disabled retention or
oversize rejection can require another sweep. A raw hit still reconstructs
missing local canonical frames under the requesting catalog's budget.
Public CGC ensures only its requested input/output chains. On an F/R channel
miss, optional canonical children from that product are committed as one batch
only if they fit; already-coherent siblings may then be published. Unrequested
outputs requiring recursion or alignment wait for demand, including their
possible generation errors.
Caps exclude container capacity, allocator metadata, transient/external clones,
returned values and RSS.

Configure once, **before any cache operation or policy observation**:

```rust
use fusion_symbols::cache::{configure_cache_budgets, CoefficientCacheBudgets, CoefficientCacheTier};
let budgets = CoefficientCacheBudgets::default()
    .with_limit(CoefficientCacheTier::SixJ, 1 << 20);
if let Err(err) = configure_cache_budgets(budgets) {
    eprintln!("cache budgets were already fixed: {err}");
}
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
misses on a key share one successful computation; waiters count as hits. Errors are not retained. B/C/D F/R misses
may also publish coherent siblings. Product loaders use only exact root mathematics
and never acquire a catalog or exceptional-family mutex.

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
channels or nested raw decompositions are retained, even if a child starts
after reset. Returned values remain valid across trim, eviction and
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

## Retained group data and allocation

Generated groups own immutable shared root-system context for repeated label
construction; clones share it. A-series context initializes on the first label
or product query, keeping large-rank group construction free of root-table
allocations. B/C/D seed-budget preflight runs before this
rank-sized allocation. There is no process-global root-data map. Cache reset and
catalog reset leave group-owned structural context intact; dropping its owners
releases it. Labels hold weak context references, so calls reuse a live context
without cache entries retaining its root tables. A standalone label whose group
is gone reconstructs temporary context when a weight or product query needs it.
The cache charge includes the allocation shell retained by each weak reference.

Generated labels own exact dimension, rounded `f64` dimension, principal sign and
both coordinate views for a complex conjugate pair. `dim()` reads stored data
without a global, catalog or family lock. `dim_exact()` clones the stored Natural
and can allocate for large values. Construction evaluates one exact Weyl quotient
in checked `u128`, with an unbounded rational fallback. `dual()` clones the shared
payload and changes its orientation without a cache lookup or allocation. The
complex E₆ public phase still uses its existing gauge-derived coefficient memo.
U(1) through `LieGroup` allocates nothing for label construction, clones, fusion
iteration and symbol/CGC queries. Composite weight multiplicities multiply
factor multiplicities in checked `u64`; the infallible query panics on overflow.

## B/C/D generator catalogs

Each B/C/D `LieGroup` owns a generator catalog; clones share it. The default
budget is 256 MiB. `LieGroup::with_catalog_budget(group,max_bytes)` sets a
retained generator charge budget, separate from coefficient-cache limits.
Construction checks the trivial and defining seeds' combined charge with checked
arithmetic before any rank-sized allocation; an insufficient budget or byte-count
overflow returns `LieError::Generation` (including a zero-byte budget).
Materialization stages a whole canonical-parent chain before committing it;
budget failure leaves committed entries unchanged. Temporary generation memory
is not bounded by the retained-byte budget. Catalog and block ladders remain
dense-only. Product ladders and raw seeds store ordered row and column adjacency
of exact nonzeros, with no dense product matrices. Product construction scans
dense factors once and reuses adjacency factors directly. See `src/lie/bcd/catalog.rs::ensure`.

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
`bytes`, and the `budget`; all three are zero without a catalog. `reset_catalog()` re-seeds base representations,
drops the exceptional coefficient memo, and is a no-op for other groups. Both report a poisoned B/C/D catalog
lock as `LieError::Generation`; a poisoned exceptional memo is dropped and rebuilt. Global coefficient caches survive catalog reset;
warm CGC/F/R calls consult those caches before locking the catalog.

## Exceptional coefficient memo

G₂, F₄, E₆, E₇ and E₈ coefficient buffers use a group-owned memo. Their exact
tensor products use the global Product tier and its budget. Each such `LieGroup` (and each such factor of a composite) owns a memo of exact
modules, CGC channels and E₆ Frobenius–Schur phases; clones share it.
After each coefficient or complex-phase query, under the same mutex, whole completed
entries are evicted until at most 512 entries and an estimated 256 MiB remain.
The eviction order is modules, rounded channels, then phase pairs,
least recently used within each class. Rounded channels own their data and survive
module eviction. A complex phase is stored once under the smaller of its dual labels;
it serves both labels and counts as two entries, so eviction cannot split the pair.
An entry that cannot fit is evicted when that query ends. Eviction never
runs while a query borrows modules, and a query's transient working set may exceed
the retained limits.

The estimate charges retained label/data buffers and per-entry value, byte-count
and recency metadata; a channel is charged for the stored clone's capacity.
Hash-table/B-tree node overhead, unused map capacity, allocator metadata, returned
buffers and query scratch are excluded; the bound is not a peak-RSS limit.
A single module-plus-ladders entry can be tens or hundreds of MiB.
Fusion outcomes and N use Product and never wait for the family memo; F, R and CGC queries on one exceptional group run one at a time.

Module size grows quickly with the dimension. Measured in release on one core, exact module construction takes
about 0.012 s for E₇ `1539`, 0.031 s for E₈ `3875`, 0.53 s for E₈ `27000`
and 0.67 s for E₈ `30380`. These timings exclude ladder flattening; estimated
retained module-plus-ladder sizes are still about 6, 15, 174 and 220 MiB,
respectively (`benches/exceptional.rs` reproduces these cases). A query
touching both of the last two exceeds the byte bound on its own; completed
channels can still stay warm after those modules are evicted.

Measured in release on one core, the E₈ `248⁴` F-unitarity check (the ignored
`tests/lie_exceptional.rs::self_dual::e8_f_unitarity_on_the_adjoint`) takes
about 6.1 s with a process peak RSS of about 0.84 GiB, and a single
`248 ⊗ 27000 → 248` or `248 ⊗ 30380 → 248` channel about 0.98 s or 1.17 s from
a fresh group. Exact module construction and CGC descent account for most of
that time. Peak RSS includes transient query allocations and is not bounded by
the retained-memo limits above. `benches/exceptional.rs` reproduces the module
and phase cases.

The Frobenius–Schur phase of a complex E₆ irrep is derived from the gauge, so
its first `frobenius_schur_phase` call (or first A symbol) builds the exact
modules of the irrep and its dual under the memo lock and memoizes the sign
for both; later calls on either label are memo hits. Admission, the indicator
(`0` for complex irreps) and self-dual phases build nothing. The first call on
a large admitted complex label therefore pays for both exact module builds
and the principal-ladder computation. Measured with the same release configuration
(irrep and dual), `1728` takes about 0.083 s and `7371` about 1.01 s; the
later calls on either label reuse the memo. Cold/warm timing and allocation
cases are in `benches/exceptional.rs`. `irrep` rejects complex E₆ labels above `u32::MAX` dimensions with
`LieError::LabelOverflow`; below that bound the first phase query is unbounded
in the same way as F or R.

`reset_catalog()` drops the memo (on any clone, and on every exceptional factor of a
composite); dropping the last clone frees it too. The memo does not appear in
`catalog_stats`, and `with_catalog_budget` rejects these groups with
`LieError::Unsupported`. Every value is exact until one rounding, so the memo
state never changes a returned bit.

## Exactness and fingerprints

Which values are exact and which are rounded is listed in
[numerical contracts](../overview.md#numerical-and-capability-contracts).
No cross-process bit identity is promised for SU(N) and B/C/D coefficients. Discrete
sign, basis or multiplicity-order changes are not ordinary round-off.

Persist `FusionCategory::authority_fingerprint()` next to derived data and
compare bytes on load; never parse them. The SU(2) and exceptional fingerprints identify
values; the SU(N) and B/C/D fingerprints identify conventions, pipeline and
tolerance policy, with numerical agreement established by gates and oracles.
Generated fingerprint bytes and generated cache statistics/ceiling are marked
unstable in rustdoc.

A value-changing specification correction requires the spec edit, affected
family's epoch bump, CHANGELOG breaking-change entry and regenerated golden
values in one change. Ordinary prose or cache edits require no epoch bump.
Fingerprints that begin with `racah:` (the crate's former name) keep that
prefix permanently. Normative rules:
[SU(2)/SU(N)](../gauge.md#status-frozen-normative-specification),
[B/C/D](../gauge_soN.md#status-frozen-normative-specification),
[exceptional](../gauge_exceptional.md#status-frozen-normative-specification).
