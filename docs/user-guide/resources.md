# Resources and reproducibility

## Coefficient caches

Every coefficient-cache tier is process-global and independently bounded by
bytes. Eviction uses Clock-PRO, a scan-resistant approximation of
least-recently-used replacement. Default retained-entry charge caps are:

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

## Retained group data and allocation

Besides the caches above, B/C/D groups retain immutable root data for the
process lifetime: one entry per constructed (series, rank), O(rank³)
integers, independent of the number of irreps or queries. `reset()` and
`reset_catalog()` do not release it. Exact SU(N) and B/C/D Weyl dimensions
are computed in checked `u128` arithmetic and fall back to allocating
big-integer arithmetic only when a dimension or an intermediate overflows.
Warm `LieGroup` dual lookups reuse shared labels retained in the
`SunProduct` and `BcdCgc` tiers, charged to their budgets. U(1) through
`LieGroup` allocates nothing for label construction, clones, fusion
iteration and symbol/CGC queries. Composite weight multiplicities multiply
factor multiplicities in checked `u64`; the infallible query panics on
overflow.

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
`bytes`, and the `budget`; all three are zero without a catalog. `reset_catalog()` re-seeds base representations,
drops the exceptional coefficient memo, and is a no-op for other groups. Both report a poisoned B/C/D catalog
lock as `LieError::Generation`; a poisoned exceptional memo is dropped and rebuilt. Global coefficient caches survive catalog reset;
warm CGC/F/R calls consult those caches before locking the catalog.

## Exceptional coefficient memo

G₂, F₄, E₆, E₇ and E₈ coefficients bypass the global coefficient caches and their
budgets. Each such `LieGroup` (and each such factor of a composite) owns a memo of exact
modules, tensor products, CGC channels and E₆ Frobenius–Schur phases; clones share it.
After each coefficient or complex-phase query, under the same mutex, whole completed
entries are evicted until at most 512 entries and an estimated 256 MiB remain.
The eviction order is modules, tensor products, rounded channels, then phase pairs,
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
Fusion outcomes and N use only root-system combinatorics and never wait for
the memo; F, R and CGC queries on one exceptional group run one at a time.

Module size grows quickly with the dimension. Measured in release on one core, exact module construction takes
about 0.018 s for E₇ `1539`, 0.048 s for E₈ `3875`, 1.23 s for E₈ `27000`
and 1.77 s for E₈ `30380`. These timings exclude ladder flattening; estimated
retained module-plus-ladder sizes are still about 6, 15, 174 and 220 MiB,
respectively (`benches/exceptional.rs` reproduces these cases). A query
touching both of the last two exceeds the byte bound on its own; completed
channels can still stay warm after those modules are evicted.

Release, sequential E₈ `248⁴` F-unitarity measurements at base `888062e`:

| Policy | Instrumented query time | Module builds (rebuilds) | Channel builds (rebuilds) | Peak RSS |
|---|---:|---:|---:|---:|
| Wholesale reset (base accounting) | 72.34 s | 38 (33) | 64 (51) | 3.03 GiB |
| Plain LRU (retained-clone accounting) | 34.36 s | 17 (12) | 34 (21) | 2.91 GiB |
| Class order above (retained-clone accounting) | 15.51 s | 8 (3) | 13 (0) | 2.76 GiB |

The public ignored test `tests/lie_exceptional.rs::self_dual::e8_f_unitarity_on_the_adjoint`
without counters took 74.61 s before and 15.80 s after (4.72×); both returned
residual `4.8405723873656825e-14`; peak RSS was 3.04 → 2.64 GiB.
These are measured whole workloads, not the
older trace projections. RSS is from `getrusage` (`time -l` cannot read sysctl
in the sandbox), and includes transient allocations. The LRU comparison uses
the same accounting and query-end boundaries as the chosen policy.
`src/lie/general.rs::tests::e8_memo_workload_report` reproduces generation counts
in release; `small_memo_workload_report` measures five fresh groups and 128 warm
calls per group. Median cold/warm times (base → chosen) were G₂ `7` singlet R
39.5 → 38.5 µs / 1.090 → 1.069 µs, F₄ `26` singlet R
259 → 256 µs / 3.388 → 3.386 µs, and E₆ `27` phase
164 → 159 µs / 0.113 → 0.117 µs. Warm calls built no modules or channels;
the E₆ warm difference is 4 ns. These microsecond measurements are descriptive,
not performance assertions.

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
