# Resources and reproducibility

## Coefficient caches

Every coefficient-cache tier is process-global, holds one kind of data and
is independently bounded by bytes, with one limit shared by every group value.
Keys are the presented root system and the ordered labels; global forms
sharing a weight share the entry, while B₂/C₂ and D₃/A₃ stay distinct.
Eviction uses Clock-PRO, a scan-resistant approximation of
least-recently-used replacement. Default retained-entry charge caps are:

| Tier (`CoefficientCacheTier`, Display) | Holds | Cap |
|---|---|---|
| `Wigner3j` (`"wigner_3j"`) | exact SU(2) 3j symbols by Regge class | 64 MiB |
| `Wigner6j` (`"wigner_6j"`) | exact SU(2) 6j symbols by Regge class | 64 MiB |
| `FSymbol` (`"f_symbol"`) | rounded SU(2) F scalars and Cartan-engine F blocks | 64 MiB |
| `Cgc` (`"cgc"`) | Cartan-engine irrep modules, completed CGC channels and computed Frobenius–Schur phases, every simple factor including A(1) | 256 MiB |
| `RSymbol` (`"r_symbol"`) | Cartan-engine R blocks (SU(2) R is a closed-form sign) | 64 MiB |
| `Product` (`"product"`) | exact simple-Lie tensor products | 128 KiB, 256 entries |

`BASE_CACHE_MAX_BYTES` (448 MiB) is the sum of the four tiers a build
without `lie-generated` can fill; `GENERATED_CACHE_MAX_BYTES` (64 MiB +
128 KiB) adds Product and R, which only higher-rank groups fill. Every build
compiles every tier.

The CGC tier holds the construction data of one operation under one limit. A
completed channel is its multiplicity plus a shared sparse entry list, so a
resident channel is served without a Product lookup; modules are needed again
only while channels of their label are built. Each value is charged when it
is published, including its labels, rational limbs, vector capacities and
`Arc` headers. A query holds the modules and channels it uses independently
of retention, so its working set may exceed the cap transiently, and an
oversize value is returned but not retained. A completed channel is retained
independently of its modules.

All materialized simple-Lie tensor products use the Product tier. Its
borrowed key is the root-system presentation, an unordered pair of exact
Dynkin labels, and the stored projection (multiplicities or fusion labels).
N reads a dimension-free channel list; fusion builds a separate immutable label
projection from that list. Each projection occupies one entry in the same tier
and is fully charged at publication. Fusion payloads include exact dimensions
and both conjugate coordinate views. Neither projection grows after
publication. Rank-one A₁/B₁/C₁ N and fusion stay allocation-free arithmetic,
without retention. Eviction, reset or a zero budget can require exact
recomputation; they never invalidate a caller's shared handle.

Caps exclude container capacity, allocator metadata, transient/external clones,
returned values, generation working memory and RSS.

Configure once, **before any cache operation or policy observation**:

```rust
use fusion_symbols::cache::{configure_cache_budgets, CoefficientCacheBudgets, CoefficientCacheTier};
let budgets = CoefficientCacheBudgets::default()
    .with_limit(CoefficientCacheTier::Wigner6j, 1 << 20);
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
Exact SU(2) values share their immutable rational payload; F and R blocks
and CGC channels are stored directly, sharing their coefficient storage.
Concurrent misses on a key share one successful computation; waiters count as
hits. Errors, including values that fail a generation gate, are not retained.
Loaders take no lock while generating, and their nested loads form a directed
graph (F/R → channel → module/product; phase → module), so no loader waits on
itself.

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
do not scan residents. Hits count retained or shared in-flight results.
Misses count successful computations/publications. Errors add neither
counter.
Eviction victims and trim survivors are unspecified.
Ghost history contains hashes, not owned labels or coefficient payloads, and
belongs to excluded container scaffolding. Reset also invalidates every
generation in flight when it begins or started while it runs, in every tier,
together with every value such a generation loads (for example the modules,
channels and products of a CGC, F or R generation): their callers receive
computed values, but those values are not retained. A generation begins when
its outermost miss reads the reset epoch, before it acquires its placeholder;
nested misses inherit that epoch. Concurrent resets run one at a time. Returned values remain valid across trim,
eviction and reset.

### Concurrency

Each tier's single shard is guarded by quick_cache's `sharded-lock` lock
(crossbeam's `ShardedLock`), which has eight reader stripes. A warm hit takes
the read lock of its thread's stripe only, which reduces contention between
readers; it does not make hits independent of each other (see below). Every
change to retained values (publication, trim, reset) takes
the tier's publication mutex and then the write lock, which acquires all eight
stripes, so it waits for every active reader. Placeholder bookkeeping for
in-flight misses takes the write lock without the publication mutex.

Coefficient generation holds no coefficient-cache lock, so it may load other
keys, including keys of its own tier. Concurrent misses on one key wait for
its single computation, and a failed or panicking computation hands the key to
one waiter, which computes again.

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
(non-inline F blocks, CGC channels, products) that is one `Arc` clone, whose reference
count is written by every concurrent hit on the same key, so throughput on one
shared block (such as the SU(3) octet F block) stays limited however many
threads query it. In hot loops, query once and hold the returned block rather
than repeating the query.

## Retained group data and allocation

Groups of rank at least two own immutable shared root-system context for
repeated label construction; clones share it. The context initializes on the
first label, keeping group construction free of root-table allocations. There
is no process-global root-data map, except that every A(1) presentation shares
one A(1) context, built on the first block-CGC request. Cache reset leaves
group-owned structural context intact; dropping its owners releases it.
Labels hold weak context references, so calls reuse a live context without
cache entries retaining its root tables. A standalone label whose group is
gone reconstructs temporary context when a weight or product query needs it.
The cache charge includes the allocation shell retained by each weak
reference.

Labels own exact dimension, rounded `f64` dimension, principal sign and both
coordinate views for a complex conjugate pair. `dim()` reads stored data
without a lock. `dim_exact()` clones the stored Natural and can allocate for
large values. Construction evaluates one exact Weyl quotient in checked
`u128`, with an unbounded rational fallback. `dual()` clones the shared
payload and changes its orientation without a cache lookup or allocation.
U(1) through `LieGroup` allocates nothing for label construction, clones,
fusion iteration and symbol/CGC queries. Composite weight multiplicities
multiply factor multiplicities in checked `u64`; the infallible query panics
on overflow.

## Cartan-engine cost

Module size grows quickly with the dimension. Measured in release on one core,
exact module construction takes about 0.012 s for E₇ `1539`, 0.031 s for E₈
`3875`, 0.53 s for E₈ `27000` and 0.67 s for E₈ `30380`; estimated retained
module-plus-ladder sizes are about 6, 15, 174 and 220 MiB
(`benches/exceptional.rs` reproduces these cases). A query touching both of
the last two exceeds the CGC cap on its own; completed channels can still stay
warm after those modules are evicted.

The Frobenius–Schur phase of a complex D(r) (r odd) or E₆ irrep is derived
from the gauge (`docs/gauge_cartan.md` §5.3), so its first
`frobenius_schur_phase` call (or first A symbol) builds the exact modules of
the irrep and its dual and retains the sign for both; later calls on either
label are hits. A failure there is returned as `LieError::Generation`, never
a substituted sign. Admission, the indicator (`0` for complex irreps), self-dual
phases and every A-series phase build nothing. `irrep` rejects such complex
labels above `u32::MAX` dimensions with `LieError::LabelOverflow`; below that
bound the first phase query is unbounded in the same way as F or R.

## Exactness and value changes

Which values are exact and which are rounded is listed in
[numerical contracts](../overview.md#numerical-and-capability-contracts).
Every CGC, F, R, A and B value of a simple factor is bit-identical across
processes and platforms with correctly rounded binary64 arithmetic
(`docs/gauge_cartan.md` §6.2). Discrete sign, basis or multiplicity-order
changes are not ordinary round-off.

Generated cache statistics and ceilings are marked unstable in rustdoc.

A value-changing specification correction requires the spec edit, a CHANGELOG
breaking-change entry, regenerated golden values and a semver-breaking
release, in one change. Ordinary prose or cache edits are not value changes.
Normative rules:
[status and SU(2)](../gauge.md#status-frozen-normative-specification),
[simple Lie factors](../gauge_cartan.md#status-frozen-normative-specification).
