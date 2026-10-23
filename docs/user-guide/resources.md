# Resources and reproducibility

## Coefficient caches

Every coefficient-cache tier is process-global, FIFO and independently bounded
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
consistent under its lock; totals combine independent snapshots.
`trim_to(tier,target)` removes oldest entries to the target charge;
`reset()` clears all tiers and counters. Concurrent fills can change occupancy
after either call. **The application owns this process-wide policy; libraries
must not call reset or trim.** Neither operation promises to lower RSS.

## B/C/D generator catalogs

Each B/C/D `LieGroup` owns a generator catalog; clones share it. The default
budget is 256 MiB. `LieGroup::with_catalog_budget(group,max_bytes)` sets a
retained generator charge budget, separate from coefficient-cache limits.
Materialization stages a whole canonical-parent chain before committing it;
budget failure leaves committed entries unchanged. Temporary generation memory
is not bounded by the retained-byte budget. See `src/lie/bcd/catalog.rs::ensure`.

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
