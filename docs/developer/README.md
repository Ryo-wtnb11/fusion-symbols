# Contributor reference

- [AGENTS.md](../../AGENTS.md): scope and acceptance gates.
- [Port guard inventories](#port-guard-inventories), below.
- [Gauge specifications](../gauge.md), [B/C/D conventions](../gauge_soN.md) and
  [exceptional conventions](../gauge_exceptional.md):
  normative coefficient rules; prose cleanup does not change gauge epochs.
- [Reference map](../references.md) and [table provenance](../references_categories.md).
- [Tools](../../tools/README.md): independent test oracles and table generation.
- [Cache ownership](../user-guide/resources.md): application policy and catalog budgets.

## Checks

[CI](../../.github/workflows/ci.yml) checks formatting, clippy, documentation,
and default, base-only and all-features test suites. Use current stable Rust:

```sh
cargo fmt --all --check
RUSTFLAGS="-D warnings" cargo clippy --all-targets
RUSTFLAGS="-D warnings" cargo clippy --all-targets --no-default-features
RUSTFLAGS="-D warnings" cargo clippy --all-targets --all-features
cargo test
cargo test --no-default-features
cargo test --all-features
RUSTDOCFLAGS="-D warnings --html-in-header doc/katex-header.html" \
  cargo doc --no-deps --all-features
```

The KaTeX header matches `Cargo.toml::package.metadata.docs.rs`.
Criterion harnesses in `benches/` are measurements, not CI gates:
`torus`, `cache_concurrency`, `wigner`, `sun_cgc`, `sun_fr`, `sun_product`,
`bcd_fr`. The `sun_*` and `bcd_fr` harnesses require `lie-generated`. Test-only cache measurement
harnesses remain in `src/cache_audit.rs`, `src/cache_budget_pressure.rs`, and
`src/cache_trim_pressure.rs`; their ignored entry points describe how to run them.

## Integration-test targets

`Cargo.toml` disables automatic test targets. Existing `tests/*.rs` members stay
in place under `tests/integration.rs` (value tests) or `tests/allocations.rs`
(allocation budgets). For example:

```sh
cargo test --test integration gauge_golden::
cargo test --test allocations lie_a1_alloc::
```

Allocation tests take `tests/common/counting_alloc.rs::serial` before any work;
the single allocator counts calls, bytes and largest allocation per thread.
Only integration tests that measure or reset global caches take
`tests/common/mod.rs::serial`; pure value tests stay parallel. The backend
identity test takes that guard to reset between independent generation runs.

The explicitly named cache-policy, cache-contract, cache-trim, `sun_*_cache`,
`lie_a1`, `lie_ar`, `lie_bcd`, and `doc_examples` targets remain separate:
one-shot policies or exact cache counters require isolation from unrelated
value-test traffic. Their root comments state the reason. Member module names
prefix test paths; the assertions and feature gates are unchanged.

Local dev/test builds use line tables for debug information. CI retains its
explicit `CARGO_PROFILE_TEST_DEBUG=1` override.

## Port guard inventories

How the guards and invariants of the ported TensorKitSectors paths map to
this crate. User-facing differences are in
[differences from TensorKitSectors](../tensorkitsectors-differences.md).

### Product symbols

The port uses TensorKitSectors 0.3.9 `product.jl::Nsymbol`, `Fsymbol`,
`Rsymbol`, `fusiontensor`, `⊗`, style methods, `auxiliary.jl::_kron` / `_kron_promote`,
and `sectors.jl` style `&` methods. Their relevant guards and assumptions map as follows:

| Reference guard or invariant | Rust mapping |
|---|---|
| `SectorTuple`, same sector type `I` for symbol inputs | Compile-time tuple/object types and factor `contains` preflight; vector lengths must equal the factor count. Foreign labels panic loudly. |
| Runtime Cartesian indices and lazy CGC axes fit their public index types | Fusion checks the product length for `usize` overflow; CGC checks each axis, without imposing a dense element-count limit. Empty products use one scalar entry. |
| `unit` rejects `GenericUnit` | N/A: `category::FusionCategory::unit` requires one simple unit. |
| Scalar `_kron_promote` fills the logical N-derived shape | Logical axes queried independently of `SymbolBlock::dims`; zero axes skip values, all-one axes stay inline. |
| `_kron` assumes matching ranks and valid Cartesian indices | Fixed block ranks; each level expands in place with per-axis digit countdowns, last factor fastest; shape/count/byte overflow is a typed `Overflow`. |
| Julia product iteration can restart each component | Clone initialized iterators; nonempty reset is a documented loud-panic invariant. A valid clone preserves the initialized iterator sequence. |
| Styles restrict braiding and symbol availability | R capability errors and twist panic after full preflight; no special-case branch bypasses validation. |
| Julia multiplicity/product arithmetic fits its representation | Checked Rust arithmetic, zero-wins N, documented `nsymbol` panic or typed `try_nsymbol`/F/R `Overflow`. |
| Single-sector symbol base cases | Singleton vectors delegate symbols to the factor authority; the empty vector supplies the scalar identity. Tuples accept 2..=12 factors. |

No reference assertions are present on these product-symbol paths. TKS
generic units and flattening are outside the approved port;
CGC value indexing follows `auxiliary.jl::_kron`: the last factor varies fastest
on every axis. `CgcBlock::at` panics for out-of-range indices, and product CGC
queries check combined axis lengths for typed `Overflow` before returning.
The scalar abelian case maps an absent channel to a zero multiplicity axis. Factor generation retains its own verification
gates; composition only consumes those authorities. Independent indexed
oracles are in `tests/product.rs`; structural identities use
`tests/category_conformance.rs::product_identities` and
`tests/category_conformance.rs::product_multiplicity_residuals`. Composition allocation checks are in
`tests/product_alloc.rs::warm_product_composition_allocates_nothing`.

### Composite Lie restriction

`src/lie.rs::{coefficient_roots,LieGroup::irrep_with_charges,LieGroup::factor_labels}`
validate engine availability and outward group identity before Product queries.
`src/lie.rs::LieGroup::composite_label` reattaches the complete datum; quotient
fusion closure follows from `ν−λ−μ ∈ Q ⊆ X`, rather than filtering channels.
No numerical generator or reference coefficient routine is ported for composites.

| Reference constraint | Mapping |
|---|---|
| TKS `product.jl::SectorTuple` common label type and factor order | Private factor labels plus the foreign-object panic at the outward GroupId boundary |
| TKS Cartesian shape/storage assumptions | `src/product.rs::ProductError::checked_block`, checked N/axis/storage arithmetic; malformed data retain their indexed source chain |
| Restartable product iterators | `src/product.rs::{advance,restarted}` clone initialized iterators; changed replay is a documented loud-panic invariant |
| Machine integer representability | Product overflow maps to `LieError::LabelOverflow`; infallible weight multiplicities panic on checked `u64` multiplication overflow |
| Root datum has Q contained in X | `src/lie/group.rs::GroupId::new` typed lattice validation; admission precedes engine label construction |
| Canonical generator resources | `src/lie/bcd/catalog.rs::CanonicalCatalog::seed_requirement` preflights checked seed totals before composite catalogs are allocated |
| Numerical generation/verification | Simple-factor gates run as for single factors, including special cases; Kronecker orthogonality and identities follow from the factor identities, verified by `tests/category_conformance.rs::composite_lie_identities` |
| CU1 constructor, SUN UInt8 labels, and new dimension/generator ports | N/A: no such constructor or numerical routine is ported here |

D₂'s convention is α₁ then α₂ in the exact A₁ gauge. N/F/R/A/B and CGCs use
Product's factor order, with the last factor fastest on each coefficient axis;
fusion varies the first factor fastest. Composite fingerprints use
Product framing of factor fingerprints. Quotienting adds no gauge change and
needs no epoch bump; equal fingerprints do not equate outward group identities.

### Connected torus integration

The arithmetic authority is `src/lie/u1.rs::{U1Irrep::from_exact,U1::fusion,U1::dual}`.
The adapters `src/lie.rs::{LieGroup::irrep_with_charges,LieGroup::fusion}` map invalid
charges and unrepresentable fusion outcomes to typed `LieError::LabelOverflow`.
`src/lie/u1.rs::U1Irrep::new` rejects `i32::MIN` with `U1Error`, proving dual
negation safe. `src/lie/u1.rs::U1::nsymbol` decides the vertex condition exactly
in `i64`; `try_nsymbol` uses the infallible default. An unrepresentable sum
cannot equal a representable output charge, so every symbol returns an exact
zero channel. `tests/lie_torus.rs::typed_u1_matches_liegroup_bitwise` checks
both paths, including boundaries and the independent dual oracle.
`LieGroup::label` checks full datum identity before every symbol shortcut.
`LieGroup::composite_label` asserts lattice closure: root containment and additive
charges keep every fusion outcome admitted. Ordered cover factors reuse Product's
checked shape/storage/multiplicity guards. Torus F/R/A/B/CGC
blocks use the same exact vertex condition, including every shortcut.
The closed forms are exact zeros and ones, so numerical generation gates are
vacuous; tensor products and lattice restriction preserve factor identities.
Closed-form U1 calls never generate or sweep, including warm repeats in the
bitwise equivalence test; no coefficient cache is needed.
No numerical or external constructor is ported here (additional SUN/CU1
reference guards are N/A). Warm generated calls are checked directly with
`src/lie.rs::tests::torus_composites_warm_calls_do_not_generate` and the
`src/lie/sun/cgc.rs::GENERATIONS` counter.

The `sun_product` benchmark/test target names remain stable; their tier control
is `CoefficientCacheTier::Product`. Product retention includes all simple
Lie families, while BcdCgc observations describe raw/CGC coefficients only.

The reset-sensitive producer-count gate runs alone in the default CI job:

```sh
cargo test --release --offline --lib lie::general::product_contract_tests::exceptional_product_cache_counted -- --ignored --exact --test-threads=1
```
