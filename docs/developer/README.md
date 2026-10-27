# Contributor reference

- [AGENTS.md](../../AGENTS.md): scope, acceptance gates and port guard inventory.
- [Gauge specifications](../gauge.md) and [B/C/D conventions](../gauge_soN.md):
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
