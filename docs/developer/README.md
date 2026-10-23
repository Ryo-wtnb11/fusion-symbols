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
`wigner`, `sun_cgc`, `sun_fr`, `sun_product`, `bcd_fr`.
All except `wigner` require `lie-generated`. Test-only cache measurement
harnesses remain in `src/cache_audit.rs`, `src/cache_budget_pressure.rs`, and
`src/cache_trim_pressure.rs`; their ignored entry points describe how to run them.
