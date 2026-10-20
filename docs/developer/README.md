# Developer documentation

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

**Not user-facing.** Everything under `docs/developer/` is evidence and
internals for people changing `fusion-symbols` itself: measurement runs, raw traces, and
the reasoning behind resource defaults. Users who only need to *bound* cache
memory want [User Guide: numerical behaviour and resources](../user-guide/resources.md);
users who need the exact meaning of a public item want [docs.rs](https://docs.rs/fusion-symbols).

## Contents

| Document | What it records |
|---|---|
| [`coefficient-cache-audit.md`](coefficient-cache-audit.md) | Measured hit/miss/charge behaviour of every base and generated cache tier, with raw traces in `coefficient-cache-audit-trace.jsonl` and per-process timings in `coefficient-cache-audit-timings.jsonl`. |
| [`coefficient-cache-budget-tradeoff.md`](coefficient-cache-budget-tradeoff.md) | Why the one-shot shrink-only budget policy has the defaults it has; raw data in `coefficient-cache-budget-pressure.jsonl`. |
| [`coefficient-cache-trim-tradeoff.md`](coefficient-cache-trim-tradeoff.md) | Per-tier `trim_to` release behaviour under pressure; raw data in `coefficient-cache-trim-pressure.jsonl`. |

The collectors that produce these are `src/cache_audit.rs`,
`src/cache_budget_pressure.rs` and `src/cache_trim_pressure.rs` — all
`#[ignore]`d test-only harnesses, never CI gates.

## Other developer material in the repo

- [`../../AGENTS.md`](../../AGENTS.md) — contribution policy: boundaries,
  acceptance criteria for coefficient-affecting changes, guard-inventory
  discipline.
- [`../../tools/README.md`](../../tools/README.md) — test-oracle rules (expected
  values readable in the test) and the AnyonWiki data-build tool.
- [`../gauge.md`](../gauge.md), [`../gauge_soN.md`](../gauge_soN.md) — the
  frozen gauge specifications. Implementation internals (GT construction,
  B/C/D bootstrap, canonical parent, sweep order, alignment) are documented
  rule by rule there and in the module rustdoc of `src/lie/sun/`, `src/bcd/`.
- [`../../CHANGELOG.md`](../../CHANGELOG.md) — the breaking-change ledger the
  gauge specification's fingerprint epochs are recorded in.
- [`benches/`](#benchmarks) — `cargo bench` harnesses (not CI gates); see the
  index below.

## Implementation notes

**Kernel routing.** All dense numerical work behind `cgc-gen` — the
nullspace/QR/least-squares factorizations and the CGC contractions producing
F/R — routes through t4a-faer with explicit `Par::Seq` and scratch workspaces
(`src/lie/sun/linalg.rs`, `src/bcd/linalg.rs`). Only `std` is enabled, without
`rayon`; global parallelism is neither set nor consulted. `fusion-symbols`
contains no hand-rolled numeric kernels; selecting a backend is **not public API**. An
extended-precision tier (the QSpace model: compute in ~128-bit precision,
tighten tolerances, store `f64`) is not supported; it would be a
dense-backend capability, not a private arithmetic stack inside this crate.

**Gauge continuity.** The SU(N) pipeline reproduces the gauge of its reference
implementation by construction: the canonical gauge is a deterministic function
of the GT basis order and the nullspace subspace, so a faithful port reproduces
reference-generated coefficient tables to numerical tolerance; the tests pin
it on representative channels, not on whole tables. SO(N)/Sp(2N) carry their
own gauge tag.

**Oracle independence.** Every oracle is independent of the code under test:
closed forms and identities written in the tests (Condon–Shortley, Racah sum
rule, Weyl dimension, Brauer–Klimyk, Gelfand–Tsetlin, Chevalley–Serre,
intertwining), a few literals checked against the reference implementations,
and Regge/tetrahedral symmetries, pentagon/hexagon identities and orthogonality
as internal consistency gates.

**Guard discipline.** Every port PR carries a guard inventory — see
[`../../AGENTS.md`](../../AGENTS.md) and issue
[#15](https://github.com/Ryo-wtnb11/fusion-symbols/issues/15).

## Local checks

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features    # CI runs with RUSTFLAGS=-D warnings
cargo test --all-features
cargo test --no-default-features
RUSTDOCFLAGS="-D warnings --html-in-header doc/katex-header.html" \
  cargo doc --no-deps --all-features
```

The `RUSTDOCFLAGS` KaTeX header is the same one docs.rs uses
(`Cargo.toml [package.metadata.docs.rs]`), so `$...$` math in doc comments
renders locally exactly as it will on docs.rs. Add `--open` to read it.

## Benchmarks

Criterion harnesses under `benches/`, none of them CI gates. All but `wigner`
need the `cgc-gen` feature.

| Bench | What it measures | How to run |
|---|---|---|
| `wigner` | Prime-factorized exact 3j/6j on doubled spins ≤ 254, the thousands tier, and the repeated-label cache-hit regime. | `cargo bench --bench wigner` |
| `sun_cgc` | SU(N) CGC generation cost across representative `(N, dim)` channels, plus the warm cache-hit path. | `cargo bench --bench sun_cgc --features cgc-gen` |
| `sun_fr` | SU(N) F-symbol cost, cold (full four-CGC contraction, CGC caches cleared) and warm (derived-f64 F cache hit). R is not benched separately: a single sparse join of two CGC, no cache. | `cargo bench --bench sun_fr --features cgc-gen` |
| `sun_product` | SU(N) tensor-product cache working set (counting global allocator) plus cold/warm decomposition timing. | `cargo bench --bench sun_product --features cgc-gen` |
| `bcd_fr` | SO(N)/Sp(2N) F-symbol cost, cold (fresh catalog + full CGC sweep + four-CGC contraction) and warm (derived-f64 bcd F cache hit). Same R rationale as `sun_fr`. | `cargo bench --bench bcd_fr --features cgc-gen` |
