# fusion-symbols Agent Policy

The repository and its single crate are `fusion-symbols` (formerly `racah`;
lib `fusion_symbols`). The public modules follow the mathematical classification:
`category` (shared contract), `lie` (positive-dimensional compact Lie groups,
including U(1) and CU(1)), `finite` (finite groups, currently Z_N), `fermion`, `table`,
`product` (ordered Deligne products), and `cache` (resource controls).
The rules below apply to every family; the `lie-generated` rules concern the
Lie-group generation path.

Design authority: the architecture, acceptance criteria, and reference-role
map recorded in the upstream design discussion. Read them before any
non-trivial change.

## Boundaries

- Pure representation and fusion-category mathematics only. Category
  vocabulary (objects, unit, duals, fusion, N/F/R symbols, Frobenius–Schur
  data, twists, fusion and braiding style) is in scope; tensor-network
  concepts (legs, fusion trees, tensor maps, sector-id codecs, contraction)
  are not, and there is no
  dependency on any tensor-network engine crate. Label types are limited to
  the plain object labels a module's symbols are indexed by (for example
  `table::TableObject`, `lie::LieIrrep`, and the `lie`, `finite` and
  `fermion` irreps); sector behaviour beyond those
  symbols belongs to the consumer.
- Base crate: `category`, exact rank-one `lie` (with `lie::su2` evaluators),
  closed-form U(1)/CU(1) in `lie`, `finite`, `fermion`, tabulated `table`, `product`, and `cache`.
  Dependencies stay minimal (big-integer arithmetic and `num-complex`).
- `lie-generated` feature (enabled by default; opt out with
  `default-features = false`): the capability switch for higher-rank simple
  Lie types (A_r with r ≥ 2, B–G). The shared exact Cartan core and the A1
  basis provider compile in every build. Coefficient generation is exact
  rational arithmetic rounded once; no dense floating-point backend enters
  coefficient generation.
- CGC are a legitimate public output of this crate; recoupling (F/R)
  derivation and its caches live here, next to the gauge contract.

## Acceptance criteria (every coefficient-affecting change)

1. Combinatorial structure exact (integer/rational; no floats in
   enumeration, multiplicities, or weights).
2. Discrete data exact (duals, Frobenius–Schur phases, signs, basis order).
3. Gauge fixing deterministic: pivot rules and sign conventions are part of
   the specification; discrete gauge flips across runs/backends are defects.
4. The gauge is a **frozen normative specification** (`docs/gauge.md`,
   `docs/gauge_cartan.md`): the documents are the authority, the code
   implements them. A change that moves a coefficient value is a bug unless it
   ships as a specification correction — spec edit, CHANGELOG breaking-change
   entry, regenerated `tests/gauge_golden.rs` values and a semver-breaking
   release, one PR (`docs/gauge.md`, "Status").
5. Floating-point stages verification-gated at generation time
   (`docs/gauge_cartan.md` §7): CGC isometry and ladder consistency, per-block
   R unitarity, and F finiteness. Violations are typed errors, never silently
   degraded or repaired coefficients. F-unitarity, pentagon and hexagon are
   category-level identities verified through `category::residual` by
   callers and CI.

## Guard inventory (every port PR)

Reference guards that live *outside* the expression being ported are the
recurring defect class tracked in issue #15 (dropped `@assert`s, type
constraints, and unreachable-by-construction assumptions). Every port PR must:

1. Enumerate, in the PR body, each guard on the reference path being ported —
   `@assert`/`@check`, type or domain constraints, and invariants the reference
   relies on as unreachable-by-construction.
2. Map each to exactly one of: a typed error, a documented loud-panic invariant,
   or an explicit N/A with justification. An unmapped guard blocks merge.
3. Ensure every fast-path / special-case branch runs the same verification
   gates as the general path, or prove those gates vacuous for that branch.

## Verification

- Oracles are independent and readable in the test (`tools/README.md`,
  "Test oracles"): closed forms, identities, dense expansions, or a few
  literals naming the reference they were checked against (WignerSymbols.jl,
  TensorKitSectors, SUNRepresentations.jl, GroupMath, Oscar, AnyonWiki).
  Values derived from the code under test are not oracles.
- **Reference-inventory rule.** Any "existing X" a spec/issue/PR relies on
  (function, symbol, test, fixture) must be cited as `file::symbol` and
  verified to exist before it is built on. Unverified references block merge.
- **Warm-no-resweep rule.** Adding or touching a coefficient-generation API
  requires a test proving a warm second call is served from cache without
  re-sweeping. Consumer-only strands (they call an existing decomposition)
  are exempt.
- `cargo fmt` and `cargo test` (all feature combinations touched) before
  every commit; fine-grained commits, each building and testing green.

## Test style

A test exists to protect a stated purpose; write the purpose first, then the
assertion that checks exactly it. Reviewers judge every test by that pair.

Rules 2–3 extend `tools/README.md` "Test oracles", which stays binding, together with the guard-inventory rule.

1. Each test protects one purpose, stated in its name. The assertion checks exactly that purpose.
2. Expected values are independent and inline: closed forms, identities, dense expansion, or a few literals naming the external reference they were re-checked against. Never use a past output of this crate or of a previous implementation as an oracle.
3. The only recorded values allowed are declared format or specification contracts: the frozen gauge goldens (`gauge_golden.rs`), cross-process bit identity, and shipped-data digests. Each says so in its module doc.
4. Allocation tests assert `== 0` on warm paths, `largest < payload` for sharing, or a count derived from the fixture's structure (levels, leaves, outputs). Never assert byte totals built from `size_of`, `Arc` header sizes, or other std/dependency layouts.
5. Cache tests assert: warm misses unchanged; rebuild counters `== 0`; `bytes <= budget` and `entries <= cap`; trim/reset accounting identities; single-flight `misses == one cold call`. Do not assert internal hit, entry-kind or eviction totals that a valid refactor may change.
6. A cost-scaling contract ("linear in leaves", "independent of size") is asserted as a relation between two fixture sizes, not as recorded per-query coefficients.
7. Do not test the configuration or internals of dependencies (quick_cache shards, capacity, weight; malachite arithmetic).
8. Each category identity (pentagon, hexagon, F-unitarity, FS, twist) runs through the public `category::residual` functions or the conformance harness, once per family. The residual functions themselves have negative controls.
9. Do not run the same gate through both the internal engine and the public API; keep the public one.
10. Error tests assert the variant (`matches!`). Use at most one `Display` test per error type. Never branch on error text.
11. A test asserts something beyond "no panic"; random-input tests assert a stated property of every output.
12. Measurements, timing and RSS reports live in `benches/` or `examples/`, never as `#[ignore]` tests. Every `#[ignore = "…"]` gives a reason, a size class, and the CI job that runs it, if any.
13. Global-cache tests run in an isolated binary or under the serial guard. Use no sleeps or timeouts for ordering; use barriers and assert the reached state.
14. No issue numbers, commit hashes, "former/migration/baseline" wording, measured timings, or the old crate name in test names, comments or ignore reasons.
15. Documentation examples are compiled from the documentation source. Contract tests are not duplicated in the doc-example file.
