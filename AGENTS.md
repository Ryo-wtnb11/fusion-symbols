# fusion-symbols Agent Policy

The repository and its single crate are `fusion-symbols` (formerly `racah`;
lib `fusion_symbols`). Each symmetry family is a top-level module: the
Lie-group families `su2`, `sun`, `bcd` (with `group`, `cache`), and the non-Lie
families added beside them. The rules below apply to every family; the
`cgc-gen` and dense-backend sections concern the Lie-group generation path.

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
  `TableObject`, the `abelian` and `cu1` irreps); sector behaviour beyond those
  symbols belongs to the consumer.
- Base crate: exact SU(2), closed-form and tabulated categories
  (`table`, `abelian`, `cu1`); dependencies stay minimal (big-integer
  arithmetic and `num-complex`).
- `cgc-gen` feature: all runtime generation (SU(N)/SO(N)/Sp(2N)) and every
  dense-kernel dependency live behind this feature. Factorizations
  (nullspace/QR/least-squares) and CGC contractions route through the
  selected dense backend; hand-rolled numeric kernels are not accepted.
- CGC are a legitimate public output of this crate; recoupling (F/R)
  derivation and its caches live here, next to the gauge contract.

## Acceptance criteria (every coefficient-affecting change)

1. Combinatorial structure exact (integer/rational; no floats in
   enumeration, multiplicities, or weights).
2. Discrete data exact (duals, Frobenius–Schur phases, signs, basis order).
3. Gauge fixing deterministic: pivot rules and sign conventions are part of
   the specification; discrete gauge flips across runs/backends are defects.
4. The gauge is a **frozen normative specification** (`docs/gauge.md`,
   `docs/gauge_soN.md`): the documents are the authority, the code implements
   them. A change that moves a coefficient value is a bug unless it ships as a
   specification correction — spec edit, fingerprint `epoch` bump, CHANGELOG
   breaking-change entry, regenerated `tests/gauge_golden.rs` values, one PR
   (`docs/gauge.md`, "Status"). Fingerprint strings keep their `racah:` prefix
   permanently (identifiers); never rename it.
5. Floating-point stages verification-gated: orthogonality, unitarity,
   pentagon/hexagon checks run at generation time; violations are typed
   errors, never silently degraded coefficients.

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

## t4a-faer dependency (cgc-gen)

The `cgc-gen` feature uses the crates.io semver range `t4a-faer = "0.24.6"`,
with default features disabled and only `std` enabled. No `rayon`: dense
operations are sequential, and the caller owns parallelism. Use explicit
`Par::Seq` for APIs taking `Par`; never set or rely on global parallelism.
Backend updates are ordinary reviewed changes and must satisfy the gauge
value and structural-identity gates below.

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
