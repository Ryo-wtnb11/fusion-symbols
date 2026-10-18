# Differences from TensorKitSectors.jl

fusion-symbols follows TensorKitSectors.jl conventions (reference: 0.3.9,
git `67d342a1`) for the sector data it provides. This page lists every
known difference and the reason recorded for it. Per-symbol provenance is in
[`references.md`](references.md) and
[`references_categories.md`](references_categories.md).

## Design

- **A standalone provider, not a sector interface.** There is no `Sector`
  trait, no `FusionStyle`/`BraidingStyle` vocabulary and no tensor engine;
  each family is a module of plain label types and functions. Reason: the
  crate answers "what are the correct numbers"; sector behaviour beyond the
  symbols, and its trait vocabulary, belong to the consumer, so that a
  consumer's engine cannot tell whether an F-block came from this crate, a
  closed form or a table ([`AGENTS.md`](../AGENTS.md) "Boundaries";
  [overview: Out of scope](overview.md#out-of-scope-deliberately)).
- **Quantities left to the consumer.** A/B symbols and other sector
  operations TensorKitSectors derives generically are not provided. Reason:
  same boundary; only the symbols a module is indexed by are in scope
  ([`references_categories.md`](references_categories.md), "Derived
  quantities").
- **More Lie groups, generated at run time.** SU(N), SU(N)/Z_k, PSU(N),
  SO(N), Spin(N) and Sp(2N) live in this crate (`cgc-gen`), with CGC as a
  public output; TensorKitSectors provides SU(2) itself and leaves SU(N) to
  SUNRepresentations.jl. Reason: no existing library computes the full
  coefficient set for these groups on demand for any admissible labels;
  tables have a label cut and single-group packages do not extend to
  SU(N ≥ 3), SO(N) or Sp(2N)
  ([overview: Why this crate exists](overview.md#why-this-crate-exists)).
- **Frozen gauge specification and authority fingerprints.** Every family
  has a written, normative gauge ([`gauge.md`](gauge.md),
  [`gauge_soN.md`](gauge_soN.md)) and a fingerprint that changes only with a
  specification correction. Reason: generated coefficients must be
  deterministic across runs and backends, and consumers persist the
  fingerprint next to anything they derive
  ([`AGENTS.md`](../AGENTS.md) acceptance criteria 3–4;
  [overview: Exactness and gauge](overview.md#exactness-and-gauge)).
- **Verification-gated generation with typed errors.** Orthogonality,
  unitarity, pentagon and hexagon checks run at generation time and a
  violation is a typed error. Reason: a silently degraded coefficient is not
  acceptable ([`AGENTS.md`](../AGENTS.md) acceptance criterion 5).
- **Checked and infallible entry points.** Each symbol has an infallible
  form returning exact zero on an inadmissible tuple and a `_checked` form
  returning a typed error; label overflow is a typed error. Reason: a
  forbidden coupling or out-of-range label must never produce a wrong answer
  ([overview: Families](overview.md#families);
  [User Guide: How errors work](user-guide/getting-started.md#how-errors-work)).

## Labels

- **SU(2) labels are doubled spins** `dj = 2j` (`u32`), not `HalfInt`.
  Reason: keeps half-integer spins exact and makes the triangle and parity
  rules integer arithmetic
  ([User Guide: representations](user-guide/representations.md#su2-doubled-spins)).
- **Z_N carries its modulus at run time** (`u32` in the label) rather than as
  a type parameter. Reason: matches symtn, the consumer it was ported from
  ([`references_categories.md`](references_categories.md), abelian).
- **U(1) charges are integer `i32`**, without `i32::MIN` and without
  half-integer charges. Reason: matches symtn; `i32::MIN` has no
  representable dual (`src/abelian.rs` module docs).
- **CU(1) `j` is stored as `twice_charge: u32`**, not `HalfInt`. Reason not
  recorded. (The full `u32` range, wider than symtn's codec, is recorded as a
  deliberate choice in [`references_categories.md`](references_categories.md),
  CU(1).)
- **`tables` objects are one-based fusion-ring indices** (`TableObject(1..=rank)`,
  `1` = unit) in AnyonWiki's order, not named TensorKitSectors objects.
  Reason: the source order is frozen and not re-sorted, so every value stays
  traceable to the database
  ([`references_categories.md`](references_categories.md), "Identity,
  ordering, gauge, fingerprint").

## Values

- **Abelian scalars are `f64`**, where TensorKitSectors declares `Int`.
  Reason: `f64` is the scalar type of the symtn providers these were ported
  from, whose values they reproduce bit for bit (`src/abelian.rs` module
  docs).
- **CU(1) inadmissible R is `+0.0`**, where TensorKitSectors returns `-0.0`
  on 30 recorded tuples. Reason: the crate-wide exact-zero contract, as in
  symtn ([`references_categories.md`](references_categories.md), CU(1)).
- **`tables` entries are in AnyonWiki's gauge**, with no conversion to
  TensorKitSectors' gauge. Fibonacci coincides with `FibonacciAnyon`; Ising
  differs from `IsingAnyon` by the vertex gauge `u^{σσ}_ψ = −1` and the object
  map `(I, σ, ψ) → (1, 3, 2)`. Reason: the tables are loaded verbatim from
  the source, so every number is traceable to it
  ([`references_categories.md`](references_categories.md), "Named
  constructors and TensorKitSectors 0.3.9").
- **Fibonacci values are correctly rounded** from AnyonWiki's exact symbols;
  `±F^{τττ}_τ[𝟙,𝟙]`, `R^{ττ}`, `dim τ` and `θ_τ` differ from
  TensorKitSectors' f64 arithmetic by 1–3 ulp. Reason: `TableCategory` is the
  single Fibonacci path, and it ships the correctly rounded projection of
  the exact data (CHANGELOG 0.4.0, fusion-symbols #134).
- **Only unitary AnyonWiki entries are shipped.** Non-unitary entries, entries
  whose pivotal structure is not unitary, and (without `anyonwiki-unitary`)
  rank 5–7 entries are refused with a typed `Unsupported` error. Reason: the
  selection rule is exactly the condition under which TensorKitSectors'
  derivations of dimension, Frobenius–Schur phase and twist from F and R
  agree with the database's own values
  ([`references_categories.md`](references_categories.md), "Selection").
