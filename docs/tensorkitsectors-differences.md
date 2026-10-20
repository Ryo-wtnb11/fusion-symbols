# Differences from TensorKitSectors.jl

fusion-symbols follows TensorKitSectors.jl conventions (reference: 0.3.9,
git `67d342a1`) for the sector data it provides. This page lists every
known difference and the reason recorded for it. Per-symbol provenance is in
[`references.md`](references.md) and
[`references_categories.md`](references_categories.md).

## Design

- **A standalone provider, not a sector interface.** There is no `Sector`
  trait or tensor engine. `category::FusionCategory` supplies mathematical
  quantities and `FusionStyle`/`BraidingStyle` for the abelian and fermion
  families; Lie, CU(1) and table families retain their existing APIs. Tensor-network
  behaviour built on the symbols belongs to the consumer
  ([`AGENTS.md`](../AGENTS.md) "Boundaries";
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
- **Frozen gauge specification and authority fingerprints.** The Lie-group
  families have a written, normative gauge: [`gauge.md`](gauge.md) for SU(2)
  and SU(N), [`gauge_soN.md`](gauge_soN.md) for SO(N), Spin(N) and Sp(2N).
  The abelian and CU(1) families use TensorKitSectors' closed-form gauge and
  `tables` uses AnyonWiki's source gauge; those conventions are recorded in
  [`references_categories.md`](references_categories.md), not in a gauge
  specification. Every family has an authority fingerprint (for example
  `su2_authority_fingerprint`, `cu1_authority_fingerprint`,
  `TableCategory::authority_fingerprint`) that changes only with a
  specification correction. Reason: generated coefficients must be
  deterministic across runs and backends, and consumers persist the
  fingerprint next to anything they derive
  ([`AGENTS.md`](../AGENTS.md) acceptance criteria 3–4;
  [overview: Exactness and gauge](overview.md#exactness-and-gauge)).
- **Verification-gated generation with typed errors.** Orthogonality,
  unitarity, pentagon and hexagon checks run at generation time and a
  violation is a typed error. Reason: a silently degraded coefficient is not
  acceptable ([`AGENTS.md`](../AGENTS.md) acceptance criterion 5).
- **Checked and infallible entry points.** `su2` and `cu1` give
  each F-symbol, R-symbol and (in `su2`) Wigner symbol an infallible form that
  returns exact zero on an inadmissible tuple and a `_checked` form that
  returns a typed error (for example `su2_f_symbol` / `su2_f_symbol_checked`,
  `cu1_r_symbol` / `cu1_r_symbol_checked`). For `abelian` and `fermion`, `FusionCategory` is the only symbol surface:
  N returns an exact multiplicity and F/R return `Result`, with zero on
  inadmissible tuples. U(1) fusion alone can overflow; cyclic and parity
  operations use `Infallible`. Foreign cyclic objects panic. CU(1)'s N
  function has no checked form and returns zero for a forbidden channel. The generated families return `Result` only:
  `sun::f_symbol`, `sun::r_symbol`, `bcd::f_symbol` and `bcd::r_symbol`
  return `Result<_, SunError>` / `Result<_, FrError>` (`src/sun/fr.rs`,
  `src/bcd/fr.rs`). In `tables`, the infallible `TableCategory` methods
  return exact zero for an unlisted entry but panic on a label that is not
  an object of the category, and `r_symbol` / `twist` also panic on a
  fusion-only category; the `_checked` methods return `TableError` instead
  (`src/tables.rs`, "# Panics" sections). Label overflow is a typed error. Reason: a
  forbidden coupling or out-of-range label must never produce a wrong answer
  ([overview: Families](overview.md#families);
  [User Guide: How errors work](user-guide/getting-started.md#how-errors-work)).

## Labels

- **SU(2) labels are doubled spins** `dj = 2j` (`u32`), not `HalfInt`.
  Reason: keeps half-integer spins exact and makes the triangle and parity
  rules integer arithmetic
  ([User Guide: representations](user-guide/representations.md#su2-doubled-spins)).
- **Z_N carries its modulus at run time** (`u32` in the label) rather than as
  a type parameter. `ZN` also carries the modulus and rejects foreign
  objects by panic. `ZNIrrep::new` reduces arbitrary integer charges;
  `ZNIrrep::from_canonical` instead rejects charges outside `0..N` without
  reduction. Z2 uses `ZN::new(2)` and its fingerprint. Reason: matches symtn,
  the consumer it was ported from
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

- **Abelian and fermion-parity scalars are `f64`**, where TensorKitSectors declares `Int`.
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
  the exact data.
- **Only unitary AnyonWiki entries are shipped.** Non-unitary entries, entries
  whose pivotal structure is not unitary, and (without `anyonwiki-unitary`)
  rank 5–7 entries are refused with a typed `Unsupported` error. Reason: the
  selection rule is exactly the condition under which TensorKitSectors'
  derivations of dimension, Frobenius–Schur phase and twist from F and R
  agree with the database's own values
  ([`references_categories.md`](references_categories.md), "Selection").
