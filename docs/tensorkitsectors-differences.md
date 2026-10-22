# Differences from TensorKitSectors.jl

fusion-symbols follows TensorKitSectors.jl conventions (reference: 0.3.9,
git `67d342a1`) for the sector data it provides. This page lists every
known difference and the reason recorded for it. Per-symbol provenance is in
[`references.md`](references.md) and
[`references_categories.md`](references_categories.md).

## Design

- **A standalone provider, not a sector interface.** There is no `Sector`
  trait or tensor engine. `category::FusionCategory` supplies mathematical
  quantities and `FusionStyle`/`BraidingStyle` for every family: `lie`
  (positive-dimensional compact Lie groups, including U(1) and CU(1)),
  `finite` (finite groups), `fermion`, `table` and `product`;
  `cache` controls resources. Tensor-network
  behaviour built on the symbols belongs to the consumer
  ([`AGENTS.md`](../AGENTS.md) "Boundaries";
  [overview: Out of scope](overview.md#out-of-scope-deliberately)).
- **A/B symbols as `RBlock`.** `FusionCategory::a_symbol` and `b_symbol`
  follow `Asymbol` / `Bsymbol` (including the abelian and product
  specializations) but always return an `RBlock` matrix, empty for an absent
  channel, instead of a scalar for multiplicity-free categories. Reason: the
  default derivation must build its result for every category, including one
  whose `RSymbol` is a block type this crate cannot construct.
- **Quantities left to the consumer.** Apart from object enumeration
  (`values`, see [Object enumeration](#object-enumeration)) and the derived
  dimensions `sqrt_dim`, `inv_sqrt_dim` and `inv_dim` (`sqrtdim`,
  `invsqrtdim`, `inv(dim)`), other sector operations TensorKitSectors derives
  generically are not provided. Reason: same
  boundary; only the symbols a module is indexed by are in scope
  ([`references_categories.md`](references_categories.md), "Derived
  quantities").
- **More Lie groups, generated at run time.** SU(N), SU(N)/Z_k, PSU(N),
  SO(N), Spin(N) and Sp(2N) live in this crate (`lie-generated`), with CGC as a
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
  `table` uses AnyonWiki's source gauge; those conventions are recorded in
  [`references_categories.md`](references_categories.md), not in a gauge
  specification. Every family has an authority fingerprint (for example
  `LieGroup` through `FusionCategory::authority_fingerprint`)
  that changes only with a specification correction. Reason: generated coefficients must be
  deterministic across runs and backends, and consumers persist the
  fingerprint next to anything they derive
  ([`AGENTS.md`](../AGENTS.md) acceptance criteria 3–4;
  [overview: Exactness and gauge](overview.md#exactness-and-gauge)).
- **Verification-gated generation with typed errors.** Internal orthogonality
  and generator-consistency checks reject invalid CGC. Generic category
  residual functions check F unitarity, pentagon and hexagon identities.
  A numeric generation-gate violation is a typed error. Reason: a silently degraded coefficient is not
  acceptable ([`AGENTS.md`](../AGENTS.md) acceptance criterion 5).
- **Validation and exact evaluators.** Only the exact 3j/6j/CG evaluators
  in `lie::su2` retain infallible and `_checked` entry points. Abelian, fermion, CU(1) and table families use only
  `FusionCategory`: N = 0 is a value, and inadmissible F/R return `Ok(0)`.
  U(1) and CU(1) fusion can report label overflow. Object constructors validate
  labels; foreign cyclic or table objects panic. On fusion-only tables,
  `r_symbol` returns `TableError::NoBraiding` and `twist` panics, both after
  rejecting foreign objects. SU(N) uses `LieGroup` too: absent
  channels return empty blocks and generation failures are opaque
  `LieError::Generation` diagnostics with an error source. The same rules apply
  to B/C/D, including spinors. Validation belongs at construction.

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
  representable dual (`src/lie/u1.rs` module docs).
- **CU(1) `j` is stored as `twice_charge: u32`**, not `HalfInt`.
  `CU1Irrep` constructors validate `s`; `lie::CU1` supplies the category
  operations, keeping labels separate from their mathematical provider.
  (The full `u32` range, wider than symtn's codec, is recorded as a
  deliberate choice in [`references_categories.md`](references_categories.md),
  CU(1).)
- **`table` objects have one-based fusion-ring indices**
  (`TableCategory::object(1..=rank)?`, `1` = unit), private catalog tags
  and a `TableObject::index()` accessor in AnyonWiki's order, not named
  TensorKitSectors objects.
  Reason: the source order is frozen and not re-sorted, so every value stays
  traceable to the database
  ([`references_categories.md`](references_categories.md), "Identity,
  ordering, gauge, fingerprint").

## Values

- **Abelian and fermion-parity scalars are `f64`**, where TensorKitSectors declares `Int`.
  Reason: `f64` is the scalar type of the symtn providers these were ported
  from, whose values they reproduce bit for bit (`src/lie/u1.rs` and `src/finite.rs` module
  docs).
- **CU(1) inadmissible R is `+0.0`**, where TensorKitSectors returns `-0.0`
  on 30 recorded tuples. Reason: the crate-wide exact-zero contract, as in
  symtn ([`references_categories.md`](references_categories.md), CU(1)).
- **`table` entries are in AnyonWiki's gauge**, with no conversion to
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

## Object enumeration

`FusionCategory::object_count`, `object_at` and `object_index` correspond to
TensorKitSectors 0.3.10 `values(I)` with `length`, `getindex` and
`findindex` (`sectors.jl::SectorValues`). The order of each category is a
frozen contract, documented on the category type.

| Category | Order (zero-based index) | Relation to the reference |
|---|---|---|
| `ZN(N)` | charges `0, …, N − 1` | `ZNIrrep{N}`, same order |
| `FermionParity` | even, odd | `FermionParity`, same order |
| `U1` | `0, 1, −1, 2, −2, …` | `U1Irrep` restricted to integer charges (below) |
| `CU1` | `0+`, `0−`, then `twice_charge = 1, 2, …` | `CU1Irrep`, same order |
| `LieGroup`, rank one | admitted doubled spins `dj`, increasing | `SU2Irrep` (`dj` = index); SO(3) keeps only even `dj` (fs convention) |
| `LieGroup`, SU(N), N ≥ 3 | Dynkin labels by total, then lexicographic | SUNRepresentations.jl 0.4.0 `sector.jl::getindex` |
| `TableCategory` | source order, index = source index − 1 | AnyonWiki order (below) |
| `Product` | bounded Manhattan order of factor indices | `product.jl::getindex` (`to_manhattan_index − 1`), without flattening |

- **Zero-based `u64` indices.** Julia indices are one-based. A finite count
  above `u64::MAX` is `EnumerationError::Overflow` while individual indices
  stay addressable; an infinite category reports `Ok(None)` even though its
  labels have a machine range, and labels past it are `Overflow`. A product
  count is infinite whenever a factor's is, even if another factor's count
  overflows; a factor count above `u64::MAX`, including one an external
  category can only report as `Overflow`, never limits a product index up to
  `u64::MAX`, because such a bound excludes only tuples whose index is already
  at least `2^64`.
  TensorKitSectors' `SizeUnknown` iterator trait for large `ZNIrrep` is a
  storage hint, not a missing count.
- **U(1) is compressed to integer charges.** TensorKitSectors enumerates
  `0, ½, −½, 1, −1, …`; the integer subsequence is reindexed densely, so
  charge `q > 0` has index `2q − 1` and `q ≤ 0` has `−2q`. Reason: fs
  labels are integer charges only.
- **Ising's source order is `(I, ψ, σ)`**, TensorKitSectors' `IsingAnyon` is
  `(I, σ, ψ)`. Fibonacci agrees (`(I, τ)`). Reason: table order is AnyonWiki's
  frozen source order, never re-sorted.
- **Products keep their nesting.** `⊠` flattens, so a nested fs product
  enumerates differently from the flat one (`((1, 0), 0)` of `(U1 ⊠ U1) ⊠ U1`
  has index 5, `(1, 0, 0)` of `U1 ⊠ U1 ⊠ U1` index 3). Unranking uses
  exact inclusion–exclusion counts and binary searches instead of the
  reference's sequential searches.
- **Not enumerable.** Quotients of SU(N), N ≥ 3 (such as PSU(N) and
  SU(N)/Z_k with `k > 1`) and the B, C and D series of rank at least two return
  `EnumerationError::NotEnumerable`; no reference package enumerates them.
  A product with such a factor is not enumerable either.
- **Product derived dimensions are factorwise.** `sqrt_dim`, `inv_sqrt_dim`
  and `inv_dim` of a product multiply the factor values left to right.
  TensorKitSectors specializes only `dim` for products and takes
  `sqrtdim`/`invsqrtdim` of that product; the values agree up to rounding.

## Deligne products

- `product::Product<C>` retains 2..=12 immediate factors in a tuple. Explicit
  nesting and trivial factors are preserved. TensorKitSectors 0.3.9 `⊠`
  flattens products and absorbs `Trivial`; Rust callers choose their tuple
  shape explicitly. Nested and flat label types and fingerprints differ.
- Fusion enumeration varies the first factor fastest, while every F/R
  multiplicity axis varies the last factor fastest, matching TKS. Rust block
  buffers are row-major; compare indexed entries with Julia arrays.
- Rust uses one `FusionCategory::Scalar`, promoted left to right with
  `category::Promote`. TKS separates fusion and braiding scalar types.
- Fermionic R and twist multiply factorwise without a cross-factor sign.
  Charge/spin and parity labels are independent; no parity constraint is imposed.
- Products provide no `GroupRepresentations` / CGC implementation. Cloneable
  initialized fusion iterators reset without querying a factor again. The
  higher-ranked iterator bound currently supports owned category types.

### Reference guard inventory

The port uses TensorKitSectors 0.3.9 `product.jl::Nsymbol`, `Fsymbol`,
`Rsymbol`, `⊗`, style methods, `auxiliary.jl::_kron` / `_kron_promote`,
and `sectors.jl` style `&` methods. Their relevant guards and assumptions map as follows:

| Reference guard or invariant | Rust mapping |
|---|---|
| `SectorTuple`, same sector type `I` for symbol inputs | Compile-time tuple/object types and factor `dim` preflight; foreign labels panic loudly. |
| `unit` rejects `GenericUnit` | N/A: `category::FusionCategory::unit` requires one simple unit. |
| Scalar `_kron_promote` fills the logical N-derived shape | Logical axes queried independently of `SymbolBlock::dims`; zero axes skip values, all-one axes stay inline. |
| `_kron` assumes matching ranks and valid Cartesian indices | Fixed block ranks and per-axis decoding; shape/count/byte overflow is a typed `Overflow`. |
| Julia product iteration can restart each component | Clone initialized `Fuse` iterators; nonempty reset is a documented loud-panic invariant. A valid clone preserves the initialized iterator sequence. |
| Styles restrict braiding and symbol availability | R capability errors and twist panic after full preflight; no special-case branch bypasses validation. |
| Julia multiplicity/product arithmetic fits its representation | Checked Rust arithmetic, zero-wins N, documented `nsymbol` panic or typed `try_nsymbol`/F/R `Overflow`. |
| Single-sector symbol base cases | N/A: Rust accepts 2..=12 immediate factors and delegates leaf symbols to the factor authority. |

No reference assertions are present on these product-symbol paths. TKS
`fusiontensor`, generic units and flattening are outside the approved port;
value indexing is described in [Object enumeration](#object-enumeration). Factor generation retains its own verification
gates; composition only consumes those authorities. Independent indexed
oracles are in `tests/product.rs`; structural identities use
`tests/category_conformance.rs::product_identities` and
`tests/category_conformance.rs::product_multiplicity_residuals`. Composition allocation checks are in
`tests/product_alloc.rs::warm_product_composition_allocates_nothing`.
