# Design and contracts

`fusion-symbols` provides representation and fusion-category mathematics.
Consumers own tensor-network representations and operations. There is no tensor
engine dependency. The shared interface is `category::FusionCategory`;
CGC are the separate `GroupRepresentations` capability, read through `CgcBlock`.
This page is the contract reference; task-oriented examples are in the
[user guide](user-guide/README.md).

## Capability boundaries

- Connected compact Lie groups are classified by `GroupId` and computed by
  `LieGroup` for every root system (A–G) and tori. Higher-rank and exceptional
  factors need `lie-generated`; without it `LieGroup::new` returns
  `LieError::Unsupported`.
- CU(1) = O(2) is the only disconnected group. Other O(N), Pin(N) and general
  disconnected groups are not provided (tracked in
  [#191](https://github.com/Ryo-wtnb11/fusion-symbols/issues/191)); noncompact
  groups are out of scope.
- Finite groups are the cyclic `ZN` and fermion parity. Finite non-abelian
  Rep(G) is not supported ([#192](https://github.com/Ryo-wtnb11/fusion-symbols/issues/192)).
- Tables are multiplicity-free, unitary, rank ≤ 7 AnyonWiki entries; other
  database entries are refused with a typed reason.

## Families and features

| Provider | Labels | Availability |
|---|---|---|
| `lie::LieGroup`: SU(2), SO(3), Spin(3), Sp(2), PSU(2) | One Dynkin label, doubled spin `dj = 2j` | Base |
| `lie::LieGroup`: SU(N), SU(N)/Z_k, PSU(N), N ≥ 3 | N−1 nonnegative Dynkin labels | `lie-generated` |
| `lie::LieGroup`: SO(N)/Spin(N), N ≥ 5; Sp(2r), r ≥ 2; supported quotients | Rank-many nonnegative Dynkin labels | `lie-generated` |
| `lie::LieGroup`: D₂ forms (Spin(4), SO(4), PSO(4), half-spin lattices) | Two ordered A₁ Dynkin labels | Base |
| `lie::LieGroup`: semisimple products and central quotients | Concatenated factor Dynkin labels, restricted to the character lattice | Union of factor features |
| `lie::LieGroup`: U(1), Tᵗ, and torus quotients | Ordered integer charges in `-i32::MAX..=i32::MAX` | Base |
| `lie::LieGroup`: U(N), semisimple×torus quotients | Dynkin labels then torus charges, restricted to the character lattice | Base for U(2); otherwise union of factor features |
| `lie::LieGroup`: trivial group | Empty Dynkin and charge slices | Base |
| `lie::U1` | `U1Irrep`: integer `i32` charge, excluding `i32::MIN` | Base |
| `lie::CU1` = O(2) | `CU1Irrep`: `(twice_charge, s)` = `(0,0)`, `(0,1)`, or `(q>0,2)` | Base |
| `finite::ZN` | `ZNIrrep`: charge and runtime nonzero `u32` modulus | Base |
| `fermion::FermionParity` | `FermionParityIrrep`: even or odd | Base |
| `table::TableCategory` | Tagged `TableObject`, source index `1..=rank`, unit = 1 | Rank ≤ 4 in base; rank 5–7 with `anyonwiki-unitary` |
| `product::Product` | Ordered tuples or homogeneous vectors, preserving explicit nesting | Base |

Base means available with `default-features = false`. `lie-generated` is on by
default; all dense dependencies are behind it. `anyonwiki-unitary` embeds more
data without adding a dependency. See `Cargo.toml::features`.

## Dedicated typed categories

Every connected compact group is computed through `LieGroup`. A dedicated
typed category exists only where fixed-size labels and closed-form symbols
remove measurable overhead, and it returns bitwise-identical values.

`U1` is the only connected group with a dedicated typed category. Its `Copy`
labels and scalar symbols avoid the general label and block overhead, with the
same values as `LieGroup::new(GroupId::u(1)?)`. `U1::group_id()` returns
`GroupId::u(1)`.
For Tᵗ at typed speed, use a tuple `Product` of `U1` factors, such as
`Product::new((U1, U1, U1))`. `ZN`, `FermionParity`, and `CU1` are not connected
groups and have no `group_id()`.

## Lie labels and global forms

Labels are the Dynkin labels of the highest weight in Bourbaki numbering,
followed by torus charges; [Dynkin labels](user-guide/labels.md) defines them
and gives numbering, dimensions, duals and the U(N) charge rule.
`GroupId::sp(2*r)`, `so(n)` and `spin(n)` take the defining dimension, not the
rank. A1, B1 and C1 use the exact SU(2) implementation; SO(3) and PSU(2) admit
only even doubled spins. D₂ expands into α₁ then α₂, each in the exact A₁
gauge: dimensions are `(a+1)(b+1)`, and SO(4) admits `a+b` even while PSO(4)
requires both labels even. Every simple factor uses the exact Cartan engine
of [gauge_cartan.md](gauge_cartan.md); factors other than A₁ need
`lie-generated`. `GroupId::{g2, f4, e6, e7, e8}` build the simply connected
forms. G₂, F₄, E₇
and E₈ irreps are all self-dual, and all are real except the E₇ irreps with
odd `a₂ + a₅ + a₇`, which are pseudo-real. E₆ duality swaps `a₁ ↔ a₆` and
`a₃ ↔ a₅`; its self-dual irreps are real. The Frobenius–Schur phase of a
complex D(r) (r odd) or E₆ irrep is derived from the gauge
([gauge_cartan.md](gauge_cartan.md) §5.3) and
generated on first use. `irrep` rejects such a complex label whose dimension
exceeds `u32::MAX` with `LieError::LabelOverflow`, because that phase is
generated in a basis with `u32` indices.

A `GroupId` is a validated classification datum: ordered root-system blocks,
a torus rank, and a full-rank character lattice, the lattice of admitted
(label, charge) coordinates, which must contain every root. A global form
filters labels through `GroupId::admits`; it never changes coefficients on
shared irreps, and fusion of admitted labels stays admitted. Spin(N) admits
spinors that SO(N) excludes. Besides the constructors in the
[user guide](user-guide/README.md#building-lie-groups), `GroupId::new` accepts
square integer row generators of any lattice; `cover` chooses the full
weight/charge lattice. D-series central-class and half-spin naming
conventions are fixed in [references](references.md#global-form--fixed-conventions).
`character_lattice` returns the canonical basis, the row Hermite normal form
with `0 ≤ H[i,j] < H[j,j]` above the positive pivots. Invalid ranks return
`GroupError::UnsupportedRank`; malformed, singular or root-excluding
lattices return `InvalidLattice`; overflowing sizes or entries return
`SizeOverflow`. Unsupported presentations are rejected before any root data
are built.

Equality, hashing and order use the ordered presentation and canonical
lattice, so SU(n)/Z1 equals SU(n), while distinct presentations such as
B1/C1/A1, D3/A3 and D2/[A1,A1] remain distinct. Objects retain the full group
tag. `LieGroup::new` succeeds exactly when every root-system block has an
engine; generated factors require `lie-generated`. Composites compute as the
ordered product of their simple covers followed by rank-one torus covers, so
each factor keeps its own gauge.

`irrep_with_charges(dynkin, charges)` checks lengths and nonnegative Dynkin
labels, then lattice admission, then engine integer limits. Charges outside
`-i32::MAX..=i32::MAX` return `LabelOverflow`. The allocation-free iterators
`dynkin` and `charges` read the semisimple and torus coordinates.
`dim_exact` returns `malachite_nz::natural::Natural`; the category's `dim` returns `f64`.
`weight_multiplicities` uses Dynkin coordinates followed by torus charges.
`gt_patterns` gives the Gelfand–Tsetlin (GT) basis for A-series rank ≥ 2, and
`None` for A1, B/C/D, exceptional groups and composites.

## Validation and errors

`FusionCategory` accepts validated objects. Every object-taking method except
`contains` panics on a foreign object before testing capabilities or channels.
Use `contains` for a non-panicking membership check. Category equality and
hashing represent mathematical identity.

- `fusion` yields distinct outcomes in deterministic order; `nsymbol` returns
  the exact `u32` multiplicity, `Ok(0)` for an absent channel. Its errors are
  genuine failures only: an unrepresentable product multiplicity, and on
  `LieGroup` generated fusion arithmetic and generation failures.
- `frobenius_schur_phase` returns a `Result`: closed forms and stored signs
  are `Ok`, while a complex D(r) (r odd) or E₆ irrep propagates a failure to
  generate its phase as `LieError::Generation`, never a substituted sign.
  The default A symbol propagates N, phase and F errors; B propagates N and F.
  `frobenius_schur_indicator` reads stored data and stays infallible.
- Inadmissible F/R vertices give zero symbols. Logical axes come from N;
  scalar storage has all-one axes. Lie blocks have zero axes for absent
  channels. See [symbol layout](user-guide/README.md#symbol-layout).
- Lie constructors distinguish invalid labels, excluded global-form weights,
  label overflow and unsupported capabilities. Generated failures are opaque
  `LieError::Generation` diagnostics with an error source.
- Generated Lie fusion and fallible N/F/R/A/B/CGC paths report `LieError::LabelOverflow`
  when a label, weight, dimension or multiplicity exceeds its integer range.
  Infallible weight-multiplicity queries panic on such failures. Dimension and
  phase formulas use exact or parity arithmetic without fixed-width overflow.
- U(1) symbols decide charge conservation exactly in `i64`, including zero
  channels when the sum is outside the label range. Only fusion reports that
  overflow (`U1Error` on `U1`, `LieError::LabelOverflow` on `LieGroup`).
  CU(1) and rank-one Lie fusion also report label overflow. `ZN` and parity
  category errors are `Infallible`; their constructors validate labels.
  `ZNIrrep::new` reduces charges modulo N; an input boundary that must reject
  a noncanonical charge checks `charge < N` first. CU(1) uses the full `u32`
  twice-charge range.
- Table lookup distinguishes unknown entries from known unsupported entries.
  Fusion-only tables return `TableError::NoBraiding` from R and panic on twist.
  Foreign objects are rejected first. Clones share parsed and derived data;
  categories with the same uuid accept the same objects.
- The exact `lie::su2` evaluators return exact zero for forbidden labels at
  every `u32`/`i32` input, with projection sums and sign changes evaluated in
  `i64`; a forbidden coupling and an accidental zero read the same.

## Numerical and capability contracts

The exact producer is `lie/general/roots.rs::CartanData`, with classical
coordinate branches (including integer A-series GT counting) in that layer.
Generated labels share immutable dimension/sign/conjugate data; materialized
simple-factor decompositions share the bounded `cache::CoefficientCacheTier::Product`.
Rank-one arithmetic and the existing coefficient generators keep their gauges.

Lie dimensions, duals, Frobenius–Schur indicators, weights and fusion
multiplicities are exact. SU(2) coefficients use signed square-rooted rationals
and round once to `f64`. The CGC of every simple factor is computed in exact
rationals, held in machine words when the reduced value fits and in the same
required Malachite 0.14 arithmetic family as the rest of the crate otherwise,
and rounded once to `f64`; F, R, A and B follow a fixed floating-point
evaluation order, so all are bit-identical across builds and platforms.
Public exact rationals are `malachite_q::Rational`; there are no arithmetic
backend conversions. Tables are offline projections of exact AnyonWiki data
to `Complex64`.
Closed-form U(1), CU(1), Z_N and parity symbols are `f64`.

Every simple factor uses one exact pipeline: a highest-weight nullspace and an
intertwiner descent from the Cartan matrix, with the Gelfand–Tsetlin basis for
the A series and a contravariant Gram–Schmidt basis otherwise. Generation
checks exact module ranks and nullities, rounded CGC isometry and ladder
consistency, R orthogonality and F finiteness; failures are typed errors.
Public F-unitarity, pentagon and hexagon residuals are explicit
caller-invoked checks with caller-selected tolerances. There is no dense
linear-algebra backend.

`has_trivial_associator_gauge` certifies admissible F = 1 in the documented gauge;
`has_canonical_unit` certifies identity F moves with an external unit leg
(`a`, `b` or `c`, not merely `d`, `e` or `f`). Both default to false and
products take the conjunction. All built-in families have a canonical unit;
U(1), Z_N and parity have trivial associators, as do qualifying pointed tables.
These are gauge-dependent certificates, not substitutes for N or braiding checks.

Gauge rules are frozen in [gauge.md](gauge.md) and
[gauge_cartan.md](gauge_cartan.md).
A coefficient value changes only in a semver-breaking release with a
CHANGELOG entry. See [resources](user-guide/resources.md).

## Enumeration and products

Enumeration uses zero-based `u64` indices and is independent of object `Ord`
and fusion order. Infinite categories report `Ok(None)` despite finite machine
label ranges; unrepresentable labels or indices return `EnumerationError::Overflow`.
Finite counts can overflow while individual indices remain addressable.
The frozen orders and unavailable enumerations are listed in the
[TensorKitSectors comparison](tensorkitsectors-differences.md#object-enumeration).

Composite Lie groups enumerate only when their lattice is the full cover
lattice and all factors enumerate. They use Product's Manhattan order
over semisimple factors then torus charges, with D₂ expanded in α₁, α₂ order. Every composite quotient
returns `NotEnumerable`, including U(N) for N ≥ 2, SO(4), and nonstandard torus lattices.

Tuple products have 2..=12 immediate factors; `Product<Vec<C>>` accepts any
number of homogeneous factors, including zero (the trivial category) and one.
Vector objects must have exactly as many entries as factors; other lengths are
foreign. Nesting and trivial factors are retained;
objects, N, dimensions, duals, Frobenius–Schur data and twists compose factorwise.
Charge/spin and parity labels are independent, with no extra cross-factor sign.
Fusion varies the first factor fastest; each F/R/A/B multiplicity axis varies
the last factor fastest. Scalar promotion and multiplication fold left to right.
Reassociation agrees to floating-point tolerance where intermediate queries are
representable. Products of `GroupRepresentations` provide lazy tuple or vector CGCs,
with the last factor fastest on each magnetic and multiplicity axis. Combined
axis overflow returns `ProductError::Overflow`. Products add no coefficient cache.
With label-owned dimension reads, warm single-factor A/B queries allocate only their
returned coefficient buffer (zero for empty or inline blocks).
Product F/R/A/B compose factor blocks on each query: beyond the factors' own
allocations, a warm query allocates one buffer per product level whose block
has more than one coefficient, and nothing when every level is a single one.
A `Vec` product nested as a factor also holds its factor blocks in one `Vec`,
and a real prefix of more than one coefficient meeting a complex factor costs
one transient real buffer.

All factors' objects are validated before capability or shape checks. R reports
`ProductError::NoBraiding { factor }` even when N = 0; twist panics on absent
braiding. Factor failures retain their immediate index. For N, a zero factor
wins over factor errors and overflow; otherwise `nsymbol` returns the first
factor error by position, then `ProductError::Overflow`. A phase returns the
first failing factor's error; it has no zero rule.
F/R/A/B check their own multiplicities and layouts for typed overflow. Nested
factor queries must themselves be representable. Fusion resets clone the
initialized factor iterators; the higher-ranked iterator bound admits owned category types only.
Runtime fusion uses checked mixed-radix front/back indices, implements
`ExactSizeIterator` when the factors do, and `DoubleEndedIterator` when the
factors do. It checks the outcome count for `usize` overflow before iteration.
Runtime factor failures use `ProductError::Factor { index, source }`.
The Lie facade retains that factor path in `LieError::Generation`'s source
chain for generation and malformed-data failures. Overflow maps to
`LabelOverflow`; factor `LabelOverflow` and `Unsupported` propagate directly
so callers can match their variants.

Before composing a nonzero F/R/A/B block, Product checks each factor's
storage axes against its checked N symbols and requires the exact row-major
storage length. A mismatch returns
`ProductError::MalformedSymbolData { factor, expected, actual, storage_len }`:
`factor` is the zero-based immediate position, `expected` and `actual` are
axis vectors (four for F, two for R/A/B), and `storage_len` is the number of
scalars returned by `SymbolBlock::as_slice`. Equal element counts do not excuse
incorrect axes. For a nested product, the enclosing `Factor0` .. `Factor11`
variants (tuples) or `Factor { index, source }` (runtime products) retain the
path to the malformed factor. Zero-channel queries skip
factor symbol evaluation. `ProductError` is non-exhaustive; matches need a
fallback arm.
