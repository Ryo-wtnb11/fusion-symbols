# Design and contracts

`fusion-symbols` provides representation and fusion-category mathematics.
Consumers own tensor-network representations and operations. There is no tensor
engine dependency. The shared interface is `src/category.rs::FusionCategory`;
CGC are the separate `GroupRepresentations` capability, read through `CgcBlock`.

## Families and features

| Provider | Labels | Availability |
|---|---|---|
| `lie::LieGroup`: SU(2), SO(3), Spin(3), Sp(2), PSU(2) | One Dynkin label, doubled spin `dj = 2j` | Base |
| `lie::LieGroup`: SU(N), SU(N)/Z_k, PSU(N), N ≥ 3 | N−1 nonnegative Dynkin labels | `lie-generated` |
| `lie::LieGroup`: SO(N), Spin(N), Sp(2r), supported quotients | Rank-many nonnegative Dynkin labels | `lie-generated` |
| `lie::U1` | `U1Irrep`: integer `i32` charge, excluding `i32::MIN` | Base |
| `lie::CU1` = O(2) | `CU1Irrep`: `(twice_charge, s)` = `(0,0)`, `(0,1)`, or `(q>0,2)` | Base |
| `finite::ZN` | `ZNIrrep`: charge and runtime nonzero `u32` modulus | Base |
| `fermion::FermionParity` | `FermionParityIrrep`: even or odd | Base |
| `table::TableCategory` | Tagged `TableObject`, source index `1..=rank`, unit = 1 | Rank ≤ 4 in base; rank 5–7 with `anyonwiki-unitary` |
| `product::Product` | Ordered tuples, preserving explicit nesting | Base |

Base means available with `default-features = false`. `lie-generated` is on by
default; all dense dependencies are behind it. `anyonwiki-unitary` embeds more
data without adding a dependency. See `Cargo.toml::features`.

## Lie labels and global forms

Construct `GroupId`, then `LieGroup`, then `LieGroup::irrep(&dynkin, &[])`.
Dynkin numbering follows Bourbaki. `GroupId::sp(2*r)` uses the defining
dimension, not the rank. A1, B1 and C1 use the exact SU(2) implementation;
SO(3) and PSU(2) admit only even doubled spins. D2 is unsupported: use an
SU(2) product. Exceptional root-system variants have no coefficient engine.
Apart from CU(1) = O(2), other O(N), Pin(N), and noncompact groups are unsupported.

A global form filters labels through `src/lie/group.rs::GroupId::admits`;
it does not change coefficients on shared irreps. Spin(N) admits spinors that
SO(N) excludes. Supported constructors also include `su_quotient`, `psu`,
`psp`, `pso`, `half_spin_plus`, and `half_spin_minus`. D-series central-class
and half-spin naming conventions are fixed in [references](references.md#global-form--fixed-conventions).
Objects retain the full group tag, including isomorphic groups.
`GroupId` is a validated classification datum: ordered root-system blocks,
torus rank, and a full-rank character lattice containing every root. `new`
accepts square `i64` row generators and stores their canonical upper triangular
row HNF, with `0 ≤ H[i,j] < H[j,j]` above the positive pivots. `cover` chooses
the full weight/charge lattice. `root_systems`, `semisimple_rank`, `torus_rank`
and `character_lattice` expose the presentation and canonical basis.
`admits(dynkin, charges)` is exact and total: lengths must match, Dynkin labels
must be nonnegative, and signed charges must lie in the character lattice.
Invalid ranks, malformed/singular/root-excluding lattices, and unaddressable
sizes or unrepresentable HNF entries return typed `GroupError`s.

`torus(t)` classifies the standard positive-rank torus; `u(n)` imposes
`q ≡ Σ i·aᵢ (mod n)`. General constructors also classify coupled products,
the empty datum, and exceptional central quotients. These classifications
**enable no additional coefficient engines**: `LieGroup::new` rejects tori,
products, D2, the trivial group, and exceptional groups with a specific
`LieError::Unsupported`. Existing generated families still require
`lie-generated`. Their B/C/D seeds are checked against the catalog budget
before allocation (see [resources](user-guide/resources.md#bcd-generator-catalogs)).

`GroupId` has private fields and shares immutable data through `Arc`; it is
`Clone`, not `Copy`. Equality/hash/order use the ordered presentation and
canonical lattice, so SU(n)/Z1 equals SU(n). Distinct root presentations such
as B1/C1/A1, D3/A3, and D2/[A1,A1] remain distinct. `LieGroup::group_id` borrows
the classification datum. `LieGroup::irrep(dynkin, charges)` checks lengths and
nonnegative Dynkin labels, then lattice admission, then engine integer limits.
Every current engine requires an empty charge slice; `LieIrrep::charges` is
an empty allocation-free exact-size iterator.

`LieIrrep::dynkin_labels` reads labels without allocation; `dynkin` collects
them. Clones share weight buffers without allocation. Warm duals reuse shared
labels in the existing `SunProduct` / `BcdCgc` tiers, charged to their byte
budgets and subject to their FIFO eviction, trim and reset controls. Each irrep
retains only its own weight; coefficient cache keys keep their existing charge.
The unit is memoized per group. `dim_exact` returns `BigInt`; the category's
`dim` returns `f64`.
B/C/D dimensions and Frobenius–Schur data reuse immutable root data: one
process-lifetime entry per constructed `(series, rank)`, with O(rank³) integers,
independent of the number of labels or queries. Exact big-integer arithmetic
can still allocate for large dimensions.
`weight_multiplicities` uses Dynkin coordinates. `gt_patterns` gives the
Gelfand–Tsetlin basis for A-series rank ≥ 2, and `None` for A1 and B/C/D.

Classification validates root ranks, basis shape, full rank, and root
containment. Invalid ranks return `UnsupportedRank`; malformed or root-excluding
bases return `InvalidLattice`. Rank-sum overflow, allocation failure, and
unrepresentable canonical entries return `SizeOverflow`. General bases are
normalized with exact unimodular row operations. Named constructors build the
canonical lattice directly in O(rank) time or less; full rank and root containment
hold by construction. Only entries differing from the identity matrix are stored;
`character_lattice()` allocates a dense matrix with rank squared entries.
Unsupported presentations are rejected before any coefficient catalog is created.

## Validation and errors

`FusionCategory` accepts validated objects. Every object-taking method except
`contains` panics on a foreign object before testing capabilities or channels.
Use `contains` for a non-panicking membership check. Category equality and
hashing represent mathematical identity, independently of fingerprints.

- `fusion` yields distinct outcomes in deterministic order; `nsymbol` gives
  the exact `u32` multiplicity, including zero. `try_nsymbol` reports
  unrepresentable multiplicities as errors where supported. On `LieGroup`, it
  also propagates generated fusion arithmetic and generation failures;
  infallible `nsymbol` panics with a clear diagnostic for those failures.
- Inadmissible F/R vertices give zero symbols. Logical axes come from N;
  scalar storage still has all-one axes. Lie blocks have zero axes for absent
  channels. See [symbol layout](user-guide/README.md#symbol-layout).
- Lie constructors distinguish invalid labels, excluded global-form weights,
  label overflow and unsupported capabilities. Generated failures are opaque
  `LieError::Generation` diagnostics with an error source.
- Lie fusion and fallible N/F/R/A/B/CGC paths report `LieError::LabelOverflow`
  when a label, weight, dimension or multiplicity exceeds its integer range.
  Infallible weight-multiplicity queries panic on such failures. Dimension and
  phase formulas use exact or parity arithmetic without fixed-width overflow.
- U(1), CU(1) and rank-one Lie fusion report label overflow. `ZN` and parity
  category errors are `Infallible`; their constructors still validate labels.
  `ZNIrrep::new` reduces charges, whereas `from_canonical` rejects them outside
  `0..N`. CU(1) uses the full `u32` twice-charge range.
- Table lookup distinguishes unknown entries from known unsupported entries.
  Fusion-only tables return `TableError::NoBraiding` from R and panic on twist.
  Foreign objects are rejected first. Clones share parsed and derived data;
  categories with the same uuid accept the same objects.
- The exact `lie::su2` evaluators return exact zero for forbidden labels;
  their `*_checked` twins distinguish forbidden couplings from admissible zeros.

## Numerical and capability contracts

Lie dimensions, duals, Frobenius–Schur indicators, weights and fusion
multiplicities are exact. SU(2) coefficients use signed square-rooted rationals
and round once to `f64`. Higher-rank Lie coefficients use verification-gated
`f64`; tables are offline projections of exact AnyonWiki data to `Complex64`.
Closed-form U(1), CU(1), Z_N and parity symbols are `f64`.

SU(N) uses a highest-weight nullspace in the GT basis and lowering descent.
B/C/D uses exact fusion plus a canonical generator catalog and a numerical
decomposition sweep. Generation checks orthogonality and generator consistency;
failures are typed errors. Public F-unitarity, pentagon and hexagon residuals
are explicit caller-invoked checks with caller-selected tolerances. Dense
factorizations and CGC contractions use sequential t4a-faer; backend selection
and extended precision are not public capabilities.

`has_trivial_associator_gauge` certifies admissible F = 1 in the current gauge;
`has_canonical_unit` certifies identity F moves with an external unit leg
(`a`, `b` or `c`, not merely `d`, `e` or `f`). Both default to false and
products take the conjunction. All built-in families have a canonical unit;
U(1), Z_N and parity have trivial associators, as do qualifying pointed tables.
These are gauge-dependent certificates, not substitutes for N or braiding checks.

Gauge rules are frozen in [gauge.md](gauge.md) and [gauge_soN.md](gauge_soN.md).
Persisted derived data should carry the opaque authority fingerprint; compare
bytes without parsing. Generated fingerprints identify conventions, not
cross-process bit-identical coefficients. See [resources](user-guide/resources.md).

## Enumeration and products

Enumeration uses zero-based `u64` indices and is independent of object `Ord`
and fusion order. Infinite categories report `Ok(None)` despite finite machine
label ranges; unrepresentable labels or indices return `EnumerationError::Overflow`.
Finite counts can overflow while individual indices remain addressable.
The frozen orders and unavailable enumerations are listed in the
[TensorKitSectors comparison](tensorkitsectors-differences.md#object-enumeration).

Products have 2..=12 immediate factors. Nesting and trivial factors are retained;
objects, N, dimensions, duals, Frobenius–Schur data and twists compose factorwise.
Charge/spin and parity labels are independent, with no extra cross-factor sign.
Fusion varies the first factor fastest; each F/R/A/B multiplicity axis varies
the last factor fastest. Scalar promotion and multiplication fold left to right.
Reassociation agrees to floating-point tolerance where intermediate queries are
representable. Products of `GroupRepresentations` provide lazy tuple CGCs,
with the last factor fastest on each magnetic and multiplicity axis. Combined
axis overflow returns `ProductError::Overflow`. Products add no coefficient cache.
With inline dimension arithmetic, warm leaf A/B queries allocate only their
returned coefficient buffer (zero for empty or inline blocks).
Product A/B, like F, compose factor blocks on
each query and also allocate their storage copies and Kronecker buffers.

All factors' objects are validated before capability or shape checks. R reports
`ProductError::NoBraiding { factor }` even when N = 0; twist panics on absent
braiding. Factor failures retain their immediate index. For N, a zero factor
wins; otherwise overflow panics in `nsymbol` and is typed in `try_nsymbol`.
F/R/A/B check their own multiplicities and layouts for typed overflow. Nested
factor queries must themselves be representable. Fusion resets clone the
initialized factor iterators; the current bound supports owned category types.

Before composing a nonzero F/R/A/B block, Product checks each factor's
storage axes against its checked N symbols and requires the exact row-major
storage length. A mismatch returns
`ProductError::MalformedSymbolData { factor, expected, actual, storage_len }`:
`factor` is the zero-based immediate position, `expected` and `actual` are
axis vectors (four for F, two for R/A/B), and `storage_len` is the number of
scalars returned by `SymbolBlock::as_slice`. Equal element counts do not excuse
incorrect axes. For a nested product, the enclosing `Factor0` .. `Factor11`
variants retain the path to the malformed factor. Zero-channel queries skip
factor symbol evaluation. `ProductError` is non-exhaustive; matches need a
fallback arm.
See `src/product.rs::Product` and `ProductError`.
