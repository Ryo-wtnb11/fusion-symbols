# Differences from TensorKitSectors.jl

Symbol conventions follow TensorKitSectors 0.3.9; enumeration follows 0.3.10.
The pinned sources are in [algorithm references](references.md) and
[category provenance](references_categories.md). This page records intentional
API and convention differences, not a claim of complete feature parity.

## Design and values

| Area | fusion-symbols contract and reason |
|---|---|
| Boundary | Standalone mathematical category values, no `Sector` interface or tensor engine; consumers own tensor operations |
| Lie coefficients | Built-in runtime SU(N)/SO(N)/Spin(N)/Sp(2r) generation with public CGC and frozen gauge specifications |
| Errors | Constructors validate objects; foreign objects panic; N = 0 is a value; absent vertices give zero symbols; numerical failures are typed errors |
| A/B symbols | Always `category::RBlock`, including scalar families, so the default can construct a result for every category |
| Scalars | One associated `Scalar` (`f64` or `Complex64`), rather than separate fusion/braiding scalar types; abelian/parity symbols use `f64` rather than TKS `Int` |
| CU(1) forbidden R | `+0.0`, including cases where TKS computes `-0.0`; uniform exact-zero convention |
| Tables | AnyonWiki source gauge retained verbatim; only unitary entries with compatible pivotal dimensions are shipped |
| Fibonacci | Same gauge and braiding as TKS; exact-source rounding can differ from TKS floating arithmetic by a few ulp |
| Ising | TKS `(I,σ,ψ)` maps to source `(1,3,2)`, with vertex gauge `u^{σσ}_ψ = −1` (R identical, six F signs differ) |
| Resource control | Bounded coefficient caches and explicit application-owned policy; opaque fingerprints for persisted derived data |

Exact SU(2) evaluators retain infallible and `*_checked` entry points. All
category operations use `FusionCategory`; braiding capability errors apply
even to absent channels. See [validation](overview.md#validation-and-errors).

## Labels

| Family | Encoding difference |
|---|---|
| Rank-one Lie | `u32` doubled spin `dj = 2j`, not `HalfInt`; projections doubled too |
| Z_N | Runtime modulus in category and label, rather than a type parameter; `new` reduces charges, `from_canonical` validates |
| U(1) | Integer `i32` charges, excluding `i32::MIN`; half-integer charges cannot be represented |
| CU(1) | `twice_charge: u32` and validated `s`; full `u32` range, separate category provider |
| Tables | Validated one-based source indices with a private catalog tag, rather than named TKS objects |

## Object enumeration

`FusionCategory::{object_count,object_at,object_index}` corresponds to TKS
`values(I)` with `length`, `getindex`, `findindex`. Enumeration order is frozen
and independent of object `Ord` or fusion-iterator order.

| Category | Order (zero-based index) |
|---|---|
| `ZN(N)` | Charges `0,…,N−1` |
| `FermionParity` | Even, odd |
| `U1` or U(1) through `LieGroup` | `0,1,−1,2,−2,…`; integer subsequence of TKS's `0,½,−½,1,−1,…`, reindexed densely |
| `CU1` | `0+`, `0−`, then `twice_charge = 1,2,…` |
| Rank-one Lie | Admitted doubled spins increasing; SO(3)/PSU(2) skip odd labels |
| SU(N), N ≥ 3, simply connected | Dynkin labels by increasing total, then lexicographic (SUNRepresentations 0.4.0 order) |
| Tables | Source index minus one; Ising `(I,ψ,σ)`, whereas TKS uses `(I,σ,ψ)` |
| Composite Lie covers (including Spin(4)) | Product order over ordered simple covers then torus charges; D₂ expands to α₁ then α₂ |
| Products | Bounded Manhattan order: increasing sum of immediate factor indices, then lexicographic, skipping finite bounds |

Indices are `u64`, not Julia's one-based indices. Infinite categories return
`Ok(None)` despite machine-limited labels; unrepresentable labels/indices are
`EnumerationError::Overflow`. Finite counts above `u64::MAX` can overflow
without preventing individual queries. SU(N) quotients (N ≥ 3, nontrivial
quotient), higher-rank simple B/C/D, and every composite Lie quotient return
`NotEnumerable`. Composite covers also require every factor to enumerate.

A product is enumerable only if every factor is; `NotEnumerable` takes
precedence over other factor errors. The product count is infinite if any
factor is, even when another finite count overflows. A factor count above
`u64::MAX` does not bound any addressable product tuple. Product unranking uses
exact counts and binary searches instead of the reference's sequential search.

## Deligne products

`Product` retains tuple factors (2..=12) or homogeneous vector factors (any
length), explicit nesting and trivial factors; TKS `⊠` flattens and absorbs
`Trivial`. Thus nested and flat labels,
fingerprints and enumerations differ. For example, index 5 of `(U(1) ⊠ U(1)) ⊠ U(1)`
is `((1,0),0)`, while index 3 of the flat product is `(1,0,0)`.

Fusion varies the first factor fastest, each F/R/A/B multiplicity axis the
last factor fastest. Buffers are Rust row-major: compare indexed values with
Julia arrays. Scalar promotion/multiplication and derived dimensions fold
left to right; TKS derives square roots/inverses from the product dimension,
which may differ by rounding. Fermionic R/twist multiply factorwise without a
cross-factor sign or imposed parity/charge relation. Products of
`GroupRepresentations` provide lazy tuple or vector CGCs through `CgcBlock`,
with the last factor fastest on all four axes; they add no coefficient cache.
Initialized fusion iterators reset by cloning;
the higher-ranked iterator bound admits owned providers only.

### Reference guard inventory

The port uses TensorKitSectors 0.3.9 `product.jl::Nsymbol`, `Fsymbol`,
`Rsymbol`, `fusiontensor`, `⊗`, style methods, `auxiliary.jl::_kron` / `_kron_promote`,
and `sectors.jl` style `&` methods. Their relevant guards and assumptions map as follows:

| Reference guard or invariant | Rust mapping |
|---|---|
| `SectorTuple`, same sector type `I` for symbol inputs | Compile-time tuple/object types and factor `contains` preflight; vector lengths must equal the factor count. Foreign labels panic loudly. |
| Runtime Cartesian indices and lazy CGC axes fit their public index types | Fusion checks the product length for `usize` overflow; CGC checks each axis, without imposing a dense element-count limit. Empty products use one scalar entry. |
| `unit` rejects `GenericUnit` | N/A: `category::FusionCategory::unit` requires one simple unit. |
| Scalar `_kron_promote` fills the logical N-derived shape | Logical axes queried independently of `SymbolBlock::dims`; zero axes skip values, all-one axes stay inline. |
| `_kron` assumes matching ranks and valid Cartesian indices | Fixed block ranks; each level expands in place with per-axis digit countdowns, last factor fastest; shape/count/byte overflow is a typed `Overflow`. |
| Julia product iteration can restart each component | Clone initialized iterators; nonempty reset is a documented loud-panic invariant. A valid clone preserves the initialized iterator sequence. |
| Styles restrict braiding and symbol availability | R capability errors and twist panic after full preflight; no special-case branch bypasses validation. |
| Julia multiplicity/product arithmetic fits its representation | Checked Rust arithmetic, zero-wins N, documented `nsymbol` panic or typed `try_nsymbol`/F/R `Overflow`. |
| Single-sector symbol base cases | Singleton vectors delegate symbols to the factor authority; the empty vector supplies the scalar identity. Tuples accept 2..=12 factors. |

No reference assertions are present on these product-symbol paths. TKS
generic units and flattening are outside the approved port;
CGC value indexing follows `auxiliary.jl::_kron`: the last factor varies fastest
on every axis. `CgcBlock::at` panics for out-of-range indices, and product CGC
queries check combined axis lengths for typed `Overflow` before returning.
The scalar abelian case maps an absent channel to a zero multiplicity axis. Factor generation retains its own verification
gates; composition only consumes those authorities. Independent indexed
oracles are in `tests/product.rs`; structural identities use
`tests/category_conformance.rs::product_identities` and
`tests/category_conformance.rs::product_multiplicity_residuals`. Composition allocation checks are in
`tests/product_alloc.rs::warm_product_composition_allocates_nothing`.

## Composite Lie restriction: guard inventory

`src/lie.rs::{coefficient_roots,LieGroup::irrep,LieGroup::factor_labels}`
validate engine availability and outward group identity before Product queries.
`src/lie.rs::LieGroup::composite_label` reattaches the complete datum; quotient
fusion closure follows from `ν−λ−μ ∈ Q ⊆ X`, rather than filtering channels.
No numerical generator or reference coefficient routine is ported for composites.

| Reference constraint | Mapping |
|---|---|
| TKS `product.jl::SectorTuple` common label type and factor order | Private factor labels plus the foreign-object panic at the outward GroupId boundary |
| TKS Cartesian shape/storage assumptions | `src/product.rs::ProductError::checked_block`, checked N/axis/storage arithmetic; malformed data retain their indexed source chain |
| Restartable product iterators | `src/product.rs::{advance,restarted}` clone initialized iterators; changed replay is a documented loud-panic invariant |
| Machine integer representability | Product overflow maps to `LieError::LabelOverflow`; infallible weight multiplicities panic on checked `u64` multiplication overflow |
| Root datum has Q contained in X | `src/lie/group.rs::GroupId::new` typed lattice validation; admission precedes engine label construction |
| Canonical generator resources | `src/lie/bcd/catalog.rs::CanonicalCatalog::seed_requirement` preflights checked seed totals before composite catalogs are allocated |
| Numerical generation/verification | Simple-factor gates run as for single factors, including special cases; Kronecker orthogonality and identities follow from the factor identities, verified by `tests/category_conformance.rs::composite_lie_identities` |
| CU1 constructor, SUN UInt8 labels, and new dimension/generator ports | N/A: no such constructor or numerical routine is ported here |

D₂'s convention is α₁ then α₂ in the exact A₁ gauge. N/F/R/A/B and CGCs use
Product's factor order, with the last factor fastest on each coefficient axis;
fusion varies the first factor fastest. Composite fingerprints use
Product framing of factor fingerprints. Quotienting adds no gauge change and
needs no epoch bump; equal fingerprints do not equate outward group identities.

### Connected torus integration guards

The arithmetic authority is `src/lie/u1.rs::{U1Irrep::from_exact,U1::fusion,U1::dual}`.
The adapters `src/lie.rs::{LieGroup::irrep,LieGroup::fusion}` map invalid
charges and unrepresentable fusion outcomes to typed `LieError::LabelOverflow`.
`src/lie/u1.rs::U1Irrep::new` rejects `i32::MIN` with `U1Error`, proving dual
negation safe. `src/lie/u1.rs::U1::nsymbol` decides the vertex condition exactly
in `i64`; `try_nsymbol` uses the infallible default. An unrepresentable sum
cannot equal a representable output charge, so every symbol returns an exact
zero channel. `tests/lie_torus.rs::typed_u1_matches_liegroup_bitwise` checks
both paths, including boundaries and the independent dual oracle.
`LieGroup::label` checks full datum identity before every symbol shortcut.
`LieGroup::composite_label` asserts lattice closure: root containment and additive
charges keep every fusion outcome admitted. Ordered cover factors reuse Product's
checked shape/storage/multiplicity guards. Torus F/R/A/B/CGC
blocks use the same exact vertex condition, including every shortcut.
The closed forms are exact zeros and ones, so numerical generation gates are
vacuous; tensor products and lattice restriction preserve factor identities.
Closed-form U1 calls never generate or sweep, including warm repeats in the
bitwise equivalence test; no coefficient cache is needed.
No numerical or external constructor is ported here (additional SUN/CU1
reference guards are N/A). Warm generated calls are checked directly with
`src/lie.rs::tests::torus_composites_warm_calls_do_not_generate` and the
`src/lie/sun/cgc.rs::GENERATIONS` counter.
