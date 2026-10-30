# Differences from TensorKitSectors.jl

Symbol conventions follow TensorKitSectors 0.3.9; enumeration follows 0.3.10.
The pinned sources are in [algorithm references](references.md) and
[category provenance](references_categories.md). This page records intentional
API and convention differences, not a claim of complete feature parity.

## Design and values

| Area | fusion-symbols contract and reason |
|---|---|
| Boundary | Standalone mathematical category values, no `Sector` interface or tensor engine; consumers own tensor operations |
| Lie coefficients | Built-in runtime SU(N)/SO(N)/Spin(N)/Sp(2r)/G₂/F₄/E₆/E₇/E₈ generation from one exact Cartan engine (TKS has no exceptional groups) with public CGC and frozen gauge specifications |
| Errors | Constructors validate objects; foreign objects panic; N = 0 is a value; absent vertices give zero symbols; numerical failures are typed errors |
| A/B symbols | Always `category::RBlock`, including scalar families, so the default can construct a result for every category |
| Scalars | One associated `Scalar` (`f64` or `Complex64`), rather than separate fusion/braiding scalar types; abelian/parity symbols use `f64` rather than TKS `Int` |
| CU(1) forbidden R | `+0.0`, including cases where TKS computes `-0.0`; uniform exact-zero convention |
| Tables | AnyonWiki source gauge retained verbatim; only unitary entries with compatible pivotal dimensions are shipped |
| Fibonacci | Same gauge and braiding as TKS; exact-source rounding can differ from TKS floating arithmetic by a few ulp |
| Ising | TKS `(I,σ,ψ)` maps to source `(1,3,2)`, with vertex gauge `u^{σσ}_ψ = −1` (R identical, six F signs differ) |
| Resource control | Bounded coefficient caches and explicit application-owned policy |
| Hexagon check | `hexagon_residual`'s second hexagon uses the adjoint of each R block for the inverse braiding; TKS `hexagon_equation` uses the elementwise `conj`, which equals the adjoint only for symmetric blocks and so rejects valid data with `N ≥ 2` vertices between different irreps |

Exact SU(2) evaluators return exact zero for forbidden labels, as in
WignerSymbols.jl. N and the Frobenius–Schur phase return `Result`: N = 0 is
`Ok(0)`, and only genuine computation failures are errors. All
category operations use `FusionCategory`; braiding capability errors apply
even to absent channels. See [validation](overview.md#validation-and-errors).

## Labels

| Family | Encoding difference |
|---|---|
| Rank-one Lie | The label is the `u32` Dynkin label `dj = 2j`; `HalfInt` spins only through `LieGroup::irrep_spin` and `LieIrrep::spin` on SU(2) and SO(3) (any rank-one presentation); projections doubled too |
| Z_N | Runtime modulus in category and label, rather than a type parameter; `new` reduces charges like `ZNIrrep{N}(n)` |
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
quotient), higher-rank simple B/C/D, G₂, F₄, E₆, E₇, E₈, and every composite Lie quotient return
`NotEnumerable`. Composite covers also require every factor to enumerate.

A product is enumerable only if every factor is; `NotEnumerable` takes
precedence over other factor errors. The product count is infinite if any
factor is, even when another finite count overflows. A factor count above
`u64::MAX` does not bound any addressable product tuple. Product unranking uses
exact counts and binary searches instead of the reference's sequential search.

## Deligne products

`Product` retains tuple factors (2..=12) or homogeneous vector factors (any
length), explicit nesting and trivial factors; TKS `⊠` flattens and absorbs
`Trivial`. Thus nested and flat labels
and enumerations differ. For example, index 5 of `(U(1) ⊠ U(1)) ⊠ U(1)`
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
