# fusion-symbols overview

What the crate covers, its feature flags, more examples, and its verification
status. The [README](../README.md) keeps only what the crate is, what it does
not do, and how to start; the sections below moved here from it.

## Families

The shared contract lives in `category`. Families are classified as `lie`
(Lie-simple groups and their global forms), `abelian`, `group` (other compact
groups: CU(1) = O(2)), `fermion`, and `table`. `product` is planned;
`cache` supplies resource controls. The first consumer is
[symtn](https://github.com/Ryo-wtnb11/symtn-rs).

For the Lie-group families, coefficients for any admissible labels are
computed on demand — exactly for SU(2), and through a deterministically
gauged, verification-gated numerical pipeline for the generated families.
Those families have no precomputed table and no generation-time label cut:
labels are bounded only by the machine-word ranges of the label types, which
report a typed overflow error rather than a wrong answer. The abelian and
CU(1) symbols are closed forms; only the `table` categories ship data, the
pinned AnyonWiki tables described below.

## Supported groups

| Family | Module | Feature | Labels |
|---|---|---|---|
| SU(2), SO(3), Spin(3), Sp(2), PSU(2) | `fusion_symbols::lie::LieGroup` and `lie::su2` | *(default)* | doubled spin `dj = 2j` |
| SU(N), SU(N)/Z_k, PSU(N) | `fusion_symbols::lie` | `cgc-gen` | `N-1` Dynkin labels |
| SO(N), Spin(N) | `fusion_symbols::lie::LieGroup` (B/D) | `cgc-gen` | `r` Dynkin labels |
| Sp(2r) | `fusion_symbols::lie::LieGroup` (C) | `cgc-gen` | `r` Dynkin labels |
| U(1), Z_N (Z2 = `ZN::new(2)`) | `fusion_symbols::abelian` | *(default)* | `U1` / `U1Irrep` (`i32`), `ZN` / `ZNIrrep` (charge, runtime `N`) |
| CU(1) = U(1) ⋊ C (`O(2)`) | `fusion_symbols::group` | *(default)* | `CU1` / `CU1Irrep` (twice charge `u32`, TensorKitSectors `s`) |
| Fermion parity | `fusion_symbols::fermion` | *(default)* | `FermionParity` / `FermionParityIrrep` (bool) |
| Unitary multiplicity-free categories, rank ≤ 4 (AnyonWiki tables: 74 entries incl. Fibonacci, Ising) | `fusion_symbols::table` | *(default)* | `TableCategory` (`by_uuid`, `by_code`, `fibonacci()`, `ising()`); tagged `TableObject` from `object(1..=rank)?` |
| … rank 5–7 (303 further AnyonWiki entries, about 4.2 MB of embedded tables) | `fusion_symbols::table` | `anyonwiki-unitary` | as above |

`lie::LieGroup`, `abelian::{U1, ZN}`, `group::CU1`, `fermion::FermionParity` and
`table::TableCategory` implement
`category::FusionCategory`, the single surface for dimension, dual, FS phase
and indicator, twist, fusion and N/F/R. Objects are validated at construction;
foreign Lie, cyclic or table objects panic. N = 0 gives scalar zero or a block
with a zero axis; U(1), rank-one Lie and CU(1) fusion overflow
is a typed error. Closed-form symbols are `f64`, table symbols are `Complex64`.
Category identity includes the modulus or table uuid;
authority fingerprints describe conventions (see the migration guide in
[CHANGELOG](../CHANGELOG.md#unreleased) for the retired Z2 fingerprint).

Table objects retain one-based source indices through `TableObject::index()`.
Cloning a table category shares its derived data through `Arc`. Fusion and
braiding styles are derived at construction. Fusion-only tables return
`TableError::NoBraiding` from `r_symbol`; `twist` panics.
All families use TensorKitSectors index order; closed forms use its gauge.
Table braiding uses `num_complex::Complex64`; CU(1) braiding is `f64`.
Provenance is in [`docs/references_categories.md`](references_categories.md).

The `table` tables come from the
[AnyonWiki](https://anyonwiki.org/) database
([`anyonwiki/AnyonWikiDatabase`](https://github.com/anyonwiki/AnyonWikiDatabase),
MIT, commit `badbfe2`), which stores every symbol exactly as an algebraic
number. An offline generator (`tools/anyonwiki/gen_category_tables.py`) rounds
each one correctly to binary64; the crate ships that output and parses it, with
no runtime dependency. Every entry is in AnyonWiki's gauge. Fibonacci is
TensorKit's `FibonacciAnyon` in TensorKit's own gauge and braiding
(`R^{ττ}_𝟙 = e^{+4πi/5}`); Ising has `IsingAnyon`'s braiding and F in a
gauge differing by the vertex gauge `u^{σσ}_ψ = −1`. Entries that are non-unitary (608), whose pivotal structure is not unitary (4),
or above rank 4 without the `anyonwiki-unitary` feature (303 not shipped) are
refused with a typed `Unsupported` error (`NonUnitary`, `PivotalNotUnitary`,
`NotShipped`); a
fusion-only category returns `NoBraiding` for R and panics on twist.

## What fusion-symbols computes

For any admissible irreps of a supported group:

- **Labels and structure** — dimensions, duals, Frobenius–Schur indicators,
  weight multiplicities. Exact integer arithmetic.
- **Fusion** — the tensor-product decomposition and its multiplicities
  `N^c_ab`. Exact.
- **Clebsch–Gordan coefficients** — m-basis tensors with the outer multiplicity
  on a trailing axis.
- **Recoupling coefficients** — F-symbols (rank-4 blocks over the four vertex
  multiplicity indices) and R-symbols (braiding).
- **Self-checks as public API** — generic F-unitarity, pentagon and hexagon
  residuals in `category`, for every family; callers choose the tolerance.

## Feature flags

| Feature | Adds | Pulls in |
|---|---|---|
| *(default)* | `category`; exact rank-one `lie`; closed-form `abelian`, `group` (CU(1)), `fermion`; rank ≤ 4 `table`; `cache` | `num-bigint`, `num-rational`, `num-traits`, `num-complex` |
| `cgc-gen` | Runtime CGC / F / R generation for SU(N) (Gelfand–Tsetlin) and SO(N)/Sp(2N) (generator bootstrap) | `t4a-faer` (sequential, `std` only) (the dense factorization + contraction backend) |
| `anyonwiki-unitary` | The 303 supported AnyonWiki `table` entries of rank 5–7 (377 in all); without it they are `NotShipped` (feature added in 0.4.1) | nothing (embedded data only, about 4.2 MB) |

The feature boundary is mathematical, not organizational: SU(2) has closed forms
and needs no matrix computation, so consumers who need only SU(2) never pull a
linear-algebra stack. No fixed MSRV — `fusion-symbols` builds and is tested on current
stable Rust.

With `cgc-gen` enabled the coefficient caches may retain several hundred MiB by
default; call `fusion_symbols::cache::configure_cache_budgets` once before first use to
lower that bound
([User Guide: resources](user-guide/resources.md#bounding-cache-memory)).

## More examples

Each is a literal copy of a crate doctest, so it compiles against the current
API.

SU(N) (`cgc-gen`). Irreps are built from Dynkin labels (length `N-1`); this
decomposes the SU(3) product `8 ⊗ 8`, where the adjoint appears twice:

```rust
use fusion_symbols::{category::{FusionCategory}, lie::{GroupId, LieGroup}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

let eight = cat.irrep(&[1, 1]).unwrap(); // SU(3) adjoint
assert_eq!(eight.dim_exact(), 8u32.into());
assert_eq!(cat.fusion(&eight, &eight).unwrap()
    .map(|c| { let n = cat.nsymbol(&eight, &eight, &c); (c, n) })
    .collect::<std::collections::BTreeMap<_, _>>()[&eight], 2);
```

SO(N)/Sp(2N) (`cgc-gen`). Generation runs through a per-(series, rank)
`LieGroup` that retains canonical generator matrices; this is an Sp(4) (`C_2`)
F-symbol block:

```rust
use fusion_symbols::{category::{FusionCategory},lie::{LieGroup,GroupId}};

let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap(); // Sp(4)
let triv = cat.unit();
let five = cat.irrep(&[0, 1]).unwrap(); // the 5
let ten = cat.irrep(&[2, 0]).unwrap();  // the adjoint 10

let block = cat.f_symbol( &triv, &five, &five, &ten, &five, &ten).unwrap();
assert_eq!(block.dims(), [1, 1, 1, 1]);
assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
```

Spinor irreps need the simply connected cover, named explicitly:

```rust
use fusion_symbols::{lie::{LieGroup,GroupId}};

let spin7 = GroupId::spin(7).unwrap();
let s = LieGroup::new(spin7).unwrap().irrep(&[0, 0, 1]).unwrap();
assert_eq!(s.dim_exact(), 8u32.into());

// The same label is not a representation of SO(7).
assert!(LieGroup::new(GroupId::so(7).unwrap()).unwrap().irrep(&[0, 0, 1]).is_err());
```

## Why this crate exists

No library — in Rust, and essentially nowhere as a standalone component —
computes the *full* representation-theory coefficient set for the compact Lie
groups on demand, for any admissible labels: fusion multiplicities, dimensions,
duals, Frobenius–Schur indicators, Clebsch–Gordan coefficients, and the
recoupling data (3j / 6j and the F- and R-symbols). The existing supply stops
short in two ways. **Precomputed tables** are complete only for *finite*
symmetry sets; a compact Lie group has infinitely many irreps and tensor
products only make them larger, so any table has a cut a large-enough
calculation will exceed. **Single-group packages** solve one group at a fixed
scope and do not extend to SU(N≥3), SO(N), or Sp(2N), where no closed forms
exist and the coefficients must be *constructed*.

`fusion-symbols` removes both limits: coefficients are computed on demand, inside the
process, in pure Rust. It is pure representation and fusion-category
mathematics — no tensor-network concepts, no dependency on any tensor
engine — so consumers build their tensor operations on its numbers. To do
this faithfully it consolidates the algorithms of three production
references, one per family (full provenance in
[`docs/references.md`](references.md)):

| Reference | What is taken from it |
|---|---|
| WignerSymbols.jl | the exact SU(2) model: big-rational Racah sums, prime-factorized factorials as the measured-need upgrade |
| SUNRepresentations.jl (Alex–Kalus–Huckleberry–von Delft, J. Math. Phys. 52, 023507 (2011)) | the SU(N) pipeline: Gelfand–Tsetlin patterns, exact ladder matrices, highest-weight nullspace, deterministic gauge canonicalization, weight-ladder descent |
| QSpace v4 (Weichselbaum) | the SO(N)/Sp(2N) pipeline: per-family defining-representation seeds feeding one family-generic decomposition loop; and the production discipline — abort on tolerance violation, per-representation error recording, precision tiers |

### Out of scope, deliberately

- **Tensor-network concepts.** Legs, fusion trees, tensor maps, sector-id
  codecs and contraction belong to the consumer (symtn); `fusion-symbols`
  answers "what are the correct numbers".
- **Pentagon solving for finite fusion categories.** Anyon models (Fibonacci,
  Ising, …) have complete exact F/R data published; converting it is a
  consumer's data problem, not a computation problem for this crate. Those
  categories are provided natively as their own modules of this crate, from
  published data and closed forms.
- **Symbolic algebraic-number coefficients.** See
  [Exactness and gauge](#exactness-and-gauge).
- **Additional disconnected or non-compact groups.** CU(1) = O(2) is
  supported in `group`; other O(N), Pin(N), and non-compact groups are unsupported.

Why each family gets a different algorithm — and why that choice is forced by
the group's branching structure rather than chosen for convenience — is argued
in [`docs/theory.pdf`](theory.pdf) §8, and the classification each family
comes from is §3–§5.

## Exactness and gauge

Structural and discrete data are exact; generated coefficient *values* are
deterministically gauged, verification-gated floating point. In detail:
combinatorial structure (patterns, multiplicities, weights) and discrete data
(duals, FS phases, signs, basis ordering) are exact integer/rational arithmetic;
gauge fixing is a deterministic function of the subspace. Internal CGC
orthogonality and generator-consistency gates reject failures with typed errors.
Generic F-unitarity, pentagon and hexagon residuals are available for explicit
checks against a caller-selected tolerance. The user-facing summary is in
[User Guide: exact vs generated](user-guide/resources.md#exact-vs-generated-values).

The conventions that fix each coefficient's basis, sign and ordering are written
down as a **frozen normative specification** — [`docs/gauge.md`](gauge.md)
(base SU(2) and SU(N)) and [`docs/gauge_soN.md`](gauge_soN.md)
(SO(N)/Sp(2N)). Frozen means the documents are the authority and the code
implements them: a change that moves a returned value is a bug unless it ships
as a specification correction with a fingerprint epoch bump, a CHANGELOG
breaking-change entry, and regenerated golden values in the same change. The per-family
authority fingerprints are therefore *specification versions* — persist them
next to anything you derive, and compare by equality
([User Guide: gauge and reproducibility](user-guide/resources.md#gauge-and-reproducibility)).

## Verification status

Feature-complete for its v0 scope; all families are implemented and
oracle-checked. The base SU(2) surface — authority fingerprint, checked
representation layer, cache resource contract — is stable; the `cgc-gen`
generated-provider surface is unstable; the
generated-family fingerprint bytes are opaque. The rustdoc marker is on `FusionCategory::authority_fingerprint`,
`cache::GENERATED_CACHE_MAX_BYTES` and `cache::GeneratedCacheStats`.

| Family | Pipeline | Independent verification |
|---|---|---|
| SU(2) | exact closed-form big-rational | Condon–Shortley closed forms and recursion, Racah's sum rule, exact large-spin orthogonality, textbook values checked against WignerSymbols.jl 2.0.0 |
| SU(N), SU(N)/Z_k, PSU(N) | Gelfand–Tsetlin | CGC isometry and intertwining plus signed CGC/F/R values checked against SUNRepresentations.jl v0.4.0; in-test Brauer–Klimyk products, rows checked against GroupMath 1.1.3 |
| SO(N) / Sp(2N) | generator bootstrap (B/C/D) | generators checked as Chevalley–Serre data and the CGC of 33 rank-2/3 channels as isometric intertwiners whose isotypic projectors sum to the identity (`src/lie/bcd/intertwiner_tests.rs`); products against an in-test Brauer–Klimyk decomposition |
| Spin(N) | same bootstrap, Clifford seeds | `Spin(6) ≅ SU(4)` and `Spin(5) ≅ Sp(4)` on the whole weight lattice — dimensions, duals, FS indicators, every ordered product, spinor labels included (`tests/isomorphism.rs`, `tests/spin.rs`) |

Internal consistency gates run alongside: Regge/tetrahedral symmetries,
pentagon/hexagon identities, orthogonality, and `tests/gauge_golden.rs`, a
committed table of coefficient values asserted at `1e-12` that fails on any
gauge drift with no reference toolchain in the loop.

Rank-one A(1), B(1), C(1) groups use one exact SU(2) path and fingerprint.
Objects keep the caller's `GroupId`; isomorphic groups remain mutually foreign.
A(r), B(r), C(r), r ≥ 2, and D(r), r ≥ 3, require `cgc-gen`; without it `LieGroup::new` returns
`LieError::Unsupported { requires: "cgc-gen" }`. B/C/D use the same `LieGroup`
surface with a shared per-value generator catalog. D(2) requires the
`SU(2) × SU(2)` product.
