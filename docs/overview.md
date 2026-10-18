# fusion-symbols overview

What the crate covers, its feature flags, more examples, and its verification
status. The [README](../README.md) keeps only what the crate is, what it does
not do, and how to start; the sections below moved here from it.

## Families

The first families are the compact Lie groups — Racah–Wigner calculus:
irreducible representations, Clebsch–Gordan coefficients, and recoupling
coefficients (3j / 6j / F / R) for SU(2), SU(N), SO(N), and Sp(2N). Non-Lie
fusion categories sit beside them as their own top-level modules: the
multiplicity-free unitary categories of rank ≤ 4 (Fibonacci, Ising, …; up to
rank 7 with the `anyonwiki-unitary` feature) from pinned AnyonWiki tables, the abelian groups Z2,
Z_N, U(1) with fermion parity, and CU(1) = U(1) ⋊ C, now; Deligne products next. The first consumer is
[symtn](https://github.com/Ryo-wtnb11/symtn-rs).

For the Lie-group families, coefficients for any admissible labels are
computed on demand — exactly for SU(2), and through a deterministically
gauged, verification-gated numerical pipeline for the generated families.
Those families have no precomputed table and no generation-time label cut:
labels are bounded only by the machine-word ranges of the label types, which
report a typed overflow error rather than a wrong answer. The abelian and
CU(1) symbols are closed forms; only the `tables` categories ship data, the
pinned AnyonWiki tables described below.

## Supported groups

| Family | Module | Feature | Labels |
|---|---|---|---|
| SU(2) | `fusion_symbols::su2` | *(default)* | doubled spin `dj = 2j` |
| SU(N), SU(N)/Z_k, PSU(N) | `fusion_symbols::sun` | `cgc-gen` | `N-1` Dynkin labels |
| SO(N), Spin(N) | `fusion_symbols::bcd` (series `B`, `D`) | `cgc-gen` | `r` Dynkin labels |
| Sp(2r) | `fusion_symbols::bcd` (series `C`) | `cgc-gen` | `r` Dynkin labels |
| Z2, Z_N, U(1), fermion parity | `fusion_symbols::abelian` | *(default)* | `Z2Irrep`, `ZNIrrep` (charge, runtime `N`), `U1Irrep` (`i32`), `FermionParity` |
| CU(1) = U(1) ⋊ C (`O(2)`) | `fusion_symbols::cu1` | *(default)* | `CU1Irrep` (twice charge `u32`, TensorKitSectors `s`) |
| Unitary multiplicity-free categories, rank ≤ 4 (AnyonWiki tables: 74 entries incl. Fibonacci, Ising) | `fusion_symbols::tables` | *(default)* | `TableCategory` (`by_uuid`, `by_code`, `fibonacci()`, `ising()`); one-based `TableObject(1..=rank)` |
| … rank 5–7 (303 further AnyonWiki entries, about 4.2 MB of embedded tables) | `fusion_symbols::tables` | `anyonwiki-unitary` | as above |

The non-Lie modules follow the `su2` conventions (label methods `dim` / `dual`
/ `frobenius_schur` / `twist` / `fusion`; `nsymbol` / `f_symbol` /
`r_symbol` with `_checked` variants; an authority fingerprint) and
TensorKitSectors index order (and, for the closed forms, gauge); R-symbols and twists are
`num_complex::Complex64` (`f64` for the real abelian and CU(1) braidings). Their provenance is in
[`docs/references_categories.md`](references_categories.md).

The `tables` tables come from the
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
fusion-only category returns `NoBraiding` for R and twists.

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
- **Self-checks as public API** — F-unitarity and the pentagon/hexagon
  identities per generated family, so they double as generation gates and as
  oracle harnesses for your own labels.

## Feature flags

| Feature | Adds | Pulls in |
|---|---|---|
| *(default)* | Exact SU(2): 3j / 6j / Clebsch–Gordan / F / R, closed-form big-rational | `num-bigint`, `num-rational`, `num-traits` only |
| `cgc-gen` | Runtime CGC / F / R generation for SU(N) (Gelfand–Tsetlin) and SO(N)/Sp(2N) (generator bootstrap) | `tenferro-linalg` / `-cpu` / `-runtime` (the dense factorization + contraction backend) |
| `anyonwiki-unitary` | The 303 supported AnyonWiki `tables` entries of rank 5–7 (377 in all); without it they are `NotShipped` (feature added in 0.4.1) | nothing (embedded data only, about 4.2 MB) |

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
use fusion_symbols::sun::{directproduct, Irrep};

let eight = Irrep::from_dynkin(&[1, 1]).unwrap(); // SU(3) adjoint
assert_eq!(eight.dim(), 8u32.into());
assert_eq!(directproduct(&eight, &eight).unwrap()[&eight], 2);
```

SO(N)/Sp(2N) (`cgc-gen`). Generation runs through a per-(series, rank)
`CanonicalCatalog` that caches the aligned CGC; this is an Sp(4) (`C_2`)
F-symbol block:

```rust
use fusion_symbols::bcd::{f_symbol, CanonicalCatalog, Irrep, Series};

let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap(); // Sp(4)
let triv = Irrep::trivial(Series::C, 2).unwrap();
let five = Irrep::from_dynkin(Series::C, &[0, 1]).unwrap(); // the 5
let ten = Irrep::from_dynkin(Series::C, &[2, 0]).unwrap();  // the adjoint 10

let block = f_symbol(&mut cat, &triv, &five, &five, &ten, &five, &ten).unwrap();
assert_eq!(block.dims(), [1, 1, 1, 1]);
assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
```

Spinor irreps need the simply connected cover, named explicitly:

```rust
use fusion_symbols::bcd::Irrep;
use fusion_symbols::group::GroupId;

let spin7 = GroupId::spin(7).unwrap();
let s = Irrep::from_dynkin_in(&spin7, &[0, 0, 1]).unwrap();
assert_eq!(s.dim(), 8u32.into());

// The same label is not a representation of SO(7).
assert!(Irrep::from_dynkin_in(&GroupId::so(7).unwrap(), &[0, 0, 1]).is_err());
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
process, in pure Rust. It is pure representation mathematics — no
fusion-category trait vocabulary, no sector-identity types, no tensor-network
concepts, no dependency on any tensor engine — so consumers translate its
numbers into their own interfaces. To do this faithfully it consolidates the
algorithms of three production references, one per family (full provenance in
[`docs/references.md`](references.md)):

| Reference | What is taken from it |
|---|---|
| WignerSymbols.jl | the exact SU(2) model: big-rational Racah sums, prime-factorized factorials as the measured-need upgrade |
| SUNRepresentations.jl (Alex–Kalus–Huckleberry–von Delft, J. Math. Phys. 52, 023507 (2011)) | the SU(N) pipeline: Gelfand–Tsetlin patterns, exact ladder matrices, highest-weight nullspace, deterministic gauge canonicalization, weight-ladder descent |
| QSpace v4 (Weichselbaum) | the SO(N)/Sp(2N) pipeline: per-family defining-representation seeds feeding one family-generic decomposition loop; and the production discipline — abort on tolerance violation, per-representation error recording, precision tiers |

### Out of scope, deliberately

- **Fusion-category trait vocabulary.** `fusion-symbols` answers "what are the correct
  numbers"; a consumer's engine should not be able to tell whether an F-block
  came from this crate, a closed form, or a checked-in table.
- **Pentagon solving for finite fusion categories.** Anyon models (Fibonacci,
  Ising, …) have complete exact F/R data published; converting it is a
  consumer's data problem, not a computation problem for this crate. Those
  categories are provided natively as their own modules of this crate, from
  published data and closed forms.
- **Symbolic algebraic-number coefficients.** See
  [Exactness and gauge](#exactness-and-gauge).
- **Non-connected and non-compact groups.** `O(N)` and `Pin(N)` are not central
  quotients of a simply connected group and are out of scope.

Why each family gets a different algorithm — and why that choice is forced by
the group's branching structure rather than chosen for convenience — is argued
in [`docs/theory.pdf`](theory.pdf) §8, and the classification each family
comes from is §3–§5.

## Exactness and gauge

Structural and discrete data are exact; generated coefficient *values* are
deterministically gauged, verification-gated floating point. In detail:
combinatorial structure (patterns, multiplicities, weights) and discrete data
(duals, FS phases, signs, basis ordering) are exact integer/rational arithmetic;
gauge fixing is a deterministic function of the subspace; and orthogonality,
unitarity and pentagon/hexagon checks run at generation time, so a tolerance
violation is a typed error and never a silently degraded coefficient. The user-
facing summary is in
[User Guide: exact vs generated](user-guide/resources.md#exact-vs-generated-values).

The conventions that fix each coefficient's basis, sign and ordering are written
down as a **frozen normative specification** — [`docs/gauge.md`](gauge.md)
(base SU(2) and SU(N)) and [`docs/gauge_soN.md`](gauge_soN.md)
(SO(N)/Sp(2N)). Frozen means the documents are the authority and the code
implements them: a change that moves a returned value is a bug unless it ships
as a specification correction with a fingerprint epoch bump, a CHANGELOG
breaking-change entry, and regenerated golden values in the same release. The per-family
authority fingerprints are therefore *specification versions* — persist them
next to anything you derive, and compare by equality
([User Guide: gauge and reproducibility](user-guide/resources.md#gauge-and-reproducibility)).

## Verification status

Feature-complete for its v0 scope; all families are implemented and
oracle-checked. The base SU(2) surface — authority fingerprint, checked
representation layer, cache resource contract — is stable; the `cgc-gen`
generated-provider surface is unstable; the
rustdoc marker is on `sun_authority_fingerprint`, `bcd_authority_fingerprint`,
`cache::GENERATED_CACHE_MAX_BYTES` and `cache::GeneratedCacheStats`.

| Family | Pipeline | Independent verification |
|---|---|---|
| SU(2) | exact closed-form big-rational | exhaustive agreement with `wigner-symbols` 0.5.1 over its label domain, plus reference fixtures beyond it |
| SU(N), SU(N)/Z_k, PSU(N) | Gelfand–Tsetlin | signed element-wise table regeneration against SUNRepresentations.jl v0.4.0 (dim ≤ 8 every `cargo test`; a full dim ≤ 27 sweep, 76,853 F blocks, run explicitly); products cross-checked against GroupMath 1.1.3 |
| SO(N) / Sp(2N) | generator bootstrap (B/C/D) | the QSpace v4 CGC projector battery — 33 rank-2/3 channels projector-tested to round-off, 0 structural-only, 9 higher-rank rows out of the anchor's scope (see `src/bcd/qspace_oracle_tests.rs`) |
| Spin(N) | same bootstrap, Clifford seeds | `Spin(6) ≅ SU(4)` and `Spin(5) ≅ Sp(4)` on the whole weight lattice — dimensions, duals, FS indicators, every ordered product, spinor labels included (`tests/isomorphism.rs`, `tests/spin.rs`) |

Internal consistency gates run alongside: Regge/tetrahedral symmetries,
pentagon/hexagon identities, orthogonality, and `tests/gauge_golden.rs`, a
committed table of coefficient values asserted at `1e-12` that fails on any
gauge drift with no reference toolchain in the loop.
