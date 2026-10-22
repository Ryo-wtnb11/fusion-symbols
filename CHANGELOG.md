# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with the
value/gauge rule noted below.

## [Unreleased]

### Fixed

- `GroupId::admits` and `d_odd_central_class` reduce each label modulo the
  center order before summing, so labels near `i64::MAX` no longer overflow
  (a panic in debug builds, a wrong answer in release builds).

## [0.6.0] - 2026-10-22

### Added

- `lie::U1` implements `GroupRepresentations` with `Cgc = f64`: the single
  coefficient of the one-dimensional irreps, exactly `1.0` when `c = a + b` and
  `0.0` otherwise (equal to `N^{ab}_c`, so also `0.0` when `a + b` leaves
  `i32`). It never returns an error.
- `FusionCategory::contains` reports, without panicking, whether an object
  belongs to a category value; every other object-taking query panics on such
  a foreign object. Built-in categories check only label identity and range;
  products check each leaf once. The default returns `true` and is correct
  only for categories whose objects carry no category parameter.
- `FusionCategory::try_nsymbol` returns an unrepresentable multiplicity as an
  error. For products it is `ProductError::Overflow`, a nested product's
  overflow keeps its immediate factor position, and a zero factor multiplicity
  still gives `Ok(0)`. Product F and R symbols use it to build their shapes, so
  a nested product whose multiplicity overflows now returns an error from
  `f_symbol` and `r_symbol` instead of panicking.
- `FusionCategory::has_trivial_associator_gauge` and
  `FusionCategory::has_canonical_unit` certify, without evaluating any symbol,
  that in the category's documented gauge every admissible F is exactly `1`,
  respectively that the unit is self-dual, fuses as `𝟙 ⊗ a = a ⊗ 𝟙 = a` once,
  and every admissible F with `a`, `b` or `c` the unit is the identity on the
  multiplicity spaces. Both default to `false`, which claims nothing; products
  take the conjunction of their factors. `ZN`, `U1` and `FermionParity`
  certify both; `CU1` and every `LieGroup` certify only the canonical unit;
  each `TableCategory` reads both off its stored F at construction (every
  shipped entry has a canonical unit; the untwisted pointed entries also have a
  trivial associator).
- `FusionCategory::a_symbol` and `FusionCategory::b_symbol` return the
  TensorKitSectors A- and B-symbols, `A^{ab}_c = √d_a √d_b / √d_c ·
  conj(κ_a F^{ā a b}_b[𝟙, c])` and `B^{ab}_c = √d_a √d_b / √d_c ·
  F^{a b b̄}_a[c, 𝟙]`, as `RBlock` matrices with axes `[N(a,b,c), N(ā,c,b)]`
  and `[N(a,b,c), N(c,b̄,a)]`, whatever the category's `RSymbol` type. They
  need no braiding, so fusion-only tables provide them. The provided defaults
  evaluate that F for every category; `U1` and `ZN` return `N` as
  TensorKitSectors' abelian irreps do, and products take the
  last-factor-fastest Kronecker product of their factors' blocks. An absent
  channel gives the empty `0 × 0` block; a `1 × 1` block does not allocate.

### Performance

- Product object validation and shape queries now cost time linear in the
  number of leaf factors, including for nested products, which previously
  re-validated and recomputed shapes at every nesting level. Building F/R
  blocks still scales with the block size, and fusion with the number of
  outcomes. Values and errors are unchanged; a factor whose F or R block
  disagrees with its own N symbols now panics in every build.
- Cloning a generated A/B/C/D `LieIrrep` no longer allocates, so a warm
  `fusion` over these groups yields its channel labels without allocation.

## [0.5.1] - 2026-10-21

### Added

- `LieIrrep::dynkin_labels()` reads every Dynkin coordinate without allocation,
  with constant time per coordinate and an exact remaining length, for rank-one
  and generated A/B/C/D representations, including quotient groups and spinors.

## [0.5.0] - 2026-10-21

### Added

- Ordered Deligne products through `product::Product`, `product(a, b)` and
  `product![a, b, c, ...]`, with 2..=12 immediate factors, explicit nesting,
  typed factor/overflow errors, scalar promotion and factorwise F/R symbols.
- Generic `category::FBlock<S = f64>` and `category::RBlock<S = f64>`;
  `lie::FBlock` / `lie::RBlock` remain concrete `f64` compatibility aliases.

### Breaking — category API migration (#146)

All symmetry families now implement `category::FusionCategory`; Lie CGC use
`category::GroupRepresentations`. Category values supply the mathematics and
validated objects carry labels. This removes duplicate free functions and
validation paths, and lets generic identity checks work across families.
Modules follow the mathematical classification: `category`, `lie`
(positive-dimensional compact Lie groups, including U(1) and CU(1)), `finite`
(finite groups, currently Z_N), `fermion`, `table`, `product`, and `cache`.
Import the traits to call their methods.

| Old API | New API |
|---|---|
| Per-object `dim`, `dual`, `frobenius_schur`, `twist`, `fusion` on abelian, parity and CU(1) labels | Category methods taking object references; `frobenius_schur_phase` and `frobenius_schur_indicator` are distinct (indicator is zero unless self-dual) |
| `z2_*`, `zn_*`, `u1_*` N/F/R and fingerprint functions | `FusionCategory::{nsymbol,f_symbol,r_symbol,authority_fingerprint}` on `finite::ZN` / `lie::U1` |
| `Z2Irrep` in the former abelian module, including `EVEN`, `ODD`, `new`, `is_odd` | `finite::ZN::new(2)?` with `ZNIrrep::from_canonical(0 or 1, 2)?`; inspect `charge()` |
| U(1) types in the former abelian module; CU(1) types in the former group module | `lie::{U1,U1Irrep,U1Error,CU1,CU1Irrep,CU1Fusion,CU1Error}` |
| Z_N types in the former abelian module | `finite::{ZN,ZNIrrep,ZNError}` (including `from_canonical`); Z2 is `ZN::new(2)` |
| Former opt-in runtime-generation feature | `lie-generated`, enabled by default; opt out with `default-features = false`; no old-name alias |
| `ZNIrrep::unit(n)` | `ZN::new(n)?.unit()`; `ZNIrrep::new` still reduces charges, `from_canonical` rejects noncanonical charges |
| `FermionParity` in the former abelian module label | `fermion::FermionParityIrrep` (same constructors/accessors); `fermion::FermionParity` is the category |
| `fermion_parity_*` symbol/fingerprint functions | `FusionCategory` methods on `fermion::FermionParity` |
| `cu1::{CU1Irrep,CU1Fusion,CU1Error}`; `cu1::CU1` | `lie::{CU1,CU1Irrep,CU1Fusion,CU1Error}`; `cu1` is removed |
| `cu1_*` symbol/fingerprint functions | `FusionCategory` methods on `lie::CU1` |
| `tables::*` | `table::*`; metadata, provenance and named category constructors remain inherent methods |
| `TableObject(index)`, `TableObject::label()` | `TableCategory::object(index)?`, `TableObject::index()`; fields and catalog tags are private |
| Inherent table `unit`, `dim`, `dual`, `frobenius_schur`, `twist`, `fusion`, `nsymbol`, F/R and fingerprint methods | `FusionCategory` methods with object references; `fusion`, F/R return `Result`; `TableFusion` borrows the category |
| Global-form types `GroupId`, `RootSystem`, `GlobalForm`, `CenterSubgroup`, `GroupError` in the former group module | Same names under `lie`; the former group module is removed |
| `su2::*` and crate-root exact evaluators, checked exact evaluators, Regge utilities, `SignedSqrtRational`, `Su2Error`, `AdmissibilityViolation` | Same names under `lie::su2` (including exact 3j/6j/CG `_checked` forms) |
| `Su2Irrep`, `Su2Fusion`, `Su2Irrep::{new,dj,dim,dual,fusion}` | `LieGroup::new(GroupId::su(2)?)?`, `irrep(&[dj])?`, `LieIrrep::{dynkin,dim_exact}`, category `dual` / `fusion` yielding `LieFusion` |
| `su2_f_symbol`, `su2_r_symbol`, `su2_frobenius_schur`, `su2_twist`, `su2_authority_fingerprint` | `FusionCategory` methods on a rank-one `lie::LieGroup`; F/R are `lie::{FBlock,RBlock}` with inline scalar storage |
| `sun::Irrep` constructors (`from_dynkin`, `from_dynkin_in`, `from_weight`, `trivial`) | `LieGroup::new(GroupId::su(n)?)?` (or another global form), `irrep(&dynkin)?`, `unit()` and `dual()`; convert partition differences to Dynkin labels |
| `sun::Irrep::{dim,dual,frobenius_schur_phase,frobenius_schur,twist,rank}` | `LieIrrep::dim_exact`, category dual/FS/twist methods, and `group_id.root_system.rank()` |
| `sun::{directproduct,shared_directproduct,SunProduct}` and `bcd::directproduct` | `FusionCategory::fusion` shares cached product channels; `nsymbol` gives multiplicity |
| `sun::{cgc,Cgc,CgcEntry}`, `bcd::CatalogCgc` | `GroupRepresentations::cgc` returns `lie::LieCgc`; use `dims()` and `at(m1,m2,m3,mu)` |
| `sun::Irrep::{patterns,weight,dynkin}`, `sun::GtPattern` | `LieIrrep::gt_patterns() -> Option<Vec<lie::GtPattern>>`, `dynkin()`; GT basis data exist only for higher A-series groups |
| `sun::Irrep::{creation,annihilation}`, `sun::LadderEntry`, internal basis/storage helpers | No public replacement; generator matrices and storage are private |
| `sun::{f_symbol,r_symbol,sun_authority_fingerprint}`, `bcd::{f_symbol,r_symbol,bcd_authority_fingerprint}` | `FusionCategory` methods on `LieGroup`; common `lie::{FBlock,RBlock}` |
| `bcd::{Irrep,Series}`, `Irrep::{from_dynkin,from_dynkin_in,trivial,dynkin,series,rank}`, `Series::root_system` | `LieGroup` / `GroupId`, `irrep(&dynkin)?`, `LieIrrep`; rank-one B/C redirect to the exact path, D(2) reports the required SU(2) × SU(2) product |
| `bcd::Irrep::{dim,dual,frobenius_schur,twist,two_weight_multiplicities,weight_multiplicities}` | `LieIrrep::{dim_exact,weight_multiplicities}` and category methods, including spinors; weights are Dynkin coordinates |
| `bcd::Irrep::{partition,two_partition,is_spinor}`, `bcd::{Seed,CommReport,Generators,Block,Decomposition,defining_seed,spinor_seeds,decompose,decompose_defining_product,cgc_sweeps}` and their accessors | No public replacement; engines are private behind `LieGroup` |
| `bcd::CanonicalCatalog`, its constructors, `with_budget`, `bytes`, `budget`, `reset`, generation/statistics methods | Catalog is owned by `LieGroup`; use `with_catalog_budget`, `catalog_stats() -> Result<(bytes,budget),LieError>`, `reset_catalog()`; clones share it |
| Combined abelian label/overflow error | `lie::U1Error` (`U1ChargeOutOfRange`) and `finite::ZNError` (`ZeroModulus`, `NonCanonicalCharge`); non-generic, with variant names unchanged |
| `SunError`, `BcdError`, `bcd::{CatalogError,SweepError,FrError}` | `LieError`; genuine generation failures are opaque `GenerationError` diagnostics with `Display` and `Error::source` |
| `check_f_unitarity`, `check_pentagon`, `check_hexagon` | `category::{f_unitarity_residual,pentagon_residual,hexagon_residual}`; return maximum absolute residual, caller chooses tolerance |
| `bcd::check_commutators` | Private generation gates; use generic residuals for public consistency checks |
| Symbol `_checked` duals (abelian, parity, CU(1), table and SU(2) F/R); table `twist_checked` | Construction-time object validation followed by category methods; exact 3j/6j/CG checked evaluators remain |
| Former abelian `NotAdmissible` / `ModulusMismatch` errors, `CU1Error::NotAdmissible`, `TableError::NotAdmissible` | Absent channels are zero; foreign objects panic. Invalid labels/overflow retain typed errors; `TableError::UnknownObject { object }` is reported by construction |

Behaviour and persisted-data changes:

- **Absent channels:** N = 0 returns zero CGC/F/R blocks for generated families,
  replacing `ZeroFusionChannel`. Block axes follow vertex multiplicities;
  scalar families store zero with all-one storage axes. Foreign Lie, cyclic
  and table objects panic before capability checks. Fusion-only table R
  returns `NoBraiding`; twist panics. Table clones share data and identity is
  the uuid. `LieGroup` / `LieIrrep` are `Clone`, not `Copy`; group identity is
  the full `GroupId`. Warm generated F/CGC/R calls avoid the catalog lock.
- **Z2 fingerprint:** `z2_authority_fingerprint` is retired. Z2 uses the
  unchanged ZN fingerprint. Coefficient values do not change; replace the
  persisted fingerprint when migrating. Other #146 values, gauge conventions,
  fingerprint bytes (including permanent `racah:`, `fusion-symbols:abelian-u1`
  and `fusion-symbols:abelian-zn` prefixes), and cache
  APIs are unchanged. A1 dense CGC skip magnetic-selection-rule zeros without
  calling the exact evaluator; all returned values are unchanged.
- **B/C/D gauge correction (#151), epoch 3:** §4a of `docs/gauge_soN.md`
  selects in-order independent columns by CGS2 before positive-diagonal thin
  QR. The old R-row rule left signs backend-chosen when a dependent middle
  column shifted the independent direction off the diagonal. In the compared
  B2–B4/C2–C4/D3–D4 defining-self-products and D3 adjoint-self-product, only
  D3 `(0,1,1) ⊗ (0,1,1) → (0,2,2)`, copy 0, moves: 33 CGC entries reverse
  sign (including entry 7055, `-1 → +1`; maximum absolute change `2`).
  Compared B/C entries and F/R blocks are unchanged. Other products coupling
  through corrected frames may move; invalidate persisted B/C/D coefficients.
  SU(2)/SU(N) epochs are unchanged. This is a separate specification correction,
  not a coefficient change caused by #146.
- **Python bindings removed (#153):** `racah-py/`, its workspace membership,
  wheels workflow and Python guide are removed. Published `racah-py` 0.1.2
  wheels (import `racah`) remain on PyPI; a revived binding is tracked in #130.

### Changed

- **lie-generated dense backend (#151):** t4a-faer replaces Tenferro for dense
  factorizations and contractions, preserving plain column-major buffers and
  verification gates. The minimum is now 0.24.7 to exclude 0.24.6's threaded-SVD
  defect (tensor4all/faer#25). Only `std` is enabled; this crate uses explicit
  `Par::Seq` without rayon, so raising the floor changes no execution behaviour.

## [0.4.1] - 2026-10-17

### Added

- **`anyonwiki-unitary` feature (fusion-symbols #135):** ships the 303
  supported AnyonWiki (`badbfe2`) entries of rank 5–7 in `tables` (same
  selection rule as rank ≤ 4: unitary, unitary gauge, dimensions `|1/F|`),
  377 entries in all (270 braided). Without the feature they stay
  `Unsupported { reason: NotShipped }`. Default builds are unchanged: same
  74 entries, table bytes and fingerprints, and none of the new data is
  compiled. Data only, no new dependency; about 4.2 MB of embedded tables
  with the feature, and every package download carries them (+450,822 B
  compressed, about 451 KB: a 4,352,954 B `.crate`).

## [0.4.0] - 2026-10-17

### Breaking

- **`tables` generalised (fusion-symbols #134):** `tables::FibonacciTable`
  (`try_new`, `PROVENANCE`) is replaced by `tables::TableCategory`, and the
  closed-form `fibonacci` module is removed (`FibonacciObject`,
  `FibonacciError`, `fibonacci_nsymbol`, `fibonacci_f_symbol[_checked]`,
  `fibonacci_r_symbol[_checked]`, `fibonacci_authority_fingerprint`).
  `TableCategory::fibonacci()` is the single Fibonacci path: its N/F/R
  values and table bytes are those of `FibonacciTable`; against the removed
  closed form, R is bit-identical and `±F^{τττ}_τ[𝟙,𝟙]`, `dim τ` and `θ_τ`
  are correctly rounded instead of TensorKitSectors' f64 arithmetic (1–2
  ulp). Its authority fingerprint changes (uniform per-entry provenance
  strings; the projection epoch is unchanged). Migrate with
  `TableCategory::fibonacci()` and one-based `TableObject(1..=2)`.

### Added

- **`tables::TableCategory` (fusion-symbols #134):** every multiplicity-free
  AnyonWiki (`badbfe2`) entry that is unitary with dimensions
  `|1/F^{aāa}_a[1,1]|`, up to rank 4 — 74 categories, 58 braided — selected
  by `by_uuid`, `by_code` (pinned commit only) or the named `fibonacci()`
  and `ising()`, in AnyonWiki's gauge (Ising differs from TensorKitSectors'
  `IsingAnyon` by the vertex gauge `u^{σσ}_ψ = −1`). New `TableError`
  variants `UnknownCategory`, `Unsupported { reason: UnsupportedReason }`
  (`NonUnitary`, `PivotalNotUnitary`, `NotShipped`) and `NoBraiding`;
  `is_braided`, `twist_checked`, `uuid`, `code`, `rank`, `provenance`,
  `shipped_uuids`. Pentagon, hexagon, unitarity and ribbon checks run over
  every shipped entry.
- **`cu1` (symtn #2016, fusion-symbols #128):** CU(1) = U(1) ⋊ C in
  closed form following TensorKitSectors 0.3.9 `CU1Irrep` (labels as
  twice charge and `s`, its F gauge and R signs), with the per-family
  convention (`cu1_nsymbol` / `cu1_f_symbol` / `cu1_r_symbol` plus
  `_checked` variants returning `CU1Error`, `cu1_authority_fingerprint`).
  Values are bit-identical to symtn `2a6c1523` and to measured
  TensorKitSectors values. Twice charges span the full `u32`; `N` is
  exact everywhere, and only `fusion` reports a channel above `u32::MAX`.
- **`abelian` (symtn #2016, fusion-symbols #127):** Z2, Z_N (runtime
  modulus carried by the label), U(1) (`i32` integer charges) and fermion
  parity, in closed form following TensorKitSectors 0.3.9 (`ZNIrrep`,
  `U1Irrep`, `FermionParity`), with the per-family convention (label
  methods, `nsymbol` / `f_symbol` / `r_symbol` plus `_checked` variants
  returning `AbelianError`, an authority fingerprint per family). Values are
  bit-identical to symtn `49d09dc4` on every admissible tuple; unlike symtn,
  an inadmissible F or R is `0.0`, and U(1) `N` is exact (`0`) where the
  charge sum leaves `i32`.

## [0.3.0] - 2026-10-16

### Added

- **Non-Lie fusion categories, first slice (symtn #2016):**
  `fibonacci` (closed form, TensorKitSectors `FibonacciAnyon`) and
  `tables` (`FibonacciTable`, tables generated offline from the
  exact symbols of the AnyonWiki database at `badbfe2`, correctly rounded to
  binary64, with provenance and a byte-bound fingerprint; cross-checked
  against smallRankUnitaryFusionData and CategoryData.jl). Both follow the
  `su2` per-family convention and TensorKitSectors index order and gauge.
  The closed-form R-symbols and twist are correctly rounded constants, equal
  bit for bit to the table's R; every value equals the implementations ported
  from symtn at `957ccc05` except those three closed-form phases, which were
  libm-evaluated there and differ by 1–2 ulp. New dependency: `num-complex`
  0.4.

### Changed

- **Renamed `racah` → `fusion-symbols`** (lib `fusion_symbols`), and the
  repository `Ryo-wtnb11/racah` → `Ryo-wtnb11/fusion-symbols` (GitHub
  redirects the old URL). The modules, API and every coefficient value are
  unchanged: replace `racah::` with `fusion_symbols::`. The published
  `racah` 0.2.4 stays on crates.io; `fusion-symbols` is its successor
  (first release of this name). The authority fingerprint bytes keep their `racah:`
  prefix, because a fingerprint changes only with a specification
  correction, never with a rename. The Python bindings keep the import name
  `racah` for now.

## [0.2.4] - 2026-10-05

### Added

- **Twists for every family
  ([#118](https://github.com/Ryo-wtnb11/fusion-symbols/issues/118)):**
  `su2_twist(dj)`, `sun::Irrep::twist()` and `bcd::Irrep::twist()`, each
  exactly `1.0`. This is TensorKitSectors' generic
  `twist_from_Rsymbol`, θ_a = Σ_c (d_c/d_a) tr R^{aa}_c, for the `Bosonic`
  braiding of group irreps; the sum equals the trace of the tensor flip
  divided by d_a. A new test evaluates that sum from racah's own R-symbols
  over SU(2) up to j = 5, SU(3)/SU(4) irreps and Spin(5)/Sp(4)/Spin(6)
  irreps, spinors included. Downstream code no longer needs to hard-code it.
- **SU(N) Frobenius–Schur data
  ([#118](https://github.com/Ryo-wtnb11/fusion-symbols/issues/118)):**
  `sun::Irrep::frobenius_schur_phase()` (`±1.0` for every irrep;
  TensorKitSectors `frobenius_schur_phase` = sign F[a,ā,a,a,1,1]) and
  `sun::Irrep::frobenius_schur()` (`1`/`0`/`-1`, as
  `bcd::Irrep::frobenius_schur`; TensorKitSectors
  `frobenius_schur_indicator`). The phase is the closed form
  (-1)^⟨λ,2ρ⟩ = (-1)^{Σ aᵢ·i(N−i)}, exact in the SUNRepresentations gauge
  and checked against the F-symbol sweep and SUNRepresentations 0.4.0. It
  is not `+1` on every complex irrep: the SU(4) fundamental has `-1`.
  Downstream code no longer needs to derive it from an F-symbol.

### Changed

- **Warm F/CGC cache hits no longer copy coefficients
  ([#118](https://github.com/Ryo-wtnb11/fusion-symbols/issues/118)):** `FBlock` and
  `sun::Cgc` now share their buffers behind an `Arc`, so cloning one (and
  every hit of `sun::f_symbol`, `bcd::f_symbol`, `sun::cgc`) is a
  reference-count bump. `sun::f_symbol` and `bcd::f_symbol` also run their
  vertex guard on a miss only. A warm hit of an SU(3) 8⁶ block went from
  15 allocations (464 B) to 6 (144 B, the owned cache key), an SU(3) 8⊗8→8
  CGC from 7 (2352 B) to 3 (72 B), and a Spin(6) adjoint F block from 731
  (27 280 B) to 6 (144 B). No signature or accessor changed, so callers
  (TeNeT `tenet-sectors`) need no migration; `Debug` output of `Cgc` now
  shows the shared wrapper.

## [0.2.3] - 2026-10-04

### Fixed

- **SU(N) F/R contractions from cached CGC now use a stable summation order
  ([#114](https://github.com/Ryo-wtnb11/fusion-symbols/issues/114)).** Recomputing R or
  an uncached F block in one process could previously change its last bits as
  randomly seeded hash maps were traversed. The CGC descent also now accumulates
  its right-hand side in sorted weight order, and the CGC orthogonality gate
  sums in sorted coupling-index order. This can shift floating-point
  results by round-off within the existing oracle tolerance; it does not change
  the frozen gauge, its tolerance tier, or the SU(N) fingerprint epoch. Separate
  processes are still not promised bit-identical results; as measured for this
  release, 1082 SU(3) F blocks and 44 R blocks over `{3, 3̄, 6, 8}` (including the
  multiplicity-2 channel `8 ⊗ 8 → 8`) hashed bit-identically across ten fresh
  processes and independently of call order, whereas 0.2.2 gave ten distinct
  hashes.

## [0.2.2] - 2026-09-25

### Changed

- **`cgc-gen` builds against Tenferro 0.7.1 (`tenferro-* = "0.7.1"`).** Downstream
  (TeNeT) moved to 0.7.1, and a workspace that pulls both TeNeT and racah with
  `cgc-gen` on would otherwise link two different tenferro versions from one
  feature build. 0.7.1 moves the strided-rs requirement to 0.4.4 (racah has no
  direct `strided-*` dependency; this is transitive through tenferro).

  0.7.0 made `tenferro_tensor_core::DefaultScalars`/`Tensor` opaque
  ([tensor4all/tenferro-rs#1861](https://github.com/tensor4all/tenferro-rs/pull/1861)),
  but racah's dense seams (`src/sun/linalg.rs`, `src/bcd/linalg.rs`) only ever
  touch the unaffected `tenferro_runtime::Tensor` through `as_slice::<f64>()`, so
  no source change was needed.

  **No coefficient value moved.** `tests/gauge_golden.rs` passes unchanged, the
  generation-time orthogonality/unitarity/pentagon/hexagon gates pass, and the
  SU(2), SU(N) and B/C/D authority fingerprints are unchanged.

## [0.2.1] - 2026-09-21

### Changed

- **`cgc-gen` builds against Tenferro 0.6.0 (`tenferro-* = "0.6.0"`).** The
  published 0.3.0 line no longer resolves to a building closure: `strided-kernel`
  0.4.1 removed the fused API that `tenferro-internal-cpu-kernels` 0.3.0 imports
  ([tensor4all/strided-rs#261](https://github.com/tensor4all/strided-rs/issues/261)),
  so a fresh `cargo add racah --features cgc-gen` failed to compile. Tenferro 0.6.0
  builds against `strided-kernel` 0.4.1. The dense seam (`src/sun/linalg.rs`,
  `src/bcd/linalg.rs`) needed no source change and still routes every SVD, QR,
  least-squares and matmul through the CPU faer provider. No tenferro type is part
  of racah's public API (the seams are private modules and backend failures surface
  as `String` payloads), so this is not a breaking change.

  **No coefficient value moved.** `tests/gauge_golden.rs` passes unchanged, the
  generation-time orthogonality/unitarity/pentagon/hexagon gates pass, and the
  SU(2), SU(N) and B/C/D authority fingerprints are unchanged.

### Added

- **`racah-py` 0.1.2 — the exact SU(2) surface is bound
  ([#107](https://github.com/Ryo-wtnb11/fusion-symbols/issues/107)).** Python could reach SU(2)
  only through the generated SU(N) pipeline at rank 1, paying a Gelfand–Tsetlin CGC
  construction, an SVD nullspace, a least-squares descent and a QR gauge fix for values
  the crate already has in closed form. Now bound: `wigner_3j`, `wigner_6j`,
  `su2_clebsch_gordan`, `su2_f_symbol`, `su2_r_symbol`, `su2_authority_fingerprint`,
  joining the `su2_frobenius_schur` that was already there. Labels are doubled
  (`dj = 2j`, `dm = 2m`), values are scalars, and an inadmissible coupling is exactly
  `0.0` rather than an error — the crate's infallible contract. The measured gap that
  motivated it: ~3.8 ms per cold SU(2) R-symbol through the generated path, against a
  sign (consumer evidence:
  [TeNeT-py#307](https://github.com/Ryo-wtnb11/TeNeT-py/issues/307)).

  **No existing surface changed and no value moved.** The SU(N) functions, their gauge
  and `sun_authority_fingerprint()` are untouched; this adds a tier rather than
  rerouting one, and which tier a consumer calls stays the consumer's choice.

  Two relations between the tiers are now pinned by tests
  (`racah-py/tests/test_su2_exact.py`) rather than left to be rediscovered: F-symbols
  and R-symbols **agree** to round-off (worst |diff| 1.3e-15 over 55 R triangles and 76
  F classes), while the **CGC differ by exactly one sign per channel**, uniform in the
  magnetic indices and equal to `su2_r_symbol(dj1, dj2, dj3)`. The two are consistent:
  F and R are gauge-invariant combinations in which that phase cancels, and CGC are
  gauge data. Mixing the tiers' CGC without the factor gives wrong signs and no error.

  The crate's `*_checked` twins are deliberately **not** bound: nothing has yet needed
  to distinguish "zero because forbidden" from "zero because zero". New User Guide
  section in [`docs/user-guide/python.md`](docs/user-guide/python.md), executed by
  `racah-py/tests/test_doc_examples.py`.

  The cost figure above is the **cold** path and is recorded here rather than asserted:
  a test process cannot measure it, because the coefficient caches are process-global
  and whichever test ran first has already warmed them.

  `racah-py/README.md` -- the PyPI long description -- now describes both surfaces and
  the CGC phase relation, and its code blocks are executed by
  `racah-py/tests/test_doc_examples.py` alongside the User Guide's, so the published
  quickstart cannot rot silently.

- **`racah-py` 0.1.1 — real Python documentation.** The wheel now ships full
  type stubs ([`racah-py/racah.pyi`](racah-py/racah.pyi)) with per-function
  docstrings (precise NumPy types, array shapes, the CGC/F/R multiplicity-axis
  semantics, raised exceptions) plus the `py.typed` marker, so
  `mypy`/`ty`/IDEs resolve the typed surface. Every PyO3 function and method
  carries a matching `__doc__`. New User Guide chapter
  [`docs/user-guide/python.md`](docs/user-guide/python.md): install from PyPI,
  quickstart, worked CGC and F-symbol examples with the multiplicity axes in
  NumPy code, and the fingerprint/epoch contract for Python consumers.
  Documentation and packaging only — no behavior, value or fingerprint change.

## [0.2.0] - 2026-08-18

**Value/gauge status of this release:** the one value-affecting change is the
[#90](https://github.com/Ryo-wtnb11/fusion-symbols/issues/90) base-case frame
correction below — `SO(2r+1)`/`SO(2r)`/`Spin(N)` coefficient values move and
`bcd_authority_fingerprint()` is now `epoch=2`; persisted B/D coefficients
generated under `epoch=1` must be regenerated. `Sp(2r)` values are
byte-identical, and the SU(2) and SU(N) fingerprints do not move. No
coefficient, gauge, normalization or fingerprint change has landed since that
`epoch=2` correction; everything else in this release is API surface
(`fusion_symbols::group`, `Spin(N)`, `from_dynkin_in`, the `BcdError` rename), docs,
tests and CI.

### Added

- **Documentation reorganized into five layers** with a top-level index
  ([`docs/README.md`](docs/README.md)). New task-oriented **User Guide**
  (`docs/user-guide/`): getting started, representations, choosing a group,
  fusion, Clebsch–Gordan, recoupling (F/R), and numerical behaviour/resources.
  New `docs/developer/` index; the coefficient-cache audit and trade-off
  documents and their raw JSONL moved there, marked non-user-facing. The README
  is roughly 60% shorter: the provider contract, cache contract, layering
  diagram, kernel routing and verification-strategy prose moved to the User
  Guide, the crate rustdoc and the developer docs rather than being duplicated.
  Rustdoc gained per-item examples and explicit **Returns** sections stating
  array shapes, axis order and multiplicity-index meaning for the CGC / F / R
  surfaces; `tests/doc_examples.rs` compiles and runs every code block in the
  README and the guide. Documentation only — no API, convention, gauge or
  coefficient value changes.

- **`sun::Irrep::from_dynkin_in(&GroupId, &[i64])`** — the `A`-series
  form-aware constructor, matching `bcd::Irrep::from_dynkin_in`
  ([#87](https://github.com/Ryo-wtnb11/fusion-symbols/issues/87)). `SU(N)/Z_k` (`k | N`)
  and `PSU(N)` now restrict the admissible highest weights by `N`-ality; the
  rule itself stays in `GroupId::admits` and `sun` does not restate it.
  `from_dynkin` is unchanged and remains the simply connected `SU(N)`. **No
  coefficient value, basis ordering, phase, normalization or fingerprint
  moves**: a global form gates which weights may be requested, and every
  admissible weight goes through the one Gelfand–Tsetlin engine and the same
  (form-free) cache entry.
- **`Spin(N)` — the spinor irreps of the `B`/`D` covers**
  ([#54](https://github.com/Ryo-wtnb11/fusion-symbols/issues/54), stage (b) of
  [#87](https://github.com/Ryo-wtnb11/fusion-symbols/issues/87)). `Irrep::from_dynkin_in`
  takes a `GroupId`, so `GroupId::spin(N)` admits the spinor labels that
  `SO(N)` rejects; their dimensions, duals, Frobenius–Schur indicators, fusion,
  CGC and F/R are produced by the same bootstrap. Three parts:
  - `bcd::Irrep` now stores the **doubled** ε-basis weight `2λ`, so a spinor's
    half-integer highest weight is exact (`Irrep::two_partition`,
    `Irrep::is_spinor`; `Irrep::partition` returns `Option`, `None` on a
    spinor, and `Irrep::weight_multiplicities` likewise, with
    `two_weight_multiplicities` always available).
  - `bcd::spinor_seeds` — the second base case: the Clifford/Fock generator
    seeds for `ω_r` (`B_r`) and `ω_{r-1}`, `ω_r` (`D_r`), gated by the same
    exact `check_commutators`, specified in
    [`docs/gauge_soN.md`](docs/gauge_soN.md) §16 and pinned by new
    `tests/gauge_golden.rs` rows (`Spin(5)`, `Spin(6)`, `Spin(7)`, `Spin(10)`).
  - The canonical-parent candidate set is **class-indexed** (§14.2): a tensor
    irrep's parents are searched in the tensor sublattice only. **Every shipped
    `SO(N)`/`Sp(2N)` coefficient is byte-identical** and the `bcd` `epoch` stays
    at `1` at this stage; `bcd_authority_fingerprint()` gains no tag (§16.4
    states why). (The separate #90 correction under **Changed** below is what
    later moves the `bcd` epoch to `2`.)

  Oracles in `tests/isomorphism.rs` and `tests/spin.rs`: `Spin(6) ≅ SU(4)` and
  `Spin(5) ≅ Sp(4)` now agree on the **whole** weight lattice (dimensions,
  duals, every ordered product, Frobenius–Schur), and `Spin(3) ≅ SU(2)` is the
  form-aware low-rank redirect.

- **`fusion_symbols::group` — root datum plus global form** (stage (a) of
  [#87](https://github.com/Ryo-wtnb11/fusion-symbols/issues/87)). `RootSystem`,
  `GlobalForm`, `CenterSubgroup`, `GroupId` with fallible named constructors
  (`GroupId::su/su_quotient/psu/sp/psp/spin/so/pso/half_spin_plus/
  half_spin_minus`), and `GroupId::admits(&[i64])` — the central-character
  predicate that says which dominant weights are representations of which
  connected compact group. Ungated (pure integer arithmetic). `bcd`'s spinor
  rejection is now the one call site of that predicate, and
  `Series::root_system(r)` is the single bridge to the label-lattice type.
  Conventions pinned in [`docs/references.md`](docs/references.md): the
  `D_r`-odd `Z4` generator `[ω_r]`, and the `D_r`-even half-spin forms named
  by the class they retain. **No coefficient value changes**; no `epoch` moves.

- **Frozen gauge specification.** [`docs/gauge.md`](docs/gauge.md) and
  [`docs/gauge_soN.md`](docs/gauge_soN.md) are declared **normative**: the
  documents are the authority and the code implements them. `docs/gauge.md` also
  now specifies the base SU(2) conventions (§12) and marks the rules the
  implementation fixes only implicitly. **No coefficient value changes** from
  this declaration itself; all three `epoch` tags stayed at `1` at this stage
  (the #90 correction below is the release's one epoch move). What changes is the meaning of the authority
  fingerprints: they are now **specification versions**, moving only on a
  specification correction (spec edit + `epoch` bump + CHANGELOG breaking-change
  entry + regenerated goldens, in one PR), never because a refactor moved a
  value. A refactor that moves a value is now a bug by definition.
  ([#84](https://github.com/Ryo-wtnb11/fusion-symbols/issues/84))
- **`tests/gauge_golden.rs`**: the in-repo gauge tripwire — a small committed
  table of SU(3) CGC (including the OM = 2 adjoint vertex), SU(3) F, and signed
  SO(5)/Sp(4) CGC and F values, asserted at `1e-12` in the default `cgc-gen` test
  run. It covers the drift the external oracles miss by default: SU(3) F symbols
  are otherwise pinned only by an `#[ignore]`d heavy table oracle, and the QSpace
  B/C/D anchor compares an isotypic projector that is blind to the coupled-side
  gauge. It is a drift detector, not an oracle.
- **`racah-py`**: PyO3/maturin Python bindings for the SU(N) surface, built as
  a workspace member with `cgc-gen` always on. Import name `racah`,
  distribution `racah-py`; abi3-py312 wheels are built by the `wheels`
  workflow. See [`racah-py/README.md`](racah-py/README.md).
- **Coverage badges** — self-hosted llvm-cov badge branch plus Codecov
  integration ([#93](https://github.com/Ryo-wtnb11/fusion-symbols/pull/93)).
- **`CITATION.cff`**, README badges and a Citation section
  ([#92](https://github.com/Ryo-wtnb11/fusion-symbols/pull/92)).

### Changed

- **BREAKING (error variant renamed): `BcdError::SpinorLabel` is now
  `BcdError::NotAdmissible { group: GroupId, dynkin: Vec<i64> }`**
  ([#87](https://github.com/Ryo-wtnb11/fusion-symbols/issues/87)). The old name was
  mathematically misleading: the variant is returned whenever
  `group.admits(dynkin)` is false, and `PSp(2r)`, `PSO(2r)` and the two
  half-spin forms all reject *non-spinor* weights through it — `C_r` has no
  spinor sector at all. It now carries the rejecting `GroupId` rather than just
  the `Series`. Pre-1.0, no deprecation shim: the variant is renamed outright,
  and the meaning is exactly "a dominant integral highest weight of the cover
  that is not a genuine representation of the selected global form", kept
  distinct from the invalid-label, rank, excluded-rank, generation and
  verification errors.
- `SunError` gains `NotAdmissible`, `UnsupportedRootSystem` and
  `LabelRankMismatch` for the new `sun::Irrep::from_dynkin_in` guards.

- **BREAKING (gauge specification correction, B/D coefficient values move):
  the B/D defining-rep base case is re-framed into the sweep's descending-weight
  order; `bcd_authority_fingerprint()` moves to `epoch=2`**
  ([#90](https://github.com/Ryo-wtnb11/fusion-symbols/issues/90)). The QSpace-ported
  defining seed was stored in QSpace's `Setup_*` state order while every
  rediscovery of the same irrep inside a product was produced in the sweep's
  descending-weight order, and `assemble_cgc` exempted base cases from both the
  coherence guard and the intertwiner alignment — so the two frames were mixed
  silently inside a contraction. Measured on `epoch=1`: the `B_2` vector
  pentagon residual `0.285_239_560_970_874_55`, the `D_3` vector pentagon
  residual `0.129_099_444_873_580_74`, and `B_2 F(1,v,v;adj|v,adj) = 0.0` where
  unitarity forces `1.0`. The seed is now put through one §1–§8 sweep pass over
  its own carrier before it enters the catalog (`docs/gauge_soN.md` §14.2,
  "Base-case frame"), so there is one frame convention in the crate, and the
  base-case exemption is **removed**: the coherence guard and the alignment now
  bind every block, base cases included, which makes this defect class
  impossible to reintroduce silently. Consequences for consumers: persisted
  `SO(2r+1)`/`SO(2r)`/`Spin(N)` coefficients derived under `epoch=1` are stale
  and must be regenerated; `Sp(2r)` (the `C` family) values are **unchanged** —
  `Setup_SpN`'s order was already descending-weight, which is why the C gates
  always closed. `tests/gauge_golden.rs` is regenerated for the moved entries.

- **`BcdError::ExcludedRank::redirect` is form-aware** (stage (b) of #87, Q3):
  the low-rank isomorphism is an isomorphism of *groups*, so `Spin(3)` now
  redirects to `"use SU(2) instead"` while `SO(3)` redirects to `"use SU(2)
  with integer j only instead"` (likewise `Spin(4)`/`SO(4)`). Diagnostic strings
  only; no coefficient value changes.
- **`bcd::Irrep::partition` returns `Option<Vec<i64>>`** and
  `bcd::Irrep::weight_multiplicities` returns `Option<…>`, both `None` on a
  spinor irrep, whose ε-basis weights are half-integers. The always-exact
  doubled forms are `two_partition` / `two_weight_multiplicities`.
- Docs: replaced the "no label ceiling" absolute with the machine-word label
  bounds that report a typed overflow, and corrected "selectable dense
  backend" to the single Tenferro seam with no public backend-selection API.
- **`wheels` workflow: full abi3 wheel matrix + PyPI trusted publishing for
  `racah-py`** ([#103](https://github.com/Ryo-wtnb11/fusion-symbols/pull/103)): linux
  x86_64/aarch64, macOS arm64/x86_64, windows x86_64, plus an sdist. Publishing
  is gated on `py-v*` tags/releases, so the crate's own `v*` releases never
  touch PyPI. `racah-py` itself is unversioned by this release (stays 0.1.0,
  `publish = false` for crates.io) and releases separately.

## [0.1.1] - 2026-08-12

This release publishes the generated-provider dependency closure against the
published Tenferro 0.3.0 registry line.

### Changed

- Use registry `tenferro-* = "0.3.0"` dependencies for `cgc-gen`.
- Document crates.io installation and the published feature configuration.

Generated-provider (`cgc-gen`) observability and convention-identity surface
(issue [#47](https://github.com/Ryo-wtnb11/fusion-symbols/issues/47)). This whole surface
is **unstable: shape may change while the generated-provider contract is
negotiated** — Cargo features cannot express instability tiers, so the rustdoc
labels plus issue #47 are the ledger.

### Added

- **Per-tier coefficient-cache trim**: `trim_to(CoefficientCacheTier, bytes)`
  releases the oldest FIFO prefix of exactly one tier and reports its
  linearization-point charged-entry accounting in `CacheTrimReport`. It is a
  process-global single-owner lifecycle operation; it preserves the one-shot
  cache budget and makes no allocator/RSS release claim.

- **One-shot coefficient-cache budgets**: `CoefficientCacheBudgets`,
  `CoefficientCacheTier`, `configure_cache_budgets`, and `cache_budgets` let
  applications shrink independent process-local tier caps before first use.
  Zero retains no entry; reset preserves the selected policy.

- **Shared SU(N) product tier** (`cgc-gen`, issue #59): exact
  `directproduct` decompositions are retained once as sorted shared channels
  under an order-normalized irrep pair. The unchanged public API reconstructs
  its `BTreeMap`; Racah-private multiplicity and channel consumers use the
  shared value directly. The tier is bounded at 256 entries and a 128 KiB
  retained-charge backstop, sized from the checked-in collector and a
  downstream SU(3)+SU(4) Generic HomSpace/topology probe.
- **Generated-tier cache stats** (`cgc-gen`): `generated_cache_stats() ->
  GeneratedCacheStats` (`#[non_exhaustive]`, reusing `TierStats`) reports the
  five generated tiers (SU(N) product / CGC / F, B/C/D CGC / F) per-tier plus a
  field-wise `total()`. `GeneratedCacheStats` `bytes` fields are conservative
  retained charges of cache-owned entries, with container scaffolding,
  allocator/RSS costs, external clones, and returned values excluded.
  `GENERATED_CACHE_MAX_BYTES` (640 MiB + 128 KiB) is the documented aggregate
  retained-charge cap, tied to the per-tier caps by a `const` assertion.
  Two-layer cache story: base = `BASE_CACHE_MAX_BYTES`, generated =
  `GENERATED_CACHE_MAX_BYTES`, whole = the documented sum; no cross-feature
  constant. `reset()` clears the generated tiers alongside the base ones.
- **Generated authority fingerprints** (`cgc-gen`):
  `sun::sun_authority_fingerprint()` and `bcd::bcd_authority_fingerprint()`
  (`&'static [u8]`). Their contract is weaker than the exact SU(2) fingerprint —
  equal fingerprints identify the same convention, generation pipeline, and
  tolerance policy, but do not imply byte-identical values or independently prove
  numerical agreement (verification gates and oracles own that). Epochs are
  per-family and independent. Backend identity is excluded by design.
- **Backend structural-identity gate** (`cgc-gen`, D2): a test asserting the
  discrete/structural generation outputs are a function of the convention alone
  (stable across independent in-process runs), the single-backend reduction of
  the cross-backend gate.

## [0.1.0] - 2026-07-24

First tagged release of the v0 scope: the full representation-theory
coefficient set for SU(2), SU(N), SO(N), and Sp(2N), computed on demand with no
label ceiling.

### Added

- **Exact SU(2)** (default build, no features): 3j, 6j, Clebsch–Gordan, and
  F / R / Frobenius–Schur symbols in closed-form big-rational arithmetic with a
  single final rounding. Dependency-light (`num-bigint` / `num-rational` /
  `num-traits` only).
- **Generated SU(N)** (`cgc-gen` feature): the Gelfand–Tsetlin pipeline — CGC,
  F, and R with outer-multiplicity indices.
- **Generated SO(N) / Sp(2N)** (`cgc-gen` feature): the generator-bootstrap
  pipeline over the B/C/D Cartan series — CGC, F, and R.
- **Base SU(2) provider contract:**
  - `su2_authority_fingerprint()` — opaque bytes identifying the value-fixing
    convention set; compared by equality, changed only on a value-affecting
    breaking release.
  - Checked representation surface — `Su2Irrep` (with `dj` / `dim` / `dual` /
    `fusion`), `Su2Fusion`, `Su2Error` / `AdmissibilityViolation`, and the
    `wigner_3j_checked` / `wigner_6j_checked` / `clebsch_gordan_checked` /
    `su2_f_symbol_checked` / `su2_r_symbol_checked` functions. Additive over the
    infallible zero-convention functions; distinguishes `Ok(0)` (an admissible
    accidental zero) from `Err(NotAdmissible)` (a forbidden coupling).
  - Cache resource contract — `BASE_CACHE_MAX_BYTES` static partition over the
    three base tiers, `base_cache_stats()` / `BaseCacheStats` / `TierStats`
    per-tier statistics (entries, bytes, hits, misses, evictions), and
    single-owner `reset()`.
- **Self-check / oracle batteries**, shipped as public API and used as
  generation gates: CGC orthogonality, F-unitarity, R-orthogonality, and the
  pentagon / hexagon identities.

### Notes

- Not published to crates.io: the `cgc-gen` feature depends on the unpublished
  `tenferro-rs`, so a crates.io release is blocked upstream. The git dependency
  is the supported path.

### Versioning policy

Coefficient *values* are floating point, but the *computation* is exact:
combinatorial structure, discrete data (duals, signs, Frobenius–Schur phases),
and gauge fixing are deterministic. Any change that can alter a coefficient
value, its normalization, or its canonical gauge is a **breaking** change, so
consumers may key caches and persisted data on the crate version. For the base
SU(2) provider this rule is mechanized by `su2_authority_fingerprint()`: its
epoch is bumped only on such a value-affecting release, so a fingerprint change
and a breaking release are one reviewable event.

[Unreleased]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Ryo-wtnb11/fusion-symbols/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Ryo-wtnb11/fusion-symbols/releases/tag/v0.1.1
[0.1.0]: https://github.com/Ryo-wtnb11/fusion-symbols/releases/tag/v0.1.0
