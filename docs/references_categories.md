# Provenance: non-Lie categories (`abelian`, `fibonacci`, `tables`)

## Abelian groups (`abelian` module)

| Field | Value |
| --- | --- |
| Reference | TensorKitSectors.jl `0.3.9`, git `67d342a130bd9e31caf3eccbf9ca46ec8b088556` (registry git-tree-sha1 `9f536263310aae90df175337d586a1655104aae3`; `src/` identical to `~/.julia/packages/TensorKitSectors/VuA9Z`) |
| Z2, Z_N, U(1): `Nsymbol`, `Fsymbol`, `Rsymbol`, FS phase | `src/irreps/irreps.jl`: `Nsymbol` (l. 57), `Fsymbol` (l. 58), `frobenius_schur_phase(::AbelianIrrep)` (l. 62), `Rsymbol` (l. 65); `BraidingStyle(::AbstractIrrep) = Bosonic()` (l. 17) |
| Z_N labels, dual, fusion | `src/irreps/znirrep.jl`: `ZNIrrep{N}(n) = mod(n, N)` (l. 24), `dual` (l. 82), `⊗` (l. 83). The modulus is a runtime `u32` carried by the label (as in symtn), not a type parameter |
| U(1) labels, dual, fusion | `src/irreps/u1irrep.jl`: `dual` (l. 30), `⊗` (l. 31). TKS stores a `HalfInt`; here integer `i32` charges without `i32::MIN` (as in symtn); half-integer charges are not representable |
| Fermion parity | `src/fermions.jl`: `dual` (l. 34), `dim` (l. 35), `BraidingStyle = Fermionic()` (l. 39), `⊗` (l. 43), `Nsymbol` (l. 45), `Fsymbol` (l. 48), `Rsymbol` (l. 51), `twist` (l. 54); FS phase from the generic `src/sectors.jl:frobenius_schur_phase_from_Fsymbol` (l. 461) |
| `dim`, bosonic twist | generic `src/sectors.jl:dim_from_Fsymbol` (UniqueFusion → 1, l. 429) and `twist_from_Rsymbol` (l. 646) |
| Scalars | `f64` (exact ±1, 0); TKS declares `Int`. Measured TKS values: `tests/fixtures/tks_abelian_oracle.jl`, hand-recorded in `src/abelian.rs::tests::matches_measured_tensorkitsectors_values` |
| Ported implementation | [symtn](https://github.com/Ryo-wtnb11/symtn-rs) `symtn-sectors/src/abelian.rs` at `49d09dc4fc0825b29d547e63a70014221c0c5406`; every value bit-identical on admissible tuples (`tests/symtn_abelian_values.rs`). Two deliberate differences: an inadmissible F or R is `0.0` here (symtn returns `±1.0` unchecked), and U(1) `N` is the exact `0` where `a + b` leaves `i32` (symtn's checked N fails) |

## Closed-form Fibonacci (`fibonacci` module)

| Field | Value |
| --- | --- |
| Reference | TensorKitSectors.jl `0.3.9`, registry git-tree-sha1 `9f536263310aae90df175337d586a1655104aae3` (verified against the local copy `~/.julia/packages/TensorKitSectors/VuA9Z`) |
| `dim`, `Nsymbol`, `Fsymbol`, `Rsymbol`, fusion order | `src/anyons.jl`: `dim(a::FibonacciAnyon)` (l. 83), `FibonacciAnyonProdIterator` iteration (l. 96), `Nsymbol` (l. 111), `Fsymbol` (l. 115), `Rsymbol` (l. 139) |
| Frobenius–Schur phase | no override; generic `src/sectors.jl:frobenius_schur_phase_from_Fsymbol` (l. 463) |
| Twist | no override; generic `src/sectors.jl:twist_from_Rsymbol` (l. 647) |
| R and twist values | correctly rounded binary64 of `e^{4πi/5}`, `e^{-3πi/5}`, `e^{-4πi/5}` from the table generator (`tools/anyonwiki/gen_category_tables.py`; check rows `R 2 2 1`, `R 2 2 2`, `T 2`), not libm `cispi`/`from_polar`. TKS 0.3.9's measured `cispi` values are ≤ 3 ulp away; F and `dim` equal TKS bit for bit (`src/fibonacci.rs::tests::matches_measured_tensorkitsectors_values`) |
| Ported implementation | [symtn](https://github.com/Ryo-wtnb11/symtn-rs) `tenet-sectors/src/fibonacci.rs` at `957ccc05737f0841b6a98047880aba201bee63a1`; values bit-identical except the three phases above, which differ by the pinned (1,1), (2,1), (1,1) ulp (`tests/tenet_values.rs`) |

The Fibonacci bodies of `anyons.jl` are textually identical in
TensorKitSectors 0.3.6 and 0.3.9.

## Pinned Fibonacci table (`tables` module)

Every number the table provider ships is traceable to the rows below. The crate
performs no derivation of the *source* data: it parses the committed tables and
derives only the API-level quantities that TensorKitSectors also derives
(dimension, Frobenius–Schur phase, twist).

### Authority — AnyonWiki database (exact symbols)

| Field | Value |
| --- | --- |
| Repository | [`anyonwiki/AnyonWikiDatabase`](https://github.com/anyonwiki/AnyonWikiDatabase) (data of [anyonwiki.org](https://anyonwiki.org/)) |
| Commit | `badbfe299dd533c12a84e817f5890059705e0a3c` (2026-09-30) |
| License | MIT, Copyright (c) 2026 AnyonWiki — copied to `data/anyonwiki-badbfe2/LICENSE` |
| Category file | `AlgebraicStructures/FusionCategories/MultFreeFusionCategories/Rank1-7.json`, SHA-256 `383cb2db58ae839f73c3d67ad79186e6ee44d69c32fd8fb202d1a4089ca7e859` (64.7 MB, 989 categories; not committed) |
| Fusion-ring file | `AlgebraicStructures/FusionRings/mult1.json`, SHA-256 `66117d72d204a5359ef43e584ed46cfe503240558aa8b9c381922b4861aa1153` (not committed) |
| Category | `anyonwiki_code [2,1,0,2,1,2,1]` (ring `2_1_0_2`, associator `f = 1`, braiding `b = 2`, pivotal `p = 1`), uuid `37fa4e42-97c6-11f1-9d0f-9b7fe5c504f9`, `in_unitary_gauge: true` |
| Fusion ring | uuid `d810a768-812e-11f1-8f24-516eff0ae265`, `mult_tab` (`N_{ab}^c = mult_tab[a][b][c]`) |
| Committed extract | `data/anyonwiki-badbfe2/fibonacci.json`: the entry's exact `f_symbols` / `r_symbols` / `p_symbols` / `quantum_dimensions` / `twists` (qqb_ids), its `numeric_*` values, uuid, code, references, software, the ring's `mult_tab`, and the source paths and hashes |

A qqb_id `a0_a1_…_an__i` is the `i`-th root of `a0 + a1 x + … + an x^n`, roots
ordered real first (ascending), then conjugate pairs by real part, then
imaginary part (database README, "Conventions used").

Key mapping onto the shipped record layout: AnyonWiki F
`a_b_c_d_e_α_β_f_γ_δ` → `a b c d α e β γ f δ` (Bridgeman's `a b c d α e β μ f ν`,
so `μ ↔ γ`, `ν ↔ δ`); R `a_b_c_α_β` → `a b c α β`. All multiplicity slots are
`1`. The pivotal coefficients are all `1` and are not shipped.

### Projection — correctly rounded binary64, offline

`tools/anyonwiki/gen_category_tables.py` (recorded run: Python 3.14.6, mpmath
1.3.0) evaluates each qqb_id with `mpmath.polyroots` at 80 significant digits,
orders the roots by the database convention, checks the residual, rounds the
real and imaginary parts to nearest binary64 (`float(mpf)`), and confirms the
rounding is unchanged at 160 digits. It writes

- `data/anyonwiki-badbfe2/fibonacci/{Nabc,F,R}.txt`, the shipped tables, as
  shortest round-trip decimals in the layouts `a b c Nabc`,
  `a b c d α e β μ f ν Re Im`, `a b c α μ Re Im`;
- `tests/fixtures/anyonwiki-fibonacci-check.txt`, one row per symbol with its
  qqb_id, the generated value and AnyonWiki's `numeric_*` value.

`tests/tables_anyonwiki.rs` checks every shipped value: (a) bit-equal to the
generator output, (b) within 4 ulp of AnyonWiki's `numeric_*` (which are not
correctly rounded: `R^{ττ}_𝟙` differs by 2 ulp in the real part, 1 in the
imaginary), (c) its minimal polynomial vanishes to rounding at it.

### The `Complex64` projection epoch

Epoch identifier:
**`fusion-symbols/tables/anyonwiki-badbfe2/projection/2026-10-16`**.

1. Each exact symbol is rounded correctly to binary64 offline (above).
2. At load, split each record on ASCII whitespace; the trailing two fields
   are the real and imaginary decimals, read with Rust's correctly rounded
   `str::parse::<f64>()`, which returns the generated bits exactly.
3. Pair them as `Complex64 { re, im }`. No re-normalisation, no gauge fixing.
4. Unlisted `N`/`F`/`R` records are exact zeros.

Changing any step is a new epoch identifier, and therefore a new authority
fingerprint. symtn's earlier epoch `tenet-category-data/projection/2026-07-27`
(step 1 replaced by "parse CategoryData.jl's 20-decimal export") maps to this
one: every table F, R and N value is bit-identical (`tests/tenet_values.rs`), only the
source, the decimal spelling of F and R, and the fingerprint are new.

### Category identity, ordering, and gauge

| Field | Value |
| --- | --- |
| Object ordering | Fusion-ring element order, one-based: `1 = 𝟙`, `2 = τ`. Frozen; not re-sorted |
| Braiding | AnyonWiki `b = 2`: `R^{ττ}_𝟙 = e^{+4πi/5}`, `R^{ττ}_τ = e^{-3πi/5}`, as TensorKitSectors `FibonacciAnyon` (`anyons.jl:Rsymbol`). The other braiding (`b = 1`) is the conjugate and is not shipped; `tables::tests::shipped_braiding_is_tensorkits_not_its_mirror` pins the choice |
| Gauge | Source gauge, loaded verbatim (unitary). Identical to the TensorKitSectors closed-form F block `[1/φ, 1/√φ; 1/√φ, −1/φ]` |

### Cross-checks (not authorities)

| Source | Pin | Check |
| --- | --- | --- |
| `JCBridgeman/smallRankUnitaryFusionData` | `a6ba1c308ddcecbb03b9651a25fa0d1afcfb7546`, `uFC/FR_2_0_2/{Nabc.txt, 0/F.txt, 0/0/R.txt}` (SHA-256 `a7a6d179…`, `00cbae54…`, `24a653d1…`; MIT, Copyright (c) 2023 Jacob C Bridgeman), copied with its license to `tests/fixtures/smallRankUnitaryFusionData-a6ba1c3/` | every record bit-equal after parsing (`tests/tables_anyonwiki.rs`) |
| CategoryData.jl `0.3.6` (`e793a08a093f6ba890a32c7e57d8e8b347441058`), artifact `data-v0.1.3` (git-tree-sha1 `90a414867a392b7b811ba6c8092e5e1a848dc4c4`, tarball SHA-256 `586f4841…5c045cf6`) | redistributes the smallRankUnitaryFusionData files byte for byte as `Nsymbols/FR_2_1_0_2.txt`, `Fsymbols/FR_2_1_0_2_0.txt`, `Rsymbols/FR_2_1_0_2_0_0.txt` (name mapping `FR_2_0_2` ↔ `FR_2_1_0_2`, braid `0/0` ↔ `_0_0`); alias `Fib = PMFC{2,1,0,2,0,0}` | its parser and TKS-derived values (`tests/fixtures/fib-categorydata-v0.1.3.txt`) via `tests/tables_reference.rs` |

### Citation

AnyonWikiDatabase asks users of its fusion-category data to cite
G. Vercleyen, *On Low-Rank Multiplicity-Free Fusion Categories*,
arXiv:2405.20075 (the entry's `references` field gives the published version,
doi:10.1007/s00220-026-05705-2), and users of fusion-ring data to cite
G. Vercleyen and J. K. Slingerland, *On low rank fusion rings*, J. Math. Phys.
64, 091703 (2023), doi:10.1063/5.0148848. The entry's software field names
Anyonica 0.9.11.5 (doi:10.5281/zenodo.10686860). The smallRankUnitaryFusionData
README states no paper citation.

### Derived quantities

Derived from the shipped tables using the TensorKitSectors 0.3.9 formulas (A/B
symbols are left to the consumer)
(`~/.julia/packages/TensorKitSectors/VuA9Z/src/sectors.jl`):

| Quantity | Formula | TKS location |
| --- | --- | --- |
| `dim(a)` | `abs(1 / F(a, ā, a, a, 1, 1))` | `dim_from_Fsymbol`, `sectors.jl:431-439` |
| `sqrtdim` / `invsqrtdim` | `sqrt(dim)` / `inv(sqrt(dim))` | `sectors.jl:440-441` |
| `κ(a)` | `sign(F(a, ā, a, a, 1, 1))` | `frobenius_schur_phase_from_Fsymbol`, `sectors.jl:463-469` |
| `θ(a)` | `Σ_{b ∈ a⊗a} dim(b)/dim(a) · R(a, a, b)`, summed in ascending `b` | `twist_from_Rsymbol`, `sectors.jl:646-647` |

### Reference fixture

`tests/fixtures/fib-categorydata-v0.1.3.txt` was emitted by
`tests/fixtures/generate_fixture.jl` run against a Julia 1.11.6 project holding
`CategoryData v0.3.6` (pinned rev above) and `TensorKitSectors v0.3.9`. It is a
genuine independent oracle: the values come out of CategoryData.jl's own
parser, sparse-array indexing, and dual/dimension/twist derivation, not out of
a re-print of the source files.

Regenerating it requires Julia and network access and is therefore an offline,
manual step. Nothing in the shipped crate invokes Julia, Mathematica, or the
network.
