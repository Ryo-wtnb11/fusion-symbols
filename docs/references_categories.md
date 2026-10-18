# Provenance: non-Lie categories (`abelian`, `cu1`, `tables`)

## Abelian groups (`abelian` module)

| Field | Value |
| --- | --- |
| Reference | TensorKitSectors.jl `0.3.9`, git `67d342a130bd9e31caf3eccbf9ca46ec8b088556` (registry git-tree-sha1 `9f536263310aae90df175337d586a1655104aae3`; `src/` identical to `~/.julia/packages/TensorKitSectors/VuA9Z`) |
| Z2, Z_N, U(1): `Nsymbol`, `Fsymbol`, `Rsymbol`, FS phase | `src/irreps/irreps.jl`: `Nsymbol` (l. 57), `Fsymbol` (l. 58), `frobenius_schur_phase(::AbelianIrrep)` (l. 62), `Rsymbol` (l. 65); `BraidingStyle(::AbstractIrrep) = Bosonic()` (l. 17) |
| Z_N labels, dual, fusion | `src/irreps/znirrep.jl`: `ZNIrrep{N}(n) = mod(n, N)` (l. 24), `dual` (l. 82), `⊗` (l. 83). The modulus is a runtime `u32` carried by the label (as in symtn), not a type parameter |
| U(1) labels, dual, fusion | `src/irreps/u1irrep.jl`: `dual` (l. 30), `⊗` (l. 31). TKS stores a `HalfInt`; here integer `i32` charges without `i32::MIN` (as in symtn); half-integer charges are not representable |
| Fermion parity | `src/fermions.jl`: `dual` (l. 34), `dim` (l. 35), `BraidingStyle = Fermionic()` (l. 39), `⊗` (l. 43), `Nsymbol` (l. 45), `Fsymbol` (l. 48), `Rsymbol` (l. 51), `twist` (l. 54); FS phase from the generic `src/sectors.jl:frobenius_schur_phase_from_Fsymbol` (l. 461) |
| `dim`, bosonic twist | generic `src/sectors.jl:dim_from_Fsymbol` (UniqueFusion → 1, l. 429) and `twist_from_Rsymbol` (l. 646) |
| Scalars | `f64` (exact ±1, 0); TKS declares `Int`. TKS values checked by hand in `src/abelian.rs::tests::matches_measured_tensorkitsectors_values` |
| Ported implementation | [symtn](https://github.com/Ryo-wtnb11/symtn-rs) `symtn-sectors/src/abelian.rs` at `49d09dc4fc0825b29d547e63a70014221c0c5406`; two deliberate differences: an inadmissible F or R is `0.0` here (symtn returns `±1.0` unchecked), and U(1) `N` is the exact `0` where `a + b` leaves `i32` (symtn's checked N fails) |

## CU(1) = U(1) ⋊ C (`cu1` module)

| Field | Value |
| --- | --- |
| Reference | TensorKitSectors.jl `0.3.9`, git `67d342a130bd9e31caf3eccbf9ca46ec8b088556` (tree `9f536263310aae90df175337d586a1655104aae3`, recomputed from `~/.julia/packages/TensorKitSectors/VuA9Z`) |
| Labels, order, unit, dual | `src/irreps/cu1irrep.jl`: constructor validity (l. 25), `isless` (l. 60), `unit` (l. 69), `dual` (l. 70). `j` is stored as `twice_charge: u32` (the full range), `s` as `u8` |
| Fusion order | `CU1IrrepProdIterator` `iterate` (l. 85) |
| `dim`, scalars | `dim` (l. 107); `fusionscalartype` / `braidingscalartype` `Float64` (l. 110-111) |
| `Nsymbol`, `Fsymbol`, `Rsymbol`, FS phase | `Nsymbol` (l. 115), `Fsymbol` (l. 125), `Rsymbol` (l. 215), `frobenius_schur_phase` (l. 254); `BraidingStyle(::AbstractIrrep) = Bosonic()` (`irreps.jl` l. 17) |
| Twist | no override; generic `src/sectors.jl:twist_from_Rsymbol` (l. 647), which is `1` for every label |
| TKS values | `tests/cu1_values.rs`: every F and R over `0±`, `2j ≤ 5` equals the dense recoupling of the `fusiontensor` Clebsch–Gordan tensors written in the test, plus literals checked against TKS 0.3.9 (Julia 1.11.6). Deliberate deviation: TKS `Rsymbol` gives `-0.0` on inadmissible `R^{ab}_{0-}` with charged `a` (`-convert(Float64, N)`, `N = 0`), this crate `+0.0` (the crate-wide exact-zero contract, as symtn) |
| Ported implementation | [symtn](https://github.com/Ryo-wtnb11/symtn-rs) `symtn-sectors/src/cu1.rs` at `2a6c152315bb271dc5e56b5f1f034ca818ef999d`. Deliberate differences: labels span the full `u32` (symtn's codec stops at `u32::MAX - 1`, which symtn validates itself), so fusions summing to exactly `u32::MAX` succeed here; and `N` is exact everywhere (symtn's checked N fails above its bound). Only `CU1Irrep::fusion`, which must produce the labels, reports an overflow, above `u32::MAX` |

## Pinned tables (`tables` module)

Every number the table provider ships is traceable to the rows below. The crate
performs no derivation of the *source* data: it parses the committed tables and
derives only the API-level quantities that TensorKitSectors also derives
(dimension, Frobenius–Schur phase, twist). `TableCategory::fibonacci()` is the
single Fibonacci path.

### Authority — AnyonWiki database (exact symbols)

| Field | Value |
| --- | --- |
| Repository | [`anyonwiki/AnyonWikiDatabase`](https://github.com/anyonwiki/AnyonWikiDatabase) (data of [anyonwiki.org](https://anyonwiki.org/)) |
| Commit | `badbfe299dd533c12a84e817f5890059705e0a3c` (2026-09-30) |
| License | MIT, Copyright (c) 2026 AnyonWiki — copied to `data/anyonwiki-badbfe2/LICENSE` |
| Category file | `AlgebraicStructures/FusionCategories/MultFreeFusionCategories/Rank1-7.json`, SHA-256 `383cb2db58ae839f73c3d67ad79186e6ee44d69c32fd8fb202d1a4089ca7e859` (64.7 MB, 989 categories; not committed) |
| Fusion-ring file | `AlgebraicStructures/FusionRings/mult1.json`, SHA-256 `66117d72d204a5359ef43e584ed46cfe503240558aa8b9c381922b4861aa1153` (not committed) |
| Identity | (commit, `uuid`). The `anyonwiki_code` `[rank, multiplicity, non-self-dual, ring, associator f, braiding b (0 = none), pivotal p]` is accepted by `TableCategory::by_code` for this commit only: codes are not stable across database versions (CategoryData.jl's differ for most braided entries, Fibonacci's chirality index among them) |

A qqb_id `a0_a1_…_an__i` is the `i`-th root of `a0 + a1 x + … + an x^n`, roots
ordered real first (ascending), then conjugate pairs by real part, then
imaginary part (database README, "Conventions used").

### Selection

All 989 entries are listed in the generated status index
(`src/tables/catalog.rs`, `INDEX`); a query for any other uuid or code is
`TableError::UnknownCategory`.

| Status | Rule | Count |
| --- | --- | --- |
| shipped | `is_unitary`, `in_unitary_gauge`, and every signed quantum dimension `d_a = \|1 / F^{a ā a}_a[1,1]\|` (checked at 80 digits), rank ≤ 4 | 74 (58 braided, 16 fusion-only) |
| shipped with `anyonwiki-unitary`, otherwise `NotShipped` | as shipped, rank 5–7 (same exact check) | 303 (212 braided, 91 fusion-only) |
| `NonUnitary` | `is_unitary` false | 608 |
| `PivotalNotUnitary` | `is_unitary` true but dimensions ≠ `\|1/F\|`: the four `[7,1,2,11,2,b,1]` entries (negative dimensions; a suspected upstream defect). The generator aborts if any other entry fails | 4 |

The dimension rule is exactly the condition under which TensorKitSectors'
derivations from F and R (below) are the database's own dimension,
Frobenius–Schur phase and twist. It is not "pivotal index 1" (the unitary
pivotal of `[4,1,0,1,1,2,·]` is `p = 2`) and not "all `p_symbols` = 1":
AnyonWiki's convention is `d_a = p_a / F^{a ā a}_a[1,1]`, so the unitary
structure has `p_a = κ_a`, which is `−1` for the semion and SU(2)₃'s spin ½.

Key mapping onto the shipped record layout: AnyonWiki F
`a_b_c_d_e_α_β_f_γ_δ` → `a b c d α e β γ f δ` (Bridgeman's `a b c d α e β μ f ν`,
so `μ ↔ γ`, `ν ↔ δ`); R `a_b_c_α_β` → `a b c α β`. All multiplicity slots are
`1`. Pivotal coefficients are not shipped.

### Projection — correctly rounded binary64, offline

`tools/anyonwiki/gen_category_tables.py` (recorded run: Python 3.14.6, mpmath
1.3.0, 3 min 10 s) reads the database checkout (both SHA-256s verified),
classifies every entry, evaluates each qqb_id of a supported entry with
`mpmath.polyroots` at 80 significant digits, orders the roots by the database
convention, checks the residual, rounds the real and imaginary parts to
nearest binary64 (`float(mpf)`), confirms the rounding is unchanged at 160
digits, and aborts if any value is more than 4 ulp from AnyonWiki's
`numeric_*`. It writes

- `data/anyonwiki-badbfe2/N/<ring code>.txt`, `F/<code>.txt` and
  `R/<code>.txt` (braided entries only), the tables, as shortest round-trip
  decimals in the layouts `a b c Nabc`, `a b c d α e β μ f ν Re Im`,
  `a b c α μ Re Im`. N and F files are shared by content; an F file is named
  after the lowest code using it. Rank ≤ 4: 16 N, 35 F, 58 R files, 137,640
  bytes; with rank 5–7 (`anyonwiki-unitary`): 60 N, 179 F, 270 R files,
  4,195,511 bytes (F de-duplication halves 210,736 records to 105,543; no two
  R files are equal);
- `src/tables/catalog.rs`, the supported entries (`include_str!` of their
  files and their `TableProvenance`; rank 5–7 entries under
  `#[cfg(feature = "anyonwiki-unitary")]`) and the status index (rank 5–7
  rows `GATED`: shipped with the feature, `NotShipped` without);
- `tests/fixtures/anyonwiki-badbfe2-check.txt` and
  `tests/fixtures/anyonwiki-badbfe2-unitary-check.txt`, the generator's own
  per-symbol audit output (qqb_id, generated value, AnyonWiki `numeric_*`).
  They are not committed and no test reads them; the generator's gates (each
  symbol within 4 ulp of `numeric_*` under its own key) run when it does.

`tests/tables_anyonwiki.rs` checks on every shipped entry that the derived
dimensions are the fusion ring's Perron–Frobenius dimensions, and pins a few
representative symbols (rank 3: irrational real F, complex R, an admissible
exact zero; rank 5 with the feature: `1/√3`, purely imaginary F, eighth- and
twenty-fourth-root R) bitwise to the correctly rounded AnyonWiki symbol, each
written in the test with its minimal polynomial, plus their entries'
dimensions and twists. Unitarity, pentagon, hexagon and ribbon run over every
shipped entry in `tests/tables_reference.rs`.

### The `Complex64` projection epoch

Epoch identifier:
**`fusion-symbols/tables/anyonwiki-badbfe2/projection/2026-10-16`**. The
entry selection and file layout are recorded in each entry's provenance and
fingerprint, not in the epoch.

1. Each exact symbol is rounded correctly to binary64 offline (above).
2. At load, split each record on ASCII whitespace; the trailing two fields
   are the real and imaginary decimals, read with Rust's correctly rounded
   `str::parse::<f64>()`, which returns the generated bits exactly.
3. Pair them as `Complex64 { re, im }`. No re-normalisation, no gauge fixing.
4. Unlisted `N`/`F`/`R` records are exact zeros.

Changing any step is a new epoch identifier, and therefore a new authority
fingerprint. symtn's earlier epoch `tenet-category-data/projection/2026-07-27`
(step 1 replaced by "parse CategoryData.jl's 20-decimal export") maps to this
one: every Fibonacci F, R and N value is bit-identical.

### Identity, ordering, gauge, fingerprint

| Field | Value |
| --- | --- |
| Object ordering | Fusion-ring element order, one-based, `1` = unit. Frozen; not re-sorted |
| Gauge | Source gauge, loaded verbatim (unitary), named constructors included. No conversion to TensorKitSectors' gauge |
| Provenance | One generated `TableProvenance` per entry (uuid and fusion-ring uuid in `source_entry`, code in `category_key`) |
| Fingerprint | Canonical provenance bytes, then the N, F and R texts verbatim (a fusion-only entry has an empty R section). All 74 prehashes and file digests are pinned (`tables::tests::shipped_tables_and_fingerprints_are_pinned`). Fibonacci's file bytes equal 0.3.0's `FibonacciTable`; its fingerprint differs only through the uniform provenance strings |
| Braiding | `TableCategory::is_braided`; R and twist of a fusion-only entry are `TableError::NoBraiding` (`_checked`) or a panic |

### Named constructors and TensorKitSectors 0.3.9

`tests/tables_reference.rs` writes out TensorKitSectors 0.3.9's
`FibonacciAnyon` and `IsingAnyon` closed forms (`src/anyons.jl`) and checks a
stated object map and vertex gauge `u` (a witness, not a search):
`N` exact, `F_here = F_TKS · u^{ab}_e u^{ec}_d / (u^{bc}_f u^{af}_d)`,
`R_here = R_TKS · u^{ab}_c / u^{ba}_c`, and dim, FS phase and twist, within 32ε.

| Constructor | Code, uuid | Relation to TensorKitSectors |
| --- | --- | --- |
| `fibonacci()` | `[2,1,0,2,1,2,1]`, `37fa4e42-97c6-11f1-9d0f-9b7fe5c504f9` | `FibonacciAnyon`, identity map, `u ≡ 1` (same gauge, same braiding `R^{ττ}_𝟙 = e^{+4πi/5}`; `[2,1,0,2,1,1,1]` is the mirror and fails). The F and R entries are the correctly rounded exact values, pinned bitwise (`1/φ` = `0.6180339887498949` here; TKS's f64 `1/φ` is `0.6180339887498948`) |
| `ising()` | `[3,1,0,1,1,1,1]`, `37fa4e56-97c6-11f1-907a-eb6ab8e8bc11` | `IsingAnyon` under TKS `(I, σ, ψ)` → `(1, 3, 2)` with `u^{σσ}_ψ = −1` (R identical; six F entries change sign). `u ≡ 1` and the mirror braiding `[3,1,0,1,1,4,1]` fail |

### Citation

AnyonWikiDatabase asks users of its fusion-category data to cite
G. Vercleyen, *On Low-Rank Multiplicity-Free Fusion Categories*,
arXiv:2405.20075 (the entry's `references` field gives the published version,
doi:10.1007/s00220-026-05705-2), and users of fusion-ring data to cite
G. Vercleyen and J. K. Slingerland, *On low rank fusion rings*, J. Math. Phys.
64, 091703 (2023), doi:10.1063/5.0148848. The entry's software field names
Anyonica 0.9.11.5 (doi:10.5281/zenodo.10686860).

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

