# Provenance: closed-form `lie`, `finite`, `fermion` and `table` categories

## Abelian groups and fermion parity (`lie`, `finite` and `fermion` modules)

| Field | Value |
| --- | --- |
| Reference | TensorKitSectors.jl `0.3.9`, git `67d342a130bd9e31caf3eccbf9ca46ec8b088556` |
| Z2, Z_N, U(1): `Nsymbol`, `Fsymbol`, `Rsymbol`, FS phase | `src/irreps/irreps.jl`: `Nsymbol` (l. 57), `Fsymbol` (l. 58), `frobenius_schur_phase(::AbelianIrrep)` (l. 62), `Rsymbol` (l. 65); `BraidingStyle(::AbstractIrrep) = Bosonic()` (l. 17) |
| Z_N labels, dual, fusion | `src/irreps/znirrep.jl`: `ZNIrrep{N}(n) = mod(n, N)` (l. 24), `dual` (l. 82), `⊗` (l. 83). The modulus is a runtime `u32` carried by both `ZN` and `ZNIrrep`, not a type parameter. `new` reduces charges; `from_canonical` rejects noncanonical charges without reduction |
| U(1) labels, dual, fusion | `src/irreps/u1irrep.jl`: `dual` (l. 30), `⊗` (l. 31). TKS stores a `HalfInt`; here integer `i32` charges without `i32::MIN`; half-integer charges are not representable |
| U(1) and Z_N CGC | `src/irreps/irreps.jl`: `fusiontensor(::AbelianIrrep, …)` (l. 67) returns `fill(Nabc, (1, 1, 1, Nabc))`; `U1` and `ZN` expose scalar `f64` blocks, and `LieGroup` exposes `LieCgc` blocks through `CgcBlock`; an absent channel has shape `1x1x1x0` |
| Fermion parity | `src/fermions.jl`: `dual` (l. 34), `dim` (l. 35), `BraidingStyle = Fermionic()` (l. 39), `⊗` (l. 43), `Nsymbol` (l. 45), `Fsymbol` (l. 48), `Rsymbol` (l. 51), `twist` (l. 54); FS phase from the generic `src/sectors.jl:frobenius_schur_phase_from_Fsymbol` (l. 461) |
| `dim`, bosonic twist | generic `src/sectors.jl:dim_from_Fsymbol` (UniqueFusion → 1, l. 429) and `twist_from_Rsymbol` (l. 646) |
| Scalars | `f64` (exact ±1, 0); TKS declares `Int`. TKS values checked by hand in `src/finite.rs::tests::matches_measured_tensorkitsectors_values` (Z2, Z_N) and `tests/lie_torus.rs::u1_values_bounds_and_fingerprint` (U(1)) |
| Overflow | U(1) symbols test charge conservation exactly in `i64`; only fusion, which must produce a label, reports an unrepresentable sum (`U1Error` or `LieError::LabelOverflow`) |

U(1) uses `lie::U1` or `lie::LieGroup` and the shared `category::FusionCategory` surface,
alongside `finite::ZN` and `fermion::FermionParity`. `U1`, `LieGroup`, and `ZN` also
implement `category::GroupRepresentations`. The parity label is
`fermion::FermionParityIrrep`. Z2 is `ZN::new(2)`. Foreign moduli panic before
computation. Inadmissible symbols are zero; `Error` is `Infallible` for ZN
and parity, while U(1) fusion reports label overflow. FS indicator is the FS
phase for self-dual objects and zero otherwise
(`src/sectors.jl::frobenius_schur_indicator`). U(1) values are checked in `tests/lie_torus.rs::u1_values_bounds_and_fingerprint` and `src/finite.rs::tests::matches_measured_tensorkitsectors_values`;
identities use `tests/category_conformance.rs::conformance`.

## CU(1) = U(1) ⋊ C (`lie` module)

| Field | Value |
| --- | --- |
| Reference | TensorKitSectors.jl `0.3.9`, git `67d342a130bd9e31caf3eccbf9ca46ec8b088556` |
| Labels, order, unit, dual | `src/irreps/cu1irrep.jl`: constructor validity (l. 25), `isless` (l. 60), `unit` (l. 69), `dual` (l. 70). `j` is stored as `twice_charge: u32` (the full range), `s` as `u8` |
| Fusion order | `CU1IrrepProdIterator` `iterate` (l. 85) |
| `dim`, scalars | `dim` (l. 107); `fusionscalartype` / `braidingscalartype` `Float64` (l. 110-111) |
| `Nsymbol`, `Fsymbol`, `Rsymbol`, FS phase | `Nsymbol` (l. 115), `Fsymbol` (l. 125), `Rsymbol` (l. 215), `frobenius_schur_phase` (l. 254); `BraidingStyle(::AbstractIrrep) = Bosonic()` (`irreps.jl` l. 17) |
| Twist | no override; generic `src/sectors.jl:twist_from_Rsymbol` (l. 647), which is `1` for every label |
| TKS values | `tests/cu1_values.rs`: every F and R over `0±`, `2j ≤ 5` equals the dense recoupling of the `fusiontensor` Clebsch–Gordan tensors written in the test, plus literals checked against TKS 0.3.9. Deliberate deviation: TKS `Rsymbol` gives `-0.0` on inadmissible `R^{ab}_{0-}` with charged `a` (`-convert(Float64, N)`, `N = 0`), this crate `+0.0` (the crate-wide exact-zero contract) |
| Label range | Labels span the full `u32`, so fusions summing to exactly `u32::MAX` succeed, and `N` is exact everywhere. Only `CU1::fusion`, which must produce the labels, reports an overflow, above `u32::MAX`. A consumer with a narrower label encoding validates its own bound |

The `lie::CU1` category implements `FusionCategory` with real scalar symbols,
simple fusion and bosonic braiding. Labels retain their validated constructors;
only fusion may fail on label overflow. The constructor still reports
`CU1Error::InvalidLabel`.

## Pinned tables (`table` module)

Tables are loaded without regauging. Source selection and binary64 projection
are part of the authority fingerprint (`src/table/provenance.rs::TableProvenance`).

### Authority — AnyonWiki database (exact symbols)

| Field | Value |
| --- | --- |
| Repository | [`anyonwiki/AnyonWikiDatabase`](https://github.com/anyonwiki/AnyonWikiDatabase) (data of [anyonwiki.org](https://anyonwiki.org/)) |
| Commit | `badbfe299dd533c12a84e817f5890059705e0a3c` |
| License | MIT, Copyright (c) 2026 AnyonWiki — copied to `data/anyonwiki-badbfe2/LICENSE` |
| Category file | `AlgebraicStructures/FusionCategories/MultFreeFusionCategories/Rank1-7.json`, SHA-256 `383cb2db58ae839f73c3d67ad79186e6ee44d69c32fd8fb202d1a4089ca7e859` (64.7 MB, 989 categories; not committed) |
| Fusion-ring file | `AlgebraicStructures/FusionRings/mult1.json`, SHA-256 `66117d72d204a5359ef43e584ed46cfe503240558aa8b9c381922b4861aa1153` (not committed) |
| Identity | (commit, `uuid`). The `anyonwiki_code` `[rank, multiplicity, non-self-dual, ring, associator f, braiding b (0 = none), pivotal p]` is accepted by `TableCategory::by_code` for this commit only: codes are not stable across database versions (CategoryData.jl's differ for most braided entries, Fibonacci's chirality index among them) |

A qqb_id `a0_a1_…_an__i` is the `i`-th root of `a0 + a1 x + … + an x^n`, roots
ordered real first (ascending), then conjugate pairs by real part, then
imaginary part (database README, "Conventions used").

### Selection

All 989 entries are listed in the generated status index
(`src/table/catalog.rs`, `INDEX`); a query for any other uuid or code is
`TableError::UnknownCategory`.

| Status | Rule | Count |
| --- | --- | --- |
| shipped | `is_unitary`, `in_unitary_gauge`, and every signed quantum dimension `d_a = \|1 / F^{a ā a}_a[1,1]\|` (checked at 80 digits), rank ≤ 4 | 74 (58 braided, 16 fusion-only) |
| shipped with `anyonwiki-unitary`, otherwise `NotShipped` | as shipped, rank 5–7 (same exact check) | 303 (212 braided, 91 fusion-only) |
| `NonUnitary` | `is_unitary` false | 608 |
| `PivotalNotUnitary` | `is_unitary` true but dimensions ≠ `\|1/F\|`: the four `[7,1,2,11,2,b,1]` entries (negative dimensions). The generator aborts if any other entry fails | 4 |

Selection ensures the dimension, FS phase and twist derived from F/R agree
with the source pivotal structure. It is not a selection on pivotal index alone.

Key mapping onto the shipped record layout: AnyonWiki F
`a_b_c_d_e_α_β_f_γ_δ` → `a b c d α e β γ f δ` (Bridgeman's `a b c d α e β μ f ν`,
so `μ ↔ γ`, `ν ↔ δ`); R `a_b_c_α_β` → `a b c α β`. All multiplicity slots are
`1`. Pivotal coefficients are not shipped.

### Projection — correctly rounded binary64, offline

`tools/anyonwiki/gen_category_tables.py` verifies both source SHA-256s,
evaluates qqb_id roots at 80 significant digits, checks residuals and stable
binary64 rounding at 160 digits, and rejects discrepancies above 4 ulp from
the source numeric data. It writes shortest round-trip decimals to
`data/anyonwiki-badbfe2/{N,F,R}/` and embeds their provenance and status in
`src/table/catalog.rs`. N and F texts are deduplicated by content.

`tests/tables_anyonwiki.rs` checks derived dimensions against independent
Perron–Frobenius dimensions and pins representative exact algebraic symbols.
`tests/tables_reference.rs` checks structural identities over every shipped entry.

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
fingerprint.

### Identity, ordering, gauge, fingerprint

| Field | Value |
| --- | --- |
| Object ordering | Validated by `TableCategory::object(index)`, with a private catalog tag; `TableObject::index()` returns the source index. Fusion-ring element order, one-based, `1` = unit. Frozen; not re-sorted |
| Gauge | Source gauge, loaded verbatim (unitary), named constructors included. No conversion to TensorKitSectors' gauge |
| Provenance | One generated `TableProvenance` per entry (uuid and fusion-ring uuid in `source_entry`, code in `category_key`) |
| Fingerprint | Canonical provenance bytes, then the N, F and R texts verbatim (a fusion-only entry has an empty R section). Prehashes and file digests are pinned in `src/table/tests.rs::shipped_tables_and_fingerprints_are_pinned` |
| Braiding | `FusionCategory::braiding_style` (`BraidingStyle::None` on fusion-only entries); `FusionCategory::r_symbol` returns `TableError::NoBraiding` on fusion-only entries; `twist` panics. Foreign objects panic before either capability check |

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
64, 091703 (2023), doi:10.1063/5.0148848.

### Derived quantities

Derived in `src/table.rs::TableCategory::from_entry` using TensorKitSectors 0.3.9 formulas.
A/B follow the `src/category.rs::FusionCategory::{a_symbol,b_symbol}` defaults
(`sectors.jl:501-512`, `543-552`):

| Quantity | Formula | TKS location |
| --- | --- | --- |
| `dim(a)` | `abs(1 / F(a, ā, a, a, 1, 1))` | `dim_from_Fsymbol`, `sectors.jl:431-439` |
| `sqrtdim` / `invsqrtdim` | `sqrt(dim)` / `inv(sqrt(dim))` | `sectors.jl:440-441` |
| `κ(a)` | `sign(F(a, ā, a, a, 1, 1))` | `frobenius_schur_phase_from_Fsymbol`, `sectors.jl:463-469` |
| `θ(a)` | `Σ_{b ∈ a⊗a} dim(b)/dim(a) · R(a, a, b)`, summed in ascending `b` | `twist_from_Rsymbol`, `sectors.jl:646-647` |

