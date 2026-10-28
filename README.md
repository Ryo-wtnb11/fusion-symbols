# fusion-symbols

[![CI](https://github.com/Ryo-wtnb11/fusion-symbols/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryo-wtnb11/fusion-symbols/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/fusion-symbols.svg)](https://crates.io/crates/fusion-symbols)
[![docs.rs](https://img.shields.io/docsrs/fusion-symbols)](https://docs.rs/fusion-symbols)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Ryo-wtnb11/fusion-symbols/badges/badge.json)](https://github.com/Ryo-wtnb11/fusion-symbols/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/gh/Ryo-wtnb11/fusion-symbols/graph/badge.svg)](https://codecov.io/gh/Ryo-wtnb11/fusion-symbols)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Fusion-category and representation data for symmetric tensors in Rust,
inspired by [TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl):
labels, fusion multiplicities N, dimensions, duals, Frobenius–Schur data,
F/R/A/B symbols and Clebsch–Gordan coefficients (CGC). It is a standalone
mathematical provider; consumers own tensors and tensor operations.

Intentional differences from TensorKitSectors: categories are runtime values
with validated labels rather than types; higher-rank Lie coefficients are
generated on demand in a frozen, documented gauge; coefficient caches are
byte-bounded and application-owned; tables keep AnyonWiki's gauge. See
[differences from TensorKitSectors](docs/tensorkitsectors-differences.md).

## Supported families

| Family | Provider | Feature |
|---|---|---|
| Rank one: SU(2), SO(3), Spin(3), Sp(2), PSU(2); exact 3j/6j/CG | `lie::LieGroup`, `lie::su2` | base |
| D₂ forms: Spin(4), SO(4), PSO(4) | `lie::LieGroup` | base |
| SU(N), SO(N), Spin(N), Sp(2r) of higher rank, and their central quotients | `lie::LieGroup` | `lie-generated` |
| G₂, F₄, E₇ and E₈, with exact coefficients | `lie::LieGroup` | `lie-generated` |
| U(1), tori Tᵗ, U(N), semisimple × torus products and quotients | `lie::LieGroup` | base when every simple factor is rank one or D₂ (such as U(2)); otherwise `lie-generated` |
| U(1) with `Copy` labels; CU(1) = O(2) | `lie::U1`, `lie::CU1` | base |
| Z_N, fermion parity | `finite::ZN`, `fermion::FermionParity` | base |
| Unitary multiplicity-free AnyonWiki categories (Fibonacci, Ising, …) | `table::TableCategory` | rank ≤ 4 (74) base; rank 5–7 (303 more) with `anyonwiki-unitary` |
| Deligne products of any of the above | `product::Product` | base |

The exceptional group E₆ is classified by `GroupId`, but
`LieGroup::new` returns `LieError::Unsupported`. CU(1) is the only disconnected group. Finite
non-abelian groups Rep(G) are not provided.

## Install

```toml
[dependencies]
fusion-symbols = "0.9"
```

| Feature | Default | Adds |
|---|---|---|
| `lie-generated` | yes | Runtime SU(N)/SO(N)/Spin(N)/Sp(2r) coefficients and the dense backend (`t4a-faer`, sequential; the caller owns parallelism) |
| `anyonwiki-unitary` | no | 303 rank 5–7 table categories, data only |

Use `default-features = false` for the base build without a dense backend.

## Example

Build a group by name, make irreps from Dynkin labels and torus charges, and
query N and CGC (SU(3) needs the default `lie-generated` feature):

```rust
use fusion_symbols::category::{CgcBlock, FusionCategory, GroupRepresentations};
use fusion_symbols::lie::{GroupId, LieGroup};

let su3 = LieGroup::new(GroupId::su(3)?)?;
let eight = su3.irrep(&[1, 1], &[])?; // adjoint, Dynkin labels (1,1)
assert_eq!(su3.nsymbol(&eight, &eight, &eight), 2);
let cgc = su3.cgc(&eight, &eight, &eight)?;
assert_eq!(cgc.dims(), [8, 8, 8, 2]); // [dim a, dim b, dim c, N]

let u2 = LieGroup::new(GroupId::u(2)?)?;
let fundamental = u2.irrep(&[1], &[1])?; // SU(2) label 1, U(1) charge 1
assert_eq!(u2.dim(&fundamental), 2.0);
assert!(u2.irrep(&[1], &[0]).is_err()); // charge must match the SU(2) label mod 2
```

Exact SU(2) recoupling, available without default features. Spins are
doubled (`dj = 2j`), so `2` means spin 1:

```rust
use fusion_symbols::lie::su2::wigner_6j;

let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
```

## Documentation

- [User guide](docs/user-guide/README.md): choosing a category, building
  groups, labels, symbols, CGC, products, tables, errors and caches.
- [API reference](https://docs.rs/fusion-symbols).
- [Design and contracts](docs/overview.md),
  [caches and reproducibility](docs/user-guide/resources.md).
- Frozen gauge specifications: [SU(2)/SU(N)](docs/gauge.md),
  [SO(N)/Spin(N)/Sp(2r)](docs/gauge_soN.md), [exceptional](docs/gauge_exceptional.md).
- [Algorithm references](docs/references.md),
  [closed-form and table conventions](docs/references_categories.md).
- [Changelog](CHANGELOG.md), [contributor notes](docs/developer/README.md).

## Acknowledgments

fusion-symbols follows the conventions of
[TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl)
and [TensorKit.jl](https://github.com/QuantumKitHub/TensorKit.jl). The `table`
data come from the [AnyonWiki](https://anyonwiki.org/) database
([`anyonwiki/AnyonWikiDatabase`](https://github.com/anyonwiki/AnyonWikiDatabase),
MIT). The Lie-group algorithms are ported from WignerSymbols.jl,
SUNRepresentations.jl and QSpace; see [`docs/references.md`](docs/references.md).

## Citation

**If you use fusion-symbols in research, please cite [TensorKit](https://github.com/QuantumKitHub/TensorKit.jl) and [TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl).**

If you use fusion-symbols in academic work, please cite it
(machine-readable metadata in [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{watanabe_fusion_symbols_2026,
  author  = {Watanabe, Ryo},
  title   = {{fusion-symbols}: fusion and recoupling symbols for symmetric tensors},
  year    = {2026},
  url     = {https://github.com/Ryo-wtnb11/fusion-symbols},
  license = {MIT OR Apache-2.0}
}
```

If you use the `table` categories, please also cite the sources the
AnyonWiki database asks for: G. Vercleyen, *On Low-Rank Multiplicity-Free
Fusion Categories*, [arXiv:2405.20075](https://arxiv.org/abs/2405.20075)
(published as [doi:10.1007/s00220-026-05705-2](https://doi.org/10.1007/s00220-026-05705-2)),
for fusion-category data, and G. Vercleyen and J. K. Slingerland, *On low rank
fusion rings*, J. Math. Phys. 64, 091703 (2023),
[doi:10.1063/5.0148848](https://doi.org/10.1063/5.0148848), for fusion-ring
data.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
