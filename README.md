# fusion-symbols

[![CI](https://github.com/Ryo-wtnb11/fusion-symbols/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryo-wtnb11/fusion-symbols/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/fusion-symbols.svg)](https://crates.io/crates/fusion-symbols)
[![docs.rs](https://img.shields.io/docsrs/fusion-symbols)](https://docs.rs/fusion-symbols)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Ryo-wtnb11/fusion-symbols/badges/badge.json)](https://github.com/Ryo-wtnb11/fusion-symbols/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/gh/Ryo-wtnb11/fusion-symbols/graph/badge.svg)](https://codecov.io/gh/Ryo-wtnb11/fusion-symbols)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Representation and fusion-category data in Rust, inspired by
[TensorKitSectors.jl](https://github.com/QuantumKitHub/TensorKitSectors.jl):
labels, fusion multiplicities, dimensions, duals, Frobenius–Schur data,
Clebsch–Gordan coefficients (CGC), and F/R symbols.

This is a standalone mathematical provider. Consumers own tensors and tensor
operations. Intentional differences include runtime generation for higher-rank
Lie groups, explicit category values, validated Rust labels, byte-bounded coefficient caches with borrowed warm lookups,
and a frozen coefficient gauge. Tables retain AnyonWiki's gauge. See
[differences from TensorKitSectors](docs/tensorkitsectors-differences.md).

## Supported families

| Module | Provides |
|---|---|
| `category` | Shared `FusionCategory` contract, symbol blocks, identity residuals |
| `lie` | Exact rank-one symbols; typed `U1` alongside `LieGroup` U(1), CU(1) = O(2); generated SU(N), SO(N), Spin(N), Sp(2r); U(N), tori, and semisimple×torus products and central quotients, including D₂ forms |
| `finite`, `fermion` | Cyclic groups Z_N; fermion parity |
| `table` | 74 unitary multiplicity-free AnyonWiki categories of rank ≤ 4, including Fibonacci and Ising |
| `product` | Ordered Deligne products: tuples of 2..=12 or homogeneous vectors of any length, preserving nesting |
| `cache` | Coefficient-cache budgets, statistics, reset and trimming |

```toml
[dependencies]
fusion-symbols = "0.9"
# Omit generated Lie coefficients and the dense backend:
# fusion-symbols = { version = "0.9", default-features = false }
```

`lie-generated` is enabled by default and uses sequential t4a-faer kernels;
the caller owns parallelism. `anyonwiki-unitary` adds 303 table categories of
rank 5–7 without a new dependency. See the [design](docs/overview.md) for
labels, capabilities and error contracts.

`lie::GroupId` separately classifies ordered connected root data with exact
character lattices, including products, tori, U(N), and exceptional quotients.
`LieGroup` executes tori, U(N), the trivial group, and semisimple×torus data
whenever every simple factor has an engine; D₂ expands to two exact A₁ factors.
U(1) and U(2) work in the base build; U(N) for N ≥ 3 requires `lie-generated`.
Exceptional blocks remain unsupported. Composite covers enumerate in Product
order when every factor enumerates; composite quotients return `NotEnumerable`.
See [Lie labels and lattices](docs/overview.md#lie-labels-and-global-forms).

## Example

Exact SU(2) 6j, available without default features. Spins are doubled
(`dj = 2j`), so `2` means spin 1:

```rust
use fusion_symbols::lie::su2::wigner_6j;

let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
```

## Documentation

- [Design and contracts](docs/overview.md), [using the API](docs/user-guide/README.md),
  [resources and reproducibility](docs/user-guide/resources.md).
- [API reference](https://docs.rs/fusion-symbols).
- Frozen gauge specifications: [SU(2)/SU(N)](docs/gauge.md),
  [SO(N)/Spin(N)/Sp(2r)](docs/gauge_soN.md).
- [Algorithm references](docs/references.md),
  [closed-form and table conventions](docs/references_categories.md).
- [Contributor checks](docs/developer/README.md), [policy](AGENTS.md),
  [test-oracle rules](tools/README.md).

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

If you use the `table` tables, please also cite the sources the
AnyonWiki database asks for: G. Vercleyen, *On Low-Rank Multiplicity-Free
Fusion Categories*, [arXiv:2405.20075](https://arxiv.org/abs/2405.20075)
(published as [doi:10.1007/s00220-026-05705-2](https://doi.org/10.1007/s00220-026-05705-2)),
for fusion-category data, and G. Vercleyen and J. K. Slingerland, *On low rank
fusion rings*, J. Math. Phys. 64, 091703 (2023),
[doi:10.1063/5.0148848](https://doi.org/10.1063/5.0148848), for fusion-ring
data.

## License

MIT OR Apache-2.0
