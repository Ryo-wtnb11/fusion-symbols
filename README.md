# fusion-symbols

[![CI](https://github.com/Ryo-wtnb11/fusion-symbols/actions/workflows/ci.yml/badge.svg)](https://github.com/Ryo-wtnb11/fusion-symbols/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/fusion-symbols.svg)](https://crates.io/crates/fusion-symbols)
[![docs.rs](https://img.shields.io/docsrs/fusion-symbols)](https://docs.rs/fusion-symbols)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Ryo-wtnb11/fusion-symbols/badges/badge.json)](https://github.com/Ryo-wtnb11/fusion-symbols/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Ryo-wtnb11/fusion-symbols/graph/badge.svg)](https://codecov.io/gh/Ryo-wtnb11/fusion-symbols)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Symmetry data for symmetric-tensor libraries in Rust, inspired by
TensorKitSectors.jl.

For each supported symmetry, fusion-symbols provides only the sector data:
which sectors exist, how they fuse, and their recoupling data (dimensions,
duals, Frobenius–Schur data, F- and R-symbols, twists, Clebsch–Gordan
coefficients), in TensorKitSectors.jl conventions. Tensors and tensor
operations are not part of fusion-symbols; symtn is the first consumer.

Unlike TensorKitSectors, it is a standalone provider with no tensor engine
behind it.
Differences in detail are listed in docs/.

## Status

The modules follow the mathematical classification:

| Module | Supported data | Feature |
|---|---|---|
| `category` | Shared `FusionCategory` contract, styles, scalar blocks and identity residuals | default |
| `lie` | SU(2), SO(3), Spin(3), Sp(2), PSU(2) through `LieGroup`; exact evaluators in `lie::su2` | default |
| `lie` | SU(N), SU(N)/Z_k, PSU(N), SO(N), Spin(N), Sp(2N) and their supported global forms | `cgc-gen` |
| `abelian` | U(1), Z_N (including Z2) via `U1`, `ZN` | default |
| `group` | Other compact groups: CU(1) = U(1) ⋊ C = O(2), via `CU1` | default |
| `fermion` | Fermion parity via `FermionParity`, with `FermionParityIrrep` labels | default |
| `table` | 74 unitary multiplicity-free AnyonWiki categories of rank ≤ 4, including Fibonacci and Ising | default |
| `table` | 303 further AnyonWiki categories of rank 5–7 | `anyonwiki-unitary` |
| `product` | Deligne products (planned; no public module yet) | — |
| `cache` | Coefficient-cache budgets, statistics, reset and trimming | default; generated tiers with `cgc-gen` |

The exact SU(2) evaluators live under `lie::su2`; category operations use
`lie::LieGroup` and `category::FusionCategory`. In the `cgc-gen` surface, the
fingerprint bytes of the generated families (SU(N), SO/Spin/Sp),
`cache::GENERATED_CACHE_MAX_BYTES` and `cache::GeneratedCacheStats` are marked
unstable in their rustdoc.

This crate succeeds the [`racah`](https://crates.io/crates/racah) crate on
crates.io (0.2.4), preserving its coefficient gauges under `fusion_symbols::`
with a common category interface.
There are no Python bindings in this repository; the last release of the
earlier bindings is `racah-py` 0.1.2 on PyPI (import name `racah`), which is no
longer developed here (#130).

```toml
[dependencies]
fusion-symbols = "0.4"
# generated SU(N)/SO(N)/Sp(2N) families:
# fusion-symbols = { version = "0.4", features = ["cgc-gen"] }
```

## Example

Exact SU(2) 6j (no features). Spins are doubled (`dj = 2j`), so `2` means
spin 1; `{1 1 1; 1 1 1} = 1/6`:

```rust
use fusion_symbols::lie::su2::wigner_6j;

let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
```

## Documentation

- [Overview](docs/overview.md): supported groups, feature flags, more
  examples, exactness and verification status.
- [Differences from TensorKitSectors.jl](docs/tensorkitsectors-differences.md).
- [User Guide](docs/user-guide/README.md), task by task.
- API reference: [docs.rs](https://docs.rs/fusion-symbols).
- [Theory](docs/theory.pdf) ([source](docs/theory.tex)).
- Gauge specification (normative, frozen): [`docs/gauge.md`](docs/gauge.md),
  [`docs/gauge_soN.md`](docs/gauge_soN.md).
- Provenance: [`docs/references.md`](docs/references.md),
  [`docs/references_categories.md`](docs/references_categories.md).
- Developer docs: [`docs/developer/`](docs/developer/README.md),
  [`AGENTS.md`](AGENTS.md), [`tools/README.md`](tools/README.md).

Documentation index: [`docs/README.md`](docs/README.md).

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
