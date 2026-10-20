# fusion-symbols documentation

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

Six layers, six audiences. Pick the one that matches what you are doing.

| Layer | Read it when | Where |
|---|---|---|
| **README** | You want to know what `fusion-symbols` is and run one calculation. | [`../README.md`](../README.md) |
| **User Guide** | You know what you want to compute and need the task-oriented path. | [`user-guide/`](user-guide/README.md) |
| **API Reference** | You know the item and want its exact semantics, shapes and errors. | [docs.rs/fusion-symbols](https://docs.rs/fusion-symbols) |
| **Theory** | You want the mathematics behind the objects the API returns. | [`theory.tex`](theory.tex) → [`theory.pdf`](theory.pdf) ([pointer](theory.md)) |
| **Specifications** | You need the normative basis/sign/ordering conventions of a returned value. | [`gauge.md`](gauge.md) (SU(2), SU(N)), [`gauge_soN.md`](gauge_soN.md) (SO(N)/Sp(2N)) |
| **Developer docs** | You are changing `fusion-symbols` itself. | [`developer/`](developer/README.md) |

Provenance — which reference implementation each algorithm is ported from,
symbol by symbol, plus the bibliography: [`references.md`](references.md).

## Module map

`category` defines the shared contract; families live in `lie`, `abelian`,
`group` (other compact groups: CU(1)), `fermion` and `table`. `product` is
planned; `cache` provides resource controls. See the
[family table](overview.md#supported-groups) for features and label types.

## Which document answers which question

- *"What can this library compute?"* → [README](../README.md), then
  [overview](overview.md) for the supported groups, feature flags and
  verification status.
- *"How does this differ from TensorKitSectors.jl?"* →
  [`tensorkitsectors-differences.md`](tensorkitsectors-differences.md).
- *"How do I build an SU(3) irrep and decompose a product?"* →
  [User Guide: representations](user-guide/representations.md),
  [fusion](user-guide/fusion.md).
- *"What is the shape of the array `f_symbol` returned, and what is axis 2?"* →
  [docs.rs](https://docs.rs/fusion-symbols) for the item, or
  [User Guide: recoupling](user-guide/recoupling.md) for the picture.
- *"Why is this coefficient this sign and not the other one?"* →
  [`gauge.md`](gauge.md) / [`gauge_soN.md`](gauge_soN.md). These are frozen and
  normative; they are specifications, not tutorials.
- *"Why Gelfand–Tsetlin for SU(N) but a generator bootstrap for SO(N)?"* →
  [`theory.pdf`](theory.pdf) §8.
- *"How much memory will the caches hold, and how do I bound it?"* →
  [User Guide: resources](user-guide/resources.md); the measurement evidence is
  in [`developer/`](developer/README.md).
