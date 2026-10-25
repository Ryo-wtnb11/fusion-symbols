# Using the API

Install and feature selection: [README](../../README.md).
Labels, validation and capabilities: [design](../overview.md).
Per-item reference: [docs.rs](https://docs.rs/fusion-symbols).

## Fusion and CGC

Import `FusionCategory` for category operations and `GroupRepresentations`
for Clebsch–Gordan coefficients (CGC), and `CgcBlock` to read them.
Fusion yields each outcome once;
`nsymbol(a,b,c)` is the number of copies of `c` in `a ⊗ b`.
This SU(3) example requires the default `lie-generated` feature:

```rust
use fusion_symbols::{category::{CgcBlock, FusionCategory, GroupRepresentations}, lie::{GroupId, LieGroup}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
let eight = cat.irrep(&[1, 1], &[]).unwrap();
assert_eq!(cat.nsymbol(&eight, &eight, &eight), 2);
let c = cat.cgc(&eight, &eight, &eight).unwrap();
assert_eq!(c.dims(), [8, 8, 8, 2]);
assert!(c.at([0, 2, 0, 1]) < 0.0);
```

`CgcBlock::dims()` is `[dim(a), dim(b), dim(c), N(a,b,c)]`;
`at([m1,m2,m3,mu])` reads that coefficient and panics outside the axes.
An absent channel has multiplicity axis zero. A1 indices ascend from the
lowest magnetic weight; higher A-series indices use `LieIrrep::gt_patterns()`.
B/C/D indices use the canonical sweep frame. The trailing copy order is gauge,
fixed in [SU(N) §8](../gauge.md#8-outer-multiplicity-axis) and
[B/C/D §9](../gauge_soN.md#9-outer-multiplicity-assignment).

Exact SU(2) evaluators under `lie::su2` return `SignedSqrtRational`:
`sign * sqrt(p/q)`. Both spin and projection labels are doubled; the CG
argument order is `(dj1, dm1, dj2, dm2, dj3, dm3)`. Convert with `to_f64()`.

## Symbol layout

`f_symbol(a,b,c,d,e,f)` is `F^{abc}_d[e,f]`, mapping the coupling
`(a ⊗ b) ⊗ c → d` through `e` to `a ⊗ (b ⊗ c) → d` through `f`.
`r_symbol(a,b,c)` is `R^{ab}_c`, exchanging the factors at a vertex.
Their logical axes follow TensorKitSectors:

| Symbol | Axes and lengths |
|---|---|
| F `[μ,ν,κ,λ]` | `[N(a,b,e), N(e,c,d), N(b,c,f), N(a,f,d)]` |
| R `[μ,ν]` | `[N(a,b,c), N(b,a,c)]` |
| A `[μ,ν]` | `[N(a,b,c), N(dual(a),c,b)]` |
| B `[μ,ν]` | `[N(a,b,c), N(c,dual(b),a)]` |

`category::SymbolBlock` exposes row-major storage through `dims`, `get` and
`as_slice`. Logical shape comes from N: a bare scalar's storage axes remain
all one even when its value is zero. Lie `FBlock`/`RBlock` provide `at` and
`data`; zero axes are empty, all-one blocks store one scalar inline, and
larger block clones share storage. A/B always return `category::RBlock`, even
for scalar families, and need no braiding. The A formula uses the
Frobenius–Schur *phase*, which differs from the indicator on non-self-dual objects;
see `src/category.rs::FusionCategory::{a_symbol,b_symbol}`.

For SU(3) `8 ⊗ 8 → 8`, all four F axes can have length 2; they cannot be
interchanged. The dense F-move check is retained in
`tests/doc_examples.rs::recoupling_5`; it need not be copied into
consumer code. F/R conventions and contraction wiring are traced in
[references](../references.md#family-generic-fr-core-and-gates).

## Products

```rust
use fusion_symbols::{lie::{U1, U1Irrep}, category::FusionCategory,
    fermion::{FermionParity, FermionParityIrrep}, product};
let cat = product(U1, FermionParity);
let odd = (U1Irrep::new(1).unwrap(), FermionParityIrrep::ODD);
assert_eq!(cat.twist(&odd), -1.0);
let three = product![U1, FermionParity, U1,];
assert_eq!(three.dim(&three.unit()), 1.0);
```

Use `Product::new((a,b,...))` for 2..=12 factors, `product(a,b)` for two,
and `product![...]` for 3..=12. `factors` borrows the tuple; `into_factors`
returns it. `product(product(a,b),c)` keeps labels `((x,y),z)`, with different
identity and enumeration from the flat product. Fusion is first-factor-fast;
within each symbol axis, `i = i_left * n_right + i_right` is last-factor-fast.
When every factor implements `GroupRepresentations`, so does the product. Its
CGC is a tuple of factor blocks, read lazily as a Kronecker product through
`CgcBlock`, with the last factor fastest on all four axes. `U1` and `ZN`
use scalar CGCs: `1.0` for a present channel, `0.0` for an absent one, with
shapes `[1,1,1,1]` and `[1,1,1,0]`. Product CGC queries report axis overflow
as `ProductError::Overflow`; a hand-built tuple's `dims()` may panic on overflow.

## Identity checks

`category::{f_unitarity_residual,pentagon_residual,hexagon_residual}` return
maximum absolute residuals, including multiplicity sums. Callers select the
tolerance. These are consistency checks; signed independent coefficient
oracles are needed to test a particular gauge. See
[verification policy](../../tools/README.md#test-oracles-binding) and
[resources](resources.md) for numerical and cache contracts.
