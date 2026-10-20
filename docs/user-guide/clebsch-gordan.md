# 5. Clebsch–Gordan coefficients

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## What is this?

A Clebsch–Gordan coefficient is one entry of the intertwiner that maps the
product basis of `a ⊗ b` onto the basis of a coupled irrep `c`. As a tensor it
has four indices:

```text
C[m1, m2, m3, μ]      m1 ∈ basis(a), m2 ∈ basis(b), m3 ∈ basis(c),
                      μ ∈ 0 .. N^c_ab   (the outer-multiplicity axis)
```

The fourth axis exists because `c` can appear in `a ⊗ b` more than once
([Fusion](fusion.md#multiplicity-greater-than-one)); `μ` selects the copy. For a
multiplicity-free family such as SU(2), it has length 1 for an allowed coupling
and 0 for an absent channel.

## When do I need it?

When you need the actual change of basis — projecting a product state onto a
coupled sector, building symmetric tensors, computing matrix elements. If you
only need recoupling *between* coupling orders, you want
[F- and R-symbols](recoupling.md) instead: those already contract the CGC away
and are cheaper to consume.

## SU(2)

Exact, closed-form, no feature flag. Both irrep and projection labels are
doubled (`dj = 2j`, `dm = 2m`).

```rust
use fusion_symbols::lie::su2::clebsch_gordan;

// ⟨½ +½, ½ −½ | 0 0⟩ = 1/√2, in doubled labels.
let cg = clebsch_gordan(1, 1, 1, -1, 0, 0);
assert!((cg.to_f64() - 0.5f64.sqrt()).abs() < 1e-15);
```

The argument order is `(dj1, dm1, dj2, dm2, dj3, dm3)` — factor label and its
projection, interleaved. The return is a [`SignedSqrtRational`]: the exact value
`sign · sqrt(p/q)`. A forbidden coupling returns exact zero; use
`clebsch_gordan_checked` if you must distinguish that from an accidental zero.

Related exact objects on the same surface: `wigner_3j` (the 3j symbol,
`(dj1 dj2 dj3; dm1 dm2 dm3)`) and `wigner_6j` (the 6j / recoupling symbol,
`{dj1 dj2 dj3; dj4 dj5 dj6}`), from which the CG and F-symbol are composed.

## SU(N)

`GroupRepresentations::cgc(s1, s2, s3)` generates the whole block at once — all multiplicity
copies together, since they share one nullspace.

```rust
use fusion_symbols::{category::{FusionCategory, GroupRepresentations}, lie::{GroupId, LieGroup}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

let three = cat.irrep(&[1, 0]).unwrap();
let anti = cat.dual(&three);
let eight = cat.irrep(&[1, 1]).unwrap();

let c = cat.cgc(&three, &anti, &eight).unwrap();
assert_eq!(c.dims(), [3, 3, 8, 1]);   // [dim(s1), dim(s2), dim(s3), N^{s3}_{s1 s2}]
assert_eq!(c.dims()[3], 1);
assert!((c.at(0, 0, 0, 0) - 1.0).abs() < 1e-12);
```

### What the result means

- `LieCgc::dims()` gives `[dim(a), dim(b), dim(c), N(a,b,c)]`;
  `at(m1,m2,m3,mu)` reads a coefficient, including zero. Indices outside
  these axes panic. The private generator retains its sparse cached layout.
- Higher A-series indices follow `LieIrrep::gt_patterns()` in the frozen
  GT order. A1 magnetic indices ascend from the lowest weight.
- **Values are real** `f64`, in the crate's canonical gauge.

### Multiplicity greater than one

```rust
use fusion_symbols::{category::GroupRepresentations, lie::{GroupId, LieGroup}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

let eight = cat.irrep(&[1, 1]).unwrap();
let c = cat.cgc(&eight, &eight, &eight).unwrap();
assert_eq!(c.dims()[3], 2);       // the 8 occurs twice in 8 ⊗ 8
assert_eq!(c.dims(), [8, 8, 8, 2]);
assert!(c.at(0, 2, 0, 1) < 0.0);
```

The two copies `μ = 0, 1` are mutually orthonormal and are fixed by the gauge
specification, not by an arbitrary choice at run time — the same input always
yields the same two copies, in the same order.

## SO(N) / Spin(N) / Sp(2N)

B/C/D generation discovers and retains canonical generator matrices inside
`LieGroup`. Clones share this catalog. CGC use the same `LieCgc` accessors as
SU(N), with magnetic indices in the frozen bootstrap basis order.

```rust
use fusion_symbols::{category::{GroupRepresentations},lie::{LieGroup,GroupId}};

let cat = LieGroup::new(GroupId::so(5).unwrap()).unwrap();   // SO(5) / Spin(5)
let v = cat.irrep(&[1, 0]).unwrap();      // the 5
let adj = cat.irrep(&[0, 2]).unwrap();    // the 10

let c = cat.cgc(&v, &v, &adj).unwrap();
assert_eq!(c.dims()[3], 1);
assert_eq!((c.dims()[0]*c.dims()[1],c.dims()[2]), (25, 10));    // (dim(s1)·dim(s2), dim(s3))
```

### What the result means

`LieCgc::dims()` returns `[dim(a), dim(b), dim(c), N(a,b,c)]`;
`at(m1,m2,m3,mu)` reads one coefficient. The private generator keeps its dense
first-factor-fast layout; these accessors preserve that gauge without exposing
storage details. Use `LieGroup::with_catalog_budget` to bound retained generator
bytes, `catalog_stats()` for `(bytes, budget)`, and `reset_catalog()` to re-seed.
Catalog control returns `Result`; a poisoned catalog is a generation error.
Global coefficient caches remain valid across catalog resets.

## Caveats

- **Generation cost.** CGC generation is the expensive step; F- and R-symbols
  are contractions of already-generated CGC. Results are cached, so repeated
  queries are cheap, but a cold first query on a large irrep is not.
- **Empty channels.** Every Lie CGC has multiplicity axis zero for an absent
  channel; F/R blocks also have zero axes.
- **Values are floating point.** Labels, dimensions and multiplicities are
  exact; generated coefficient *values* are verification-gated `f64`. See
  [Resources § exact vs generated](resources.md#exact-vs-generated-values).
- **Index conventions are frozen, not incidental.** Basis order, sign, and
  multiplicity-copy order are specified in [`../gauge.md`](../gauge.md) (SU(N))
  and [`../gauge_soN.md`](../gauge_soN.md) (B/C/D). Most users need not read
  them; consult them if you are comparing against another implementation.

## Related API

- [`su2::clebsch_gordan`], `clebsch_gordan_checked`, `wigner_3j`, `wigner_6j`
- [`GroupRepresentations::cgc`], [`lie::LieCgc`], [`LieCgc::at`]

[`SignedSqrtRational`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/su2/struct.SignedSqrtRational.html
[`su2::clebsch_gordan`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/su2/fn.clebsch_gordan.html
[`GroupRepresentations::cgc`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.GroupRepresentations.html#tymethod.cgc
[`lie::LieCgc`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieCgc.html
[`LieCgc::at`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieCgc.html#method.at
