# 1. Getting started

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## Install

```toml
[dependencies]
fusion-symbols = "0.4"
```

That is the **base** build: exact SU(2) only, three small numeric dependencies,
no linear-algebra stack. To compute for SU(N), SO(N), Spin(N) or Sp(2N), turn on
the `cgc-gen` feature:

```toml
[dependencies]
fusion-symbols = { version = "0.4", features = ["cgc-gen"] }
```

There is no fixed MSRV; `fusion-symbols` builds and is tested on current stable Rust.

### Which feature flag do I need?

| Feature | Gives you | Cost |
|---|---|---|
| *(default)* | SU(2): 3j, 6j, Clebsch–Gordan, F, R, Frobenius–Schur — closed-form, big-rational exact | `num-bigint`, `num-rational`, `num-traits` |
| `cgc-gen` | SU(N) (`fusion_symbols::lie`), SO(N)/Spin(N)/Sp(2N) (`fusion_symbols::bcd`), and the `fusion_symbols::lie` global forms applied to them | pulls the sequential t4a-faer dense-linear-algebra backend |

The split is mathematical, not organizational: SU(2) has closed-form
coefficients and needs no matrix computation; every other family must be
*constructed* numerically. If you only need SU(2), you never pull the backend.

## First calculation: SU(2) (no features)

```rust
use fusion_symbols::lie::su2::wigner_6j;

let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
```

**Spins are doubled.** Every SU(2) label in this crate is `dj = 2j`, so `2`
means spin 1 and `1` means spin ½. The same holds for projections: `dm = 2m`.
This is not a stylistic choice — it keeps half-integer spins as exact integers,
so no label is ever a float. See
[Representations § doubled spins](representations.md#su2-doubled-spins).

The returned [`SignedSqrtRational`] is an *exact* value of the form
`sign · sqrt(p/q)`; call `.to_f64()` when you want a number.

## First calculation: SU(3) (`cgc-gen`)

```rust
use fusion_symbols::{category::{FusionCategory}, lie::{GroupId, LieGroup}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

let fund = cat.irrep(&[1, 0]).unwrap();   // the 3 of SU(3)
assert_eq!(fund.dim_exact(), 3u32.into());
assert_eq!(cat.dual(&fund).dynkin(), vec![0, 1]);      // the 3-bar
```

An SU(N) irrep is named by its `N-1` Dynkin labels. See
[Representations](representations.md).

## How errors work

Abelian, fermion, CU(1) and table categories use `category::FusionCategory`.
Constructors validate objects, N = 0 is a value, and F/R return `Ok(0)` on
inadmissible vertices. Only fusion can overflow U(1)/CU(1) labels.
`CU1Irrep::new` retains `InvalidLabel` for invalid `(twice_charge, s)` pairs.
`TableCategory::object(index)` alone reports an out-of-range table index.
Foreign cyclic/table objects panic. Fusion-only tables return
`TableError::NoBraiding` from `r_symbol`; `twist` panics. Both reject foreign
objects before checking braiding.

Rank-one `LieGroup` category symbols return inline blocks, with a zero axis
for inadmissible vertices. Labels are validated at construction, and fusion
reports `LieError::LabelOverflow`. The exact evaluators and generated families
retain these error styles:

**The exact SU(2) evaluators return exact zero for a forbidden coupling.**
`lie::su2::{wigner_6j, wigner_3j, clebsch_gordan}` never
fail: a label set that violates a triangle or parity rule returns `0`, matching
the physics convention where a forbidden channel simply contributes nothing.

**The checked SU(2) functions distinguish "forbidden" from "zero".** An
*admissible* 6j can still be accidentally zero. When you need to tell the two
apart, use the `*_checked` twins:

```rust
use fusion_symbols::lie::su2::{wigner_6j_checked, Su2Error};

// Admissible, and genuinely nonzero.
assert!(wigner_6j_checked(2, 2, 2, 2, 2, 2).is_ok());

// Triangle violation: not a real zero, a forbidden label set.
assert!(matches!(
    wigner_6j_checked(2, 2, 20, 2, 2, 2),
    Err(Su2Error::NotAdmissible(_))
));
```

Both paths share one admissibility predicate, so they can never disagree.

**SU(N) generation is fallible.** `LieGroup::irrep` validates rank,
non-negativity and global-form admissibility. Foreign objects panic. An absent
channel returns an empty CGC/F/R block; a genuine generation failure returns
`LieError::Generation(GenerationError)`. The diagnostic implements `Display`
and exposes its underlying reason through `std::error::Error::source`.
B/C/D keeps its existing family errors and absent-channel behavior in this slice.

A verification-gate failure is always an error, never a silently degraded
number: `fusion-symbols` will not hand you a coefficient that failed its orthogonality,
unitarity, or pentagon/hexagon check.

## Caveats

- With `cgc-gen` on, the coefficient caches may retain several hundred MiB by
  default. If that matters, call `fusion_symbols::cache::configure_cache_budgets` once
  before first use — see [Resources](resources.md#bounding-cache-memory).
- The generated (`cgc-gen`) surface is marked **unstable**: its shape may still
  change. The base SU(2) surface is stable.

## Related API

- [`fusion_symbols::lie::su2::wigner_6j`](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/su2/fn.wigner_6j.html),
  [`fusion_symbols::lie::su2`](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/su2/index.html)
- [`fusion_symbols::lie`](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html),
  [`fusion_symbols::bcd`](https://docs.rs/fusion-symbols/latest/fusion_symbols/bcd/index.html)
- [`fusion_symbols::cache`](https://docs.rs/fusion-symbols/latest/fusion_symbols/cache/index.html)

[`SignedSqrtRational`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/struct.SignedSqrtRational.html
