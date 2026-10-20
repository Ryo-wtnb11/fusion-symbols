# 4. Tensor products and fusion

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## What is this?

The tensor product of two irreps decomposes into a direct sum of irreps:

```text
a ⊗ b  ≅  ⨁_c  N^c_ab · c
```

The non-negative integer `N^c_ab` is the **fusion multiplicity** (also: outer
multiplicity): how many independent copies of `c` sit inside `a ⊗ b`. `fusion-symbols`
computes it exactly, in integer arithmetic, for every family.

## When do I need it?

Before anything else. `N^c_ab` tells you which coupling channels exist at all,
and it is the length of the multiplicity axis on every CGC, F and R block you
ask for. If `N^c_ab = 0` there is no coefficient to compute, and the
`FusionCategory` APIs return zero symbols. Generated Lie groups return
blocks whose axis lengths are the vertex multiplicities, including zero.

## SU(2)

The classic triangle rule. `LieGroup::fusion` returns an allocation-free
iterator over the coupled doubled spins `|dj1 − dj2| ..= dj1 + dj2` in steps of
2, each with multiplicity 1 (SU(2) is multiplicity-free).

```rust
use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup}};
let group = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
let half = group.irrep(&[1]).unwrap();
let channels: Vec<i64> = group.fusion(&half, &half).unwrap().map(|s| s.dynkin()[0]).collect();
assert_eq!(channels, vec![0, 2]);
```

The only failure is `LieError::LabelOverflow`, when `dj1 + dj2` exceeds `u32`.

## SU(N)

`cat.fusion(&a, &b)` yields each outcome once in deterministic order.
`cat.nsymbol(&a, &b, &c)` gives its exact integer multiplicity. Collect a
map only when needed; the iterator shares the cached product list.

```rust
use fusion_symbols::{category::{FusionCategory}, lie::{GroupId, LieGroup}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

let three = cat.irrep(&[1, 0]).unwrap();
let anti = cat.dual(&three);
let out = cat.fusion(&three, &anti).unwrap()
    .map(|c| { let n = cat.nsymbol(&three, &anti, &c); (c, n) })
    .collect::<std::collections::BTreeMap<_, _>>();

// 3 ⊗ 3-bar = 1 ⊕ 8, each once.
assert_eq!(out.len(), 2);
assert_eq!(out[&cat.unit()], 1);
assert_eq!(out[&cat.irrep(&[1, 1]).unwrap()], 1);
```

### Multiplicity greater than one

`N^c_ab > 1` is the normal case for `N ≥ 3`, and it is what makes the
coefficient blocks arrays instead of scalars.

```rust
use fusion_symbols::{category::{FusionCategory}, lie::{GroupId, LieGroup}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

let eight = cat.irrep(&[1, 1]).unwrap();       // SU(3) adjoint
let out = cat.fusion(&eight, &eight).unwrap()
    .map(|c| { let n = cat.nsymbol(&eight, &eight, &c); (c, n) })
    .collect::<std::collections::BTreeMap<_, _>>();
assert_eq!(out[&eight], 2);     // the 8 appears twice in 8 ⊗ 8
```

Repeated `fusion` and `nsymbol` calls use the same process-global cache;
a warm second call does not sweep the weights again. Each yielded label is cloned.

## SO(N) / Sp(2N) / Spin(N)

B/C/D use the same `fusion` iterator and `nsymbol` queries, backed by exact
Brauer–Klimyk / Racah–Speiser decomposition. These queries do not generate CGC.
Products are cached independently of each group's retained generators.

```rust
use fusion_symbols::{category::{FusionCategory},lie::{LieGroup,GroupId}};

// SO(5) = B_2: 5 ⊗ 5 = 1 ⊕ 10 ⊕ 14.
let cat = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
    let v = cat.irrep(&[1, 0]).unwrap();
let out = cat.fusion(&v, &v).map(|it| it.map(|c| { let n = cat.nsymbol(&v,&v,&c); (c,n) }).collect::<std::collections::BTreeMap<_,_>>()).unwrap();
assert_eq!(out.len(), 3);
for (irrep, mult) in &out {
    assert_eq!(*mult, 1);
    let _ = irrep.dim_exact();
}
```

## CU(1) and shipped categories

Both use `category::FusionCategory`. CU(1) has simple fusion and bosonic
braiding; the iterator keeps TensorKitSectors' source order and reports
`CU1Error::TwiceChargeOutOfRange` if a charge sum exceeds `u32::MAX`.

```rust
use fusion_symbols::{category::FusionCategory, cu1::{CU1, CU1Irrep}};

let a = CU1Irrep::from_twice_charge(1);
let channels: Vec<_> = CU1.fusion(&a, &a).unwrap().collect();
assert_eq!(channels, vec![CU1Irrep::VACUUM, CU1Irrep::PSEUDOSCALAR,
                         CU1Irrep::from_twice_charge(2)]);
```

`table::TableCategory` yields tagged objects in ascending one-based index
order. Its fusion style is `Unique` if every pair has one outcome, otherwise
`Simple`. Braiding style is `None` for fusion-only tables, `Bosonic` for
symmetric braiding with all twists one, `Fermionic` for symmetric braiding
with some twist minus one, and `Anyonic` otherwise.

## What does the result mean?

- The map's **keys** are the irreps `c` that occur; irreps with `N^c_ab = 0` are
  absent, not present with value `0`.
- The **values** are `N^c_ab`. Summing `N^c_ab · dim(c)` over the map reproduces
  `dim(a) · dim(b)` — a cheap sanity check.
- The map is a `BTreeMap`, so iteration order is the deterministic `Ord` order
  of the label type. That order is stable, but it is *not* a physics ordering
  (not by dimension, not by highest weight height).

## Caveats

- SU(N) category methods panic on foreign objects. B/C/D decomposition
  retains its typed family/rank mismatch errors.
- Multiplicity is `u32`. The B/C/D weight-multiplicity accumulation is `i128`
  internally and reports overflow as a typed error rather than wrapping.
- Fusion respects global forms automatically: if `a` and `b` are admissible for
  a form, every `c` in the decomposition is too, so no re-filtering is needed.

## Related API

- [`category::FusionCategory::fusion`], [`lie::LieFusion`]
- [`FusionCategory::fusion`], [`FusionCategory::fusion`], [`lie::LieFusion`]

[`category::FusionCategory::fusion`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.fusion
[`lie::LieFusion`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieFusion.html
[`FusionCategory::fusion`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html
[`FusionCategory::fusion`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html
[`lie::LieFusion`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html
