# 2. Representations

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## What is an irrep here?

An **irrep** (irreducible representation) is identified by a label. For Lie
groups this is a dominant integral highest weight; CU(1) uses a charge and
conjugation label. Table categories instead use validated simple-object
indices. Dimensions, duals and coefficients are queried through the family
API or category value.

Each family has its own label type, because each family has its own natural
encoding:

| Family | Type | Label |
|---|---|---|
| SU(2) | [`lie::LieIrrep`] | group tag plus an inline `u32`, the **doubled spin** `dj = 2j` |
| SU(N) | [`lie::LieIrrep`] | `N-1` Dynkin labels (stored internally as a normalized weight) |
| SO(N), Spin(N), Sp(2N) | [`lie::LieIrrep`] | `r` Dynkin labels plus the Cartan series `B`/`C`/`D` |
| U(1), Z_N | `lie::U1Irrep`, `finite::ZNIrrep` | integer charge; cyclic labels carry their modulus |
| CU(1) | `lie::CU1Irrep` | twice charge `u32` and `s`: `(0,0)`, `(0,1)`, or `(q>0,2)` |
| Fermion parity | `fermion::FermionParityIrrep` | boolean parity |
| Shipped categories | `table::TableObject` | validated one-based index and private catalog tag |

## When do I need this?

Always — every coefficient function takes irrep labels. Getting the label
convention right is the single most common source of confusion, so this chapter
is worth reading once before anything else.

## CU(1) and table objects

CU(1) labels keep `CU1Irrep::new(twice_charge, s)?`,
`CU1Irrep::from_twice_charge(q)`, `VACUUM` and `PSEUDOSCALAR`. Mathematical
queries go through the `lie::CU1` category and `category::FusionCategory`.

Table labels come from their category, not from a public tuple constructor:

```rust
use fusion_symbols::{category::FusionCategory, table::TableCategory};

let fib = TableCategory::fibonacci();
let tau = fib.object(2).unwrap();
assert_eq!(tau.index(), 2);
assert_eq!(fib.fusion(&tau, &tau).unwrap().count(), 2);
```

`object(index)` rejects indices outside `1..=rank`. Every table query panics
on an object from another uuid, even if its index fits. Clones and independently
constructed categories with the same uuid accept the same objects. Clones
share parsed and derived data; category equality and hashing use the uuid.

## SU(2): doubled spins

```rust
use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup}};
let group = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
let s = group.irrep(&[3]).unwrap(); // doubled spin 3, i.e. spin 3/2
assert_eq!(s.dynkin(), vec![3]);
assert_eq!(s.dim_exact(), 4.into());
assert_eq!(group.dual(&s), s);
assert_eq!(group.frobenius_schur_phase(&s), -1.0);
```

Dynkin labels are validated by `LieGroup::irrep`: one nonnegative label fitting
`u32`, admitted by the global form. SO(3) and PSU(2) reject odd labels.
Projections for the exact evaluators are doubled too: `dm = 2m`, an `i32`.

**Why doubled.** Spin ½ is not an integer, and a float label would make
admissibility tests approximate. `dj = 2j` keeps every label exact and turns the
triangle and parity rules into integer arithmetic. The B/C/D families use the
same trick for the same reason — see [doubled weights](#son-sp2n-dynkin-labels-and-doubled-weights).

## SU(N): Dynkin labels

An SU(N) irrep is built from its `N-1` Dynkin labels `(a₁,…,a_{N-1})`, all
non-negative integers, in Bourbaki numbering (`a₁` is attached to the first node
of the `A_{N-1}` chain).

```rust
use fusion_symbols::{category::{FusionCategory}, lie::{GroupId, LieGroup}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

let three = cat.irrep(&[1, 0]).unwrap();   // SU(3) fundamental
let eight = cat.irrep(&[1, 1]).unwrap();   // SU(3) adjoint
let singlet = cat.unit();
assert_eq!(singlet.dim_exact(), 1.into());

assert_eq!(three.dim_exact(), 3u32.into());
assert_eq!(eight.dim_exact(), 8u32.into());
assert_eq!(cat.dual(&three).dynkin(), vec![0, 1]);      // 3-bar
assert_eq!(cat.dual(&eight), eight);                    // the adjoint is self-dual
```

The category fixes `N`; `LieGroup::irrep` requires exactly `N-1` nonnegative
Dynkin labels and checks the global form. `dim_exact()` returns an unbounded
`BigInt`; `cat.dim(&a)` returns `f64`. `LieIrrep` is `Clone`, not `Copy`.

For a partition `λ`, convert to Dynkin labels with `aᵢ = λᵢ − λᵢ₊₁`
before constructing the label. `gt_patterns()` returns `Some(Vec<GtPattern>)`
for A(r), r ≥ 2, in the frozen GT basis order, and `None` for A1 and B/C/D.
Generator matrices and partition storage are private implementation details.

## SO(N), Sp(2N): Dynkin labels and doubled weights

`LieGroup` covers B(r) (`Spin(2r+1)` and its quotients), C(r) (`Sp(2r)`
and its quotients), and D(r) (`Spin(2r)` and its quotients). Choose the global
form with `GroupId`, then construct rank-many integer Dynkin labels with `irrep`.

```rust
use fusion_symbols::{category::{FusionCategory},lie::{LieGroup,GroupId}};

// SO(5) = B_2. Dynkin [1, 0] is the 5-dimensional vector representation.
let cat = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
    let v = cat.irrep(&[1, 0]).unwrap();
assert_eq!(v.dim_exact(), 5u32.into());
assert_eq!(cat.dual(&v), v);
assert_eq!(cat.frobenius_schur_indicator(&v), 1.0);       // real (orthogonal) self-duality
```

`weight_multiplicities()` returns all weights in Dynkin coordinates, including
spinor weights, with exact integer multiplicities. `gt_patterns()` is `None`
for B/C/D. The internal doubled epsilon weights are private.

B(1) and C(1) use the exact A1 implementation. D(2) returns
`LieError::Unsupported`, naming `Spin(4) = SU(2) × SU(2)`.

## Duals and Frobenius–Schur indicators

The **dual** `ā` of an irrep `a` is the conjugate representation; `a ⊗ ā` always
contains the singlet exactly once. The **Frobenius–Schur indicator** is only
meaningful for a *self-dual* irrep (`a == ā`), where it says *how* the irrep is
self-dual:

| FS value | Meaning |
|---|---|
| `+1` | real / orthogonal self-duality (a symmetric invariant bilinear form) |
| `−1` | pseudo-real / symplectic self-duality (an antisymmetric one) |
| `0` | not self-dual (complex) |

```rust
use fusion_symbols::{category::{FusionCategory},lie::{LieGroup,GroupId}};

// Sp(4) = C_2. Its defining 4 is pseudo-real; the 5 is real.
let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
    let four = cat.irrep(&[1, 0]).unwrap();
let five = cat.irrep(&[0, 1]).unwrap();
assert_eq!(four.dim_exact(), 4u32.into());
assert_eq!(cat.frobenius_schur_indicator(&four), -1.0);
assert_eq!(five.dim_exact(), 5u32.into());
assert_eq!(cat.frobenius_schur_indicator(&five), 1.0);
```

For SU(2), `FusionCategory::frobenius_schur_phase(dj)` is `(-1)^dj`: `+1` for integer spin, `−1`
for half-integer. For every Lie group, `cat.frobenius_schur_indicator(&a)` is zero when
`a != cat.dual(&a)` and otherwise equals `cat.frobenius_schur_phase(&a)`.

## Caveats

- `LieGroup` fixes the rank: a wrong-length label is `LieError::InvalidLabel`.
- Objects from different groups (including different global forms) are distinct;
  passing a foreign object to a category method panics.
- Negative Dynkin labels are rejected everywhere. In the `D` series a *negative
  last partition entry* `λ_r` is legal (it is the chirality sign) but that is
  the partition, not the Dynkin label.

## Related API

- [`lie::LieGroup`], [`lie::LieIrrep`], [`fusion_symbols::category::FusionCategory::frobenius_schur_phase`]
- `LieGroup::irrep`, `LieIrrep::dynkin`, `dim_exact`, `gt_patterns`,
  `weight_multiplicities`; other queries use `FusionCategory`.

[`lie::LieIrrep`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieIrrep.html
[`fusion_symbols::category::FusionCategory::frobenius_schur_phase`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.frobenius_schur_phase
[`lie::LieGroup`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html
