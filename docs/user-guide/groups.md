# 3. Choosing a group

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## What is this?

For the `lie` family, three different things get called "the group", and `fusion-symbols` keeps them apart:

- the **Lie algebra / root system** (`A_{N-1}`, `B_r`, `C_r`, `D_r`) — fixes the
  weight lattice and all the combinatorics;
- the **compact group** whose representations you want (`SU(N)`, `Spin(N)`,
  `Sp(2r)`, …) — several groups share one root system;
- the **global form** — which of those groups: `SU(N)` vs `PSU(N)`, `Spin(N)` vs
  `SO(N)`.

The load-bearing fact is small: **a global form changes only which highest
weights are allowed, never the value of any coefficient of a surviving irrep.**
Dimensions, duals, fusion rules, CGC, F and R are all shared. So picking a
global form is a construction-time filter, not a different computation.

## When do I need this?

Choosing the *family* (SU(2)/SU(N)/B/C/D): always, at the first line.
Choosing the *global form*: only if you need spinors, or need to exclude irreps
that your physical system does not carry. Basics do not require it.

## SU(2)

Use `lie::LieGroup` for spin category operations and `lie::su2` for exact
3j/6j/CG evaluators. Labels are doubled spins, values
are exact closed forms, and no feature flag is needed.

```rust
use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup}};
let group = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
let one = group.irrep(&[2]).unwrap();
let channels: Vec<i64> = group.fusion(&one, &one).unwrap().map(|s| s.dynkin()[0]).collect();
assert_eq!(channels, vec![0, 2, 4]);
```

`GroupId::su(2)` always selects the exact rank-one path, including with
`lie-generated` enabled.
SO(3), Spin(3), Sp(2), and PSU(2) use the same implementation; SO(3)
and PSU(2) admit only even doubled spins. Objects retain distinct group tags.

## SU(N), N ≥ 3

Use `lie::LieGroup` with `GroupId::su(n)` (`lie-generated` for n ≥ 3). Coefficients are built by the Gelfand–Tsetlin
construction: GT patterns label the basis of an irrep uniquely, so ladder
operators have exact closed-form matrix elements and the CGC follow from a
highest-weight nullspace solve.

```rust
use fusion_symbols::{category::{FusionCategory}, lie::{GroupId, LieGroup}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

let eight = cat.irrep(&[1, 1]).unwrap();          // SU(3) adjoint
let decomposition = cat.fusion(&eight, &eight).unwrap()
    .map(|c| { let n = cat.nsymbol(&eight, &eight, &c); (c, n) })
    .collect::<std::collections::BTreeMap<_, _>>();
// 8 ⊗ 8 = 1 ⊕ 8 ⊕ 8 ⊕ 10 ⊕ 10-bar ⊕ 27 — note the 8 appears twice.
assert_eq!(decomposition[&eight], 2);
```

## SO(N) and Sp(2N): the B/C/D series

Use `fusion_symbols::lie::LieGroup` (`lie-generated`). These families have no practical GT-type basis, so
`fusion-symbols` builds them by a **generator bootstrap**: seed the defining
representation's generators explicitly, take tensor products, decompose them
numerically, harvest the new generator sets, recurse.

```rust
use fusion_symbols::{category::{FusionCategory},lie::{LieGroup,GroupId}};

// SO(5) = B_2: 5 ⊗ 5 = 1 ⊕ 10 ⊕ 14.
let cat = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
    let v = cat.irrep(&[1, 0]).unwrap();
let out = cat.fusion(&v, &v).map(|it| it.map(|c| { let n = cat.nsymbol(&v,&v,&c); (c,n) }).collect::<std::collections::BTreeMap<_,_>>()).unwrap();
let mut dims: Vec<String> = out.keys().map(|s| s.dim_exact().to_string()).collect();
dims.sort();
assert_eq!(dims, vec!["1", "10", "14"]);
```

The label combinatorics here — dimensions, duals, FS indicators, weight
multiplicities, the decomposition `N^c_ab` — are exact integer arithmetic and
do not materialize generators. CGC and F/R generation use the catalog
held inside `LieGroup`.

## Spin(N): when you need spinors

`Spin(N)` is the simply connected double cover of `SO(N)`. Use it when your
problem carries spinor representations — the ones whose ε-basis highest weights
are half-integers. `SO(N)` does not have them: a spinor is a representation of
the cover, not of `SO(N)` itself, which is why asking for a spinor label as an
`SO(N)` irrep is an error rather than a rounding.

```rust
use fusion_symbols::{lie::{LieGroup,GroupId}};

let spin5 = GroupId::spin(5).unwrap();
let s = LieGroup::new(spin5).unwrap().irrep(&[0, 1]).unwrap();
assert_eq!(s.dim_exact(), 4u32.into());        // the Spin(5) Dirac spinor

// The same label is not a representation of SO(5).
let so5 = GroupId::so(5).unwrap();
assert!(LieGroup::new(so5).unwrap().irrep(&[0, 1]).is_err());
```

Choose `GroupId::so(N)` for tensor irreps or `GroupId::spin(N)` to admit
spinors, then construct labels with `LieGroup::irrep`. Their coefficients come from the same
bootstrap, started from a second base case (the Clifford/Fock seeds).

`Spin(2r)` has two spinor chiralities (the two half-spin irreps `ω_{r-1}`,
`ω_r`); `Spin(2r+1)` has one.

## Sp(2N): naming, rank, and the defining dimension

This is the easiest place to be off by a factor of two, so it is spelled out.

- `GroupId::sp(2 * r)` names the group `Sp(2r)`, of Cartan type `C_r` and **rank `r`**.
- Its **defining representation has dimension `2r`**, Dynkin label
  `[1, 0, …, 0]`.
- So `GroupId::sp(4)` has `r = 2`, and its defining rep is the **4**.

```rust
use fusion_symbols::{category::{FusionCategory},lie::{LieGroup,GroupId}};

// Sp(4) = C_2. The defining 4, and 4 ⊗ 4 = 1 ⊕ 5 ⊕ 10.
let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
    let four = cat.irrep(&[1, 0]).unwrap();
assert_eq!(four.dim_exact(), 4u32.into());
let out = cat.fusion(&four,&four).map(|it| it.map(|c| { let n = cat.nsymbol(&four,&four,&c); (c,n) }).collect::<std::collections::BTreeMap<_,_>>()).unwrap();
let mut dims: Vec<String> = out.keys().map(|s| s.dim_exact().to_string()).collect();
dims.sort();
assert_eq!(dims, vec!["1", "10", "5"]); // string sort
```

`Sp(2r)` is simply connected, so it has no spinor sector and no
second global form to choose. (`PSp(2r) = Sp(2r)/Z₂` exists and is available via
`GroupId::psp`, but there is no cover above `Sp(2r)`.)

## Global forms and admissibility

Every connected compact group with a given root system is `G_sc/Γ` for a
subgroup `Γ` of the center of the simply connected form `G_sc`. Its irreps are
exactly those highest weights whose central character is trivial on `Γ`. That
predicate is [`GroupId::admits`], and it is the *whole* content of a global
form here.

```rust
use fusion_symbols::{lie::{GroupId, LieGroup}};
let psu3 = GroupId::psu(3).unwrap();       // SU(3)/Z₃

// The adjoint 8 has zero triality, so it is a genuine PSU(3) representation.
assert!(LieGroup::new(psu3).unwrap().irrep(&[1, 1]).is_ok());

// The fundamental 3 has triality 1: PSU(3) has no such representation.
assert!(LieGroup::new(psu3).unwrap().irrep(&[1, 0]).is_err());
```

Available forms: `GroupId::su`, `su_quotient(n, k)`, `psu`, `sp`, `psp`, `spin`,
`so`, `pso`, `half_spin_plus`, `half_spin_minus`.

**What "admissible" means, precisely.** Two different things in this crate are
called admissibility and they are unrelated:

1. *Global-form admissibility* (this section): is this weight a representation
   of this group at all? Enforced once, at `LieGroup::irrep` construction.
   Failure is `LieError::NotAdmissible`.
2. *Coupling admissibility* (SU(2)): does this label tuple satisfy the triangle
   and parity rules of a 3j/6j? Failure is an exact zero, or
   `Su2Error::NotAdmissible` on the `*_checked` functions.

Because the central-character class map is a group homomorphism, admissibility
is closed under fusion and duality: a product of admissible irreps decomposes
into admissible irreps only, and never needs re-checking. That is also why the
coefficient caches are form-free — two forms that share an irrep share its
basis, gauge, and cached values.

## Caveats

- `lie` covers positive-dimensional compact Lie groups: `LieGroup` handles
  supported simple root systems and global forms, while `U1` and `CU1`
  supply closed forms. CU(1) = O(2) is disconnected; other O(N), Pin(N),
  and non-compact groups are unsupported.
- Choosing a global form never changes a number. If you were hoping `SO(N)`
  coefficients would differ from `Spin(N)` ones on shared irreps: they do not,
  by construction.
- The `G2`–`E8` variants of [`RootSystem`] exist for type completeness but carry
  no coefficient engine.

## Related API

- [`fusion_symbols::lie`]: `GroupId`, `RootSystem`, `GlobalForm`, `CenterSubgroup`,
  `GroupId::admits`
- [`LieGroup::irrep`]
- Deeper background: [`../theory.pdf`](../theory.pdf) §3 (the Killing–Cartan
  classification and the Dynkin diagrams), §4 (each family's compact group,
  its centre, and the global forms as centre quotients), §5 (the
  classification-to-code correspondence table), §8 (why each family gets its
  algorithm), and the `fusion_symbols::lie` module docs for the lattice arithmetic.

[`GroupId::admits`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.GroupId.html#method.admits
[`RootSystem`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/enum.RootSystem.html
[`fusion_symbols::lie`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/index.html
[`LieGroup::irrep`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html
