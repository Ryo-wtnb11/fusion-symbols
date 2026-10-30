# Dynkin labels

`LieGroup::irrep(&[a₁, …, a_r])` names an irreducible representation by the
Dynkin labels of its highest weight. This page explains what those numbers
mean, how they are numbered for each root system, and how to read off
dimensions and duals. Every code block is mirrored in `tests/doc_examples.rs`.

## Definition

A finite-dimensional irrep of a compact connected Lie group is fixed by its
highest weight λ. Write λ in the basis of fundamental weights ωᵢ:

λ = Σᵢ aᵢ ωᵢ,  aᵢ = ⟨λ, αᵢ∨⟩ = 2(λ, αᵢ)/(αᵢ, αᵢ) ∈ ℤ≥0,

where α₁, …, α_r are the simple roots and αᵢ∨ their coroots. The rank r is the
number of labels. Every tuple of nonnegative integers is the highest weight of
exactly one irrep of the simply connected group; other global forms admit only
some tuples (see [global forms](#global-forms)). For a product of root systems,
the labels of the factors are concatenated in factor order.

## Node numbering

fusion-symbols numbers simple roots as Bourbaki does. In the diagrams, an arrow
between two nodes points toward the shorter root.

| Type | Group | Diagram |
|---|---|---|
| A_r | SU(r+1) | `1 — 2 — … — r` |
| B_r | Spin(2r+1), SO(2r+1) | `1 — 2 — … — (r−1) ⇒ r` (r short) |
| C_r | Sp(2r) | `1 — 2 — … — (r−1) ⇐ r` (r long) |
| D_r | Spin(2r), SO(2r) | `1 — 2 — … — (r−2)`, which branches to both `r−1` and `r` |
| G₂ | G₂ | `1 ⇐ 2` (1 short) |
| F₄ | F₄ | `1 — 2 ⇒ 3 — 4` (1, 2 long) |
| E₆, E₇, E₈ | E₆, E₇, E₈ | `1 — 3 — 4 — 5 — 6 (— 7 — 8)`, with `2` attached to `4` |

`GroupId::sp(m)`, `so(m)` and `spin(m)` take the defining matrix dimension m,
not the rank: `sp(6)` is C₃.

## Common irreps

| Group | Irrep | Labels | Dimension |
|---|---|---|---|
| SU(2) | spin j | `[2j]` | 2j + 1 |
| SU(3) | 3, 3̄ | `[1,0]`, `[0,1]` | 3 |
| SU(3) | 8 (adjoint) | `[1,1]` | 8 |
| SU(3) | 6, 10 | `[2,0]`, `[3,0]` | 6, 10 |
| SU(N) | fundamental | `[1,0,…,0]` | N |
| SU(N), N ≥ 3 | adjoint | `[1,0,…,0,1]` | N² − 1 |
| SO(2r+1), r ≥ 2 | vector | `ω₁ = [1,0,…,0]` | 2r + 1 |
| Spin(2r+1), r ≥ 2 | spinor | `ω_r = [0,…,0,1]` | 2^r |
| SO(2r), r ≥ 3 | vector | `ω₁` | 2r |
| Spin(2r), r ≥ 3 | half-spinors | `ω_{r−1}`, `ω_r` | 2^(r−1) each |
| Sp(2r) | defining | `ω₁` | 2r |
| G₂ | 7, 14 | `[1,0]`, `[0,1]` | 7, 14 |
| F₄ | 26, 52 | `ω₄`, `ω₁` | 26, 52 |
| E₆ | 27, 27̄, 78 | `ω₁`, `ω₆`, `ω₂` | 27, 27, 78 |
| E₇ | 56, 133 | `ω₇`, `ω₁` | 56, 133 |
| E₈ | 248 | `ω₈` | 248 |

The smallest ranks follow the rank-one and D₂ conventions instead. The SU(2)
adjoint is `[2]`. SO(3) = B₁ uses the doubled-spin label, so its vector is
`[2]` and `[1]` (spin ½) is not an SO(3) irrep. SO(4) = D₂ has two A₁ labels
`[a, b]` of dimension (a+1)(b+1), so its vector is `[1,1]`.

Here `ωᵢ` is the label tuple with a single 1 in position i. Labels follow the
Bourbaki plates; the dimensions agree with the tables of Slansky (1981), which
number some diagrams differently (see [references](#references)).

```rust
use fusion_symbols::lie::{GroupId, LieGroup};

let su3 = LieGroup::new(GroupId::su(3)?)?;
assert_eq!(su3.irrep(&[1, 1])?.dim_exact(), 8u32);
assert_eq!(su3.irrep(&[3, 0])?.dim_exact(), 10u32);

let spin7 = LieGroup::new(GroupId::spin(7)?)?;
assert_eq!(spin7.irrep(&[0, 0, 1])?.dim_exact(), 8u32); // spinor

let e7 = LieGroup::new(GroupId::e7())?;
assert_eq!(e7.irrep(&[0, 0, 0, 0, 0, 0, 1])?.dim_exact(), 56u32);
```

## SU(N) and Young diagrams

An SU(N) irrep is also given by a Young diagram with row lengths
λ₁ ≥ λ₂ ≥ … ≥ λ_N, where only the differences matter. The labels are the
differences of consecutive rows:

aᵢ = λᵢ − λᵢ₊₁ (i = 1, …, N−1).

The diagram with one row of length k is `[k, 0, …, 0]`, the symmetric power.
The diagram with one column of height k is `ω_k`, the antisymmetric power.

## Dimension

Generated labels store their exact Weyl dimension and its rounded `f64` value
at construction; repeated `dim()` reads need no global lock. Conjugate views
share one label payload, so `dual()` needs no cache lookup.

The Weyl dimension formula gives the dimension from the labels:

dim λ = Π_{α > 0} ⟨λ + ρ, α∨⟩ / ⟨ρ, α∨⟩,

with ρ = Σᵢ ωᵢ, the product taken over the positive roots α. For SU(3) with
labels `[a, b]` it reduces to

dim = (a + 1)(b + 1)(a + b + 2) / 2.

`LieIrrep::dim_exact` returns the exact value; `FusionCategory::dim` returns
it as `f64`.

```rust
use fusion_symbols::lie::{GroupId, LieGroup};

let su3 = LieGroup::new(GroupId::su(3)?)?;
for a in 0..4u32 {
    for b in 0..4u32 {
        let dim = (a + 1) * (b + 1) * (a + b + 2) / 2;
        assert_eq!(su3.irrep(&[a.into(), b.into()])?.dim_exact(), dim);
    }
}
```

## Duals

The dual of the irrep with highest weight λ has highest weight −w₀λ, where w₀
is the longest Weyl group element. On labels, −w₀ is a symmetry of the diagram:

| Type | Dual labels |
|---|---|
| A_r | reversed: aᵢ ↔ a_{r+1−i} |
| D_r, r odd | a_{r−1} ↔ a_r, the others unchanged |
| E₆ | a₁ ↔ a₆ and a₃ ↔ a₅ |
| B_r, C_r, D_r (r even), G₂, F₄, E₇, E₈ | unchanged: every irrep is self-dual |

A self-dual irrep is real or pseudo-real: its Frobenius–Schur indicator is
(−1)^⟨λ, 2ρ∨⟩, where 2ρ∨ is the sum of the positive coroots. For example, the
SU(2) spin-½ irrep and the defining irrep of Sp(2r) are pseudo-real, and so
are the E₇ irreps with odd a₂ + a₅ + a₇.

```rust
use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup}};

let su3 = LieGroup::new(GroupId::su(3)?)?;
assert!(su3.dual(&su3.irrep(&[2, 0])?).dynkin().eq([0, 2]));

let e6 = LieGroup::new(GroupId::e6())?;
let v = e6.irrep(&[1, 0, 1, 0, 0, 0])?;
assert!(e6.dual(&v).dynkin().eq([0, 0, 0, 0, 1, 1]));

let sp4 = LieGroup::new(GroupId::sp(4)?)?;
assert_eq!(sp4.frobenius_schur_indicator(&sp4.irrep(&[1, 0])?), -1.0);
```

## Global forms

Groups with the same root system differ in which labels they admit. The
simply connected form (`su`, `spin`, `sp`, `g2`, …, `e8`) admits every
label. A quotient admits only labels on which its kernel acts trivially:

- SO(2r+1) admits `a_r` even, so it has no spinor.
- SO(2r) admits `a_{r−1} + a_r` even.
- SU(N)/Z_k admits `Σ i·aᵢ ≡ 0 (mod k)`; PSU(N) is k = N.

`irrep` returns `LieError::NotAdmissible` for an excluded label. The exact
rules for each constructor are on [`GroupId`](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.GroupId.html).

## Torus charges and U(N)

A group with a torus, such as U(1), Tᵗ or U(N), takes integer charges after
the Dynkin labels: `irrep_with_charges(dynkin, charges)`. For U(N) = (SU(N) ×
U(1))/Z_N, the charge q of the fundamental is +1, so q is the number of boxes
Σᵢ λᵢ of the Young diagram, and the determinant has charge N. Here the
diagram is not normalized to λ_N = 0: adding a column of height N leaves the
Dynkin labels unchanged and raises q by N. A pair
(labels, q) is admitted exactly when

q ≡ Σᵢ i·aᵢ (mod N).

```rust
use fusion_symbols::lie::{GroupId, LieError, LieGroup};

let u3 = LieGroup::new(GroupId::u(3)?)?;
assert_eq!(u3.irrep_with_charges(&[1, 0], &[1])?.dim_exact(), 3u32);
assert!(u3.irrep_with_charges(&[1, 0], &[-2]).is_ok()); // −2 ≡ 1 (mod 3)
assert_eq!(u3.irrep_with_charges(&[1, 0], &[0]), Err(LieError::NotAdmissible));
assert!(u3.irrep_with_charges(&[1, 1], &[3]).is_ok()); // 1 + 2 ≡ 0 (mod 3)
assert_eq!(u3.irrep_with_charges(&[1, 1], &[1]), Err(LieError::NotAdmissible));
```

## Rank one and spins

SU(2) and SO(3) have one label a = 2j, so spin ½ is `[1]`. They also accept a
spin directly through `irrep_spin` with a `HalfInt`, and `LieIrrep::spin` reads
it back. SO(3) admits only integer spins.

```rust
use fusion_symbols::lie::{GroupId, HalfInt, LieGroup};

let su2 = LieGroup::new(GroupId::su(2)?)?;
let three_halves = su2.irrep_spin(HalfInt::half(3))?;
assert!(three_halves.dynkin().eq([3]));
assert_eq!(three_halves.dim_exact(), 4u32);
assert_eq!(three_halves.spin(), Some(HalfInt::half(3)));

let so3 = LieGroup::new(GroupId::so(3)?)?;
assert!(so3.irrep_spin(HalfInt::half(1)).is_err());
```

## References

- N. Bourbaki, *Lie Groups and Lie Algebras*, Chapters 4–6, Springer (2002),
  Chapter VI, Plates I–IX: simple roots, numbering, fundamental weights and −w₀.
- R. Slansky, *Group theory for unified model building*, Phys. Rep. 79 (1981)
  1–128, [doi:10.1016/0370-1573(81)90092-2](https://doi.org/10.1016/0370-1573(81)90092-2):
  irrep tables with Dynkin labels and dimensions.
