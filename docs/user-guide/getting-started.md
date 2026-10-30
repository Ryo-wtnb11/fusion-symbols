# Getting started

This tutorial goes from installing the crate to reading an F symbol and a
Clebsch–Gordan coefficient, and checks one value of each against a textbook
table. The [user guide](README.md) covers each topic in more depth and has an
[API at a glance](README.md#api-at-a-glance) table.

Every example compiles with the default features and is mirrored in
`tests/doc_examples.rs`. Examples that use `?` run inside a function returning
`Result<(), Box<dyn std::error::Error>>`.

## Features

```toml
[dependencies]
fusion-symbols = "0.10"
```

| You use | Feature |
|---|---|
| SU(2), SO(3), U(1), U(2), tori, Spin(4)/SO(4), Z_N, fermion parity, rank ≤ 4 anyon tables, products of these | none (`default-features = false` is enough) |
| SU(N) and U(N) for N ≥ 3, SO(N)/Spin(N) for N ≥ 5, Sp(2r) for r ≥ 2, G₂, F₄, E₆, E₇, E₈ | `lie-generated` (on by default) |
| Anyon tables of rank 5–7 | `anyonwiki-unitary` |

Without `lie-generated`, building a group that needs it returns
`LieError::Unsupported { requires: "lie-generated" }`.

## Groups and irreps

A `GroupId` names a group; `LieGroup::new` turns it into a category whose
objects are irreps (`LieIrrep`). Irreps are named by the Dynkin labels of
their highest weight ([Dynkin labels](labels.md) explains them). SU(2) also
takes a spin, and groups with a torus, such as U(N), take integer charges.

```rust
use fusion_symbols::lie::{GroupId, HalfInt, LieGroup};

let su2 = LieGroup::new(GroupId::su(2)?)?;
let half = su2.irrep_spin(HalfInt::half(1))?; // spin 1/2 (HalfInt::half takes 2j)
let one = su2.irrep(&[2])?; // spin 1: the Dynkin label is 2j
assert_eq!(half.spin(), Some(HalfInt::half(1)));
assert_eq!(one.spin(), HalfInt::integer(1));

let su3 = LieGroup::new(GroupId::su(3)?)?;
let eight = su3.irrep(&[1, 1])?; // the adjoint
assert_eq!(eight.dim_exact(), 8u32);

let u2 = LieGroup::new(GroupId::u(2)?)?;
let doublet = u2.irrep_with_charges(&[1], &[-1])?; // SU(2) doublet, U(1) charge -1
assert_eq!(doublet.charges().collect::<Vec<_>>(), [-1]);
```

## Fusion and N

`fusion(a, b)` yields each irrep in `a ⊗ b` once; `nsymbol(a, b, c)` is the
number of times `c` occurs, zero when it does not occur.

```rust
use fusion_symbols::{category::FusionCategory, lie::{GroupId, HalfInt, LieGroup}};

let su2 = LieGroup::new(GroupId::su(2)?)?;
let one = su2.irrep_spin(HalfInt::half(2))?;
let spins: Vec<_> = su2.fusion(&one, &one)?.map(|c| c.spin()).collect();
let expected = [0, 2, 4].map(|dj| Some(HalfInt::half(dj)));
assert_eq!(spins, expected); // 1 ⊗ 1 = 0 ⊕ 1 ⊕ 2

let su3 = LieGroup::new(GroupId::su(3)?)?;
let eight = su3.irrep(&[1, 1])?;
assert_eq!(su3.nsymbol(&eight, &eight, &eight), 2); // the 8 occurs twice in 8 ⊗ 8
// Dimensions add up: Σ_c N(8,8,c) dim(c) = 8 · 8.
let total: f64 = su3
    .fusion(&eight, &eight)?
    .map(|c| f64::from(su3.nsymbol(&eight, &eight, &c)) * su3.dim(&c))
    .sum();
assert_eq!(total, 64.0);
```

## Dimensions and duals

`dim` is the dimension as `f64` (the quantum dimension, for anyon tables),
`dim_exact` the exact integer for a Lie irrep. `dual` gives the dual irrep;
`frobenius_schur_indicator` is `+1` for real, `-1` for pseudoreal and `0` for
irreps that are not self-dual.

```rust
use fusion_symbols::{category::FusionCategory, lie::{GroupId, HalfInt, LieGroup}};

let su3 = LieGroup::new(GroupId::su(3)?)?;
let three = su3.irrep(&[1, 0])?;
assert_eq!(su3.dim(&three), 3.0);
assert_eq!(su3.dual(&three).dynkin(), [0, 1]); // the 3̄
assert_eq!(su3.frobenius_schur_indicator(&three), 0.0);

let su2 = LieGroup::new(GroupId::su(2)?)?;
let half = su2.irrep_spin(HalfInt::half(1))?;
assert_eq!(su2.dual(&half), half);
assert_eq!(su2.frobenius_schur_indicator(&half), -1.0); // pseudoreal
```

## F and R symbols

`f_symbol(a, b, c, d, e, f)` is `F^{abc}_d[e, f]`: it maps the basis
`(a ⊗ b → e) ⊗ c → d` to `a ⊗ (b ⊗ c → f) → d`. `r_symbol(a, b, c)` is
`R^{ab}_c`, the exchange of `a` and `b` at a vertex `a ⊗ b → c`.

Each vertex can occur several times, so both return blocks. An F block has
axes `[μ, ν, κ, λ]` labeling the four vertices, with lengths
`[N(a,b,e), N(e,c,d), N(b,c,f), N(a,f,d)]`; an R block has axes `[μ, ν]` with
lengths `[N(a,b,c), N(b,a,c)]`. Read an entry with `at`. For SU(2) every
length is 1 and the value is `at(0, 0, 0, 0)`.

```rust
use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup}};

let su3 = LieGroup::new(GroupId::su(3)?)?;
let eight = su3.irrep(&[1, 1])?;
let unit = su3.unit();
let (a, b, c, d, e, f) = (&eight, &eight, &eight, &eight, &unit, &eight);
let block = su3.f_symbol(a, b, c, d, e, f)?;
let n = |x, y, z| su3.nsymbol(x, y, z) as usize;
let axes = [n(a, b, e), n(e, c, d), n(b, c, f), n(a, f, d)];
assert_eq!(block.dims(), axes);
assert_eq!(axes, [1, 1, 2, 2]); // N(8,8,1) = N(1,8,8) = 1, N(8,8,8) = 2
let r = su3.r_symbol(&eight, &eight, &eight)?;
assert_eq!(r.dim(), 2); // a 2 × 2 block: R(μ, ν) = r.at(μ, ν)
```

A vertex that fusion forbids gives a zero-length axis, not an error.

## Checking F against the Wigner 6j symbol

The SU(2) F symbol is a normalized Wigner 6j symbol. In the crate's frozen
convention ([gauge.md §1](../gauge.md#1-su2-closed-forms), the
TensorKitSectors convention), with spins `a, b, c, d, e, f`:

```text
F^{abc}_d[e, f] = (-1)^(a+b+c+d) · sqrt((2e+1)(2f+1)) · {a b e; c d f}
```

Take four spin-½ objects, `a = b = c = d = ½`. Then `(-1)^(a+b+c+d) = +1`
and `e, f ∈ {0, 1}`. The 6j symbols are standard values:

- With a zero entry, `{j₁ j₂ j₃; j₂ j₁ 0} = (-1)^(j₁+j₂+j₃) / sqrt((2j₁+1)(2j₂+1))`.
  So `{½ ½ 0; ½ ½ 0} = -1/2`, and `{½ ½ 0; ½ ½ 1} = {½ ½ 1; ½ ½ 0} = +1/2`
  (a 6j symbol is invariant under exchanging the upper and lower entries in
  two columns).
- `{½ ½ 1; ½ ½ 1} = 1/6` from Racah's single-sum formula: each of the four
  triangle factors is `Δ(½ ½ 1)² = 0!·1!·1!/3! = 1/6`, and the sum has the
  single term `t = 2`, equal to `(-1)² 3! / (0!⁴ · 0!·1!·1!) = 6`, so the
  symbol is `(1/6)² · 6 = 1/6`.

Both formulas are in Edmonds, *Angular Momentum in Quantum Mechanics*, §6.3,
and DLMF §34.4. Multiplying by `sqrt((2e+1)(2f+1))` (1, √3, √3 and 3):

```text
              f = 0           f = 1
e = 0     1 · (-1/2)      √3 · (1/2)        [ -1/2   √3/2 ]
e = 1     √3 · (1/2)      3 · (1/6)    =    [ √3/2   1/2  ]
```

This is an orthogonal matrix, as F must be. The exact evaluator
`lie::su2::wigner_6j` takes doubled spins:

```rust
use fusion_symbols::{category::FusionCategory, lie::{su2::wigner_6j, GroupId, HalfInt, LieGroup}};

assert!((wigner_6j(1, 1, 2, 1, 1, 2).to_f64() - 1.0 / 6.0).abs() < 1e-15);

let su2 = LieGroup::new(GroupId::su(2)?)?;
let h = su2.irrep_spin(HalfInt::half(1))?;
let inner = [su2.irrep_spin(HalfInt::half(0))?, su2.irrep_spin(HalfInt::half(2))?];
let r3 = 3f64.sqrt() / 2.0;
let expected = [[-0.5, r3], [r3, 0.5]];
for (i, e) in inner.iter().enumerate() {
    for (j, f) in inner.iter().enumerate() {
        let x = su2.f_symbol(&h, &h, &h, &h, e, f)?.at(0, 0, 0, 0);
        assert!((x - expected[i][j]).abs() < 1e-15);
    }
}
```

## Clebsch–Gordan coefficients

`cgc(a, b, c)` returns axes `[m₁, m₂, m₃, μ]`
([layout](README.md#clebschgordan-coefficients)). For SU(2), the magnetic
index is `m + j`, from the lowest weight up. The entry
`at([m₁ + j₁, m₂ + j₂, m₃ + j₃, 0])` is the Condon–Shortley coefficient
`⟨j₁ m₁; j₂ m₂ | j₃ m₃⟩`.

For `1 ⊗ ½ → ½`, the standard `j₁ ⊗ ½` formulas in the Condon–Shortley
convention, `⟨j₁ m+½; ½ -½ | j₁-½ m⟩ = +sqrt((j₁+m+½)/(2j₁+1))` and
`⟨j₁ m-½; ½ +½ | j₁-½ m⟩ = -sqrt((j₁-m+½)/(2j₁+1))`, give at `j₁ = 1`,
`m = ½` the values of the Particle Data Group table *Clebsch–Gordan
coefficients* (its `1 × 1/2` block):

- `⟨1 +1; ½ -½ | ½ +½⟩ = +sqrt(2/3)`, indices `[2, 0, 1, 0]`;
- `⟨1 0; ½ +½ | ½ +½⟩ = -sqrt(1/3)`, indices `[1, 1, 1, 0]`.

```rust
use fusion_symbols::{category::{CgcBlock, GroupRepresentations}, lie::{GroupId, HalfInt, LieGroup}};

let su2 = LieGroup::new(GroupId::su(2)?)?;
let one = su2.irrep_spin(HalfInt::half(2))?;
let half = su2.irrep_spin(HalfInt::half(1))?;
let c = su2.cgc(&one, &half, &half)?;
assert_eq!(c.dims(), [3, 2, 2, 1]);
assert!((c.at([2, 0, 1, 0]) - (2.0f64 / 3.0).sqrt()).abs() < 1e-15);
assert!((c.at([1, 1, 1, 0]) + (1.0f64 / 3.0).sqrt()).abs() < 1e-15);
assert_eq!(c.at([2, 1, 1, 0]), 0.0); // m₁ + m₂ ≠ m₃
```

## Products with U(1)

`Product::new` combines categories; objects are tuples. Symbols are products
of the factor symbols. `lie::U1` is U(1) with `Copy` labels.

```rust
use fusion_symbols::{category::FusionCategory,
    lie::{GroupId, HalfInt, LieGroup, U1Irrep, U1},
    product::Product};

let su2 = LieGroup::new(GroupId::su(2)?)?;
let cat = Product::new((su2.clone(), U1));
let electron = (su2.irrep_spin(HalfInt::half(1))?, U1Irrep::new(1)?);
let pairs: Vec<_> = cat.fusion(&electron, &electron)?.collect();
assert_eq!(pairs.len(), 2); // spin 0 and spin 1
assert!(pairs.iter().all(|(_, q)| q.charge() == 2));
assert_eq!(cat.dim(&electron), 2.0);

let charges = Product::new((U1, U1));
let x = (U1Irrep::new(1)?, U1Irrep::new(-1)?);
assert_eq!(charges.dual(&x), (U1Irrep::new(-1)?, U1Irrep::new(1)?));
```

## Common errors

```rust
use fusion_symbols::{category::FusionCategory, lie::{GroupError, GroupId, LieError, LieGroup}};

assert!(matches!(GroupId::su(1), Err(GroupError::UnsupportedRank { .. })));

let su3 = LieGroup::new(GroupId::su(3)?)?;
assert_eq!(su3.irrep(&[1]), Err(LieError::InvalidLabel)); // SU(3) takes two labels

let u2 = LieGroup::new(GroupId::u(2)?)?;
assert_eq!(u2.irrep(&[1]), Err(LieError::InvalidLabel)); // U(2) also needs a charge
assert_eq!(u2.irrep_with_charges(&[1], &[0]), Err(LieError::NotAdmissible));

let so3 = LieGroup::new(GroupId::so(3)?)?;
assert_eq!(so3.irrep(&[1]), Err(LieError::NotAdmissible)); // spin ½ is not an SO(3) irrep

let three = su3.irrep(&[1, 0])?;
assert_eq!(su3.nsymbol(&three, &three, &su3.unit()), 0); // not an error
```

- `LieError::InvalidLabel`: wrong label or charge count, or a negative label.
- `LieError::NotAdmissible`: the label is excluded by this global form.
- `LieError::Unsupported { requires: "lie-generated" }`: a needed feature is off.
- A panic: an object from another category value; check with `contains`.

A forbidden vertex is not an error: `nsymbol` is 0 and F, R and CGC return
zero-length blocks. Every case is in the guide's [errors table](README.md#errors).

## Where to go next

- [User guide](README.md): every category, A and B symbols, tables, caches,
  and the API at a glance.
- [Dynkin labels](labels.md): labels of common irreps, duals, U(N) charges.
- [Caches and reproducibility](resources.md): memory budgets and reproducibility.
- [Gauge specifications](../gauge.md): the exact basis and sign conventions.
- [API reference](https://docs.rs/fusion-symbols).
