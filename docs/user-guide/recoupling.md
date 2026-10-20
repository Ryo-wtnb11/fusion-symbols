# 6. Recoupling: F- and R-symbols

> **AI-generated, for agentic coding.** This document was written by an AI agent
> as reference material for AI agents (and humans) working on this repository.
> It may contain errors — check it against the code and tests rather than
> trusting it blindly.

## What is this?

Coupling three irreps `a, b, c` to a total `d` can be done in two orders, and
both give a complete basis of the same space:

```text
(a ⊗ b) ⊗ c → d      via an intermediate e         ("left" tree)
a ⊗ (b ⊗ c) → d      via an intermediate f         ("right" tree)
```

The **F-symbol** `F^{abc}_d[e, f]` is the change of basis between them. The
**R-symbol** `R^{ab}_c` is the change of basis under swapping two factors,
`a ⊗ b → c` versus `b ⊗ a → c`.

Both are contractions of Clebsch–Gordan tensors over all magnetic indices; what
survives are the *multiplicity* indices, one per coupling vertex.

For SU(2) these reduce to the classical closed forms: the F-symbol is the 6j
symbol times a dimension factor and a phase, and R is a sign.

## When do I need them?

When you work in a coupled basis and need to reassociate or braid without ever
touching magnetic indices — the standard situation in tensor networks with
non-abelian symmetry, in recoupling algebra, and in fusion-tree manipulations.
If you need explicit product-basis vectors instead, you want
[CGC](clebsch-gordan.md).

## Abelian categories and fermion parity

Import `FusionCategory` to query `abelian::{U1, ZN}` or
`fermion::FermionParity`. Labels retain their validating constructors:

```rust
use fusion_symbols::abelian::{ZN, ZNIrrep};
use fusion_symbols::category::{FusionCategory, SymbolBlock};

let category = ZN::new(3).unwrap();
let one = ZNIrrep::from_canonical(1, 3).unwrap();
let two = ZNIrrep::from_canonical(2, 3).unwrap();
assert_eq!(category.fusion(&one, &one).unwrap().next(), Some(two));
assert_eq!(category.r_symbol(&one, &one, &two), Ok(1.0));
let zero = category.f_symbol(&one, &one, &one, &one, &one, &one).unwrap();
assert_eq!(zero, 0.0); // an inadmissible vertex is a value
assert_eq!(<f64 as SymbolBlock<4>>::dims(&zero), [1; 4]);
```

Logical F shape is `[N(a,b,e), N(e,c,d), N(b,c,f), N(a,f,d)]`;
`SymbolBlock::dims` describes storage only. Bare scalars keep all-one axes
even for zero symbols. A foreign modulus panics on every category operation.
`ZN` and parity have `Infallible` errors; U(1) fusion reports overflow.
`frobenius_schur_phase` is +1 for these families, while
`frobenius_schur_indicator` is zero for non-self-dual objects.
Odd parity has R(odd, odd, even) = −1 and twist −1.

## SU(2): exact values in inline blocks

```rust
use fusion_symbols::{category::FusionCategory, lie::{su2::wigner_6j, GroupId, LieGroup}};
assert!((wigner_6j(2, 2, 2, 2, 2, 2).to_f64() - 1.0 / 6.0).abs() < 1e-14);
let group = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
let half = group.irrep(&[1]).unwrap();
let unit = group.unit();
let f = group.f_symbol(&half, &half, &half, &half, &unit, &unit).unwrap();
assert_eq!(f.at(0, 0, 0, 0), -0.5);
let r = group.r_symbol(&half, &half, &unit).unwrap();
assert_eq!(r.at(0, 0), -1.0);
```

`group.f_symbol(a,b,c,d,e,f)` returns `F^{abc}_d[e,f]` with axes
`[N(a,b,e), N(e,c,d), N(b,c,f), N(a,f,d)]`; `r_symbol(a,b,c)` has
axes `[N(a,b,c), N(b,a,c)]`. Admissible A1 blocks store one `f64`
inline. Inadmissible vertices give an empty block with a zero axis.
The exact 3j/6j/CG evaluators retain their `_checked` forms and `Su2Error`.

## SU(N): blocks over multiplicity indices

```rust
use fusion_symbols::{category::{FusionCategory}, lie::{GroupId, LieGroup}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

let three = cat.irrep(&[1, 0]).unwrap();
let anti = cat.dual(&three);
let eight = cat.irrep(&[1, 1]).unwrap();

// F^{3 3bar 3}_{3}[8, 8]
let block = cat.f_symbol(&three, &anti, &three, &three, &eight, &eight).unwrap();
assert_eq!(block.dims(), [1, 1, 1, 1]);
assert!((block.at(0, 0, 0, 0) - 1.0 / 3.0).abs() < 1e-12);

// R^{3 3bar}_{8}
let r = cat.r_symbol(&three, &anti, &eight).unwrap();
assert_eq!(r.dim(), 1);
assert!((r.at(0, 0).abs() - 1.0).abs() < 1e-12);
```

The argument order is `f_symbol(a, b, c, d, e, f)` for `F^{abc}_d[e, f]` and
`r_symbol(a, b, c)` for `R^{ab}_c`.

## SO(N) / Sp(2N)

Identical, with a [`CanonicalCatalog`](clebsch-gordan.md#son-sp2n-you-need-a-catalog)
threaded through as the first argument:

```rust
use fusion_symbols::bcd::{f_symbol, CanonicalCatalog, Irrep, Series};

let mut cat = CanonicalCatalog::new(Series::C, 2).unwrap();   // Sp(4) = C_2
let triv = Irrep::trivial(Series::C, 2).unwrap();
let five = Irrep::from_dynkin(Series::C, &[0, 1]).unwrap();   // the 5
let ten = Irrep::from_dynkin(Series::C, &[2, 0]).unwrap();    // the adjoint 10

let block = f_symbol(&mut cat, &triv, &five, &five, &ten, &five, &ten).unwrap();
assert_eq!(block.dims(), [1, 1, 1, 1]);
assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
```

(With `a` trivial the F-move is the identity, so the single entry is 1.)

## What does the result mean?

### `FBlock` — the F-symbol

A dense rank-4 array over the four **multiplicity** indices `[μ, ν, κ, λ]`,
stored row-major. One index per vertex of the recoupling move:

| Axis | Index | Vertex | Length |
|---|---|---|---|
| 0 | `μ` | `a ⊗ b → e` | `N^e_ab` |
| 1 | `ν` | `e ⊗ c → d` | `N^d_ec` |
| 2 | `κ` | `b ⊗ c → f` | `N^f_bc` |
| 3 | `λ` | `a ⊗ f → d` | `N^d_af` |

`FBlock::dims()` returns exactly those four lengths;
`FBlock::at(mu, nu, kappa, lambda)` reads one element; `FBlock::data()` is the
flat row-major buffer. In a multiplicity-free situation all four lengths are 1
and the block holds one scalar at `at(0, 0, 0, 0)`.

This axis order matches the `GenericFusion` convention of TensorKitSectors, so
blocks can be handed to a consumer expecting that layout without a permutation.

### `RBlock` — the R-symbol

A dense `N^c_ab × N^c_ba` matrix. `RBlock::dim()` is `N^c_ab`,
`RBlock::at(mu, nu)` reads an element, `RBlock::data()` is the flat row-major
buffer. Multiplicity-free means a `1×1` block holding a phase.

## Worked example: the multiplicity indices at work

The SU(3) product `8 ⊗ 8` contains the adjoint **twice** (`N^8_88 = 2`), so
with `a = b = c = d = e = 8` every row of the axis table above has length 2:
`f_symbol(&eight, &eight, &eight, &eight, &eight, &eight)` returns a
`[2, 2, 2, 2]` block, and no multiplicity index can be ignored. The contraction
below verifies the F-move itself in the m-basis, coefficient by coefficient:
the left tree contracted over `m_e` must equal the F-weighted sum of right
trees contracted over `m_f`, for every magnetic assignment and every
`(μ, ν, κ, λ)`. Each CGC factor carries its vertex's multiplicity index on the
trailing axis — `μ` on `a ⊗ b → e`, `ν` on `e ⊗ c → d`, `κ` on `b ⊗ c → f`,
`λ` on `a ⊗ f → d` — exactly as in the table.

```rust
use fusion_symbols::{category::{FusionCategory, GroupRepresentations}, lie::{GroupId, LieGroup, LieIrrep}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

let eight = cat.irrep(&[1, 1]).unwrap();
let dim = 8usize;

// N^8_88 = 2, so every multiplicity axis below has length 2.
assert_eq!(cat.fusion(&eight, &eight).unwrap()
    .map(|c| { let n = cat.nsymbol(&eight, &eight, &c); (c, n) })
    .collect::<std::collections::BTreeMap<_, _>>()[&eight], 2);

// Materialize a CGC into (row-major buffer, dims), indexed [m1][m2][m3][mu].
let dense = |s1: &LieIrrep, s2: &LieIrrep, s3: &LieIrrep| {
    let t = cat.cgc(s1, s2, s3).unwrap();
    let [d1, d2, d3, n] = t.dims();
    let mut buf = vec![0.0f64; d1 * d2 * d3 * n];
    for m1 in 0..d1 { for m2 in 0..d2 { for m3 in 0..d3 { for mu in 0..n {
        buf[((m1 * d2 + m2) * d3 + m3) * n + mu] = t.at(m1, m2, m3, mu);
    }}}}
    (buf, [d1, d2, d3, n])
};
let at = |t: &(Vec<f64>, [usize; 4]), i: usize, j: usize, k: usize, l: usize| {
    t.0[((i * t.1[1] + j) * t.1[2] + k) * t.1[3] + l]
};

// Left tree, e = 8: C^{ab->e} carries mu, C^{ec->d} carries nu (both [8, 8, 8, 2]).
let left = dense(&eight, &eight, &eight);

// Right tree: for every channel f of 8 x 8, the F block F^{888}_8[e=8, f]
// (shape [2, 2, N^f_88, N^8_8f]) and the two CGC carrying kappa and lambda.
let right: Vec<_> = cat.fusion(&eight, &eight).unwrap()
    .map(|c| { let n = cat.nsymbol(&eight, &eight, &c); (c, n) })
    .collect::<std::collections::BTreeMap<_, _>>()
    .keys()
    .map(|f| {
        (
            cat.f_symbol(&eight, &eight, &eight, &eight, &eight, f).unwrap(),
            dense(&eight, &eight, f), // C^{bc->f}[mb, mc, mf, kappa]
            dense(&eight, f, &eight), // C^{af->d}[ma, mf, md, lambda]
        )
    })
    .collect();

// The F-move, one magnetic assignment at a time:
//   sum_me C^{ab->e}_mu C^{ec->d}_nu
//     = sum_f sum_{kappa, lambda} F[mu, nu, kappa, lambda] sum_mf C^{bc->f}_kappa C^{af->d}_lambda
let mut worst = 0.0f64;
for ma in 0..dim {
    for mb in 0..dim {
        for mc in 0..dim {
            for md in 0..dim {
                for mu in 0..2 {
                    for nu in 0..2 {
                        let mut lhs = 0.0;
                        for me in 0..dim {
                            lhs += at(&left, ma, mb, me, mu) * at(&left, me, mc, md, nu);
                        }
                        let mut rhs = 0.0;
                        for (fblock, c_bcf, c_afd) in &right {
                            let [_, _, n_kappa, n_lambda] = fblock.dims();
                            let dim_f = c_bcf.1[2];
                            for kappa in 0..n_kappa {
                                for lambda in 0..n_lambda {
                                    let mut pair = 0.0;
                                    for mf in 0..dim_f {
                                        pair += at(c_bcf, mb, mc, mf, kappa)
                                            * at(c_afd, ma, mf, md, lambda);
                                    }
                                    rhs += fblock.at(mu, nu, kappa, lambda) * pair;
                                }
                            }
                        }
                        worst = worst.max((lhs - rhs).abs());
                    }
                }
            }
        }
    }
}
assert!(worst < 1e-9, "F-move residual {worst}");
```

The `f = 8` term is the `[2, 2, 2, 2]` block; the other channels of `8 ⊗ 8`
(`1`, `10`, `10bar`, `27`) are multiplicity-free on `κ` and `λ` but still
carry the full `2 × 2` left-tree `(μ, ν)` axes.

## Self-checks you can call

Generic residual functions evaluate the identities against `FusionCategory`,
including multiplicity sums. They return the maximum absolute residual; the
caller chooses the tolerance:

```rust
use fusion_symbols::{category::{FusionCategory}, lie::{GroupId, LieGroup}};
let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

let three = cat.irrep(&[1, 0]).unwrap();
let anti = cat.dual(&three);
assert!(fusion_symbols::category::f_unitarity_residual(&cat, &three, &anti, &three, &three).unwrap() < 1e-10);
assert!(fusion_symbols::category::pentagon_residual(&cat, &three, &anti, &three, &anti).unwrap() < 1e-10);
assert!(fusion_symbols::category::hexagon_residual(&cat, &three, &anti, &three).unwrap() < 1e-10);
```

`fusion_symbols::bcd` retains its generation checks (taking
`&mut CanonicalCatalog` first), plus `check_commutators` for the seed algebra. CGC orthonormality and R-orthogonality
run as generation gates only — they are not callable, and a violation surfaces
as a typed error.

## Caveats

- **B/C/D (`bcd` module): an empty vertex is an error.** If any of the four F-symbol vertices has zero
  fusion multiplicity, the call returns a typed error rather than a zero block.
  `LieGroup` returns a zero block for an empty vertex.
- **Cost.** The first F-symbol on a set of labels generates the underlying CGC;
  after that both the CGC and the F block are cached. R is a single sparse join
  of two CGC and is not separately cached.
- **The multiplicity axes are not interchangeable.** `μ` and `ν` belong to the
  left tree, `κ` and `λ` to the right one. Transposing them silently produces a
  different (wrong) recoupling.
- **Gauge.** F and R values depend on the CGC gauge. `fusion-symbols` publishes them in
  one frozen canonical gauge; see [Resources § gauge](resources.md#gauge-and-reproducibility).

## Related API

- [`fusion_symbols::category::FusionCategory::f_symbol`], [`fusion_symbols::category::FusionCategory::r_symbol`], [`fusion_symbols::lie::su2::wigner_6j`]
- [`FusionCategory::f_symbol`], [`FusionCategory::r_symbol`], [`category::f_unitarity_residual`],
  [`category::pentagon_residual`], [`category::hexagon_residual`]
- [`bcd::f_symbol`], [`bcd::r_symbol`]
- [`FBlock`], [`RBlock`]

[`fusion_symbols::category::FusionCategory::f_symbol`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.f_symbol
[`fusion_symbols::category::FusionCategory::r_symbol`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.r_symbol
[`fusion_symbols::lie::su2::wigner_6j`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/su2/fn.wigner_6j.html
[`FusionCategory::f_symbol`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html
[`FusionCategory::r_symbol`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html
[`category::f_unitarity_residual`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/category/fn.f_unitarity_residual.html
[`category::pentagon_residual`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/category/fn.pentagon_residual.html
[`category::hexagon_residual`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/category/fn.hexagon_residual.html
[`bcd::f_symbol`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/bcd/fn.f_symbol
[`bcd::r_symbol`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/bcd/fn.r_symbol
[`FBlock`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html
[`RBlock`]: https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html
