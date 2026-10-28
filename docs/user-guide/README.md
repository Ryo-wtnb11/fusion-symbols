# User guide

New to the crate? Start with [getting started](getting-started.md).

This guide shows how to do the common tasks. Install and features are in the
[README](../../README.md); the precise contracts (validation order, overflow,
enumeration, product layout) are in the [design reference](../overview.md);
per-item details are on [docs.rs](https://docs.rs/fusion-symbols).

Every example compiles with the default features and is mirrored in
`tests/doc_examples.rs`. Examples that use `?` run inside a function returning
`Result<(), Box<dyn std::error::Error>>`.

## API at a glance

Object-taking calls are `FusionCategory` methods on the category value unless
another type is named.

| I want | Call | Reference |
|---|---|---|
| Build SU(N), SO(N), Spin(N), Sp(2r), U(N), a torus or G₂…E₈ | `LieGroup::new(GroupId::su(3)?)`, `GroupId::{so, spin, sp, u, torus, g2, …}` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html#method.new) |
| Build a product group or a quotient from root data | `GroupId::cover`, `GroupId::new` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.GroupId.html#method.cover) |
| An irrep from Dynkin labels | `LieGroup::irrep(&[1, 1])` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html#method.irrep) |
| An irrep with torus charges (U(N), tori) | `LieGroup::irrep_with_charges(&[1], &[1])` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html#method.irrep_with_charges) |
| An SU(2) or SO(3) irrep from a spin | `LieGroup::irrep_spin(HalfInt::half(1))` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieGroup.html#method.irrep_spin) |
| Read a label back | `LieIrrep::{dynkin, dynkin_labels, charges, spin}` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.LieIrrep.html) |
| Dimension | `dim` (f64), `LieIrrep::dim_exact` (exact) | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.dim) |
| Dual and Frobenius–Schur data | `dual`, `frobenius_schur_indicator`, `frobenius_schur_phase` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.dual) |
| The unit object | `unit` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.unit) |
| Fusion outcomes of `a ⊗ b` | `fusion(a, b)` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.fusion) |
| Fusion multiplicity N | `nsymbol(a, b, c)`, `try_nsymbol` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.nsymbol) |
| F symbol | `f_symbol(a, b, c, d, e, f)` → `[μ,ν,κ,λ]` block | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.f_symbol) |
| R symbol, twist | `r_symbol(a, b, c)`, `twist(a)` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.r_symbol) |
| A and B symbols | `a_symbol`, `b_symbol` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#method.a_symbol) |
| Read a symbol block | `FBlock::{dims, at}`, `RBlock::{dim, at}`, `SymbolBlock` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/struct.FBlock.html) |
| Clebsch–Gordan coefficients | `GroupRepresentations::cgc`, read with `CgcBlock::{dims, at}` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.GroupRepresentations.html#tymethod.cgc) |
| Exact SU(2) 3j, 6j, CG | `lie::su2::{wigner_3j, wigner_6j, clebsch_gordan}` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/su2/index.html) |
| U(1), Z_N, fermion parity, CU(1) | `U1` + `U1Irrep::new`, `ZN::new` + `ZNIrrep::new`, `FermionParity`, `CU1` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/lie/struct.U1.html) |
| Fibonacci, Ising or another anyon table | `TableCategory::{fibonacci, ising, by_uuid, object}` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/table/struct.TableCategory.html) |
| Several symmetries at once | `Product::new((a, b))`, `Product::new(vec![…])` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/product/struct.Product.html#method.new) |
| Test an object before use (no panic) | `contains(a)` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#method.contains) |
| Enumerate objects | `object_count`, `object_at`, `object_index` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#method.object_count) |
| Check pentagon, hexagon or F unitarity | `category::{pentagon_residual, hexagon_residual, f_unitarity_residual}` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/index.html) |
| Limit or inspect cache memory | `cache::{configure_cache_budgets, tier_stats, trim_to, reset}` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/cache/index.html) |
| Record the value convention | `authority_fingerprint` | [docs](https://docs.rs/fusion-symbols/latest/fusion_symbols/category/trait.FusionCategory.html#tymethod.authority_fingerprint) |

## Choosing a category

All categories implement `category::FusionCategory`: objects, unit, duals,
dimensions, fusion, N, F, R, A, B, Frobenius–Schur data and twists.
Categories that are representation categories of groups also implement
`category::GroupRepresentations`, which adds CGC.

| You need | Use | Objects |
|---|---|---|
| A compact connected Lie group: SU(N), SO(N), Spin(N), Sp(2r), G₂, F₄, E₆, E₇, E₈, U(N), tori, products, quotients | `lie::LieGroup` built from a `lie::GroupId` | `LieIrrep` |
| Exact SU(2) 3j, 6j or CG values as rationals under a square root | `lie::su2::{wigner_3j, wigner_6j, clebsch_gordan}` | doubled spins `u32` |
| U(1) with `Copy` labels and scalar symbols | `lie::U1` | `U1Irrep` |
| CU(1) = O(2) | `lie::CU1` | `CU1Irrep` |
| Z_N | `finite::ZN` | `ZNIrrep` |
| Fermion parity | `fermion::FermionParity` | `FermionParityIrrep` |
| Fibonacci, Ising, or another unitary multiplicity-free category of rank ≤ 7 | `table::TableCategory` | `TableObject` |
| Several symmetries at once | `product::Product` | tuples or vectors of factor objects |

Not provided: disconnected groups other
than CU(1), noncompact groups, finite non-abelian groups Rep(G), and
categories with multiplicity that are not Lie groups.

A category is a value. Objects are created by that value and validated once.
Equal category values accept each other's objects; an object from a category
that is not equal to this one (a different `GroupId`, table uuid, Z_N modulus,
…) makes every method except `contains` panic.

## Building Lie groups

A `GroupId` names a connected compact group: an ordered list of root systems,
a torus rank, and a character lattice that fixes which labels exist.
`LieGroup::new` turns it into a category.

By name:

```rust
use fusion_symbols::lie::{GroupId, LieGroup};

let su3 = LieGroup::new(GroupId::su(3)?)?;
let so5 = LieGroup::new(GroupId::so(5)?)?;
let spin5 = LieGroup::new(GroupId::spin(5)?)?;
let sp4 = LieGroup::new(GroupId::sp(4)?)?; // defining dimension 4, rank 2
let psu3 = LieGroup::new(GroupId::psu(3)?)?; // SU(3)/Z3
let u2 = LieGroup::new(GroupId::u(2)?)?;
let t2 = LieGroup::new(GroupId::torus(2)?)?;
assert_eq!(su3.group_id().semisimple_rank(), 2);
assert_eq!(sp4.group_id().semisimple_rank(), 2);
assert_eq!((u2.group_id().semisimple_rank(), u2.group_id().torus_rank()), (1, 1));
assert_eq!(t2.group_id().torus_rank(), 2);
// Same root system, different global forms: different groups.
assert_ne!(so5.group_id(), spin5.group_id());
assert_ne!(psu3.group_id(), su3.group_id());
```

Named constructors: `su(n)`, `su_quotient(n, k)`, `psu(n)`, `u(n)`,
`torus(t)`, `so(n)`, `spin(n)`, `pso(n)`, `sp(m)`, `psp(m)`,
`half_spin_plus(n)`, `half_spin_minus(n)`. Orthogonal and symplectic
arguments are the defining matrix dimension, not the rank.

By root data, for products and arbitrary global forms:

```rust
use fusion_symbols::category::FusionCategory;
use fusion_symbols::lie::{GroupId, LieGroup, RootSystem};

// SU(2) × SU(3) × U(1): the full weight and charge lattice.
let cover = GroupId::cover(&[RootSystem::A(1), RootSystem::A(2)], 1)?;
let g = LieGroup::new(cover)?;
assert_eq!(g.group_id().semisimple_rank(), 3);

// U(2) = (SU(2) × U(1))/Z2 from lattice generators (rows over the Dynkin
// label and the charge). The lattice is normalized, so this equals u(2).
let u2 = GroupId::new(&[RootSystem::A(1)], 1, &[vec![1, 1], vec![0, 2]])?;
assert_eq!(u2, GroupId::u(2)?);

// G₂, F₄, E₆, E₇ and E₈ have exact coefficients (with `lie-generated`).
// E₆'s 27 and 27̄ (Bourbaki labels) are each other's duals.
let e8 = LieGroup::new(GroupId::e8())?;
assert_eq!(e8.group_id(), &GroupId::cover(&[RootSystem::E8], 0)?);
let e6 = LieGroup::new(GroupId::e6())?;
let v = e6.irrep(&[1, 0, 0, 0, 0, 0])?;
assert_eq!(e6.dual(&v).dynkin(), [0, 0, 0, 0, 0, 1]);
assert_eq!(e6.frobenius_schur_indicator(&v), 0.0);
```

`GroupId::new(roots, torus_rank, generators)` takes square integer row
generators; the lattice must have full rank and contain every root. The
[design reference](../overview.md#lie-labels-and-global-forms) gives the
validation rules and the D-series naming conventions.

Which groups need `lie-generated`: every simple factor of type A(r ≥ 2),
B(r ≥ 2), C(r ≥ 2), D(r ≥ 3), G₂, F₄, E₆, E₇ or E₈. Rank-one factors (A1, B1, C1), D2, and tori
work in the base build. Without the feature, `LieGroup::new` returns
`LieError::Unsupported { requires: "lie-generated" }`.

## Labels

`LieGroup::irrep(dynkin)` takes the highest weight as nonnegative Dynkin
labels (Bourbaki numbering, factors in order); groups with a torus, such as
U(N), take integer torus charges through `irrep_with_charges(dynkin, charges)`.
SU(2) and SO(3), in any rank-one presentation, also take a spin, `irrep_spin(HalfInt::half(1))` for spin ½,
whose Dynkin label is `2j`. [Dynkin labels](labels.md) explains the labels,
lists common irreps with their labels and dimensions, and gives the duality
and U(N) charge rules.

```rust
use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieError, LieGroup}};

let su3 = LieGroup::new(GroupId::su(3)?)?;
let three = su3.irrep(&[1, 0])?;
assert_eq!(su3.dual(&three).dynkin(), vec![0, 1]); // 3-bar
assert_eq!(three.dim_exact(), 3u32.into());

let u2 = LieGroup::new(GroupId::u(2)?)?;
let det = u2.irrep_with_charges(&[0], &[2])?; // determinant: charge 2
assert_eq!(det.charges().collect::<Vec<_>>(), vec![2]);
assert_eq!(u2.irrep_with_charges(&[1], &[0]), Err(LieError::NotAdmissible));

// The global form decides which labels exist: Spin(5) has the spinor, SO(5) not.
let spinor = [0, 1];
assert!(LieGroup::new(GroupId::spin(5)?)?.irrep(&spinor).is_ok());
assert!(LieGroup::new(GroupId::so(5)?)?.irrep(&spinor).is_err());
```

Read labels back with `dynkin_labels()` (allocation-free) or `dynkin()`, and
charges with `charges()`. `dim_exact()` is the exact dimension;
`weight_multiplicities()` lists weights; `gt_patterns()` gives the
Gelfand–Tsetlin basis of A-series groups of rank ≥ 2.

## Fusion and symbols

```rust
use fusion_symbols::{category::FusionCategory, lie::{GroupId, HalfInt, LieGroup}};

let su2 = LieGroup::new(GroupId::su(2)?)?;
let half = su2.irrep_spin(HalfInt::half(1))?;
let unit = su2.unit();
let channels: Vec<i64> = su2.fusion(&half, &half)?.map(|c| c.dynkin()[0]).collect();
assert_eq!(channels, vec![0, 2]); // spin 0 and spin 1
assert_eq!(su2.frobenius_schur_indicator(&half), -1.0); // pseudoreal

let f = su2.f_symbol(&half, &half, &half, &half, &unit, &unit)?;
assert_eq!(f.at(0, 0, 0, 0), -0.5);
let r = su2.r_symbol(&half, &half, &unit)?;
assert_eq!(r.at(0, 0), -1.0);
```

- `fusion(a, b)` yields each outcome once; `nsymbol(a, b, c)` is its
  multiplicity, zero when `c` does not occur.
- `dim`, `dual`, `frobenius_schur_phase`, `frobenius_schur_indicator` and
  `twist` are per-object data.
- `f_symbol`, `r_symbol`, `a_symbol` and `b_symbol` return blocks whose axes
  are vertex multiplicities (next section). An inadmissible vertex gives a
  zero block, not an error; errors mean a genuine failure or a missing
  capability, such as R on a fusion-only table.
- `object_count`, `object_at` and `object_index` enumerate objects where the
  order is defined; see [enumeration](../overview.md#enumeration-and-products).

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

With multiplicities these are genuine blocks; for SU(3) `8 ⊗ 8 → 8` all four F
axes can have length 2:

```rust
use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup}};

let su3 = LieGroup::new(GroupId::su(3)?)?;
let eight = su3.irrep(&[1, 1])?;
let f = su3.f_symbol(&eight, &eight, &eight, &eight, &eight, &eight)?;
assert_eq!(f.dims(), [2, 2, 2, 2]);
let r = su3.r_symbol(&eight, &eight, &eight)?;
assert_eq!(r.dim(), 2); // square: N(a,b,c) = N(b,a,c) = 2
```

All blocks implement `category::SymbolBlock` (`dims`, `get`, `as_slice`, row
major). `category::FBlock` and `RBlock` add `at` and `data`. Scalar families
(U(1), Z_N, parity, tables) return a bare scalar whose storage axes are all one
even when its logical value is zero; read the logical shape from N. A and B
always return `RBlock` and need no braiding. The multiplicity copy order is
gauge, fixed in [SU(N) §8](../gauge.md#8-outer-multiplicity-axis) and
[B/C/D §9](../gauge_soN.md#9-outer-multiplicity-assignment).

## Clebsch–Gordan coefficients

Import `GroupRepresentations` for `cgc` and `CgcBlock` to read the result:

```rust
use fusion_symbols::{category::{CgcBlock, GroupRepresentations}, lie::{GroupId, LieGroup}};

let su3 = LieGroup::new(GroupId::su(3)?)?;
let eight = su3.irrep(&[1, 1])?;
let c = su3.cgc(&eight, &eight, &eight)?;
assert_eq!(c.dims(), [8, 8, 8, 2]); // [dim a, dim b, dim c, N(a,b,c)]
assert!(c.at([0, 2, 0, 1]) < 0.0);
```

`at([m1, m2, m3, mu])` reads one coefficient and panics outside the axes. An
absent channel has a zero-length multiplicity axis. Magnetic indices: A1
ascends from the lowest weight; higher A-series follow `gt_patterns()`;
B/C/D follow the canonical frame of [gauge_soN.md](../gauge_soN.md). `U1` and
`ZN` return a scalar CGC: `1.0` for a present channel, `0.0` for an absent one.

The exact SU(2) evaluators under `lie::su2` take doubled spins and doubled
projections, in the argument order `(dj1, dm1, dj2, dm2, dj3, dm3)` for
`clebsch_gordan`. They return `SignedSqrtRational` (`sign * sqrt(p/q)`);
compare exactly with `signed_square()` or convert with `to_f64()`. A
forbidden label set gives exact zero; the `*_checked` variants report it as
`Su2Error::NotAdmissible` instead.

```rust
use fusion_symbols::lie::su2::clebsch_gordan;

// <1/2 +1/2, 1/2 -1/2 | 0 0> = 1/sqrt(2)
let cg = clebsch_gordan(1, 1, 1, -1, 0, 0);
assert!((cg.to_f64() - 0.5f64.sqrt()).abs() < 1e-15);
```

## Abelian and finite categories

These have closed-form scalar symbols (`f64`) and need no feature.

```rust
use fusion_symbols::{category::FusionCategory,
    fermion::{FermionParity, FermionParityIrrep},
    finite::{ZNIrrep, ZN},
    lie::{CU1Irrep, U1Irrep, CU1, U1}};

let q = U1Irrep::new(2)?;
assert_eq!(U1.dual(&q).charge(), -2);

let z3 = ZN::new(3)?;
let one = ZNIrrep::from_canonical(1, 3)?;
assert_eq!(z3.fusion(&one, &one)?.next(), Some(ZNIrrep::from_canonical(2, 3)?));

assert_eq!(FermionParity.twist(&FermionParityIrrep::ODD), -1.0);

let a = CU1Irrep::from_twice_charge(1);
let channels: Vec<_> = CU1.fusion(&a, &a)?.collect();
assert_eq!(channels, [CU1Irrep::VACUUM, CU1Irrep::PSEUDOSCALAR, CU1Irrep::from_twice_charge(2)]);
```

`U1` is the same group as `LieGroup::new(GroupId::u(1)?)`, with `Copy` labels
and bitwise-identical symbols; use it, or a tuple `Product` of `U1`, on hot
paths. `ZNIrrep::new` reduces any charge modulo N; `from_canonical` requires
`0..N`.

## Tables

`TableCategory` serves finite categories from the pinned AnyonWiki database.
Symbols are `Complex64`; objects are one-based source indices.

```rust
use fusion_symbols::{category::FusionCategory, table::{TableCategory, TableError}};

let fib = TableCategory::fibonacci();
let tau = fib.object(2)?; // index 1 is the unit
assert_eq!(fib.nsymbol(&tau, &tau, &tau), 1);
let golden = (1.0 + 5f64.sqrt()) / 2.0;
assert!((fib.dim(&tau) - golden).abs() < 1e-12);
let f = fib.f_symbol(&tau, &tau, &tau, &tau, &fib.unit(), &fib.unit())?;
assert!((f.re - 1.0 / golden).abs() < 1e-12);

assert!(TableCategory::shipped_uuids().len() >= 74);
assert!(matches!(TableCategory::by_uuid("not-a-uuid"), Err(TableError::UnknownCategory)));
```

Open a category with `fibonacci()`, `ising()`, `by_uuid(uuid)` or
`by_code(code)`; `shipped_uuids()` lists what this build ships. 74
categories of rank ≤ 4 ship by default and 303 more of rank 5–7 with the
`anyonwiki-unitary` feature. A known but unshipped entry returns
`TableError::Unsupported` with an `UnsupportedReason`. Fusion-only tables
return `TableError::NoBraiding` from R and panic on `twist`. Values are in
AnyonWiki's gauge; Ising differs from TensorKitSectors' `IsingAnyon` by a
vertex gauge and object order (see [differences](../tensorkitsectors-differences.md)).
Provenance and pins are in [category conventions](../references_categories.md).

## Products

`Product::new` takes a tuple of 2 to 12 categories, or a `Vec` of any number
of categories of one type. Objects are tuples or vectors of factor objects.

```rust
use fusion_symbols::{category::{CgcBlock, FusionCategory, GroupRepresentations},
    fermion::{FermionParity, FermionParityIrrep},
    lie::{GroupId, HalfInt, LieGroup, U1Irrep, U1},
    product::Product};

let cat = Product::new((U1, FermionParity));
let odd = (U1Irrep::new(1)?, FermionParityIrrep::ODD);
assert_eq!(cat.twist(&odd), -1.0);

let su2 = LieGroup::new(GroupId::su(2)?)?;
let spin_charge = Product::new((su2.clone(), U1));
let x = (su2.irrep_spin(HalfInt::half(1))?, U1Irrep::new(1)?);
let y = (su2.irrep_spin(HalfInt::half(2))?, U1Irrep::new(2)?);
let c = spin_charge.cgc(&x, &x, &y)?;
assert_eq!(c.dims(), [2, 2, 3, 1]);

let chain = Product::new(vec![U1; 4]);
assert_eq!(chain.unit(), vec![U1Irrep::UNIT; 4]);
```

Nesting is kept: `Product::new((Product::new((a, b)), c))` has objects
`((x, y), z)`, distinct from the flat product. Products add no cross-factor
sign; a fermionic product's signs come from its parity factor. A product is a
`GroupRepresentations` when all factors are. Axis order, validation and errors
are in [enumeration and products](../overview.md#enumeration-and-products). For a connected group such as
SU(2) × U(1) you can also build one `LieGroup` from
`GroupId::cover(&[RootSystem::A(1)], 1)`; the values are the same, and only
`LieGroup` can express quotients such as U(2).

## Errors

Each category has one error type: `LieError`, `U1Error`, `CU1Error`,
`TableError`, `ProductError`, or `Infallible` for `ZN` and `FermionParity`.
`GroupId` constructors return `GroupError`.

| Situation | Result |
|---|---|
| Vertex not allowed by fusion | `nsymbol` is 0; F/R/A/B/CGC return zero blocks |
| Object from another category value | panic (check first with `contains`) |
| Wrong label lengths (including `irrep` on a group with a torus), negative Dynkin label, or `irrep_spin` outside SU(2) and SO(3) | `LieError::InvalidLabel` |
| Label excluded by the global form | `LieError::NotAdmissible` |
| Label, weight or multiplicity beyond its integer type | `LieError::LabelOverflow` (and `U1Error`/`CU1Error::LabelOverflow`) |
| E₆ irrep that is not self-dual, with dimension above `u32::MAX` (its Frobenius–Schur phase is generated in a basis with `u32` indices) | `LieError::LabelOverflow` from `irrep`, and from `fusion` when an outcome is such a label |
| Z_N modulus zero, or `ZNIrrep::from_canonical` charge outside `0..N` | `ZNError::ZeroModulus`, `ZNError::NonCanonicalCharge` |
| Invalid CU(1) label `(twice_charge, s)` | `CU1Error::InvalidLabel` |
| Cache budgets configured after a cache operation or a second time | `CacheBudgetError::AlreadyInitialized` |
| Group needs `lie-generated` | `LieError::Unsupported { requires }` |
| Numerical generation failed its checks | `LieError::Generation` with an error source |
| Invalid rank or lattice | `GroupError::UnsupportedRank`, `InvalidLattice`, `NotACenterSubgroup`, `SizeOverflow` |
| R on a fusion-only table or product factor | `TableError::NoBraiding`, `ProductError::NoBraiding { factor }` |

The full validation order is in [validation and errors](../overview.md#validation-and-errors).

## Caches

Generated coefficients and exact SU(2) values are cached in process-global,
byte-bounded tiers. The defaults are bounded (192 MiB for the SU(2) tiers,
768 MiB + 128 KiB for the generated tiers), so most programs do nothing.

```rust
use fusion_symbols::cache::{tier_stats, CoefficientCacheTier};

let _ = fusion_symbols::lie::su2::wigner_6j(2, 2, 2, 2, 2, 2);
let six_j = tier_stats(CoefficientCacheTier::SixJ);
println!("{} entries, {} bytes, {} hits", six_j.entries, six_j.bytes, six_j.hits);
// A sum over tiers combines independent snapshots; use it for reporting only.
let total: usize = CoefficientCacheTier::ALL.iter().map(|&t| tier_stats(t).bytes).sum();
println!("{total} bytes retained across all tiers");
```

The application, not a library, owns the policy: configure smaller budgets
once with `cache::configure_cache_budgets` before any cache use, and release
memory with `cache::trim_to` or `cache::reset`. Each B/C/D `LieGroup` also
owns a generator catalog (256 MiB default, `LieGroup::with_catalog_budget`).
Persist `authority_fingerprint()` next to any derived data. Details:
[caches and reproducibility](resources.md).

## Identity checks

`category::{f_unitarity_residual, pentagon_residual, hexagon_residual}` return
the maximum absolute residual of each identity, including multiplicity sums;
choose the tolerance yourself. They check consistency, not a particular gauge.
