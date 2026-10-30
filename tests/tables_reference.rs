//! Reference tests for the `table` catalog.
//!
//! Two independent kinds of check, deliberately kept apart:
//!
//! 1. **Gauge equivalence with TensorKitSectors' closed forms.**
//!    `FibonacciAnyon` and `IsingAnyon` of TensorKitSectors 0.3.9
//!    (`src/anyons.jl`: `Nsymbol`, `Fsymbol`, `Rsymbol`, `dim`) are written
//!    out below; FS phase and twist follow from them by TensorKitSectors'
//!    generic `frobenius_schur_phase` and `twist_from_Rsymbol`. Each named
//!    constructor must equal them under a stated object map and vertex gauge,
//!    the documented one, and not under the wrong gauge or the mirror
//!    braiding. A few Fibonacci entries are also pinned bitwise as the
//!    correctly rounded exact values.
//! 2. **Consistency checks** over every shipped entry: unitarity, triangle,
//!    pentagon, hexagon and the ribbon relation. They are never generators.
//!
//! Planar tensor workflows need a tensor engine and are out of scope here. A/B symbols are checked in
//! `tests/ab_symbols.rs`.

use fusion_symbols::category::{BraidingStyle, FusionCategory};
use std::collections::HashMap;

use fusion_symbols::table::{TableCategory, TableObject};
use num_complex::Complex64;

/// Documented absolute bound for every approximate comparison: a fixed
/// multiple of the unit round-off covering the at most rank-4 sums of triple
/// products of unit-modulus coefficients in the pentagon. Not per-category
/// tunable: a category that needs its own tolerance has an import bug.
const FP_BOUND: f64 = 32.0 * f64::EPSILON;

fn table() -> TableCategory {
    TableCategory::fibonacci()
}

fn shipped() -> impl Iterator<Item = TableCategory> {
    TableCategory::shipped_uuids().map(|uuid| TableCategory::by_uuid(uuid).expect("shipped"))
}

/// Exact comparison for entries pinned bitwise.
fn assert_bits(actual: Complex64, expected: Complex64, what: &str) {
    assert_eq!(
        (actual.re.to_bits(), actual.im.to_bits()),
        (expected.re.to_bits(), expected.im.to_bits()),
        "{what}: {actual} != {expected}"
    );
}

// -------------------------------------------------------------------------
// 1. Gauge equivalence with TensorKitSectors 0.3.9
// -------------------------------------------------------------------------

/// One TensorKitSectors category, one-based `values(I)` labels.
struct Tks {
    rank: u16,
    n: HashMap<[u16; 3], u32>,
    f: HashMap<[u16; 6], f64>,
    r: HashMap<[u16; 3], Complex64>,
    dim: HashMap<u16, f64>,
    fs: HashMap<u16, f64>,
    twist: HashMap<u16, Complex64>,
}

/// Tabulates closed-form `N`, `F`, `R` and `dim` on labels `1..=rank` (unit
/// `1`, every label self-dual as in both anyon models), and derives
/// `frobenius_schur_phase(a) = sign(F^{a a a}_a[1, 1])` and
/// `twist_from_Rsymbol(a) = Σ_b d_b / d_a R^{aa}_b`.
fn tks(
    rank: u16,
    n: impl Fn(u16, u16, u16) -> bool,
    f: impl Fn([u16; 6]) -> f64,
    r: impl Fn(u16, u16, u16) -> Complex64,
    dim: impl Fn(u16) -> f64,
) -> Tks {
    let labels: Vec<u16> = (1..=rank).collect();
    let mut tks = Tks {
        rank,
        n: HashMap::new(),
        f: HashMap::new(),
        r: HashMap::new(),
        dim: HashMap::new(),
        fs: HashMap::new(),
        twist: HashMap::new(),
    };
    for &a in &labels {
        for &b in &labels {
            for &c in &labels {
                if !n(a, b, c) {
                    continue;
                }
                tks.n.insert([a, b, c], 1);
                tks.r.insert([a, b, c], r(a, b, c));
            }
        }
    }
    let nn = |a, b, c| tks.n.contains_key(&[a, b, c]);
    for &a in &labels {
        for &b in &labels {
            for &c in &labels {
                for &d in &labels {
                    for &e in &labels {
                        for &x in &labels {
                            if nn(a, b, e) && nn(e, c, d) && nn(b, c, x) && nn(a, x, d) {
                                tks.f.insert([a, b, c, d, e, x], f([a, b, c, d, e, x]));
                            }
                        }
                    }
                }
            }
        }
    }
    for &a in &labels {
        tks.dim.insert(a, dim(a));
        tks.fs.insert(a, tks.f[&[a, a, a, a, 1, 1]].signum());
        let twist = labels
            .iter()
            .filter(|&&b| nn(a, a, b))
            .map(|&b| tks.r[&[a, a, b]] * (dim(b) / dim(a)))
            .sum();
        tks.twist.insert(a, twist);
    }
    tks
}

const PHI: f64 = std::f64::consts::GOLDEN_RATIO;

/// `FibonacciAnyon`: labels `(I, τ) = (1, 2)`.
fn tks_fibonacci() -> Tks {
    let unit = |a| a == 1;
    tks(
        2,
        |a, b, c| u8::from(unit(a)) + u8::from(unit(b)) + u8::from(unit(c)) != 2,
        |[a, b, c, d, e, f]| match ([a, b, c, d] == [2; 4], e, f) {
            (true, 1, 1) => 1.0 / PHI,
            (true, 2, 2) => -1.0 / PHI,
            (true, _, _) => 1.0 / PHI.sqrt(),
            _ => 1.0,
        },
        |a, b, c| match (unit(a) || unit(b), c) {
            (true, _) => Complex64::new(1.0, 0.0),
            (false, 1) => Complex64::cis(0.8 * std::f64::consts::PI),
            (false, _) => Complex64::cis(-0.6 * std::f64::consts::PI),
        },
        |a| if unit(a) { 1.0 } else { PHI },
    )
}

/// `IsingAnyon`: labels `(I, σ, ψ) = (1, 2, 3)`.
fn tks_ising() -> Tks {
    let (i, s, p) = (1, 2, 3);
    tks(
        3,
        |a, b, c| {
            (a == i && b == c)
                || (b == i && a == c)
                || (c == i && a == b)
                || [a, b, c] == [s, s, p]
                || [a, b, c] == [s, p, s]
                || [a, b, c] == [p, s, s]
        },
        |[a, b, c, d, e, f]| {
            if [a, b, c, d] == [s; 4] {
                if e == p && f == p {
                    -1.0 / 2.0f64.sqrt()
                } else {
                    1.0 / 2.0f64.sqrt()
                }
            } else if e == s
                && f == s
                && ([a, b, c, d] == [s, p, s, p] || [a, b, c, d] == [p, s, p, s])
            {
                -1.0
            } else {
                1.0
            }
        },
        |a, b, c| {
            use std::f64::consts::PI;
            match (a, b, c) {
                (2, 2, 1) => Complex64::cis(-PI / 8.0),
                (3, 3, 1) => Complex64::new(-1.0, 0.0),
                (2, 3, 2) | (3, 2, 2) => Complex64::new(0.0, -1.0),
                (2, 2, 3) => Complex64::cis(3.0 * PI / 8.0),
                _ => Complex64::new(1.0, 0.0),
            }
        },
        |a| if a == s { 2.0f64.sqrt() } else { 1.0 },
    )
}

/// Checks that `category` is `tks` under the object map `map` (TKS label
/// `i` is `map[i - 1]` here) and the vertex gauge `u` (on TKS labels, unit
/// vertices `1`): `N` exactly; within `FP_BOUND`, on every tuple,
/// `F_here = F_TKS · u^{ab}_e u^{ec}_d / (u^{bc}_f u^{af}_d)` and
/// `R_here = R_TKS · u^{ab}_c / u^{ba}_c`; and the gauge invariants dim, FS
/// phase and twist equal. A witness that passes proves gauge equivalence;
/// it is not searched for.
fn gauge_equivalent(
    category: &TableCategory,
    tks: &Tks,
    map: &[u16],
    u: impl Fn(u16, u16, u16) -> f64,
) -> Result<(), String> {
    assert_eq!(category.object_count(), Ok(Some(u64::from(tks.rank))));
    let labels: Vec<u16> = (1..=tks.rank).collect();
    let here = |a: u16| category.object(map[usize::from(a) - 1]).unwrap();
    let close = |what: String, got: Complex64, want: Complex64| {
        if (got - want).norm() <= FP_BOUND {
            Ok(())
        } else {
            Err(format!("{what}: {got} vs {want}"))
        }
    };
    for &a in &labels {
        close(
            format!("dim {a}"),
            category.dim(&here(a)).into(),
            tks.dim[&a].into(),
        )?;
        close(
            format!("fs {a}"),
            category.frobenius_schur_phase(&here(a)).unwrap(),
            tks.fs[&a].into(),
        )?;
        close(
            format!("twist {a}"),
            category.twist(&here(a)),
            tks.twist[&a],
        )?;
        for &b in &labels {
            for &c in &labels {
                let n = tks.n.get(&[a, b, c]).copied().unwrap_or(0);
                if category.nsymbol(&here(a), &here(b), &here(c)).unwrap() != n {
                    return Err(format!("N {a} {b} {c}"));
                }
                let r_tks = tks.r.get(&[a, b, c]).copied().unwrap_or_default();
                let gauge = if n == 0 { 1.0 } else { u(a, b, c) / u(b, a, c) };
                close(
                    format!("R {a} {b} {c}"),
                    category.r_symbol(&here(a), &here(b), &here(c)).unwrap(),
                    r_tks * gauge,
                )?;
                for &d in &labels {
                    for &e in &labels {
                        for &f in &labels {
                            let f_tks = tks.f.get(&[a, b, c, d, e, f]).copied().unwrap_or(0.0);
                            let gauge = if f_tks == 0.0 {
                                1.0
                            } else {
                                u(a, b, e) * u(e, c, d) / (u(b, c, f) * u(a, f, d))
                            };
                            close(
                                format!("F {a} {b} {c} {d} {e} {f}"),
                                category
                                    .f_symbol(
                                        &here(a),
                                        &here(b),
                                        &here(c),
                                        &here(d),
                                        &here(e),
                                        &here(f),
                                    )
                                    .unwrap(),
                                (f_tks * gauge).into(),
                            )?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn fibonacci_is_tensorkitsectors_fibonacci_in_the_same_gauge() {
    let fib = TableCategory::fibonacci();
    let tks = tks_fibonacci();
    gauge_equivalent(&fib, &tks, &[1, 2], |_, _, _| 1.0).expect("identity gauge");
    // Negative control: the mirror braiding is not TensorKitSectors'.
    let mirror = TableCategory::by_code([2, 1, 0, 2, 1, 1, 1]).expect("shipped");
    assert!(gauge_equivalent(&mirror, &tks, &[1, 2], |_, _, _| 1.0).is_err());
    for a in [fib.object(1).unwrap(), fib.object(2).unwrap()] {
        assert_eq!(fib.dual(&a), a);
    }
}

#[test]
fn fibonacci_entries_are_the_correctly_rounded_exact_values() {
    // Checked against AnyonWiki badbfe2 (exact symbols `1/φ`, `1/√φ`,
    // `e^{4πi/5}`, `e^{-3πi/5}`) and a 60-digit decimal evaluation: the
    // nearest binary64 values. `1.0 / PHI` in f64 is one ulp below the first.
    let fib = table();
    let tau = fib.object(2).unwrap();
    let unit = fib.object(1).unwrap();
    let real = |re| Complex64::new(re, 0.0);
    assert_bits(
        fib.f_symbol(&tau, &tau, &tau, &tau, &unit, &unit).unwrap(),
        real(0.618_033_988_749_894_9),
        "F[1,1]",
    );
    assert_bits(
        fib.f_symbol(&tau, &tau, &tau, &tau, &unit, &tau).unwrap(),
        real(0.786_151_377_757_423_3),
        "F[1,τ]",
    );
    assert_bits(
        fib.f_symbol(&tau, &tau, &tau, &tau, &tau, &tau).unwrap(),
        real(-0.618_033_988_749_894_9),
        "F[τ,τ]",
    );
    assert_bits(
        fib.r_symbol(&tau, &tau, &unit).unwrap(),
        Complex64::new(-0.809_016_994_374_947_5, 0.587_785_252_292_473_1),
        "R^{ττ}_1",
    );
    assert_bits(
        fib.r_symbol(&tau, &tau, &tau).unwrap(),
        Complex64::new(-0.309_016_994_374_947_45, -0.951_056_516_295_153_5),
        "R^{ττ}_τ",
    );
    assert!((fib.dim(&tau) - PHI).abs() <= FP_BOUND);
}

#[test]
fn ising_is_tensorkitsectors_ising_up_to_the_documented_vertex_gauge() {
    let ising = TableCategory::ising();
    let tks = tks_ising();
    // TensorKitSectors (I, σ, ψ) = 1, 2, 3; here 1, ψ = 2, σ = 3.
    let map = [1, 3, 2];
    let (sigma, psi) = (2, 3);
    let documented = |a, b, c| {
        if (a, b, c) == (sigma, sigma, psi) {
            -1.0
        } else {
            1.0
        }
    };
    gauge_equivalent(&ising, &tks, &map, documented).expect("documented gauge");
    // Negative controls: not entrywise equal, and the mirror braiding is not
    // equivalent under the same witness.
    assert!(gauge_equivalent(&ising, &tks, &map, |_, _, _| 1.0).is_err());
    let mirror = TableCategory::by_code([3, 1, 0, 1, 1, 4, 1]).expect("shipped");
    assert!(gauge_equivalent(&mirror, &tks, &map, documented).is_err());
}

// -------------------------------------------------------------------------
// 2. Consistency checks — never generators — on every shipped entry
// -------------------------------------------------------------------------

fn channels(p: &TableCategory, a: TableObject, b: TableObject) -> Vec<TableObject> {
    p.fusion(&a, &b).unwrap().collect()
}

#[test]
fn f_blocks_are_unitary() {
    for p in shipped() {
        let name = p.uuid();
        let objects: Vec<TableObject> = crate::common::objects(&p).collect();
        for &a in &objects {
            for &b in &objects {
                for &c in &objects {
                    for &d in &objects {
                        let rows: Vec<_> = channels(&p, a, b)
                            .into_iter()
                            .filter(|&e| p.nsymbol(&e, &c, &d).unwrap() > 0)
                            .collect();
                        let columns: Vec<_> = channels(&p, b, c)
                            .into_iter()
                            .filter(|&f| p.nsymbol(&a, &f, &d).unwrap() > 0)
                            .collect();
                        assert_eq!(rows.len(), columns.len(), "{name}: F block not square");
                        for (i, &f1) in columns.iter().enumerate() {
                            for (j, &f2) in columns.iter().enumerate() {
                                let entry: Complex64 = rows
                                    .iter()
                                    .map(|&e| {
                                        p.f_symbol(&a, &b, &c, &d, &e, &f1).unwrap().conj()
                                            * p.f_symbol(&a, &b, &c, &d, &e, &f2).unwrap()
                                    })
                                    .sum();
                                let expected = Complex64::new(f64::from(u8::from(i == j)), 0.0);
                                assert!(
                                    (entry - expected).norm() <= FP_BOUND,
                                    "{name}: F block {:?} not unitary at [{i},{j}]: {entry}",
                                    (a, b, c, d)
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn r_symbols_are_unit_modulus_on_admissible_channels() {
    for p in shipped().filter(|cat| cat.braiding_style() != BraidingStyle::None) {
        let name = p.uuid();
        let objects: Vec<TableObject> = crate::common::objects(&p).collect();
        for &a in &objects {
            for &b in &objects {
                for c in channels(&p, a, b) {
                    let modulus = p.r_symbol(&a, &b, &c).unwrap().norm();
                    assert!(
                        (modulus - 1.0).abs() <= FP_BOUND,
                        "{name}: |R{:?}| = {modulus}",
                        (a, b, c)
                    );
                }
            }
        }
    }
}

#[test]
fn unit_associators_are_bitwise_one() {
    // The unitor associators of TensorKitSectors' `triangle_equation`,
    // asserted bitwise: consumers drop these factors as exact identities.
    let one = Complex64::new(1.0, 0.0);
    for p in shipped() {
        let unit = p.unit();
        let name = p.uuid();
        let objects: Vec<TableObject> = crate::common::objects(&p).collect();
        for &a in &objects {
            for &b in &objects {
                for c in channels(&p, a, b) {
                    for (which, value) in [
                        (
                            "left unitor",
                            p.f_symbol(&unit, &a, &b, &c, &a, &c).unwrap(),
                        ),
                        (
                            "middle unitor",
                            p.f_symbol(&a, &unit, &b, &c, &a, &b).unwrap(),
                        ),
                        (
                            "right unitor",
                            p.f_symbol(&a, &b, &unit, &c, &c, &b).unwrap(),
                        ),
                    ] {
                        assert_bits(value, one, &format!("{name} {which}{:?}", (a, b, c)));
                    }
                }
            }
        }
    }
}

// Every pentagon instance (a, …, i): 74 entries of rank <= 4 by default, all
// 377 with `anyonwiki-unitary`.
const PENTAGON_CASES: usize = if cfg!(feature = "anyonwiki-unitary") {
    5_243_551
} else {
    55_043
};

#[test]
fn pentagon_equation_holds() {
    // Structure of TensorKitSectors `pentagon_equation`, multiplicity-free
    // branch, read as a residual check.
    let mut total = 0usize;
    for p in shipped() {
        let name = p.uuid();
        let objects: Vec<TableObject> = crate::common::objects(&p).collect();
        let mut cases = 0usize;
        for &a in &objects {
            for &b in &objects {
                for &c in &objects {
                    for &d in &objects {
                        for f in channels(&p, a, b) {
                            for h in channels(&p, c, d) {
                                for g in channels(&p, f, c) {
                                    for i in channels(&p, b, h) {
                                        for &e in &objects {
                                            if p.nsymbol(&g, &d, &e).unwrap() == 0
                                                || p.nsymbol(&a, &i, &e).unwrap() == 0
                                            {
                                                continue;
                                            }
                                            let lhs = p.f_symbol(&f, &c, &d, &e, &g, &h).unwrap()
                                                * p.f_symbol(&a, &b, &h, &e, &f, &i).unwrap();
                                            let rhs: Complex64 = channels(&p, b, c)
                                                .into_iter()
                                                .map(|j| {
                                                    p.f_symbol(&a, &b, &c, &g, &f, &j).unwrap()
                                                        * p.f_symbol(&a, &j, &d, &e, &g, &i)
                                                            .unwrap()
                                                        * p.f_symbol(&b, &c, &d, &i, &j, &h)
                                                            .unwrap()
                                                })
                                                .sum();
                                            assert!(
                                                (lhs - rhs).norm() <= FP_BOUND,
                                                "{name}: pentagon{:?}: {lhs} vs {rhs}",
                                                (a, b, c, d, e, f, g, h, i)
                                            );
                                            cases += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(cases > 0, "{name}: the pentagon loop never ran");
        total += cases;
    }
    assert_eq!(total, PENTAGON_CASES);
}

#[test]
fn both_hexagon_equations_hold() {
    // Structure of TensorKitSectors `hexagon_equation`, multiplicity-free
    // branch. `conj(R)` stands for `inv(R)`, valid because R is unitary
    // (asserted independently above).
    for p in shipped().filter(|cat| cat.braiding_style() != BraidingStyle::None) {
        let name = p.uuid();
        let objects: Vec<TableObject> = crate::common::objects(&p).collect();
        let mut cases = 0usize;
        for &a in &objects {
            for &b in &objects {
                for &c in &objects {
                    for e in channels(&p, c, a) {
                        for f in channels(&p, c, b) {
                            for &d in &objects {
                                if p.nsymbol(&e, &b, &d).unwrap() == 0
                                    || p.nsymbol(&a, &f, &d).unwrap() == 0
                                {
                                    continue;
                                }
                                let facbdef = p.f_symbol(&a, &c, &b, &d, &e, &f).unwrap();
                                let over = p.r_symbol(&c, &a, &e).unwrap()
                                    * facbdef
                                    * p.r_symbol(&c, &b, &f).unwrap();
                                let under = p.r_symbol(&a, &c, &e).unwrap().conj()
                                    * facbdef
                                    * p.r_symbol(&b, &c, &f).unwrap().conj();
                                let mut path_over = Complex64::new(0.0, 0.0);
                                let mut path_under = Complex64::new(0.0, 0.0);
                                for g in channels(&p, a, b) {
                                    let fcabdeg = p.f_symbol(&c, &a, &b, &d, &e, &g).unwrap();
                                    let fabcdgf = p.f_symbol(&a, &b, &c, &d, &g, &f).unwrap();
                                    path_over +=
                                        fcabdeg * p.r_symbol(&c, &g, &d).unwrap() * fabcdgf;
                                    path_under +=
                                        fcabdeg * p.r_symbol(&g, &c, &d).unwrap().conj() * fabcdgf;
                                }
                                assert!(
                                    (over - path_over).norm() <= FP_BOUND,
                                    "{name}: hexagon (over){:?}",
                                    (a, b, c, d, e, f)
                                );
                                assert!(
                                    (under - path_under).norm() <= FP_BOUND,
                                    "{name}: hexagon (under){:?}",
                                    (a, b, c, d, e, f)
                                );
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(cases > 0, "{name}: the hexagon loop never ran");
    }
}

#[test]
fn ribbon_relation_ties_twists_to_the_braid() {
    // θ_c = θ_a θ_b R^{ba}_c R^{ab}_c on every admissible channel.
    for p in shipped().filter(|cat| cat.braiding_style() != BraidingStyle::None) {
        let name = p.uuid();
        let objects: Vec<TableObject> = crate::common::objects(&p).collect();
        for &a in &objects {
            for &b in &objects {
                for c in channels(&p, a, b) {
                    let actual = p.twist(&a)
                        * p.twist(&b)
                        * p.r_symbol(&b, &a, &c).unwrap()
                        * p.r_symbol(&a, &b, &c).unwrap();
                    assert!(
                        (actual - p.twist(&c)).norm() <= FP_BOUND,
                        "{name}: ribbon{:?}",
                        (a, b, c)
                    );
                }
            }
        }
    }
}

// -------------------------------------------------------------------------
// Provider contract
// -------------------------------------------------------------------------
#[test]
fn objects_are_validated_and_symbols_on_absent_channels_are_zero() {
    use fusion_symbols::table::TableError;
    let fib = table();
    for index in [0, 3, u16::MAX] {
        assert_eq!(
            fib.object(index),
            Err(TableError::UnknownObject { object: index })
        );
    }
    assert_eq!(
        crate::common::objects(&fib)
            .map(|a| a.index())
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    let (unit, tau) = (fib.unit(), fib.object(2).unwrap());
    assert_eq!(
        fib.f_symbol(&tau, &tau, &unit, &unit, &tau, &tau).unwrap(),
        Complex64::new(0.0, 0.0)
    );
    assert_eq!(
        fib.r_symbol(&unit, &unit, &tau).unwrap(),
        Complex64::new(0.0, 0.0)
    );
}

#[test]
fn foreign_objects_panic_before_capability_checks_in_every_argument() {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    let fib = table();
    let foreign = fib.unit();
    let fusion_only = TableCategory::by_code([3, 1, 0, 2, 2, 0, 1]).unwrap();
    for cat in [TableCategory::ising(), fusion_only] {
        let ok = cat.unit();
        let rejects = |call: &dyn Fn()| {
            let payload =
                catch_unwind(AssertUnwindSafe(call)).expect_err("foreign object must panic");
            let message = payload.downcast_ref::<String>().unwrap();
            assert!(message.contains("foreign table object"), "{message}");
        };
        for slot in 0..6 {
            let mut labels = [ok; 6];
            labels[slot] = foreign;
            let [a, b, c, d, e, f] = labels;
            rejects(&|| {
                let _ = cat.f_symbol(&a, &b, &c, &d, &e, &f);
            });
        }
        for slot in 0..3 {
            let mut labels = [ok; 3];
            labels[slot] = foreign;
            let [a, b, c] = labels;
            rejects(&|| {
                let _ = cat.r_symbol(&a, &b, &c);
            });
            rejects(&|| {
                cat.nsymbol(&a, &b, &c).unwrap();
            });
        }
        for slot in 0..2 {
            let mut labels = [ok; 2];
            labels[slot] = foreign;
            rejects(&|| {
                let _ = cat.fusion(&labels[0], &labels[1]);
            });
        }
        rejects(&|| {
            cat.dim(&foreign);
        });
        rejects(&|| {
            cat.dual(&foreign);
        });
        rejects(&|| {
            cat.frobenius_schur_phase(&foreign).unwrap();
        });
        rejects(&|| {
            cat.frobenius_schur_indicator(&foreign);
        });
        rejects(&|| {
            cat.twist(&foreign);
        });
    }
}

#[test]
fn fusion_only_entries_refuse_braiding() {
    use fusion_symbols::table::TableError;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    for cat in shipped().filter(|cat| cat.braiding_style() == BraidingStyle::None) {
        let unit = cat.unit();
        assert_eq!(
            cat.r_symbol(&unit, &unit, &unit),
            Err(TableError::NoBraiding)
        );
        assert!(catch_unwind(AssertUnwindSafe(|| cat.twist(&unit))).is_err());
        assert_eq!(
            cat.f_symbol(&unit, &unit, &unit, &unit, &unit, &unit)
                .unwrap(),
            Complex64::new(1.0, 0.0)
        );
    }
}
