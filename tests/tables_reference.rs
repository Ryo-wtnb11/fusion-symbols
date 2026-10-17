//! Reference tests for the `tables` catalog.
//!
//! Three independent kinds of check, deliberately kept apart:
//!
//! 1. **Oracle comparison against a pinned Julia environment.** Every `N`,
//!    `F` and `R` entry and every derived value of Fibonacci is compared with
//!    `fixtures/fib-categorydata-v0.1.3.txt`, emitted by CategoryData v0.3.6 /
//!    TensorKitSectors v0.3.9 (`fixtures/generate_fixture.jl`). CategoryData.jl
//!    repackages the same category (`smallRankUnitaryFusionData`), so this is
//!    an independent cross-check of the AnyonWiki-sourced tables
//!    (`tests/tables_anyonwiki.rs` checks every entry against the authority).
//! 2. **Gauge equivalence with TensorKitSectors' closed forms.**
//!    `fixtures/tks-0.3.9-anyons-values.txt` (`fixtures/tks_anyons_oracle.jl`)
//!    holds `FibonacciAnyon` and `IsingAnyon` as TensorKitSectors 0.3.9
//!    computes them. Each named constructor must equal them under a stated
//!    object map and vertex gauge, the documented one, and not under the
//!    wrong gauge or the mirror braiding.
//! 3. **Consistency checks** over every shipped entry: unitarity, triangle,
//!    pentagon, hexagon and the ribbon relation. They are never generators.
//!
//! The planar tensor workflow of the original symtn file needs a tensor
//! engine and stays with the consumer. The A/B symbols are derived by the
//! consumer and are not part of this crate.

use std::collections::HashMap;

use fusion_symbols::tables::{TableCategory, TableError, TableObject};
use num_complex::Complex64;

const FIXTURE: &str = include_str!("fixtures/fib-categorydata-v0.1.3.txt");

const TKS: &str = include_str!("fixtures/tks-0.3.9-anyons-values.txt");

const OBJECTS: [TableObject; 2] = [TableObject(1), TableObject(2)];

/// Documented absolute bound for every approximate comparison: a fixed
/// multiple of the unit round-off covering the at most rank-4 sums of triple
/// products of unit-modulus coefficients in the pentagon. Not per-category
/// tunable: a category that needs its own tolerance has an import bug.
const FP_BOUND: f64 = 32.0 * f64::EPSILON;

/// The workspace tolerance rule of the original tests,
/// `32 * sqrt(terms) * eps * max(1, |want|)`.
fn assert_close(what: &str, got: Complex64, want: Complex64, terms: usize) {
    let bound = 32.0 * (terms.max(1) as f64).sqrt() * f64::EPSILON * want.norm().max(1.0);
    let error = (got - want).norm();
    assert!(
        error <= bound,
        "{what}: {got} vs {want} ({error:e} > {bound:e})"
    );
}

fn table() -> TableCategory {
    TableCategory::fibonacci()
}

fn shipped() -> impl Iterator<Item = TableCategory> {
    TableCategory::shipped_uuids().map(|uuid| TableCategory::by_uuid(uuid).expect("shipped"))
}

/// Exact comparison for imported entries: both sides parse the same
/// shortest-round-trip decimals.
fn assert_bits(actual: Complex64, expected: Complex64, what: &str) {
    assert_eq!(
        (actual.re.to_bits(), actual.im.to_bits()),
        (expected.re.to_bits(), expected.im.to_bits()),
        "{what}: {actual} != {expected}"
    );
}

/// `tag` and one-based labels -> value, as emitted by the Julia generator.
fn fixture() -> HashMap<(String, Vec<u16>), Complex64> {
    let mut entries = HashMap::new();
    for line in FIXTURE.lines() {
        if line.trim_start().starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        let tag = fields[0].to_owned();
        let (labels, value) = if matches!(tag.as_str(), "dual" | "N") {
            let (head, tail) = fields[1..].split_at(fields.len() - 2);
            (head, Complex64::new(tail[0].parse().unwrap(), 0.0))
        } else {
            let (head, tail) = fields[1..].split_at(fields.len() - 3);
            (
                head,
                Complex64::new(tail[0].parse().unwrap(), tail[1].parse().unwrap()),
            )
        };
        let key = (tag, labels.iter().map(|f| f.parse().unwrap()).collect());
        assert!(
            entries.insert(key, value).is_none(),
            "duplicate fixture row"
        );
    }
    entries
}

fn fixture_value(
    entries: &HashMap<(String, Vec<u16>), Complex64>,
    tag: &str,
    labels: &[TableObject],
) -> Complex64 {
    let key = (tag.to_owned(), labels.iter().map(|a| a.label()).collect());
    *entries
        .get(&key)
        .unwrap_or_else(|| panic!("fixture is missing {tag}{labels:?}"))
}

// -------------------------------------------------------------------------
// 1. The pinned CategoryData.jl / TensorKitSectors fixture
// -------------------------------------------------------------------------

#[test]
fn every_fusion_and_symbol_entry_matches_the_pinned_julia_environment() {
    let fib = table();
    let entries = fixture();
    let mut checked = 0usize;
    for a in OBJECTS {
        assert_eq!(
            f64::from(fib.dual(a).label()),
            fixture_value(&entries, "dual", &[a]).re
        );
        checked += 1;
        for b in OBJECTS {
            for c in OBJECTS {
                assert_eq!(
                    f64::from(fib.nsymbol(a, b, c)),
                    fixture_value(&entries, "N", &[a, b, c]).re
                );
                assert_bits(
                    fib.r_symbol(a, b, c),
                    fixture_value(&entries, "R", &[a, b, c]),
                    "R",
                );
                checked += 2;
                for d in OBJECTS {
                    for e in OBJECTS {
                        for f in OBJECTS {
                            assert_bits(
                                fib.f_symbol(a, b, c, d, e, f),
                                fixture_value(&entries, "F", &[a, b, c, d, e, f]),
                                "F",
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    // 2 duals + 8 N + 8 R + 64 F: the full rank-2 tables, nothing sampled.
    assert_eq!(checked, 2 + 8 + 8 + 64);
}

#[test]
fn every_derived_value_matches_the_pinned_julia_environment() {
    let fib = table();
    let entries = fixture();
    for a in OBJECTS {
        assert_close(
            "dim",
            fib.dim(a).into(),
            fixture_value(&entries, "dim", &[a]),
            4,
        );
        assert_close(
            "frobenius-schur",
            fib.frobenius_schur(a),
            fixture_value(&entries, "frobeniusschur", &[a]),
            4,
        );
        assert_close(
            "twist",
            fib.twist(a),
            fixture_value(&entries, "twist", &[a]),
            4,
        );
    }
}

// -------------------------------------------------------------------------
// 2. Gauge equivalence with TensorKitSectors 0.3.9
// -------------------------------------------------------------------------

/// One TensorKitSectors category from the fixture, one-based `values(I)`
/// labels.
struct Tks {
    rank: u16,
    n: HashMap<[u16; 3], u32>,
    f: HashMap<[u16; 6], f64>,
    r: HashMap<[u16; 3], Complex64>,
    dim: HashMap<u16, f64>,
    fs: HashMap<u16, f64>,
    twist: HashMap<u16, Complex64>,
}

fn tks(name: &str) -> Tks {
    let bits = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).expect("hex bits"));
    let mut tks = Tks {
        rank: 0,
        n: HashMap::new(),
        f: HashMap::new(),
        r: HashMap::new(),
        dim: HashMap::new(),
        fs: HashMap::new(),
        twist: HashMap::new(),
    };
    for line in TKS.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f[0] != name {
            continue;
        }
        let label = |i: usize| f[i].parse::<u16>().unwrap();
        match f[1] {
            "dual" => tks.rank += 1,
            "dim" => drop(tks.dim.insert(label(2), bits(f[3]))),
            "fs" => drop(tks.fs.insert(label(2), bits(f[3]))),
            "twist" => drop(
                tks.twist
                    .insert(label(2), Complex64::new(bits(f[3]), bits(f[4]))),
            ),
            "N" => drop(
                tks.n
                    .insert([label(2), label(3), label(4)], f[5].parse().unwrap()),
            ),
            "R" => drop(tks.r.insert(
                [label(2), label(3), label(4)],
                Complex64::new(bits(f[5]), bits(f[6])),
            )),
            "F" => drop(tks.f.insert(
                [label(2), label(3), label(4), label(5), label(6), label(7)],
                bits(f[8]),
            )),
            other => panic!("unknown fixture row {other}"),
        }
    }
    assert!(tks.rank > 0, "fixture has no {name}");
    tks
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
    assert_eq!(category.rank(), usize::from(tks.rank));
    let labels: Vec<u16> = (1..=tks.rank).collect();
    let here = |a: u16| TableObject(map[usize::from(a) - 1]);
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
            category.dim(here(a)).into(),
            tks.dim[&a].into(),
        )?;
        close(
            format!("fs {a}"),
            category.frobenius_schur(here(a)),
            tks.fs[&a].into(),
        )?;
        close(format!("twist {a}"), category.twist(here(a)), tks.twist[&a])?;
        for &b in &labels {
            for &c in &labels {
                let n = tks.n.get(&[a, b, c]).copied().unwrap_or(0);
                if category.nsymbol(here(a), here(b), here(c)) != n {
                    return Err(format!("N {a} {b} {c}"));
                }
                let r_tks = tks.r.get(&[a, b, c]).copied().unwrap_or_default();
                let gauge = if n == 0 { 1.0 } else { u(a, b, c) / u(b, a, c) };
                close(
                    format!("R {a} {b} {c}"),
                    category.r_symbol(here(a), here(b), here(c)),
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
                                category.f_symbol(
                                    here(a),
                                    here(b),
                                    here(c),
                                    here(d),
                                    here(e),
                                    here(f),
                                ),
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

/// Signed distance in representable `f64`s.
fn ulps(a: f64, b: f64) -> u64 {
    let key = |x: f64| {
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    };
    key(a).abs_diff(key(b))
}

#[test]
fn fibonacci_is_tensorkitsectors_fibonacci_in_the_same_gauge() {
    let fib = TableCategory::fibonacci();
    let tks = tks("fib");
    gauge_equivalent(&fib, &tks, &[1, 2], |_, _, _| 1.0).expect("identity gauge");
    // Negative control: the mirror braiding is not TensorKitSectors'.
    let mirror = TableCategory::by_code([2, 1, 0, 2, 1, 1, 1]).expect("shipped");
    assert!(gauge_equivalent(&mirror, &tks, &[1, 2], |_, _, _| 1.0).is_err());

    // Entrywise, the correctly rounded table and TensorKitSectors' own f64
    // arithmetic (`1/φ`, `cispi`) differ by exactly these ulp distances
    // (re, im); every other F and R entry and the unit's data are bitwise
    // equal.
    let tau = TableObject(2);
    let mut distances = Vec::new();
    for (labels, &want) in &tks.f {
        let l = labels.map(TableObject);
        let got = fib.f_symbol(l[0], l[1], l[2], l[3], l[4], l[5]);
        assert_eq!(got.im, 0.0);
        let d = ulps(got.re, want);
        if d != 0 {
            distances.push((format!("F{labels:?}"), d, 0));
        }
    }
    for (labels, &want) in &tks.r {
        let l = labels.map(TableObject);
        let got = fib.r_symbol(l[0], l[1], l[2]);
        let d = (ulps(got.re, want.re), ulps(got.im, want.im));
        if d != (0, 0) {
            distances.push((format!("R{labels:?}"), d.0, d.1));
        }
    }
    let twist = fib.twist(tau);
    distances.push((
        "twist".into(),
        ulps(twist.re, tks.twist[&2].re),
        ulps(twist.im, tks.twist[&2].im),
    ));
    distances.push(("dim".into(), ulps(fib.dim(tau), tks.dim[&2]), 0));
    distances.sort();
    assert_eq!(
        distances,
        PINNED_FIBONACCI_DISTANCES
            .iter()
            .map(|&(what, re, im)| (what.to_owned(), re, im))
            .collect::<Vec<_>>()
    );
}

const PINNED_FIBONACCI_DISTANCES: [(&str, u64, u64); 6] = [
    ("F[2, 2, 2, 2, 1, 1]", 1, 0),
    ("F[2, 2, 2, 2, 2, 2]", 1, 0),
    ("R[2, 2, 1]", 1, 1),
    ("R[2, 2, 2]", 2, 1),
    ("dim", 1, 0),
    ("twist", 2, 3),
];

#[test]
fn ising_is_tensorkitsectors_ising_up_to_the_documented_vertex_gauge() {
    let ising = TableCategory::ising();
    let tks = tks("ising");
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
// 3. Consistency checks — never generators — on every shipped entry
// -------------------------------------------------------------------------

fn channels(p: &TableCategory, a: TableObject, b: TableObject) -> Vec<TableObject> {
    p.fusion(a, b).collect()
}

#[test]
fn f_blocks_are_unitary() {
    for p in shipped() {
        let name = p.uuid();
        let objects: Vec<TableObject> = p.objects().collect();
        for &a in &objects {
            for &b in &objects {
                for &c in &objects {
                    for &d in &objects {
                        let rows: Vec<_> = channels(&p, a, b)
                            .into_iter()
                            .filter(|&e| p.nsymbol(e, c, d) > 0)
                            .collect();
                        let columns: Vec<_> = channels(&p, b, c)
                            .into_iter()
                            .filter(|&f| p.nsymbol(a, f, d) > 0)
                            .collect();
                        assert_eq!(rows.len(), columns.len(), "{name}: F block not square");
                        for (i, &f1) in columns.iter().enumerate() {
                            for (j, &f2) in columns.iter().enumerate() {
                                let entry: Complex64 = rows
                                    .iter()
                                    .map(|&e| {
                                        p.f_symbol(a, b, c, d, e, f1).conj()
                                            * p.f_symbol(a, b, c, d, e, f2)
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
    for p in shipped().filter(TableCategory::is_braided) {
        let name = p.uuid();
        let objects: Vec<TableObject> = p.objects().collect();
        for &a in &objects {
            for &b in &objects {
                for c in channels(&p, a, b) {
                    let modulus = p.r_symbol(a, b, c).norm();
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
    let unit = TableObject(1);
    let one = Complex64::new(1.0, 0.0);
    for p in shipped() {
        let name = p.uuid();
        let objects: Vec<TableObject> = p.objects().collect();
        for &a in &objects {
            for &b in &objects {
                for c in channels(&p, a, b) {
                    for (which, value) in [
                        ("left unitor", p.f_symbol(unit, a, b, c, a, c)),
                        ("middle unitor", p.f_symbol(a, unit, b, c, a, b)),
                        ("right unitor", p.f_symbol(a, b, unit, c, c, b)),
                    ] {
                        assert_bits(value, one, &format!("{name} {which}{:?}", (a, b, c)));
                    }
                }
            }
        }
    }
}

const PENTAGON_CASES: usize = 55_043;

#[test]
fn pentagon_equation_holds() {
    // Structure of TensorKitSectors `pentagon_equation`, multiplicity-free
    // branch, read as a residual check.
    let mut total = 0usize;
    for p in shipped() {
        let name = p.uuid();
        let objects: Vec<TableObject> = p.objects().collect();
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
                                            if p.nsymbol(g, d, e) == 0 || p.nsymbol(a, i, e) == 0 {
                                                continue;
                                            }
                                            let lhs = p.f_symbol(f, c, d, e, g, h)
                                                * p.f_symbol(a, b, h, e, f, i);
                                            let rhs: Complex64 = channels(&p, b, c)
                                                .into_iter()
                                                .map(|j| {
                                                    p.f_symbol(a, b, c, g, f, j)
                                                        * p.f_symbol(a, j, d, e, g, i)
                                                        * p.f_symbol(b, c, d, i, j, h)
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
    // Every pentagon instance (a, …, i) of the 74 shipped entries.
    assert_eq!(total, PENTAGON_CASES);
}

#[test]
fn both_hexagon_equations_hold() {
    // Structure of TensorKitSectors `hexagon_equation`, multiplicity-free
    // branch. `conj(R)` stands for `inv(R)`, valid because R is unitary
    // (asserted independently above).
    for p in shipped().filter(TableCategory::is_braided) {
        let name = p.uuid();
        let objects: Vec<TableObject> = p.objects().collect();
        let mut cases = 0usize;
        for &a in &objects {
            for &b in &objects {
                for &c in &objects {
                    for e in channels(&p, c, a) {
                        for f in channels(&p, c, b) {
                            for &d in &objects {
                                if p.nsymbol(e, b, d) == 0 || p.nsymbol(a, f, d) == 0 {
                                    continue;
                                }
                                let facbdef = p.f_symbol(a, c, b, d, e, f);
                                let over = p.r_symbol(c, a, e) * facbdef * p.r_symbol(c, b, f);
                                let under = p.r_symbol(a, c, e).conj()
                                    * facbdef
                                    * p.r_symbol(b, c, f).conj();
                                let mut path_over = Complex64::new(0.0, 0.0);
                                let mut path_under = Complex64::new(0.0, 0.0);
                                for g in channels(&p, a, b) {
                                    let fcabdeg = p.f_symbol(c, a, b, d, e, g);
                                    let fabcdgf = p.f_symbol(a, b, c, d, g, f);
                                    path_over += fcabdeg * p.r_symbol(c, g, d) * fabcdgf;
                                    path_under += fcabdeg * p.r_symbol(g, c, d).conj() * fabcdgf;
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
    for p in shipped().filter(TableCategory::is_braided) {
        let name = p.uuid();
        let objects: Vec<TableObject> = p.objects().collect();
        for &a in &objects {
            for &b in &objects {
                for c in channels(&p, a, b) {
                    let actual =
                        p.twist(a) * p.twist(b) * p.r_symbol(b, a, c) * p.r_symbol(a, b, c);
                    assert!(
                        (actual - p.twist(c)).norm() <= FP_BOUND,
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
fn checked_and_infallible_queries_agree() {
    for p in shipped() {
        let objects: Vec<TableObject> = p.objects().collect();
        for &a in &objects {
            for &b in &objects {
                for &c in &objects {
                    if p.is_braided() {
                        let checked = p.r_symbol_checked(a, b, c);
                        assert_eq!(checked.is_ok(), p.nsymbol(a, b, c) == 1);
                        if let Ok(value) = checked {
                            assert_bits(value, p.r_symbol(a, b, c), "checked R");
                        }
                    }
                    for &d in &objects {
                        for &e in &objects {
                            for &f in &objects {
                                match p.f_symbol_checked(a, b, c, d, e, f) {
                                    Ok(value) => assert_bits(
                                        value,
                                        p.f_symbol(a, b, c, d, e, f),
                                        "checked F",
                                    ),
                                    Err(TableError::NotAdmissible { .. }) => assert_eq!(
                                        p.f_symbol(a, b, c, d, e, f),
                                        Complex64::new(0.0, 0.0)
                                    ),
                                    Err(other) => panic!("unexpected {other}"),
                                }
                            }
                        }
                    }
                }
            }
        }
        if p.is_braided() {
            for &a in &objects {
                assert_bits(
                    p.twist_checked(a).expect("braided"),
                    p.twist(a),
                    "checked twist",
                );
            }
        }
    }
    let fib = table();
    let unit = fib.unit();
    let tau = TableObject(2);
    assert_eq!(
        fib.f_symbol_checked(tau, tau, unit, unit, tau, tau),
        Err(TableError::NotAdmissible {
            left: tau,
            right: unit,
            coupled: unit
        })
    );
    for label in [0u16, 3, u16::MAX] {
        let bad = TableObject(label);
        assert_eq!(
            fib.r_symbol_checked(unit, unit, bad),
            Err(TableError::UnknownObject { object: bad })
        );
        assert_eq!(
            fib.f_symbol_checked(bad, unit, unit, unit, unit, unit),
            Err(TableError::UnknownObject { object: bad })
        );
        assert_eq!(
            fib.twist_checked(bad),
            Err(TableError::UnknownObject { object: bad })
        );
    }
}

/// A category without a braiding has N and F only: R and the twist are a
/// typed capability error, never zeros.
#[test]
fn fusion_only_entries_refuse_r_symbols_and_twists() {
    use std::panic::{self, AssertUnwindSafe};

    let mut fusion_only = 0;
    for p in shipped().filter(|p| !p.is_braided()) {
        let unit = p.unit();
        let any = TableObject(2);
        assert_eq!(
            p.r_symbol_checked(unit, unit, unit),
            Err(TableError::NoBraiding)
        );
        // NoBraiding is reported before any label check.
        assert_eq!(
            p.r_symbol_checked(TableObject(0), unit, unit),
            Err(TableError::NoBraiding)
        );
        assert_eq!(p.twist_checked(any), Err(TableError::NoBraiding));
        for (what, call) in [
            (
                "r_symbol",
                Box::new(|| {
                    let _ = p.r_symbol(unit, unit, unit);
                }) as Box<dyn Fn()>,
            ),
            (
                "twist",
                Box::new(|| {
                    let _ = p.twist(any);
                }),
            ),
        ] {
            let payload = panic::catch_unwind(AssertUnwindSafe(call)).expect_err(what);
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_default();
            assert!(
                message.contains(what) && message.contains(&TableError::NoBraiding.to_string()),
                "{what}: {message:?}"
            );
        }
        // N and F are still there.
        assert_eq!(
            p.f_symbol(unit, unit, unit, unit, unit, unit),
            Complex64::new(1.0, 0.0)
        );
        fusion_only += 1;
    }
    assert_eq!(fusion_only, 16);
}

/// An out-of-domain label must panic with the typed domain error on the
/// infallible surface — never alias another table entry, never fail on a raw
/// slice index.
#[test]
fn infallible_queries_panic_with_the_domain_error_on_unknown_labels() {
    use std::panic::{self, AssertUnwindSafe};

    fn assert_domain_panic<T>(what: &str, bad: TableObject, call: impl FnOnce() -> T) {
        let payload = panic::catch_unwind(AssertUnwindSafe(call))
            .err()
            .unwrap_or_else(|| panic!("{what} returned a value for {bad:?}"));
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        let expected = TableError::UnknownObject { object: bad }.to_string();
        assert!(
            message.contains(what) && message.contains(&expected),
            "{what} on {bad:?} panicked with {message:?}"
        );
    }

    let fib = table();
    let ok = TableObject(2);
    for bad in [0u16, 3, 5, u16::MAX].map(TableObject) {
        for slot in 0..6 {
            let mut ids = [ok; 6];
            ids[slot] = bad;
            let [a, b, c, d, e, f] = ids;
            assert_domain_panic("f_symbol", bad, || fib.f_symbol(a, b, c, d, e, f));
        }
        for slot in 0..3 {
            let mut ids = [ok; 3];
            ids[slot] = bad;
            let [a, b, c] = ids;
            assert_domain_panic("r_symbol", bad, || fib.r_symbol(a, b, c));
            assert_domain_panic("nsymbol", bad, || fib.nsymbol(a, b, c));
        }
        assert_domain_panic("fusion", bad, || fib.fusion(bad, ok).count());
        assert_domain_panic("dim", bad, || fib.dim(bad));
        assert_domain_panic("dual", bad, || fib.dual(bad));
        assert_domain_panic("twist", bad, || fib.twist(bad));
        assert_domain_panic("frobenius_schur", bad, || fib.frobenius_schur(bad));
    }
}
