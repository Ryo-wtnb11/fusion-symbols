//! Reference tests for the Fibonacci providers, ported from symtn's
//! `tenet-category-data/tests/reference.rs` at 957ccc05.
//!
//! Three independent kinds of check, deliberately kept apart:
//!
//! 1. **Oracle comparison against the pinned Julia environment.** Every `N`,
//!    `F` and `R` entry and every derived value of the table provider is
//!    compared with `fixtures/fib-categorydata-v0.1.3.txt`, emitted by
//!    CategoryData v0.3.6 / TensorKitSectors v0.3.9
//!    (`fixtures/generate_fixture.jl`). CategoryData.jl repackages the same
//!    category (`smallRankUnitaryFusionData`), so this is an independent
//!    cross-check of the AnyonWiki-sourced tables (`tests/tables_anyonwiki.rs`
//!    checks them against the authority).
//! 2. **Oracle comparison between the two providers.** The closed form and the
//!    table are independent implementations of one category and must agree to
//!    within a stated bound, while carrying different fingerprints.
//! 3. **Consistency checks.** Unitarity, triangle, pentagon, hexagon and the
//!    ribbon relation, on both providers. They are never generators.
//!
//! The planar tensor workflow of the original file needs a tensor engine and
//! stays with the consumer. The A/B symbols are derived by the consumer and
//! are not part of this crate.

use std::collections::HashMap;

use fusion_symbols::fibonacci::{
    fibonacci_authority_fingerprint, fibonacci_f_symbol, fibonacci_nsymbol, fibonacci_r_symbol,
    FibonacciObject,
};
use fusion_symbols::tables::{FibonacciTable, TableError, TableObject};
use num_complex::Complex64;

const FIXTURE: &str = include_str!("fixtures/fib-categorydata-v0.1.3.txt");

const OBJECTS: [TableObject; 2] = [TableObject(1), TableObject(2)];

/// Documented absolute bound for every approximate comparison: a fixed
/// multiple of the unit round-off covering the at most rank-2 sums of triple
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

fn table() -> FibonacciTable {
    FibonacciTable::try_new().expect("the shipped tables load")
}

fn closed(a: TableObject) -> FibonacciObject {
    match a.label() {
        1 => FibonacciObject::Vacuum,
        2 => FibonacciObject::Tau,
        _ => unreachable!("rank 2"),
    }
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
// 2. The closed-form provider as an independent oracle
// -------------------------------------------------------------------------

#[test]
fn table_and_closed_form_agree_within_the_documented_bound() {
    let fib = table();
    let check = |actual: Complex64, expected: Complex64, what: String| {
        let distance = (actual - expected).norm();
        assert!(
            distance <= FP_BOUND,
            "{what}: {actual} vs {expected} ({distance:e})"
        );
    };
    for a in OBJECTS {
        let ca = closed(a);
        assert_eq!(closed(fib.dual(a)), ca.dual());
        check(fib.dim(a).into(), ca.dim().into(), format!("dim{a:?}"));
        check(fib.twist(a), ca.twist(), format!("twist{a:?}"));
        check(
            fib.frobenius_schur(a),
            ca.frobenius_schur().into(),
            format!("kappa{a:?}"),
        );
        for b in OBJECTS {
            let cb = closed(b);
            assert_eq!(
                fib.fusion(a, b).map(closed).collect::<Vec<_>>(),
                ca.fusion(cb)
            );
            for c in OBJECTS {
                let cc = closed(c);
                assert_eq!(fib.nsymbol(a, b, c), fibonacci_nsymbol(ca, cb, cc));
                check(
                    fib.r_symbol(a, b, c),
                    fibonacci_r_symbol(ca, cb, cc),
                    format!("R{:?}", (a, b, c)),
                );
                for d in OBJECTS {
                    for e in OBJECTS {
                        for f in OBJECTS {
                            check(
                                fib.f_symbol(a, b, c, d, e, f),
                                fibonacci_f_symbol(ca, cb, cc, closed(d), closed(e), closed(f))
                                    .into(),
                                format!("F{:?}", (a, b, c, d, e, f)),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn table_and_closed_form_r_symbols_are_bit_identical() {
    // Both are the correctly rounded binary64 values of the same exact
    // phases (closed form: literals; table: the AnyonWiki projection).
    let fib = table();
    for a in OBJECTS {
        for b in OBJECTS {
            for c in OBJECTS {
                let (t, k) = (
                    fib.r_symbol(a, b, c),
                    fibonacci_r_symbol(closed(a), closed(b), closed(c)),
                );
                assert_eq!(
                    (t.re.to_bits(), t.im.to_bits()),
                    (k.re.to_bits(), k.im.to_bits()),
                    "R{:?}",
                    (a, b, c)
                );
            }
        }
    }
}

#[test]
fn table_and_closed_form_have_different_fingerprints() {
    assert_ne!(
        table().authority_fingerprint(),
        fibonacci_authority_fingerprint()
    );
    assert_eq!(
        table().authority_fingerprint(),
        table().authority_fingerprint()
    );
}

// -------------------------------------------------------------------------
// 3. Consistency checks — never generators — on both providers
// -------------------------------------------------------------------------

/// The provider surface the coherence checks read, over one-based labels.
trait Symbols {
    fn n(&self, a: TableObject, b: TableObject, c: TableObject) -> u32;
    fn f(&self, labels: [TableObject; 6]) -> Complex64;
    fn r(&self, a: TableObject, b: TableObject, c: TableObject) -> Complex64;
    fn twist(&self, a: TableObject) -> Complex64;

    fn channels(&self, a: TableObject, b: TableObject) -> Vec<TableObject> {
        OBJECTS
            .into_iter()
            .filter(|&c| self.n(a, b, c) > 0)
            .collect()
    }
}

impl Symbols for FibonacciTable {
    fn n(&self, a: TableObject, b: TableObject, c: TableObject) -> u32 {
        self.nsymbol(a, b, c)
    }
    fn f(&self, [a, b, c, d, e, f]: [TableObject; 6]) -> Complex64 {
        self.f_symbol(a, b, c, d, e, f)
    }
    fn r(&self, a: TableObject, b: TableObject, c: TableObject) -> Complex64 {
        self.r_symbol(a, b, c)
    }
    fn twist(&self, a: TableObject) -> Complex64 {
        FibonacciTable::twist(self, a)
    }
}

struct Closed;

impl Symbols for Closed {
    fn n(&self, a: TableObject, b: TableObject, c: TableObject) -> u32 {
        fibonacci_nsymbol(closed(a), closed(b), closed(c))
    }
    fn f(&self, labels: [TableObject; 6]) -> Complex64 {
        let [a, b, c, d, e, f] = labels.map(closed);
        fibonacci_f_symbol(a, b, c, d, e, f).into()
    }
    fn r(&self, a: TableObject, b: TableObject, c: TableObject) -> Complex64 {
        fibonacci_r_symbol(closed(a), closed(b), closed(c))
    }
    fn twist(&self, a: TableObject) -> Complex64 {
        closed(a).twist()
    }
}

fn providers() -> [(&'static str, Box<dyn Symbols>); 2] {
    [("table", Box::new(table())), ("closed", Box::new(Closed))]
}

#[test]
fn f_blocks_are_unitary() {
    for (name, p) in providers() {
        for a in OBJECTS {
            for b in OBJECTS {
                for c in OBJECTS {
                    for d in OBJECTS {
                        let rows: Vec<_> = p
                            .channels(a, b)
                            .into_iter()
                            .filter(|&e| p.n(e, c, d) > 0)
                            .collect();
                        let columns: Vec<_> = p
                            .channels(b, c)
                            .into_iter()
                            .filter(|&f| p.n(a, f, d) > 0)
                            .collect();
                        assert_eq!(rows.len(), columns.len(), "{name}: F block not square");
                        for (i, &f1) in columns.iter().enumerate() {
                            for (j, &f2) in columns.iter().enumerate() {
                                let entry: Complex64 = rows
                                    .iter()
                                    .map(|&e| {
                                        p.f([a, b, c, d, e, f1]).conj() * p.f([a, b, c, d, e, f2])
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
    for (name, p) in providers() {
        for a in OBJECTS {
            for b in OBJECTS {
                for c in p.channels(a, b) {
                    let modulus = p.r(a, b, c).norm();
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
    for (name, p) in providers() {
        for a in OBJECTS {
            for b in OBJECTS {
                for c in p.channels(a, b) {
                    for (which, value) in [
                        ("left unitor", p.f([unit, a, b, c, a, c])),
                        ("middle unitor", p.f([a, unit, b, c, a, b])),
                        ("right unitor", p.f([a, b, unit, c, c, b])),
                    ] {
                        assert_bits(value, one, &format!("{name} {which}{:?}", (a, b, c)));
                    }
                }
            }
        }
    }
}

#[test]
fn pentagon_equation_holds() {
    // Structure of TensorKitSectors `pentagon_equation`, multiplicity-free
    // branch, read as a residual check.
    for (name, p) in providers() {
        let mut cases = 0usize;
        for a in OBJECTS {
            for b in OBJECTS {
                for c in OBJECTS {
                    for d in OBJECTS {
                        for f in p.channels(a, b) {
                            for h in p.channels(c, d) {
                                for g in p.channels(f, c) {
                                    for i in p.channels(b, h) {
                                        for e in OBJECTS {
                                            if p.n(g, d, e) == 0 || p.n(a, i, e) == 0 {
                                                continue;
                                            }
                                            let lhs =
                                                p.f([f, c, d, e, g, h]) * p.f([a, b, h, e, f, i]);
                                            let rhs: Complex64 = p
                                                .channels(b, c)
                                                .into_iter()
                                                .map(|j| {
                                                    p.f([a, b, c, g, f, j])
                                                        * p.f([a, j, d, e, g, i])
                                                        * p.f([b, c, d, i, j, h])
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
    }
}

#[test]
fn both_hexagon_equations_hold() {
    // Structure of TensorKitSectors `hexagon_equation`, multiplicity-free
    // branch. `conj(R)` stands for `inv(R)`, valid because R is unitary
    // (asserted independently above).
    for (name, p) in providers() {
        let mut cases = 0usize;
        for a in OBJECTS {
            for b in OBJECTS {
                for c in OBJECTS {
                    for e in p.channels(c, a) {
                        for f in p.channels(c, b) {
                            for d in OBJECTS {
                                if p.n(e, b, d) == 0 || p.n(a, f, d) == 0 {
                                    continue;
                                }
                                let facbdef = p.f([a, c, b, d, e, f]);
                                let over = p.r(c, a, e) * facbdef * p.r(c, b, f);
                                let under = p.r(a, c, e).conj() * facbdef * p.r(b, c, f).conj();
                                let mut path_over = Complex64::new(0.0, 0.0);
                                let mut path_under = Complex64::new(0.0, 0.0);
                                for g in p.channels(a, b) {
                                    let fcabdeg = p.f([c, a, b, d, e, g]);
                                    let fabcdgf = p.f([a, b, c, d, g, f]);
                                    path_over += fcabdeg * p.r(c, g, d) * fabcdgf;
                                    path_under += fcabdeg * p.r(g, c, d).conj() * fabcdgf;
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
    for (name, p) in providers() {
        for a in OBJECTS {
            for b in OBJECTS {
                for c in p.channels(a, b) {
                    let actual = p.twist(a) * p.twist(b) * p.r(b, a, c) * p.r(a, b, c);
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
    let fib = table();
    for a in OBJECTS {
        for b in OBJECTS {
            for c in OBJECTS {
                let checked = fib.r_symbol_checked(a, b, c);
                assert_eq!(checked.is_ok(), fib.nsymbol(a, b, c) == 1);
                if let Ok(value) = checked {
                    assert_bits(value, fib.r_symbol(a, b, c), "checked R");
                }
                for d in OBJECTS {
                    for e in OBJECTS {
                        for f in OBJECTS {
                            match fib.f_symbol_checked(a, b, c, d, e, f) {
                                Ok(value) => {
                                    assert_bits(value, fib.f_symbol(a, b, c, d, e, f), "checked F")
                                }
                                Err(TableError::NotAdmissible { .. }) => assert_eq!(
                                    fib.f_symbol(a, b, c, d, e, f),
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
    }
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
