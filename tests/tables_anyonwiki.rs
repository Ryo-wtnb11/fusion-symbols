//! The shipped `tables` tables against their AnyonWiki authority and
//! the independent smallRankUnitaryFusionData cross-check.
//!
//! `fixtures/anyonwiki-fibonacci-check.txt` is written by
//! `tools/anyonwiki/gen_category_tables.py generate` together with the shipped
//! tables. Each row carries the exact AnyonWiki symbol (`qqb_id`: minimal
//! polynomial coefficients `a0_a1_…_an`, then `__` and the root index), the
//! generator's correctly rounded binary64 value, and AnyonWiki's own
//! `numeric_*` value. For every shipped F- and R-symbol:
//!
//! - (a) the provider's value is bit-equal to the generator output;
//! - (b) it is within 4 ulp of AnyonWiki's `numeric_*` value (those are not
//!   correctly rounded; today's largest gap is 2 ulp);
//! - (c) the minimal polynomial nearly vanishes at it, relative to the
//!   rounding scale of a Horner evaluation.

use fusion_symbols::fibonacci::FibonacciObject;
use fusion_symbols::tables::{FibonacciTable, TableObject};
use num_complex::Complex64;

const CHECK: &str = include_str!("fixtures/anyonwiki-fibonacci-check.txt");

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

fn objects(fields: &[&str]) -> Vec<TableObject> {
    fields
        .iter()
        .map(|s| TableObject(s.parse().unwrap()))
        .collect()
}

/// `|p(z)|` and the bound `4 n ε Σ |a_k| |z|^k` for a Horner evaluation.
fn residual(qqb: &str, z: Complex64) -> (f64, f64) {
    let (coefficients, _) = qqb.split_once("__").expect("qqb_id has a root index");
    let a: Vec<f64> = coefficients
        .split('_')
        .map(|c| c.parse::<i64>().unwrap() as f64)
        .collect();
    let value = a
        .iter()
        .rev()
        .fold(Complex64::new(0.0, 0.0), |acc, &c| acc * z + c);
    let scale: f64 = a
        .iter()
        .enumerate()
        .map(|(k, c)| c.abs() * z.norm().powi(k as i32))
        .sum();
    (
        value.norm(),
        4.0 * a.len() as f64 * f64::EPSILON * scale.max(1.0),
    )
}

#[test]
fn shipped_values_match_the_exact_anyonwiki_symbols() {
    let fib = FibonacciTable::try_new().expect("tables load");
    let (mut f_rows, mut r_rows) = (0, 0);
    for line in CHECK.lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields[0] == "T" {
            continue; // twists: `twist_rows_fix_the_closed_form_and_bound_the_table`
        }
        // Labels are the shipped record layout: F `a b c d α e β μ f ν`,
        // R `a b c α μ`; multiplicity slots are always 1.
        let (got, rest) = match fields[0] {
            "F" => {
                let l = objects(&fields[1..11]);
                f_rows += 1;
                (
                    fib.f_symbol(l[0], l[1], l[2], l[3], l[5], l[8]),
                    &fields[11..],
                )
            }
            "R" => {
                let l = objects(&fields[1..6]);
                r_rows += 1;
                (fib.r_symbol(l[0], l[1], l[2]), &fields[6..])
            }
            other => panic!("unknown tag {other}"),
        };
        let qqb = rest[0];
        let parse = |s: &str| s.parse::<f64>().unwrap();
        let generated = Complex64::new(parse(rest[1]), parse(rest[2]));
        let numeric = Complex64::new(parse(rest[3]), parse(rest[4]));

        assert_eq!(
            (got.re.to_bits(), got.im.to_bits()),
            (generated.re.to_bits(), generated.im.to_bits()),
            "(a) {line}"
        );
        assert!(
            ulps(got.re, numeric.re) <= 4 && ulps(got.im, numeric.im) <= 4,
            "(b) {line}"
        );
        let (value, bound) = residual(qqb, got);
        assert!(value <= bound, "(c) {line}: |p| = {value:e} > {bound:e}");
    }
    // Every nonzero entry of the rank-2 tables: 15 F, 5 R.
    assert_eq!((f_rows, r_rows), (15, 5));
}

/// The same category in `JCBridgeman/smallRankUnitaryFusionData` at
/// `a6ba1c3` (`uFC/FR_2_0_2`, categorification 0, braiding 0; MIT, copied
/// verbatim with its license) — the files CategoryData.jl v0.1.3
/// redistributes as `FR_2_1_0_2*`. Same gauge, same braiding, and its
/// 20-digit decimals round to the same binary64 values.
#[test]
fn shipped_tables_equal_the_small_rank_unitary_fusion_data_cross_check() {
    const DIR: &str = "fixtures/smallRankUnitaryFusionData-a6ba1c3/uFC/FR_2_0_2";
    let tables = [
        (
            include_str!("fixtures/smallRankUnitaryFusionData-a6ba1c3/uFC/FR_2_0_2/Nabc.txt"),
            include_str!("../data/anyonwiki-badbfe2/fibonacci/Nabc.txt"),
            3,
        ),
        (
            include_str!("fixtures/smallRankUnitaryFusionData-a6ba1c3/uFC/FR_2_0_2/0/F.txt"),
            include_str!("../data/anyonwiki-badbfe2/fibonacci/F.txt"),
            10,
        ),
        (
            include_str!("fixtures/smallRankUnitaryFusionData-a6ba1c3/uFC/FR_2_0_2/0/0/R.txt"),
            include_str!("../data/anyonwiki-badbfe2/fibonacci/R.txt"),
            5,
        ),
    ];
    for (reference, shipped, labels) in tables {
        let records = |text: &str| {
            let mut rows: Vec<(Vec<String>, Vec<u64>)> = text
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    let f: Vec<&str> = l.split_whitespace().collect();
                    (
                        f[..labels].iter().map(|s| s.to_string()).collect(),
                        f[labels..]
                            .iter()
                            .map(|s| s.parse::<f64>().unwrap().to_bits())
                            .collect(),
                    )
                })
                .collect();
            rows.sort();
            rows
        };
        assert_eq!(records(reference), records(shipped), "{DIR}");
    }
}

/// `T` rows: the correctly rounded `e^{2πiθ}` of AnyonWiki's exact twist
/// fraction `θ`. The closed-form provider returns exactly these bits; the
/// table provider derives its twist from F and R by the TensorKitSectors sum,
/// so it meets them within the documented 32ε bound.
#[test]
fn twist_rows_fix_the_closed_form_and_bound_the_table() {
    let fib = FibonacciTable::try_new().expect("tables load");
    let mut rows = 0;
    for line in CHECK.lines().filter(|l| l.starts_with("T ")) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let label: u16 = f[1].parse().unwrap();
        let want = Complex64::new(f[3].parse().unwrap(), f[4].parse().unwrap());
        let closed =
            [FibonacciObject::Vacuum, FibonacciObject::Tau][usize::from(label) - 1].twist();
        assert_eq!(
            (closed.re.to_bits(), closed.im.to_bits()),
            (want.re.to_bits(), want.im.to_bits()),
            "closed twist {line}"
        );
        let derived = fib.twist(TableObject(label));
        assert!(
            (derived - want).norm() <= 32.0 * f64::EPSILON,
            "table twist {line}: {derived}"
        );
        rows += 1;
    }
    assert_eq!(rows, 2);
}
