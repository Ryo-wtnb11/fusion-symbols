//! The shipped `tables` catalog against its AnyonWiki authority and the
//! independent smallRankUnitaryFusionData cross-check.
//!
//! `fixtures/anyonwiki-badbfe2-check.txt` is written by
//! `tools/anyonwiki/gen_category_tables.py` together with the shipped tables:
//! one `C uuid code` block per shipped entry. Each `F`/`R` row carries the
//! exact AnyonWiki symbol (`qqb_id`: minimal polynomial coefficients
//! `a0_a1_…_an`, then `__` and the root index), the generator's correctly
//! rounded binary64 value, and AnyonWiki's own `numeric_*` value. For every
//! shipped F- and R-symbol of every entry:
//!
//! - (a) the provider's value is bit-equal to the generator output;
//! - (b) it is within 4 ulp of AnyonWiki's `numeric_*` value (those are not
//!   correctly rounded);
//! - (c) the minimal polynomial nearly vanishes at it, relative to the
//!   rounding scale of a Horner evaluation;
//!
//! and the rows are every nonzero symbol of the provider. `D` and `T` rows are
//! AnyonWiki's exact quantum dimensions and twists: the provider derives both
//! from F and R (the TensorKitSectors formulas) and must meet them.

use fusion_symbols::tables::{TableCategory, TableObject};
use num_complex::Complex64;

const CHECK: &str = include_str!("fixtures/anyonwiki-badbfe2-check.txt");

/// Bound for the derived dimension and twist: a fixed multiple of the unit
/// round-off, as in `tests/tables_reference.rs`.
const FP_BOUND: f64 = 32.0 * f64::EPSILON;

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

/// The fixture's blocks: the category and its rows.
fn blocks() -> Vec<(TableCategory, Vec<Vec<&'static str>>)> {
    let mut blocks: Vec<(TableCategory, Vec<Vec<&'static str>>)> = Vec::new();
    for line in CHECK.lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields[0] == "C" {
            let category = TableCategory::by_uuid(fields[1]).expect("shipped");
            let code: Vec<u8> = fields[2].split(',').map(|c| c.parse().unwrap()).collect();
            assert_eq!(category.code().to_vec(), code);
            blocks.push((category, Vec::new()));
        } else {
            blocks.last_mut().expect("a C row first").1.push(fields);
        }
    }
    blocks
}

#[test]
fn shipped_values_match_the_exact_anyonwiki_symbols() {
    let blocks = blocks();
    assert_eq!(blocks.len(), TableCategory::shipped_uuids().len());
    let (mut f_total, mut r_total) = (0, 0);
    for (category, rows) in &blocks {
        let uuid = category.uuid();
        let (mut f_rows, mut r_rows) = (0, 0);
        for fields in rows.iter().filter(|f| matches!(f[0], "F" | "R")) {
            // Labels are the shipped record layout: F `a b c d α e β μ f ν`,
            // R `a b c α μ`; multiplicity slots are always 1.
            let (got, rest) = if fields[0] == "F" {
                let l = objects(&fields[1..11]);
                f_rows += 1;
                (
                    category.f_symbol(l[0], l[1], l[2], l[3], l[5], l[8]),
                    &fields[11..],
                )
            } else {
                let l = objects(&fields[1..6]);
                r_rows += 1;
                (category.r_symbol(l[0], l[1], l[2]), &fields[6..])
            };
            let qqb = rest[0];
            let parse = |s: &str| s.parse::<f64>().unwrap();
            let generated = Complex64::new(parse(rest[1]), parse(rest[2]));
            let numeric = Complex64::new(parse(rest[3]), parse(rest[4]));
            let line = fields.join(" ");
            assert_eq!(
                (got.re.to_bits(), got.im.to_bits()),
                (generated.re.to_bits(), generated.im.to_bits()),
                "(a) {uuid} {line}"
            );
            assert!(
                ulps(got.re, numeric.re) <= 4 && ulps(got.im, numeric.im) <= 4,
                "(b) {uuid} {line}"
            );
            let (value, bound) = residual(qqb, got);
            assert!(
                value <= bound,
                "(c) {uuid} {line}: |p| = {value:e} > {bound:e}"
            );
        }
        // The rows are the whole table: one per admissible symbol (a few
        // admissible F entries are exact zeros, e.g. in Rep(S₃)), and every
        // other entry of the provider is zero.
        let objects: Vec<TableObject> = category.objects().collect();
        let mut f_admissible = 0;
        let mut r_admissible = 0;
        for &a in &objects {
            for &b in &objects {
                for &c in &objects {
                    if category.is_braided() {
                        if category.r_symbol_checked(a, b, c).is_ok() {
                            r_admissible += 1;
                        } else {
                            assert_eq!(category.r_symbol(a, b, c), Complex64::new(0.0, 0.0));
                        }
                    }
                    for &d in &objects {
                        for &e in &objects {
                            for &f in &objects {
                                if category.f_symbol_checked(a, b, c, d, e, f).is_ok() {
                                    f_admissible += 1;
                                } else {
                                    assert_eq!(
                                        category.f_symbol(a, b, c, d, e, f),
                                        Complex64::new(0.0, 0.0)
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!((f_rows, r_rows), (f_admissible, r_admissible), "{uuid}");
        f_total += f_rows;
        r_total += r_rows;
    }
    assert_eq!((f_total, r_total), (6600, 868));
}

#[test]
fn derived_dimensions_and_twists_meet_the_exact_anyonwiki_values() {
    let (mut dims, mut twists) = (0, 0);
    for (category, rows) in blocks() {
        let uuid = category.uuid();
        for fields in rows.iter().filter(|f| matches!(f[0], "D" | "T")) {
            let a = TableObject(fields[1].parse().unwrap());
            let line = fields.join(" ");
            if fields[0] == "D" {
                let want: f64 = fields[3].parse().unwrap();
                let got = category.dim(a);
                assert!(
                    (got - want).abs() <= FP_BOUND * want,
                    "{uuid} {line}: {got}"
                );
                dims += 1;
            } else {
                let want = Complex64::new(fields[3].parse().unwrap(), fields[4].parse().unwrap());
                let got = category.twist(a);
                assert!((got - want).norm() <= FP_BOUND, "{uuid} {line}: {got}");
                twists += 1;
            }
        }
        assert_eq!(
            rows.iter().filter(|f| f[0] == "D").count(),
            category.rank(),
            "{uuid}: a D row per object"
        );
    }
    // 35 shipped entries are modular and carry AnyonWiki twists.
    assert_eq!((dims, twists), (261, 117));
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
            include_str!("../data/anyonwiki-badbfe2/N/2_1_0_2.txt"),
            3,
        ),
        (
            include_str!("fixtures/smallRankUnitaryFusionData-a6ba1c3/uFC/FR_2_0_2/0/F.txt"),
            include_str!("../data/anyonwiki-badbfe2/F/2_1_0_2_1_1_1.txt"),
            10,
        ),
        (
            include_str!("fixtures/smallRankUnitaryFusionData-a6ba1c3/uFC/FR_2_0_2/0/0/R.txt"),
            include_str!("../data/anyonwiki-badbfe2/R/2_1_0_2_1_2_1.txt"),
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
