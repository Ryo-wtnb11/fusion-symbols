//! The shipped `tables` catalog against its AnyonWiki authority.
//!
//! Contracts:
//!
//! - every shipped entry's derived quantum dimensions are its fusion ring's
//!   Perron–Frobenius dimensions (`d_a d_b = Σ_c N^{ab}_c d_c`, `d_a ≥ 1`,
//!   `d_ā = d_a`), so F carries the right normalization;
//! - a representative symbol is the correctly rounded binary64 value of the
//!   exact AnyonWiki symbol (its minimal polynomial `qqb_id`, coefficients
//!   from the constant term up), including an admissible exact zero and
//!   complex R-symbols, with the derived dimensions and twists equal to
//!   AnyonWiki's.
//!
//! Unitarity, pentagon, hexagon and ribbon over every shipped entry are in
//! `tests/tables_reference.rs`.

use fusion_symbols::tables::{TableCategory, TableObject};
use std::f64::consts::FRAC_1_SQRT_2;

use num_complex::Complex64;

/// Bound for derived dimensions and twists, as in `tests/tables_reference.rs`.
const FP_BOUND: f64 = 32.0 * f64::EPSILON;

fn shipped() -> impl Iterator<Item = TableCategory> {
    TableCategory::shipped_uuids().map(|uuid| TableCategory::by_uuid(uuid).expect("shipped"))
}

/// `|p(z)| ≤ 4 n ε Σ |a_k| |z|^k`: `z` is a root of `p = Σ a_k x^k` up to the
/// rounding of a Horner evaluation.
fn is_root(a: &[i64], z: Complex64) -> bool {
    let value = a
        .iter()
        .rev()
        .fold(Complex64::new(0.0, 0.0), |acc, &c| acc * z + c as f64);
    let scale: f64 = a
        .iter()
        .enumerate()
        .map(|(k, &c)| (c as f64).abs() * z.norm().powi(k as i32))
        .sum();
    value.norm() <= 4.0 * a.len() as f64 * f64::EPSILON * scale.max(1.0)
}

#[test]
fn derived_dimensions_are_the_perron_frobenius_dimensions() {
    let mut entries = 0;
    for p in shipped() {
        let uuid = p.uuid();
        let objects: Vec<TableObject> = p.objects().collect();
        assert_eq!(p.dim(p.unit()), 1.0, "{uuid}");
        for &a in &objects {
            assert!(p.dim(a) >= 1.0 - FP_BOUND, "{uuid} {a:?}");
            assert!((p.dim(p.dual(a)) - p.dim(a)).abs() <= FP_BOUND * p.dim(a));
            for &b in &objects {
                let sum: f64 = p.fusion(a, b).map(|c| p.dim(c)).sum();
                let want = p.dim(a) * p.dim(b);
                assert!((sum - want).abs() <= FP_BOUND * want, "{uuid} {a:?} {b:?}");
            }
        }
        entries += 1;
    }
    let expected = if cfg!(feature = "anyonwiki-unitary") {
        377
    } else {
        74
    };
    assert_eq!(entries, expected);
}

/// `(kind, labels, qqb_id coefficients, re, im)`. F labels `a b c d e f`, R
/// labels `a b c` (multiplicity slots are 1).
type Symbol = (char, &'static [u16], &'static [i64], f64, f64);

fn check_symbols(category: &TableCategory, symbols: &[Symbol]) {
    let uuid = category.uuid();
    for &(kind, labels, poly, re, im) in symbols {
        let l: Vec<TableObject> = labels.iter().map(|&x| TableObject(x)).collect();
        let got = match kind {
            'F' => category.f_symbol_checked(l[0], l[1], l[2], l[3], l[4], l[5]),
            _ => category.r_symbol_checked(l[0], l[1], l[2]),
        }
        .unwrap_or_else(|e| panic!("{uuid} {kind}{labels:?}: {e}"));
        let want = Complex64::new(re, im);
        assert_eq!(
            (got.re.to_bits(), got.im.to_bits()),
            (re.to_bits(), im.to_bits()),
            "{uuid} {kind}{labels:?}: {got}"
        );
        assert!(is_root(poly, want), "{uuid} {kind}{labels:?} not a root");
    }
}

fn check_dims_and_twists(category: &TableCategory, dims: &[f64], twists: &[(f64, f64)]) {
    let uuid = category.uuid();
    for (a, &want) in category.objects().zip(dims) {
        assert!((category.dim(a) - want).abs() <= FP_BOUND * want, "{uuid}");
    }
    for (a, &(re, im)) in category.objects().zip(twists) {
        let got = category.twist(a);
        assert!((got - Complex64::new(re, im)).norm() <= FP_BOUND, "{uuid}");
    }
}

#[test]
fn rank_three_symbols_are_the_exact_anyonwiki_symbols() {
    // Checked against AnyonWiki badbfe2: the modular rank-3 entry with
    // `d = 2cos(π/7), 1 + 2cos(2π/7)` (AnyonWiki `numeric_*` of R^{22}_3 has
    // imaginary part 0.9749279121818235, one ulp below the correctly rounded
    // value shipped here).
    let p = TableCategory::by_code([3, 1, 0, 3, 1, 1, 1]).expect("shipped");
    assert_eq!(p.uuid(), "37fa4ea8-97c6-11f1-8b25-ad9b74351a3e");
    check_symbols(
        &p,
        &[
            (
                'F',
                &[2, 2, 2, 2, 1, 1],
                &[1, -1, -2, 1],
                0.554_958_132_087_371_2,
                0.0,
            ),
            (
                'F',
                &[2, 3, 2, 3, 2, 3],
                &[-1, 0, -1, 0, 2, 0, 1],
                0.895_509_763_098_559_6,
                0.0,
            ),
            (
                'F',
                &[2, 2, 2, 2, 3, 3],
                &[-1, -1, 2, 1],
                -0.554_958_132_087_371_2,
                0.0,
            ),
            (
                'R',
                &[2, 2, 1],
                &[1, 1, 1, 1, 1, 1, 1],
                0.623_489_801_858_733_5,
                -0.781_831_482_468_029_8,
            ),
            (
                'R',
                &[2, 2, 3],
                &[1, -1, 1, -1, 1, -1, 1],
                0.222_520_933_956_314_4,
                0.974_927_912_181_823_6,
            ),
        ],
    );
    check_dims_and_twists(
        &p,
        &[1.0, 1.801_937_735_804_838_3, 2.246_979_603_717_467],
        &[
            (1.0, 0.0),
            (0.623_489_801_858_733_5, 0.781_831_482_468_029_8),
            (-0.222_520_933_956_314_4, -0.974_927_912_181_823_6),
        ],
    );
    // Checked against AnyonWiki badbfe2: an admissible F-symbol that is
    // exactly zero (`qqb_id` `0_1`), in the rank-3 entry with
    // `σ ⊗ σ = 1 ⊕ ψ ⊕ σ`.
    let p = TableCategory::by_code([3, 1, 0, 2, 1, 1, 1]).expect("shipped");
    check_symbols(&p, &[('F', &[3, 3, 3, 3, 3, 3], &[0, 1], 0.0, 0.0)]);
}

#[cfg(feature = "anyonwiki-unitary")]
#[test]
fn rank_five_symbols_are_the_exact_anyonwiki_symbols() {
    // Checked against AnyonWiki badbfe2: the rank-5 entry with
    // `d = 1, 1, √3, √3, 2`. `1/√3` correctly rounded is 0.5773502691896257
    // (`1.0 / 3f64.sqrt()` gives ...258).
    let p = TableCategory::by_code([5, 1, 0, 3, 1, 1, 1]).expect("shipped");
    assert_eq!(p.uuid(), "37fa51e4-97c6-11f1-8e38-175327b017ad");
    check_symbols(
        &p,
        &[
            (
                'F',
                &[3, 3, 3, 3, 1, 1],
                &[-1, 0, 3],
                0.577_350_269_189_625_7,
                0.0,
            ),
            (
                'F',
                &[3, 3, 3, 3, 1, 5],
                &[2, 0, 3],
                0.0,
                -0.816_496_580_927_726,
            ),
            (
                'R',
                &[3, 3, 1],
                &[1, 0, 0, 0, 1],
                -FRAC_1_SQRT_2,
                FRAC_1_SQRT_2,
            ),
            (
                'R',
                &[3, 3, 5],
                &[1, 0, 0, 0, -1, 0, 0, 0, 1],
                -0.258_819_045_102_520_74,
                -0.965_925_826_289_068_3,
            ),
        ],
    );
    let r3 = 1.732_050_807_568_877_2;
    check_dims_and_twists(
        &p,
        &[1.0, 1.0, r3, r3, 2.0],
        &[
            (1.0, 0.0),
            (1.0, 0.0),
            (-FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
            (FRAC_1_SQRT_2, FRAC_1_SQRT_2),
            (-0.5, -0.866_025_403_784_438_6),
        ],
    );
}
