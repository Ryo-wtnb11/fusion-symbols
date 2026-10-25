//! SU(2) F- and R-symbols follow the TensorKitSectors `SU2Irrep` convention.
//!
//! Contract: `F^{abc}_d[e, f] = (−1)^{a+b+c+d} √((2e+1)(2f+1)) {a b e; c d f}`
//! (`su2irrep.jl:Fsymbol`, `sqrtdim(e) sqrtdim(f) racahW(a, b, d, c, e, f)`)
//! and `R^{ab}_c = (−1)^{a+b−c}` on every tuple of doubled spins `≤ 12`, with
//! the 6j symbol pinned independently in `tests/oracle.rs`; a few values are
//! checked against TensorKitSectors itself. F is composed exactly and rounded
//! once, so the tolerance covers only the reference's own last-ulp rounding.

use crate::common::a1;
use a1::{exact_f, exact_r};
use fusion_symbols::lie::su2::wigner_6j;

fn tri(a: u32, b: u32, c: u32) -> bool {
    (a + b + c).is_multiple_of(2) && c >= a.abs_diff(b) && c <= a + b
}

/// `(−1)^{k/2}` for an even doubled exponent `k`.
fn parity(k: u32) -> f64 {
    if k.is_multiple_of(4) {
        1.0
    } else {
        -1.0
    }
}

#[test]
fn f_and_r_are_the_tensorkitsectors_formulas() {
    const MAX: u32 = 12;
    let mut admissible = 0;
    for a in 0..=MAX {
        for b in 0..=MAX {
            for c in 0..=MAX {
                let r = if tri(a, b, c) { parity(a + b - c) } else { 0.0 };
                assert_eq!(exact_r(a, b, c), r, "R {a} {b} {c}");
                for d in 0..=MAX {
                    for e in 0..=MAX {
                        for f in 0..=MAX {
                            let got = exact_f(a, b, c, d, e, f);
                            if !(tri(a, b, e) && tri(e, c, d) && tri(b, c, f) && tri(a, f, d)) {
                                assert_eq!(got, 0.0, "F {:?}", [a, b, c, d, e, f]);
                                continue;
                            }
                            let want = parity(a + b + c + d)
                                * f64::from((e + 1) * (f + 1)).sqrt()
                                * wigner_6j(a, b, e, c, d, f).to_f64();
                            assert!((got - want).abs() <= 1e-14, "F {:?}", [a, b, c, d, e, f]);
                            admissible += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(admissible > 100_000, "{admissible}");
}

#[test]
fn values_checked_against_tensorkitsectors() {
    // Checked against TensorKitSectors 0.3.9 (`SU2Irrep`, Julia 1.11.6);
    // exact values `−1/2`, `1/2`, `√3/2`, `2√2/3`, `1/2`, `√(5/6)`, `√5/3`
    // (the reference prints `0.4999999999999999` for the two halves).
    for (d, want) in [
        ([1, 1, 1, 1, 0, 0], -0.5),
        ([1, 1, 1, 1, 2, 2], 0.5),
        ([1, 1, 1, 1, 0, 2], 0.866_025_403_784_438_6),
        ([1, 2, 1, 2, 1, 3], 0.942_809_041_582_063_4),
        ([2, 2, 2, 2, 2, 2], 0.5),
        ([3, 1, 2, 2, 2, 3], 0.912_870_929_175_276_9),
        ([2, 2, 2, 2, 0, 4], 0.745_355_992_499_929_9),
    ] {
        let got = exact_f(d[0], d[1], d[2], d[3], d[4], d[5]);
        assert!((got - want).abs() <= 2.0 * f64::EPSILON, "F {d:?}: {got}");
    }
    for (d, want) in [
        ([1, 1, 0], -1.0),
        ([1, 1, 2], 1.0),
        ([2, 2, 2], -1.0),
        ([2, 1, 1], -1.0),
        ([3, 3, 4], -1.0),
        ([2, 2, 0], 1.0),
    ] {
        assert_eq!(exact_r(d[0], d[1], d[2]), want, "R {d:?}");
    }
}
