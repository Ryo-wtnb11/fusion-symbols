//! `LieGroup` SU(2) F- and R-symbols follow the TensorKitSectors `SU2Irrep`
//! convention.
//!
//! Contract: `F^{abc}_d[e, f] = (−1)^{a+b+c+d} √((2e+1)(2f+1)) {a b e; c d f}`
//! (`su2irrep.jl:Fsymbol`, `sqrtdim(e) sqrtdim(f) racahW(a, b, d, c, e, f)`)
//! and `R^{ab}_c = (−1)^{a+b−c}` on every tuple of doubled spins `≤ 12`, with
//! the 6j symbol pinned independently in `oracle.rs`.

use super::wigner_6j;
use crate::category::{FusionCategory, SymbolBlock};
use crate::lie::{GroupId, LieGroup, LieIrrep};

fn su2() -> LieGroup {
    LieGroup::new(GroupId::su(2).unwrap()).unwrap()
}

fn ir(cat: &LieGroup, dj: u32) -> LieIrrep {
    cat.irrep(&[i64::from(dj)]).unwrap()
}

fn exact_f(a: u32, b: u32, c: u32, d: u32, e: u32, f: u32) -> f64 {
    let cat = su2();
    let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|dj| ir(&cat, dj));
    let block = cat.f_symbol(&a, &b, &c, &d, &e, &f).unwrap();
    block.as_slice().first().copied().unwrap_or(0.0)
}

fn exact_r(a: u32, b: u32, c: u32) -> f64 {
    let cat = su2();
    let [a, b, c] = [a, b, c].map(|dj| ir(&cat, dj));
    let block = cat.r_symbol(&a, &b, &c).unwrap();
    block.as_slice().first().copied().unwrap_or(0.0)
}

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
