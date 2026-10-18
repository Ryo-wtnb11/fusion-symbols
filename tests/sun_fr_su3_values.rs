//! SU(3) F- and R-symbols in the SUNRepresentations gauge.
//!
//! Contract: the generated blocks equal SUNRepresentations.jl 0.4.0 (through
//! TensorKitSectors 0.3.9 `Fsymbol` / `Rsymbol`) signed and element-wise,
//! including the multiplicity-axis order of `8 ⊗ 8 → 8`. Pentagon, hexagon and
//! unitarity over the small irreps are in `tests/sun_fr_gates.rs`; they are
//! gauge invariant, so these literals are what pins the gauge.
#![cfg(feature = "cgc-gen")]

use fusion_symbols::sun::{f_symbol, r_symbol, Irrep};

fn irr(d: &[i64]) -> Irrep {
    Irrep::from_dynkin(d).unwrap()
}

const ONE: &[i64] = &[0, 0];
const THREE: &[i64] = &[1, 0];
const BAR: &[i64] = &[0, 1];
const SIX: &[i64] = &[2, 0];
const OCT: &[i64] = &[1, 1];

#[test]
fn f_symbols_checked_against_sunrepresentations() {
    // Multiplicity-free blocks: `F^{abc}_d[e, f]`.
    for (l, want) in [
        ([THREE, THREE, THREE, ONE, BAR, BAR], 1.0),
        (
            [THREE, BAR, THREE, THREE, ONE, OCT],
            0.942_809_041_582_063_1,
        ),
        (
            [THREE, BAR, THREE, THREE, OCT, OCT],
            0.333_333_333_333_333_3,
        ),
        (
            [THREE, BAR, THREE, THREE, ONE, ONE],
            0.333_333_333_333_333_3,
        ),
        (
            [THREE, THREE, BAR, THREE, SIX, OCT],
            0.577_350_269_189_625_8,
        ),
    ] {
        let [a, b, c, d, e, f] = l.map(irr);
        let block = f_symbol(&a, &b, &c, &d, &e, &f).unwrap();
        assert_eq!(block.dims(), [1, 1, 1, 1], "{l:?}");
        assert!((block.at(0, 0, 0, 0) - want).abs() < 1e-12, "{l:?}");
    }
    // `F^{888}_8[8, 8]`, a 2×2×2×2 block over `(μ, ν, κ, λ)`.
    let o = irr(OCT);
    let block = f_symbol(&o, &o, &o, &o, &o, &o).unwrap();
    assert_eq!(block.dims(), [2, 2, 2, 2]);
    let x = -0.383_325_938_999_963_3;
    let y = -0.142_857_142_857_142_8;
    for (idx, want) in [
        ([0, 0, 0, 0], 0.857_142_857_142_856),
        ([1, 0, 0, 0], 0.0),
        ([1, 1, 0, 0], y),
        ([1, 0, 1, 0], y),
        ([0, 1, 1, 0], y),
        ([1, 1, 1, 0], x),
        ([0, 1, 1, 1], x),
        ([1, 1, 1, 1], 0.628_571_428_571_427_4),
    ] {
        let got = block.at(idx[0], idx[1], idx[2], idx[3]);
        assert!((got - want).abs() < 1e-12, "F888 {idx:?}: {got}");
    }
}

#[test]
fn r_symbols_checked_against_sunrepresentations() {
    for (a, b, c, want) in [
        (THREE, THREE, SIX, 1.0),
        (THREE, THREE, BAR, -1.0),
        (THREE, BAR, ONE, 1.0),
        (THREE, BAR, OCT, 1.0),
        (OCT, OCT, ONE, 1.0),
    ] {
        let block = r_symbol(&irr(a), &irr(b), &irr(c)).unwrap();
        assert_eq!(block.dim(), 1);
        assert!((block.at(0, 0) - want).abs() < 1e-12, "R {a:?}{b:?}{c:?}");
    }
    // `R^{88}_8`, 2×2 over the multiplicity axes.
    let o = irr(OCT);
    let block = r_symbol(&o, &o, &o).unwrap();
    assert_eq!(block.dim(), 2);
    for (idx, want) in [
        ([0, 0], -0.285_714_285_714_285_3),
        ([1, 0], 0.958_314_847_499_908_9),
        ([0, 1], 0.958_314_847_499_908_8),
        ([1, 1], 0.285_714_285_714_285_5),
    ] {
        let got = block.at(idx[0], idx[1]);
        assert!((got - want).abs() < 1e-12, "R888 {idx:?}: {got}");
    }
}
