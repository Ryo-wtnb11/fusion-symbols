//! Strongest CGC oracle: the SU(N) generation pipeline specialized to N = 2
//! must reproduce the crate's *exact* closed-form SU(2) Clebsch-Gordan
//! coefficients.
//!
//! The exact core ([`crate::lie::su2::clebsch_gordan`], big-rational Racah sums rounded
//! once) is an independent value source, so this is a signed, exact check of
//! the numerical pipeline -- not a self-consistency identity.
//!
//! Gauge note: SUNRepresentations' deterministic gauge and the Condon-Shortley
//! convention of the exact core can differ by one overall sign per coupling
//! channel `(j1, j2, j3)` (the highest-weight `qrpos!` sign). The test aligns
//! that single global sign per channel, then requires *every* element to agree
//! -- so all magnitudes and all relative signs (the full ladder structure) are
//! checked exactly.

#![cfg(feature = "lie-generated")]

use super::{cgc, Irrep};
use crate::lie::su2::clebsch_gordan;
use rand::{Rng, SeedableRng};

/// GT basis index `x` (0..=dj) of an SU(2) irrep maps to the doubled magnetic
/// number `dm = 2x - dj`.
fn dm(x: usize, dj: u32) -> i32 {
    2 * x as i32 - dj as i32
}

fn exact_cg(dj1: u32, x1: usize, dj2: u32, x2: usize, dj3: u32, x3: usize) -> f64 {
    clebsch_gordan(dj1, dm(x1, dj1), dj2, dm(x2, dj2), dj3, dm(x3, dj3)).to_f64()
}

fn assert_channel(dj1: u32, dj2: u32, dj3: u32) {
    // s3 must be in s1 ⊗ s2 (triangle + parity); skip otherwise.
    let admissible =
        dj3 >= dj1.abs_diff(dj2) && dj3 <= dj1 + dj2 && (dj1 + dj2 + dj3).is_multiple_of(2);
    if !admissible {
        return;
    }
    let s1 = Irrep::from_dynkin(&[dj1 as i64]).unwrap();
    let s2 = Irrep::from_dynkin(&[dj2 as i64]).unwrap();
    let s3 = Irrep::from_dynkin(&[dj3 as i64]).unwrap();
    let c = cgc(&s1, &s2, &s3).unwrap();
    assert_eq!(c.dims()[3], 1, "SU(2) fusion is multiplicity-free");

    // Align one global sign per channel from the largest-magnitude element.
    let mut sign = 1.0f64;
    let mut best = 0.0f64;
    for e in c.entries() {
        let ex = exact_cg(dj1, e.m1 as usize, dj2, e.m2 as usize, dj3, e.m3 as usize);
        if ex.abs() > best {
            best = ex.abs();
            sign = if (ex.signum() - e.value.signum()).abs() < 0.5 {
                1.0
            } else {
                -1.0
            };
        }
    }

    // Every stored entry must match the exact value (with the global sign).
    for e in c.entries() {
        let ex = exact_cg(dj1, e.m1 as usize, dj2, e.m2 as usize, dj3, e.m3 as usize);
        assert!(
            (sign * e.value - ex).abs() < 1e-10,
            "SU(2) {dj1}⊗{dj2}→{dj3} at ({},{},{}): got {}, exact {ex}",
            e.m1,
            e.m2,
            e.m3,
            sign * e.value
        );
    }

    // Conversely, every exact nonzero must be present (no dropped coefficient).
    let d1 = c.dims()[0];
    let d2 = c.dims()[1];
    for x1 in 0..d1 {
        for x2 in 0..d2 {
            // weight additivity: dm3 = dm1 + dm2 => x3 fixed.
            let dm3 = dm(x1, dj1) + dm(x2, dj2);
            if dm3.unsigned_abs() > dj3 {
                continue;
            }
            let x3 = ((dm3 + dj3 as i32) / 2) as usize;
            let ex = exact_cg(dj1, x1, dj2, x2, dj3, x3);
            if ex.abs() < 1e-12 {
                continue;
            }
            let got = c
                .entries()
                .iter()
                .find(|e| e.m1 as usize == x1 && e.m2 as usize == x2 && e.m3 as usize == x3)
                .map(|e| sign * e.value)
                .unwrap_or(0.0);
            assert!(
                (got - ex).abs() < 1e-10,
                "SU(2) {dj1}⊗{dj2}→{dj3} missing/wrong at ({x1},{x2},{x3}): got {got}, exact {ex}"
            );
        }
    }
}

#[test]
fn su2_singlet_from_halves() {
    // The canonical 1/2 ⊗ 1/2 → 0 case (minimal wide-nullspace system).
    assert_channel(1, 1, 0);
}

#[test]
fn su2_small_channels_match_exact() {
    for dj1 in 0..=4u32 {
        for dj2 in 0..=4u32 {
            for dj3 in dj1.abs_diff(dj2)..=(dj1 + dj2) {
                assert_channel(dj1, dj2, dj3);
            }
        }
    }
}

#[test]
fn su2_randomized_sweep_matches_exact() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x00C0_FFEE_5107);
    for _ in 0..40 {
        let dj1 = rng.gen_range(0..=6u32);
        let dj2 = rng.gen_range(0..=6u32);
        let dj3 = rng.gen_range(dj1.abs_diff(dj2)..=(dj1 + dj2));
        assert_channel(dj1, dj2, dj3);
    }
}

use super::directproduct;
use std::collections::BTreeMap;

fn channels(a: &Irrep, b: &Irrep) -> BTreeMap<Vec<i64>, u32> {
    directproduct(a, b)
        .unwrap()
        .into_iter()
        .map(|(c, m)| (c.dynkin(), m))
        .collect()
}

/// The highest-weight Clebsch–Gordan coefficient is nonzero iff the coupling
/// is allowed, and SU(2)
/// multiplicities are 0/1. This is an independent code path from the GT engine.
fn su2_mult_base(dj1: i32, dj2: i32, dj3: i32) -> u32 {
    // Highest-weight state of j3: m3 = dj3, m1 = dj1, m2 = dj3 - dj1.
    let dm2 = dj3 - dj1;
    if dm2.abs() > dj2 {
        return 0;
    }
    let cg = clebsch_gordan(dj1 as u32, dj1, dj2 as u32, dm2, dj3 as u32, dj3);
    (cg.sign() != 0) as u32
}

#[test]
fn so4_su2xsu2_two_paths_agree() {
    // SO(4) tensor irreps (p + q even), spins (p/2, q/2). Check both the
    // dimension identity and that the two SU(2) code paths give the same
    // fusion multiplicities on each factor.
    let spins = [0i64, 1, 2, 3, 4]; // doubled spins dj
    for &p in &spins {
        for &q in &spins {
            if (p + q) % 2 != 0 {
                continue; // not a tensor (linear) SO(4) irrep
            }
            // dim (p+1)(q+1) via the sun N=2 dims, integer-exact.
            let dl = Irrep::from_dynkin(&[p]).unwrap().dim();
            let dr = Irrep::from_dynkin(&[q]).unwrap().dim();
            assert_eq!(dl.clone() * dr, num_bigint::BigInt::from((p + 1) * (q + 1)));

            // Factor products: SU(2) via GT engine (sun N=2) vs exact base. Only
            // the left factor (p, p2) is exercised — the right factor (q) obeys the
            // identical SU(2) rule, so a q2 loop would add redundant, not new,
            // coverage.
            for &p2 in &spins {
                let gt_left = channels(
                    &Irrep::from_dynkin(&[p]).unwrap(),
                    &Irrep::from_dynkin(&[p2]).unwrap(),
                );
                // Compare against base-module multiplicities for every dj3.
                for dj3 in 0..=(p + p2) {
                    let gt = *gt_left.get(&vec![dj3]).unwrap_or(&0);
                    let base = su2_mult_base(p as i32, p2 as i32, dj3 as i32);
                    assert_eq!(gt, base, "SU(2) mult {p} x {p2} -> {dj3}");
                }
            }
        }
    }
}

#[test]
fn su2_frobenius_schur_matches_exact_and_generated_f() {
    let one = Irrep::from_dynkin(&[0]).unwrap();
    for dj in 0..12 {
        let a = Irrep::from_dynkin(&[dj]).unwrap();
        let phase = crate::lie::su2::fs_phase(dj as u32);
        assert_eq!(a.frobenius_schur_phase(), phase);
        if dj <= 6 {
            let f = super::f_symbol(&a, &a.dual(), &a, &a, &one, &one).unwrap();
            assert!((f.at(0, 0, 0, 0) * (dj + 1) as f64 - phase).abs() < 1e-9);
        }
    }
}
