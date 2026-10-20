//! SU(N) Frobenius–Schur phase and indicator.
//!
//! Oracles: SUNRepresentations 0.4.0 / TensorKitSectors 0.3.9 values, the sign
//! of racah's own `F[a, ā, a, a, 1, 1]` (the reference definition,
//! `sectors.jl:frobenius_schur_phase_from_Fsymbol`), and `phase`.

#![cfg(feature = "cgc-gen")]

#[path = "common/lie.rs"]
mod lie_test;
use fusion_symbols::{category::FusionCategory, lie::LieIrrep as Irrep};
use lie_test::f_symbol;

fn irr(d: &[i64]) -> Irrep {
    lie_test::irrep(d).unwrap()
}

/// Every Dynkin label of rank `n - 1` with label sum `<= max`.
fn labels(n: usize, max: i64) -> Vec<Vec<i64>> {
    let mut out = vec![vec![]];
    for _ in 1..n {
        out = out
            .into_iter()
            .flat_map(|v: Vec<i64>| {
                (0..=max).map(move |x| {
                    let mut w = v.clone();
                    w.push(x);
                    w
                })
            })
            .collect();
    }
    out.retain(|v| v.iter().sum::<i64>() <= max);
    out
}

/// `frobenius_schur_phase(SUNIrrep{N}(d))` from SUNRepresentations 0.4.0 with
/// TensorKitSectors 0.3.9 (Julia 1.11.6, TeNeT `tools/sun-table-gen`
/// environment), each equal to `dim(a) * Fsymbol(a, dual(a), a, a, one, one)`.
#[test]
fn phase_matches_sunrepresentations() {
    let reference: &[(&[i64], f64)] = &[
        (&[1], -1.0),
        (&[2], 1.0),
        (&[3], -1.0),
        (&[1, 0], 1.0),
        (&[1, 1], 1.0),
        (&[1, 0, 0], -1.0),
        (&[0, 0, 1], -1.0),
        (&[0, 1, 0], 1.0),
        (&[1, 1, 0], -1.0),
        (&[0, 1, 1], -1.0),
        (&[1, 0, 1], 1.0),
        (&[2, 0, 0], 1.0),
        (&[1, 0, 0, 0], 1.0),
        (&[1, 0, 0, 0, 0], -1.0),
        (&[0, 0, 1, 0, 0], -1.0),
        (&[0, 1, 0, 0, 0], 1.0),
        (&[0, 0, 0, 0, 1], -1.0),
    ];
    for &(d, want) in reference {
        assert_eq!(
            lie_test::group(&irr(d)).frobenius_schur_phase(&irr(d)),
            want,
            "{d:?}"
        );
    }
}

#[test]
fn indicator_is_phase_on_self_dual_irreps_and_zero_otherwise() {
    for n in 2..=7 {
        for d in labels(n, 3) {
            let a = irr(&d);
            let want = if a == lie_test::dual(&a) {
                lie_test::group(&a).frobenius_schur_phase(&a) as i8
            } else {
                0
            };
            assert_eq!(
                lie_test::group(&a).frobenius_schur_indicator(&a) as i8,
                want,
                "{d:?}"
            );
        }
    }
    // Representation-theoretic anchors: SU(4) 6 = SO(6) vector is real, the
    // SU(6) 20 = Λ³C⁶ is quaternionic, the adjoint is always real.
    assert_eq!(
        lie_test::group(&irr(&[0, 1, 0])).frobenius_schur_indicator(&irr(&[0, 1, 0])) as i8,
        1
    );
    assert_eq!(
        lie_test::group(&irr(&[0, 0, 1, 0, 0])).frobenius_schur_indicator(&irr(&[0, 0, 1, 0, 0]))
            as i8,
        -1
    );
    assert_eq!(
        lie_test::group(&irr(&[1, 0, 0, 0, 1])).frobenius_schur_indicator(&irr(&[1, 0, 0, 0, 1]))
            as i8,
        1
    );
}

/// The reference definition, evaluated on racah's own F-symbols:
/// `phase = sign(F[a, ā, a, a, 1, 1])` with `|F| = 1 / dim(a)`.
#[test]
fn phase_is_sign_of_f_symbol() {
    for (n, max) in [(3usize, 3i64), (4, 2), (5, 2), (6, 1), (7, 1)] {
        let one = lie_test::trivial(n).unwrap();
        for d in labels(n, max) {
            let a = irr(&d);
            let f = f_symbol(&a, &lie_test::dual(&a), &a, &a, &one, &one).unwrap();
            let dim: f64 = a.dim_exact().to_string().parse().unwrap();
            let scaled = f.at(0, 0, 0, 0) * dim;
            assert!(
                (scaled - lie_test::group(&a).frobenius_schur_phase(&a)).abs() < 1e-9,
                "{d:?}: dim * F = {scaled}"
            );
        }
    }
}
