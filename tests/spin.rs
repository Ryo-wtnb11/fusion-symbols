//! `Spin(N)` acceptance suite: the spinor irreps
//! of the simply-connected form of the `B`/`D` families.
//!
//! Role (`tools/README.md` matrix): *internal consistency* strand. The
//! independent cross-family oracles — `Spin(5) ≅ Sp(4)`, `Spin(6) ≅ SU(4)`,
//! `Spin(3) ≅ SU(2)` — live in `tests/isomorphism.rs`; this file pins the
//! things that are statements about the spinor sector itself: the seed's exact
//! self-check, the label admissibility split between `Spin(N)` and `SO(N)`,
//! the reality types, the CGC isometry property, and the recursion reaching a
//! non-base spinor.
//!
//! Each public operation gets its own focused test rather than one umbrella
//! run, so a regression names the operation that broke.
//!
//! # Recoupling through the defining channel
//!
//! The recoupling gates (`check_f_unitarity` / `check_pentagon` /
//! `check_hexagon`) on spinor channels that route through a *defining*-rep
//! coupled channel cross several CGC bases, so they check the consistency of
//! the spinor and defining bases, which single CGC values cannot.

#![cfg(feature = "lie-generated")]

use crate::common::bcd as bcd_test;
use bcd_test::{check_f_unitarity, check_hexagon, check_pentagon, Series};
use fusion_symbols::lie::GroupId;
use fusion_symbols::{
    category::{CgcBlock, FusionCategory, GroupRepresentations},
    lie::{LieCgc, LieError, LieGroup, LieIrrep as Irrep},
};

/// The `B`/`D` families in scope, as `(series, rank, N)` with `N` the matrix
/// dimension of `Spin(N)`.
const FAMILIES: &[(Series, usize, usize)] = &[
    (Series::B, 2, 5),
    (Series::B, 3, 7),
    (Series::D, 3, 6),
    (Series::D, 4, 8),
];

fn ir(n: usize, dynkin: &[i64]) -> Irrep {
    bcd_test::irrep_in(&GroupId::spin(n).unwrap(), dynkin).expect("Spin(N) label")
}

// ---------------------------------------------------------------------------
// The seed.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Labels.
// ---------------------------------------------------------------------------

/// A spinor label is admissible in `Spin(N)` and rejected by `SO(N)` — the
/// whole content of "global form" at the label layer.
#[test]
fn spinor_labels_are_admissible_only_in_the_cover() {
    for &(_, r, n) in FAMILIES {
        for label in spinor_labels(n, r) {
            assert!(
                bcd_test::irrep_in(&GroupId::spin(n).unwrap(), &label).is_ok(),
                "Spin({n}) must admit {label:?}"
            );
            assert!(
                matches!(
                    bcd_test::irrep_in(&GroupId::so(n).unwrap(), &label),
                    Err(LieError::NotAdmissible)
                ),
                "SO({n}) must reject {label:?}"
            );
            assert!(
                matches!(
                    LieGroup::new(GroupId::so(n).unwrap())
                        .unwrap()
                        .irrep(&label),
                    Err(LieError::NotAdmissible)
                ),
                "the published constructor must still reject {label:?}"
            );
        }
    }
}

/// Weyl dimensions of the fundamental spinors: `2^r` for `B_r`, `2^{r-1}` per
/// chirality for `D_r`. Computed by the exact Weyl formula on the half-integer
/// weight, i.e. an independent statement from the seed's carrier size.
#[test]
fn spinor_dimensions_are_the_clifford_module_dimensions() {
    for &(_, r, n) in FAMILIES {
        let want: u64 = if n % 2 == 1 { 1 << r } else { 1 << (r - 1) };
        for label in spinor_labels(n, r) {
            assert_eq!(
                ir(n, &label).dim_exact(),
                want,
                "dim of Spin({n}) {label:?}"
            );
        }
    }
}

/// The Frobenius–Schur indicator of the fundamental spinor is the `N mod 8`
/// reality type: real for `N ≡ 0, ±1, 2`,
/// quaternionic for `N ≡ 3, 4, 5`, and `0` for the `D_r`-odd half-spinors,
/// which are a conjugate pair rather than self-dual.
#[test]
fn spinor_frobenius_schur_is_the_n_mod_8_reality_type() {
    // (N, dynkin, indicator).
    let cases: &[(usize, &[i64], i32)] = &[
        (5, &[0, 1], -1),              // Spin(5) 4, quaternionic (N ≡ 5)
        (7, &[0, 0, 1], 1),            // Spin(7) 8, real (N ≡ 7)
        (9, &[0, 0, 0, 1], 1),         // Spin(9) 16, real (N ≡ 1)
        (11, &[0, 0, 0, 0, 1], -1),    // Spin(11) 32, quaternionic (N ≡ 3)
        (6, &[0, 0, 1], 0),            // Spin(6) 4, complex (D_3 odd)
        (8, &[0, 0, 0, 1], 1),         // Spin(8) 8_s, real (N ≡ 0)
        (10, &[0, 0, 0, 0, 1], 0),     // Spin(10) 16, complex (D_5 odd)
        (12, &[0, 0, 0, 0, 0, 1], -1), // Spin(12) 32, quaternionic (N ≡ 4)
    ];
    for &(n, dynkin, want) in cases {
        assert_eq!(
            spin_group(n).frobenius_schur_indicator(&ir(n, dynkin)),
            f64::from(want),
            "FS of Spin({n}) {dynkin:?}"
        );
    }
}

/// `D_r` with `r` odd exchanges the two half-spinors under duality; with `r`
/// even each is self-dual: the general `dual` rule applied to the spinor
/// labels.
#[test]
fn half_spinor_duality_follows_the_d_series_chirality_rule() {
    let s = ir(6, &[0, 0, 1]);
    assert_eq!(
        spin_group(6).dual(&s).dynkin().collect::<Vec<_>>(),
        vec![0, 1, 0],
        "Spin(6) 4* = 4bar"
    );
    let s8 = ir(8, &[0, 0, 0, 1]);
    assert_eq!(
        spin_group(8).dual(&s8).dynkin().collect::<Vec<_>>(),
        vec![0, 0, 0, 1],
        "Spin(8) 8_s self-dual"
    );
}

/// Spinor fusion, from the exact Racah–Speiser layer: the textbook `Spin(7)`
/// and `Spin(8)` products. Independent of the numeric pipeline.
#[test]
fn spinor_fusion_matches_the_textbook_products() {
    let channels = |n, a: &Irrep, b: &Irrep| -> Vec<(Vec<i64>, u32)> {
        let mut v: Vec<(Vec<i64>, u32)> = bcd_test::product(&spin_group(n), a, b)
            .unwrap()
            .into_iter()
            .map(|(k, m)| (k.dynkin().collect::<Vec<_>>(), m))
            .collect();
        v.sort();
        v
    };
    // Spin(7): 8 ⊗ 8 = 1 + 7 + 21 + 35.
    assert_eq!(
        channels(7, &ir(7, &[0, 0, 1]), &ir(7, &[0, 0, 1])),
        vec![
            (vec![0, 0, 0], 1),
            (vec![0, 0, 2], 1),
            (vec![0, 1, 0], 1),
            (vec![1, 0, 0], 1),
        ]
    );
    // Spin(8) triality: 8_s ⊗ 8_v = 8_c + 56_c.
    assert_eq!(
        channels(8, &ir(8, &[0, 0, 0, 1]), &ir(8, &[1, 0, 0, 0])),
        vec![(vec![0, 0, 1, 0], 1), (vec![1, 0, 0, 1], 1)]
    );
    // Spin(5): 4 ⊗ 4 = 1 + 5 + 10 (the SO(5) global-form diagnostic).
    assert_eq!(
        channels(5, &ir(5, &[0, 1]), &ir(5, &[0, 1])),
        vec![(vec![0, 0], 1), (vec![0, 2], 1), (vec![1, 0], 1)]
    );
}

// ---------------------------------------------------------------------------
// The CGC.
// ---------------------------------------------------------------------------

/// Worst `|(CᵀC − I)|` over the outer-multiplicity copies of a CGC block.
fn isometry_residual(c: &LieCgc) -> f64 {
    let (rows, cols) = (c.dims()[0] * c.dims()[1], c.dims()[2]);
    let mut worst: f64 = 0.0;
    for mu in 0..c.dims()[3] {
        let m = bcd_test::copy(c, mu);
        for i in 0..cols {
            for j in 0..cols {
                let dot: f64 = (0..rows).map(|k| m[i * rows + k] * m[j * rows + k]).sum();
                let target = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((dot - target).abs());
            }
        }
    }
    worst
}

/// Every spinor CGC is an isometry — orthonormality asserted on the returned
/// coefficients rather than only inside the generation gate.
#[test]
fn spinor_cgc_columns_are_orthonormal() {
    for &(_, r, n) in FAMILIES {
        let cat = spin_group(n);
        let spinors: Vec<Irrep> = spinor_labels(n, r)
            .into_iter()
            .map(|label| ir(n, &label))
            .collect();
        let s = spinors[0].clone();
        for partner in [s.clone(), cat.dual(&s)] {
            for (c, _) in bcd_test::product(&cat, &s, &partner).unwrap() {
                let cgc = cat.cgc(&s, &partner, &c).unwrap();
                let residual = isometry_residual(&cgc);
                assert!(
                    residual < 1e-10,
                    "Spin({n}) {:?}⊗{:?}→{:?} is not an isometry: {residual:e}",
                    s.dynkin().collect::<Vec<_>>(),
                    partner.dynkin().collect::<Vec<_>>(),
                    c.dynkin().collect::<Vec<_>>()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Recoupling gates on the spinor channels that close (see the module note).
// ---------------------------------------------------------------------------

/// `F` is unitary on the `Spin(6)` half-spinor vertices — the `4 ⊗ 4̄` and
/// `4 ⊗ 4` families, which between them cover both the tensor channels
/// (`1 + 15`) and the chiral ones (`6 + 10`).
#[test]
fn spinor_f_symbols_are_unitary_on_spin6() {
    let cat = spin_group(6);
    let s = ir(6, &[0, 0, 1]);
    let sb = ir(6, &[0, 1, 0]);
    check_f_unitarity(&cat, &s, &sb, &s, &sb).unwrap();
    check_f_unitarity(&cat, &s, &s, &s, &s).unwrap();
}

/// The pentagon closes on the `Spin(6)` `4 ⊗ 4̄` spinor family.
#[test]
fn spinor_pentagon_closes_on_spin6() {
    let cat = spin_group(6);
    let s = ir(6, &[0, 0, 1]);
    let sb = ir(6, &[0, 1, 0]);
    check_pentagon(&cat, &s, &sb, &s, &sb).unwrap();
}

/// The hexagon closes on a mixed spinor/vector `Spin(6)` triple — braiding
/// with a spinor leg.
#[test]
fn spinor_hexagon_closes_on_spin6() {
    let cat = spin_group(6);
    let s = ir(6, &[0, 0, 1]);
    let sb = ir(6, &[0, 1, 0]);
    let v = ir(6, &[1, 0, 0]);
    check_hexagon(&cat, &s, &sb, &v).unwrap();
}

/// The `Spin(6)` `4 ⊗ 4 = 6 + 1̄0` family routes through the **defining**-rep
/// coupled channel, so its pentagon and hexagon cross the spinor and defining
/// bases; both close.
#[test]
fn spinor_recoupling_closes_through_the_defining_channel_on_spin6() {
    let s = ir(6, &[0, 0, 1]);
    let cat = spin_group(6);
    check_pentagon(&cat, &s, &s, &s, &s).unwrap();
    let cat = spin_group(6);
    check_hexagon(&cat, &s, &s, &s).unwrap();
}

fn spin_group(n: usize) -> LieGroup {
    LieGroup::new(GroupId::spin(n).unwrap()).unwrap()
}
fn spinor_labels(n: usize, r: usize) -> Vec<Vec<i64>> {
    let mut d = vec![0; r];
    d[r - 1] = 1;
    let mut labels = vec![d.clone()];
    if n.is_multiple_of(2) {
        d[r - 1] = 0;
        d[r - 2] = 1;
        labels.push(d);
    }
    labels
}
