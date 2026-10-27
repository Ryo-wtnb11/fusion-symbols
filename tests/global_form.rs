//! Global form: the per-family congruence tables, the equivalence of the
//! admissibility predicate with the rejection `bcd` ships, and closure of
//! admissibility under fusion.
//!
//! A global form only selects admissible labels; it changes no coefficient
//! value (the cross-global-form section checks this).

#[cfg(feature = "lie-generated")]
use crate::common::lie as lie_test;

use fusion_symbols::lie::{GroupError, GroupId, RootSystem};

// ---------------------------------------------------------------------------
// A_r — N-ality
// ---------------------------------------------------------------------------

#[test]
fn a_series_nality_congruences() {
    // SU(4): κ = a₁ + 2a₂ + 3a₃.
    let su4 = GroupId::su(4).unwrap();
    let su4_z2 = GroupId::su_quotient(4, 2).unwrap();
    let psu4 = GroupId::psu(4).unwrap();

    // (label, κ mod 4)
    let table = [
        ([0, 0, 0], 0), // trivial
        ([1, 0, 0], 1), // 4
        ([0, 0, 1], 3), // 4bar
        ([0, 1, 0], 2), // 6
        ([1, 0, 1], 0), // 15, the adjoint
        ([2, 0, 0], 2), // 10
        ([0, 2, 0], 0), // 20'
    ];
    for (label, kappa) in table {
        assert!(
            su4.admits(&label, &[]),
            "SU(4) admits everything: {label:?}"
        );
        assert_eq!(
            su4_z2.admits(&label, &[]),
            kappa % 2 == 0,
            "SU(4)/Z2 on {label:?}"
        );
        assert_eq!(
            psu4.admits(&label, &[]),
            kappa % 4 == 0,
            "PSU(4) on {label:?}"
        );
    }

    // SU(3): only k ∈ {1, 3}; PSU(3) is triality-0, i.e. adjoint and up.
    let psu3 = GroupId::psu(3).unwrap();
    assert!(psu3.admits(&[0, 0], &[])); // 1
    assert!(!psu3.admits(&[1, 0], &[])); // 3
    assert!(!psu3.admits(&[0, 1], &[])); // 3bar
    assert!(psu3.admits(&[1, 1], &[])); // 8
    assert!(psu3.admits(&[3, 0], &[])); // 10

    // k must divide N.
    assert_eq!(
        GroupId::su_quotient(6, 4),
        Err(GroupError::NotACenterSubgroup { n: 6, k: 4 })
    );
    // Trivial quotients have exactly the cover lattice and identity.
    assert_eq!(GroupId::su_quotient(4, 1).unwrap(), su4);
}

#[test]
fn a_series_trivial_quotient_admits_everything() {
    let g = GroupId::su_quotient(5, 1).unwrap();
    for a in 0..3i64 {
        for b in 0..3i64 {
            assert!(g.admits(&[a, b, 0, 0], &[]));
        }
    }
}

// ---------------------------------------------------------------------------
// B_r — a_r even
// ---------------------------------------------------------------------------

#[test]
fn b_series_last_label_even() {
    for r in 2..=5usize {
        let spin = GroupId::spin(2 * r + 1).unwrap();
        let so = GroupId::so(2 * r + 1).unwrap();
        assert_eq!(so.root_systems(), &[RootSystem::B(r)]);
        // PSO(2r+1) == SO(2r+1): the B_r center is already quotiented out.
        assert_eq!(GroupId::pso(2 * r + 1).unwrap(), so);
        for label in sweep(r, 3) {
            assert!(spin.admits(&label, &[]), "Spin admits {label:?}");
            assert_eq!(
                so.admits(&label, &[]),
                label[r - 1] % 2 == 0,
                "SO on {label:?}"
            );
        }
    }
    // Spin(5) spinor ω₂ = (0,1) is not an SO(5) rep; the vector ω₁ = (1,0) is.
    assert!(!GroupId::so(5).unwrap().admits(&[0, 1], &[]));
    assert!(GroupId::so(5).unwrap().admits(&[1, 0], &[]));
    assert!(GroupId::spin(5).unwrap().admits(&[0, 1], &[]));
}

// ---------------------------------------------------------------------------
// C_r — odd-index sum even
// ---------------------------------------------------------------------------

#[test]
fn c_series_odd_index_sum_even() {
    for r in 2..=5usize {
        let sp = GroupId::sp(2 * r).unwrap();
        let psp = GroupId::psp(2 * r).unwrap();
        assert_eq!(sp.root_systems(), &[RootSystem::C(r)]);
        for label in sweep(r, 3) {
            assert!(sp.admits(&label, &[]), "Sp is simply connected: {label:?}");
            let odd: i64 = label.iter().step_by(2).sum();
            assert_eq!(psp.admits(&label, &[]), odd % 2 == 0, "PSp on {label:?}");
        }
    }
    // PSp(4): the defining 4 = (1,0) is not a PSp rep; 5 = (0,1) and the
    // adjoint 10 = (2,0) are.
    let psp4 = GroupId::psp(4).unwrap();
    assert!(!psp4.admits(&[1, 0], &[]));
    assert!(psp4.admits(&[0, 1], &[]));
    assert!(psp4.admits(&[2, 0], &[]));
    assert_eq!(
        GroupId::sp(3),
        Err(GroupError::UnsupportedRank {
            family: "sp",
            value: 3
        })
    );
}

// ---------------------------------------------------------------------------
// D_r, r even — the three index-2 subgroups, and the r mod 4 naming hazard
// ---------------------------------------------------------------------------

/// The half-spin congruences are stated by the class they RETAIN, so they are
/// identical formulas at `r ≡ 0` and `r ≡ 2 (mod 4)`. Were they named by the
/// central element killed, the two would swap between `r = 4` and `r = 6`.
#[test]
fn d_even_three_quotients_at_both_r_mod_4() {
    for r in [4usize, 6] {
        let n = 2 * r;
        let so = GroupId::so(n).unwrap();
        let pso = GroupId::pso(n).unwrap();
        let hs_plus = GroupId::half_spin_plus(n).unwrap();
        let hs_minus = GroupId::half_spin_minus(n).unwrap();

        for label in sweep(r, 3) {
            let t: i64 = label[..r - 2].iter().step_by(2).sum();
            let p = (label[r - 2] + t) % 2;
            let q = (label[r - 1] + t) % 2;
            assert_eq!(
                so.admits(&label, &[]),
                (label[r - 2] + label[r - 1]) % 2 == 0,
                "SO({n}) on {label:?}"
            );
            assert_eq!(
                hs_plus.admits(&label, &[]),
                p == 0,
                "half_spin_plus({n}) on {label:?}"
            );
            assert_eq!(
                hs_minus.admits(&label, &[]),
                q == 0,
                "half_spin_minus({n}) on {label:?}"
            );
            assert_eq!(
                pso.admits(&label, &[]),
                p == 0 && q == 0,
                "PSO({n}) on {label:?}"
            );
        }

        // ω_r (the retained chirality of `plus`) and ω_{r-1} (of `minus`).
        let mut omega_last = vec![0i64; r];
        omega_last[r - 1] = 1;
        let mut omega_prev = vec![0i64; r];
        omega_prev[r - 2] = 1;
        assert!(hs_plus.admits(&omega_last, &[]), "r={r}: plus retains ω_r");
        assert!(!hs_plus.admits(&omega_prev, &[]));
        assert!(
            hs_minus.admits(&omega_prev, &[]),
            "r={r}: minus retains ω_{{r-1}}"
        );
        assert!(!hs_minus.admits(&omega_last, &[]));

        // A half-spin group has no vector representation.
        let mut vector = vec![0i64; r];
        vector[0] = 1;
        assert!(!hs_plus.admits(&vector, &[]));
        assert!(!hs_minus.admits(&vector, &[]));
        assert!(so.admits(&vector, &[]));

        // The adjoint is in every form.
        let mut adjoint = vec![0i64; r];
        adjoint[1] = 1;
        for g in [so, pso, hs_plus, hs_minus] {
            assert!(g.admits(&adjoint, &[]), "{g:?} must admit the adjoint");
        }
    }

    // Half-spin needs r even, i.e. n ≡ 0 (mod 4).
    assert_eq!(
        GroupId::half_spin_plus(10),
        Err(GroupError::UnsupportedRank {
            family: "half_spin",
            value: 10
        })
    );
}

// ---------------------------------------------------------------------------
// D_r, r odd — the Z4 class, anchored on Slansky Table 41 (SO(10))
// ---------------------------------------------------------------------------

#[test]
fn d_odd_z4_class_slansky_so10_anchors() {
    // Slansky Table 41: "the adjoint is in the 0 class, the spinor in 1, the
    // conjugate spinors in -1, and the vector … in the 2 class."
    assert_eq!(d_odd_class(&[0, 1, 0, 0, 0]), 0); // 45, adjoint
    assert_eq!(d_odd_class(&[0, 0, 0, 0, 1]), 1); // 16 = ω₅
    assert_eq!(d_odd_class(&[0, 0, 0, 1, 0]), 3); // 16bar = ω₄, i.e. -1
    assert_eq!(d_odd_class(&[1, 0, 0, 0, 0]), 2); // 10, vector
    assert_eq!(d_odd_class(&[0, 0, 1, 0, 0]), 2); // 120 = ω₃, class v

    // The generator convention: κ agrees with `dual` (λ_r ↦ -λ_r), so
    // conjugation negates the class.
    for label in sweep(5, 3) {
        let mut conj = label.clone();
        conj.swap(3, 4);
        assert_eq!(
            (d_odd_class(&label) + d_odd_class(&conj)) % 4,
            0,
            "κ(λ*) = -κ(λ) on {label:?}"
        );
    }

    let so10 = GroupId::so(10).unwrap();
    let pso10 = GroupId::pso(10).unwrap();
    let spin10 = GroupId::spin(10).unwrap();
    for label in sweep(5, 3) {
        let kappa = d_odd_class(&label);
        assert!(spin10.admits(&label, &[]));
        // κ mod 2 reproduces the shipped SO condition a_{r-1}+a_r even.
        assert_eq!(
            so10.admits(&label, &[]),
            kappa % 2 == 0,
            "SO(10) on {label:?}"
        );
        assert_eq!(so10.admits(&label, &[]), (label[3] + label[4]) % 2 == 0);
        assert_eq!(
            pso10.admits(&label, &[]),
            kappa == 0,
            "PSO(10) on {label:?}"
        );
    }

    // There is no half-spin group for odd r: Z4 has one proper subgroup.
    assert!(matches!(
        GroupId::half_spin_plus(10),
        Err(GroupError::UnsupportedRank { .. })
    ));
}

// ---------------------------------------------------------------------------
// Ill-formed input
// ---------------------------------------------------------------------------

#[test]
fn admits_rejects_wrong_length_and_negative_labels() {
    let su4 = GroupId::su(4).unwrap();
    assert!(!su4.admits(&[0, 0], &[]));
    assert!(!su4.admits(&[0, 0, 0, 0], &[]));
    assert!(!su4.admits(&[-1, 0, 0], &[]));
    assert!(!GroupId::so(9).unwrap().admits(&[0, 0, -2, 0], &[]));
}

// ---------------------------------------------------------------------------
// Equivalence with the rejection `bcd` already ships, and fusion closure
// ---------------------------------------------------------------------------

#[cfg(feature = "lie-generated")]
mod against_bcd {
    use super::*;
    use crate::common::bcd as bcd_test;
    use bcd_test::Series;
    use fusion_symbols::lie::{LieError, LieGroup, LieIrrep as Irrep};

    /// The group `bcd` publishes for `series` at rank `r` .
    fn published(series: Series, r: usize) -> GroupId {
        match series {
            Series::B => GroupId::so(2 * r + 1).unwrap(),
            Series::C => GroupId::sp(2 * r).unwrap(),
            Series::D => GroupId::so(2 * r).unwrap(),
        }
    }

    /// `from_dynkin` rejects exactly what the
    /// published form fails to admit, over a sector sweep.
    #[test]
    fn predicate_matches_shipped_rejection() {
        let mut checked = 0usize;
        for (series, min_r) in [(Series::B, 2), (Series::C, 2), (Series::D, 3)] {
            for r in min_r..=min_r + 2 {
                let group = published(series, r);
                for label in sweep(r, 3) {
                    let built = LieGroup::new(group.clone()).unwrap().irrep(&label, &[]);
                    let admitted = group.admits(&label, &[]);
                    match (&built, admitted) {
                        (Ok(_), true) => {}
                        (Err(LieError::NotAdmissible), false) => {}
                        other => panic!(
                            "series {series:?} rank {r} label {label:?}: {other:?} \
                             disagrees with admits = {admitted}"
                        ),
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 500, "sweep too small: {checked}");
    }

    /// Admissibility is closed under fusion (the class map is a homomorphism),
    /// which is why enforcement is construction-only.
    #[test]
    fn admissibility_is_closed_under_fusion() {
        for (series, r) in [(Series::B, 3), (Series::C, 3), (Series::D, 4)] {
            let group = published(series, r);
            let labels: Vec<_> = sweep(r, 2)
                .into_iter()
                .filter(|l| group.admits(l, &[]))
                .collect();
            for a in &labels {
                for b in &labels {
                    let ia = LieGroup::new(group.clone()).unwrap().irrep(a, &[]).unwrap();
                    let ib = LieGroup::new(group.clone()).unwrap().irrep(b, &[]).unwrap();
                    for c in bcd_test::product(&LieGroup::new(group.clone()).unwrap(), &ia, &ib)
                        .unwrap()
                        .keys()
                    {
                        assert!(
                            group.admits(&c.dynkin(), &[]),
                            "{series:?} r={r}: {a:?} ⊗ {b:?} produced {:?}",
                            c.dynkin()
                        );
                    }
                }
            }
        }
    }

    /// The same closure for a *quotient* form the engine does not itself know
    /// about: PSp(6) and PSO(8) label subsets of the shipped catalogs.
    #[test]
    fn quotient_forms_are_closed_under_fusion() {
        for (_series, r, group) in [
            (Series::C, 3, GroupId::psp(6).unwrap()),
            (Series::D, 4, GroupId::pso(8).unwrap()),
        ] {
            let labels: Vec<_> = sweep(r, 2)
                .into_iter()
                .filter(|l| {
                    group.admits(l, &[])
                        && LieGroup::new(group.clone()).unwrap().irrep(l, &[]).is_ok()
                })
                .collect();
            assert!(labels.len() > 2, "{group:?}: nothing to fuse");
            for a in &labels {
                for b in &labels {
                    let ia = LieGroup::new(group.clone()).unwrap().irrep(a, &[]).unwrap();
                    let ib = LieGroup::new(group.clone()).unwrap().irrep(b, &[]).unwrap();
                    for c in bcd_test::product(&LieGroup::new(group.clone()).unwrap(), &ia, &ib)
                        .unwrap()
                        .keys()
                    {
                        assert!(
                            group.admits(&c.dynkin(), &[]),
                            "{group:?}: {a:?} ⊗ {b:?} produced {:?}",
                            c.dynkin()
                        );
                    }
                }
            }
        }
    }

    /// `sun` is the simply connected form of `A`: it admits every non-negative
    /// Dynkin label, and the full A(n−1) weight lattice says exactly that.
    #[test]
    fn sun_is_the_simply_connected_a_form() {
        for n in 2..=5usize {
            let su = GroupId::su(n).unwrap();
            for label in sweep(n - 1, 3) {
                assert!(su.admits(&label, &[]));
                assert!(lie_test::irrep(&label).is_ok());
            }
        }
    }

    /// **Diagnostic, not an assertion of policy** (a flagged verification gap). Run with `--ignored`.
    ///
    /// The hand example: for `B₂ = so(5)` the adjoint `10` has both the
    /// spinor `4` and the vector `5` strictly below it in `≺` (box counts
    /// 1, 1 versus 2), and both `4⊗4` and `5⊗5` contain it. Under a
    /// cover-wide parent search (option (A)) the pair key `dim_a+dim_b`
    /// would pick `(4,4) = 8` over `(5,5) = 10`, moving the canonical parent
    /// of the SO(5) adjoint and hence its frame.
    ///
    /// racah cannot build Spin(5) spinors today, so this is measured through
    /// the isomorphism `Spin(5) ≅ Sp(4)`, where the `B₂` spinor is the `Sp(4)`
    /// defining rep `(1,0)` and the `B₂` vector is `(0,1)`.
    #[test]
    #[ignore = "diagnostic: SO(5) canonical-parent hand example; run with --ignored"]
    fn so5_canonical_parent_hand_example() {
        use num_bigint::BigInt;
        let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
        let irrep = |a: i64, b: i64| cat.irrep(&[a, b], &[]).unwrap();
        let spinor = irrep(1, 0); // B₂ ω₂, the 4
        let vector = irrep(0, 1); // B₂ ω₁, the 5
        let adjoint = irrep(2, 0); // B₂ 2ω₂, the 10

        assert_eq!(spinor.dim_exact(), BigInt::from(4));
        assert_eq!(vector.dim_exact(), BigInt::from(5));
        assert_eq!(adjoint.dim_exact(), BigInt::from(10));

        let ss = bcd_test::product(&cat, &spinor, &spinor).unwrap();
        let vv = bcd_test::product(&cat, &vector, &vector).unwrap();
        let dims = |m: &std::collections::BTreeMap<Irrep, u32>| {
            let mut d: Vec<i64> = m
                .keys()
                .map(|k| k.dim_exact().to_string().parse().unwrap())
                .collect();
            d.sort_unstable();
            d
        };
        assert_eq!(dims(&ss), vec![1, 5, 10], "4⊗4 = 1+5+10");
        assert_eq!(dims(&vv), vec![1, 10, 14], "5⊗5 = 1+10+14");
        assert!(ss.contains_key(&adjoint) && vv.contains_key(&adjoint));

        let key_ss = 4 + 4;
        let key_vv = 5 + 5;
        println!(
            "SO(5) adjoint (10): (spinor,spinor) pair key {key_ss}, (V,V) pair key {key_vv} \
             — spinor pair {} under the shipped `dim_a+dim_b` rule; box counts are 1, 1, 2 \
             so both parents are ≺ the adjoint.",
            if key_ss < key_vv { "WINS" } else { "loses" }
        );
        assert!(key_ss < key_vv, "the spinor pair wins: {key_ss} < {key_vv}");
    }
}

/// The `Z4` central class of a `D_r` weight, `r` odd, under the generator
/// `c = [ω_r]`: `κ(λ) ≡ a_r − a_{r-1} + 2·Σ_{i odd ≤ r-2} aᵢ (mod 4)`
/// (derivation in `GroupId::pso`). The oracle the PSO/SO admission is
/// checked against; the Slansky anchors below pin its convention.
fn d_odd_class(dynkin: &[i64]) -> i64 {
    let r = dynkin.len();
    let odd: i64 = dynkin[..r - 2].iter().step_by(2).sum();
    (dynkin[r - 1] - dynkin[r - 2] + 2 * odd).rem_euclid(4)
}

/// Every non-negative Dynkin label of rank `r` with each entry `< cap`.
fn sweep(r: usize, cap: i64) -> Vec<Vec<i64>> {
    let mut out = vec![vec![]];
    for _ in 0..r {
        out = out
            .into_iter()
            .flat_map(|p| {
                (0..cap).map(move |a| {
                    let mut p = p.clone();
                    p.push(a);
                    p
                })
            })
            .collect();
    }
    out
}

// ---------------------------------------------------------------------------
// `sun` in the global-form model
// ---------------------------------------------------------------------------

/// The A-series form-aware constructor, mirroring `bcd::Irrep::from_dynkin_in`.
#[cfg(feature = "lie-generated")]
mod sun_forms {
    use super::*;
    use crate::common::lie as lie_test;
    use fusion_symbols::category::FusionCategory;
    use fusion_symbols::lie::LieError;

    /// `N`-ality `Σ i·aᵢ`, computed here independently of the library so the
    /// tests are not checking `src/lie/group.rs::admits` against itself.
    fn nality(dynkin: &[i64]) -> i64 {
        dynkin
            .iter()
            .enumerate()
            .map(|(i, &a)| (i as i64 + 1) * a)
            .sum()
    }

    /// (1) `SU(N)` is simply connected: every dominant integral weight is a
    /// representation, through either constructor.
    #[test]
    fn su_n_accepts_every_dominant_integral_weight() {
        for n in 2..=5usize {
            let su = GroupId::su(n).unwrap();
            for label in sweep(n - 1, 3) {
                let via_form = lie_test::irrep_in(&su, &label).expect("SU(N) admits everything");
                assert_eq!(via_form, lie_test::irrep(&label).unwrap());
            }
        }
    }

    /// (2) `PSU(N)` rejects every non-zero `N`-ality, and (3) accepts the
    /// adjoint. The rejection is the global-form error, not a label error.
    #[test]
    fn psu_n_rejects_nonzero_nality_and_accepts_the_adjoint() {
        for n in 2..=5usize {
            let psu = GroupId::psu(n).unwrap();
            for label in sweep(n - 1, 4) {
                let built = lie_test::irrep_in(&psu, &label);
                let zero_nality = nality(&label).rem_euclid(n as i64) == 0;
                match (&built, zero_nality) {
                    (Ok(_), true) => {}
                    (Err(LieError::NotAdmissible), false) => {}
                    other => panic!("PSU({n}) on {label:?}: {other:?}"),
                }
            }
            // The adjoint is (1,0,…,0,1) for N ≥ 3, (2) for N = 2 — always
            // zero N-ality, and every group with this algebra has it.
            let mut adjoint = vec![0i64; n - 1];
            adjoint[0] += 1;
            adjoint[n - 2] += 1;
            assert_eq!(nality(&adjoint).rem_euclid(n as i64), 0);
            assert!(lie_test::irrep_in(&psu, &adjoint).is_ok());
        }
        // The stated example: PSU(3) takes the adjoint 8, not the fundamental 3.
        let psu3 = GroupId::psu(3).unwrap();
        assert!(lie_test::irrep_in(&psu3, &[1, 1]).is_ok());
        assert!(matches!(
            lie_test::irrep_in(&psu3, &[1, 0]),
            Err(LieError::NotAdmissible)
        ));
        assert!(matches!(
            lie_test::irrep_in(&psu3, &[0, 1]),
            Err(LieError::NotAdmissible)
        ));
    }

    /// (4) `SU(N)/Z_k` needs `k | N`, and then admits exactly `N`-ality
    /// `≡ 0 (mod k)`.
    #[test]
    fn su_quotient_divisibility_condition() {
        assert!(matches!(
            GroupId::su_quotient(6, 4),
            Err(GroupError::NotACenterSubgroup { n: 6, k: 4 })
        ));
        for (n, k) in [(4usize, 2usize), (6, 2), (6, 3), (6, 6), (4, 1)] {
            let g = GroupId::su_quotient(n, k).unwrap();
            for label in sweep(n - 1, 3) {
                let ok = nality(&label).rem_euclid(k as i64) == 0;
                assert_eq!(
                    lie_test::irrep_in(&g, &label).is_ok(),
                    ok,
                    "SU({n})/Z{k} on {label:?}"
                );
            }
        }
    }

    /// (5) `from_dynkin` is untouched: still SU(N), still the same object, and
    /// still the same errors.
    #[test]
    fn from_dynkin_is_unchanged_and_equals_the_simply_connected_form() {
        for n in 2..=5usize {
            let su = GroupId::su(n).unwrap();
            for label in sweep(n - 1, 3) {
                let a = lie_test::irrep(&label).unwrap();
                let b = lie_test::irrep_in(&su, &label).unwrap();
                assert_eq!(a, b);
                assert_eq!(a.dim_exact(), b.dim_exact());
                assert_eq!(a.dynkin(), b.dynkin());
                assert_eq!(a.dynkin().len() + 1, n);
            }
        }
        assert!(matches!(
            lie_test::irrep(&[-1, 0]),
            Err(LieError::InvalidLabel)
        ));
        assert_eq!(
            fusion_symbols::lie::LieGroup::new(GroupId::su(3).unwrap())
                .unwrap()
                .irrep(&[], &[]),
            Err(LieError::InvalidLabel)
        );
    }

    /// (6) Global-form validation is a gate, not a transform: the GT data of a
    /// weight admissible in a quotient is bit-identical to the cover's.
    #[test]
    fn global_form_validation_does_not_alter_gt_data() {
        let su3 = GroupId::su(3).unwrap();
        let psu3 = GroupId::psu(3).unwrap();
        for label in sweep(2, 4).into_iter().filter(|l| psu3.admits(l, &[])) {
            let cover = lie_test::irrep_in(&su3, &label).unwrap();
            let quotient = lie_test::irrep_in(&psu3, &label).unwrap();
            assert_ne!(cover, quotient);
            assert_eq!(cover.dim_exact(), quotient.dim_exact());
            assert_eq!(
                lie_test::dual(&cover).dynkin(),
                fusion_symbols::lie::LieGroup::new(psu3.clone())
                    .unwrap()
                    .dual(&quotient)
                    .dynkin()
            );
            assert_eq!(cover.gt_patterns(), quotient.gt_patterns());
        }
    }

    /// The constructor's own guards, distinct from a form violation.
    #[test]
    fn constructor_rejects_an_unsupported_root_system_and_a_wrong_length_label() {
        assert!(matches!(
            lie_test::irrep_in(&GroupId::cover(&[RootSystem::G2], 0).unwrap(), &[1, 0]),
            Err(LieError::Unsupported { .. })
        ));
        assert!(matches!(
            lie_test::irrep_in(&GroupId::su(3).unwrap(), &[1, 0, 0]),
            Err(LieError::InvalidLabel)
        ));
        assert!(matches!(
            lie_test::irrep_in(&GroupId::su(3).unwrap(), &[-1, 0]),
            Err(LieError::InvalidLabel)
        ));
    }

    /// A1 stays consistent: `SU(2)` is `A_1` simply connected, `PSU(2) = SO(3)`
    /// is its even-Dynkin (integer-spin) sublattice — the same statement the
    /// `B_1` low-rank redirect makes.
    #[test]
    fn a1_semantics_agree_with_the_su2_layer() {
        let psu2 = GroupId::psu(2).unwrap();
        for dj in 0..8i64 {
            assert_eq!(
                lie_test::irrep_in(&psu2, &[dj]).is_ok(),
                dj % 2 == 0,
                "PSU(2) = SO(3) admits integer spin only: dj={dj}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The admissibility error is generic, not spinor-specific
// ---------------------------------------------------------------------------

/// (9) `PSp(2N)` has no spinors at all — `C_r` is simply connected — yet it
/// rejects weights. The error must be the generic global-form violation, and
/// must name the group that did the rejecting.
#[cfg(feature = "lie-generated")]
#[test]
fn psp_rejection_uses_the_generic_not_admissible_error() {
    use fusion_symbols::lie::{LieError, LieGroup};
    for (group, label) in [
        (GroupId::psp(6).unwrap(), vec![1, 0, 0]),
        (GroupId::pso(8).unwrap(), vec![1, 0, 0, 0]),
        (GroupId::half_spin_plus(8).unwrap(), vec![1, 0, 0, 0]),
        (GroupId::so(7).unwrap(), vec![0, 0, 1]),
    ] {
        let error = LieGroup::new(group.clone())
            .unwrap()
            .irrep(&label, &[])
            .unwrap_err();
        assert_eq!(error, LieError::NotAdmissible);
        assert!(!error.to_string().to_lowercase().contains("spinor"));
    }
}

// ---------------------------------------------------------------------------
// GroupId construction invariants
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-global-form coefficient identity
// ---------------------------------------------------------------------------

/// The invariant this whole design rests on: for a `λ` admissible in **both**
/// the cover and a quotient, the published coefficient data are *identical* —
/// same values, same basis ordering, same multiplicity-axis ordering, same
/// cache entry. A global form gates the request; it never forks the engine.
///
/// Equality here is **literal** (`==` on `f64` / on the entry vectors), not a
/// tolerance: the code path is shared, so any difference at all would mean a
/// second engine had appeared.
#[cfg(feature = "lie-generated")]
mod cross_form_coefficients {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn hash_of<T: Hash>(v: &T) -> u64 {
        let mut h = DefaultHasher::new();
        v.hash(&mut h);
        h.finish()
    }

    /// (11) `Spin(N)` and `SO(N)` on their common tensor representations.
    #[test]
    fn spin_and_so_share_tensor_coefficient_data() {
        use fusion_symbols::{
            category::{CgcBlock, FusionCategory, GroupRepresentations},
            lie::LieGroup,
        };
        for (n, v_label, c_label) in [
            (5, vec![1, 0], vec![0, 2]),
            (6, vec![1, 0, 0], vec![0, 1, 1]),
        ] {
            let spin = LieGroup::new(GroupId::spin(n).unwrap()).unwrap();
            let so = LieGroup::new(GroupId::so(n).unwrap()).unwrap();
            let cv = spin.irrep(&v_label, &[]).unwrap();
            let qv = so.irrep(&v_label, &[]).unwrap();
            let cc = spin.irrep(&c_label, &[]).unwrap();
            let qc = so.irrep(&c_label, &[]).unwrap();
            assert_ne!(cv, qv, "objects retain the global form tag");
            let cover = spin.cgc(&cv, &cv, &cc).unwrap();
            let quot = so.cgc(&qv, &qv, &qc).unwrap();
            assert_eq!(cover.dims(), quot.dims());
            assert_eq!(
                crate::common::bcd::data(&cover),
                crate::common::bcd::data(&quot)
            );
            let f_cover = spin
                .f_symbol(&spin.unit(), &cv, &cv, &cc, &cv, &cc)
                .unwrap();
            let f_quot = so.f_symbol(&so.unit(), &qv, &qv, &qc, &qv, &qc).unwrap();
            assert_eq!(f_cover, f_quot);
        }
    }

    /// (12) `SU(3)` and `PSU(3)` on their common representations: the adjoint
    /// (a multiplicity-2 channel, so the multiplicity-axis ordering is tested)
    /// and a second nontrivial one, the decuplet.
    #[test]
    fn su_and_psu_share_coefficient_data() {
        use crate::common::lie as lie_test;
        use fusion_symbols::category::{CgcBlock, FusionCategory, GroupRepresentations};
        use fusion_symbols::lie::LieGroup;

        let su3 = GroupId::su(3).unwrap();
        let psu3 = GroupId::psu(3).unwrap();
        let adj = vec![1i64, 1];
        let ten = vec![3i64, 0];
        let tenbar = vec![0i64, 3];

        // 8 ⊗ 8 → 8 (multiplicity 2), and 10 ⊗ 10bar → 8.
        let channels = [
            (adj.clone(), adj.clone(), adj.clone()),
            (ten.clone(), tenbar.clone(), adj.clone()),
        ];
        for (a, b, c) in channels {
            for l in [&a, &b, &c] {
                assert!(psu3.admits(l, &[]), "{l:?} must be a PSU(3) rep");
            }
            let cov = |l: &[i64]| lie_test::irrep_in(&su3, l).unwrap();
            let quo = |l: &[i64]| lie_test::irrep_in(&psu3, l).unwrap();
            assert_ne!(cov(&a), quo(&a));
            assert_ne!(
                hash_of(&(cov(&a), cov(&b), cov(&c))),
                hash_of(&(quo(&a), quo(&b), quo(&c))),
                "the SU(N) CGC cache key carries no global form"
            );

            let cover = LieGroup::new(su3.clone())
                .unwrap()
                .cgc(&cov(&a), &cov(&b), &cov(&c))
                .unwrap();
            let quot = LieGroup::new(psu3.clone())
                .unwrap()
                .cgc(&quo(&a), &quo(&b), &quo(&c))
                .unwrap();
            assert_eq!(cover.dims()[3], quot.dims()[3]);
            assert_eq!(cover.dims(), quot.dims());
            assert_eq!(
                lie_test::entries(&cover).len(),
                lie_test::entries(&quot).len()
            );
            assert_eq!(
                lie_test::entries(&cover),
                lie_test::entries(&quot),
                "SU(3) vs PSU(3) CGC entries for {a:?} ⊗ {b:?} → {c:?}"
            );
        }
        assert!(
            lie_test::cgc(
                &lie_test::irrep(&adj).unwrap(),
                &lie_test::irrep(&adj).unwrap(),
                &lie_test::irrep(&adj).unwrap()
            )
            .unwrap()
            .dims()[3]
                == 2,
            "8 ⊗ 8 → 8 is the multiplicity-2 channel this test wants"
        );

        // An F-symbol over the common reps agrees bit for bit.

        let f = |g: &GroupId| {
            let i = |l: &[i64]| lie_test::irrep_in(g, l).unwrap();
            let cat = LieGroup::new(g.clone()).unwrap();
            cat.f_symbol(
                &cat.unit(),
                &i(&adj),
                &i(&adj),
                &i(&adj),
                &i(&adj),
                &i(&adj),
            )
            .unwrap()
        };
        let (fc, fq) = (f(&su3), f(&psu3));
        assert_eq!(fc.dims(), fq.dims());
        let [d0, d1, d2, d3] = fc.dims();
        for i in 0..d0 {
            for j in 0..d1 {
                for k in 0..d2 {
                    for l in 0..d3 {
                        assert_eq!(fc.at(i, j, k, l), fq.at(i, j, k, l));
                    }
                }
            }
        }
    }

    /// (13)/(14) The cache is shared where it should be and cannot be a bypass
    /// where it should not be.
    ///
    /// Warming the cover's entry for a weight the quotient does not admit gives
    /// the quotient no way to reach it: there is no `Irrep` for an inadmissible
    /// weight, so there is no key with which to read the entry. Validation is at
    /// construction, strictly before any cache is touched.
    #[test]
    fn an_inadmissible_rep_cannot_reach_the_cache_through_the_quotient_api() {
        use crate::common::lie as lie_test;
        use fusion_symbols::category::GroupRepresentations;
        use fusion_symbols::lie::{LieError, LieGroup};

        let su3 = GroupId::su(3).unwrap();
        let psu3 = GroupId::psu(3).unwrap();
        let three = lie_test::irrep_in(&su3, &[1, 0]).unwrap();
        let threebar = lie_test::irrep_in(&su3, &[0, 1]).unwrap();
        let adj = lie_test::irrep_in(&su3, &[1, 1]).unwrap();

        // Populate the cover's entry for 3 ⊗ 3bar → 8.
        let warm = lie_test::cgc(&three, &threebar, &adj).unwrap();
        assert!(!lie_test::entries(&warm).is_empty());

        // The quotient still cannot name the fundamental — before, and after.
        for label in [[1i64, 0], [0, 1], [2, 0], [0, 2]] {
            assert!(matches!(
                lie_test::irrep_in(&psu3, &label),
                Err(LieError::NotAdmissible),
            ));
        }
        // And a warm entry for a *shared* weight is genuinely shared: the
        // quotient's request returns the identical data.
        let again = LieGroup::new(psu3.clone())
            .unwrap()
            .cgc(
                &lie_test::irrep_in(&psu3, &[1, 1]).unwrap(),
                &lie_test::irrep_in(&psu3, &[1, 1]).unwrap(),
                &lie_test::irrep_in(&psu3, &[1, 1]).unwrap(),
            )
            .unwrap();
        let cover_again = lie_test::cgc(&adj, &adj, &adj).unwrap();
        assert_eq!(lie_test::entries(&again), lie_test::entries(&cover_again));
    }

    /// The same bypass argument on the B/D side: `SO(N)` cannot reach a spinor
    /// entry warmed by `Spin(N)`.
    #[test]
    fn so_cannot_reach_a_spin_only_entry() {
        use fusion_symbols::{
            category::GroupRepresentations,
            lie::{LieError, LieGroup},
        };
        let spin = LieGroup::new(GroupId::spin(5).unwrap()).unwrap();
        let so = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
        let spinor = spin.irrep(&[0, 1], &[]).unwrap();
        let vector = spin.irrep(&[1, 0], &[]).unwrap();
        spin.cgc(&spinor, &spinor, &vector).unwrap();
        assert_eq!(so.irrep(&[0, 1], &[]), Err(LieError::NotAdmissible));
    }
}

// ---------------------------------------------------------------------------
// Large labels — the congruences are computed mod k, never as a plain i64 sum
// ---------------------------------------------------------------------------

const MAX: i64 = i64::MAX; // 2^63 − 1 ≡ 1 (mod 2), ≡ 3 (mod 4), ≡ 1 (mod 3)

#[test]
fn a_series_nality_does_not_overflow() {
    let psu3 = GroupId::psu(3).unwrap();
    // κ = 1·a₁ + 2·a₂ with 2·(2^62 + 2) = 2^63 + 4 > i64::MAX.
    // 2^62 = 4^31 ≡ 1 (mod 3), so 2^62 + 2 ≡ 0 and κ ≡ 0: admitted.
    assert!(psu3.admits(&[0, (1 << 62) + 2], &[]));
    // 2^62 + 1 ≡ 2, κ ≡ 2·2 = 4 ≡ 1: the 3 class.
    assert!(!psu3.admits(&[0, (1 << 62) + 1], &[]));
    // 2^63 = 2·2^62 ≡ 2, so MAX ≡ 1 and κ ≡ 1 + 2·1 = 3 ≡ 0.
    assert!(psu3.admits(&[MAX, MAX], &[]));
    // κ ≡ 1 + 0 = 1.
    assert!(!psu3.admits(&[MAX, 0], &[]));
}

#[test]
fn bcd_congruences_do_not_overflow() {
    // B₂ = SO(5): a₂ even.
    let so5 = GroupId::so(5).unwrap();
    assert!(!so5.admits(&[0, MAX], &[]));
    assert!(so5.admits(&[MAX, MAX - 1], &[]));

    // C₃ = PSp(6): a₁ + a₃ even; MAX + MAX overflows as a plain sum.
    let psp6 = GroupId::psp(6).unwrap();
    assert!(psp6.admits(&[MAX, 0, MAX], &[]));
    assert!(!psp6.admits(&[MAX, 0, MAX - 1], &[]));

    // D₄ = SO(8): a₃ + a₄ even.
    let so8 = GroupId::so(8).unwrap();
    assert!(so8.admits(&[0, 0, MAX, MAX], &[]));
    assert!(!so8.admits(&[0, 0, MAX, MAX - 1], &[]));

    // D₄ half-spin forms: t = a₁, p = a₃ + t, q = a₄ + t (mod 2).
    // [MAX, 0, MAX, 0]: t = 1, p = 1 + 1 ≡ 0, q = 0 + 1 = 1.
    let label = [MAX, 0, MAX, 0];
    assert!(GroupId::half_spin_plus(8).unwrap().admits(&label, &[]));
    assert!(!GroupId::half_spin_minus(8).unwrap().admits(&label, &[]));
    assert!(!GroupId::pso(8).unwrap().admits(&label, &[]));

    // D₅ = PSO(10): κ = a₅ − a₄ + 2(a₁ + a₃) (mod 4).
    let pso10 = GroupId::pso(10).unwrap();
    // κ ≡ 3 − 3 + 2·(1 + 1) = 4 ≡ 0.
    assert!(pso10.admits(&[MAX, 0, MAX, MAX, MAX], &[]));
    // κ ≡ 3 − 0 + 2·1 = 5 ≡ 1.
    assert!(!pso10.admits(&[MAX, 0, 0, 0, MAX], &[]));
}
