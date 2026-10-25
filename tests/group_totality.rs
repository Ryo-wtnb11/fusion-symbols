//! Validated root data, canonical lattice identity, and the execution boundary.
use fusion_symbols::lie::{GroupError, GroupId, LieError, LieGroup, RootSystem::*};
use std::hash::{DefaultHasher, Hash, Hasher};

fn hash(g: &GroupId) -> u64 {
    let mut h = DefaultHasher::new();
    g.hash(&mut h);
    h.finish()
}

#[test]
fn hnf_identity_is_independent_of_row_basis() {
    let a = GroupId::new(&[D(2)], 0, &[vec![2, 0], vec![1, 1]]).unwrap();
    let b = GroupId::new(&[D(2)], 0, &[vec![1, 1], vec![1, -1]]).unwrap();
    let expected = vec![vec![1, 1], vec![0, 2]];
    assert_eq!(a.character_lattice(), expected);
    assert_eq!(b.character_lattice(), expected);
    assert_eq!(a, b);
    assert_eq!(a.cmp(&b), std::cmp::Ordering::Equal);
    assert_eq!(hash(&a), hash(&b));
    assert_eq!(a, GroupId::so(4).unwrap());
    assert_ne!(a, GroupId::new(&[A(1), A(1)], 0, &expected).unwrap());
    // A large shear and negative pivots preserve the embedded lattice.
    let c = GroupId::new(&[D(2)], 0, &[vec![-1, -1 - 2 * 1000], vec![0, -2]]).unwrap();
    assert_eq!(a, c);
}

#[test]
fn named_and_general_lattices_agree() {
    assert_eq!(GroupId::su(3).unwrap(), GroupId::cover(&[A(2)], 0).unwrap());
    assert_eq!(
        GroupId::psu(3).unwrap(),
        GroupId::new(&[A(2)], 0, &[vec![2, -1], vec![-1, 2]]).unwrap()
    );
    assert_eq!(GroupId::su(4).unwrap(), GroupId::su_quotient(4, 1).unwrap());
    // D3 root rows in the crate's unchanged spinor numbering.
    let d3_q = [vec![2, -1, -1], vec![-1, 2, 0], vec![-1, 0, 2]];
    assert_eq!(
        GroupId::pso(6).unwrap(),
        GroupId::new(&[D(3)], 0, &d3_q).unwrap()
    );
    // The former D-odd Z2 and DVector descriptions are the same sublattice.
    assert_eq!(
        GroupId::so(6).unwrap(),
        GroupId::new(&[D(3)], 0, &[vec![1, 0, 0], vec![0, 1, 1], vec![0, 0, 2]]).unwrap()
    );
    assert_eq!(
        GroupId::pso(4).unwrap(),
        GroupId::new(&[D(2)], 0, &[vec![2, 0], vec![0, 2]]).unwrap()
    );
    for a in 0..5 {
        for b in 0..5 {
            assert_eq!(
                GroupId::so(4).unwrap().admits(&[a, b], &[]),
                (a + b) % 2 == 0
            );
        }
    }
    for roots in [[A(1)], [B(1)], [C(1)]] {
        let g = GroupId::cover(&roots, 0).unwrap();
        assert_eq!(g.root_systems(), roots);
        assert_eq!(g.semisimple_rank(), 1);
        assert_eq!(g.torus_rank(), 0);
    }
    assert_ne!(GroupId::cover(&[A(1)], 0), GroupId::cover(&[B(1)], 0));
    assert_ne!(GroupId::cover(&[B(1)], 0), GroupId::cover(&[C(1)], 0));
    assert_ne!(GroupId::cover(&[D(3)], 0), GroupId::cover(&[A(3)], 0));
    assert_ne!(
        GroupId::cover(&[A(1), A(2)], 0),
        GroupId::cover(&[A(2), A(1)], 0)
    );
}

#[test]
fn unitary_groups_use_the_cover_charge_congruence() {
    assert_eq!(GroupId::u(1), GroupId::torus(1));
    let u2 = GroupId::u(2).unwrap();
    assert!(u2.admits(&[1], &[1]));
    assert!(!u2.admits(&[1], &[0]));
    let u3 = GroupId::u(3).unwrap();
    for (a, q, yes) in [
        ([1, 0], 1, true),
        ([0, 0], 3, true),
        ([0, 1], -1, true),
        ([1, 0], 0, false),
    ] {
        assert_eq!(u3.admits(&a, &[q]), yes);
    }
    for n in 2..=7 {
        let g = GroupId::u(n).unwrap();
        for i in 0..n - 1 {
            let mut a = vec![0; n - 1];
            a[i] = 1;
            for q in -10i64..=10 {
                assert_eq!(
                    g.admits(&a, &[q]),
                    (q - i as i64 - 1).rem_euclid(n as i64) == 0
                );
            }
        }
    }
    assert!(u3.admits(&[i64::MAX, i64::MAX], &[i64::MIN + 2]));
    assert!(!u3.admits(&[1, 0], &[]));
    assert!(!u3.admits(&[-1, 0], &[1]));
    assert!(!u3.admits(&[1], &[1]));
    assert!(!u3.admits(&[1, 0], &[1, 0]));
}

#[test]
fn invalid_shapes_singular_bases_and_missing_roots_are_typed_errors() {
    for rows in [
        vec![],
        vec![vec![1]],
        vec![vec![1, 0], vec![0]],
        vec![vec![1, 1], vec![2, 2]],
    ] {
        assert!(matches!(
            GroupId::new(&[A(2)], 0, &rows),
            Err(GroupError::InvalidLattice { .. })
        ));
    }
    assert!(matches!(
        GroupId::new(&[A(1)], 0, &[vec![4]]),
        Err(GroupError::InvalidLattice { .. })
    ));
    // B/C must use the crate's Dynkin orientation; swapping them omits a root.
    assert!(matches!(
        GroupId::new(&[B(2)], 0, &[vec![2, -1], vec![-2, 2]]),
        Err(GroupError::InvalidLattice { .. })
    ));
    assert!(matches!(
        GroupId::new(&[C(2)], 0, &[vec![2, -2], vec![-1, 2]]),
        Err(GroupError::InvalidLattice { .. })
    ));
    for root in [A(0), B(0), C(0), D(0), D(1)] {
        assert!(matches!(
            GroupId::cover(&[root], 0),
            Err(GroupError::UnsupportedRank { .. })
        ));
        assert!(matches!(
            GroupId::new(&[root], 0, &[]),
            Err(GroupError::UnsupportedRank { .. })
        ));
    }
    for result in [
        GroupId::torus(0),
        GroupId::u(0),
        GroupId::so(2),
        GroupId::spin(2),
    ] {
        assert!(matches!(result, Err(GroupError::UnsupportedRank { .. })));
    }
}

#[test]
fn overflowing_rank_sums_dense_sizes_and_entries_return_size_overflow() {
    for result in [
        GroupId::u(usize::MAX),
        GroupId::cover(&[A(1)], usize::MAX),
        GroupId::cover(&[A(usize::MAX), B(1)], 0),
        GroupId::new(&[], usize::MAX, &[]),
    ] {
        assert_eq!(result, Err(GroupError::SizeOverflow));
    }
    // det = MAX² − MAX + 1: the HNF's second pivot is not representable.
    let m = i64::MAX;
    assert_eq!(
        GroupId::new(&[], 2, &[vec![m, 1], vec![m - 1, m]]),
        Err(GroupError::SizeOverflow)
    );
    assert_eq!(
        GroupId::new(&[], 1, &[vec![i64::MIN]]),
        Err(GroupError::SizeOverflow)
    );
    // Euclidean intermediates are exact even with signed endpoint entries.
    let g = GroupId::new(&[], 2, &[vec![m, m - 1], vec![-m + 1, -m + 2]]).unwrap();
    assert_eq!(g, GroupId::torus(2).unwrap()); // determinant 1
}

#[test]
fn exceptional_classification_uses_bourbaki_roots() {
    // Independent Bourbaki E6 Cartan rows (branch 2 at node 4).
    let q6 = [
        vec![2, 0, -1, 0, 0, 0],
        vec![0, 2, 0, -1, 0, 0],
        vec![-1, 0, 2, -1, 0, 0],
        vec![0, -1, -1, 2, -1, 0],
        vec![0, 0, 0, -1, 2, -1],
        vec![0, 0, 0, 0, -1, 2],
    ];
    let adjoint = GroupId::new(&[E6], 0, &q6).unwrap();
    let cover = GroupId::cover(&[E6], 0).unwrap();
    // E6 central character: a1 − a3 + a5 − a6 mod 3.
    for mut code in 0..3usize.pow(6) {
        let a: Vec<_> = (0..6)
            .map(|_| {
                let x = (code % 3) as i64;
                code /= 3;
                x
            })
            .collect();
        assert!(cover.admits(&a, &[]));
        assert_eq!(
            adjoint.admits(&a, &[]),
            (a[0] - a[2] + a[4] - a[5]).rem_euclid(3) == 0
        );
    }
    let h = adjoint.character_lattice();
    assert_eq!((0..6).map(|i| h[i][i]).product::<i64>(), 3);
    for (root, rows, index) in [
        (G2, vec![vec![2, -1], vec![-3, 2]], 1),
        (
            F4,
            vec![
                vec![2, -1, 0, 0],
                vec![-1, 2, -2, 0],
                vec![0, -1, 2, -1],
                vec![0, 0, -1, 2],
            ],
            1,
        ),
    ] {
        let g = GroupId::new(&[root], 0, &rows).unwrap();
        assert_eq!(g, GroupId::cover(&[root], 0).unwrap());
        let h = g.character_lattice();
        assert_eq!((0..root.rank()).map(|i| h[i][i]).product::<i64>(), index);
    }
    // Extend the independent E6 diagram by its terminal chain for E7/E8.
    for (root, index) in [(E7, 2), (E8, 1)] {
        let n = root.rank();
        let mut rows = vec![vec![0; n]; n];
        for i in 0..6 {
            rows[i][..6].copy_from_slice(&q6[i]);
        }
        for i in 6..n {
            rows[i][i] = 2;
            rows[i][i - 1] = -1;
            rows[i - 1][i] = -1;
        }
        let g = GroupId::new(&[root], 0, &rows).unwrap();
        let h = g.character_lattice();
        assert_eq!((0..n).map(|i| h[i][i]).product::<i64>(), index);
        for i in 0..n {
            let mut a = vec![0; n];
            a[i] = 1;
            assert_eq!(
                g.admits(&a, &[]),
                root == E8 || (a[1] + a[4] + a[6]) % 2 == 0
            );
        }
    }
    for g in [cover, adjoint] {
        assert_eq!(
            LieGroup::new(g),
            Err(LieError::Unsupported {
                requires: "E6 coefficient engine"
            })
        );
    }
}

#[test]
fn tori_trivial_and_exceptional_data_remain_unsupported() {
    let empty = GroupId::new(&[], 0, &[]).unwrap();
    assert!(empty.admits(&[], &[]));
    assert!(empty.character_lattice().is_empty());
    assert!(!empty.admits(&[0], &[]));
    // Three independent even-charge conditions exercise generic membership.
    let torus = GroupId::new(&[], 3, &[vec![2, 1, 1], vec![0, 2, 1], vec![0, 0, 2]]).unwrap();
    for a in -3..=3 {
        for b in -3..=3 {
            for c in -3..=3 {
                let x = a / 2;
                let y = (b - x) / 2;
                // Solve the hand-written upper row basis from its first coordinate.
                let yes = a % 2 == 0 && (b - x) % 2 == 0 && (c - x - y) % 2 == 0;
                assert_eq!(torus.admits(&[], &[a, b, c]), yes);
            }
        }
    }
    let coupled = GroupId::new(&[A(1), A(1)], 0, &[vec![1, 1], vec![0, 2]]).unwrap();
    assert!(coupled.admits(&[1, 1], &[]));
    assert!(!coupled.admits(&[1, 0], &[]));
    for (g, requires) in [
        (empty, "trivial group"),
        (GroupId::u(2).unwrap(), "torus"),
        (GroupId::torus(1).unwrap(), "torus"),
        (GroupId::cover(&[G2], 0).unwrap(), "G2 coefficient engine"),
        (GroupId::cover(&[F4], 0).unwrap(), "F4 coefficient engine"),
        (GroupId::cover(&[E7], 0).unwrap(), "E7 coefficient engine"),
        (GroupId::cover(&[E8], 0).unwrap(), "E8 coefficient engine"),
    ] {
        let expected = LieError::Unsupported { requires };
        assert_eq!(LieGroup::new(g.clone()), Err(expected.clone()));
        assert_eq!(LieGroup::with_catalog_budget(g, 0), Err(expected));
    }
}

#[test]
fn label_errors_precede_admission_then_engine_limits() {
    let g = LieGroup::new(GroupId::psu(2).unwrap()).unwrap();
    assert_eq!(g.irrep(&[i64::MAX], &[0]), Err(LieError::InvalidLabel));
    assert_eq!(g.irrep(&[-1], &[]), Err(LieError::InvalidLabel));
    assert_eq!(g.irrep(&[i64::MAX], &[]), Err(LieError::NotAdmissible));
    assert_eq!(g.irrep(&[i64::MAX - 1], &[]), Err(LieError::LabelOverflow));
    #[cfg(feature = "lie-generated")]
    {
        let g = LieGroup::new(GroupId::su_quotient(4, 2).unwrap()).unwrap();
        assert_eq!(
            g.irrep(&[i64::MAX, i64::MAX, 0], &[]),
            Err(LieError::NotAdmissible)
        );
        assert_eq!(
            g.irrep(&[i64::MAX, 0, i64::MAX], &[]),
            Err(LieError::LabelOverflow)
        );
    }
}

#[test]
fn hnf_membership_matches_two_by_two_determinants() {
    // Cramer's rule on the supplied row basis is independent of HNF reduction.
    for a in -2..=2 {
        for b in -2..=2 {
            for c in -2..=2 {
                for d in -2..=2 {
                    let det: i64 = a * d - b * c;
                    if det == 0 {
                        continue;
                    }
                    let g = GroupId::new(&[], 2, &[vec![a, b], vec![c, d]]).unwrap();
                    let h = g.character_lattice();
                    assert_eq!(h[0][0] * h[1][1], det.abs());
                    assert_eq!(h[1][0], 0);
                    assert!((0..h[1][1]).contains(&h[0][1]));
                    for x in -2..=2 {
                        for y in -2..=2 {
                            assert_eq!(
                                g.admits(&[], &[x, y]),
                                (x * d - y * c) % det == 0 && (a * y - b * x) % det == 0
                            );
                        }
                    }
                    assert_eq!(
                        g,
                        GroupId::new(&[], 2, &[vec![a + c, b + d], vec![-c, -d]]).unwrap()
                    );
                }
            }
        }
    }
    let scaled = GroupId::new(&[], 1, &[vec![4]]).unwrap();
    assert!(scaled.admits(&[], &[i64::MIN]));
    assert!(!scaled.admits(&[], &[-1]));
    assert_ne!(scaled, GroupId::torus(1).unwrap());
}

#[test]
fn unaddressable_su_weight_returns_a_typed_error() {
    // Only scalars are constructed: the internal weight would need N i64s.
    let group = GroupId::su(isize::MAX as usize / 8 + 1).unwrap();
    for result in [
        LieGroup::new(group.clone()),
        LieGroup::with_catalog_budget(group.clone(), usize::MAX),
    ] {
        #[cfg(feature = "lie-generated")]
        assert!(matches!(result, Err(LieError::LabelOverflow)), "{result:?}");
        #[cfg(not(feature = "lie-generated"))]
        assert!(
            matches!(
                result,
                Err(LieError::Unsupported {
                    requires: "lie-generated"
                })
            ),
            "{result:?}"
        );
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn huge_bcd_seeds_return_a_typed_error() {
    let r = 1usize << 40;
    for group in [
        GroupId::spin(2 * r + 1),
        GroupId::so(2 * r + 1),
        GroupId::sp(2 * r),
        GroupId::spin(2 * r),
        GroupId::so(2 * r),
    ] {
        let group = group.unwrap();
        for result in [
            LieGroup::new(group.clone()),
            LieGroup::with_catalog_budget(group.clone(), usize::MAX),
        ] {
            #[cfg(feature = "lie-generated")]
            assert!(
                matches!(result, Err(LieError::Generation(_))),
                "{group:?}: {result:?}"
            );
            #[cfg(not(feature = "lie-generated"))]
            assert!(
                matches!(
                    result,
                    Err(LieError::Unsupported {
                        requires: "lie-generated"
                    })
                ),
                "{group:?}: {result:?}"
            );
        }
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn unaddressable_named_lattices_return_size_overflow() {
    for result in [
        GroupId::psu(1usize << 62),
        GroupId::psp(1usize << 63),
        GroupId::half_spin_plus(1usize << 63),
    ] {
        assert_eq!(result, Err(GroupError::SizeOverflow));
    }
}

#[test]
fn large_named_lattices_construct_and_compare_without_dense_storage() {
    let huge = 1 << 20;
    for (construct, semisimple, torus) in [
        (
            GroupId::su as fn(usize) -> Result<GroupId, GroupError>,
            huge - 1,
            0,
        ),
        (GroupId::psu, huge - 1, 0),
        (GroupId::u, huge - 1, 1),
        (GroupId::so, huge / 2, 0),
        (GroupId::psp, huge / 2, 0),
        (GroupId::pso, huge / 2, 0),
        (GroupId::half_spin_plus, huge / 2, 0),
        (GroupId::half_spin_minus, huge / 2, 0),
    ] {
        let g = construct(huge).unwrap();
        assert_eq!(g, construct(huge).unwrap());
        assert_eq!(g.semisimple_rank(), semisimple);
        assert_eq!(g.torus_rank(), torus);
    }
    let so = GroupId::so(1 << 21).unwrap();
    assert_eq!(so, GroupId::so(1 << 21).unwrap());
    assert_eq!(so.semisimple_rank(), huge);
    assert_eq!(so.torus_rank(), 0);
    assert!(GroupId::su(usize::MAX).is_ok());
    assert!(GroupId::torus(usize::MAX).is_ok());
    assert!(GroupId::sp(1usize << (usize::BITS - 2)).is_ok());
    for group in [GroupId::su(huge).unwrap(), GroupId::psu(huge).unwrap()] {
        let result = LieGroup::new(group);
        #[cfg(feature = "lie-generated")]
        assert!(result.is_ok());
        #[cfg(not(feature = "lie-generated"))]
        assert!(matches!(
            result,
            Err(LieError::Unsupported {
                requires: "lie-generated"
            })
        ));
    }
}

#[test]
fn named_lattices_contain_roots_across_ranks() {
    for n in 2..=24 {
        let mut groups = vec![GroupId::psu(n).unwrap(), GroupId::u(n).unwrap()];
        if n >= 3 {
            groups.extend([GroupId::so(n).unwrap(), GroupId::pso(n).unwrap()]);
        }
        if n % 2 == 0 {
            groups.push(GroupId::psp(n).unwrap());
        }
        if n >= 8 && n % 4 == 0 {
            groups.extend([
                GroupId::half_spin_plus(n).unwrap(),
                GroupId::half_spin_minus(n).unwrap(),
            ]);
        }
        for group in groups {
            // General validation checks every Cartan row, including signed roots.
            let validated = GroupId::new(
                group.root_systems(),
                group.torus_rank(),
                &group.character_lattice(),
            )
            .unwrap();
            assert_eq!(group, validated);
        }
    }
}

#[test]
fn sparse_membership_falls_back_to_exact_large_intermediates() {
    let m = i64::MAX;
    let g = GroupId::new(
        &[],
        5,
        &[
            vec![1, 0, 0, m - 1, 0],
            vec![0, 1, 0, m - 1, 0],
            vec![0, 0, 1, m - 1, 0],
            vec![0, 0, 0, m, m - 1],
            vec![0, 0, 0, 0, m],
        ],
    )
    .unwrap();
    // The fourth row coefficient is -3(m-1); its contribution to the
    // fifth coordinate exceeds i128. Modulo m it is -3, so charge -3 cancels.
    assert!(g.admits(&[], &[m, m, m, 0, -3]));
    assert!(!g.admits(&[], &[m, m, m, 0, -2]));
}
