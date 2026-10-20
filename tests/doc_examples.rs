//! Runnable copies of the Rust examples in the public guides.
#![cfg(feature = "cgc-gen")]

/// `README.md`, Rust block 1.
#[test]
fn readme_1() {
    use fusion_symbols::lie::su2::wigner_6j;

    let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
    assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
}

/// `docs/overview.md`, Rust block 1.
#[test]
fn overview_1() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

    let eight = cat.irrep(&[1, 1]).unwrap(); // SU(3) adjoint
    assert_eq!(eight.dim_exact(), 8u32.into());
    assert_eq!(
        cat.fusion(&eight, &eight)
            .unwrap()
            .map(|c| {
                let n = cat.nsymbol(&eight, &eight, &c);
                (c, n)
            })
            .collect::<std::collections::BTreeMap<_, _>>()[&eight],
        2
    );
}

/// `docs/overview.md`, Rust block 2.
#[test]
fn overview_2() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };

    let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap(); // Sp(4)
    let triv = cat.unit();
    let five = cat.irrep(&[0, 1]).unwrap(); // the 5
    let ten = cat.irrep(&[2, 0]).unwrap(); // the adjoint 10

    let block = cat
        .f_symbol(&triv, &five, &five, &ten, &five, &ten)
        .unwrap();
    assert_eq!(block.dims(), [1, 1, 1, 1]);
    assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
}

/// `docs/overview.md`, Rust block 3.
#[test]
fn overview_3() {
    use fusion_symbols::lie::{GroupId, LieGroup};

    let spin7 = GroupId::spin(7).unwrap();
    let s = LieGroup::new(spin7).unwrap().irrep(&[0, 0, 1]).unwrap();
    assert_eq!(s.dim_exact(), 8u32.into());

    // The same label is not a representation of SO(7).
    assert!(LieGroup::new(GroupId::so(7).unwrap())
        .unwrap()
        .irrep(&[0, 0, 1])
        .is_err());
}

/// `docs/user-guide/clebsch-gordan.md`, Rust block 1.
#[test]
fn clebsch_gordan_1() {
    use fusion_symbols::lie::su2::clebsch_gordan;

    // ⟨½ +½, ½ −½ | 0 0⟩ = 1/√2, in doubled labels.
    let cg = clebsch_gordan(1, 1, 1, -1, 0, 0);
    assert!((cg.to_f64() - 0.5f64.sqrt()).abs() < 1e-15);
}

/// `docs/user-guide/clebsch-gordan.md`, Rust block 2.
#[test]
fn clebsch_gordan_2() {
    use fusion_symbols::{
        category::{FusionCategory, GroupRepresentations},
        lie::{GroupId, LieGroup},
    };
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

    let three = cat.irrep(&[1, 0]).unwrap();
    let anti = cat.dual(&three);
    let eight = cat.irrep(&[1, 1]).unwrap();

    let c = cat.cgc(&three, &anti, &eight).unwrap();
    assert_eq!(c.dims(), [3, 3, 8, 1]); // [dim(s1), dim(s2), dim(s3), N^{s3}_{s1 s2}]
    assert_eq!(c.dims()[3], 1);
    assert!((c.at(0, 0, 0, 0) - 1.0).abs() < 1e-12);
}

/// `docs/user-guide/clebsch-gordan.md`, Rust block 3.
#[test]
fn clebsch_gordan_3() {
    use fusion_symbols::{
        category::GroupRepresentations,
        lie::{GroupId, LieGroup},
    };
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

    let eight = cat.irrep(&[1, 1]).unwrap();
    let c = cat.cgc(&eight, &eight, &eight).unwrap();
    assert_eq!(c.dims()[3], 2); // the 8 occurs twice in 8 ⊗ 8
    assert_eq!(c.dims(), [8, 8, 8, 2]);
    assert!(c.at(0, 2, 0, 1) < 0.0);
}

/// `docs/user-guide/clebsch-gordan.md`, Rust block 4.
#[test]
fn clebsch_gordan_4() {
    use fusion_symbols::{
        category::GroupRepresentations,
        lie::{GroupId, LieGroup},
    };

    let cat = LieGroup::new(GroupId::so(5).unwrap()).unwrap(); // SO(5) / Spin(5)
    let v = cat.irrep(&[1, 0]).unwrap(); // the 5
    let adj = cat.irrep(&[0, 2]).unwrap(); // the 10

    let c = cat.cgc(&v, &v, &adj).unwrap();
    assert_eq!(c.dims()[3], 1);
    assert_eq!((c.dims()[0] * c.dims()[1], c.dims()[2]), (25, 10)); // (dim(s1)·dim(s2), dim(s3))
}

/// `docs/user-guide/fusion.md`, Rust block 1.
#[test]
fn fusion_1() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let group = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = group.irrep(&[1]).unwrap();
    let channels: Vec<i64> = group
        .fusion(&half, &half)
        .unwrap()
        .map(|s| s.dynkin()[0])
        .collect();
    assert_eq!(channels, vec![0, 2]);
}

/// `docs/user-guide/fusion.md`, Rust block 2.
#[test]
fn fusion_2() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

    let three = cat.irrep(&[1, 0]).unwrap();
    let anti = cat.dual(&three);
    let out = cat
        .fusion(&three, &anti)
        .unwrap()
        .map(|c| {
            let n = cat.nsymbol(&three, &anti, &c);
            (c, n)
        })
        .collect::<std::collections::BTreeMap<_, _>>();

    // 3 ⊗ 3-bar = 1 ⊕ 8, each once.
    assert_eq!(out.len(), 2);
    assert_eq!(out[&cat.unit()], 1);
    assert_eq!(out[&cat.irrep(&[1, 1]).unwrap()], 1);
}

/// `docs/user-guide/fusion.md`, Rust block 3.
#[test]
fn fusion_3() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

    let eight = cat.irrep(&[1, 1]).unwrap(); // SU(3) adjoint
    let out = cat
        .fusion(&eight, &eight)
        .unwrap()
        .map(|c| {
            let n = cat.nsymbol(&eight, &eight, &c);
            (c, n)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(out[&eight], 2); // the 8 appears twice in 8 ⊗ 8
}

/// `docs/user-guide/fusion.md`, Rust block 4.
#[test]
fn fusion_4() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };

    // SO(5) = B_2: 5 ⊗ 5 = 1 ⊕ 10 ⊕ 14.
    let cat = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
    let v = cat.irrep(&[1, 0]).unwrap();
    let out = cat
        .fusion(&v, &v)
        .map(|it| {
            it.map(|c| {
                let n = cat.nsymbol(&v, &v, &c);
                (c, n)
            })
            .collect::<std::collections::BTreeMap<_, _>>()
        })
        .unwrap();
    assert_eq!(out.len(), 3);
    for (irrep, mult) in &out {
        assert_eq!(*mult, 1);
        let _ = irrep.dim_exact();
    }
}

/// `docs/user-guide/fusion.md`, Rust block 5.
#[test]
fn fusion_5() {
    use fusion_symbols::{
        category::FusionCategory,
        cu1::{CU1Irrep, CU1},
    };

    let a = CU1Irrep::from_twice_charge(1);
    let channels: Vec<_> = CU1.fusion(&a, &a).unwrap().collect();
    assert_eq!(
        channels,
        vec![
            CU1Irrep::VACUUM,
            CU1Irrep::PSEUDOSCALAR,
            CU1Irrep::from_twice_charge(2)
        ]
    );
}

/// `docs/user-guide/getting-started.md`, Rust block 1.
#[test]
fn getting_started_1() {
    use fusion_symbols::lie::su2::wigner_6j;

    let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
    assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
}

/// `docs/user-guide/getting-started.md`, Rust block 2.
#[test]
fn getting_started_2() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

    let fund = cat.irrep(&[1, 0]).unwrap(); // the 3 of SU(3)
    assert_eq!(fund.dim_exact(), 3u32.into());
    assert_eq!(cat.dual(&fund).dynkin(), vec![0, 1]); // the 3-bar
}

/// `docs/user-guide/getting-started.md`, Rust block 3.
#[test]
fn getting_started_3() {
    use fusion_symbols::lie::su2::{wigner_6j_checked, Su2Error};

    // Admissible, and genuinely nonzero.
    assert!(wigner_6j_checked(2, 2, 2, 2, 2, 2).is_ok());

    // Triangle violation: not a real zero, a forbidden label set.
    assert!(matches!(
        wigner_6j_checked(2, 2, 20, 2, 2, 2),
        Err(Su2Error::NotAdmissible(_))
    ));
}

/// `docs/user-guide/groups.md`, Rust block 1.
#[test]
fn groups_1() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let group = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let one = group.irrep(&[2]).unwrap();
    let channels: Vec<i64> = group
        .fusion(&one, &one)
        .unwrap()
        .map(|s| s.dynkin()[0])
        .collect();
    assert_eq!(channels, vec![0, 2, 4]);
}

/// `docs/user-guide/groups.md`, Rust block 2.
#[test]
fn groups_2() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

    let eight = cat.irrep(&[1, 1]).unwrap(); // SU(3) adjoint
    let decomposition = cat
        .fusion(&eight, &eight)
        .unwrap()
        .map(|c| {
            let n = cat.nsymbol(&eight, &eight, &c);
            (c, n)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    // 8 ⊗ 8 = 1 ⊕ 8 ⊕ 8 ⊕ 10 ⊕ 10-bar ⊕ 27 — note the 8 appears twice.
    assert_eq!(decomposition[&eight], 2);
}

/// `docs/user-guide/groups.md`, Rust block 3.
#[test]
fn groups_3() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };

    // SO(5) = B_2: 5 ⊗ 5 = 1 ⊕ 10 ⊕ 14.
    let cat = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
    let v = cat.irrep(&[1, 0]).unwrap();
    let out = cat
        .fusion(&v, &v)
        .map(|it| {
            it.map(|c| {
                let n = cat.nsymbol(&v, &v, &c);
                (c, n)
            })
            .collect::<std::collections::BTreeMap<_, _>>()
        })
        .unwrap();
    let mut dims: Vec<String> = out.keys().map(|s| s.dim_exact().to_string()).collect();
    dims.sort();
    assert_eq!(dims, vec!["1", "10", "14"]);
}

/// `docs/user-guide/groups.md`, Rust block 4.
#[test]
fn groups_4() {
    use fusion_symbols::lie::{GroupId, LieGroup};

    let spin5 = GroupId::spin(5).unwrap();
    let s = LieGroup::new(spin5).unwrap().irrep(&[0, 1]).unwrap();
    assert_eq!(s.dim_exact(), 4u32.into()); // the Spin(5) Dirac spinor

    // The same label is not a representation of SO(5).
    let so5 = GroupId::so(5).unwrap();
    assert!(LieGroup::new(so5).unwrap().irrep(&[0, 1]).is_err());
}

/// `docs/user-guide/groups.md`, Rust block 5.
#[test]
fn groups_5() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };

    // Sp(4) = C_2. The defining 4, and 4 ⊗ 4 = 1 ⊕ 5 ⊕ 10.
    let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
    let four = cat.irrep(&[1, 0]).unwrap();
    assert_eq!(four.dim_exact(), 4u32.into());
    let out = cat
        .fusion(&four, &four)
        .map(|it| {
            it.map(|c| {
                let n = cat.nsymbol(&four, &four, &c);
                (c, n)
            })
            .collect::<std::collections::BTreeMap<_, _>>()
        })
        .unwrap();
    let mut dims: Vec<String> = out.keys().map(|s| s.dim_exact().to_string()).collect();
    dims.sort();
    assert_eq!(dims, vec!["1", "10", "5"]); // string sort
}

/// `docs/user-guide/groups.md`, Rust block 6.
#[test]
fn groups_6() {
    use fusion_symbols::lie::{GroupId, LieGroup};
    let psu3 = GroupId::psu(3).unwrap(); // SU(3)/Z₃

    // The adjoint 8 has zero triality, so it is a genuine PSU(3) representation.
    assert!(LieGroup::new(psu3).unwrap().irrep(&[1, 1]).is_ok());

    // The fundamental 3 has triality 1: PSU(3) has no such representation.
    assert!(LieGroup::new(psu3).unwrap().irrep(&[1, 0]).is_err());
}

/// `docs/user-guide/recoupling.md`, Rust block 1.
#[test]
fn recoupling_1() {
    use fusion_symbols::abelian::{ZNIrrep, ZN};
    use fusion_symbols::category::{FusionCategory, SymbolBlock};

    let category = ZN::new(3).unwrap();
    let one = ZNIrrep::from_canonical(1, 3).unwrap();
    let two = ZNIrrep::from_canonical(2, 3).unwrap();
    assert_eq!(category.fusion(&one, &one).unwrap().next(), Some(two));
    assert_eq!(category.r_symbol(&one, &one, &two), Ok(1.0));
    let zero = category
        .f_symbol(&one, &one, &one, &one, &one, &one)
        .unwrap();
    assert_eq!(zero, 0.0); // an inadmissible vertex is a value
    assert_eq!(<f64 as SymbolBlock<4>>::dims(&zero), [1; 4]);
}

/// `docs/user-guide/recoupling.md`, Rust block 2.
#[test]
fn recoupling_2() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{su2::wigner_6j, GroupId, LieGroup},
    };
    assert!((wigner_6j(2, 2, 2, 2, 2, 2).to_f64() - 1.0 / 6.0).abs() < 1e-14);
    let group = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = group.irrep(&[1]).unwrap();
    let unit = group.unit();
    let f = group
        .f_symbol(&half, &half, &half, &half, &unit, &unit)
        .unwrap();
    assert_eq!(f.at(0, 0, 0, 0), -0.5);
    let r = group.r_symbol(&half, &half, &unit).unwrap();
    assert_eq!(r.at(0, 0), -1.0);
}

/// `docs/user-guide/recoupling.md`, Rust block 3.
#[test]
fn recoupling_3() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

    let three = cat.irrep(&[1, 0]).unwrap();
    let anti = cat.dual(&three);
    let eight = cat.irrep(&[1, 1]).unwrap();

    // F^{3 3bar 3}_{3}[8, 8]
    let block = cat
        .f_symbol(&three, &anti, &three, &three, &eight, &eight)
        .unwrap();
    assert_eq!(block.dims(), [1, 1, 1, 1]);
    assert!((block.at(0, 0, 0, 0) - 1.0 / 3.0).abs() < 1e-12);

    // R^{3 3bar}_{8}
    let r = cat.r_symbol(&three, &anti, &eight).unwrap();
    assert_eq!(r.dim(), 1);
    assert!((r.at(0, 0).abs() - 1.0).abs() < 1e-12);
}

/// `docs/user-guide/recoupling.md`, Rust block 4.
#[test]
fn recoupling_4() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };

    let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap(); // Sp(4) = C_2
    let triv = cat.unit();
    let five = cat.irrep(&[0, 1]).unwrap(); // the 5
    let ten = cat.irrep(&[2, 0]).unwrap(); // the adjoint 10

    let block = cat
        .f_symbol(&triv, &five, &five, &ten, &five, &ten)
        .unwrap();
    assert_eq!(block.dims(), [1, 1, 1, 1]);
    assert!((block.at(0, 0, 0, 0) - 1.0).abs() < 1e-9);
}

/// `docs/user-guide/recoupling.md`, Rust block 5.
#[test]
fn recoupling_5() {
    use fusion_symbols::{
        category::{FusionCategory, GroupRepresentations},
        lie::{GroupId, LieGroup, LieIrrep},
    };
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

    let eight = cat.irrep(&[1, 1]).unwrap();
    let dim = 8usize;

    // N^8_88 = 2, so every multiplicity axis below has length 2.
    assert_eq!(
        cat.fusion(&eight, &eight)
            .unwrap()
            .map(|c| {
                let n = cat.nsymbol(&eight, &eight, &c);
                (c, n)
            })
            .collect::<std::collections::BTreeMap<_, _>>()[&eight],
        2
    );

    // Materialize a CGC into (row-major buffer, dims), indexed [m1][m2][m3][mu].
    let dense = |s1: &LieIrrep, s2: &LieIrrep, s3: &LieIrrep| {
        let t = cat.cgc(s1, s2, s3).unwrap();
        let [d1, d2, d3, n] = t.dims();
        let mut buf = vec![0.0f64; d1 * d2 * d3 * n];
        for m1 in 0..d1 {
            for m2 in 0..d2 {
                for m3 in 0..d3 {
                    for mu in 0..n {
                        buf[((m1 * d2 + m2) * d3 + m3) * n + mu] = t.at(m1, m2, m3, mu);
                    }
                }
            }
        }
        (buf, [d1, d2, d3, n])
    };
    let at = |t: &(Vec<f64>, [usize; 4]), i: usize, j: usize, k: usize, l: usize| {
        t.0[((i * t.1[1] + j) * t.1[2] + k) * t.1[3] + l]
    };

    // Left tree, e = 8: C^{ab->e} carries mu, C^{ec->d} carries nu (both [8, 8, 8, 2]).
    let left = dense(&eight, &eight, &eight);

    // Right tree: for every channel f of 8 x 8, the F block F^{888}_8[e=8, f]
    // (shape [2, 2, N^f_88, N^8_8f]) and the two CGC carrying kappa and lambda.
    let right: Vec<_> = cat
        .fusion(&eight, &eight)
        .unwrap()
        .map(|c| {
            let n = cat.nsymbol(&eight, &eight, &c);
            (c, n)
        })
        .collect::<std::collections::BTreeMap<_, _>>()
        .keys()
        .map(|f| {
            (
                cat.f_symbol(&eight, &eight, &eight, &eight, &eight, f)
                    .unwrap(),
                dense(&eight, &eight, f), // C^{bc->f}[mb, mc, mf, kappa]
                dense(&eight, f, &eight), // C^{af->d}[ma, mf, md, lambda]
            )
        })
        .collect();

    // The F-move, one magnetic assignment at a time:
    //   sum_me C^{ab->e}_mu C^{ec->d}_nu
    //     = sum_f sum_{kappa, lambda} F[mu, nu, kappa, lambda] sum_mf C^{bc->f}_kappa C^{af->d}_lambda
    let mut worst = 0.0f64;
    for ma in 0..dim {
        for mb in 0..dim {
            for mc in 0..dim {
                for md in 0..dim {
                    for mu in 0..2 {
                        for nu in 0..2 {
                            let mut lhs = 0.0;
                            for me in 0..dim {
                                lhs += at(&left, ma, mb, me, mu) * at(&left, me, mc, md, nu);
                            }
                            let mut rhs = 0.0;
                            for (fblock, c_bcf, c_afd) in &right {
                                let [_, _, n_kappa, n_lambda] = fblock.dims();
                                let dim_f = c_bcf.1[2];
                                for kappa in 0..n_kappa {
                                    for lambda in 0..n_lambda {
                                        let mut pair = 0.0;
                                        for mf in 0..dim_f {
                                            pair += at(c_bcf, mb, mc, mf, kappa)
                                                * at(c_afd, ma, mf, md, lambda);
                                        }
                                        rhs += fblock.at(mu, nu, kappa, lambda) * pair;
                                    }
                                }
                            }
                            worst = worst.max((lhs - rhs).abs());
                        }
                    }
                }
            }
        }
    }
    assert!(worst < 1e-9, "F-move residual {worst}");
}

/// `docs/user-guide/recoupling.md`, Rust block 6.
#[test]
fn recoupling_6() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

    let three = cat.irrep(&[1, 0]).unwrap();
    let anti = cat.dual(&three);
    assert!(
        fusion_symbols::category::f_unitarity_residual(&cat, &three, &anti, &three, &three)
            .unwrap()
            < 1e-10
    );
    assert!(
        fusion_symbols::category::pentagon_residual(&cat, &three, &anti, &three, &anti).unwrap()
            < 1e-10
    );
    assert!(
        fusion_symbols::category::hexagon_residual(&cat, &three, &anti, &three).unwrap() < 1e-10
    );
}

/// `docs/user-guide/representations.md`, Rust block 1.
#[test]
fn representations_1() {
    use fusion_symbols::{category::FusionCategory, table::TableCategory};

    let fib = TableCategory::fibonacci();
    let tau = fib.object(2).unwrap();
    assert_eq!(tau.index(), 2);
    assert_eq!(fib.fusion(&tau, &tau).unwrap().count(), 2);
}

/// `docs/user-guide/representations.md`, Rust block 2.
#[test]
fn representations_2() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let group = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let s = group.irrep(&[3]).unwrap(); // doubled spin 3, i.e. spin 3/2
    assert_eq!(s.dynkin(), vec![3]);
    assert_eq!(s.dim_exact(), 4.into());
    assert_eq!(group.dual(&s), s);
    assert_eq!(group.frobenius_schur_phase(&s), -1.0);
}

/// `docs/user-guide/representations.md`, Rust block 3.
#[test]
fn representations_3() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();

    let three = cat.irrep(&[1, 0]).unwrap(); // SU(3) fundamental
    let eight = cat.irrep(&[1, 1]).unwrap(); // SU(3) adjoint
    let singlet = cat.unit();
    assert_eq!(singlet.dim_exact(), 1.into());

    assert_eq!(three.dim_exact(), 3u32.into());
    assert_eq!(eight.dim_exact(), 8u32.into());
    assert_eq!(cat.dual(&three).dynkin(), vec![0, 1]); // 3-bar
    assert_eq!(cat.dual(&eight), eight); // the adjoint is self-dual
}

/// `docs/user-guide/representations.md`, Rust block 4.
#[test]
fn representations_4() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };

    // SO(5) = B_2. Dynkin [1, 0] is the 5-dimensional vector representation.
    let cat = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
    let v = cat.irrep(&[1, 0]).unwrap();
    assert_eq!(v.dim_exact(), 5u32.into());
    assert_eq!(cat.dual(&v), v);
    assert_eq!(cat.frobenius_schur_indicator(&v), 1.0); // real (orthogonal) self-duality
}

/// `docs/user-guide/representations.md`, Rust block 5.
#[test]
fn representations_5() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };

    // Sp(4) = C_2. Its defining 4 is pseudo-real; the 5 is real.
    let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
    let four = cat.irrep(&[1, 0]).unwrap();
    let five = cat.irrep(&[0, 1]).unwrap();
    assert_eq!(four.dim_exact(), 4u32.into());
    assert_eq!(cat.frobenius_schur_indicator(&four), -1.0);
    assert_eq!(five.dim_exact(), 5u32.into());
    assert_eq!(cat.frobenius_schur_indicator(&five), 1.0);
}

/// `docs/user-guide/resources.md`, Rust block 1.
#[test]
fn resources_1() {
    use fusion_symbols::cache::{
        configure_cache_budgets, CoefficientCacheBudgets, CoefficientCacheTier,
    };

    let budgets =
        CoefficientCacheBudgets::default().with_limit(CoefficientCacheTier::SixJ, 1 << 20); // 1 MiB of 6j
    let _ = configure_cache_budgets(budgets);
}

/// `docs/user-guide/resources.md`, Rust block 2.
#[test]
fn resources_2() {
    use fusion_symbols::cache::{base_cache_stats, reset, trim_to, CoefficientCacheTier};

    let stats = base_cache_stats();
    let _ = stats.six_j.entries; // per-tier entries / bytes / hits / misses / evictions
    let _ = stats.total(); // field-wise sum across the base tiers

    // Release one tier down to a target charge, oldest entries first.
    let report = trim_to(CoefficientCacheTier::SixJ, 0);
    let _ = report.removed_entries;

    reset(); // all tiers: entries, bytes, counters to zero
}

/// `docs/user-guide/resources.md`, Rust block 3.
#[test]
fn resources_3() {
    use fusion_symbols::{
        category::FusionCategory,
        lie::{GroupId, LieGroup},
    };
    let group = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let fingerprint = group.authority_fingerprint();
    assert!(!fingerprint.is_empty());
}
