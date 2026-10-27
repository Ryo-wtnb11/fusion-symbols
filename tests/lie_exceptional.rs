//! G2 through the public `LieGroup` API (`docs/gauge_exceptional.md`), and the
//! explicit boundaries of the exceptional factors without coefficients.
//! Expected values are literals: Weyl dimensions and §9.3 decompositions.

use fusion_symbols::lie::{GroupId, LieError, LieGroup, RootSystem};

#[test]
fn exceptional_constructors_are_the_simply_connected_covers() {
    use RootSystem::*;
    for (id, root) in [
        (GroupId::g2(), G2),
        (GroupId::f4(), F4),
        (GroupId::e6(), E6),
        (GroupId::e7(), E7),
        (GroupId::e8(), E8),
    ] {
        assert_eq!(id, GroupId::cover(&[root], 0).unwrap());
    }
}

/// F4, E6, E7 and E8 have no coefficients yet; G2 has no B/C/D catalog.
#[test]
fn unsupported_boundaries() {
    for (id, requires) in [
        (GroupId::f4(), "F4 coefficient engine"),
        (GroupId::e6(), "E6 coefficient engine"),
        (GroupId::e7(), "E7 coefficient engine"),
        (GroupId::e8(), "E8 coefficient engine"),
    ] {
        assert_eq!(LieGroup::new(id), Err(LieError::Unsupported { requires }));
    }
    #[cfg(feature = "lie-generated")]
    for id in [
        GroupId::g2(),
        GroupId::cover(&[RootSystem::G2, RootSystem::A(1)], 1).unwrap(),
    ] {
        assert_eq!(
            LieGroup::with_catalog_budget(id, usize::MAX),
            Err(LieError::Unsupported {
                requires: "B/C/D catalog"
            })
        );
    }
    #[cfg(not(feature = "lie-generated"))]
    assert_eq!(
        LieGroup::new(GroupId::g2()),
        Err(LieError::Unsupported {
            requires: "lie-generated"
        })
    );
}

#[cfg(feature = "lie-generated")]
mod g2 {
    use fusion_symbols::category::{
        f_unitarity_residual, hexagon_residual, pentagon_residual, FusionCategory,
        GroupRepresentations, SymbolBlock,
    };
    use fusion_symbols::lie::{GroupId, LieGroup, LieIrrep, RootSystem};
    use fusion_symbols::product::Product;
    use num_bigint::BigInt;
    use std::time::Instant;

    fn g2() -> LieGroup {
        LieGroup::new(GroupId::g2()).unwrap()
    }

    fn l(g: &LieGroup, a: i64, b: i64) -> LieIrrep {
        g.irrep(&[a, b], &[]).unwrap()
    }

    /// `(dim, N)` of every channel of `a ⊗ b`, sorted.
    fn decomposition(g: &LieGroup, a: &LieIrrep, b: &LieIrrep) -> Vec<(u64, u32)> {
        let mut out: Vec<(u64, u32)> = g
            .fusion(a, b)
            .unwrap()
            .map(|c| (g.dim(&c) as u64, g.nsymbol(a, b, &c)))
            .collect();
        out.sort_unstable();
        out
    }

    fn bits<const R: usize>(x: impl SymbolBlock<R, Scalar = f64>) -> ([usize; R], Vec<u64>) {
        (x.dims(), x.as_slice().iter().map(|v| v.to_bits()).collect())
    }

    #[test]
    fn weyl_dimensions() {
        let g = g2();
        for (label, dim) in [
            ([0, 0], 1),
            ([1, 0], 7),
            ([0, 1], 14),
            ([2, 0], 27),
            ([1, 1], 64),
            ([0, 2], 77),
            ([3, 0], 77),
        ] {
            let a = g.irrep(&label, &[]).unwrap();
            assert_eq!(a.dim_exact(), BigInt::from(dim), "{label:?}");
            assert_eq!(g.dim(&a), f64::from(dim));
            assert_eq!(a.dynkin(), label);
            assert_eq!(a.dynkin_labels().collect::<Vec<_>>(), label);
            let weights = a.weight_multiplicities();
            assert_eq!(weights.values().sum::<u64>(), dim as u64);
        }
        assert_eq!(g.unit(), l(&g, 0, 0));
    }

    /// §9.3 literals: `7⊗7 = 1+7+14+27` and the first G2 product with an
    /// outer multiplicity 2, `27⊗27`.
    #[test]
    fn decompositions_match_the_spec() {
        let g = g2();
        let (v, x) = (l(&g, 1, 0), l(&g, 2, 0));
        assert_eq!(
            decomposition(&g, &v, &v),
            [(1, 1), (7, 1), (14, 1), (27, 1)]
        );
        assert_eq!(
            decomposition(&g, &x, &x),
            [
                (1, 1),
                (7, 1),
                (14, 1),
                (27, 2),
                (64, 2),
                (77, 1),
                (77, 1),
                (182, 1),
                (189, 1)
            ]
        );
        let labels: Vec<Vec<i64>> = g.fusion(&v, &v).unwrap().map(|c| c.dynkin()).collect();
        assert_eq!(labels, [[0, 0], [0, 1], [1, 0], [2, 0]]);
        let reversed: Vec<Vec<i64>> = g
            .fusion(&v, &v)
            .unwrap()
            .rev()
            .map(|c| c.dynkin())
            .collect();
        assert_eq!(reversed, [[2, 0], [1, 0], [0, 1], [0, 0]]);
        assert_eq!(g.nsymbol(&v, &v, &x), 1);
        assert_eq!(g.nsymbol(&v, &v, &l(&g, 1, 1)), 0);
    }

    /// §6.2: every G2 irrep is self-dual and real.
    #[test]
    fn self_dual_and_real() {
        let g = g2();
        for a in 0..4 {
            for b in 0..4 {
                let x = l(&g, a, b);
                assert_eq!(g.dual(&x), x);
                assert_eq!(g.frobenius_schur_indicator(&x), 1.0, "{a},{b}");
                assert_eq!(g.frobenius_schur_phase(&x), 1.0);
                assert_eq!(g.twist(&x), 1.0);
            }
        }
    }

    fn timed(gate: &str, run: impl FnOnce() -> f64) -> f64 {
        let t = Instant::now();
        let r = run();
        eprintln!("{gate} residual {r:e} in {:?}", t.elapsed());
        r
    }

    const TOL: f64 = 1e-12;

    #[test]
    fn gates_on_small_irreps() {
        let g = g2();
        let (v, adj) = (l(&g, 1, 0), l(&g, 0, 1));
        assert!(timed("P", || pentagon_residual(&g, &v, &v, &v, &v).unwrap()) <= TOL);
        assert!(timed("H", || hexagon_residual(&g, &v, &v, &adj).unwrap()) <= TOL);
        assert!(
            timed("F", || f_unitarity_residual(&g, &v, &adj, &v, &adj)
                .unwrap())
                <= TOL
        );
    }

    /// `F^{7,27,27}_{27}` runs through the `N = 2` vertex `27 ⊗ 27 → 27`.
    #[test]
    fn f_unitarity_through_a_multiplicity_two_vertex() {
        let g = g2();
        let (v, x) = (l(&g, 1, 0), l(&g, 2, 0));
        assert_eq!(g.nsymbol(&x, &x, &x), 2);
        assert!(timed("F", || f_unitarity_residual(&g, &v, &x, &x, &x).unwrap()) <= TOL);
    }

    /// The vertex `27 ⊗ 27 → 27` has `N = 2`.
    #[test]
    #[ignore = "release 14 s, debug 121 s"]
    fn gates_through_a_multiplicity_two_vertex() {
        let g = g2();
        let (v, x) = (l(&g, 1, 0), l(&g, 2, 0));
        assert_eq!(g.nsymbol(&x, &x, &x), 2);
        assert!(timed("F", || f_unitarity_residual(&g, &v, &x, &x, &x).unwrap()) <= TOL);
        assert!(timed("H", || hexagon_residual(&g, &x, &x, &v).unwrap()) <= TOL);
        assert!(timed("P", || pentagon_residual(&g, &v, &v, &x, &x).unwrap()) <= TOL);
    }

    /// Cold cost of one F and one R query on a fresh group per irrep.
    #[test]
    #[ignore = "timing report; run in release with --nocapture"]
    fn cold_query_report() {
        for (a, b) in [(1, 0), (0, 1), (2, 0)] {
            let g = g2();
            let x = l(&g, a, b);
            let t = Instant::now();
            let f = g.f_symbol(&x, &x, &x, &x, &x, &x).unwrap();
            let tf = t.elapsed();
            let g = g2();
            let t = Instant::now();
            let r = g.r_symbol(&x, &x, &x).unwrap();
            println!(
                "G2 {}: F^xxx_x[x,x] {:?} cold {tf:?}; R^xx_x {:?} cold {:?}",
                g.dim(&x),
                f.dims(),
                r.dims(),
                t.elapsed()
            );
        }
    }

    /// Two group values generate the same bits, and A/B agree with F.
    #[test]
    fn determinism_and_a_b() {
        let (g, h) = (g2(), g2());
        let (v, adj) = (l(&g, 1, 0), l(&g, 0, 1));
        let f = |g: &LieGroup| bits(g.f_symbol(&v, &v, &v, &v, &adj, &adj).unwrap());
        assert_eq!(f(&g), f(&h));
        let r = |g: &LieGroup| bits(g.r_symbol(&v, &v, &adj).unwrap());
        assert_eq!(r(&g), r(&h));
        // κ = +1, so A^{ab}_c = √(d_a d_b / d_c) F^{ā a b}_b[𝟙, c].
        let a = g.a_symbol(&v, &v, &adj).unwrap();
        let fab = g.f_symbol(&v, &v, &v, &v, &g.unit(), &adj).unwrap();
        let want = (7.0f64 * 7.0 / 14.0).sqrt() * fab.as_slice()[0];
        assert!((a.as_slice()[0] - want).abs() <= 1e-14);
        assert_eq!(g.b_symbol(&v, &v, &adj).unwrap().dims(), [1, 1]);
    }

    /// G2 × SU(2) × U(1) equals the ordered Product of its factor groups.
    #[test]
    fn composite_matches_the_factor_product() {
        let id = GroupId::cover(&[RootSystem::G2, RootSystem::A(1)], 1).unwrap();
        let cat = LieGroup::new(id).unwrap();
        let factors = [
            LieGroup::new(GroupId::g2()).unwrap(),
            LieGroup::new(GroupId::su(2).unwrap()).unwrap(),
            LieGroup::new(GroupId::torus(1).unwrap()).unwrap(),
        ];
        let product = Product::new(factors.to_vec());
        let split = |dynkin: [i64; 3], q: i64| {
            (
                cat.irrep(&dynkin, &[q]).unwrap(),
                vec![
                    factors[0].irrep(&dynkin[..2], &[]).unwrap(),
                    factors[1].irrep(&dynkin[2..], &[]).unwrap(),
                    factors[2].irrep(&[], &[q]).unwrap(),
                ],
            )
        };
        let (a, pa) = split([1, 0, 1], 1);
        let (b, pb) = split([1, 0, 1], -1);
        let got: Vec<(Vec<i64>, Vec<i64>)> = cat
            .fusion(&a, &b)
            .unwrap()
            .map(|c| (c.dynkin(), c.charges().collect()))
            .collect();
        let want: Vec<(Vec<i64>, Vec<i64>)> = product
            .fusion(&pa, &pb)
            .unwrap()
            .map(|c| {
                let dynkin = c[..2].iter().flat_map(|x| x.dynkin()).collect();
                (dynkin, c[2].charges().collect())
            })
            .collect();
        assert_eq!(got, want);
        assert_eq!(got.len(), 8);
        let (e, pe) = split([0, 1, 0], 0);
        let (d, pd) = split([1, 0, 1], 1);
        assert_eq!(
            bits(cat.f_symbol(&a, &b, &d, &d, &e, &e).unwrap()),
            bits(product.f_symbol(&pa, &pb, &pd, &pd, &pe, &pe).unwrap())
        );
    }

    /// The dense CGC is an isometry onto the coupled irrep.
    #[test]
    fn cgc_is_an_isometry() {
        use fusion_symbols::category::CgcBlock;
        let g = g2();
        let (v, adj) = (l(&g, 1, 0), l(&g, 0, 1));
        let c = g.cgc(&v, &v, &adj).unwrap();
        assert_eq!(c.dims(), [7, 7, 14, 1]);
        for m in 0..14 {
            for n in 0..14 {
                let s: f64 = (0..7)
                    .flat_map(|i| (0..7).map(move |j| (i, j)))
                    .map(|(i, j)| c.at([i, j, m, 0]) * c.at([i, j, n, 0]))
                    .sum();
                assert!((s - f64::from(u8::from(m == n))).abs() <= 1e-14);
            }
        }
        assert_eq!(g.cgc(&v, &v, &l(&g, 1, 1)).unwrap().dims(), [7, 7, 64, 0]);
    }
}
