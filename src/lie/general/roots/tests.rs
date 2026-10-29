//! Independent oracles: closed forms, literal tables from
//! `docs/gauge_exceptional.md` §6.2 and §9 (Bourbaki Plates V–IX), and the
//! literal classical products and chirality orbits.

use super::*;
use std::collections::BTreeSet;

/// Every label of length `rank` with entries summing to at most `total`.
fn labels(rank: usize, total: i64) -> Vec<Vec<i64>> {
    let mut out = vec![vec![]];
    for _ in 0..rank {
        out = out
            .into_iter()
            .flat_map(|l: Vec<i64>| {
                let used: i64 = l.iter().sum();
                (0..=total - used).map(move |x| {
                    let mut l = l.clone();
                    l.push(x);
                    l
                })
            })
            .collect();
    }
    out
}

fn unit(rank: usize, i: usize, n: i64) -> Vec<i64> {
    let mut l = vec![0; rank];
    l[i] = n;
    l
}

fn multiset(entries: &[(&[i64], u32)]) -> BTreeMap<Vec<i64>, u32> {
    let mut out = BTreeMap::new();
    for &(l, n) in entries {
        *out.entry(l.to_vec()).or_insert(0) += n;
    }
    out
}

// ---- positive roots, symmetrizer, ρ ---------------------------------------

#[test]
fn positive_root_counts() {
    // Closed forms: |Φ+(A_r)| = r(r+1)/2, |Φ+(B_r)| = |Φ+(C_r)| = r², |Φ+(D_r)| = r(r−1).
    for r in 1..=6 {
        assert_eq!(
            CartanData::new(RootSystem::A(r)).positive_roots().count(),
            r * (r + 1) / 2
        );
        assert_eq!(
            CartanData::new(RootSystem::B(r)).positive_roots().count(),
            r * r
        );
        assert_eq!(
            CartanData::new(RootSystem::C(r)).positive_roots().count(),
            r * r
        );
    }
    for r in 3..=6 {
        assert_eq!(
            CartanData::new(RootSystem::D(r)).positive_roots().count(),
            r * (r - 1)
        );
    }
    for (root, n) in [
        (RootSystem::G2, 6),
        (RootSystem::F4, 24),
        (RootSystem::E6, 36),
        (RootSystem::E7, 63),
        (RootSystem::E8, 120),
    ] {
        assert_eq!(CartanData::new(root).positive_roots().count(), n, "{root}");
    }
}

#[test]
fn g2_positive_roots_and_highest_roots() {
    // Bourbaki Plate IX: α₁ short; roots α₁, α₂, α₁+α₂, 2α₁+α₂, 3α₁+α₂, 3α₁+2α₂.
    let g2: BTreeSet<Vec<i64>> = CartanData::new(RootSystem::G2)
        .positive_roots()
        .map(<[_]>::to_vec)
        .collect();
    let expected = [[1, 0], [0, 1], [1, 1], [2, 1], [3, 1], [3, 2]];
    assert_eq!(g2, expected.into_iter().map(Vec::from).collect());
    // Highest roots (Bourbaki Plates V–VIII), in the code's node order.
    for (root, highest) in [
        (RootSystem::F4, vec![2, 3, 4, 2]),
        (RootSystem::E6, vec![1, 2, 2, 3, 2, 1]),
        (RootSystem::E7, vec![2, 2, 3, 4, 3, 2, 1]),
        (RootSystem::E8, vec![2, 3, 4, 6, 5, 4, 3, 2]),
    ] {
        assert_eq!(
            CartanData::new(root).positive_roots().last(),
            Some(&highest[..]),
            "{root}"
        );
    }
}

#[test]
fn symmetrizer_and_form() {
    let r = |n, d| Rational::from_signeds(n, d);
    // §2.2: long roots have length² 2.
    assert_eq!(
        CartanData::new(RootSystem::G2).symmetrizer(),
        [r(1, 3), r(1, 1)]
    );
    assert_eq!(
        CartanData::new(RootSystem::F4).symmetrizer(),
        [r(1, 1), r(1, 1), r(1, 2), r(1, 2)]
    );
    assert_eq!(
        CartanData::new(RootSystem::B(3)).symmetrizer(),
        [r(1, 1), r(1, 1), r(1, 2)]
    );
    assert_eq!(
        CartanData::new(RootSystem::C(3)).symmetrizer(),
        [r(1, 2), r(1, 2), r(1, 1)]
    );
    for root in [
        RootSystem::E6,
        RootSystem::E7,
        RootSystem::E8,
        RootSystem::D(5),
    ] {
        assert!(CartanData::new(root)
            .symmetrizer()
            .iter()
            .all(|d| *d == r(1, 1)));
    }
    // The form is symmetric with (α_i, α_i) = 2 d_i. G2: |α₁|² = 2/3, |α₂|² = 2
    // at 150°, so (α₁, α₂) = −1; F4: |α₂|² = 2, |α₃|² = 1 at 135°, so −1.
    for root in [
        RootSystem::G2,
        RootSystem::F4,
        RootSystem::B(3),
        RootSystem::E8,
    ] {
        let d = CartanData::new(root);
        for (i, di) in d.symmetrizer().iter().enumerate() {
            assert_eq!(d.form(i, i), di * Rational::from(2));
            for j in 0..root.rank() {
                assert_eq!(d.form(i, j), d.form(j, i), "{root} ({i},{j})");
            }
        }
    }
    assert_eq!(CartanData::new(RootSystem::G2).form(0, 1), r(-1, 1));
    assert_eq!(CartanData::new(RootSystem::F4).form(1, 2), r(-1, 1));
}

#[test]
fn two_rho_and_two_rho_vee() {
    // §6.2 table: 2ρ∨ in simple coroots.
    for (root, c) in [
        (RootSystem::G2, vec![6, 10]),
        (RootSystem::F4, vec![22, 42, 30, 16]),
        (RootSystem::E6, vec![16, 22, 30, 42, 30, 16]),
        (RootSystem::E7, vec![34, 49, 66, 96, 75, 52, 27]),
        (RootSystem::E8, vec![92, 136, 182, 270, 220, 168, 114, 58]),
    ] {
        assert_eq!(CartanData::new(root).two_rho_vee(), c, "{root}");
    }
    // 2ρ in simple roots (Bourbaki Plates VIII–IX): the reverse for G2 and F4,
    // equal for simply laced types.
    assert_eq!(CartanData::new(RootSystem::G2).two_rho(), [10, 6]);
    assert_eq!(CartanData::new(RootSystem::F4).two_rho(), [16, 30, 42, 22]);
    assert_eq!(
        CartanData::new(RootSystem::E7).two_rho(),
        [34, 49, 66, 96, 75, 52, 27]
    );
    // A_r: 2ρ = Σ_i i(r+1−i) α_i.
    assert_eq!(CartanData::new(RootSystem::A(4)).two_rho(), [4, 6, 6, 4]);
    assert_eq!(
        CartanData::new(RootSystem::A(4)).two_rho_vee(),
        [4, 6, 6, 4]
    );
}

// ---- exceptional tables ------------------------------------------------------

#[test]
fn exceptional_fundamental_dimensions() {
    // §9.1, Bourbaki order.
    let table: [(RootSystem, &[u64]); 5] = [
        (RootSystem::G2, &[7, 14]),
        (RootSystem::F4, &[52, 1274, 273, 26]),
        (RootSystem::E6, &[27, 78, 351, 2925, 351, 27]),
        (RootSystem::E7, &[133, 912, 8645, 365750, 27664, 1539, 56]),
        (
            RootSystem::E8,
            &[
                3875, 147250, 6696000, 6899079264, 146325270, 2450240, 30380, 248,
            ],
        ),
    ];
    for (root, dims) in table {
        let d = CartanData::new(root);
        assert_eq!(
            d.weyl_dimension(&vec![0; root.rank()]).unwrap(),
            Integer::from(1)
        );
        for (i, &dim) in dims.iter().enumerate() {
            let l = unit(root.rank(), i, 1);
            assert_eq!(
                d.weyl_dimension(&l).unwrap(),
                Integer::from(dim),
                "{root} ω{}",
                i + 1
            );
        }
    }
}

#[test]
fn g2_weyl_dimension_closed_form() {
    // Fulton–Harris §22.3: with a on the short node,
    // dim = (a+1)(b+1)(a+b+2)(a+2b+3)(a+3b+4)(2a+3b+5)/120.
    let g2 = CartanData::new(RootSystem::G2);
    let closed = |a: i64, b: i64| {
        let f = |x: i64| Integer::from(x);
        f(a + 1)
            * f(b + 1)
            * f(a + b + 2)
            * f(a + 2 * b + 3)
            * f(a + 3 * b + 4)
            * f(2 * a + 3 * b + 5)
            / f(120)
    };
    for a in 0..6 {
        for b in 0..6 {
            assert_eq!(
                g2.weyl_dimension(&[a, b]).unwrap(),
                closed(a, b),
                "({a},{b})"
            );
        }
    }
}

#[test]
fn exceptional_duals() {
    // §6.1: −w₀ = 1 for G2, F4, E7, E8; for E6 it swaps 0↔5 and 2↔4.
    for root in [
        RootSystem::G2,
        RootSystem::F4,
        RootSystem::E7,
        RootSystem::E8,
    ] {
        let d = CartanData::new(root);
        for l in labels(root.rank(), 2) {
            assert_eq!(d.dual(&l).unwrap(), l, "{root}");
        }
    }
    let e6 = CartanData::new(RootSystem::E6);
    for l in labels(6, 2) {
        let swapped = vec![l[5], l[1], l[4], l[3], l[2], l[0]];
        assert_eq!(e6.dual(&l).unwrap(), swapped, "{l:?}");
    }
    // The 27/27̄ pair.
    assert_eq!(e6.dual(&unit(6, 0, 1)).unwrap(), unit(6, 5, 1));
    assert_eq!(
        e6.weyl_dimension(&unit(6, 5, 1)).unwrap(),
        Integer::from(27)
    );
}

#[test]
fn exceptional_frobenius_schur_signs() {
    // §6.2: G2, F4, E6 and E8 are all even; E7 is pseudo-real exactly when
    // a₂ + a₅ + a₇ (Bourbaki) is odd, i.e. coordinates 1, 4, 6 here.
    for root in [
        RootSystem::G2,
        RootSystem::F4,
        RootSystem::E6,
        RootSystem::E8,
    ] {
        let d = CartanData::new(root);
        for l in labels(root.rank(), 3) {
            assert_eq!(d.two_rho_vee_sign(&l).unwrap(), 1, "{root} {l:?}");
        }
    }
    let e7 = CartanData::new(RootSystem::E7);
    for l in labels(7, 3) {
        let expected = if (l[1] + l[4] + l[6]) % 2 == 1 { -1 } else { 1 };
        assert_eq!(e7.two_rho_vee_sign(&l).unwrap(), expected, "{l:?}");
    }
    // 56 (a₇) and 912 (a₂) pseudo-real; 133 (a₁) and 1539 (a₆) real.
    assert_eq!(e7.two_rho_vee_sign(&unit(7, 6, 1)).unwrap(), -1);
    assert_eq!(e7.two_rho_vee_sign(&unit(7, 1, 1)).unwrap(), -1);
    assert_eq!(e7.two_rho_vee_sign(&unit(7, 0, 1)).unwrap(), 1);
    assert_eq!(e7.two_rho_vee_sign(&unit(7, 5, 1)).unwrap(), 1);
}

#[test]
fn exceptional_adjoint_and_minuscule_multiplicities() {
    // The adjoint has the |Φ| roots with multiplicity 1 and the zero weight
    // with multiplicity rank; a minuscule irrep is one Weyl orbit.
    for (root, adjoint) in [
        (RootSystem::G2, unit(2, 1, 1)),
        (RootSystem::F4, unit(4, 0, 1)),
        (RootSystem::E6, unit(6, 1, 1)),
        (RootSystem::E7, unit(7, 0, 1)),
        (RootSystem::E8, unit(8, 7, 1)),
    ] {
        let d = CartanData::new(root);
        let w = d.weight_multiplicities(&adjoint).unwrap();
        let roots = 2 * d.positive_roots().count();
        assert_eq!(w.len(), roots + 1, "{root}");
        assert_eq!(w[&vec![0; root.rank()]], root.rank() as u64, "{root}");
        assert_eq!(w.values().filter(|&&m| m == 1).count(), roots, "{root}");
    }
    let e6 = CartanData::new(RootSystem::E6)
        .weight_multiplicities(&unit(6, 0, 1))
        .unwrap();
    assert_eq!((e6.len(), e6.values().all(|&m| m == 1)), (27, true));
    let e7 = CartanData::new(RootSystem::E7)
        .weight_multiplicities(&unit(7, 6, 1))
        .unwrap();
    assert_eq!((e7.len(), e7.values().all(|&m| m == 1)), (56, true));
    // G2 27 = (2,0): the zero weight has multiplicity 3 (§10 table).
    let g2 = CartanData::new(RootSystem::G2)
        .weight_multiplicities(&[2, 0])
        .unwrap();
    assert_eq!(g2[&vec![0, 0]], 3);
    assert_eq!(g2.values().sum::<u64>(), 27);
}

// ---- decompositions (§9.3) ---------------------------------------------------

#[test]
fn g2_decompositions() {
    let g2 = CartanData::new(RootSystem::G2);
    let p = |a: [i64; 2], b: [i64; 2]| g2.tensor_product(&a, &b).unwrap();
    // 7 = (1,0), 14 = (0,1), 27 = (2,0), 64 = (1,1), 77 = (0,2) and (3,0),
    // 182 = (4,0), 189 = (2,1).
    assert_eq!(
        p([1, 0], [1, 0]),
        multiset(&[(&[0, 0], 1), (&[1, 0], 1), (&[0, 1], 1), (&[2, 0], 1)])
    );
    assert_eq!(
        p([1, 0], [0, 1]),
        multiset(&[(&[1, 0], 1), (&[2, 0], 1), (&[1, 1], 1)])
    );
    assert_eq!(
        p([0, 1], [0, 1]),
        multiset(&[
            (&[0, 0], 1),
            (&[0, 1], 1),
            (&[2, 0], 1),
            (&[0, 2], 1),
            (&[3, 0], 1)
        ])
    );
    assert_eq!(
        p([1, 0], [2, 0]),
        multiset(&[
            (&[1, 0], 1),
            (&[0, 1], 1),
            (&[2, 0], 1),
            (&[1, 1], 1),
            (&[3, 0], 1)
        ])
    );
    assert_eq!(
        p([2, 0], [2, 0]),
        multiset(&[
            (&[0, 0], 1),
            (&[1, 0], 1),
            (&[0, 1], 1),
            (&[2, 0], 2),
            (&[1, 1], 2),
            (&[0, 2], 1),
            (&[3, 0], 1),
            (&[4, 0], 1),
            (&[2, 1], 1),
        ])
    );
}

#[test]
fn exceptional_decompositions() {
    // F4: 26 = ω4, 52 = ω1, 273 = ω3, 324 = 2ω4.
    let f4 = CartanData::new(RootSystem::F4);
    assert_eq!(
        f4.tensor_product(&[0, 0, 0, 1], &[0, 0, 0, 1]).unwrap(),
        multiset(&[
            (&[0, 0, 0, 0], 1),
            (&[0, 0, 0, 1], 1),
            (&[1, 0, 0, 0], 1),
            (&[0, 0, 1, 0], 1),
            (&[0, 0, 0, 2], 1),
        ])
    );
    // E6: 27 = ω1, 27̄ = ω6, 351 = ω3, 351′ = 2ω1, 78 = ω2, 650 = ω1 + ω6.
    let e6 = CartanData::new(RootSystem::E6);
    let e27 = unit(6, 0, 1);
    assert_eq!(
        e6.tensor_product(&e27, &e27).unwrap(),
        multiset(&[
            (&[0, 0, 0, 0, 0, 1], 1),
            (&[0, 0, 1, 0, 0, 0], 1),
            (&[2, 0, 0, 0, 0, 0], 1),
        ])
    );
    assert_eq!(
        e6.tensor_product(&e27, &unit(6, 5, 1)).unwrap(),
        multiset(&[
            (&[0, 0, 0, 0, 0, 0], 1),
            (&[0, 1, 0, 0, 0, 0], 1),
            (&[1, 0, 0, 0, 0, 1], 1),
        ])
    );
    // E7: 56 = ω7, 133 = ω1, 1463 = 2ω7, 1539 = ω6.
    let e7 = CartanData::new(RootSystem::E7);
    let e56 = unit(7, 6, 1);
    assert_eq!(
        e7.tensor_product(&e56, &e56).unwrap(),
        multiset(&[
            (&[0; 7], 1),
            (&unit(7, 0, 1), 1),
            (&unit(7, 6, 2), 1),
            (&unit(7, 5, 1), 1),
        ])
    );
    // The channel dimensions named in §9.3.
    assert_eq!(
        e7.weyl_dimension(&unit(7, 6, 2)).unwrap(),
        Integer::from(1463)
    );
    assert_eq!(
        e6.weyl_dimension(&[1, 0, 0, 0, 0, 1]).unwrap(),
        Integer::from(650)
    );
    assert_eq!(
        f4.weyl_dimension(&[0, 0, 0, 2]).unwrap(),
        Integer::from(324)
    );
}

/// Brauer–Klimyk walks the 241 distinct weights of 248 and reflects each
/// shifted weight into the chamber; it never visits W (|W(E8)| = 696,729,600).
#[test]
fn e8_adjoint_square() {
    // E8: 248 = ω8, 3875 = ω1, 27000 = 2ω8, 30380 = ω7.
    let e8 = CartanData::new(RootSystem::E8);
    let e248 = unit(8, 7, 1);
    assert_eq!(
        e8.tensor_product(&e248, &e248).unwrap(),
        multiset(&[
            (&[0; 8], 1),
            (&e248, 1),
            (&unit(8, 0, 1), 1),
            (&unit(8, 7, 2), 1),
            (&unit(8, 6, 1), 1),
        ])
    );
    for (l, dim) in [(unit(8, 7, 2), 27000), (unit(8, 6, 1), 30380)] {
        assert_eq!(e8.weyl_dimension(&l).unwrap(), Integer::from(dim));
    }
}

// ---- errors and overflow -----------------------------------------------------

#[test]
fn invalid_and_huge_labels() {
    let g2 = CartanData::new(RootSystem::G2);
    assert_eq!(g2.weyl_dimension(&[1]), Err(LieError::InvalidLabel));
    assert_eq!(g2.dual(&[-1, 0]), Err(LieError::InvalidLabel));
    assert_eq!(
        g2.tensor_product(&[0, 0], &[0, 0, 0]),
        Err(LieError::InvalidLabel)
    );
    assert_eq!(
        g2.dominant_multiplicities(&[0, -2]),
        Err(LieError::InvalidLabel)
    );
    assert_eq!(g2.two_rho_vee_sign(&[0]), Err(LieError::InvalidLabel));

    let m = i64::MAX;
    // The Weyl dimension is exact past u128 (G2 closed form, as above).
    let f = |x: i64| Integer::from(x);
    let expected =
        (f(m) + f(1)) * (f(m) + f(2)) * (f(m) + f(3)) * (f(m) + f(4)) * (f(m) * f(2) + f(5))
            / f(120);
    assert_eq!(g2.weyl_dimension(&[m, 0]).unwrap(), expected);
    assert_eq!(g2.two_rho_vee_sign(&[m, m]), Ok(1));
    // −(m, 0) → s₁ → (m, −m) → s₂ needs m − 3m: out of range.
    assert_eq!(g2.dual(&[m, 0]), Ok(vec![m, 0]));
    assert_eq!(g2.check_dual_domain(&[m, 0]), Err(LieError::LabelOverflow));
    assert_eq!(
        g2.dominant_multiplicities(&[m, 0]),
        Err(LieError::LabelOverflow)
    );
    assert_eq!(
        g2.weight_multiplicities(&[0, m - 1]),
        Err(LieError::LabelOverflow)
    );
    assert_eq!(
        g2.tensor_product(&[m, 0], &[1, 0]),
        Err(LieError::LabelOverflow)
    );
    assert_eq!(
        g2.tensor_product(&[1, 0], &[m - 1, 0]),
        Err(LieError::LabelOverflow)
    );
    let e8 = CartanData::new(RootSystem::E8);
    assert!(e8.weyl_dimension(&[m; 8]).unwrap() > u128::MAX);
}

#[test]
fn payload_identity_and_legacy_product_order() {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let hash = |x: &RootIrrep| {
        let mut h = DefaultHasher::new();
        x.hash(&mut h);
        h.finish()
    };
    for (root, a, b, expected) in [
        // SU(3): 3 x 6bar = 3bar + 15bar, ordered by partitions, not Dynkin labels.
        (
            RootSystem::A(2),
            vec![1, 0],
            vec![0, 2],
            vec![vec![0, 1], vec![1, 2]],
        ),
        // D3: 4 x 4bar = 1 + 15 (odd-D dual pair).
        (
            RootSystem::D(3),
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![vec![0, 0, 0], vec![0, 1, 1]],
        ),
        // B2: 4 x 4 = 1 + 5 + 10 (equal-dimension spinor factors).
        (
            RootSystem::B(2),
            vec![0, 1],
            vec![0, 1],
            vec![vec![0, 0], vec![1, 0], vec![0, 2]],
        ),
        // C2: 4 x 4 = 1 + 5 + 10.
        (
            RootSystem::C(2),
            vec![1, 0],
            vec![1, 0],
            vec![vec![0, 0], vec![0, 1], vec![2, 0]],
        ),
    ] {
        let data = CartanData::new(root);
        let (a, b) = (data.irrep(&a).unwrap(), data.irrep(&b).unwrap());
        let product = data.product(&a, &b).unwrap();
        assert_eq!(
            product
                .iter()
                .map(|(c, n)| {
                    assert_eq!(*n, 1);
                    c.dynkin().to_vec()
                })
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(data.product(&b, &a).unwrap(), product);
        assert_eq!(
            product
                .iter()
                .rev()
                .map(|(c, _)| c.dynkin().to_vec())
                .collect::<Vec<_>>(),
            expected.into_iter().rev().collect::<Vec<_>>()
        );
        for (c, n) in product.iter() {
            assert_eq!(product.multiplicity(c), *n);
        }
        let same = data.irrep(a.dynkin()).unwrap();
        assert_eq!(a, same);
        assert_eq!(hash(&a), hash(&same));
        assert_eq!(a.dual().dual(), a);
        let unit = data.irrep(&vec![0; root.rank()]).unwrap();
        let unit_product = data
            .load_product(
                data.multiplicities(unit.dynkin(), a.dynkin(), Some((&unit, &a)))
                    .unwrap(),
                Some((&unit, &a)),
            )
            .unwrap();
        assert_eq!(unit_product.len(), 1);
        assert_eq!(unit_product[0].1, 1);
        assert!(Arc::ptr_eq(&unit_product[0].0.data, &a.data));
    }
    let data = CartanData::new(RootSystem::A(2));
    assert!(
        data.irrep(&[1, 0]).unwrap() < data.irrep(&[0, 2]).unwrap(),
        "preserve partition order, which differs from Dynkin order"
    );
}

#[test]
fn classical_overflow_precedes_unit_shortcuts() {
    for root in [RootSystem::B(2), RootSystem::C(2), RootSystem::D(3)] {
        let data = CartanData::new(root);
        let r = root.rank();
        let mut huge = vec![0; r];
        huge[0] = i64::MAX / 2;
        assert_eq!(
            data.tensor_product(&vec![0; r], &huge),
            Err(LieError::LabelOverflow)
        );
    }
    let data = CartanData::new(RootSystem::D(3));
    assert_eq!(
        data.dynkin(&[i64::MAX, i64::MAX, i64::MAX]).unwrap(),
        vec![0, 0, i64::MAX]
    );
    assert_eq!(data.dynkin(&[i64::MIN, 0, 0]), Err(LieError::LabelOverflow));
}

#[test]
fn d_chirality_orbits() {
    let orbit = |w: &[i64]| {
        classical::try_weyl_orbit(Series::D, w)
            .unwrap()
            .into_iter()
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(
        orbit(&[1, 1, -1]),
        [
            vec![-1, -1, -1],
            vec![-1, 1, 1],
            vec![1, -1, 1],
            vec![1, 1, -1]
        ]
        .into_iter()
        .collect()
    );
    assert_eq!(
        orbit(&[1, 1, 1]),
        [
            vec![1, 1, 1],
            vec![1, -1, -1],
            vec![-1, 1, -1],
            vec![-1, -1, 1]
        ]
        .into_iter()
        .collect()
    );
    assert_eq!(
        orbit(&[2, 0, 0]),
        [
            vec![2, 0, 0],
            vec![-2, 0, 0],
            vec![0, 2, 0],
            vec![0, -2, 0],
            vec![0, 0, 2],
            vec![0, 0, -2]
        ]
        .into_iter()
        .collect()
    );
    for series in [Series::B, Series::C, Series::D] {
        assert_eq!(
            classical::try_weyl_orbit(series, &[0; 32]),
            Err(LieError::LabelOverflow)
        );
        assert_eq!(
            classical::try_weyl_orbit(series, &[i64::MIN, 0]),
            Err(LieError::LabelOverflow)
        );
    }
}

#[test]
fn n_builds_no_channel_dimensions_and_fusion_reuses_the_product() {
    let _guard = crate::cache::TEST_CACHE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for root in [
        RootSystem::A(5),
        RootSystem::B(3),
        RootSystem::C(4),
        RootSystem::D(6),
        RootSystem::E6,
    ] {
        let data = CartanData::new(root);
        let a = data.irrep(&unit(root.rank(), 0, 1)).unwrap();
        let b = a.dual();
        let one = data.irrep(&vec![0; root.rank()]).unwrap();
        crate::cache::reset();
        EVALUATIONS.with(|n| n.set((0, 0)));
        assert_eq!(data.multiplicity(&a, &b, &one).unwrap(), 1);
        assert_eq!(data.multiplicity(&b, &a, &one).unwrap(), 1);
        EVALUATIONS.with(|n| assert_eq!(n.get(), (0, 0)));
        let product = data.product(&a, &b).unwrap();
        assert_eq!(product.multiplicity(&one), 1);
        EVALUATIONS.with(|n| n.set((0, 0)));
        assert!(product.ptr_eq(&data.product(&b, &a).unwrap()));
        EVALUATIONS.with(|n| assert_eq!(n.get(), (0, 0)));
    }
}

#[test]
fn labels_reuse_live_root_context_without_retaining_its_tables() {
    let data = CartanData::new(RootSystem::A(2));
    let weak = Arc::downgrade(&data);
    let a = data.irrep(&[1, 0]).unwrap();
    EVALUATIONS.with(|n| n.set((0, 0)));
    assert!(Arc::ptr_eq(&data, &a.roots()));
    assert_eq!(a.weight_multiplicities(&a.roots()).unwrap().len(), 3);
    EVALUATIONS.with(|n| assert_eq!(n.get(), (0, 0)));
    drop(data);
    assert!(
        weak.upgrade().is_none(),
        "labels must not retain root tables"
    );
    assert_eq!(a.weight_multiplicities(&a.roots()).unwrap().len(), 3);
}

#[test]
#[ignore = "isolated Product cache and root-context counters"]
fn warm_adapter_does_not_rebuild_context_after_group_drop() {
    let data = CartanData::new(RootSystem::A(2));
    let a = data.irrep(&[1, 0]).unwrap();
    let one = data.irrep(&[0, 0]).unwrap();
    crate::cache::reset();
    let first = shared_product(&a, &one).unwrap();
    drop(data);
    EVALUATIONS.with(|n| n.set((0, 0)));
    assert!(first.ptr_eq(&shared_product(&a, &one).unwrap()));
    EVALUATIONS.with(|n| assert_eq!(n.get(), (0, 0)));
}

#[test]
fn weyl_dimension_of_scaled_rho() {
    use malachite_base::num::arithmetic::traits::Pow;
    // Every Weyl factor of k*rho is k+1, including beyond the u128 boundary.
    for root in [
        RootSystem::A(16),
        RootSystem::B(12),
        RootSystem::C(12),
        RootSystem::D(12),
        RootSystem::G2,
        RootSystem::F4,
        RootSystem::E6,
        RootSystem::E7,
        RootSystem::E8,
    ] {
        let data = CartanData::new(root);
        for k in [0, 1, 1000] {
            assert_eq!(
                data.weyl_dimension(&vec![k; root.rank()]).unwrap(),
                Integer::from(Natural::from((k + 1) as u64).pow(data.positive.len() as u64)),
                "{root} k={k}"
            );
        }
    }
}

#[test]
fn coefficient_checks_use_dimension_free_multiplicities() {
    use crate::frcore::Family;
    let _guard = crate::cache::TEST_CACHE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let roots = CartanData::new(RootSystem::C(2));
    let a = crate::lie::bcd::Irrep(roots.irrep(&[1, 0]).unwrap());
    let one = crate::lie::bcd::Irrep(roots.irrep(&[0, 0]).unwrap());
    crate::cache::reset();
    EVALUATIONS.with(|n| n.set((0, 0)));
    assert_eq!(crate::lie::bcd::fr::mult(&a, &a, &one).unwrap(), 1);
    assert_eq!(crate::lie::bcd::fr::mult(&a, &a, &one).unwrap(), 1);
    EVALUATIONS.with(|n| assert_eq!(n.get(), (0, 0)));
    let mut family = super::super::cgc::GeneralFamily::new(RootSystem::G2);
    crate::cache::reset();
    LABEL_PROJECTIONS.set(0);
    EVALUATIONS.with(|n| n.set((0, 0)));
    assert_eq!(
        family.mult(&vec![1, 0], &vec![1, 0], &vec![0, 0]).unwrap(),
        1
    );
    EVALUATIONS.with(|n| n.set((0, 0)));
    assert_eq!(
        family.mult(&vec![1, 0], &vec![1, 0], &vec![0, 0]).unwrap(),
        1
    );
    EVALUATIONS.with(|n| assert_eq!(n.get(), (0, 0)));
    // Observe this key and this thread; unrelated tests may populate other projections.
    assert!(crate::cache::product_multiplicities_are_resident(
        RootSystem::G2,
        &[1, 0],
        &[1, 0]
    ));
    assert_eq!(LABEL_PROJECTIONS.get(), 0);

    let engine = super::super::Engine::new(RootSystem::G2);
    let a = engine.irrep(&[1, 0]).unwrap();
    let one = engine.irrep(&[0, 0]).unwrap();
    crate::cache::reset();
    LABEL_PROJECTIONS.set(0);
    let first = engine.cgc(&a, &a, &one).unwrap();
    assert_eq!(engine.cgc(&a, &a, &one).unwrap(), first);
    assert!(crate::cache::product_multiplicities_are_resident(
        RootSystem::G2,
        &[1, 0],
        &[1, 0]
    ));
    assert_eq!(LABEL_PROJECTIONS.get(), 0);
}
