//! Independent oracles: closed forms, literal tables from
//! `docs/gauge_exceptional.md` §6.2 and §9 (Bourbaki Plates V–IX), and the
//! existing SU(N) and B/C/D implementations as a differential check.

use super::*;
use crate::lie::bcd::{self, Series};
use crate::lie::sun;
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
    let r = |n, d| Ratio::new(n, d);
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
        for (i, &di) in d.symmetrizer().iter().enumerate() {
            assert_eq!(d.form(i, i), di * 2);
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

// ---- differential against SU(N) and B/C/D -----------------------------------

fn sun_irrep(l: &[i64]) -> sun::Irrep {
    sun::Irrep::from_dynkin(l).unwrap()
}

fn bcd_irrep(series: Series, l: &[i64]) -> bcd::Irrep {
    bcd::Irrep::from_dynkin_in(&series.cover_group(l.len()), l).unwrap()
}

/// SU(N) weights from GT patterns: `w_l = σ_l − σ_{l−1}` with `σ_l` the row sum
/// of row `l`, and Dynkin coordinate `i` equal to `w_i − w_{i+1}`.
fn sun_weights(irrep: &sun::Irrep) -> BTreeMap<Vec<i64>, u64> {
    let n = irrep.rank();
    let mut out = BTreeMap::new();
    for p in irrep.patterns() {
        let sigma: Vec<i64> = (0..=n)
            .map(|l| (1..=l).map(|k| p.get(k, l)).sum())
            .collect();
        let w: Vec<i64> = (1..=n).map(|l| sigma[l] - sigma[l - 1]).collect();
        let dynkin = w.windows(2).map(|x| x[0] - x[1]).collect();
        *out.entry(dynkin).or_insert(0) += 1;
    }
    out
}

#[test]
fn differential_sun() {
    for r in 1..=4 {
        let d = CartanData::new(RootSystem::A(r));
        let box_ = labels(r, if r <= 2 { 3 } else { 2 });
        for l in &box_ {
            let irrep = sun_irrep(l);
            assert_eq!(d.weyl_dimension(l).unwrap(), irrep.dim(), "A{r} {l:?}");
            assert_eq!(d.dual(l).unwrap(), irrep.dual().dynkin(), "A{r} {l:?}");
            assert_eq!(
                d.weight_multiplicities(l).unwrap(),
                sun_weights(&irrep),
                "A{r} {l:?}"
            );
            if irrep.dual() == irrep {
                let sign = f64::from(d.two_rho_vee_sign(l).unwrap());
                assert_eq!(sign, irrep.frobenius_schur_phase(), "A{r} {l:?}");
            }
        }
        for a in labels(r, 2) {
            for b in labels(r, if r <= 2 { 2 } else { 1 }) {
                let expected: BTreeMap<Vec<i64>, u32> =
                    sun::directproduct(&sun_irrep(&a), &sun_irrep(&b))
                        .unwrap()
                        .into_iter()
                        .map(|(c, n)| (c.dynkin(), n))
                        .collect();
                assert_eq!(
                    d.tensor_product(&a, &b).unwrap(),
                    expected,
                    "A{r} {a:?}⊗{b:?}"
                );
            }
        }
    }
}

#[test]
fn differential_bcd() {
    let cases = [
        (Series::B, 2, 3, 2),
        (Series::B, 3, 2, 1),
        (Series::C, 2, 3, 2),
        (Series::C, 3, 2, 1),
        (Series::D, 4, 2, 1),
        (Series::D, 5, 1, 1),
    ];
    for (series, r, total, product_total) in cases {
        let d = CartanData::new(series.root_system(r));
        for l in labels(r, total) {
            let irrep = bcd_irrep(series, &l);
            let tag = format!("{series:?}{r} {l:?}");
            assert_eq!(d.weyl_dimension(&l).unwrap(), irrep.dim(), "{tag}");
            assert_eq!(d.dual(&l).unwrap(), irrep.dual().dynkin(), "{tag}");
            assert_eq!(
                d.weight_multiplicities(&l).unwrap(),
                irrep.dynkin_weight_multiplicities(),
                "{tag}"
            );
            if irrep.dual() == irrep {
                let sign = f64::from(d.two_rho_vee_sign(&l).unwrap());
                assert_eq!(sign, irrep.frobenius_schur_phase(), "{tag}");
            }
        }
        for a in labels(r, product_total) {
            for b in labels(r, product_total) {
                let expected: BTreeMap<Vec<i64>, u32> =
                    bcd::directproduct(&bcd_irrep(series, &a), &bcd_irrep(series, &b))
                        .unwrap()
                        .into_iter()
                        .map(|(c, n)| (c.dynkin(), n))
                        .collect();
                assert_eq!(
                    d.tensor_product(&a, &b).unwrap(),
                    expected,
                    "{series:?}{r} {a:?}⊗{b:?}"
                );
            }
        }
    }
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
            BigInt::one()
        );
        for (i, &dim) in dims.iter().enumerate() {
            let l = unit(root.rank(), i, 1);
            assert_eq!(
                d.weyl_dimension(&l).unwrap(),
                BigInt::from(dim),
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
        let f = |x: i64| BigInt::from(x);
        f(a + 1)
            * f(b + 1)
            * f(a + b + 2)
            * f(a + 2 * b + 3)
            * f(a + 3 * b + 4)
            * f(2 * a + 3 * b + 5)
            / 120
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
    assert_eq!(e6.weyl_dimension(&unit(6, 5, 1)).unwrap(), BigInt::from(27));
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
        BigInt::from(1463)
    );
    assert_eq!(
        e6.weyl_dimension(&[1, 0, 0, 0, 0, 1]).unwrap(),
        BigInt::from(650)
    );
    assert_eq!(f4.weyl_dimension(&[0, 0, 0, 2]).unwrap(), BigInt::from(324));
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
        assert_eq!(e8.weyl_dimension(&l).unwrap(), BigInt::from(dim));
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
    let f = |x: i64| BigInt::from(x);
    let expected = (f(m) + 1) * (f(m) + 2) * (f(m) + 3) * (f(m) + 4) * (f(m) * 2 + 5) / 120;
    assert_eq!(g2.weyl_dimension(&[m, 0]).unwrap(), expected);
    assert_eq!(g2.two_rho_vee_sign(&[m, m]), Ok(1));
    // −(m, 0) → s₁ → (m, −m) → s₂ needs m − 3m: out of range.
    assert_eq!(g2.dual(&[m, 0]), Err(LieError::LabelOverflow));
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
    assert!(e8.weyl_dimension(&[m; 8]).unwrap() > BigInt::from(u128::MAX));
}
