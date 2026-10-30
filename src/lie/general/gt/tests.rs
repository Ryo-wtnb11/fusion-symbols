//! Independent checks of the Gelfand–Tsetlin basis: Molev's (2.6), (2.7) and
//! Proposition 2.4 written out again here, SUNRepresentations.jl v0.4.0 and
//! Condon–Shortley ladder literals, the Chevalley relations and recipe
//! identity shared with the Gram–Schmidt basis, Weyl and Freudenthal counts,
//! and exact CGC identities on channels built from this basis.

use super::*;
use crate::lie::general::cgc::tests::check_exact;
use crate::lie::general::cgc::{check_rounded, Channel, Ladders};
use crate::lie::general::module::depth_order;
use crate::lie::general::module::tests::{check_recipes, check_relations_of};
use crate::lie::RootSystem;
use std::sync::Arc;

fn build(lambda: &[i64]) -> (Arc<CartanData>, Ladders) {
    let data = CartanData::new(RootSystem::A(lambda.len()));
    let (m, public) = module(&data, lambda).unwrap();
    (data, Ladders::from_module(m, public))
}

fn q(n: i64) -> Q {
    Q::from(n)
}

/// `l_{ki} = λ_{ki} − i + 1`.
fn ell(p: &GtPattern, k: usize, i: usize) -> i64 {
    p.get(i, k) - i as i64 + 1
}

/// Molev (2.7): the coefficient of `ξ_{Λ−δ_{ki}}` in `E_{k+1,k} ξ_Λ`.
fn molev_b(p: &GtPattern, k: usize, i: usize) -> Q {
    let num: Q = (1..k).map(|j| q(ell(p, k, i) - ell(p, k - 1, j))).product();
    let den: Q = (1..=k)
        .filter(|&j| j != i)
        .map(|j| q(ell(p, k, i) - ell(p, k, j)))
        .product();
    num / den
}

/// Molev (2.6): the coefficient of `ξ_{Λ+δ_{ki}}` in `E_{k,k+1} ξ_Λ`.
fn molev_a(p: &GtPattern, k: usize, i: usize) -> Q {
    let num: Q = (1..=k + 1)
        .map(|j| q(ell(p, k, i) - ell(p, k + 1, j)))
        .product();
    let den: Q = (1..=k)
        .filter(|&j| j != i)
        .map(|j| q(ell(p, k, i) - ell(p, k, j)))
        .product();
    -num / den
}

fn factorial(n: i64) -> Q {
    assert!(n >= 0);
    (2..=n).map(q).product()
}

/// Molev Proposition 2.4: `⟨ξ_Λ, ξ_Λ⟩`.
fn molev_norm(p: &GtPattern) -> Q {
    let mut x = q(1);
    for k in 2..=p.rank() {
        for i in 1..k {
            for j in i..k {
                x *= factorial(ell(p, k, i) - ell(p, k - 1, j))
                    / factorial(ell(p, k - 1, i) - ell(p, k - 1, j));
            }
        }
        for i in 1..=k {
            for j in i + 1..=k {
                x *= factorial(ell(p, k, i) - ell(p, k, j) - 1)
                    / factorial(ell(p, k - 1, i) - ell(p, k, j) - 1);
            }
        }
    }
    x
}

/// `p` with `λ_{ki}` shifted by `delta`, if betweenness holds.
fn shift(patterns: &[GtPattern], p: &GtPattern, k: usize, i: usize, delta: i64) -> Option<usize> {
    let mut t = p.clone();
    t.set(i, k, p.get(i, k) + delta);
    patterns.iter().position(|x| *x == t)
}

fn engine(l: &Ladders) -> Vec<usize> {
    let mut e = vec![usize::MAX; l.dim()];
    for (m, &g) in l.public.iter().enumerate() {
        e[g as usize] = m;
    }
    assert!(
        e.iter().all(|&m| m != usize::MAX),
        "public is a permutation"
    );
    e
}

const CASES: [&[i64]; 10] = [
    &[1],
    &[4],
    &[1, 1],
    &[2, 2],
    &[3, 1],
    &[1, 0, 1],
    &[2, 0, 2],
    &[1, 2, 1],
    &[1, 0, 0, 0, 1],
    &[0, 1, 1, 0, 1],
];

/// Every `f_j` entry is (2.7) and every `e_j` entry, the contravariant
/// transpose through the propagated norms, is (2.6); the norms are
/// Proposition 2.4; every entry is positive, so the orthonormal ladders are
/// nonnegative; and the highest pattern has norm 1.
#[test]
fn molev_ladders_and_norms() {
    for lambda in CASES {
        let (data, l) = build(lambda);
        let patterns = data.gt_patterns(lambda).unwrap();
        let e = engine(&l);
        assert_eq!(l.norm[0], q(1));
        for (g, p) in patterns.iter().enumerate() {
            assert_eq!(l.norm[e[g]], molev_norm(p), "{lambda:?} {p:?}");
        }
        let mut count = 0;
        for (g, p) in patterns.iter().enumerate() {
            for j in 0..lambda.len() {
                let k = j + 1;
                for i in 1..=k {
                    let Some(h) = shift(&patterns, p, k, i, -1) else {
                        continue;
                    };
                    let x = molev_b(p, k, i);
                    let got = l.lower[j][e[g]].iter().find(|(t, _)| *t == e[h]);
                    if x == 0 {
                        assert!(got.is_none());
                        continue;
                    }
                    count += 1;
                    assert!(x > 0, "{lambda:?}: f_{j} sign");
                    assert_eq!(got.map(|(_, x)| x), Some(&x), "{lambda:?} f_{j} {p:?}");
                    let y = l.raise[j][e[h]].iter().find(|(t, _)| *t == e[g]).unwrap();
                    assert_eq!(y.1, molev_a(&patterns[h], k, i), "{lambda:?} e_{j}");
                    assert!(y.1 > 0, "{lambda:?}: e_{j} sign");
                }
            }
        }
        let stored: usize = l.lower.iter().flatten().map(Vec::len).sum();
        assert_eq!(stored, count, "{lambda:?}: no other ladder entries");
    }
}

/// SUNRepresentations.jl v0.4.0 `creation(SU3Irrep(2,1,0))`, as pinned in
/// `sun.rs::tests::su3_adjoint_creation_matches_reference`:
/// `(row, col, ⟨ζ_row, e_l ζ_col⟩²)` with 1-based pattern indices. In this
/// basis that is `F² n_col / n_row` for `f_{l−1}` from `row` to `col`.
#[test]
fn su3_adjoint_matches_sunrepresentations() {
    let (_, l) = build(&[1, 1]);
    let e = engine(&l);
    let reference: [&[(usize, usize, i64, i64)]; 2] = [
        &[(2, 1, 1, 1), (5, 4, 2, 1), (6, 5, 2, 1), (8, 7, 1, 1)],
        &[
            (3, 2, 3, 2),
            (4, 1, 1, 1),
            (5, 2, 1, 2),
            (7, 3, 3, 2),
            (7, 5, 1, 2),
            (8, 6, 1, 1),
        ],
    ];
    for (j, entries) in reference.iter().enumerate() {
        let stored: usize = l.lower[j].iter().map(Vec::len).sum();
        assert_eq!(stored, entries.len());
        for &(row, col, num, den) in *entries {
            let (src, tgt) = (e[row - 1], e[col - 1]);
            let (_, x) = l.lower[j][src].iter().find(|(t, _)| *t == tgt).unwrap();
            let square = x * x * &l.norm[tgt] / &l.norm[src];
            assert_eq!(
                square,
                Q::from_integers(num.into(), den.into()),
                "f_{j} {row}->{col}"
            );
        }
    }
}

/// Public order is the pattern enumeration order, with the highest weight at
/// `d − 1`: the SU(3) fundamental literally, and `A_1` patterns `(dj, 0; x)`
/// at `x = m + j` with the Condon–Shortley ladder `(j+m)(j−m+1) = x(dj−x+1)`.
#[test]
fn gt_order_literals() {
    let (data, l) = build(&[1, 0]);
    let patterns = data.gt_patterns(&[1, 0]).unwrap();
    let rows: Vec<&[i64]> = patterns.iter().map(|p| p.data()).collect();
    assert_eq!(
        rows,
        [
            &[1, 0, 0, 0, 0, 0][..],
            &[1, 0, 0, 1, 0, 0],
            &[1, 0, 0, 1, 0, 1]
        ]
    );
    assert_eq!(l.public, [2, 1, 0]);
    for dj in 0..=6i64 {
        let (data, l) = build(&[dj]);
        let rows: Vec<Vec<i64>> = data
            .gt_patterns(&[dj])
            .unwrap()
            .iter()
            .map(|p| p.data().to_vec())
            .collect();
        assert_eq!(rows, (0..=dj).map(|x| vec![dj, 0, x]).collect::<Vec<_>>());
        let want: Vec<u32> = (0..=dj as u32).rev().collect();
        assert_eq!(l.public, want);
        assert_eq!(l.norm[0], q(1));
        let e = engine(&l);
        for x in 1..=dj {
            let (src, tgt) = (e[x as usize], e[x as usize - 1]);
            let (_, f) = l.lower[0][src].iter().find(|(t, _)| *t == tgt).unwrap();
            assert_eq!(
                f * f * &l.norm[tgt] / &l.norm[src],
                q(x * (dj - x + 1)),
                "dj {dj} x {x}"
            );
        }
    }
}

/// The pattern count is the Weyl dimension, and every dominant weight space
/// has its Freudenthal multiplicity (an independent recursion for `A_r`).
#[test]
fn weyl_and_freudenthal_counts() {
    for lambda in CASES {
        let (data, l) = build(lambda);
        assert_eq!(
            Integer::from(l.dim() as u64),
            data.weyl_dimension(lambda).unwrap()
        );
        for (mu, m) in data.dominant_multiplicities(lambda).unwrap() {
            let w = l.module.weight_index(&mu).unwrap();
            assert_eq!(
                l.module.weights[w].norms.len() as u64,
                m,
                "{lambda:?} {mu:?}"
            );
        }
    }
}

/// Each recipe reproduces its basis vector exactly, and the recipes are not
/// all triangular: some weight space has a non-unit diagonal or a later term.
#[test]
fn recipes_reconstruct_basis_exactly() {
    let mut general = false;
    for lambda in CASES {
        let (_, l) = build(lambda);
        check_recipes(&l.module);
        for s in &l.module.weights[1..] {
            for (k, row) in s.recipe.iter().enumerate() {
                general |= row[k] != 1 || row[k + 1..].iter().any(|x| *x != 0);
            }
        }
    }
    assert!(general);
}

/// The Chevalley relations hold exactly in this basis.
#[test]
fn generator_relations_exact() {
    for lambda in [&[4][..], &[1, 1], &[2, 2], &[3, 1], &[1, 0, 1]] {
        let (data, l) = build(lambda);
        check_relations_of(&data, &l.module);
    }
}

/// Gate 1: a missing pattern is a typed rank failure, not a short module.
#[test]
fn missing_pattern_is_a_rank_failure() {
    let data = CartanData::new(RootSystem::A(2));
    let patterns = data.gt_patterns(&[1, 0]).unwrap();
    let weights = |p: &[GtPattern]| p.iter().map(|p| data.gt_weight(p).unwrap()).collect();
    let order = || depth_order(&data, &[1, 0]).unwrap();
    let err = from_patterns(&data, order(), &patterns[1..], weights(&patterns[1..])).err();
    assert_eq!(
        err,
        Some(
            ModuleError::RankMismatch {
                weight: data.gt_weight(&patterns[0]).unwrap(),
                rank: 0,
                multiplicity: 1
            }
            .into()
        )
    );
    assert_eq!(
        from_patterns(&data, order(), &[], Vec::new()).err(),
        Some(
            ModuleError::RankMismatch {
                weight: vec![1, 0],
                rank: 0,
                multiplicity: 1
            }
            .into()
        )
    );
}

/// A dimension beyond the `u32` magnetic index is a typed overflow, decided
/// before any of its `2^32` patterns would be enumerated.
#[test]
fn oversized_label_overflows_before_enumeration() {
    let data = CartanData::new(RootSystem::A(1));
    assert_eq!(
        module(&data, &[i64::from(u32::MAX)]).err(),
        Some(LieError::LabelOverflow)
    );
}

/// Channels built on the Gelfand–Tsetlin basis with the shared descent are
/// exact isometric intertwiners, and their rounded entries at public indices
/// pass the rounded gates through the non-identity index map.
#[test]
fn gt_channels_are_exact() {
    for (a, b) in [([1, 0], [0, 1]), ([1, 0], [1, 0]), ([1, 1], [1, 1])] {
        let data = CartanData::new(RootSystem::A(2));
        let [ma, mb] = [&a, &b].map(|x| build(x).1);
        let product = data.multiplicities(&a, &b, None).unwrap();
        for (c, n) in data.tensor_product(&a, &b).unwrap() {
            assert_eq!(product.multiplicity(&c), n);
            let mc = build(&c).1;
            let ch = Channel::new(&data, &ma, &mb, &mc, n as usize).unwrap();
            check_exact(&ch, &ma, &mb, &mc);
            let entries = ch.rounded_entries(&ma, &mb, &mc).unwrap();
            check_rounded(&data, &ma, &mb, &mc, n as usize, &entries).unwrap();
        }
    }
}

/// The recipe solve on literal candidates `c_0 = b_0 + b_1`, `c_1 = b_1`,
/// `c_2 = 2 b_1` (dependent, dropped): `b_0 = c_0 − c_1` and `b_1 = c_1`.
#[test]
fn recipe_solve_literal() {
    let col = |x: [i64; 2]| x.iter().map(|&v| vec![q(v)]).collect::<Vec<_>>();
    let mut space = WeightSpace {
        mu: vec![0, 0],
        k: vec![1, 1],
        offset: 2,
        norms: vec![q(1); 2],
        kept: Vec::new(),
        recipe: Vec::new(),
        lower: vec![
            Some((0, col([1, 1]))),
            Some((1, col([0, 1]))),
            Some((1, col([0, 2]))),
        ],
    };
    solve_recipe(&mut space, &[vec![0], vec![0]]).unwrap();
    assert_eq!(space.kept, [(0, 0), (1, 0)]);
    assert_eq!(space.recipe, [vec![q(1), q(-1)], vec![q(0), q(1)]]);
    space.lower[0] = Some((0, col([0, 3])));
    assert_eq!(
        solve_recipe(&mut space, &[vec![0], vec![0]]).err(),
        Some(
            ModuleError::RankMismatch {
                weight: vec![0, 0],
                rank: 1,
                multiplicity: 2
            }
            .into()
        )
    );
}

/// The direct `A_1` chain is the module the pattern enumeration builds:
/// identical weights, depths, offsets, norms, candidates, recipes, lowering
/// matrices and public indices, including the trivial irrep.
#[test]
fn rank_one_chain_is_the_enumerated_module() {
    let data = CartanData::new(RootSystem::A(1));
    for dj in 0..=12 {
        let weyl = (dj + 1) as usize;
        let (direct, dp) = rank_one(dj, weyl);
        let (general, gp) = from_enumeration(&data, &[dj], weyl).unwrap();
        assert_eq!(dp, gp, "public {dj}");
        assert_eq!(direct.dim, general.dim);
        assert_eq!(direct.weights.len(), general.weights.len());
        for (x, y) in direct.weights.iter().zip(&general.weights) {
            assert_eq!((&x.mu, &x.k, x.offset), (&y.mu, &y.k, y.offset), "{dj}");
            assert_eq!(
                (&x.norms, &x.kept, &x.recipe),
                (&y.norms, &y.kept, &y.recipe)
            );
            assert_eq!(x.lower, y.lower, "{dj} {:?}", x.mu);
            assert_eq!(direct.weight_index(&x.mu), general.weight_index(&x.mu));
        }
    }
}
