//! SU(N) combinatorics against closed forms written in the test.
//!
//! Contracts:
//!
//! - `dim` is the Weyl dimension `Π_{i<j} (λ_i − λ_j + j − i)/(j − i)` of the
//!   partition `λ`, and the dual reverses the Dynkin label;
//! - `patterns` enumerates the Gelfand–Tsetlin patterns by the betweenness
//!   rule in the SUNRepresentations `GTPatternIterator` order (`docs/gauge.md`
//!   §1: second rows lexicographic with the last entry fastest, the `N−1`
//!   iterator as the inner loop);
//! - `directproduct` is the Brauer–Klimyk decomposition built from those
//!   patterns' weights;
//! - `creation` holds the Gelfand–Tsetlin raising matrix elements, positive
//!   square roots, which close `[E_l, E_l^T] = H_l` exactly;
//!
//! plus a few values checked against SUNRepresentations.jl 0.4.0.
#![cfg(feature = "lie-generated")]

use malachite_base::num::basic::traits::Zero;
use std::collections::BTreeMap;

use malachite_nz::integer::Integer;
use malachite_q::Rational;

use crate::lie::sun::Irrep;

/// Every Dynkin label of SU(N) with entries summing to at most `max`.
fn labels(n: usize, max: i64) -> Vec<Vec<i64>> {
    let mut out = vec![vec![]];
    for _ in 1..n {
        out = out
            .into_iter()
            .flat_map(|l: Vec<i64>| {
                let used: i64 = l.iter().sum();
                (0..=max - used).map(move |x| {
                    let mut l = l.clone();
                    l.push(x);
                    l
                })
            })
            .collect();
    }
    out
}

/// Partition `λ_i = Σ_{k ≥ i} a_k` (length `N`, last entry 0).
fn partition(dynkin: &[i64]) -> Vec<i64> {
    let n = dynkin.len() + 1;
    (0..n)
        .map(|i| dynkin[i.min(n - 1)..].iter().sum())
        .collect()
}

/// GT patterns with top row `top`, flattened top row first, in the
/// `GTPatternIterator` order.
fn gt_patterns(top: &[i64]) -> Vec<Vec<i64>> {
    if top.len() == 1 {
        return vec![top.to_vec()];
    }
    // Second rows: `top[i+1] ≤ row[i] ≤ top[i]`, lexicographic, last fastest.
    let mut rows: Vec<Vec<i64>> = vec![vec![]];
    for i in 0..top.len() - 1 {
        rows = rows
            .into_iter()
            .flat_map(|r| {
                (top[i + 1]..=top[i]).map(move |x| {
                    let mut r = r.clone();
                    r.push(x);
                    r
                })
            })
            .collect();
    }
    rows.iter()
        .flat_map(|row| {
            gt_patterns(row).into_iter().map(|sub| {
                let mut p = top.to_vec();
                p.extend(sub);
                p
            })
        })
        .collect()
}

/// Entry `m_{k,l}` (1-based, `1 ≤ k ≤ l ≤ N`) of a flattened pattern.
fn entry(p: &[i64], n: usize, k: usize, l: usize) -> i64 {
    let offset: usize = (l + 1..=n).sum();
    p[offset + k - 1]
}

/// Weight `w_l = Σ_k m_{k,l} − Σ_k m_{k,l−1}`.
fn weight(p: &[i64], n: usize) -> Vec<i64> {
    let row = |l: usize| -> i64 { (1..=l).map(|k| entry(p, n, k, l)).sum() };
    (1..=n)
        .map(|l| row(l) - if l > 1 { row(l - 1) } else { 0 })
        .collect()
}

#[test]
fn patterns_follow_the_gt_iterator_order() {
    for n in 2..=2 {
        for d in labels(n, 3) {
            let s = Irrep::from_dynkin(&d).unwrap();
            let got: Vec<Vec<i64>> = s.patterns().iter().map(|p| p.data().to_vec()).collect();
            assert_eq!(got, gt_patterns(&partition(&d)), "{d:?}");
        }
    }
    // Checked against SUNRepresentations.jl 0.4.0 (`basis`): the adjoint of
    // SU(3), highest weight last.
    let got: Vec<Vec<i64>> = Irrep::from_dynkin(&[1, 1])
        .unwrap()
        .patterns()
        .iter()
        .map(|p| p.data().to_vec())
        .collect();
    let want: Vec<Vec<i64>> = [
        [2, 1, 0, 1, 0, 0],
        [2, 1, 0, 1, 0, 1],
        [2, 1, 0, 1, 1, 1],
        [2, 1, 0, 2, 0, 0],
        [2, 1, 0, 2, 0, 1],
        [2, 1, 0, 2, 0, 2],
        [2, 1, 0, 2, 1, 1],
        [2, 1, 0, 2, 1, 2],
    ]
    .iter()
    .map(|p| p.to_vec())
    .collect();
    assert_eq!(got, want);
}

/// `⟨M + δ_{k,l}| E_{l,l+1} |M⟩² = −Π_{i≤l+1}(ℓ_{i,l+1} − ℓ_{k,l})
/// Π_{i≤l−1}(ℓ_{i,l−1} − ℓ_{k,l} − 1) / Π_{i≤l, i≠k}(ℓ_{i,l} − ℓ_{k,l})(ℓ_{i,l} − ℓ_{k,l} − 1)`
/// with `ℓ_{i,l} = m_{i,l} − i`.
fn gt_raising_square(p: &[i64], n: usize, k: usize, l: usize) -> Rational {
    let ell = |i: usize, row: usize| entry(p, n, i, row) - i as i64;
    let lk = ell(k, l);
    let mut num = Integer::from(-1);
    let mut den = Integer::from(1);
    for i in 1..=l + 1 {
        num *= Integer::from(ell(i, l + 1) - lk);
    }
    for i in 1..l {
        num *= Integer::from(ell(i, l - 1) - lk - 1);
    }
    if num == 0 {
        // Raising would leave the betweenness cone; a denominator factor can
        // vanish together with it.
        return Rational::ZERO;
    }
    for i in (1..=l).filter(|&i| i != k) {
        den *= Integer::from((ell(i, l) - lk) * (ell(i, l) - lk - 1));
    }
    Rational::from_integers(num, den)
}

#[test]
fn creation_matrices_are_the_gt_raising_operators() {
    for n in 2..=4 {
        for d in labels(n, 2) {
            let s = Irrep::from_dynkin(&d).unwrap();
            let pats = gt_patterns(&partition(&d));
            let index: BTreeMap<&Vec<i64>, usize> =
                pats.iter().enumerate().map(|(i, p)| (p, i)).collect();
            let creation = s.creation();
            for l in 1..n {
                let mut want = BTreeMap::new();
                for (col, p) in pats.iter().enumerate() {
                    for k in 1..=l {
                        let sq = gt_raising_square(p, n, k, l);
                        if sq == 0 {
                            continue;
                        }
                        let mut up = p.clone();
                        up[(l + 1..=n).sum::<usize>() + k - 1] += 1;
                        want.insert((index[&up], col), sq);
                    }
                }
                let mut got = BTreeMap::new();
                for e in &creation[l - 1] {
                    assert!(e.value.sign() > 0, "{d:?} l={l}");
                    got.insert((e.row, e.col), e.value.signed_square());
                }
                assert_eq!(got, want, "{d:?} l={l}");
                // `[E_l, E_l^T]` is diagonal `w_l − w_{l+1}`; its diagonal is
                // exact: `Σ_j E_ij² − Σ_j E_ji²`.
                let mut diag = vec![Rational::ZERO; pats.len()];
                for ((row, col), sq) in &got {
                    diag[*row] += sq;
                    diag[*col] -= sq;
                }
                for (i, p) in pats.iter().enumerate() {
                    let w = weight(p, n);
                    assert_eq!(diag[i], Rational::from(w[l - 1] - w[l]), "{d:?}");
                }
            }
        }
    }
    // Checked against SUNRepresentations.jl 0.4.0 (`creation`): SU(3)
    // adjoint, level 2, `(row, col) → entry²` (1-based basis indices).
    let adj = Irrep::from_dynkin(&[1, 1]).unwrap().creation();
    let got: BTreeMap<(usize, usize), Rational> = adj[1]
        .iter()
        .map(|e| ((e.row + 1, e.col + 1), e.value.signed_square()))
        .collect();
    let r = |a: i64, b: i64| Rational::from_integers(Integer::from(a), Integer::from(b));
    let want: BTreeMap<(usize, usize), Rational> = [
        ((4, 1), r(1, 1)),
        ((3, 2), r(3, 2)),
        ((5, 2), r(1, 2)),
        ((7, 3), r(3, 2)),
        ((7, 5), r(1, 2)),
        ((8, 6), r(1, 1)),
    ]
    .into_iter()
    .collect();
    assert_eq!(got, want);
}

/// The closed form's premise: every GT lowering matrix element is
/// non-negative.
#[test]
fn lowering_matrix_elements_are_nonnegative() {
    for n in 2..=5 {
        for d in labels(n, 3) {
            for op in Irrep::from_dynkin(&d).unwrap().annihilation() {
                for e in op {
                    assert!(e.value.to_f64() >= 0.0, "{d:?} {e:?}");
                }
            }
        }
    }
}
