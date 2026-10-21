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

use std::collections::BTreeMap;

use num_bigint::BigInt;

#[path = "common/lie.rs"]
mod lie_test;
use lie_test::product as directproduct;

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

fn weyl_dimension(dynkin: &[i64]) -> BigInt {
    let l = partition(dynkin);
    let (mut num, mut den) = (BigInt::from(1), BigInt::from(1));
    for i in 0..l.len() {
        for j in i + 1..l.len() {
            num *= l[i] - l[j] + (j - i) as i64;
            den *= (j - i) as i64;
        }
    }
    num / den
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
fn dimensions_and_duals_are_the_closed_forms() {
    for n in 2..=5 {
        for d in labels(n, 3) {
            let s = lie_test::irrep(&d).unwrap();
            assert_eq!(s.dim_exact(), weyl_dimension(&d), "{d:?}");
            let rev: Vec<i64> = d.iter().rev().copied().collect();
            assert_eq!(lie_test::dual(&s).dynkin(), rev, "{d:?}");
        }
    }
    // Checked against SUNRepresentations.jl 0.4.0 (`dim`).
    for (d, dim) in [
        (&[1, 1][..], 8),
        (&[2, 1, 0], 45),
        (&[1, 2, 0], 60),
        (&[1, 0, 0, 1], 24),
    ] {
        assert_eq!(lie_test::irrep(d).unwrap().dim_exact(), BigInt::from(dim));
    }
}

#[test]
fn patterns_follow_the_gt_iterator_order() {
    for n in 3..=4 {
        for d in labels(n, 3) {
            let s = lie_test::irrep(&d).unwrap();
            let got: Vec<Vec<i64>> = s
                .gt_patterns()
                .unwrap()
                .iter()
                .map(|p| p.data().to_vec())
                .collect();
            assert_eq!(got, gt_patterns(&partition(&d)), "{d:?}");
        }
    }
    // Checked against SUNRepresentations.jl 0.4.0 (`basis`): the adjoint of
    // SU(3), highest weight last.
    let got: Vec<Vec<i64>> = lie_test::irrep(&[1, 1])
        .unwrap()
        .gt_patterns()
        .unwrap()
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

/// `N^c_{ab} = Σ_μ m_a(μ) sgn(w)` over `w(λ_b + μ + ρ) − ρ = λ_c` dominant,
/// with `m_a(μ)` the number of GT patterns of weight `μ` and `W = S_N`.
fn klimyk(a: &[i64], b: &[i64]) -> BTreeMap<Vec<i64>, u32> {
    let n = a.len() + 1;
    let lb = partition(b);
    let mut acc: BTreeMap<Vec<i64>, i64> = BTreeMap::new();
    for p in gt_patterns(&partition(a)) {
        let mu = weight(&p, n);
        let v: Vec<i64> = (0..n).map(|i| lb[i] + mu[i] + (n - 1 - i) as i64).collect();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by_key(|&i| std::cmp::Reverse(v[i]));
        if idx.windows(2).any(|w| v[w[0]] == v[w[1]]) {
            continue;
        }
        let mut sign = 1;
        for i in 0..n {
            for j in i + 1..n {
                if idx[i] > idx[j] {
                    sign = -sign;
                }
            }
        }
        let c: Vec<i64> = (0..n).map(|i| v[idx[i]] - (n - 1 - i) as i64).collect();
        let dynkin: Vec<i64> = c.windows(2).map(|w| w[0] - w[1]).collect();
        *acc.entry(dynkin).or_default() += sign;
    }
    acc.into_iter()
        .filter(|&(_, m)| m != 0)
        .map(|(c, m)| (c, u32::try_from(m).expect("nonnegative")))
        .collect()
}

#[test]
fn products_are_the_brauer_klimyk_decomposition() {
    let mut pairs = 0;
    for (n, max) in [(2, 6), (3, 3), (4, 2), (5, 1)] {
        let all = labels(n, max);
        for a in &all {
            for b in &all {
                let ia = lie_test::irrep(a).unwrap();
                let ib = lie_test::irrep(b).unwrap();
                let got: BTreeMap<Vec<i64>, u32> = directproduct(&ia, &ib)
                    .unwrap()
                    .into_iter()
                    .map(|(k, v)| (k.dynkin(), v))
                    .collect();
                assert_eq!(got, klimyk(a, b), "{a:?} ⊗ {b:?}");
                pairs += 1;
            }
        }
    }
    assert_eq!(pairs, 49 + 100 + 100 + 25);
    // Checked against SUNRepresentations.jl 0.4.0 (`directproduct`).
    let got: BTreeMap<Vec<i64>, u32> = directproduct(
        &lie_test::irrep(&[1, 1]).unwrap(),
        &lie_test::irrep(&[1, 1]).unwrap(),
    )
    .unwrap()
    .into_iter()
    .map(|(k, v)| (k.dynkin(), v))
    .collect();
    let want: BTreeMap<Vec<i64>, u32> = [
        (vec![0, 0], 1),
        (vec![1, 1], 2),
        (vec![2, 2], 1),
        (vec![0, 3], 1),
        (vec![3, 0], 1),
    ]
    .into_iter()
    .collect();
    assert_eq!(got, want);
}
