//! Independent oracles (`docs/gauge_exceptional.md` §9): the Chevalley
//! relations of §2.3, Weyl and Freudenthal counts, an `e_j` built from the
//! module axioms alone against the transpose of `f_j`, the lowering recipe
//! against the stored ladders, and basis facts derived by hand from §3–§4.
//! The relation and recipe checks are shared with the Gelfand–Tsetlin basis.

use super::*;
use crate::exact::SignedSqrtRational;
use crate::lie::RootSystem;
use std::collections::{BTreeMap, BTreeSet};

/// A vector in the rational basis `b`, keyed by magnetic index.
type Vector = BTreeMap<usize, Q>;
/// A sparse operator in the rational basis: `op[source]` lists `(target, coefficient)`.
type Op = Vec<Vec<(usize, Q)>>;

fn unit(rank: usize, i: usize) -> Vec<i64> {
    let mut l = vec![0; rank];
    l[i] = 1;
    l
}

fn build(root: RootSystem, lambda: &[i64]) -> (std::sync::Arc<CartanData>, IrrepModule) {
    let data = CartanData::new(root);
    let m = IrrepModule::new(&data, lambda).unwrap();
    (data, m)
}

/// `(weight space, local index)` of every magnetic index.
fn owners(m: &IrrepModule) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(m.dim);
    for (w, s) in m.weights.iter().enumerate() {
        assert_eq!(s.offset, out.len());
        out.extend((0..s.norms.len()).map(|a| (w, a)));
    }
    out
}

/// `f_j` from the stored `F`.
fn lowering(m: &IrrepModule, j: usize) -> Op {
    let mut op: Op = vec![Vec::new(); m.dim];
    for s in &m.weights {
        if let Some((src, f)) = &s.lower[j] {
            let src = &m.weights[*src];
            for (a, row) in f.iter().enumerate() {
                for (b, x) in row.iter().enumerate() {
                    if x.numerator_ref() != &0 {
                        op[src.offset + b].push((s.offset + a, x.clone()));
                    }
                }
            }
        }
    }
    op
}

/// `e_j` as the contravariant transpose `E[b][a] = F[a][b] n_a / n_b`.
fn raising_by_form(m: &IrrepModule, j: usize) -> Op {
    let mut op: Op = vec![Vec::new(); m.dim];
    for s in &m.weights {
        if let Some((src, f)) = &s.lower[j] {
            let src = &m.weights[*src];
            for (a, row) in f.iter().enumerate() {
                for (b, x) in row.iter().enumerate() {
                    if x.numerator_ref() != &0 {
                        let e = x * &s.norms[a] / &src.norms[b];
                        op[s.offset + a].push((src.offset + b, e));
                    }
                }
            }
        }
    }
    op
}

/// `e_j` from the module axioms only: `b_a = Σ_t recipe[a][t] f_{j_t} b_{q_t}`
/// and `e_j f_i = f_i e_j + δ_ij h_j`, depth by depth.
fn raising_by_recursion(m: &IrrepModule, j: usize, lowers: &[Op]) -> Op {
    let mut op: Op = vec![Vec::new(); m.dim];
    for s in m.weights.iter().skip(1) {
        for (a, row) in s.recipe.iter().enumerate() {
            let mut out = Vector::new();
            for (&(jt, q), r) in s.kept.iter().zip(row) {
                let src = &m.weights[s.lower[jt].as_ref().unwrap().0];
                let g = src.offset + q;
                let mut e_c = apply(&lowers[jt], &to_vec(&op[g]));
                if jt == j {
                    add(&mut e_c, g, Q::from(src.mu[j]));
                }
                for (i, x) in e_c {
                    add(&mut out, i, r * x);
                }
            }
            op[s.offset + a] = out
                .into_iter()
                .filter(|(_, x)| x.numerator_ref() != &0)
                .collect();
        }
    }
    op
}

fn to_vec(col: &[(usize, Q)]) -> Vector {
    col.iter().cloned().collect()
}

fn add(v: &mut Vector, i: usize, x: Q) {
    let e = v.entry(i).or_insert_with(|| Q::from(0));
    *e += x;
    if e.numerator_ref() == &0 {
        v.remove(&i);
    }
}

fn apply(op: &Op, v: &Vector) -> Vector {
    let mut out = Vector::new();
    for (&i, x) in v {
        for (t, y) in &op[i] {
            add(&mut out, *t, x * y);
        }
    }
    out
}

fn word(ops: &[&Op], v: &Vector) -> Vector {
    ops.iter().rev().fold(v.clone(), |v, op| apply(op, &v))
}

fn binomial(n: i64, k: i64) -> i64 {
    (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

/// `ad(x_i)^n x_j = Σ_k (−1)^k C(n, k) x_i^{n−k} x_j x_i^k` on `v`.
fn serre(x: &[Op], i: usize, j: usize, n: i64, v: &Vector) -> Vector {
    let mut out = Vector::new();
    for k in 0..=n {
        let mut ops: Vec<&Op> = vec![&x[i]; (n - k) as usize];
        ops.push(&x[j]);
        ops.extend(std::iter::repeat_n(&x[i], k as usize));
        let c = binomial(n, k) * if k % 2 == 0 { 1 } else { -1 };
        for (t, y) in word(&ops, v) {
            add(&mut out, t, y * Q::from(c));
        }
    }
    out
}

/// §9.4 and e_i† = f_i, exactly in the rational basis.
fn check_relations(root: RootSystem, lambda: &[i64]) {
    let (data, m) = build(root, lambda);
    check_relations_of(&data, &m);
}

/// The Chevalley relations of §2.3 on `m`, with `e_j` both as the
/// contravariant transpose of `f_j` and from the recipe and module axioms.
pub(crate) fn check_relations_of(data: &CartanData, m: &IrrepModule) {
    let (root, lambda) = (data.root, &m.weights[0].mu);
    let r = lambda.len();
    let f: Vec<Op> = (0..r).map(|j| lowering(m, j)).collect();
    let e: Vec<Op> = (0..r).map(|j| raising_by_form(m, j)).collect();
    for (j, e_j) in e.iter().enumerate() {
        assert_eq!(
            *e_j,
            raising_by_recursion(m, j, &f),
            "{root:?} {lambda:?} e_{j}"
        );
    }
    let own = owners(m);
    for g in 0..m.dim {
        let mu = &m.weights[own[g].0].mu;
        let v = Vector::from([(g, Q::from(1))]);
        for i in 0..r {
            // [h_i, e_j] = cartan(j, i) e_j: e_j raises the weight by α_j.
            for j in 0..r {
                for &(t, _) in &e[j][g] {
                    let nu = &m.weights[own[t].0].mu;
                    assert!((0..r).all(|l| nu[l] == mu[l] + data.c(j, l)));
                }
                let mut comm = word(&[&e[i], &f[j]], &v);
                for (t, y) in word(&[&f[j], &e[i]], &v) {
                    add(&mut comm, t, -y);
                }
                let expected = if i == j && mu[i] != 0 {
                    Vector::from([(g, Q::from(mu[i]))])
                } else {
                    Vector::new()
                };
                assert_eq!(comm, expected, "{root:?} {lambda:?} [e_{i}, f_{j}] at {g}");
                if i != j {
                    let n = 1 - data.c(j, i);
                    assert!(serre(&e, i, j, n, &v).is_empty(), "e Serre {i} {j}");
                    assert!(serre(&f, i, j, n, &v).is_empty(), "f Serre {i} {j}");
                }
            }
        }
    }
}

/// Each recipe reproduces its basis vector exactly: with the candidate
/// `c_t = f_{j_t} b_{q_t}` having coordinates `F_{j_t}[·][q_t]` in the
/// weight space, `Σ_t recipe[k][t] F_{j_t}[a][q_t] = δ_ak`.
pub(crate) fn check_recipes(m: &IrrepModule) {
    for s in &m.weights[1..] {
        let dim = s.norms.len();
        assert_eq!((s.recipe.len(), s.kept.len()), (dim, dim), "{:?}", s.mu);
        for (k, row) in s.recipe.iter().enumerate() {
            for a in 0..dim {
                let got = row
                    .iter()
                    .zip(&s.kept)
                    .map(|(r, &(j, q))| r * &s.lower[j].as_ref().unwrap().1[a][q])
                    .fold(Q::from(0), |acc, x| acc + x);
                assert_eq!(got, Q::from(u8::from(a == k)), "{:?} b_{k} at {a}", s.mu);
            }
        }
    }
}

/// Weyl dimension and Freudenthal multiplicity of every weight space (§4.2, §9.1–9.2).
fn check_ranks(root: RootSystem, lambda: &[i64], weyl: usize) {
    let (data, m) = build(root, lambda);
    assert_eq!(m.dim, weyl);
    let mult = data.weight_multiplicities(lambda).unwrap();
    assert_eq!(mult.len(), m.weights.len());
    for s in &m.weights {
        assert_eq!(s.norms.len() as u64, mult[&s.mu], "{root:?} {:?}", s.mu);
        assert_eq!(m.weight_index(&s.mu).map(|w| &m.weights[w].mu), Some(&s.mu));
    }
}

fn cases() -> Vec<(RootSystem, Vec<i64>, usize)> {
    vec![
        (RootSystem::G2, vec![0, 0], 1),
        (RootSystem::G2, vec![1, 0], 7),
        (RootSystem::G2, vec![0, 1], 14),
        (RootSystem::G2, vec![2, 0], 27),
        (RootSystem::G2, vec![1, 1], 64),
        (RootSystem::F4, unit(4, 3), 26),
        (RootSystem::E6, unit(6, 0), 27),
        (RootSystem::E6, unit(6, 5), 27),
        (RootSystem::E7, unit(7, 6), 56),
        (RootSystem::E8, unit(8, 7), 248),
    ]
}

#[test]
fn generator_relations_exact() {
    for (root, lambda, _) in cases() {
        check_relations(root, &lambda);
    }
}

/// The Gram–Schmidt recipe is unit lower triangular (§4.3) and reproduces
/// each basis vector from the kept candidates.
#[test]
fn gram_schmidt_recipes_are_unit_lower_triangular_and_exact() {
    for (root, lambda, _) in cases()
        .into_iter()
        .chain([(RootSystem::F4, unit(4, 2), 273)])
    {
        let (_, m) = build(root, &lambda);
        check_recipes(&m);
        for s in &m.weights[1..] {
            for (k, row) in s.recipe.iter().enumerate() {
                assert_eq!(row[k], Q::from(1));
                assert!(row[k + 1..].iter().all(|x| x.numerator_ref() == &0));
            }
        }
    }
}

/// Gate 1 rejects a weight space whose selected rank differs from the
/// declared multiplicity, and an assembled module whose dimension differs
/// from the Weyl dimension.
#[test]
fn rank_and_dimension_gates_reject_discrepancies() {
    let data = CartanData::new(RootSystem::G2);
    let order = depth_order(&data, &[2, 0]).unwrap();
    let m = IrrepModule::new(&data, &[2, 0]).unwrap();
    let w = order.index[&vec![0, 0]];
    let (k, mu) = order.spaces[w].clone();
    let err = build_space(&data, &m.weights[..w], &order.index, mu, k, 0, 4).err();
    assert_eq!(
        err,
        Some(
            ModuleError::RankMismatch {
                weight: vec![0, 0],
                rank: 3,
                multiplicity: 4
            }
            .into()
        )
    );
    let mut wrong = depth_order(&data, &[2, 0]).unwrap();
    wrong.weyl += 1;
    assert_eq!(
        IrrepModule::from_spaces(m.weights, wrong).err(),
        Some(
            ModuleError::DimensionMismatch {
                built: 27,
                weyl: 28
            }
            .into()
        )
    );
}

#[test]
fn rank_gates() {
    for (root, lambda, weyl) in cases() {
        check_ranks(root, &lambda, weyl);
    }
}

/// G2 `7 = (1, 0)` with α₀ = (2, −1), α₁ = (−3, 2) (§2.1). Lowering from λ
/// along the only allowed simple roots gives one weight per depth (§3):
/// (1,0) →f₀ (−1,1) →f₁ (2,−1) →f₀ (0,0) →f₀ (−2,1) →f₁ (1,−1) →f₀ (−1,0).
/// Each is multiplicity-free with the single candidate shown, so each entry is
/// `+√‖f_j u‖²` with `‖f_j u‖² = μ_j + ‖e_j u‖²`: 1, 1, √2, √2, 1, 1 (the α₀
/// string (2,−1), (0,0), (−2,1) is a spin-1 triplet).
#[test]
fn g2_7_order_candidates_and_entries() {
    let (_, m) = build(RootSystem::G2, &[1, 0]);
    let mus: Vec<Vec<i64>> = m.weights.iter().map(|s| s.mu.clone()).collect();
    let expected: Vec<Vec<i64>> = [[1, 0], [-1, 1], [2, -1], [0, 0], [-2, 1], [1, -1], [-1, 0]]
        .iter()
        .map(|x| x.to_vec())
        .collect();
    assert_eq!(mus, expected);
    let ks: Vec<Vec<i64>> = m.weights.iter().map(|s| s.k.clone()).collect();
    assert_eq!(
        ks,
        [[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [3, 2], [4, 2]].map(|x| x.to_vec())
    );
    let js = [0, 1, 0, 0, 1, 0];
    let radicands = [1, 1, 2, 2, 1, 1];
    for (w, (&j, &q)) in js.iter().zip(&radicands).enumerate() {
        let s = &m.weights[w + 1];
        assert_eq!(s.kept, vec![(j, 0)]);
        let x = m.f_entry(w + 1, j, 0, 0);
        assert_eq!(
            (x.sign(), x.radicand().clone()),
            (
                1,
                malachite_q::Rational::from(malachite_nz::integer::Integer::from(q))
            )
        );
    }
}

/// E6 27 and 27̄ are minuscule: every `f_i` entry is 0 or +1 (§6.3 proof).
#[test]
fn e6_minuscule_entries_are_zero_or_one() {
    for lambda in [unit(6, 0), unit(6, 5)] {
        let (_, m) = build(RootSystem::E6, &lambda);
        let mut ones = 0;
        for (w, s) in m.weights.iter().enumerate() {
            for (j, l) in s.lower.iter().enumerate() {
                if l.is_some() {
                    let x = m.f_entry(w, j, 0, 0);
                    assert_eq!(
                        (x.sign(), x.radicand().clone()),
                        (
                            1,
                            malachite_q::Rational::from(malachite_nz::integer::Integer::from(1))
                        )
                    );
                    ones += 1;
                }
            }
        }
        // One entry per edge μ+α_j → μ of the weight graph.
        assert_eq!(ones, 36);
    }
}

/// G2 27 = (2, 0), zero weight. Dominant weights of 27 are (2,0), (0,1),
/// (1,0), (0,0) with multiplicities 1, 1, 2, 3 (6 + 6 + 12 + 3 = 27).
/// Candidates (§4.1): j = 0 from 0 + α₀ = (2,−1), Weyl-conjugate to (1,0)
/// (s₁ then s₀), so 2 vectors; j = 1 from 0 + α₁ = (−3,2), conjugate to (0,1)
/// (s₀ then s₁), so 1 vector. Three candidates span a 3-dimensional space, so
/// §4.2 keeps all of them: (0,0), (0,1), (1,0).
#[test]
fn g2_27_zero_weight_keeps_every_candidate() {
    let (_, m) = build(RootSystem::G2, &[2, 0]);
    let s = &m.weights[m.weight_index(&[0, 0]).unwrap()];
    assert_eq!(s.kept, vec![(0, 0), (0, 1), (1, 0)]);
    // §4.3: b_k = c_k − (earlier b), so its own coefficient is 1 and later ones 0.
    for (k, row) in s.recipe.iter().enumerate() {
        assert_eq!(row[k], Q::from(1));
        assert!(row[k + 1..].iter().all(|x| x.numerator_ref() == &0));
    }
}

#[test]
fn determinism() {
    for (root, lambda, _) in cases() {
        let (_, a) = build(root, &lambda);
        let (_, b) = build(root, &lambda);
        for (x, y) in a.weights.iter().zip(&b.weights) {
            assert_eq!(
                (&x.mu, &x.norms, &x.kept, &x.recipe, &x.lower),
                (&y.mu, &y.norms, &y.kept, &y.recipe, &y.lower)
            );
        }
    }
}

#[test]
fn invalid_and_huge_labels() {
    let g2 = CartanData::new(RootSystem::G2);
    for bad in [&[1][..], &[0, -1], &[0, 0, 0]] {
        assert_eq!(
            IrrepModule::new(&g2, bad).err(),
            Some(LieError::InvalidLabel)
        );
    }
    for huge in [[i64::MAX, 0], [0, 1 << 20], [i64::MAX, i64::MAX]] {
        assert_eq!(
            IrrepModule::new(&g2, &huge).err(),
            Some(LieError::LabelOverflow)
        );
    }
}

/// Release build times (`cargo test --release -- --ignored --nocapture`).
#[test]
#[ignore]
fn build_times() {
    for (root, lambda) in [
        (RootSystem::G2, vec![1, 1]),
        (RootSystem::F4, unit(4, 3)),
        (RootSystem::E7, unit(7, 6)),
        (RootSystem::E8, unit(8, 7)),
    ] {
        let data = CartanData::new(root);
        let t = std::time::Instant::now();
        let m = IrrepModule::new(&data, &lambda).unwrap();
        println!("{root:?} {lambda:?} dim {} in {:?}", m.dim, t.elapsed());
    }
}

/// Single-thread release module-build measurements. Filter a
/// case with FS_PERF_CASE (e.g. E8-27000); repeat with FS_PERF_SAMPLES.
#[test]
#[ignore = "release performance report"]
fn performance_modules() {
    let samples: usize = std::env::var("FS_PERF_SAMPLES")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .unwrap();
    for (name, root, lambda) in [
        ("G2-27", RootSystem::G2, vec![2, 0]),
        ("G2-64", RootSystem::G2, vec![1, 1]),
        ("F4-273", RootSystem::F4, unit(4, 2)),
        ("F4-324", RootSystem::F4, vec![0, 0, 0, 2]),
        ("E7-912", RootSystem::E7, unit(7, 1)),
        ("E7-1539", RootSystem::E7, unit(7, 5)),
        ("E8-3875", RootSystem::E8, unit(8, 0)),
        ("E8-27000", RootSystem::E8, vec![0, 0, 0, 0, 0, 0, 0, 2]),
        ("E8-30380", RootSystem::E8, unit(8, 6)),
        ("E6-7371", RootSystem::E6, vec![1, 0, 0, 0, 1, 0]),
    ] {
        if std::env::var("FS_PERF_CASE").is_ok_and(|filter| filter != name) {
            continue;
        }
        let data = CartanData::new(root);
        for sample in 0..samples {
            let t = std::time::Instant::now();
            let m = IrrepModule::new(&data, &lambda).unwrap();
            let elapsed = t.elapsed();
            let max_candidates = m
                .weights
                .iter()
                .map(|w| {
                    w.lower
                        .iter()
                        .flatten()
                        .map(|(src, _)| m.weights[*src].norms.len())
                        .sum::<usize>()
                })
                .max()
                .unwrap();
            let mut depths = BTreeMap::<i64, usize>::new();
            let mut max_mult = 0;
            for w in &m.weights {
                *depths.entry(w.k.iter().sum()).or_default() += 1;
                max_mult = max_mult.max(w.norms.len());
            }
            println!("{name} sample={sample} seconds={:.9} dim={} spaces={} depths={} max_width={} max_mult={max_mult} max_candidates={max_candidates} bytes={}", elapsed.as_secs_f64(), m.dim, m.weights.len(), depths.len(), depths.values().max().unwrap(), m.bytes());
        }
    }
}

fn q(n: i64, d: i64) -> Q {
    Q::from_integers(n.into(), d.into())
}

/// Golden values, cross-checked against an independent exact computation of
/// the PBW Shapovalov form from the spec text (not from this module).
#[test]
fn golden_values() {
    let (_, g27) = build(RootSystem::G2, &[2, 0]);
    for mu in [[2, -1], [1, -1]] {
        let x = g27.f_entry(g27.weight_index(&mu).unwrap(), 1, 1, 1);
        assert_eq!((x.sign(), x.radicand().clone()), (-1, q(2, 3)), "{mu:?}");
    }
    let zero = &g27.weights[g27.weight_index(&[0, 0]).unwrap()];
    assert_eq!(zero.norms, vec![q(96, 1), q(16, 1), q(84, 1)]);

    let (_, f26) = build(RootSystem::F4, &unit(4, 3));
    let w = f26.weight_index(&[0; 4]).unwrap();
    let zero = &f26.weights[w];
    assert_eq!(zero.kept, vec![(2, 0), (3, 0)]);
    assert_eq!(zero.norms, vec![q(2, 1), q(3, 2)]);
    let radicands: BTreeSet<_> = entries(&f26)
        .into_iter()
        .filter(|x| x.sign() != 0)
        .map(|x| x.radicand().clone())
        .collect();
    let expected = [q(1, 1), q(1, 2), q(2, 1), q(3, 2)];
    assert_eq!(radicands, BTreeSet::from(expected));

    let (_, g64) = build(RootSystem::G2, &[1, 1]);
    let all = entries(&g64);
    let nonzero = all.iter().filter(|x| x.sign() != 0).count();
    let negative = all.iter().filter(|x| x.sign() < 0).count();
    assert_eq!((negative, nonzero), (19, 143));
}

/// Every stored ladder entry `⟨u_a, f_j u_b⟩`.
fn entries(m: &IrrepModule) -> Vec<SignedSqrtRational> {
    let mut out = Vec::new();
    for (w, s) in m.weights.iter().enumerate() {
        for (j, l) in s.lower.iter().enumerate() {
            if let Some((src, _)) = l {
                for a in 0..s.norms.len() {
                    for b in 0..m.weights[*src].norms.len() {
                        out.push(m.f_entry(w, j, a, b));
                    }
                }
            }
        }
    }
    out
}

/// Gate 1 and the recipe identity on large modules (release).
#[test]
#[ignore = "large exact modules; run in release"]
fn large_modules_satisfy_rank_gates_and_recipes() {
    for (root, lambda, weyl) in [
        (RootSystem::E8, unit(8, 0), 3875),
        (RootSystem::E8, vec![0, 0, 0, 0, 0, 0, 0, 2], 27000),
        (RootSystem::E8, unit(8, 6), 30380),
        (RootSystem::E6, vec![1, 0, 0, 0, 1, 0], 7371),
    ] {
        check_ranks(root, &lambda, weyl);
        check_recipes(&build(root, &lambda).1);
    }
}
