//! Independent oracles (`docs/gauge_exceptional.md` §9): the Chevalley
//! relations of §2.3, Weyl and Freudenthal counts, an `e_j` built from the
//! module axioms alone against the transpose of `f_j`, and basis facts derived
//! by hand from §3–§4.

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

fn build(root: RootSystem, lambda: &[i64]) -> (CartanData, IrrepModule) {
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
                    if !x.is_zero() {
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
                    if !x.is_zero() {
                        let e = x * &s.norms[a] / &src.norms[b];
                        op[s.offset + a].push((src.offset + b, e));
                    }
                }
            }
        }
    }
    op
}

/// `e_j` from the module axioms only: `b_a = Σ_t gs[a][t] f_{j_t} b_{q_t}` and
/// `e_j f_i = f_i e_j + δ_ij h_j`, depth by depth.
fn raising_by_recursion(m: &IrrepModule, j: usize, lowers: &[Op]) -> Op {
    let mut op: Op = vec![Vec::new(); m.dim];
    for s in m.weights.iter().skip(1) {
        for (a, row) in s.gs.iter().enumerate() {
            let mut out = Vector::new();
            for (&(jt, q), r) in s.kept.iter().zip(row) {
                let src = &m.weights[s.lower[jt].as_ref().unwrap().0];
                let g = src.offset + q;
                let mut e_c = apply(&lowers[jt], &to_vec(&op[g]));
                if jt == j {
                    add(&mut e_c, g, Q::from_integer(src.mu[j].into()));
                }
                for (i, x) in e_c {
                    add(&mut out, i, r * x);
                }
            }
            op[s.offset + a] = out.into_iter().filter(|(_, x)| !x.is_zero()).collect();
        }
    }
    op
}

fn to_vec(col: &[(usize, Q)]) -> Vector {
    col.iter().cloned().collect()
}

fn add(v: &mut Vector, i: usize, x: Q) {
    let e = v.entry(i).or_insert_with(Q::zero);
    *e += x;
    if e.is_zero() {
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
            add(&mut out, t, y * Q::from_integer(c.into()));
        }
    }
    out
}

/// §9.4 and e_i† = f_i, exactly in the rational basis.
fn check_relations(root: RootSystem, lambda: &[i64]) {
    let (data, m) = build(root, lambda);
    let r = lambda.len();
    let f: Vec<Op> = (0..r).map(|j| lowering(&m, j)).collect();
    let e: Vec<Op> = (0..r).map(|j| raising_by_form(&m, j)).collect();
    for (j, e_j) in e.iter().enumerate() {
        assert_eq!(
            *e_j,
            raising_by_recursion(&m, j, &f),
            "{root:?} {lambda:?} e_{j}"
        );
    }
    let own = owners(&m);
    for g in 0..m.dim {
        let mu = &m.weights[own[g].0].mu;
        let v = Vector::from([(g, Q::from_integer(1.into()))]);
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
                    Vector::from([(g, Q::from_integer(mu[i].into()))])
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
            (1, Q::from_integer(q.into()))
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
                        (1, Q::from_integer(1.into()))
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
    for (k, row) in s.gs.iter().enumerate() {
        assert_eq!(row[k], Q::from_integer(1.into()));
        assert!(row[k + 1..].iter().all(Zero::is_zero));
    }
}

#[test]
fn determinism() {
    for (root, lambda, _) in cases() {
        let (_, a) = build(root, &lambda);
        let (_, b) = build(root, &lambda);
        for (x, y) in a.weights.iter().zip(&b.weights) {
            assert_eq!(
                (&x.mu, &x.norms, &x.kept, &x.gs, &x.lower),
                (&y.mu, &y.norms, &y.kept, &y.gs, &y.lower)
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

/// Single-thread release measurements and exact-module drift checks. Filter a
/// case with FS_PERF_CASE (e.g. E8-27000); repeat with FS_PERF_SAMPLES.
#[test]
#[ignore = "release performance report"]
fn performance_modules() {
    use std::hash::{DefaultHasher, Hash, Hasher};
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
            let mut hash = DefaultHasher::new();
            let mut depths = BTreeMap::<i64, usize>::new();
            let mut max_mult = 0;
            for w in &m.weights {
                (&w.mu, &w.k, w.offset, &w.norms, &w.kept, &w.gs, &w.lower).hash(&mut hash);
                *depths.entry(w.k.iter().sum()).or_default() += 1;
                max_mult = max_mult.max(w.norms.len());
            }
            // Exact internal-state hashes pinned on 854d838e; drift detectors.
            let expected = match name {
                "G2-27" => 0x98dca8b5927fdde9,
                "G2-64" => 0x9f32818cded4dda3,
                "F4-273" => 0xb6c9b017f19398d7,
                "F4-324" => 0x72f327c59a64ebaa,
                "E7-912" => 0x293f5a027dd425e7,
                "E7-1539" => 0xa74a83768eb11a82,
                "E8-3875" => 0x76563c5ac7e240cc,
                "E8-27000" => 0x7d4ed4abef86235f,
                "E8-30380" => 0x46ebcf415bfd3db9,
                "E6-7371" => 0xd6f8ec25d1d901d4,
                _ => unreachable!(),
            };
            assert_eq!(hash.finish(), expected, "{name}");
            println!("{name} sample={sample} seconds={:.9} dim={} spaces={} depths={} max_width={} max_mult={max_mult} max_candidates={max_candidates} bytes={} exact_hash={:016x}", elapsed.as_secs_f64(), m.dim, m.weights.len(), depths.len(), depths.values().max().unwrap(), m.bytes(), hash.finish());
        }
    }
}

fn q(n: i64, d: i64) -> Q {
    Q::new(n.into(), d.into())
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
    let radicands: BTreeSet<Q> = entries(&f26)
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

// The c236d39 §4 recursion is a drift reference, not an independent oracle.
fn old_build_space(
    data: &CartanData,
    weights: &[WeightSpace],
    index: &HashMap<Vec<i64>, usize>,
    mu: Vec<i64>,
    k: Vec<i64>,
    offset: usize,
    multiplicity: u64,
) -> Result<WeightSpace, LieError> {
    let rank = mu.len();
    let raise =
        |nu: &[i64], j: usize| -> Vec<i64> { (0..rank).map(|i| nu[i] + data.c(j, i)).collect() };
    // §4.1: candidates (j, source weight, q); `start[j]` is where j's block begins.
    let mut cands: Vec<(usize, usize, usize)> = Vec::new();
    let mut start = vec![None; rank];
    for (j, slot) in start.iter_mut().enumerate() {
        if let Some(&src) = index.get(&raise(&mu, j)) {
            *slot = Some(cands.len());
            cands.extend((0..weights[src].norms.len()).map(|q| (j, src, q)));
        }
    }

    // §4.4: ⟨f_j w, f_k w′⟩ = ⟨e_k w, e_j w′⟩ + δ_jk (μ+α_j)_j ⟨w, w′⟩, with
    // ⟨e_k b_q, e_j b_q′⟩ = Σ_p F_k[q][p] F_j[q′][p] n_q n_q′ / n_p by contravariance.
    let c = cands.len();
    let mut gram = vec![vec![Q::zero(); c]; c];
    for s in 0..c {
        let (j, v1, q) = cands[s];
        for t in s..c {
            let (kk, v2, q2) = cands[t];
            let (w1, w2) = (&weights[v1], &weights[v2]);
            let mut g = Q::zero();
            if let (Some((tau, f1)), Some((_, f2))) = (&w1.lower[kk], &w2.lower[j]) {
                let n_tau = &weights[*tau].norms;
                for (p, n_p) in n_tau.iter().enumerate() {
                    if !f1[q][p].is_zero() && !f2[q2][p].is_zero() {
                        g += &f1[q][p] * &f2[q2][p] / n_p;
                    }
                }
                if !g.is_zero() {
                    g *= &w1.norms[q] * &w2.norms[q2];
                }
            }
            if j == kk && q == q2 {
                g += Q::from_integer(w1.mu[j].into()) * &w1.norms[q];
            }
            gram[t][s] = g.clone();
            gram[s][t] = g;
        }
    }

    // §4.2/§4.3: greedy selection by exact Gram–Schmidt residual, in
    // candidate order; coefficients are over all candidates.
    let mut full: Vec<Vec<Q>> = Vec::new();
    let mut norms: Vec<Q> = Vec::new();
    let mut kept_idx: Vec<usize> = Vec::new();
    for t in 0..c {
        let mut coeffs = vec![Q::zero(); c];
        coeffs[t] = Q::from_integer(1.into());
        let mut n = gram[t][t].clone();
        for (r, n_l) in full.iter().zip(&norms) {
            let proj: Q = r
                .iter()
                .zip(&gram[t])
                .filter(|(x, g)| !x.is_zero() && !g.is_zero())
                .map(|(x, g)| x * g)
                .sum();
            if proj.is_zero() {
                continue;
            }
            let scale = &proj / n_l;
            n -= &scale * &proj;
            for (x, y) in coeffs.iter_mut().zip(r) {
                if !y.is_zero() {
                    *x -= &scale * y;
                }
            }
        }
        if n.is_negative() {
            return Err(ModuleError::IndefiniteForm { weight: mu }.into());
        }
        if !n.is_zero() {
            full.push(coeffs);
            norms.push(n);
            kept_idx.push(t);
        }
    }
    if u64::try_from(norms.len()).ok() != Some(multiplicity) {
        return Err(ModuleError::RankMismatch {
            weight: mu,
            rank: norms.len(),
            multiplicity,
        }
        .into());
    }

    // §4.5: F[a][b] = ⟨f_j b_b, b_a⟩ / n_a, and f_j b_b is candidate (j, b).
    let lower = (0..rank)
        .map(|j| {
            let s0 = start[j]?;
            let src = cands[s0].1;
            let f = full
                .iter()
                .zip(&norms)
                .map(|(r, n_a)| {
                    (0..weights[src].norms.len())
                        .map(|b| {
                            let col = s0 + b;
                            let dot: Q = r
                                .iter()
                                .zip(&gram)
                                .filter(|(x, row)| !x.is_zero() && !row[col].is_zero())
                                .map(|(x, row)| x * &row[col])
                                .sum();
                            dot / n_a
                        })
                        .collect()
                })
                .collect();
            Some((src, f))
        })
        .collect();
    let gs = full
        .iter()
        .map(|r| kept_idx.iter().map(|&t| r[t].clone()).collect())
        .collect();
    let kept = kept_idx.iter().map(|&t| (cands[t].0, cands[t].2)).collect();
    Ok(WeightSpace {
        mu,
        k,
        offset,
        norms,
        kept,
        gs,
        lower,
    })
}

fn compare_old_module(root: RootSystem, lambda: &[i64]) {
    let data = CartanData::new(root);
    let new = IrrepModule::new(&data, lambda).unwrap();
    let mut old = vec![WeightSpace {
        mu: lambda.to_vec(),
        k: vec![0; lambda.len()],
        offset: 0,
        norms: vec![Q::from_integer(1.into())],
        kept: Vec::new(),
        gs: vec![Vec::new()],
        lower: vec![None; lambda.len()],
    }];
    for w in &new.weights[1..] {
        let space = old_build_space(
            &data,
            &old,
            &new.index,
            w.mu.clone(),
            w.k.clone(),
            w.offset,
            w.norms.len() as u64,
        )
        .unwrap();
        assert_eq!(
            (
                &space.mu,
                &space.k,
                space.offset,
                &space.norms,
                &space.kept,
                &space.gs,
                &space.lower
            ),
            (&w.mu, &w.k, w.offset, &w.norms, &w.kept, &w.gs, &w.lower),
            "{root:?} {lambda:?} {:?}",
            w.mu
        );
        old.push(space);
    }
    assert_eq!(old.iter().map(|w| w.norms.len()).sum::<usize>(), new.dim);
}

#[test]
fn module_matches_old_path_exactly() {
    for (root, lambda, _) in cases() {
        compare_old_module(root, &lambda);
    }
    compare_old_module(RootSystem::F4, &unit(4, 2));
}

#[test]
#[ignore = "large exact old/new module comparisons; run in release"]
fn large_modules_match_old_path_exactly() {
    for (root, lambda) in [
        (RootSystem::E8, unit(8, 0)),
        (RootSystem::E8, vec![0, 0, 0, 0, 0, 0, 0, 2]),
        (RootSystem::E8, unit(8, 6)),
        (RootSystem::E6, vec![1, 0, 0, 0, 1, 0]),
    ] {
        compare_old_module(root, &lambda);
    }
}
