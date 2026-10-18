//! B/C/D generators and CGC as Lie-algebra data, with no reference toolchain.
//!
//! Contracts:
//!
//! - every irrep's generator set is the irrep: with `F_i = E_iᵀ`,
//!   `[E_i, F_j] = 0` for `i ≠ j`, the coroots `h_i ∝ [E_i, F_i]` give
//!   `[h_i, E_j] = A_ij E_j` for the series' Cartan matrix, the Serre relations
//!   `ad(E_i)^{1−A_ij} E_j = 0` hold, and the only state every `E_i`
//!   annihilates has weight `λ` (with `dim` the Weyl dimension, pinned in
//!   `bcd::tests`, that makes it `V(λ)`). This is independent of how the crate
//!   normalizes `E_i` and its Cartan basis;
//! - every outer-multiplicity copy `C_μ` of `a ⊗ b → c` is an isometry,
//!   orthogonal to the other copies, that intertwines the simple raising and
//!   lowering operators: `(E^a_i ⊗ 1 + 1 ⊗ E^b_i) C_μ = C_μ E^c_i` and the
//!   same for `F_i`. Simple generators generate the algebra, so
//!   `P_c = Σ_μ C_μ C_μᵀ` is the isotypic projector of `c`, and `Σ_c P_c = 1`.
//!
//! All of this is gauge invariant. The signed coupled-side gauge is the frozen
//! spec of `docs/gauge_soN.md`, pinned by `tests/gauge_golden.rs`.

use super::linalg::Dense;
use super::{directproduct, CanonicalCatalog, Irrep, Series};

const TOL: f64 = 1e-10;

/// `2⟨α_i, α_j⟩/⟨α_i, α_i⟩`, simple roots in the crate's node order:
/// `ε_k − ε_{k+1}`, then `ε_r` (B), `2ε_r` (C), `ε_{r−1} + ε_r` (D).
fn cartan(series: Series, r: usize) -> Vec<Vec<i64>> {
    let root = |i: usize| -> Vec<i64> {
        let mut v = vec![0; r];
        if i + 1 < r {
            v[i] = 1;
            v[i + 1] = -1;
        } else {
            match series {
                Series::B => v[r - 1] = 1,
                Series::C => v[r - 1] = 2,
                Series::D => {
                    v[r - 2] = 1;
                    v[r - 1] = 1;
                }
            }
        }
        v
    };
    let dot = |a: &[i64], b: &[i64]| -> i64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    (0..r)
        .map(|i| {
            (0..r)
                .map(|j| 2 * dot(&root(i), &root(j)) / dot(&root(i), &root(i)))
                .collect()
        })
        .collect()
}

type M = Vec<Vec<f64>>;

fn dense(d: &Dense, n: usize) -> M {
    (0..n)
        .map(|r| (0..n).map(|c| d.at(r, c)).collect())
        .collect()
}

fn mul(a: &M, b: &M) -> M {
    let (n, k, m) = (a.len(), b.len(), b[0].len());
    (0..n)
        .map(|i| {
            (0..m)
                .map(|j| (0..k).map(|t| a[i][t] * b[t][j]).sum())
                .collect()
        })
        .collect()
}

fn sub(a: &M, b: &M) -> M {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p - q).collect())
        .collect()
}

fn transpose(a: &M) -> M {
    (0..a[0].len())
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect()
}

fn comm(a: &M, b: &M) -> M {
    sub(&mul(a, b), &mul(b, a))
}

fn max_abs(a: &M) -> f64 {
    a.iter().flatten().fold(0.0, |m, x| m.max(x.abs()))
}

fn diag(v: &[f64]) -> M {
    (0..v.len())
        .map(|i| {
            (0..v.len())
                .map(|j| if i == j { v[i] } else { 0.0 })
                .collect()
        })
        .collect()
}

fn scale(a: &M, s: f64) -> M {
    a.iter()
        .map(|r| r.iter().map(|x| x * s).collect())
        .collect()
}

/// `(E_i, H_i)` of `irrep`, dense.
fn generators(cat: &mut CanonicalCatalog, irrep: &Irrep) -> (Vec<M>, Vec<M>) {
    let g = cat.generators(irrep).unwrap();
    let n = g.dim();
    let e = (0..g.rank()).map(|i| dense(g.raising(i), n)).collect();
    let h = (0..g.rank()).map(|i| diag(g.cartan_diag(i))).collect();
    (e, h)
}

/// `c` with `[x, e] = c e`, asserting `e` is an eigenvector of `ad x`.
fn eigen(x: &M, e: &M) -> f64 {
    let c = comm(x, e);
    let (mut num, mut den) = (0.0, 0.0);
    for (cr, er) in c.iter().zip(e) {
        for (p, q) in cr.iter().zip(er) {
            num += p * q;
            den += q * q;
        }
    }
    let k = num / den;
    assert!(
        max_abs(&sub(&c, &scale(e, k))) < TOL,
        "not an ad-eigenvector"
    );
    k
}

/// Generator `k` carries simple root `α_{node(k)}` (0-based): QSpace's node
/// order, `α_{r−1−k}` for `k < r−1` and `α_r` last in 1-based terms
/// (`docs/gauge_soN.md` §16.2).
fn node(k: usize, r: usize) -> usize {
    if k + 1 < r {
        r - 2 - k
    } else {
        k
    }
}

fn check_irrep(cat: &mut CanonicalCatalog, series: Series, irrep: &Irrep) {
    let r = irrep.rank();
    let cm = cartan(series, r);
    let a: Vec<Vec<i64>> = (0..r)
        .map(|i| (0..r).map(|j| cm[node(i, r)][node(j, r)]).collect())
        .collect();
    let (e, h) = generators(cat, irrep);
    let n = e[0].len();
    assert_eq!(irrep.dim(), n.into(), "{irrep:?}");
    if n == 1 {
        assert!(e.iter().chain(&h).all(|x| max_abs(x) < TOL), "trivial");
        return;
    }
    // Every `E_j` is a weight vector of the Cartan generators.
    for hk in &h {
        for ej in &e {
            eigen(hk, ej);
        }
    }
    // Coroots `h_i = 2 [E_i, F_i] / α_i([E_i, F_i])`, whatever the
    // normalization of `E_i` and of the Cartan basis.
    let coroots: Vec<M> = (0..r)
        .map(|i| {
            let d = comm(&e[i], &transpose(&e[i]));
            scale(&d, 2.0 / eigen(&d, &e[i]))
        })
        .collect();
    for i in 0..r {
        for j in 0..r {
            if i != j {
                let mixed = comm(&e[i], &transpose(&e[j]));
                assert!(max_abs(&mixed) < TOL, "[E{i},F{j}] {irrep:?}");
            }
            assert!(
                (eigen(&coroots[i], &e[j]) - a[i][j] as f64).abs() < TOL,
                "A{i}{j} {irrep:?}"
            );
            if i != j {
                let mut ad = e[j].clone();
                for _ in 0..(1 - a[i][j]) {
                    ad = comm(&e[i], &ad);
                }
                assert!(max_abs(&ad) < TOL, "Serre ({i},{j}) {irrep:?}");
            }
        }
    }
    // The unique state every `E_i` annihilates has weight `λ` (Dynkin label).
    let top: Vec<usize> = (0..n)
        .filter(|&s| e.iter().all(|ei| (0..n).all(|row| ei[row][s].abs() < TOL)))
        .collect();
    assert_eq!(top.len(), 1, "highest weight {irrep:?}");
    let label = irrep.dynkin();
    for (i, h) in coroots.iter().enumerate() {
        let want = label[node(i, r)] as f64;
        assert!((h[top[0]][top[0]] - want).abs() < TOL, "λ {irrep:?}");
    }
}

/// `(X ⊗ 1 + 1 ⊗ Y)` on the product basis `m1 + d1·m2`.
fn on_product(x: &M, y: &M) -> M {
    let (d1, d2) = (x.len(), y.len());
    let mut out = vec![vec![0.0; d1 * d2]; d1 * d2];
    for m2 in 0..d2 {
        for m1 in 0..d1 {
            for k in 0..d1 {
                out[k + d1 * m2][m1 + d1 * m2] += x[k][m1];
            }
            for k in 0..d2 {
                out[m1 + d1 * k][m1 + d1 * m2] += y[k][m2];
            }
        }
    }
    out
}

fn check_products(series: Series, rank: usize, factors: &[&[i64]]) -> usize {
    let mut cat = CanonicalCatalog::new(series, rank).unwrap();
    let mut channels = 0;
    let irreps: Vec<Irrep> = factors
        .iter()
        .map(|d| Irrep::from_dynkin(series, d).unwrap())
        .collect();
    for x in &irreps {
        check_irrep(&mut cat, series, x);
    }
    for (i, a) in irreps.iter().enumerate() {
        for b in &irreps[i..] {
            let (ea, _) = generators(&mut cat, a);
            let (eb, _) = generators(&mut cat, b);
            let rows = ea[0].len() * eb[0].len();
            let mut total = vec![vec![0.0; rows]; rows];
            for (c, mult) in directproduct(a, b).unwrap() {
                check_irrep(&mut cat, series, &c);
                let (ec, _) = generators(&mut cat, &c);
                let t = cat.cgc(a, b, &c).unwrap();
                assert_eq!(t.multiplicity(), mult as usize, "{c:?}");
                let (_, d3) = t.copy_shape();
                let copies: Vec<M> = (0..t.multiplicity())
                    .map(|mu| {
                        let col = t.copy(mu);
                        (0..rows)
                            .map(|row| (0..d3).map(|k| col[k * rows + row]).collect())
                            .collect()
                    })
                    .collect();
                for (mu, cm) in copies.iter().enumerate() {
                    for (nu, cn) in copies.iter().enumerate() {
                        let g = mul(&transpose(cm), cn);
                        let want = if mu == nu {
                            diag(&vec![1.0; d3])
                        } else {
                            vec![vec![0.0; d3]; d3]
                        };
                        assert!(max_abs(&sub(&g, &want)) < TOL, "isometry {a:?}⊗{b:?}→{c:?}");
                    }
                    for l in 0..rank {
                        for (x, y, z) in [
                            (&ea[l], &eb[l], &ec[l]),
                            (&transpose(&ea[l]), &transpose(&eb[l]), &transpose(&ec[l])),
                        ] {
                            let lhs = mul(&on_product(x, y), cm);
                            let rhs = mul(cm, z);
                            assert!(
                                max_abs(&sub(&lhs, &rhs)) < TOL,
                                "intertwiner {a:?}⊗{b:?}→{c:?} l={l}"
                            );
                        }
                    }
                    let p = mul(cm, &transpose(cm));
                    for (row, prow) in total.iter_mut().zip(&p) {
                        for (x, y) in row.iter_mut().zip(prow) {
                            *x += y;
                        }
                    }
                }
                channels += 1;
            }
            assert!(
                max_abs(&sub(&total, &diag(&vec![1.0; rows]))) < TOL,
                "completeness {a:?}⊗{b:?}"
            );
        }
    }
    channels
}

#[test]
fn so5_generators_and_cgc_are_intertwiners() {
    // Vector `[1,0]` (5) and adjoint `[0,2]` (10).
    assert_eq!(check_products(Series::B, 2, &[&[1, 0], &[0, 2]]), 12);
}

#[test]
fn sp4_generators_and_cgc_are_intertwiners() {
    // Defining `[1,0]` (4) and `[0,1]` (5).
    assert_eq!(check_products(Series::C, 2, &[&[1, 0], &[0, 1]]), 8);
}

#[test]
fn so6_generators_and_cgc_are_intertwiners() {
    // Vector `[1,0,0]` (6) and adjoint `[0,1,1]` (15), incl. the 84 and the
    // multiplicity-2 adjoint channel of `15 ⊗ 15`.
    assert_eq!(check_products(Series::D, 3, &[&[1, 0, 0], &[0, 1, 1]]), 13);
}
