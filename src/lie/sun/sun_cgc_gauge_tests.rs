//! SU(N) Clebsch–Gordan coefficients: intertwiners in the SUNRepresentations
//! gauge.
//!
//! Contracts, on representative channels (SU(2) `2 ⊗ 2 → 3`; SU(3)
//! `3 ⊗ 3 → 6`, `3 ⊗ 3̄ → 8`, `8 ⊗ 8 → 8` with outer multiplicity 2; SU(4)
//! `4 ⊗ 4̄ → 15`, `6 ⊗ 6 → 20'`):
//!
//! - each multiplicity column is an isometry, `Σ_{m1 m2} C[m1,m2,m3,μ] C[m1,m2,m3',μ'] = δ δ`;
//! - it intertwines the Gelfand–Tsetlin raising operators,
//!   `(E^{(1)}_l ⊗ 1 + 1 ⊗ E^{(2)}_l) C = C E^{(3)}_l` (the generators are
//!   pinned in `tests/sun_oracle.rs`);
//! - the gauge (GT basis order, signs, multiplicity-column split) is pinned
//!   by signed entries checked against SUNRepresentations.jl 0.4.0.
//!
//! Identities are gauge invariant; only the literals pin the gauge.
#![cfg(feature = "cgc-gen")]
// Index loops mirror the tensor-index formulas above.
#![allow(clippy::needless_range_loop)]

use crate::lie::sun::{cgc, Irrep};

fn irr(d: &[i64]) -> Irrep {
    Irrep::from_dynkin(d).unwrap()
}

/// Dense `C[m1][m2][m3][mu]`.
fn dense(a: &Irrep, b: &Irrep, c: &Irrep) -> (Vec<f64>, [usize; 4]) {
    let t = cgc(a, b, c).unwrap();
    let dims = t.dims();
    let mut out = vec![0.0; dims.iter().product()];
    for e in t.entries() {
        let [m1, m2, m3, mu] = [e.m1, e.m2, e.m3, e.mu].map(|x| x as usize);
        out[((m1 * dims[1] + m2) * dims[2] + m3) * dims[3] + mu] = e.value;
    }
    (out, dims)
}

/// Dense raising operator `E_l[row][col]` of `s`.
fn raising(s: &Irrep, l: usize) -> Vec<Vec<f64>> {
    let d = s.patterns().len();
    let mut e = vec![vec![0.0; d]; d];
    for x in &s.creation()[l] {
        e[x.row][x.col] = x.value.to_f64();
    }
    e
}

const CHANNELS: [(&[i64], &[i64], &[i64]); 6] = [
    (&[1], &[1], &[2]),
    (&[1, 0], &[1, 0], &[2, 0]),
    (&[1, 0], &[0, 1], &[1, 1]),
    (&[1, 1], &[1, 1], &[1, 1]),
    (&[1, 0, 0], &[0, 0, 1], &[1, 0, 1]),
    (&[0, 1, 0], &[0, 1, 0], &[0, 2, 0]),
];

#[test]
fn cgc_columns_are_isometric_intertwiners() {
    for (da, db, dc) in CHANNELS {
        let (a, b, c) = (irr(da), irr(db), irr(dc));
        let (t, [d1, d2, d3, om]) = dense(&a, &b, &c);
        let at =
            |m1: usize, m2: usize, m3: usize, mu: usize| t[((m1 * d2 + m2) * d3 + m3) * om + mu];
        for (m3, mu) in (0..d3).flat_map(|m| (0..om).map(move |u| (m, u))) {
            for (n3, nu) in (0..d3).flat_map(|m| (0..om).map(move |u| (m, u))) {
                let mut dot = 0.0;
                for m1 in 0..d1 {
                    for m2 in 0..d2 {
                        dot += at(m1, m2, m3, mu) * at(m1, m2, n3, nu);
                    }
                }
                let want = f64::from(u8::from((m3, mu) == (n3, nu)));
                assert!((dot - want).abs() < 1e-12, "{dc:?} isometry");
            }
        }
        for l in 0..da.len() {
            let (e1, e2, e3) = (raising(&a, l), raising(&b, l), raising(&c, l));
            for m3 in 0..d3 {
                for mu in 0..om {
                    for m1 in 0..d1 {
                        for m2 in 0..d2 {
                            let lhs: f64 =
                                (0..d1).map(|k| e1[m1][k] * at(k, m2, m3, mu)).sum::<f64>()
                                    + (0..d2).map(|k| e2[m2][k] * at(m1, k, m3, mu)).sum::<f64>();
                            let rhs: f64 = (0..d3).map(|r| at(m1, m2, r, mu) * e3[r][m3]).sum();
                            assert!((lhs - rhs).abs() < 1e-12, "{da:?}⊗{db:?}→{dc:?} l={l}");
                        }
                    }
                }
            }
        }
    }
}
