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
#![cfg(feature = "lie-generated")]
// Index loops mirror the tensor-index formulas above.
#![allow(clippy::needless_range_loop)]

use fusion_symbols::category::CgcBlock;

use crate::common::lie as lie_test;
use fusion_symbols::lie::LieIrrep as Irrep;
use lie_test::cgc;

fn irr(d: &[i64]) -> Irrep {
    lie_test::irrep(d).unwrap()
}

/// Dense `C[m1][m2][m3][mu]`.
fn dense(a: &Irrep, b: &Irrep, c: &Irrep) -> (Vec<f64>, [usize; 4]) {
    let t = cgc(a, b, c).unwrap();
    let dims = t.dims();
    let mut out = vec![0.0; dims.iter().product()];
    for e in lie_test::entries(&t) {
        let [m1, m2, m3, mu] = [e.m1, e.m2, e.m3, e.mu].map(|x| x as usize);
        out[((m1 * dims[1] + m2) * dims[2] + m3) * dims[3] + mu] = e.value;
    }
    (out, dims)
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
fn cgc_columns_are_isometric() {
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
    }
}

#[test]
fn cgc_values_checked_against_sunrepresentations() {
    // Checked against SUNRepresentations.jl 0.4.0 (`CGC(Float64, s1, s2, s3)`):
    // `(m1, m2, m3, mu)` 0-based in the GT basis order.
    let s = std::f64::consts::FRAC_1_SQRT_2;
    type Entries<'a> = &'a [([usize; 4], f64)];
    let rows: [(usize, Entries); 6] = [
        (
            0,
            &[
                ([0, 0, 0, 0], 1.0),
                ([1, 0, 1, 0], s),
                ([0, 1, 1, 0], s),
                ([1, 1, 2, 0], 1.0),
            ],
        ),
        (
            1,
            &[
                ([0, 0, 0, 0], 1.0),
                ([1, 0, 1, 0], s),
                ([2, 0, 2, 0], s),
                ([1, 1, 3, 0], 1.0),
            ],
        ),
        (
            2,
            &[
                ([0, 0, 0, 0], 1.0),
                ([2, 0, 2, 0], -0.408_248_290_463_863_1),
                ([1, 1, 2, 0], 0.408_248_290_463_863_1),
                ([0, 2, 2, 0], 0.816_496_580_927_726_1),
                ([1, 1, 4, 0], s),
            ],
        ),
        (
            3,
            &[
                // Both multiplicity columns at the same `(m1, m2, m3)`: a
                // swapped or re-signed `mu` axis moves these.
                ([0, 2, 0, 0], 0.534_522_483_824_848),
                ([0, 2, 0, 1], -0.119_522_860_933_438_75),
                ([0, 4, 0, 1], 0.483_045_891_539_648_1),
                ([0, 5, 1, 1], 0.683_130_051_063_973_3),
                ([0, 6, 3, 0], 0.654_653_670_707_976_9),
                ([0, 7, 2, 0], -0.267_261_241_912_424_4),
            ],
        ),
        (
            4,
            &[
                ([0, 1, 1, 0], 1.0),
                ([3, 0, 3, 0], 0.288_675_134_594_812_9),
                ([2, 1, 3, 0], -0.288_675_134_594_812_9),
                ([1, 2, 3, 0], 0.288_675_134_594_812_9),
            ],
        ),
        (
            5,
            &[
                ([0, 0, 0, 0], 1.0),
                ([1, 0, 1, 0], s),
                ([0, 2, 2, 0], s),
                ([1, 1, 3, 0], 1.0),
            ],
        ),
    ];
    for (channel, entries) in rows {
        let (da, db, dc) = CHANNELS[channel];
        let (t, [_, d2, d3, om]) = dense(&irr(da), &irr(db), &irr(dc));
        for &([m1, m2, m3, mu], want) in entries {
            let got = t[((m1 * d2 + m2) * d3 + m3) * om + mu];
            assert!(
                (got - want).abs() < 1e-12,
                "{da:?}⊗{db:?}→{dc:?} {:?}: {got}",
                [m1, m2, m3, mu]
            );
        }
    }
}
