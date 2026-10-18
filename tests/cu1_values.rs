//! The `cu1` module reproduces TensorKitSectors' `CU1Irrep` (`O(2)`).
//!
//! Contract: F and R are the recoupling and braiding coefficients of the
//! TensorKitSectors `CU1Irrep` fusion tensors, so every value follows from a
//! dense contraction of the 2×2 (or 1×1) Clebsch–Gordan tensors of
//! `cu1irrep.jl:fusiontensor`, written out below. The unary data and fusion
//! order are TensorKitSectors' closed forms. A few literals pin the sign
//! convention against the reference itself.

use fusion_symbols::cu1::{cu1_f_symbol, cu1_nsymbol, cu1_r_symbol, CU1Irrep};

const OP: CU1Irrep = CU1Irrep::VACUUM;
const OM: CU1Irrep = CU1Irrep::PSEUDOSCALAR;

fn j(twice_charge: u32) -> CU1Irrep {
    CU1Irrep::from_twice_charge(twice_charge)
}

/// `0+`, `0-`, `2j = 1..=5`.
fn labels() -> Vec<CU1Irrep> {
    [OP, OM].into_iter().chain((1..=5).map(j)).collect()
}

fn dim(a: CU1Irrep) -> usize {
    if a.twice_charge() == 0 {
        1
    } else {
        2
    }
}

/// `cu1irrep.jl:fusiontensor(a, b, c)` as `C[i][k][m]` (basis `+j, -j` of a
/// charged irrep), all zero when `c ∉ a ⊗ b`.
fn fusion_tensor(a: CU1Irrep, b: CU1Irrep, c: CU1Irrep) -> [[[f64; 2]; 2]; 2] {
    let mut t = [[[0.0; 2]; 2]; 2];
    let (qa, qb, qc) = (
        i64::from(a.twice_charge()),
        i64::from(b.twice_charge()),
        i64::from(c.twice_charge()),
    );
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let sign = |x: CU1Irrep| if x.s() == 1 { -1.0 } else { 1.0 };
    if qc == 0 {
        if qa == 0 && qb == 0 {
            if a.s() ^ b.s() == c.s() {
                t[0][0][0] = 1.0;
            }
        } else if qa == qb {
            t[0][1][0] = r;
            t[1][0][0] = r * sign(c);
        }
    } else if qa == 0 {
        if qb == qc {
            t[0][0][0] = 1.0;
            t[0][1][1] = sign(a);
        }
    } else if qb == 0 {
        if qa == qc {
            t[0][0][0] = 1.0;
            t[1][0][1] = sign(b);
        }
    } else if qc == qa + qb {
        t[0][0][0] = 1.0;
        t[1][1][1] = 1.0;
    } else if qc == qa - qb {
        t[0][1][0] = 1.0;
        t[1][0][1] = 1.0;
    } else if qc == qb - qa {
        t[1][0][0] = 1.0;
        t[0][1][1] = 1.0;
    }
    t
}

/// `F^{abc}_d[e, f] = (1/d_d) Σ C^{ab}_e[i,k,μ] C^{ec}_d[μ,l,m] C^{bc}_f[k,l,ν] C^{af}_d[i,ν,m]`
/// (the real fusion-tensor identity TensorKitSectors' sector tests assert).
fn dense_f(l: [CU1Irrep; 6]) -> f64 {
    let [a, b, c, d, e, f] = l;
    let (x1, x2, y1, y2) = (
        fusion_tensor(a, b, e),
        fusion_tensor(e, c, d),
        fusion_tensor(b, c, f),
        fusion_tensor(a, f, d),
    );
    let mut sum = 0.0;
    for i in 0..dim(a) {
        for k in 0..dim(b) {
            for l in 0..dim(c) {
                for m in 0..dim(d) {
                    for mu in 0..dim(e) {
                        for nu in 0..dim(f) {
                            sum += x1[i][k][mu] * x2[mu][l][m] * y1[k][l][nu] * y2[i][nu][m];
                        }
                    }
                }
            }
        }
    }
    sum / dim(d) as f64
}

/// `C^{ba}_c[k,i,m] = R^{ab}_c C^{ab}_c[i,k,m]`: the ratio on any nonzero entry.
fn dense_r(a: CU1Irrep, b: CU1Irrep, c: CU1Irrep) -> f64 {
    let (ab, ba) = (fusion_tensor(a, b, c), fusion_tensor(b, a, c));
    for i in 0..2 {
        for k in 0..2 {
            for m in 0..2 {
                if ab[i][k][m] != 0.0 {
                    return ba[k][i][m] / ab[i][k][m];
                }
            }
        }
    }
    0.0
}

#[test]
fn f_and_r_are_the_fusion_tensor_recouplings() {
    let all = labels();
    let mut nonzero = (0, 0);
    for &a in &all {
        for &b in &all {
            for &c in &all {
                let r = cu1_r_symbol(a, b, c);
                assert_eq!(r, dense_r(a, b, c), "R {a:?} {b:?} {c:?}");
                nonzero.0 += usize::from(r != 0.0);
                for &d in &all {
                    for &e in &all {
                        for &f in &all {
                            let l = [a, b, c, d, e, f];
                            let got = cu1_f_symbol(a, b, c, d, e, f);
                            assert!((got - dense_f(l)).abs() < 1e-15, "F {l:?}");
                            nonzero.1 += usize::from(got != 0.0);
                        }
                    }
                }
            }
        }
    }
    // Nonzero counts over the seven labels (TensorKitSectors 0.3.9 has the
    // same counts): a vacuous sweep would not reach them.
    assert_eq!(nonzero, (64, 628));
}

#[test]
fn values_checked_against_tensorkitsectors() {
    // Checked against TensorKitSectors 0.3.9 (`CU1Irrep`, Julia 1.11.6).
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let f = [
        ([j(1), j(1), j(1), j(1), OP, OP], 0.5),
        ([j(1), j(1), j(1), j(1), OP, OM], -0.5),
        ([j(1), j(1), j(1), j(1), OM, OM], -0.5),
        ([j(1), j(1), j(1), j(1), j(2), OP], s),
        ([j(1), j(1), j(1), j(1), j(2), j(2)], 0.0),
        ([j(1), j(1), j(3), j(3), OM, j(4)], -s),
        ([j(1), j(3), j(1), j(3), j(2), j(4)], 1.0),
        ([OM, j(1), j(3), j(2), j(1), j(2)], -1.0),
    ];
    for ([a, b, c, d, e, x], want) in f {
        assert_eq!(
            cu1_f_symbol(a, b, c, d, e, x),
            want,
            "{a:?}{b:?}{c:?}{d:?}{e:?}{x:?}"
        );
    }
    assert_eq!(cu1_r_symbol(j(1), j(1), OM), -1.0);
    assert_eq!(cu1_r_symbol(j(1), j(1), OP), 1.0);
    assert_eq!(cu1_r_symbol(j(1), j(1), j(2)), 1.0);
    assert_eq!(cu1_r_symbol(OM, j(1), j(1)), 1.0);
    // Known deviation: TensorKitSectors returns `-0.0` on this inadmissible
    // tuple (`-convert(Float64, N)` with `N = 0`); here every inadmissible
    // symbol is `+0.0`.
    assert_eq!(cu1_r_symbol(j(1), j(2), OM).to_bits(), 0.0f64.to_bits());
}

#[test]
fn unary_data_and_fusion_order_follow_tensorkitsectors() {
    // `dim = 1` for `0±`, `2` when charged; self-dual; FS phase and twist 1.
    for a in labels() {
        assert_eq!(a.dim(), dim(a) as f64);
        assert_eq!(a.dual(), a);
        assert_eq!((a.frobenius_schur(), a.twist()), (1.0, 1.0));
    }
    // `SectorProductIterator{CU1Irrep}` order: `0± ⊗ 0± = 0^{s xor s'}`,
    // `0± ⊗ j = j`, `j ⊗ j = [0+, 0-, 2j]`, `j ⊗ k = [|j-k|, j+k]`.
    let all = labels();
    for &a in &all {
        for &b in &all {
            let (qa, qb) = (a.twice_charge(), b.twice_charge());
            let want = match (qa, qb) {
                (0, 0) => vec![CU1Irrep::new(0, a.s() ^ b.s()).unwrap()],
                (0, _) => vec![b],
                (_, 0) => vec![a],
                _ if qa == qb => vec![OP, OM, j(2 * qa)],
                _ => vec![j(qa.abs_diff(qb)), j(qa + qb)],
            };
            let got: Vec<_> = a.fusion(b).unwrap().collect();
            assert_eq!(got, want, "{a:?} ⊗ {b:?}");
            for &c in &all {
                assert_eq!(cu1_nsymbol(a, b, c), u32::from(want.contains(&c)));
            }
        }
    }
}
