//! SU(2) 3j, Clebsch–Gordan and 6j symbols against readable references.
//!
//! Contracts:
//!
//! - textbook values, exact (sign and squared rational), checked against
//!   WignerSymbols.jl 2.0.0;
//! - Clebsch–Gordan coefficients obey the Condon–Shortley convention: the
//!   stretched and singlet closed forms, the lowering recursion (which with
//!   the positive `⟨j1 j1; j2 J−j1|J J⟩` fixes every coefficient), and the
//!   3j relation `(j1 j2 j3; m1 m2 m3) = (−1)^{j1−j2−m3} ⟨j1 m1 j2 m2|j3 −m3⟩ / √(2j3+1)`;
//! - 6j symbols obey the `{a b c; 0 c b}` closed form and Racah's sum rule
//!   (Edmonds 6.2.8), which is sign sensitive on generic arguments.
//!
//! Exact orthogonality and Regge symmetry are in `tests/properties.rs`.
//! Every spin is doubled (`dj = 2j`).

use num_bigint::BigInt;
use num_rational::Ratio;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use fusion_symbols::lie::su2::{clebsch_gordan, wigner_3j, wigner_6j, SignedSqrtRational};

fn tri(a: u32, b: u32, c: u32) -> bool {
    (a + b + c).is_multiple_of(2) && c >= a.abs_diff(b) && c <= a + b
}

/// `sign · num / den` as the signed square of the expected value.
fn signed(sign: i64, num: i64, den: i64) -> Ratio<BigInt> {
    Ratio::new(BigInt::from(sign * num), BigInt::from(den))
}

fn projections(dj: u32) -> impl Iterator<Item = i32> {
    let dj = dj as i32;
    (-dj..=dj).step_by(2)
}

fn binomial(n: u32, k: u32) -> BigInt {
    (0..k).fold(BigInt::from(1), |acc, i| acc * (n - i) / (i + 1))
}

fn parity(half_units: i64) -> i64 {
    if half_units.rem_euclid(4) == 0 {
        1
    } else {
        -1
    }
}

#[test]
fn textbook_values_checked_against_wignersymbols() {
    // Checked against WignerSymbols.jl 2.0.0 (`RationalRoot{BigInt}`).
    let six: [([u32; 6], Ratio<BigInt>); 7] = [
        ([2, 2, 2, 2, 2, 2], signed(1, 1, 36)),
        ([1, 1, 2, 1, 1, 2], signed(1, 1, 36)),
        ([1, 1, 0, 1, 1, 0], signed(-1, 1, 4)),
        ([4, 4, 4, 4, 4, 4], signed(-1, 9, 4900)),
        ([2, 2, 2, 2, 2, 0], signed(-1, 1, 9)),
        ([3, 3, 2, 1, 1, 2], signed(1, 5, 72)),
        ([4, 2, 2, 2, 4, 4], signed(1, 7, 300)),
    ];
    for (d, want) in six {
        let got = wigner_6j(d[0], d[1], d[2], d[3], d[4], d[5]);
        assert_eq!(got.signed_square(), want, "6j {d:?}");
    }
    let three: [([u32; 3], [i32; 3], Ratio<BigInt>); 5] = [
        ([2, 2, 0], [0, 0, 0], signed(-1, 1, 3)),
        ([2, 2, 4], [2, -2, 0], signed(1, 1, 30)),
        ([1, 1, 2], [1, 1, -2], signed(-1, 1, 3)),
        ([2, 2, 2], [2, 0, -2], signed(-1, 1, 6)),
        ([4, 2, 2], [0, 2, -2], signed(1, 1, 30)),
    ];
    for (j, m, want) in three {
        let got = wigner_3j(j[0], j[1], j[2], m[0], m[1], m[2]);
        assert_eq!(got.signed_square(), want, "3j {j:?} {m:?}");
    }
    // `(dj1, dm1, dj2, dm2, dJ, dM)`.
    let cg: [([u32; 3], [i32; 3], Ratio<BigInt>); 7] = [
        ([1, 1, 2], [1, -1, 0], signed(1, 1, 2)),
        ([1, 1, 0], [1, -1, 0], signed(1, 1, 2)),
        ([2, 1, 1], [2, -1, 1], signed(1, 2, 3)),
        ([2, 1, 1], [0, 1, 1], signed(-1, 1, 3)),
        ([2, 2, 4], [0, 0, 0], signed(1, 2, 3)),
        ([2, 2, 2], [0, 0, 0], signed(0, 0, 1)),
        ([4, 2, 2], [2, -2, 0], signed(1, 3, 10)),
    ];
    for (j, m, want) in cg {
        let got = clebsch_gordan(j[0], m[0], j[1], m[1], j[2], m[2]);
        assert_eq!(got.signed_square(), want, "CG {j:?} {m:?}");
    }
}

/// Every admissible `(dj1, dm1, dj2, dm2, dJ, dM)` with `dj1, dj2 ≤ max`.
fn cg_labels(max: u32) -> Vec<[i64; 6]> {
    let mut out = Vec::new();
    for dj1 in 0..=max {
        for dj2 in 0..=max {
            for dj in (dj1.abs_diff(dj2)..=dj1 + dj2).step_by(2) {
                for dm1 in projections(dj1) {
                    for dm2 in projections(dj2) {
                        if (dm1 + dm2).unsigned_abs() <= dj {
                            out.push([
                                i64::from(dj1),
                                i64::from(dm1),
                                i64::from(dj2),
                                i64::from(dm2),
                                i64::from(dj),
                                i64::from(dm1 + dm2),
                            ]);
                        }
                    }
                }
            }
        }
    }
    out
}

fn cg(l: [i64; 6]) -> SignedSqrtRational {
    clebsch_gordan(
        l[0] as u32,
        l[1] as i32,
        l[2] as u32,
        l[3] as i32,
        l[4] as u32,
        l[5] as i32,
    )
}

#[test]
fn clebsch_gordan_stretched_and_singlet_closed_forms() {
    // `⟨j1 m1 j2 m2|j1+j2 M⟩² = C(2j1, j1+m1) C(2j2, j2+m2) / C(2J, J+M)`,
    // positive; `⟨j m j −m|0 0⟩ = (−1)^{j−m} / √(2j+1)`.
    let k = |dj: i64, dm: i64| ((dj + dm) / 2) as u32;
    let mut checked = 0;
    // Every doubled spin `≤ 30`, then a few in the hundreds.
    let pairs = (0..=30i64)
        .flat_map(|a| (0..=30i64).map(move |b| (a, b)))
        .chain([(101, 140), (200, 199), (254, 3)]);
    for (dj1, dj2) in pairs {
        let dj = dj1 + dj2;
        // Every projection while `dj1 + dj2 ≤ 60`, a sparse grid above.
        let step = if dj <= 60 { 2 } else { 26 };
        for dm1 in (-dj1..=dj1).step_by(step) {
            for dm2 in (-dj2..=dj2).step_by(step) {
                let l = [dj1, dm1, dj2, dm2, dj, dm1 + dm2];
                let want = Ratio::new(
                    binomial(dj1 as u32, k(dj1, dm1)) * binomial(dj2 as u32, k(dj2, dm2)),
                    binomial(dj as u32, k(dj, dm1 + dm2)),
                );
                assert_eq!(cg(l).signed_square(), want, "stretched {l:?}");
                checked += 1;
            }
        }
        if dj1 == dj2 {
            for dm1 in (-dj1..=dj1).step_by(2) {
                let l = [dj1, dm1, dj1, -dm1, 0, 0];
                let want = signed(parity(dj1 - dm1), 1, dj1 + 1);
                assert_eq!(cg(l).signed_square(), want, "singlet {l:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 10_000, "{checked}");
}

#[test]
fn clebsch_gordan_obey_the_lowering_recursion_and_condon_shortley_sign() {
    // `√((J+M)(J−M+1)) ⟨m1 m2|J M−1⟩ = √((j1−m1)(j1+m1+1)) ⟨m1+1 m2|J M⟩
    //   + √((j2−m2)(j2+m2+1)) ⟨m1 m2+1|J M⟩`, doubled: `(J+M)(J−M+1)` is
    // `(dJ+dM)(dJ−dM+2)/4`.
    let c = |l: [i64; 6]| {
        let [dj1, dm1, dj2, dm2, dj, dm] = l;
        if dm1.abs() > dj1 || dm2.abs() > dj2 || dm.abs() > dj {
            0.0
        } else {
            cg(l).to_f64()
        }
    };
    let ladder = |dj: i64, dm: i64| (((dj - dm) * (dj + dm + 2)) as f64 / 4.0).sqrt();
    let mut checked = 0;
    for l @ [dj1, dm1, dj2, dm2, dj, dm] in cg_labels(10) {
        if dj1 + dj2 >= dj && dm == dj && dm1 == dj1 {
            assert!(cg(l).sign() > 0, "Condon–Shortley {l:?}");
        }
        // `l` holds `M − 1 = dm − 2` in doubled units.
        let up = dm + 2;
        if up > dj {
            continue;
        }
        let lhs = ladder(dj, dm) * c(l);
        let rhs = ladder(dj1, dm1) * c([dj1, dm1 + 2, dj2, dm2, dj, up])
            + ladder(dj2, dm2) * c([dj1, dm1, dj2, dm2 + 2, dj, up]);
        assert!((lhs - rhs).abs() < 1e-13, "{l:?}: {lhs} vs {rhs}");
        checked += 1;
    }
    assert!(checked > 10_000, "{checked}");
}

#[test]
fn three_j_is_the_rescaled_clebsch_gordan() {
    for [dj1, dm1, dj2, dm2, dj, dm] in cg_labels(8) {
        let three = wigner_3j(
            dj1 as u32, dj2 as u32, dj as u32, dm1 as i32, dm2 as i32, -dm as i32,
        );
        let want = cg([dj1, dm1, dj2, dm2, dj, dm]).signed_square()
            * signed(parity(dj1 - dj2 + dm), 1, dj + 1);
        assert_eq!(
            three.signed_square(),
            want,
            "{:?}",
            [dj1, dm1, dj2, dm2, dj]
        );
    }
}

#[test]
fn six_j_with_a_zero_entry_is_the_closed_form() {
    // `{a b c; 0 c b} = (−1)^{a+b+c} / √((2b+1)(2c+1))`, up to doubled spins
    // in the thousands.
    let mut cases = vec![];
    for a in 0..=12 {
        for b in 0..=12 {
            for c in 0..=12 {
                cases.push([a, b, c]);
            }
        }
    }
    cases.extend([
        [2000, 1001, 1001],
        [1, 3001, 3000],
        [4000, 2500, 1500],
        [999, 2000, 2001],
    ]);
    for [a, b, c] in cases {
        let got = wigner_6j(a, b, c, 0, c, b);
        let want = if tri(a, b, c) {
            let s = parity(i64::from(a + b + c));
            signed(s, 1, i64::from((b + 1) * (c + 1)))
        } else {
            signed(0, 0, 1)
        };
        assert_eq!(got.signed_square(), want, "{:?}", [a, b, c]);
    }
}

/// Racah's sum rule (Edmonds 6.2.8):
/// `Σ_x (−1)^{p+q+x} (2x+1) {a b x; c d p}{a b x; d c q} = {a c q; b d p}`.
fn racah_sum_rule(l: [u32; 6]) -> Option<(f64, f64)> {
    let [a, b, c, d, p, q] = l;
    if !(tri(a, d, p) && tri(c, b, p) && tri(a, c, q) && tri(d, b, q)) {
        return None;
    }
    let lo = a.abs_diff(b).max(c.abs_diff(d));
    let hi = (a + b).min(c + d);
    let sum: f64 = (lo..=hi)
        .step_by(2)
        .map(|x| {
            let s = parity(i64::from(p + q + x)) as f64;
            s * f64::from(x + 1)
                * wigner_6j(a, b, x, c, d, p).to_f64()
                * wigner_6j(a, b, x, d, c, q).to_f64()
        })
        .sum();
    Some((sum, wigner_6j(a, c, q, b, d, p).to_f64()))
}

#[test]
fn six_j_obey_racah_sum_rule() {
    let mut checked = 0;
    for a in 0..=6 {
        for b in 0..=6 {
            for c in 0..=6 {
                for d in 0..=6 {
                    for p in 0..=6 {
                        for q in 0..=6 {
                            if let Some((sum, want)) = racah_sum_rule([a, b, c, d, p, q]) {
                                assert!((sum - want).abs() < 1e-13, "{:?}", [a, b, c, d, p, q]);
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    // Every admissible tuple of doubled spins ≤ 6.
    assert_eq!(checked, 3418);
    // Random generic arguments on the doubled-spin ≤ 254 range.
    let mut rng = ChaCha8Rng::seed_from_u64(0xA55A_1234);
    let mut samples = 0;
    while samples < 40 {
        let l = [(); 6].map(|_| rng.gen_range(0..=254u32));
        if let Some((sum, want)) = racah_sum_rule(l) {
            assert!((sum - want).abs() < 1e-12, "{l:?}: {sum} vs {want}");
            samples += 1;
        }
    }
}
