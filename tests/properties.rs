//! Self-consistency and robustness gates that do not depend on the oracle:
//! exact orthogonality identities, the zero convention, and a no-panic
//! property over arbitrary inputs. Regge orbit invariance is a unit test of the
//! crate-private key in `src/lie/su2.rs`.

use malachite_base::num::{
    arithmetic::traits::{Abs, FloorSqrt},
    basic::traits::{One, Zero},
};
use malachite_nz::integer::Integer;
use malachite_q::Rational;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use fusion_symbols::lie::su2::{clebsch_gordan, wigner_3j, wigner_6j, SignedSqrtRational};

fn tri(a: u32, b: u32, c: u32) -> bool {
    let (a, b, c) = (a as i64, b as i64, c as i64);
    (a + b + c) % 2 == 0 && c >= (a - b).abs() && c <= a + b
}

/// Exact rational square root of a nonnegative rational that is a perfect
/// square; `None` otherwise. Used to divide out the common irrational factor.
fn exact_rational_sqrt(r: &Rational) -> Option<Rational> {
    if r == &0 {
        return Some(Rational::ZERO);
    }
    let n = r.numerator_ref();
    let d = r.denominator_ref();
    let ns = n.floor_sqrt();
    let ds = d.floor_sqrt();
    if &(&ns * &ns) == n && &(&ds * &ds) == d {
        Some(Rational::from_integers(
            Integer::from(ns),
            Integer::from(ds),
        ))
    } else {
        None
    }
}

/// Sum the values whose signed squares are `terms` (`sign * radicand`). They
/// share a common irrational factor `sqrt(K)` (every pairwise radicand ratio is
/// a perfect square); return the rational coefficient `R` such that the sum
/// equals `R * sqrt(rho0)`, where `rho0` is the first nonzero term's radicand.
/// Also returns `rho0`.
fn factor_common_sqrt(terms: &[Rational]) -> (Rational, Rational) {
    let mut rho0: Option<Rational> = None;
    let mut acc = Rational::ZERO;
    for t in terms {
        if t == &0 {
            continue;
        }
        let sign = Rational::from(if t < &0 { -1 } else { 1 });
        let rho = t.abs();
        match &rho0 {
            None => {
                rho0 = Some(rho);
                acc += sign;
            }
            Some(r0) => {
                // g = sqrt(rho / rho0), guaranteed rational for a shared factor.
                let g = exact_rational_sqrt(&(rho / r0)).expect("shared sqrt factor");
                acc += sign * g;
            }
        }
    }
    (acc, rho0.unwrap_or(Rational::ZERO))
}

/// Signed square of the product `a * b * k`: `ssq(a) * ssq(b) * k * |k|`.
fn product_signed_square(a: &SignedSqrtRational, b: &SignedSqrtRational, k: i64) -> Rational {
    a.signed_square()
        * b.signed_square()
        * Rational::from(Integer::from(k) * Integer::from(k.abs()))
}

#[test]
fn six_j_orthogonality_exact() {
    // sum_x (2x+1)(2j3+1) {j1 j2 x; j4 j5 j3}{j1 j2 x; j4 j5 j6} = delta(j3,j6).
    // All terms share the same irrational factor sqrt(K) (independent of x), so
    // the sum is (rational) * sqrt(K); we assert the exact rational identity.
    let mut rng = ChaCha8Rng::seed_from_u64(0x0447_0007);
    let mut cases = 0;
    let mut attempts = 0;
    while cases < 40 && attempts < 200_000 {
        attempts += 1;
        let dj1 = rng.gen_range(0..=12);
        let dj2 = rng.gen_range(0..=12);
        let dj4 = rng.gen_range(0..=12);
        let dj5 = rng.gen_range(0..=12);
        let dj3 = rng.gen_range(0..=12);
        let dj6 = rng.gen_range(0..=12);
        // j3 must be admissible with (j4,j5) and (j1,j2)-reachable via some x;
        // require both dj3 and dj6 valid outer couplings.
        if !tri(dj1, dj5, dj6) || !tri(dj4, dj2, dj6) {
            continue;
        }
        if !tri(dj1, dj5, dj3) || !tri(dj4, dj2, dj3) {
            continue;
        }

        // x ranges over admissible couplings of (j1,j2) and (j4,j5).
        let xmin = (dj1 as i64 - dj2 as i64)
            .abs()
            .max((dj4 as i64 - dj5 as i64).abs()) as u32;
        let xmax = (dj1 + dj2).min(dj4 + dj5);
        let mut terms = Vec::new();
        let mut x = xmin;
        while x <= xmax {
            let a = wigner_6j(dj1, dj2, x, dj4, dj5, dj3);
            let b = wigner_6j(dj1, dj2, x, dj4, dj5, dj6);
            terms.push(product_signed_square(
                &a,
                &b,
                (x as i64 + 1) * (dj3 as i64 + 1),
            ));
            x += 2;
        }
        if terms.iter().all(|x| x == &0) {
            continue;
        }
        let (r, rho0) = factor_common_sqrt(&terms);
        if dj3 == dj6 {
            // sqrt(rho0) must be rational and R*sqrt(rho0) == 1.
            let s = exact_rational_sqrt(&rho0).expect("diagonal radicand is a square");
            assert_eq!(r * s, Rational::ONE, "6j orthogonality diagonal != 1");
        } else {
            assert!((r == 0), "6j orthogonality off-diagonal != 0");
        }
        cases += 1;
    }
    assert!(cases >= 10, "only {cases} orthogonality cases");
}

#[test]
fn cg_orthogonality_sum_of_squares_is_one() {
    // For fixed admissible (j1,j2,j3,m3): sum_{m1} CG(j1 m1, j2 (m3-m1) | j3 m3)^2 = 1.
    // Each square is exactly the radicand (a rational), so the sum is exact.
    let mut rng = ChaCha8Rng::seed_from_u64(0xC6_u64 * 99991);
    let mut cases = 0;
    let mut attempts = 0;
    while cases < 60 && attempts < 200_000 {
        attempts += 1;
        let dj1 = rng.gen_range(0..=14);
        let dj2 = rng.gen_range(0..=14);
        let dj3 = rng.gen_range(0..=14);
        if !tri(dj1, dj2, dj3) {
            continue;
        }
        // pick m3 with valid parity/range
        let m3choices: Vec<i32> = (-(dj3 as i32)..=dj3 as i32).step_by(2).collect();
        let dm3 = m3choices[rng.gen_range(0..m3choices.len())];

        let mut sum: Rational = Rational::ZERO;
        let mut dm1 = -(dj1 as i32);
        while dm1 <= dj1 as i32 {
            let dm2 = dm3 - dm1;
            if dm2.unsigned_abs() <= dj2 && (dj2 as i32 + dm2) % 2 == 0 {
                let cg = clebsch_gordan(dj1, dm1, dj2, dm2, dj3, dm3);
                // CG^2 = radicand (nonneg rational).
                sum += cg.radicand();
            }
            dm1 += 2;
        }
        assert_eq!(
            sum,
            Rational::ONE,
            "CG sum of squares != 1 for j=({dj1},{dj2},{dj3}) m3={dm3}"
        );
        cases += 1;
    }
    assert!(cases >= 20, "only {cases} CG cases");
}

#[test]
fn cg_cross_j3_orthogonality_zero() {
    // sum_{m1} CG(j1 m1, j2 (M-m1) | j3 M) CG(j1 m1, j2 (M-m1) | j3' M) = 0 for
    // j3 != j3'. Terms share a common sqrt factor across m1; assert the exact
    // rational coefficient is zero.
    let mut rng = ChaCha8Rng::seed_from_u64(0xC7_u64 * 40009);
    let mut cases = 0;
    let mut attempts = 0;
    while cases < 30 && attempts < 200_000 {
        attempts += 1;
        let dj1 = rng.gen_range(0..=12);
        let dj2 = rng.gen_range(0..=12);
        let dj3 = rng.gen_range(0..=12);
        let dj3p = rng.gen_range(0..=12);
        if dj3 == dj3p || !tri(dj1, dj2, dj3) || !tri(dj1, dj2, dj3p) {
            continue;
        }
        // Total M must be valid for both j3 and j3'.
        let mmax = (dj3.min(dj3p)) as i32;
        let mchoices: Vec<i32> = (-mmax..=mmax).step_by(2).collect();
        if mchoices.is_empty() {
            continue;
        }
        let big_m = mchoices[rng.gen_range(0..mchoices.len())];

        let mut terms = Vec::new();
        let mut dm1 = -(dj1 as i32);
        while dm1 <= dj1 as i32 {
            let dm2 = big_m - dm1;
            if dm2.unsigned_abs() <= dj2 && (dj2 as i32 + dm2) % 2 == 0 {
                let a = clebsch_gordan(dj1, dm1, dj2, dm2, dj3, big_m);
                let b = clebsch_gordan(dj1, dm1, dj2, dm2, dj3p, big_m);
                terms.push(product_signed_square(&a, &b, 1));
            }
            dm1 += 2;
        }
        if terms.iter().all(|x| x == &0) {
            continue;
        }
        let (r, _rho0) = factor_common_sqrt(&terms);
        assert!((r == 0), "CG cross-j3 orthogonality != 0");
        cases += 1;
    }
    assert!(cases >= 10, "only {cases} CG cross cases");
}

#[test]
fn zero_convention() {
    // Each non-admissible class returns exact zero.
    // 6j: a broken triangle.
    assert_eq!(wigner_6j(1, 1, 1, 1, 1, 1), SignedSqrtRational::zero()); // parity
    assert_eq!(wigner_6j(2, 2, 10, 2, 2, 2), SignedSqrtRational::zero()); // range

    // 3j: triangle violation, |m|>j, parity mismatch, m-sum != 0.
    assert_eq!(wigner_3j(2, 2, 10, 0, 0, 0), SignedSqrtRational::zero());
    assert_eq!(wigner_3j(2, 2, 2, 4, -4, 0), SignedSqrtRational::zero()); // |m1|>j1
    assert_eq!(wigner_3j(2, 2, 2, 1, -1, 0), SignedSqrtRational::zero()); // parity: dj+dm odd
    assert_eq!(wigner_3j(2, 2, 2, 2, 2, 0), SignedSqrtRational::zero()); // m-sum != 0

    // CG inherits: total m mismatch and triangle violation.
    assert_eq!(clebsch_gordan(1, 1, 1, 1, 2, 0), SignedSqrtRational::zero()); // m3 != m1+m2
    assert_eq!(
        clebsch_gordan(2, 0, 2, 0, 10, 0),
        SignedSqrtRational::zero()
    ); // triangle
}

#[test]
fn no_panic_arbitrary_inputs() {
    // Arbitrary (bounded) doubled labels, including negative and out-of-range
    // projections, must never panic and must produce a well-formed value.
    let mut rng = ChaCha8Rng::seed_from_u64(0xDEAD_BEEF);
    for _ in 0..20_000 {
        let dj = |r: &mut ChaCha8Rng| r.gen_range(0u32..=40);
        let dm = |r: &mut ChaCha8Rng| r.gen_range(-50i32..=50);
        let (a, b, c, d, e, f) = (
            dj(&mut rng),
            dj(&mut rng),
            dj(&mut rng),
            dj(&mut rng),
            dj(&mut rng),
            dj(&mut rng),
        );
        let _ = wigner_6j(a, b, c, d, e, f).to_f64();
        let (m1, m2, m3) = (dm(&mut rng), dm(&mut rng), dm(&mut rng));
        let _ = wigner_3j(a, b, c, m1, m2, m3).to_f64();
        let _ = clebsch_gordan(a, m1, b, m2, c, m3).to_f64();
    }
}
