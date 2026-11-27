//! The certified fast path on CGC entries, against the exact conversion,
//! on `2 ⊗ 2 → 1` fixtures with two coprime integer coordinates per
//! multiplicity index, so that the stored `X` are the coordinates and the
//! vertex is `1/N_μ` (both asserted before rounding): midpoints and their
//! neighbours, representable values, the range gates, and the stack limit
//! of the exact conversion.

use super::*;
use crate::exact::radical::words_natural;
use crate::lie::general::cgc::exact::Column;
use malachite_base::num::basic::traits::One;
use malachite_base::num::logic::traits::SignificantBits;
use malachite_q::Rational;

fn pow2(k: u64) -> Natural {
    Natural::ONE << k
}

fn nat(x: u64) -> Natural {
    Natural::from(x)
}

/// The norm table of an irrep of dimension `dim` with every norm 1.
fn unit(dim: usize) -> NormTable {
    NormTable::from_parts(vec![Q::from(1); dim].into(), vec![0; dim].into(), 1, 0).unwrap()
}

/// Per entry `(fast bits or None, exact bits)` of `2 ⊗ 2 → 1` with unit
/// norms, the coordinates `x` at `(0, 0)` and `(1, 1)` and vertex `ρ =
/// (num, den)` (reduced), after asserting the stored `X`, their signs and
/// the vertex.
fn entries(x: [(bool, Natural); 2], rho: (Natural, Natural)) -> Vec<(Option<u64>, u64)> {
    let n = Q::from(Rational::from_naturals(rho.1.clone(), rho.0.clone()));
    let coordinate = |(neg, m): &(bool, Natural)| {
        let q = Q::from(Rational::from(m.clone()));
        if *neg {
            q * Q::from(-1)
        } else {
            q
        }
    };
    let column = Column::new(vec![(0, 0, coordinate(&x[0])), (1, 1, coordinate(&x[1]))]);
    let exact = ExactCgc::from_columns([2, 2, 1], vec![column], &[n], false, None).unwrap();
    let stored = exact.column_coords(0);
    for (j, (neg, m)) in x.iter().enumerate() {
        assert_eq!(&words_natural(stored.magnitude(j)), m, "stored X");
        assert_eq!(stored.is_negative(j), *neg, "stored sign");
    }
    assert_eq!(
        (stored.vertex().num(), stored.vertex().den()),
        (&rho.0, &rho.1)
    );
    let (two, one) = (unit(2), unit(1));
    fast_and_exact(&exact, [&two, &two, &one])
}

/// Entry `j` of a fixture: `(fast, exact)`, with a certified value equal
/// to the exact one.
fn entry(got: &[(Option<u64>, u64)], j: usize) -> (Option<u64>, u64) {
    let (fast, exact) = got[j];
    if let Some(f) = fast {
        assert_eq!(f, exact, "entry {j}: fast and exact differ");
    }
    (fast, exact)
}

const ONE: u64 = 0x3ff0_0000_0000_0000;
const ONE_UP: u64 = 0x3ff0_0000_0000_0001;

/// Entry 1 is `(2^53 + 1)·√ρ`: at `ρ = 2^-106` exactly the midpoint
/// `1 + 2^-53`, which falls back and ties to even; perturbed by `1 ± 2^-120`
/// (about 64 units from it) it falls back and rounds to the side of the
/// perturbation; by `1 ± 2^-110` (about `2^16` units) the estimate decides it.
/// Entry 2 rounds to `2^-52` through the fast path.
#[test]
fn midpoints_fall_back_and_their_far_neighbours_do_not() {
    let x = || [(false, pow2(53) + Natural::ONE), (false, nat(2))];
    let rows = [
        ((Natural::ONE, pow2(106)), None, ONE),
        ((pow2(120) + Natural::ONE, pow2(226)), None, ONE_UP),
        ((pow2(120) - Natural::ONE, pow2(226)), None, ONE),
        ((pow2(110) + Natural::ONE, pow2(216)), Some(ONE_UP), ONE_UP),
        ((pow2(110) - Natural::ONE, pow2(216)), Some(ONE), ONE),
    ];
    for (rho, fast, exact) in rows {
        let got = entries(x(), rho);
        assert_eq!(entry(&got, 0), (fast, exact));
        let two_pow_m52 = 0x3cb0_0000_0000_0000;
        assert_eq!(entry(&got, 1), (Some(two_pow_m52), two_pow_m52));
    }
}

/// Representable values are returned by the fast path: `(3, 2)` at
/// `ρ = 1/4` gives `1.5` and `1.0`, `(2^53 − 1, 2)` at `ρ = 1` gives
/// `2^53 − 1` and `2.0`.
#[test]
fn representable_values_take_the_fast_path() {
    let got = entries([(false, nat(3)), (false, nat(2))], (nat(1), nat(4)));
    assert_eq!(entry(&got, 0).0, Some(1.5f64.to_bits()));
    assert_eq!(entry(&got, 1).0, Some(1.0f64.to_bits()));
    let big = (1u64 << 53) - 1;
    let got = entries([(false, nat(big)), (false, nat(2))], (nat(1), nat(1)));
    assert_eq!(entry(&got, 0).0, Some((big as f64).to_bits()));
    assert_eq!(entry(&got, 1).0, Some(2.0f64.to_bits()));
}

/// Values outside the fast path's binades fall back and round exactly: a
/// nonzero subnormal (`(2^60 + 1)·2^-1100`, the subnormal `2^34·2^-1074`), a
/// signed-zero underflow (`−2^-1100`), an overflow (`2^1050`), and the
/// binades `−1022` and `1023` themselves; the next binades inward decide.
#[test]
fn range_gates_fall_back_to_the_exact_conversion() {
    let tiny = (Natural::ONE, pow2(2200));
    let got = entries(
        [(false, pow2(60) + Natural::ONE), (false, nat(2))],
        tiny.clone(),
    );
    assert_eq!(entry(&got, 0), (None, 0x0000_0004_0000_0000));
    let got = entries([(true, nat(1)), (false, nat(2))], tiny);
    assert_eq!(entry(&got, 0), (None, 0x8000_0000_0000_0000));
    let got = entries(
        [(false, nat(1)), (false, nat(2))],
        (pow2(2100), Natural::ONE),
    );
    assert_eq!(entry(&got, 0), (None, f64::INFINITY.to_bits()));
    let got = entries(
        [(false, nat(1)), (false, nat(2))],
        (Natural::ONE, pow2(2044)),
    );
    assert_eq!(entry(&got, 0), (None, f64::MIN_POSITIVE.to_bits()));
    let min_times_two = (2.0 * f64::MIN_POSITIVE).to_bits();
    assert_eq!(entry(&got, 1), (Some(min_times_two), min_times_two));
    let got = entries(
        [(false, nat(1)), (false, nat(2))],
        (pow2(2046), Natural::ONE),
    );
    assert_eq!(entry(&got, 0), (None, 2f64.powi(1023).to_bits()));
    assert_eq!(entry(&got, 1), (None, f64::INFINITY.to_bits()));
}

/// The midpoint scaled by an odd `k` in both coordinate and vertex,
/// `((2^53 + 1)·k, 2)` at `ρ = 1/(k²·2^106)`, falls back and rounds to `1.0`
/// whether the wider full-list product has 383 bits (`k = 2^138 + 1`, the
/// exact conversion's stack path) or 385 (`k = 2^139 + 1`, its big-integer
/// path).
#[test]
fn midpoints_round_exactly_on_both_sides_of_the_stack_limit() {
    for (shift, width) in [(138, 383), (139, 385)] {
        let k = pow2(shift) + Natural::ONE;
        let x = (pow2(53) + Natural::ONE) * &k;
        let rho = (Natural::ONE, &k * &k * pow2(106));
        // The full lists are `[X, X, ρ.num, 1, 1, 1]` and `[ρ.den, 1, 1, 1]`.
        let num = (&x * &x).significant_bits();
        let den = rho.1.significant_bits();
        assert_eq!(num.max(den), width);
        let got = entries([(false, x), (false, nat(2))], rho);
        assert_eq!(entry(&got, 0), (None, ONE));
        entry(&got, 1);
    }
}

/// Roots are formed exactly when `entries · 150 ≥ roots · 370` (ns): one
/// root pays from 3 entries, fifteen from 37, and no root needs nothing.
#[test]
fn roots_are_formed_from_the_break_even() {
    assert!(!pays(2, 1));
    assert!(pays(3, 1));
    assert!(!pays(36, 15));
    assert!(pays(37, 15));
    assert!(pays(0, 0));
    assert!(!pays(0, 1));
}

/// The gate routes columns within one channel: in `2 ⊗ 2 → 4` with `a = b`
/// sharing one table, a distinct `c`, `N = 1` and columns of 2, 3, 3 and 3
/// entries, the channel's 3 serial roots pay on the 9 entries of its
/// 3-entry columns, and only the 2-entry column skips its coupled root.
/// Every entry, through `round_column`, is `X/√2`, the correctly rounded
/// `√(X²/2)` that IEEE `sqrt` gives for the exactly representable `X²/2`.
#[test]
fn the_gate_routes_each_column_and_rounds_every_entry() {
    let q = Q::from;
    let cols: [&[(u32, u32, i64)]; 4] = [
        &[(0, 0, 1), (1, 1, -3)],
        &[(0, 0, 2), (0, 1, -5), (1, 0, 7)],
        &[(0, 1, 4), (1, 0, 6), (1, 1, -9)],
        &[(0, 0, 11), (1, 0, -13), (1, 1, 8)],
    ];
    let built = cols
        .iter()
        .map(|c| Column::new(c.iter().map(|&(ma, mb, x)| (ma, mb, q(x))).collect()))
        .collect();
    let exact = ExactCgc::from_columns([2, 2, 4], built, &[q(2)], false, None).unwrap();
    let (two, four) = (unit(2), unit(4));
    let norms = [&two, &two, &four];
    let lengths: Vec<u64> = cols.iter().map(|c| c.len() as u64).collect();
    let estimates = ChannelEstimates::for_channel(exact.vertices(), norms, &lengths)
        .expect("3 roots pay on 9 entries");
    for (k, col) in cols.iter().enumerate() {
        let coords = exact.column_coords(k);
        assert_eq!(
            estimates.column(&coords, &four).is_some(),
            col.len() >= 3,
            "column {k}"
        );
        let mut out = view_slots(col.len());
        round_column(&coords, norms, Some(&estimates), &mut out);
        for (e, &(ma, mb, x)) in out.iter().zip(col.iter()) {
            let want = ((x * x) as f64 / 2.0).sqrt();
            let want = if x < 0 { -want } else { want };
            assert_eq!((e.m1, e.m2, e.m3, e.mu), (ma, mb, k as u32, 0));
            assert_eq!(e.value.to_bits(), want.to_bits(), "column {k}, X = {x}");
        }
    }
}
