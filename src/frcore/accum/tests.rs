//! The plan's boundaries on synthetic bounds, the fixed accumulators against
//! malachite integers at the maximal width of every class, and the
//! canonical slab of a sum that cancels.

use super::*;
use rand::{RngExt, SeedableRng};

fn joined(bits: u64, terms: u64) -> Side {
    Side::Joined {
        bits: [bits, bits],
        terms,
    }
}

fn bounds(sides: [Side; 2], w_bits: u64, terms: u64) -> Bounds {
    Bounds {
        sides,
        w_bits,
        terms,
    }
}

fn fixed(join: Join, limbs: usize) -> Plan {
    Plan::Fixed {
        join,
        limbs,
        w_on_left: true,
    }
}

/// The join word count `J` and the final width `L` switch exactly at their
/// bounds: legs of 63 and 64 bits, joined widths of 127 and 128, legs of 256
/// and 257 bits, final operands of 1024 and 1025 bits, and term counts
/// `2^64/q − 1` (accepted, `q·t = 2^64 − q`) and `2^64/q` (`Big`, `q·t = 2^64`)
/// for every `q = 2J` and `q = 2L`; an overflowing bound is `Big`.
#[test]
fn plans_switch_exactly_at_their_bounds() {
    let direct = |bits| Side::Direct { bits };
    // Word against Wide(1): a 63-bit leg pair with one term joins in 127 bits.
    let p = choose(&bounds([direct(1), joined(63, 1)], 0, 1));
    assert_eq!(p, fixed(Join::Word, 4));
    let p = choose(&bounds([direct(1), joined(64, 1)], 0, 1));
    assert_eq!(p, fixed(Join::Wide(1), 4));
    // A joined width of 128 bits with 63-bit legs (127 was the Word case).
    let p = choose(&bounds([joined(63, 2), direct(1)], 0, 1));
    assert!(
        matches!(
            p,
            Plan::Fixed {
                join: Join::Wide(1),
                ..
            }
        ),
        "{p:?}"
    );
    // Legs of 256 against 257 bits.
    let p = choose(&bounds([joined(256, 1), direct(1)], 0, 1));
    assert!(
        matches!(
            p,
            Plan::Fixed {
                join: Join::Wide(4),
                ..
            }
        ),
        "{p:?}"
    );
    assert_eq!(
        choose(&bounds([joined(257, 1), direct(1)], 0, 1)),
        Plan::Big
    );
    // Final operands of 1024 against 1025 bits.
    let p = choose(&bounds([direct(1), direct(1024)], 0, 1));
    assert_eq!(p, fixed(Join::Word, 16));
    assert_eq!(choose(&bounds([direct(1025), direct(1)], 0, 1)), Plan::Big);
    // Join headroom for q = 2J.
    for (bits, q) in [(64, 2u64), (128, 4), (256, 8)] {
        let ok = (1u64 << 63) / (q / 2) - 1;
        let p = choose(&bounds([joined(bits, ok), direct(1)], 0, 1));
        assert!(matches!(p, Plan::Fixed { .. }), "q = {q}");
        let reject = (1u64 << 63) / (q / 2);
        assert_eq!(
            choose(&bounds([joined(bits, reject), direct(1)], 0, 1)),
            Plan::Big,
            "q = {q}"
        );
    }
    // Sum headroom for q = 2L.
    for (bits, q) in [(64u64, 2u64), (256, 8), (512, 16), (1024, 32)] {
        let ok = (1u64 << 63) / (q / 2) - 1;
        assert!(matches!(
            choose(&bounds([direct(bits), direct(1)], 0, ok)),
            Plan::Fixed { .. }
        ));
        let reject = (1u64 << 63) / (q / 2);
        assert_eq!(
            choose(&bounds([direct(bits), direct(1)], 0, reject)),
            Plan::Big
        );
    }
    // A bound that overflows u64.
    assert_eq!(
        choose(&bounds([joined(u64::MAX, 1), direct(1)], 0, 1)),
        Plan::Big
    );
    // The weight goes on the narrower side.
    assert!(matches!(
        choose(&bounds([direct(10), direct(20)], 5, 1)),
        Plan::Fixed {
            w_on_left: true,
            ..
        }
    ));
    assert!(matches!(
        choose(&bounds([direct(30), direct(20)], 5, 1)),
        Plan::Fixed {
            w_on_left: false,
            ..
        }
    ));
}

/// `n` random words of which the top one has all of its 64 bits when
/// `full`, else random.
fn words(rng: &mut rand_chacha::ChaCha8Rng, n: usize, full: bool) -> Vec<u64> {
    (0..n)
        .map(|_| if full { u64::MAX } else { rng.random::<u64>() })
        .collect()
}

/// One synthetic term: entry, then sign and magnitude of each factor.
type Term = (usize, bool, Vec<u64>, bool, Vec<u64>);

/// Runs the same terms through a plan and through `Big`, and asserts equal
/// slabs.
fn sums_agree(plan: Plan, entries: usize, terms: &[Term]) {
    let mut fast = SumAcc::new(&plan, entries);
    let mut big = SumAcc::new(&Plan::Big, entries);
    for (i, xn, x, yn, y) in terms {
        let (x, y) = (Signed { neg: *xn, mag: x }, Signed { neg: *yn, mag: y });
        fast.add(*i, x, y);
        big.add(*i, x, y);
    }
    assert_eq!(fast.finish(), big.finish(), "{plan:?}");
}

/// The fixed entry sums equal malachite's at the maximal operand width of
/// every `L`, with mixed signs, all-ones operands (maximal carries) and
/// all-positive banks whose settlement needs `2L + 1` words.
#[test]
fn fixed_sums_equal_big_sums() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x325b);
    for limbs in [1usize, 4, 8, 16] {
        let plan = fixed(Join::Word, limbs);
        let mut terms = Vec::new();
        for t in 0..40 {
            let full = t % 4 == 0;
            let (xn, yn) = (rng.random::<u64>() % 2 == 0, rng.random::<u64>() % 3 == 0);
            terms.push((
                t % 5,
                xn,
                words(&mut rng, limbs, full),
                yn,
                words(&mut rng, limbs, full),
            ));
        }
        sums_agree(plan, 5, &terms);
        // All-positive maximal terms: the settled value needs 2L + 1 words.
        let max = vec![u64::MAX; limbs];
        let terms: Vec<_> = (0..3)
            .map(|_| (0, false, max.clone(), false, max.clone()))
            .collect();
        sums_agree(plan, 1, &terms);
        let s = {
            let mut acc = SumAcc::new(&plan, 1);
            for (i, xn, x, yn, y) in &terms {
                acc.add(*i, Signed { neg: *xn, mag: x }, Signed { neg: *yn, mag: y });
            }
            acc.finish()
        };
        assert_eq!(s.stride(), 2 * limbs + 1);
    }
}

/// The fixed join cells equal malachite's at the maximal leg width of every
/// class, `Word` (63 bits) and `Wide(1, 2, 4)`, with mixed signs and maximal
/// carries; a read cell is cleared.
#[test]
fn fixed_joins_equal_big_joins() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x0325);
    for (join, words_per_leg) in [
        (Join::Word, 1usize),
        (Join::Wide(1), 1),
        (Join::Wide(2), 2),
        (Join::Wide(4), 4),
    ] {
        let plan = fixed(join, 16);
        let mut fast = JoinAcc::new(&plan, 3);
        let mut big = JoinAcc::new(&Plan::Big, 3);
        for t in 0..30 {
            let mut x = words(&mut rng, words_per_leg, t % 3 == 0);
            let mut y = words(&mut rng, words_per_leg, t % 3 == 0);
            if join == Join::Word {
                // Ten 61-bit products per cell stay within 127 bits.
                x[0] >>= 3;
                y[0] >>= 3;
            }
            let (x, y) = (
                Signed {
                    neg: t % 2 == 0,
                    mag: &x,
                },
                Signed {
                    neg: t % 5 == 0,
                    mag: &y,
                },
            );
            fast.add(t % 3, x, y);
            big.add(t % 3, x, y);
        }
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for cell in 0..3 {
            let (na, nb) = (fast.take(cell, &mut a), big.take(cell, &mut b));
            assert_eq!((na, &a), (nb, &b), "{join:?} cell {cell}");
            assert!(!fast.take(cell, &mut a) && a.is_empty(), "cleared");
        }
    }
}

/// A sum that cancels, `+(2^64 − 1)² − (2^64 − 1)²`, settles in three words
/// and is stored canonically: stride 0, no magnitude words, no sign.
#[test]
fn a_cancelled_sum_is_canonical_zero() {
    let m = [u64::MAX];
    let mut acc = SumAcc::new(&fixed(Join::Word, 1), 1);
    acc.add(
        0,
        Signed {
            neg: false,
            mag: &m,
        },
        Signed {
            neg: false,
            mag: &m,
        },
    );
    acc.add(
        0,
        Signed { neg: true, mag: &m },
        Signed {
            neg: false,
            mag: &m,
        },
    );
    let s = acc.finish();
    assert_eq!((s.len(), s.stride()), (1, 0));
    assert!(s.words(0, 1).is_empty());
    assert!(!s.is_negative(0));
}
