//! The estimate's contracts, each against exact rationals formed from the
//! inputs: `top`, `mul` and `root` truncate from below by less than one
//! unit, the stack root equals the big-integer one, an entry lies in `[V(1 + u)^-9, V]`, exponents are exact and
//! checked, and `decide` rounds and gates as stated.

use super::*;
use malachite_base::num::arithmetic::traits::PowerOf2;
use malachite_base::num::basic::traits::{One, Zero};
use malachite_q::Rational;
use rand::{RngExt, SeedableRng};

use crate::exact::radical::words_natural;

/// The value `m·2^e`.
fn value(v: Est) -> Rational {
    Rational::from(Natural::from(v.m)) * Rational::power_of_2(v.e)
}

/// `n·u` with `u = 2^-127`.
fn units(n: u32) -> Rational {
    Rational::from(n) * Rational::power_of_2(-127i64)
}

/// Whether `got/want` lies in `[1 − lo, 1 + hi]`, exactly.
fn within(got: &Rational, want: &Rational, lo: Rational, hi: Rational) -> bool {
    *got >= want * (Rational::ONE - lo) && *got <= want * (Rational::ONE + hi)
}

fn rng() -> rand_chacha::ChaCha8Rng {
    rand_chacha::ChaCha8Rng::seed_from_u64(0x0327)
}

/// `n` random words with a nonzero top word.
fn words(rng: &mut rand_chacha::ChaCha8Rng, n: usize) -> Vec<u64> {
    let mut w: Vec<u64> = (0..n).map(|_| rng.random::<u64>()).collect();
    if let Some(top) = w.last_mut() {
        *top |= 1 << (rng.random::<u64>() % 64);
    }
    w
}

/// `top` is exact up to 128 bits, normalizes (`1` and `2` at both exponent
/// parities, `2^126 + 1` to mantissa `2^127 + 2`), ignores high zero words,
/// agrees between words and big integers, and truncates a wider integer
/// within one unit from below.
#[test]
fn top_is_exact_to_128_bits_and_truncates_within_a_unit() {
    assert_eq!(top_words(&[1]), Some(Est::new(1 << 127, -127)));
    assert_eq!(top_words(&[2]), Some(Est::new(1 << 127, -126)));
    assert_eq!(top_words(&[1, 1 << 62]), Some(Est::new((1 << 127) + 2, -1)));
    assert_eq!(top_words(&[5, 0, 0]), top_words(&[5]));
    assert_eq!(top_words(&[0, 0]), None);
    assert_eq!(top_natural(&Natural::ZERO), None);
    let mut rng = rng();
    for n in [1, 2, 3, 4, 7] {
        for _ in 0..200 {
            let w = words(&mut rng, n);
            let x = words_natural(&w);
            let t = top_words(&w).unwrap();
            assert_eq!(top_natural(&x), Some(t));
            let exact = Rational::from(x);
            assert!(within(&value(t), &exact, units(1), Rational::ZERO));
            if n <= 2 {
                assert_eq!(value(t), exact);
            }
        }
    }
}

/// `mul` is malachite's product of the mantissas truncated to its top 128
/// bits, with the exponent sum plus the dropped bits: minimum, maximum and
/// regression-pair mantissas and random ones; two minimum mantissas drop 127
/// bits and two maximum ones 128.
#[test]
fn mul_is_the_truncated_big_integer_product() {
    let max = u128::MAX;
    let min = Est::new(1 << 127, 0);
    assert_eq!(mul(min, min), Some(Est::new(1 << 127, 127)));
    assert_eq!(
        mul(Est::new(max, 0), Est::new(max, 0)),
        Some(Est::new(max - 1, 128))
    );
    let pair = [
        top_words(&[1, 1 << 62]).unwrap(),
        top_words(&[u64::MAX, 1 << 63]).unwrap(),
    ];
    let mut cases = vec![
        (min, min),
        (Est::new(max, -3), Est::new(max, 5)),
        (Est::new(max, 0), min),
        (pair[0], pair[1]),
    ];
    let mut rng = rng();
    for _ in 0..1000 {
        let m = |rng: &mut rand_chacha::ChaCha8Rng| {
            (u128::from(rng.random::<u64>()) << 64 | u128::from(rng.random::<u64>())) | (1 << 127)
        };
        cases.push((Est::new(m(&mut rng), -200), Est::new(m(&mut rng), 77)));
    }
    for (a, b) in cases {
        let product = Natural::from(a.m) * Natural::from(b.m);
        let dropped = if product.significant_bits() == 256 {
            128
        } else {
            127
        };
        let m = u128::try_from(&(product >> dropped)).unwrap();
        assert_eq!(mul(a, b), Some(Est::new(m, a.e + b.e + dropped as i64)));
    }
}

/// Exponents are exact `i64` far beyond `f64`'s range and checked at the
/// end of `i64`: at `2^30` the multiply adds them exactly in both branches
/// (`+127` for two minimum mantissas, `+128` for two maximum ones), and an
/// exponent sum past `i64::MAX` is `None`, which sends the entry to the
/// exact conversion.
#[test]
fn exponents_are_exact_and_checked() {
    let e = 1i64 << 30;
    let (min, max) = (1u128 << 127, u128::MAX);
    assert_eq!(
        mul(Est::new(min, e), Est::new(min, e)),
        Some(Est::new(min, (1 << 31) + 127))
    );
    assert_eq!(
        mul(Est::new(max, e), Est::new(max, e)),
        Some(Est::new(max - 1, (1 << 31) + 128))
    );
    let big = i64::MAX / 2;
    assert_eq!(mul(Est::new(min, big), Est::new(min, big)), None);
    assert_eq!(mul(Est::new(min, -big), Est::new(min, -big - 3)), None);
}

/// `root(p, q)` is `√(p/q)` truncated from below by less than one unit,
/// checked on exact squares: ratios whose root is exactly `1`, `2` and
/// `3/2` (the minimum mantissa, an exactly representable one), roots just
/// below `2` and `1` (the maximum mantissa) at even and odd differences of
/// the operands' bit lengths, either operand far wider, and random word
/// and big operands of one to seven words.
#[test]
fn root_truncates_from_below_within_a_unit() {
    let mut rng = rng();
    let pow2 = |k: u32| Natural::ONE << k;
    let mut pairs: Vec<(Natural, Natural)> = vec![
        (Natural::from(1u32), Natural::from(1u32)),
        (Natural::from(4u32), Natural::from(1u32)),
        (Natural::from(9u32), Natural::from(4u32)),
        (pow2(202) - Natural::ONE, pow2(200)),
        (pow2(201) - Natural::ONE, pow2(201)),
        (pow2(203) - Natural::ONE, pow2(200)),
        (Natural::from(3u32), pow2(900) + Natural::ONE),
        (pow2(900) + Natural::ONE, Natural::from(3u32)),
    ];
    for (n, d) in [(1, 1), (1, 3), (2, 2), (7, 1), (3, 7)] {
        for _ in 0..100 {
            pairs.push((
                words_natural(&words(&mut rng, n)),
                words_natural(&words(&mut rng, d)),
            ));
        }
    }
    for (p, q) in pairs {
        let pw = crate::exact::radical::natural_words(&p);
        let s = root_of(Factor::Words(&pw), Factor::Natural(&q)).unwrap();
        assert_eq!(root(&p, &q), Some(s));
        let ratio = Rational::from_naturals(p.clone(), q.clone());
        let up = Rational::ONE + units(1);
        let s2 = value(s) * value(s);
        assert!(s2 <= ratio && ratio < &s2 * (&up * &up), "√({p}/{q})");
    }
    assert_eq!(
        root(&Natural::from(9u32), &Natural::from(4u32)),
        Some(Est::new(3 << 126, -127))
    );
    assert_eq!(root(&Natural::ZERO, &Natural::ONE), None);
}

/// An entry `|X|·√ρ/√n^c·√n^a·√n^b` is a truncation from below of the exact
/// value by its nine one-sided steps, `V̂² ≤ V² < V̂²(1 + u)^18`, with `V²`
/// formed from the original inputs, for `X` of one, two, three and seven
/// words and random word and big norms and vertices.
#[test]
fn entry_is_within_nine_one_sided_units() {
    let mut rng = rng();
    let mut nat = |n: usize| words_natural(&words(&mut rng, n));
    let up = Rational::ONE + units(1);
    let mut bound = Rational::ONE;
    for _ in 0..18 {
        bound *= &up;
    }
    for xn in [1, 2, 3, 7] {
        for case in 0..60 {
            let w = (case % 3) + 1;
            let x = nat(xn);
            let [rn, rd, pa, qa, pb, qb, pc, qc] = [5, 4, w, 1, 1, w, w, w].map(&mut nat);
            let k = mul(root(&rn, &rd).unwrap(), root(&qc, &pc).unwrap()).unwrap();
            let xw = crate::exact::radical::natural_words(&x);
            let sa = root(&pa, &qa).unwrap();
            let sb = root(&pb, &qb).unwrap();
            let v = value(entry(&xw, k, sa, sb).unwrap());
            let q = |p: &Natural, q: &Natural| Rational::from_naturals(p.clone(), q.clone());
            let x = Rational::from(x);
            let exact = &x * &x * q(&rn, &rd) * q(&pa, &qa) * q(&pb, &qb) * q(&qc, &pc);
            let v2 = &v * &v;
            assert!(v2 <= exact && exact < &v2 * &bound);
        }
    }
}

/// `decide` returns the common rounding of the ends `m ∓ MARGIN`: down when
/// both lie below the midpoint `1 + 2^-53` (the upper end may be the
/// midpoint itself, which ties to the even `1.0`), up when both lie above
/// it, and `None` when the midpoint lies strictly between them or, at the
/// lower end, ties to even away from the upper end's rounding. An upper end
/// past `u128` rounds to `2^128`, so a floor estimate just below a power of
/// two returns it; it falls back outside the binades `−1021..=1022`,
/// accepts a minimum mantissa whose lower end crosses into the binade
/// below, scales exactly at both ends of the range, and sets the sign.
#[test]
fn decide_rounds_the_ends_and_gates() {
    let half = 1u128 << 74;
    let one = 1u128 << 127;
    // Value 1 + r·2^-127 at exponent −127.
    let at = |r: u128| Est::new(one + r, -127);
    assert_eq!(decide(false, at(0)), Some(1.0));
    assert_eq!(decide(true, at(0)), Some(-1.0));
    assert_eq!(decide(false, at(half - MARGIN - 1)), Some(1.0));
    assert_eq!(decide(false, at(half - MARGIN)), Some(1.0));
    assert_eq!(decide(false, at(half - MARGIN + 1)), None);
    assert_eq!(decide(false, at(half)), None);
    assert_eq!(decide(false, at(half + MARGIN)), None);
    assert_eq!(
        decide(false, at(half + MARGIN + 1)),
        Some(f64::from_bits(0x3ff0_0000_0000_0001))
    );
    // A floor estimate just below a power of two: the upper end passes
    // `u128` and rounds to `2^128`, at an ordinary exponent and at the top
    // binade, where `2^1023` is still finite.
    assert_eq!(decide(false, Est::new(u128::MAX, -128)), Some(1.0));
    assert_eq!(decide(true, Est::new(u128::MAX - 3, -129)), Some(-0.5));
    assert_eq!(
        decide(false, Est::new(u128::MAX, 1022 - 127)),
        Some(f64::from_bits(2046 << 52))
    );
    assert_eq!(decide(false, Est::new(u128::MAX - MARGIN, -128)), Some(1.0));
    // Binades −1022 and 1023 fall back; −1021 and 1022 decide exactly.
    assert_eq!(decide(false, Est::new(one, -1022 - 127)), None);
    assert_eq!(decide(false, Est::new(one, 1023 - 127)), None);
    assert_eq!(
        decide(false, Est::new(one, -1021 - 127)),
        Some(f64::from_bits(2 << 52))
    );
    assert_eq!(
        decide(false, Est::new(one, 1022 - 127)),
        Some(f64::from_bits(2045 << 52))
    );
}

/// Values of every boundary class for the stack root: small integers,
/// `2^i` and `2^i ± 1` across the range, `3·2^(i−1)`, `2^127 − 1`,
/// `2^128 − 1`, `(2^64 − 1)²` and `3^80`.
fn boundary_values() -> Vec<u128> {
    let mut v: Vec<u128> = (1..=9).collect();
    for i in [1u32, 2, 31, 32, 33, 63, 64, 65, 95, 96, 126, 127] {
        let t = 1u128 << i;
        v.extend([t - 1, t, t + 1, 3 * (t >> 1)]);
    }
    v.extend([
        (1 << 127) - 1,
        u128::MAX - 1,
        u128::MAX,
        u128::from(u64::MAX) * u128::from(u64::MAX),
        3u128.pow(80),
    ]);
    v.sort_unstable();
    v.dedup();
    v
}

/// The stack root equals the big-integer root, `⌊√⌊p·2^(2k)/q⌋⌋` by
/// malachite's `floor_sqrt`, normalized: on every pair of boundary values
/// (`p` or `q` = 1, `p = q`, powers of two and their neighbours, both ends
/// of two words), on perfect-square ratios `a²c/b²c` and their `±1`
/// neighbours, and on seeded random pairs of random bit lengths.
#[test]
fn stack_root_equals_the_big_integer_root() {
    let oracle = |p: u128, q: u128| root(&Natural::from(p), &Natural::from(q));
    let values = boundary_values();
    let mut pairs: Vec<(u128, u128)> = Vec::new();
    for &p in &values {
        for &q in &values {
            pairs.push((p, q));
        }
    }
    for a in [1u128, 2, 3, 7, 1 << 40, (1 << 63) - 1] {
        for b in [1u128, 5, 9, (1 << 31) + 1] {
            for c in [1u128, 3, 1 << 20] {
                let (Some(p), Some(q)) = (a.checked_mul(a * c), b.checked_mul(b * c)) else {
                    continue;
                };
                for (dp, dq) in [(0, 0), (1, 0), (0, 1)] {
                    if let (Some(up), Some(uq)) = (p.checked_add(dp), q.checked_add(dq)) {
                        pairs.push((up, uq));
                    }
                    pairs.push((p - dp, q.max(2) - dq));
                }
            }
        }
    }
    let mut rng = rng();
    for _ in 0..3000 {
        let r = |rng: &mut rand_chacha::ChaCha8Rng| {
            let bits = 1 + rng.random::<u64>() % 128;
            let x = u128::from(rng.random::<u64>()) << 64 | u128::from(rng.random::<u64>());
            (x >> (128 - bits)) | (1 << (bits - 1))
        };
        pairs.push((r(&mut rng), r(&mut rng)));
    }
    for (p, q) in pairs {
        assert_eq!(stack_root(p, q), oracle(p, q), "√({p}/{q})");
    }
    assert_eq!(stack_root(0, 1), None);
    assert_eq!(stack_root(1, 0), None);
}
