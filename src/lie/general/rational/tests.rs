use super::*;
use malachite_base::num::arithmetic::traits::Gcd;
use malachite_nz::integer::Integer;
use rand::{Rng, SeedableRng};

fn r(n: impl Into<Integer>, d: impl Into<Integer>) -> Rational {
    Rational::from_integers(n.into(), d.into())
}

fn p2(k: u64) -> Integer {
    Integer::from(1) << k
}

/// The representation invariant, checked from the payload alone.
fn canonical(q: &Q) -> bool {
    match &q.0 {
        Repr::Small(n, d) => {
            Natural::from(n.unsigned_abs()).gcd(Natural::from(d.get())) == 1u32
                && (*n != 0 || d.get() == 1)
        }
        Repr::Big(r) => word_pair(r).is_none(),
    }
}

fn is_small(q: &Q) -> bool {
    matches!(q.0, Repr::Small(..))
}

/// `q` holds `want` in its unique representation, whose tag follows from
/// the value alone.
fn check(q: &Q, want: &Rational, what: &str) {
    assert_eq!(q.to_rational(), *want, "{what}");
    assert!(canonical(q), "{what}: {q:?}");
    assert_eq!(*q, Q::from(want.clone()), "{what}: one form per value");
}

/// Values at the word boundaries, both inside and just outside.
fn boundary() -> Vec<Rational> {
    let nums: Vec<Integer> = vec![
        Integer::from(0),
        Integer::from(1),
        Integer::from(-1),
        Integer::from(3),
        Integer::from(-6),
        Integer::from(i64::MIN),
        Integer::from(i64::MIN + 1),
        Integer::from(i64::MAX),
        Integer::from(i64::MAX - 1),
        p2(63),
        -p2(63) - Integer::from(1),
        p2(64) - Integer::from(1),
        p2(100),
    ];
    let dens: Vec<Integer> = vec![
        Integer::from(1),
        Integer::from(2),
        Integer::from(3),
        p2(63) - Integer::from(1),
        p2(63),
        p2(63) + Integer::from(1),
        p2(64) - Integer::from(1),
        p2(64) - Integer::from(3),
        p2(64),
        p2(64) + Integer::from(1),
    ];
    let mut out = Vec::new();
    for n in &nums {
        for d in &dens {
            out.push(Rational::from_integers_ref(n, d));
        }
    }
    out
}

fn random(seed: u64, count: usize) -> Vec<Rational> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let bits = |rng: &mut rand_chacha::ChaCha8Rng| -> Integer {
        let b = rng.gen_range(1..=70u32);
        let lo: u64 = rng.gen();
        let hi: u64 = rng.gen();
        let x = (Integer::from(hi) << 64u64) + Integer::from(lo);
        (x >> (128 - u64::from(b))) | (Integer::from(1) << u64::from(b - 1))
    };
    (0..count)
        .map(|_| {
            let n = bits(&mut rng);
            let n = if rng.gen() { -n } else { n };
            Rational::from_integers(n, bits(&mut rng))
        })
        .collect()
}

/// An operation's name, its Malachite result and every ownership form of it.
type Case = (&'static str, Rational, Box<dyn Fn() -> Vec<Q>>);

/// Every operation and ownership form against Malachite, with the tag.
fn check_pair(x: &Rational, y: &Rational) {
    let (qx, qy) = (Q::from(x.clone()), Q::from(y.clone()));
    let mut cases: Vec<Case> = vec![
        ("add", x + y, {
            let (a, b) = (qx.clone(), qy.clone());
            Box::new(move || {
                let mut s = a.clone();
                s += &b;
                let mut t = a.clone();
                t += b.clone();
                vec![
                    &a + &b,
                    a.clone() + &b,
                    a.clone() + b.clone(),
                    &a + b.clone(),
                    s,
                    t,
                ]
            })
        }),
        ("sub", x - y, {
            let (a, b) = (qx.clone(), qy.clone());
            Box::new(move || {
                let mut s = a.clone();
                s -= &b;
                let mut t = a.clone();
                t -= b.clone();
                vec![
                    &a - &b,
                    a.clone() - &b,
                    a.clone() - b.clone(),
                    &a - b.clone(),
                    s,
                    t,
                ]
            })
        }),
        ("mul", x * y, {
            let (a, b) = (qx.clone(), qy.clone());
            Box::new(move || {
                let mut s = a.clone();
                s *= &b;
                let mut t = a.clone();
                t *= b.clone();
                vec![
                    &a * &b,
                    a.clone() * &b,
                    a.clone() * b.clone(),
                    &a * b.clone(),
                    s,
                    t,
                ]
            })
        }),
    ];
    if *y != 0u32 {
        cases.push(("div", x / y, {
            let (a, b) = (qx.clone(), qy.clone());
            Box::new(move || {
                let mut s = a.clone();
                s /= &b;
                vec![
                    &a / &b,
                    a.clone() / &b,
                    a.clone() / b.clone(),
                    &a / b.clone(),
                    s,
                ]
            })
        }));
    }
    for (name, want, got) in cases {
        for (form, q) in got().iter().enumerate() {
            check(q, &want, &format!("{x} {name} {y} (form {form})"));
        }
    }
    // Fused forms, accumulating into either operand.
    for (acc, qacc) in [(x, &qx), (y, &qy)] {
        let mut z = qacc.clone();
        z.add_mul(&qx, &qy);
        check(&z, &(acc + x * y), &format!("{acc} + {x}·{y}"));
        let mut z = qacc.clone();
        z.sub_mul(&qx, &qy);
        check(&z, &(acc - x * y), &format!("{acc} - {x}·{y}"));
    }
    check(&-qx.clone(), &-x.clone(), &format!("-({x})"));
    check(&qx, x, &format!("ingress {x}"));
}

#[test]
fn operations_match_malachite_at_word_boundaries() {
    let values = boundary();
    for x in &values {
        for y in &values {
            check_pair(x, y);
        }
    }
}

#[test]
fn operations_match_malachite_on_random_values() {
    let xs = random(0x8b01, 3000);
    let ys = random(0x8b02, 3000);
    for (x, y) in xs.iter().zip(&ys) {
        check_pair(x, y);
    }
}

#[test]
fn representation_does_not_depend_on_history() {
    let big = Q::from(Rational::from(p2(100)));
    assert!(!is_small(&big));
    // Cancellation after promotion returns to words, in every form.
    let one = &big - &Q::from(Rational::from(p2(100) - Integer::from(1)));
    assert!(is_small(&one) && one.is_one());
    let mut zero = big.clone();
    zero -= &big;
    assert!(is_small(&zero) && zero.is_zero());
    let mut unit = big.clone();
    unit /= big.clone();
    assert!(is_small(&unit) && unit.is_one());
    // (2^70 / 3) · (3 / 2^70) = 1.
    let a = Q::from(r(p2(70), 3));
    let b = Q::from(r(3, p2(70)));
    assert_eq!(&a * &b, Q::from(1));
    // -i64::MIN = 2^63 needs a Big; negating back returns to words.
    let min = Q::from(i64::MIN);
    let pos = -min.clone();
    assert!(!is_small(&pos));
    assert_eq!(-pos, min);
    // Equal values from different histories are equal.
    assert_eq!(Q::from(r(2, 4)), &Q::from(1) / &Q::from(2));
    assert_ne!(Q::from(r(1, 2)), Q::from(r(-1, 2)));
}

#[test]
fn checked_sums_fall_back_exactly() {
    // The cross products fit i128 but their sum does not (leaf-8 review §2).
    let x = r(i64::MAX, p2(64) - Integer::from(1));
    let y = r(i64::MAX, p2(64) - Integer::from(3));
    let sum = &Q::from(x.clone()) + &Q::from(y.clone());
    check(&sum, &(&x + &y), "overflowing sum");
    assert!(!is_small(&sum));
    // A sum of exactly -2^127 must use the unsigned magnitude.
    let x = r(i64::MIN, 1);
    let y = r(i64::MIN, p2(64) - Integer::from(1));
    check(
        &(&Q::from(x.clone()) + &Q::from(y.clone())),
        &(&x + &y),
        "i128::MIN sum",
    );
}

#[test]
fn division_moves_the_sign_and_promotes_reciprocals() {
    let q = |x: Rational| Q::from(x);
    check(
        &(&Q::from(1) / &Q::from(i64::MIN)),
        &r(-1, p2(63)),
        "1/i64::MIN",
    );
    let tiny = q(r(1, p2(64) - Integer::from(1)));
    let inv = &Q::from(1) / &tiny;
    assert!(!is_small(&inv));
    check(
        &inv,
        &Rational::from(p2(64) - Integer::from(1)),
        "1/(1/u64::MAX)",
    );
    check(&(&Q::from(3) / &Q::from(-6)), &r(-1, 2), "3/-6");
}

#[test]
#[should_panic(expected = "division by zero")]
fn division_by_zero_panics() {
    let _ = &Q::from(1) / &Q::from(0);
}

#[test]
#[should_panic(expected = "division by zero")]
fn zero_over_zero_panics() {
    let _ = &Q::from(0) / &Q::from(0);
}

#[test]
#[should_panic(expected = "division by zero")]
fn big_division_by_zero_panics() {
    let mut x = Q::from(Rational::from(p2(100)));
    x /= Q::from(0);
}

#[test]
fn ingress_reduces_and_selects_by_value() {
    let q = |x: Rational| Q::from(x);
    assert!(is_small(&q(r(-6, -4))) && q(r(-6, -4)) == q(r(3, 2)));
    assert!(is_small(&q(Rational::from(Integer::from(i64::MIN)))));
    assert!(!is_small(&q(Rational::from(p2(63)))));
    assert!(is_small(&q(r(1, p2(64) - Integer::from(1)))));
    assert!(!is_small(&q(r(1, p2(64)))));
    assert!(q(Rational::from(0)).is_zero());
    let x = q(r(-7, 3));
    let (n, d) = x.magnitudes();
    assert_eq!(
        (n.into_owned(), d.into_owned()),
        (Natural::from(7u32), Natural::from(3u32))
    );
}

/// Word operations with word results, and rounding from word values, never
/// touch the heap.
#[cfg(feature = "lie-generated")]
#[test]
fn word_arithmetic_does_not_allocate() {
    // Operands below 2^30 keep every result of the four operations in words.
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x8b03);
    let xs: Vec<Q> = (0..2000)
        .map(|_| {
            let n: i64 = rng.gen_range(-(1 << 30)..1 << 30);
            let d: i64 = rng.gen_range(1..1 << 30);
            Q::from(r(if n == 0 { 1 } else { n }, d))
        })
        .collect();
    let (a, b, s, t) = (
        Q::from(r(3, 7)),
        Q::from(r(5, 11)),
        Q::from(r(-2, 9)),
        Q::from(r(13, 4)),
    );
    let calls = crate::audit_alloc::thread_allocation_calls;
    let mut out = Vec::with_capacity(5 * xs.len());
    let before = calls();
    for (x, y) in xs.iter().zip(xs.iter().skip(1)) {
        out.extend([x + y, x - y, x * y, x / y]);
        let mut z = x.clone();
        z += x * y;
        z -= y;
        z.add_mul(x, y);
        z.sub_mul(y, x);
        out.push(z);
    }
    let v = rounded_radical(&s, &a, &b, &t);
    assert_eq!(calls() - before, 0);
    assert!(out.iter().all(is_small));
    assert!(v < 0.0);
}

/// A Big value is charged at least the bytes that dropping it frees, once
/// compacted, even when Malachite handed over a vector with spare capacity.
#[cfg(feature = "lie-generated")]
#[test]
fn compacted_big_values_are_fully_charged() {
    let live = crate::audit_alloc::thread_live_bytes;
    let bloated = || {
        let mut limbs = Vec::with_capacity(1024);
        limbs.extend([1u64, 1]);
        Q::from(Rational::from(Natural::from_owned_limbs_asc(limbs)))
    };
    // Without compaction, the charge misses the spare capacity.
    let q = bloated();
    let charged = q.heap_bytes();
    let before = live();
    drop(q);
    assert!((before - live()) as usize > charged + 4096);
    // After compaction the charge bounds what the value owns.
    let mut q = bloated();
    q.compact();
    assert_eq!(q.to_rational(), Rational::from(p2(64) + Integer::from(1)));
    let charged = q.heap_bytes();
    let before = live();
    drop(q);
    let freed = (before - live()) as usize;
    assert!(
        freed >= std::mem::size_of::<Rational>() && freed <= charged,
        "{freed} > {charged}"
    );
}
