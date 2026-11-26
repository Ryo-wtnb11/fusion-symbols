//! The integralization's contracts: every arithmetic width gives the
//! integers of the definition, the column stages give the serial run's
//! slab at every width, and a two-word run allocates per column, never per
//! entry.

use super::*;
use malachite_base::num::logic::traits::SignificantBits;
use malachite_nz::integer::Integer;
use malachite_q::Rational;

/// `±p/q` reduced.
fn q(negative: bool, p: Natural, d: Natural) -> Q {
    Q::from(Rational::from_sign_and_naturals(!negative, p, d))
}

fn pow2(k: u32) -> Natural {
    Natural::ONE << k
}

fn nat(x: u128) -> Natural {
    Natural::from(x)
}

/// The definition, in big rationals: `D = lcm` of the denominators, `g`
/// the gcd of the integers `x·D` (1 when all are zero), `X = x·D/g`, negated
/// with `negate`.
fn oracle(xs: &[Q], negate: bool) -> (Vec<Integer>, Natural, Natural) {
    let rs: Vec<Rational> = xs.iter().map(Q::to_rational).collect();
    let mut den = Natural::ONE;
    for r in &rs {
        let d = r.denominator_ref();
        den = &den / (&den).gcd(d) * d;
    }
    let scaled: Vec<Rational> = rs.iter().map(|r| r * Rational::from(&den)).collect();
    let mut content = Natural::ZERO;
    for s in &scaled {
        assert_eq!(*s.denominator_ref(), Natural::ONE, "x·D is an integer");
        content = content.gcd(s.numerator_ref());
    }
    if content == Natural::ZERO {
        content = Natural::ONE;
    }
    let xs = scaled
        .iter()
        .map(|s| {
            let m = Integer::from(s.numerator_ref() / &content);
            if (*s < 0u32) != negate {
                -m
            } else {
                m
            }
        })
        .collect();
    (xs, den, content)
}

/// Words of the widest `|X|`, the canonical stride.
fn widest_words(xs: &[Integer]) -> usize {
    xs.iter()
        .map(|x| x.unsigned_abs_ref().significant_bits().div_ceil(64) as usize)
        .max()
        .unwrap_or(0)
}

/// Boundary inputs written out: numerators and denominators at `2^64 ± 1`
/// and `2^127`, a common denominator of exactly `2^128 − 1`, one beyond
/// two words, a bound that overestimates the widest `X` by a word, mixed
/// signs, zeros and the empty run. On each, the two-word integralization
/// (hardware words where its operands fit) and the big-integer one give the
/// integers, denominator and content of the definition, at the canonical
/// stride, with and without the vertex sign.
#[test]
fn every_width_gives_the_integers_of_the_definition() {
    let m64 = nat(u64::MAX as u128);
    let p64 = nat(1u128 << 64) + Natural::ONE;
    let cases: Vec<(&str, Vec<Q>)> = vec![
        ("empty", vec![]),
        ("zeros", vec![Q::from(0), Q::from(0)]),
        (
            "words",
            vec![
                q(false, nat(3), nat(4)),
                q(true, nat(5), nat(6)),
                Q::from(0),
            ],
        ),
        (
            "2^64 - 1 denominators",
            vec![q(false, nat(1), m64.clone()), q(true, nat(3), m64.clone())],
        ),
        (
            "2^64 + 1 numerator",
            vec![
                q(false, p64.clone(), nat(3)),
                q(true, nat(1), nat(2)),
                Q::from(0),
            ],
        ),
        (
            "2^127 numerators",
            vec![q(false, pow2(127), nat(1)), q(true, pow2(127), nat(3))],
        ),
        (
            "2^127 denominator",
            vec![q(true, nat(1), pow2(127)), Q::from(1)],
        ),
        (
            "lcm 2^128 - 1",
            vec![q(false, nat(1), p64.clone()), q(true, nat(1), m64.clone())],
        ),
        (
            "lcm beyond two words",
            vec![
                q(false, nat(1), p64.clone()),
                q(true, nat(1), m64.clone()),
                q(false, nat(1), nat(7)),
            ],
        ),
        (
            "numerator beyond two words",
            vec![q(false, pow2(130) + Natural::ONE, nat(3)), Q::from(-2)],
        ),
        (
            "bound one word wide",
            vec![
                q(false, nat(3u128.pow(30)), pow2(40)),
                q(true, nat(1), nat(1)),
            ],
        ),
    ];
    for (name, xs) in &cases {
        for negate in [false, true] {
            let (want, den, content) = oracle(xs, negate);
            let fast = integral(xs.iter(), negate).unwrap();
            let big = natural_integral(xs.iter(), negate).unwrap();
            assert_eq!(fast, big, "{name}, negate {negate}");
            let got: Vec<Integer> = (0..xs.len()).map(|i| fast.x.integer(i)).collect();
            assert_eq!(got, want, "{name}, negate {negate}");
            assert_eq!((&fast.den, &fast.content), (&den, &content), "{name}");
            assert_eq!(fast.x.stride(), widest_words(&want), "{name}");
        }
    }
}

/// Columns of four `μ` (four per `μ`): words only; a bound one word above
/// the widest `X`, narrowed; two words with numerators near `2^64`; and a
/// common denominator beyond two words, in big integers.
fn boundary_columns() -> (Vec<Column>, Vec<Q>) {
    let m64 = nat(u64::MAX as u128);
    let p64 = nat(1u128 << 64) + Natural::ONE;
    let mut cols: Vec<Column> = Vec::new();
    for mu in 0..4u32 {
        for k in 0..4u32 {
            let col = (0..(3 + 5 * k))
                .map(|i| {
                    let small = nat(u128::from(i + 1));
                    let (p, d) = match mu {
                        0 => (nat(u128::from(2 * i + 1)), nat(u128::from(k + 2))),
                        1 if i % 3 == 0 => (nat(3u128.pow(30)), pow2(40)),
                        1 => (small, nat(1)),
                        2 if i % 2 == 0 => (p64.clone() + nat(u128::from(2 * i)), nat(1)),
                        2 => (small, nat(2)),
                        _ if i == 0 && k == 3 => (nat(1), nat(7)),
                        _ if i % 2 == 0 => (small, p64.clone()),
                        _ => (small, m64.clone()),
                    };
                    (k, i, q((i + k) % 3 == 0, p, d))
                })
                .collect();
            cols.push(col);
        }
    }
    let norms = vec![Q::from(2), q(false, nat(3), nat(5)), Q::from(7), Q::from(1)];
    (cols, norms)
}

/// A real team of `width` threads.
fn team(width: usize) -> threadteam::Team {
    threadteam::Pool::builder()
        .threads(width)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap()
        .into_team(width)
        .unwrap()
}

/// The column stages give each `μ` the slab, scale and width of one serial
/// run over its entries, and the same coordinates through real sections of
/// widths 2–4 as inline, with and without the vertex sign.
#[test]
fn column_stages_give_the_serial_run_at_every_width() {
    let (cols, norms) = boundary_columns();
    let team = team(4);
    for negate in [false, true] {
        let serial = Coords::new(cols.clone(), 4, &norms, negate, None).unwrap();
        for (mu, coords) in serial.per_mu.iter().enumerate() {
            let run = integral(
                cols[4 * mu..4 * mu + 4].iter().flatten().map(|e| &e.2),
                negate,
            )
            .unwrap();
            assert_eq!(coords.x, run.x, "μ {mu}");
            assert_eq!(coords.bits, run.x.max_bits(), "μ {mu}");
            assert_eq!(coords.vertex, vertex(&norms[mu], &run.den, &run.content));
        }
        let strides: Vec<usize> = serial.per_mu.iter().map(|m| m.x.stride()).collect();
        assert_eq!(strides, [1, 1, 2, 2], "the four tiers");
        for width in [2, 3, 4] {
            let _force = crate::stage::force_width(width);
            INTEGRAL_SECTIONS.with(|s| s.set(0));
            let got =
                Coords::new(cols.clone(), 4, &norms, negate, Some(&mut team.exec(4))).unwrap();
            assert_eq!(got, serial, "width {width}");
            assert!(
                INTEGRAL_SECTIONS.with(std::cell::Cell::get) >= 2,
                "width {width}"
            );
        }
    }
}

/// On the two-word path the integralization allocates per column and per
/// `μ`, never per entry: eight columns of 64 and of 1024 entries cost the
/// same number of allocations, at a stride of two words.
#[cfg(feature = "lie-generated")]
#[test]
fn two_word_integralization_allocates_per_column() {
    let calls = crate::audit_alloc::thread_allocation_calls;
    let cost = |len: u64| {
        let cols: Vec<Column> = (0..8u64)
            .map(|k| {
                (0..len)
                    .map(|i| {
                        let p = (1u64 << 50) + 6 * (k * len + i) + 1;
                        let d = if i % 2 == 0 {
                            1u128 << 40
                        } else {
                            3u128.pow(25)
                        };
                        (0, i as u32, q(false, nat(u128::from(p)), nat(d)))
                    })
                    .collect()
            })
            .collect();
        let norms = [Q::from(1), Q::from(3)];
        let before = calls();
        let coords = Coords::new(cols, 4, &norms, false, None).unwrap();
        let spent = calls() - before;
        assert!(coords.per_mu.iter().all(|m| m.x.stride() == 2));
        spent
    };
    assert_eq!(cost(64), cost(1024));
}
