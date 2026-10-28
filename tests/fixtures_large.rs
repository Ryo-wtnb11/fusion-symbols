//! Exact 6j symbols at large doubled spins (hundreds and thousands): the
//! evaluator has no spin ceiling.
//!
//! Contracts: the exact orthogonality `Σ_x (2x+1)(2f+1) {a b x; c d f}² = 1`
//! (every term is a rational square, so the sum is exact) on generic large
//! arguments; and a few values checked against WignerSymbols.jl 2.0.0. Both
//! tiers run by default: the prime-factorized engine evaluates them in a
//! fraction of a second.

use malachite_base::num::basic::traits::{One, Zero};
use malachite_nz::integer::Integer;
use malachite_q::Rational;

use fusion_symbols::lie::su2::wigner_6j;

fn tri(a: u32, b: u32, c: u32) -> bool {
    (a + b + c).is_multiple_of(2) && c >= a.abs_diff(b) && c <= a + b
}

/// `Σ_x (dx+1)(df+1) {a b x; c d f}²` over every admissible `x`.
fn norm(a: u32, b: u32, c: u32, d: u32, f: u32) -> Rational {
    assert!(tri(a, d, f) && tri(c, b, f));
    let lo = a.abs_diff(b).max(c.abs_diff(d));
    let hi = (a + b).min(c + d);
    let mut sum = Rational::ZERO;
    for x in (lo..=hi).step_by(2) {
        let v = wigner_6j(a, b, x, c, d, f);
        sum += v.radicand() * Rational::from(Integer::from((x + 1) * (f + 1)));
    }
    sum
}

#[test]
fn six_j_orthogonality_is_exact_at_large_spins() {
    // `(a, b, c, d, f)`: hundreds, then thousands. A small `c` keeps the
    // `x` range short while `a, b, d, f` stay large and generic.
    for l in [
        [447, 545, 6, 300, 543],
        [331, 278, 12, 281, 270],
        [600, 401, 9, 404, 400],
        [1041, 1016, 8, 1019, 1010],
        [2337, 3101, 5, 3001, 3100],
    ] {
        assert_eq!(norm(l[0], l[1], l[2], l[3], l[4]), Rational::ONE, "{l:?}");
    }
}

#[test]
fn six_j_large_values_checked_against_wignersymbols() {
    // Checked against WignerSymbols.jl 2.0.0 (`RationalRoot{BigInt}`): the
    // exact signed square.
    let exact: [([u32; 6], i8, &str, &str); 2] = [
        (
            [447, 545, 394, 277, 299, 272],
            -1,
            "78046261490065802627398211543868345102171842545290655302489799888888045415657489467102722404493007963157151441992221317758531188582578482151791344018923225",
            "337684961380007806770126791871460413086757841015114340969912434689156595516612234330912524759535321733476513267724995090505560859434923920439699726469549165471657",
        ),
        (
            [331, 278, 545, 352, 255, 324],
            1,
            "2410406216429875681011000150282327032647750606995451298153064751626475238876681537390373756138321561050414372161610441776006632661823044642712345386301604032112609",
            "5727498848764300309634251429422987216379282932774155818930436926160098749599464243644351177313912007898080995591688037727474852123874547160454902339212230105276389840183358477680",
        ),
    ];
    for (d, sign, num, den) in exact {
        let got = wigner_6j(d[0], d[1], d[2], d[3], d[4], d[5]);
        assert_eq!(got.sign(), sign, "{d:?}");
        let want = Rational::from_integers(num.parse::<Integer>().unwrap(), den.parse().unwrap());
        assert_eq!(got.radicand(), &want, "{d:?}");
    }
    // Doubled spins in the thousands: `Float64` of the exact value.
    for (d, want) in [
        (
            [1041, 1016, 1297, 1231, 1038, 1233],
            1.741_071_424_311_935_8e-5,
        ),
        (
            [1024, 1338, 1332, 1258, 1388, 1066],
            -2.804_215_379_725_395e-5,
        ),
    ] {
        let got = wigner_6j(d[0], d[1], d[2], d[3], d[4], d[5]).to_f64();
        assert!(
            (got - want).abs() <= 2.0 * f64::EPSILON * want.abs(),
            "{d:?}: {got}"
        );
    }
}
