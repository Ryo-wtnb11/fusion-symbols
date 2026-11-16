//! Prime-factorized integer arithmetic for the Racah/Wigner factorial engine.
//!
//! Ported from WignerSymbols.jl v2.0.0 `src/primefactorization.jl`
//! (`PrimeFactorization`, `primefactor`, `primefactorial`, `mul!`,
//! `divexact!`, `gcd!`, `lcm!`, `divgcd!`, `splitsquare`). A nonnegative
//! integer is stored as the exponent vector of its prime factorization plus a
//! sign; products and exact quotients are integer vector add/sub, so no big
//! integer is multiplied until a single final reconstruction. Factorials and
//! their ratios (the whole cost of a 3j/6j evaluation) stay in exponent space.
//!
//! ## Thread-safety and the memory contract
//!
//! Two process-global tables grow monotonically and are shared across threads:
//! the prime list and the factorial exponent-vector table, together in one
//! [`RwLock`]. An evaluation takes it once through [`with_tables`], which
//! first grows the tables to the evaluation's largest factorial argument; only
//! that growth takes the write lock.
//!
//! The tables never shrink. Their size is bounded by the largest factorial
//! argument `N` any call has requested: the factorial table holds `N + 1` rows
//! of `~pi(N)` `u32` exponents each, and the prime list holds `~pi(N)` values.
//! For doubled spins in the thousands `N` is a few thousand, so the tables are
//! a few megabytes and are amortized across every subsequent symbol.

use malachite_base::num::{
    arithmetic::traits::Pow,
    basic::traits::{One, Zero},
};
use std::sync::RwLock;

use malachite_nz::integer::Integer;

/// A signed integer stored as `sign * prod_i prime(i)^powers[i]`.
///
/// `powers[i]` is the exponent of the `i`-th prime (`powers[0]` -> 2). The
/// vector is normalized so its last entry is nonzero (an empty vector is the
/// integer 1, or 0 when `sign == 0`). `sign` is `-1`, `0`, or `+1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Pf {
    powers: Vec<u32>,
    sign: i8,
}

impl Pf {
    /// The integer 1.
    pub(crate) fn one() -> Self {
        Pf {
            powers: Vec::new(),
            sign: 1,
        }
    }

    /// Wrap a normalized (trailing-zero-trimmed) positive exponent vector.
    fn from_powers(mut powers: Vec<u32>) -> Self {
        trim(&mut powers);
        Pf { powers, sign: 1 }
    }

    /// Negate in place (flip the sign).
    pub(crate) fn neg(mut self) -> Self {
        self.sign = -self.sign;
        self
    }

    fn is_zero(&self) -> bool {
        self.sign == 0
    }

    /// `self *= other` — exponent vectors add, signs multiply
    /// (`primefactorization.jl::mul!`).
    pub(crate) fn mul_assign(&mut self, other: &Pf) {
        if self.sign == 0 || other.sign == 0 {
            self.sign = 0;
            self.powers.clear();
            return;
        }
        self.sign *= other.sign;
        if other.powers.len() > self.powers.len() {
            self.powers.resize(other.powers.len(), 0);
        }
        for (a, b) in self.powers.iter_mut().zip(other.powers.iter()) {
            *a += *b;
        }
    }

    /// `self *= prod_i prime(i)^powers[i]`, folding a raw positive exponent
    /// slice (a factorial table row) straight into `self` without allocating a
    /// [`Pf`] for it. This is the clone-free multiplicand path: callers
    /// accumulate a product of factorials in one buffer instead of cloning each
    /// `O(pi(N))` row only to add and drop it. Sign is unchanged (the slice is a
    /// factorial, hence positive).
    fn mul_by_powers(&mut self, powers: &[u32]) {
        if self.sign == 0 {
            return;
        }
        if powers.len() > self.powers.len() {
            self.powers.resize(powers.len(), 0);
        }
        for (a, b) in self.powers.iter_mut().zip(powers.iter()) {
            *a += *b;
        }
    }

    /// `self /= other`, exact division — exponent vectors subtract
    /// (`primefactorization.jl::divexact!`). `other` must divide `self`, which
    /// is guaranteed at every call site (a factorial ratio, or a divisor pulled
    /// out by gcd/lcm); the debug assertion documents that invariant.
    pub(crate) fn divexact_assign(&mut self, other: &Pf) {
        debug_assert!(other.sign != 0, "exact division by zero");
        if self.sign == 0 {
            return;
        }
        self.sign *= other.sign;
        debug_assert!(
            other.powers.len() <= self.powers.len(),
            "divexact: divisor has a prime the dividend lacks"
        );
        for (a, b) in self.powers.iter_mut().zip(other.powers.iter()) {
            debug_assert!(*a >= *b, "divexact: non-divisible exponent");
            *a -= *b;
        }
        trim(&mut self.powers);
    }

    /// `self = lcm(self, other)` (positive) — exponent-wise max
    /// (`primefactorization.jl::lcm!`).
    fn lcm_assign(&mut self, other: &Pf) {
        self.sign = 1;
        if other.powers.len() > self.powers.len() {
            self.powers.resize(other.powers.len(), 0);
        }
        for (a, b) in self.powers.iter_mut().zip(other.powers.iter()) {
            *a = (*a).max(*b);
        }
    }

    /// `self = gcd(self, other)` (positive) — exponent-wise min
    /// (`primefactorization.jl::gcd!`).
    fn gcd_assign(&mut self, other: &Pf) {
        self.sign = 1;
        let l = self.powers.len().min(other.powers.len());
        self.powers.truncate(l);
        for (a, b) in self.powers.iter_mut().zip(other.powers.iter()) {
            *a = (*a).min(*b);
        }
        trim(&mut self.powers);
    }

    /// Split off the common gcd of `a` and `b` in place, so afterwards
    /// `gcd(a, b) == 1` (`primefactorization.jl::divgcd!`). Keeps the two
    /// reconstructed big integers as small as possible.
    pub(crate) fn divgcd(a: &mut Pf, b: &mut Pf) {
        let l = a.powers.len().min(b.powers.len());
        for k in 0..l {
            let g = a.powers[k].min(b.powers[k]);
            a.powers[k] -= g;
            b.powers[k] -= g;
        }
        trim(&mut a.powers);
        trim(&mut b.powers);
    }

    /// Split `a = s^2 * r` with `r` square-free (all exponents 0 or 1)
    /// (`primefactorization.jl::splitsquare`). `r` carries the sign.
    pub(crate) fn splitsquare(&self) -> (Pf, Pf) {
        let s = Pf::from_powers(self.powers.iter().map(|&e| e >> 1).collect());
        let mut r = Pf::from_powers(self.powers.iter().map(|&e| e & 1).collect());
        r.sign = self.sign;
        (s, r)
    }

    /// Reconstruct the big integer (`primefactorization.jl::_convert!`). This is
    /// the only place a prime is raised to a power and multiplied out.
    pub(crate) fn to_bigint(&self, t: &Tables) -> Integer {
        if self.sign == 0 {
            return Integer::ZERO;
        }
        let mut acc = Integer::ONE;
        for (i, &e) in self.powers.iter().enumerate() {
            if e != 0 {
                acc *= Integer::from(t.prime(i)).pow(u64::from(e));
            }
        }
        if self.sign < 0 {
            -acc
        } else {
            acc
        }
    }
}

/// Drop trailing zero exponents so the representation is canonical.
fn trim(powers: &mut Vec<u32>) {
    while matches!(powers.last(), Some(0)) {
        powers.pop();
    }
}

// ---------------------------------------------------------------------------
// Growing global tables (prime list, factorial exponent vectors).
// ---------------------------------------------------------------------------

/// The shared prime list and factorial exponent rows.
///
/// Invariant: `fact` holds rows `0..fact.len()` and `primes` holds every prime
/// up to `fact.len() - 1`, so every row's exponent vector indexes `primes`.
pub(crate) struct Tables {
    primes: Vec<u64>,
    fact: Vec<Vec<u32>>,
}

// One lock taken once per evaluation. Reading through it per prime or per
// factorial row put a lock round trip on every element of the hot loop; the
// reference reads a lock-free append-only list, which here would need new
// `unsafe` or a new dependency.
static TABLES: RwLock<Tables> = RwLock::new(Tables {
    primes: Vec::new(),
    fact: Vec::new(),
});

#[cfg(debug_assertions)]
thread_local! {
    static HELD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks this thread as holding [`TABLES`] for the duration of one evaluation.
/// A nested acquisition could deadlock behind a queued writer, so debug builds
/// reject it.
struct Held;

impl Held {
    fn enter() -> Self {
        #[cfg(debug_assertions)]
        HELD.with(|h| assert!(!h.replace(true), "nested with_tables"));
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        HELD.with(|h| h.set(false));
    }
}

/// Run `f` with tables holding every factorial row `0..=n` (and so every prime
/// `<= n`). Growth takes the write lock once; the evaluation itself takes the
/// read lock once. `f` must not call `with_tables` again.
pub(crate) fn with_tables<R>(n: u64, f: impl FnOnce(&Tables) -> R) -> R {
    let _held = Held::enter();
    let n = usize::try_from(n).expect("factorial table index fits usize");
    {
        let t = TABLES.read().unwrap();
        if n < t.fact.len() {
            return f(&t);
        }
    }
    TABLES.write().unwrap().grow(n);
    // The tables never shrink, so they still cover `n`.
    f(&TABLES.read().unwrap())
}

impl Tables {
    /// The `idx`-th prime (`idx == 0` -> 2) (`primefactorization.jl::prime`).
    fn prime(&self, idx: usize) -> u64 {
        self.primes[idx]
    }

    /// Extend so row `n` exists (`primefactorization.jl::primefactorial`,
    /// growth half). Built incrementally:
    /// `factorial(m) = factorial(m-1) * primefactor(m)` in exponent space.
    fn grow(&mut self, n: usize) {
        if self.fact.is_empty() {
            self.fact.push(Vec::new()); // 0! = 1
        }
        while self.fact.len() <= n {
            let m = self.fact.len() as u64;
            let fm = primefactor_powers(&mut self.primes, m);
            let mut next = self.fact[self.fact.len() - 1].clone();
            if fm.len() > next.len() {
                next.resize(fm.len(), 0);
            }
            for (a, b) in next.iter_mut().zip(fm.iter()) {
                *a += *b;
            }
            self.fact.push(next);
        }
    }

    /// Multiply `acc` by `n!` in place, folding the memoized exponent row into
    /// `acc` with no per-call allocation or row clone.
    pub(crate) fn mul_factorial(&self, acc: &mut Pf, n: u64) {
        acc.mul_by_powers(&self.fact[n as usize]);
    }

    /// `n!` as a fresh [`Pf`].
    pub(crate) fn factorial(&self, n: u64) -> Pf {
        let mut p = Pf::one();
        self.mul_factorial(&mut p, n);
        p
    }
}

/// Append the next prime after the largest in `primes`.
fn push_next_prime(primes: &mut Vec<u64>) {
    let Some(&last) = primes.last() else {
        primes.push(2);
        return;
    };
    // Trial-divide by the already-known primes: all primes up to
    // sqrt(candidate) are present because the candidate starts just above the
    // current largest prime.
    let mut cand = last + 1;
    while !is_prime_by(cand, primes) {
        cand += 1;
    }
    primes.push(cand);
}

fn is_prime_by(n: u64, primes: &[u64]) -> bool {
    for &p in primes {
        if p * p > n {
            break;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }
    true
}

/// Exponent vector of the prime factorization of `n >= 1`, extending `primes`
/// as needed (`primefactorization.jl::primefactor`).
fn primefactor_powers(primes: &mut Vec<u64>, mut n: u64) -> Vec<u32> {
    let mut powers = Vec::new();
    let mut idx = 0;
    while n > 1 {
        if idx == primes.len() {
            push_next_prime(primes);
        }
        let p = primes[idx];
        let mut e = 0u32;
        while n.is_multiple_of(p) {
            n /= p;
            e += 1;
        }
        powers.push(e);
        idx += 1;
    }
    trim(&mut powers);
    powers
}

// ---------------------------------------------------------------------------
// Series over a common denominator (compute3jseries / compute6jseries core).
// ---------------------------------------------------------------------------

/// Sum a list of exact fractions `nums[i] / dens[i]`, each already prime
/// factorized, into one reduced [`malachite_q::Rational`]
/// (`primefactorization.jl::commondenominator!` + `sumlist!`,
/// `WignerSymbols.jl::compute{3,6}jseries`).
///
/// Puts every term over the lcm of the denominators, factors the gcd out of the
/// numerators before reconstruction, sums the small remainders as big integers,
/// then multiplies the common factor back. Big-integer work is confined here.
pub(crate) fn sum_series(t: &Tables, mut terms: Vec<(Pf, Pf)>) -> malachite_q::Rational {
    use malachite_q::Rational;
    if terms.is_empty() {
        return Rational::ZERO;
    }

    // Common denominator = lcm of all denominators; rescale each numerator.
    let mut den = terms[0].1.clone();
    for (_, d) in &terms[1..] {
        den.lcm_assign(d);
    }
    for (num, d) in &mut terms {
        num.mul_assign(&den);
        num.divexact_assign(d);
    }

    // Pull the gcd of all numerators out before reconstructing big integers, so
    // the summed integers stay as small as the series allows (`sumlist!`).
    let mut g = terms[0].0.clone();
    for (num, _) in &terms[1..] {
        g.gcd_assign(num);
    }
    let mut total = Integer::ZERO;
    for (num, _) in &mut terms {
        if !g.is_zero() {
            num.divexact_assign(&g);
        }
        total += num.to_bigint(t);
    }
    total *= g.to_bigint(t);

    Rational::from_integers(total, den.to_bigint(t))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct big-integer factorial, independent of the prime-factorized path.
    fn direct_factorial(n: u64) -> Integer {
        let mut f = Integer::ONE;
        for k in 2..=n {
            f *= Integer::from(k);
        }
        f
    }

    #[test]
    fn nth_prime_matches_known() {
        let known = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];
        with_tables(47, |t| {
            for (i, &p) in known.iter().enumerate() {
                assert_eq!(t.prime(i), p, "prime index {i}");
            }
        });
    }

    #[test]
    fn primefactor_reconstructs() {
        with_tables(200, |t| {
            for n in 1u64..=200 {
                let mut primes = Vec::new();
                let pf = Pf::from_powers(primefactor_powers(&mut primes, n));
                assert_eq!(pf.to_bigint(t), Integer::from(n), "factoring {n}");
            }
        });
    }

    #[test]
    fn factorial_exponents_vs_direct() {
        with_tables(30, |t| {
            for n in 0u64..=30 {
                assert_eq!(
                    t.factorial(n).to_bigint(t),
                    direct_factorial(n),
                    "factorial({n})"
                );
            }
        });
    }

    #[test]
    fn factorial_large_still_exact() {
        with_tables(257, |t| {
            assert_eq!(t.factorial(100).to_bigint(t), direct_factorial(100));
            assert_eq!(t.factorial(257).to_bigint(t), direct_factorial(257));
        });
    }

    #[test]
    fn mul_factorial_accumulates_clone_free() {
        with_tables(7, |t| {
            let mut acc = Pf::one();
            t.mul_factorial(&mut acc, 3);
            t.mul_factorial(&mut acc, 5);
            t.mul_factorial(&mut acc, 7);
            assert_eq!(
                acc.to_bigint(t),
                direct_factorial(3) * direct_factorial(5) * direct_factorial(7)
            );
        });
    }

    #[test]
    fn mul_and_divexact_roundtrip() {
        with_tables(12, |t| {
            let a = t.factorial(12);
            let b = t.factorial(7);
            let mut p = a.clone();
            p.mul_assign(&b);
            assert_eq!(
                p.to_bigint(t),
                direct_factorial(12) * direct_factorial(7),
                "12! * 7!"
            );
            p.divexact_assign(&b);
            assert_eq!(p.to_bigint(t), a.to_bigint(t), "(12! * 7!) / 7! == 12!");
        });
    }

    #[test]
    fn divgcd_reduces_to_coprime() {
        with_tables(9, |t| {
            let mut a = t.factorial(9);
            let mut b = t.factorial(6);
            Pf::divgcd(&mut a, &mut b);
            // gcd(9!,6!) = 6!, so a = 9!/6! = 7*8*9 = 504, b = 1.
            assert_eq!(a.to_bigint(t), Integer::from(504));
            assert_eq!(b.to_bigint(t), Integer::ONE);
        });
    }

    #[test]
    fn splitsquare_roundtrips() {
        with_tables(120, |t| {
            for n in 1u64..=120 {
                let mut primes = Vec::new();
                let a = Pf::from_powers(primefactor_powers(&mut primes, n));
                let (s, r) = a.splitsquare();
                let recon = s.to_bigint(t).pow(2) * r.to_bigint(t);
                assert_eq!(recon, Integer::from(n), "splitsquare {n}");
            }
            let a = t.factorial(20);
            let (s, r) = a.splitsquare();
            assert_eq!(s.to_bigint(t).pow(2) * r.to_bigint(t), direct_factorial(20));
        });
    }

    #[test]
    fn sign_propagates() {
        with_tables(5, |t| {
            let mut a = t.factorial(5).neg();
            assert_eq!(a.to_bigint(t), -direct_factorial(5));
            let b = t.factorial(3).neg();
            a.mul_assign(&b); // (-5!) * (-3!) = +5!*3!
            assert_eq!(a.to_bigint(t), direct_factorial(5) * direct_factorial(3));
        });
    }

    #[test]
    fn sum_series_matches_direct_rational() {
        use malachite_q::Rational;
        // Sum 1/3! - 1/(2!*4!) + 3!/(5!) as prime-factorized terms.
        let got = with_tables(5, |t| {
            let mut d2 = t.factorial(2);
            d2.mul_assign(&t.factorial(4));
            let terms = vec![
                (Pf::one(), t.factorial(3)),
                (Pf::one().neg(), d2),
                (t.factorial(3), t.factorial(5)),
            ];
            sum_series(t, terms)
        });
        let want = Rational::from_integers(Integer::ONE, Integer::from(6))
            - Rational::from_integers(Integer::ONE, Integer::from(48))
            + Rational::from_integers(Integer::from(6), Integer::from(120));
        assert_eq!(got, want);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "nested with_tables")]
    fn nested_with_tables_panics() {
        with_tables(3, |_| with_tables(3, |_| ()));
    }
}
