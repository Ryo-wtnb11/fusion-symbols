//! Prime-factorized integer arithmetic for the Racah/Wigner factorial engine.
//!
//! Ported from WignerSymbols.jl v2.0.0 `src/primefactorization.jl`
//! (`PrimeFactorization`, `primefactor`, `primefactorial`, `mul!`,
//! `divexact!`, `gcd!`, `lcm!`, `divgcd!`, `splitsquare`,
//! `commondenominator!`, `sumlist!`). A positive integer is stored as the
//! exponent vector of its prime factorization; products and exact quotients
//! are integer vector add/sub, so no big integer is multiplied until a single
//! final reconstruction. Factorials and their ratios (the whole cost of a
//! 3j/6j evaluation) stay in exponent space.
//!
//! Reconstruction multiplies prime powers into one machine word and folds the
//! word into the big integer only when the next factor would overflow it, so a
//! term costs about one limb pass per 64 bits instead of one big-integer
//! multiply per prime. A series keeps every term in fixed-width rows of two
//! buffers and reconstructs into one reused limb buffer, so a term allocates
//! nothing.
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

use malachite_base::num::basic::traits::Zero;
use malachite_nz::{
    integer::Integer,
    natural::Natural,
    platform::{DoubleLimb, Limb},
};
use malachite_q::Rational;
use std::sync::RwLock;

/// A positive integer stored as `prod_i prime(i)^powers[i]`.
///
/// `powers[i]` is the exponent of the `i`-th prime (`powers[0]` -> 2). The
/// vector is normalized so its last entry is nonzero (an empty vector is the
/// integer 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Pf {
    powers: Vec<u32>,
}

impl Pf {
    /// The integer 1.
    pub(crate) fn one() -> Self {
        Pf { powers: Vec::new() }
    }

    /// Wrap a normalized (trailing-zero-trimmed) exponent vector.
    fn from_powers(mut powers: Vec<u32>) -> Self {
        trim(&mut powers);
        Pf { powers }
    }

    /// `self *= other` — exponent vectors add (`primefactorization.jl::mul!`).
    pub(crate) fn mul_assign(&mut self, other: &Pf) {
        self.mul_by_powers(&other.powers);
    }

    /// `self *= prod_i prime(i)^powers[i]`, folding a raw exponent slice (a
    /// factorial table row) straight into `self` without allocating a [`Pf`]
    /// for it.
    fn mul_by_powers(&mut self, powers: &[u32]) {
        if powers.len() > self.powers.len() {
            self.powers.resize(powers.len(), 0);
        }
        add_rows(&mut self.powers, powers);
    }

    /// Split off the common gcd of `a` and `b` in place, so afterwards
    /// `gcd(a, b) == 1` (`primefactorization.jl::divgcd!`). Keeps the two
    /// reconstructed big integers as small as possible.
    pub(crate) fn divgcd(a: &mut Pf, b: &mut Pf) {
        divgcd_rows(&mut a.powers, &mut b.powers);
        trim(&mut a.powers);
        trim(&mut b.powers);
    }

    /// Split `a = s^2 * r` with `r` square-free (all exponents 0 or 1)
    /// (`primefactorization.jl::splitsquare`).
    pub(crate) fn splitsquare(&self) -> (Pf, Pf) {
        let s = Pf::from_powers(self.powers.iter().map(|&e| e >> 1).collect());
        let r = Pf::from_powers(self.powers.iter().map(|&e| e & 1).collect());
        (s, r)
    }

    /// Reconstruct the big integer (`primefactorization.jl::_convert!`).
    pub(crate) fn to_bigint(&self, t: &Tables) -> Integer {
        let mut limbs = Vec::new();
        let w = t.mul_out(&self.powers, &mut limbs);
        if limbs.is_empty() {
            return Integer::from(w);
        }
        mul_limb(&mut limbs, w);
        Integer::from(Natural::from_owned_limbs_asc(limbs))
    }
}

/// `a[i] += b[i]` over `b`; `a` must be at least as long.
fn add_rows(a: &mut [u32], b: &[u32]) {
    for (x, y) in a[..b.len()].iter_mut().zip(b) {
        *x += *y;
    }
}

/// Remove the exponent-wise minimum of `a` and `b` from both.
fn divgcd_rows(a: &mut [u32], b: &mut [u32]) {
    for (x, y) in a.iter_mut().zip(b.iter_mut()) {
        let g = (*x).min(*y);
        *x -= g;
        *y -= g;
    }
}

/// `xs *= w` in place on little-endian limbs.
fn mul_limb(xs: &mut Vec<Limb>, w: Limb) {
    let mut carry: Limb = 0;
    for x in xs.iter_mut() {
        let t = DoubleLimb::from(*x) * DoubleLimb::from(w) + DoubleLimb::from(carry);
        *x = t as Limb;
        carry = (t >> Limb::BITS) as Limb;
    }
    if carry != 0 {
        xs.push(carry);
    }
}

/// `acc += xs` in place on little-endian limbs.
fn add_limbs(acc: &mut Vec<Limb>, xs: &[Limb]) {
    if acc.len() < xs.len() {
        acc.resize(xs.len(), 0);
    }
    let mut carry = false;
    for (i, a) in acc.iter_mut().enumerate() {
        let Some(&x) = xs.get(i) else {
            if !carry {
                return;
            }
            (*a, carry) = a.overflowing_add(1);
            continue;
        };
        let (s, c1) = a.overflowing_add(x);
        let (s, c2) = s.overflowing_add(Limb::from(carry));
        *a = s;
        carry = c1 || c2;
    }
    if carry {
        acc.push(1);
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
/// `word_pow[i]` is `(p, k, p^k)` for `p = primes[i]` and the largest `k` with
/// `p^k <= Limb::MAX`, with one entry per prime.
pub(crate) struct Tables {
    primes: Vec<u64>,
    word_pow: Vec<(Limb, u32, Limb)>,
    fact: Vec<Vec<u32>>,
}

// One lock taken once per evaluation. Reading through it per prime or per
// factorial row put a lock round trip on every element of the hot loop; the
// reference reads a lock-free append-only list, which here would need new
// `unsafe` or a new dependency.
static TABLES: RwLock<Tables> = RwLock::new(Tables {
    primes: Vec::new(),
    word_pow: Vec::new(),
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
        for &p in &self.primes[self.word_pow.len()..] {
            let p = Limb::try_from(p).expect("a tabulated prime fits a limb");
            let (mut k, mut pk) = (1, p);
            while let Some(next) = pk.checked_mul(p) {
                (k, pk) = (k + 1, next);
            }
            self.word_pow.push((p, k, pk));
        }
    }

    /// `prod_i prime(i)^powers[i]` as `out * w`: the little-endian limbs left
    /// in `out` (empty means 1) times the returned word `w`.
    ///
    /// Prime powers are multiplied into one machine word; the word is folded
    /// into `out` only when the next factor would overflow it. Every factor is
    /// at most `word_pow[i].2`, so no word product is ever unchecked. A product
    /// that fits one word leaves `out` empty and so never allocates.
    fn mul_out(&self, powers: &[u32], out: &mut Vec<Limb>) -> Limb {
        out.clear();
        let mut w: Limb = 1;
        let mut fold = |x: Limb| match w.checked_mul(x) {
            Some(y) => w = y,
            None => {
                if out.is_empty() {
                    out.push(w);
                } else {
                    mul_limb(out, w);
                }
                w = x;
            }
        };
        for (i, &e) in powers.iter().enumerate() {
            if e == 0 {
                continue;
            }
            let (p, k, pk) = self.word_pow[i];
            for _ in 0..e / k {
                fold(pk);
            }
            if e % k != 0 {
                // p^(e mod k) < p^k <= Limb::MAX.
                fold(p.pow(e % k));
            }
        }
        w
    }

    /// An empty Racah series whose factorial arguments are all `<= n`, with
    /// room for `terms` terms.
    pub(crate) fn series(&self, n: u64, terms: usize) -> Series<'_> {
        // Rows of 0! and 1! are empty; one zero column keeps `chunks_exact` valid.
        let width = self.fact[usize::try_from(n).expect("factorial table index fits usize")]
            .len()
            .max(1);
        Series {
            t: self,
            width,
            nums: Vec::with_capacity(terms * width),
            dens: Vec::with_capacity(terms * width),
            negative: Vec::with_capacity(terms),
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

/// A sum of exact fractions `(-1)^negative[i] * nums[i] / dens[i]`, each a
/// product of factorials in exponent space
/// (`primefactorization.jl::commondenominator!` + `sumlist!`,
/// `WignerSymbols.jl::compute{3,6}jseries`).
///
/// Every term's exponents live in one fixed-width row of `nums` and `dens`
/// (`width` = number of primes up to the largest factorial argument), so a
/// term costs no allocation, and [`Series::sum`] reconstructs every term into
/// one reused limb buffer.
pub(crate) struct Series<'t> {
    t: &'t Tables,
    width: usize,
    nums: Vec<u32>,
    dens: Vec<u32>,
    negative: Vec<bool>,
}

impl Series<'_> {
    /// Append `(-1)^negative * prod_a num[a]! / prod_b den[b]!`, reduced by
    /// its own gcd (`divgcd!`). Every argument must be `<=` the series bound;
    /// a longer row panics on the slice index.
    pub(crate) fn push(&mut self, negative: bool, num: &[u64], den: &[u64]) {
        let w = self.width;
        let start = self.nums.len();
        self.nums.resize(start + w, 0);
        self.dens.resize(start + w, 0);
        let (nrow, drow) = (&mut self.nums[start..], &mut self.dens[start..]);
        for &n in num {
            add_rows(nrow, &self.t.fact[n as usize]);
        }
        for &n in den {
            add_rows(drow, &self.t.fact[n as usize]);
        }
        divgcd_rows(nrow, drow);
        self.negative.push(negative);
    }

    /// The exact sum as one reduced rational.
    ///
    /// Puts every term over the lcm of the denominators, factors the gcd out of
    /// the numerators before reconstruction, sums the small remainders, then
    /// multiplies the common factor back.
    pub(crate) fn sum(mut self) -> Rational {
        let w = self.width;
        if self.negative.is_empty() {
            return Rational::ZERO;
        }

        // Common denominator = lcm of all denominators; rescale each numerator.
        let mut den = self.dens[..w].to_vec();
        for d in self.dens.chunks_exact(w).skip(1) {
            for (a, &b) in den.iter_mut().zip(d) {
                *a = (*a).max(b);
            }
        }
        for (num, d) in self.nums.chunks_exact_mut(w).zip(self.dens.chunks_exact(w)) {
            for ((a, &l), &b) in num.iter_mut().zip(&den).zip(d) {
                *a += l - b;
            }
        }

        // Pull the gcd of all numerators out before reconstructing big integers,
        // so the summed integers stay as small as the series allows (`sumlist!`).
        let mut g = self.nums[..w].to_vec();
        for num in self.nums.chunks_exact(w).skip(1) {
            for (a, &b) in g.iter_mut().zip(num) {
                *a = (*a).min(b);
            }
        }
        let (mut term, mut pos, mut neg) = (Vec::new(), Vec::new(), Vec::new());
        for (num, &negative) in self.nums.chunks_exact_mut(w).zip(&self.negative) {
            for (a, &b) in num.iter_mut().zip(&g) {
                *a -= b;
            }
            let w = self.t.mul_out(num, &mut term);
            let acc = if negative { &mut neg } else { &mut pos };
            if term.is_empty() {
                add_limbs(acc, &[w]);
            } else {
                mul_limb(&mut term, w);
                add_limbs(acc, &term);
            }
        }
        let mut total = Integer::from(Natural::from_owned_limbs_asc(pos))
            - Integer::from(Natural::from_owned_limbs_asc(neg));
        total *= Pf::from_powers(g).to_bigint(self.t);

        Rational::from_integers(total, Pf::from_powers(den).to_bigint(self.t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use malachite_base::num::{arithmetic::traits::Pow, basic::traits::One};

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
                assert_eq!(t.primes[i], p, "prime index {i}");
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
    fn series_sum_matches_direct_rational() {
        use malachite_q::Rational;
        // Sum 1/3! - 1/(2!*4!) + 3!/(5!) as prime-factorized terms.
        let got = with_tables(5, |t| {
            let mut s = t.series(5, 3);
            s.push(false, &[], &[3]);
            s.push(true, &[], &[2, 4]);
            s.push(false, &[3], &[5]);
            s.sum()
        });
        let want = Rational::from_integers(Integer::ONE, Integer::from(6))
            - Rational::from_integers(Integer::ONE, Integer::from(48))
            + Rational::from_integers(Integer::from(6), Integer::from(120));
        assert_eq!(got, want);
    }

    #[test]
    fn add_limbs_carries_past_the_shorter_operand() {
        let mut acc: Vec<Limb> = vec![Limb::MAX, Limb::MAX];
        add_limbs(&mut acc, &[1]);
        assert_eq!(acc, [0, 0, 1]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "nested with_tables")]
    fn nested_with_tables_panics() {
        with_tables(3, |_| with_tables(3, |_| ()));
    }
}
