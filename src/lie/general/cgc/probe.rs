//! Measurement probe for exact F and R (`exact-fr-probe` feature, not a
//! public API): every block entry is evaluated as one exact signed square
//! root of a rational and rounded once. Production values are untouched.
//!
//! With the CGC entry `C = x·√(n^a n^b / (N_μ n^c))` (§4.7), the four norms
//! of each shared leg cancel and
//!
//! ```text
//! F[μ,ν,κ,λ] = S / (n^d_0 √(N_A,μ N_B,ν N_C,κ N_D,λ)),
//! S = Σ_{m_a,m_b,m_c} n^a n^b n^c (Σ_{m_e} x_A x_B)(Σ_{m_f} x_C x_D),
//! R[μ,ν]     = T / (n^c_0 √(N_μ N'_ν)),   T = Σ_{m_a,m_b} n^a n^b x_A x_B.
//! ```
//!
//! An exact intertwiner gives the same value on every `d` (resp. `c`) state,
//! so `B`, `D` and both R channels are read at engine index 0, the highest
//! weight, for every basis: they are the elimination seeds of §4.2–§4.3 and
//! need no descent. Each `(channel, μ)` and each module's norms are put on
//! one common denominator and their content divided out, so `S` and `T` are
//! integer multiply–accumulates, and every denominator and content enters
//! only the final radicand `U / V`, which
//! `exact::signed_sqrt_to_f64` rounds once. Product groups take the exact
//! Kronecker product of the factor radicands (factor order, row-major, as
//! `product.rs`) and round once.
//!
//! The F final sum `S[μν, κλ] = Σ_key ν_a ν_b ν_c AB[key][μν] CD[key][κλ]`
//! reads each partial join once per key; its cost is `K · n1 n2 n3 n4`
//! multiply–accumulates for `K` keys. [`Accum::Auto`] runs it in
//! sign–magnitude stack limbs (`L ∈ {1, 4, 8, 16}`, chosen before the loop
//! from the widest operand) into per-entry lazy-carry `u128` columns, with
//! no allocation or gcd per term, and falls back to `Q` above 1024 bits.
//!
//! The tolerance functions bound `|current − exact|` for the production
//! binary64 evaluation of §6.2 from its own rounded channels:
//! `γ_k · Σ|terms|` with `k = 2(n + 7)` for F and `k = 2(n + 3)` for R
//! (`n` summed terms per entry; four, resp. two, correctly rounded CGC and
//! three, resp. one, rounded products per term; the factor 2 absorbs the
//! rounding of the absolute sum itself).

use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Range;
use std::time::Instant;

use malachite_base::num::arithmetic::traits::{Gcd, Lcm, Square};
use malachite_base::num::logic::traits::SignificantBits;
use malachite_nz::natural::Natural;
use malachite_nz::platform::Limb;
use malachite_q::Rational;

use super::{descend, highest, GeneralFamily, Ladders, ProductScratch, RequestModules, RowScratch};
use super::{CartanData, Seeds, Q};
use crate::cache::Label;
use crate::category::SymbolBlock;
use crate::lie::{su2, LieError, LieGroup, LieIrrep, LiePayload};

/// Unit roundoff of binary64.
const U: f64 = f64::EPSILON / 2.0;

/// Counters of one exact evaluation. Bit lengths are of nonzero entries.
#[derive(Clone, Debug, Default)]
pub struct ProbeStats {
    /// Modules, exact channels (descent of `A`, `Cb`; seeds of `B`, `D`),
    /// and common denominators.
    pub gen_secs: f64,
    /// Joins, integer multiply–accumulate, radicand formation, rounding.
    pub contract_secs: f64,
    /// Of `contract_secs`: the partial joins `Σ_me X_A X_B`, `Σ_mf X_C X_D`.
    pub join_secs: f64,
    /// Of `contract_secs`: the final sum over keys.
    pub sum_secs: f64,
    /// Accumulator of each simple block's final sum (`words`, `limbs4`,
    /// `limbs8`, `limbs16`, `big`, `q` for R; `=big` when checked), joined
    /// by `+` over product factors.
    pub accum: String,
    /// Bit length of each exact integer numerator `S` (resp. `T`); for a
    /// product group, of each factor's numerators.
    pub s_bits: Vec<u64>,
    /// Bit length of each radicand numerator `U` passed to rounding.
    pub u_bits: Vec<u64>,
    /// Bit length of each radicand denominator `V` passed to rounding.
    pub v_bits: Vec<u64>,
    /// Of `gen_secs`: capturing exact coordinates inside the descents.
    pub capture_secs: f64,
    /// Of `gen_secs`: common denominators, contents and module norms.
    pub prep_secs: f64,
    /// Exact CGC coordinates `x` read.
    pub x_values: u64,
    /// Of those, values outside the inline word representation.
    pub x_big: u64,
    /// Integer accumulators `S` (resp. `T`).
    pub s_values: u64,
    /// Of those, values outside the inline word representation.
    pub s_big: u64,
}

/// An exact block rounded once, row-major in the production axis order.
#[derive(Clone, Debug)]
pub struct ProbeBlock {
    /// Axis lengths.
    pub dims: Vec<usize>,
    /// The correctly rounded exact entries.
    pub values: Vec<f64>,
    /// Counters and timings.
    pub stats: ProbeStats,
}

/// How the final sum `S` of an F block is accumulated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accum {
    /// Word joins and fixed-width stack limbs when their bounds hold, else
    /// the `Q` joins with stack limbs, else `Q` throughout.
    Auto,
    /// The `Q` hash-map joins and a `Q` final sum.
    Big,
    /// Both; panics unless every entry's `S` (hence `(sign, U, V)`) agrees.
    Check,
}

/// `sign·√(u/v)`, unreduced; zero has sign 0.
#[derive(Clone, Debug, PartialEq)]
struct Exact {
    sign: i8,
    u: Natural,
    v: Natural,
}

impl Exact {
    fn zero() -> Self {
        Self {
            sign: 0,
            u: Natural::from(0u32),
            v: Natural::from(1u32),
        }
    }

    fn times(&self, other: &Self) -> Self {
        if self.sign == 0 || other.sign == 0 {
            return Self::zero();
        }
        Self {
            sign: self.sign * other.sign,
            u: &self.u * &other.u,
            v: &self.v * &other.v,
        }
    }
}

struct Block {
    dims: Vec<usize>,
    entries: Vec<Exact>,
}

impl Block {
    fn zeros(dims: Vec<usize>) -> Self {
        let len = dims.iter().product();
        Self {
            dims,
            entries: vec![Exact::zero(); len],
        }
    }
}

/// Exact entries grouped by one engine index → `(index, multiplicity, X)`.
type ByIndex<'q> = HashMap<usize, Vec<(usize, usize, &'q Q)>>;
/// Partial sums keyed by `(m_a, m_b, m_c)`, dense over two multiplicities.
type Keyed = HashMap<(usize, usize, usize), Vec<Q>>;
/// R seeds keyed by `(m_a, m_b)` → `(ν, X)`.
type ByPair<'q> = HashMap<(usize, usize), Vec<(usize, &'q Q)>>;
/// Rounded `|C|` grouped by one public index → `(index, multiplicity, |C|)`.
type Grouped = HashMap<u32, Vec<(u32, u32, f64)>>;
/// Rounded absolute partial products keyed by `(m_a, m_b, m_c)`.
type Joined = HashMap<(u32, u32, u32), Vec<(u32, u32, f64)>>;
/// Rounded `|C|` of R keyed by `(m_a, m_b)` → `(ν, |C|)`.
type PairAbs = HashMap<(u32, u32), Vec<(u32, f64)>>;

/// One exact entry `(left engine, right engine, coupled engine, μ, value)`.
type Raw = (usize, usize, usize, usize, Q);

/// An exact channel on per-`μ` common denominators with the content divided
/// out: `x = content_μ · X / D_μ` with `X` an integer.
struct IntChannel {
    entries: Vec<Raw>,
    den: Vec<Natural>,
    content: Vec<Natural>,
    norms: Vec<Q>,
}

impl IntChannel {
    /// Per `μ`: the radicand factors `q(N_μ)·content_μ²` (numerator side)
    /// and `D_μ²·p(N_μ)` (denominator side).
    fn radicand_parts(&self) -> (Vec<Natural>, Vec<Natural>) {
        self.norms
            .iter()
            .zip(&self.den)
            .zip(&self.content)
            .map(|((n, d), g)| {
                let (p, q) = n.magnitudes();
                (q.into_owned() * g.square(), d.square() * p.as_ref())
            })
            .unzip()
    }
}

fn one() -> Natural {
    Natural::from(1u32)
}

fn to_q(n: &Natural) -> Q {
    Q::from(Rational::from(n.clone()))
}

fn bits(q: &Q) -> u64 {
    q.magnitudes().0.significant_bits()
}

fn max_bits(v: &[Q]) -> u64 {
    v.iter().map(bits).max().unwrap_or(0)
}

/// Progress of the exact path on stderr, so a long run shows its phase.
fn progress(phase: &str, since: Instant) {
    eprintln!(
        "exact-fr-probe: {phase} done at {:.3}s",
        since.elapsed().as_secs_f64()
    );
}

/// The content (gcd) of the integers `values`; 1 when there are none.
fn divide_content<'v>(values: impl Iterator<Item = &'v Q>) -> Natural {
    let mut g = Natural::from(0u32);
    for x in values {
        if g == one() {
            break;
        }
        g = g.gcd(x.magnitudes().0.into_owned());
    }
    if g == Natural::from(0u32) {
        one()
    } else {
        g
    }
}

fn integral(raw: Vec<Raw>, norms: Vec<Q>, st: &mut ProbeStats) -> IntChannel {
    let t = Instant::now();
    let n = norms.len();
    let mut den = vec![one(); n];
    for (.., mu, x) in &raw {
        st.x_values += 1;
        st.x_big += u64::from(x.heap_bytes() != 0);
        let (_, q) = x.magnitudes();
        if *q != one() {
            let d = std::mem::replace(&mut den[*mu], one());
            den[*mu] = d.lcm(q.into_owned());
        }
    }
    let scale: Vec<Q> = den.iter().map(to_q).collect();
    let scaled: Vec<Raw> = raw
        .into_iter()
        .map(|(p, q, m, mu, x)| (p, q, m, mu, x * &scale[mu]))
        .collect();
    let content: Vec<Natural> = (0..n)
        .map(|mu| divide_content(scaled.iter().filter(|e| e.3 == mu).map(|e| &e.4)))
        .collect();
    let div: Vec<Q> = content.iter().map(to_q).collect();
    let entries = scaled
        .into_iter()
        .map(|(p, q, m, mu, x)| {
            let x = if div[mu].is_one() { x } else { x / &div[mu] };
            (p, q, m, mu, x)
        })
        .collect();
    st.prep_secs += t.elapsed().as_secs_f64();
    IntChannel {
        entries,
        den,
        content,
        norms,
    }
}

/// `n_m = g ν_m / Δ` with `Δ` the lcm of every norm denominator of `m`, `g`
/// the content of `n_m Δ`, and `ν_m` an integer; returns `(ν, Δ, g)`.
fn integral_norms(m: &Ladders) -> (Vec<Q>, Natural, Natural) {
    let mut den = one();
    for n in &m.norm {
        let (_, q) = n.magnitudes();
        if *q != one() {
            den = den.lcm(q.into_owned());
        }
    }
    let scale = to_q(&den);
    let nu: Vec<Q> = m.norm.iter().map(|n| n * &scale).collect();
    let g = divide_content(nu.iter());
    let div = to_q(&g);
    let nu = if div.is_one() {
        nu
    } else {
        nu.into_iter().map(|x| x / &div).collect()
    };
    (nu, den, g)
}

/// The full exact channel `a ⊗ b → c` by the production descent, keeping
/// the entries `(m_a, m_b, m_c)` that `keep` accepts; the capture time is
/// added to `capture`.
fn full(
    data: &CartanData,
    (a, b, c): (&Ladders, &Ladders, &Ladders),
    n: usize,
    keep: &dyn Fn(usize, usize, usize) -> bool,
    capture: &mut f64,
) -> Result<(Vec<Raw>, Vec<Q>), LieError> {
    let mut raw = Vec::new();
    let mut norms = Vec::new();
    let (start, dim) = (Instant::now(), c.dim());
    let mut done = 0usize;
    descend(data, a, b, c, n, &mut |seed_norms, m, space, per_mu| {
        if norms.is_empty() {
            norms = seed_norms.to_vec();
        }
        done += 1;
        if done % (dim / 10).max(1) == 0 {
            eprintln!(
                "exact-fr-probe:   descent {done}/{dim} states at {:.3}s",
                start.elapsed().as_secs_f64()
            );
        }
        let t = Instant::now();
        for (mu, v) in per_mu.iter().enumerate() {
            for (&(pa, pb), x) in space.pairs.iter().zip(v) {
                if !x.is_zero() && keep(pa, pb, m) {
                    raw.push((pa, pb, m, mu, x.clone()));
                }
            }
        }
        *capture += t.elapsed().as_secs_f64();
        Ok(())
    })?;
    Ok((raw, norms))
}

/// The highest-weight column (engine `m_c = 0`) of `a ⊗ b → c`: elimination
/// and Gram–Schmidt only (§4.2–§4.3).
fn seeds(
    data: &CartanData,
    (a, b, c): (&Ladders, &Ladders, &Ladders),
    n: usize,
) -> Result<(Vec<Raw>, Vec<Q>), LieError> {
    let (_, seeded) = highest(
        data,
        a,
        b,
        c,
        n,
        &mut RowScratch::new(a),
        &mut ProductScratch::new(a, b),
    )?;
    let Some(Seeds { p, seeds, norms }) = seeded else {
        return Ok((Vec::new(), Vec::new()));
    };
    let mut raw = Vec::new();
    for (mu, v) in seeds.into_iter().enumerate() {
        for (&(pa, pb), x) in p.pairs.iter().zip(v) {
            if !x.is_zero() {
                raw.push((pa, pb, 0, mu, x));
            }
        }
    }
    Ok((raw, norms))
}

fn sign_of(q: &Q) -> i8 {
    match q.signum() {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

/// An exact integer as `(sign, |value|)`.
type Signed = (i8, Natural);

fn signed(q: &Q) -> Signed {
    (sign_of(q), q.magnitudes().0.into_owned())
}

/// `sign·√(|s|²·u_rest / v_rest)`.
fn radical((sign, mag): Signed, u_rest: &Natural, v_rest: Natural, st: &mut ProbeStats) -> Exact {
    st.s_values += 1;
    if sign == 0 {
        return Exact::zero();
    }
    let b = mag.significant_bits();
    st.s_bits.push(b);
    st.s_big += u64::from(b > 63);
    Exact {
        sign,
        u: mag.square() * u_rest,
        v: v_rest,
    }
}

/// A signed integer of at most `L` little-endian limbs, as sign and
/// magnitude; `len` counts limbs up to the highest nonzero one.
#[derive(Clone, Copy)]
struct Mag<const L: usize> {
    neg: bool,
    len: usize,
    limbs: [u64; L],
}

impl<const L: usize> Mag<L> {
    /// The integer `q`, which the caller's bit bound fits into `L` limbs.
    fn from_q(q: &Q) -> Self {
        let (p, _) = q.magnitudes();
        let mut limbs = [0u64; L];
        for (i, w) in p.limbs().enumerate() {
            let bit = i * Limb::BITS as usize;
            // A Malachite limb is 64 bits, or 32 under `32_bit_limbs`.
            #[allow(clippy::useless_conversion)]
            let w = u64::from(w);
            limbs[bit / 64] |= w << (bit % 64);
        }
        Self {
            neg: q.signum() == Ordering::Less,
            len: p.significant_bits().div_ceil(64) as usize,
            limbs,
        }
    }

    /// The word sum `v`, which the caller's bit bound fits into `L` limbs.
    fn from_i128(v: i128) -> Self {
        let m = v.unsigned_abs();
        let (lo, hi) = (m as u64, (m >> 64) as u64);
        let mut limbs = [0u64; L];
        limbs[0] = lo;
        if hi != 0 {
            limbs[1] = hi;
        }
        Self {
            neg: v < 0,
            len: if hi != 0 { 2 } else { usize::from(lo != 0) },
            limbs,
        }
    }

    /// `self · other`, which the caller's bit bound fits into `L` limbs.
    fn mul(&self, other: &Self) -> Self {
        let mut r = [0u64; L];
        for (i, &x) in self.limbs[..self.len].iter().enumerate() {
            let mut carry = 0u128;
            for (j, &y) in other.limbs[..other.len].iter().enumerate() {
                let t = u128::from(x) * u128::from(y) + u128::from(r[i + j]) + carry;
                r[i + j] = t as u64;
                carry = t >> 64;
            }
            if carry != 0 {
                r[i + other.len] = carry as u64;
            }
        }
        let mut len = (self.len + other.len).min(L);
        while len > 0 && r[len - 1] == 0 {
            len -= 1;
        }
        Self {
            neg: self.neg != other.neg,
            len,
            limbs: r,
        }
    }

    /// Adds the limb products of `|self · other|` into the `2L` lazy-carry
    /// columns `col` (column `k` weighs `2^(64k)`); carries are resolved once
    /// per entry by [`column_natural`].
    #[inline]
    fn mac_into(&self, other: &Self, col: &mut [u128]) {
        for (i, &x) in self.limbs[..self.len].iter().enumerate() {
            for (j, &y) in other.limbs[..other.len].iter().enumerate() {
                let p = u128::from(x) * u128::from(y);
                col[i + j] += u128::from(p as u64);
                col[i + j + 1] += p >> 64;
            }
        }
    }
}

/// `Σ_k col[k] 2^(64k)`. A column receives two summands below `2^64` per
/// limb product, so it stays below `2^128` for fewer than `2^63` products.
fn column_natural(col: &[u128]) -> Natural {
    let mut limbs = Vec::with_capacity(col.len() + 2);
    let mut carry = 0u128;
    for &c in col {
        let t = c + carry;
        limbs.push(t as u64);
        carry = t >> 64;
    }
    while carry != 0 {
        limbs.push(carry as u64);
        carry >>= 64;
    }
    limbs.iter().rev().fold(Natural::from(0u32), |acc, &w| {
        (acc << 64u32) + Natural::from(w)
    })
}

/// `S[μν, κλ] = Σ_key w_key · AB[key][μν] · CD[key][κλ]`, `w = ν_a ν_b ν_c`:
/// one multiply–accumulate per key and entry pair, no per-term allocation.
/// `w` multiplies the narrower side once per `(key, μν)` or `(key, κλ)`.
fn sum_fixed<const L: usize>(
    ab: &Keyed,
    cd: &Keyed,
    nu: [&[Q]; 3],
    len: usize,
    n34: usize,
    w_on_ab: bool,
) -> Vec<Signed> {
    let c = 2 * L;
    let mut cols = vec![0u128; len * 2 * c];
    let nus: [Vec<Mag<L>>; 3] = nu.map(|v| v.iter().map(Mag::from_q).collect());
    let (mut left, mut right): (Vec<Mag<L>>, Vec<Mag<L>>) = (Vec::new(), Vec::new());
    for (key, cdv) in cd {
        let w = nus[0][key.0].mul(&nus[1][key.1]).mul(&nus[2][key.2]);
        left.clear();
        right.clear();
        if w_on_ab {
            left.extend(ab[key].iter().map(|x| Mag::from_q(x).mul(&w)));
            right.extend(cdv.iter().map(Mag::from_q));
        } else {
            left.extend(ab[key].iter().map(Mag::from_q));
            right.extend(cdv.iter().map(|x| Mag::from_q(x).mul(&w)));
        }
        for (i, l) in left.iter().enumerate() {
            if l.len == 0 {
                continue;
            }
            for (j, r) in right.iter().enumerate() {
                if r.len == 0 {
                    continue;
                }
                let base = ((i * n34 + j) * 2 + usize::from(l.neg != r.neg)) * c;
                l.mac_into(r, &mut cols[base..base + c]);
            }
        }
    }
    settle(&cols, c)
}

/// The same sum accumulated in `Q`.
fn sum_big(ab: &Keyed, cd: &Keyed, nu: [&[Q]; 3], len: usize, n34: usize) -> Vec<Signed> {
    let mut s = vec![Q::from(0); len];
    for (key, cdv) in cd {
        let w = &(&nu[0][key.0] * &nu[1][key.1]) * &nu[2][key.2];
        for (i, xab) in ab[key].iter().enumerate() {
            if xab.is_zero() {
                continue;
            }
            let t = &w * xab;
            for (j, xcd) in cdv.iter().enumerate() {
                if !xcd.is_zero() {
                    s[i * n34 + j].add_mul(&t, xcd);
                }
            }
        }
    }
    s.iter().map(signed).collect()
}

/// Chooses the accumulator from a bound computed before the loop: the
/// widest operand is `bits(w) + bits(narrower side)` or the wider side, with
/// `bits(w) ≤ Σ max bits(ν)`; the lazy columns hold any sum, so only the
/// operand width selects `L`.
fn sum_auto(
    ab: &Keyed,
    cd: &Keyed,
    nu: [&[Q]; 3],
    len: usize,
    n34: usize,
    st: &mut ProbeStats,
) -> Vec<Signed> {
    let w_bits: u64 = nu.iter().map(|v| max_bits(v)).sum();
    let ab_bits = cd.keys().map(|k| max_bits(&ab[k])).max().unwrap_or(0);
    let cd_bits = cd.values().map(|v| max_bits(v)).max().unwrap_or(0);
    let w_on_ab = ab_bits <= cd_bits;
    let widest = if w_on_ab {
        (w_bits + ab_bits).max(cd_bits)
    } else {
        ab_bits.max(w_bits + cd_bits)
    };
    let (kind, out) = match widest.div_ceil(64) {
        0..=1 => ("words", sum_fixed::<1>(ab, cd, nu, len, n34, w_on_ab)),
        2..=4 => ("limbs4", sum_fixed::<4>(ab, cd, nu, len, n34, w_on_ab)),
        5..=8 => ("limbs8", sum_fixed::<8>(ab, cd, nu, len, n34, w_on_ab)),
        9..=16 => ("limbs16", sum_fixed::<16>(ab, cd, nu, len, n34, w_on_ab)),
        _ => ("big", sum_big(ab, cd, nu, len, n34)),
    };
    st.accum.push_str(kind);
    out
}

/// Splits per-entry lazy-carry columns `[pos; c][neg; c]` into `(sign, |S|)`.
fn settle(cols: &[u128], c: usize) -> Vec<Signed> {
    cols.chunks_exact(2 * c)
        .map(|e| {
            let (pos, neg) = (column_natural(&e[..c]), column_natural(&e[c..]));
            match pos.cmp(&neg) {
                Ordering::Greater => (1, pos - neg),
                Ordering::Less => (-1, neg - pos),
                Ordering::Equal => (0, Natural::from(0u32)),
            }
        })
        .collect()
}

/// A channel entry regrouped for the word join: `(g0, g1, m_b, multiplicity,
/// X)`, sorted by the group `(g0, g1)`.
type WordEntry = (usize, usize, usize, usize, i64);
/// A sliced channel grouped by one engine index: `(other index, multiplicity,
/// X)`.
type WordGroup = HashMap<usize, Vec<(usize, usize, i64)>>;

/// `q` as a word, when it fits.
fn word(q: &Q) -> Option<i64> {
    let (p, _) = q.magnitudes();
    let v = i64::try_from(u64::try_from(p.as_ref()).ok()?).ok()?;
    Some(if q.signum() == Ordering::Less { -v } else { v })
}

fn word_bits(v: i64) -> u64 {
    u64::from(64 - v.unsigned_abs().leading_zeros())
}

fn bit_len(n: usize) -> u64 {
    u64::from(usize::BITS - n.leading_zeros())
}

/// `(g0, g1)` → the range of that group in a sorted entry list.
type Ranges = HashMap<(usize, usize), Range<usize>>;

/// One flag per engine index of a module, set for `idx`.
fn marks(n: usize, idx: impl Iterator<Item = usize>) -> Vec<bool> {
    let mut v = vec![false; n];
    for i in idx {
        v[i] = true;
    }
    v
}

/// Entries of a full channel as `(g0, g1, m_b, μ, X)` sorted by `(g0, g1)`,
/// with the range of each group: one sort, no per-key allocation.
fn sorted_groups(
    ch: &IntChannel,
    regroup: impl Fn(&Raw) -> (usize, usize, usize),
) -> Option<(Vec<WordEntry>, Ranges)> {
    let mut v = ch
        .entries
        .iter()
        .map(|e| {
            let (g0, g1, mb) = regroup(e);
            Some((g0, g1, mb, e.3, word(&e.4)?))
        })
        .collect::<Option<Vec<WordEntry>>>()?;
    v.sort_unstable_by_key(|e| (e.0, e.1));
    let mut at = HashMap::new();
    let mut start = 0;
    for g in v.chunk_by(|x, y| (x.0, x.1) == (y.0, y.1)) {
        at.insert((g[0].0, g[0].1), start..start + g.len());
        start += g.len();
    }
    Some((v, at))
}

/// A sliced channel grouped by `key(entry)`, with `(other, multiplicity, X)`.
fn word_group(ch: &IntChannel, key: impl Fn(&Raw) -> (usize, usize)) -> Option<WordGroup> {
    let mut g: WordGroup = HashMap::new();
    for e in &ch.entries {
        let (k, other) = key(e);
        g.entry(k).or_default().push((other, e.3, word(&e.4)?));
    }
    Some(g)
}

/// The inputs of [`fused`], with every bound checked.
struct Fused<'a> {
    /// `A` as `(m_a, m_e, m_b, μ, X)` and its `(m_a, m_e)` ranges.
    a: Vec<WordEntry>,
    a_at: Ranges,
    /// `Cb` as `(m_c, m_f, m_b, κ, X)` and its `(m_c, m_f)` ranges.
    c: Vec<WordEntry>,
    c_at: Ranges,
    /// `B` seeds by `m_c` → `(m_e, ν, X)`; `D` seeds by `m_a` → `(m_f, λ, X)`.
    b_by_mc: WordGroup,
    d_by_ma: WordGroup,
    nu: [&'a [Q]; 3],
    /// Position of each `m_b` within its weight space, and the largest
    /// weight multiplicity of `b`.
    rank: &'a [u32],
    slots: usize,
    dims: [usize; 4],
    w_on_ab: bool,
}

/// The joins and the final sum per `(m_a, m_c)`: all keys `(m_a, m_b, m_c)`
/// of one pair share the weight `top(d) − wt(m_a) − wt(m_c)` of `m_b`, so
/// `m_b` indexes a dense slot by its rank in that weight space. Each partial
/// join is a word × word multiply–accumulate into an `i128` slot, read once
/// by the final sum as stack limbs; slots are cleared as they are read.
fn fused<const L: usize>(x: &Fused<'_>, st: &mut ProbeStats) -> Vec<Signed> {
    let [n1, n2, n3, n4] = x.dims;
    let (n12, n34, c) = (n1 * n2, n3 * n4, 2 * L);
    let mut cols = vec![0u128; n12 * n34 * 2 * c];
    let nus: [Vec<Mag<L>>; 3] = x.nu.map(|v| v.iter().map(Mag::from_q).collect());
    let mut ab = vec![0i128; x.slots * n12];
    let mut cd = vec![0i128; x.slots * n34];
    // The `m_b` of a slot reached by the AB join, `usize::MAX` otherwise.
    let mut slot_mb = vec![usize::MAX; x.slots];
    let mut has_cd = vec![false; x.slots];
    let mut touched: Vec<usize> = Vec::with_capacity(x.slots);
    let (mut left, mut right): (Vec<Mag<L>>, Vec<Mag<L>>) =
        (Vec::with_capacity(n12), Vec::with_capacity(n34));
    let (mut join, mut sum) = (0.0, 0.0);
    for (&ma, dl) in &x.d_by_ma {
        for (&mc, bl) in &x.b_by_mc {
            let t0 = Instant::now();
            for &(me, nu, xb) in bl {
                let Some(r) = x.a_at.get(&(ma, me)) else {
                    continue;
                };
                for &(_, _, mb, mu, xa) in &x.a[r.clone()] {
                    let s = x.rank[mb] as usize;
                    if slot_mb[s] == usize::MAX {
                        slot_mb[s] = mb;
                        touched.push(s);
                    }
                    ab[s * n12 + mu * n2 + nu] += i128::from(xa) * i128::from(xb);
                }
            }
            if touched.is_empty() {
                join += t0.elapsed().as_secs_f64();
                continue;
            }
            for &(mf, lambda, xd) in dl {
                let Some(r) = x.c_at.get(&(mc, mf)) else {
                    continue;
                };
                for &(_, _, mb, kappa, xc) in &x.c[r.clone()] {
                    let s = x.rank[mb] as usize;
                    if slot_mb[s] == mb {
                        has_cd[s] = true;
                        cd[s * n34 + kappa * n4 + lambda] += i128::from(xc) * i128::from(xd);
                    }
                }
            }
            let t1 = Instant::now();
            join += (t1 - t0).as_secs_f64();
            let wac = nus[0][ma].mul(&nus[2][mc]);
            for &s in &touched {
                let (abr, cdr) = (s * n12..(s + 1) * n12, s * n34..(s + 1) * n34);
                if has_cd[s] {
                    let w = wac.mul(&nus[1][slot_mb[s]]);
                    left.clear();
                    right.clear();
                    if x.w_on_ab {
                        left.extend(ab[abr.clone()].iter().map(|&v| Mag::from_i128(v).mul(&w)));
                        right.extend(cd[cdr.clone()].iter().map(|&v| Mag::from_i128(v)));
                    } else {
                        left.extend(ab[abr.clone()].iter().map(|&v| Mag::from_i128(v)));
                        right.extend(cd[cdr.clone()].iter().map(|&v| Mag::from_i128(v).mul(&w)));
                    }
                    for (i, l) in left.iter().enumerate() {
                        if l.len == 0 {
                            continue;
                        }
                        for (j, r) in right.iter().enumerate() {
                            if r.len == 0 {
                                continue;
                            }
                            let base = ((i * n34 + j) * 2 + usize::from(l.neg != r.neg)) * c;
                            l.mac_into(r, &mut cols[base..base + c]);
                        }
                    }
                    cd[cdr].fill(0);
                    has_cd[s] = false;
                }
                ab[abr].fill(0);
                slot_mb[s] = usize::MAX;
            }
            touched.clear();
            sum += t1.elapsed().as_secs_f64();
        }
    }
    st.join_secs += join;
    st.sum_secs += sum;
    settle(&cols, c)
}

/// The word join and final sum when every bound holds before the loop:
/// each `X` a word; `bits(AB) ≤ bits X_A + bits X_B + bit_len(max |B[m_c]|)`
/// within `i128`, likewise CD with `max |D[m_a]|`; and the final operands
/// within 16 limbs. `None` leaves the block to the `Q` join.
fn fused_sum(
    (ca, cb, cc, cd): (&IntChannel, &IntChannel, &IntChannel, &IntChannel),
    nu: [&[Q]; 3],
    lb: &Ladders,
    dims: [usize; 4],
    st: &mut ProbeStats,
) -> Option<Vec<Signed>> {
    let t = Instant::now();
    let (a, a_at) = sorted_groups(ca, |e| (e.0, e.2, e.1))?;
    let (c, c_at) = sorted_groups(cc, |e| (e.1, e.2, e.0))?;
    let b_by_mc = word_group(cb, |e| (e.1, e.0))?;
    let d_by_ma = word_group(cd, |e| (e.0, e.1))?;
    let bits = |v: &[WordEntry]| v.iter().map(|e| word_bits(e.4)).max().unwrap_or(0);
    let gbits = |g: &WordGroup| {
        g.values()
            .flatten()
            .map(|e| word_bits(e.2))
            .max()
            .unwrap_or(0)
    };
    let longest = |g: &WordGroup| g.values().map(Vec::len).max().unwrap_or(0);
    let ab_bits = bits(&a) + gbits(&b_by_mc) + bit_len(longest(&b_by_mc));
    let cd_bits = bits(&c) + gbits(&d_by_ma) + bit_len(longest(&d_by_ma));
    if ab_bits > 127 || cd_bits > 127 {
        return None;
    }
    let w_bits: u64 = nu.iter().map(|v| max_bits(v)).sum();
    let w_on_ab = ab_bits <= cd_bits;
    let widest = if w_on_ab {
        (w_bits + ab_bits).max(cd_bits)
    } else {
        ab_bits.max(w_bits + cd_bits)
    };
    let x = Fused {
        a,
        a_at,
        c,
        c_at,
        b_by_mc,
        d_by_ma,
        nu,
        rank: &lb.rank,
        slots: lb
            .module
            .weights
            .iter()
            .map(|s| s.norms.len())
            .max()
            .unwrap_or(0),
        dims,
        w_on_ab,
    };
    st.join_secs += t.elapsed().as_secs_f64();
    let (kind, out) = match widest.div_ceil(64) {
        0..=1 => ("words", fused::<1>(&x, st)),
        2..=4 => ("limbs4", fused::<4>(&x, st)),
        5..=8 => ("limbs8", fused::<8>(&x, st)),
        9..=16 => ("limbs16", fused::<16>(&x, st)),
        _ => return None,
    };
    st.accum.push_str("join128+");
    st.accum.push_str(kind);
    Some(out)
}

/// The `Q` hash-map joins of every key `(m_a, m_b, m_c)` and the final sum
/// over them: the fallback, and the reference of [`Accum::Check`].
fn legacy_sum(
    (ca, cb, cc, cd): (&IntChannel, &IntChannel, &IntChannel, &IntChannel),
    nu: [&[Q]; 3],
    dims: [usize; 4],
    accum: Accum,
    st: &mut ProbeStats,
) -> Vec<Signed> {
    let [n1, n2, n3, n4] = dims;
    let t1 = Instant::now();
    // AB[(m_a,m_b,m_c)][μ,ν] = Σ_{m_e} X_A X_B.
    let mut b_by_me: ByIndex<'_> = HashMap::new();
    for (pe, pc, _, nu, x) in &cb.entries {
        b_by_me.entry(*pe).or_default().push((*pc, *nu, x));
    }
    let mut ab: Keyed = HashMap::new();
    for (pa, pb, pe, mu, xa) in &ca.entries {
        let Some(bl) = b_by_me.get(pe) else { continue };
        for &(pc, nu, xb) in bl {
            ab.entry((*pa, *pb, pc))
                .or_insert_with(|| vec![Q::from(0); n1 * n2])[mu * n2 + nu]
                .add_mul(xa, xb);
        }
    }
    // CD[(m_a,m_b,m_c)][κ,λ] = Σ_{m_f} X_C X_D, on the keys of AB.
    let mut d_by_mf: ByIndex<'_> = HashMap::new();
    for (pa, pf, _, lambda, x) in &cd.entries {
        d_by_mf.entry(*pf).or_default().push((*pa, *lambda, x));
    }
    let mut cdm: Keyed = HashMap::new();
    for (pb, pc, pf, kappa, xc) in &cc.entries {
        let Some(dl) = d_by_mf.get(pf) else { continue };
        for &(pa, lambda, xd) in dl {
            let key = (pa, *pb, *pc);
            if ab.contains_key(&key) {
                cdm.entry(key).or_insert_with(|| vec![Q::from(0); n3 * n4])[kappa * n4 + lambda]
                    .add_mul(xc, xd);
            }
        }
    }
    let t2 = Instant::now();
    st.join_secs += (t2 - t1).as_secs_f64();
    st.accum.push_str("qjoin+");
    let s = final_sum(&ab, &cdm, nu, n1 * n2 * n3 * n4, n3 * n4, accum, st);
    st.sum_secs += t2.elapsed().as_secs_f64();
    s
}

/// The `Q` final sum: [`Accum::Big`] in `Q`, otherwise fixed limbs.
fn final_sum(
    ab: &Keyed,
    cd: &Keyed,
    nu: [&[Q]; 3],
    len: usize,
    n34: usize,
    accum: Accum,
    st: &mut ProbeStats,
) -> Vec<Signed> {
    if cd.is_empty() {
        st.accum.push_str("none");
        return vec![(0, Natural::from(0u32)); len];
    }
    if accum == Accum::Big {
        st.accum.push_str("big");
        sum_big(ab, cd, nu, len, n34)
    } else {
        sum_auto(ab, cd, nu, len, n34, st)
    }
}

fn simple_f(
    fam: &GeneralFamily,
    [a, b, c, d, e, f]: [Label<'_>; 6],
    accum: Accum,
    st: &mut ProbeStats,
) -> Result<Block, LieError> {
    let dims = [
        fam.mult(a, b, e)?,
        fam.mult(e, c, d)?,
        fam.mult(b, c, f)?,
        fam.mult(a, f, d)?,
    ];
    let [n1, n2, n3, n4] = dims;
    let len = crate::lie::checked_size(dims)?;
    if len == 0 {
        return Ok(Block::zeros(dims.to_vec()));
    }
    let t0 = Instant::now();
    let data = fam.data();
    let mut modules = RequestModules::default();
    let la = modules.load(fam, a)?;
    let lb = modules.load(fam, b)?;
    let lc = modules.load(fam, c)?;
    let ld = modules.load(fam, d)?;
    let le = modules.load(fam, e)?;
    let lf = modules.load(fam, f)?;
    progress("modules", t0);
    let (raw, norms) = seeds(data, (&le, &lc, &ld), n2)?;
    let cb = integral(raw, norms, st);
    let (raw, norms) = seeds(data, (&la, &lf, &ld), n4)?;
    let cd = integral(raw, norms, st);
    progress("seeds B, D", t0);
    // Only A entries whose m_e meets B and whose m_a meets D, and Cb entries
    // whose m_f meets D and whose m_c meets B, reach a key of both joins.
    let me_in = marks(le.dim(), cb.entries.iter().map(|x| x.0));
    let mc_in = marks(lc.dim(), cb.entries.iter().map(|x| x.1));
    let ma_in = marks(la.dim(), cd.entries.iter().map(|x| x.0));
    let mf_in = marks(lf.dim(), cd.entries.iter().map(|x| x.1));
    let mut capture = 0.0;
    let (raw, norms) = full(
        data,
        (&la, &lb, &le),
        n1,
        &|pa, _, pe| me_in[pe] && ma_in[pa],
        &mut capture,
    )?;
    let ca = integral(raw, norms, st);
    progress("descent A", t0);
    let (raw, norms) = full(
        data,
        (&lb, &lc, &lf),
        n3,
        &|_, pc, pf| mf_in[pf] && mc_in[pc],
        &mut capture,
    )?;
    let cc = integral(raw, norms, st);
    st.capture_secs += capture;
    progress("descent Cb", t0);
    let tn = Instant::now();
    let (nu_a, da, ga) = integral_norms(&la);
    let (nu_b, db, gb) = integral_norms(&lb);
    let (nu_c, dc, gc) = integral_norms(&lc);
    st.prep_secs += tn.elapsed().as_secs_f64();
    let (p_d0, q_d0) = ld.norm[0].magnitudes();
    let (p_d0, q_d0) = (p_d0.into_owned(), q_d0.into_owned());
    drop(modules);
    let t1 = Instant::now();
    st.gen_secs += (t1 - t0).as_secs_f64();

    // S = Σ_key ν_a ν_b ν_c AB·CD. The sum is exact, so key order is free.
    let nu = [nu_a.as_slice(), nu_b.as_slice(), nu_c.as_slice()];
    let chans = (&ca, &cb, &cc, &cd);
    if !st.accum.is_empty() {
        st.accum.push('+');
    }
    let s = match accum {
        Accum::Big => legacy_sum(chans, nu, dims, Accum::Big, st),
        Accum::Auto | Accum::Check => match fused_sum(chans, nu, &lb, dims, st) {
            Some(s) => s,
            None => legacy_sum(chans, nu, dims, Accum::Auto, st),
        },
    };
    if accum == Accum::Check {
        let mut reference = ProbeStats::default();
        let big = legacy_sum(chans, nu, dims, Accum::Big, &mut reference);
        st.accum.push('=');
        st.accum.push_str(&reference.accum);
        assert!(s == big, "word/limb and Q evaluations of S differ");
    }
    progress("joins and final sum", t0);
    // U = S² q(n_d0)² (g_a g_b g_c)² Π q(N) g²;  V = (Δa Δb Δc)² p(n_d0)² Π D² p(N).
    let (uq_a, vp_a) = ca.radicand_parts();
    let (uq_b, vp_b) = cb.radicand_parts();
    let (uq_c, vp_c) = cc.radicand_parts();
    let (uq_d, vp_d) = cd.radicand_parts();
    let u_common = q_d0.square() * (&(&ga * &gb) * &gc).square();
    let v_common = (&(&da * &db) * &dc).square() * p_d0.square();
    let mut entries = Vec::with_capacity(len);
    for (idx, x) in s.into_iter().enumerate() {
        let (mu, nu, kappa, lambda) = (
            idx / (n2 * n3 * n4),
            idx / (n3 * n4) % n2,
            idx / n4 % n3,
            idx % n4,
        );
        let u = &(&(&(&u_common * &uq_a[mu]) * &uq_b[nu]) * &uq_c[kappa]) * &uq_d[lambda];
        let v = &(&(&(&v_common * &vp_a[mu]) * &vp_b[nu]) * &vp_c[kappa]) * &vp_d[lambda];
        entries.push(radical(x, &u, v, st));
    }
    st.contract_secs += t1.elapsed().as_secs_f64();
    Ok(Block {
        dims: dims.to_vec(),
        entries,
    })
}

fn simple_r(
    fam: &GeneralFamily,
    [a, b, c]: [Label<'_>; 3],
    st: &mut ProbeStats,
) -> Result<Block, LieError> {
    let n = fam.mult(a, b, c)?;
    if n == 0 {
        return Ok(Block::zeros(vec![0, 0]));
    }
    let len = crate::lie::checked_size([n, n])?;
    let t0 = Instant::now();
    let data = fam.data();
    let mut modules = RequestModules::default();
    let la = modules.load(fam, a)?;
    let lb = modules.load(fam, b)?;
    let lc = modules.load(fam, c)?;
    progress("modules", t0);
    let (raw, norms) = seeds(data, (&la, &lb, &lc), n)?;
    let ca = integral(raw, norms, st);
    let (raw, norms) = seeds(data, (&lb, &la, &lc), n)?;
    let cb = integral(raw, norms, st);
    progress("seeds", t0);
    let (nu_a, da, ga) = integral_norms(&la);
    let (nu_b, db, gb) = integral_norms(&lb);
    let (p_c0, q_c0) = lc.norm[0].magnitudes();
    let (p_c0, q_c0) = (p_c0.into_owned(), q_c0.into_owned());
    drop(modules);
    let t1 = Instant::now();
    st.gen_secs += (t1 - t0).as_secs_f64();

    // C(b,a,c) keyed by (m_a, m_b) = (its right, its left).
    let mut b_map: ByPair<'_> = HashMap::new();
    for (pb, pa, _, nu, x) in &cb.entries {
        b_map.entry((*pa, *pb)).or_default().push((*nu, x));
    }
    st.join_secs += t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    // T is seeds-only and small, so it stays in `Q`.
    if !st.accum.is_empty() {
        st.accum.push('+');
    }
    st.accum.push_str("q");
    let mut t = vec![Q::from(0); len];
    for (pa, pb, _, mu, xa) in &ca.entries {
        let Some(bl) = b_map.get(&(*pa, *pb)) else {
            continue;
        };
        let w = &(&nu_a[*pa] * &nu_b[*pb]) * xa;
        for &(nu, xb) in bl {
            t[mu * n + nu].add_mul(&w, xb);
        }
    }
    st.sum_secs += t2.elapsed().as_secs_f64();
    progress("final sum", t0);
    // U = T² q(n_c0)² (g_a g_b)² q(N_μ) g_μ² q(N'_ν) g'_ν²;
    // V = (Δa Δb)² p(n_c0)² D_μ² p(N_μ) D'_ν² p(N'_ν).
    let (uq_a, vp_a) = ca.radicand_parts();
    let (uq_b, vp_b) = cb.radicand_parts();
    let u_common = q_c0.square() * (&ga * &gb).square();
    let v_common = (&da * &db).square() * p_c0.square();
    let mut entries = Vec::with_capacity(len);
    for (idx, x) in t.iter().enumerate() {
        let (mu, nu) = (idx / n, idx % n);
        let u = &(&u_common * &uq_a[mu]) * &uq_b[nu];
        let v = &(&v_common * &vp_a[mu]) * &vp_b[nu];
        entries.push(radical(signed(x), &u, v, st));
    }
    st.contract_secs += t1.elapsed().as_secs_f64();
    Ok(Block {
        dims: vec![n, n],
        entries,
    })
}

/// Position of `(l, r)` in the row-major Kronecker product of blocks with
/// axis lengths `ld`, `rd`: axis index `l · right + r` (`product.rs`).
fn kron_pos(mut li: usize, mut ri: usize, ld: &[usize], rd: &[usize]) -> usize {
    let (mut o, mut stride) = (0, 1);
    for (&l, &r) in ld.iter().zip(rd).rev() {
        o += (li % l * r + ri % r) * stride;
        stride *= l * r;
        li /= l;
        ri /= r;
    }
    o
}

fn kron_dims(ld: &[usize], rd: &[usize]) -> Vec<usize> {
    ld.iter().zip(rd).map(|(l, r)| l * r).collect()
}

fn kron(l: &Block, r: &Block) -> Block {
    let mut out = Block::zeros(kron_dims(&l.dims, &r.dims));
    for (li, x) in l.entries.iter().enumerate() {
        for (ri, y) in r.entries.iter().enumerate() {
            out.entries[kron_pos(li, ri, &l.dims, &r.dims)] = x.times(y);
        }
    }
    out
}

fn unsupported() -> LieError {
    LieError::Unsupported {
        requires: "exact-fr-probe: simple rank >= 2, SU(2) or products of them",
    }
}

fn block_f(
    g: &LieGroup,
    labels: [&LieIrrep; 6],
    accum: Accum,
    st: &mut ProbeStats,
) -> Result<Block, LieError> {
    if let Some(product) = &g.composite {
        let mut acc: Option<Block> = None;
        for (i, factor) in product.factors().iter().enumerate() {
            let fl = labels.map(|x| &x.factors().expect("composite label")[i]);
            let block = block_f(factor, fl, accum, st)?;
            let t = Instant::now();
            acc = Some(match acc {
                None => block,
                Some(left) => kron(&left, &block),
            });
            st.contract_secs += t.elapsed().as_secs_f64();
        }
        return acc.ok_or_else(unsupported);
    }
    match labels[0].payload {
        LiePayload::Simple(_) => {
            let ls = labels.map(|x| x.simple().expect("simple label").label());
            simple_f(g.engine(), ls, accum, st)
        }
        LiePayload::RankOne(_) => {
            let t = Instant::now();
            let [a, b, c, d, e, f] = labels.map(|x| x.dj());
            let dims = [(a, b, e), (e, c, d), (b, c, f), (a, f, d)]
                .map(|(x, y, z)| usize::from(su2::triangle_ok(x, y, z)));
            if dims.contains(&0) {
                return Ok(Block::zeros(dims.to_vec()));
            }
            let v = su2::f_symbol_exact(a, b, c, d, e, f);
            let entry = if v.sign() == 0 {
                Exact::zero()
            } else {
                Exact {
                    sign: v.sign(),
                    u: v.radicand().numerator_ref().clone(),
                    v: v.radicand().denominator_ref().clone(),
                }
            };
            st.gen_secs += t.elapsed().as_secs_f64();
            Ok(Block {
                dims: vec![1; 4],
                entries: vec![entry],
            })
        }
        _ => Err(unsupported()),
    }
}

fn block_r(g: &LieGroup, labels: [&LieIrrep; 3], st: &mut ProbeStats) -> Result<Block, LieError> {
    if let Some(product) = &g.composite {
        let mut acc: Option<Block> = None;
        for (i, factor) in product.factors().iter().enumerate() {
            let fl = labels.map(|x| &x.factors().expect("composite label")[i]);
            let block = block_r(factor, fl, st)?;
            let t = Instant::now();
            acc = Some(match acc {
                None => block,
                Some(left) => kron(&left, &block),
            });
            st.contract_secs += t.elapsed().as_secs_f64();
        }
        return acc.ok_or_else(unsupported);
    }
    match labels[0].payload {
        LiePayload::Simple(_) => {
            let ls = labels.map(|x| x.simple().expect("simple label").label());
            simple_r(g.engine(), ls, st)
        }
        LiePayload::RankOne(_) => {
            let [a, b, c] = labels.map(|x| x.dj());
            if !su2::triangle_ok(a, b, c) {
                return Ok(Block::zeros(vec![0, 0]));
            }
            // R is exactly ±1.
            let sign = if su2::r_symbol(a, b, c) < 0.0 { -1 } else { 1 };
            Ok(Block {
                dims: vec![1, 1],
                entries: vec![Exact {
                    sign,
                    u: one(),
                    v: one(),
                }],
            })
        }
        _ => Err(unsupported()),
    }
}

/// Rounds every entry once, timed as contraction.
fn finish(block: Block, mut st: ProbeStats) -> ProbeBlock {
    let t = Instant::now();
    let mut values = Vec::with_capacity(block.entries.len());
    for x in &block.entries {
        if x.sign != 0 {
            st.u_bits.push(x.u.significant_bits());
            st.v_bits.push(x.v.significant_bits());
        }
        values.push(crate::exact::signed_sqrt_to_f64(x.sign, &x.u, &x.v));
    }
    st.contract_secs += t.elapsed().as_secs_f64();
    ProbeBlock {
        dims: block.dims,
        values,
        stats: st,
    }
}

/// `F^{abc}_d[e, f]` evaluated exactly and rounded once, its final sum
/// accumulated as `accum` says. Reads modules through the CGC tier (reset it
/// for a cold measurement), retains no exact data, and reports each phase on
/// stderr.
pub fn exact_f(g: &LieGroup, labels: [&LieIrrep; 6], accum: Accum) -> Result<ProbeBlock, LieError> {
    let mut st = ProbeStats::default();
    let block = block_f(g, labels, accum, &mut st)?;
    Ok(finish(block, st))
}

/// `R^{ab}_c` evaluated exactly and rounded once; see [`exact_f`].
pub fn exact_r(g: &LieGroup, labels: [&LieIrrep; 3]) -> Result<ProbeBlock, LieError> {
    let mut st = ProbeStats::default();
    let block = block_r(g, labels, &mut st)?;
    Ok(finish(block, st))
}

fn gamma(k: u64) -> f64 {
    let ku = k as f64 * U;
    ku / (1.0 - ku)
}

fn simple_tol_f(
    fam: &GeneralFamily,
    [a, b, c, d, e, f]: [Label<'_>; 6],
) -> Result<Vec<f64>, LieError> {
    let dims = [
        fam.mult(a, b, e)?,
        fam.mult(e, c, d)?,
        fam.mult(b, c, f)?,
        fam.mult(a, f, d)?,
    ];
    let len = crate::lie::checked_size(dims)?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let cab = fam.cgc_entries(a, b, e, None)?;
    let cecd = fam.cgc_entries(e, c, d, None)?;
    let cbcf = fam.cgc_entries(b, c, f, None)?;
    let cafd = fam.cgc_entries(a, f, d, None)?;
    let mut b_by_me: Grouped = HashMap::new();
    for x in cecd.iter().filter(|x| x.m3 == 0) {
        b_by_me
            .entry(x.m1)
            .or_default()
            .push((x.m2, x.mu, x.value.abs()));
    }
    let mut ab: Joined = HashMap::new();
    for x in cab.iter() {
        let Some(bl) = b_by_me.get(&x.m3) else {
            continue;
        };
        for &(mc, nu, vb) in bl {
            ab.entry((x.m1, x.m2, mc))
                .or_default()
                .push((x.mu, nu, x.value.abs() * vb));
        }
    }
    let mut d_by_mf: Grouped = HashMap::new();
    for x in cafd.iter().filter(|x| x.m3 == 0) {
        d_by_mf
            .entry(x.m2)
            .or_default()
            .push((x.m1, x.mu, x.value.abs()));
    }
    let mut cd: Joined = HashMap::new();
    for x in cbcf.iter() {
        let Some(dl) = d_by_mf.get(&x.m3) else {
            continue;
        };
        for &(ma, lambda, vd) in dl {
            let key = (ma, x.m1, x.m2);
            if ab.contains_key(&key) {
                cd.entry(key)
                    .or_default()
                    .push((x.mu, lambda, x.value.abs() * vd));
            }
        }
    }
    let mut m = vec![0.0; len];
    let mut n = vec![0u64; len];
    for (key, cdl) in &cd {
        for &(mu, nu, x) in &ab[key] {
            for &(kappa, lambda, y) in cdl {
                let idx = ((mu as usize * dims[1] + nu as usize) * dims[2] + kappa as usize)
                    * dims[3]
                    + lambda as usize;
                m[idx] += x * y;
                n[idx] += 1;
            }
        }
    }
    Ok(m.iter()
        .zip(&n)
        .map(|(m, &n)| gamma(2 * (n + 7)) * m)
        .collect())
}

fn simple_tol_r(fam: &GeneralFamily, [a, b, c]: [Label<'_>; 3]) -> Result<Vec<f64>, LieError> {
    let n = fam.mult(a, b, c)?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let cab = fam.cgc_entries(a, b, c, None)?;
    let cba = fam.cgc_entries(b, a, c, None)?;
    let mut b_map: PairAbs = HashMap::new();
    for x in cba.iter().filter(|x| x.m3 == 0) {
        b_map
            .entry((x.m2, x.m1))
            .or_default()
            .push((x.mu, x.value.abs()));
    }
    let mut m = vec![0.0; n * n];
    let mut k = vec![0u64; n * n];
    for x in cab.iter().filter(|x| x.m3 == 0) {
        let Some(bl) = b_map.get(&(x.m1, x.m2)) else {
            continue;
        };
        for &(nu, vb) in bl {
            let idx = x.mu as usize * n + nu as usize;
            m[idx] += x.value.abs() * vb;
            k[idx] += 1;
        }
    }
    Ok(m.iter()
        .zip(&k)
        .map(|(m, &k)| gamma(2 * (k + 3)) * m)
        .collect())
}

/// `(dims, production values, bounds)` of one block.
type Tolerated = (Vec<usize>, Vec<f64>, Vec<f64>);

/// Production values and bounds folded left to right as the product path
/// multiplies them: `|F_l F_r − fl(c_l c_r)| ≤ t_l(|c_r| + t_r) + |c_l| t_r + u|c_l c_r|`.
fn kron_tol((ld, lc, lt): Tolerated, (rd, rc, rt): Tolerated) -> Tolerated {
    let dims = kron_dims(&ld, &rd);
    let len = dims.iter().product();
    let (mut cur, mut tol) = (vec![0.0; len], vec![0.0; len]);
    for (li, (&cl, &tl)) in lc.iter().zip(&lt).enumerate() {
        for (ri, (&cr, &tr)) in rc.iter().zip(&rt).enumerate() {
            let o = kron_pos(li, ri, &ld, &rd);
            cur[o] = cl * cr;
            tol[o] = tl * (cr.abs() + tr) + cl.abs() * tr + U * cur[o].abs();
        }
    }
    (dims, cur, tol)
}

fn tolerated_f(g: &LieGroup, labels: [&LieIrrep; 6]) -> Result<Tolerated, LieError> {
    if let Some(product) = &g.composite {
        let mut acc: Option<Tolerated> = None;
        for (i, factor) in product.factors().iter().enumerate() {
            let fl = labels.map(|x| &x.factors().expect("composite label")[i]);
            let part = tolerated_f(factor, fl)?;
            acc = Some(match acc {
                None => part,
                Some(left) => kron_tol(left, part),
            });
        }
        return acc.ok_or_else(unsupported);
    }
    let cur = g.f_symbol_in(labels, None)?;
    let values = cur.as_slice().to_vec();
    let dims = cur.dims().to_vec();
    let tol = match labels[0].payload {
        LiePayload::Simple(_) => simple_tol_f(
            g.engine(),
            labels.map(|x| x.simple().expect("simple label").label()),
        )?,
        // The closed form is correctly rounded.
        LiePayload::RankOne(_) => values.iter().map(|x| U * x.abs()).collect(),
        _ => return Err(unsupported()),
    };
    Ok((dims, values, tol))
}

fn tolerated_r(g: &LieGroup, labels: [&LieIrrep; 3]) -> Result<Tolerated, LieError> {
    if let Some(product) = &g.composite {
        let mut acc: Option<Tolerated> = None;
        for (i, factor) in product.factors().iter().enumerate() {
            let fl = labels.map(|x| &x.factors().expect("composite label")[i]);
            let part = tolerated_r(factor, fl)?;
            acc = Some(match acc {
                None => part,
                Some(left) => kron_tol(left, part),
            });
        }
        return acc.ok_or_else(unsupported);
    }
    let [a, b, c] = labels;
    let cur = g.r_symbol_in(a, b, c, None)?;
    let values = cur.as_slice().to_vec();
    let dims = cur.dims().to_vec();
    let tol = match labels[0].payload {
        LiePayload::Simple(_) => simple_tol_r(
            g.engine(),
            labels.map(|x| x.simple().expect("simple label").label()),
        )?,
        // Exactly ±1.
        LiePayload::RankOne(_) => vec![0.0; values.len()],
        _ => return Err(unsupported()),
    };
    Ok((dims, values, tol))
}

/// A bound on `|F_current − F_exact|` per entry of the production F block,
/// from the rounded channels it reads (warm after a production query).
pub fn tol_f(g: &LieGroup, labels: [&LieIrrep; 6]) -> Result<Vec<f64>, LieError> {
    Ok(tolerated_f(g, labels)?.2)
}

/// A bound on `|R_current − R_exact|` per entry; see [`tol_f`].
pub fn tol_r(g: &LieGroup, labels: [&LieIrrep; 3]) -> Result<Vec<f64>, LieError> {
    Ok(tolerated_r(g, labels)?.2)
}
