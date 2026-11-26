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
//! one common denominator, so `S` and `T` are integer multiply–accumulates in
//! `Q`, and every denominator enters only the final radicand `U / V`, which
//! `exact::signed_sqrt_to_f64` rounds once. Product groups take the exact
//! Kronecker product of the factor radicands (factor order, row-major, as
//! `product.rs`) and round once.
//!
//! The tolerance functions bound `|current − exact|` for the production
//! binary64 evaluation of §6.2 from its own rounded channels:
//! `γ_k · Σ|terms|` with `k = 2(n + 7)` for F and `k = 2(n + 3)` for R
//! (`n` summed terms per entry; four, resp. two, correctly rounded CGC and
//! three, resp. one, rounded products per term; the factor 2 absorbs the
//! rounding of the absolute sum itself).

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use malachite_base::num::arithmetic::traits::{Lcm, Square};
use malachite_base::num::logic::traits::SignificantBits;
use malachite_nz::natural::Natural;
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
    /// Bit length of each exact integer numerator `S` (resp. `T`); for a
    /// product group, of each factor's numerators.
    pub s_bits: Vec<u64>,
    /// Bit length of each radicand numerator `U` passed to rounding.
    pub u_bits: Vec<u64>,
    /// Bit length of each radicand denominator `V` passed to rounding.
    pub v_bits: Vec<u64>,
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

/// `sign·√(u/v)`, unreduced; zero has sign 0.
#[derive(Clone)]
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

/// An exact channel on per-`μ` common denominators: `X = x·D_μ` is an
/// integer.
struct IntChannel {
    entries: Vec<Raw>,
    den: Vec<Natural>,
    norms: Vec<Q>,
}

impl IntChannel {
    /// Per `μ`: the radicand factors `q(N_μ)` (numerator side) and
    /// `D_μ²·p(N_μ)` (denominator side).
    fn radicand_parts(&self) -> (Vec<Natural>, Vec<Natural>) {
        self.norms
            .iter()
            .zip(&self.den)
            .map(|(n, d)| {
                let (p, q) = n.magnitudes();
                (q.into_owned(), d.square() * p.as_ref())
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

fn integral(raw: Vec<Raw>, norms: Vec<Q>, st: &mut ProbeStats) -> IntChannel {
    let mut den = vec![one(); norms.len()];
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
    let entries = raw
        .into_iter()
        .map(|(p, q, m, mu, x)| (p, q, m, mu, x * &scale[mu]))
        .collect();
    IntChannel {
        entries,
        den,
        norms,
    }
}

/// `ν_m = n_m Δ` with `Δ` the lcm of every norm denominator of `m`.
fn integral_norms(m: &Ladders) -> (Vec<Q>, Natural) {
    let mut den = one();
    for n in &m.norm {
        let (_, q) = n.magnitudes();
        if *q != one() {
            den = den.lcm(q.into_owned());
        }
    }
    let scale = to_q(&den);
    (m.norm.iter().map(|n| n * &scale).collect(), den)
}

/// The full exact channel `a ⊗ b → c` by the production descent, keeping
/// the coupled states in `keep`.
fn full(
    data: &CartanData,
    (a, b, c): (&Ladders, &Ladders, &Ladders),
    n: usize,
    keep: &HashSet<usize>,
) -> Result<(Vec<Raw>, Vec<Q>), LieError> {
    let mut raw = Vec::new();
    let mut norms = Vec::new();
    descend(data, a, b, c, n, &mut |seed_norms, m, space, per_mu| {
        if norms.is_empty() {
            norms = seed_norms.to_vec();
        }
        if keep.contains(&m) {
            for (mu, v) in per_mu.iter().enumerate() {
                for (&(pa, pb), x) in space.pairs.iter().zip(v) {
                    if !x.is_zero() {
                        raw.push((pa, pb, m, mu, x.clone()));
                    }
                }
            }
        }
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

/// `sign(s)·√(s²·u_rest / v_rest)` with `s` an integer.
fn radical(s: &Q, u_rest: &Natural, v_rest: Natural, st: &mut ProbeStats) -> Exact {
    st.s_values += 1;
    st.s_big += u64::from(s.heap_bytes() != 0);
    let sign = sign_of(s);
    if sign == 0 {
        return Exact::zero();
    }
    let (ps, _) = s.magnitudes();
    st.s_bits.push(ps.as_ref().significant_bits());
    Exact {
        sign,
        u: ps.as_ref().square() * u_rest,
        v: v_rest,
    }
}

fn simple_f(
    fam: &GeneralFamily,
    [a, b, c, d, e, f]: [Label<'_>; 6],
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
    let (raw, norms) = seeds(data, (&le, &lc, &ld), n2)?;
    let cb = integral(raw, norms, st);
    let me: HashSet<usize> = cb.entries.iter().map(|x| x.0).collect();
    let (raw, norms) = full(data, (&la, &lb, &le), n1, &me)?;
    let ca = integral(raw, norms, st);
    let (raw, norms) = seeds(data, (&la, &lf, &ld), n4)?;
    let cd = integral(raw, norms, st);
    let mf: HashSet<usize> = cd.entries.iter().map(|x| x.1).collect();
    let (raw, norms) = full(data, (&lb, &lc, &lf), n3, &mf)?;
    let cc = integral(raw, norms, st);
    let (nu_a, da) = integral_norms(&la);
    let (nu_b, db) = integral_norms(&lb);
    let (nu_c, dc) = integral_norms(&lc);
    let (p_d0, q_d0) = ld.norm[0].magnitudes();
    let (p_d0, q_d0) = (p_d0.into_owned(), q_d0.into_owned());
    drop(modules);
    let t1 = Instant::now();
    st.gen_secs += (t1 - t0).as_secs_f64();

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
    // S = Σ_key ν_a ν_b ν_c AB·CD. The sum is exact, so key order is free.
    let mut s = vec![Q::from(0); len];
    for (key, cdv) in &cdm {
        let w = &(&nu_a[key.0] * &nu_b[key.1]) * &nu_c[key.2];
        for (i, xab) in ab[key].iter().enumerate() {
            if xab.is_zero() {
                continue;
            }
            let t = &w * xab;
            let base = i * n3 * n4;
            for (j, xcd) in cdv.iter().enumerate() {
                if !xcd.is_zero() {
                    s[base + j].add_mul(&t, xcd);
                }
            }
        }
    }
    // U = S² q(n_d0)² Π q(N);  V = (Δa Δb Δc)² p(n_d0)² Π D² p(N).
    let (uq_a, vp_a) = ca.radicand_parts();
    let (uq_b, vp_b) = cb.radicand_parts();
    let (uq_c, vp_c) = cc.radicand_parts();
    let (uq_d, vp_d) = cd.radicand_parts();
    let u_common = q_d0.square();
    let v_common = (&(&da * &db) * &dc).square() * p_d0.square();
    let mut entries = Vec::with_capacity(len);
    for (idx, x) in s.iter().enumerate() {
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
    let (raw, norms) = seeds(data, (&la, &lb, &lc), n)?;
    let ca = integral(raw, norms, st);
    let (raw, norms) = seeds(data, (&lb, &la, &lc), n)?;
    let cb = integral(raw, norms, st);
    let (nu_a, da) = integral_norms(&la);
    let (nu_b, db) = integral_norms(&lb);
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
    // U = T² q(n_c0)² q(N_μ) q(N'_ν);  V = (Δa Δb)² p(n_c0)² D_μ² p(N_μ) D'_ν² p(N'_ν).
    let (uq_a, vp_a) = ca.radicand_parts();
    let (uq_b, vp_b) = cb.radicand_parts();
    let u_common = q_c0.square();
    let v_common = (&da * &db).square() * p_c0.square();
    let mut entries = Vec::with_capacity(len);
    for (idx, x) in t.iter().enumerate() {
        let (mu, nu) = (idx / n, idx % n);
        let u = &(&u_common * &uq_a[mu]) * &uq_b[nu];
        let v = &(&v_common * &vp_a[mu]) * &vp_b[nu];
        entries.push(radical(x, &u, v, st));
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

fn block_f(g: &LieGroup, labels: [&LieIrrep; 6], st: &mut ProbeStats) -> Result<Block, LieError> {
    if let Some(product) = &g.composite {
        let mut acc: Option<Block> = None;
        for (i, factor) in product.factors().iter().enumerate() {
            let fl = labels.map(|x| &x.factors().expect("composite label")[i]);
            let block = block_f(factor, fl, st)?;
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
            simple_f(g.engine(), ls, st)
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

/// `F^{abc}_d[e, f]` evaluated exactly and rounded once. Reads modules
/// through the CGC tier (reset it for a cold measurement) and retains no
/// exact data.
pub fn exact_f(g: &LieGroup, labels: [&LieIrrep; 6]) -> Result<ProbeBlock, LieError> {
    let mut st = ProbeStats::default();
    let block = block_f(g, labels, &mut st)?;
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
