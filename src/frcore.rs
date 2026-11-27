//! The F- and R-symbol contractions of the Cartan engine
//! (`docs/gauge_cartan.md` §6). The gates on completed blocks (§7, gates 6
//! and 7) are in the rounding layer, [`crate::rounding::blocks`].
//!
//! The F-symbol is the contraction of four Clebsch–Gordan tensors over their
//! magnetic indices, leaving four outer multiplicity indices `[μ, ν, κ, λ]`
//! (SUNRepresentations.jl `sector.jl:_Fsymbol`, `:58-89`; the contraction wiring
//! and axis order match TensorKitSectors `sectors.jl:Fsymbol_from_fusiontensor`,
//! `:406-418`). The CGC are real, so conjugation is the identity and is elided.
//!
//! # Exact contraction: sparse joins in integers
//!
//! CGC are sparse, and each F/R element is a sum over matched magnetic-index
//! keys, so the contraction is a sequence of sparse joins over shared magnetic
//! indices. Every CGC entry is `X·√(ρ_μ n^a n^b/n^c)` with an integer `X`
//! (the exact channel, `docs/gauge_cartan.md` §9), and with `n = ν g/Δ` the
//! square roots of a contraction factor out: the entry is an integer sum `S`
//! under one radical per leg ([`RadicalBlock`]). The sums are exact, so their
//! order is free; the block is rounded once, in the rounding layer.

mod accum;

use std::collections::{HashMap, HashSet};

use malachite_base::num::arithmetic::traits::Gcd;
use malachite_nz::natural::Natural;

use crate::exact::limbs::{mul_into, trimmed_words};
use crate::exact::radical::{RadicalBlock, Radicand};
use crate::lie::general::cgc::exact::{ColumnCoords, ExactCgc, HighestColumn, NormTable};
use accum::{choose, Bounds, JoinAcc, Plan, Side, Signed, SumAcc};

/// One nonzero entry of a sparse CGC channel: `value` at public magnetic
/// indices `(m1, m2, m3)` and outer-multiplicity `mu` (`docs/gauge_cartan.md`
/// §4.7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MEntry {
    /// 0-based magnetic index in the left factor.
    pub m1: u32,
    /// 0-based magnetic index in the right factor.
    pub m2: u32,
    /// 0-based magnetic index in the coupled irrep.
    pub m3: u32,
    /// 0-based outer-multiplicity index (trailing axis).
    pub mu: u32,
    /// The coefficient, rounded once.
    pub value: f64,
}

use crate::lie::LieError;

/// A rounded CGC, F or R value that failed a generation gate. The gates only
/// verify; a failing value is never repaired or returned.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GateError {
    /// A CGC entry outside its index range or off its weight.
    Support,
    /// A nonfinite entry or residual in `gate`.
    NonFinite { gate: &'static str },
    /// The largest residual of `gate` exceeds its tolerance.
    Residual { gate: &'static str, residual: f64 },
}

impl std::fmt::Display for GateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Support => f.write_str("CGC entry outside its index range or weight"),
            Self::NonFinite { gate } => write!(f, "{gate} gate: nonfinite value"),
            Self::Residual { gate, residual } => {
                write!(f, "{gate} gate: residual {residual:e} exceeds tolerance")
            }
        }
    }
}

impl std::error::Error for GateError {}

// ---------------------------------------------------------------------------
// F (sector.jl:_Fsymbol, :58-89) and R (:91-110) in exact integers.
// ---------------------------------------------------------------------------

/// The inputs of one F block: the full channels `A = (a,b,e)` and
/// `C = (b,c,f)`, the seed columns `B = (e,c,d)` and `D = (a,f,d)` (engine
/// index 0 of `d`), the norm tables of `a, b, c, d` and the four
/// multiplicities.
pub(crate) struct FInputs<'a> {
    pub(crate) a: &'a ExactCgc,
    pub(crate) cb: &'a ExactCgc,
    pub(crate) b: &'a HighestColumn,
    pub(crate) d: &'a HighestColumn,
    pub(crate) norms: [&'a NormTable; 4],
    pub(crate) dims: [usize; 4],
}

/// The inputs of one R block: the seed columns of `(a,b,c)` and `(b,a,c)`,
/// the norm tables of `a, b, c` and `N^c_{ab}`.
pub(crate) struct RInputs<'a> {
    pub(crate) ab: &'a HighestColumn,
    pub(crate) ba: &'a HighestColumn,
    pub(crate) norms: [&'a NormTable; 3],
    pub(crate) n: usize,
}

/// Significant bits of little-endian words.
fn bit_len(w: &[u64]) -> u64 {
    let w = trimmed_words(w);
    w.last().map_or(0, |&top| {
        64 * w.len() as u64 - u64::from(top.leading_zeros())
    })
}

/// Entry `j` of `col`, read in place.
fn entry<'a>(col: &ColumnCoords<'a>, j: usize) -> Signed<'a> {
    Signed {
        neg: col.is_negative(j),
        mag: col.magnitude(j),
    }
}

/// Every entry `(m_a, m_b, m_c, μ, X)` of an exact channel.
fn channel_entries<'a>(
    ch: &'a ExactCgc,
) -> impl Iterator<Item = (u32, u32, u32, usize, Signed<'a>)> + 'a {
    (0..ch.column_ends().len()).flat_map(move |k| {
        let col = ch.column_coords(k);
        let (mc, mu) = (col.mc() as u32, col.mu());
        (0..col.pairs().len()).map(move |j| {
            let (ma, mb) = col.pairs()[j];
            (ma, mb, mc, mu, entry(&col, j))
        })
    })
}

/// Every entry `(m_a, m_b, μ, X)` of a seed set.
fn seed_entries(s: &HighestColumn) -> impl Iterator<Item = (u32, u32, usize, Signed<'_>)> + '_ {
    (0..s.mult()).flat_map(move |mu| {
        let col = s.column_coords(mu);
        (0..col.pairs().len()).map(move |j| {
            let (ma, mb) = col.pairs()[j];
            (ma, mb, mu, entry(&col, j))
        })
    })
}

fn seed_vertices(s: &HighestColumn) -> Vec<Radicand> {
    (0..s.mult())
        .map(|mu| s.column_coords(mu).vertex().clone())
        .collect()
}

/// `v·w` into `v`, through `tmp`.
fn scale(v: &mut Vec<u64>, w: &[u64], tmp: &mut Vec<u64>) {
    tmp.clear();
    tmp.resize(v.len() + w.len(), 0);
    mul_into(v, w, tmp);
    let len = trimmed_words(tmp).len();
    tmp.truncate(len);
    std::mem::swap(v, tmp);
}

/// `Π_k ν_k[m_k]` into `out`.
fn weight(out: &mut Vec<u64>, parts: &[&[u64]], tmp: &mut Vec<u64>) {
    out.clear();
    out.push(1);
    for p in parts {
        scale(out, p, tmp);
    }
}

/// `ρ_common = (q·Π g / (Π Δ · p))²` for the top norm `p/q` of the coupled
/// label and the tables of the summed labels, reduced by one gcd.
fn common(top: &NormTable, summed: &[&NormTable]) -> Result<Radicand, LieError> {
    let (p, q) = top.norm[top.top as usize].magnitudes();
    let (mut num, mut den): (Natural, Natural) = (q.into_owned(), p.into_owned());
    for t in summed {
        num *= t.content();
        den *= t.den();
    }
    let g = (&num).gcd(&den);
    let (num, den) = (num / &g, den / &g);
    Radicand::new(&num * &num, &den * &den).ok_or(LieError::LabelOverflow)
}

/// `F^{abc}_d[e, f]` as an exact radical block (`docs/gauge_cartan.md` §9):
///
/// ```text
/// AB[(m_a,m_b,m_c)][μ,ν] = Σ_{m_e} X_A[m_a,m_b,m_e,μ]·X_B[m_e,m_c,ν]
/// CD[(m_a,m_b,m_c)][κ,λ] = Σ_{m_f} X_C[m_b,m_c,m_f,κ]·X_D[m_a,m_f,λ]
/// S[μ,ν,κ,λ]             = Σ_{(m_a,m_b,m_c)} ν_a ν_b ν_c·AB·CD
/// ```
///
/// with entry `sign(S)·√(S²·ρ_common·ρ_A[μ]ρ_B[ν]ρ_C[κ]ρ_D[λ])`. An empty
/// vertex gives the zero block of its dimensions (the reference semantics).
/// Allocates per block, never per term, unless a bound selects the
/// malachite path ([`accum::Plan::Big`]), which is slow.
pub(crate) fn f_exact(x: &FInputs<'_>) -> Result<RadicalBlock<4>, LieError> {
    let dims = x.dims;
    if dims.contains(&0) {
        return RadicalBlock::zeros(dims).ok_or(LieError::LabelOverflow);
    }
    let [n1, n2, n3, n4] = dims;
    let (n12, n34) = (n1 * n2, n3 * n4);
    let mut bits = [0u64; 4];
    // B by m_e → (m_c, ν, X); D by m_f → (m_a, λ, X).
    let mut b_by: HashMap<u32, Vec<(u32, usize, Signed<'_>)>> = HashMap::new();
    for (me, mc, nu, s) in seed_entries(x.b) {
        bits[1] = bits[1].max(bit_len(s.mag));
        b_by.entry(me).or_default().push((mc, nu, s));
    }
    let mut d_by: HashMap<u32, Vec<(u32, usize, Signed<'_>)>> = HashMap::new();
    for (ma, mf, lambda, s) in seed_entries(x.d) {
        bits[3] = bits[3].max(bit_len(s.mag));
        d_by.entry(mf).or_default().push((ma, lambda, s));
    }
    // The keys `(m_a, m_b, m_c)` both joins reach.
    let mut ab_keys = HashSet::new();
    for (ma, mb, me, _, s) in channel_entries(x.a) {
        bits[0] = bits[0].max(bit_len(s.mag));
        for &(mc, ..) in b_by.get(&me).into_iter().flatten() {
            ab_keys.insert((ma, mb, mc));
        }
    }
    let mut keys = HashMap::new();
    for (mb, mc, mf, _, s) in channel_entries(x.cb) {
        bits[2] = bits[2].max(bit_len(s.mag));
        for &(ma, ..) in d_by.get(&mf).into_iter().flatten() {
            if ab_keys.contains(&(ma, mb, mc)) {
                let next = keys.len();
                keys.entry((ma, mb, mc)).or_insert(next);
            }
        }
    }
    drop(ab_keys);
    let [na, nb, nc, nd] = x.norms;
    let plan = choose(&Bounds {
        sides: [
            Side::Joined {
                bits: [bits[0], bits[1]],
                terms: b_by.len() as u64,
            },
            Side::Joined {
                bits: [bits[2], bits[3]],
                terms: d_by.len() as u64,
            },
        ],
        w_bits: na.nu().max_bits() + nb.nu().max_bits() + nc.nu().max_bits(),
        terms: keys.len() as u64,
    });
    let cells = |n: usize| crate::lie::checked_size([keys.len().max(1), n]);
    let mut ab = JoinAcc::new(&plan, cells(n12)?);
    for (ma, mb, me, mu, sa) in channel_entries(x.a) {
        for &(mc, nu, sb) in b_by.get(&me).into_iter().flatten() {
            if let Some(&i) = keys.get(&(ma, mb, mc)) {
                ab.add(i * n12 + mu * n2 + nu, sa, sb);
            }
        }
    }
    let mut cd = JoinAcc::new(&plan, cells(n34)?);
    for (mb, mc, mf, kappa, sc) in channel_entries(x.cb) {
        for &(ma, lambda, sd) in d_by.get(&mf).into_iter().flatten() {
            if let Some(&i) = keys.get(&(ma, mb, mc)) {
                cd.add(i * n34 + kappa * n4 + lambda, sc, sd);
            }
        }
    }
    let w_on_left = !matches!(
        plan,
        Plan::Fixed {
            w_on_left: false,
            ..
        }
    );
    let mut sum = SumAcc::new(&plan, crate::lie::checked_size(dims)?);
    let (mut l, mut r) = (
        vec![(false, Vec::new()); n12],
        vec![(false, Vec::new()); n34],
    );
    let (mut w, mut tmp) = (Vec::new(), Vec::new());
    for (&(ma, mb, mc), &i) in &keys {
        let parts = [
            na.nu().magnitude(ma as usize),
            nb.nu().magnitude(mb as usize),
            nc.nu().magnitude(mc as usize),
        ];
        weight(&mut w, &parts, &mut tmp);
        for (c, (neg, mag)) in l.iter_mut().enumerate() {
            *neg = ab.take(i * n12 + c, mag);
        }
        for (c, (neg, mag)) in r.iter_mut().enumerate() {
            *neg = cd.take(i * n34 + c, mag);
        }
        for (_, mag) in if w_on_left { &mut l } else { &mut r } {
            scale(mag, &w, &mut tmp);
        }
        for (p, (ln, lm)) in l.iter().enumerate().filter(|(_, (_, m))| !m.is_empty()) {
            for (q, (rn, rm)) in r.iter().enumerate().filter(|(_, (_, m))| !m.is_empty()) {
                sum.add(
                    p * n34 + q,
                    Signed { neg: *ln, mag: lm },
                    Signed { neg: *rn, mag: rm },
                );
            }
        }
    }
    let legs = [
        x.a.vertices().cloned().collect(),
        seed_vertices(x.b),
        x.cb.vertices().cloned().collect(),
        seed_vertices(x.d),
    ];
    RadicalBlock::from_legs(dims, sum.finish(), legs, common(nd, &[na, nb, nc])?)
        .ok_or(LieError::LabelOverflow)
}

/// `R^{ab}_c` as an exact radical block:
/// `T[μ,ν] = Σ_{(m_a,m_b)} ν_a ν_b·X_{ab}[m_a,m_b,μ]·X_{ba}[m_b,m_a,ν]` over
/// the seed pairs (engine index 0 of `c`), with entry
/// `sign(T)·√(T²·ρ_common·ρ_{ab}[μ]ρ_{ba}[ν])`. `N = 0` is the empty block.
pub(crate) fn r_exact(x: &RInputs<'_>) -> Result<RadicalBlock<2>, LieError> {
    let n = x.n;
    if n == 0 {
        return RadicalBlock::zeros([0, 0]).ok_or(LieError::LabelOverflow);
    }
    let mut bits = [0u64; 2];
    // `(b,a,c)` by `(m_a, m_b)` → (ν, X).
    let mut by: HashMap<(u32, u32), Vec<(usize, Signed<'_>)>> = HashMap::new();
    for (mb, ma, nu, s) in seed_entries(x.ba) {
        bits[1] = bits[1].max(bit_len(s.mag));
        by.entry((ma, mb)).or_default().push((nu, s));
    }
    for (.., s) in seed_entries(x.ab) {
        bits[0] = bits[0].max(bit_len(s.mag));
    }
    let [na, nb, nc] = x.norms;
    let plan = choose(&Bounds {
        sides: [
            Side::Direct { bits: bits[0] },
            Side::Direct { bits: bits[1] },
        ],
        w_bits: na.nu().max_bits() + nb.nu().max_bits(),
        terms: by.len() as u64,
    });
    let w_on_left = !matches!(
        plan,
        Plan::Fixed {
            w_on_left: false,
            ..
        }
    );
    let mut sum = SumAcc::new(&plan, crate::lie::checked_size([n, n])?);
    let (mut w, mut tmp, mut side) = (Vec::new(), Vec::new(), Vec::new());
    for (ma, mb, mu, sa) in seed_entries(x.ab) {
        let Some(list) = by.get(&(ma, mb)) else {
            continue;
        };
        let parts = [
            na.nu().magnitude(ma as usize),
            nb.nu().magnitude(mb as usize),
        ];
        weight(&mut w, &parts, &mut tmp);
        for &(nu, sb) in list {
            let (scaled, other) = if w_on_left { (sa, sb) } else { (sb, sa) };
            side.clear();
            side.extend_from_slice(trimmed_words(scaled.mag));
            scale(&mut side, &w, &mut tmp);
            let scaled = Signed {
                neg: scaled.neg,
                mag: &side,
            };
            let (l, r) = if w_on_left {
                (scaled, other)
            } else {
                (other, scaled)
            };
            sum.add(mu * n + nu, l, r);
        }
    }
    let legs = [seed_vertices(x.ab), seed_vertices(x.ba)];
    RadicalBlock::from_legs([n, n], sum.finish(), legs, common(nc, &[na, nb])?)
        .ok_or(LieError::LabelOverflow)
}
