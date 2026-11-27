//! The rounding layer of the Cartan CGC: the rounded view of an exact
//! channel ([`round_channel`]), and the gates defined on rounded channels
//! (`docs/gauge_cartan.md` §7, gates 4 and 5: [`check_rounded`]). Gates are
//! sanity checks of a rounded channel, never certificates of its rounding,
//! and never change a value.

use std::borrow::Cow;
use std::collections::HashMap;

use threadteam::Exec;

use super::estimate::{self, Est};
use super::{round_radical, Factor};
use crate::exact::radical::Radicand;
use crate::frcore::{GateError, MEntry};
use crate::lie::general::cgc::exact::{ColumnCoords, ExactCgc, NormTable};
use crate::lie::general::cgc::{CgcChannel, Ladders};
use crate::lie::general::module::Q;
use crate::lie::general::roots::CartanData;
use malachite_nz::natural::Natural;

/// The rounded view of `exact`: one conversion per entry,
/// `sign(X)·√(X² ρ_μ n^a n^b/n^c)` with the norms of `norms` at the entry's
/// public indices, in the channel's §4.7 order. An entry is kept even when it
/// rounds to zero; an exactly zero coordinate has no entry.
///
/// The norms are resolved by the caller, so no participant reads a cache.
/// The items are the channel's columns ([`round_column`]), as one stage of
/// [`crate::stage::run_items`] whose planner prices them by their entries;
/// each column writes its own range of the one view allocation, so nothing
/// is merged. Every entry is the same conversion at every width, so the bits
/// are too. A freshly generated channel is rounded column by column in the
/// stage that writes it instead (`lie::general::family`).
pub(crate) fn round_channel(
    exact: &ExactCgc,
    norms: [&NormTable; 3],
    exec: Option<&mut Exec<'_>>,
) -> CgcChannel {
    use crate::stage::{part_units, run_items, split_parts, take_part, StageBuf};
    let ends = exact.column_ends();
    let units = part_units(ends);
    let estimates = ChannelEstimates::for_channel(exact.vertices(), norms, &units);
    let mut entries = view_slots(exact.len());
    {
        let parts = split_parts(&mut entries, ends.iter().copied());
        let _width = run_items(exec, &units, &mut StageBuf::default(), |k| {
            round_column(
                &exact.column_coords(k),
                norms,
                estimates.as_ref(),
                take_part(&parts, k),
            );
        });
        #[cfg(test)]
        if _width.get() > 1 {
            ROUND_SECTIONS.with(|s| s.set(s.get() + 1));
        }
    }
    CgcChannel {
        mult: exact.mult(),
        entries: entries.into_boxed_slice(),
    }
}

/// The `len` entry slots of a view, every one overwritten by its column.
///
/// Why not leave them uninitialized for the columns to fill in parallel: a
/// `Vec` of written-but-unset elements needs `unsafe` to assume them
/// initialized, and an `MEntry` has no zeroed allocation in `std`; so the
/// slots are filled once here, a memset of the view.
pub(crate) fn view_slots(len: usize) -> Vec<MEntry> {
    let unset = MEntry {
        m1: 0,
        m2: 0,
        m3: 0,
        mu: 0,
        value: 0.0,
    };
    vec![unset; len]
}

/// A norm's numerator and denominator as conversion factors: words for a
/// word value, its limbs in place otherwise.
enum NormParts<'a> {
    Words([u64; 2]),
    Big(Cow<'a, Natural>, Cow<'a, Natural>),
}

impl<'a> NormParts<'a> {
    fn of(q: &'a Q) -> Self {
        match q.word_parts() {
            Some((_, p, d)) => Self::Words([p, d]),
            None => {
                let (p, d) = q.magnitudes();
                Self::Big(p, d)
            }
        }
    }

    fn factors(&self) -> [Factor<'_>; 2] {
        match self {
            Self::Words(w) => [Factor::Words(&w[..1]), Factor::Words(&w[1..])],
            Self::Big(p, d) => [Factor::Natural(p), Factor::Natural(d)],
        }
    }
}

/// `√n` (or `1/√n` when `inverse`) of a positive rational, `None` when an
/// exponent leaves the estimate's range.
fn norm_root(n: &Q, inverse: bool) -> Option<Est> {
    let parts = NormParts::of(n);
    let [p, q] = parts.factors();
    if inverse {
        estimate::root_of(q, p)
    } else {
        estimate::root_of(p, q)
    }
}

/// Nanoseconds the estimate saves per entry against [`exact_entry`],
/// conservatively. Measured on lg (Xeon Platinum 9242, release, one
/// thread): the exact conversion 194–305 ns per entry and the fast path
/// 32–45 ns on channels of at least 300 entries; on channels of 30–277
/// entries 263–471 ns against 92–112 ns. The smallest difference, about
/// 150 ns, is taken.
const SAVING_NS: u64 = 150;

/// Nanoseconds one exact root costs, conservatively: the stack root's cost
/// in the first roots of a fresh process, the larger of two lg jobs (Xeon
/// Platinum 9242, `reviews/fs-root-bench`: 214–230 ns warm, 240–370 ns cold,
/// no allocation). With [`SAVING_NS`] a root pays for itself from about 2.5
/// entries (`370/150`), one coupled root from 3. Roots of operands beyond 128
/// bits take the big-integer path (about 0.3–1.1 µs) but occur only on the
/// largest channels' vertices and norms, where the estimate pays anyway.
const ROOT_NS: u64 = 370;

/// Whether forming `roots` exact roots costs no more than the estimate saves
/// on the `entries` they serve.
fn pays(entries: u64, roots: u64) -> bool {
    entries.saturating_mul(SAVING_NS) >= roots.saturating_mul(ROOT_NS)
}

/// The roots one label's norms contribute: `√n_m` as a left or right
/// factor, `1/√n_m` as the coupled one.
type Roots = (Option<Est>, Option<Est>);

/// Which norm tables are the same allocation, as the CGC tier shares them:
/// whether `b` repeats `a`, and the factor table (`0` for `a`, `1` for a
/// distinct `b`) a repeated `c` reuses.
fn shared_labels([na, nb, nc]: [&NormTable; 3]) -> (bool, Option<usize>) {
    let b_is_a = std::ptr::eq(na, nb);
    let c = if std::ptr::eq(nc, na) {
        Some(0)
    } else if std::ptr::eq(nc, nb) {
        Some(1)
    } else {
        None
    };
    (b_is_a, c)
}

/// The factor estimates of one channel call, built serially before its
/// stage and never retained: per index of every distinct label among `a`,
/// `b` and `c` the roots it is used with (a repeated label shares one
/// table), and `√ρ_μ` per `μ`. A distinct `c` label's roots are formed per
/// column, inside the stage, since each `m_c` occurs in `N` columns only.
pub(crate) struct ChannelEstimates {
    tables: Vec<Box<[Roots]>>,
    /// The tables of `a` and `b`, and of `c` when it repeats either.
    of: [usize; 2],
    c: Option<usize>,
    vertices: Box<[Option<Est>]>,
    /// Whether a column's own coupled root is formed only when it [`pays`].
    gate_columns: bool,
}

impl ChannelEstimates {
    /// The estimates of a channel whose columns hold `columns` entries each,
    /// or `None` when its roots cost more than they save, when every entry
    /// takes [`exact_entry`].
    ///
    /// One comparison, [`pays`], at the two places roots are formed: a
    /// column of a distinct `c` label adds one root, so it uses the estimate
    /// when it pays alone; the channel's serial roots (the factor tables, the
    /// repeated `c`'s reciprocal roots and one per `μ`) are formed when they
    /// pay on the entries of the columns that use the estimate.
    pub(crate) fn for_channel<'a>(
        vertices: impl ExactSizeIterator<Item = &'a Radicand>,
        norms: [&NormTable; 3],
        columns: &[u64],
    ) -> Option<Self> {
        let (b_is_a, c) = shared_labels(norms);
        let [da, db] = [norms[0].norm.len(), norms[1].norm.len()].map(|d| d as u64);
        let tables = da + if b_is_a { 0 } else { db };
        let inverse = match c {
            Some(0) => da,
            Some(_) => db,
            None => 0,
        };
        let roots = tables + inverse + vertices.len() as u64;
        let served: u64 = columns.iter().filter(|&&n| c.is_some() || pays(n, 1)).sum();
        pays(served, roots).then(|| Self::new(vertices, norms, true))
    }

    /// The estimates for a channel with vertices `vertices` (in `μ` order)
    /// over the norm tables `[a, b, c]`, every root formed; with
    /// `gate_columns`, a distinct `c` label's column forms its root only
    /// when it [`pays`].
    fn new<'a>(
        vertices: impl Iterator<Item = &'a Radicand>,
        norms: [&NormTable; 3],
        gate_columns: bool,
    ) -> Self {
        let (b_is_a, c) = shared_labels(norms);
        let [na, nb, nc] = norms;
        let table = |t: &NormTable| -> Box<[Roots]> {
            let inverse = std::ptr::eq(t, nc);
            t.norm
                .iter()
                .map(|n| {
                    let inv = if inverse { norm_root(n, true) } else { None };
                    (norm_root(n, false), inv)
                })
                .collect()
        };
        let mut tables = vec![table(na)];
        let b = if b_is_a {
            0
        } else {
            tables.push(table(nb));
            1
        };
        let c = c.map(|t| if t == 0 { 0 } else { b });
        let vertices = vertices
            .map(|r| estimate::root_of(Factor::Natural(r.num()), Factor::Natural(r.den())))
            .collect();
        Self {
            tables,
            of: [0, b],
            c,
            vertices,
            gate_columns,
        }
    }

    /// `√ρ_μ/√n^c_{m_c}` of the column, or `None` when a factor has no
    /// estimate or the column's own root does not pay.
    fn column(&self, column: &ColumnCoords<'_>, nc: &NormTable) -> Option<Est> {
        let mc = column.mc();
        let tc = match self.c {
            Some(t) => self.tables[t][mc].1?,
            None if self.gate_columns && !pays(column.pairs().len() as u64, 1) => return None,
            None => norm_root(&nc.norm[mc], true)?,
        };
        estimate::mul(self.vertices[column.mu()]?, tc)
    }

    /// The fast value of entry `j` of a column whose constant is `k`
    /// ([`Self::column`]), or `None` when the estimate cannot certify it.
    fn entry(&self, k: Option<Est>, column: &ColumnCoords<'_>, j: usize) -> Option<f64> {
        let (ma, mb) = column.pairs()[j];
        let sa = self.tables[self.of[0]][ma as usize].0?;
        let sb = self.tables[self.of[1]][mb as usize].0?;
        let v = estimate::entry(column.magnitude(j), k?, sa, sb)?;
        estimate::decide(column.is_negative(j), v)
    }
}

/// The exact value of entry `j` of `column`: [`round_radical`] over the
/// full factor lists, the definition every fast value equals.
fn exact_entry(column: &ColumnCoords<'_>, [na, nb, nc]: [&NormTable; 3], j: usize) -> f64 {
    let (ma, mb) = column.pairs()[j];
    let a = NormParts::of(&na.norm[ma as usize]);
    let b = NormParts::of(&nb.norm[mb as usize]);
    let c = NormParts::of(&nc.norm[column.mc()]);
    let (num, den) = radical_factors(column, j, [&a, &b, &c]);
    round_radical(if column.is_negative(j) { -1 } else { 1 }, &num, &den)
}

/// The radicand of entry `j` of `column`, `X² ρ_μ n^a n^b / n^c`, as factor
/// lists, with the norms at the entry's public indices.
fn radical_factors<'a>(
    column: &ColumnCoords<'a>,
    j: usize,
    [a, b, c]: [&'a NormParts<'a>; 3],
) -> ([Factor<'a>; 6], [Factor<'a>; 4]) {
    let vertex = column.vertex();
    let x = column.magnitude(j);
    let ([pa, qa], [pb, qb], [pc, qc]) = (a.factors(), b.factors(), c.factors());
    (
        [
            Factor::Words(x),
            Factor::Words(x),
            Factor::Natural(vertex.num()),
            pa,
            pb,
            qc,
        ],
        [Factor::Natural(vertex.den()), qa, qb, pc],
    )
}

/// A product leaf's exact channel and the norm tables its entries read
/// (`None` for an empty channel).
pub(crate) struct ExactLeaf {
    pub(crate) exact: std::sync::Arc<ExactCgc>,
    pub(crate) norms: Option<[std::sync::Arc<NormTable>; 3]>,
}

/// The composite channel of a product's Radical leaves, `None` when a
/// composite axis of a nonempty channel exceeds `u32`.
///
/// Every combination of one exact entry per leaf is one product entry, the
/// single radical `Π sign · √(Π X² ρ_μ n^a n^b / n^c)` over the
/// concatenated factor lists, rounded once by [`round_radical`]: no rounded
/// leaf value is multiplied. Indices combine last leaf fastest on every axis,
/// `μ` included, and the entries are sorted into §4.7 order. The cost is one
/// conversion per product entry, `Π nnz_i` of them.
pub(crate) fn compose_channel(leaves: &[ExactLeaf]) -> Option<CgcChannel> {
    let shape = |l: &ExactLeaf| {
        let [a, b, c] = l.exact.dims();
        [a, b, c, l.exact.mult()]
    };
    // An empty leaf (`N = 0`) makes the channel empty whatever the other
    // axes, so it is decided before any axis is multiplied.
    if leaves.iter().any(|l| l.exact.is_empty()) {
        return Some(CgcChannel {
            mult: 0,
            entries: Box::new([]),
        });
    }
    let mut dims = [1usize; 4];
    for l in leaves {
        for (d, x) in dims.iter_mut().zip(shape(l)) {
            *d = d.checked_mul(x)?;
        }
    }
    if dims.iter().any(|&d| d > u32::MAX as usize) {
        return None;
    }
    struct Entry<'a> {
        at: [u32; 4],
        negative: bool,
        num: [Factor<'a>; 6],
        den: [Factor<'a>; 4],
    }
    let parts: Vec<[Vec<NormParts<'_>>; 3]> = leaves
        .iter()
        .map(|l| {
            let norms = l.norms.as_ref().expect("a nonempty channel has norms");
            norms
                .each_ref()
                .map(|t| t.norm.iter().map(NormParts::of).collect())
        })
        .collect();
    let lists: Vec<Vec<Entry<'_>>> = leaves
        .iter()
        .zip(&parts)
        .map(|(l, [na, nb, nc])| {
            let mut out = Vec::with_capacity(l.exact.len());
            for k in 0..l.exact.column_ends().len() {
                let column = l.exact.column_coords(k);
                let (mu, mc) = (column.mu() as u32, column.mc());
                for (j, &(ma, mb)) in column.pairs().iter().enumerate() {
                    let norms = [&na[ma as usize], &nb[mb as usize], &nc[mc]];
                    let (num, den) = radical_factors(&column, j, norms);
                    out.push(Entry {
                        at: [ma, mb, mc as u32, mu],
                        negative: column.is_negative(j),
                        num,
                        den,
                    });
                }
            }
            out
        })
        .collect();
    let shapes: Vec<[usize; 4]> = leaves.iter().map(shape).collect();
    let mut entries = Vec::with_capacity(lists.iter().map(Vec::len).product());
    let (mut num, mut den) = (Vec::new(), Vec::new());
    let mut pick = vec![0usize; lists.len()];
    'odometer: loop {
        let (mut at, mut negative) = ([0usize; 4], false);
        num.clear();
        den.clear();
        for ((list, &i), d) in lists.iter().zip(&pick).zip(&shapes) {
            let e = &list[i];
            for axis in 0..4 {
                at[axis] = at[axis] * d[axis] + e.at[axis] as usize;
            }
            negative ^= e.negative;
            num.extend_from_slice(&e.num);
            den.extend_from_slice(&e.den);
        }
        // Every composite index is below its axis, which fits `u32`.
        let [m1, m2, m3, mu] = at.map(|x| x as u32);
        entries.push(MEntry {
            m1,
            m2,
            m3,
            mu,
            value: round_radical(if negative { -1 } else { 1 }, &num, &den),
        });
        for (i, list) in pick.iter_mut().zip(&lists).rev() {
            *i += 1;
            if *i < list.len() {
                continue 'odometer;
            }
            *i = 0;
        }
        break;
    }
    entries.sort_unstable_by_key(|e| (e.mu, e.m3, e.m1, e.m2));
    Some(CgcChannel {
        mult: dims[3],
        entries: entries.into_boxed_slice(),
    })
}

/// Per entry of `exact`, in order: the bits [`ChannelEstimates`] certifies,
/// or `None`, and the bits of [`exact_entry`]; every root is formed, whatever
/// [`pays`] would decide, so every entry is tried on the estimate.
#[cfg(test)]
pub(crate) fn fast_and_exact(exact: &ExactCgc, norms: [&NormTable; 3]) -> Vec<(Option<u64>, u64)> {
    let estimates = ChannelEstimates::new(exact.vertices(), norms, false);
    let mut out = Vec::new();
    for k in 0..exact.column_ends().len() {
        let column = exact.column_coords(k);
        let kc = estimates.column(&column, norms[2]);
        for j in 0..column.pairs().len() {
            out.push((
                estimates.entry(kc, &column, j).map(f64::to_bits),
                exact_entry(&column, norms, j).to_bits(),
            ));
        }
    }
    out
}

/// Rounds the column `column` into `out`, its entries: the certified
/// estimate of [`ChannelEstimates`] where the channel has one and it decides,
/// [`exact_entry`] otherwise, with the same bits.
pub(crate) fn round_column(
    column: &ColumnCoords<'_>,
    norms: [&NormTable; 3],
    estimates: Option<&ChannelEstimates>,
    out: &mut [MEntry],
) {
    let (mu, mc) = (column.mu(), column.mc());
    let k = estimates.and_then(|e| e.column(column, norms[2]));
    for (j, (slot, &(ma, mb))) in out.iter_mut().zip(column.pairs()).enumerate() {
        let value = estimates
            .and_then(|e| e.entry(k, column, j))
            .unwrap_or_else(|| exact_entry(column, norms, j));
        // The multiplicity and every dimension fit `u32` (§7 gate 1).
        *slot = MEntry {
            m1: ma,
            m2: mb,
            m3: mc as u32,
            mu: mu as u32,
            value,
        };
    }
}

impl Ladders {
    /// `⟨u_m, f_j u_src⟩ = x·√(n_m/n_src)` rounded once, for an `(m, x)`
    /// entry of `lower[j][src]`.
    pub(crate) fn rounded_lower(&self, src: usize, m: usize, x: &Q) -> f64 {
        let sign = match x.signum() {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        };
        let (px, qx) = x.magnitudes();
        let (pm, qm) = self.norm[m].magnitudes();
        let (ps, qs) = self.norm[src].magnitudes();
        round_radical(
            sign,
            &[
                Factor::Natural(&px),
                Factor::Natural(&px),
                Factor::Natural(&pm),
                Factor::Natural(&qs),
            ],
            &[
                Factor::Natural(&qx),
                Factor::Natural(&qx),
                Factor::Natural(&qm),
                Factor::Natural(&ps),
            ],
        )
    }
}

/// A rounded entry in one output weight block: `((m_a, m_b), column, value)`.
pub(crate) type BlockEntry = ((u32, u32), usize, f64);

/// Verification tolerance of the isometry and ladder gates (not value-fixing).
const TOL_ROUNDED: f64 = 1e-9;

/// The largest `|x|` over `residuals`, with a nonfinite residual a failure of
/// `gate` (a float `max` would hide NaN).
pub(crate) fn worst(
    gate: &'static str,
    residuals: impl Iterator<Item = f64>,
) -> Result<(), GateError> {
    let mut worst = 0.0f64;
    for r in residuals {
        if !r.is_finite() {
            return Err(GateError::NonFinite { gate });
        }
        worst = worst.max(r.abs());
    }
    if worst > TOL_ROUNDED {
        return Err(GateError::Residual {
            gate,
            residual: worst,
        });
    }
    Ok(())
}

/// Element counts the isometry gate requests for its three buffers of one
/// weight block: the gathered block, its column order and its Gram matrix.
pub(crate) type GateCounts = [usize; 3];

/// Element sizes of the [`GateCounts`] buffers.
const GATE_SIZES: [usize; 3] = [
    std::mem::size_of::<BlockEntry>(),
    std::mem::size_of::<(usize, usize, usize)>(),
    std::mem::size_of::<f64>(),
];

/// Requested bytes of buffers with element counts `c`. Allocator slack is
/// outside the account, as in the cache resource contract.
pub(crate) fn gate_bytes(c: GateCounts) -> usize {
    c.iter()
        .zip(GATE_SIZES)
        .fold(0usize, |t, (&n, s)| t.saturating_add(n.saturating_mul(s)))
}

/// The buffers a block needing `need` replaces when `held` are requested and
/// the participant's charge is `high`, already raised to `gate_bytes(need)`.
///
/// While the componentwise maxima fit the charge, only a buffer that must
/// grow is replaced. Otherwise every buffer of another size is replaced
/// exactly, so the held counts become `need`. Either way the held bytes stay
/// within `high`, also mid-replacement, since every replaced buffer is
/// dropped before any is allocated.
pub(crate) fn gate_replaced(held: GateCounts, need: GateCounts, high: usize) -> [bool; 3] {
    let kept = std::array::from_fn(|c| held[c].max(need[c]));
    let fits = gate_bytes(kept) <= high;
    std::array::from_fn(|c| need[c] > held[c] || (!fits && held[c] != need[c]))
}

/// One participant's isometry-gate buffers, with their requested element
/// counts and its high-water charge: the largest `gate_bytes` of a block it
/// has run. Its live scratch never exceeds that charge, so participants with
/// disjoint blocks hold at most the sum of their largest blocks.
#[derive(Default)]
pub(crate) struct GateScratch {
    block: Vec<BlockEntry>,
    order: Vec<(usize, usize, usize)>,
    gram: Vec<f64>,
    held: GateCounts,
    high: usize,
}

impl GateScratch {
    /// Makes room for a block needing `need`. Accounting uses the counts
    /// requested here, never `capacity()`; every fill stays within them, so
    /// no fill requests memory.
    fn reserve(&mut self, need: GateCounts) {
        self.high = self.high.max(gate_bytes(need));
        let replace = gate_replaced(self.held, need, self.high);
        // Release first: an assignment would hold the old and new buffer.
        if replace[0] {
            self.block = Vec::new();
        }
        if replace[1] {
            self.order = Vec::new();
        }
        if replace[2] {
            self.gram = Vec::new();
        }
        if replace[0] {
            self.block = Vec::with_capacity(need[0]);
        }
        if replace[1] {
            self.order = Vec::with_capacity(need[1]);
        }
        if replace[2] {
            self.gram = Vec::with_capacity(need[2]);
        }
        for c in 0..3 {
            if replace[c] {
                self.held[c] = need[c];
            }
        }
    }
}

/// The isometry block of one output weight: `k` columns from column run
/// `offset`, holding `e` entries.
#[derive(Clone, Copy)]
pub(crate) struct GateBlock {
    pub(crate) offset: usize,
    pub(crate) k: usize,
    pub(crate) e: usize,
}

impl GateBlock {
    pub(crate) fn need(self) -> GateCounts {
        [self.e, self.k, self.k * self.k]
    }
}

/// `max |MᵀM − I| ≤ 1e-9` over one weight block whose column runs are
/// `cols`, with its buffers in `s`.
pub(crate) fn isometry_block(
    s: &mut GateScratch,
    entries: &[MEntry],
    cols: &[(usize, usize)],
    w: GateBlock,
) -> Result<(), GateError> {
    let k = w.k;
    s.reserve(w.need());
    let GateScratch {
        block, order, gram, ..
    } = s;
    // Columns are gathered in their order in `entries`, so the unstable
    // sort below sees the same sequence as a scan of the whole list: a
    // duplicated (row, column) key then sums in the same order.
    order.clear();
    order.extend(
        cols.iter()
            .enumerate()
            .filter(|(_, &(s, t))| t > s)
            .map(|(col, &(s, t))| (s, t, col)),
    );
    order.sort_unstable();
    block.clear();
    for &(s, t, col) in order.iter() {
        block.extend(entries[s..t].iter().map(|e| ((e.m1, e.m2), col, e.value)));
    }
    gram.clear();
    gram.resize(k * k, 0.0);
    // Index a slice, not the scratch's `Vec`: indexing through `&mut Vec`
    // made the accumulation below measurably slower.
    let gram = &mut gram[..];
    block.sort_unstable_by_key(|&(row, col, _)| (row, col));
    for run in block.chunk_by(|x, y| x.0 == y.0) {
        for (i, &(_, ci, vi)) in run.iter().enumerate() {
            for &(_, cj, vj) in &run[i..] {
                gram[ci * k + cj] += vi * vj;
            }
        }
    }
    let gram = &*gram;
    worst(
        "isometry",
        (0..k).flat_map(|i| (i..k).map(move |j| gram[i * k + j] - f64::from(u8::from(i == j)))),
    )
}

/// The isometry gate over every weight block, in weight order. On a team
/// token the blocks are one stage of [`crate::stage::run_bounded_stage`]:
/// live scratch stays within the serial peak plus the CGC tier's budget, the
/// allowance [`crate::cache::CoefficientCacheBudgets`] documents for scratch
/// beyond the serial peak, and the first failing block in weight order is
/// the result at every width.
fn isometry_gate(
    c: &Ladders,
    n: usize,
    entries: &[MEntry],
    runs: &[(usize, usize)],
    exec: Option<&mut Exec<'_>>,
) -> Result<(), GateError> {
    let blocks = c.module.weights.iter().map(|space| {
        let k = space.norms.len() * n;
        let cols = &runs[space.offset * n..space.offset * n + k];
        let e = cols.iter().map(|&(s, t)| t - s).sum();
        GateBlock {
            offset: space.offset * n,
            k,
            e,
        }
    });
    let cols = |w: GateBlock| &runs[w.offset..w.offset + w.k];
    let exec = match exec {
        Some(exec) if exec.carries_team() && !exec.is_serial() => exec,
        _ => {
            let mut s = GateScratch::default();
            for w in blocks {
                isometry_block(&mut s, entries, cols(w), w)?;
            }
            return Ok(());
        }
    };
    use crate::stage::{memory_width, run_bounded_stage, StageBuf, SystemClock};
    let blocks: Vec<GateBlock> = blocks.collect();
    let sigma: Vec<usize> = blocks.iter().map(|&w| gate_bytes(w.need())).collect();
    // Work per block: the gather and its sort, then the Gram matrix.
    let units: Vec<u64> = blocks
        .iter()
        .map(|w| (w.e as u64).saturating_add((w.k as u64).saturating_mul(w.k as u64)))
        .collect();
    let allowance = gate_allowance();
    let ceiling = sigma
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .saturating_add(allowance);
    let memory = |rest: &[usize]| memory_width(&sigma, rest, ceiling);
    let mut buf = StageBuf::default();
    run_bounded_stage(
        exec,
        &units,
        &SystemClock,
        &memory,
        &mut None,
        &mut buf,
        GateScratch::default,
        |s, w| isometry_block(s, entries, cols(blocks[w]), blocks[w]),
    );
    buf.drain().collect()
}

/// Entries per item of the support stage on a team token: large enough to
/// amortize a stage item's scheduling and keep the stage's slots and queue
/// at `O(nnz / 2^16)`, while the planner prices dispatch. It is a fixed
/// grain, not a crossover threshold.
const SUPPORT_CHUNK: usize = 1 << 16;

/// The support predicate of one entry of `a ⊗ b → c` at multiplicity `n`, in
/// the gates' order: every index in range, before any engine lookup; a finite
/// value; the entry on its weight. Returns its column `p_c·n + μ` at engine
/// `p_c`.
///
/// Always inlined: the serial loop otherwise calls it per entry, reloading
/// the module dimensions and engine maps and returning through memory,
/// where the former inline loop hoisted them.
#[inline(always)]
fn entry_column(
    [a, b, c]: [&Ladders; 3],
    n: usize,
    [ea, eb, ec]: &[Vec<usize>; 3],
    e: &MEntry,
) -> Result<usize, GateError> {
    let (m1, m2, m3, mu) = (e.m1 as usize, e.m2 as usize, e.m3 as usize, e.mu as usize);
    if m1 >= a.dim() || m2 >= b.dim() || m3 >= c.dim() || mu >= n {
        return Err(GateError::Support);
    }
    if !e.value.is_finite() {
        return Err(GateError::NonFinite { gate: "isometry" });
    }
    let (pa, pb, pc) = (ea[m1], eb[m2], ec[m3]);
    let on_weight = a
        .weight(pa)
        .iter()
        .zip(b.weight(pb))
        .zip(c.weight(pc))
        .all(|((x, y), z)| x.checked_add(*y) == Some(*z));
    if !on_weight {
        return Err(GateError::Support);
    }
    Ok(pc * n + mu)
}

/// Extends column `col`'s run in `runs` by entry `i`, where `prev` is the
/// previous entry's column; a column that reappears after another is split,
/// a support failure.
fn extend_run(
    runs: &mut [(usize, usize)],
    prev: &mut usize,
    i: usize,
    col: usize,
) -> Result<(), GateError> {
    if col != *prev {
        if runs[col].1 != 0 {
            return Err(GateError::Support);
        }
        runs[col].0 = i;
        *prev = col;
    }
    runs[col].1 = i + 1;
    Ok(())
}

/// The support check of [`check_rounded`] and its column runs:
/// `runs[m_c·n + μ]` (engine `m_c`) is the column's `entries[start..end]`,
/// and `end == 0` marks an absent column.
///
/// Serially every entry passes [`entry_column`] and then extends its run, so
/// the first failing entry decides. On a team token the predicate runs as one
/// stage of [`crate::stage::run_stage`] over chunks of [`SUPPORT_CHUNK`]
/// entries, each returning its first failure; the leader then extends the
/// runs over the entries before the earliest failure `f` only. A split
/// column there precedes `f`; otherwise the failure at `f` is the serial one,
/// since the serial loop tests the predicate of an entry before its run. The
/// stage's slots, queue and units end here, before the isometry gate.
pub(crate) fn support_runs(
    modules: [&Ladders; 3],
    n: usize,
    engines: &[Vec<usize>; 3],
    entries: &[MEntry],
    exec: Option<&mut Exec<'_>>,
) -> Result<Vec<(usize, usize)>, GateError> {
    let exec = match exec {
        Some(exec) if exec.carries_team() && !exec.is_serial() => exec,
        _ => return serial_support_runs(modules[0], modules[1], modules[2], n, engines, entries),
    };
    let mut runs = vec![(0usize, 0usize); modules[2].dim() * n];
    let mut prev = usize::MAX;
    use crate::stage::{run_stage, StageBuf, SystemClock};
    let chunk = support_chunk();
    let units: Vec<u64> = entries.chunks(chunk).map(|x| x.len() as u64).collect();
    let mut buf = StageBuf::default();
    let _ran = run_stage(
        exec,
        &units,
        &SystemClock,
        &mut None,
        &mut buf,
        || (),
        |(), j| {
            let start = j * chunk;
            entries[start..(start + chunk).min(entries.len())]
                .iter()
                .enumerate()
                .find_map(|(i, e)| {
                    entry_column(modules, n, engines, e)
                        .err()
                        .map(|err| (start + i, err))
                })
        },
    );
    #[cfg(test)]
    if _ran.width.get() > 1 {
        SUPPORT_SECTIONS.with(|s| s.set(s.get() + 1));
    }
    let failure = buf.drain().flatten().next();
    let valid = failure.as_ref().map_or(entries.len(), |&(f, _)| f);
    let ec = &engines[2];
    for (i, e) in entries[..valid].iter().enumerate() {
        extend_run(
            &mut runs,
            &mut prev,
            i,
            ec[e.m3 as usize] * n + e.mu as usize,
        )?;
    }
    match failure {
        Some((_, err)) => Err(err),
        None => Ok(runs),
    }
}

/// The serial support check of [`support_runs`]: every entry passes
/// [`entry_column`] and then extends its run, so the first failing entry
/// decides.
///
/// Never inlined, and with each module a parameter of its own: inside
/// [`check_rounded`] the loop shares registers with the team path and the
/// later gates and spills its loop invariants to the stack, and modules
/// passed in one array are not known to be unaliased, so their weight tables
/// are reloaded on every entry. As here, the loop keeps them in registers,
/// as the former inline loop did.
#[inline(never)]
fn serial_support_runs(
    a: &Ladders,
    b: &Ladders,
    c: &Ladders,
    n: usize,
    engines: &[Vec<usize>; 3],
    entries: &[MEntry],
) -> Result<Vec<(usize, usize)>, GateError> {
    let mut runs = vec![(0usize, 0usize); c.dim() * n];
    let mut prev = usize::MAX;
    for (i, e) in entries.iter().enumerate() {
        let col = entry_column([a, b, c], n, engines, e)?;
        extend_run(&mut runs, &mut prev, i, col)?;
    }
    Ok(runs)
}

/// [`SUPPORT_CHUNK`], or this thread's test override.
fn support_chunk() -> usize {
    #[cfg(test)]
    if let Some(chunk) = SUPPORT_CHUNK_OVERRIDE.with(std::cell::Cell::get) {
        return chunk;
    }
    SUPPORT_CHUNK
}

/// The allowance for isometry-gate scratch beyond the serial peak.
fn gate_allowance() -> usize {
    #[cfg(test)]
    if let Some(a) = GATE_ALLOWANCE.with(std::cell::Cell::get) {
        return a;
    }
    crate::cache::cache_budgets().limit(crate::cache::CoefficientCacheTier::Cgc)
}

/// The rounded gates on one channel `a ⊗ b → c` of multiplicity `n`, with
/// `entries` at public magnetic indices:
///
/// - every entry in range, finite and on its weight, `wt(m_a) + wt(m_b) = wt(m_c)`;
/// - isometry, `max |MᵀM − I| ≤ 1e-9` with `M[(m_a, m_b), (m_c, μ)]` over every
///   expected column `(m_c, μ)`, including absent ones. Validated support makes
///   columns of different output weights disjoint, so the weight blocks are
///   the whole condition;
/// - ladder consistency at the highest weight: for each `μ` and each `j` with
///   `λ_c − α_j` a weight, `Δ(f_j) C[·, ·, top, μ] = Σ_m ⟨u_m, f_j u_top⟩
///   C[·, ·, m, μ]` within `1e-9`, with correctly rounded ladder entries.
///
/// Each column `(m_c, μ)` must be one contiguous run of `entries`, as the
/// §4.7 order makes it; a split column is a support failure. The gates then
/// read the entries in place through those runs: one weight block of
/// `BlockEntry` at a time, so no copy of the whole channel is made.
pub(crate) fn check_rounded(
    data: &CartanData,
    a: &Ladders,
    b: &Ladders,
    c: &Ladders,
    n: usize,
    entries: &[MEntry],
    mut exec: Option<&mut Exec<'_>>,
) -> Result<(), GateError> {
    let engines = [a.engine(), b.engine(), c.engine()];
    let runs = support_runs([a, b, c], n, &engines, entries, exec.as_deref_mut())?;
    let [ea, eb, _] = &engines;
    let column = |m: usize, mu: usize| {
        let (s, t) = runs[m * n + mu];
        &entries[s..t]
    };

    isometry_gate(c, n, entries, &runs, exec)?;

    // First descent: f_j u_top lies in the weight space λ_c − α_j, which
    // exists exactly when (λ_c)_j > 0 (an unbroken α_j-string).
    let top = &c.module.weights[0].mu;
    let mut descents: Vec<(usize, Vec<(usize, f64)>)> = Vec::new();
    for j in 0..top.len() {
        if top[j] <= 0 {
            continue;
        }
        let nu = (0..top.len())
            .map(|i| top[i].checked_sub(data.c(j, i)))
            .collect::<Option<Vec<i64>>>();
        let Some(w) = nu.and_then(|nu| c.module.weight_index(&nu)) else {
            return Err(GateError::Support);
        };
        let space = &c.module.weights[w];
        let ladder = (space.offset..space.offset + space.norms.len())
            .map(|m| {
                let x = c.lower[j][0].iter().find(|(t, _)| *t == m);
                (m, x.map_or(0.0, |(_, x)| c.rounded_lower(0, m, x)))
            })
            .collect();
        descents.push((j, ladder));
    }
    for (j, ladder) in &descents {
        for mu in 0..n {
            let mut diff: HashMap<(usize, usize), f64> = HashMap::new();
            for e in column(0, mu) {
                let (pa, pb, v) = (ea[e.m1 as usize], eb[e.m2 as usize], e.value);
                for (pa2, x) in &a.lower[*j][pa] {
                    *diff.entry((*pa2, pb)).or_default() += a.rounded_lower(pa, *pa2, x) * v;
                }
                for (pb2, x) in &b.lower[*j][pb] {
                    *diff.entry((pa, *pb2)).or_default() += b.rounded_lower(pb, *pb2, x) * v;
                }
            }
            for &(m, l) in ladder {
                for e in column(m, mu) {
                    *diff
                        .entry((ea[e.m1 as usize], eb[e.m2 as usize]))
                        .or_default() -= l * e.value;
                }
            }
            worst("ladder", diff.into_values())?;
        }
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// Rounding stages of [`round_channel`] on this thread that ran a
    /// section of width two or more.
    pub(crate) static ROUND_SECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Support stages of [`check_rounded`] on this thread that ran a section
    /// of width two or more.
    pub(crate) static SUPPORT_SECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Replaces [`SUPPORT_CHUNK`] on this thread.
    pub(crate) static SUPPORT_CHUNK_OVERRIDE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    /// Replaces the isometry gate's allowance beyond the serial peak.
    pub(crate) static GATE_ALLOWANCE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
mod tests;
