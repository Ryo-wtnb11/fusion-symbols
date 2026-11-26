//! The rounding layer of the Cartan CGC: the rounded view of an exact
//! channel ([`round_channel`]), and the gates defined on rounded channels
//! (`docs/gauge_cartan.md` §7, gates 4 and 5: [`check_rounded`]). Gates are
//! sanity checks of a rounded channel, never certificates of its rounding,
//! and never change a value.

use std::collections::HashMap;

use threadteam::Exec;

use super::{round_radical, Factor, SharedFactors};
use crate::frcore::{GateError, MEntry};
use crate::lie::general::cgc::exact::{ExactCgc, NormTable};
use crate::lie::general::cgc::{CgcChannel, Ladders};
use crate::lie::general::module::Q;
use crate::lie::general::roots::CartanData;

/// The rounded view of `exact`: one conversion per entry,
/// `sign(X)·√(X² ρ_μ n^a n^b/n^c)` with the norms of `norms` at the entry's
/// public indices, in the channel's §4.7 order. An entry is kept even when it
/// rounds to zero; an exactly zero coordinate has no entry.
///
/// The norms are resolved by the caller, so no participant reads a cache.
/// The items are the channel's columns, the grain of the descent's phase B
/// that emitted them, as one stage of [`crate::stage::run_items`] whose
/// planner prices them by their entries; each column writes its own range of
/// the one view allocation, so nothing is merged. Every entry is the same
/// conversion at every width, so the bits are too.
pub(crate) fn round_channel(
    exact: &ExactCgc,
    norms: [&NormTable; 3],
    exec: Option<&mut Exec<'_>>,
) -> CgcChannel {
    use crate::stage::{run_items, split_parts, take_part, StageBuf};
    let ends = exact.column_ends();
    let mut start = 0;
    let units: Vec<u64> = ends
        .iter()
        .map(|&end| {
            let n = (end - start) as u64;
            start = end;
            n
        })
        .collect();
    // Every entry is overwritten by its column below.
    let unset = MEntry {
        m1: 0,
        m2: 0,
        m3: 0,
        mu: 0,
        value: 0.0,
    };
    let mut entries = vec![unset; exact.len()];
    {
        let parts = split_parts(&mut entries, ends.iter().copied());
        let _width = run_items(exec, &units, &mut StageBuf::default(), |k| {
            round_column(exact, norms, k, take_part(&parts, k));
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

/// Rounds column `k = μ·d_c + m_c` of `exact` into `out`, its entries.
///
/// `ρ_μ` and `n^c` are the column's constants, multiplied once into
/// [`SharedFactors`], so an entry multiplies only `X²` and the words of
/// `n^a n^b`; an entry outside that stack path converts all its factors in
/// one [`round_radical`], with the same bits.
fn round_column(exact: &ExactCgc, [na, nb, nc]: [&NormTable; 3], k: usize, out: &mut [MEntry]) {
    let dc = exact.dims()[2];
    let (mu, mc) = (k / dc, k % dc);
    let coords = exact.coords(mu);
    let base = exact.mu_range(mu).start;
    let (rn, rd) = (coords.vertex.num(), coords.vertex.den());
    let (pc, qc) = nc.norm[mc].magnitudes();
    let shared = SharedFactors::new(
        &[Factor::Natural(rn), Factor::Natural(&qc)],
        &[Factor::Natural(rd), Factor::Natural(&pc)],
    );
    for (slot, i) in out.iter_mut().zip(exact.column(k)) {
        let j = i - base;
        let (ma, mb) = exact.m_ab()[i];
        let x = coords.x.magnitude(j);
        let sign = if coords.x.is_negative(j) { -1 } else { 1 };
        let (a, b) = (&na.norm[ma as usize], &nb.norm[mb as usize]);
        let fast = match (shared.as_ref(), a.word_parts(), b.word_parts()) {
            (Some(s), Some((_, pa, qa)), Some((_, pb, qb))) => {
                s.round_square(sign, x, [pa, pb], [qa, qb])
            }
            _ => None,
        };
        let value = fast.unwrap_or_else(|| {
            let (pa, qa) = a.magnitudes();
            let (pb, qb) = b.magnitudes();
            round_radical(
                sign,
                &[
                    Factor::Words(x),
                    Factor::Words(x),
                    Factor::Natural(rn),
                    Factor::Natural(&pa),
                    Factor::Natural(&pb),
                    Factor::Natural(&qc),
                ],
                &[
                    Factor::Natural(rd),
                    Factor::Natural(&qa),
                    Factor::Natural(&qb),
                    Factor::Natural(&pc),
                ],
            )
        });
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
