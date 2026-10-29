//! SU(N) Clebsch-Gordan coefficient generation: highest-weight nullspace, deterministic gauge canonicalization, and
//! reverse-lex weight-ladder descent.
//!
//! Ported from SUNRepresentations.jl v0.4.0 `src/clebschgordan.jl`. The
//! gauge produced here is a semver contract of this crate: any change that can
//! alter a returned coefficient value is a breaking release. See
//! `docs/gauge.md` for the full specification and per-choice citations.
//!
//! Pipeline (`clebschgordan.jl:_CGC`):
//! 1. `highest_weight_CGC` — build the sparse raising-operator system over the
//!    coupling subspace at `s3`'s highest weight, take its dense right
//!    nullspace (SVD with full V, cut at `TOL_NULLSPACE`), and gauge-fix the block.
//! 2. `lower_weight_CGC!` — descend to every lower weight in reverse
//!    lexicographic order, solving a per-weight least-squares system.
//! 3. `purge!` — drop coefficients below `TOL_PURGE`.
//!
//! Numerical seams (SVD with full V, positive-diagonal QR, least squares) route
//! through [`super::linalg`] (t4a-faer public APIs only).

use super::{add_weight, magnetic_index, weight_sum};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::linalg::{self, Mat};
use super::{GtPattern, Irrep, LadderEntry, SunError};

/// Absolute singular-value tolerance for the highest-weight nullspace rank cut.
/// Reference: `clebschgordan.jl:TOL_NULLSPACE = 1.0e-13`.
pub const TOL_NULLSPACE: f64 = 1.0e-13;

/// Pivot tolerance for the `cref!` column-echelon gauge step. Reference:
/// `clebschgordan.jl:TOL_GAUGE = 1.0e-11` (deliberately looser than
/// `TOL_NULLSPACE`, per the reference comment).
pub const TOL_GAUGE: f64 = 1.0e-11;

/// Magnitude below which an assembled coefficient is dropped as an approximate
/// zero. Reference: `clebschgordan.jl:TOL_PURGE = 1.0e-14`.
pub const TOL_PURGE: f64 = 1.0e-14;

/// Generation gate: worst tolerated CGC orthonormality residual. Not a
/// reference constant; sized well above the f64 SVD/QR round-off floor
/// (`~sqrt(dim) * eps`) so a true gauge/algebra defect trips it while faithful
/// round-off does not. See `docs/gauge.md`.
pub const TOL_ORTHO: f64 = 1.0e-9;

/// Generation gate: worst tolerated ladder-intertwiner residual (same sizing
/// rationale as `TOL_ORTHO`).
pub const TOL_LADDER: f64 = 1.0e-9;

/// One nonzero element of a sparse m-basis CGC tensor: value at
/// `(m1, m2, m3, mu)`, where `m1`/`m2`/`m3` are 0-based GT basis indices into
/// [`Irrep::patterns`] of `s1`/`s2`/`s3` and `mu` is the outer-multiplicity
/// (trailing) axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CgcEntry {
    /// 0-based GT basis index in `s1`.
    pub m1: u32,
    /// 0-based GT basis index in `s2`.
    pub m2: u32,
    /// 0-based GT basis index in `s3`.
    pub m3: u32,
    /// 0-based outer-multiplicity index (trailing axis).
    pub mu: u32,
    /// The (real, standard-gauge) coefficient.
    pub value: f64,
}

/// The Clebsch-Gordan coefficients coupling $s_1 \otimes s_2 \to s_3$, as a sparse
/// m-basis tensor with the outer multiplicity carried on a trailing axis.
///
/// Shape is $[\dim(s_1), \dim(s_2), \dim(s_3), N]$ with $N = N^{s_3}_{s_1 s_2}$ the
/// exact fusion multiplicity ([`crate::lie::sun::directproduct`]). Only nonzero entries (after
/// the `TOL_PURGE` cut) are stored, sorted by `(m1, m2, m3, mu)`.
///
/// Coefficient values realize the SUNRepresentations.jl v0.4.0 gauge (see
/// `docs/gauge.md`); they are a versioned part of this crate's contract.
///
/// Cloning is cheap: the tensor is shared (`Arc`), so a cache hit hands out the
/// cached coefficients without copying them.
#[derive(Clone, Debug, PartialEq)]
pub struct Cgc(std::sync::Arc<CgcData>);

#[derive(Debug, PartialEq)]
struct CgcData {
    s1: Irrep,
    s2: Irrep,
    s3: Irrep,
    dims: [usize; 4],
    entries: Vec<CgcEntry>,
}

impl Cgc {
    /// The outer multiplicity `N^{s3}_{s1 s2}` (length of the trailing axis).
    pub fn multiplicity(&self) -> usize {
        self.0.dims[3]
    }
    /// The tensor shape `[dim(s1), dim(s2), dim(s3), multiplicity]`.
    pub fn dims(&self) -> [usize; 4] {
        self.0.dims
    }
    /// The stored nonzero entries, sorted by `(m1, m2, m3, mu)`.
    pub fn entries(&self) -> &[CgcEntry] {
        &self.0.entries
    }
    /// The number of stored nonzero entries.
    #[cfg(test)]
    pub fn nnz(&self) -> usize {
        self.0.entries.len()
    }
    /// Conservative retained charge used by the cache: the entry vector plus
    /// the three irrep weight buffers and value shell. Container scaffolding,
    /// allocator metadata/RSS, and external clones are outside this charge.
    pub(crate) fn storage_bytes(&self) -> usize {
        self.0
            .entries
            .len()
            .saturating_mul(std::mem::size_of::<CgcEntry>())
            .saturating_add(std::mem::size_of::<CgcData>())
            .saturating_add(
                [&self.0.s1, &self.0.s2, &self.0.s3]
                    .into_iter()
                    .fold(0usize, |sum, s| {
                        sum.saturating_add(s.rank().saturating_mul(std::mem::size_of::<i64>()))
                    }),
            )
    }
}

// ---------------------------------------------------------------------------
// GT weight helpers (port of gtpatterns.jl:weight for a pattern).
// ---------------------------------------------------------------------------

/// The physical weight of a GT pattern, an `N`-tuple. Ported from
/// `gtpatterns.jl:weight`: component `l` (1-based) is `rowsum(l) - rowsum(l-1)`
/// with `rowsum(l) = Σ_{k=1..l} m[k, l]` and `rowsum(0) = 0`.
pub(in crate::lie) fn pattern_weight(p: &GtPattern, n: usize) -> Result<Vec<i64>, SunError> {
    let mut previous = 0i64;
    (1..=n)
        .map(|l| {
            let sum = (1..=l).try_fold(0i64, |sum, k| add_weight(sum, p.get(k, l)))?;
            let w = sum.checked_sub(previous).ok_or(SunError::LabelOverflow)?;
            previous = sum;
            Ok(w)
        })
        .collect()
}

/// Map from weight tuple to the 0-based basis indices carrying it, over the GT
/// basis of `s`. Ports `clebschgordan.jl:weightmap`.
fn weight_map(patterns: &[GtPattern], n: usize) -> Result<HashMap<Vec<i64>, Vec<usize>>, SunError> {
    let mut m: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();
    for (i, p) in patterns.iter().enumerate() {
        m.entry(pattern_weight(p, n)?).or_default().push(i);
    }
    Ok(m)
}

/// Sparse GT adjacency. Lowering columns are in ascending destination-index order.
#[derive(Default)]
struct LadderOp {
    by_col: HashMap<usize, Vec<(usize, f64)>>,
    #[cfg(test)]
    elem: HashMap<(usize, usize), f64>,
}

fn build_ops(mats: Vec<Vec<LadderEntry>>) -> (Vec<LadderOp>, Vec<LadderOp>) {
    mats.into_iter()
        .map(|mat| {
            let mut up = LadderOp::default();
            let mut down = LadderOp::default();
            for e in mat {
                let v = e.value.to_f64();
                up.by_col.entry(e.col).or_default().push((e.row, v));
                // Creation visits source indices in ascending order, so its
                // transpose has exactly the old weight-space traversal order.
                down.by_col.entry(e.row).or_default().push((e.col, v));
                #[cfg(test)]
                {
                    up.elem.insert((e.row, e.col), v);
                    down.elem.insert((e.col, e.row), v);
                }
            }
            (up, down)
        })
        .unzip()
}

fn is_trivial(s: &Irrep) -> bool {
    s.weight().iter().all(|&x| x == 0)
}

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

/// Generate the Clebsch-Gordan coefficients for `s1 ⊗ s2 → s3`.
///
/// Returns the sparse m-basis [`Cgc`] tensor with the outer multiplicity on the
/// trailing axis. All multiplicity columns are generated together (they share
/// one nullspace) and gauge-fixed as a block, reproducing the
/// SUNRepresentations.jl v0.4.0 gauge (`docs/gauge.md`).
///
/// # Errors
///
/// - [`SunError::RankMismatch`] if `s1`, `s2`, `s3` are not all SU(N) for one
///   `N`.
/// - [`SunError::NullspaceDimMismatch`] if the SVD rank cut disagrees with the
///   exact fusion multiplicity (`clebschgordan.jl` `@assert N123 == …`).
/// - [`SunError::NotOrthonormal`] / [`SunError::LadderInconsistent`] if a
///   generation gate fails.
/// - [`SunError::Linalg`] if a dense factorization backend call fails.
///
/// # Examples
///
pub fn cgc(s1: &Irrep, s2: &Irrep, s3: &Irrep) -> Result<Cgc, SunError> {
    // Transparent, byte-accounted cache (WignerSymbols.jl / SUNRepresentations
    // `_get_CGC` model): a warm hit skips the whole SVD/QR/descent pipeline.
    // A hit clones the shared `Cgc` handle (a reference-count bump, no
    // coefficient copy). A miss surfaces its error uncached (errors are not
    // stored).
    use std::sync::Arc;
    let stored = crate::cache::sun_cgc(s1, s2, s3, || generate(s1, s2, s3, None).map(Arc::new))?;
    Ok((*stored).clone())
}

#[cfg(test)]
thread_local! {
    pub(super) static LADDER_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static MATCH_BUILDS: std::cell::RefCell<BTreeMap<(Vec<i64>, usize), usize>> = const { std::cell::RefCell::new(BTreeMap::new()) };
    pub(in crate::lie) static GENERATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The pure generation pipeline (no cache). `expected_override` forces the
/// multiplicity-gate target for the `#[cfg(test)]` forced-mismatch scenario;
/// `None` uses the exact fusion multiplicity.
fn generate(
    s1: &Irrep,
    s2: &Irrep,
    s3: &Irrep,
    expected_override: Option<usize>,
) -> Result<Cgc, SunError> {
    #[cfg(test)]
    GENERATIONS.with(|n| n.set(n.get() + 1));
    let n = s1.rank();
    if s2.rank() != n || s3.rank() != n {
        return Err(SunError::RankMismatch {
            a: n,
            b: if s2.rank() != n { s2.rank() } else { s3.rank() },
        });
    }
    let [d1, d2, d3] = [s1, s2, s3].map(|s| {
        let d = super::super::checked_dimension(&s.dim()).map_err(|_| SunError::LabelOverflow)?;
        magnetic_index(d - 1)?;
        Ok(d)
    });
    let (d1, d2, d3) = (d1?, d2?, d3?);
    let expected = match expected_override {
        Some(expected) => expected,
        None => super::fr::mult(s1, s2, s3)?,
    };

    // Trivial couplings (clebschgordan.jl:trivial_CGC): 1 ⊗ s → s and s ⊗ 1 → s
    // are identity embeddings, no linear algebra.
    if is_trivial(s1) {
        // Reference `_CGC`: isone(s1) => @assert s2 == s3. Without this the
        // trivial embedding would fabricate a mult-1 identity for a channel
        // whose real multiplicity is 0 (e.g. 1 ⊗ 3 → 3̄), silently skipping the
        // multiplicity gate. `expected` is the true fusion multiplicity (0 here).
        if s3 != s2 {
            return Err(SunError::NullspaceDimMismatch { expected, found: 1 });
        }
        return trivial_cgc(s1.clone(), s2.clone(), s3.clone(), [d1, d2, d3, 1], true);
    }
    if is_trivial(s2) {
        if s3 != s1 {
            return Err(SunError::NullspaceDimMismatch { expected, found: 1 });
        }
        return trivial_cgc(s1.clone(), s2.clone(), s3.clone(), [d1, d2, d3, 1], false);
    }

    let mut data: HashMap<(u32, u32, u32, u32), f64> = HashMap::new();
    let ctx = Ctx::new(s1, s2, s3)?;
    let n123 = highest_weight_cgc(&mut data, &ctx, expected)?;
    lower_weight_cgc(&mut data, &ctx, n123)?;
    purge(&mut data);

    let cgc = freeze(s1.clone(), s2.clone(), s3.clone(), [d1, d2, d3, n123], data);
    check_orthonormal(&cgc, &ctx)?;
    check_ladder(&cgc, &ctx)?;
    Ok(cgc)
}

fn trivial_cgc(
    s1: Irrep,
    s2: Irrep,
    s3: Irrep,
    dims: [usize; 4],
    is_left: bool,
) -> Result<Cgc, SunError> {
    // isleft: 1 ⊗ s → s, CGC[0, m, m, 0] = 1. else s ⊗ 1 → s, CGC[m, 0, m, 0] = 1.
    let d = if is_left { dims[1] } else { dims[0] };
    let mut entries = Vec::with_capacity(d);
    for m in 0..d {
        let m = magnetic_index(m)?;
        let (m1, m2) = if is_left { (0, m) } else { (m, 0) };
        entries.push(CgcEntry {
            m1,
            m2,
            m3: m,
            mu: 0,
            value: 1.0,
        });
    }
    Ok(Cgc(std::sync::Arc::new(CgcData {
        s1,
        s2,
        s3,
        dims,
        entries,
    })))
}

/// One basis, weight map and rounded ladder pair per distinct irrep in a context.
struct IrrepData {
    pats: Vec<GtPattern>,
    map: HashMap<Vec<i64>, Vec<usize>>,
    creation: Vec<LadderOp>,
    annih: Vec<LadderOp>,
}

impl IrrepData {
    fn new(s: &Irrep) -> Result<Self, SunError> {
        let pats = s.patterns();
        let map = weight_map(&pats, s.rank())?;
        let (creation, annih) = build_ops(s.try_creation_with_patterns(&pats)?);
        Ok(Self {
            pats,
            map,
            creation,
            annih,
        })
    }
}

/// Per-generation data shared by the highest-weight and descent stages.
struct Ctx {
    n: usize,
    d3: usize,
    one: std::sync::Arc<IrrepData>,
    two: std::sync::Arc<IrrepData>,
    three: std::sync::Arc<IrrepData>,
    w3_top: Vec<i64>,
    wshift: i64,
}

impl Ctx {
    fn new(s1: &Irrep, s2: &Irrep, s3: &Irrep) -> Result<Self, SunError> {
        use std::sync::Arc;
        let n = s1.rank();
        let wshift = coupling_shift(s1.weight(), s2.weight(), s3.weight(), n)?;
        let one = Arc::new(IrrepData::new(s1)?);
        let two = if s2 == s1 {
            Arc::clone(&one)
        } else {
            Arc::new(IrrepData::new(s2)?)
        };
        let three = if s3 == s1 {
            Arc::clone(&one)
        } else if s3 == s2 {
            Arc::clone(&two)
        } else {
            Arc::new(IrrepData::new(s3)?)
        };
        Ok(Ctx {
            n,
            d3: three.pats.len(),
            one,
            two,
            three,
            w3_top: s3.weight().to_vec(),
            wshift,
        })
    }
}

fn coupling_shift(a: &[i64], b: &[i64], c: &[i64], n: usize) -> Result<i64, SunError> {
    add_weight(weight_sum(a)?, weight_sum(b)?)?
        .checked_sub(weight_sum(c)?)
        .and_then(|w| w.checked_div(i64::try_from(n).ok()?))
        .ok_or(SunError::LabelOverflow)
}
fn matching_weight(total: &[i64], first: &[i64], shift: i64) -> Result<Vec<i64>, SunError> {
    total
        .iter()
        .zip(first)
        .map(|(&a, &b)| add_weight(a.checked_sub(b).ok_or(SunError::LabelOverflow)?, shift))
        .collect()
}

const EMPTY: &[usize] = &[];

fn get_indices<'a>(m: &'a HashMap<Vec<i64>, Vec<usize>>, w: &[i64]) -> &'a [usize] {
    m.get(w).map(Vec::as_slice).unwrap_or(EMPTY)
}

// ---------------------------------------------------------------------------
// Stage 1: highest-weight CGC (clebschgordan.jl:highest_weight_CGC).
// ---------------------------------------------------------------------------

fn highest_weight_cgc(
    data: &mut HashMap<(u32, u32, u32, u32), f64>,
    ctx: &Ctx,
    expected: usize,
) -> Result<usize, SunError> {
    let n = ctx.n;
    // Columns = coupling pairs (m1, m2) at the target (highest) weight of s3.
    let mut cols: Vec<(usize, usize)> = Vec::new();
    let mut col_index: HashMap<(usize, usize), usize> = HashMap::new();
    // Sparse raising-operator equations, keyed by row triple (l, m1', m2').
    let mut acc: Vec<((usize, usize, usize), usize, f64)> = Vec::new();
    let mut rowset: BTreeSet<(usize, usize, usize)> = BTreeSet::new();

    for (m1, pat1) in ctx.one.pats.iter().enumerate() {
        let w1 = pattern_weight(pat1, n)?;
        // w2 = w3_top - w1 + wshift  (componentwise, wshift a scalar broadcast).
        let w2 = matching_weight(&ctx.w3_top, &w1, ctx.wshift)?;
        for &m2 in get_indices(&ctx.two.map, &w2) {
            let col = cols.len();
            cols.push((m1, m2));
            col_index.insert((m1, m2), col);
            for l in 0..n - 1 {
                // Jp1 raises m1 (m2 fixed): column m1 of creation1[l].
                if let Some(entries) = ctx.one.creation[l].by_col.get(&m1) {
                    for &(m1p, v) in entries {
                        let key = (l, m1p, m2);
                        rowset.insert(key);
                        acc.push((key, col, v));
                    }
                }
                // Jp2 raises m2 (m1 fixed): column m2 of creation2[l].
                if let Some(entries) = ctx.two.creation[l].by_col.get(&m2) {
                    for &(m2p, v) in entries {
                        let key = (l, m1, m2p);
                        rowset.insert(key);
                        acc.push((key, col, v));
                    }
                }
            }
        }
    }

    let ncols = cols.len();
    let rows: Vec<(usize, usize, usize)> = rowset.into_iter().collect();
    let row_index: HashMap<(usize, usize, usize), usize> =
        rows.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let mut eqs = Mat::try_zeros(rows.len(), ncols)?;
    for (rk, col, v) in acc {
        eqs.add(row_index[&rk], col, v);
    }

    let solutions = linalg::nullspace(&eqs, TOL_NULLSPACE)?; // n_cols x N123
    let n123 = solutions.cols;
    if n123 != expected {
        return Err(SunError::NullspaceDimMismatch {
            expected,
            found: n123,
        });
    }

    // Gauge-fix the whole multiplicity block: first(qrpos!(cref!(solutions))).
    let fixed = gaugefix(solutions)?; // n_cols x N123

    // Scatter into CGC at m3 = highest-weight pattern of s3 = last basis index.
    let m3_top = magnetic_index(ctx.d3 - 1)?;
    for alpha in 0..n123 {
        for (i, &(m1, m2)) in cols.iter().enumerate() {
            let v = fixed.at(i, alpha);
            if v != 0.0 {
                *data
                    .entry((
                        magnetic_index(m1)?,
                        magnetic_index(m2)?,
                        m3_top,
                        magnetic_index(alpha)?,
                    ))
                    .or_insert(0.0) += v;
            }
        }
    }
    Ok(n123)
}

/// `gaugefix!(C) = first(qrpos!(cref!(C, TOL_GAUGE)))`.
fn gaugefix(mut solutions: Mat) -> Result<Mat, SunError> {
    cref(&mut solutions, TOL_GAUGE);
    linalg::qr_positive_q(&solutions)
}

/// Column-pivoted reduced echelon form, ported pivot-for-pivot from
/// `clebschgordan.jl:cref!`. Operates in place on the `nr x nc` matrix.
///
/// Per pivot row `i`, the pivot column is the one with the largest
/// `|A[i, j]|` over the not-yet-pinned columns `j..nc`; ties resolve to the
/// leftmost such column (`findabsmax` uses a strict `>` update). A row whose
/// remaining max is `<= eps` is zeroed and skipped. This pivot selection *is*
/// the gauge; see `docs/gauge.md`.
fn cref(a: &mut Mat, eps: f64) {
    let (nr, nc) = (a.rows, a.cols);
    let (mut i, mut j) = (0usize, 0usize);
    while i < nr && j < nc {
        // Pivot column = largest |A[i, ·]| over the free columns j..nc, leftmost
        // on tie. Note (rows 0..nr vs the reference's i..nr in the swap/scale/
        // eliminate below): rows < i of a free column are already zero by the
        // time it is processed (each earlier pivot step zeroes its own row in
        // every other column), so operating on all rows is a no-op there and
        // matches the reference exactly.
        let (mval, off) = findabsmax((j..nc).map(|k| a.at(i, k)));
        let mj = j + off;
        if mval <= eps {
            if eps > 0.0 {
                for k in j..nc {
                    a.data[i + k * nr] = 0.0;
                }
            }
            i += 1;
        } else {
            for k in 0..nr {
                a.data.swap(k + j * nr, k + mj * nr);
            }
            let d = a.at(i, j);
            for k in 0..nr {
                a.data[k + j * nr] /= d;
            }
            for k in 0..nc {
                if k != j {
                    let dk = a.at(i, k);
                    if dk != 0.0 {
                        for l in 0..nr {
                            a.data[l + k * nr] -= dk * a.at(l, j);
                        }
                    }
                }
            }
            i += 1;
            j += 1;
        }
    }
}

/// `(max |v|, offset)` of the first entry of maximal absolute value in `vals`.
///
/// Ties resolve to the **leftmost** entry: the running maximum updates only on
/// a strict `>` (`clebschgordan.jl:findabsmax`). This leftmost rule is part of
/// the gauge specification (`docs/gauge.md §4a`). It is, however, **value-
/// neutral in the final `cref` output** — reduced column echelon form is unique,
/// so a different tie rule cannot change any returned coefficient; the rule is
/// pinned by a unit test at the selection site because no value fixture can
/// catch it.
fn findabsmax(vals: impl Iterator<Item = f64>) -> (f64, usize) {
    let mut m = f64::NEG_INFINITY;
    let mut mi = 0;
    for (k, v) in vals.enumerate() {
        if v.abs() > m {
            m = v.abs();
            mi = k;
        }
    }
    (m, mi)
}

// ---------------------------------------------------------------------------
// Stage 2: lower-weight descent (clebschgordan.jl:lower_weight_CGC!).
// ---------------------------------------------------------------------------

struct DescentRoot<'a> {
    l: usize,
    parents: &'a [usize],
    sources: Vec<((u32, u32), std::ops::Range<usize>)>,
    edges: Vec<((usize, usize), f64)>,
}

fn lower_weight_cgc(
    data: &mut HashMap<(u32, u32, u32, u32), f64>,
    ctx: &Ctx,
    n123: usize,
) -> Result<(), SunError> {
    if n123 == 0 {
        return Ok(());
    }
    let mut w3list: Vec<_> = ctx.three.map.keys().collect();
    w3list.sort_unstable();
    w3list.reverse();
    let mut w1list: Vec<_> = ctx.one.map.keys().collect();
    w1list.sort_unstable();

    // All metadata is coefficient-independent. Keep alpha outermost when
    // replaying it: first-seen masks and each individual solve stay unchanged.
    let mut plans = Vec::new();
    for w3 in w3list.into_iter().skip(1) {
        let m3list = &ctx.three.map[w3];
        let mut roots = Vec::new();
        let mut imax = 0;
        let mut capacity = 0;
        for l in 0..ctx.n - 1 {
            let w3p = raise(w3, l)?;
            let parents = get_indices(&ctx.three.map, &w3p);
            if parents.is_empty() {
                continue;
            }
            #[cfg(test)]
            MATCH_BUILDS.with(|counts| {
                *counts.borrow_mut().entry((w3.clone(), l)).or_default() += 1;
            });
            imax = crate::lie::checked_count(imax, parents.len())
                .map_err(|_| SunError::LabelOverflow)?;
            let mut root = DescentRoot {
                l,
                parents,
                sources: Vec::new(),
                edges: Vec::new(),
            };
            for &w1p in &w1list {
                let w2p = matching_weight(&w3p, w1p, ctx.wshift)?;
                let m2plist = get_indices(&ctx.two.map, &w2p);
                if m2plist.is_empty() {
                    continue;
                }
                // Retain checked weight arithmetic from the old path. Exact
                // GT ladders can only reach these lowered weight spaces.
                lower(w1p, l)?;
                lower(&w2p, l)?;
                for &m2p in m2plist {
                    for &m1p in &ctx.one.map[w1p] {
                        let start = root.edges.len();
                        if let Some(entries) = ctx.one.annih[l].by_col.get(&m1p) {
                            for &(m1, v) in entries {
                                root.edges.push(((m1, m2p), v));
                            }
                        }
                        if let Some(entries) = ctx.two.annih[l].by_col.get(&m2p) {
                            for &(m2, v) in entries {
                                root.edges.push(((m1p, m2), v));
                            }
                        }
                        root.sources.push((
                            (magnetic_index(m1p)?, magnetic_index(m2p)?),
                            start..root.edges.len(),
                        ));
                    }
                }
            }
            let contributions = crate::lie::checked_size([parents.len(), root.edges.len()])
                .map_err(|_| SunError::LabelOverflow)?;
            capacity = crate::lie::checked_count(capacity, contributions)
                .map_err(|_| SunError::LabelOverflow)?;
            roots.push(root);
        }
        let mut eqs = Mat::try_zeros(imax, m3list.len())?;
        let mut i = 0;
        for root in &roots {
            for &m3p in root.parents {
                if let Some(entries) = ctx.three.annih[root.l].by_col.get(&m3p) {
                    for &(m3, v) in entries {
                        if let Ok(j) = m3list.binary_search(&m3) {
                            eqs.add(i, j, v);
                        }
                    }
                }
                i += 1;
            }
        }
        plans.push((m3list, eqs, roots, capacity, None));
    }

    for alpha in 0..n123 {
        let alpha = magnetic_index(alpha)?;
        for (m3list, eqs, roots, capacity, factorization) in &mut plans {
            // Upper bounds count every structural edge for every parent,
            // including zero coefficients: replay never grows either buffer.
            let mut mask_index = HashMap::with_capacity(*capacity);
            let mut rhs_acc = Vec::with_capacity(*capacity);
            let mut i = 0;
            for root in roots {
                for &m3p in root.parents {
                    let m3p = magnetic_index(m3p)?;
                    for &((m1p, m2p), ref edges) in &root.sources {
                        let coeff = *data.get(&(m1p, m2p, m3p, alpha)).unwrap_or(&0.0);
                        if coeff == 0.0 {
                            continue;
                        }
                        for &(pair, v) in &root.edges[edges.clone()] {
                            push_rhs(&mut mask_index, &mut rhs_acc, i, pair, v * coeff);
                        }
                    }
                    i += 1;
                }
            }
            let nmask = mask_index.len();
            if nmask == 0 {
                continue;
            }
            let mut mask = vec![(0, 0); nmask];
            for (&pair, &c) in &mask_index {
                mask[c] = pair;
            }
            let mut rhs = Mat::try_zeros(eqs.rows, nmask)?;
            for (row, pair, val) in rhs_acc {
                rhs.add(row, mask_index[&pair], val);
            }
            #[cfg(test)]
            descent_tests::record_solve(eqs, &rhs, &mask);
            // A plan with no RHS must never factor (or report a singular R).
            if factorization.is_none() {
                *factorization = Some(linalg::LeastSquares::factor(eqs, &rhs)?);
            }
            let sols = factorization.as_mut().unwrap().solve(&rhs)?;
            for (c, &(m1, m2)) in mask.iter().enumerate() {
                for (j, &m3) in m3list.iter().enumerate() {
                    let v = sols.at(j, c);
                    if v != 0.0 {
                        *data
                            .entry((
                                magnetic_index(m1)?,
                                magnetic_index(m2)?,
                                magnetic_index(m3)?,
                                alpha,
                            ))
                            .or_insert(0.0) += v;
                    }
                }
            }
        }
    }
    Ok(())
}

fn push_rhs(
    mask_index: &mut HashMap<(usize, usize), usize>,
    rhs_acc: &mut Vec<(usize, (usize, usize), f64)>,
    row: usize,
    pair: (usize, usize),
    val: f64,
) {
    let next = mask_index.len();
    mask_index.entry(pair).or_insert(next);
    rhs_acc.push((row, pair, val));
}

/// Raise weight component `l` by 1 and `l+1` by -1 (0-based `l`). Ports the
/// `Base.setindex(w, w[l]+1, l); setindex(_, w[l+1]-1, l+1)` parent step.
fn raise(w: &[i64], l: usize) -> Result<Vec<i64>, SunError> {
    let mut v = w.to_vec();
    v[l] = add_weight(v[l], 1)?;
    v[l + 1] = add_weight(v[l + 1], -1)?;
    Ok(v)
}

/// Lower weight component `l` by 1 and `l+1` by +1 (0-based `l`).
fn lower(w: &[i64], l: usize) -> Result<Vec<i64>, SunError> {
    let mut v = w.to_vec();
    v[l] = add_weight(v[l], -1)?;
    v[l + 1] = add_weight(v[l + 1], 1)?;
    Ok(v)
}

// ---------------------------------------------------------------------------
// Stage 3: purge and freeze (clebschgordan.jl:purge!).
// ---------------------------------------------------------------------------

fn purge(data: &mut HashMap<(u32, u32, u32, u32), f64>) {
    data.retain(|_, v| v.abs() > TOL_PURGE);
}

fn freeze(
    s1: Irrep,
    s2: Irrep,
    s3: Irrep,
    dims: [usize; 4],
    data: HashMap<(u32, u32, u32, u32), f64>,
) -> Cgc {
    let mut entries: Vec<CgcEntry> = data
        .into_iter()
        .map(|((m1, m2, m3, mu), value)| CgcEntry {
            m1,
            m2,
            m3,
            mu,
            value,
        })
        .collect();
    entries.sort_by_key(|e| (e.m1, e.m2, e.m3, e.mu));
    Cgc(std::sync::Arc::new(CgcData {
        s1,
        s2,
        s3,
        dims,
        entries,
    }))
}

// ---------------------------------------------------------------------------
// Generation gates.
// ---------------------------------------------------------------------------

/// Orthonormality gate: the CGC reshaped as `M[(m1,m2), (m3,α)]` is an
/// isometry, `Σ_{m1,m2} C[..,m3,α] C[..,m3',β] = δ_{m3 m3'} δ_{αβ}`. Overlap is
/// contracted over the coupling indices `(m1,m2)` only, per output column
/// `(m3,α)` -- not summed over `m3`.
fn check_orthonormal(cgc: &Cgc, ctx: &Ctx) -> Result<(), SunError> {
    let [weights1, weights2, weights3] = [&ctx.one.map, &ctx.two.map, &ctx.three.map].map(|map| {
        let mut weights = vec![&[][..]; map.values().map(Vec::len).sum()];
        for (weight, indices) in map {
            for &m in indices {
                weights[m] = weight.as_slice();
            }
        }
        weights
    });
    // GT weights differ from additive SU(N) weights by the uniform coupling shift.
    for e in cgc.entries() {
        let w1 = weights1[usize::try_from(e.m1).map_err(|_| SunError::LabelOverflow)?];
        let w2 = weights2[usize::try_from(e.m2).map_err(|_| SunError::LabelOverflow)?];
        let w3 = weights3[usize::try_from(e.m3).map_err(|_| SunError::LabelOverflow)?];
        for l in 0..ctx.n {
            if add_weight(w1[l], w2[l])? != add_weight(w3[l], ctx.wshift)? {
                return Err(SunError::NotOrthonormal {
                    residual: f64::INFINITY,
                });
            }
        }
    }
    // Columns keyed by (m3, mu); each is a map (m1,m2) -> value.
    let mut columns: BTreeMap<(u32, u32), BTreeMap<(u32, u32), f64>> = BTreeMap::new();
    for e in cgc.entries() {
        columns
            .entry((e.m3, e.mu))
            .or_default()
            .insert((e.m1, e.m2), e.value);
    }
    // Validated support makes columns in different output weights disjoint.
    let mut blocks: BTreeMap<&[i64], Vec<(u32, u32)>> = BTreeMap::new();
    for &key in columns.keys() {
        let weight = weights3[usize::try_from(key.0).map_err(|_| SunError::LabelOverflow)?];
        blocks.entry(weight).or_default().push(key);
    }
    let mut worst = 0.0f64;
    for keys in blocks.values() {
        for (ia, ka) in keys.iter().enumerate() {
            let ca = &columns[ka];
            for kb in keys.iter().skip(ia) {
                let cb = &columns[kb];
                // Dot over shared (m1,m2); iterate the smaller column.
                let (small, big) = if ca.len() <= cb.len() {
                    (ca, cb)
                } else {
                    (cb, ca)
                };
                let mut dot = 0.0;
                for (idx, &v) in small {
                    if let Some(&w) = big.get(idx) {
                        dot += v * w;
                    }
                }
                let target = if ka == kb { 1.0 } else { 0.0 };
                worst = worst.max((dot - target).abs());
            }
        }
    }
    if worst > TOL_ORTHO {
        return Err(SunError::NotOrthonormal { residual: worst });
    }
    Ok(())
}

/// Ladder-consistency spot check: the lowering intertwiner at level `l = 0`,
/// evaluated at the highest-weight parent `m3' = d3-1`, for every multiplicity
/// column. Verifies `Σ (Jm1⊗I + I⊗Jm2) C[·,·,m3',α] = Σ_m3 C[·,·,m3,α] Jm3[m3,m3']`
/// element-wise over the coupling states, coupling the highest-weight block to
/// its first descent.
fn check_ladder(cgc: &Cgc, ctx: &Ctx) -> Result<(), SunError> {
    if ctx.n < 2 || cgc.multiplicity() == 0 {
        return Ok(());
    }
    let l = 0usize;
    let m3p = magnetic_index(ctx.d3 - 1)?; // highest-weight pattern of s3
    let n123 = cgc.multiplicity();
    // Index CGC for random access.
    let mut idx: HashMap<(u32, u32, u32, u32), f64> = HashMap::new();
    for e in cgc.entries() {
        idx.insert((e.m1, e.m2, e.m3, e.mu), e.value);
    }
    let get = |m1: usize, m2: usize, m3: usize, a: usize| -> Result<f64, SunError> {
        Ok(idx
            .get(&(
                magnetic_index(m1)?,
                magnetic_index(m2)?,
                magnetic_index(m3)?,
                magnetic_index(a)?,
            ))
            .copied()
            .unwrap_or(0.0))
    };

    let mut worst = 0.0f64;
    for alpha in 0..n123 {
        // LHS[(m1,m2)] = Σ over lowering of the parent state m3'.
        let mut lhs: HashMap<(usize, usize), f64> = HashMap::new();
        // parent (m1',m2') states carrying nonzero CGC at m3'.
        for e in cgc.entries() {
            if e.m3 != m3p || usize::try_from(e.mu).map_err(|_| SunError::LabelOverflow)? != alpha {
                continue;
            }
            let (m1p, m2p) = (
                usize::try_from(e.m1).map_err(|_| SunError::LabelOverflow)?,
                usize::try_from(e.m2).map_err(|_| SunError::LabelOverflow)?,
            );
            let coeff = e.value;
            // Jm1 lowers m1'
            if let Some(col) = ctx.one.annih[l].by_col.get(&m1p) {
                for &(m1, v) in col {
                    *lhs.entry((m1, m2p)).or_insert(0.0) += v * coeff;
                }
            }
            // Jm2 lowers m2'
            if let Some(col) = ctx.two.annih[l].by_col.get(&m2p) {
                for &(m2, v) in col {
                    *lhs.entry((m1p, m2)).or_insert(0.0) += v * coeff;
                }
            }
        }
        // RHS[(m1,m2)] = Σ_m3 C[m1,m2,m3,α] Jm3[m3, m3'].
        let mut rhs: HashMap<(usize, usize), f64> = HashMap::new();
        if let Some(col) = ctx.three.annih[l]
            .by_col
            .get(&(usize::try_from(m3p).map_err(|_| SunError::LabelOverflow)?))
        {
            for &(m3, jm3) in col {
                // m3 is a child (lower) state; find CGC entries at that m3.
                for e in cgc.entries() {
                    if usize::try_from(e.m3).map_err(|_| SunError::LabelOverflow)? != m3
                        || usize::try_from(e.mu).map_err(|_| SunError::LabelOverflow)? != alpha
                    {
                        continue;
                    }
                    *rhs.entry((
                        usize::try_from(e.m1).map_err(|_| SunError::LabelOverflow)?,
                        usize::try_from(e.m2).map_err(|_| SunError::LabelOverflow)?,
                    ))
                    .or_insert(0.0) += get(
                        usize::try_from(e.m1).map_err(|_| SunError::LabelOverflow)?,
                        usize::try_from(e.m2).map_err(|_| SunError::LabelOverflow)?,
                        m3,
                        alpha,
                    )? * jm3;
                }
            }
        }
        let mut keys: BTreeSet<(usize, usize)> = BTreeSet::new();
        keys.extend(lhs.keys().copied());
        keys.extend(rhs.keys().copied());
        for k in keys {
            let d = lhs.get(&k).copied().unwrap_or(0.0) - rhs.get(&k).copied().unwrap_or(0.0);
            worst = worst.max(d.abs());
        }
    }
    if worst > TOL_LADDER {
        return Err(SunError::LadderInconsistent { residual: worst });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn irr(d: &[i64]) -> Irrep {
        Irrep::from_dynkin(d).unwrap()
    }

    #[test]
    fn forced_multiplicity_mismatch_is_typed_error() {
        // SU(3) 3 ⊗ 3̄ → 8 is multiplicity 1; forcing the gate target to 2 must
        // surface a typed NullspaceDimMismatch (the ported `@assert N123 == …`).
        let e = generate(&irr(&[1, 0]), &irr(&[0, 1]), &irr(&[1, 1]), Some(2)).unwrap_err();
        assert_eq!(
            e,
            SunError::NullspaceDimMismatch {
                expected: 2,
                found: 1
            }
        );
    }

    #[test]
    fn private_generate_does_not_reconstruct_public_product_maps() {
        crate::lie::sun::reset_public_directproduct_reconstructions();

        let trivial = Irrep::trivial(3).unwrap();
        let three = irr(&[1, 0]);
        let c = generate(&trivial, &three, &three, None).unwrap();

        assert_eq!(c.multiplicity(), 1);
        assert_eq!(crate::lie::sun::public_directproduct_reconstructions(), 0);
    }

    #[test]
    fn su3_adjoint_channel_generates_and_caches_equal() {
        let c = cgc(&irr(&[1, 0]), &irr(&[0, 1]), &irr(&[1, 1])).unwrap();
        assert_eq!(c.multiplicity(), 1);
        assert!(c.nnz() > 0);
        // Warm hit returns byte-identical values.
        let generations = GENERATIONS.get();
        let c2 = cgc(&irr(&[1, 0]), &irr(&[0, 1]), &irr(&[1, 1])).unwrap();
        assert_eq!(c, c2);
        assert_eq!(
            GENERATIONS.get(),
            generations,
            "warm call must not generate"
        );
    }

    #[test]
    fn su3_octet_squared_has_outer_multiplicity_two() {
        // 8 ⊗ 8 → 8 has N = 2: the multiplicity block is generated and
        // gauge-fixed together, and must come out orthonormal.
        let c = cgc(&irr(&[1, 1]), &irr(&[1, 1]), &irr(&[1, 1])).unwrap();
        assert_eq!(c.multiplicity(), 2);
        let ctx = Ctx::new(&irr(&[1, 1]), &irr(&[1, 1]), &irr(&[1, 1])).unwrap();
        check_orthonormal(&c, &ctx).unwrap();
    }

    #[test]
    fn orthonormal_gate_rejects_off_weight_support() {
        let (a, b, c) = (irr(&[1, 0]), irr(&[0, 1]), irr(&[1, 1]));
        let ctx = Ctx::new(&a, &b, &c).unwrap();
        let mut cgc = generate(&a, &b, &c, None).unwrap();
        let mut entry = cgc.entries()[0];
        let other_weight = ctx
            .three
            .map
            .values()
            .find(|indices| !indices.contains(&usize::try_from(entry.m3).unwrap()))
            .unwrap();
        entry.m3 = magnetic_index(other_weight[0]).unwrap();
        // Below the numerical gate tolerance: exact support must still reject it.
        entry.value = TOL_ORTHO / 4.0;
        std::sync::Arc::get_mut(&mut cgc.0)
            .unwrap()
            .entries
            .push(entry);
        assert_eq!(
            check_orthonormal(&cgc, &ctx),
            Err(SunError::NotOrthonormal {
                residual: f64::INFINITY
            })
        );
    }

    #[test]
    fn orthonormal_gate_rejects_parallel_within_weight_columns() {
        let a = irr(&[1, 1]);
        let ctx = Ctx::new(&a, &a, &a).unwrap();
        let mut cgc = generate(&a, &a, &a, None).unwrap();
        let indices = ctx
            .three
            .map
            .values()
            .find(|indices| indices.len() > 1)
            .unwrap();
        let source = magnetic_index(indices[0]).unwrap();
        let target = magnetic_index(indices[1]).unwrap();
        // Duplicate a column across two states of the same weight: both norms
        // stay one, but their overlap becomes one.
        let mut duplicate: Vec<_> = cgc
            .entries()
            .iter()
            .filter(|e| e.m3 == source && e.mu == 0)
            .copied()
            .collect();
        for e in &mut duplicate {
            e.m3 = target;
        }
        let entries = &mut std::sync::Arc::get_mut(&mut cgc.0).unwrap().entries;
        entries.retain(|e| e.m3 != target || e.mu != 0);
        entries.extend(duplicate);
        assert!(matches!(
            check_orthonormal(&cgc, &ctx),
            Err(SunError::NotOrthonormal { residual }) if residual > 0.9 && residual.is_finite()
        ));
    }

    #[test]
    fn rank_mismatch_is_typed_error() {
        let e = generate(&irr(&[1, 0]), &irr(&[1, 0, 0]), &irr(&[1, 0]), None).unwrap_err();
        assert!(matches!(e, SunError::RankMismatch { .. }));
    }

    #[test]
    fn findabsmax_breaks_ties_leftmost() {
        // The cref pivot rule. RCEF is unique, so a wrong tie rule is
        // value-neutral in cref's output and no coefficient fixture can catch
        // it (docs/gauge.md §4a); pin it here at the selection site. Exact
        // tie -> leftmost. The `>`->`>=` mutant returns the rightmost tie and
        // fails all three.
        assert_eq!(findabsmax([2.0, 2.0, 1.0].into_iter()), (2.0, 0));
        assert_eq!(findabsmax([-3.0, 3.0, 3.0].into_iter()), (3.0, 0));
        assert_eq!(findabsmax([1.0, 4.0, 4.0, 2.0].into_iter()), (4.0, 1));
    }

    #[test]
    fn trivial_left_correct_target_is_identity() {
        // 1 ⊗ 3 → 3 : the sole admissible channel, an identity embedding.
        let triv = Irrep::trivial(3).unwrap();
        let c = cgc(&triv, &irr(&[1, 0]), &irr(&[1, 0])).unwrap();
        assert_eq!(c.multiplicity(), 1);
        assert_eq!(c.nnz(), 3); // identity on dim 3
    }

    #[test]
    fn trivial_left_wrong_target_is_typed_error() {
        // 1 ⊗ 3 → 3̄ : 3̄ ∉ 1 ⊗ 3 = {3} (mult 0). Must NOT fabricate an identity
        // (reference `@assert s2 == s3`).
        let triv = Irrep::trivial(3).unwrap();
        let e = cgc(&triv, &irr(&[1, 0]), &irr(&[0, 1])).unwrap_err();
        assert_eq!(
            e,
            SunError::NullspaceDimMismatch {
                expected: 0,
                found: 1
            }
        );
    }

    #[test]
    fn trivial_left_wrong_dim_target_is_typed_error() {
        // 1 ⊗ 3 → 8 : dims would be nonsense ([1,3,8,1]); reject.
        let triv = Irrep::trivial(3).unwrap();
        let e = cgc(&triv, &irr(&[1, 0]), &irr(&[1, 1])).unwrap_err();
        assert_eq!(
            e,
            SunError::NullspaceDimMismatch {
                expected: 0,
                found: 1
            }
        );
    }

    #[test]
    fn trivial_right_wrong_target_is_typed_error() {
        // 3 ⊗ 1 → 3̄ : symmetric guard (`@assert s1 == s3`).
        let triv = Irrep::trivial(3).unwrap();
        let e = cgc(&irr(&[1, 0]), &triv, &irr(&[0, 1])).unwrap_err();
        assert_eq!(
            e,
            SunError::NullspaceDimMismatch {
                expected: 0,
                found: 1
            }
        );
    }
}

#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    #[test]
    fn coupling_weight_bounds() {
        assert_eq!(
            coupling_shift(&[i64::MAX], &[1], &[0], 1),
            Err(SunError::LabelOverflow)
        );
        assert_eq!(
            coupling_shift(&[i64::MIN], &[0], &[1], 1),
            Err(SunError::LabelOverflow)
        );
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            coupling_shift(&[0], &[0], &[0], i64::MAX as usize + 1),
            Err(SunError::LabelOverflow)
        );
        assert_eq!(
            matching_weight(&[i64::MAX], &[0], 1),
            Err(SunError::LabelOverflow)
        );
        assert_eq!(
            matching_weight(&[i64::MIN], &[1], 0),
            Err(SunError::LabelOverflow)
        );
        assert_eq!(raise(&[i64::MAX, 0], 0), Err(SunError::LabelOverflow));
        assert_eq!(raise(&[0, i64::MIN], 0), Err(SunError::LabelOverflow));
        assert_eq!(lower(&[i64::MIN, 0], 0), Err(SunError::LabelOverflow));
        assert_eq!(lower(&[0, i64::MAX], 0), Err(SunError::LabelOverflow));
        // Row-sum overflow is rejected even if the final difference would fit.
        let p = GtPattern {
            n: 2,
            data: vec![i64::MAX, i64::MAX, i64::MAX].into_boxed_slice(),
        };
        assert_eq!(pattern_weight(&p, 2), Err(SunError::LabelOverflow));
        let p = GtPattern {
            n: 2,
            data: vec![i64::MAX, i64::MAX, 0].into_boxed_slice(),
        };
        assert_eq!(pattern_weight(&p, 2), Err(SunError::LabelOverflow));
    }
}

#[cfg(test)]
#[path = "cgc_descent_tests.rs"]
mod descent_tests;
