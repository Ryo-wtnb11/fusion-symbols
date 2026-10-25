//! The `CanonicalCatalog`: the single, append-only owner of discovered
//! generator sets for one B/C/D family instance (series + rank fixed at
//! construction), providing on-demand, query-order-independent generator and
//! Clebsch–Gordan materialization on top of the decomposition sweep.
//!
//! The **canonical parent rule** — which product `a ⊗ b` produces each irrep's generators, and
//! why the choice is a deterministic function of the exact decomposition data rather than
//! of discovery order — is specified to re-derivation standard in
//! `docs/gauge_soN.md` §14 (it IS gauge: the parent's sweep fixes `c`'s basis).
//!
//! # What it owns
//!
//! Exactly one generator set [`Generators`] per irrep, **append-only** for the
//! catalog's lifetime (no per-entry eviction; the whole catalog may be
//! [`reset`](CanonicalCatalog::reset)). CGC/F/R **values do not live here** —
//! they are returned to the caller (and, for SU(N), go to the byte-bounded value
//! caches in `crate::cache`); the catalog owns only generator sets plus the
//! minimal bookkeeping (a byte counter) to bound them.
//!
//! # On-demand materialization
//!
//! [`generators`](CanonicalCatalog::generators) and
//! [`cgc`](CanonicalCatalog::cgc) recursively materialize an irrep's
//! canonical-parent chain as needed. The recursion is well-founded (§14 of the
//! gauge doc): each parent is strictly smaller than its child in a fixed
//! well-order over the exact irrep data, and the chain bottoms out at the base
//! cases: the trivial and defining reps, seeded at construction, and — for the
//! `B`/`D` spinor class — the fundamental spinors, whose Clifford seeds
//! (`docs/gauge_soN.md` §16) are built on demand, so a catalog only ever asked
//! for tensor irreps never materializes one.
//! QSpace's fixed-pass `dmax` enumeration (`clebsch.cc` bootstrap loop) is **not**
//! ported as semantics — see §14.
//!
//! # Atomic byte budget
//!
//! A request whose recursive materialization would exceed the byte budget fails
//! atomically with [`CatalogError::BudgetExceeded`], leaving **no** partial
//! state: the whole chain is assembled in a staging buffer, its cost checked
//! against the budget, and only then committed (compute-fully-then-commit).
//!
//! # Single-threaded
//!
//! The API is `&mut self`; there is no global state and no interior mutability,
//! so sharing one catalog across threads requires external synchronization.

use std::collections::HashMap;

use num_bigint::BigInt;

use super::seeds::spinor_seeds;
use super::sweep::{align_block, decompose, Block, Generators, SweepError};
use super::{defining_seed, directproduct, BcdError, Irrep, Series};

/// Default byte budget for a catalog (256 MiB). Generator sets are dense `f64`
/// `D×D` blocks; a family exercised over modest ranks stays far below this,
/// while a runaway recursion (or a deliberately tiny budget in a test) trips
/// [`CatalogError::BudgetExceeded`] before committing.
pub(in crate::lie) const DEFAULT_MAX_BYTES: usize = 256 << 20;

/// Coherence tolerance for the QSpace cross-copy check
/// ([`CatalogError::BasisIncoherent`]): two embeddings of one irrep must present
/// the same canonical basis to this element-wise generator residual. Provenance:
/// QSpace `normDiff <= 1e-10` (`clebsch.cc:6710-6718 @ dd2cc7e`). A coherent pair
/// agrees to ~1e-15 (well-conditioned sweep gauge); an ill-conditioned rotation
/// is O(1) — the tolerance cleanly separates the two.
const TOL_BASIS_COHERENT: f64 = 1.0e-10;

// ---- typed errors -----------------------------------------------------------

/// Failure of a [`CanonicalCatalog`] request. Every ill-posed input is a typed
/// error (never a panic, never a silent zero): every `N^c_ab = 0` triple is
/// [`CatalogError::ZeroFusionChannel`].
///
/// Not `Eq`: [`CatalogError::Sweep`] carries a [`SweepError`], several of whose
/// variants hold an `f64` residual.
#[derive(Clone, Debug, PartialEq)]
pub enum CatalogError {
    /// An irrep passed to the catalog belongs to a different family than the
    /// catalog owns (different series or rank). A catalog instance is fixed to
    /// one `(series, rank)` at construction; a foreign irrep is ill-posed.
    WrongGroup {
        /// The catalog's `(series, rank)`.
        catalog: (Series, usize),
        /// The offending irrep's `(series, rank)`.
        got: (Series, usize),
    },
    /// A malformed or out-of-scope label surfaced while constructing the family
    /// or an intermediate irrep (empty/negative/spinor/excluded-rank). Wraps the
    /// label-level [`BcdError`].
    Label(BcdError),
    /// [`cgc`](CanonicalCatalog::cgc) was asked for a triple with `N^c_ab = 0`
    /// (the coupled irrep `c` does not appear in `a ⊗ b`, per the exact
    /// [`directproduct`]). The reference sweep would simply never emit such a
    /// block; a query API must reject the ill-posed question loudly instead of
    /// returning an empty or zero block. Carries the Dynkin labels.
    ZeroFusionChannel {
        /// Dynkin label of the left factor `a`.
        a: Vec<i64>,
        /// Dynkin label of the right factor `b`.
        b: Vec<i64>,
        /// Dynkin label of the requested coupled irrep `c`.
        c: Vec<i64>,
    },
    /// Construction or recursive materialization would push the catalog's
    /// retained generator bytes past its budget. Reported **before** any commit,
    /// so no partial chain is ever observable.
    BudgetExceeded {
        /// The byte budget.
        limit: usize,
        /// The bytes that would be retained after committing this request;
        /// `usize::MAX` if the construction charge overflows.
        needed: usize,
    },
    /// The decomposition sweep (or a product-generator composition) failed while
    /// materializing a canonical-parent chain. Surfaced, not panicked: the
    /// floating-point stages are verification-gated.
    Sweep(SweepError),
    /// A coupled multiplet discovered in one product does **not** present the
    /// same canonical carrier basis as the stored embedding of that irrep — its
    /// projected generators differ beyond the coherence tolerance (the QSpace
    /// `normDiff` cross-copy check, `clebsch.cc:6710-6718 @ dd2cc7e`). An ill-conditioned QR can leave an irrep in a
    /// rotated frame between two embeddings, which would silently corrupt every
    /// F/R contraction that shares that irrep across its coupled and factor
    /// roles; this crate refuses to return such a value. Intertwiner alignment
    /// first tries to rotate the frame onto the canonical stored
    /// basis; this error is what remains when even the aligned frame disagrees
    /// beyond tolerance (a genuinely different irrep, or a numerically hopeless
    /// embedding whose remedy would be the out-of-scope extended-precision tier).
    BasisIncoherent {
        /// Dynkin label of the incoherent coupled irrep.
        irrep: Vec<i64>,
        /// Dynkin labels of the product `(s1, s2)` that produced the rotated
        /// embedding.
        product: (Vec<i64>, Vec<i64>),
        /// The worst generator-element residual against the stored basis.
        residual: f64,
    },
    /// A non-base irrep had **no** admissible canonical-parent pair (§14.4).
    /// This is **unreachable by the box-count-first existence theorem**
    /// (`(defining, c-minus-a-box)` is always admissible); it is surfaced as a
    /// typed error rather than an `unreachable!` panic as defense-in-depth while
    /// the corrected proof beds in — a wrong theorem should fail loudly and
    /// recoverably at the exact label, not abort the process. Carries the
    /// offending Dynkin label.
    NoCanonicalParent {
        /// The Dynkin label with no admissible parent pair.
        dynkin: Vec<i64>,
    },
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::WrongGroup { catalog, got } => write!(
                f,
                "irrep of group {got:?} passed to a catalog owning group {catalog:?}"
            ),
            CatalogError::Label(e) => write!(f, "invalid label: {e}"),
            CatalogError::ZeroFusionChannel { a, b, c } => write!(
                f,
                "ill-posed coupling: irrep {c:?} does not appear in {a:?} ⊗ {b:?} (N^c_ab = 0)"
            ),
            CatalogError::BudgetExceeded { limit, needed } => write!(
                f,
                "byte budget exceeded: request needs {needed} bytes, budget is {limit}"
            ),
            CatalogError::Sweep(e) => write!(f, "sweep failed during materialization: {e}"),
            CatalogError::BasisIncoherent {
                irrep,
                product,
                residual,
            } => write!(
                f,
                "irrep {irrep:?} from product {:?}⊗{:?} could not be aligned onto its \
                 stored canonical basis (post-alignment generator residual {residual:e} > \
                 coherence tol) — genuinely different frame or a numerically hopeless embedding",
                product.0, product.1
            ),
            CatalogError::NoCanonicalParent { dynkin } => write!(
                f,
                "no admissible canonical parent for irrep {dynkin:?} \
                 (unreachable by the box-count-first existence theorem)"
            ),
        }
    }
}

impl std::error::Error for CatalogError {}

impl From<BcdError> for CatalogError {
    fn from(e: BcdError) -> Self {
        CatalogError::Label(e)
    }
}

impl From<SweepError> for CatalogError {
    fn from(e: SweepError) -> Self {
        CatalogError::Sweep(e)
    }
}

// ---- public CGC result -----------------------------------------------------

/// The Clebsch–Gordan isometry coupling `s1 ⊗ s2 → s3`, produced by decomposing
/// the queried product `s1 ⊗ s2` and selecting the `s3` blocks.
///
/// Layout mirrors the SU(N) surface (`crate::lie::sun::Cgc`) but stays **dense**: the
/// sweep hands back each coupled multiplet as a dense isometry, so this holds the
/// concatenation of the outer-multiplicity copies, column-major, in
/// outer-multiplicity index order. `PartialEq` is bitwise over the coefficient
/// buffer, so the query-order-independence acceptance test can compare two
/// materializations directly.
#[derive(Clone, Debug, PartialEq)]
pub struct CatalogCgc {
    s1: Irrep,
    s2: Irrep,
    s3: Irrep,
    /// `d1·d2` (rows of each copy's isometry).
    rows: usize,
    /// `d3 = dim(s3)` (columns of each copy's isometry).
    d3: usize,
    /// Outer multiplicity `N^{s3}_{s1 s2}` (number of copies).
    multiplicity: usize,
    /// Concatenated copies: copy `mu` occupies `cols[mu·rows·d3 .. (mu+1)·rows·d3]`,
    /// each a column-major `rows × d3` isometry.
    cols: Vec<f64>,
}

impl CatalogCgc {
    /// The coupled irrep `s3`.
    pub fn s3(&self) -> &Irrep {
        &self.s3
    }
    /// The outer multiplicity `N^{s3}_{s1 s2}` (number of copies).
    pub fn multiplicity(&self) -> usize {
        self.multiplicity
    }
    /// `(rows, cols)` of one copy's isometry: `(d1·d2, d3)`.
    pub fn copy_shape(&self) -> (usize, usize) {
        (self.rows, self.d3)
    }
    /// The isometry of outer-multiplicity copy `mu` (`< multiplicity`) as a flat
    /// column-major `d1·d2 × d3` buffer.
    ///
    /// Indexing, in full: element `(row, m3)` is at `copy[m3 * rows + row]`
    /// with `rows = d1·d2` from [`copy_shape`](CatalogCgc::copy_shape). The
    /// product-basis row index is **first factor fast**, `row = m1 + d1·m2`
    /// with `d1 = dim(s1)`, so `m1 = row % d1` and `m2 = row / d1`.
    ///
    /// # Panics
    ///
    /// If `mu >= multiplicity()`.
    pub fn copy(&self, mu: usize) -> &[f64] {
        let stride = self.rows * self.d3;
        &self.cols[mu * stride..(mu + 1) * stride]
    }

    /// Conservative retained-byte charge for the value cache tier
    /// ([`crate::cache::cache_bcd_cgc`]): the dense coefficient buffer plus a
    /// fixed shell. Mirrors [`crate::lie::sun::Cgc::storage_bytes`].
    pub(crate) fn storage_bytes(&self) -> usize {
        self.cols
            .len()
            .saturating_mul(std::mem::size_of::<f64>())
            .saturating_add(std::mem::size_of::<Self>())
    }
}

// ---- the catalog ------------------------------------------------------------

/// Append-only owner of generator sets for one B/C/D family instance.
///
/// See the module docs for the ownership, canonical-parent, and atomicity
/// contracts; `docs/gauge_soN.md` §14 for the canonical-parent order and its
/// well-foundedness argument.
#[derive(Debug)]
pub struct CanonicalCatalog {
    series: Series,
    rank: usize,
    /// The discovered generator sets, keyed by irrep. Append-only; contains the
    /// two base cases (trivial, defining) from construction onward.
    store: HashMap<Irrep, Generators>,
    /// Retained generator bytes (conservative charge), the quantity bounded by
    /// `max_bytes`.
    bytes: usize,
    max_bytes: usize,
}

impl CanonicalCatalog {
    /// Build a catalog for `series` at rank `r` with the default byte budget.
    ///
    /// Seeds the two base cases — the trivial rep and the defining rep, the
    /// latter from the exact [`defining_seed`]. Rejects the excluded
    /// low-rank isomorphisms (`SO(3)`, `Sp(2)`, `SO(4)`) with
    /// [`CatalogError::Label`] carrying [`BcdError::ExcludedRank`].
    pub fn new(series: Series, r: usize) -> Result<Self, CatalogError> {
        Self::with_budget(series, r, DEFAULT_MAX_BYTES)
    }

    /// Build a catalog with an explicit byte budget (see [`new`](Self::new)).
    /// Rejects base seeds exceeding the budget (or overflowing byte arithmetic)
    /// with [`CatalogError::BudgetExceeded`] before any rank-sized allocation.
    pub fn with_budget(series: Series, r: usize, max_bytes: usize) -> Result<Self, CatalogError> {
        // Preserve the seed layer's excluded-rank errors before checking its charge.
        if r >= series.min_rank() {
            Self::seed_requirement(series, r, max_bytes)?;
        }
        let mut cat = CanonicalCatalog {
            series,
            rank: r,
            store: HashMap::new(),
            bytes: 0,
            max_bytes,
        };
        cat.seed_base()?;
        Ok(cat)
    }

    /// Preflight the base fixtures without allocating generators.
    pub(in crate::lie) fn seed_requirement(
        series: Series,
        r: usize,
        max_bytes: usize,
    ) -> Result<usize, CatalogError> {
        let needed = r
            .checked_mul(2)
            .and_then(|d| d.checked_add(usize::from(series == Series::B)))
            .and_then(|d| gen_bytes_for(r, d))
            .and_then(|defining| gen_bytes_for(r, 1)?.checked_add(defining));
        match needed {
            Some(needed) if needed <= max_bytes => Ok(needed),
            _ => Err(CatalogError::BudgetExceeded {
                limit: max_bytes,
                needed: needed.unwrap_or(usize::MAX),
            }),
        }
    }

    /// The family series.
    #[cfg(test)]
    pub fn series(&self) -> Series {
        self.series
    }
    /// The family rank.
    #[cfg(test)]
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// Retained generator bytes (the quantity bounded by the budget).
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    /// The byte budget.
    pub fn budget(&self) -> usize {
        self.max_bytes
    }
    /// Number of generator sets currently held (including the two base cases).
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.store.len()
    }
    /// Seed the two base cases: the trivial rep (a 1-dimensional carrier, all
    /// generators zero) and the defining rep (from the exact defining seed). Both
    /// are `≺`-minimal (§14), so every canonical-parent chain bottoms out here.
    ///
    /// The defining seed is stored in QSpace's `Setup_*` state order, which is
    /// **not** the sweep's descending-weight order; it is therefore brought into
    /// the canonical frame by one pass of the §1–§8 sweep over its own carrier
    /// (`docs/gauge_soN.md` §14.2) — exactly what the spinor base
    /// cases already do (§16.3). One frame convention for every catalog entry:
    /// a rediscovered defining block is then coherent with the stored one, so
    /// the §15 coherence guard and the alignment apply to it unchanged.
    fn seed_base(&mut self) -> Result<(), CatalogError> {
        // Rank guard (excluded low-rank isomorphisms) fires here via the seed.
        let seed = defining_seed(self.series, self.rank)?;
        let defining_irrep = self.defining_irrep()?;
        let expected = std::collections::BTreeMap::from([(defining_irrep.clone(), 1u32)]);
        let decomp = decompose(&Generators::from_seed(&seed), &expected)?;
        let defining = decomp.blocks()[0].generators().clone();
        let trivial_irrep = Irrep::trivial(self.series, self.rank)?;
        let trivial = Generators::trivial(self.series, self.rank);
        self.commit_one(trivial_irrep, trivial);
        self.commit_one(defining_irrep, defining);
        Ok(())
    }

    /// The defining (vector / fundamental) irrep `(1, 0, …, 0)` of this family.
    fn defining_irrep(&self) -> Result<Irrep, CatalogError> {
        let mut dynkin = vec![0i64; self.rank];
        dynkin[0] = 1;
        Ok(Irrep::from_dynkin(self.series, &dynkin)?)
    }

    /// Validate that `c` belongs to this family (same series and rank).
    fn check_group(&self, c: &Irrep) -> Result<(), CatalogError> {
        if c.series() != self.series || c.rank() != self.rank {
            return Err(CatalogError::WrongGroup {
                catalog: (self.series, self.rank),
                got: (c.series(), c.rank()),
            });
        }
        Ok(())
    }

    /// The generator set of `c`, materializing its canonical-parent chain on
    /// demand (atomic under the byte budget). Idempotent: a second call is a map
    /// lookup.
    ///
    /// # Errors
    /// - [`CatalogError::WrongGroup`] if `c` is not of this family.
    /// - [`CatalogError::BudgetExceeded`] if materialization would exceed the
    ///   budget (no partial state is committed).
    /// - [`CatalogError::Sweep`] if a sweep in the chain fails a production gate.
    #[cfg(test)]
    pub fn generators(&mut self, c: &Irrep) -> Result<&Generators, CatalogError> {
        self.check_group(c)?;
        self.ensure(c)?;
        Ok(self.store.get(c).expect("ensure guarantees presence"))
    }

    /// The Clebsch–Gordan isometry of `s1 ⊗ s2 → s3` (this crate's B/C/D public
    /// CGC surface, mirroring `crate::lie::sun::cgc`).
    ///
    /// The coupling is computed from the **queried** product `s1 ⊗ s2` (not from
    /// `s3`'s canonical parent): the CGC is defined relative to the canonical
    /// bases of `s1`, `s2` (materialized from the catalog) and `s3`. Every
    /// ill-posed triple is a typed error.
    ///
    /// # Errors
    /// - [`CatalogError::WrongGroup`] if the three irreps are not one family.
    /// - [`CatalogError::ZeroFusionChannel`] if `N^{s3}_{s1 s2} = 0` (`s3` does
    ///   not appear in `s1 ⊗ s2`) — the ill-posed-input guard.
    /// - [`CatalogError::BudgetExceeded`] / [`CatalogError::Sweep`] as for
    ///   [`generators`](Self::generators).
    pub fn cgc(&mut self, s1: &Irrep, s2: &Irrep, s3: &Irrep) -> Result<CatalogCgc, CatalogError> {
        self.check_group(s1)?;
        self.check_group(s2)?;
        self.check_group(s3)?;

        // Ill-posed-input guard: every N^c_ab = 0 is a typed error, decided by
        // the exact decomposition before any float work.
        let expected = directproduct(s1, s2)?;
        if expected.get(s3).copied().unwrap_or(0) == 0 {
            return Err(CatalogError::ZeroFusionChannel {
                a: s1.dynkin(),
                b: s2.dynkin(),
                c: s3.dynkin(),
            });
        }

        // Canonical bases of the two factors (and s3, for the debug-assert).
        self.ensure(s1)?;
        self.ensure(s2)?;
        self.ensure(s3)?;

        let g1 = self
            .store
            .get(s1)
            .expect("ensure succeeded for this irrep without an intervening reset")
            .clone();
        let g2 = self
            .store
            .get(s2)
            .expect("ensure succeeded for this irrep without an intervening reset")
            .clone();
        let product = Generators::product(&g1, &g2)?;
        let decomp = decompose(&product, &expected)?;

        // Collect the s3 copies in outer-multiplicity index order.
        let mut copies: Vec<&Block> = decomp.blocks().iter().filter(|b| b.irrep() == s3).collect();
        copies.sort_by_key(|b| b.outer_multiplicity().0);
        self.assemble_cgc(s1, s2, s3, &copies)
    }

    /// Every coupled channel of `s1 ⊗ s2` from a **single** decomposition sweep,
    /// one [`CatalogCgc`] per distinct coupled irrep `s3` (each byte-identical to
    /// [`cgc`](Self::cgc)`(s1, s2, s3)`).
    ///
    /// This is the sweep-once-per-product primitive the B/C/D F/R value tier
    /// ([`crate::cache::cache_bcd_cgc`]) is built on: the associativity/braiding
    /// gates request many different `s3` from the same `s1 ⊗ s2`, and
    /// [`cgc`](Self::cgc) re-runs the full `s1 ⊗ s2` sweep for each — this runs it
    /// once and hands back all channels. `pub(crate)`: the
    /// per-channel [`cgc`](Self::cgc) stays the public surface.
    ///
    /// # Errors
    /// - [`CatalogError::WrongGroup`] if `s1`/`s2` are not of this family.
    /// - [`CatalogError::BudgetExceeded`] / [`CatalogError::Sweep`] as for
    ///   [`generators`](Self::generators).
    pub(crate) fn cgc_product(
        &mut self,
        s1: &Irrep,
        s2: &Irrep,
    ) -> Result<Vec<CatalogCgc>, CatalogError> {
        self.check_group(s1)?;
        self.check_group(s2)?;
        self.ensure(s1)?;
        self.ensure(s2)?;

        let expected = directproduct(s1, s2)?;
        let g1 = self
            .store
            .get(s1)
            .expect("ensure succeeded for this irrep without an intervening reset")
            .clone();
        let g2 = self
            .store
            .get(s2)
            .expect("ensure succeeded for this irrep without an intervening reset")
            .clone();
        let product = Generators::product(&g1, &g2)?;
        let decomp = decompose(&product, &expected)?;

        // Group blocks by coupled irrep; assemble one CatalogCgc per channel.
        let mut by_irrep: std::collections::BTreeMap<Irrep, Vec<&Block>> =
            std::collections::BTreeMap::new();
        for b in decomp.blocks() {
            by_irrep.entry(b.irrep().clone()).or_default().push(b);
        }
        // Materialize each channel's canonical basis so the coherence guard in
        // `assemble_cgc` can compare against it (a production check now, so this
        // runs in release too, not only debug).
        let channels: Vec<Irrep> = by_irrep.keys().cloned().collect();
        for c in &channels {
            self.ensure(c)?;
        }
        let mut out = Vec::with_capacity(by_irrep.len());
        for (c, mut copies) in by_irrep {
            copies.sort_by_key(|b| b.outer_multiplicity().0);
            out.push(self.assemble_cgc(s1, s2, &c, &copies)?);
        }
        Ok(out)
    }

    /// Assemble a [`CatalogCgc`] for `s1 ⊗ s2 → s3` from its outer-multiplicity
    /// copies (already OM-sorted). Shared by [`cgc`](Self::cgc) and
    /// [`cgc_product`](Self::cgc_product) so both produce the identical isometry.
    ///
    /// Each copy is brought into `s3`'s stored canonical frame before its columns
    /// are appended: a copy already coherent to [`TOL_BASIS_COHERENT`]
    /// is used verbatim (bit-exact fast path); a copy in a rotated frame (an
    /// ill-conditioned QR) is aligned by [`align_block`] and re-verified. The
    /// coherence guard fires on the **post-alignment** residual, as
    /// [`CatalogError::BasisIncoherent`], when a frame cannot be aligned within tolerance (a genuinely different irrep or a
    /// numerically hopeless embedding). **No irrep is exempt** — every catalog
    /// entry, base cases included, is stored in the sweep's descending-weight
    /// frame, so the element-wise comparison is meaningful for all of them.
    fn assemble_cgc(
        &self,
        s1: &Irrep,
        s2: &Irrep,
        s3: &Irrep,
        copies: &[&Block],
    ) -> Result<CatalogCgc, CatalogError> {
        let stored = self
            .store
            .get(s3)
            .expect("callers ensure s3 before assembling its CGC");
        let (rows, d3) = copies[0].cgc_shape();
        let len = crate::lie::checked_size([rows, d3, copies.len()])
            .map_err(|_| BcdError::LabelOverflow)?;
        let mut cols = Vec::with_capacity(len);
        for b in copies {
            debug_assert_cartan_matches(b, stored)?;
            let raw = b.generators().coherence_residual(stored);
            if raw <= TOL_BASIS_COHERENT {
                // Already in the canonical frame: use the block CGC verbatim
                // (bit-exact fast path — alignment on a coherent block is the
                // identity up to sign, so this avoids perturbing stored values).
                cols.extend_from_slice(b.cgc());
            } else {
                // Rotated frame (ill-conditioned QR): align to the canonical
                // stored frame instead of bricking. The coherence guard runs on
                // the POST-alignment residual: a frame that still disagrees is a genuinely different irrep
                // or a numerically hopeless embedding and stays BasisIncoherent.
                let (aligned, residual) = align_block(b, stored)?;
                if residual > TOL_BASIS_COHERENT {
                    return Err(CatalogError::BasisIncoherent {
                        irrep: s3.dynkin(),
                        product: (s1.dynkin(), s2.dynkin()),
                        residual,
                    });
                }
                cols.extend_from_slice(&aligned.data);
            }
        }
        Ok(CatalogCgc {
            s1: s1.clone(),
            s2: s2.clone(),
            s3: s3.clone(),
            rows,
            d3,
            multiplicity: copies.len(),
            cols,
        })
    }

    /// Drop every discovered generator set and re-seed the base cases, returning
    /// the catalog to its just-constructed state. Re-materialization afterward is
    /// bitwise identical (the canonical-parent chain is a deterministic function
    /// of the exact data).
    #[cfg(test)]
    pub fn reset(&mut self) {
        self.try_reset()
            .expect("test catalog can regenerate its base fixtures");
    }
    pub(in crate::lie) fn try_reset(&mut self) -> Result<(), CatalogError> {
        self.store.clear();
        self.bytes = 0;
        self.seed_base()
    }

    // ---- materialization (compute-fully-then-commit) -----------------------

    /// Ensure `c`'s generators are committed, materializing its canonical-parent
    /// chain atomically: assemble every new set into a staging buffer, check the
    /// total against the budget, and commit only if it fits.
    fn ensure(&mut self, c: &Irrep) -> Result<(), CatalogError> {
        if self.store.contains_key(c) {
            return Ok(());
        }
        let mut staged: Vec<(Irrep, Generators)> = Vec::new();
        build_into(self.series, self.rank, &self.store, &mut staged, c)?;

        let needed = staged
            .iter()
            .try_fold(self.bytes, |n, (_, g)| {
                n.checked_add(gen_bytes_for(g.rank(), g.dim())?)
            })
            .ok_or(CatalogError::BudgetExceeded {
                limit: self.max_bytes,
                needed: usize::MAX,
            })?;
        if needed > self.max_bytes {
            // Atomic failure: discard the staging buffer, commit nothing.
            return Err(CatalogError::BudgetExceeded {
                limit: self.max_bytes,
                needed,
            });
        }
        for (k, v) in staged {
            self.commit_one(k, v);
        }
        Ok(())
    }

    /// Commit one generator set, charging its bytes.
    fn commit_one(&mut self, irrep: Irrep, gens: Generators) {
        self.bytes = self
            .bytes
            .checked_add(gen_bytes(&gens))
            .expect("construction or ensure checked the total charge before committing");
        self.store.insert(irrep, gens);
    }

    // ---- test / bench inspection -------------------------------------------

    /// Whether `c`'s generators are currently committed (no materialization).
    #[cfg(test)]
    pub(crate) fn is_materialized(&self, c: &Irrep) -> bool {
        self.store.contains_key(c)
    }

    /// The worst commutator residual of `c`'s stored generators (chain-depth
    /// error watch). `c` must already be materialized.
    #[cfg(test)]
    pub(crate) fn stored_commutator_residual(&self, c: &Irrep) -> Option<f64> {
        self.store.get(c).map(|g| g.max_commutator_residual())
    }
}

// ---- byte accounting -------------------------------------------------------

/// Conservative retained-byte charge for one generator set: the `r` dense
/// `D×D` raising operators plus the `r` length-`D` Cartan diagonals, over the
/// `f64` coefficient buffers, plus a fixed shell. This is a conservative charge
/// of catalog-owned generator entries, not allocator RSS or map scaffolding.
fn gen_bytes(g: &Generators) -> usize {
    gen_bytes_for(g.rank(), g.dim()).unwrap_or(usize::MAX)
}

fn gen_bytes_for(r: usize, d: usize) -> Option<usize> {
    d.checked_mul(d)?
        .checked_add(d)?
        .checked_mul(r)?
        .checked_mul(std::mem::size_of::<f64>())?
        .checked_add(std::mem::size_of::<Generators>())
}

// ---- the canonical parent rule (docs/gauge_soN.md §14) ---------------------

/// Whether `c` is a **fundamental spinor** — `ω_r` for `B_r`, `ω_{r-1}`/`ω_r`
/// for `D_r` — i.e. one of the spinor base cases (§14.2, §16). These are
/// exactly the spinor labels whose doubled weight is `(±1,…,±1)`: the
/// `≺`-minimal irreps of the spinor class, and the only ones that carry a
/// Clifford seed rather than a canonical parent.
fn is_spinor_base(c: &Irrep) -> bool {
    c.is_spinor() && c.two_partition().iter().all(|x| x.abs() == 1)
}

/// The **doubled** box count of an irrep's highest weight: `Σ_i |2λ_i|` over
/// the ε-basis partition (§14.1). Strictly monotone under adding/removing a box
/// (which moves it by 2), and — unlike `dim` — monotone in **every** coordinate
/// including the D-series sign-carrying last part. The primary `≺` component.
///
/// Doubling is a uniform rescaling of `Σ|λ_i|`, so `≺` — and every parent
/// chosen through it — agrees with the undoubled order on the tensor irreps; what doubling
/// buys is that a spinor's half-integer weight has an integer box count too
/// (`box'(ω_r) = r`).
fn box_count(c: &Irrep) -> Result<i64, BcdError> {
    c.two_partition().iter().try_fold(0i64, |sum, &x| {
        x.checked_abs()
            .and_then(|x| sum.checked_add(x))
            .ok_or(BcdError::LabelOverflow)
    })
}

/// The `≺` sort key of an irrep: `(box_count, dim, dynkin)` (§14.1). Box count
/// is the primary component so that removing a box always yields a strictly
/// smaller irrep — the fact the existence proof (§14.4) needs and that `dim`
/// alone fails for the D-series chirality pair (`dim` is not monotone in the last
/// partition coordinate: partition `(1,1,0)` has dim 15 > `(1,1,±1)` dim 10).
/// `≺` is a **well order**: box count is a non-negative integer and, at a fixed
/// box count and rank, only finitely many irreps exist (§14.1).
#[cfg(test)]
fn prec_key(c: &Irrep) -> (i64, BigInt, Vec<i64>) {
    try_prec_key(c).expect("test fixture has a representable box count")
}
fn try_prec_key(c: &Irrep) -> Result<(i64, BigInt, Vec<i64>), BcdError> {
    Ok((box_count(c)?, c.dim(), c.dynkin()))
}

/// The canonical parent pair `(a, b)` of a non-base irrep `c` (§14): among all
/// pairs with `a ≺ c`, `b ≺ c`, and `c ∈ a ⊗ b` (exact decomposition), the minimum under
/// the pair order `(dim_a + dim_b, dim_a, dynkin_a, dynkin_b)`. Returns the pair
/// in canonical `a ⪯ b` form (the order's tie-break fixes which is `a`).
///
/// Existence is guaranteed for every non-base `c` (§14: the pair
/// `(defining, c-minus-a-box)` is always admissible), so the returned `Option`
/// is `None` only if called on a base case (trivial/defining), which the caller
/// never does — those are pre-seeded and short-circuited.
#[cfg(test)]
fn canonical_parent(series: Series, rank: usize, c: &Irrep) -> Option<(Irrep, Irrep)> {
    try_canonical_parent(series, rank, c)
        .expect("test fixture has representable canonical-parent arithmetic")
}
fn try_canonical_parent(
    series: Series,
    rank: usize,
    c: &Irrep,
) -> Result<Option<(Irrep, Irrep)>, CatalogError> {
    /// A candidate parent pair with its `key(a,b)` (§14.2).
    struct Cand {
        sum: BigInt,
        dim_a: BigInt,
        dynkin_a: Vec<i64>,
        dynkin_b: Vec<i64>,
        a: Irrep,
        b: Irrep,
    }
    impl Cand {
        /// The pair order `(dim_a + dim_b, dim_a, dynkin_a, dynkin_b)`.
        fn key(&self) -> (&BigInt, &BigInt, &Vec<i64>, &Vec<i64>) {
            (&self.sum, &self.dim_a, &self.dynkin_a, &self.dynkin_b)
        }
    }

    let key_c = try_prec_key(c)?;
    // All irreps strictly `≺ c` — the finite candidate set for `a` and `b`.
    let below = try_irreps_below(series, rank, c)?;

    let mut best: Option<Cand> = None;
    // Iterate `a` in ascending `≺` order; prune once `2·dim_a` exceeds the best
    // sum found (with `a ⪯ b`, `dim_a + dim_b ≥ 2·dim_a`; the minimum pair is
    // always reached via its smaller factor before this fires — §14.4).
    for a in &below {
        let dim_a = a.dim();
        if let Some(cur) = &best {
            if &dim_a * 2 > cur.sum {
                break; // `below` is sorted ascending by (dim, dynkin).
            }
        }
        // `c ∈ a ⊗ b`  ⟺  `b ∈ a* ⊗ c` (Frobenius reciprocity). Enumerate the
        // candidate `b` directly from that product rather than looping all irreps.
        let prod = directproduct(&a.dual(), c)?;
        for b in prod.keys() {
            if try_prec_key(b)? >= key_c {
                continue; // require b ≺ c
            }
            let cand = Cand {
                sum: &dim_a + b.dim(),
                dim_a: dim_a.clone(),
                dynkin_a: a.dynkin(),
                dynkin_b: b.dynkin(),
                a: a.clone(),
                b: b.clone(),
            };
            if best.as_ref().is_none_or(|cur| cand.key() < cur.key()) {
                best = Some(cand);
            }
        }
    }
    Ok(best.map(|c| (c.a, c.b)))
}

/// The irreps `x` of `(series, rank)` with `x ≺ c` that are **admissible as
/// parents of `c`** (§14.2, class-indexed candidate-set restriction), sorted
/// ascending by `(dim, dynkin)` — the order the pruning in
/// [`canonical_parent`] relies on.
///
/// **Class restriction.** If `c` lies in the
/// tensor sublattice of `P/Q`, the candidate set is restricted to the tensor
/// sublattice; a spinor is never a parent of a tensor irrep, so every
/// `SO(N)`/`Sp(2N)` coefficient is independent of the `Spin(N)` labels. If
/// `c` is a spinor, both classes are candidates (a spinor's parents are a
/// spinor and a tensor irrep — no product of tensor irreps contains a spinor).
///
/// Enumerated by a depth-first walk over doubled weights `2λ` (ε-basis,
/// nonincreasing, `≥ 0`, all of one parity; the D series additionally emits the
/// `λ_r < 0` chiral partner) bounded by the doubled **box count**
/// `Σ|2λ_i| ≤ box_count(c)`. Box count is monotone in every coordinate
/// (including the D-series last part, where `dim` is not), so the
/// prune is exact for all three series. Every `x ≺ c` has
/// `box_count(x) ≤ box_count(c)`, so the walk is a complete superset; the
/// `retain` keeps exactly `{ x : x ≺ c }`.
#[cfg(test)]
fn irreps_below(series: Series, rank: usize, c: &Irrep) -> Vec<Irrep> {
    try_irreps_below(series, rank, c).expect("test fixture has representable box count")
}
fn try_irreps_below(series: Series, rank: usize, c: &Irrep) -> Result<Vec<Irrep>, BcdError> {
    let max_boxes = box_count(c)?;
    let key_c = try_prec_key(c)?;
    let mut out: Vec<Irrep> = Vec::new();
    let mut cur = vec![0i64; rank];
    // Parity 0 = the tensor sublattice, parity 1 = the spinor class.
    enum_partitions(series, rank, max_boxes, 0, 0, 0, &mut cur, &mut out)?;
    if c.is_spinor() {
        enum_partitions(series, rank, max_boxes, 1, 0, 0, &mut cur, &mut out)?;
    }
    let mut below = Vec::new();
    for x in out {
        if try_prec_key(&x)? < key_c {
            below.push(x);
        }
    }
    below.sort_by_key(|x| (x.dim(), x.dynkin()));
    Ok(below)
}

/// Walk the doubled dominant weights of one class (`parity` 0 = tensor, 1 =
/// spinor) with `Σ|2λ_i| ≤ max_boxes`.
#[allow(clippy::too_many_arguments)]
fn enum_partitions(
    series: Series,
    rank: usize,
    max_boxes: i64,
    parity: i64,
    pos: usize,
    used: i64,
    cur: &mut Vec<i64>,
    out: &mut Vec<Irrep>,
) -> Result<(), BcdError> {
    if pos == rank {
        push_partition_irrep(series, cur, out);
        return Ok(());
    }
    let upper = if pos == 0 { max_boxes } else { cur[pos - 1] };
    let mut v = parity;
    while v <= upper {
        // Prune on box count: monotone in v for every coordinate ⇒ safe break.
        if v > max_boxes - used {
            break;
        }
        cur[pos] = v;
        enum_partitions(
            series,
            rank,
            max_boxes,
            parity,
            pos + 1,
            super::add_weight(used, v)?,
            cur,
            out,
        )?;
        let Some(next) = v.checked_add(2) else {
            break;
        };
        v = next;
    }
    cur[pos] = parity;
    Ok(())
}

/// Emit the (non-negative) partition `cur` as an irrep, and — for the D series
/// with `λ_r > 0` — its chiral partner `λ_r ↦ -λ_r` (a distinct tensor irrep of
/// the same box count and dim). `irreps_below`'s `retain` applies the `≺` filter.
fn push_partition_irrep(series: Series, cur: &[i64], out: &mut Vec<Irrep>) {
    out.push(make_irrep(series, cur.to_vec()));
    if series == Series::D {
        let last = cur.len() - 1;
        if cur[last] > 0 {
            let mut w = cur.to_vec();
            w[last] = -w[last];
            out.push(make_irrep(series, w));
        }
    }
}

/// Construct an [`Irrep`] directly from a doubled ε-basis weight `2λ` (a
/// descendant module of `bcd` may build the private struct). The enumeration
/// only ever produces valid dominant weights of the cover, so no validation is
/// needed here.
fn make_irrep(series: Series, two_weight: Vec<i64>) -> Irrep {
    super::Irrep::from_two_weight(series, two_weight)
}

// ---- recursive build into the staging buffer -------------------------------

/// Look up `c`'s generators in the committed store or the staging buffer.
fn lookup<'a>(
    store: &'a HashMap<Irrep, Generators>,
    staged: &'a [(Irrep, Generators)],
    c: &Irrep,
) -> Option<&'a Generators> {
    store
        .get(c)
        .or_else(|| staged.iter().find(|(k, _)| k == c).map(|(_, g)| g))
}

/// Recursively assemble the generator sets `c`'s canonical-parent chain needs
/// but the store does not yet have, into `staged` (no commit, no budget check —
/// [`CanonicalCatalog::ensure`] does both once the whole chain is staged).
///
/// Harvest discipline: decomposing the canonical parent yields blocks
/// for several irreps; a block's generators are staged **only** if that irrep has
/// no generators yet **and** this product is its canonical parent. A rediscovery
/// (the irrep already committed/staged) never writes — instead it debug-asserts
/// Cartan-spectrum agreement (the `clebsch.cc:6710-6718 @ dd2cc7e` cross-copy
/// `normDiff` check, replaced by-design; §14).
fn build_into(
    series: Series,
    rank: usize,
    store: &HashMap<Irrep, Generators>,
    staged: &mut Vec<(Irrep, Generators)>,
    c: &Irrep,
) -> Result<(), CatalogError> {
    if lookup(store, staged, c).is_some() {
        return Ok(()); // already committed or staged (includes the base cases)
    }

    // Spinor base case (§14.2): the fundamental spinors are not in
    // any product of already-materialized irreps — they carry their own exact
    // Clifford seed. Seeded on demand rather than at construction, so a catalog
    // that is only ever asked for tensor irreps builds no spinor matrices and
    // its byte accounting counts tensor irreps only.
    if is_spinor_base(c) {
        for (label, seed) in spinor_seeds(series, rank)? {
            if Irrep::from_dynkin_in(&series.cover_group(rank), &label)? == *c {
                // The Clifford seed is brought into the canonical frame by one
                // pass of the §1–§8 sweep over its own carrier (§16). That is
                // what makes a spinor base case behave exactly like any other
                // catalog entry: every later rediscovery of `c` is produced by
                // the same sweep, so the §15 coherence guard and the alignment
                // apply to it unchanged.
                let raw = Generators::from_seed(&seed);
                let expected = std::collections::BTreeMap::from([(c.clone(), 1u32)]);
                let decomp = decompose(&raw, &expected)?;
                let gens = decomp.blocks()[0].generators().clone();
                staged.push((c.clone(), gens));
                return Ok(());
            }
        }
    }

    // Non-base c: its canonical parent exists (§14.4 existence argument). The
    // error path is unreachable by that theorem; kept as defense-in-depth.
    let (a, b) = try_canonical_parent(series, rank, c)?
        .ok_or_else(|| CatalogError::NoCanonicalParent { dynkin: c.dynkin() })?;
    build_into(series, rank, store, staged, &a)?;
    build_into(series, rank, store, staged, &b)?;

    let ga = lookup(store, staged, &a)
        .expect("successful recursive build_into stages this irrep or finds it in the store")
        .clone();
    let gb = lookup(store, staged, &b)
        .expect("successful recursive build_into stages this irrep or finds it in the store")
        .clone();
    let product = Generators::product(&ga, &gb)?;
    let expected = directproduct(&a, &b)?;
    let decomp = decompose(&product, &expected)?;

    for block in decomp.blocks() {
        let ci = block.irrep();
        if let Some(existing) = lookup(store, staged, ci) {
            // Rediscovery: never write; assert the intrinsic Cartan spectrum
            // agrees (deviation-by-design from QSpace's normDiff replacement).
            debug_assert_cartan_matches(block, existing)?;
            continue;
        }
        // Append only the outer-multiplicity-0 copy, and only when THIS product
        // is ci's canonical parent (query-order-independent gauge).
        if block.outer_multiplicity().0 != 0 {
            continue;
        }
        if try_canonical_parent(series, rank, ci)?.as_ref() == Some(&(a.clone(), b.clone())) {
            staged.push((ci.clone(), block.generators().clone()));
        }
    }

    debug_assert!(
        lookup(store, staged, c).is_some(),
        "the canonical parent of c must produce c's block"
    );
    Ok(())
}

/// Debug-assert that a rediscovered block's Cartan (weight) spectrum matches the
/// stored generator set's — the cheap, loud analogue of QSpace's `normDiff`
/// cross-copy check (`clebsch.cc:6712-6718 @ dd2cc7e`).
///
/// Compared as a **multiset** of per-state weight vectors, not state-by-state:
/// the weight *content* of an irrep is gauge-independent, but the state *order*
/// is not, and this cheap check deliberately makes no order claim — the
/// element-wise statement (which binds every entry, base cases included) is the production coherence guard in
/// [`assemble_cgc`](CanonicalCatalog::assemble_cgc). Weights are
/// integer Cartan eigenvalues (snapped in the sweep, §6), so they compare exactly
/// after rounding.
fn debug_assert_cartan_matches(block: &Block, stored: &Generators) -> Result<(), SweepError> {
    debug_assert_eq!(
        block.dim(),
        stored.dim(),
        "rediscovered block dim disagrees with stored generators"
    );
    if !cfg!(debug_assertions) {
        return Ok(());
    }
    let rank = stored.rank();
    let d = stored.dim();
    // Compared at the doubled scale: a carrier with a spinor factor has
    // half-integer Cartan eigenvalues, and `2·weight` is an integer for every
    // irrep of the cover (§6, §16).
    let round = |x: f64| super::sweep::rounded_weight((2.0 * x).round());
    let mut block_w: Vec<Vec<i64>> = (0..d)
        .map(|s| {
            (0..rank)
                .map(|j| round(block.weight(s, j)))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    let mut stored_w: Vec<Vec<i64>> = (0..d)
        .map(|s| {
            (0..rank)
                .map(|j| round(stored.cartan_diag(j)[s]))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    block_w.sort_unstable();
    stored_w.sort_unstable();
    debug_assert_eq!(
        block_w, stored_w,
        "rediscovered block weight multiset disagrees with stored generators"
    );
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod arithmetic_tests {
    use super::*;
    #[test]
    fn box_count_bound() {
        assert_eq!(gen_bytes_for(usize::MAX, 1), None);
        let c = Irrep::from_two_weight(Series::B, vec![i64::MAX; 2]);
        assert_eq!(
            try_irreps_below(Series::B, 2, &c),
            Err(BcdError::LabelOverflow)
        );
        assert_eq!(
            try_canonical_parent(Series::B, 2, &c),
            Err(CatalogError::Label(BcdError::LabelOverflow))
        );
    }
}
