//! Finite fusion categories from pinned, auditable tables instead of formulas.
//!
//! The classification of small fusion categories is a body of computed data,
//! not a body of formulas. The authority is the AnyonWiki database
//! (`anyonwiki/AnyonWikiDatabase`, MIT, commit `badbfe2`), which stores every
//! symbol exactly, as an algebraic number given by its minimal polynomial and
//! root index:
//!
//! ```text
//! exact algebraic symbol  ->  correctly rounded binary64 (offline generator)
//!                         ->  shortest round-trip decimal table  ->  Complex64
//! ```
//!
//! The rounding is done once, offline, by `tools/anyonwiki/gen_category_tables.py`;
//! the crate parses the committed output and adds no runtime dependency.
//!
//! - Not a solver. Nothing here solves the pentagon or hexagon equations or
//!   regenerates `F`/`R` from `N`. Those equations appear only as
//!   *consistency* tests, run over every shipped entry.
//! - Not online. Nothing invokes Julia, Mathematica, or the network.
//!
//! # What ships
//!
//! Every multiplicity-free entry of the pinned database that is unitary and
//! whose quantum dimensions are `|1 / F^{a ā a}_a[𝟙, 𝟙]|` — exactly the
//! entries for which the TensorKitSectors derivations of dimension,
//! Frobenius–Schur phase and twist used here are the database's own — up to
//! rank 4: 74 categories, 58 of them braided. The `anyonwiki-unitary` feature
//! adds the 303 such entries of rank 5–7 (377 in all, 270 braided); without
//! it they are refused as [`UnsupportedReason::NotShipped`]. Every other entry
//! of the database is known by uuid and code and refused with
//! [`TableError::Unsupported`] and its [`UnsupportedReason`].
//!
//! A category's identity is the pair (database commit, uuid). The
//! `anyonwiki_code` is a version-dependent label: [`TableCategory::by_code`]
//! is valid only for the pinned commit (CategoryData.jl's codes, for example,
//! differ for most braided entries).
//!
//! # Gauge
//!
//! Every entry is the database's gauge verbatim, named constructors
//! included; nothing is converted to TensorKitSectors' gauge. Where
//! TensorKitSectors has the category in closed form the two are
//! gauge-equivalent: [`TableCategory::fibonacci`] is in TensorKitSectors'
//! gauge exactly, [`TableCategory::ising`] differs from `IsingAnyon` by a
//! vertex gauge (see there).
//!
//! Every upstream pin, hash and the source-to-`Complex64` conversion epoch is
//! tabulated in the crate's `docs/references_categories.md`; the
//! machine-readable form is [`TableProvenance`].

use num_complex::Complex64;

// Generated (`tools/anyonwiki/gen_category_tables.py`): kept byte-equal to
// the generator output, so rustfmt leaves it alone.
#[rustfmt::skip]
mod catalog;
mod provenance;
mod table;

pub use provenance::{TableProvenance, PROJECTION_EPOCH};
pub use table::{TableError, TableObject, UnsupportedReason};

use table::MultiplicityFreeTable;

/// One shipped entry, as generated into `catalog.rs`. The texts are the
/// committed files embedded verbatim: they are the identity-bearing bytes,
/// so they are `include_str!`-ed rather than re-encoded.
#[derive(Debug)]
struct CategoryEntry {
    uuid: &'static str,
    code: [u8; 7],
    n: &'static str,
    f: &'static str,
    /// `None` for a fusion-only entry.
    r: Option<&'static str>,
    provenance: TableProvenance,
}

const FIBONACCI_UUID: &str = "37fa4e42-97c6-11f1-9d0f-9b7fe5c504f9";
const ISING_UUID: &str = "37fa4e56-97c6-11f1-907a-eb6ab8e8bc11";

/// A multiplicity-free unitary fusion category from the pinned AnyonWiki
/// database, in the source gauge.
///
/// Objects are the one-based [`TableObject`] labels of the source files, in
/// the fusion ring's element order: `TableObject(1)` is the unit. Index
/// order follows TensorKitSectors: `F^{abc}_d[e, f]`, `R^{ab}_c`.
///
/// The infallible queries take labels by precondition and panic on a label
/// outside `1..=rank` (a flat table cannot tell an out-of-range label from
/// another entry), and [`r_symbol`](Self::r_symbol) / [`twist`](Self::twist)
/// panic on a category without a braiding ([`is_braided`](Self::is_braided));
/// an in-range inadmissible symbol is exact zero. The `_checked` variants
/// return [`TableError`] instead.
///
/// # Examples
///
/// ```
/// use fusion_symbols::tables::{TableCategory, TableObject};
///
/// let fib = TableCategory::fibonacci();
/// let tau = TableObject(2);
/// assert_eq!(fib.fusion(tau, tau).count(), 2);
/// assert!((fib.dim(tau) - (1.0 + 5f64.sqrt()) / 2.0).abs() < 1e-15);
///
/// let rep_s3 = TableCategory::by_code([3, 1, 0, 2, 2, 0, 1])?; // fusion-only
/// assert!(!rep_s3.is_braided());
/// # Ok::<(), fusion_symbols::tables::TableError>(())
/// ```
#[derive(Debug)]
pub struct TableCategory {
    entry: &'static CategoryEntry,
    table: MultiplicityFreeTable,
    // Derived once at construction from F and R only, exactly as
    // TensorKitSectors 0.3.9 derives them (docs/references_categories.md,
    // "Derived quantities"). Initialisation, not a cache layer with a policy.
    dims: Vec<f64>,
    frobenius_schur: Vec<Complex64>,
    /// `None` for a fusion-only entry.
    twists: Option<Vec<Complex64>>,
    fingerprint: Vec<u8>,
}

impl TableCategory {
    /// The category with AnyonWiki uuid `uuid` (database commit `badbfe2`).
    ///
    /// Parses and validates the embedded tables: there is no unchecked
    /// constructor, because a provider that skipped the import checks would be
    /// indistinguishable from a correct one at every later call site.
    ///
    /// # Errors
    ///
    /// [`TableError::UnknownCategory`] if the pinned database has no such
    /// entry, [`TableError::Unsupported`] if it has one this crate does not
    /// provide. The import variants are unreachable for the shipped data (the
    /// crate's tests construct every entry).
    pub fn by_uuid(uuid: &str) -> Result<Self, TableError> {
        let (_, _, unsupported) = catalog::INDEX
            .iter()
            .find(|(id, _, _)| *id == uuid)
            .ok_or(TableError::UnknownCategory)?;
        if let Some(reason) = *unsupported {
            return Err(TableError::Unsupported { reason });
        }
        let entry = catalog::SHIPPED
            .iter()
            .find(|entry| entry.uuid == uuid)
            .expect("generator invariant: every supported INDEX row is in SHIPPED (tested)");
        Self::from_entry(entry)
    }

    /// The category with `anyonwiki_code` `code` = `[rank, multiplicity,
    /// non-self-dual objects, ring, associator, braiding (0 = none),
    /// pivotal]`, **as numbered in database commit `badbfe2`**. Codes are not
    /// stable across database versions; [`by_uuid`](Self::by_uuid) is.
    ///
    /// # Errors
    ///
    /// As [`by_uuid`](Self::by_uuid).
    pub fn by_code(code: [u8; 7]) -> Result<Self, TableError> {
        let (uuid, _, _) = catalog::INDEX
            .iter()
            .find(|(_, other, _)| *other == code)
            .ok_or(TableError::UnknownCategory)?;
        Self::by_uuid(uuid)
    }

    /// Fibonacci, `anyonwiki_code [2,1,0,2,1,2,1]` (uuid
    /// `37fa4e42-97c6-11f1-9d0f-9b7fe5c504f9`): `TableObject(1)` = `𝟙`,
    /// `TableObject(2)` = `τ`.
    ///
    /// This is TensorKitSectors' `FibonacciAnyon` in its own gauge and
    /// braiding (`R^{ττ}_𝟙 = e^{+4πi/5}`; code `[2,1,0,2,1,1,1]` is the
    /// mirror). The values are correctly rounded, so they can differ from
    /// TensorKitSectors' by an ulp (`F^{τττ}_τ[𝟙, 𝟙]`: `0.6180339887498949`
    /// here, `1/φ = 0.6180339887498948` there).
    pub fn fibonacci() -> Self {
        // Infallible: the entry is compiled in, and
        // `tests::every_shipped_entry_constructs` builds it.
        Self::by_uuid(FIBONACCI_UUID).expect("the shipped Fibonacci entry is valid (tested)")
    }

    /// Ising, `anyonwiki_code [3,1,0,1,1,1,1]` (uuid
    /// `37fa4e56-97c6-11f1-907a-eb6ab8e8bc11`), in AnyonWiki's object order:
    /// `TableObject(1)` = `𝟙`, `TableObject(2)` = `ψ`, `TableObject(3)` = `σ`
    /// (TensorKitSectors' `IsingAnyon` orders `I, σ, ψ`).
    ///
    /// Its braiding is `IsingAnyon`'s (`R^{σσ}_𝟙 = e^{-iπ/8}`; code
    /// `[3,1,0,1,1,4,1]` is the mirror) and its R-symbols are identical. Its
    /// F-symbols are `IsingAnyon`'s in a different gauge: with the vertex gauge
    /// `u^{σσ}_ψ = −1` (every other vertex `1`),
    /// `F_here = F_TKS · u^{ab}_e u^{ec}_d / (u^{bc}_f u^{af}_d)`, which flips the
    /// sign of six entries, e.g. `F^{σσσ}_σ[𝟙, ψ] = −1/√2` here.
    pub fn ising() -> Self {
        // Infallible: as `fibonacci`.
        Self::by_uuid(ISING_UUID).expect("the shipped Ising entry is valid (tested)")
    }

    /// The uuids of every shipped entry, in `anyonwiki_code` order.
    pub fn shipped_uuids() -> impl ExactSizeIterator<Item = &'static str> {
        catalog::SHIPPED.iter().map(|entry| entry.uuid)
    }

    fn from_entry(entry: &'static CategoryEntry) -> Result<Self, TableError> {
        let rank = usize::from(entry.code[0]);
        let table = MultiplicityFreeTable::parse(rank, entry.n, entry.f, entry.r.unwrap_or(""))?;

        let unit_f = |a: usize| table.fsymbol(a, table.dual(a), a, a, 0, 0);
        // TensorKitSectors `dim_from_Fsymbol`: abs(1 / F[a, ā, a, a, 1, 1]).
        let dims: Vec<f64> = (0..rank)
            .map(|a| (Complex64::new(1.0, 0.0) / unit_f(a)).norm())
            .collect();
        // `frobenius_schur_phase_from_Fsymbol`: sign(F[a, ā, a, a, 1, 1]).
        let frobenius_schur: Vec<Complex64> = (0..rank)
            .map(|a| {
                let value = unit_f(a);
                value / value.norm()
            })
            .collect();
        // `twist_from_Rsymbol`: Σ_{b ∈ a⊗a} d_b/d_a R^{aa}_b, ascending b.
        let twists = entry.r.map(|_| {
            (0..rank)
                .map(|a| {
                    table
                        .channels(a, a)
                        .map(|b| Complex64::new(dims[b] / dims[a], 0.0) * table.rsymbol(a, a, b))
                        .fold(Complex64::new(0.0, 0.0), |sum, term| sum + term)
                })
                .collect()
        });

        Ok(Self {
            entry,
            table,
            dims,
            frobenius_schur,
            twists,
            fingerprint: fingerprint_bytes(&entry.provenance, entry.n, entry.f, entry.r),
        })
    }

    /// The AnyonWiki uuid: with the database commit, the category's identity.
    pub fn uuid(&self) -> &'static str {
        self.entry.uuid
    }

    /// The `anyonwiki_code` in database commit `badbfe2`.
    pub fn code(&self) -> [u8; 7] {
        self.entry.code
    }

    /// The frozen lineage of this category's tables.
    pub fn provenance(&self) -> &'static TableProvenance {
        &self.entry.provenance
    }

    /// Number of simple objects.
    pub fn rank(&self) -> usize {
        self.table.rank()
    }

    /// Whether the category is braided. Without a braiding there are no
    /// R-symbols and no twists ([`TableError::NoBraiding`]).
    pub fn is_braided(&self) -> bool {
        self.twists.is_some()
    }

    /// The category's objects in the frozen source order.
    pub fn objects(&self) -> impl Iterator<Item = TableObject> {
        (1..=self.table.rank() as u16).map(TableObject)
    }

    /// The tensor unit, `TableObject(1)`.
    pub fn unit(&self) -> TableObject {
        TableObject(1)
    }

    /// Opaque authority fingerprint: the provenance record plus the pinned
    /// source files, byte for byte.
    ///
    /// Compare by equality only, never parse. It changes exactly when a table
    /// byte, a provenance field or the projection epoch changes, including an
    /// edit too small to move any `f64`.
    pub fn authority_fingerprint(&self) -> &[u8] {
        &self.fingerprint
    }

    /// Quantum dimension `d_a = |1 / F^{a ā a}_a[𝟙, 𝟙]|`.
    ///
    /// # Panics
    ///
    /// If `a` is not an object of the category.
    pub fn dim(&self, a: TableObject) -> f64 {
        self.dims[self.index_or_panic("dim", a)]
    }

    /// The dual object: the unique `b` with `N^{ab}_𝟙 = 1`.
    ///
    /// # Panics
    ///
    /// If `a` is not an object of the category.
    pub fn dual(&self, a: TableObject) -> TableObject {
        object(self.table.dual(self.index_or_panic("dual", a)))
    }

    /// Frobenius–Schur phase `κ_a = F^{a ā a}_a[𝟙, 𝟙] / |F^{a ā a}_a[𝟙, 𝟙]|`.
    ///
    /// # Panics
    ///
    /// If `a` is not an object of the category.
    pub fn frobenius_schur(&self, a: TableObject) -> Complex64 {
        self.frobenius_schur[self.index_or_panic("frobenius_schur", a)]
    }

    /// Twist `θ_a = Σ_{b ∈ a⊗a} d_b/d_a · R^{aa}_b`.
    ///
    /// # Panics
    ///
    /// If the category is not braided, or `a` is not an object of it.
    pub fn twist(&self, a: TableObject) -> Complex64 {
        self.twist_checked(a)
            .unwrap_or_else(|error| panic!("twist precondition violated: {error}"))
    }

    /// Checked [`twist`](Self::twist): [`TableError::NoBraiding`] or
    /// [`TableError::UnknownObject`] instead of a panic.
    pub fn twist_checked(&self, a: TableObject) -> Result<Complex64, TableError> {
        let twists = self.twists.as_ref().ok_or(TableError::NoBraiding)?;
        Ok(twists[self.index(a)?])
    }

    /// Fusion channels of `a ⊗ b` in ascending label order (TensorKitSectors
    /// filters `values(I)` in that order).
    ///
    /// # Panics
    ///
    /// If `a` or `b` is not an object of the category.
    pub fn fusion(&self, a: TableObject, b: TableObject) -> impl Iterator<Item = TableObject> + '_ {
        self.table
            .channels(
                self.index_or_panic("fusion", a),
                self.index_or_panic("fusion", b),
            )
            .map(object)
    }

    /// Fusion multiplicity `N^{ab}_c`.
    ///
    /// # Panics
    ///
    /// If a label is not an object of the category.
    pub fn nsymbol(&self, a: TableObject, b: TableObject, c: TableObject) -> u32 {
        let i = |x| self.index_or_panic("nsymbol", x);
        self.table.nsymbol(i(a), i(b), i(c))
    }

    /// F-symbol `F^{abc}_d[e, f]`; exact zero for an unlisted entry.
    ///
    /// # Panics
    ///
    /// If a label is not an object of the category.
    #[allow(clippy::too_many_arguments)]
    pub fn f_symbol(
        &self,
        a: TableObject,
        b: TableObject,
        c: TableObject,
        d: TableObject,
        e: TableObject,
        f: TableObject,
    ) -> Complex64 {
        let i = |x| self.index_or_panic("f_symbol", x);
        self.table.fsymbol(i(a), i(b), i(c), i(d), i(e), i(f))
    }

    /// Checked [`f_symbol`](Self::f_symbol): an error for an unknown label or
    /// an inadmissible vertex instead of a panic or a zero.
    #[allow(clippy::too_many_arguments)]
    pub fn f_symbol_checked(
        &self,
        a: TableObject,
        b: TableObject,
        c: TableObject,
        d: TableObject,
        e: TableObject,
        f: TableObject,
    ) -> Result<Complex64, TableError> {
        for x in [a, b, c, d, e, f] {
            self.index(x)?;
        }
        for (left, right, coupled) in [(a, b, e), (e, c, d), (b, c, f), (a, f, d)] {
            self.check_vertex(left, right, coupled)?;
        }
        Ok(self.f_symbol(a, b, c, d, e, f))
    }

    /// R-symbol `R^{ab}_c`; exact zero for an unlisted entry.
    ///
    /// # Panics
    ///
    /// If the category is not braided, or a label is not an object of it.
    pub fn r_symbol(&self, a: TableObject, b: TableObject, c: TableObject) -> Complex64 {
        if !self.is_braided() {
            panic!("r_symbol precondition violated: {}", TableError::NoBraiding);
        }
        let i = |x| self.index_or_panic("r_symbol", x);
        self.table.rsymbol(i(a), i(b), i(c))
    }

    /// Checked [`r_symbol`](Self::r_symbol): [`TableError::NoBraiding`] for a
    /// category without a braiding, otherwise an error for an unknown label
    /// or an inadmissible vertex, instead of a panic or a zero.
    pub fn r_symbol_checked(
        &self,
        a: TableObject,
        b: TableObject,
        c: TableObject,
    ) -> Result<Complex64, TableError> {
        if !self.is_braided() {
            return Err(TableError::NoBraiding);
        }
        for x in [a, b, c] {
            self.index(x)?;
        }
        self.check_vertex(a, b, c)?;
        Ok(self.r_symbol(a, b, c))
    }

    fn check_vertex(
        &self,
        left: TableObject,
        right: TableObject,
        coupled: TableObject,
    ) -> Result<(), TableError> {
        if self.nsymbol(left, right, coupled) == 0 {
            Err(TableError::NotAdmissible {
                left,
                right,
                coupled,
            })
        } else {
            Ok(())
        }
    }

    fn index(&self, a: TableObject) -> Result<usize, TableError> {
        let label = usize::from(a.label());
        if (1..=self.rank()).contains(&label) {
            Ok(label - 1)
        } else {
            Err(TableError::UnknownObject { object: a })
        }
    }

    fn index_or_panic(&self, method: &str, a: TableObject) -> usize {
        self.index(a)
            .unwrap_or_else(|error| panic!("{method} precondition violated: {error}"))
    }
}

/// The exact byte composition of the authority fingerprint.
fn fingerprint_bytes(
    provenance: &TableProvenance,
    n_text: &str,
    f_text: &str,
    r_text: Option<&str>,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    provenance.write_canonical_bytes(&mut bytes);
    // The source text, not a re-serialisation of the parsed f64s, which would
    // hide a parser change behind identical output. A fusion-only entry keeps
    // an empty, still tagged, R section.
    for (tag, text) in [
        ("Nsymbols", n_text),
        ("Fsymbols", f_text),
        ("Rsymbols", r_text.unwrap_or("")),
    ] {
        bytes.extend_from_slice(b"--- ");
        bytes.extend_from_slice(tag.as_bytes());
        bytes.push(b'\n');
        bytes.extend_from_slice(text.as_bytes());
    }
    bytes
}

fn object(index: usize) -> TableObject {
    TableObject(index as u16 + 1)
}

#[cfg(test)]
mod tests;
