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

use crate::category::{BraidingStyle, EnumerationError, FusionCategory, FusionStyle};
use num_complex::Complex64;
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

// Generated (`tools/anyonwiki/gen_category_tables.py`): kept byte-equal to
// the generator output, so rustfmt leaves it alone.
#[rustfmt::skip]
mod catalog;
mod parser;
mod provenance;

pub use parser::{TableError, TableObject, UnsupportedReason};
pub use provenance::{TableProvenance, PROJECTION_EPOCH};

use parser::MultiplicityFreeTable;

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
/// Objects are validated by [`Self::object`] and carry a private catalog tag.
/// Every object-taking query except `contains` panics on foreign objects before
/// checking capabilities.
/// Inadmissible symbols are exact zero. On fusion-only entries, `r_symbol`
/// returns [`TableError::NoBraiding`] and `twist` panics.
///
/// Enumeration (frozen): the source object order of [`Self::objects`]; the
/// object with one-based source index `i` has enumeration index `i − 1`, and
/// the count is [`FusionCategory::object_count`]. Ising is therefore `(I, ψ, σ)`, unlike
/// TensorKitSectors' `IsingAnyon` order `(I, σ, ψ)`.
///
/// # Examples
///
/// ```
/// use fusion_symbols::{category::{BraidingStyle, FusionCategory}, table::TableCategory};
///
/// let fib = TableCategory::fibonacci();
/// let tau = fib.object(2)?;
/// assert_eq!(fib.fusion(&tau, &tau)?.count(), 2);
/// assert!((fib.dim(&tau) - (1.0 + 5f64.sqrt()) / 2.0).abs() < 1e-15);
///
/// let rep_s3 = TableCategory::by_code([3, 1, 0, 2, 2, 0, 1])?; // fusion-only
/// assert_eq!(rep_s3.braiding_style(), BraidingStyle::None);
/// # Ok::<(), fusion_symbols::table::TableError>(())
/// ```
#[derive(Clone, Debug)]
pub struct TableCategory {
    table_id: u32,
    data: Arc<TableData>,
    entry: &'static CategoryEntry,
}

#[derive(Debug)]
struct TableData {
    fusion_style: FusionStyle,
    braiding_style: BraidingStyle,
    table: MultiplicityFreeTable,
    // Derived once at construction from F and R only, exactly as
    // TensorKitSectors 0.3.9 derives them (docs/references_categories.md,
    // "Derived quantities"). Initialisation, not a cache layer with a policy.
    dims: Vec<f64>,
    frobenius_schur: Vec<Complex64>,
    /// `None` for a fusion-only entry.
    twists: Option<Vec<Complex64>>,
    /// Gauge certificates, read off the stored F exactly (no tolerance).
    trivial_associator: bool,
    canonical_unit: bool,
    fingerprint: Vec<u8>,
}

impl PartialEq for TableCategory {
    fn eq(&self, other: &Self) -> bool {
        self.uuid() == other.uuid()
    }
}
impl Eq for TableCategory {}
impl Hash for TableCategory {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uuid().hash(state);
    }
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
        let (table_id, entry) = catalog::SHIPPED
            .iter()
            .enumerate()
            .find(|(_, entry)| entry.uuid == uuid)
            .expect("generator invariant: every supported INDEX row is in SHIPPED (tested)");
        Self::from_entry(table_id as u32, entry)
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
    /// `37fa4e42-97c6-11f1-9d0f-9b7fe5c504f9`): index `1` = `𝟙`,
    /// index `2` = `τ`.
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
    /// index `1` = `𝟙`, index `2` = `ψ`, index `3` = `σ`
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

    fn from_entry(table_id: u32, entry: &'static CategoryEntry) -> Result<Self, TableError> {
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
        let twists: Option<Vec<Complex64>> = entry.r.map(|_| {
            (0..rank)
                .map(|a| {
                    table
                        .channels(a, a)
                        .map(|b| Complex64::new(dims[b] / dims[a], 0.0) * table.rsymbol(a, a, b))
                        .fold(Complex64::new(0.0, 0.0), |sum, term| sum + term)
                })
                .collect()
        });

        let fusion_style =
            if (0..rank).all(|a| (0..rank).all(|b| table.channels(a, b).count() == 1)) {
                FusionStyle::Unique
            } else {
                FusionStyle::Simple
            };
        // Only derived floating quantities use a tolerance; N and labels remain exact.
        let close = |a: Complex64, b: Complex64| (a - b).norm() <= 1e-12;
        let one = Complex64::new(1.0, 0.0);
        let braiding_style = match &twists {
            None => BraidingStyle::None,
            Some(twists) => {
                let symmetric = (0..rank).all(|a| {
                    (0..rank).all(|b| {
                        table
                            .channels(a, b)
                            .all(|c| close(table.rsymbol(a, b, c) * table.rsymbol(b, a, c), one))
                    })
                });
                if symmetric && twists.iter().all(|&t| close(t, one)) {
                    BraidingStyle::Bosonic
                } else if symmetric && twists.iter().any(|&t| close(t, -one)) {
                    BraidingStyle::Fermionic
                } else {
                    BraidingStyle::Anyonic
                }
            }
        };
        // Exact comparisons: the certificates promise the stored values
        // themselves, and a correctly rounded exact `1` is exactly `1.0`. The
        // unit laws and the unit's self-duality are already `parse` errors.
        let admissible = |[a, b, c, d, e, f]: [usize; 6]| {
            [(a, b, e), (e, c, d), (b, c, f), (a, f, d)]
                .iter()
                .all(|&(x, y, z)| table.nsymbol(x, y, z) > 0)
        };
        let admissible_f = (0..rank.pow(6))
            .map(|flat| {
                let mut labels = [0; 6];
                for (axis, label) in labels.iter_mut().enumerate() {
                    *label = flat / rank.pow(5 - axis as u32) % rank;
                }
                labels
            })
            .filter(|&labels| admissible(labels));
        let is_one = |[a, b, c, d, e, f]: [usize; 6]| table.fsymbol(a, b, c, d, e, f) == one;
        let trivial_associator = admissible_f.clone().all(is_one);
        let canonical_unit = admissible_f
            .filter(|labels| labels[..3].contains(&0))
            .all(is_one);
        Ok(Self {
            entry,
            table_id,
            data: Arc::new(TableData {
                table,
                dims,
                frobenius_schur,
                twists,
                trivial_associator,
                canonical_unit,
                fusion_style,
                braiding_style,
                fingerprint: fingerprint_bytes(&entry.provenance, entry.n, entry.f, entry.r),
            }),
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

    /// Validate a one-based source index.
    ///
    /// # Errors
    /// Returns [`TableError::UnknownObject`] outside `1..=rank`.
    pub fn object(&self, index: u16) -> Result<TableObject, TableError> {
        if (1..=self.data.table.rank()).contains(&usize::from(index)) {
            Ok(TableObject {
                table_id: self.table_id,
                index,
            })
        } else {
            Err(TableError::UnknownObject { object: index })
        }
    }

    /// The category's tagged objects in the frozen source order.
    pub fn objects(&self) -> impl Iterator<Item = TableObject> + '_ {
        (1..=self.data.table.rank() as u16).map(|index| self.object(index).expect("in-range index"))
    }

    fn index(&self, a: &TableObject) -> usize {
        assert_eq!(a.table_id, self.table_id, "foreign table object");
        usize::from(a.index) - 1
    }
    fn at_index(&self, index: usize) -> TableObject {
        TableObject {
            table_id: self.table_id,
            index: index as u16 + 1,
        }
    }
}

/// Allocation-free iterator over fusion channels in the frozen source order.
#[derive(Clone, Debug)]
pub struct TableFusion<'a> {
    category: &'a TableCategory,
    a: usize,
    b: usize,
    remaining: std::ops::Range<usize>,
}

impl Iterator for TableFusion<'_> {
    type Item = TableObject;

    fn next(&mut self) -> Option<TableObject> {
        self.remaining
            .find(|&c| self.category.data.table.nsymbol(self.a, self.b, c) > 0)
            .map(|c| self.category.at_index(c))
    }
}

impl FusionCategory for TableCategory {
    type Object = TableObject;
    type Scalar = Complex64;
    type FSymbol = Complex64;
    type RSymbol = Complex64;
    type Fusion<'a> = TableFusion<'a>;
    type Error = TableError;
    fn fusion_style(&self) -> FusionStyle {
        self.data.fusion_style
    }
    fn braiding_style(&self) -> BraidingStyle {
        self.data.braiding_style
    }
    fn unit(&self) -> TableObject {
        self.at_index(0)
    }
    fn contains(&self, a: &TableObject) -> bool {
        a.table_id == self.table_id && (1..=self.data.table.rank()).contains(&usize::from(a.index))
    }
    fn dual(&self, a: &TableObject) -> TableObject {
        self.at_index(self.data.table.dual(self.index(a)))
    }
    fn dim(&self, a: &TableObject) -> f64 {
        self.data.dims[self.index(a)]
    }
    fn frobenius_schur_phase(&self, a: &TableObject) -> Complex64 {
        self.data.frobenius_schur[self.index(a)]
    }
    fn frobenius_schur_indicator(&self, a: &TableObject) -> Complex64 {
        let phase = self.frobenius_schur_phase(a);
        if *a == self.dual(a) {
            phase
        } else {
            Complex64::new(0.0, 0.0)
        }
    }
    /// Twist derived from R.
    ///
    /// # Panics
    /// Panics on foreign objects first, then if braiding is absent.
    fn twist(&self, a: &TableObject) -> Complex64 {
        let i = self.index(a);
        self.data
            .twists
            .as_ref()
            .expect("twist: the category has no braiding")[i]
    }
    fn fusion<'a>(
        &'a self,
        a: &TableObject,
        b: &TableObject,
    ) -> Result<Self::Fusion<'a>, TableError> {
        let (a, b) = (self.index(a), self.index(b));
        Ok(TableFusion {
            category: self,
            a,
            b,
            remaining: 0..self.data.table.rank(),
        })
    }
    fn nsymbol(&self, a: &TableObject, b: &TableObject, c: &TableObject) -> u32 {
        self.data
            .table
            .nsymbol(self.index(a), self.index(b), self.index(c))
    }
    fn f_symbol(
        &self,
        a: &TableObject,
        b: &TableObject,
        c: &TableObject,
        d: &TableObject,
        e: &TableObject,
        f: &TableObject,
    ) -> Result<Complex64, TableError> {
        Ok(self.data.table.fsymbol(
            self.index(a),
            self.index(b),
            self.index(c),
            self.index(d),
            self.index(e),
            self.index(f),
        ))
    }
    fn r_symbol(
        &self,
        a: &TableObject,
        b: &TableObject,
        c: &TableObject,
    ) -> Result<Complex64, TableError> {
        let (a, b, c) = (self.index(a), self.index(b), self.index(c));
        if self.data.twists.is_none() {
            return Err(TableError::NoBraiding);
        }
        Ok(self.data.table.rsymbol(a, b, c))
    }
    /// Read off the stored F at construction: `true` exactly when every
    /// admissible entry is `1` (the pointed entries with a trivial 3-cocycle).
    fn has_trivial_associator_gauge(&self) -> bool {
        self.data.trivial_associator
    }
    /// Read off the stored F at construction: `true` exactly when every
    /// admissible F with `a`, `b` or `c` the unit is `1` (multiplicity free,
    /// so the identity is `1`); `parse` already enforces the unit laws.
    fn has_canonical_unit(&self) -> bool {
        self.data.canonical_unit
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        Ok(Some(self.data.table.rank() as u64))
    }
    fn object_at(&self, index: u64) -> Result<TableObject, EnumerationError> {
        if index < self.data.table.rank() as u64 {
            Ok(self.at_index(index as usize))
        } else {
            Err(EnumerationError::OutOfRange)
        }
    }
    fn object_index(&self, a: &TableObject) -> Result<u64, EnumerationError> {
        Ok(self.index(a) as u64)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        &self.data.fingerprint
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

#[cfg(test)]
mod tests;
