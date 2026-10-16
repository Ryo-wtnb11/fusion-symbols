//! Fusion categories from pinned, auditable tables instead of formulas.
//!
//! The classification of small fusion categories is a body of computed data,
//! not a body of formulas. The authority is the AnyonWiki database
//! (`anyonwiki/AnyonWikiDatabase`, MIT), which stores every symbol exactly, as
//! an algebraic number given by its minimal polynomial and root index:
//!
//! ```text
//! exact algebraic symbol  ->  correctly rounded binary64 (offline generator)
//!                         ->  shortest round-trip decimal table  ->  Complex64
//! ```
//!
//! The rounding is done once, offline, by `tools/anyonwiki/gen_category_tables.py`;
//! the crate parses the committed output and adds no runtime dependency.
//! `JCBridgeman/smallRankUnitaryFusionData` and its CategoryData.jl
//! repackaging hold the same category in the same gauge and serve only as
//! independent cross-checks.
//!
//! - Not a solver. Nothing here solves the pentagon or hexagon equations or
//!   regenerates `F`/`R` from `N`. Those equations appear only as
//!   *consistency* tests.
//! - Not online. Nothing invokes Julia, Mathematica, or the network.
//!
//! Every upstream pin, hash, gauge statement and the source-to-`Complex64`
//! conversion epoch is tabulated in the crate's `docs/references_categories.md`; the
//! machine-readable form is [`TableProvenance`].
//!
//! # Relationship to the closed-form Fibonacci
//!
//! [`crate::fibonacci`] evaluates the same category from formulas. The two
//! agree bitwise on R, but not on F, dimensions and twist (within a few
//! ulp), so they have different authority
//! fingerprints by design: sharing one would let data computed under one set
//! of coefficients be reused under the other.

use num_complex::Complex64;

mod provenance;
mod table;

pub use provenance::{TableProvenance, PROJECTION_EPOCH};
pub use table::{TableError, TableObject};

use table::MultiplicityFreeTable;

const RANK: usize = 2;

// The three pinned files, embedded verbatim under their upstream paths. They
// are the identity-bearing bytes, so they are `include_str!`-ed rather than
// re-encoded: any change to them, deliberate or accidental, changes the
// authority fingerprint.
const N_TEXT: &str = include_str!("../data/anyonwiki-badbfe2/fibonacci/Nabc.txt");
const F_TEXT: &str = include_str!("../data/anyonwiki-badbfe2/fibonacci/F.txt");
const R_TEXT: &str = include_str!("../data/anyonwiki-badbfe2/fibonacci/R.txt");

/// Fibonacci from the AnyonWiki database (`anyonwiki_code [2,1,0,2,1,2,1]`:
/// ring `2_1_0_2`, associator 1, braiding 2, pivotal 1), in the source gauge.
/// Its braiding is TensorKit's: `R^{ττ}_𝟙 = e^{+4πi/5}`.
///
/// Objects are the one-based [`TableObject`] labels of the source files:
/// `TableObject(1)` is the unit `𝟙`, `TableObject(2)` is `τ`. Index
/// order follows TensorKitSectors: `F^{abc}_d[e, f]`, `R^{ab}_c`.
///
/// The infallible queries take labels by precondition and panic on a label
/// outside `1..=rank` (a flat table cannot tell an out-of-range label from
/// another entry); an in-range inadmissible symbol is exact zero. The
/// `_checked` variants return [`TableError`] instead.
///
/// # Examples
///
/// ```
/// use fusion_symbols::tables::{FibonacciTable, TableObject};
///
/// let fib = FibonacciTable::try_new()?;
/// let tau = TableObject(2);
/// assert_eq!(fib.fusion(tau, tau).count(), 2);
/// assert!((fib.dim(tau) - (1.0 + 5f64.sqrt()) / 2.0).abs() < 1e-15);
/// # Ok::<(), fusion_symbols::tables::TableError>(())
/// ```
#[derive(Debug)]
pub struct FibonacciTable {
    table: MultiplicityFreeTable,
    // Derived once at construction from F and R only, exactly as
    // TensorKitSectors 0.3.9 derives them (docs/references_categories.md, "Derived
    // quantities"). Initialisation, not a cache layer with a policy.
    dims: Vec<f64>,
    frobenius_schur: Vec<Complex64>,
    twists: Vec<Complex64>,
    fingerprint: Vec<u8>,
}

impl FibonacciTable {
    /// The frozen lineage of this provider's tables.
    pub const PROVENANCE: TableProvenance = TableProvenance {
        source_repository: "https://github.com/anyonwiki/AnyonWikiDatabase",
        source_commit: "badbfe299dd533c12a84e817f5890059705e0a3c",
        source_license: "MIT, Copyright (c) 2026 AnyonWiki",
        source_files: "AlgebraicStructures/FusionCategories/MultFreeFusionCategories/Rank1-7.json sha256=383cb2db58ae839f73c3d67ad79186e6ee44d69c32fd8fb202d1a4089ca7e859; AlgebraicStructures/FusionRings/mult1.json sha256=66117d72d204a5359ef43e584ed46cfe503240558aa8b9c381922b4861aa1153",
        source_entry: "category 37fa4e42-97c6-11f1-9d0f-9b7fe5c504f9 (f_symbols, r_symbols); fusion ring d810a768-812e-11f1-8f24-516eff0ae265 (mult_tab)",
        projection: "tools/anyonwiki/gen_category_tables.py: exact qqb_id evaluated at 80 digits (confirmed at 160), correctly rounded to binary64, written as shortest round-trip decimals",
        category_key: "anyonwiki_code [2,1,0,2,1,2,1] (Fibonacci)",
        object_order: "fusion-ring element order, one-based: 1 = vacuum, 2 = tau",
        gauge: "source gauge verbatim; only the one-based to zero-based label shift is applied",
        projection_epoch: PROJECTION_EPOCH,
    };

    /// Parses, validates and publishes the provider.
    ///
    /// There is no unchecked constructor: a provider that skipped the import
    /// checks would be indistinguishable from a correct one at every later
    /// call site.
    ///
    /// # Errors
    ///
    /// Returns an import [`TableError`] if the embedded tables do not
    /// parse or violate an exact `N` axiom. For the shipped data this cannot
    /// happen; the constructor is fallible because it is the import boundary,
    /// and the crate's tests are what hold that invariant.
    pub fn try_new() -> Result<Self, TableError> {
        let table = MultiplicityFreeTable::parse(RANK, N_TEXT, F_TEXT, R_TEXT)?;

        let unit_f = |a: usize| table.fsymbol(a, table.dual(a), a, a, 0, 0);
        // TensorKitSectors `dim_from_Fsymbol`: abs(1 / F[a, ā, a, a, 1, 1]).
        let dims: Vec<f64> = (0..RANK)
            .map(|a| (Complex64::new(1.0, 0.0) / unit_f(a)).norm())
            .collect();
        // `frobenius_schur_phase_from_Fsymbol`: sign(F[a, ā, a, a, 1, 1]).
        let frobenius_schur: Vec<Complex64> = (0..RANK)
            .map(|a| {
                let value = unit_f(a);
                value / value.norm()
            })
            .collect();
        // `twist_from_Rsymbol`: Σ_{b ∈ a⊗a} d_b/d_a R^{aa}_b, ascending b.
        let twists: Vec<Complex64> = (0..RANK)
            .map(|a| {
                table
                    .channels(a, a)
                    .map(|b| Complex64::new(dims[b] / dims[a], 0.0) * table.rsymbol(a, a, b))
                    .fold(Complex64::new(0.0, 0.0), |sum, term| sum + term)
            })
            .collect();

        Ok(Self {
            table,
            dims,
            frobenius_schur,
            twists,
            fingerprint: fingerprint_bytes(N_TEXT, F_TEXT, R_TEXT),
        })
    }

    /// The category's objects in the frozen source order.
    pub fn objects(&self) -> impl Iterator<Item = TableObject> {
        (1..=self.table.rank() as u16).map(TableObject)
    }

    /// The tensor unit, `TableObject(1)`.
    pub fn unit(&self) -> TableObject {
        TableObject(1)
    }

    /// Opaque authority fingerprint: the provenance record plus the three
    /// pinned source files, byte for byte.
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
        self.dims[index_or_panic("dim", a)]
    }

    /// The dual object: the unique `b` with `N^{ab}_𝟙 = 1`.
    ///
    /// # Panics
    ///
    /// If `a` is not an object of the category.
    pub fn dual(&self, a: TableObject) -> TableObject {
        object(self.table.dual(index_or_panic("dual", a)))
    }

    /// Frobenius–Schur phase `κ_a = F^{a ā a}_a[𝟙, 𝟙] / |F^{a ā a}_a[𝟙, 𝟙]|`.
    ///
    /// # Panics
    ///
    /// If `a` is not an object of the category.
    pub fn frobenius_schur(&self, a: TableObject) -> Complex64 {
        self.frobenius_schur[index_or_panic("frobenius_schur", a)]
    }

    /// Twist `θ_a = Σ_{b ∈ a⊗a} d_b/d_a · R^{aa}_b`.
    ///
    /// # Panics
    ///
    /// If `a` is not an object of the category.
    pub fn twist(&self, a: TableObject) -> Complex64 {
        self.twists[index_or_panic("twist", a)]
    }

    /// Fusion channels of `a ⊗ b` in ascending label order (TensorKitSectors
    /// filters `values(I)` in that order).
    ///
    /// # Panics
    ///
    /// If `a` or `b` is not an object of the category.
    pub fn fusion(&self, a: TableObject, b: TableObject) -> impl Iterator<Item = TableObject> + '_ {
        self.table
            .channels(index_or_panic("fusion", a), index_or_panic("fusion", b))
            .map(object)
    }

    /// Fusion multiplicity `N^{ab}_c`.
    ///
    /// # Panics
    ///
    /// If a label is not an object of the category.
    pub fn nsymbol(&self, a: TableObject, b: TableObject, c: TableObject) -> u32 {
        let i = |x| index_or_panic("nsymbol", x);
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
        let i = |x| index_or_panic("f_symbol", x);
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
            index(x)?;
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
    /// If a label is not an object of the category.
    pub fn r_symbol(&self, a: TableObject, b: TableObject, c: TableObject) -> Complex64 {
        let i = |x| index_or_panic("r_symbol", x);
        self.table.rsymbol(i(a), i(b), i(c))
    }

    /// Checked [`r_symbol`](Self::r_symbol): an error for an unknown label or
    /// an inadmissible vertex instead of a panic or a zero.
    pub fn r_symbol_checked(
        &self,
        a: TableObject,
        b: TableObject,
        c: TableObject,
    ) -> Result<Complex64, TableError> {
        for x in [a, b, c] {
            index(x)?;
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
}

/// The exact byte composition of the authority fingerprint.
fn fingerprint_bytes(n_text: &str, f_text: &str, r_text: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    FibonacciTable::PROVENANCE.write_canonical_bytes(&mut bytes);
    // The source text, not a re-serialisation of the parsed f64s, which would
    // hide a parser change behind identical output.
    for (tag, text) in [
        ("Nsymbols", n_text),
        ("Fsymbols", f_text),
        ("Rsymbols", r_text),
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

fn index(a: TableObject) -> Result<usize, TableError> {
    let label = usize::from(a.label());
    if (1..=RANK).contains(&label) {
        Ok(label - 1)
    } else {
        Err(TableError::UnknownObject { object: a })
    }
}

fn index_or_panic(method: &str, a: TableObject) -> usize {
    index(a).unwrap_or_else(|error| panic!("{method} precondition violated: {error}"))
}

#[cfg(test)]
mod tests {
    use super::provenance::fnv1a64;
    use super::*;

    #[test]
    fn embedded_tables_load_and_expose_the_frozen_ordering() {
        let fib = FibonacciTable::try_new().expect("embedded tables are valid");
        assert_eq!(
            fib.objects().collect::<Vec<_>>(),
            vec![TableObject(1), TableObject(2)]
        );
        assert_eq!(fib.unit(), TableObject(1));
    }

    #[test]
    fn fingerprint_is_reproducible_and_epoch_bound() {
        let first = FibonacciTable::try_new().expect("valid");
        let second = FibonacciTable::try_new().expect("valid");
        assert_eq!(
            first.authority_fingerprint(),
            second.authority_fingerprint()
        );

        let mut baseline = Vec::new();
        FibonacciTable::PROVENANCE.write_canonical_bytes(&mut baseline);
        let mut bumped = Vec::new();
        TableProvenance {
            projection_epoch: "other",
            ..FibonacciTable::PROVENANCE
        }
        .write_canonical_bytes(&mut bumped);
        assert_ne!(baseline, bumped);
    }

    // Golden FNV-1a digests of the three shipped files and of the composed
    // fingerprint. This is the only test that can fail on a *tamper*: every
    // other test reads the shipped bytes and would agree with an edited table
    // whose values still round to the same `f64`. Updating a constant here
    // means the pinned source changed, and `docs/references_categories.md` must change with
    // it.
    //
    // Golden digests of the three generated files. `Nabc.txt` is byte-equal
    // to the earlier source; F and R are not symtn's bytes: the decimals are now the shortest round-trip form of correctly rounded
    // binary64, a different spelling of bit-identical values
    // (`tests/tenet_values.rs`). `FINGERPRINT_PREHASH` differs from symtn's
    // `IDENTITY_PREHASH` (0xee32_d986_edef_625b) because source and epoch are
    // new.
    const N_TEXT_FNV: u64 = 0xaba7_7aa3_3176_9df0;
    const F_TEXT_FNV: u64 = 0x9ade_e873_8a99_da25;
    const R_TEXT_FNV: u64 = 0x75d1_cc53_3d67_65a1;
    const FINGERPRINT_PREHASH: u64 = 0x52e5_b6dc_1a15_96c5;

    #[test]
    fn shipped_table_bytes_are_pinned() {
        assert_eq!(fnv1a64(N_TEXT.as_bytes()), N_TEXT_FNV, "Nsymbols changed");
        assert_eq!(fnv1a64(F_TEXT.as_bytes()), F_TEXT_FNV, "Fsymbols changed");
        assert_eq!(fnv1a64(R_TEXT.as_bytes()), R_TEXT_FNV, "Rsymbols changed");
        let provider = FibonacciTable::try_new().expect("valid");
        assert_eq!(
            fnv1a64(provider.authority_fingerprint()),
            FINGERPRINT_PREHASH
        );
    }

    #[test]
    fn shipped_braiding_is_tensorkits_not_its_mirror() {
        // The other braiding of the same fusion category (AnyonWiki b = 1)
        // has the conjugate R-symbols; TensorKitSectors' `FibonacciAnyon`
        // has R^{ττ}_𝟙 = e^{+4πi/5}.
        let fib = FibonacciTable::try_new().expect("valid");
        let tau = TableObject(2);
        let r = fib.r_symbol(tau, tau, fib.unit());
        assert!((r.arg() - 0.8 * std::f64::consts::PI).abs() < 1e-15, "{r}");
    }

    #[test]
    fn fingerprint_moves_on_a_source_edit_too_small_to_move_a_coefficient() {
        // Appends one trailing zero digit to every decimal literal: value
        // preserving and below binary64 resolution.
        let edited: String = F_TEXT
            .lines()
            .map(|line| {
                let fields: Vec<String> = line
                    .split_whitespace()
                    .map(|field| {
                        if field.contains('.') {
                            format!("{field}0")
                        } else {
                            field.to_owned()
                        }
                    })
                    .collect();
                format!("{}\n", fields.join(" "))
            })
            .collect();
        assert_ne!(edited, F_TEXT, "the edit did not apply");
        let fields = |text: &str| {
            text.split_whitespace()
                .map(|field| field.parse::<f64>().expect("numeric field").to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            fields(&edited),
            fields(F_TEXT),
            "the edit moved a coefficient"
        );

        let provider = FibonacciTable::try_new().expect("valid");
        assert_eq!(
            fingerprint_bytes(N_TEXT, F_TEXT, R_TEXT),
            provider.authority_fingerprint()
        );
        assert_ne!(
            fingerprint_bytes(N_TEXT, &edited, R_TEXT),
            provider.authority_fingerprint()
        );
    }
}
