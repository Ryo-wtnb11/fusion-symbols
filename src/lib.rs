//! Fusion-category and representation data for symmetric tensors: labels,
//! fusion multiplicities, dimensions, duals, Frobenius–Schur data, F/R/A/B
//! symbols and Clebsch–Gordan coefficients (CGC).
//!
//! Families: compact connected Lie groups through [`lie::LieGroup`] — SU(N),
//! SO(N), Spin(N), Sp(2r), G₂, U(N), tori, their products and central quotients —
//! plus exact SU(2) evaluators in [`lie::su2`]; closed-form [`lie::U1`] and
//! CU(1) = O(2) ([`lie::CU1`]); cyclic groups [`finite::ZN`]; fermion parity
//! [`fermion::FermionParity`]; tabulated unitary multiplicity-free categories
//! ([`table`]); and ordered Deligne products of all of these ([`mod@product`]).
//!
//! Not provided: the exceptional groups F₄, E₆, E₇ and E₈ (classified by
//! [`lie::GroupId`], but [`lie::LieGroup::new`] returns
//! [`lie::LieError::Unsupported`]), disconnected
//! groups other than CU(1), and finite non-abelian Rep(G).
//!
//! Lie-group coefficients are computed on demand for any admissible labels;
//! there is no precomputed table or label cutoff. This crate holds only the
//! mathematics: tensors, fusion trees and contraction belong to the consumer.
//!
//! # Quick start
//!
//! Build a group from a [`lie::GroupId`], create irreps from Dynkin labels and
//! torus charges, and query symbols through [`category::FusionCategory`]:
//!
//! ```
//! use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup}};
//!
//! let su2 = LieGroup::new(GroupId::su(2)?)?;
//! let half = su2.irrep(&[1], &[])?; // doubled spin: 1 means spin 1/2
//! let unit = su2.unit();
//! assert_eq!(su2.nsymbol(&half, &half, &unit), 1);
//! let f = su2.f_symbol(&half, &half, &half, &half, &unit, &unit)?;
//! assert_eq!(f.at(0, 0, 0, 0), -0.5);
//! let r = su2.r_symbol(&half, &half, &unit)?;
//! assert_eq!(r.at(0, 0), -1.0);
//!
//! // U(2): an SU(2) label followed by a U(1) charge of the same parity.
//! let u2 = LieGroup::new(GroupId::u(2)?)?;
//! let fundamental = u2.irrep(&[1], &[1])?;
//! assert_eq!(u2.dim(&fundamental), 2.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Exact SU(2) recoupling values, with doubled spins (`2` means spin 1):
//!
//! ```
//! use fusion_symbols::lie::su2::wigner_6j;
//!
//! let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
//! assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
//! ```
//!
//! Products combine categories factorwise:
//!
//! ```
//! use fusion_symbols::{finite::ZN, category::FusionCategory,
//!     fermion::{FermionParity, FermionParityIrrep}, product::Product};
//! let cat = Product::new((ZN::new(2).unwrap(), FermionParity));
//! let odd = (cat.factors().0.unit(), FermionParityIrrep::ODD);
//! assert_eq!(cat.twist(&odd), -1.0);
//! ```
//!
//! # Where to look
//!
//! | You want | Go to |
//! |---|---|
//! | The shared category contract, symbol blocks, CGC and identity residuals | [`category`] |
//! | Classifying a group and its global form | [`lie::GroupId`] |
//! | Any supported Lie group as a category | [`lie::LieGroup`], objects [`lie::LieIrrep`] |
//! | Exact SU(2) 3j, 6j and CG values | [`lie::su2`] |
//! | U(1) with `Copy` labels; CU(1) = O(2) | [`lie::U1`], [`lie::CU1`] |
//! | Z_N; fermion parity | [`finite::ZN`]; [`fermion::FermionParity`] |
//! | Fibonacci, Ising and other AnyonWiki categories | [`table::TableCategory`] |
//! | Deligne products, tuples or runtime-length vectors | [`mod@product`] |
//! | Cache budgets, statistics, trimming | [`cache`] |
//!
//! # Features
//!
//! - **base** (`default-features = false`): everything except the items below,
//!   including every rank-one and D₂ Lie factor, tori and U(2), exact SU(2)
//!   values, all closed-form families, rank ≤ 4 [`table`] categories (74),
//!   products and caches. Dependencies are the `num-*` arithmetic crates and
//!   `quick_cache`; there is no dense linear algebra.
//! - **`lie-generated`** (default): runtime coefficient generation for A, B, C
//!   factors of rank ≥ 2 and D factors of rank ≥ 3, that is SU(N ≥ 3),
//!   SO(N ≥ 5), Spin(N ≥ 5), Sp(2r ≥ 4), U(N ≥ 3) and products containing them.
//!   Dense kernels run sequentially through `t4a-faer`; the caller owns
//!   parallelism.
//! - **`anyonwiki-unitary`**: the 303 rank 5–7 [`table`] categories (embedded
//!   data only).
//!
//! # Exactness and conventions
//!
//! Labels, dimensions, duals, Frobenius–Schur indicators and fusion
//! multiplicities are exact. SU(2) values are exact rationals under a square
//! root, rounded once to `f64`. Generated CGC/F/R values are `f64`, checked
//! by numerical gates that return typed errors on failure. Closed-form
//! families use `f64`; [`table`] uses [`num_complex::Complex64`]. Index order
//! follows TensorKitSectors.jl (`F^{abc}_d[e, f]` with `e ∈ a ⊗ b`,
//! `f ∈ b ⊗ c`; `R^{ab}_c`). Closed forms use TensorKitSectors' gauge;
//! [`table`] keeps AnyonWiki's; generated Lie coefficients follow the frozen
//! gauge specifications. Each category publishes an opaque
//! [`category::FusionCategory::authority_fingerprint`]: persist it next to
//! derived data and compare bytes on load.
//!
//! # Documentation
//!
//! - [User guide]: task-oriented — choose a category, build groups and labels,
//!   query symbols and CGC, use products and tables, handle errors and caches.
//! - [`docs/overview.md`]: capability boundaries and contracts.
//! - [Caches and reproducibility][guide-resources].
//! - [`docs/gauge.md`] / [`docs/gauge_soN.md`]: the frozen gauge specifications.
//! - [`docs/references.md`] and [`docs/references_categories.md`]: algorithm
//!   and data provenance.
//!
//! [User guide]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/user-guide/README.md
//! [guide-resources]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/user-guide/resources.md
//! [`docs/overview.md`]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/overview.md
//! [`docs/references.md`]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/references.md
//! [`docs/references_categories.md`]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/references_categories.md
//! [`docs/gauge.md`]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/gauge.md
//! [`docs/gauge_soN.md`]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/gauge_soN.md
#![warn(missing_docs)]

// The audit harness measures only allocations routed through Rust's System
// allocator. Keeping it test-only avoids changing the library allocator
// contract or attributing C/backend allocations to Racah.
#[cfg(all(test, feature = "lie-generated"))]
#[global_allocator]
static TEST_ALLOCATOR: audit_alloc::TrackingAllocator = audit_alloc::TrackingAllocator;

#[cfg(all(test, feature = "lie-generated"))]
mod audit_alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
    static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

    pub(crate) struct TrackingAllocator;

    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
                let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
                PEAK.fetch_max(live, Ordering::Relaxed);
            }
            ptr
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc_zeroed(layout) };
            if !ptr.is_null() {
                ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
                let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
                PEAK.fetch_max(live, Ordering::Relaxed);
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) };
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            let new = unsafe { System.realloc(ptr, layout, size) };
            if !new.is_null() {
                if size >= layout.size() {
                    let delta = size - layout.size();
                    ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
                    ALLOC_BYTES.fetch_add(delta, Ordering::Relaxed);
                    let live = LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
                    PEAK.fetch_max(live, Ordering::Relaxed);
                } else {
                    LIVE.fetch_sub(layout.size() - size, Ordering::Relaxed);
                }
            }
            new
        }
    }

    pub(crate) fn snapshot() -> (usize, usize) {
        (LIVE.load(Ordering::Relaxed), PEAK.load(Ordering::Relaxed))
    }

    pub(crate) fn reset_peak_to_live() {
        PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    /// Cumulative successful allocation requests. `realloc` contributes only
    /// its growth delta; deallocations never subtract from these counters.
    pub(crate) fn allocation_totals() -> (usize, usize) {
        (
            ALLOC_CALLS.load(Ordering::Relaxed),
            ALLOC_BYTES.load(Ordering::Relaxed),
        )
    }
}

pub mod cache;

pub mod lie;

mod exact;
mod primefactor;

#[cfg(all(test, feature = "lie-generated"))]
mod cache_audit;
#[cfg(all(test, feature = "lie-generated"))]
mod cache_budget_pressure;
#[cfg(all(test, feature = "lie-generated"))]
mod cache_trim_pressure;

// Family-neutral contractions for generated Lie coefficients.
#[cfg(feature = "lie-generated")]
mod frcore;

// Module docs are the files' inner `//!` docs: an outer `///` here would make
// rustdoc resolve their intra-doc links in the crate-root scope.
pub mod category;
pub mod fermion;
pub mod finite;
pub mod table;

pub mod product;
