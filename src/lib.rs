//! Fusion and recoupling data for symmetric tensors: the compact Lie groups
//! SU(2), SU(N), SO(N), Spin(N) and Sp(2N) (irreducible representations,
//! Clebsch–Gordan coefficients, 3j / 6j / F / R), the abelian groups and
//! CU(1), fermion parity, and tabulated finite fusion categories.
//!
//! `fusion-symbols` is the successor of the `racah` crate (published up to
//! 0.2.4). The Rust API uses category values under `fusion_symbols::`.
//!
//! Lie-group coefficients for any admissible labels are computed on demand —
//! there is no precomputed table and no generation-time label cut. Pure
//! representation and fusion-category mathematics: no tensor-network concepts
//! (legs, fusion trees, tensor maps, contraction). Consumers build those on top
//! of these numbers.
//!
//! # Quick start
//!
//! Exact SU(2) recoupling needs no features. Spins are doubled (`dj = 2j`), so
//! `2` means spin 1; a non-admissible label set returns exact zero, never an
//! error. Here `{1 1 1; 1 1 1} = 1/6`:
//!
//! ```
//! use fusion_symbols::lie::su2::wigner_6j;
//!
//! let sixj = wigner_6j(2, 2, 2, 2, 2, 2);
//! assert!((sixj.to_f64() - 1.0 / 6.0).abs() < 1e-14);
//! ```
//!
//! Lie categories use validated Dynkin labels. A1 blocks are stored inline;
//! A(r), r >= 2, uses cached generated blocks with `cgc-gen`:
//!
//! ```
//! use fusion_symbols::{category::FusionCategory, lie::{su2::wigner_6j, GroupId, LieGroup}};
//! assert!((wigner_6j(2, 2, 2, 2, 2, 2).to_f64() - 1.0 / 6.0).abs() < 1e-14);
//! let group = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
//! let half = group.irrep(&[1]).unwrap();
//! let unit = group.unit();
//! let f = group.f_symbol(&half, &half, &half, &half, &unit, &unit).unwrap();
//! assert_eq!(f.at(0, 0, 0, 0), -0.5);
//! let r = group.r_symbol(&half, &half, &unit).unwrap();
//! assert_eq!(r.at(0, 0), -1.0);
//! ```
//!
//! # Where to look
//!
//! | You want | Go to |
//! |---|---|
//! | Shared category contract, styles, scalar storage and identity residuals | [`category`] |
//! | SU(2): exact 3j, 6j, CG evaluators; category F/R and Frobenius–Schur | [`lie::su2`] and [`lie::LieGroup`] |
//! | SU(N), SU(N)/Z_k, PSU(N) — Gelfand–Tsetlin construction | [`lie::LieGroup`] (`cgc-gen`) |
//! | SO(N), Spin(N), Sp(2r) — generator bootstrap, B/C/D series | [`lie::LieGroup`] (`cgc-gen`) |
//! | Which highest weights a global form admits | [`lie::GroupId`] |
//! | U(1), Z_N (including Z2) in closed form | [`abelian::U1`], [`abelian::ZN`] |
//! | Other compact groups: CU(1) = U(1) ⋊ C = O(2), in closed form | [`group`] |
//! | Fermion parity in closed form | [`fermion::FermionParity`] |
//! | Finite multiplicity-free unitary categories (Fibonacci, Ising, …, rank ≤ 4; rank ≤ 7 with `anyonwiki-unitary`) from pinned AnyonWiki tables | [`table`] |
//! | Deligne products | Planned; no public `product` module yet |
//! | Cache ceilings, budgets, statistics | [`cache`] |
//!
//! SU(N) uses the same [`lie::LieGroup`] and [`lie::LieIrrep`] layer as A1.
//! With `cgc-gen`, higher A-series fusion shares cached product lists, and
//! [`category::GroupRepresentations::cgc`] returns [`lie::LieCgc`]. Empty
//! vertices give zero blocks; genuine generation failures are opaque
//! [`lie::LieError::Generation`] diagnostics with an error source.
//!
//! # Non-Lie families
//!
//! These families use closed forms or pinned, provenance-tracked tables
//! rather than runtime coefficient generation, and need no feature flag for
//! the default catalog. [`abelian`], [`group`],
//! [`fermion`] and [`table`] expose their
//! validated objects through [`category::FusionCategory`]: N = 0 is a value,
//! F/R return zero for inadmissible vertices, and errors mean genuine failure
//! or an absent capability. Foreign objects panic. Logical symbol shape comes
//! from N; scalar storage has all-one axes even for a zero symbol.
//! Table objects are validated by [`table::TableCategory::object`]; their private
//! catalog tag rejects foreign objects. Table categories share data when cloned
//! and compare by uuid. Fusion-only tables return a braiding error from R and
//! panic on twist. CU(1) fusion reports label overflow.
//! Index order follows TensorKitSectors.jl (`F^{abc}_d[e, f]` with
//! `e ∈ a ⊗ b`, `f ∈ b ⊗ c`; `R^{ab}_c`). Symbols in [`abelian`],
//! [`group`] and [`fermion`] are `f64`; [`table`] uses
//! [`num_complex::Complex64`]. Closed forms use TensorKitSectors' gauge;
//! [`table`] uses AnyonWiki's (gauge-equivalent, and for Fibonacci identical).
//! Provenance: [`docs/references_categories.md`].
//!
//! # Layers
//!
//! - **base** (no feature): exact SU(2) — closed-form 3j/6j/CGC in
//!   big-rational arithmetic with a single final rounding to floating point;
//!   the shared [`category`] contract, closed-form [`abelian`], [`group`]
//!   (CU(1)), [`fermion`], rank ≤ 4 [`table`] categories, and [`cache`].
//! - **`anyonwiki-unitary`**: rank 5–7 [`table`] categories (embedded data only).
//! - **`cgc-gen`**: runtime coefficient generation for SU(N) (Gelfand–Tsetlin
//!   construction), and SO(N)/Sp(2N) (defining-representation seeds plus a
//!   family-generic decomposition loop). Dense factorizations and the CGC
//!   contractions producing F/R route through t4a-faer with explicit `Par::Seq`
//!   and scratch workspaces; no hand-rolled kernels,
//!   and backend selection is not public API.
//!
//! The boundary is mathematical, not organizational: SU(2) has closed forms and
//! needs no matrix computation, so a consumer needing only SU(2) never pulls a
//! linear-algebra stack.
//!
//! # Exactness contract
//!
//! Combinatorial structure and discrete data are exact; gauge fixing is a
//! deterministic function of the subspace; floating-point stages are
//! verification-gated and versioned. Concretely: labels, dimensions, duals,
//! Frobenius–Schur indicators and fusion multiplicities are exact integer or
//! rational arithmetic. Generated CGC / F / R *values* are `f64`; numeric
//! generation gates reject invalid isometries with typed errors. Generic
//! category residual functions verify F unitarity, pentagon and hexagon
//! identities, including outer multiplicities.
//!
//! # Provider contract
//!
//! Each family publishes an opaque **authority fingerprint** naming the
//! convention set its coefficients are computed in
//! ([`category::FusionCategory::authority_fingerprint`]). Persist the bytes next to anything you
//! derive from these coefficients and compare by equality on load; never parse
//! them. The base SU(2) provider adds a checked representation surface
//! ([`lie::LieIrrep`] and [`lie::LieGroup`]) and a cache
//! resource contract ([`cache::base_cache_stats`],
//! [`cache::BASE_CACHE_MAX_BYTES`], [`cache::reset`]). The
//! [User Guide][guide-resources] carries the prose.
//!
//! # Documentation
//!
//! - [User Guide] — task-oriented: pick a group, build irreps, fuse, get CGC,
//!   get F/R, bound the caches. Start here if you are new.
//! - [`docs/theory.pdf`] — a self-contained note on the objects this API
//!   computes (irreps, fusion multiplicities, CGC and gauge, recoupling, the
//!   two constructions, the exactness contract).
//! - [`docs/gauge.md`] / [`docs/gauge_soN.md`] — the frozen normative gauge
//!   specifications. Reference documents, not tutorials.
//! - [`docs/references.md`] — porting provenance (`file:symbol`-level) and the
//!   verified bibliography.
//!
//! [User Guide]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/user-guide/README.md
//! [guide-resources]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/user-guide/resources.md
//! [`docs/theory.pdf`]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/theory.pdf
//! [`docs/references.md`]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/references.md
//! [`docs/references_categories.md`]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/references_categories.md
//! [`docs/gauge.md`]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/gauge.md
//! [`docs/gauge_soN.md`]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/gauge_soN.md
#![warn(missing_docs)]

// The audit harness measures only allocations routed through Rust's System
// allocator. Keeping it test-only avoids changing the library allocator
// contract or attributing C/backend allocations to Racah.
#[cfg(all(test, feature = "cgc-gen"))]
#[global_allocator]
static TEST_ALLOCATOR: audit_alloc::TrackingAllocator = audit_alloc::TrackingAllocator;

#[cfg(all(test, feature = "cgc-gen"))]
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

/// Lie groups, validated irreps, and exact SU(2) evaluators.
pub mod lie;

mod exact;
mod primefactor;

#[cfg(all(test, feature = "cgc-gen"))]
mod cache_audit;
#[cfg(all(test, feature = "cgc-gen"))]
mod cache_budget_pressure;
#[cfg(all(test, feature = "cgc-gen"))]
mod cache_trim_pressure;

// Family-neutral contractions for generated Lie coefficients.
#[cfg(feature = "cgc-gen")]
mod frcore;

// Module docs are the files' inner `//!` docs: an outer `///` here would make
// rustdoc resolve their intra-doc links in the crate-root scope.
pub mod abelian;
pub mod category;
pub mod fermion;
pub mod group;
pub mod table;
