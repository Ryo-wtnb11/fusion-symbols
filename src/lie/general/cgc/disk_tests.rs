//! The disk hooks at the real generation boundaries. Each test runs alone in
//! a fresh process with the disk cache enabled (`cache::disk::tests::isolated`),
//! so no other test shares its coefficient tiers or its store.

use super::*;
use crate::cache::disk::tests::{isolated, pending_rows, AFTER_READ, CHUNK, FAULT_WRITE};
use crate::cache::{
    disk_cache_stats, reset, Label, DISK_CHANNEL, DISK_F, DISK_NORMS, DISK_PHASE, DISK_SEEDS,
};
use crate::category::{FBlock, RBlock, SymbolBlock};
use crate::lie::general::{GeneralFamily, RequestModules};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const OCTET: [i64; 2] = [1, 1];

fn su3() -> GeneralFamily {
    GeneralFamily::new(RootSystem::A(2))
}

fn octet_channel(fam: &GeneralFamily) -> Arc<CgcChannel> {
    let o = Label::new(&OCTET);
    fam.cgc_entries(o, o, o, None).unwrap()
}

fn octet_f(fam: &GeneralFamily) -> Result<FBlock<f64>, LieError> {
    let o = Label::new(&OCTET);
    fam.f_symbol([o; 6], None)
}

fn octet_r(fam: &GeneralFamily) -> RBlock<f64> {
    let o = Label::new(&OCTET);
    fam.r_symbol([o; 3], None).unwrap()
}

fn bits(c: &CgcChannel) -> Vec<(u32, u32, u32, u32, u64)> {
    c.entries
        .iter()
        .map(|e| (e.m1, e.m2, e.m3, e.mu, e.value.to_bits()))
        .collect()
}

fn block_bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn reset_builds() {
    BUILDS.with(|b| b.set([0; 4]));
}

fn panic_if_contracted(_: f64) -> f64 {
    panic!("a disk hit reached the contraction")
}

/// A disk-warm view miss loads the exact channel and the octet's norm table
/// (one label, one table) and rounds the view from them: it builds no
/// module, runs no generation and gives the cold bits. Only exact rows are
/// stored: one channel row and one norm row.
#[test]
fn valid_hit_rebuilds_nothing() {
    isolated(module_path!(), "valid_hit_rebuilds_nothing", |db| {
        let fam = su3();
        let cold = octet_channel(&fam);
        assert_eq!(
            (db.rows_of_kind(DISK_CHANNEL), db.rows_of_kind(DISK_NORMS)),
            (1, 1)
        );
        reset();
        reset_builds();
        let hits = disk_cache_stats().hits;
        let warm = octet_channel(&fam);
        assert_eq!(BUILDS.with(std::cell::Cell::get), [0; 4]);
        assert_eq!(disk_cache_stats().hits, hits + 2);
        assert_eq!(bits(&warm), bits(&cold));
    });
}

/// A channel row whose norm rows were removed (the store trims rows
/// independently) still loads: the missing norm table is rebuilt from its
/// module, and nothing else is generated.
#[test]
fn channel_hit_without_norm_rows_rebuilds_only_the_norms() {
    isolated(
        module_path!(),
        "channel_hit_without_norm_rows_rebuilds_only_the_norms",
        |db| {
            let fam = su3();
            let cold = octet_channel(&fam);
            db.delete_kind(DISK_NORMS);
            reset();
            reset_builds();
            let warm = octet_channel(&fam);
            assert_eq!(BUILDS.with(std::cell::Cell::get), [1, 0, 0, 1]);
            assert_eq!(bits(&warm), bits(&cold));
            assert_eq!(db.rows_of_kind(DISK_NORMS), 1, "the rebuilt table persists");
        },
    );
}

/// Seeds round-trip through their own rows and load with no module.
#[test]
fn seeds_round_trip_through_disk() {
    isolated(module_path!(), "seeds_round_trip_through_disk", |db| {
        let fam = su3();
        let o = Label::new(&OCTET);
        let seeds = || {
            fam.cgc_seeds_in(o, o, o, &mut RequestModules::default())
                .unwrap()
        };
        let cold = seeds();
        assert_eq!(db.rows_of_kind(DISK_SEEDS), 1);
        reset();
        reset_builds();
        assert!(*seeds() == *cold);
        assert_eq!(BUILDS.with(std::cell::Cell::get), [0; 4]);
    });
}

#[test]
fn f_disk_hit_skips_contraction() {
    isolated(module_path!(), "f_disk_hit_skips_contraction", |_| {
        let fam = su3();
        let cold = octet_f(&fam).unwrap();
        reset();
        let hits = disk_cache_stats().hits;
        CORRUPT_BLOCK.with(|x| x.set(Some(panic_if_contracted)));
        let warm = octet_f(&fam).unwrap();
        CORRUPT_BLOCK.with(|x| x.set(None));
        assert_eq!(disk_cache_stats().hits, hits + 1);
        assert_eq!(block_bits(warm.as_slice()), block_bits(cold.as_slice()));
    });
}

#[test]
fn r_disk_hit_skips_contraction() {
    isolated(module_path!(), "r_disk_hit_skips_contraction", |_| {
        let fam = su3();
        let cold = octet_r(&fam);
        reset();
        let hits = disk_cache_stats().hits;
        CORRUPT_BLOCK.with(|x| x.set(Some(panic_if_contracted)));
        let warm = octet_r(&fam);
        CORRUPT_BLOCK.with(|x| x.set(None));
        assert_eq!(disk_cache_stats().hits, hits + 1);
        assert_eq!(block_bits(warm.as_slice()), block_bits(cold.as_slice()));
    });
}

/// A D(5) spinor is complex, so its phase is computed from two modules.
#[test]
fn phase_round_trips_through_disk() {
    isolated(module_path!(), "phase_round_trips_through_disk", |db| {
        let fam = GeneralFamily::new(RootSystem::D(5));
        let (s, sbar) = ([0i64, 0, 0, 1, 0], [0i64, 0, 0, 0, 1]);
        let phase = || {
            fam.frobenius_schur_phase(Label::new(&s), Label::new(&sbar))
                .unwrap()
        };
        let cold = phase();
        assert_eq!(db.rows_of_kind(DISK_PHASE), 1);
        reset();
        reset_builds();
        assert_eq!(phase(), cold);
        assert_eq!(BUILDS.with(std::cell::Cell::get), [0; 4], "no module built");
    });
}

#[test]
fn nested_success_then_error_writes_inner_rows() {
    isolated(
        module_path!(),
        "nested_success_then_error_writes_inner_rows",
        |db| {
            let fam = su3();
            CORRUPT_BLOCK.with(|x| x.set(Some(|_| f64::INFINITY)));
            assert!(octet_f(&fam).is_err());
            CORRUPT_BLOCK.with(|x| x.set(None));
            assert_eq!(
                db.rows_of_kind(DISK_CHANNEL),
                1,
                "the gated channel is kept"
            );
            assert_eq!(db.rows_of_kind(DISK_F), 0, "the failed block is not");
            assert!(octet_f(&fam).is_ok());
            assert_eq!(db.rows_of_kind(DISK_F), 1);
        },
    );
}

fn panicking_f(fam: &GeneralFamily) {
    CORRUPT_BLOCK.with(|x| x.set(Some(panic_if_contracted)));
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| octet_f(fam)));
    CORRUPT_BLOCK.with(|x| x.set(None));
    assert!(unwound.is_err());
}

#[test]
fn nested_success_then_panic_discards_unflushed_rows() {
    isolated(
        module_path!(),
        "nested_success_then_panic_discards_unflushed_rows",
        |db| {
            let fam = su3();
            panicking_f(&fam);
            assert_eq!(pending_rows(), 0);
            assert_eq!(db.rows_of_kind(DISK_CHANNEL), 0, "queued rows are dropped");
            reset();
            assert!(octet_f(&fam).is_ok());
            assert_eq!(db.rows_of_kind(DISK_CHANNEL), 1);
            assert_eq!(db.rows_of_kind(DISK_F), 1);

            // A row already committed is not rolled back by a later panic.
            reset();
            crate::cache::clear_disk_cache().unwrap();
            CHUNK.with(|c| c.set(Some(1)));
            panicking_f(&fam);
            CHUNK.with(|c| c.set(None));
            assert_eq!(
                db.rows_of_kind(DISK_CHANNEL),
                1,
                "written through before the panic"
            );
            assert_eq!(db.rows_of_kind(DISK_F), 0);
        },
    );
}

static REACHED: AtomicBool = AtomicBool::new(false);
static RELEASE: AtomicBool = AtomicBool::new(false);

fn park() {
    REACHED.store(true, Ordering::SeqCst);
    while !RELEASE.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }
}

/// The leader parks after its disk read; it is released only once the waiter
/// has registered on the leader's placeholder (or finished without waiting),
/// so a read outside the claim would already have happened. No step waits on
/// a condition a faulty build can leave unmet.
#[test]
fn one_leader_reads_disk_with_a_parked_waiter() {
    isolated(
        module_path!(),
        "one_leader_reads_disk_with_a_parked_waiter",
        |db| {
            let cold = octet_channel(&su3());
            assert_eq!(db.rows_of_kind(DISK_CHANNEL), 1);
            reset();
            let before = disk_cache_stats();
            let waits = crate::cache::placeholder_waits();
            let leader = std::thread::spawn(|| {
                AFTER_READ.with(|h| h.set(Some(park)));
                let value = octet_channel(&su3());
                AFTER_READ.with(|h| h.set(None));
                value
            });
            while !REACHED.load(Ordering::SeqCst) && !leader.is_finished() {
                std::thread::yield_now();
            }
            let waiter = std::thread::spawn(|| octet_channel(&su3()));
            while crate::cache::placeholder_waits() == waits && !waiter.is_finished() {
                std::thread::yield_now();
            }
            let parked = crate::cache::placeholder_waits() > waits;
            RELEASE.store(true, Ordering::SeqCst);
            let (a, b) = (leader.join().unwrap(), waiter.join().unwrap());
            assert!(parked, "the waiter never reached the placeholder");
            // One read of the exact channel and one of the octet's norms.
            let after = disk_cache_stats();
            assert_eq!(
                (after.hits, after.misses, after.rejected),
                (before.hits + 2, before.misses, before.rejected)
            );
            assert_eq!(bits(&a), bits(&cold));
            assert_eq!(bits(&b), bits(&cold));
        },
    );
}

#[test]
fn failed_flush_leaves_cache_usable() {
    isolated(module_path!(), "failed_flush_leaves_cache_usable", |db| {
        let fam = su3();
        let failures = disk_cache_stats().failures;
        FAULT_WRITE.with(|f| f.set(true));
        let cold = octet_channel(&fam);
        FAULT_WRITE.with(|f| f.set(false));
        assert_eq!(disk_cache_stats().failures, failures + 1);
        assert_eq!(db.rows_of_kind(DISK_CHANNEL), 0);
        reset();
        assert_eq!(bits(&octet_channel(&fam)), bits(&cold));
        assert_eq!(db.rows_of_kind(DISK_CHANNEL), 1, "a fresh miss persists");
        reset();
        let hits = disk_cache_stats().hits;
        assert_eq!(bits(&octet_channel(&fam)), bits(&cold));
        assert_eq!(disk_cache_stats().hits, hits + 2, "and reloads");
    });
}
