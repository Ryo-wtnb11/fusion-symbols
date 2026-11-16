//! Fresh-process disk-cache probe (a measurement, not a test).
//!
//! ```text
//! disk_cache <case> <off|fill|disk> <db> [--e2e]
//! disk_cache --list
//! ```
//!
//! - `off`: no store; `fill`: the store at `<db>`, which the caller empties
//!   first; `disk`: the store at `<db>` as a previous `fill` left it.
//! - Default timing: groups and labels are built and one unrelated query is
//!   warmed outside the timer, then the store is enabled outside the timer,
//!   then one target call is timed.
//! - `--e2e`: timed from before `enable_disk_cache` (open, setup, trim) to
//!   the end of the case, with group and label construction inside.
//!
//! One tab-separated line: case, mode, e2e, ns, peak RSS KiB (Linux
//! `VmHWM`, else -1), disk hits, misses, rejected, failures, bytes written,
//! stored payload bytes, file bytes, WAL bytes, page size and
//! `wal_autocheckpoint`.
//!
//! Without the `disk-cache` feature every mode runs without a store and the
//! disk columns are zero, so the same probe measures builds that predate it.

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use fusion_symbols::category::{FusionCategory, GroupRepresentations};
use fusion_symbols::lie::{GroupId, LieGroup, LieIrrep};

type Case = fn() -> Box<dyn FnOnce()>;

fn group(id: GroupId) -> LieGroup {
    LieGroup::new(id).unwrap()
}

fn cgc(id: GroupId, a: &'static [i64], b: &'static [i64], c: &'static [i64]) -> Box<dyn FnOnce()> {
    let g = group(id);
    let [a, b, c] = [a, b, c].map(|l| g.irrep(l).unwrap());
    Box::new(move || {
        black_box(g.cgc(&a, &b, &c).unwrap());
    })
}

/// `c` is the dual of `a` when `c` is empty.
fn cgc_dual(id: GroupId, a: &'static [i64]) -> Box<dyn FnOnce()> {
    let g = group(id);
    let a = g.irrep(a).unwrap();
    let c = g.dual(&a);
    Box::new(move || {
        black_box(g.cgc(&a, &a, &c).unwrap());
    })
}

fn f_all(id: GroupId, a: &'static [i64]) -> Box<dyn FnOnce()> {
    let g = group(id);
    let a = g.irrep(a).unwrap();
    Box::new(move || {
        black_box(g.f_symbol(&a, &a, &a, &a, &a, &a).unwrap());
    })
}

fn r_all(id: GroupId, a: &'static [i64]) -> Box<dyn FnOnce()> {
    let g = group(id);
    let a = g.irrep(a).unwrap();
    Box::new(move || {
        black_box(g.r_symbol(&a, &a, &a).unwrap());
    })
}

fn phase(id: GroupId, a: &'static [i64]) -> Box<dyn FnOnce()> {
    let g = group(id);
    let a = g.irrep(a).unwrap();
    Box::new(move || {
        black_box(g.frobenius_schur_phase(&a).unwrap());
    })
}

/// Every admissible CGC and R over `set`, and every F whose four outer
/// labels lie in `outer`: the coefficient plan of a small symmetric network.
fn workset(id: GroupId, set: &'static [&'static [i64]], outer: usize) -> Box<dyn FnOnce()> {
    Box::new(move || {
        let g = group(id);
        let set: Vec<LieIrrep> = set.iter().map(|l| g.irrep(l).unwrap()).collect();
        for a in &set {
            for b in &set {
                for (c, _) in g.fusion(a, b).unwrap() {
                    if set.contains(&c) {
                        black_box(g.cgc(a, b, &c).unwrap());
                        black_box(g.r_symbol(a, b, &c).unwrap());
                    }
                }
            }
        }
        let outer = &set[..outer];
        for a in outer {
            for b in outer {
                for c in outer {
                    for d in outer {
                        for (e, _) in g.fusion(a, b).unwrap() {
                            if g.nsymbol(&e, c, d).unwrap() == 0 {
                                continue;
                            }
                            for (f, _) in g.fusion(b, c).unwrap() {
                                if g.nsymbol(a, &f, d).unwrap() > 0 {
                                    black_box(g.f_symbol(a, b, c, d, &e, &f).unwrap());
                                }
                            }
                        }
                    }
                }
            }
        }
    })
}

fn su(n: usize) -> GroupId {
    GroupId::su(n).unwrap()
}

const CASES: &[(&str, Case)] = &[
    // Declared large rows.
    ("cgc2_l", || cgc(su(2), &[200], &[200], &[200])),
    ("cgc3_d", || cgc(su(3), &[3, 3], &[3, 3], &[3, 3])),
    ("cgc4_c", || cgc(su(4), &[1, 1, 1], &[1, 1, 1], &[2, 0, 2])),
    ("cgc6_adj", || {
        let a: &[i64] = &[1, 0, 0, 0, 1];
        cgc(su(6), a, a, a)
    }),
    ("f6_adj", || f_all(su(6), &[1, 0, 0, 0, 1])),
    ("cgcd4_adj", || {
        let a: &[i64] = &[0, 1, 0, 0];
        cgc(GroupId::spin(8).unwrap(), a, a, a)
    }),
    ("cgcg2_27", || cgc(GroupId::g2(), &[2, 0], &[2, 0], &[2, 0])),
    ("cgcf4_26", || {
        let a: &[i64] = &[0, 0, 0, 1];
        cgc(GroupId::f4(), a, a, a)
    }),
    ("cgce6_27", || cgc_dual(GroupId::e6(), &[1, 0, 0, 0, 0, 0])),
    // Mechanism rows.
    ("cgc3_b", || cgc(su(3), &[1, 1], &[1, 1], &[1, 1])),
    ("cgc3_c", || cgc(su(3), &[2, 2], &[2, 2], &[2, 2])),
    ("f3_b", || f_all(su(3), &[1, 1])),
    ("f3_c", || f_all(su(3), &[2, 2])),
    ("r3_c", || r_all(su(3), &[2, 2])),
    ("cgc4_b", || cgc(su(4), &[1, 0, 1], &[1, 0, 1], &[1, 0, 1])),
    ("f4_b", || f_all(su(4), &[1, 0, 1])),
    ("cgcb3_adj", || {
        let a: &[i64] = &[0, 1, 0];
        cgc(GroupId::spin(7).unwrap(), a, a, a)
    }),
    ("cgcc3_14", || {
        let a: &[i64] = &[0, 1, 0];
        cgc(GroupId::sp(6).unwrap(), a, a, a)
    }),
    ("fb2_adj", || f_all(GroupId::spin(5).unwrap(), &[0, 2])),
    ("rg2_27", || r_all(GroupId::g2(), &[2, 0])),
    ("phasee6_27", || phase(GroupId::e6(), &[1, 0, 0, 0, 0, 0])),
    ("cgc3_a", || cgc(su(3), &[1, 0], &[0, 1], &[1, 1])),
    // End-to-end working sets (run with --e2e).
    ("ws_su3", || {
        workset(
            su(3),
            &[
                &[1, 0],
                &[0, 1],
                &[1, 1],
                &[2, 0],
                &[0, 2],
                &[3, 0],
                &[0, 3],
                &[2, 2],
            ],
            3,
        )
    }),
    ("ws_su4", || {
        workset(
            su(4),
            &[&[1, 0, 0], &[0, 0, 1], &[1, 0, 1], &[0, 1, 0], &[0, 2, 0]],
            3,
        )
    }),
    // A sustained fill: every CGC and R over the SU(3) irreps with Dynkin
    // labels up to 3, far beyond one auto-checkpoint of WAL pages.
    ("sustain_su3", || {
        workset(
            su(3),
            &[
                &[0, 1],
                &[0, 2],
                &[0, 3],
                &[1, 0],
                &[1, 1],
                &[1, 2],
                &[1, 3],
                &[2, 0],
                &[2, 1],
                &[2, 2],
                &[2, 3],
                &[3, 0],
                &[3, 1],
                &[3, 2],
                &[3, 3],
            ],
            0,
        )
    }),
];

/// `(hits, misses, rejected, failures, bytes written)` of this process.
#[cfg(feature = "disk-cache")]
fn disk_counters() -> [u64; 5] {
    let s = fusion_symbols::cache::disk_cache_stats();
    [s.hits, s.misses, s.rejected, s.failures, s.bytes_written]
}

#[cfg(not(feature = "disk-cache"))]
fn disk_counters() -> [u64; 5] {
    [0; 5]
}

#[cfg(feature = "disk-cache")]
fn enable(mode: &str, db: &Path) {
    if mode != "off" {
        fusion_symbols::cache::enable_disk_cache(db, 1 << 30).unwrap();
    }
}

#[cfg(not(feature = "disk-cache"))]
fn enable(_: &str, _: &Path) {}

fn peak_rss_kib() -> i64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(-1)
}

fn file_bytes(path: &Path, suffix: &str) -> u64 {
    std::fs::metadata(format!("{}{suffix}", path.display())).map_or(0, |m| m.len())
}

/// Stored payload, page size and auto-checkpoint pages, read after the run.
#[cfg(not(feature = "disk-cache"))]
fn store_facts(_: &Path) -> (i64, i64, i64) {
    (0, -1, -1)
}

/// Stored payload, page size and auto-checkpoint pages, read after the run.
#[cfg(feature = "disk-cache")]
fn store_facts(path: &Path) -> (i64, i64, i64) {
    let Ok(conn) = rusqlite::Connection::open(path) else {
        return (-1, -1, -1);
    };
    let q = |sql: &str| conn.query_row(sql, [], |r| r.get(0)).unwrap_or(-1);
    (
        q("SELECT bytes FROM meta WHERE id = 0"),
        q("PRAGMA page_size"),
        q("PRAGMA wal_autocheckpoint"),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--list") {
        for (name, _) in CASES {
            println!("{name}");
        }
        return;
    }
    let [name, mode, db, ..] = &args[..] else {
        eprintln!("usage: disk_cache <case> <off|fill|disk> <db> [--e2e]");
        std::process::exit(2);
    };
    let e2e = args.iter().any(|a| a == "--e2e");
    let case = CASES
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("unknown case {name}"))
        .1;
    let db = Path::new(db);
    let start_store = || enable(mode, db);
    let ns = if e2e {
        let start = Instant::now();
        start_store();
        case()();
        start.elapsed().as_nanos()
    } else {
        let query = case();
        cgc(su(2), &[1], &[1], &[2])();
        start_store();
        let start = Instant::now();
        query();
        start.elapsed().as_nanos()
    };
    let [hits, misses, rejected, failures, written] = disk_counters();
    let (stored, page, autockpt) = if mode == "off" {
        (0, -1, -1)
    } else {
        store_facts(db)
    };
    println!(
        "{name}\t{mode}\t{e2e}\t{ns}\t{}\t{hits}\t{misses}\t{rejected}\t{failures}\t{written}\t{stored}\t{}\t{}\t{page}\t{autockpt}",
        peak_rss_kib(),
        file_bytes(db, ""),
        file_bytes(db, "-wal"),
    );
}
