//! The disk cache across process boundaries and through the public API. The
//! store is process-global, so every test runs its body alone in a fresh
//! child process of this binary. Cross-process bit identity is a declared
//! contract: a value served from disk equals the value generated without it.

#[path = "common/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOC: counting_alloc::Counting = counting_alloc::Counting;

/// For `counting_alloc::serial`; each test here already owns its process.
pub fn serial() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    SERIAL.lock().unwrap_or_else(|p| p.into_inner())
}

use std::path::{Path, PathBuf};

use fusion_symbols::cache::{disk_cache_stats, enable_disk_cache, DiskCacheError, DiskCacheStats};
use fusion_symbols::category::{CgcBlock, GroupRepresentations};
use fusion_symbols::lie::{GroupId, LieGroup};
#[cfg(feature = "lie-generated")]
use fusion_symbols::{
    cache::reset,
    category::{FusionCategory, SymbolBlock},
};

const CHILD: &str = "FUSION_SYMBOLS_DISK_TEST";

/// A database path in the temporary directory, removed with its WAL files.
struct TempDb(PathBuf);

impl TempDb {
    fn new(name: &str) -> Self {
        Self(std::env::temp_dir().join(format!("fs-disk-it-{}-{name}.sqlite", std::process::id())))
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", self.0.display()));
        }
    }
}

/// Runs the test `name` alone in a child with `envs`; returns its stdout.
fn run_child(name: &str, envs: &[(&str, &Path)]) -> String {
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", name, "--test-threads=1", "--nocapture"])
        .env(CHILD, "1");
    for (k, v) in envs {
        command.env(k, v);
    }
    let out = command.output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "{name}\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

/// `body` in a fresh process; the parent only checks that it passed.
fn isolated(name: &str, body: impl FnOnce()) {
    if std::env::var_os(CHILD).is_some() {
        body();
    } else {
        run_child(name, &[]);
    }
}

fn put(out: &mut Vec<u8>, dims: &[usize], values: impl IntoIterator<Item = f64>) {
    for d in dims {
        out.extend_from_slice(&(*d as u64).to_le_bytes());
    }
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

fn cgc_bytes(out: &mut Vec<u8>, g: &LieGroup, labels: [&[i64]; 3]) {
    let [a, b, c] = labels.map(|l| g.irrep(l).unwrap());
    let cg = g.cgc(&a, &b, &c).unwrap();
    let [n1, n2, n3, n4] = cg.dims();
    let mut values = Vec::new();
    for i in 0..n1 {
        for j in 0..n2 {
            for k in 0..n3 {
                for mu in 0..n4 {
                    values.push(cg.get([i, j, k, mu]));
                }
            }
        }
    }
    put(out, &cg.dims(), values);
}

/// The bytes of every persisted kind: A(1) CGC in every build; higher-rank
/// CGC, F, R and a complex phase with `lie-generated`.
fn emit() -> Vec<u8> {
    let mut out = Vec::new();
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    cgc_bytes(&mut out, &su2, [&[20], &[20], &[20]]);
    #[cfg(feature = "lie-generated")]
    {
        let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        cgc_bytes(&mut out, &su3, [&[1, 1], &[1, 1], &[1, 1]]);
        let o = su3.irrep(&[1, 1]).unwrap();
        let f = su3.f_symbol(&o, &o, &o, &o, &o, &o).unwrap();
        put(&mut out, &f.dims(), f.as_slice().iter().copied());
        let r = su3.r_symbol(&o, &o, &o).unwrap();
        put(&mut out, &r.dims(), r.as_slice().iter().copied());
        let b2 = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
        cgc_bytes(&mut out, &b2, [&[1, 0], &[1, 0], &[0, 2]]);
        let v = b2.irrep(&[1, 0]).unwrap();
        let one = b2.unit();
        let f = b2.f_symbol(&v, &v, &v, &v, &one, &one).unwrap();
        put(&mut out, &f.dims(), f.as_slice().iter().copied());
        let g2 = LieGroup::new(GroupId::g2()).unwrap();
        cgc_bytes(&mut out, &g2, [&[1, 0], &[1, 0], &[1, 0]]);
        let spin10 = LieGroup::new(GroupId::spin(10).unwrap()).unwrap();
        let spinor = spin10.irrep(&[0, 0, 0, 1, 0]).unwrap();
        put(
            &mut out,
            &[],
            [spin10.frobenius_schur_phase(&spinor).unwrap()],
        );
    }
    out
}

const OUT: &str = "FUSION_SYMBOLS_DISK_OUT";
const DB: &str = "FUSION_SYMBOLS_DISK_DB";

fn stats_line(s: DiskCacheStats) -> String {
    format!("disk-stats hits={} misses={}", s.hits, s.misses)
}

/// Child of `second_process_reuses_disk_values_bit_identically`: emits with
/// the store at `DB` if given, otherwise without one.
#[test]
#[ignore = "child process of second_process_reuses_disk_values_bit_identically; light; run by that test"]
fn emit_child() {
    let Some(out) = std::env::var_os(OUT) else {
        return;
    };
    if let Some(db) = std::env::var_os(DB) {
        enable_disk_cache(db, 64 << 20).unwrap();
    }
    std::fs::write(out, emit()).unwrap();
    println!("{}", stats_line(disk_cache_stats()));
}

#[test]
fn second_process_reuses_disk_values_bit_identically() {
    let db = TempDb::new("reuse");
    let run = |name: &str, with_db: bool| {
        let out = TempDb::new(name);
        let mut envs = vec![(OUT, out.0.as_path())];
        if with_db {
            envs.push((DB, db.0.as_path()));
        }
        let stdout = run_child_ignored("emit_child", &envs);
        let bytes = std::fs::read(&out.0).unwrap();
        let at = stdout.find("disk-stats").unwrap();
        let stats = stdout[at..].lines().next().unwrap().to_owned();
        (bytes, stats)
    };
    let (off, _) = run("off", false);
    let (fill, fill_stats) = run("fill", true);
    let (disk, disk_stats) = run("disk", true);
    assert!(!off.is_empty());
    assert_eq!(fill, off, "a filling process generates the same bits");
    assert_eq!(disk, off, "a disk-served process returns the same bits");
    assert!(fill_stats.contains("hits=0 ") && !fill_stats.ends_with("misses=0"));
    assert!(disk_stats.ends_with("misses=0") && !disk_stats.contains("hits=0 "));
}

fn run_child_ignored(name: &str, envs: &[(&str, &Path)]) -> String {
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command.args([
        "--exact",
        name,
        "--include-ignored",
        "--test-threads=1",
        "--nocapture",
    ]);
    for (k, v) in envs {
        command.env(k, v);
    }
    let out = command.output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "{name}\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

#[test]
fn warm_memory_hit_reads_no_disk_and_allocates_nothing() {
    isolated(
        "warm_memory_hit_reads_no_disk_and_allocates_nothing",
        || {
            let db = TempDb::new("warm");
            enable_disk_cache(&db.0, 64 << 20).unwrap();
            let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
            let x = su2.irrep(&[4]).unwrap();
            let query = || su2.cgc(&x, &x, &x).unwrap();
            std::hint::black_box(query());
            let before = disk_cache_stats();
            let (calls, bytes, _, value) = counting_alloc::measure(query);
            std::hint::black_box(value);
            assert_eq!((calls, bytes), (0, 0), "warm allocation");
            assert_eq!(disk_cache_stats(), before, "a warm hit read the disk");
            #[cfg(feature = "lie-generated")]
            {
                let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
                let o = su3.irrep(&[1, 1]).unwrap();
                let f = || su3.f_symbol(&o, &o, &o, &o, &o, &o).unwrap();
                std::hint::black_box(f());
                let before = disk_cache_stats();
                let (calls, bytes, _, value) = counting_alloc::measure(f);
                std::hint::black_box(value);
                assert_eq!((calls, bytes), (0, 0), "warm F allocation");
                assert_eq!(disk_cache_stats(), before, "a warm F hit read the disk");
            }
        },
    );
}

#[cfg(feature = "lie-generated")]
#[test]
fn prefill_on_team_writes_and_rereads() {
    use fusion_symbols::lie::{prefill, PrefillRequest};
    use std::time::Duration;
    use threadteam::{DispatchCosts, Pool, WidthCosts};
    isolated("prefill_on_team_writes_and_rereads", || {
        let db = TempDb::new("prefill");
        enable_disk_cache(&db.0, 64 << 20).unwrap();
        let zero = WidthCosts::flat(Duration::ZERO);
        let team = Pool::builder()
            .threads(4)
            .acquire_timeout(Duration::from_secs(10))
            .dispatch_costs(DispatchCosts::new(zero, zero, zero))
            .build()
            .unwrap()
            .into_team(4)
            .unwrap();
        let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        let (o, t) = (su3.irrep(&[1, 1]).unwrap(), su3.irrep(&[2, 2]).unwrap());
        let mut request = PrefillRequest::new(&su3);
        request
            .cgc(&o, &o, &o)
            .cgc(&o, &o, &t)
            .f_symbol(&o, &o, &o, &o, &o, &o)
            .r_symbol(&o, &o, &o);
        let read = || {
            let f = su3.f_symbol(&o, &o, &o, &o, &o, &o).unwrap();
            let c = su3.cgc(&o, &o, &t).unwrap();
            (
                f.as_slice().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                c.dims(),
            )
        };
        prefill(&request, team.exec(4)).unwrap();
        let filled = read();
        let written = disk_cache_stats();
        assert!(written.bytes_written > 0 && written.hits == 0);
        reset();
        prefill(&request, team.exec(4)).unwrap();
        let reread = disk_cache_stats();
        assert_eq!(
            reread.misses, written.misses,
            "the second prefill generated"
        );
        assert!(reread.hits > 0);
        assert_eq!(read(), filled);
    });
}

#[test]
fn enable_twice_is_rejected() {
    isolated("enable_twice_is_rejected", || {
        let db = TempDb::new("twice");
        enable_disk_cache(&db.0, 1 << 20).unwrap();
        assert!(matches!(
            enable_disk_cache(&db.0, 1 << 20),
            Err(DiskCacheError::AlreadyEnabled)
        ));
    });
}

#[test]
fn failed_enable_can_be_retried() {
    isolated("failed_enable_can_be_retried", || {
        let dir = std::env::temp_dir().join(format!("fs-disk-it-{}-dir", std::process::id()));
        let db = TempDb(dir.join("cache.sqlite"));
        assert!(matches!(
            enable_disk_cache(&db.0, 1 << 20),
            Err(DiskCacheError::Open { .. })
        ));
        std::fs::create_dir_all(&dir).unwrap();
        enable_disk_cache(&db.0, 1 << 20).unwrap();
        drop(db);
        let _ = std::fs::remove_dir(&dir);
    });
}

#[test]
fn clear_before_enable_is_rejected_with_a_message() {
    isolated("clear_before_enable_is_rejected_with_a_message", || {
        let e = fusion_symbols::cache::clear_disk_cache().unwrap_err();
        assert!(matches!(e, DiskCacheError::NotEnabled));
        assert_eq!(e.to_string(), "the disk cache is not enabled");
    });
}
