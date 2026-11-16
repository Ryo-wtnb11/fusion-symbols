//! Codec and store contracts on local files, plus the harness the hook tests
//! in `lie::general::cgc::disk_tests` share. Expected values are inline:
//! hand-built rows whose shape is stated next to them.

use super::*;
use std::cell::Cell;
use std::sync::atomic::AtomicUsize;

thread_local! {
    /// Replaces the returned journal mode with `delete` on this thread.
    pub(crate) static FAULT_WAL_MODE: Cell<bool> = const { Cell::new(false) };
    /// Fails every write transaction on this thread.
    pub(crate) static FAULT_WRITE: Cell<bool> = const { Cell::new(false) };
    /// Replaces the queue's byte limit on this thread.
    pub(crate) static CHUNK: Cell<Option<usize>> = const { Cell::new(None) };
    /// Largest queued byte count reached on this thread.
    pub(crate) static PEAK: Cell<usize> = const { Cell::new(0) };
    /// Replaces the setup lock-wait budget on this thread.
    pub(crate) static SETUP_BUDGET_OVERRIDE: Cell<Option<Duration>> = const { Cell::new(None) };
    /// Sleeps setup work this long after the WAL switch, on this thread.
    pub(crate) static SLOW_SETUP: Cell<Option<Duration>> = const { Cell::new(None) };
    /// Lock-wait sleeps of setups on this thread.
    pub(crate) static SETUP_WAITS: std::cell::RefCell<Vec<Duration>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Run on this thread after a row was read, before it is checked.
    pub(crate) static AFTER_READ: Cell<Option<fn()>> = const { Cell::new(None) };
}

pub(super) fn after_read() {
    if let Some(hook) = AFTER_READ.with(Cell::get) {
        hook();
    }
}

/// Rows queued on this thread.
pub(crate) fn pending_rows() -> usize {
    PENDING.with(|p| p.borrow().rows.len())
}

/// A database file in the temporary directory, removed with its WAL files.
pub(crate) struct TempDb(PathBuf);

impl TempDb {
    pub(crate) fn new(name: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = name.replace("::", "-");
        Self(std::env::temp_dir().join(format!("fs-disk-{}-{n}-{name}.sqlite", std::process::id())))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }

    /// Stored rows of `kind`, read through a separate connection.
    pub(crate) fn rows_of_kind(&self, kind: u8) -> i64 {
        let conn = Connection::open(&self.0).unwrap();
        conn.query_row(
            "SELECT count(*) FROM entries WHERE substr(key, 7, 1) = ?1",
            (&[kind][..],),
            |row| row.get(0),
        )
        .unwrap()
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{suffix}", self.0.display()));
        }
    }
}

const CHILD: &str = "FUSION_SYMBOLS_DISK_CHILD";

/// Runs `body` as the only test of a fresh process whose disk cache is
/// enabled at a new file, so no other test shares its coefficient tiers or
/// its store. `module` is the caller's `module_path!()`.
pub(crate) fn isolated(module: &str, name: &str, body: impl FnOnce(&TempDb)) {
    let test = format!(
        "{}::{name}",
        module.strip_prefix("fusion_symbols::").unwrap_or(module)
    );
    if std::env::var_os(CHILD).is_some() {
        let db = TempDb::new(name);
        enable_disk_cache(db.path(), 64 << 20).unwrap();
        body(&db);
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test, "--test-threads=1"])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "{test}\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn store(db: &TempDb, max_bytes: u64) -> Store {
    Store::open(db.path(), max_bytes, Identity::CURRENT).unwrap()
}

fn su3_key(kind: u8, labels: [[i64; 2]; 3]) -> Vec<u8> {
    key_bytes(
        kind,
        RootSystem::A(2),
        &labels.each_ref().map(|l| Label::new(l)),
    )
}

fn row(key: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut row = payload.to_vec();
    row.extend_from_slice(&checksum(key, payload).to_le_bytes());
    row
}

fn payload<V: Persist>(value: &V) -> Vec<u8> {
    let mut out = Vec::new();
    value.encode(&mut out);
    assert_eq!(out.len(), value.payload_len().unwrap());
    out
}

fn entry(m1: u32, m2: u32, m3: u32, mu: u32, value: f64) -> MEntry {
    MEntry {
        m1,
        m2,
        m3,
        mu,
        value,
    }
}

/// `2 ⊗ 2 → 1` with `N = 2`: two columns `(μ, m_c) = (0, 0), (1, 0)`.
fn two_column_channel() -> (Arc<CgcChannel>, Shape) {
    let channel = Arc::new(CgcChannel {
        mult: 2,
        entries: Box::new([
            entry(0, 1, 0, 0, std::f64::consts::FRAC_1_SQRT_2),
            entry(1, 0, 0, 0, -0.0),
            entry(0, 1, 0, 1, f64::from_bits(1)),
            entry(1, 0, 0, 1, -std::f64::consts::FRAC_1_SQRT_2),
        ]),
    });
    let shape = Shape::Channel {
        dims: [2, 2, 1],
        mult: 2,
    };
    (channel, shape)
}

fn channel_bits(c: &CgcChannel) -> Vec<(u32, u32, u32, u32, u64)> {
    c.entries
        .iter()
        .map(|e| (e.m1, e.m2, e.m3, e.mu, e.value.to_bits()))
        .collect()
}

fn block_bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn each_kind_round_trips_bit_identically() {
    let (channel, shape) = two_column_channel();
    let back = Arc::<CgcChannel>::decode(&payload(&channel), &shape).unwrap();
    assert_eq!(back.mult, 2);
    assert_eq!(channel_bits(&back), channel_bits(&channel));

    for k in [1i8, -1] {
        assert_eq!(i8::decode(&payload(&k), &Shape::Phase), Some(k));
    }

    let f = FBlock::from_vec([1, 2, 1, 2], vec![0.5, -0.0, f64::MIN_POSITIVE / 2.0, -1.0]);
    let back = FBlock::decode(&payload(&f), &Shape::F([1, 2, 1, 2])).unwrap();
    assert_eq!(block_bits(back.as_slice()), block_bits(f.as_slice()));
    let empty = FBlock::<f64>::zeros([0, 2, 1, 1]);
    let back = FBlock::decode(&payload(&empty), &Shape::F([0, 2, 1, 1])).unwrap();
    assert_eq!(back.dims(), [0, 2, 1, 1]);

    // A rotation: orthogonal, with a signed zero nowhere needed.
    let r = RBlock::from_vec(2, vec![0.6, 0.8, -0.8, 0.6]);
    let back = RBlock::decode(&payload(&r), &Shape::R(2)).unwrap();
    assert_eq!(block_bits(back.as_slice()), block_bits(r.as_slice()));
}

#[test]
fn signed_zero_entry_round_trips() {
    let (channel, shape) = two_column_channel();
    let back = Arc::<CgcChannel>::decode(&payload(&channel), &shape).unwrap();
    assert_eq!(back.entries[1].value.to_bits(), (-0.0f64).to_bits());
}

#[test]
fn transplanted_row_is_rejected() {
    let (channel, shape) = two_column_channel();
    let a = su3_key(super::super::DISK_CHANNEL, [[1, 0], [1, 0], [0, 1]]);
    let b = su3_key(super::super::DISK_CHANNEL, [[0, 1], [0, 1], [1, 0]]);
    let stored = row(&a, &payload(&channel));
    assert!(decode_row::<Arc<CgcChannel>>(&a, &stored, &shape).is_some());
    assert!(decode_row::<Arc<CgcChannel>>(&b, &stored, &shape).is_none());
}

/// The key has 15 + 48 bytes and the payload 8 + 24·4, so the message ends in
/// a 7-byte tail: bytes 1..8 of the last coefficient.
#[test]
fn tail_bit_flip_is_caught_only_by_checksum() {
    let (channel, shape) = two_column_channel();
    let key = su3_key(super::super::DISK_CHANNEL, [[1, 0], [1, 0], [0, 1]]);
    let mut stored = row(&key, &payload(&channel));
    assert_eq!((key.len() + stored.len() - 8) % 8, 7);
    let last_value = stored.len() - 8 - 8;
    stored[last_value + 1] ^= 0x10;
    let (body, _) = stored.split_at(stored.len() - 8);
    assert!(
        Arc::<CgcChannel>::decode(body, &shape).is_some(),
        "structurally valid"
    );
    assert!(decode_row::<Arc<CgcChannel>>(&key, &stored, &shape).is_none());
}

#[test]
fn truncated_and_overlong_values_are_rejected() {
    let (channel, shape) = two_column_channel();
    let key = su3_key(super::super::DISK_CHANNEL, [[1, 0], [1, 0], [0, 1]]);
    let stored = row(&key, &payload(&channel));
    for len in 0..stored.len() {
        assert!(decode_row::<Arc<CgcChannel>>(&key, &stored[..len], &shape).is_none());
    }
    let mut long = payload(&channel);
    long.push(0);
    assert!(decode_row::<Arc<CgcChannel>>(&key, &row(&key, &long), &shape).is_none());
}

/// Every row below has a valid checksum; only the structure is wrong.
#[test]
fn structurally_wrong_rows_with_valid_checksum_are_rejected() {
    let (good, shape) = two_column_channel();
    let key = su3_key(super::super::DISK_CHANNEL, [[1, 0], [1, 0], [0, 1]]);
    let channel = |mult, entries: Vec<MEntry>| {
        payload(&Arc::new(CgcChannel {
            mult,
            entries: entries.into_boxed_slice(),
        }))
    };
    let e = good.entries.to_vec();
    let swapped = vec![e[1], e[0], e[2], e[3]];
    let duplicate = vec![e[0], e[0], e[2], e[3]];
    let mut far = e.clone();
    far[3].m3 = 1;
    let mut high_mu = e.clone();
    high_mu[3].mu = 2;
    let mut nan = e.clone();
    nan[2].value = f64::NAN;
    let mut wide = e.clone();
    wide[1].m1 = 2;
    let cases: [(&str, Vec<u8>); 8] = [
        ("unsorted", channel(2, swapped)),
        ("duplicate", channel(2, duplicate)),
        ("m3 out of range", channel(2, far)),
        ("mu out of range", channel(2, high_mu)),
        ("nan", channel(2, nan)),
        ("m1 out of range", channel(2, wide)),
        ("empty with N = 2", channel(2, vec![])),
        ("missing column", channel(2, e[..2].to_vec())),
    ];
    for (name, bytes) in cases {
        let stored = row(&key, &bytes);
        assert!(
            decode_row::<Arc<CgcChannel>>(&key, &stored, &shape).is_none(),
            "{name}"
        );
    }
    let stored = row(&key, &payload(&good));
    let n1 = Shape::Channel {
        dims: [2, 2, 1],
        mult: 1,
    };
    assert!(
        decode_row::<Arc<CgcChannel>>(&key, &stored, &n1).is_none(),
        "N mismatch"
    );
    let mut header = payload(&good);
    header[0] = 3;
    let header_row = row(&key, &header);
    assert!(
        decode_row::<Arc<CgcChannel>>(&key, &header_row, &shape).is_none(),
        "stored N"
    );
    let zero = Shape::Channel {
        dims: [2, 2, 1],
        mult: 0,
    };
    let row_in_zero = row(&key, &channel(0, vec![e[0]]));
    assert!(
        decode_row::<Arc<CgcChannel>>(&key, &row_in_zero, &zero).is_none(),
        "zero channel"
    );

    let f = FBlock::from_vec([1, 1, 1, 2], vec![0.5, 0.5]);
    let f_row = row(&key, &payload(&f));
    assert!(
        decode_row::<FBlock<f64>>(&key, &f_row, &Shape::F([1, 1, 2, 1])).is_none(),
        "F dims"
    );
    let mut huge = Vec::new();
    for _ in 0..4 {
        put_u32(&mut huge, u32::MAX as usize);
    }
    let huge_shape = Shape::F([u32::MAX as usize; 4]);
    assert!(
        decode_row::<FBlock<f64>>(&key, &row(&key, &huge), &huge_shape).is_none(),
        "F overflow"
    );
    // 2^16 per axis: a wrapping product would be 0 and accept no values.
    let mut wraps = Vec::new();
    for _ in 0..4 {
        put_u32(&mut wraps, 1 << 16);
    }
    let wrap_shape = Shape::F([1 << 16; 4]);
    assert!(
        decode_row::<FBlock<f64>>(&key, &row(&key, &wraps), &wrap_shape).is_none(),
        "F wrap"
    );
    let bad_f = FBlock::from_vec([1, 1, 1, 2], vec![0.5, f64::INFINITY]);
    let bad_f_row = row(&key, &payload(&bad_f));
    assert!(
        decode_row::<FBlock<f64>>(&key, &bad_f_row, &Shape::F([1, 1, 1, 2])).is_none(),
        "F gate"
    );

    let r1 = RBlock::from_vec(1, vec![1.0]);
    let r1_row = row(&key, &payload(&r1));
    assert!(
        decode_row::<RBlock<f64>>(&key, &r1_row, &Shape::R(2)).is_none(),
        "R n"
    );
    let mut r_header = payload(&RBlock::from_vec(2, vec![0.6, 0.8, -0.8, 0.6]));
    r_header[0] = 3;
    let r_header_row = row(&key, &r_header);
    assert!(
        decode_row::<RBlock<f64>>(&key, &r_header_row, &Shape::R(2)).is_none(),
        "stored R n"
    );
    let skew = RBlock::from_vec(2, vec![1.0, 0.5, 0.0, 1.0]);
    let skew_row = row(&key, &payload(&skew));
    assert!(
        decode_row::<RBlock<f64>>(&key, &skew_row, &Shape::R(2)).is_none(),
        "R gate"
    );
    let phase_row = row(&key, &[2]);
    assert!(
        decode_row::<i8>(&key, &phase_row, &Shape::Phase).is_none(),
        "phase"
    );
}

fn meta_bytes(store: &Store) -> (i64, i64) {
    let conn = store.lock();
    let meta = conn
        .query_row("SELECT bytes FROM meta WHERE id = 0", [], |r| r.get(0))
        .unwrap();
    let sum = conn
        .query_row(
            "SELECT coalesce(sum(length(key) + length(value)), 0) FROM entries",
            [],
            |r| r.get(0),
        )
        .unwrap();
    (meta, sum)
}

fn has_key(store: &Store, key: &[u8]) -> bool {
    matches!(store.read(key), Read::Row(_))
}

/// A row of `n` value bytes under a key numbered `i`.
fn raw(i: u8, n: usize) -> (Vec<u8>, Vec<u8>) {
    (vec![i; 16], vec![i; n])
}

#[test]
fn other_version_is_cleared_at_enable() {
    let db = TempDb::new("other-version");
    let old = Identity {
        minor: Identity::CURRENT.minor + 1,
        ..Identity::CURRENT
    };
    let a = Store::open(db.path(), 1 << 20, old).unwrap();
    a.write(&[raw(1, 100)]);
    assert!(has_key(&a, &[1; 16]));
    drop(a);
    let b = store(&db, 1 << 20);
    assert!(!has_key(&b, &[1; 16]));
    assert_eq!(meta_bytes(&b), (0, 0));
}

#[test]
fn meta_bytes_equal_sql_sum() {
    let db = TempDb::new("meta-bytes");
    let s = store(&db, 1000);
    s.write(&[raw(1, 100), raw(2, 200)]);
    assert_eq!(meta_bytes(&s), (332, 332));
    s.write(&[(vec![1; 16], vec![9; 50])]);
    assert_eq!(meta_bytes(&s), (282, 282), "repair update");
    s.write(&[raw(3, 900)]);
    let (meta, sum) = meta_bytes(&s);
    assert_eq!(meta, sum, "after trim");
    assert!(meta <= 1000);
}

#[test]
fn stored_bytes_stay_within_bound() {
    let db = TempDb::new("bound");
    let s = store(&db, 1000);
    for i in 0..20 {
        s.write(&[raw(i, 84)]);
        assert!(meta_bytes(&s).0 <= 1000);
    }
    assert!(has_key(&s, &[19; 16]), "the newest row survives");
    assert!(!has_key(&s, &[0; 16]), "the oldest row goes first");
}

#[test]
fn over_budget_file_trimmed_at_enable() {
    let db = TempDb::new("reopen");
    let big = store(&db, 1 << 20);
    for i in 0..20 {
        big.write(&[raw(i, 84)]);
    }
    drop(big);
    let small = store(&db, 500);
    assert!(meta_bytes(&small).0 <= 500);
    // 20 rows of 100 bytes under a 500-byte bound: 15 removed at enable.
    assert_eq!(small.counters.snapshot().evictions, 15);
    assert!(has_key(&small, &[19; 16]));
}

#[test]
fn oversized_entry_is_not_enqueued() {
    let db = TempDb::new("oversized");
    let (channel, _) = two_column_channel();
    let key = su3_key(super::super::DISK_CHANNEL, [[1, 0], [1, 0], [0, 1]]);
    let row_len = key.len() + channel.payload_len().unwrap() + 8;
    let exact = store(&db, row_len as u64);
    enqueue(&exact, key.clone(), &channel);
    exact.write(&take_pending());
    assert!(
        has_key(&exact, &key),
        "a row of exactly the bound is stored"
    );
    drop(exact);
    let db = TempDb::new("oversized-2");
    let tight = store(&db, row_len as u64 - 1);
    enqueue(&tight, key.clone(), &channel);
    assert_eq!(pending_rows(), 0);
    assert_eq!(tight.counters.snapshot().evictions, 1);
    assert_eq!(
        tight.counters.snapshot().bytes_written,
        0,
        "refused, not written then trimmed"
    );
    assert!(!has_key(&tight, &key));
}

#[test]
fn pending_is_bounded_independent_of_request_size() {
    let (channel, _) = two_column_channel();
    let row_len = |key: &[u8]| key.len() + channel.payload_len().unwrap() + 8;
    let chunk = 3 * row_len(&su3_key(1, [[0, 0], [0, 0], [0, i64::from(0u8)]]));
    CHUNK.with(|c| c.set(Some(chunk)));
    for n in [8i64, 16] {
        let db = TempDb::new("pending");
        let s = store(&db, 1 << 20);
        PEAK.with(|p| p.set(0));
        for i in 0..n {
            enqueue(&s, su3_key(1, [[i, 0], [0, 0], [0, 0]]), &channel);
        }
        s.write(&take_pending());
        assert!(PEAK.with(Cell::get) <= chunk, "{n} rows");
        let stored: i64 = s
            .lock()
            .query_row("SELECT count(*) FROM entries", [], |r| r.get(0))
            .unwrap();
        assert_eq!(stored, n);
    }
    // A row larger than the queue is written through, never queued.
    CHUNK.with(|c| c.set(Some(10)));
    let db = TempDb::new("through");
    let s = store(&db, 1 << 20);
    let key = su3_key(1, [[9, 9], [0, 0], [0, 0]]);
    PEAK.with(|p| p.set(0));
    enqueue(&s, key.clone(), &channel);
    assert_eq!((pending_rows(), PEAK.with(Cell::get)), (0, 0));
    assert!(has_key(&s, &key));
    CHUNK.with(|c| c.set(None));
}

#[test]
fn stale_owner_goes_inert_on_flush() {
    let db = TempDb::new("stale");
    let old = Identity {
        minor: Identity::CURRENT.minor + 1,
        ..Identity::CURRENT
    };
    let a = Store::open(db.path(), 1 << 20, old).unwrap();
    let b = store(&db, 1 << 20);
    b.write(&[raw(2, 10)]);
    a.write(&[raw(1, 10)]);
    assert!(a.inert.load(Ordering::Relaxed));
    assert_eq!(a.counters.snapshot().failures, 1);
    assert!(!has_key(&b, &[1; 16]), "the stale owner inserted nothing");
    assert!(has_key(&b, &[2; 16]));
    assert_eq!(owner(&b.lock()).unwrap(), Some(Identity::CURRENT));
    assert_eq!(meta_bytes(&b), (26, 26));
}

#[test]
fn setup_rejects_memory_and_empty_paths() {
    for path in ["", ":memory:"] {
        assert!(matches!(
            Store::open(Path::new(path), 1, Identity::CURRENT),
            Err(DiskCacheError::Open { .. })
        ));
    }
}

#[test]
fn setup_requires_wal() {
    let db = TempDb::new("wal");
    FAULT_WAL_MODE.with(|f| f.set(true));
    let refused = Store::open(db.path(), 1, Identity::CURRENT);
    FAULT_WAL_MODE.with(|f| f.set(false));
    assert!(matches!(refused, Err(DiskCacheError::Open { .. })));
}

#[test]
fn concurrent_first_creation_yields_one_store() {
    let db = TempDb::new("create");
    let barrier = std::sync::Barrier::new(2);
    let stores: Vec<Store> = std::thread::scope(|s| {
        let open = || {
            // This test is about one creation winning, not about how long the
            // loser may wait for a slow creator: give it ample lock-wait time.
            SETUP_BUDGET_OVERRIDE.with(|d| d.set(Some(Duration::from_secs(60))));
            barrier.wait();
            store(&db, 1 << 20)
        };
        let (x, y) = (s.spawn(open), s.spawn(open));
        vec![x.join().unwrap(), y.join().unwrap()]
    });
    stores[0].write(&[raw(1, 10)]);
    stores[1].write(&[raw(2, 10)]);
    let metas: i64 = stores[0]
        .lock()
        .query_row("SELECT count(*) FROM meta", [], |r| r.get(0))
        .unwrap();
    assert_eq!(metas, 1);
    assert_eq!(meta_bytes(&stores[1]), (52, 52));
}

/// A file another connection keeps write-locked: setup gives up once its one
/// lock-wait budget is spent, and its sleeps never add up to more.
#[test]
fn locked_setup_shares_one_wait_budget() {
    let db = TempDb::new("locked");
    drop(store(&db, 1 << 20));
    let holder = Connection::open(db.path()).unwrap();
    holder.execute_batch("BEGIN IMMEDIATE").unwrap();
    let budget = Duration::from_millis(60);
    SETUP_BUDGET_OVERRIDE.with(|d| d.set(Some(budget)));
    SETUP_WAITS.with(|w| w.borrow_mut().clear());
    let refused = Store::open(db.path(), 1 << 20, Identity::CURRENT);
    SETUP_BUDGET_OVERRIDE.with(|d| d.set(None));
    assert!(matches!(refused, Err(DiskCacheError::Open { .. })));
    let waits = SETUP_WAITS.with(|w| w.take());
    assert!(!waits.is_empty());
    assert!(waits.iter().sum::<Duration>() <= budget, "{waits:?}");
    holder.execute_batch("ROLLBACK").unwrap();
    let ok = store(&db, 1 << 20);
    assert_eq!(owner(&ok.lock()).unwrap(), Some(Identity::CURRENT));
}

/// Setup work that outlasts the whole lock-wait budget (slow file creation
/// or fsync on a loaded machine) is not a lock wait: an uncontended file
/// still opens.
#[test]
fn slow_uncontended_setup_still_opens() {
    let db = TempDb::new("slow");
    SETUP_BUDGET_OVERRIDE.with(|d| d.set(Some(Duration::from_millis(20))));
    SLOW_SETUP.with(|s| s.set(Some(Duration::from_millis(100))));
    let opened = Store::open(db.path(), 1 << 20, Identity::CURRENT);
    SLOW_SETUP.with(|s| s.set(None));
    SETUP_BUDGET_OVERRIDE.with(|d| d.set(None));
    let opened = opened.unwrap();
    assert_eq!(owner(&opened.lock()).unwrap(), Some(Identity::CURRENT));
}
