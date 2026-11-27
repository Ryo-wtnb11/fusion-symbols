//! Codec and store contracts on local files, plus the harness the hook tests
//! in `lie::general::cgc::disk_tests` share. Expected values are inline:
//! hand-built rows whose shape is stated next to them.

use super::*;
use crate::exact::radical::{RadicalBlock, Radicand, SignedInts};
use malachite_nz::natural::Natural;
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

    /// Removes every stored row of `kind`, through a separate connection.
    pub(crate) fn delete_kind(&self, kind: u8) {
        let conn = Connection::open(&self.0).unwrap();
        conn.execute(
            "DELETE FROM entries WHERE substr(key, 7, 1) = ?1",
            (&[kind][..],),
        )
        .unwrap();
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

/// A radicand part as stored: `u32` word count, then the words.
fn part(words: &[u64]) -> Vec<u8> {
    let mut out = (words.len() as u32).to_le_bytes().to_vec();
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// The radicand `num/den` of one-word parts, as stored.
fn radicand(num: u64, den: u64) -> Vec<u8> {
    let word = |x: u64| if x == 0 { part(&[]) } else { part(&[x]) };
    [word(num), word(den)].concat()
}

/// One μ of a stored channel: its slab of `(negative, |X|)` and its radicand.
fn mu(x: &[(bool, u64)], rho: Vec<u8>) -> (SignedInts, Vec<u8>) {
    (SignedInts::from_words(x).unwrap(), rho)
}

/// The channel layout, written out field by field: `u32 mult`, `3 × u32`
/// dims, `u32 count`, the pairs, the column ends, then per μ the slab and
/// the radicand.
fn exact_row(
    mult: u32,
    dims: [u32; 3],
    pairs: &[(u32, u32)],
    ends: &[u32],
    mus: &[(SignedInts, Vec<u8>)],
) -> Vec<u8> {
    let mut out = Vec::new();
    for x in [mult, dims[0], dims[1], dims[2], pairs.len() as u32] {
        out.extend_from_slice(&x.to_le_bytes());
    }
    for &(a, b) in pairs {
        out.extend_from_slice(&a.to_le_bytes());
        out.extend_from_slice(&b.to_le_bytes());
    }
    for e in ends {
        out.extend_from_slice(&e.to_le_bytes());
    }
    for (x, r) in mus {
        x.encode(&mut out);
        out.extend_from_slice(r);
    }
    out
}

/// The seeds layout: a channel row without its dims, one column per μ.
fn seeds_row(
    mult: u32,
    pairs: &[(u32, u32)],
    ends: &[u32],
    mus: &[(SignedInts, Vec<u8>)],
) -> Vec<u8> {
    let row = exact_row(mult, [0; 3], pairs, ends, mus);
    [&row[..4], &row[16..]].concat()
}

const PAIRS: [(u32, u32); 4] = [(0, 1), (1, 0), (0, 1), (1, 0)];

/// `2 ⊗ 2 → 1` with `N = 2`: columns `(μ, m_c) = (0, 0), (1, 0)` of two
/// entries each, `X = (1, −1)` with `ρ_0 = 1/2` and `X = (2, 3)` with
/// `ρ_1 = 1/13`.
fn good_mus() -> Vec<(SignedInts, Vec<u8>)> {
    vec![
        mu(&[(false, 1), (true, 1)], radicand(1, 2)),
        mu(&[(false, 2), (false, 3)], radicand(1, 13)),
    ]
}

fn two_column_row() -> Vec<u8> {
    exact_row(2, [2, 2, 1], &PAIRS, &[2, 4], &good_mus())
}

fn two_column_channel() -> (Arc<ExactCgc>, Shape) {
    let shape = Shape::Channel {
        dims: [2, 2, 1],
        mult: 2,
    };
    let channel = ExactCgc::decode(&two_column_row(), [2, 2, 1], 2).unwrap();
    (Arc::new(channel), shape)
}

/// The norm-table layout, written out field by field.
fn norm_row(dim: u32, norms: &[(u64, u64)], top: u32) -> Vec<u8> {
    let mut out = dim.to_le_bytes().to_vec();
    for &(n, d) in norms {
        out.extend(radicand(n, d));
    }
    out.extend_from_slice(&top.to_le_bytes());
    out
}

/// A three-dimensional table: norms `1, 1/2, 3/2`.
fn good_norms() -> Vec<u8> {
    norm_row(3, &[(1, 1), (1, 2), (3, 2)], 0)
}

fn rad(n: u32, d: u32) -> Radicand {
    Radicand::new(Natural::from(n), Natural::from(d)).unwrap()
}

/// `[1, 2, 1, 2]`: `S = (1, −2, 0, 3)`, axes 1 and 3 stored.
fn exact_f() -> Arc<RadicalBlock<4>> {
    let nums = SignedInts::from_words(&[(false, 1), (true, 2), (false, 0), (false, 3)]).unwrap();
    let two = || -> Box<[Radicand]> { vec![rad(1, 3), rad(2, 3)].into() };
    Arc::new(
        RadicalBlock::new(
            [1, 2, 1, 2],
            nums,
            [Box::default(), two(), Box::default(), two()],
            rad(1, 2),
        )
        .unwrap(),
    )
}

/// `2 × 2`: `S = (−2, 3, 3, 2)` over `ρ = 1/49` per leg pair, both axes
/// `2 × 2` orthogonal (it passes gate 6): `S = (−3, 4, 4, 3)` over
/// `common = 1/25`, both axes stored.
fn exact_r() -> Arc<RadicalBlock<2>> {
    let nums = SignedInts::from_words(&[(true, 3), (false, 4), (false, 4), (false, 3)]).unwrap();
    let axis = || -> Box<[Radicand]> { vec![rad(1, 1), rad(1, 1)].into() };
    Arc::new(RadicalBlock::new([2, 2], nums, [axis(), axis()], rad(1, 25)).unwrap())
}

#[test]
fn each_kind_round_trips_bit_identically() {
    let (channel, shape) = two_column_channel();
    assert_eq!(payload(&channel), two_column_row(), "the stated layout");
    let back = Arc::<ExactCgc>::decode(&payload(&channel), &shape).unwrap();
    assert_eq!(back.mult(), 2);
    assert!(*back == *channel);

    let seeds = seeds_row(2, &PAIRS, &[2, 4], &good_mus());
    let seed_shape = Shape::Seeds {
        dims: [2, 2],
        mult: 2,
    };
    let s = Arc::<HighestColumn>::decode(&seeds, &seed_shape).unwrap();
    assert_eq!(payload(&s), seeds);

    let norms = Arc::<NormTable>::decode(&good_norms(), &Shape::Norms { dim: 3 }).unwrap();
    assert_eq!(payload(&norms), good_norms());

    for k in [1i8, -1] {
        assert_eq!(i8::decode(&payload(&k), &Shape::Phase), Some(k));
    }

    let f = exact_f();
    let back = Arc::<RadicalBlock<4>>::decode(&payload(&f), &Shape::F([1, 2, 1, 2])).unwrap();
    assert_eq!(back, f);
    let empty = Arc::new(RadicalBlock::<4>::zeros([0, 2, 1, 1]).unwrap());
    let back = Arc::<RadicalBlock<4>>::decode(&payload(&empty), &Shape::F([0, 2, 1, 1]));
    assert_eq!(back, Some(empty));
    let r = exact_r();
    let back = Arc::<RadicalBlock<2>>::decode(&payload(&r), &Shape::R(2)).unwrap();
    assert_eq!(back, r);
}

#[test]
fn transplanted_row_is_rejected() {
    let (channel, shape) = two_column_channel();
    let a = su3_key(super::super::DISK_CHANNEL, [[1, 0], [1, 0], [0, 1]]);
    let b = su3_key(super::super::DISK_CHANNEL, [[0, 1], [0, 1], [1, 0]]);
    let stored = row(&a, &payload(&channel));
    assert!(decode_row::<Arc<ExactCgc>>(&a, &stored, &shape).is_some());
    assert!(decode_row::<Arc<ExactCgc>>(&b, &stored, &shape).is_none());
}

/// The key has 15 + 48 bytes and the payload 172, so the message ends in a
/// 3-byte tail: the top bytes of the last radicand's denominator word.
#[test]
fn tail_bit_flip_is_caught_only_by_checksum() {
    let (channel, shape) = two_column_channel();
    let key = su3_key(super::super::DISK_CHANNEL, [[1, 0], [1, 0], [0, 1]]);
    let mut stored = row(&key, &payload(&channel));
    assert_eq!((key.len() + stored.len() - 8) % 8, 3);
    let last = stored.len() - 8 - 1;
    stored[last] ^= 0x10;
    let (body, _) = stored.split_at(stored.len() - 8);
    assert!(
        Arc::<ExactCgc>::decode(body, &shape).is_some(),
        "structurally valid"
    );
    assert!(decode_row::<Arc<ExactCgc>>(&key, &stored, &shape).is_none());
}

#[test]
fn truncated_and_overlong_values_are_rejected() {
    let (channel, shape) = two_column_channel();
    let key = su3_key(super::super::DISK_CHANNEL, [[1, 0], [1, 0], [0, 1]]);
    let stored = row(&key, &payload(&channel));
    for len in 0..stored.len() {
        assert!(decode_row::<Arc<ExactCgc>>(&key, &stored[..len], &shape).is_none());
    }
    let mut long = payload(&channel);
    long.push(0);
    assert!(decode_row::<Arc<ExactCgc>>(&key, &row(&key, &long), &shape).is_none());
    let norms = Shape::Norms { dim: 3 };
    let good = good_norms();
    for len in 0..good.len() {
        assert!(
            Arc::<NormTable>::decode(&good[..len], &norms).is_none(),
            "a truncated table, {len} bytes"
        );
    }
}

/// Every row below has a valid checksum and changes one field of the good
/// channel, seeds or norm table, so each structural check is reached.
#[test]
fn structurally_wrong_rows_with_valid_checksum_are_rejected() {
    let (_, shape) = two_column_channel();
    let key = su3_key(super::super::DISK_CHANNEL, [[1, 0], [1, 0], [0, 1]]);
    let good = good_mus;
    let one_mu = |x: &[(bool, u64)]| {
        let mut m = good_mus();
        m[1] = mu(x, radicand(1, 13));
        m
    };
    let row2 = |pairs: &[(u32, u32)], ends: &[u32], mus: &[(SignedInts, Vec<u8>)]| {
        exact_row(2, [2, 2, 1], pairs, ends, mus)
    };
    let swapped = [(1, 0), (0, 1), (0, 1), (1, 0)];
    let duplicate = [(0, 1), (0, 1), (0, 1), (1, 0)];
    let wide_a = [(0, 1), (2, 0), (0, 1), (1, 0)];
    let wide_b = [(0, 1), (1, 2), (0, 1), (1, 0)];
    let cases = [
        (
            "stored N",
            exact_row(3, [2, 2, 1], &PAIRS, &[2, 4], &good()),
        ),
        ("dims", exact_row(2, [2, 3, 1], &PAIRS, &[2, 4], &good())),
        ("m_a out of range", row2(&wide_a, &[2, 4], &good())),
        ("m_b out of range", row2(&wide_b, &[2, 4], &good())),
        ("unsorted", row2(&swapped, &[2, 4], &good())),
        ("duplicate", row2(&duplicate, &[2, 4], &good())),
        // The existing negative control: a column with no entry.
        ("missing column", row2(&PAIRS, &[4, 4], &good())),
        ("decreasing ends", row2(&PAIRS, &[3, 2], &good())),
        ("last end short", row2(&PAIRS, &[1, 3], &good())),
        ("end past count", row2(&PAIRS, &[2, 5], &good())),
        ("empty with N = 2", row2(&[], &[], &[])),
        (
            "slab too short",
            row2(&PAIRS, &[2, 4], &one_mu(&[(false, 2)])),
        ),
        (
            "slab too long",
            row2(
                &PAIRS,
                &[2, 4],
                &one_mu(&[(false, 2), (false, 3), (false, 1)]),
            ),
        ),
        (
            "zero coordinate",
            row2(&PAIRS, &[2, 4], &one_mu(&[(false, 2), (false, 0)])),
        ),
        (
            "zero radicand",
            row2(
                &PAIRS,
                &[2, 4],
                &[
                    good().remove(0),
                    mu(&[(false, 2), (false, 3)], radicand(0, 13)),
                ],
            ),
        ),
        ("a missing μ", row2(&PAIRS, &[2, 4], &good()[..1])),
    ];
    for (name, bytes) in cases {
        let stored = row(&key, &bytes);
        assert!(
            decode_row::<Arc<ExactCgc>>(&key, &stored, &shape).is_none(),
            "{name}"
        );
    }
    let stored = row(&key, &two_column_row());
    let n1 = Shape::Channel {
        dims: [2, 2, 1],
        mult: 1,
    };
    assert!(
        decode_row::<Arc<ExactCgc>>(&key, &stored, &n1).is_none(),
        "N mismatch"
    );
    let zero = Shape::Channel {
        dims: [2, 2, 1],
        mult: 0,
    };
    assert!(
        decode_row::<Arc<ExactCgc>>(
            &key,
            &row(&key, &exact_row(0, [2, 2, 1], &[], &[], &[])),
            &zero
        )
        .is_some(),
        "the empty channel"
    );
    let row_in_zero = row(&key, &exact_row(0, [2, 2, 1], &PAIRS[..1], &[], &[]));
    assert!(
        decode_row::<Arc<ExactCgc>>(&key, &row_in_zero, &zero).is_none(),
        "zero channel"
    );

    let seed_shape = Shape::Seeds {
        dims: [2, 2],
        mult: 2,
    };
    let seeds = |mult, pairs: &[(u32, u32)], ends: &[u32]| {
        row(&key, &seeds_row(mult, pairs, ends, &good()))
    };
    assert!(
        decode_row::<Arc<HighestColumn>>(&key, &seeds(2, &PAIRS, &[2, 4]), &seed_shape).is_some()
    );
    for (name, stored) in [
        ("seeds N", seeds(1, &PAIRS, &[2, 4])),
        ("seeds m_a", seeds(2, &wide_a, &[2, 4])),
        ("empty seed", seeds(2, &PAIRS, &[0, 4])),
    ] {
        assert!(
            decode_row::<Arc<HighestColumn>>(&key, &stored, &seed_shape).is_none(),
            "{name}"
        );
    }

    let norms = Shape::Norms { dim: 3 };
    let table = |norms: &[(u64, u64)], top| row(&key, &norm_row(3, norms, top));
    let n = [(1, 1), (1, 2), (3, 2)];
    // The derived fields are those of the norms: `Δ = 2`, `g = 1`, so
    // `ν = (2, 1, 3)`.
    let decoded = decode_row::<Arc<NormTable>>(&key, &table(&n, 0), &norms).unwrap();
    let want = SignedInts::from_words(&[(false, 2), (false, 1), (false, 3)]).unwrap();
    assert_eq!(
        (decoded.nu(), decoded.den(), decoded.content()),
        (&want, &Natural::from(2u32), &Natural::from(1u32))
    );
    assert!(
        decode_row::<Arc<NormTable>>(&key, &row(&key, &good_norms()), &Shape::Norms { dim: 4 })
            .is_none(),
        "dim"
    );
    for (name, stored) in [
        ("zero norm", table(&[(1, 1), (0, 2), (3, 2)], 0)),
        ("top past dim", table(&n, 3)),
    ] {
        assert!(
            decode_row::<Arc<NormTable>>(&key, &stored, &norms).is_none(),
            "{name}"
        );
    }

    let f_row = row(&key, &payload(&exact_f()));
    assert!(
        decode_row::<Arc<RadicalBlock<4>>>(&key, &f_row, &Shape::F([1, 2, 2, 1])).is_none(),
        "F dims"
    );
    assert!(
        decode_row::<Arc<RadicalBlock<4>>>(&key, &f_row, &Shape::R(2)).is_none(),
        "F row as R"
    );
    // Dims alone, of a product past `usize` or of 2^64 entries: refused
    // before any slab is read.
    for (name, d) in [("F overflow", u32::MAX), ("F wrap", 1 << 16)] {
        let dims: Vec<u8> = (0..4).flat_map(|_| d.to_le_bytes()).collect();
        assert!(
            decode_row::<Arc<RadicalBlock<4>>>(&key, &row(&key, &dims), &Shape::F([d as usize; 4]))
                .is_none(),
            "{name}"
        );
    }
    let r_row = row(&key, &payload(&exact_r()));
    assert!(
        decode_row::<Arc<RadicalBlock<2>>>(&key, &r_row, &Shape::R(3)).is_none(),
        "R n"
    );
    let mut r_header = payload(&exact_r());
    r_header[0] = 3;
    assert!(
        decode_row::<Arc<RadicalBlock<2>>>(&key, &row(&key, &r_header), &Shape::R(2)).is_none(),
        "stored R n"
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

/// A structurally valid exact R row whose view fails gate 6 (`[1, 1]`,
/// `S = 1`, `common = 1/4`, so `R = 1/2`: residual `3/4`) is rejected like a
/// malformed one, and so regenerated; the valid rows decode.
#[test]
fn gate_failing_exact_rows_are_rejected() {
    let bad = RadicalBlock::<2>::new(
        [1, 1],
        SignedInts::from_words(&[(false, 1)]).unwrap(),
        [Box::default(), Box::default()],
        rad(1, 4),
    )
    .unwrap();
    let mut payload = Vec::new();
    bad.encode(&mut payload);
    assert!(<Arc<RadicalBlock<2>> as Persist>::decode(&payload, &Shape::R(1)).is_none());
    let good = exact_r();
    let mut payload = Vec::new();
    good.encode(&mut payload);
    assert_eq!(
        <Arc<RadicalBlock<2>> as Persist>::decode(&payload, &Shape::R(2)),
        Some(good)
    );
}
