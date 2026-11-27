//! The opt-in persistent tier: one SQLite file that serves generated
//! coefficients to later processes.
//!
//! A miss of a persisted kind reads its row before generating
//! ([`through`]), inside the claiming leader's compute, so a warm in-memory
//! hit never reaches this module and only one thread per key reads the disk.
//! A loaded value is published to the memory tier like a generated one.
//!
//! **Identity.** A row's key is the storage schema, this crate's
//! major.minor, the kind, the presented root and the labels in in-memory key
//! order. A file holds one identity; enabling another one recreates it.
//!
//! **Contents.** CGC data is stored exactly: exact channels, highest-weight
//! seeds and norm tables, never a rounded channel. A loaded exact channel is
//! rounded into its view as after a generation. F and R rows keep their
//! rounded blocks, their only authority until exact F and R.
//!
//! **Integrity.** A row is used only if its checksum (over key and payload)
//! matches and its payload has exactly the shape its labels determine
//! ([`Shape`]), checked by structural decoders before any arithmetic or
//! indexing: Weyl dimensions and multiplicities, every CGC index in range
//! and in the order `LieCgc::get` searches, every multiplicity column
//! present, canonical integer slabs with nonzero coordinates and positive
//! radicands, complete norm tables, no trailing bytes, finite F and R values
//! and the F and R generation gates. The CGC isometry and ladder gates are
//! not re-run on a loaded channel: they need the irrep modules that
//! persistence exists to avoid, and no channel row is written before its
//! view passed them, so a row is trusted to be what a gated generation of
//! this crate wrote. The checksum detects later accidental damage, not a
//! faulty writer or an adversary.
//!
//! **Writes.** Generated rows are encoded at once and queued per thread; the
//! queue holds at most [`PENDING_MAX_BYTES`] (or the store bound, if smaller)
//! and [`PENDING_MAX_ENTRIES`] rows. A full queue, a row larger than the
//! queue, and the end of each outermost generation commit one transaction.
//! An unwinding generation drops its queued rows without I/O; rows already
//! committed stay.
//!
//! **Bound.** The stored key and value bytes are at most the `max_bytes` of
//! the process that last committed: oldest rows are removed first, at enable
//! and in every write transaction. File pages, the free list and the WAL are
//! outside that bound.
//!
//! No disk failure fails a coefficient call: a failed read is a miss, a
//! failed write is skipped, and both are counted.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior};

use super::fingerprint::{fold, root_code, Label, P1, P2, P3};
use super::Shape;
use crate::category::{FBlock, RBlock, SymbolBlock};
use crate::lie::general::cgc::exact::{ExactCgc, HighestColumn, NormTable};
use crate::lie::RootSystem;
use crate::rounding::blocks::{check_f_block, check_r_block};

/// Storage format version: the key and value encodings and the table layout.
/// Schema 2 stores exact CGC channels, seeds and norm tables.
const SCHEMA: u16 = 2;

const fn parse_u16(s: &str) -> u16 {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut value = 0u16;
    while i < bytes.len() {
        value = value * 10 + (bytes[i] - b'0') as u16;
        i += 1;
    }
    value
}

/// The identity a file is owned by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Identity {
    schema: u16,
    major: u16,
    minor: u16,
}

impl Identity {
    const CURRENT: Self = Self {
        schema: SCHEMA,
        major: parse_u16(env!("CARGO_PKG_VERSION_MAJOR")),
        minor: parse_u16(env!("CARGO_PKG_VERSION_MINOR")),
    };
}

/// Private batching limits of the per-thread write queue. They bound its
/// memory and amortize commits; they do not bound the WAL.
const PENDING_MAX_BYTES: usize = 4 << 20;
const PENDING_MAX_ENTRIES: usize = 1024;

/// How long one SQLite call waits for another connection's lock.
const BUSY_TIMEOUT: Duration = Duration::from_millis(1000);
/// The total time `enable_disk_cache` may spend waiting for other
/// connections' locks, across every statement and retry of its setup. File
/// creation and I/O are not waits and are not charged.
const SETUP_WAIT_BUDGET: Duration = Duration::from_millis(1000);
/// Longest single sleep of a setup lock wait.
const SETUP_MAX_STEP: Duration = Duration::from_millis(50);
/// WAL truncation target after a checkpoint; not a bound on the live WAL.
const JOURNAL_SIZE_LIMIT: i64 = 64 << 20;

/// Checksum seed (`"fs-disk1"`), a format constant.
const SEED: u64 = 0x6673_2d64_6973_6b31;

const META: &str = "CREATE TABLE IF NOT EXISTS meta (
    id INTEGER PRIMARY KEY CHECK (id = 0),
    schema INTEGER NOT NULL, major INTEGER NOT NULL, minor INTEGER NOT NULL,
    bytes INTEGER NOT NULL)";

const RECREATE: &str = "DROP TABLE IF EXISTS entries;
CREATE TABLE entries (id INTEGER PRIMARY KEY, key BLOB NOT NULL UNIQUE, value BLOB NOT NULL);
CREATE TRIGGER entries_insert AFTER INSERT ON entries BEGIN
    UPDATE meta SET bytes = bytes + length(NEW.key) + length(NEW.value) WHERE id = 0; END;
CREATE TRIGGER entries_delete AFTER DELETE ON entries BEGIN
    UPDATE meta SET bytes = bytes - length(OLD.key) - length(OLD.value) WHERE id = 0; END;
CREATE TRIGGER entries_update AFTER UPDATE OF value ON entries BEGIN
    UPDATE meta SET bytes = bytes + length(NEW.value) - length(OLD.value) WHERE id = 0; END;";

const UPSERT: &str = "INSERT INTO entries (key, value) VALUES (?1, ?2)
    ON CONFLICT (key) DO UPDATE SET value = excluded.value";

/// This process's disk-cache counters, from [`disk_cache_stats`]. They count
/// lookups and writes of this process only and read no file.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskCacheStats {
    /// Rows loaded instead of generated. Each is also one publication in
    /// [`super::TierStats::misses`] of its tier.
    pub hits: u64,
    /// Lookups that found no row and generated.
    pub misses: u64,
    /// Rows that failed their checksum or shape checks; the value was
    /// generated and the row is overwritten.
    pub rejected: u64,
    /// Failed reads and writes (including rows dropped because another
    /// connection held the write lock beyond the busy timeout, or because the
    /// file was taken over by another version).
    pub failures: u64,
    /// Rows removed by this process to keep the bound, plus values not
    /// stored because a row would exceed the bound.
    pub evictions: u64,
    /// Row bytes loaded.
    pub bytes_read: u64,
    /// Row bytes committed.
    pub bytes_written: u64,
}

#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    rejected: AtomicU64,
    failures: AtomicU64,
    evictions: AtomicU64,
    bytes_read: AtomicU64,
    bytes_written: AtomicU64,
}

impl Counters {
    fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Ordering::Relaxed);
    }

    fn snapshot(&self) -> DiskCacheStats {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        DiskCacheStats {
            hits: get(&self.hits),
            misses: get(&self.misses),
            rejected: get(&self.rejected),
            failures: get(&self.failures),
            evictions: get(&self.evictions),
            bytes_read: get(&self.bytes_read),
            bytes_written: get(&self.bytes_written),
        }
    }
}

/// Why the disk cache could not be enabled or cleared.
#[non_exhaustive]
#[derive(Debug)]
pub enum DiskCacheError {
    /// [`enable_disk_cache`] already succeeded in this process.
    AlreadyEnabled,
    /// [`clear_disk_cache`] before [`enable_disk_cache`].
    NotEnabled,
    /// The path is not a database file this crate can own: empty or
    /// `:memory:`, not openable or creatable, WAL mode not granted, or still
    /// locked after setup spent its lock-wait budget. Nothing was installed.
    Open {
        /// The path as given.
        path: PathBuf,
        /// The underlying cause.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Removing the rows failed; nothing changed.
    Clear {
        /// The underlying cause.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The rows were removed, but compacting the file failed.
    Compact {
        /// The underlying cause.
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl std::fmt::Display for DiskCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyEnabled => f.write_str("the disk cache is already enabled"),
            Self::NotEnabled => f.write_str("the disk cache is not enabled"),
            Self::Open { path, source } => {
                write!(
                    f,
                    "cannot use {} as the disk cache: {source}",
                    path.display()
                )
            }
            Self::Clear { source } => write!(f, "cannot clear the disk cache: {source}"),
            Self::Compact { source } => {
                write!(f, "cleared the disk cache but cannot compact it: {source}")
            }
        }
    }
}

impl std::error::Error for DiskCacheError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open { source, .. } | Self::Clear { source } | Self::Compact { source } => {
                Some(source.as_ref())
            }
            Self::AlreadyEnabled | Self::NotEnabled => None,
        }
    }
}

/// A cause of this crate's own.
#[derive(Debug)]
struct Reason(&'static str);

impl std::fmt::Display for Reason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for Reason {}

static STORE: OnceLock<Store> = OnceLock::new();
static ENABLE: Mutex<()> = Mutex::new(());

/// Enables the disk cache for this process at the database file `path`,
/// keeping at most `max_bytes` of stored key and value bytes.
///
/// From then on, a miss of a Cartan-engine CGC channel, Frobenius–Schur
/// phase, F or R block reads its row before generating, and generated values
/// are written back. Exact SU(2) symbols, modules and products are never
/// stored. Rows written by another fusion-symbols major.minor are removed:
/// a file holds one version, so give each version its own path.
///
/// `path` is a literal file system path (no `file:` URIs). Its directory
/// must exist; the file is created if absent. Use a node-local file system:
/// SQLite's WAL mode needs shared memory, which network file systems do not
/// provide across hosts. Builds from unreleased sources should use a
/// disposable path, since only releases fix the values behind a version.
///
/// Like [`super::reset`], this is a single-owner, application-level policy;
/// a library must not call it. It succeeds at most once per process; a
/// failed attempt installs nothing and may be retried.
///
/// # Errors
///
/// [`DiskCacheError::AlreadyEnabled`] after a success, and
/// [`DiskCacheError::Open`] when the file cannot be opened as a WAL database
/// while other connections hold its locks for the whole one-second setup
/// lock-wait budget (time spent creating or syncing the file is not charged).
///
/// # Examples
///
/// ```
/// use fusion_symbols::cache::{disk_cache_stats, enable_disk_cache};
///
/// let path = std::env::temp_dir().join(format!("fs-doc-{}.sqlite", std::process::id()));
/// enable_disk_cache(&path, 64 << 20).unwrap();
/// assert_eq!(disk_cache_stats().hits, 0);
/// # for suffix in ["", "-wal", "-shm"] {
/// #     let _ = std::fs::remove_file(format!("{}{suffix}", path.display()));
/// # }
/// ```
pub fn enable_disk_cache(path: impl AsRef<Path>, max_bytes: u64) -> Result<(), DiskCacheError> {
    let _serial = ENABLE.lock().unwrap_or_else(PoisonError::into_inner);
    if STORE.get().is_some() {
        return Err(DiskCacheError::AlreadyEnabled);
    }
    let store = Store::open(path.as_ref(), max_bytes, Identity::CURRENT)?;
    // Under `ENABLE`, after the emptiness check: this set cannot fail.
    let _ = STORE.set(store);
    Ok(())
}

/// Removes every stored row, then compacts the file.
///
/// Rows still queued in a running generation, and rows other processes write
/// afterwards, may repopulate the file. The two steps are not atomic: when
/// compaction fails the rows are already gone.
///
/// # Errors
///
/// [`DiskCacheError::NotEnabled`] before [`enable_disk_cache`];
/// [`DiskCacheError::Clear`] when nothing could be removed (including a file
/// another version has taken over); [`DiskCacheError::Compact`] when only
/// compaction failed.
pub fn clear_disk_cache() -> Result<(), DiskCacheError> {
    STORE.get().ok_or(DiskCacheError::NotEnabled)?.clear()
}

/// This process's disk-cache counters; all zero before
/// [`enable_disk_cache`].
pub fn disk_cache_stats() -> DiskCacheStats {
    STORE
        .get()
        .map_or_else(DiskCacheStats::default, |s| s.counters.snapshot())
}

/// One open database file.
pub(super) struct Store {
    // ponytail: one connection per process; parallel cold readers serialize
    // on it. Per-thread connections if a prefill-team measurement shows
    // contention.
    conn: Mutex<Connection>,
    max_bytes: u64,
    identity: Identity,
    /// Set when another version took the file over: neither read nor write.
    inert: AtomicBool,
    counters: Counters,
}

/// Lock-wait time left to one `enable_disk_cache` setup, shared by SQLite's
/// busy handler and the retry loop. Only time spent sleeping for a lock is
/// charged, so a slow but uncontended setup is never reported as locked.
struct WaitBudget {
    left_nanos: AtomicU64,
}

impl WaitBudget {
    fn new(budget: Duration) -> Self {
        Self {
            left_nanos: AtomicU64::new(u64::try_from(budget.as_nanos()).unwrap_or(u64::MAX)),
        }
    }

    /// Sleeps the `attempt`-th lock wait (1 ms doubling to
    /// [`SETUP_MAX_STEP`], never past the budget) and charges the time
    /// slept; `false`, without sleeping, once the budget is spent.
    fn wait(&self, attempt: usize) -> bool {
        let left = Duration::from_nanos(self.left_nanos.load(Ordering::Relaxed));
        if left.is_zero() {
            return false;
        }
        let step = Duration::from_millis(1u64 << attempt.min(6))
            .min(SETUP_MAX_STEP)
            .min(left);
        #[cfg(test)]
        tests::SETUP_WAITS.with(|w| w.borrow_mut().push(step));
        let start = Instant::now();
        std::thread::sleep(step);
        let slept = u64::try_from(start.elapsed().max(step).as_nanos()).unwrap_or(u64::MAX);
        // One setup at a time uses a budget: its retry loop and the busy
        // handler of its current connection run on the calling thread.
        let left = u64::try_from(left.as_nanos()).unwrap_or(u64::MAX);
        self.left_nanos
            .store(left.saturating_sub(slept), Ordering::Relaxed);
        true
    }
}

enum SetupError {
    /// Locked by another connection; worth retrying.
    Busy(Box<dyn std::error::Error + Send + Sync>),
    Fatal(Box<dyn std::error::Error + Send + Sync>),
}

impl From<rusqlite::Error> for SetupError {
    fn from(e: rusqlite::Error) -> Self {
        match e.sqlite_error_code() {
            Some(rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked) => {
                Self::Busy(Box::new(e))
            }
            _ => Self::Fatal(Box::new(e)),
        }
    }
}

enum Read {
    Row(Vec<u8>),
    Absent,
    Failed,
}

enum Written {
    Committed(u64),
    TakenOver,
}

impl Store {
    fn open(path: &Path, max_bytes: u64, identity: Identity) -> Result<Self, DiskCacheError> {
        let open_error = |source| DiskCacheError::Open {
            path: path.to_path_buf(),
            source,
        };
        if path.as_os_str().is_empty() || path == Path::new(":memory:") {
            return Err(open_error(Box::new(Reason("not a database file path"))));
        }
        #[cfg(test)]
        let budget = tests::SETUP_BUDGET_OVERRIDE
            .with(std::cell::Cell::get)
            .unwrap_or(SETUP_WAIT_BUDGET);
        #[cfg(not(test))]
        let budget = SETUP_WAIT_BUDGET;
        let waits = Arc::new(WaitBudget::new(budget));
        let mut attempt = 0;
        loop {
            match Self::try_open(path, max_bytes, identity, &waits) {
                Ok(store) => return Ok(store),
                // Every busy outcome retries while lock-wait budget is left.
                Err(SetupError::Busy(e)) => {
                    if !waits.wait(attempt) {
                        return Err(open_error(e));
                    }
                    attempt += 1;
                }
                Err(SetupError::Fatal(e)) => return Err(open_error(e)),
            }
        }
    }

    /// One setup attempt. Its lock waits, in SQLite's busy handler, draw on
    /// the shared `waits` budget; the installed connection then gets the
    /// ordinary busy timeout.
    fn try_open(
        path: &Path,
        max_bytes: u64,
        identity: Identity,
        waits: &Arc<WaitBudget>,
    ) -> Result<Self, SetupError> {
        // No `SQLITE_OPEN_URI`: the path is a literal file name.
        let mut conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        if conn.path().is_none_or(str::is_empty) {
            return Err(SetupError::Fatal(Box::new(Reason("not a file database"))));
        }
        let handler = Arc::clone(waits);
        conn.busy_handler(Some(move |count: i32| {
            handler.wait(usize::try_from(count).unwrap_or(usize::MAX))
        }))?;
        let mode: String =
            conn.pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get(0))?;
        #[cfg(test)]
        if let Some(slow) = tests::SLOW_SETUP.with(std::cell::Cell::get) {
            // Stands in for slow file creation and fsync: work, not a wait.
            std::thread::sleep(slow);
        }
        #[cfg(test)]
        let mode = if tests::FAULT_WAL_MODE.with(std::cell::Cell::get) {
            "delete".to_owned()
        } else {
            mode
        };
        // SQLite keeps the old mode, without an error, where WAL is unavailable.
        if !mode.eq_ignore_ascii_case("wal") {
            return Err(SetupError::Fatal(Box::new(Reason(
                "the file system does not grant WAL mode",
            ))));
        }
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "journal_size_limit", JOURNAL_SIZE_LIMIT)?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(META)?;
        if owner(&tx)? != Some(identity) {
            tx.execute_batch(RECREATE)?;
            tx.execute(
                "INSERT OR REPLACE INTO meta VALUES (0, ?1, ?2, ?3, 0)",
                (identity.schema, identity.major, identity.minor),
            )?;
        }
        let evicted = trim(&tx, max_bytes)?;
        tx.commit()?;
        // Replaces the setup handler.
        conn.busy_timeout(BUSY_TIMEOUT)?;
        let counters = Counters::default();
        // Rows this process removed to bring an existing file within its bound.
        Counters::add(&counters.evictions, evicted);
        Ok(Self {
            conn: Mutex::new(conn),
            max_bytes,
            identity,
            inert: AtomicBool::new(false),
            counters,
        })
    }

    fn lock(&self) -> MutexGuard<'_, Connection> {
        // Statements are RAII and no invariant of ours lives under this lock.
        self.conn.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The queue's byte limit.
    fn chunk(&self) -> usize {
        let chunk = usize::try_from(self.max_bytes)
            .unwrap_or(usize::MAX)
            .min(PENDING_MAX_BYTES);
        #[cfg(test)]
        let chunk = tests::CHUNK.with(std::cell::Cell::get).unwrap_or(chunk);
        chunk
    }

    /// The stored row, copied out so the read transaction ends before any
    /// shape derivation reaches the memory tiers.
    fn read(&self, key: &[u8]) -> Read {
        let conn = self.lock();
        match conn
            .query_row("SELECT value FROM entries WHERE key = ?1", [key], |row| {
                row.get::<_, Vec<u8>>(0)
            })
            .optional()
        {
            Ok(Some(row)) => Read::Row(row),
            Ok(None) => Read::Absent,
            Err(_) => {
                Counters::add(&self.counters.failures, 1);
                Read::Failed
            }
        }
    }

    /// Commits `rows` in one transaction, after checking that this version
    /// still owns the file. Failures are counted, never returned.
    fn write(&self, rows: &[(Vec<u8>, Vec<u8>)]) {
        if rows.is_empty() {
            return;
        }
        let count = rows.len() as u64;
        if self.inert.load(Ordering::Relaxed) {
            Counters::add(&self.counters.failures, count);
            return;
        }
        #[cfg(test)]
        if tests::FAULT_WRITE.with(std::cell::Cell::get) {
            Counters::add(&self.counters.failures, count);
            return;
        }
        let mut conn = self.lock();
        let result = (|| -> rusqlite::Result<Written> {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            if owner(&tx)? != Some(self.identity) {
                return Ok(Written::TakenOver);
            }
            {
                let mut upsert = tx.prepare(UPSERT)?;
                for (key, value) in rows {
                    upsert.execute((key, value))?;
                }
            }
            let evicted = trim(&tx, self.max_bytes)?;
            tx.commit()?;
            Ok(Written::Committed(evicted))
        })();
        drop(conn);
        match result {
            Ok(Written::Committed(evicted)) => {
                let bytes: usize = rows.iter().map(|(k, v)| k.len() + v.len()).sum();
                Counters::add(&self.counters.bytes_written, bytes as u64);
                Counters::add(&self.counters.evictions, evicted);
            }
            Ok(Written::TakenOver) => {
                self.inert.store(true, Ordering::Relaxed);
                Counters::add(&self.counters.failures, count);
            }
            Err(_) => Counters::add(&self.counters.failures, count),
        }
    }

    fn clear(&self) -> Result<(), DiskCacheError> {
        let mut conn = self.lock();
        let cleared = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
            if owner(&tx)? != Some(self.identity) {
                return Err(Box::new(Reason("another version owns the file")));
            }
            tx.execute("DELETE FROM entries", [])?;
            tx.commit()?;
            Ok(())
        })();
        cleared.map_err(|source| DiskCacheError::Clear { source })?;
        conn.execute_batch("VACUUM")
            .map_err(|e| DiskCacheError::Compact {
                source: Box::new(e),
            })
    }
}

/// The identity recorded in the file, if any.
fn owner(conn: &Connection) -> rusqlite::Result<Option<Identity>> {
    conn.query_row(
        "SELECT schema, major, minor FROM meta WHERE id = 0",
        [],
        |row| {
            Ok(Identity {
                schema: row.get(0)?,
                major: row.get(1)?,
                minor: row.get(2)?,
            })
        },
    )
    .optional()
}

/// Removes oldest rows until the stored bytes fit `max_bytes`; returns how
/// many it removed.
fn trim(conn: &Connection, max_bytes: u64) -> rusqlite::Result<u64> {
    let mut bytes = conn.prepare("SELECT bytes FROM meta WHERE id = 0")?;
    let mut oldest =
        conn.prepare("DELETE FROM entries WHERE id = (SELECT min(id) FROM entries)")?;
    let mut removed = 0;
    while bytes.query_row([], |row| row.get::<_, i64>(0))?
        > i64::try_from(max_bytes).unwrap_or(i64::MAX)
    {
        if oldest.execute([])? == 0 {
            break;
        }
        removed += 1;
    }
    Ok(removed)
}

/// A key: schema, version, kind, presented root and the label words.
fn key_bytes<const N: usize>(kind: u8, root: RootSystem, parts: &[Label<'_>; N]) -> Vec<u8> {
    let words: usize = parts.iter().map(|p| p.words.len()).sum();
    let mut key = Vec::with_capacity(15 + 8 * words);
    let id = Identity::CURRENT;
    for x in [id.schema, id.major, id.minor] {
        key.extend_from_slice(&x.to_le_bytes());
    }
    key.push(kind);
    key.extend_from_slice(&root_code(root).to_le_bytes());
    for part in parts {
        for w in part.words {
            key.extend_from_slice(&w.to_le_bytes());
        }
    }
    key
}

/// The checksum of `key ‖ payload`: one fold per 8-byte word, the tail
/// zero-padded and tagged with its length, and the message length in the seed.
fn checksum(key: &[u8], payload: &[u8]) -> u64 {
    let len = key.len() + payload.len();
    let mut h = fold(SEED ^ len as u64, P1);
    let mut carry = [0u8; 8];
    let mut filled = 0;
    for mut part in [key, payload] {
        if filled > 0 {
            let take = (8 - filled).min(part.len());
            carry[filled..filled + take].copy_from_slice(&part[..take]);
            filled += take;
            part = &part[take..];
            if filled < 8 {
                continue;
            }
            h = fold(h ^ u64::from_le_bytes(carry), P1);
        }
        let (words, tail) = part.as_chunks::<8>();
        for w in words {
            h = fold(h ^ u64::from_le_bytes(*w), P1);
        }
        carry[..tail.len()].copy_from_slice(tail);
        filled = tail.len();
    }
    carry[filled..].fill(0);
    let tail = u64::from_le_bytes(carry) ^ ((len % 8) as u64) << 56;
    fold(fold(h ^ tail, P2), P3)
}

/// A value of a persisted kind.
pub(crate) trait Persist: Sized {
    /// Payload bytes, or `None` when a count does not fit its `u32` field.
    fn payload_len(&self) -> Option<usize>;
    /// Appends the payload; called only after `payload_len` succeeded.
    fn encode(&self, out: &mut Vec<u8>);
    /// The value, if `payload` is exactly a value of `shape`.
    fn decode(payload: &[u8], shape: &Shape) -> Option<Self>;
}

fn u32_at(bytes: &[u8], at: usize) -> usize {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes")) as usize
}

fn f64_at(bytes: &[u8], at: usize) -> f64 {
    f64::from_bits(u64::from_le_bytes(
        bytes[at..at + 8].try_into().expect("8 bytes"),
    ))
}

fn put_u32(out: &mut Vec<u8>, x: usize) {
    out.extend_from_slice(&(x as u32).to_le_bytes());
}

/// `count` little-endian `f64` after `at`, if they are all there is.
fn values(bytes: &[u8], at: usize, count: usize) -> Option<Vec<f64>> {
    if bytes.len().checked_sub(at)? != count.checked_mul(8)? {
        return None;
    }
    Some((0..count).map(|i| f64_at(bytes, at + 8 * i)).collect())
}

/// The exact channel ([`ExactCgc::encode`]).
impl Persist for Arc<ExactCgc> {
    fn payload_len(&self) -> Option<usize> {
        self.encoded_len()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        ExactCgc::encode(self, out);
    }

    fn decode(payload: &[u8], shape: &Shape) -> Option<Self> {
        let &Shape::Channel { dims, mult } = shape else {
            return None;
        };
        ExactCgc::decode(payload, dims, mult).map(Arc::new)
    }
}

/// The highest-weight column ([`HighestColumn::encode`]).
impl Persist for Arc<HighestColumn> {
    fn payload_len(&self) -> Option<usize> {
        self.encoded_len()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        HighestColumn::encode(self, out);
    }

    fn decode(payload: &[u8], shape: &Shape) -> Option<Self> {
        let &Shape::Seeds { dims, mult } = shape else {
            return None;
        };
        HighestColumn::decode(payload, dims, mult).map(Arc::new)
    }
}

/// The norm table of one irrep ([`NormTable::encode`]).
impl Persist for Arc<NormTable> {
    fn payload_len(&self) -> Option<usize> {
        self.encoded_len()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        NormTable::encode(self, out);
    }

    fn decode(payload: &[u8], shape: &Shape) -> Option<Self> {
        let &Shape::Norms { dim } = shape else {
            return None;
        };
        NormTable::decode(payload, dim).map(Arc::new)
    }
}

/// One byte, `±1`.
impl Persist for i8 {
    fn payload_len(&self) -> Option<usize> {
        Some(1)
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }

    fn decode(payload: &[u8], shape: &Shape) -> Option<Self> {
        match (shape, payload) {
            (Shape::Phase, &[x]) => Some(x as i8).filter(|&k| k == 1 || k == -1),
            _ => None,
        }
    }
}

/// `4 × u32` dims, then the row-major values.
impl Persist for FBlock<f64> {
    fn payload_len(&self) -> Option<usize> {
        for d in self.dims() {
            u32::try_from(d).ok()?;
        }
        self.as_slice().len().checked_mul(8)?.checked_add(16)
    }

    fn encode(&self, out: &mut Vec<u8>) {
        for d in self.dims() {
            put_u32(out, d);
        }
        for x in self.as_slice() {
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }

    fn decode(payload: &[u8], shape: &Shape) -> Option<Self> {
        let Shape::F(expected) = shape else {
            return None;
        };
        if payload.len() < 16 {
            return None;
        }
        let dims = [0, 4, 8, 12].map(|at| u32_at(payload, at));
        if &dims != expected {
            return None;
        }
        let count = if dims.contains(&0) {
            0
        } else {
            dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d))?
        };
        let block = FBlock::from_vec(dims, values(payload, 16, count)?);
        check_f_block(&block).ok()?;
        Some(block)
    }
}

/// `u32 n`, then the row-major `n × n` values.
impl Persist for RBlock<f64> {
    fn payload_len(&self) -> Option<usize> {
        u32::try_from(self.dims()[0]).ok()?;
        self.as_slice().len().checked_mul(8)?.checked_add(4)
    }

    fn encode(&self, out: &mut Vec<u8>) {
        put_u32(out, self.dims()[0]);
        for x in self.as_slice() {
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }

    fn decode(payload: &[u8], shape: &Shape) -> Option<Self> {
        let &Shape::R(expected) = shape else {
            return None;
        };
        if payload.len() < 4 || u32_at(payload, 0) != expected {
            return None;
        }
        let block = RBlock::from_vec(
            expected,
            values(payload, 4, expected.checked_mul(expected)?)?,
        );
        check_r_block(&block).ok()?;
        Some(block)
    }
}

/// The value of a stored row, if its checksum and shape hold.
fn decode_row<V: Persist>(key: &[u8], row: &[u8], shape: &Shape) -> Option<V> {
    let (payload, sum) = row.split_at_checked(row.len().checked_sub(8)?)?;
    if checksum(key, payload) != u64::from_le_bytes(sum.try_into().ok()?) {
        return None;
    }
    V::decode(payload, shape)
}

/// Serves a miss of a persisted kind from the store when its row is valid,
/// otherwise generates and queues the row. `shape` runs only on a found row;
/// if it fails, `compute` decides the call's result.
pub(crate) fn through<V: Persist, E, const N: usize>(
    kind: u8,
    root: RootSystem,
    parts: [Label<'_>; N],
    shape: impl FnOnce() -> Result<Shape, E>,
    compute: impl FnOnce() -> Result<V, E>,
) -> Result<V, E> {
    let Some(store) = STORE.get() else {
        return compute();
    };
    if store.inert.load(Ordering::Relaxed) {
        return compute();
    }
    let key = key_bytes(kind, root, &parts);
    match store.read(&key) {
        Read::Row(row) => {
            #[cfg(test)]
            tests::after_read();
            if let Ok(shape) = shape() {
                if let Some(value) = decode_row(&key, &row, &shape) {
                    Counters::add(&store.counters.hits, 1);
                    Counters::add(&store.counters.bytes_read, row.len() as u64);
                    return Ok(value);
                }
                Counters::add(&store.counters.rejected, 1);
            }
        }
        Read::Absent => Counters::add(&store.counters.misses, 1),
        Read::Failed => {}
    }
    let value = compute()?;
    enqueue(store, key, &value);
    Ok(value)
}

/// Encoded rows of this thread's running generation, not yet committed.
#[derive(Default)]
struct Pending {
    rows: Vec<(Vec<u8>, Vec<u8>)>,
    bytes: usize,
}

thread_local! {
    static PENDING: RefCell<Pending> = RefCell::new(Pending::default());
}

fn take_pending() -> Vec<(Vec<u8>, Vec<u8>)> {
    PENDING.with(|p| {
        let mut p = p.borrow_mut();
        p.bytes = 0;
        std::mem::take(&mut p.rows)
    })
}

/// Queues `value`'s row, keeping the queue within its limits. A row that
/// would exceed the store bound is not stored; a row larger than the queue is
/// written through on its own.
fn enqueue<V: Persist>(store: &Store, key: Vec<u8>, value: &V) {
    let row = value
        .payload_len()
        .and_then(|n| n.checked_add(key.len())?.checked_add(8))
        .filter(|&row| row as u64 <= store.max_bytes);
    let Some(row) = row else {
        Counters::add(&store.counters.evictions, 1);
        return;
    };
    let mut encoded = Vec::with_capacity(row - key.len());
    value.encode(&mut encoded);
    let sum = checksum(&key, &encoded);
    encoded.extend_from_slice(&sum.to_le_bytes());
    let chunk = store.chunk();
    let full = PENDING.with(|p| {
        let p = p.borrow();
        p.bytes + row > chunk || p.rows.len() >= PENDING_MAX_ENTRIES
    });
    if full {
        store.write(&take_pending());
    }
    if row > chunk {
        store.write(&[(key, encoded)]);
        return;
    }
    PENDING.with(|p| {
        let mut p = p.borrow_mut();
        p.bytes += row;
        p.rows.push((key, encoded));
        #[cfg(test)]
        tests::PEAK.with(|peak| peak.set(peak.get().max(p.bytes)));
    });
}

/// Commits this thread's queued rows; called at the end of each outermost
/// generation.
pub(super) fn flush() {
    if let Some(store) = STORE.get() {
        store.write(&take_pending());
    }
}

/// Drops the queued rows of a generation that unwinds, without I/O.
pub(super) struct DiscardOnUnwind;

impl Drop for DiscardOnUnwind {
    fn drop(&mut self) {
        if std::thread::panicking() && STORE.get().is_some() {
            drop(take_pending());
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
