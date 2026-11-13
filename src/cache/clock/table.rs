//! The published slot table of a CLOCK tier and its ring, after MemoizationKit
//! `containers/clock.jl` (`findentry`, `addentry!`, `placeentry!`,
//! `removeentry!`), with seize reclamation standing in for the GC.
//!
//! The table is an open-addressing hash table with linear probing, at most
//! half full, read without the writer lock. As in `clock.jl`:
//! - a node never changes after construction, apart from its reference bit;
//! - a published table never changes length, it is replaced as a whole;
//! - a slot only changes, under the writer lock, from empty or tombstone to a
//!   node, from a node to a tombstone, or to empty when the next slot is
//!   empty. Nodes never move, so a probe finds every node present when it
//!   started.
//!
//! Slots are loaded with acquire (seize `protect`) and stored with release
//! ordering. Empty is null and the tombstone is the address of [`TOMB`], which
//! is never dereferenced. A node is one `Box` shared by every table
//! generation, so a resize never clones a key.
//!
//! Every `unsafe` below relies on one of these invariants:
//! 1. `slots` always points to a live `Slots` made by [`Slots::new`]; every
//!    non-null, non-tombstone slot pointer was made by `Box::into_raw` in
//!    [`Locked::insert`] and stays valid until it is retired.
//! 2. Only the writer-lock holder unlinks and retires a pointer, exactly once,
//!    and only after unlinking it from the current table. A reader
//!    dereferences only pointers it loaded under its own seize guard, so seize
//!    frees nothing a reader can still reach. `K` and `V` are `Send + Sync +
//!    'static` because seize may drop a retired node on any thread, at any
//!    later time.
//! 3. A ring position holds exactly the nodes linked in the current table;
//!    only the writer-lock holder dereferences them.
//! 4. `Drop` has `&mut self`: no reader or writer is left, and every linked
//!    node is in the ring exactly once, while retired nodes are not.
//! 5. A [`Graveyard`] holds only pointers unlinked by its lock section, each
//!    once, and is their sole owner: it is retired to seize as one object
//!    when the section ends, and freeing it frees them.
//!
//! Retiring one graveyard per lock section, rather than one seize object per
//! node, makes a trim one entry of the thread's seize batch and a reset at
//! most six (one per retiring tier section), whatever the number of removed
//! entries. Each submission attempt is a process-wide barrier: a call makes
//! one explicit attempt ([`release`]) plus at most one automatic attempt per
//! retiring tier section, so at most two barriers in the ordinary case and at
//! most seven when `reset` retries for its six retirements (seize retries
//! while more threads are inside a guard than its batch holds objects).

use super::Equivalent;
use seize::{Collector, Guard, LocalGuard};
use std::hash::{BuildHasher, Hash};
use std::ptr::{self, NonNull};
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicBool, AtomicPtr};
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

/// One collector for every table, so a reader enters one reservation.
static COLLECTOR: LazyLock<Collector> = LazyLock::new(Collector::new);
static TOMB: u8 = 0;
/// The length of a new table (`emptytable`).
const INITIAL_SLOTS: usize = 16;

/// Submits this thread's seize batch for reclamation (one barrier). Its
/// graveyards are freed once every thread that was inside a guard has left
/// it; seize keeps a batch with fewer objects than active threads for this
/// thread's next retirements.
pub(crate) fn release() {
    COLLECTOR.enter().flush();
}

fn tomb<K, V>() -> *mut Node<K, V> {
    ptr::from_ref(&TOMB).cast_mut().cast()
}

struct Node<K, V> {
    hash: u64,
    key: K,
    value: V,
    used: AtomicBool,
}

impl<K, V> Node<K, V> {
    /// Sets the reference bit only when it is clear, so a hot node is only
    /// read (`usevalue!`).
    #[inline]
    fn touch(&self) -> &V {
        if !self.used.load(Relaxed) {
            self.used.store(true, Relaxed);
        }
        &self.value
    }
}

/// One ring position: a linked node and its charge.
struct Ring<K, V> {
    node: NonNull<Node<K, V>>,
    bytes: usize,
}

/// Bytes one entry retains beyond its key and value charges: the boxed node
/// shell and its ring position.
pub(crate) const fn entry_shell<K, V>() -> usize {
    std::mem::size_of::<Node<K, V>>() + std::mem::size_of::<Option<Ring<K, V>>>()
}

struct Slots<K, V>(Box<[AtomicPtr<Node<K, V>>]>);

impl<K, V> Slots<K, V> {
    fn new(len: usize) -> *mut Self {
        let slots = (0..len).map(|_| AtomicPtr::new(ptr::null_mut())).collect();
        Box::into_raw(Box::new(Self(slots)))
    }

    #[inline]
    fn start(&self, hash: u64) -> usize {
        hash as usize & (self.0.len() - 1)
    }

    #[inline]
    fn next(&self, i: usize) -> usize {
        (i + 1) & (self.0.len() - 1)
    }

    /// Stores `node` in the first slot of its probe without a node; whether
    /// that slot was empty rather than a tombstone (`placeentry!`).
    fn place(&self, hash: u64, node: *mut Node<K, V>) -> bool {
        let mut i = self.start(hash);
        loop {
            let old = self.0[i].load(Relaxed);
            if old.is_null() || old == tomb() {
                self.0[i].store(node, Release);
                return old.is_null();
            }
            i = self.next(i);
        }
    }

    /// The node equivalent to `key` (`findentry`), every slot loaded by `load`.
    ///
    /// # Safety
    ///
    /// Every non-null, non-tombstone pointer `load` returns stays valid for
    /// `'a`.
    #[inline]
    unsafe fn find<'a, Q: Hash + Equivalent<K> + ?Sized>(
        &'a self,
        hash: u64,
        key: &Q,
        load: impl Fn(&AtomicPtr<Node<K, V>>) -> *mut Node<K, V>,
    ) -> Option<&'a Node<K, V>> {
        let mut i = self.start(hash);
        // Bounds the probe also while slots change.
        for _ in 0..self.0.len() {
            let p = load(&self.0[i]);
            if p.is_null() {
                return None;
            }
            if p != tomb() {
                // SAFETY: guaranteed by the caller.
                let node = unsafe { &*p };
                if node.hash == hash && key.equivalent(&node.key) {
                    return Some(node);
                }
            }
            i = self.next(i);
        }
        None
    }
}

/// Writer state: the ring and the table's occupancy, plus the owner's `S`.
struct Writer<K, V, S> {
    ring: Vec<Option<Ring<K, V>>>,
    free: Vec<usize>,
    /// Slots holding a node or a tombstone.
    occupied: usize,
    count: usize,
    state: S,
}

// SAFETY: the ring's pointers name nodes that readers share as `&Node` and
// that seize may drop on any thread, so moving the writer state across threads
// needs exactly `K, V: Send + Sync`; they are dereferenced only by the lock
// holder (invariant 3).
unsafe impl<K: Send + Sync, V: Send + Sync, S: Send> Send for Writer<K, V, S> {}

/// A concurrent map whose hits take no lock. Mutations go through
/// [`Map::lock`], which also guards the owner's state `S`.
pub(crate) struct Map<K, V, B, S> {
    slots: AtomicPtr<Slots<K, V>>,
    hasher: B,
    writer: Mutex<Writer<K, V, S>>,
}

impl<K, V, B, S> Map<K, V, B, S>
where
    K: Eq + Hash + Send + Sync + 'static,
    V: Send + Sync + 'static,
    B: BuildHasher,
{
    pub(crate) fn new(hasher: B, state: S) -> Self {
        Self {
            slots: AtomicPtr::new(Slots::new(INITIAL_SLOTS)),
            hasher,
            writer: Mutex::new(Writer {
                ring: Vec::new(),
                free: Vec::new(),
                occupied: 0,
                count: 0,
                state,
            }),
        }
    }

    /// The lock-free hit: `f` of the value equivalent to `key`, run under
    /// this thread's seize guard. Sets the reference bit when it is clear.
    #[inline]
    pub(crate) fn get_with<Q, R>(&self, key: &Q, f: impl FnOnce(&V) -> R) -> Option<R>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hasher.hash_one(key);
        let guard = COLLECTOR.enter();
        // SAFETY: invariant 1; the table stays valid under `guard` (invariant 2).
        let slots = unsafe { &*guard.protect(&self.slots, Acquire) };
        // SAFETY: every node loaded under `guard` outlives this call (invariant 2).
        let node = unsafe { slots.find(hash, key, |slot| guard.protect(slot, Acquire)) };
        node.map(|node| f(node.touch()))
    }

    pub(crate) fn lock(&self) -> Locked<'_, K, V, B, S> {
        Locked {
            map: self,
            writer: self.writer.lock().unwrap_or_else(PoisonError::into_inner),
            guard: COLLECTOR.enter(),
            dead: Graveyard {
                nodes: Vec::new(),
                tables: Vec::new(),
            },
        }
    }
}

impl<K, V, B, S> Drop for Map<K, V, B, S> {
    fn drop(&mut self) {
        let writer = self
            .writer
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        for ring in writer.ring.drain(..).flatten() {
            // SAFETY: invariant 4: each linked node is freed once, here.
            drop(unsafe { Box::from_raw(ring.node.as_ptr()) });
        }
        // SAFETY: invariants 1 and 4: the current table is not retired.
        drop(unsafe { Box::from_raw(*self.slots.get_mut()) });
    }
}

/// The allocations one lock section unlinked (invariant 5). Seize frees it on
/// whichever thread is last to leave a guard that was active when its batch
/// was submitted: the retiring thread, or any thread then inside a hit or
/// another lock section, which then drops every entry it holds.
struct Graveyard<K, V> {
    nodes: Vec<NonNull<Node<K, V>>>,
    tables: Vec<NonNull<Slots<K, V>>>,
}

impl<K, V> Drop for Graveyard<K, V> {
    fn drop(&mut self) {
        for node in self.nodes.drain(..) {
            // SAFETY: invariant 5: unlinked, owned only here, freed once.
            drop(unsafe { Box::from_raw(node.as_ptr()) });
        }
        for table in self.tables.drain(..) {
            // SAFETY: invariant 5; a table owns only pointer storage.
            drop(unsafe { Box::from_raw(table.as_ptr()) });
        }
    }
}

/// The writer lock, held with a seize guard so that nothing retired while it
/// is held is freed before the guard drops (after the lock is released).
pub(crate) struct Locked<'a, K, V, B, S> {
    map: &'a Map<K, V, B, S>,
    writer: MutexGuard<'a, Writer<K, V, S>>,
    guard: LocalGuard<'static>,
    dead: Graveyard<K, V>,
}

impl<K, V, B, S> Drop for Locked<'_, K, V, B, S> {
    /// Retires this section's graveyard as one seize object, under the guard.
    fn drop(&mut self) {
        if self.dead.nodes.is_empty() && self.dead.tables.is_empty() {
            return;
        }
        let dead = Graveyard {
            nodes: std::mem::take(&mut self.dead.nodes),
            tables: std::mem::take(&mut self.dead.tables),
        };
        // SAFETY: invariants 2 and 5: every pointer in it is unlinked and
        // owned by it alone; `boxed` frees the graveyard, which frees them.
        unsafe {
            self.guard
                .defer_retire(Box::into_raw(Box::new(dead)), seize::reclaim::boxed)
        };
    }
}

impl<K, V, B, S> std::ops::Deref for Locked<'_, K, V, B, S> {
    type Target = S;
    fn deref(&self) -> &S {
        &self.writer.state
    }
}

impl<K, V, B, S> std::ops::DerefMut for Locked<'_, K, V, B, S> {
    fn deref_mut(&mut self) -> &mut S {
        &mut self.writer.state
    }
}

impl<K, V, B, S> Locked<'_, K, V, B, S>
where
    K: Eq + Hash,
    B: BuildHasher,
{
    fn slots(&self) -> &Slots<K, V> {
        // SAFETY: invariant 1; only this lock holder replaces or retires it.
        unsafe { &*self.map.slots.load(Relaxed) }
    }

    fn node(&self, pos: usize) -> Option<&Node<K, V>> {
        let ring = self.writer.ring.get(pos)?.as_ref()?;
        // SAFETY: invariant 3: linked, and only this lock holder retires it.
        Some(unsafe { ring.node.as_ref() })
    }

    /// Entries in the table.
    pub(crate) fn len(&self) -> usize {
        self.writer.count
    }

    /// Ring positions, occupied or free.
    pub(crate) fn positions(&self) -> usize {
        self.writer.ring.len()
    }

    /// The value equivalent to `key`, without touching its reference bit.
    pub(crate) fn get<Q: Hash + Equivalent<K> + ?Sized>(&self, key: &Q) -> Option<&V> {
        let hash = self.map.hasher.hash_one(key);
        // SAFETY: only this lock holder retires a node (invariant 2).
        unsafe { self.slots().find(hash, key, |slot| slot.load(Relaxed)) }.map(|n| &n.value)
    }

    /// Inserts an absent key charged `bytes` (`addentry!`), first publishing
    /// a larger table without tombstones when this would fill more than half.
    pub(crate) fn insert(&mut self, key: K, value: V, bytes: usize) {
        debug_assert!(self.get(&key).is_none(), "insert of a resident key");
        let hash = self.map.hasher.hash_one(&key);
        if 2 * (self.writer.occupied + 1) > self.slots().0.len() {
            let len = INITIAL_SLOTS.max((4 * (self.writer.count + 1)).next_power_of_two());
            let fresh = Slots::new(len);
            for ring in self.writer.ring.iter().flatten() {
                // SAFETY: `fresh` is unpublished; invariant 3 for the node.
                unsafe { (*fresh).place(ring.node.as_ref().hash, ring.node.as_ptr()) };
            }
            self.writer.occupied = self.writer.count;
            let old = self.map.slots.swap(fresh, Release);
            // The swap unlinked `old`; its nodes stay linked from `fresh`.
            let old = NonNull::new(old).expect("a table pointer is not null");
            self.dead.tables.push(old);
        }
        let node = Box::into_raw(Box::new(Node {
            hash,
            key,
            value,
            used: AtomicBool::new(false),
        }));
        if self.slots().place(hash, node) {
            self.writer.occupied += 1;
        }
        self.writer.count += 1;
        let ring = Some(Ring {
            node: NonNull::new(node).expect("a Box pointer is not null"),
            bytes,
        });
        match self.writer.free.pop() {
            Some(pos) => self.writer.ring[pos] = ring,
            None => self.writer.ring.push(ring),
        }
    }

    /// Clears the reference bit at `pos`; whether it was set, or `None` for a
    /// free position.
    pub(crate) fn clear_used(&mut self, pos: usize) -> Option<bool> {
        Some(self.node(pos)?.used.swap(false, Relaxed))
    }

    /// Unlinks the node at `pos` (`removeentry!`) into this section's
    /// graveyard; its charge, or `None` for a free position. Reclamation
    /// waits for [`release`] or later retirements on this thread.
    pub(crate) fn remove(&mut self, pos: usize) -> Option<usize> {
        let hash = self.node(pos)?.hash;
        let ring = self.writer.ring[pos].take().expect("an occupied position");
        self.writer.free.push(pos);
        let p = ring.node.as_ptr();
        let slots = self.slots();
        let mut i = slots.start(hash);
        while slots.0[i].load(Relaxed) != p {
            i = slots.next(i);
        }
        let mut emptied = 0;
        if slots.0[slots.next(i)].load(Relaxed).is_null() {
            // No probe passes this slot or the tombstones before it.
            let mask = slots.0.len() - 1;
            loop {
                slots.0[i].store(ptr::null_mut(), Release);
                emptied += 1;
                i = i.wrapping_sub(1) & mask;
                if slots.0[i].load(Relaxed) != tomb() {
                    break;
                }
            }
        } else {
            slots.0[i].store(tomb(), Release);
        }
        self.writer.occupied -= emptied;
        self.writer.count -= 1;
        // Unlinked above, and its ring position is gone (invariant 5).
        self.dead.nodes.push(ring.node);
        Some(ring.bytes)
    }

    /// Unlinks every node by publishing an empty table and releases the
    /// ring's capacity. An empty tier at the initial table length is left as
    /// is, so clearing it retires nothing. The unlinked allocations go to this
    /// section's graveyard; reclamation starts at the next [`release`].
    pub(crate) fn clear(&mut self) {
        let ring = std::mem::take(&mut self.writer.ring);
        self.writer.free = Vec::new();
        if self.writer.count == 0 && self.slots().0.len() == INITIAL_SLOTS {
            return;
        }
        let old = self.map.slots.swap(Slots::new(INITIAL_SLOTS), Release);
        // The swap unlinked the table and every node of the ring, each once
        // (invariants 3 and 5).
        let old = NonNull::new(old).expect("a table pointer is not null");
        self.dead.tables.push(old);
        self.dead
            .nodes
            .extend(ring.into_iter().flatten().map(|ring| ring.node));
        self.writer.occupied = 0;
        self.writer.count = 0;
    }

    /// Every entry with its charge, in ring order.
    #[cfg(test)]
    pub(crate) fn for_each(&self, mut f: impl FnMut(&K, &V, usize)) {
        for (pos, ring) in self.writer.ring.iter().enumerate() {
            if let (Some(ring), Some(node)) = (ring, self.node(pos)) {
                f(&node.key, &node.value, ring.bytes);
            }
        }
    }
}
