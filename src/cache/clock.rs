//! A byte-budgeted CLOCK cache with lock-free hits and single-flight misses,
//! after MemoizationKit's `ClockCache` (`containers/clock.jl`: `evict!`,
//! `addentry!`).
//!
//! This module and [`table`] are generic over the key, value and hasher and
//! take each entry's charge from the caller; they depend only on `std` and
//! `seize`. A hit reads the published table under a seize guard and writes
//! nothing shared except the entry's reference bit when it is clear.
//! Insertion, eviction, trim, clear and the single-flight bookkeeping run
//! under one writer lock.
//!
//! Single flight is kept although `clock.jl` computes outside its lock and
//! may duplicate work: a miss claims its key, one leader computes without the
//! lock, and waiters block until it publishes. A leader that fails or panics
//! abandons the claim (see [`Lead`]), and exactly one waiter claims again.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};
use std::sync::{Arc, Condvar, Mutex, PoisonError};

mod table;
pub(crate) use table::{entry_shell, release};

/// A borrowed lookup form of the key `K` (the `equivalent` crate's trait):
/// it hashes like the key it is equivalent to.
pub(crate) trait Equivalent<K: ?Sized> {
    fn equivalent(&self, key: &K) -> bool;
}

impl<Q: ?Sized + Eq, K: ?Sized + std::borrow::Borrow<Q>> Equivalent<K> for Q {
    fn equivalent(&self, key: &K) -> bool {
        self == key.borrow()
    }
}

enum Flight<V> {
    Pending,
    Done(V),
    Abandoned,
}

/// One in-flight miss, shared by its leader and waiters.
pub(crate) struct Placeholder<V> {
    /// The outcome and the number of blocked waiters.
    state: Mutex<(Flight<V>, usize)>,
    ready: Condvar,
}

impl<V: Clone> Placeholder<V> {
    fn new() -> Self {
        Self {
            state: Mutex::new((Flight::Pending, 0)),
            ready: Condvar::new(),
        }
    }

    /// Wakes waiters only when there are any: std's futex `notify_all` is a
    /// syscall even without waiters, on every published miss.
    fn finish(&self, outcome: Flight<V>) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.0 = outcome;
        if state.1 > 0 {
            self.ready.notify_all();
        }
    }

    /// The leader's value, or `None` when it failed (the caller claims again).
    fn wait(&self) -> Option<V> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.1 += 1;
        loop {
            match &state.0 {
                Flight::Pending => {
                    state = self
                        .ready
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner)
                }
                Flight::Done(v) => return Some(v.clone()),
                Flight::Abandoned => return None,
            }
        }
    }
}

/// Accounting and single-flight state, behind the writer lock.
pub(crate) struct State<K, V, B> {
    /// Sum of admission weights, `max(bytes, minimum)` per entry.
    weight: usize,
    /// Sum of entry charges.
    pub(crate) bytes: usize,
    pub(crate) evictions: u64,
    hand: usize,
    flights: HashMap<K, Arc<Placeholder<V>>, B>,
}

/// The held writer lock of a [`Clock`].
pub(crate) type Locked<'a, K, V, B> = table::Locked<'a, K, V, B, State<K, V, B>>;

/// A CLOCK cache bounded by a byte budget and an entry cap.
pub(crate) struct Clock<K, V, B> {
    map: table::Map<K, V, B, State<K, V, B>>,
    max_bytes: usize,
    /// The minimum admission weight `ceil(max_bytes / max_entries)`, so at
    /// most `max_entries` entries are resident.
    minimum: usize,
}

/// Outcome of claiming a key on a miss.
pub(crate) enum Claim<V> {
    /// This thread computes; it must [`Clock::publish`] or [`Clock::abandon`]
    /// (through [`Lead`]).
    Lead(Arc<Placeholder<V>>),
    /// Resident, or delivered by another leader.
    Value(V),
}

/// How [`Clock::publish`] ended a flight.
pub(crate) enum Publication {
    Retained,
    /// Another flight had already published the key; its value is returned.
    Resident,
    /// Larger than the budget, or not retained by request.
    Rejected,
    NotRetained,
}

impl<K, V, B> Clock<K, V, B>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    B: BuildHasher + Clone,
{
    /// `max_entries` must be positive.
    pub(crate) fn new(max_entries: usize, max_bytes: usize, hasher: B) -> Self {
        let state = State {
            weight: 0,
            bytes: 0,
            evictions: 0,
            hand: 0,
            flights: HashMap::with_hasher(hasher.clone()),
        };
        Self {
            map: table::Map::new(hasher, state),
            max_bytes,
            minimum: max_bytes.div_ceil(max_entries),
        }
    }

    /// The lock-free hit: `f` of the resident value.
    #[inline]
    pub(crate) fn get_with<Q, R>(&self, key: &Q, f: impl FnOnce(&V) -> R) -> Option<R>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.map.get_with(key, f)
    }

    pub(crate) fn lock(&self) -> Locked<'_, K, V, B> {
        self.map.lock()
    }

    /// One leader per key; waiters block until it publishes or fails.
    pub(crate) fn claim(&self, key: &K) -> Claim<V> {
        loop {
            let flight = {
                let mut st = self.lock();
                if let Some(v) = st.get(key) {
                    return Claim::Value(v.clone());
                }
                match st.flights.get(key) {
                    Some(f) => Arc::clone(f),
                    None => {
                        let f = Arc::new(Placeholder::new());
                        st.flights.insert(key.clone(), Arc::clone(&f));
                        return Claim::Lead(f);
                    }
                }
            };
            if let Some(v) = flight.wait() {
                return Claim::Value(v);
            }
        }
    }

    /// The leader failed or panicked: the next claimant leads.
    pub(crate) fn abandon(&self, key: &K, flight: &Arc<Placeholder<V>>) {
        let mut st = self.lock();
        if st.flights.get(key).is_some_and(|f| Arc::ptr_eq(f, flight)) {
            st.flights.remove(key);
        }
        drop(st);
        flight.finish(Flight::Abandoned);
    }

    /// Ends a flight with `value` charged `bytes` under the held lock,
    /// retaining it when `retain` and it fits, evicting by CLOCK as needed.
    /// Returns the value to hand out and how the flight ended.
    pub(crate) fn publish(
        &self,
        st: &mut Locked<'_, K, V, B>,
        key: K,
        flight: &Arc<Placeholder<V>>,
        value: V,
        bytes: usize,
        retain: bool,
    ) -> (V, Publication) {
        if st.flights.get(&key).is_some_and(|f| Arc::ptr_eq(f, flight)) {
            st.flights.remove(&key);
        }
        let weight = bytes.max(self.minimum);
        let (value, outcome) = if !retain {
            (value, Publication::NotRetained)
        } else if let Some(stored) = st.get(&key) {
            (stored.clone(), Publication::Resident)
        } else if weight > self.max_bytes {
            st.evictions += 1;
            (value, Publication::Rejected)
        } else {
            while st.weight + weight > self.max_bytes && st.len() > 0 {
                self.evict(st);
            }
            st.insert(key, value.clone(), bytes);
            st.weight += weight;
            st.bytes += bytes;
            (value, Publication::Retained)
        };
        flight.finish(Flight::Done(value.clone()));
        (value, outcome)
    }

    fn remove_at(&self, st: &mut Locked<'_, K, V, B>, pos: usize) -> bool {
        let Some(bytes) = st.remove(pos) else {
            return false;
        };
        st.weight -= bytes.max(self.minimum);
        st.bytes -= bytes;
        st.evictions += 1;
        true
    }

    /// The hand clears set reference bits until it finds a clear one; after
    /// two sweeps it evicts anyway (`evict!`). Requires a resident entry.
    fn evict(&self, st: &mut Locked<'_, K, V, B>) {
        let n = st.positions();
        let mut steps = 0;
        loop {
            st.hand = if st.hand + 1 >= n { 0 } else { st.hand + 1 };
            steps += 1;
            let hand = st.hand;
            match st.clear_used(hand) {
                None => continue,
                Some(true) if steps <= 2 * n => continue,
                Some(_) => {
                    self.remove_at(st, hand);
                    return;
                }
            }
        }
    }

    /// Removes every entry and zeroes the accounting, evictions included.
    /// In-flight claims survive. Reclamation starts at the next [`release`].
    pub(crate) fn clear(&self, st: &mut Locked<'_, K, V, B>) {
        st.clear();
        st.weight = 0;
        st.bytes = 0;
        st.evictions = 0;
        st.hand = 0;
    }

    /// Removes entries in ring order until the charge is at most `target`
    /// (all of them for `target == 0`); returns the number removed, each
    /// counted as an eviction. In-flight claims survive. Reclamation starts at
    /// the next [`release`].
    pub(crate) fn trim(&self, st: &mut Locked<'_, K, V, B>, target: usize) -> usize {
        if target == 0 {
            let removed = st.len();
            st.clear();
            st.weight = 0;
            st.bytes = 0;
            st.evictions += removed as u64;
            st.hand = 0;
            return removed;
        }
        let mut removed = 0;
        for pos in 0..st.positions() {
            if st.bytes <= target {
                break;
            }
            removed += usize::from(self.remove_at(st, pos));
        }
        removed
    }
}

#[cfg(test)]
impl<K, V, B> Clock<K, V, B>
where
    K: Clone + Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    B: BuildHasher + Clone,
{
    /// Every entry with its charge, in ring order.
    pub(crate) fn entries(&self) -> Vec<(K, V, usize)> {
        let mut out = Vec::new();
        self.lock()
            .for_each(|k, v, bytes| out.push((k.clone(), v.clone(), bytes)));
        out
    }

    /// A lookup that does not touch the reference bit.
    pub(crate) fn peek<Q: Hash + Equivalent<K> + ?Sized>(&self, key: &Q) -> Option<V> {
        self.lock().get(key).cloned()
    }
}
