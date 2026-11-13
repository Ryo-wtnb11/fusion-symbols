//! One stage of independent items inside a coefficient generation: a serial
//! calibration prefix on the leader, then at most one team section at the
//! width [`crate::width::choose_width`] picks for the rest.
//!
//! Only a token that carries a team with a budget of at least two forks; any
//! other token, and a stage whose feasible sections all cost `Duration::MAX`,
//! runs every item inline in item order without reading the clock. The first
//! item with positive units always runs before the prefix compares its
//! elapsed time with the cheapest section, so a sample is never empty.
//!
//! A section hands the remaining items out in LPT order (largest units
//! first, ties by index) through one atomic counter, and each item writes its
//! own result slot, so results come back in item order whatever the width.
//! Shares never touch the coefficient cache: everything they read is
//! borrowed from the leader.

use std::any::Any;
use std::num::{NonZeroU64, NonZeroUsize};
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use threadteam::Exec;

use crate::width::{cheapest_section, choose_width, Sample};

/// The stage's time source: [`Instant::now`] in production, scripted in
/// tests.
pub(crate) trait Clock: Sync {
    fn now(&self) -> Instant;
}

/// The monotonic system clock.
pub(crate) struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// How a stage ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ran {
    /// Width of its section, or 1 when every item ran on the leader.
    pub(crate) width: NonZeroUsize,
    /// Scratch values created by this stage, including replacements of
    /// scratch dropped after a panic. The leader's carries over between
    /// stages and is participant 0's in a section.
    pub(crate) scratches: usize,
}

/// The result slots and item queue of a stage, kept by a caller across its
/// stages of one kind so that a stage allocates nothing of its own when its
/// items run on the leader.
pub(crate) struct StageBuf<T> {
    out: Vec<Option<T>>,
    queue: Vec<usize>,
}

impl<T> Default for StageBuf<T> {
    fn default() -> Self {
        Self {
            out: Vec::new(),
            queue: Vec::new(),
        }
    }
}

impl<T> StageBuf<T> {
    /// The last stage's results, in item order.
    pub(crate) fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.out
            .drain(..)
            .map(|r| r.expect("every item of a stage runs exactly once"))
    }

    /// The last stage's results, borrowed, in item order.
    pub(crate) fn results(&self) -> impl Iterator<Item = &T> {
        self.out.iter().flatten()
    }
}

/// Runs `kernel(scratch, i)` for every item `i` of a stage whose items have
/// units `units[i]`, leaving the results in `buf` in item order. Every item
/// is attempted even when others panic; the lowest-index item's panic is
/// then re-raised.
///
/// `scratch` builds a participant's working state, at most once per
/// participant and only when it takes an item; the leader's lives in
/// `leader`, so a caller keeps it across its stages.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_stage<S: Send, T: Send + Sync>(
    exec: &mut Exec<'_>,
    units: &[u64],
    clock: &impl Clock,
    leader: &mut Option<S>,
    buf: &mut StageBuf<T>,
    scratch: impl Fn() -> S + Sync,
    kernel: impl Fn(&mut S, usize) -> T + Sync,
) -> Ran {
    run_priced(
        exec,
        |e, k| e.dispatch_cost(k),
        units,
        clock,
        leader,
        buf,
        scratch,
        kernel,
    )
}

/// [`run_stage`] with the section price `D(k)` of `exec` given by `price`.
#[allow(clippy::too_many_arguments)]
fn run_priced<S: Send, T: Send + Sync>(
    exec: &mut Exec<'_>,
    price: impl Fn(&Exec<'_>, NonZeroUsize) -> Duration,
    units: &[u64],
    clock: &impl Clock,
    leader: &mut Option<S>,
    buf: &mut StageBuf<T>,
    scratch: impl Fn() -> S + Sync,
    kernel: impl Fn(&mut S, usize) -> T + Sync,
) -> Ran {
    let StageBuf { out, queue: rest } = buf;
    out.clear();
    out.resize_with(units.len(), || None);
    rest.clear();
    let made = AtomicUsize::new(0);
    let panics = Panics::default();
    let attempt =
        |s: &mut Option<S>, i: usize| attempt_item(s, i, &scratch, &kernel, &made, &panics);
    let inline = |i: usize, out: &mut [Option<T>], leader: &mut Option<S>| {
        out[i] = attempt(leader, i);
    };
    // Zero-unit items carry no rate: run them first, unmeasured.
    for (i, &u) in units.iter().enumerate() {
        if u == 0 {
            inline(i, out, leader);
        } else {
            rest.push(i);
        }
    }
    let width = 'run: {
        let budget = exec.budget();
        let feasible = |n: usize| NonZeroUsize::new(n).map(|n| n.min(budget));
        let cost = |k| price(exec, k);
        let Some(cap) = feasible(rest.len()).filter(|_| exec.carries_team()) else {
            rest.iter().for_each(|&i| inline(i, out, leader));
            break 'run NonZeroUsize::MIN;
        };
        #[cfg(test)]
        if let Some(k) = FORCE_WIDTH.with(std::cell::Cell::get) {
            let k = k.min(cap);
            if k.get() >= 2 && cost(k) != Duration::MAX {
                lpt(rest, units);
                break 'run section(exec, k, rest, leader, &attempt, out);
            }
        }
        let Some(d_g) = cheapest_section(cap, cost) else {
            rest.iter().for_each(|&i| inline(i, out, leader));
            break 'run NonZeroUsize::MIN;
        };
        let t0 = clock.now();
        inline(rest[0], out, leader);
        let mut done = 1;
        let mut sampled = units[rest[0]];
        while done < rest.len() && clock.now().saturating_duration_since(t0) < d_g {
            inline(rest[done], out, leader);
            sampled += units[rest[done]];
            done += 1;
        }
        if done == rest.len() {
            break 'run NonZeroUsize::MIN;
        }
        let sample = Sample {
            elapsed_ns: u64::try_from(clock.now().saturating_duration_since(t0).as_nanos())
                .unwrap_or(u64::MAX),
            units: NonZeroU64::new(sampled).unwrap_or(NonZeroU64::MIN),
        };
        let rest = &mut rest[done..];
        lpt(rest, units);
        let sorted: Vec<u64> = rest.iter().map(|&i| units[i]).collect();
        let cap = feasible(rest.len()).unwrap_or(NonZeroUsize::MIN);
        let k = choose_width(&sorted, sample, cap, cost);
        #[cfg(test)]
        PLANS.with(|p| p.borrow_mut().push((done, sample, k)));
        if k.get() == 1 {
            // Item order again for the inline rest.
            rest.sort_unstable();
            rest.iter().for_each(|&i| inline(i, out, leader));
            break 'run k;
        }
        section(exec, k, rest, leader, &attempt, out)
    };
    // Every item was attempted; now report the lowest-index panic.
    if let Some((_, payload)) = panics
        .0
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
    {
        std::panic::resume_unwind(payload);
    }
    Ran {
        width,
        scratches: made.into_inner(),
    }
}

/// The section queue: largest units first; the sort is stable, so equal
/// units keep item order.
fn lpt(items: &mut [usize], units: &[u64]) {
    items.sort_by_key(|&i| std::cmp::Reverse(units[i]));
}

/// The panic of the lowest-index item that panicked.
#[derive(Default)]
struct Panics(Mutex<Option<(usize, Box<dyn Any + Send>)>>);

/// Runs item `i` on `s` (built first if absent). A panic is kept, and the
/// scratch it may have left inconsistent is dropped, so the stage goes on
/// with its other items and re-raises the lowest-index panic at its end.
fn attempt_item<S, T>(
    s: &mut Option<S>,
    i: usize,
    scratch: &impl Fn() -> S,
    kernel: &impl Fn(&mut S, usize) -> T,
    made: &AtomicUsize,
    panics: &Panics,
) -> Option<T> {
    let state = s.get_or_insert_with(|| {
        made.fetch_add(1, Ordering::Relaxed);
        scratch()
    });
    match std::panic::catch_unwind(AssertUnwindSafe(|| kernel(state, i))) {
        Ok(t) => Some(t),
        Err(payload) => {
            *s = None;
            let mut first = panics.0.lock().unwrap_or_else(PoisonError::into_inner);
            if first.as_ref().is_none_or(|&(j, _)| i < j) {
                *first = Some((i, payload));
            }
            None
        }
    }
}

/// One section of width `k` over `items` (in LPT order); returns the width.
/// Participant 0 runs on the leader's scratch, so a section holds at most
/// `k` scratch values.
///
/// Shares record no busy time: a per-share slowdown measured against a
/// one-item calibration sample amplifies that item's noise, and the planner
/// would never re-measure a width it priced out (`crate::width`). Narrowing
/// on measured loss needs section-level timings first.
fn section<S: Send, T: Send + Sync>(
    exec: &mut Exec<'_>,
    k: NonZeroUsize,
    items: &[usize],
    leader: &mut Option<S>,
    attempt: &(impl Fn(&mut Option<S>, usize) -> Option<T> + Sync),
    out: &mut [Option<T>],
) -> NonZeroUsize {
    #[cfg(test)]
    QUEUES.with(|q| q.borrow_mut().push(items.to_vec()));
    let next = AtomicUsize::new(0);
    let slots: Vec<OnceLock<T>> = items.iter().map(|_| OnceLock::new()).collect();
    let lead = Mutex::new(leader.take());
    let ran = exec.reborrow().limit(k).team(|team| {
        team.run_with(|p| {
            #[cfg(test)]
            let _no_cache = crate::cache::forbid_cache_access();
            let lock = || lead.lock().unwrap_or_else(PoisonError::into_inner);
            let mut s = if p.index() == 0 { lock().take() } else { None };
            loop {
                let t = next.fetch_add(1, Ordering::Relaxed);
                let Some(&i) = items.get(t) else { break };
                if let Some(v) = attempt(&mut s, i) {
                    let _ = slots[t].set(v);
                }
            }
            if p.index() == 0 {
                *lock() = s;
            }
        });
    });
    *leader = lead.into_inner().unwrap_or_else(PoisonError::into_inner);
    // A carried team is entered without acquisition, so this never refuses;
    // if it did, nothing ran and the leader takes every item.
    if ran.is_err() {
        for &i in items {
            out[i] = attempt(leader, i);
        }
        return NonZeroUsize::MIN;
    }
    for (&i, slot) in items.iter().zip(slots) {
        out[i] = slot.into_inner();
    }
    #[cfg(test)]
    SECTIONS.with(|s| s.borrow_mut().push(k.get()));
    k
}

#[cfg(test)]
thread_local! {
    /// Forces the next sections of this leader to the given width (capped by
    /// the stage), skipping calibration.
    pub(crate) static FORCE_WIDTH: std::cell::Cell<Option<NonZeroUsize>> = const { std::cell::Cell::new(None) };
    /// The queue of every section this leader ran, in dispatch order.
    pub(crate) static QUEUES: std::cell::RefCell<Vec<Vec<usize>>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Widths of the sections this leader ran.
    pub(crate) static SECTIONS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Per planned stage of this leader: items in the sample, the sample, the width.
    pub(crate) static PLANS: std::cell::RefCell<Vec<(usize, Sample, NonZeroUsize)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Forces sections of `width` on this thread until the guard drops.
#[cfg(test)]
#[must_use]
pub(crate) fn force_width(width: usize) -> impl Drop {
    struct Restore(Option<NonZeroUsize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_WIDTH.with(|f| f.set(self.0));
        }
    }
    Restore(FORCE_WIDTH.with(|f| f.replace(NonZeroUsize::new(width))))
}

#[cfg(test)]
mod tests;
