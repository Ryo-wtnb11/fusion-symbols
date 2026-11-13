use std::sync::atomic::AtomicU64;
use std::sync::{Barrier, Mutex};

use threadteam::{Pool, Team};

use super::*;

/// A clock whose `i`-th read is `ticks[i]` ns after an origin (the last tick
/// repeats), counting its reads.
struct Script {
    origin: Instant,
    ticks: Vec<u64>,
    reads: AtomicU64,
}

impl Script {
    fn new(ticks: &[u64]) -> Self {
        Self {
            origin: Instant::now(),
            ticks: ticks.to_vec(),
            reads: AtomicU64::new(0),
        }
    }

    fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }
}

impl Clock for Script {
    fn now(&self) -> Instant {
        let i = self.reads.fetch_add(1, Ordering::Relaxed) as usize;
        let tick = self.ticks[i.min(self.ticks.len() - 1)];
        self.origin + Duration::from_nanos(tick)
    }
}

/// [`run_stage`] with its results collected in item order.
fn run_vec<S: Send, T: Send + Sync>(
    exec: &mut Exec<'_>,
    units: &[u64],
    clock: &impl Clock,
    leader: &mut Option<S>,
    scratch: impl Fn() -> S + Sync,
    kernel: impl Fn(&mut S, usize) -> T + Sync,
) -> (Vec<T>, Ran) {
    let mut buf = StageBuf::default();
    let ran = run_stage(exec, units, clock, leader, &mut buf, scratch, kernel);
    (buf.drain().collect(), ran)
}

/// [`run_priced`] with its results collected in item order.
fn priced_vec<S: Send, T: Send + Sync>(
    exec: &mut Exec<'_>,
    price: impl Fn(&Exec<'_>, NonZeroUsize) -> Duration,
    units: &[u64],
    clock: &impl Clock,
    leader: &mut Option<S>,
    scratch: impl Fn() -> S + Sync,
    kernel: impl Fn(&mut S, usize) -> T + Sync,
) -> (Vec<T>, Ran) {
    let mut buf = StageBuf::default();
    let ran = run_priced(
        exec, price, units, clock, None, leader, &mut buf, scratch, kernel,
    );
    (buf.drain().collect(), ran)
}

fn new_team(width: usize) -> Team {
    Pool::builder()
        .threads(width)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap()
        .into_team(width)
        .unwrap()
}

fn ns(n: u64) -> Duration {
    Duration::from_nanos(n)
}

/// Runs a stage of `units` whose item `i` returns `i`, with `D(k) = d` for
/// every width, and checks that results come back in item order. Returns how
/// it ran and the plans it made.
fn stage(
    exec: &mut Exec<'_>,
    units: &[u64],
    d: Duration,
    clock: &impl Clock,
) -> (Ran, Vec<(usize, Sample, NonZeroUsize)>) {
    PLANS.with(|p| p.borrow_mut().clear());
    let (out, ran) = priced_vec(exec, |_, _| d, units, clock, &mut None, || (), |(), i| i);
    assert_eq!(out, (0..units.len()).collect::<Vec<_>>());
    (ran, PLANS.with(|p| p.take()))
}

/// r4 §2: a zero cheapest section samples exactly one item, before the
/// planner runs, and the planner returns a width.
#[test]
fn zero_dispatch_cost_samples_exactly_one_item() {
    let team = new_team(4);
    // Reads: t0 = 0, then 10 after the first item, then the sample end.
    let clock = Script::new(&[0, 10, 10]);
    let (_, plans) = stage(&mut team.exec(4), &[5; 8], ns(0), &clock);
    let [(done, sample, k)] = plans[..] else {
        panic!("one plan: {plans:?}")
    };
    assert_eq!((done, sample.units.get(), sample.elapsed_ns), (1, 5, 10));
    // T(k) = ĉ·LS_k with ĉ = 2 ns: 4 workers over seven items of 5.
    assert_eq!(k.get(), 4);
}

/// r4 §2: a first clock delta already above a positive `D_g` still leaves a
/// one-item sample.
#[test]
fn first_item_over_the_threshold_samples_one_item() {
    let team = new_team(4);
    let clock = Script::new(&[0, 1_000, 1_000]);
    let (_, plans) = stage(&mut team.exec(4), &[5; 8], ns(100), &clock);
    let [(done, sample, _)] = plans[..] else {
        panic!("one plan: {plans:?}")
    };
    assert_eq!((done, sample.units.get(), sample.elapsed_ns), (1, 5, 1_000));
    // Negative control: below the threshold, the prefix grows.
    let clock = Script::new(&[0, 10, 20, 30, 120, 120]);
    let (_, plans) = stage(&mut team.exec(4), &[5; 8], ns(100), &clock);
    assert_eq!(plans[0].0, 4);
}

/// r4 §2: a zero clock delta prices the stage at zero and selects width 1.
#[test]
fn zero_clock_delta_selects_width_one() {
    let team = new_team(4);
    let clock = Script::new(&[7]);
    let (ran, plans) = stage(&mut team.exec(4), &[5; 8], ns(0), &clock);
    assert_eq!(plans[0].1.elapsed_ns, 0);
    assert_eq!(ran.width.get(), 1);
    // Negative control: one nanosecond more and the stage goes parallel.
    let clock = Script::new(&[7, 8, 8]);
    assert!(
        stage(&mut team.exec(4), &[5; 8], ns(0), &clock)
            .0
            .width
            .get()
            > 1
    );
}

/// r4 §2: stages that cannot price a section never read the clock: all-zero
/// units, one item, a cap of one and all-`MAX` costs.
#[test]
fn unpriceable_stages_run_inline_without_the_clock() {
    let team = new_team(4);
    for (units, budget, d) in [
        (&[0u64; 6][..], 4, ns(0)),
        (&[5][..], 4, ns(0)),
        (&[5; 6][..], 1, ns(0)),
        (&[5; 6][..], 4, Duration::MAX),
    ] {
        let clock = Script::new(&[0, 10]);
        let (ran, plans) = stage(&mut team.exec(budget), units, d, &clock);
        assert_eq!(
            (ran.width.get(), clock.reads()),
            (1, 0),
            "{units:?} {budget} {d:?}"
        );
        assert!(plans.is_empty());
    }
    // Negative control: two items on a team reach the planner.
    let clock = Script::new(&[0, 10, 10]);
    assert_eq!(stage(&mut team.exec(4), &[5, 5], ns(0), &clock).1.len(), 1);
}

/// Only a team token forks: a plain pool token and the serial token run
/// every item inline without the clock.
#[test]
fn only_team_tokens_fork() {
    let pool = Pool::builder()
        .threads(4)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    for mut exec in [Exec::serial(), Exec::new(&pool, 4).unwrap()] {
        let clock = Script::new(&[0, 10]);
        let (ran, _) = stage(&mut exec, &[5; 8], ns(0), &clock);
        assert_eq!((ran.width.get(), clock.reads()), (1, 0));
    }
}

/// The text of a panic payload.
fn text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

/// Runs a 12-item stage whose kernel panics with "item i" where `fail(i)`
/// holds, catching the re-raised panic: the items attempted, the items that
/// panicked, and the re-raised message.
fn panicking_stage(
    exec: &mut Exec<'_>,
    d: Duration,
    clock: &impl Clock,
    fail: impl Fn(usize) -> bool + Sync,
) -> (Vec<usize>, Vec<usize>, String) {
    let attempted = Mutex::new(Vec::new());
    let failed = Mutex::new(Vec::new());
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        priced_vec(
            exec,
            |_, _| d,
            &[1; 12],
            clock,
            &mut None,
            || (),
            |(), i| {
                attempted.lock().unwrap().push(i);
                if fail(i) {
                    failed.lock().unwrap().push(i);
                    panic!("item {i}");
                }
            },
        )
    }));
    let message = text(&*caught.expect_err("the stage re-raises a panic"));
    let (mut a, mut f) = (
        attempted.into_inner().unwrap(),
        failed.into_inner().unwrap(),
    );
    a.sort_unstable();
    f.sort_unstable();
    (a, f, message)
}

/// A panic never stops a stage: every item is attempted, on the first
/// inline item of a serial token, on the first calibration item of a team
/// token, when every section item panics on every participant, and when
/// only non-leader shares panic; then the lowest-index panic is re-raised.
#[test]
fn panics_never_stop_the_stage() {
    let all: Vec<usize> = (0..12).collect();
    let (a, f, m) = panicking_stage(&mut Exec::serial(), ns(0), &SystemClock, |i| i == 0);
    assert_eq!((a, f, m), (all.clone(), vec![0], "item 0".to_string()));

    let team = new_team(3);
    let clock = Script::new(&[0, 10, 10]);
    let (a, f, m) = panicking_stage(&mut team.exec(3), ns(0), &clock, |i| i == 0);
    assert_eq!((a, f, m), (all.clone(), vec![0], "item 0".to_string()));
    assert_eq!(
        SECTIONS.with(|s| s.take()).len(),
        1,
        "the rest ran in a section"
    );

    let _force = force_width(3);
    let (a, f, m) = panicking_stage(&mut team.exec(3), ns(0), &SystemClock, |_| true);
    assert_eq!(
        (a, f.clone(), m),
        (all.clone(), all.clone(), "item 0".to_string())
    );

    // Only shares on member threads panic. The first three items each wait
    // for the others, so three participants hold one item at once.
    let leader = std::thread::current().id();
    let entered = AtomicUsize::new(0);
    let barrier = Barrier::new(3);
    let (a, f, m) = panicking_stage(&mut team.exec(3), ns(0), &SystemClock, |_| {
        if entered.fetch_add(1, Ordering::SeqCst) < 3 {
            barrier.wait();
        }
        std::thread::current().id() != leader
    });
    assert_eq!(a, all);
    assert!(!f.is_empty() && f.len() < 12, "{f:?}");
    assert_eq!(m, format!("item {}", f[0]));
}

/// A section holds at most one scratch per participant, the leader's
/// included: participant 0 runs on the scratch the leader carries, which
/// comes back to it after the section.
#[test]
fn sections_reuse_the_leaders_scratch() {
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    struct Scratch;
    impl Scratch {
        fn new() -> Self {
            let now = LIVE.fetch_add(1, Ordering::SeqCst) + 1;
            PEAK.fetch_max(now, Ordering::SeqCst);
            Scratch
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            LIVE.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let team = new_team(3);
    let _force = force_width(3);
    let mut leader = Some(Scratch::new());
    let entered = AtomicUsize::new(0);
    let barrier = Barrier::new(3);
    let (_, ran) = run_vec(
        &mut team.exec(3),
        &[1; 9],
        &SystemClock,
        &mut leader,
        Scratch::new,
        |_, _| {
            // Three participants hold their scratch at once.
            if entered.fetch_add(1, Ordering::SeqCst) < 3 {
                barrier.wait();
            }
        },
    );
    assert_eq!(ran.width.get(), 3);
    assert_eq!(ran.scratches, 2, "two members built one each");
    assert_eq!(
        PEAK.load(Ordering::SeqCst),
        3,
        "at most one per participant"
    );
    assert!(leader.is_some(), "the leader's scratch came back");
    drop(leader);
    assert_eq!(LIVE.load(Ordering::SeqCst), 0);
}

/// Forced and planned sections both dispatch largest units first, ties in
/// item order: `[1, 1, 6, 6, 10]` is queued as items 4, 2, 3, 0, 1, not in
/// item order (whose greedy makespan on two workers is 17 against 12).
#[test]
fn sections_dispatch_largest_units_first() {
    let team = new_team(2);
    let units = [1, 1, 6, 6, 10];
    let queue = |exec: &mut Exec<'_>, clock: &Script| {
        QUEUES.with(|q| q.borrow_mut().clear());
        priced_vec(
            exec,
            |_, _| ns(0),
            &units,
            clock,
            &mut None,
            || (),
            |(), i| i,
        );
        QUEUES.with(|q| q.take())
    };
    {
        let _force = force_width(2);
        assert_eq!(
            queue(&mut team.exec(2), &Script::new(&[0])),
            [vec![4, 2, 3, 0, 1]]
        );
    }
    // Planned: item 0 is the calibration sample, the rest is queued.
    let planned = queue(&mut team.exec(2), &Script::new(&[0, 10, 10]));
    assert_eq!(planned, [vec![4, 2, 3, 1]]);
    assert_ne!(planned[0], [1, 2, 3, 4], "not item order");
}

/// Results come back in item order through real sections of every width,
/// including widths above the item count and a narrow child token, and each
/// participant builds its scratch at most once.
#[test]
fn sections_return_results_in_item_order() {
    let team = new_team(5);
    let units: Vec<u64> = (0..23).map(|i| 1 + (i * 7) % 5).collect();
    for k in [2, 3, 4, 5, 8] {
        for budget in [5, 2] {
            let _force = force_width(k);
            SECTIONS.with(|s| s.borrow_mut().clear());
            let made = AtomicUsize::new(0);
            let (out, ran) = run_vec(
                &mut team.exec(budget),
                &units,
                &SystemClock,
                &mut None,
                || made.fetch_add(1, Ordering::Relaxed),
                |_, i| i * i,
            );
            assert_eq!(out, (0..23).map(|i| i * i).collect::<Vec<_>>());
            let width = k.min(budget);
            assert_eq!(SECTIONS.with(|s| s.take()), [width]);
            assert_eq!(ran.width.get(), width);
            assert!(ran.scratches <= width && ran.scratches == made.into_inner());
        }
    }
    // A child token of a team section: two children of width 2.
    let team4 = new_team(4);
    team4.exec(4).fork(2, |_, mut child| {
        let _force = force_width(4);
        let (out, ran) = run_vec(
            &mut child,
            &units,
            &SystemClock,
            &mut None,
            || (),
            |(), i| i + 1,
        );
        assert_eq!(out, (1..=23).collect::<Vec<_>>());
        assert!(ran.width.get() <= 2);
    });
}

/// Shares hold the no-cache guard: a kernel that touches the coefficient
/// cache inside a section panics, and the panic reaches the caller after
/// every item was attempted.
#[test]
fn shares_cannot_touch_the_cache() {
    let team = new_team(3);
    let attempted = AtomicUsize::new(0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _force = force_width(3);
        run_vec(
            &mut team.exec(3),
            &[1; 12],
            &SystemClock,
            &mut None,
            || (),
            |(), i| {
                attempted.fetch_add(1, Ordering::Relaxed);
                if i == 5 {
                    let _ = crate::cache::tier_stats(crate::cache::CoefficientCacheTier::Cgc);
                }
            },
        )
    }));
    let message = text(&*result.unwrap_err());
    assert!(
        message.contains("accessed the coefficient cache"),
        "{message}"
    );
    assert_eq!(attempted.into_inner(), 12);
    // Negative control: the same kernel inline on the leader may read it.
    let (out, _) = run_vec(
        &mut Exec::serial(),
        &[1; 2],
        &SystemClock,
        &mut None,
        || (),
        |(), _| crate::cache::tier_stats(crate::cache::CoefficientCacheTier::Cgc).entries,
    );
    assert_eq!(out.len(), 2);
}

/// Errors are per item: every item runs, and the lowest failing index is
/// the first `Err` in item order, whichever share produced it.
#[test]
fn errors_keep_item_order_and_every_item_runs() {
    let team = new_team(4);
    let _force = force_width(4);
    let attempted = Mutex::new(Vec::new());
    let (out, _) = run_vec(
        &mut team.exec(4),
        &(0..40).map(|i| 40 - i).collect::<Vec<u64>>(),
        &SystemClock,
        &mut None,
        || (),
        |(), i| {
            attempted.lock().unwrap().push(i);
            if i % 13 == 7 {
                Err(i)
            } else {
                Ok(i)
            }
        },
    );
    let mut attempted = attempted.into_inner().unwrap();
    attempted.sort_unstable();
    assert_eq!(attempted, (0..40).collect::<Vec<_>>());
    assert_eq!(out.into_iter().collect::<Result<Vec<_>, _>>(), Err(7));
}

/// A reused `StageBuf` holds only the current stage's results: an early
/// stop of `drain` drops the unread tail, a smaller or empty next stage
/// sees none of the old values, and a panicking stage leaves nothing that a
/// later stage returns.
#[test]
fn reused_stage_buffers_hold_only_the_current_stage() {
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    struct Payload(usize);
    impl Payload {
        fn new(v: usize) -> Self {
            LIVE.fetch_add(1, Ordering::SeqCst);
            Payload(v)
        }
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            LIVE.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let live = || LIVE.load(Ordering::SeqCst);
    let mut buf = StageBuf::default();
    let run = |n: usize, base: usize, fail: Option<usize>, buf: &mut StageBuf<Payload>| {
        run_stage(
            &mut Exec::serial(),
            &vec![1; n],
            &SystemClock,
            &mut None,
            buf,
            || (),
            |(), i| {
                if Some(i) == fail {
                    panic!("item {i}");
                }
                Payload::new(base + i)
            },
        )
    };

    run(6, 100, None, &mut buf);
    assert_eq!(live(), 6);
    // Stop after two results: the unread four are dropped with the drain.
    assert_eq!(
        buf.drain().take(2).map(|p| p.0).collect::<Vec<_>>(),
        [100, 101]
    );
    assert_eq!(live(), 0);

    run(3, 200, None, &mut buf);
    assert_eq!(
        buf.drain().map(|p| p.0).collect::<Vec<_>>(),
        [200, 201, 202]
    );
    assert_eq!(live(), 0);

    // A panicking stage leaves partial results behind; the next stage,
    // empty and then of two items, returns none of them.
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(5, 300, Some(1), &mut buf);
    }));
    assert_eq!(text(&*caught.unwrap_err()), "item 1");
    assert_eq!(live(), 4, "items 0, 2, 3 and 4 completed");
    run(0, 400, None, &mut buf);
    assert_eq!(live(), 0, "the empty stage dropped the partial results");
    assert_eq!(buf.drain().count(), 0);
    run(2, 500, None, &mut buf);
    assert_eq!(buf.drain().map(|p| p.0).collect::<Vec<_>>(), [500, 501]);
    assert_eq!(live(), 0);
}

/// The memory width is the planner's ceiling, not a clip of its choice:
/// eight equal items of 10 units on a team of 8, with `D(4) = 100` and
/// `D(k) = 1` otherwise, and an allowance of three blocks beyond the serial
/// peak, so `k_mem = 4`. Width 4 does not pay at `D(4) = 100` against the
/// serial 70 ns of the seven items left after the one-item sample, so the
/// planner runs 3 (`T(3) = 1 + 30`), not the clipped 4 (`100 + 20`); without
/// the ceiling it would run 7.
#[test]
fn memory_width_caps_the_planner() {
    let team = new_team(8);
    let price = |_: &Exec<'_>, k: NonZeroUsize| ns(if k.get() == 4 { 100 } else { 1 });
    let charge = [5usize; 8];
    let memory = |rest: &[usize]| memory_width(&charge, rest, 5 + 3 * 5);
    assert_eq!(memory(&[1, 2, 3, 4, 5, 6, 7]), 4);
    let run = |memory: Option<MemoryWidth<'_>>| {
        // The sample: item 0 of 10 units in 10 ns.
        let clock = Script::new(&[0, 10, 10]);
        SECTIONS.with(|s| s.borrow_mut().clear());
        let mut buf = StageBuf::default();
        run_priced(
            &mut team.exec(8),
            price,
            &[10; 8],
            &clock,
            memory,
            &mut None,
            &mut buf,
            || (),
            |(), i| i,
        );
        assert_eq!(buf.drain().collect::<Vec<_>>(), (0..8).collect::<Vec<_>>());
        SECTIONS.with(|s| s.take())
    };
    assert_eq!(run(Some(&memory)), [3]);
    assert_eq!(run(None), [7]);
}

/// A bounded section's scratch is its own: the leader's is dropped before
/// dispatch, so three participants hold three scratch values, and every
/// participant drops its own before the section returns.
#[test]
fn bounded_sections_own_their_scratch() {
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    struct Scratch;
    impl Scratch {
        fn new() -> Self {
            let now = LIVE.fetch_add(1, Ordering::SeqCst) + 1;
            PEAK.fetch_max(now, Ordering::SeqCst);
            Scratch
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            LIVE.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let team = new_team(3);
    let _force = force_width(3);
    let mut leader = Some(Scratch::new());
    let entered = AtomicUsize::new(0);
    let barrier = Barrier::new(3);
    let ran = run_bounded_stage(
        &mut team.exec(3),
        &[1; 9],
        &SystemClock,
        &|rest| rest.len(),
        &mut leader,
        &mut StageBuf::<()>::default(),
        Scratch::new,
        |_, _| {
            if entered.fetch_add(1, Ordering::SeqCst) < 3 {
                barrier.wait();
            }
        },
    );
    assert_eq!(ran.width.get(), 3);
    assert_eq!(ran.scratches, 3, "every participant built its own");
    assert_eq!(
        PEAK.load(Ordering::SeqCst),
        3,
        "the leader's was dropped first"
    );
    assert!(leader.is_none());
    assert_eq!(LIVE.load(Ordering::SeqCst), 0, "shares dropped theirs");
}
