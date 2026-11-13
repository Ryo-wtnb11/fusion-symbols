//! Batch cache fill of Lie-group coefficients on the caller's execution token.

use threadteam::Exec;

use super::{LieError, LieGroup, LieIrrep};
use crate::category::{FusionCategory, GroupRepresentations};

/// A batch of coefficients of one [`LieGroup`] for [`prefill`] to make
/// resident: CGC channels, F and R blocks, tensor products (every `N^c_{ab}`
/// and the fusion labels of `a ⊗ b`), and Frobenius–Schur phases.
///
/// Each method adds one item and returns the request, so items chain. An item
/// is evaluated exactly as the serial entry of the same name evaluates it:
/// the same dispatch (closed forms, rank-one and composite paths included),
/// the same cache lookups and the same error. `a_symbol` and `b_symbol` have
/// no item: `A^{ab}_c` reads `F^{ā a b}_b[𝟙, c]` and the Frobenius–Schur
/// phase of `a`, and `B^{ab}_c` reads `F^{a b b̄}_a[c, 𝟙]`, so request
/// `f_symbol(ā, a, b, b, 𝟙, c)` with `frobenius_schur_phase(a)`, or
/// `f_symbol(a, b, b̄, a, c, 𝟙)`.
///
/// # Panics
/// Every method panics on an object of another group, at insertion, as the
/// category methods do.
#[derive(Clone, Debug)]
pub struct PrefillRequest {
    group: LieGroup,
    items: Vec<Item>,
}

#[derive(Clone, Debug)]
enum Item {
    Cgc([LieIrrep; 3]),
    F([LieIrrep; 6]),
    R([LieIrrep; 3]),
    Fusion([LieIrrep; 2]),
    Phase(LieIrrep),
}

impl PrefillRequest {
    /// An empty request over `group`.
    pub fn new(group: &LieGroup) -> Self {
        Self {
            group: group.clone(),
            items: Vec::new(),
        }
    }

    fn push<const N: usize>(
        &mut self,
        labels: [&LieIrrep; N],
        item: fn([LieIrrep; N]) -> Item,
    ) -> &mut Self {
        for a in labels {
            self.group.label(a);
        }
        self.items.push(item(labels.map(LieIrrep::clone)));
        self
    }

    /// The channel `a ⊗ b → c` of [`GroupRepresentations::cgc`](crate::category::GroupRepresentations::cgc).
    pub fn cgc(&mut self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> &mut Self {
        self.push([a, b, c], Item::Cgc)
    }

    /// `F^{abc}_d[e, f]` of [`FusionCategory::f_symbol`](crate::category::FusionCategory::f_symbol).
    #[allow(clippy::too_many_arguments)]
    pub fn f_symbol(
        &mut self,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
        d: &LieIrrep,
        e: &LieIrrep,
        f: &LieIrrep,
    ) -> &mut Self {
        self.push([a, b, c, d, e, f], Item::F)
    }

    /// `R^{ab}_c` of [`FusionCategory::r_symbol`](crate::category::FusionCategory::r_symbol).
    pub fn r_symbol(&mut self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> &mut Self {
        self.push([a, b, c], Item::R)
    }

    /// The decomposition of `a ⊗ b`: every `N^c_{ab}` of
    /// [`FusionCategory::nsymbol`](crate::category::FusionCategory::nsymbol)
    /// and the labels of [`FusionCategory::fusion`](crate::category::FusionCategory::fusion).
    pub fn fusion(&mut self, a: &LieIrrep, b: &LieIrrep) -> &mut Self {
        self.push([a, b], Item::Fusion)
    }

    /// `κ_a` of [`FusionCategory::frobenius_schur_phase`](crate::category::FusionCategory::frobenius_schur_phase).
    pub fn frobenius_schur_phase(&mut self, a: &LieIrrep) -> &mut Self {
        self.push([a], |[a]| Item::Phase(a))
    }
}

/// Evaluates every item of `request` as its serial entry would, using up to
/// `exec.budget()` threads inside each generation. The order in which values
/// are published, and therefore which of them a full tier keeps, is not
/// specified.
///
/// Values are bit-identical to the serial entries', at every width. What is
/// retained follows each tier's policy exactly as for a serial query, so
/// `Ok(())` means every item was evaluated successfully, not that it is still
/// resident: a tier with a smaller budget may have evicted or rejected it.
/// A key another thread is generating is waited for; it is computed again
/// only if that thread failed or its value was not retained. Work is split
/// only on a token that carries a team (`Exec::carries_team`: minted by a
/// `threadteam::Team`, or a child of one); any other token runs serially on
/// the calling thread. A reset during the call retains nothing the call
/// generates after the reset began.
///
/// # Errors
/// Every item is attempted. A failure does not roll back successful items or
/// dependencies; retention remains subject to tier policy and the call's reset
/// epoch. The error of the lowest-index failing item is returned, with its
/// serial entry's error precedence (for example [`LieError::LabelOverflow`],
/// [`LieError::Unsupported`] or [`LieError::Generation`] for a failed gate).
///
/// # Examples
///
/// ```no_run
/// use fusion_symbols::category::{FusionCategory, GroupRepresentations};
/// use fusion_symbols::lie::{prefill, GroupId, LieGroup, PrefillRequest};
/// use threadteam::{Exec, Pool};
///
/// let su3 = LieGroup::new(GroupId::su(3)?)?;
/// let adj = su3.irrep(&[1, 1])?;
/// let mut request = PrefillRequest::new(&su3);
/// request
///     .fusion(&adj, &adj)
///     .cgc(&adj, &adj, &adj)
///     .f_symbol(&adj, &adj, &adj, &adj, &adj, &adj)
///     .r_symbol(&adj, &adj, &adj);
///
/// // The pool owner's team; a library passes the token it was given.
/// let team = Pool::builder().threads(8).build()?.into_team(8)?;
/// prefill(&request, team.exec(8))?;
/// // Serial, for a caller without threads:
/// prefill(&request, Exec::serial())?;
///
/// // Warm hits now.
/// let _ = su3.cgc(&adj, &adj, &adj)?;
/// let _ = su3.f_symbol(&adj, &adj, &adj, &adj, &adj, &adj)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn prefill(request: &PrefillRequest, exec: Exec<'_>) -> Result<(), LieError> {
    // Each item runs its serial entry unchanged on the calling thread, which
    // every token permits.
    let _ = exec;
    let group = &request.group;
    crate::cache::in_call_epoch(|| {
        let mut first = Ok(());
        for item in &request.items {
            #[cfg(test)]
            if let Some(hook) = BEFORE_ITEM.with(std::cell::Cell::get) {
                hook();
            }
            let result = match item {
                Item::Cgc([a, b, c]) => group.cgc(a, b, c).map(drop),
                Item::F([a, b, c, d, e, f]) => group.f_symbol(a, b, c, d, e, f).map(drop),
                Item::R([a, b, c]) => group.r_symbol(a, b, c).map(drop),
                Item::Fusion([a, b]) => group.fusion(a, b).map(drop),
                Item::Phase(a) => group.frobenius_schur_phase(a).map(drop),
            };
            if first.is_ok() {
                first = result;
            }
        }
        first
    })
}

#[cfg(test)]
thread_local! {
    /// A test hook run before each item, inside the call's reset epoch.
    static BEFORE_ITEM: std::cell::Cell<Option<fn()>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{cgc_channel_is_resident, reset, AFTER_GUARD, TEST_CACHE_LOCK};
    use crate::category::{CgcBlock, SymbolBlock};
    use crate::lie::{general, GroupId, RootSystem};

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Modules, channels and tensor products this thread generated.
    fn builds() -> ([usize; 2], usize) {
        (
            general::cgc::BUILDS.with(std::cell::Cell::get),
            general::roots::THREAD_PRODUCT_BUILDS.with(std::cell::Cell::get),
        )
    }

    /// A serial token, a plain pool token and a team token.
    fn on_every_token(mut f: impl FnMut(&str, Exec<'_>)) {
        f("serial", Exec::serial());
        let pool = threadteam::Pool::builder().threads(2).build().unwrap();
        f("pool", Exec::new(&pool, 2).unwrap());
        let team = pool.into_team(2).unwrap();
        f("team", team.exec(2));
    }

    /// What the serial entry of `item` returns, as bits: dimensions, then
    /// entries in storage order (dense for a CGC), fusion labels with their
    /// multiplicities, or the phase.
    fn observe(group: &LieGroup, item: &Item) -> Result<Vec<u64>, LieError> {
        let bits = |dims: &[usize], x: &[f64]| {
            dims.iter()
                .map(|&n| n as u64)
                .chain(x.iter().map(|y| y.to_bits()))
                .collect()
        };
        Ok(match item {
            Item::Cgc([a, b, c]) => {
                let x = group.cgc(a, b, c)?;
                bits(&x.dims(), &x.to_dense())
            }
            Item::F([a, b, c, d, e, f]) => {
                let x = group.f_symbol(a, b, c, d, e, f)?;
                bits(&x.dims(), x.as_slice())
            }
            Item::R([a, b, c]) => {
                let x = group.r_symbol(a, b, c)?;
                bits(&x.dims(), x.as_slice())
            }
            Item::Fusion([a, b]) => {
                let mut out = Vec::new();
                for c in group.fusion(a, b)? {
                    out.extend(c.dynkin().chain(c.charges()).map(|x| x as u64));
                    out.push(u64::from(group.nsymbol(a, b, &c)?));
                }
                out
            }
            Item::Phase(a) => vec![group.frobenius_schur_phase(a)?.to_bits()],
        })
    }

    fn su2() -> (LieGroup, impl Fn(i64) -> LieIrrep) {
        let g = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
        let h = g.clone();
        (g, move |dj| h.irrep(&[dj]).unwrap())
    }

    /// The requests of the value test: SU(2) closed forms and A1 channels,
    /// and with `lie-generated` the SU(3) octet (multiplicity two) and a
    /// generated D5 phase.
    fn requests() -> Vec<PrefillRequest> {
        let (g, j) = su2();
        let (one, h, s1) = (j(0), j(1), j(2));
        let mut su2 = PrefillRequest::new(&g);
        su2.fusion(&h, &s1)
            .cgc(&h, &h, &s1)
            .cgc(&s1, &h, &h)
            .f_symbol(&h, &h, &h, &h, &one, &one)
            .f_symbol(&h, &h, &h, &h, &s1, &s1)
            .r_symbol(&h, &h, &s1)
            .frobenius_schur_phase(&h);
        #[allow(unused_mut)]
        let mut all = vec![su2];
        #[cfg(feature = "lie-generated")]
        {
            let g = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
            let [three, adj] = [[1, 0], [1, 1]].map(|d| g.irrep(&d).unwrap());
            let mut su3 = PrefillRequest::new(&g);
            su3.fusion(&adj, &adj)
                .cgc(&adj, &adj, &adj)
                .f_symbol(&adj, &adj, &adj, &adj, &adj, &adj)
                .r_symbol(&adj, &adj, &adj)
                .frobenius_schur_phase(&three);
            let d5 = LieGroup::new(GroupId::spin(10).unwrap()).unwrap();
            let mut phase = PrefillRequest::new(&d5);
            phase.frobenius_schur_phase(&d5.irrep(&[0, 0, 0, 1, 0]).unwrap());
            all.extend([su3, phase]);
        }
        all
    }

    /// On every token, `prefill` publishes exactly what the serial entries
    /// compute: after it, every entry is served without generation and
    /// returns the serial bits, gauge and entry order.
    #[test]
    fn prefill_publishes_the_serial_entry_values_on_every_token() {
        let _guard = lock();
        for request in requests() {
            let serial = |r: &PrefillRequest| {
                r.items
                    .iter()
                    .map(|item| observe(&r.group, item).unwrap())
                    .collect::<Vec<_>>()
            };
            reset();
            let want = serial(&request);
            on_every_token(|name, exec| {
                reset();
                prefill(&request, exec).unwrap();
                let before = builds();
                assert_eq!(serial(&request), want, "{name}");
                assert_eq!(builds(), before, "{name}: entries after prefill generated");
            });
        }
    }

    /// Every item is attempted after a failure, earlier and later successes
    /// stay published, and the error is the lowest-index item's.
    #[test]
    fn lowest_index_error_wins_and_successes_are_kept() {
        let _guard = lock();
        let (g, j) = su2();
        let (zero, huge, max) = (j(0), j(1 << 31), j(i64::from(u32::MAX)));
        assert!(matches!(
            g.cgc(&huge, &zero, &huge),
            Err(LieError::Unsupported { .. })
        ));
        assert!(matches!(g.fusion(&max, &max), Err(LieError::LabelOverflow)));
        let (a, b, c, d) = (j(13), j(11), j(6), j(9));
        let resident = || {
            [
                cgc_channel_is_resident(RootSystem::A(1), [&[13], &[11], &[6]]),
                cgc_channel_is_resident(RootSystem::A(1), [&[11], &[9], &[6]]),
            ]
        };
        for unsupported_first in [true, false] {
            reset();
            let mut request = PrefillRequest::new(&g);
            request.cgc(&a, &b, &c);
            if unsupported_first {
                request.cgc(&huge, &zero, &huge).fusion(&max, &max);
            } else {
                request.fusion(&max, &max).cgc(&huge, &zero, &huge);
            }
            request.cgc(&b, &d, &c);
            let error = prefill(&request, Exec::serial()).unwrap_err();
            if unsupported_first {
                assert!(matches!(error, LieError::Unsupported { .. }));
            } else {
                assert!(matches!(error, LieError::LabelOverflow));
            }
            assert_eq!(resident(), [true; 2]);
        }
    }

    /// Resets the caches when the item of index `AT` is about to start;
    /// `ITEMS` counts the items started.
    static AT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static ITEMS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    fn reset_before_item() {
        use std::sync::atomic::Ordering::SeqCst;
        if ITEMS.fetch_add(1, SeqCst) == AT.load(SeqCst) {
            reset();
        }
    }

    /// The call is one generation: a reset before the first item's guard,
    /// between two items, or inside an item retains nothing the call
    /// generates afterwards; a later, distinct call retains again.
    #[test]
    fn reset_anywhere_in_the_call_retains_nothing_generated_afterwards() {
        let _guard = lock();
        let (g, j) = su2();
        let (a, b, c, d) = (j(13), j(11), j(6), j(9));
        let mut request = PrefillRequest::new(&g);
        request.cgc(&a, &b, &c).cgc(&b, &d, &c);
        let resident = || {
            [
                cgc_channel_is_resident(RootSystem::A(1), [&[13], &[11], &[6]]),
                cgc_channel_is_resident(RootSystem::A(1), [&[11], &[9], &[6]]),
            ]
        };
        let hooked = |before_item: Option<usize>, in_item: bool| {
            reset();
            if let Some(at) = before_item {
                AT.store(at, std::sync::atomic::Ordering::SeqCst);
                ITEMS.store(0, std::sync::atomic::Ordering::SeqCst);
                BEFORE_ITEM.with(|h| h.set(Some(reset_before_item)));
            }
            if in_item {
                fn hook() {
                    std::thread::spawn(reset).join().unwrap();
                }
                AFTER_GUARD.with(|h| h.set(Some(hook)));
            }
            prefill(&request, Exec::serial()).unwrap();
            BEFORE_ITEM.with(|h| h.set(None));
            assert!(AFTER_GUARD.with(std::cell::Cell::get).is_none());
            resident()
        };
        // Before the first guard: the call's epoch predates the reset.
        assert_eq!(hooked(Some(0), false), [false; 2]);
        // Between the items: the second item's generation is discarded.
        assert!(!hooked(Some(1), false)[1]);
        // Inside the first item's generation: both items are discarded.
        assert_eq!(hooked(None, true), [false; 2]);
        // A later, distinct call retains again.
        prefill(&request, Exec::serial()).unwrap();
        assert_eq!(resident(), [true; 2]);
    }

    /// Composite items keep Product's dispatch: a zero torus axis answers
    /// before any factor generates, an aggregate overflow and the first of
    /// several factor failures are the serial entry's errors.
    #[cfg(feature = "lie-generated")]
    #[test]
    fn composite_items_keep_the_product_dispatch() {
        let _guard = lock();
        let first = |r: &PrefillRequest| observe(&r.group, &r.items[0]).map(drop);
        // SU(3)×U(1): the zero-axis F builds no CGC; the conserving control does.
        let g = LieGroup::new(GroupId::cover(&[RootSystem::A(2)], 1).unwrap()).unwrap();
        let q = |n| g.irrep_with_charges(&[1, 1], &[n]).unwrap();
        reset();
        for (e, generates) in [(0, false), (2, true)] {
            let mut request = PrefillRequest::new(&g);
            request.f_symbol(&q(1), &q(1), &q(1), &q(3), &q(e), &q(2));
            let before = builds().0;
            prefill(&request, Exec::serial()).unwrap();
            assert_eq!(builds().0 != before, generates, "torus charge {e}");
            assert_eq!(first(&request), Ok(()));
        }
        // 32 SU(3) factors: the 6^32 labels of 8 ⊗ 8 overflow.
        let g = LieGroup::new(GroupId::cover(&[RootSystem::A(2); 32], 0).unwrap()).unwrap();
        let adj = g.irrep(&[1; 64]).unwrap();
        let mut request = PrefillRequest::new(&g);
        request.fusion(&adj, &adj);
        assert_eq!(
            prefill(&request, Exec::serial()),
            Err(LieError::LabelOverflow)
        );
        assert_eq!(first(&request), Err(LieError::LabelOverflow));
        // Two generated phases both fail: the error names the first factor.
        struct Clear;
        impl Drop for Clear {
            fn drop(&mut self) {
                general::cgc::FAIL_PHASE.with(|f| f.set(None));
            }
        }
        fn injected() -> LieError {
            LieError::Generation(super::super::GenerationError(std::sync::Arc::new(
                std::io::Error::other("injected phase failure"),
            )))
        }
        let g = LieGroup::new(GroupId::cover(&[RootSystem::D(5); 2], 0).unwrap()).unwrap();
        let mut request = PrefillRequest::new(&g);
        request.frobenius_schur_phase(&g.irrep(&[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]).unwrap());
        let _clear = Clear;
        general::cgc::FAIL_PHASE.with(|f| f.set(Some(injected)));
        let error = prefill(&request, Exec::serial()).unwrap_err();
        assert_eq!(Err(error.clone()), first(&request));
        let source = std::error::Error::source(&error).unwrap().source().unwrap();
        assert!(matches!(
            source.downcast_ref::<super::super::CompositeError>(),
            Some(super::super::CompositeError(
                crate::product::ProductError::Factor { index: 0, .. }
            ))
        ));
    }

    /// Runs [`cache_policy_child`] in a fresh process per budget policy.
    #[test]
    fn cache_policy_contracts_in_fresh_processes() {
        for policy in ["fit", "zero", "undersized"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "lie::prefill::tests::cache_policy_child",
                    "--ignored",
                    "--test-threads=1",
                ])
                .env("FS_PREFILL_POLICY", policy)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "{policy} child failed:\n{stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    /// With budgets that fit, every requested entry and a repeated prefill
    /// are served from the cache (unchanged misses, nothing built), and
    /// rank-one F fills the F tier; a resident F needs no CGC after the CGC
    /// tier is trimmed. With zero or undersized budgets, prefill still
    /// generates cold items, returns a failing item's error, and the entries
    /// keep their values.
    #[test]
    #[ignore = "child process of cache_policy_contracts_in_fresh_processes; light; run by that test in every CI test job"]
    fn cache_policy_child() {
        use crate::cache::{
            cache_budgets, configure_cache_budgets, tier_stats, CoefficientCacheBudgets,
            CoefficientCacheTier as Tier,
        };
        let misses = || Tier::ALL.iter().map(|&t| tier_stats(t).misses).sum::<u64>();
        let policy = std::env::var("FS_PREFILL_POLICY").unwrap();
        match policy.as_str() {
            "fit" => {}
            "zero" => configure_cache_budgets(CoefficientCacheBudgets::disabled()).unwrap(),
            "undersized" => configure_cache_budgets(
                CoefficientCacheBudgets::default()
                    .with_limit(Tier::Cgc, 1024)
                    .with_limit(Tier::FSymbol, 64),
            )
            .unwrap(),
            _ => unreachable!("unknown policy {policy}"),
        }
        if policy != "fit" {
            for request in requests() {
                let values = || {
                    let r = &request;
                    r.items
                        .iter()
                        .map(|x| observe(&r.group, x).unwrap())
                        .collect::<Vec<_>>()
                };
                let want = values();
                // Every request holds a cold channel or phase after a reset,
                // so prefill itself must generate.
                reset();
                let before = builds().0;
                prefill(&request, Exec::serial()).unwrap();
                assert_ne!(builds().0, before, "{policy}: prefill generated nothing");
                prefill(&request, Exec::serial()).unwrap();
                assert_eq!(values(), want, "{policy}");
                for &tier in Tier::ALL {
                    assert!(tier_stats(tier).bytes <= cache_budgets().limit(tier));
                }
            }
            let (g, j) = su2();
            let (zero, huge) = (j(0), j(1 << 31));
            let mut request = PrefillRequest::new(&g);
            request.cgc(&huge, &zero, &huge);
            assert!(matches!(
                prefill(&request, Exec::serial()),
                Err(LieError::Unsupported { .. })
            ));
            return;
        }
        let (g, j) = su2();
        let (one, h) = (j(0), j(1));
        let mut request = PrefillRequest::new(&g);
        request.f_symbol(&h, &h, &h, &h, &one, &one);
        prefill(&request, Exec::serial()).unwrap();
        let before = tier_stats(Tier::FSymbol).misses;
        g.f_symbol(&h, &h, &h, &h, &one, &one).unwrap();
        assert_eq!(tier_stats(Tier::FSymbol).misses, before);
        for request in requests() {
            prefill(&request, Exec::serial()).unwrap();
            let before = (misses(), builds());
            // Every requested entry is resident: querying it misses nothing.
            for item in &request.items {
                observe(&request.group, item).unwrap();
            }
            assert_eq!((misses(), builds()), before, "entries after prefill");
            prefill(&request, Exec::serial()).unwrap();
            assert_eq!((misses(), builds()), before, "repeated prefill");
        }
        #[cfg(feature = "lie-generated")]
        {
            let g = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
            let adj = g.irrep(&[1, 1]).unwrap();
            let mut request = PrefillRequest::new(&g);
            request.f_symbol(&adj, &adj, &adj, &adj, &adj, &adj);
            prefill(&request, Exec::serial()).unwrap();
            crate::cache::trim_to(Tier::Cgc, 0);
            let before = builds();
            prefill(&request, Exec::serial()).unwrap();
            assert_eq!((tier_stats(Tier::Cgc).entries, builds()), (0, before));
        }
    }
}
