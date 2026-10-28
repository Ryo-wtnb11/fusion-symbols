use crate::{
    cache,
    category::{CgcBlock, FusionCategory, GroupRepresentations},
    lie::{GroupId, LieError, LieGroup},
};
use std::{sync::mpsc, time::Duration};

#[test]
fn warm_calls_do_not_lock_or_resweep() {
    let _guard = cache::TEST_CACHE_LOCK.lock().unwrap();
    cache::reset();
    let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
    let one = cat.unit();
    let v = cat.irrep(&[0, 1]).unwrap();
    let adj = cat.irrep(&[2, 0]).unwrap();
    let other = cat.irrep(&[0, 2]).unwrap();
    let first = cat.f_symbol(&one, &v, &v, &adj, &v, &adj).unwrap();
    let cgc = cat.cgc(&v, &v, &adj).unwrap();
    // Optional canonical-parent harvesting publishes this coherent sibling.
    assert!(cache::bcd_cgc_is_resident(
        v.bcd().unwrap(),
        v.bcd().unwrap(),
        other.bcd().unwrap()
    ));
    let sweeps = super::CGC_SWEEPS.load(std::sync::atomic::Ordering::Relaxed);
    let clone = cat.clone();
    let lock = cat.catalog.as_ref().unwrap().lock().unwrap();
    let (tx, rx) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let f = clone.f_symbol(&one, &v, &v, &adj, &v, &adj).unwrap();
        let c = clone.cgc(&v, &v, &adj).unwrap();
        let r = clone.r_symbol(&v, &v, &adj).unwrap();
        tx.send((f, c, r)).unwrap();
    });
    let result = rx.recv_timeout(Duration::from_secs(5));
    drop(lock);
    thread.join().unwrap();
    let (f, c, r) = result.expect("warm F/CGC/R must finish while catalog is locked");
    assert_eq!(first, f);
    assert_eq!(cgc, c);
    assert_eq!(r.dim(), 1);
    assert_eq!(
        super::CGC_SWEEPS.load(std::sync::atomic::Ordering::Relaxed),
        sweeps
    );
}

#[test]
fn cached_r_is_bit_identical_to_the_raw_join() {
    let _guard = cache::TEST_CACHE_LOCK.lock().unwrap();
    let bits = |r: &super::RBlock<f64>| {
        (
            r.dim(),
            r.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        )
    };
    let b2 = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
    let c3 = LieGroup::new(GroupId::sp(6).unwrap()).unwrap();
    let samples: [(&LieGroup, [&[i64]; 3], usize); 7] = [
        (&b2, [&[1, 0], &[1, 0], &[0, 2]], 1),
        (&b2, [&[0, 2], &[0, 2], &[0, 2]], 1),
        (&b2, [&[1, 0], &[1, 0], &[1, 0]], 0),
        (&b2, [&[2, 0], &[2, 0], &[2, 0]], 1),
        (&c3, [&[1, 0, 0], &[1, 0, 0], &[0, 1, 0]], 1),
        (&c3, [&[2, 0, 0], &[2, 0, 0], &[2, 0, 0]], 1),
        (&c3, [&[1, 0, 0], &[1, 0, 0], &[1, 0, 0]], 0),
    ];
    for (cat, labels, dim) in samples {
        let [a, b, c] = labels.map(|d| cat.irrep(d).unwrap());
        let [a, b, c] = [&a, &b, &c].map(|x| x.bcd().unwrap());
        let catalog = cat.catalog.as_ref().unwrap();
        cache::trim_to(cache::CoefficientCacheTier::BcdR, 0);
        let cold = super::r_symbol(catalog, a, b, c).unwrap();
        let warm = super::r_symbol(catalog, a, b, c).unwrap();
        let raw = super::r_block_raw(&mut super::BcdFamily { cat: catalog }, a, b, c).unwrap();
        assert_eq!(cold.dim(), dim, "{labels:?}");
        assert_eq!(bits(&cold), bits(&raw), "cold R differs from the raw join");
        assert_eq!(bits(&warm), bits(&cold), "warm R differs from cold R");
    }
    // Errors are not retained: a repeat reports the same typed error.
    let v = b2.irrep(&[1, 0]).unwrap();
    let six = c3.irrep(&[1, 0, 0]).unwrap();
    let (v, six) = (v.bcd().unwrap(), six.bcd().unwrap());
    let catalog = b2.catalog.as_ref().unwrap();
    let first = super::r_symbol(catalog, v, six, v).unwrap_err();
    assert_eq!(super::r_symbol(catalog, v, six, v).unwrap_err(), first);
}

#[test]
fn poisoned_catalog_is_a_generation_error() {
    let _guard = cache::TEST_CACHE_LOCK.lock().unwrap();
    cache::reset();
    let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
    let clone = cat.clone();
    assert!(std::thread::spawn(move || {
        let _lock = clone.catalog.as_ref().unwrap().lock().unwrap();
        panic!("poison the test catalog");
    })
    .join()
    .is_err());
    let v = cat.irrep(&[0, 1]).unwrap();
    assert!(matches!(
        cat.cgc(&v, &v, &cat.unit()),
        Err(LieError::Generation(_))
    ));
    assert!(matches!(cat.catalog_stats(), Err(LieError::Generation(_))));
    assert!(matches!(cat.reset_catalog(), Err(LieError::Generation(_))));
}

#[test]
fn warm_product_is_shared_without_resweeping() {
    let _guard = cache::TEST_CACHE_LOCK.lock().unwrap();
    cache::reset();
    let cat = LieGroup::new(GroupId::spin(5).unwrap()).unwrap();
    let a = cat.irrep(&[0, 1]).unwrap();
    let first = cat.fusion(&a, &a).unwrap().collect::<Vec<_>>();
    let product = super::super::shared_directproduct(a.bcd().unwrap(), a.bcd().unwrap()).unwrap();
    assert_eq!(cat.fusion(&a, &a).unwrap().collect::<Vec<_>>(), first);
    for c in first {
        assert_eq!(cat.nsymbol(&a, &a, &c), 1);
    }
    let second = super::super::shared_directproduct(a.bcd().unwrap(), a.bcd().unwrap()).unwrap();
    assert!(std::sync::Arc::ptr_eq(&product, &second));
    let before_owners = std::sync::Arc::strong_count(&product);
    cache::trim_to(cache::CoefficientCacheTier::BcdCgc, 0);
    assert_eq!(std::sync::Arc::strong_count(&product), before_owners - 1);
    assert_eq!(product.len(), 3, "external products survive trimming");
}

#[test]
fn d3_adjoint_square_84_channel_cgc_is_coherent_or_bricks() {
    // The QSpace coherence guard on the CGC path:
    // a coupled channel is either coherent (Ok) or refused with a typed
    // BasisIncoherent naming irrep/product/residual — never a silent wrong CGC.
    //
    // WHICH happens for a near-rank-deficient channel (the D3 84 = (0,2,2))
    // is platform-dependent (a deterministic near-tie resolution: rotated on
    // dev macOS ARM at residual 3.65, may be coherent on CI Linux x86), so we
    // assert the disjunction. The guard's firing is pinned deterministically by
    // `sweep::tests::coherence_residual_detects_degenerate_rotation`.
    use crate::lie::bcd::{directproduct, CatalogError};
    let g = super::super::Irrep::from_dynkin(super::super::Series::D, &[0, 1, 1]).unwrap();
    assert_eq!(
        directproduct(&g, &g).unwrap().get(&g).copied().unwrap(),
        2,
        "exact layer must predict OM=2 for the D3 adjoint square"
    );

    let mut cat = super::super::CanonicalCatalog::new(super::super::Series::D, 3).unwrap();
    let eightyfour = super::super::Irrep::from_dynkin(super::super::Series::D, &[0, 2, 2]).unwrap();
    match cat.cgc(&g, &g, &eightyfour) {
        Ok(_) => {}
        Err(CatalogError::BasisIncoherent {
            irrep,
            product,
            residual,
        }) => {
            // If it bricks, it must name THIS channel and be O(1), not noise.
            assert_eq!(irrep, vec![0, 2, 2]);
            assert_eq!(product, (vec![0, 1, 1], vec![0, 1, 1]));
            assert!(
                residual > 1e-3,
                "residual {residual} must be O(1), not noise"
            );
        }
        other => panic!("expected Ok or BasisIncoherent, got {other:?}"),
    }
}

#[test]
fn d3_adjoint_f_symbol_closes_or_bricks() {
    let _guard = cache::TEST_CACHE_LOCK.lock().unwrap();
    let cat = LieGroup::new(GroupId::so(6).unwrap()).unwrap();
    let g = cat.irrep(&[0, 1, 1]).unwrap();
    match cat.f_symbol(&g, &g, &g, &g, &g, &g) {
        Ok(_) => {}
        Err(LieError::Generation(error))
            if matches!(
                error.0.downcast_ref::<super::super::CatalogError>(),
                Some(super::super::CatalogError::BasisIncoherent { .. })
            ) => {}
        other => panic!("expected Ok or BasisIncoherent, got {other:?}"),
    }
}

#[test]
fn cgc_budget_covers_only_the_requested_channel() {
    use super::super::{CanonicalCatalog, Irrep, Series};
    let _guard = cache::TEST_CACHE_LOCK.lock().unwrap();
    cache::reset();
    let group = GroupId::spin(5).unwrap();
    let s = Irrep::from_dynkin_in(&group, &[0, 1]).unwrap();
    let one = Irrep::trivial(Series::B, 2).unwrap();
    let mut probe = CanonicalCatalog::new(Series::B, 2).unwrap();
    let expected = probe.cgc(&s, &s, &one).unwrap();
    let needed = probe.bytes();
    for mode in ["cold", "retained", "evicted"] {
        cache::reset();
        if mode != "cold" {
            let mut large = CanonicalCatalog::new(Series::B, 2).unwrap();
            for c in super::super::directproduct(&s, &s).unwrap().keys() {
                cache::with_bcd_test_sweep(&s, &s, c, |sweep| {
                    large.cgc_with(&s, &s, c, Some(sweep), false)
                })
                .unwrap();
            }
            if mode == "evicted" {
                cache::trim_to(cache::CoefficientCacheTier::BcdCgc, 0);
            }
        }
        let cat = LieGroup::with_catalog_budget(group.clone(), needed).unwrap();
        let s = cat.irrep(&[0, 1]).unwrap();
        let got = cat.cgc(&s, &s, &cat.unit()).unwrap();
        let stats = cat.catalog_stats().unwrap();
        assert_eq!(
            (stats.entries, stats.bytes, stats.budget),
            (probe.len(), needed, needed)
        );
        for m2 in 0..4 {
            for m1 in 0..4 {
                assert_eq!(
                    got.at([m1, m2, 0, 0]).to_bits(),
                    expected.copy(0)[m1 + 4 * m2].to_bits()
                );
            }
        }
    }
}

#[test]
fn resident_ordered_products_are_swept_once() {
    use super::super::sweep::DECOMPOSITIONS;
    let _guard = cache::TEST_CACHE_LOCK.lock().unwrap();
    for mode in ["public", "f", "r"] {
        cache::reset();
        let cat = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
        let a = cat.irrep(&[2, 0]).unwrap();
        let channels: Vec<_> = cat.fusion(&a, &a).unwrap().collect();
        DECOMPOSITIONS.set(0); // exclude defining seed
        if mode == "f" {
            cat.f_symbol(&a, &a, &a, &a, &a, &a).unwrap();
        } else if mode == "r" {
            cat.r_symbol(&a, &a, &a).unwrap();
        }
        if mode != "public" {
            assert_eq!(DECOMPOSITIONS.get(), 2, "cold {mode}");
            assert_eq!(cat.catalog_stats().unwrap().entries, 5);
        }
        let mut increments = Vec::new();
        for c in &channels {
            let before = DECOMPOSITIONS.get();
            cat.cgc(&a, &a, c).unwrap();
            increments.push(DECOMPOSITIONS.get() - before);
        }
        assert_eq!(
            increments,
            if mode == "public" {
                [2, 0, 0, 1, 1, 0]
            } else {
                [0, 0, 0, 1, 1, 0]
            }
        );
        assert_eq!(DECOMPOSITIONS.get(), 4);
        assert_eq!(cat.catalog_stats().unwrap().entries, 7);
        for c in &channels {
            cat.cgc(&a, &a, c).unwrap();
        }
        cat.f_symbol(&a, &a, &a, &a, &a, &a).unwrap();
        cat.r_symbol(&a, &a, &a).unwrap();
        assert_eq!(DECOMPOSITIONS.get(), 4, "warm {mode}");
        // The last two channels were lazy; replay them while the catalog is locked.
        let clone = cat.clone();
        let a = a.clone();
        let c = channels[4].clone();
        let lock = cat.catalog.as_ref().unwrap().lock().unwrap();
        let (tx, rx) = mpsc::channel();
        let worker = std::thread::spawn(move || tx.send(clone.cgc(&a, &a, &c).unwrap()).unwrap());
        let result = rx.recv_timeout(Duration::from_secs(5));
        drop(lock);
        worker.join().unwrap();
        result.expect("lazy published channel must avoid the catalog lock");
    }
}

#[test]
fn simultaneous_siblings_and_canonical_parent_collision_complete() {
    use super::super::sweep::DECOMPOSITIONS;
    use std::sync::{Arc, Barrier};
    let _guard = cache::TEST_CACHE_LOCK.lock().unwrap();
    // In the last case one thread needs 55's parent 14⊗14 while another
    // requests a channel of that very product.
    for (independent, collision) in [(false, false), (true, false), (true, true)] {
        cache::reset();
        let first = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
        let second = if independent {
            LieGroup::new(GroupId::so(5).unwrap()).unwrap()
        } else {
            first.clone()
        };
        let start = Arc::new(Barrier::new(3));
        let (tx, rx) = mpsc::channel();
        let workers: Vec<_> = [first, second]
            .into_iter()
            .enumerate()
            .map(|(i, cat)| {
                let start = start.clone();
                let tx = tx.clone();
                std::thread::spawn(move || {
                    let a = cat.irrep(&[2, 0]).unwrap();
                    let c = cat
                        .irrep(if collision && i == 1 {
                            &[4, 0]
                        } else if i == 0 {
                            &[0, 0]
                        } else {
                            &[0, 2]
                        })
                        .unwrap();
                    DECOMPOSITIONS.set(0);
                    start.wait();
                    let got = cat.cgc(&a, &a, &c).unwrap();
                    tx.send((got.dims(), DECOMPOSITIONS.get())).unwrap();
                })
            })
            .collect();
        start.wait();
        let results: Vec<_> = (0..2)
            .map(|_| {
                rx.recv_timeout(Duration::from_secs(10))
                    .expect("raw/channel lock order must complete")
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(
            results.iter().map(|(_, n)| n).sum::<usize>(),
            2,
            "shared raw computation"
        );
    }
}

#[test]
fn cached_and_uncached_fr_preserve_every_bit_and_fingerprint() {
    use super::super::{CanonicalCatalog, Irrep};
    use crate::frcore::{Family, MEntry};
    struct Uncached(CanonicalCatalog);
    impl Family for Uncached {
        type Irrep = Irrep;
        type Error = LieError;
        fn overflow() -> LieError {
            LieError::LabelOverflow
        }
        fn mult(&mut self, a: &Irrep, b: &Irrep, c: &Irrep) -> Result<usize, LieError> {
            super::mult(a, b, c)
        }
        fn cgc_entries(
            &mut self,
            a: &Irrep,
            b: &Irrep,
            c: &Irrep,
        ) -> Result<Vec<MEntry>, LieError> {
            super::sparse_entries(a, &self.0.cgc(a, b, c)?)
        }
    }
    let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let _guard = cache::TEST_CACHE_LOCK.lock().unwrap();
    for (id, labels, middle) in [
        (GroupId::so(5).unwrap(), vec![2, 0], vec![2, 0]),
        (GroupId::sp(4).unwrap(), vec![1, 0], vec![2, 0]),
        (GroupId::spin(5).unwrap(), vec![0, 1], vec![0, 2]),
        (GroupId::spin(6).unwrap(), vec![0, 1, 0], vec![1, 0, 0]),
        (GroupId::so(6).unwrap(), vec![0, 1, 1], vec![0, 1, 1]),
    ] {
        cache::reset();
        let cat = LieGroup::new(id.clone()).unwrap();
        let a = cat.irrep(&labels).unwrap();
        let e = cat.irrep(&middle).unwrap();
        let d = cat.dual(&a);
        let (ia, output, ie) = (a.bcd().unwrap(), d.bcd().unwrap(), e.bcd().unwrap());
        let mut uncached = Uncached(CanonicalCatalog::new(ia.series(), ia.rank()).unwrap());
        let expected_f = super::f_block_raw(&mut uncached, ia, ia, ia, output, ie, ie).unwrap();
        let expected_r = super::r_block_raw(&mut uncached, ia, ia, ie).unwrap();
        let fingerprint = cat.authority_fingerprint().to_vec();
        for public_first in [false, true] {
            cache::reset();
            let cat = LieGroup::new(id.clone()).unwrap();
            let a = cat.irrep(&labels).unwrap();
            let e = cat.irrep(&middle).unwrap();
            let d = cat.dual(&a);
            if public_first {
                cat.cgc(&a, &a, &e).unwrap();
            }
            for _ in 0..2 {
                let f = cat.f_symbol(&a, &a, &a, &d, &e, &e).unwrap();
                let r = cat.r_symbol(&a, &a, &e).unwrap();
                assert_eq!(
                    (f.dims(), bits(f.data())),
                    (expected_f.dims(), bits(expected_f.data()))
                );
                assert_eq!(
                    (r.dim(), bits(r.data())),
                    (expected_r.dim(), bits(expected_r.data()))
                );
                assert_eq!(cat.authority_fingerprint(), fingerprint);
            }
        }
    }
}
