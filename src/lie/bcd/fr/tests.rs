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
    let v = cat.irrep(&[0, 1], &[]).unwrap();
    let adj = cat.irrep(&[2, 0], &[]).unwrap();
    let other = cat.irrep(&[0, 2], &[]).unwrap();
    let first = cat.f_symbol(&one, &v, &v, &adj, &v, &adj).unwrap();
    let cgc = cat.cgc(&v, &v, &adj).unwrap();
    // The same sweep populates every channel, including this unrequested one.
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
    let v = cat.irrep(&[0, 1], &[]).unwrap();
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
    let a = cat.irrep(&[0, 1], &[]).unwrap();
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
    let g = cat.irrep(&[0, 1, 1], &[]).unwrap();
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
    let cat = LieGroup::with_catalog_budget(group, needed).unwrap();
    let s = cat.irrep(&[0, 1], &[]).unwrap();
    let got = cat.cgc(&s, &s, &cat.unit()).unwrap();
    assert_eq!(cat.catalog_stats().unwrap(), (needed, needed));
    for m2 in 0..4 {
        for m1 in 0..4 {
            assert_eq!(got.at([m1, m2, 0, 0]), expected.copy(0)[m1 + 4 * m2]);
        }
    }
}
