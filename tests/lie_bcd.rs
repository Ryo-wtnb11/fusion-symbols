//! Isolated process: reset/warm counter and catalog lifecycle assertions exclude unrelated cache traffic.
mod common;

#[cfg(feature = "lie-generated")]
use fusion_symbols::category::FusionCategory;
use fusion_symbols::lie::{CatalogStats, GroupId, LieError, LieGroup};

#[test]
fn excluded_rank_and_absent_catalog() {
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    assert_eq!(su2.catalog_stats().unwrap(), CatalogStats::default());
    su2.reset_catalog().unwrap();
    assert!(matches!(
        LieGroup::with_catalog_budget(GroupId::su(2).unwrap(), 0),
        Err(LieError::Unsupported { .. })
    ));
    #[cfg(not(feature = "lie-generated"))]
    for id in [
        GroupId::spin(5).unwrap(),
        GroupId::sp(4).unwrap(),
        GroupId::so(6).unwrap(),
    ] {
        assert_eq!(
            LieGroup::new(id),
            Err(LieError::Unsupported {
                requires: "lie-generated"
            })
        );
    }
}

#[cfg(feature = "lie-generated")]
#[test]
fn labels_weights_zero_blocks_and_catalog_budget() {
    use fusion_symbols::{
        cache,
        category::{CgcBlock, FusionCategory, FusionStyle, GroupRepresentations},
    };
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };
    let cat = LieGroup::new(GroupId::spin(5).unwrap()).unwrap();
    let s = cat.irrep(&[0, 1], &[]).unwrap();
    assert_eq!(s.dim_exact(), 4.into());
    assert!(s.gt_patterns().is_none());
    assert_eq!(
        s.weight_multiplicities(),
        [vec![0, 1], vec![1, -1], vec![-1, 1], vec![0, -1]]
            .into_iter()
            .map(|w| (w, 1))
            .collect()
    );
    assert_eq!(cat.frobenius_schur_indicator(&s), -1.0);
    assert_eq!(cat.fusion_style(), FusionStyle::Generic);
    assert_eq!(cat.irrep(&[1], &[]), Err(LieError::InvalidLabel));
    assert_eq!(cat.irrep(&[-1, 0], &[]), Err(LieError::InvalidLabel));
    assert_eq!(cat.irrep(&[i64::MAX, 1], &[]), Err(LieError::LabelOverflow));
    assert_eq!(
        format!("{cat:?}"),
        format!("LieGroup {{ group: {:?} }}", GroupId::spin(5).unwrap())
    );
    let v = cat.irrep(&[1, 0], &[]).unwrap();
    let adj = cat.irrep(&[0, 2], &[]).unwrap();
    assert_eq!(
        cat.f_symbol(&v, &v, &v, &v, &v, &adj).unwrap().dims(),
        [0, 0, 1, 1]
    );
    assert_eq!(cat.r_symbol(&v, &v, &v).unwrap().dim(), 0);
    assert_eq!(cat.cgc(&v, &v, &v).unwrap().dims(), [5, 5, 5, 0]);
    assert_eq!(adj.weight_multiplicities()[&vec![0, 0]], 2);

    for (id, label, weights) in [
        (
            GroupId::sp(4).unwrap(),
            vec![1, 0],
            vec![vec![1, 0], vec![-1, 0], vec![-1, 1], vec![1, -1]],
        ),
        (
            GroupId::spin(6).unwrap(),
            vec![0, 0, 1],
            vec![
                vec![0, 0, 1],
                vec![1, 0, -1],
                vec![-1, 1, 0],
                vec![0, -1, 0],
            ],
        ),
    ] {
        let group = LieGroup::new(id).unwrap();
        let a = group.irrep(&label, &[]).unwrap();
        assert_eq!(
            a.weight_multiplicities(),
            weights.into_iter().map(|w| (w, 1)).collect()
        );
    }
    for (id, label) in [
        (GroupId::sp(4).unwrap(), vec![1, 0]),
        (GroupId::spin(6).unwrap(), vec![0, 0, 1]),
        (GroupId::so(6).unwrap(), vec![0, 2, 0]),
        (GroupId::spin(10).unwrap(), vec![0, 0, 0, 0, 1]),
    ] {
        let group = LieGroup::new(id).unwrap();
        let a = group.irrep(&label, &[]).unwrap();
        let dual = group.dual(&a);
        let unit = group.unit();
        // Only these four channels are needed; a full product builds large unrelated irreps.
        for (left, right, out) in [
            (&a, &dual, &unit),
            (&unit, &a, &a),
            (&dual, &a, &unit),
            (&a, &unit, &a),
        ] {
            group.cgc(left, right, out).unwrap();
        }
        let f = group
            .f_symbol(&a, &dual, &a, &a, &unit, &unit)
            .unwrap()
            .at(0, 0, 0, 0);
        assert!((group.frobenius_schur_phase(&a) - f / f.abs()).abs() < 1e-12);
    }
    cache::reset();
    let product = cat.fusion(&s, &s).unwrap().collect::<Vec<_>>();
    let before = cache::tier_stats(cache::CoefficientCacheTier::BcdCgc);
    assert_eq!(cat.fusion(&s, &s).unwrap().collect::<Vec<_>>(), product);
    for c in product {
        assert_eq!(cat.nsymbol(&s, &s, &c), 1);
    }
    let warm = cache::tier_stats(cache::CoefficientCacheTier::BcdCgc);
    assert_eq!(warm.misses, before.misses);
    assert!(warm.hits > before.hits);
    cache::reset();
    let seed_bytes = cat.catalog_stats().unwrap().bytes;
    let tiny = LieGroup::with_catalog_budget(GroupId::spin(5).unwrap(), seed_bytes).unwrap();
    assert_eq!(tiny, cat);
    let hash = |x: &LieGroup| {
        let mut h = DefaultHasher::new();
        x.hash(&mut h);
        h.finish()
    };
    assert_eq!(hash(&tiny), hash(&cat));
    assert_eq!(tiny.catalog_stats().unwrap().budget, seed_bytes);
    assert!(matches!(
        tiny.cgc(&s, &s, &adj),
        Err(LieError::Generation(_))
    ));
    let baseline = cat.catalog_stats().unwrap();
    let clone = cat.clone();
    cat.cgc(&s, &s, &adj).unwrap();
    let grown = clone.catalog_stats().unwrap();
    assert!(grown.bytes > baseline.bytes);
    assert!(grown.entries > baseline.entries);
    assert_eq!(grown.budget, baseline.budget);
    clone.reset_catalog().unwrap();
    assert_eq!(cat.catalog_stats().unwrap(), baseline);
}

#[cfg(feature = "lie-generated")]
#[test]
fn doubled_weight_overflow_is_a_label_error() {
    for (id, label) in [
        (GroupId::sp(4).unwrap(), vec![1 << 62, 0]),
        (GroupId::sp(4).unwrap(), vec![0, 1 << 62]),
        (GroupId::sp(4).unwrap(), vec![1 << 61, 1 << 61]),
        (GroupId::spin(5).unwrap(), vec![1 << 62, 0]),
        (GroupId::spin(5).unwrap(), vec![1 << 61, 1 << 62]),
        (GroupId::spin(6).unwrap(), vec![1 << 62, 0, 0]),
        (GroupId::spin(6).unwrap(), vec![1 << 61, 1 << 62, 0]),
    ] {
        assert_eq!(
            LieGroup::new(id).unwrap().irrep(&label, &[]),
            Err(LieError::LabelOverflow)
        );
    }
}

#[cfg(feature = "lie-generated")]
#[test]
#[should_panic(expected = "foreign Lie object")]
fn cover_object_is_foreign_to_quotient() {
    let spin = LieGroup::new(GroupId::spin(5).unwrap()).unwrap();
    let so = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
    so.f_symbol(
        &spin.unit(),
        &so.unit(),
        &so.unit(),
        &so.unit(),
        &so.unit(),
        &so.unit(),
    )
    .unwrap();
}
