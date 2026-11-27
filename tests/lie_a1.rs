//! Isolated process: exact warm CGC miss counts exclude unrelated cache traffic.
mod common;

use fusion_symbols::{
    category::{CgcBlock, FusionCategory, GroupRepresentations, SymbolBlock},
    lie::{GroupId, LieError, LieGroup},
};

#[test]
fn labels_forms_and_rank_one_isomorphisms() {
    let su = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    assert_eq!(su.irrep(&[]), Err(LieError::InvalidLabel));
    assert_eq!(su.irrep(&[1, 0]), Err(LieError::InvalidLabel));
    assert_eq!(su.irrep(&[-1]), Err(LieError::InvalidLabel));
    assert_eq!(
        su.irrep(&[i64::from(u32::MAX) + 1]),
        Err(LieError::LabelOverflow)
    );
    let max = su.irrep(&[i64::from(u32::MAX)]).unwrap();
    assert_eq!(max.dim_exact(), (u64::from(u32::MAX) + 1));
    assert!(matches!(
        su.fusion(&max, &su.irrep(&[1]).unwrap()),
        Err(LieError::LabelOverflow)
    ));
    #[cfg(not(feature = "lie-generated"))]
    assert!(matches!(
        LieGroup::new(GroupId::su(3).unwrap()),
        Err(LieError::Unsupported {
            requires: "lie-generated"
        })
    ));
    let half = su.irrep(&[1]).unwrap();
    assert_eq!(
        half.weight_multiplicities(),
        [(vec![-1], 1), (vec![1], 1)].into()
    );
    for id in [
        GroupId::so(3).unwrap(),
        GroupId::spin(3).unwrap(),
        GroupId::sp(2).unwrap(),
        GroupId::su_quotient(2, 2).unwrap(),
    ] {
        let cat = LieGroup::new(id.clone()).unwrap();
        if id == GroupId::so(3).unwrap() || id == GroupId::su_quotient(2, 2).unwrap() {
            assert_eq!(cat.irrep(&[1]), Err(LieError::NotAdmissible));
        } else {
            assert!(cat.irrep(&[1]).is_ok());
        }
        let one = cat.irrep(&[2]).unwrap();
        let unit = cat.unit();
        let su_one = su.irrep(&[2]).unwrap();
        assert_ne!(one, su_one);
        assert_eq!(
            cat.f_symbol(&one, &one, &one, &one, &unit, &unit).unwrap(),
            su.f_symbol(&su_one, &su_one, &su_one, &su_one, &su.unit(), &su.unit())
                .unwrap()
        );
        assert_eq!(
            cat.r_symbol(&one, &one, &unit).unwrap(),
            su.r_symbol(&su_one, &su_one, &su.unit()).unwrap()
        );
        assert_eq!(cat.frobenius_schur_phase(&one).unwrap(), 1.0);
        assert_eq!(cat.twist(&one), 1.0);
        assert!(std::panic::catch_unwind(|| cat.dim(&su_one)).is_err());
    }
}

#[test]
#[should_panic(expected = "foreign Lie object")]
fn foreign_object_precedes_zero_channel() {
    let su = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let so = LieGroup::new(GroupId::so(3).unwrap()).unwrap();
    su.r_symbol(&su.unit(), &su.unit(), &so.irrep(&[2]).unwrap())
        .unwrap();
}

#[test]
fn warm_cgc_is_a_hit_and_oversized_or_empty_blocks_are_typed() {
    let cat = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    // A warm block is a CGC-tier hit, not a regeneration.
    let half = cat.irrep(&[1]).unwrap();
    cat.cgc(&half, &half, &cat.unit()).unwrap();
    let cached =
        fusion_symbols::cache::tier_stats(fusion_symbols::cache::CoefficientCacheTier::Cgc);
    cat.cgc(&half, &half, &cat.unit()).unwrap();
    let warm = fusion_symbols::cache::tier_stats(fusion_symbols::cache::CoefficientCacheTier::Cgc);
    assert_eq!(warm.misses, cached.misses);
    assert!(warm.hits > cached.hits);
    let huge = cat.irrep(&[i64::from(u32::MAX)]).unwrap();
    assert!(matches!(
        cat.cgc(&huge, &huge, &cat.unit()),
        Err(LieError::Unsupported { .. })
    ));
    assert_eq!(
        cat.cgc(&huge, &cat.unit(), &cat.unit()).unwrap().dims()[3],
        0
    );
    let zero = cat.unit();
    let half = cat.irrep(&[1]).unwrap();
    let f = cat
        .f_symbol(&half, &zero, &zero, &zero, &zero, &zero)
        .unwrap();
    assert_eq!(f.dims(), [0, 1, 1, 0]);
    assert!(f.as_slice().is_empty());
    let r = cat.r_symbol(&half, &zero, &zero).unwrap();
    assert_eq!(r.dims(), [0, 0]);
    assert!(r.as_slice().is_empty());
}

#[test]
fn every_valid_rank_one_datum_admits_the_unit() {
    for id in [
        GroupId::su(2).unwrap(),
        GroupId::su_quotient(2, 2).unwrap(),
        GroupId::so(3).unwrap(),
    ] {
        let cat = LieGroup::new(id.clone()).unwrap();
        assert_eq!(cat.group_id(), &id);
        assert!(id.admits(&[0], &[]));
        assert_eq!(cat.unit().charges().len(), 0);
    }
}
