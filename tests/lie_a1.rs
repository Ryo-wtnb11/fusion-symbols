//! Isolated process: exact warm 3j miss counts exclude unrelated cache traffic.
mod common;

use fusion_symbols::{
    category::{CgcBlock, FusionCategory, GroupRepresentations, SymbolBlock},
    lie::{su2::clebsch_gordan, GroupId, LieError, LieGroup},
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
        GroupId::psu(2).unwrap(),
    ] {
        let cat = LieGroup::new(id.clone()).unwrap();
        if id == GroupId::so(3).unwrap() || id == GroupId::psu(2).unwrap() {
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
fn dense_cgc_matches_exact_in_ascending_magnetic_order() {
    let cat = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    // Every block for doubled spins a, b ≤ 8 and every admissible c (plus one
    // inadmissible c each), bitwise against the exact closed form, so the
    // signs of zeros are compared too.
    for a in 0..=8u32 {
        for b in 0..=8u32 {
            for c in 0..=a + b + 1 {
                let [ia, ib, ic] = [a, b, c].map(|dj| cat.irrep(&[i64::from(dj)]).unwrap());
                let tensor = cat.cgc(&ia, &ib, &ic).unwrap();
                let n = cat.nsymbol(&ia, &ib, &ic).unwrap() as usize;
                assert_eq!(
                    tensor.dims(),
                    [a as usize + 1, b as usize + 1, c as usize + 1, n]
                );
                if n == 0 {
                    continue;
                }
                for x in 0..=a {
                    for y in 0..=b {
                        for z in 0..=c {
                            let exact = clebsch_gordan(
                                a,
                                2 * x as i32 - a as i32,
                                b,
                                2 * y as i32 - b as i32,
                                c,
                                2 * z as i32 - c as i32,
                            )
                            .to_f64();
                            assert_eq!(
                                tensor
                                    .get([x as usize, y as usize, z as usize, 0])
                                    .to_bits(),
                                exact.to_bits(),
                                "{a} {b} {c} [{x},{y},{z}]"
                            );
                        }
                    }
                }
            }
        }
    }
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
        GroupId::psu(2).unwrap(),
        GroupId::so(3).unwrap(),
    ] {
        let cat = LieGroup::new(id.clone()).unwrap();
        assert_eq!(cat.group_id(), &id);
        assert!(id.admits(&[0], &[]));
        assert_eq!(cat.unit().charges().len(), 0);
    }
}
