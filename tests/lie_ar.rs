//! Isolated process: reset/warm miss counts exclude unrelated cache traffic.
#![cfg(feature = "lie-generated")]
mod common;

use fusion_symbols::{
    cache::{self, CoefficientCacheTier as Tier},
    category::{CgcBlock, FusionCategory, FusionStyle, GroupRepresentations},
    lie::{GroupId, LieError, LieGroup},
};

#[test]
fn labels_zero_channels_and_warm_generation() {
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    assert_eq!(cat.fusion_style(), FusionStyle::Generic);
    assert_eq!(cat.irrep(&[1]), Err(LieError::InvalidLabel));
    assert_eq!(cat.irrep(&[-1, 0]), Err(LieError::InvalidLabel));
    assert_eq!(cat.irrep(&[i64::MAX, 1]), Err(LieError::LabelOverflow));
    let quotient = LieGroup::new(GroupId::psu(3).unwrap()).unwrap();
    assert_eq!(quotient.irrep(&[1, 0]), Err(LieError::NotAdmissible));
    assert_eq!(quotient.irrep(&[1, 1]).unwrap().dim_exact(), 8.into());
    let three = cat.irrep(&[1, 0]).unwrap();
    let eight = cat.irrep(&[1, 1]).unwrap();
    let unit = cat.unit();
    assert_eq!(three.gt_patterns().unwrap().len(), 3);
    let a1 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    assert!(a1.unit().gt_patterns().is_none());
    assert_eq!(eight.weight_multiplicities()[&vec![0, 0]], 2);
    assert_eq!(cat.frobenius_schur_indicator(&three), 0.0);
    assert_eq!(cat.nsymbol(&eight, &eight, &eight), 2);
    let block = cat
        .f_symbol(&three, &three, &three, &three, &unit, &unit)
        .unwrap();
    assert_eq!(block.dims(), [0, 1, 0, 1]);
    assert!(block.data().is_empty());
    assert_eq!(cat.r_symbol(&three, &three, &unit).unwrap().dim(), 0);
    assert_eq!(cat.cgc(&three, &three, &unit).unwrap().dims(), [3, 3, 1, 0]);
    assert_eq!(cat.authority_fingerprint(),b"racah:sun-gt:ref=sunrep-0.4:basis=gt-order:gauge=qrpos-cref:descent=ladder-lstsq:tol=sunrep-tol-tier:epoch=1");

    cache::reset();
    let cold: Vec<_> = cat.fusion(&eight, &eight).unwrap().collect();
    let misses = cache::tier_stats(Tier::SunProduct).misses;
    assert_eq!(
        cat.fusion(&eight, &eight).unwrap().collect::<Vec<_>>(),
        cold
    );
    assert_eq!(cache::tier_stats(Tier::SunProduct).misses, misses);
    let cgc = cat.cgc(&eight, &eight, &eight).unwrap();
    let misses = cache::tier_stats(Tier::SunCgc).misses;
    assert_eq!(cat.cgc(&eight, &eight, &eight).unwrap(), cgc);
    assert_eq!(cache::tier_stats(Tier::SunCgc).misses, misses);
    let product = fusion_symbols::product::Product::new((cat.clone(), a1.clone()));
    let channel = (eight.clone(), a1.irrep(&[2]).unwrap());
    let first = product.cgc(&channel, &channel, &channel).unwrap();
    let tiers = [Tier::SunCgc, Tier::SunProduct, Tier::ThreeJ];
    let [cgc_stats, product_stats, three_j] = tiers.map(cache::tier_stats);
    assert_eq!(product.cgc(&channel, &channel, &channel).unwrap(), first);
    let [cgc_warm, product_warm, three_j_warm] = tiers.map(cache::tier_stats);
    assert_eq!(cgc_warm.misses, cgc_stats.misses);
    assert_eq!(product_warm.misses, product_stats.misses);
    assert!(cgc_warm.hits > cgc_stats.hits);
    assert_eq!(three_j_warm.misses, three_j.misses);
    let r = cat.r_symbol(&eight, &eight, &eight).unwrap();
    let tiers = [Tier::SunCgc, Tier::SunProduct];
    let misses = tiers.map(|tier| cache::tier_stats(tier).misses);
    assert_eq!(cat.r_symbol(&eight, &eight, &eight).unwrap(), r);
    assert_eq!(tiers.map(|tier| cache::tier_stats(tier).misses), misses);
    let f = cat
        .f_symbol(&eight, &eight, &eight, &eight, &eight, &eight)
        .unwrap();
    let tiers = [Tier::SunF, Tier::SunCgc];
    let misses = tiers.map(|tier| cache::tier_stats(tier).misses);
    assert_eq!(
        cat.f_symbol(&eight, &eight, &eight, &eight, &eight, &eight)
            .unwrap(),
        f
    );
    assert_eq!(tiers.map(|tier| cache::tier_stats(tier).misses), misses);
}

#[test]
#[should_panic(expected = "foreign Lie object")]
fn foreign_object_precedes_zero() {
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let su4 = LieGroup::new(GroupId::su(4).unwrap()).unwrap();
    su3.r_symbol(&su3.unit(), &su3.unit(), &su4.unit()).unwrap();
}
