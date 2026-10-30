use crate::common::counting_alloc;
use fusion_symbols::category::SymbolBlock;
use fusion_symbols::{
    category::{FBlock, FusionCategory, RBlock},
    lie::{GroupId, LieGroup},
};

#[test]
fn inline_blocks_and_warm_a1_symbols_allocate_nothing() {
    let _serial = crate::common::counting_alloc::serial();
    let cat = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = cat.irrep(&[1]).unwrap();
    let zero = cat.unit();
    let expected = cat
        .f_symbol(&half, &half, &half, &half, &zero, &zero)
        .unwrap();
    let stats =
        fusion_symbols::cache::tier_stats(fusion_symbols::cache::CoefficientCacheTier::FSymbol);
    let before = counting_alloc::calls();
    let label = cat.irrep(&[1]).unwrap();
    std::hint::black_box(label.clone());
    std::hint::black_box(cat.group_id().clone());
    assert_eq!(label.charges().len(), 0);
    std::hint::black_box(FBlock::one(0.5f64));
    std::hint::black_box(RBlock::one(-1.0f64));
    let f = cat
        .f_symbol(&half, &half, &half, &half, &zero, &zero)
        .unwrap();
    let r = cat.r_symbol(&half, &half, &zero).unwrap();
    let empty_f = cat
        .f_symbol(&half, &zero, &zero, &zero, &zero, &zero)
        .unwrap();
    let empty_r = cat.r_symbol(&half, &zero, &zero).unwrap();
    let fusion_count = cat.fusion(&half, &half).unwrap().count();
    let after = counting_alloc::calls();
    assert_eq!(after, before);
    assert_eq!(f, expected);
    assert_eq!(r.get([0, 0]), -1.0);
    assert!(empty_f.as_slice().is_empty());
    assert!(empty_r.as_slice().is_empty());
    assert_eq!(fusion_count, 2);
    let warm =
        fusion_symbols::cache::tier_stats(fusion_symbols::cache::CoefficientCacheTier::FSymbol);
    assert_eq!(warm.misses, stats.misses);
    assert_eq!(warm.hits, stats.hits + 1);
}
