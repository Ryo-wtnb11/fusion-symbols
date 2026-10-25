use crate::common::counting_alloc;
use fusion_symbols::{
    category::FusionCategory,
    lie::{GroupId, LieGroup},
};

#[test]
fn warm_product_composition_allocates_nothing() {
    let _serial = crate::common::counting_alloc::serial();
    use fusion_symbols::{
        fermion::{FermionParity, FermionParityIrrep as Parity},
        product,
    };
    let cat = product(
        LieGroup::new(GroupId::su(2).unwrap()).unwrap(),
        FermionParity,
    );
    let half = (cat.factors().0.irrep(&[1]).unwrap(), Parity::ODD);
    let zero = cat.unit();
    let _ = cat
        .f_symbol(&half, &half, &half, &half, &zero, &zero)
        .unwrap();
    let stats = fusion_symbols::cache::base_cache_stats();
    let before = counting_alloc::calls();
    let f = cat
        .f_symbol(&half, &half, &half, &half, &zero, &zero)
        .unwrap();
    let r = cat.r_symbol(&half, &half, &zero).unwrap();
    let empty = cat.r_symbol(&half, &zero, &zero).unwrap();
    let outcomes = cat.fusion(&half, &half).unwrap().count();
    let after = counting_alloc::calls();
    assert_eq!(after, before);
    assert_eq!(f.at(0, 0, 0, 0), -0.5);
    assert_eq!(r.at(0, 0), 1.0);
    assert!(empty.data().is_empty());
    assert_eq!(outcomes, 2);
    let warm = fusion_symbols::cache::base_cache_stats();
    assert_eq!(warm.derived_f.misses, stats.derived_f.misses);
    assert_eq!(warm.derived_f.hits, stats.derived_f.hits + 1);
}
