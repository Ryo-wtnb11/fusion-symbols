use crate::common::counting_alloc;
use fusion_symbols::{
    category::{CgcBlock, FusionCategory, GroupRepresentations},
    lie::{GroupId, LieGroup},
};

#[test]
fn inline_torus_labels_and_queries_allocate_nothing() {
    let _serial = counting_alloc::serial();
    let g = LieGroup::new(GroupId::u(1).unwrap()).unwrap();
    let a = g.irrep_with_charges(&[], &[3]).unwrap();
    let b = g.irrep_with_charges(&[], &[-5]).unwrap();
    let c = g.irrep_with_charges(&[], &[-2]).unwrap();
    let one = g.unit();
    // Closed-form arithmetic has no generation sweep or coefficient cache.
    let before = counting_alloc::calls();
    std::hint::black_box(g.irrep_with_charges(&[], &[3]).unwrap());
    std::hint::black_box(a.clone());
    std::hint::black_box(g.dual(&a));
    assert_eq!(a.charges().next(), Some(3));
    assert_eq!(g.object_index(&a), Ok(5));
    assert_eq!(g.object_at(5), Ok(a.clone()));
    let mut fusion = g.fusion(&a, &b).unwrap();
    assert_eq!(fusion.len(), 1);
    assert_eq!(fusion.next_back(), Some(c.clone()));
    assert_eq!(fusion.next(), None);
    assert_eq!(g.nsymbol(&a, &b, &c), 1);
    assert_eq!(g.dim(&a), 1.0);
    assert_eq!(
        g.f_symbol(&a, &b, &one, &c, &c, &b).unwrap().at(0, 0, 0, 0),
        1.0
    );
    assert_eq!(g.r_symbol(&a, &b, &c).unwrap().at(0, 0), 1.0);
    assert_eq!(g.a_symbol(&a, &b, &c).unwrap().at(0, 0), 1.0);
    assert_eq!(g.b_symbol(&a, &b, &c).unwrap().at(0, 0), 1.0);
    assert_eq!(g.cgc(&a, &b, &c).unwrap().at([0; 4]), 1.0);
    assert_eq!(counting_alloc::calls(), before);
}
