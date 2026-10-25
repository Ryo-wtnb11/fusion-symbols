//! Public checked arithmetic must stay fallible through every symbol entry point.
#![cfg(feature = "lie-generated")]
use fusion_symbols::{
    category::{FusionCategory, GroupRepresentations},
    lie::{GroupId, LieError, LieGroup},
};

#[test]
fn overflow_propagates_through_fusion_n_f_r_a_b_and_cgc() {
    for (group, huge, small) in [
        (GroupId::su(3).unwrap(), &[i64::MAX, 0][..], &[1, 0][..]),
        (GroupId::spin(5).unwrap(), &[0, i64::MAX][..], &[1, 0][..]),
        (GroupId::sp(4).unwrap(), &[i64::MAX / 2, 0][..], &[1, 0][..]),
        (
            GroupId::spin(6).unwrap(),
            &[i64::MAX / 2, 0, 0][..],
            &[1, 0, 0][..],
        ),
    ] {
        let cat = LieGroup::new(group).unwrap();
        let a = cat.irrep(huge, &[]).unwrap();
        let b = cat.irrep(small, &[]).unwrap();
        let unit = cat.unit();
        assert_eq!(cat.dual(&cat.dual(&a)), a);
        assert!(matches!(cat.fusion(&a, &b), Err(LieError::LabelOverflow)));
        assert_eq!(cat.try_nsymbol(&a, &b, &unit), Err(LieError::LabelOverflow));
        assert_eq!(
            cat.f_symbol(&a, &b, &unit, &unit, &unit, &unit),
            Err(LieError::LabelOverflow)
        );
        assert_eq!(cat.r_symbol(&a, &b, &unit), Err(LieError::LabelOverflow));
        assert_eq!(cat.a_symbol(&a, &b, &unit), Err(LieError::LabelOverflow));
        assert_eq!(cat.b_symbol(&a, &b, &unit), Err(LieError::LabelOverflow));
        assert!(matches!(
            cat.cgc(&a, &b, &unit),
            Err(LieError::LabelOverflow)
        ));
    }
}
