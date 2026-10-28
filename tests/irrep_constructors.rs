//! Irrep constructors: `irrep` without charges, `irrep_with_charges`, and SU(2)
//! spins through `HalfInt`.

use fusion_symbols::{
    category::FusionCategory,
    lie::{GroupId, HalfInt, LieError, LieGroup, RootSystem},
};

fn group(id: GroupId) -> LieGroup {
    LieGroup::new(id).unwrap()
}

#[cfg(feature = "lie-generated")]
#[test]
fn irrep_takes_dynkin_labels_only() {
    let su3 = group(GroupId::su(3).unwrap());
    assert_eq!(su3.dim(&su3.irrep(&[1, 1]).unwrap()), 8.0);
}

#[test]
fn torus_groups_need_charges() {
    let u2 = group(GroupId::u(2).unwrap());
    assert_eq!(u2.irrep(&[1]), Err(LieError::InvalidLabel));
    assert!(u2.irrep_with_charges(&[1], &[1]).is_ok());
    assert_eq!(
        u2.irrep_with_charges(&[1], &[0]),
        Err(LieError::NotAdmissible)
    );
}

#[test]
fn spin_round_trips_on_su2_forms() {
    for id in [
        GroupId::su(2).unwrap(),
        GroupId::spin(3).unwrap(),
        GroupId::sp(2).unwrap(),
    ] {
        let g = group(id);
        let half = g.irrep_spin(HalfInt::half(1)).unwrap();
        assert_eq!(g.dim(&half), 2.0);
        assert_eq!(half.spin(), Some(HalfInt::half(1)));
        let one = g.irrep_spin(HalfInt::integer(1).unwrap()).unwrap();
        assert_eq!(g.dim(&one), 3.0);
        assert_eq!(one.spin(), Some(HalfInt::integer(1).unwrap()));
    }
}

#[test]
fn so3_forms_admit_integer_spins_only() {
    for id in [
        GroupId::so(3).unwrap(),
        GroupId::psu(2).unwrap(),
        GroupId::psp(2).unwrap(),
    ] {
        let g = group(id);
        assert_eq!(g.irrep_spin(HalfInt::half(1)), Err(LieError::NotAdmissible));
        let one = g.irrep_spin(HalfInt::integer(1).unwrap()).unwrap();
        assert_eq!(g.dim(&one), 3.0);
        assert_eq!(one.spin(), Some(HalfInt::integer(1).unwrap()));
    }
}

#[test]
fn spin_is_only_for_su2_and_so3() {
    let mut others = vec![
        GroupId::u(2).unwrap(),
        GroupId::cover(&[RootSystem::A(1), RootSystem::A(1)], 0).unwrap(),
    ];
    if cfg!(feature = "lie-generated") {
        others.push(GroupId::su(3).unwrap());
    }
    for id in others {
        let g = group(id);
        assert_eq!(g.irrep_spin(HalfInt::half(1)), Err(LieError::InvalidLabel));
        assert_eq!(g.unit().spin(), None);
    }
}

#[test]
fn half_int_values() {
    let shown: Vec<_> = (0..4).map(|n| HalfInt::half(n).to_string()).collect();
    assert_eq!(shown, ["0", "1/2", "1", "3/2"]);
    assert_eq!(HalfInt::half(2), HalfInt::integer(1).unwrap());
    assert_eq!(HalfInt::half(3).doubled(), 3);
    assert_eq!(
        HalfInt::integer(u32::MAX / 2).unwrap().doubled(),
        u32::MAX - 1
    );
    assert_eq!(HalfInt::integer(u32::MAX / 2 + 1), None);
}
