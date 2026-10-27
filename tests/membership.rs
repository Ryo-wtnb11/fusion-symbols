//! `contains` is the non-panicking membership pre-check: it accepts exactly the
//! objects the other queries accept and rejects foreign ones without panicking.
use fusion_symbols::{
    category::FusionCategory,
    fermion::{FermionParity, FermionParityIrrep},
    finite::{ZNIrrep, ZN},
    lie::{CU1Irrep, GroupId, LieGroup, U1Irrep, CU1, U1},
    product::Product,
    table::TableCategory,
};

#[test]
fn leaf_categories_accept_own_and_reject_foreign_objects() {
    let z2 = ZN::new(2).unwrap();
    let z3 = ZN::new(3).unwrap();
    assert!(z2.contains(&z2.unit()) && z2.contains(&ZNIrrep::new(1, 2).unwrap()));
    assert!(!z2.contains(&ZNIrrep::new(1, 3).unwrap()) && !z3.contains(&z2.unit()));

    for q in [0, 1, -7, i32::MAX, i32::MIN + 1] {
        assert!(U1.contains(&U1Irrep::new(q).unwrap()));
    }
    for a in [
        CU1Irrep::VACUUM,
        CU1Irrep::PSEUDOSCALAR,
        CU1Irrep::from_twice_charge(1),
        CU1Irrep::from_twice_charge(u32::MAX),
    ] {
        assert!(CU1.contains(&a));
    }
    for a in [FermionParityIrrep::EVEN, FermionParityIrrep::ODD] {
        assert!(FermionParity.contains(&a));
    }

    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let so3 = LieGroup::new(GroupId::so(3).unwrap()).unwrap();
    let spin = su2.irrep(&[1], &[]).unwrap();
    assert!(su2.contains(&spin) && su2.contains(&su2.unit()));
    assert!(!so3.contains(&spin) && !su2.contains(&so3.irrep(&[2], &[]).unwrap()));

    let fib = TableCategory::fibonacci();
    let ising = TableCategory::ising();
    assert!(fib.objects().all(|a| fib.contains(&a)));
    // Ising has rank 3, so its third object is also outside Fibonacci's index range.
    assert!(ising.objects().all(|a| !fib.contains(&a)));
    assert!(fib.objects().all(|a| !ising.contains(&a)));
    // Another value of the same table shares its identity and objects.
    assert!(fib
        .objects()
        .all(|a| TableCategory::fibonacci().contains(&a)));
}

#[cfg(feature = "lie-generated")]
#[test]
fn generated_lie_groups_reject_other_groups() {
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let su4 = LieGroup::new(GroupId::su(4).unwrap()).unwrap();
    let sp4 = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
    let octet = su3.irrep(&[1, 1], &[]).unwrap();
    assert!(su3.contains(&octet));
    assert!(!su4.contains(&octet) && !sp4.contains(&octet));
    assert!(!su3.contains(&sp4.irrep(&[1, 0], &[]).unwrap()));
    assert!(!su3.contains(&su4.unit()));
}

#[test]
fn products_reject_a_foreign_component_in_any_position() {
    let z2 = ZN::new(2).unwrap();
    let z3 = ZN::new(3).unwrap();
    let cat = Product::new((z2, U1, z2));
    let u = cat.unit();
    assert!(cat.contains(&u));
    assert!(cat.contains(&(
        ZNIrrep::new(1, 2).unwrap(),
        U1Irrep::new(5).unwrap(),
        ZNIrrep::new(1, 2).unwrap()
    )));
    assert!(!cat.contains(&(z3.unit(), u.1, u.2)));
    assert!(!cat.contains(&(u.0, u.1, z3.unit())));

    let fib = TableCategory::fibonacci();
    let nested = Product::new((Product::new((z2, fib.clone())), Product::new((U1, z2))));
    let u = nested.unit();
    assert!(nested.contains(&u));
    let ising = TableCategory::ising();
    assert!(!nested.contains(&((z3.unit(), u.0 .1), u.1)));
    assert!(!nested.contains(&((u.0 .0, ising.unit()), u.1)));
    assert!(!nested.contains(&(u.0, (u.1 .0, z3.unit()))));
    // A rejected object would panic in every other query; `contains` did not.
    let foreign = (u.0, (u.1 .0, z3.unit()));
    assert!(std::panic::catch_unwind(|| nested.dim(&foreign)).is_err());
}
