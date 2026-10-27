//! `has_trivial_associator_gauge` and `has_canonical_unit` are checked three
//! ways: a literal per-category expectation, a direct reading of F over a
//! bounded label set for every `true`, and a literal F entry for every `false`.
use fusion_symbols::{
    category::{FusionCategory, Scalar, SymbolBlock},
    fermion::{FermionParity, FermionParityIrrep},
    finite::{ZNIrrep, ZN},
    lie::{CU1Irrep, GroupId, LieGroup, U1Irrep, CU1, U1},
    product::Product,
    table::TableCategory,
};
use num_complex::Complex64;

fn su2() -> LieGroup {
    LieGroup::new(GroupId::su(2).unwrap()).unwrap()
}
fn su2_irreps() -> Vec<<LieGroup as FusionCategory>::Object> {
    let g = su2();
    (0..4).map(|dj| g.irrep(&[dj], &[]).unwrap()).collect()
}
fn zn(n: u32) -> (ZN, Vec<ZNIrrep>) {
    let objects = (0..i64::from(n))
        .map(|q| ZNIrrep::new(q, n).unwrap())
        .collect();
    (ZN::new(n).unwrap(), objects)
}
fn u1_irreps() -> Vec<U1Irrep> {
    (-2..=2).map(|q| U1Irrep::new(q).unwrap()).collect()
}
fn parities() -> Vec<FermionParityIrrep> {
    vec![FermionParityIrrep::EVEN, FermionParityIrrep::ODD]
}
fn cu1_irreps() -> Vec<CU1Irrep> {
    let mut objects = vec![CU1Irrep::VACUUM, CU1Irrep::PSEUDOSCALAR];
    objects.extend((1..=3).map(CU1Irrep::from_twice_charge));
    objects
}
fn table(
    code: [u8; 7],
) -> (
    TableCategory,
    Vec<<TableCategory as FusionCategory>::Object>,
) {
    let cat = TableCategory::by_code(code).unwrap();
    let objects = cat.objects().collect();
    (cat, objects)
}

fn admissible<C: FusionCategory>(cat: &C, [a, b, c, d, e, f]: [&C::Object; 6]) -> bool {
    [(a, b, e), (e, c, d), (b, c, f), (a, f, d)]
        .iter()
        .all(|&(x, y, z)| cat.nsymbol(x, y, z) > 0)
}

/// Every admissible F over `objects` is the 1x1x1x1 block holding exactly 1.
fn assert_trivial_associator<C: FusionCategory>(cat: &C, objects: &[C::Object]) -> usize {
    let mut checked = 0;
    for a in objects {
        for b in objects {
            for c in objects {
                for d in objects {
                    for e in objects {
                        for f in objects {
                            if !admissible(cat, [a, b, c, d, e, f]) {
                                continue;
                            }
                            let block = cat.f_symbol(a, b, c, d, e, f).unwrap();
                            assert_eq!(block.dims(), [1; 4], "F^{a:?}{b:?}{c:?}_{d:?}");
                            assert_eq!(block.as_slice(), [C::Scalar::one()]);
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    checked
}

/// The unit is self-dual and fuses trivially, and every admissible F over
/// `objects` with a unit among `a, b, c` is the identity between its two
/// non-unit vertices, to within `tol`. Returns the number of F checked.
fn assert_canonical_unit<C: FusionCategory>(cat: &C, objects: &[C::Object], tol: f64) -> usize {
    let one = cat.unit();
    assert_eq!(cat.dual(&one), one);
    for a in objects {
        assert_eq!(
            cat.fusion(&one, a).unwrap().collect::<Vec<_>>(),
            std::slice::from_ref(a)
        );
        assert_eq!(
            cat.fusion(a, &one).unwrap().collect::<Vec<_>>(),
            std::slice::from_ref(a)
        );
        assert_eq!(cat.nsymbol(&one, a, a), 1);
        assert_eq!(cat.nsymbol(a, &one, a), 1);
    }
    let mut checked = 0;
    for a in objects {
        for b in objects {
            for c in objects {
                // Axes [N(a,b,e), N(e,c,d), N(b,c,f), N(a,f,d)]: a unit leg makes
                // two of them 1, and the identity pairs the other two.
                let (left, right) = if *a == one {
                    (1, 2)
                } else if *b == one {
                    (1, 3)
                } else if *c == one {
                    (0, 3)
                } else {
                    continue;
                };
                for d in objects {
                    for e in objects {
                        for f in objects {
                            if !admissible(cat, [a, b, c, d, e, f]) {
                                continue;
                            }
                            let block = cat.f_symbol(a, b, c, d, e, f).unwrap();
                            let dims = block.dims();
                            assert_eq!(dims[left], dims[right]);
                            for (axis, &n) in dims.iter().enumerate() {
                                assert!(axis == left || axis == right || n == 1);
                            }
                            for i in 0..dims[left] {
                                for j in 0..dims[right] {
                                    let mut index = [0; 4];
                                    index[left] = i;
                                    index[right] = j;
                                    let expected = if i == j {
                                        C::Scalar::one()
                                    } else {
                                        C::Scalar::zero()
                                    };
                                    let error = (block.get(index) - expected).abs();
                                    assert!(error <= tol, "F^{a:?}{b:?}{c:?}_{d:?}[{e:?},{f:?}]");
                                }
                            }
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    checked
}

fn certificates<C: FusionCategory>(cat: &C) -> (bool, bool) {
    (cat.has_trivial_associator_gauge(), cat.has_canonical_unit())
}

#[test]
fn hand_stated_certificates() {
    assert_eq!(certificates(&ZN::new(1).unwrap()), (true, true));
    assert_eq!(certificates(&ZN::new(2).unwrap()), (true, true));
    assert_eq!(certificates(&ZN::new(7).unwrap()), (true, true));
    assert_eq!(certificates(&U1), (true, true));
    assert_eq!(certificates(&FermionParity), (true, true));
    assert_eq!(certificates(&CU1), (false, true));
    assert_eq!(certificates(&su2()), (false, true));
    let so3 = LieGroup::new(GroupId::so(3).unwrap()).unwrap();
    assert_eq!(certificates(&so3), (false, true));
    assert_eq!(certificates(&TableCategory::fibonacci()), (false, true));
    assert_eq!(certificates(&TableCategory::ising()), (false, true));
    // Vec_Z2 with the nontrivial 3-cocycle (the semion fusion category).
    assert_eq!(certificates(&table([2, 1, 0, 1, 1, 1, 1]).0), (false, true));
    assert_eq!(certificates(&table([2, 1, 0, 1, 2, 1, 1]).0), (true, true));
}

#[cfg(feature = "lie-generated")]
#[test]
fn hand_stated_generated_certificates() {
    for group in [
        GroupId::su(3).unwrap(),
        GroupId::sp(4).unwrap(),
        GroupId::so(5).unwrap(),
    ] {
        assert_eq!(certificates(&LieGroup::new(group).unwrap()), (false, true));
    }
}

#[test]
fn abelian_true_certificates_hold_in_f() {
    for n in [1, 2, 3, 4] {
        let (cat, objects) = zn(n);
        assert!(assert_trivial_associator(&cat, &objects) > 0);
        assert!(assert_canonical_unit(&cat, &objects, 0.0) > 0);
    }
    assert!(assert_trivial_associator(&U1, &u1_irreps()) > 0);
    assert!(assert_canonical_unit(&U1, &u1_irreps(), 0.0) > 0);
    assert!(assert_trivial_associator(&FermionParity, &parities()) > 0);
    assert!(assert_canonical_unit(&FermionParity, &parities(), 0.0) > 0);
}

#[test]
fn closed_form_canonical_units_hold_exactly_in_f() {
    assert!(assert_canonical_unit(&CU1, &cu1_irreps(), 0.0) > 0);
    assert!(assert_canonical_unit(&su2(), &su2_irreps(), 0.0) > 0);
}

#[cfg(feature = "lie-generated")]
#[test]
fn generated_canonical_units_hold_in_f_to_rounding() {
    let cases: [(GroupId, &[&[i64]]); 3] = [
        // 8 ⊗ 8 → 8 has multiplicity two, so the identity is a 2x2 block.
        (
            GroupId::su(3).unwrap(),
            &[&[0, 0], &[1, 0], &[0, 1], &[1, 1]],
        ),
        (
            GroupId::sp(4).unwrap(),
            &[&[0, 0], &[1, 0], &[0, 1], &[2, 0]],
        ),
        (
            GroupId::spin(5).unwrap(),
            &[&[0, 0], &[1, 0], &[0, 1], &[0, 2]],
        ),
    ];
    for (group, labels) in cases {
        let cat = LieGroup::new(group).unwrap();
        let objects: Vec<_> = labels.iter().map(|l| cat.irrep(l, &[]).unwrap()).collect();
        assert!(assert_canonical_unit(&cat, &objects, 1e-12) > 0);
    }
}

#[test]
fn tables_certificates_hold_in_f() {
    for uuid in TableCategory::shipped_uuids() {
        let cat = TableCategory::by_uuid(uuid).unwrap();
        let objects: Vec<_> = cat.objects().collect();
        assert!(cat.has_canonical_unit(), "{uuid}");
        assert_canonical_unit(&cat, &objects, 0.0);
        if cat.has_trivial_associator_gauge() {
            assert_trivial_associator(&cat, &objects);
            // Only a pointed category can have every F equal to 1.
            assert!(objects.iter().all(|a| cat.dim(a) == 1.0), "{uuid}");
        }
    }
}

#[test]
fn trivial_associator_tables_are_the_untwisted_pointed_entries() {
    // Rank ≤ 4: Vec_1, Vec_Z2, Vec_Z3 and Vec_Z4 / Vec_{Z2×Z2} with the trivial
    // 3-cocycle, each with every braiding the database lists.
    #[allow(unused_mut)]
    let mut expected: Vec<[u8; 7]> = vec![
        [1, 1, 0, 1, 1, 1, 1],
        [2, 1, 0, 1, 2, 1, 1],
        [2, 1, 0, 1, 2, 2, 1],
        [3, 1, 2, 1, 1, 1, 1],
        [3, 1, 2, 1, 1, 2, 1],
        [3, 1, 2, 1, 1, 3, 1],
        [4, 1, 0, 1, 2, 4, 1],
        [4, 1, 0, 1, 2, 5, 1],
        [4, 1, 0, 1, 2, 5, 3],
        [4, 1, 0, 1, 2, 6, 1],
        [4, 1, 2, 1, 2, 1, 1],
        [4, 1, 2, 1, 2, 2, 1],
        [4, 1, 2, 1, 2, 3, 1],
        [4, 1, 2, 1, 2, 4, 1],
    ];
    // Rank 5–7: untwisted Vec_Z5, Vec_Z6 (one fusion-only entry) and Vec_Z7.
    #[cfg(feature = "anyonwiki-unitary")]
    expected.extend([
        [5, 1, 4, 1, 1, 1, 1],
        [5, 1, 4, 1, 1, 2, 1],
        [5, 1, 4, 1, 1, 3, 1],
        [6, 1, 2, 1, 1, 0, 1],
        [6, 1, 4, 1, 2, 1, 1],
        [6, 1, 4, 1, 2, 2, 1],
        [6, 1, 4, 1, 2, 3, 1],
        [6, 1, 4, 1, 2, 4, 1],
        [6, 1, 4, 1, 2, 5, 1],
        [6, 1, 4, 1, 2, 6, 1],
        [7, 1, 6, 1, 1, 1, 1],
        [7, 1, 6, 1, 1, 2, 1],
        [7, 1, 6, 1, 1, 3, 1],
    ]);
    // Every shipped entry: listed ⇒ true, unlisted ⇒ false.
    for uuid in TableCategory::shipped_uuids() {
        let cat = TableCategory::by_uuid(uuid).unwrap();
        assert_eq!(
            cat.has_trivial_associator_gauge(),
            expected.contains(&cat.code()),
            "{:?}",
            cat.code()
        );
    }
    // And every listed code is shipped in this build.
    for code in &expected {
        assert!(TableCategory::by_code(*code).is_ok(), "{code:?}");
    }
}

#[test]
fn false_trivial_associators_have_a_non_unit_f() {
    // SU(2), exact 6j: F^{½½½}_½[0, 0] = {½ ½ 0; ½ ½ 0} = -1/2.
    let g = su2();
    let (zero, half) = (g.irrep(&[0], &[]).unwrap(), g.irrep(&[1], &[]).unwrap());
    let f = g
        .f_symbol(&half, &half, &half, &half, &zero, &zero)
        .unwrap();
    assert_eq!(f.as_slice(), [-0.5]);

    // CU(1), `cu1irrep.jl:Fsymbol`: F^{jjj}_j[0+, 0+] = 1/2 for j = 1/2.
    let j = CU1Irrep::from_twice_charge(1);
    let v = CU1Irrep::VACUUM;
    assert_eq!(CU1.f_symbol(&j, &j, &j, &j, &v, &v).unwrap(), 0.5);

    // Fibonacci: F^{τττ}_τ[𝟙, 𝟙] = 1/φ.
    let fib = TableCategory::fibonacci();
    let (one, tau) = (fib.object(1).unwrap(), fib.object(2).unwrap());
    assert_eq!(
        fib.f_symbol(&tau, &tau, &tau, &tau, &one, &one).unwrap(),
        Complex64::new(0.6180339887498949, 0.0)
    );

    // Ising (AnyonWiki gauge): F^{σσσ}_σ[𝟙, ψ] = -1/√2.
    let ising = TableCategory::ising();
    let (one, psi, sigma) = (
        ising.object(1).unwrap(),
        ising.object(2).unwrap(),
        ising.object(3).unwrap(),
    );
    let f = ising
        .f_symbol(&sigma, &sigma, &sigma, &sigma, &one, &psi)
        .unwrap();
    assert!((f - Complex64::new(-0.5f64.sqrt(), 0.0)).norm() < 1e-15);

    // Twisted Vec_Z2: F^{sss}_s[𝟙, 𝟙] = -1.
    let (semion, objects) = table([2, 1, 0, 1, 1, 1, 1]);
    let (one, s) = (&objects[0], &objects[1]);
    assert_eq!(
        semion.f_symbol(s, s, s, s, one, one).unwrap(),
        Complex64::new(-1.0, 0.0)
    );
}

#[cfg(feature = "lie-generated")]
#[test]
fn generated_trivial_associator_counterexample() {
    // |F^{3 3̄ 3}_3[𝟙, 𝟙]| = 1 / dim(3) = 1/3.
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let (one, three, bar) = (
        su3.unit(),
        su3.irrep(&[1, 0], &[]).unwrap(),
        su3.irrep(&[0, 1], &[]).unwrap(),
    );
    let f = su3
        .f_symbol(&three, &bar, &three, &three, &one, &one)
        .unwrap();
    assert!((f.as_slice()[0].abs() - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn products_take_the_conjunction() {
    let z2 = ZN::new(2).unwrap();
    let fib = TableCategory::fibonacci();
    assert_eq!(certificates(&Product::new((z2, U1))), (true, true));
    assert_eq!(
        certificates(&Product::new((z2, U1, FermionParity))),
        (true, true)
    );
    assert_eq!(certificates(&Product::new((U1, CU1))), (false, true));
    assert_eq!(
        certificates(&Product::new((su2(), fib.clone()))),
        (false, true)
    );
    // Nested: one non-trivial leaf anywhere decides the conjunction.
    let inner = Product::new((z2, FermionParity));
    assert_eq!(certificates(&Product::new((inner, U1))), (true, true));
    assert_eq!(
        certificates(&Product::new((
            U1,
            Product::new((z2, Product::new((FermionParity, CU1))))
        ))),
        (false, true)
    );
    assert_eq!(
        certificates(&Product::new((
            Product::new((z2, U1)),
            U1,
            Product::new((fib, z2))
        ))),
        (false, true)
    );
}

#[test]
fn product_certificates_hold_in_f() {
    let (z3, z3_objects) = zn(3);
    let cat = Product::new((z3, FermionParity));
    let objects: Vec<_> = z3_objects
        .iter()
        .flat_map(|&a| parities().into_iter().map(move |p| (a, p)))
        .collect();
    assert!(assert_trivial_associator(&cat, &objects) > 0);
    assert!(assert_canonical_unit(&cat, &objects, 0.0) > 0);

    let fib = TableCategory::fibonacci();
    let cat = Product::new((CU1, fib.clone()));
    let objects: Vec<_> = cu1_irreps()[..3]
        .iter()
        .flat_map(|&a| fib.objects().map(move |t| (a, t)))
        .collect();
    assert!(assert_canonical_unit(&cat, &objects, 0.0) > 0);
}
