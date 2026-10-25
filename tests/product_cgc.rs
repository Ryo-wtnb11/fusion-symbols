//! CGC isometries and the overlap of the two three-object coupling bases.
use fusion_symbols::{
    category::{CgcBlock, FusionCategory, GroupRepresentations, SymbolBlock},
    finite::{ZNIrrep, ZN},
    lie::{GroupId, LieGroup},
    product,
};

fn su2() -> LieGroup {
    LieGroup::new(GroupId::su(2).unwrap()).unwrap()
}

fn assert_orthonormal(block: &impl CgcBlock) {
    let [da, db, dc, n] = block.dims();
    for c in 0..dc {
        for mu in 0..n {
            for other_c in 0..dc {
                for other_mu in 0..n {
                    let mut dot = 0.0;
                    for a in 0..da {
                        for b in 0..db {
                            dot += block.at([a, b, c, mu]) * block.at([a, b, other_c, other_mu]);
                        }
                    }
                    let want = f64::from(c == other_c && mu == other_mu);
                    assert!(
                        (dot - want).abs() < 1e-10,
                        "Gram {c},{mu};{other_c},{other_mu}: {dot}"
                    );
                }
            }
        }
    }
}

fn assert_f_from_cgc<C: GroupRepresentations<Scalar = f64>>(cat: &C, labels: [&C::Object; 6]) {
    let [a, b, c, d, e, f] = labels;
    let ab = cat.cgc(a, b, e).unwrap();
    let ec = cat.cgc(e, c, d).unwrap();
    let bc = cat.cgc(b, c, f).unwrap();
    let af = cat.cgc(a, f, d).unwrap();
    let expected = cat.f_symbol(a, b, c, d, e, f).unwrap();
    let [da, db, de, nmu] = ab.dims();
    let [_, dc, _, nnu] = ec.dims();
    let [_, _, df, nkappa] = bc.dims();
    let nlambda = af.dims()[3];
    // Schur's lemma: the coupling-basis overlap is F times identity on d.
    // Fix md = 0 and sum over the three uncoupled magnetic indices.
    let mut overlap = vec![0.0; nmu * nnu * nkappa * nlambda];
    for ma in 0..da {
        for mb in 0..db {
            for mc in 0..dc {
                let left: Vec<f64> = (0..nmu * nnu)
                    .map(|i| {
                        (0..de)
                            .map(|me| ab.at([ma, mb, me, i / nnu]) * ec.at([me, mc, 0, i % nnu]))
                            .sum()
                    })
                    .collect();
                let right: Vec<f64> = (0..nkappa * nlambda)
                    .map(|i| {
                        (0..df)
                            .map(|mf| {
                                bc.at([mb, mc, mf, i / nlambda]) * af.at([ma, mf, 0, i % nlambda])
                            })
                            .sum()
                    })
                    .collect();
                for (i, l) in left.iter().enumerate() {
                    for (j, r) in right.iter().enumerate() {
                        overlap[i * right.len() + j] += l * r;
                    }
                }
            }
        }
    }
    for mu in 0..nmu {
        for nu in 0..nnu {
            for kappa in 0..nkappa {
                for lambda in 0..nlambda {
                    let index = [mu, nu, kappa, lambda];
                    let got = overlap[((mu * nnu + nu) * nkappa + kappa) * nlambda + lambda];
                    let want = expected.get(index);
                    assert!(
                        (got - want).abs() < 1e-10,
                        "F {index:?}: {got}, expected {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn su2_su2_isometry_and_f() {
    let spin = su2();
    let one = spin.irrep(&[2], &[]).unwrap();
    let cat = product(spin.clone(), spin);
    let a = (one.clone(), one);
    assert_orthonormal(&cat.cgc(&a, &a, &a).unwrap());
    assert_f_from_cgc(&cat, [&a; 6]);
}

#[test]
fn su2_zn_isometry_and_f() {
    let spin = su2();
    let one = spin.irrep(&[2], &[]).unwrap();
    let zn = ZN::new(3).unwrap();
    let z = |q| ZNIrrep::new(q, 3).unwrap();
    let cat = product(spin, zn);
    let a = (one.clone(), z(1));
    let e = (one.clone(), z(2));
    let d = (one, z(0));
    assert_orthonormal(&cat.cgc(&a, &a, &e).unwrap());
    assert_f_from_cgc(&cat, [&a, &a, &a, &d, &e, &e]);
    let absent = cat.cgc(&a, &a, &a).unwrap();
    assert_eq!(absent.dims()[3], 0);
    assert!(std::panic::catch_unwind(|| absent.at([0; 4])).is_err());
}

#[cfg(feature = "lie-generated")]
#[test]
fn su3_su2_outer_multiplicity_isometry_and_f() {
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let eight = su3.irrep(&[1, 1], &[]).unwrap();
    let spin = su2();
    let half = spin.irrep(&[1], &[]).unwrap();
    let one = spin.irrep(&[2], &[]).unwrap();
    let cat = product(su3, spin);
    let a = (eight.clone(), half);
    let e = (eight, one);
    let block = cat.cgc(&a, &a, &e).unwrap();
    assert_eq!(block.dims(), [16, 16, 24, 2]);
    assert_orthonormal(&block);
    assert_f_from_cgc(&cat, [&a, &a, &a, &a, &e, &e]);
}

#[test]
fn last_factor_fastest_and_nested_products() {
    let spin = su2();
    let half = spin.irrep(&[1], &[]).unwrap();
    let one = spin.irrep(&[2], &[]).unwrap();
    let unit = spin.unit();
    let cat = product(spin.clone(), spin.clone());
    let a = (half.clone(), one.clone());
    let c = (one.clone(), unit.clone());
    let block = cat.cgc(&a, &a, &c).unwrap();
    let half_half_to_one = spin.cgc(&half, &half, &one).unwrap();
    let one_one_to_unit = spin.cgc(&one, &one, &unit).unwrap();
    let combined = [3, 2, 1, 0]; // [1*3+0, 0*3+2, 1*1+0, 0*1+0]
    let want = half_half_to_one.at([1, 0, 1, 0]) * one_one_to_unit.at([0, 2, 0, 0]);
    assert!(want.abs() > 0.1);
    assert_eq!(block.at(combined), want);
    assert!(std::panic::catch_unwind(|| block.at([6, 0, 0, 0])).is_err());
    let nested = product(cat, ZN::new(3).unwrap());
    let charge = ZNIrrep::new(0, 3).unwrap();
    let nested_block = nested
        .cgc(&(a.clone(), charge), &(a, charge), &(c, charge))
        .unwrap();
    assert_eq!(nested_block.dims(), block.dims());
    assert_eq!(nested_block.at(combined), want);
    // The arity-12 implementation remains lazy, including explicitly nested tuples.
    let z = ZN::new(3).unwrap();
    let twelve = fusion_symbols::product![z, z, z, z, z, z, z, z, z, z, z, z];
    assert_eq!(
        twelve
            .cgc(&twelve.unit(), &twelve.unit(), &twelve.unit())
            .unwrap()
            .at([0; 4]),
        1.0
    );
}

#[cfg(feature = "lie-generated")]
#[test]
fn su3_su3_multiplicity_last_factor_fastest() {
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let eight = su3.irrep(&[1, 1], &[]).unwrap();
    let c = su3.cgc(&eight, &eight, &eight).unwrap();
    let cat = product(su3.clone(), su3);
    let a = (eight.clone(), eight);
    let block = cat.cgc(&a, &a, &a).unwrap();
    assert_eq!(block.dims(), [64, 64, 64, 4]);
    let want_10 = c.at([0, 2, 0, 1]) * c.at([2, 0, 0, 0]);
    let want_01 = c.at([0, 2, 0, 0]) * c.at([2, 0, 0, 1]);
    assert!((want_10 - want_01).abs() > 0.1);
    // Last factor fastest: magnetic [0*8+2, 2*8+0, 0*8+0], mu = mu0*2+mu1.
    let [i, j, k] = [2, 16, 0];
    assert_eq!(block.at([i, j, k, 2]), want_10); // 1*2+0
    assert_eq!(block.at([i, j, k, 1]), want_01); // 0*2+1
}

#[test]
fn zn_scalar_channels_and_bounds() {
    let cat = ZN::new(3).unwrap();
    let z = |q| ZNIrrep::new(q, 3).unwrap();
    let present = cat.cgc(&z(1), &z(1), &z(2)).unwrap();
    let absent = cat.cgc(&z(1), &z(1), &z(0)).unwrap();
    assert_eq!(present, 1.0);
    assert_eq!(absent, 0.0);
    assert_eq!(CgcBlock::dims(&present), [1; 4]);
    assert_eq!(CgcBlock::dims(&absent), [1, 1, 1, 0]);
    assert_eq!(present.at([0; 4]), 1.0);
    assert!(std::panic::catch_unwind(|| absent.at([0; 4])).is_err());
    for axis in 0..4 {
        let mut index = [0; 4];
        index[axis] = 1;
        assert!(std::panic::catch_unwind(|| present.at(index)).is_err());
    }
    assert!(
        std::panic::catch_unwind(|| cat.cgc(&z(1), &z(1), &ZNIrrep::new(2, 4).unwrap())).is_err()
    );
}

#[test]
fn axis_overflow_factor_errors_and_foreign_objects() {
    use fusion_symbols::{lie::LieError, product::ProductError};
    let spin = su2();
    let huge = spin.irrep(&[i64::from(u32::MAX)], &[]).unwrap();
    let unit = spin.unit();
    let cat = product(spin.clone(), spin);
    let h = (huge.clone(), huge.clone());
    let u = (unit.clone(), unit.clone());
    // Absent channels need no dense allocation, but each magnetic axis must fit.
    for labels in [[&h, &u, &u], [&u, &h, &u], [&u, &u, &h]] {
        assert!(matches!(
            cat.cgc(labels[0], labels[1], labels[2]),
            Err(ProductError::Overflow)
        ));
    }
    assert!(matches!(
        cat.cgc(&h, &h, &u),
        Err(ProductError::Factor0(LieError::Unsupported { .. }))
    ));
    let right_huge = (unit, huge);
    assert!(matches!(
        cat.cgc(&right_huge, &right_huge, &u),
        Err(ProductError::Factor1(LieError::Unsupported { .. }))
    ));
    let zn = ZN::new(3).unwrap();
    let zero = zn.unit();
    let mixed = product(cat, zn);
    let foreign = ZNIrrep::new(0, 4).unwrap();
    // The foreign last label must panic before the first factor can return an error.
    assert!(std::panic::catch_unwind(|| mixed.cgc(
        &(h.clone(), zero),
        &(h.clone(), zero),
        &(u.clone(), foreign)
    ))
    .is_err());
}

#[test]
fn d2_isometry_and_f() {
    let cat = LieGroup::new(GroupId::spin(4).unwrap()).unwrap();
    let a = cat.irrep(&[2, 2], &[]).unwrap();
    assert_orthonormal(&cat.cgc(&a, &a, &a).unwrap());
    assert_f_from_cgc(&cat, [&a; 6]);
}

#[cfg(feature = "lie-generated")]
#[test]
fn a1_a2_isometry_and_f() {
    use fusion_symbols::lie::RootSystem::A;
    let cat = LieGroup::new(GroupId::cover(&[A(1), A(2)], 0).unwrap()).unwrap();
    let a = cat.irrep(&[1, 1, 1], &[]).unwrap();
    let e = cat.irrep(&[2, 1, 1], &[]).unwrap();
    assert_orthonormal(&cat.cgc(&a, &a, &e).unwrap());
    assert_f_from_cgc(&cat, [&a, &a, &a, &a, &e, &e]);
}
