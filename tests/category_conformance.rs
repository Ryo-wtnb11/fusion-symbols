//! Fusion-category identities from TensorKitSectors 0.3.9 `src/sectors.jl`:
//! triangle_equation, pentagon_equation, hexagon_equation,
//! frobenius_schur_phase_from_Fsymbol and twist_from_Rsymbol.
//! Fusion outcomes are enumerated by the category, including intermediate
//! U(1) charges outside the finite input sample. Shape is determined by N.
use fusion_symbols::{
    category::{BraidingStyle, FusionCategory, FusionStyle, Promote, Scalar, SymbolBlock},
    fermion::{FermionParity, FermionParityIrrep},
    finite::{ZNIrrep, ZN},
    lie::{CU1Irrep, CU1},
    lie::{U1Irrep, U1},
    table::TableCategory,
};
use num_complex::Complex64;

const TOLERANCE: f64 = 1e-12;

fn close<S: Scalar>(lhs: S, rhs: S) {
    assert!((lhs - rhs).abs() <= TOLERANCE, "{lhs:?} != {rhs:?}");
}

fn outcomes<C: FusionCategory>(cat: &C, a: &C::Object, b: &C::Object) -> Vec<C::Object> {
    cat.fusion(a, b).unwrap().collect()
}

fn f<C: FusionCategory>(cat: &C, [a, b, c, d, e, f]: [&C::Object; 6]) -> C::Scalar {
    let block = cat.f_symbol(a, b, c, d, e, f).unwrap();
    let shape = [
        cat.nsymbol(a, b, e),
        cat.nsymbol(e, c, d),
        cat.nsymbol(b, c, f),
        cat.nsymbol(a, f, d),
    ];
    if shape.contains(&0) {
        assert!(block.as_slice().iter().all(|&x| x == C::Scalar::zero()));
        C::Scalar::zero()
    } else {
        assert_eq!(shape, [1; 4]);
        block.get([0; 4])
    }
}

fn conformance<C: FusionCategory>(cat: C, labels: Vec<C::Object>) {
    use fusion_symbols::category::{f_unitarity_residual, hexagon_residual, pentagon_residual};
    let one = C::Scalar::one();
    let zero = C::Scalar::zero();
    let unit = cat.unit();
    close(cat.dim(&unit), 1.0);
    for a in &labels {
        assert_eq!(outcomes(&cat, &unit, a), std::slice::from_ref(a));
        assert_eq!(outcomes(&cat, a, &unit), std::slice::from_ref(a));
        assert_eq!(cat.dual(&cat.dual(a)), *a);
        assert_eq!(cat.nsymbol(a, &cat.dual(a), &unit), 1);
        let phase_f = f(&cat, [a, &cat.dual(a), a, a, &unit, &unit]);
        close(
            cat.frobenius_schur_phase(a),
            phase_f * C::Scalar::from_f64(1.0 / phase_f.abs()),
        );
        close(
            cat.frobenius_schur_indicator(a),
            if *a == cat.dual(a) {
                cat.frobenius_schur_phase(a)
            } else {
                zero
            },
        );
        if cat.braiding_style() != BraidingStyle::None {
            let mut twist = zero;
            for b in outcomes(&cat, a, a) {
                let block = cat.r_symbol(a, a, &b).unwrap();
                for mu in 0..cat.nsymbol(a, a, &b) as usize {
                    twist =
                        twist + C::Scalar::from_f64(cat.dim(&b) / cat.dim(a)) * block.get([mu, mu]);
                }
            }
            close(cat.twist(a), twist);
        }
        for b in &labels {
            let dimension_sum: f64 = outcomes(&cat, a, b)
                .iter()
                .map(|c| f64::from(cat.nsymbol(a, b, c)) * cat.dim(c))
                .sum();
            close(cat.dim(a) * cat.dim(b), dimension_sum);
            for c in outcomes(&cat, a, b) {
                let n = cat.nsymbol(a, b, &c) as usize;
                let blocks = [
                    (cat.f_symbol(&unit, a, b, &c, a, &c).unwrap(), 0),
                    (cat.f_symbol(a, &unit, b, &c, a, b).unwrap(), 1),
                    (cat.f_symbol(a, b, &unit, &c, &c, b).unwrap(), 2),
                ];
                for (block, kind) in blocks {
                    for mu in 0..n {
                        for nu in 0..n {
                            let indices = match kind {
                                0 => [0, mu, nu, 0],
                                1 => [0, mu, 0, nu],
                                _ => [mu, 0, 0, nu],
                            };
                            close(block.get(indices), if mu == nu { one } else { zero });
                        }
                    }
                }
            }
            for c in &labels {
                if matches!(
                    cat.braiding_style(),
                    BraidingStyle::Bosonic | BraidingStyle::Fermionic
                ) {
                    let n = cat.nsymbol(a, b, c) as usize;
                    assert_eq!(n, cat.nsymbol(b, a, c) as usize);
                    if n != 0 {
                        let ab = cat.r_symbol(a, b, c).unwrap();
                        let ba = cat.r_symbol(b, a, c).unwrap();
                        for mu in 0..n {
                            for nu in 0..n {
                                let mut dot = zero;
                                for k in 0..n {
                                    dot = dot + ab.get([mu, k]) * ba.get([k, nu]);
                                }
                                close(dot, if mu == nu { one } else { zero });
                            }
                        }
                    }
                }
                for e in outcomes(&cat, a, b) {
                    for d in outcomes(&cat, &e, c) {
                        close(f_unitarity_residual(&cat, a, b, c, &d).unwrap(), 0.0);
                    }
                }
                if cat.braiding_style() != BraidingStyle::None {
                    close(hexagon_residual(&cat, a, b, c).unwrap(), 0.0);
                }
                for d in &labels {
                    close(pentagon_residual(&cat, a, b, c, d).unwrap(), 0.0);
                }
            }
        }
    }
}

#[test]
fn category_identities() {
    assert_eq!(U1.fusion_style(), FusionStyle::Unique);
    assert_eq!(U1.braiding_style(), BraidingStyle::Bosonic);
    assert_eq!(FermionParity.fusion_style(), FusionStyle::Unique);
    assert_eq!(FermionParity.braiding_style(), BraidingStyle::Fermionic);
    conformance(U1, (-3..=3).map(|q| U1Irrep::new(q).unwrap()).collect());
    let u1 = crate::common::lie::torus();
    let labels = (-3..=3)
        .map(|q| u1.irrep_with_charges(&[], &[q]).unwrap())
        .collect();
    conformance(u1, labels);
    for n in [2, 3, 5] {
        assert_eq!(ZN::new(n).unwrap().fusion_style(), FusionStyle::Unique);
        assert_eq!(ZN::new(n).unwrap().braiding_style(), BraidingStyle::Bosonic);
        conformance(
            ZN::new(n).unwrap(),
            (0..n)
                .map(|q| ZNIrrep::from_canonical(q, n).unwrap())
                .collect(),
        );
    }
    conformance(
        FermionParity,
        vec![FermionParityIrrep::EVEN, FermionParityIrrep::ODD],
    );
}

#[test]
fn scalar_storage_and_promotion() {
    fn storage<S: Scalar + SymbolBlock<4, Scalar = S>>(x: S) {
        assert_eq!(x.dims(), [1; 4]);
        assert_eq!(x.get([0; 4]), x);
        assert_eq!(x.as_slice(), &[x]);
    }
    storage(0.0);
    storage(Complex64::new(0.0, 0.0));
    storage(Complex64::new(2.0, 3.0));
    assert_eq!(<f64 as Promote<f64>>::left(2.0), 2.0);
    assert_eq!(<f64 as Promote<f64>>::right(3.0), 3.0);
    let z = Complex64::new(2.0, 3.0);
    assert_eq!(
        <f64 as Promote<Complex64>>::left(2.0),
        Complex64::new(2.0, 0.0)
    );
    assert_eq!(<f64 as Promote<Complex64>>::right(z), z);
    assert_eq!(<Complex64 as Promote<f64>>::left(z), z);
    assert_eq!(
        <Complex64 as Promote<f64>>::right(2.0),
        Complex64::new(2.0, 0.0)
    );
    assert_eq!(<Complex64 as Promote<Complex64>>::left(z), z);
    assert_eq!(<Complex64 as Promote<Complex64>>::right(z), z);
    assert_eq!(Scalar::conj(z), Complex64::new(2.0, -3.0));
    close(Scalar::abs(z), 13.0_f64.sqrt());
}

#[test]
#[should_panic(expected = "symbol index out of range")]
fn real_storage_rejects_out_of_range() {
    <f64 as SymbolBlock<4>>::get(&0.0, [0, 0, 1, 0]);
}

#[test]
#[should_panic(expected = "symbol index out of range")]
fn complex_storage_rejects_out_of_range() {
    <Complex64 as SymbolBlock<2>>::get(&Complex64::new(0.0, 0.0), [1, 0]);
}

#[test]
fn cu1_identities() {
    assert_eq!(CU1.fusion_style(), FusionStyle::Simple);
    assert_eq!(CU1.braiding_style(), BraidingStyle::Bosonic);
    conformance(
        CU1,
        [CU1Irrep::VACUUM, CU1Irrep::PSEUDOSCALAR]
            .into_iter()
            .chain((1..=4).map(CU1Irrep::from_twice_charge))
            .collect(),
    );
}

#[test]
fn shipped_table_identities() {
    for uuid in TableCategory::shipped_uuids() {
        let cat = TableCategory::by_uuid(uuid).unwrap();
        let labels = cat.objects().collect();
        conformance(cat, labels);
    }
}

#[test]
fn su2_identities_through_dj6() {
    use fusion_symbols::lie::{GroupId, LieGroup};
    let cat = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    conformance(
        cat.clone(),
        (0..=6).map(|dj| cat.irrep(&[dj]).unwrap()).collect(),
    );
}

#[test]
fn so3_identities() {
    use fusion_symbols::lie::{GroupId, LieGroup};
    let cat = LieGroup::new(GroupId::so(3).unwrap()).unwrap();
    conformance(
        cat.clone(),
        (0..=6)
            .step_by(2)
            .map(|dj| cat.irrep(&[dj]).unwrap())
            .collect(),
    );
}

#[cfg(feature = "lie-generated")]
#[test]
fn su3_and_su4_identities() {
    use fusion_symbols::lie::{GroupId, LieGroup};
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    conformance(
        su3.clone(),
        (0..=2)
            .flat_map(|p| (0..=2 - p).map(move |q| [p, q]))
            .map(|x| su3.irrep(&x).unwrap())
            .collect(),
    );
    let su4 = LieGroup::new(GroupId::su(4).unwrap()).unwrap();
    conformance(
        su4.clone(),
        [[0, 0, 0], [1, 0, 0], [0, 0, 1]]
            .map(|x| su4.irrep(&x).unwrap())
            .into(),
    );
}

#[cfg(feature = "lie-generated")]
mod bcd_conformance {
    use super::*;
    use fusion_symbols::lie::{GroupId, LieGroup};

    fn identities(group: GroupId, label: &[i64]) {
        let cat = LieGroup::new(group).unwrap();
        conformance(cat.clone(), vec![cat.unit(), cat.irrep(label).unwrap()]);
    }
    #[test]
    fn spin5() {
        identities(GroupId::spin(5).unwrap(), &[0, 1]);
    }
    #[test]
    fn sp4() {
        identities(GroupId::sp(4).unwrap(), &[1, 0]);
    }
    #[test]
    fn so5() {
        identities(GroupId::so(5).unwrap(), &[1, 0]);
    }
    #[test]
    fn spin7() {
        identities(GroupId::spin(7).unwrap(), &[0, 0, 1]);
    }
    #[test]
    fn multiplicity_two() {
        // C2: (2,0) ⊗ (1,1) contains (1,1) twice (GroupMath 1.1.3).
        let cat = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
        let a = cat.irrep(&[2, 0]).unwrap();
        let b = cat.irrep(&[1, 1]).unwrap();
        assert_eq!(cat.nsymbol(&a, &b, &b), 2);
        // Unit associativity and symmetric braiding on that multiplicity space.
        let f = cat.f_symbol(&cat.unit(), &a, &b, &b, &a, &b).unwrap();
        let r = cat.r_symbol(&a, &b, &b).unwrap();
        let reverse = cat.r_symbol(&b, &a, &b).unwrap();
        assert_eq!(f.dims(), [1, 2, 2, 1]);
        for mu in 0..2 {
            for nu in 0..2 {
                close(f.at(0, mu, nu, 0), if mu == nu { 1.0 } else { 0.0 });
                close(
                    (0..2).map(|k| r.at(mu, k) * reverse.at(k, nu)).sum::<f64>(),
                    if mu == nu { 1.0 } else { 0.0 },
                );
            }
        }
    }
}

#[test]
fn product_identities() {
    use fusion_symbols::{
        lie::{GroupId, LieGroup},
        product::Product,
    };
    let parity = FermionParityIrrep::ODD;
    let fu1 = Product::new((U1, FermionParity));
    conformance(
        fu1,
        vec![
            (U1.unit(), FermionParityIrrep::EVEN),
            (U1Irrep::new(1).unwrap(), parity),
        ],
    );
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = su2.irrep(&[1]).unwrap();
    let fsu2 = Product::new((su2.clone(), FermionParity));
    conformance(fsu2.clone(), vec![fsu2.unit(), (half.clone(), parity)]);
    let fib = TableCategory::fibonacci();
    let tau = fib.object(2).unwrap();
    let fibsu2 = Product::new((fib, su2.clone()));
    conformance(fibsu2.clone(), vec![fibsu2.unit(), (tau, half.clone())]);
    let ss = Product::new((su2.clone(), su2));
    conformance(ss.clone(), vec![ss.unit(), (half.clone(), half)]);
}

#[cfg(feature = "lie-generated")]
#[test]
fn product_multiplicity_residuals() {
    use fusion_symbols::{
        category::{f_unitarity_residual, hexagon_residual, pentagon_residual},
        lie::{GroupId, LieGroup},
        product::Product,
    };
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let octet = su3.irrep(&[1, 1]).unwrap();
    let cat = Product::new((su3, FermionParity));
    let o = (octet, FermionParityIrrep::ODD);
    let u = cat.unit();
    assert_eq!(
        cat.nsymbol(&o, &o, &(o.0.clone(), FermionParityIrrep::EVEN)),
        2
    );
    close(
        f_unitarity_residual(&cat, &u, &o, &o, &(o.0.clone(), FermionParityIrrep::EVEN)).unwrap(),
        0.0,
    );
    close(hexagon_residual(&cat, &u, &o, &o).unwrap(), 0.0);
    close(pentagon_residual(&cat, &u, &u, &o, &o).unwrap(), 0.0);
}

#[test]
fn composite_lie_identities() {
    use fusion_symbols::lie::{GroupId, LieGroup};
    for id in [GroupId::spin(4).unwrap(), GroupId::so(4).unwrap()] {
        let cat = LieGroup::new(id).unwrap();
        let labels = vec![cat.unit(), cat.irrep(&[1, 1]).unwrap()];
        conformance(cat, labels);
    }
}

#[test]
fn u2_identities() {
    use fusion_symbols::lie::{GroupId, LieGroup};
    let cat = LieGroup::new(GroupId::u(2).unwrap()).unwrap();
    let labels = vec![
        cat.unit(),
        cat.irrep_with_charges(&[1], &[1]).unwrap(),
        cat.irrep_with_charges(&[1], &[-1]).unwrap(),
        cat.irrep_with_charges(&[0], &[2]).unwrap(),
    ];
    conformance(cat, labels);
}

#[cfg(feature = "lie-generated")]
#[test]
fn g2_identities() {
    use fusion_symbols::lie::{GroupId, LieGroup};
    let cat = LieGroup::new(GroupId::g2()).unwrap();
    let labels = vec![cat.unit(), cat.irrep(&[1, 0]).unwrap()];
    conformance(cat, labels);
}
