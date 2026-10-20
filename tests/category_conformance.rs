//! Multiplicity-free identities from TensorKitSectors 0.3.9 `src/sectors.jl`:
//! triangle_equation, pentagon_equation, hexagon_equation,
//! frobenius_schur_phase_from_Fsymbol and twist_from_Rsymbol.
//! Fusion outcomes are enumerated by the category, including intermediate
//! U(1) charges outside the finite input sample. Shape is determined by N.
use fusion_symbols::{
    abelian::{U1Irrep, ZNIrrep, U1, ZN},
    category::{BraidingStyle, FusionCategory, FusionStyle, Promote, Scalar, SymbolBlock},
    fermion::{FermionParity, FermionParityIrrep},
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

fn r<C: FusionCategory>(cat: &C, a: &C::Object, b: &C::Object, c: &C::Object) -> C::Scalar {
    let block = cat.r_symbol(a, b, c).unwrap();
    if cat.nsymbol(a, b, c) == 0 {
        assert!(block.as_slice().iter().all(|&x| x == C::Scalar::zero()));
        C::Scalar::zero()
    } else {
        block.get([0; 2])
    }
}

// These slices implement multiplicity-free categories. Higher-multiplicity
// conformance needs sums over vertex indices when that family is migrated.
fn conformance<C: FusionCategory>(cat: C, labels: Vec<C::Object>) {
    assert_ne!(cat.fusion_style(), FusionStyle::Generic);
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
        let mut twist = zero;
        for b in outcomes(&cat, a, a) {
            twist = twist + C::Scalar::from_f64(cat.dim(&b) / cat.dim(a)) * r(&cat, a, a, &b);
        }
        close(cat.twist(a), twist);
        for b in &labels {
            let dimension_sum: f64 = outcomes(&cat, a, b)
                .iter()
                .map(|c| f64::from(cat.nsymbol(a, b, c)) * cat.dim(c))
                .sum();
            close(cat.dim(a) * cat.dim(b), dimension_sum);
            for c in outcomes(&cat, a, b) {
                // Triangle/unit F moves.
                close(f(&cat, [&unit, a, b, &c, a, &c]), one);
                close(f(&cat, [a, &unit, b, &c, a, b]), one);
                close(f(&cat, [a, b, &unit, &c, &c, b]), one);
            }
            for c in &labels {
                if matches!(
                    cat.braiding_style(),
                    BraidingStyle::Bosonic | BraidingStyle::Fermionic
                ) {
                    assert_eq!(cat.nsymbol(a, b, c), cat.nsymbol(b, a, c));
                    if cat.nsymbol(a, b, c) != 0 {
                        close(r(&cat, a, b, c) * r(&cat, b, a, c), one);
                    }
                }
                // Complete F row orthogonality over all right channels.
                for e1 in outcomes(&cat, a, b) {
                    for d in outcomes(&cat, &e1, c) {
                        for e2 in outcomes(&cat, a, b) {
                            if cat.nsymbol(&e2, c, &d) == 0 {
                                continue;
                            }
                            let mut dot = zero;
                            for right in outcomes(&cat, b, c) {
                                dot = dot
                                    + f(&cat, [a, b, c, &d, &e1, &right])
                                        * f(&cat, [a, b, c, &d, &e2, &right]).conj();
                            }
                            close(dot, if e1 == e2 { one } else { zero });
                        }
                    }
                }
                // Both hexagons, using the reference's axis/argument order.
                for e in outcomes(&cat, c, a) {
                    for right in outcomes(&cat, c, b) {
                        for d in outcomes(&cat, &e, b) {
                            if cat.nsymbol(a, &right, &d) == 0 {
                                continue;
                            }
                            let middle = f(&cat, [a, c, b, &d, &e, &right]);
                            let lhs = r(&cat, c, a, &e) * middle * r(&cat, c, b, &right);
                            let lhs_under =
                                r(&cat, a, c, &e).conj() * middle * r(&cat, b, c, &right).conj();
                            let (mut rhs, mut rhs_under) = (zero, zero);
                            for g in outcomes(&cat, a, b) {
                                let left = f(&cat, [c, a, b, &d, &e, &g]);
                                let right_f = f(&cat, [a, b, c, &d, &g, &right]);
                                rhs = rhs + left * r(&cat, c, &g, &d) * right_f;
                                rhs_under = rhs_under + left * r(&cat, &g, c, &d).conj() * right_f;
                            }
                            close(lhs, rhs);
                            close(lhs_under, rhs_under);
                        }
                    }
                }
                for d in &labels {
                    // Pentagon, including every internal channel.
                    for left in outcomes(&cat, a, b) {
                        for h in outcomes(&cat, c, d) {
                            for g in outcomes(&cat, &left, c) {
                                for i in outcomes(&cat, b, &h) {
                                    for e in outcomes(&cat, &g, d) {
                                        if cat.nsymbol(a, &i, &e) == 0 {
                                            continue;
                                        }
                                        let lhs = f(&cat, [&left, c, d, &e, &g, &h])
                                            * f(&cat, [a, b, &h, &e, &left, &i]);
                                        let mut rhs = zero;
                                        for j in outcomes(&cat, b, c) {
                                            rhs = rhs
                                                + f(&cat, [a, b, c, &g, &left, &j])
                                                    * f(&cat, [a, &j, d, &e, &g, &i])
                                                    * f(&cat, [b, c, d, &i, &j, &h]);
                                        }
                                        close(lhs, rhs);
                                    }
                                }
                            }
                        }
                    }
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
