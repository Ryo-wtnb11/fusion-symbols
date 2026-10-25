//! Runtime products share the tuple composition law, including its two axis orders.
use fusion_symbols::{
    category::{
        CgcBlock, EnumerationError, FusionCategory, FusionStyle, GroupRepresentations, SymbolBlock,
    },
    finite::{ZNIrrep, ZN},
    lie::{GroupId, LieError, LieGroup},
    product::{Product, ProductError},
};

fn su(n: usize) -> LieGroup {
    LieGroup::new(GroupId::su(n).unwrap()).unwrap()
}
fn bits<const R: usize>(b: impl SymbolBlock<R, Scalar = f64>) -> ([usize; R], Vec<u64>) {
    (b.dims(), b.as_slice().iter().map(|x| x.to_bits()).collect())
}

fn compare<C, T>(
    runtime: &Product<Vec<C>>,
    tuple: &T,
    flatten: impl Fn(&T::Object) -> Vec<C::Object>,
    labels: [T::Object; 6],
) where
    C: GroupRepresentations<Scalar = f64>,
    for<'a> C::Fusion<'a>: Clone,
    T: GroupRepresentations<Scalar = f64>,
{
    assert_eq!(runtime.unit(), flatten(&tuple.unit()));
    assert_eq!(runtime.fusion_style(), tuple.fusion_style());
    assert_eq!(runtime.braiding_style(), tuple.braiding_style());
    assert_eq!(runtime.has_canonical_unit(), tuple.has_canonical_unit());
    assert_eq!(
        runtime.has_trivial_associator_gauge(),
        tuple.has_trivial_associator_gauge()
    );
    assert_eq!(runtime.object_count(), tuple.object_count());
    // A singleton still has Product fingerprint framing; its coefficient
    // values and enumeration agree with the bare factor.
    if runtime.factors().len() != 1 {
        assert_eq!(
            runtime.authority_fingerprint(),
            tuple.authority_fingerprint()
        );
    }
    for i in 0..24 {
        match tuple.object_at(i) {
            Ok(a) => {
                assert_eq!(runtime.object_at(i), Ok(flatten(&a)));
                assert_eq!(runtime.object_index(&flatten(&a)), tuple.object_index(&a));
            }
            Err(e) => assert_eq!(runtime.object_at(i), Err(e)),
        }
    }
    for a in &labels {
        let va = flatten(a);
        assert_eq!(runtime.dual(&va), flatten(&tuple.dual(a)));
        for (x, y) in [
            (runtime.dim(&va), tuple.dim(a)),
            (runtime.sqrt_dim(&va), tuple.sqrt_dim(a)),
            (runtime.inv_dim(&va), tuple.inv_dim(a)),
            (runtime.inv_sqrt_dim(&va), tuple.inv_sqrt_dim(a)),
            (
                runtime.frobenius_schur_phase(&va),
                tuple.frobenius_schur_phase(a),
            ),
            (
                runtime.frobenius_schur_indicator(&va),
                tuple.frobenius_schur_indicator(a),
            ),
            (runtime.twist(&va), tuple.twist(a)),
        ] {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        for b in &labels {
            let vb = flatten(b);
            assert_eq!(
                runtime.fusion(&va, &vb).unwrap().collect::<Vec<_>>(),
                tuple
                    .fusion(a, b)
                    .unwrap()
                    .map(|c| flatten(&c))
                    .collect::<Vec<_>>()
            );
            for c in &labels {
                let vc = flatten(c);
                assert_eq!(runtime.nsymbol(&va, &vb, &vc), tuple.nsymbol(a, b, c));
                assert_eq!(
                    runtime.try_nsymbol(&va, &vb, &vc).unwrap(),
                    tuple.try_nsymbol(a, b, c).unwrap()
                );
                assert_eq!(
                    bits(runtime.r_symbol(&va, &vb, &vc).unwrap()),
                    bits(tuple.r_symbol(a, b, c).unwrap())
                );
                assert_eq!(
                    bits(runtime.a_symbol(&va, &vb, &vc).unwrap()),
                    bits(tuple.a_symbol(a, b, c).unwrap())
                );
                assert_eq!(
                    bits(runtime.b_symbol(&va, &vb, &vc).unwrap()),
                    bits(tuple.b_symbol(a, b, c).unwrap())
                );
            }
        }
    }
    let mut f_labels: Vec<_> = labels.iter().collect();
    f_labels.sort();
    f_labels.dedup();
    let mut unequal_channels = 0;
    for &a in &f_labels {
        for &b in &f_labels {
            for &e in &f_labels {
                if tuple.nsymbol(a, b, e) == 0 {
                    continue;
                }
                for &c in &f_labels {
                    for &d in &f_labels {
                        if tuple.nsymbol(e, c, d) == 0 {
                            continue;
                        }
                        for &f in &f_labels {
                            if tuple.nsymbol(b, c, f) == 0 || tuple.nsymbol(a, f, d) == 0 {
                                continue;
                            }
                            let [va, vb, vc, vd, ve, vf] = [a, b, c, d, e, f].map(&flatten);
                            assert_eq!(
                                bits(runtime.f_symbol(&va, &vb, &vc, &vd, &ve, &vf).unwrap()),
                                bits(tuple.f_symbol(a, b, c, d, e, f).unwrap()),
                                "F({a:?}, {b:?}, {c:?}, {d:?}, {e:?}, {f:?})"
                            );
                            unequal_channels += usize::from(e != f);
                        }
                    }
                }
            }
        }
    }
    assert!(
        unequal_channels > 0,
        "F coverage must include unequal e/f channels"
    );
    let [a, b, c, d, e, f] = &labels;
    let [va, vb, vc, vd, ve, vf] = labels.each_ref().map(&flatten);
    assert_eq!(
        bits(runtime.f_symbol(&va, &vb, &vc, &vd, &ve, &vf).unwrap()),
        bits(tuple.f_symbol(a, b, c, d, e, f).unwrap())
    );
    compare_cgc(
        &runtime.cgc(&va, &vb, &ve).unwrap(),
        &tuple.cgc(a, b, e).unwrap(),
    );
}

fn compare_cgc(a: &impl CgcBlock, b: &impl CgcBlock) {
    assert_eq!(a.dims(), b.dims());
    let [da, db, dc, n] = a.dims();
    for i in 0..da {
        for j in 0..db {
            for k in 0..dc {
                for mu in 0..n {
                    let index = [i, j, k, mu];
                    assert_eq!(a.at(index).to_bits(), b.at(index).to_bits(), "{index:?}");
                }
            }
        }
    }
}

#[test]
fn runtime_matches_tuples() {
    let s = su(2);
    let h = s.irrep(&[1], &[]).unwrap();
    let one = s.irrep(&[2], &[]).unwrap();
    let u = s.unit();
    let two = Product::new((s.clone(), s.clone()));
    compare(
        &Product::new(vec![s.clone(), s.clone()]),
        &two,
        |(a, b)| vec![a.clone(), b.clone()],
        [
            (h.clone(), one.clone()),
            (h.clone(), one.clone()),
            (one.clone(), one.clone()),
            (u.clone(), u.clone()),
            (u.clone(), one.clone()),
            (one.clone(), u.clone()),
        ],
    );
    let so3 = LieGroup::new(GroupId::so(3).unwrap()).unwrap();
    let vector = so3.irrep(&[2], &[]).unwrap();
    let a = (one.clone(), vector, one.clone());
    compare(
        &Product::new(vec![s.clone(), so3.clone(), s.clone()]),
        &Product::new((s.clone(), so3.clone(), s.clone())),
        |(a, b, c)| vec![a.clone(), b.clone(), c.clone()],
        [
            a.clone(),
            a.clone(),
            a.clone(),
            a.clone(),
            a,
            (s.unit(), so3.unit(), s.unit()),
        ],
    );
    let z = |q, n| ZNIrrep::new(q, n).unwrap();
    let factors = [
        ZN::new(2).unwrap(),
        ZN::new(3).unwrap(),
        ZN::new(4).unwrap(),
    ];
    let a = (z(1, 2), z(1, 3), z(1, 4));
    let b = (z(1, 2), z(2, 3), z(3, 4));
    let u = (z(0, 2), z(0, 3), z(0, 4));
    let runtime = Product::new(factors.to_vec());
    compare(
        &runtime,
        &Product::new((factors[0], factors[1], factors[2])),
        |&(a, b, c)| vec![a, b, c],
        [a, b, a, a, u, u],
    );
    // Independent charge-sum oracle: (1+1 mod 2, 1+2 mod 3, 1+3 mod 4) = (0,0,0).
    assert_eq!(
        runtime
            .fusion(&vec![a.0, a.1, a.2], &vec![b.0, b.1, b.2])
            .unwrap()
            .collect::<Vec<_>>(),
        [vec![u.0, u.1, u.2]]
    );
    // SU(2) singlet recoupling F(h,h,h,h,0,0) = -1/2 in the frozen gauge.
    // Two factors therefore give (+1/4), independently of tuple composition.
    let runtime = Product::new(vec![s.clone(), s.clone()]);
    let hh = vec![h.clone(), h];
    let uu = runtime.unit();
    assert_eq!(
        runtime
            .f_symbol(&hh, &hh, &hh, &hh, &uu, &uu)
            .unwrap()
            .as_slice(),
        &[0.25]
    );
    let hu = vec![hh[0].clone(), s.unit()];
    assert_eq!(
        runtime
            .f_symbol(&hu, &hu, &hu, &hu, &uu, &uu)
            .unwrap()
            .as_slice(),
        &[-0.5]
    );
    let a = vec![s.irrep(&[1], &[]).unwrap(), one.clone()];
    let c = vec![one, s.unit()];
    let cgc = runtime.cgc(&a, &a, &c).unwrap();
    // Last-fastest [3,2,1,0] decodes as [1,0,1,0] x [0,2,0,0].
    // Triplet m=0 has 1/sqrt(2); spin-one singlet (+1,-1) has 1/sqrt(3).
    assert!((cgc.at([3, 2, 1, 0]) - 1.0 / 6.0_f64.sqrt()).abs() < 1e-14);
}

#[cfg(feature = "lie-generated")]
#[test]
fn runtime_generated_multiplicity_matches_tuple() {
    let s = su(3);
    let t = su(2);
    let a = (s.irrep(&[1, 1], &[]).unwrap(), t.irrep(&[2], &[]).unwrap());
    let runtime = Product::new(vec![s.clone(), t.clone()]);
    let va = vec![a.0.clone(), a.1.clone()];
    assert_eq!(runtime.nsymbol(&va, &va, &va), 2);
    let u = (s.unit(), t.unit());
    compare(
        &runtime,
        &Product::new((s, t)),
        |(a, b)| vec![a.clone(), b.clone()],
        [a.clone(), a.clone(), a.clone(), a.clone(), a, u],
    );
}

#[test]
fn empty_and_singleton() {
    let empty = Product::new(Vec::<LieGroup>::new());
    let u = empty.unit();
    assert!(u.is_empty());
    assert_eq!(empty.object_count(), Ok(Some(1)));
    assert_eq!(empty.object_at(0), Ok(u.clone()));
    assert_eq!(empty.object_at(1), Err(EnumerationError::OutOfRange));
    assert_eq!(empty.object_index(&u), Ok(0));
    assert_eq!(empty.fusion_style(), FusionStyle::Unique);
    assert!(empty.has_trivial_associator_gauge() && empty.has_canonical_unit());
    for v in [
        empty.dim(&u),
        empty.sqrt_dim(&u),
        empty.inv_dim(&u),
        empty.inv_sqrt_dim(&u),
        empty.frobenius_schur_phase(&u),
        empty.frobenius_schur_indicator(&u),
        empty.twist(&u),
    ] {
        assert_eq!(v, 1.0);
    }
    assert_eq!(empty.nsymbol(&u, &u, &u), 1);
    assert_eq!(empty.try_nsymbol(&u, &u, &u), Ok(1));
    assert_eq!(
        empty.fusion(&u, &u).unwrap().collect::<Vec<_>>(),
        std::slice::from_ref(&u)
    );
    assert_eq!(
        bits(empty.f_symbol(&u, &u, &u, &u, &u, &u).unwrap()),
        ([1; 4], vec![1.0_f64.to_bits()])
    );
    for b in [
        empty.r_symbol(&u, &u, &u).unwrap(),
        empty.a_symbol(&u, &u, &u).unwrap(),
        empty.b_symbol(&u, &u, &u).unwrap(),
    ] {
        assert_eq!(bits(b), ([1; 2], vec![1.0_f64.to_bits()]));
    }
    let cgc = empty.cgc(&u, &u, &u).unwrap();
    assert_eq!(cgc.dims(), [1; 4]);
    assert_eq!(cgc.at([0; 4]), 1.0);
    assert!(std::panic::catch_unwind(|| cgc.at([1, 0, 0, 0])).is_err());
    let s = su(2);
    let a = s.irrep(&[1], &[]).unwrap();
    let u = s.unit();
    compare(
        &Product::new(vec![s.clone()]),
        &s,
        |a| vec![a.clone()],
        [
            a.clone(),
            a.clone(),
            s.irrep(&[2], &[]).unwrap(),
            a,
            u,
            s.irrep(&[2], &[]).unwrap(),
        ],
    );
}

#[test]
fn fusion_indices_and_factor_errors() {
    let s = su(2);
    let cat = Product::new(vec![s.clone(); 3]);
    let a = [1, 2, 3].map(|n| s.irrep(&[n], &[]).unwrap()).to_vec();
    let iter = cat.fusion(&a, &a).unwrap();
    let forward: Vec<_> = iter.clone().collect();
    assert_eq!(iter.len(), 24);
    assert_eq!(
        iter.clone().rev().collect::<Vec<_>>(),
        forward.iter().rev().cloned().collect::<Vec<_>>()
    );
    for n in [0, 1, 7, 23, 24, usize::MAX] {
        let mut it = iter.clone();
        assert_eq!(it.nth(n), forward.get(n).cloned());
        assert_eq!(it.len(), forward.len().saturating_sub(n.saturating_add(1)));
        assert_eq!(
            it.collect::<Vec<_>>(),
            forward[n.saturating_add(1).min(forward.len())..]
        );
        let mut it = iter.clone();
        assert_eq!(
            it.nth_back(n),
            forward
                .len()
                .checked_sub(n)
                .and_then(|i| i.checked_sub(1))
                .map(|i| forward[i].clone())
        );
        assert_eq!(it.len(), forward.len().saturating_sub(n.saturating_add(1)));
        assert_eq!(
            it.collect::<Vec<_>>(),
            forward[..forward.len().saturating_sub(n.saturating_add(1))]
        );
    }
    let mut mixed = iter;
    assert_eq!(mixed.next(), forward.first().cloned());
    assert_eq!(mixed.next_back(), forward.last().cloned());
    assert_eq!(mixed.nth(3), Some(forward[4].clone()));
    assert_eq!(mixed.nth_back(2), Some(forward[20].clone()));
    assert_eq!(mixed.len(), 15);
    assert_eq!(mixed.collect::<Vec<_>>(), forward[5..20]);
    let huge = s.irrep(&[i64::from(u32::MAX)], &[]).unwrap();
    for index in 0..3 {
        let mut a = cat.unit();
        a[index] = huge.clone();
        assert!(
            matches!(cat.fusion(&a,&a),Err(ProductError::Factor { index: i, source: LieError::LabelOverflow }) if i == index)
        );
    }
    let h = vec![huge.clone(); 3];
    let u = cat.unit();
    assert!(matches!(cat.cgc(&h, &u, &u), Err(ProductError::Overflow)));
    for index in 0..3 {
        let mut a = u.clone();
        a[index] = huge.clone();
        assert!(
            matches!(cat.cgc(&a,&a,&u),Err(ProductError::Factor { index: i, source: LieError::Unsupported { .. } }) if i == index)
        );
    }
    let foreign = vec![s.unit(); 2];
    assert!(!cat.contains(&foreign));
    assert!(std::panic::catch_unwind(|| cat.f_symbol(&a, &a, &a, &a, &a, &foreign)).is_err());
    assert!(std::panic::catch_unwind(|| cat.dim(&foreign)).is_err());
}
