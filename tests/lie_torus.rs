//! Integer charges and classical highest-weight identities are independent oracles.
use fusion_symbols::{
    category::{
        CgcBlock, EnumerationError, FusionCategory, FusionStyle, GroupRepresentations, SymbolBlock,
    },
    lie::{GroupId, LieError, LieGroup, LieIrrep, RootSystem::A, U1Error, U1Irrep, U1},
    product::Product,
};

fn group(id: GroupId) -> LieGroup {
    LieGroup::new(id).unwrap()
}
fn charge(g: &LieGroup, q: i64) -> LieIrrep {
    g.irrep(&[], &[q]).unwrap()
}

#[test]
fn u1_values_bounds_and_fingerprint() {
    let g = group(GroupId::u(1).unwrap());
    let q = |n| charge(&g, n);
    assert_eq!(g.fusion_style(), FusionStyle::Unique);
    assert!(g.has_trivial_associator_gauge() && g.has_canonical_unit());
    assert_eq!(g.authority_fingerprint(), b"fusion-symbols:abelian-u1:labels=i32-integer-no-min:n=exact:f=tks-irreps:r=tks-irreps:dim=one:fs=tks-irreps-one:twist=one:scalar=f64:epoch=1");
    for n in [-i64::from(i32::MAX), -5, 0, 3, i64::from(i32::MAX)] {
        let a = q(n);
        assert!(a.dynkin().is_empty());
        assert_eq!(a.charges().collect::<Vec<_>>(), [n]);
        let index = (if n > 0 { 2 * n - 1 } else { -2 * n }) as u64;
        assert_eq!(g.object_index(&a), Ok(index));
        assert_eq!(g.object_at(index), Ok(a.clone()));
        assert_eq!(g.dual(&a), q(-n));
        assert_eq!(g.dual(&g.dual(&a)), a);
        assert_eq!(a.dim_exact(), 1.into());
        assert_eq!(a.weight_multiplicities(), [(vec![n], 1)].into());
        assert_eq!(
            (g.dim(&a), g.frobenius_schur_phase(&a), g.twist(&a)),
            (1.0, 1.0, 1.0)
        );
        assert_eq!(
            g.frobenius_schur_indicator(&a),
            if n == 0 { 1.0 } else { 0.0 }
        );
    }
    for (a, b, c) in [(3, -5, -2), (-4, -7, -11), (5, -5, 0), (0, 0, 0)] {
        let (a, b, c) = (q(a), q(b), q(c));
        assert_eq!(
            g.fusion(&a, &b).unwrap().collect::<Vec<_>>(),
            std::slice::from_ref(&c)
        );
        assert_eq!(g.nsymbol(&a, &b, &c), 1);
        for block in [
            g.r_symbol(&a, &b, &c),
            g.a_symbol(&a, &b, &c),
            g.b_symbol(&a, &b, &c),
        ] {
            assert_eq!(block.unwrap().data(), [1.0]);
        }
        let cg = g.cgc(&a, &b, &c).unwrap();
        assert_eq!(cg.dims(), [1; 4]);
        assert_eq!(cg.at([0; 4]), 1.0);
        let absent = q(c.charges().next().unwrap() + 1);
        assert_eq!(g.nsymbol(&a, &b, &absent), 0);
        assert_eq!(g.cgc(&a, &b, &absent).unwrap().dims(), [1, 1, 1, 0]);
        assert!(g.r_symbol(&a, &b, &absent).unwrap().data().is_empty());
        assert!(g.a_symbol(&a, &b, &absent).unwrap().data().is_empty());
        assert!(g.b_symbol(&a, &b, &absent).unwrap().data().is_empty());
    }
    // 3 + (-5) + 4 = 2, with intermediate charges -2 and -1.
    assert_eq!(
        g.f_symbol(&q(3), &q(-5), &q(4), &q(2), &q(-2), &q(-1))
            .unwrap()
            .data(),
        [1.0]
    );
    assert_eq!(
        g.f_symbol(&q(3), &q(-5), &q(4), &q(2), &q(0), &q(-1))
            .unwrap()
            .dims(),
        [0, 0, 1, 1]
    );
    for n in [
        i64::from(i32::MAX) + 1,
        -i64::from(i32::MAX) - 1,
        i64::from(i32::MIN),
        i64::MIN,
        i64::MAX,
    ] {
        assert_eq!(g.irrep(&[], &[n]), Err(LieError::LabelOverflow));
    }
    for (a, b) in [(i64::from(i32::MAX), 1), (-i64::from(i32::MAX), -1)] {
        let (a, b) = (q(a), q(b));
        assert_eq!(g.fusion(&a, &b).unwrap_err(), LieError::LabelOverflow);
        assert_eq!(g.nsymbol(&a, &b, &g.unit()), 0);
        assert_eq!(g.try_nsymbol(&a, &b, &g.unit()), Ok(0));
        assert!(g
            .f_symbol(&a, &b, &g.unit(), &g.unit(), &g.unit(), &b)
            .unwrap()
            .data()
            .is_empty());
        for block in [
            g.r_symbol(&a, &b, &g.unit()),
            g.a_symbol(&a, &b, &g.unit()),
            g.b_symbol(&a, &b, &g.unit()),
        ] {
            assert_eq!(block.unwrap().dims(), [0, 0]);
        }
        assert_eq!(g.cgc(&a, &b, &g.unit()).unwrap().dims(), [1, 1, 1, 0]);
    }
}

#[test]
fn typed_u1_matches_liegroup_bitwise() {
    let g = group(U1.group_id());
    assert_eq!(U1.group_id(), GroupId::u(1).unwrap());
    assert_eq!(
        U1.unit().charge(),
        g.unit().charges().next().unwrap() as i32
    );
    assert_eq!(U1.fusion_style(), g.fusion_style());
    assert_eq!(U1.braiding_style(), g.braiding_style());
    assert_eq!(
        U1.has_trivial_associator_gauge(),
        g.has_trivial_associator_gauge()
    );
    assert_eq!(U1.has_canonical_unit(), g.has_canonical_unit());
    assert_eq!(U1.authority_fingerprint(), g.authority_fingerprint());
    assert_eq!(U1.object_count(), g.object_count());
    for index in (0..100).chain([4_294_967_293, 4_294_967_294, 4_294_967_295, u64::MAX]) {
        assert_eq!(
            U1.object_at(index).map(|q| i64::from(q.charge())),
            g.object_at(index).map(|q| q.charges().next().unwrap())
        );
    }
    let charges = [
        -i32::MAX,
        -i32::MAX + 1,
        -3,
        -1,
        0,
        1,
        3,
        i32::MAX - 1,
        i32::MAX,
    ];
    // Both calls are closed-form: no generation sweep or coefficient cache exists.
    for _ in 0..2 {
        for qa in charges {
            let (a, la) = (U1Irrep::new(qa).unwrap(), charge(&g, i64::from(qa)));
            // Independent oracle: equality alone cannot catch a shared broken dual.
            assert_eq!(U1.dual(&a).charge(), -qa);
            assert_eq!(g.dual(&la).charges().next(), Some(i64::from(-qa)));
            assert_eq!(U1.object_index(&a), g.object_index(&la));
            for (typed, general) in [
                (U1.dim(&a), g.dim(&la)),
                (U1.frobenius_schur_phase(&a), g.frobenius_schur_phase(&la)),
                (
                    U1.frobenius_schur_indicator(&a),
                    g.frobenius_schur_indicator(&la),
                ),
                (U1.twist(&a), g.twist(&la)),
            ] {
                assert_eq!(typed.to_bits(), general.to_bits());
            }
            for qb in charges {
                let (b, lb) = (U1Irrep::new(qb).unwrap(), charge(&g, i64::from(qb)));
                let sum = i64::from(qa) + i64::from(qb);
                match (U1.fusion(&a, &b), g.fusion(&la, &lb)) {
                    (Ok(typed), Ok(general)) => {
                        let typed: Vec<_> = typed.map(|q| i64::from(q.charge())).collect();
                        let general: Vec<_> =
                            general.map(|q| q.charges().next().unwrap()).collect();
                        assert_eq!(typed, [sum]);
                        assert_eq!(typed, general);
                    }
                    (Err(U1Error::U1ChargeOutOfRange { charge }), Err(LieError::LabelOverflow)) => {
                        assert_eq!(charge, sum);
                        assert!(sum.abs() > i64::from(i32::MAX));
                    }
                    other => panic!("fusion mismatch: {other:?}"),
                }
                for qc in charges {
                    let (c, lc) = (U1Irrep::new(qc).unwrap(), charge(&g, i64::from(qc)));
                    let n = u32::from(sum == i64::from(qc));
                    assert_eq!(U1.nsymbol(&a, &b, &c), n);
                    assert_eq!(g.nsymbol(&la, &lb, &lc), n);
                    assert_eq!(U1.try_nsymbol(&a, &b, &c), Ok(n));
                    assert_eq!(g.try_nsymbol(&la, &lb, &lc), Ok(n));
                    let r = g.r_symbol(&la, &lb, &lc).unwrap();
                    assert_eq!(r.dims(), [n as usize; 2]);
                    assert_eq!(
                        U1.r_symbol(&a, &b, &c).unwrap().to_bits(),
                        r.data().first().copied().unwrap_or(0.0).to_bits()
                    );
                    same_block(
                        U1.a_symbol(&a, &b, &c).unwrap(),
                        g.a_symbol(&la, &lb, &lc).unwrap(),
                    );
                    same_block(
                        U1.b_symbol(&a, &b, &c).unwrap(),
                        g.b_symbol(&la, &lb, &lc).unwrap(),
                    );
                    let cg = g.cgc(&la, &lb, &lc).unwrap();
                    assert_eq!(cg.dims(), [1, 1, 1, n as usize]);
                    assert_eq!(
                        U1.cgc(&a, &b, &c).unwrap().to_bits(),
                        (if n == 0 { 0.0 } else { cg.at([0; 4]) }).to_bits()
                    );
                }
            }
        }
    }
    // A separately constructed, equal datum must pass the identity check too.
    let equal = group(GroupId::u(1).unwrap());
    let charges = [-i64::from(i32::MAX), -1, 0, 1, i64::from(i32::MAX)];
    let labels = charges.map(|q| charge(&equal, q));
    for mut index in 0..charges.len().pow(6) {
        let [a, b, c, d, e, f] = std::array::from_fn(|_| {
            let i = index % charges.len();
            index /= charges.len();
            i
        });
        let vertices = [(a, b, e), (e, c, d), (b, c, f), (a, f, d)];
        let sums = vertices.map(|(x, y, _)| charges[x] + charges[y]);
        let result = g.f_symbol(
            &labels[a], &labels[b], &labels[c], &labels[d], &labels[e], &labels[f],
        );
        let dims = std::array::from_fn(|i| usize::from(sums[i] == charges[vertices[i].2]));
        let block = result.unwrap();
        assert_eq!(block.dims(), dims);
        assert_eq!(
            block.data(),
            if dims.contains(&0) {
                &[][..]
            } else {
                &[1.0][..]
            }
        );
        let [a, b, c, d, e, f] =
            [a, b, c, d, e, f].map(|i| U1Irrep::new(charges[i] as i32).unwrap());
        assert_eq!(
            U1.f_symbol(&a, &b, &c, &d, &e, &f).unwrap().to_bits(),
            block.data().first().copied().unwrap_or(0.0).to_bits()
        );
    }
    let foreign = group(GroupId::su(2).unwrap()).unit();
    let one = g.unit();
    for i in 0..6 {
        let mut args = [&one; 6];
        args[i] = &foreign;
        let [a, b, c, d, e, f] = args;
        assert!(std::panic::catch_unwind(|| g.f_symbol(a, b, c, d, e, f)).is_err());
    }
    let mut fusion = g.fusion(&labels[1], &labels[3]).unwrap();
    assert_eq!(fusion.clone().next_back(), Some(one.clone()));
    assert_eq!(fusion.next(), Some(one));
    assert_eq!(fusion.clone().next_back(), None);
    assert_eq!(fusion.len(), 0);
}

#[test]
fn u2_highest_weights_and_charge_conservation() {
    let g = group(GroupId::u(2).unwrap());
    let a = g.irrep(&[1], &[1]).unwrap();
    let shifted = g.irrep(&[1], &[3]).unwrap();
    let det = g.irrep(&[0], &[2]).unwrap();
    assert_ne!(a, shifted); // highest weights (1,0) and (2,1)
    assert_eq!((g.dim(&a), g.dim(&shifted), g.dim(&det)), (2.0, 2.0, 1.0));
    assert_eq!(g.dual(&det), g.irrep(&[0], &[-2]).unwrap());
    assert_eq!(g.irrep(&[1], &[0]), Err(LieError::NotAdmissible));
    // Fundamental weights are ±1 in doubled-spin coordinates, both at charge +1.
    assert_eq!(
        a.weight_multiplicities(),
        [(vec![-1, 1], 1), (vec![1, 1], 1)].into()
    );
    assert_eq!(
        g.fusion(&a, &a).unwrap().collect::<Vec<_>>(),
        [det.clone(), g.irrep(&[2], &[2]).unwrap()]
    );
    for x in [&a, &shifted, &det] {
        assert_eq!(g.frobenius_schur_indicator(x), 0.0);
        assert_eq!(g.dual(x).charges().next(), x.charges().next().map(|q| -q));
        for y in [&a, &shifted, &det] {
            for z in g.fusion(x, y).unwrap() {
                assert_eq!(
                    z.charges().next().unwrap(),
                    x.charges().next().unwrap() + y.charges().next().unwrap()
                );
                assert!(g
                    .group_id()
                    .admits(&z.dynkin(), &z.charges().collect::<Vec<_>>()));
            }
        }
    }
    for result in [
        g.object_count().map(|_| ()),
        g.object_at(0).map(|_| ()),
        g.object_index(&a).map(|_| ()),
    ] {
        assert_eq!(result, Err(EnumerationError::NotEnumerable));
    }
}

fn same_block<const R: usize>(
    a: impl SymbolBlock<R, Scalar = f64>,
    b: impl SymbolBlock<R, Scalar = f64>,
) {
    assert_eq!(a.dims(), b.dims());
    assert_eq!(a.as_slice().len(), b.as_slice().len());
    assert!(a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .all(|(x, y)| x.to_bits() == y.to_bits()));
}

#[test]
fn torus_composites_equal_product() {
    let su2 = group(GroupId::su(2).unwrap());
    let u1 = group(GroupId::u(1).unwrap());
    let product = Product::new(vec![su2, u1]);
    for id in [GroupId::u(2).unwrap(), GroupId::cover(&[A(1)], 1).unwrap()] {
        let g = group(id);
        let split = |x: &LieIrrep| {
            vec![
                product.factors()[0].irrep(&x.dynkin(), &[]).unwrap(),
                charge(&product.factors()[1], x.charges().next().unwrap()),
            ]
        };
        let labels: Vec<_> = [(0, 0), (1, 1), (0, 2), (1, -1), (2, 0)]
            .into_iter()
            .map(|(a, q)| g.irrep(&[a], &[q]).unwrap())
            .collect();
        assert_eq!(g.authority_fingerprint(), product.authority_fingerprint());
        for a in &labels {
            for b in &labels {
                let (pa, pb) = (split(a), split(b));
                for c in g.fusion(a, b).unwrap() {
                    let pc = split(&c);
                    assert_eq!(g.nsymbol(a, b, &c), product.nsymbol(&pa, &pb, &pc));
                    same_block(
                        g.r_symbol(a, b, &c).unwrap(),
                        product.r_symbol(&pa, &pb, &pc).unwrap(),
                    );
                    same_block(
                        g.a_symbol(a, b, &c).unwrap(),
                        product.a_symbol(&pa, &pb, &pc).unwrap(),
                    );
                    same_block(
                        g.b_symbol(a, b, &c).unwrap(),
                        product.b_symbol(&pa, &pb, &pc).unwrap(),
                    );
                    let cg = g.cgc(a, b, &c).unwrap();
                    let pg = product.cgc(&pa, &pb, &pc).unwrap();
                    assert_eq!(cg.dims(), pg.dims());
                    let [da, db, dc, n] = cg.dims();
                    for i in 0..da {
                        for j in 0..db {
                            for k in 0..dc {
                                for m in 0..n {
                                    assert_eq!(
                                        cg.at([i, j, k, m]).to_bits(),
                                        pg.at([i, j, k, m]).to_bits()
                                    );
                                }
                            }
                        }
                    }
                    for d in g.fusion(&c, a).unwrap() {
                        for f in g.fusion(b, a).unwrap() {
                            same_block(
                                g.f_symbol(a, b, a, &d, &c, &f).unwrap(),
                                product
                                    .f_symbol(&pa, &pb, &pa, &split(&d), &pc, &split(&f))
                                    .unwrap(),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn tori_quotients_and_trivial_group() {
    let t = group(GroupId::torus(2).unwrap());
    assert_eq!(t.fusion_style(), FusionStyle::Unique);
    let prefix = [
        vec![0, 0],
        vec![0, 1],
        vec![1, 0],
        vec![0, -1],
        vec![1, 1],
        vec![-1, 0],
    ];
    let product = Product::new(vec![group(GroupId::u(1).unwrap()); 2]);
    for i in 0..40 {
        let x = t.object_at(i).unwrap();
        let qs = x.charges().collect::<Vec<_>>();
        if (i as usize) < prefix.len() {
            assert_eq!(qs, prefix[i as usize]);
        }
        assert_eq!(
            qs,
            product
                .object_at(i)
                .unwrap()
                .iter()
                .flat_map(LieIrrep::charges)
                .collect::<Vec<_>>()
        );
        assert_eq!(t.object_index(&x), Ok(i));
    }
    let even = group(GroupId::new(&[], 1, &[vec![2]]).unwrap());
    assert_eq!(even.irrep(&[], &[1]), Err(LieError::NotAdmissible));
    let q = charge(&even, 2);
    assert_eq!(even.dual(&q), charge(&even, -2));
    assert_eq!(even.fusion(&q, &q).unwrap().next(), Some(charge(&even, 4)));
    assert_eq!(even.object_count(), Err(EnumerationError::NotEnumerable));
    assert_eq!(even.object_at(0), Err(EnumerationError::NotEnumerable));
    assert_eq!(even.object_index(&q), Err(EnumerationError::NotEnumerable));
    let g = group(GroupId::new(&[], 0, &[]).unwrap());
    let one = g.unit();
    assert!(one.dynkin().is_empty() && one.charges().len() == 0);
    assert_eq!(g.object_count(), Ok(Some(1)));
    assert_eq!(g.object_at(0), Ok(one.clone()));
    assert_eq!(g.object_at(1), Err(EnumerationError::OutOfRange));
    assert_eq!(g.object_index(&one), Ok(0));
    assert_eq!(g.dual(&one), one);
    assert_eq!(
        g.fusion(&one, &one).unwrap().collect::<Vec<_>>(),
        std::slice::from_ref(&one)
    );
    assert_eq!(g.nsymbol(&one, &one, &one), 1);
    assert_eq!(g.dim(&one), 1.0);
    assert_eq!(one.dim_exact(), 1.into());
    assert_eq!(
        g.f_symbol(&one, &one, &one, &one, &one, &one)
            .unwrap()
            .data(),
        [1.0]
    );
    for block in [
        g.r_symbol(&one, &one, &one),
        g.a_symbol(&one, &one, &one),
        g.b_symbol(&one, &one, &one),
    ] {
        assert_eq!(block.unwrap().data(), [1.0]);
    }
    assert_eq!(g.cgc(&one, &one, &one).unwrap().at([0; 4]), 1.0);
}

#[cfg(feature = "lie-generated")]
#[test]
fn u3_fundamental_antifundamental_and_leaf_symbols() {
    let g = group(GroupId::u(3).unwrap());
    let su = group(GroupId::su(3).unwrap());
    let a = g.irrep(&[1, 0], &[1]).unwrap();
    let b = g.irrep(&[0, 1], &[-1]).unwrap();
    assert_eq!(g.dual(&a), b);
    let octet = g.irrep(&[1, 1], &[0]).unwrap();
    let channels = g.fusion(&a, &b).unwrap().collect::<Vec<_>>();
    assert_eq!(channels.len(), 2);
    assert!(channels.contains(&g.unit()) && channels.contains(&octet));
    assert_eq!(g.dim(&a) * g.dim(&b), 9.0);
    assert_eq!((g.dim(&octet), g.dim(&g.unit())), (8.0, 1.0));
    let sa = su.irrep(&[1, 0], &[]).unwrap();
    let sb = su.dual(&sa);
    for e in &channels {
        for f in &channels {
            same_block(
                g.f_symbol(&a, &b, &a, &a, e, f).unwrap(),
                su.f_symbol(
                    &sa,
                    &sb,
                    &sa,
                    &sa,
                    &su.irrep(&e.dynkin(), &[]).unwrap(),
                    &su.irrep(&f.dynkin(), &[]).unwrap(),
                )
                .unwrap(),
            );
            same_block(
                g.r_symbol(&a, &b, e).unwrap(),
                su.r_symbol(&sa, &sb, &su.irrep(&e.dynkin(), &[]).unwrap())
                    .unwrap(),
            );
        }
    }
}

#[cfg(not(feature = "lie-generated"))]
#[test]
fn u3_requires_generated() {
    assert_eq!(
        LieGroup::new(GroupId::u(3).unwrap()),
        Err(LieError::Unsupported {
            requires: "lie-generated"
        })
    );
}
