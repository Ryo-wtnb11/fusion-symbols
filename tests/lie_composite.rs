//! Composite groups restrict cover objects and reuse the ordered Product gauge.
use fusion_symbols::{
    category::{
        CgcBlock, EnumerationError, FusionCategory, FusionStyle, GroupRepresentations, SymbolBlock,
    },
    lie::{CatalogStats, GroupId, LieError, LieGroup, LieIrrep, RootSystem::*},
    product::Product,
};
use std::collections::BTreeMap;

fn label(cat: &LieGroup, dynkin: &[i64]) -> LieIrrep {
    cat.irrep(dynkin).unwrap()
}

fn same_block<const R: usize>(
    a: impl SymbolBlock<R, Scalar = f64>,
    b: impl SymbolBlock<R, Scalar = f64>,
) {
    assert_eq!(a.dims(), b.dims());
    assert_eq!(
        a.as_slice().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b.as_slice().iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn d2_forms_admission_and_hand_values() {
    let forms = [
        (GroupId::spin(4).unwrap(), 0),
        (GroupId::so(4).unwrap(), 1),
        (GroupId::pso(4).unwrap(), 2),
        (
            GroupId::new(&[D(2)], 0, &[vec![2, 0], vec![0, 1]]).unwrap(),
            3,
        ),
        (
            GroupId::new(&[D(2)], 0, &[vec![1, 0], vec![0, 2]]).unwrap(),
            4,
        ),
    ];
    for (id, form) in forms {
        let cat = LieGroup::new(id.clone()).unwrap();
        assert_eq!(cat.fusion_style(), FusionStyle::Simple);
        assert_eq!(
            cat.object_count(),
            if form == 0 {
                Ok(None)
            } else {
                Err(EnumerationError::NotEnumerable)
            }
        );
        assert_eq!(cat.irrep(&[0]), Err(LieError::InvalidLabel));
        assert_eq!(
            cat.irrep_with_charges(&[0, 0], &[0]),
            Err(LieError::InvalidLabel)
        );
        assert_eq!(cat.irrep(&[-1, 0]), Err(LieError::InvalidLabel));
        for a in 0..=3 {
            for b in 0..=3 {
                let admitted = match form {
                    0 => true,
                    1 => (a + b) % 2 == 0,
                    2 => a % 2 == 0 && b % 2 == 0,
                    3 => a % 2 == 0,
                    _ => b % 2 == 0,
                };
                let result = cat.irrep(&[a, b]);
                if !admitted {
                    assert_eq!(result, Err(LieError::NotAdmissible));
                    continue;
                }
                let x = result.unwrap();
                assert_eq!(x.dynkin(), [a, b]);
                assert!(x.dynkin_labels().eq([a, b]));
                assert_eq!(x.charges().len(), 0);
                assert_eq!(x.dim_exact(), ((a + 1) * (b + 1)).into());
                assert!(x.gt_patterns().is_none());
                // Independent A1 weights a,a-2,...,-a in each coordinate.
                let weights: BTreeMap<_, _> = (0..=a)
                    .flat_map(|i| (0..=b).map(move |j| (vec![a - 2 * i, b - 2 * j], 1)))
                    .collect();
                assert_eq!(x.weight_multiplicities(), weights);
                for y in cat.fusion(&x, &x).unwrap() {
                    assert!(id.admits(&y.dynkin(), &[]));
                    assert!(cat.contains(&y));
                }
            }
        }
    }
    let cat = LieGroup::new(GroupId::so(4).unwrap()).unwrap();
    let vector = label(&cat, &[1, 1]);
    assert_eq!(
        cat.fusion(&vector, &vector)
            .unwrap()
            .map(|x| x.dynkin())
            .collect::<Vec<_>>(),
        [vec![0, 0], vec![2, 0], vec![0, 2], vec![2, 2]]
    );
    // SU2 R^{1/2,1/2}_{1}=+1, R^{1/2,1/2}_{0}=-1.
    assert_eq!(
        cat.r_symbol(&vector, &vector, &label(&cat, &[2, 0]))
            .unwrap()
            .get([0, 0]),
        -1.0
    );
    assert_eq!(
        cat.r_symbol(&vector, &vector, &cat.unit())
            .unwrap()
            .get([0, 0]),
        1.0
    );
    // SU2 F^{1/2,1/2,1/2}_{1/2}[0,0] = -1/2 in each factor.
    assert_eq!(
        cat.f_symbol(&vector, &vector, &vector, &vector, &cat.unit(), &cat.unit())
            .unwrap()
            .get([0; 4]),
        (-0.5) * (-0.5)
    );
    let cover = LieGroup::new(GroupId::spin(4).unwrap()).unwrap();
    assert_eq!(cat.authority_fingerprint(), cover.authority_fingerprint());
    let cv = label(&cover, &[1, 1]);
    for x in cat.fusion(&vector, &vector).unwrap() {
        let cx = label(&cover, &x.dynkin());
        same_block(
            cat.r_symbol(&vector, &vector, &x).unwrap(),
            cover.r_symbol(&cv, &cv, &cx).unwrap(),
        );
        same_block(
            cat.a_symbol(&vector, &vector, &x).unwrap(),
            cover.a_symbol(&cv, &cv, &cx).unwrap(),
        );
        same_block(
            cat.b_symbol(&vector, &vector, &x).unwrap(),
            cover.b_symbol(&cv, &cv, &cx).unwrap(),
        );
        same_block(
            cat.f_symbol(&vector, &vector, &vector, &vector, &x, &cat.unit())
                .unwrap(),
            cover
                .f_symbol(&cv, &cv, &cv, &cv, &cx, &cover.unit())
                .unwrap(),
        );
    }
    assert!(!cat.contains(&cover.unit()));
    assert!(std::panic::catch_unwind(|| cat.try_nsymbol(&vector, &vector, &cover.unit())).is_err());
}

// Product is the composition reference; hand values above independently pin the gauge.
fn compare_product(id: GroupId, roots: &[fusion_symbols::lie::RootSystem], samples: &[&[i64]]) {
    let cat = LieGroup::new(id).unwrap();
    let leaves: Vec<_> = roots
        .iter()
        .map(|r| LieGroup::new(GroupId::cover(&[*r], 0).unwrap()).unwrap())
        .collect();
    let product = Product::new(leaves);
    let labels: Vec<_> = samples.iter().map(|d| label(&cat, d)).collect();
    let parts: Vec<Vec<_>> = samples
        .iter()
        .map(|d| {
            let mut offset = 0;
            product
                .factors()
                .iter()
                .map(|leaf| {
                    let end = offset + leaf.group_id().semisimple_rank();
                    let x = label(leaf, &d[offset..end]);
                    offset = end;
                    x
                })
                .collect()
        })
        .collect();
    assert_eq!(cat.authority_fingerprint(), product.authority_fingerprint());
    assert_eq!(cat.fusion_style(), product.fusion_style());
    assert_eq!(cat.braiding_style(), product.braiding_style());
    assert_eq!(cat.has_canonical_unit(), product.has_canonical_unit());
    assert_eq!(
        cat.has_trivial_associator_gauge(),
        product.has_trivial_associator_gauge()
    );
    for (a, pa) in labels.iter().zip(&parts) {
        for (got, want) in [
            (cat.dim(a), product.dim(pa)),
            (cat.sqrt_dim(a), product.sqrt_dim(pa)),
            (cat.inv_dim(a), product.inv_dim(pa)),
            (cat.inv_sqrt_dim(a), product.inv_sqrt_dim(pa)),
            (cat.twist(a), product.twist(pa)),
            (
                cat.frobenius_schur_phase(a),
                product.frobenius_schur_phase(pa),
            ),
            (
                cat.frobenius_schur_indicator(a),
                product.frobenius_schur_indicator(pa),
            ),
        ] {
            assert_eq!(got.to_bits(), want.to_bits());
        }
        assert_eq!(
            cat.dual(a).dynkin(),
            product
                .dual(pa)
                .iter()
                .flat_map(LieIrrep::dynkin)
                .collect::<Vec<_>>()
        );
        for (b, pb) in labels.iter().zip(&parts) {
            let fusion: Vec<_> = cat.fusion(a, b).unwrap().collect();
            assert!(fusion
                .iter()
                .all(|x| cat.group_id().admits(&x.dynkin(), &[])));
            assert_eq!(
                fusion.iter().map(LieIrrep::dynkin).collect::<Vec<_>>(),
                product
                    .fusion(pa, pb)
                    .unwrap()
                    .map(|xs| xs.iter().flat_map(LieIrrep::dynkin).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                cat.fusion(a, b).unwrap().rev().collect::<Vec<_>>(),
                fusion.into_iter().rev().collect::<Vec<_>>()
            );
            for (c, pc) in labels.iter().zip(&parts) {
                assert_eq!(
                    cat.try_nsymbol(a, b, c).unwrap(),
                    product.try_nsymbol(pa, pb, pc).unwrap()
                );
                same_block(
                    cat.r_symbol(a, b, c).unwrap(),
                    product.r_symbol(pa, pb, pc).unwrap(),
                );
                same_block(
                    cat.a_symbol(a, b, c).unwrap(),
                    product.a_symbol(pa, pb, pc).unwrap(),
                );
                same_block(
                    cat.b_symbol(a, b, c).unwrap(),
                    product.b_symbol(pa, pb, pc).unwrap(),
                );
                let (cg, pg) = match (cat.cgc(a, b, c), product.cgc(pa, pb, pc)) {
                    (Ok(cg), Ok(pg)) => (cg, pg),
                    // Preserve the simple engine's loud rejection of a mismatched
                    // unit coupling, even when the combined channel is absent.
                    (Err(cg), Err(pg)) => {
                        assert_eq!(cg.to_string(), pg.to_string());
                        continue;
                    }
                    result => panic!("different CGC results: {result:?}"),
                };
                assert_eq!(cg.dims(), pg.dims());
                let [da, db, dc, n] = cg.dims();
                for i in 0..da {
                    for j in 0..db {
                        for k in 0..dc {
                            for mu in 0..n {
                                assert_eq!(
                                    cg.at([i, j, k, mu]).to_bits(),
                                    pg.at([i, j, k, mu]).to_bits()
                                );
                            }
                        }
                    }
                }
            }
            // All admissible recouplings for a,b,a use every intermediate channel.
            for e in cat.fusion(a, b).unwrap() {
                for d in cat.fusion(&e, a).unwrap() {
                    for f in cat.fusion(b, a).unwrap() {
                        let split = |x: &LieIrrep| {
                            let ds = x.dynkin();
                            let mut offset = 0;
                            product
                                .factors()
                                .iter()
                                .map(|leaf| {
                                    let end = offset + leaf.group_id().semisimple_rank();
                                    let x = label(leaf, &ds[offset..end]);
                                    offset = end;
                                    x
                                })
                                .collect::<Vec<_>>()
                        };
                        same_block(
                            cat.f_symbol(a, b, a, &d, &e, &f).unwrap(),
                            product
                                .f_symbol(pa, pb, pa, &split(&d), &split(&e), &split(&f))
                                .unwrap(),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn d2_equals_product_of_covers() {
    // Unequal factor labels make a reversed factor map observably wrong.
    compare_product(
        GroupId::spin(4).unwrap(),
        &[A(1), A(1)],
        &[&[0, 0], &[1, 0], &[0, 2]],
    );
}

#[test]
fn quotient_only_restricts_objects_and_enumeration() {
    let roots = [A(1), A(1)];
    let diagonal = GroupId::new(&roots, 0, &[vec![1, 1], vec![0, 2]]).unwrap();
    compare_product(diagonal.clone(), &roots, &[&[0, 0], &[1, 1], &[2, 0]]);
    compare_product(
        GroupId::so(4).unwrap(),
        &roots,
        &[&[0, 0], &[1, 1], &[2, 0]],
    );
    let cat = LieGroup::new(diagonal).unwrap();
    assert_eq!(cat.object_count(), Err(EnumerationError::NotEnumerable));
    assert_eq!(cat.object_at(0), Err(EnumerationError::NotEnumerable));
    assert_eq!(
        cat.object_index(&cat.unit()),
        Err(EnumerationError::NotEnumerable)
    );
    let quotient = LieGroup::new(GroupId::so(4).unwrap()).unwrap();
    assert_eq!(
        quotient.object_count(),
        Err(EnumerationError::NotEnumerable)
    );
    let cover = LieGroup::new(GroupId::spin(4).unwrap()).unwrap();
    let spin = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let product = Product::new(vec![spin.clone(), spin]);
    assert_eq!(cover.object_count(), Ok(None));
    for i in 0..32 {
        let x = cover.object_at(i).unwrap();
        assert_eq!(
            x.dynkin(),
            product
                .object_at(i)
                .unwrap()
                .iter()
                .flat_map(LieIrrep::dynkin)
                .collect::<Vec<_>>()
        );
        assert_eq!(cover.object_index(&x), Ok(i));
    }
    assert_eq!(cover.catalog_stats().unwrap(), CatalogStats::default());
    assert!(matches!(
        LieGroup::with_catalog_budget(GroupId::spin(4).unwrap(), 0),
        Err(LieError::Unsupported {
            requires: "B/C/D catalog"
        })
    ));
}

#[cfg(feature = "lie-generated")]
#[test]
fn a1_a2_and_diagonal_su3_quotient() {
    compare_product(
        GroupId::cover(&[A(1), A(2)], 0).unwrap(),
        &[A(1), A(2)],
        &[&[0, 0, 0], &[1, 1, 0], &[0, 0, 1]],
    );
    // SU3 central class a+2b; the diagonal Z3 requires a+2b+c+2d=0 mod 3.
    let roots = [A(2), A(2)];
    let id = GroupId::new(
        &roots,
        0,
        &[
            vec![1, 0, 0, 1],
            vec![0, 1, 0, 2],
            vec![0, 0, 1, 1],
            vec![0, 0, 0, 3],
        ],
    )
    .unwrap();
    let cat = LieGroup::new(id.clone()).unwrap();
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                for d in 0..3 {
                    assert_eq!(
                        cat.irrep(&[a, b, c, d]).is_ok(),
                        (a + 2 * b + c + 2 * d) % 3 == 0
                    );
                }
            }
        }
    }
    assert_eq!(cat.object_count(), Err(EnumerationError::NotEnumerable));
    compare_product(id, &roots, &[&[0, 0, 0, 0], &[1, 0, 0, 1]]);
}

#[cfg(not(feature = "lie-generated"))]
#[test]
fn composite_feature_boundary() {
    assert_eq!(
        LieGroup::new(GroupId::cover(&[D(2), A(2)], 0).unwrap()),
        Err(LieError::Unsupported {
            requires: "lie-generated"
        })
    );
}

#[cfg(feature = "lie-generated")]
#[test]
fn aggregate_catalog_budget() {
    let b = LieGroup::new(GroupId::cover(&[B(2)], 0).unwrap()).unwrap();
    let c = LieGroup::new(GroupId::cover(&[C(2)], 0).unwrap()).unwrap();
    let (b_seed, c_seed) = (b.catalog_stats().unwrap(), c.catalog_stats().unwrap());
    let seed_sum = b_seed.bytes + c_seed.bytes;
    let byte_budget = |s: CatalogStats| (s.bytes, s.budget);
    let id = GroupId::cover(&[B(2), C(2)], 0).unwrap();
    assert!(matches!(
        LieGroup::with_catalog_budget(id.clone(), seed_sum - 1),
        Err(LieError::Generation(_))
    ));
    let cat = LieGroup::with_catalog_budget(id, seed_sum + 5).unwrap();
    let stats = cat.catalog_stats().unwrap();
    assert_eq!(byte_budget(stats), (seed_sum, seed_sum + 5));
    assert_eq!(stats.entries, b_seed.entries + c_seed.entries);
    cat.reset_catalog().unwrap();
    assert_eq!(cat.catalog_stats().unwrap(), stats);
    assert_eq!(cat.object_count(), Err(EnumerationError::NotEnumerable));
}

#[cfg(feature = "lie-generated")]
#[test]
fn checked_composite_n_overflow_and_zero_wins() {
    // Each SU3 adjoint product contains the adjoint twice: 2^32 does not fit u32.
    let cat = LieGroup::new(GroupId::cover(&[A(2); 32], 0).unwrap()).unwrap();
    let a = label(&cat, &[1; 64]);
    assert_eq!(cat.try_nsymbol(&a, &a, &a), Err(LieError::LabelOverflow));
    let mut absent = [1; 64];
    absent[62..].copy_from_slice(&[1, 0]); // triality excludes the fundamental from 8 x 8
    let absent = label(&cat, &absent);
    assert_eq!(cat.try_nsymbol(&a, &a, &absent), Ok(0));
}

/// The feature boundary precedes any B/C/D catalog seed charge.
#[cfg(not(feature = "lie-generated"))]
#[test]
fn unsupported_composite_precedence() {
    for (roots, torus, requires) in [
        (vec![B(1 << 20), E6], 0, "lie-generated"),
        (vec![E6, A(1)], 1, "lie-generated"),
    ] {
        let group = GroupId::cover(&roots, torus).unwrap();
        assert_eq!(
            LieGroup::new(group.clone()),
            Err(LieError::Unsupported { requires })
        );
        assert_eq!(
            LieGroup::with_catalog_budget(group, 0),
            Err(LieError::Unsupported { requires })
        );
    }
}
