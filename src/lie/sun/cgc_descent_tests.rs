//! Structural regression against the pre-indexing descent, not a mathematical oracle.
use super::*;

fn old_lower_weight_cgc(
    data: &mut HashMap<(u32, u32, u32, u32), f64>,
    ctx: &Ctx,
    n123: usize,
) -> Result<(), SunError> {
    let n = ctx.n;
    // Reverse-lexicographic weight order: parents (higher weights) come first
    // and are already solved. Skip the first (the highest weight, done above).
    let mut w3list: Vec<Vec<i64>> = ctx.three.map.keys().cloned().collect();
    w3list.sort();
    w3list.reverse();
    let mut w1list: Vec<_> = ctx.one.map.keys().collect();
    w1list.sort_unstable();

    for alpha in 0..n123 {
        for w3 in w3list.iter().skip(1) {
            let m3list = &ctx.three.map[w3];
            let jmax = m3list.len();

            // Parent rows: one block per level l for the raised weight w3'.
            // imax = Σ_l |map3[w3' (l)]|.
            let mut parent_blocks: Vec<(usize, Vec<usize>)> = Vec::new();
            let mut imax = 0usize;
            for l in 0..n - 1 {
                let w3p = raise(w3, l)?;
                let list = get_indices(&ctx.three.map, &w3p).to_vec();
                imax = crate::lie::checked_count(imax, list.len())
                    .map_err(|_| SunError::LabelOverflow)?;
                parent_blocks.push((l, list));
            }

            let mut eqs = Mat::try_zeros(imax, jmax)?;
            // RHS over the unique (m1, m2) columns encountered.
            let mut mask_index: HashMap<(usize, usize), usize> = HashMap::new();
            let mut rhs_acc: Vec<(usize, (usize, usize), f64)> = Vec::new();

            let mut i = 0usize;
            for (l, m3plist) in &parent_blocks {
                let l = *l;
                let w3p = raise(w3, l)?;
                for &m3p in m3plist {
                    // eqs[i, j] = Jm3[m3, m3']  (annihilation3[l], element (m3, m3')).
                    for (j, &m3) in m3list.iter().enumerate() {
                        if let Some(&v) = ctx.three.annih[l].elem.get(&(m3, m3p)) {
                            eqs.add(i, j, v);
                        }
                    }
                    MATCH_BUILDS.with(|counts| {
                        *counts.borrow_mut().entry((w3.clone(), l)).or_default() += 1;
                    });
                    // RHS: (Jm1 ⊗ I + I ⊗ Jm2) acting on parent CGC[m1',m2',m3',α].
                    for &w1p in &w1list {
                        let m1plist = &ctx.one.map[w1p];
                        let w2p = matching_weight(&w3p, w1p, ctx.wshift)?;
                        let m2plist = get_indices(&ctx.two.map, &w2p);
                        if m2plist.is_empty() {
                            continue;
                        }
                        for &m2p in m2plist {
                            for &m1p in m1plist {
                                let cgc_coeff = *data
                                    .get(&(
                                        magnetic_index(m1p)?,
                                        magnetic_index(m2p)?,
                                        magnetic_index(m3p)?,
                                        magnetic_index(alpha)?,
                                    ))
                                    .unwrap_or(&0.0);
                                if cgc_coeff == 0.0 {
                                    continue;
                                }
                                // Apply Jm1: lower m1' at level l -> m1.
                                let w1 = lower(w1p, l)?;
                                for &m1 in get_indices(&ctx.one.map, &w1) {
                                    if let Some(&jm1) = ctx.one.annih[l].elem.get(&(m1, m1p)) {
                                        push_rhs(
                                            &mut mask_index,
                                            &mut rhs_acc,
                                            i,
                                            (m1, m2p),
                                            jm1 * cgc_coeff,
                                        );
                                    }
                                }
                                // Apply Jm2: lower m2' at level l -> m2.
                                let w2 = lower(&w2p, l)?;
                                for &m2 in get_indices(&ctx.two.map, &w2) {
                                    if let Some(&jm2) = ctx.two.annih[l].elem.get(&(m2, m2p)) {
                                        push_rhs(
                                            &mut mask_index,
                                            &mut rhs_acc,
                                            i,
                                            (m1p, m2),
                                            jm2 * cgc_coeff,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    i += 1;
                }
            }

            let nmask = mask_index.len();
            if nmask == 0 {
                continue;
            }
            // Ordered mask (column index -> (m1,m2)); mask_index gives the order.
            let mut mask: Vec<(usize, usize)> = vec![(0, 0); nmask];
            for (&pair, &c) in &mask_index {
                mask[c] = pair;
            }
            let mut rhs = Mat::try_zeros(imax, nmask)?;
            for (row, pair, val) in rhs_acc {
                rhs.add(row, mask_index[&pair], val);
            }

            // sols = eqs \ rhs  (least squares), shape jmax x nmask.
            record_solve(&eqs, &rhs, &mask);
            let sols = linalg::lstsq(&eqs, &rhs)?;
            for (c, &(m1, m2)) in mask.iter().enumerate() {
                for (j, &m3) in m3list.iter().enumerate() {
                    let v = sols.at(j, c);
                    if v != 0.0 {
                        *data
                            .entry((
                                magnetic_index(m1)?,
                                magnetic_index(m2)?,
                                magnetic_index(m3)?,
                                magnetic_index(alpha)?,
                            ))
                            .or_insert(0.0) += v;
                    }
                }
            }
        }
    }
    Ok(())
}

fn old_ops(mats: Vec<Vec<LadderEntry>>) -> Vec<LadderOp> {
    mats.into_iter()
        .map(|mat| {
            let mut op = LadderOp::default();
            for e in mat {
                let v = e.value.to_f64();
                op.by_col.entry(e.col).or_default().push((e.row, v));
                op.elem.insert((e.row, e.col), v);
            }
            op
        })
        .collect()
}

// Ctx::new before context-local sharing: three basis/maps and five ladders.
fn old_context(s1: &Irrep, s2: &Irrep, s3: &Irrep) -> Ctx {
    let n = s1.rank();
    let [one, two, three] = [s1, s2, s3].map(|s| {
        let pats = s.patterns();
        IrrepData {
            map: weight_map(&pats, n).unwrap(),
            pats,
            creation: Vec::new(),
            annih: Vec::new(),
        }
    });
    let mut one = one;
    let mut two = two;
    let mut three = three;
    one.creation = old_ops(s1.try_creation().unwrap());
    two.creation = old_ops(s2.try_creation().unwrap());
    one.annih = old_ops(s1.try_annihilation().unwrap());
    two.annih = old_ops(s2.try_annihilation().unwrap());
    three.annih = old_ops(s3.try_annihilation().unwrap());
    Ctx {
        n,
        d3: three.pats.len(),
        one: std::sync::Arc::new(one),
        two: std::sync::Arc::new(two),
        three: std::sync::Arc::new(three),
        w3_top: s3.weight().to_vec(),
        wshift: coupling_shift(s1.weight(), s2.weight(), s3.weight(), n).unwrap(),
    }
}

#[derive(Debug, PartialEq)]
struct SolveTrace {
    shape: [usize; 3],
    mask: Vec<(usize, usize)>,
    eqs: Vec<u64>,
    rhs: Vec<u64>,
}

thread_local! {
    static SOLVES: std::cell::RefCell<Option<Vec<SolveTrace>>> = const { std::cell::RefCell::new(None) };
}

pub(super) fn record_solve(eqs: &Mat, rhs: &Mat, mask: &[(usize, usize)]) {
    SOLVES.with(|solves| {
        if let Some(solves) = &mut *solves.borrow_mut() {
            solves.push(SolveTrace {
                shape: [eqs.rows, eqs.cols, rhs.cols],
                mask: mask.to_vec(),
                eqs: eqs.data.iter().map(|v| v.to_bits()).collect(),
                rhs: rhs.data.iter().map(|v| v.to_bits()).collect(),
            });
        }
    });
}

#[test]
fn indexed_descent_preserves_bits_and_builds_structure_once() {
    // Drift detector against the original ordered scalar descent. Independent
    // mathematical oracles remain in tests/sun_cgc_gauge.rs and gauge_golden.rs.
    let cases: [(&[i64], &[i64], &[i64]); 7] = [
        (&[3, 3], &[3, 3], &[3, 3]),
        (&[2, 0, 2], &[2, 0, 2], &[2, 0, 2]),
        (&[1, 1], &[1, 1], &[1, 1]), // 8 x 8 -> 8, multiplicity two
        (&[1, 0], &[0, 1], &[1, 1]), // three distinct irreps
        (&[1, 0], &[1, 1], &[1, 0]), // output equals first factor
        (&[1, 1], &[1, 0], &[1, 0]), // output equals second factor
        (&[1, 0], &[1, 0], &[1, 1]), // absent channel, no descent
    ];
    for (a, b, c) in cases {
        let [a, b, c] = [a, b, c].map(|d| Irrep::from_dynkin(d).unwrap());
        let distinct = [&a, &b, &c].into_iter().collect::<BTreeSet<_>>().len();
        LADDER_BUILDS.set(0);
        let ctx = Ctx::new(&a, &b, &c).unwrap();
        assert_eq!(LADDER_BUILDS.get(), distinct);
        LADDER_BUILDS.set(0);
        let old_ctx = old_context(&a, &b, &c);
        assert_eq!(LADDER_BUILDS.get(), 5);
        for (new, old) in [
            (&ctx.one, &old_ctx.one),
            (&ctx.two, &old_ctx.two),
            (&ctx.three, &old_ctx.three),
        ] {
            assert_eq!(new.pats, old.pats);
            assert_eq!(new.map, old.map);
            for (up, down) in new.creation.iter().zip(&new.annih) {
                for (&(row, col), v) in &up.elem {
                    assert_eq!(v.to_bits(), down.elem[&(col, row)].to_bits());
                }
                for col in down.by_col.values() {
                    assert!(col.windows(2).all(|w| w[0].0 < w[1].0));
                }
            }
            for (new_op, old_op) in new
                .creation
                .iter()
                .zip(&old.creation)
                .chain(new.annih.iter().zip(&old.annih))
            {
                assert_eq!(new_op.elem.len(), old_op.elem.len());
                for (key, v) in &new_op.elem {
                    assert_eq!(v.to_bits(), old_op.elem[key].to_bits());
                }
                assert_eq!(new_op.by_col, old_op.by_col);
            }
        }
        let expected = super::super::fr::mult(&a, &b, &c).unwrap();
        let mut data = HashMap::new();
        let mut old = HashMap::new();
        let n = highest_weight_cgc(&mut data, &ctx, expected).unwrap();
        assert_eq!(n, highest_weight_cgc(&mut old, &old_ctx, expected).unwrap());
        MATCH_BUILDS.with(|counts| counts.borrow_mut().clear());
        SOLVES.set(Some(Vec::new()));
        linalg::FACTORIZATIONS.set(0);
        lower_weight_cgc(&mut data, &ctx, n).unwrap();
        let factorizations = linalg::FACTORIZATIONS.get();
        let builds = MATCH_BUILDS.take();
        let solves = SOLVES.take().unwrap();
        SOLVES.set(Some(Vec::new()));
        old_lower_weight_cgc(&mut old, &old_ctx, n).unwrap();
        let old_builds = MATCH_BUILDS.take();
        let old_solves = SOLVES.take().unwrap();
        assert_eq!(
            builds.keys().collect::<Vec<_>>(),
            old_builds.keys().collect::<Vec<_>>()
        );
        for ((weight, l), &count) in &builds {
            assert_eq!(count, 1, "matching must be shared across parents and alpha");
            assert_eq!(
                old_builds[&(weight.clone(), *l)],
                n * get_indices(&ctx.three.map, &raise(weight, *l).unwrap()).len()
            );
        }
        for (key, value) in &data {
            assert_eq!(
                value.to_bits(),
                old.get(key).unwrap_or(&0.0).to_bits(),
                "coefficient {key:?}"
            );
        }
        assert_eq!(data.len(), old.len());
        assert_eq!(solves, old_solves, "solve shapes, masks and inputs changed");
        if n > 0 {
            assert_eq!(factorizations, ctx.three.map.len() - 1);
        } else {
            assert_eq!(factorizations, 0);
        }
        // Both paths still pass the same generation gates after purging.
        for mut entries in [data, old] {
            purge(&mut entries);
            let cgc = freeze(
                a.clone(),
                b.clone(),
                c.clone(),
                [ctx.one.pats.len(), ctx.two.pats.len(), ctx.d3, n],
                entries,
            );
            check_orthonormal(&cgc, &ctx).unwrap();
            check_ladder(&cgc, &ctx).unwrap();
        }
    }
}

#[test]
fn descent_factors_only_plans_with_rhs() {
    let a = Irrep::from_dynkin(&[1, 1]).unwrap();
    let ctx = Ctx::new(&a, &a, &a).unwrap();
    let mut data = HashMap::new();
    assert_eq!(highest_weight_cgc(&mut data, &ctx, 2).unwrap(), 2);
    SOLVES.set(Some(Vec::new()));
    linalg::FACTORIZATIONS.set(0);
    lower_weight_cgc(&mut data, &ctx, 2).unwrap();
    let solves = SOLVES.take().unwrap();
    // The octet has six root weights and zero: seven distinct weights. Skip
    // the highest, leaving six plans, each with an RHS for both alpha columns.
    assert_eq!(ctx.three.map.len(), 7);
    assert_eq!(solves.len(), 12);
    assert_eq!(linalg::FACTORIZATIONS.get(), 6);

    // Make every plan singular, but give none an RHS: factoring eagerly would
    // report a singular R even though the old descent performed no solves.
    let [a, b, c] = [&[1, 0][..], &[0, 1], &[1, 1]].map(|d| Irrep::from_dynkin(d).unwrap());
    let mut ctx = Ctx::new(&a, &b, &c).unwrap();
    let mut data = HashMap::new();
    highest_weight_cgc(&mut data, &ctx, 1).unwrap();
    for op in &mut std::sync::Arc::get_mut(&mut ctx.three).unwrap().annih {
        op.by_col.clear();
    }
    linalg::FACTORIZATIONS.set(0);
    lower_weight_cgc(&mut HashMap::new(), &ctx, 1).unwrap();
    assert_eq!(linalg::FACTORIZATIONS.get(), 0);
    assert_eq!(
        lower_weight_cgc(&mut data, &ctx, 1).err(),
        Some(SunError::Linalg("lstsq: singular R diagonal".into()))
    );
    assert_eq!(linalg::FACTORIZATIONS.get(), 1);
}

#[test]
fn coefficient_multiplicity_checks_do_not_build_channel_dimensions() {
    use crate::lie::general::roots::{CartanData, EVALUATIONS};
    let _guard = crate::cache::TEST_CACHE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let roots = CartanData::new(crate::lie::RootSystem::A(2));
    let a = Irrep(roots.irrep(&[1, 0]).unwrap());
    let b = Irrep(a.0.dual());
    let one = Irrep(roots.irrep(&[0, 0]).unwrap());
    crate::cache::reset();
    EVALUATIONS.with(|n| n.set((0, 0)));
    assert_eq!(super::super::fr::mult(&a, &b, &one).unwrap(), 1);
    assert_eq!(super::super::fr::mult(&b, &a, &one).unwrap(), 1);
    EVALUATIONS.with(|n| assert_eq!(n.get(), (0, 0)));
    crate::cache::reset();
    generate(&a, &b, &one, None).unwrap();
    EVALUATIONS.with(|n| assert_eq!(n.get(), (0, 0)));
}
