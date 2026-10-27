//! Generated labels share their weight storage: a warm fusion yields owned
//! channel labels without allocating.
#![cfg(feature = "lie-generated")]
use crate::common::counting_alloc;
use fusion_symbols::{
    category::FusionCategory,
    lie::{GroupId, LieGroup},
};

// Warm means:
// the labels are constructed and one identical `fusion(a, b)` has been fully
// consumed, so the decomposition is in the product cache.
#[test]
fn warm_generated_fusion_yields_labels_without_allocation() {
    let _serial = crate::common::counting_alloc::serial();
    for (id, a, b, expected) in [
        // SU(3): 8 ⊗ 8 = 1 ⊕ 8 ⊕ 8 ⊕ 10 ⊕ 10̄ ⊕ 27 (distinct outcomes).
        (
            GroupId::su(3).unwrap(),
            vec![1, 1],
            vec![1, 1],
            vec![vec![0, 0], vec![0, 3], vec![1, 1], vec![2, 2], vec![3, 0]],
        ),
        // Sp(4): 4 ⊗ 4 = 1 ⊕ 5 ⊕ 10.
        (
            GroupId::sp(4).unwrap(),
            vec![1, 0],
            vec![1, 0],
            vec![vec![0, 0], vec![0, 1], vec![2, 0]],
        ),
        // Spin(5): 4 ⊗ 5 = 4 ⊕ 16 (spinor ⊗ vector).
        (
            GroupId::spin(5).unwrap(),
            vec![0, 1],
            vec![1, 0],
            vec![vec![0, 1], vec![1, 1]],
        ),
    ] {
        let cat = LieGroup::new(id.clone()).unwrap();
        let a = cat.irrep(&a, &[]).unwrap();
        let b = cat.irrep(&b, &[]).unwrap();
        let mut cold: Vec<Vec<i64>> = cat.fusion(&a, &b).unwrap().map(|c| c.dynkin()).collect();
        cold.sort();
        assert_eq!(cold, expected, "{id:?}");
        let mut sums = [0i64; 8];
        let before = counting_alloc::calls();
        let mut channels = 0;
        for c in cat.fusion(&a, &b).unwrap() {
            let copy = c.clone();
            for (i, x) in copy.dynkin_labels().enumerate() {
                sums[i] += x;
            }
            channels += 1;
        }
        let after = counting_alloc::calls();
        assert_eq!(after, before, "{id:?}");
        assert_eq!(channels, expected.len());
        for (i, s) in sums.iter().enumerate() {
            assert_eq!(*s, expected.iter().filter_map(|d| d.get(i)).sum::<i64>());
        }
    }
}

// Warm means the label and the group's root data exist; `dim` and `sqrt_dim`
// then evaluate the Weyl product in machine integers.
#[test]
fn warm_generated_dim_and_sqrt_dim_do_not_allocate() {
    let _serial = crate::common::counting_alloc::serial();
    for (id, label, dim) in [
        (GroupId::su(3).unwrap(), vec![1, 1], 8.0_f64),
        (GroupId::su(5).unwrap(), vec![2, 1, 0, 3], 2_625.0),
        (GroupId::spin(5).unwrap(), vec![1, 1], 16.0),
        (GroupId::sp(6).unwrap(), vec![1, 0, 1], 70.0),
    ] {
        let cat = LieGroup::new(id.clone()).unwrap();
        let a = cat.irrep(&label, &[]).unwrap();
        cat.dim(&a);
        let (calls, _, _, (d, s)) = counting_alloc::measure(|| (cat.dim(&a), cat.sqrt_dim(&a)));
        assert_eq!(calls, 0, "{id:?}");
        assert_eq!((d, s), (dim, dim.sqrt()), "{id:?}");
    }
}
