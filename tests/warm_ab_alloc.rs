//! Warm leaf A/B allocate only their returned block; a product level adds only its own.
#![cfg(feature = "lie-generated")]

use crate::common::counting_alloc;
use fusion_symbols::{
    cache,
    category::FusionCategory,
    lie::{GroupId, LieGroup},
    product::Product,
};

use std::sync::atomic::AtomicUsize;

fn measure<T>(f: impl FnOnce() -> T) -> ((usize, usize), T) {
    let (calls, bytes, _, result) = counting_alloc::measure(f);
    ((calls, bytes), result)
}

// Each non-inline Arc<[f64]> owns two reference counts and its coefficient payload.
fn buffers(lengths: &[usize]) -> (usize, usize) {
    let allocated = lengths.iter().filter(|&&n| n > 1);
    (
        allocated.clone().count(),
        allocated
            .map(|&n| std::mem::size_of::<[AtomicUsize; 2]>() + n * std::mem::size_of::<f64>())
            .sum(),
    )
}

fn check<C: FusionCategory<Scalar = f64>>(
    name: &str,
    cat: &C,
    a: &C::Object,
    b: &C::Object,
    c: &C::Object,
    composition_buffers: &[usize],
) -> usize {
    let mut elements = 0;
    for is_a in [true, false] {
        let query = || {
            if is_a {
                cat.a_symbol(a, b, c)
            } else {
                cat.b_symbol(a, b, c)
            }
            .unwrap()
        };
        let cold = query();
        elements = cold.data().len();
        let base = crate::common::stats_sum(&crate::common::BASE_TIERS);
        let generated = crate::common::stats_sum(crate::common::generated_tiers());
        let (allocation, warm) = measure(query);
        let expected = buffers(if composition_buffers.is_empty() {
            std::slice::from_ref(&elements)
        } else {
            composition_buffers
        });
        println!(
            "{name} {}: {}/{}, elements {elements}",
            if is_a { "A" } else { "B" },
            allocation.0,
            allocation.1
        );
        assert_eq!(
            allocation,
            expected,
            "{name} {}",
            if is_a { "A" } else { "B" }
        );
        assert_eq!(warm, cold);
        assert!(warm
            .data()
            .iter()
            .zip(cold.data())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        let base_warm = crate::common::stats_sum(&crate::common::BASE_TIERS);
        let generated_warm = crate::common::stats_sum(crate::common::generated_tiers());
        assert_eq!(
            base_warm.misses, base.misses,
            "{name}: no base regeneration"
        );
        assert_eq!(
            generated_warm.misses, generated.misses,
            "{name}: no generated re-sweep"
        );
        assert!(base_warm.hits + generated_warm.hits > base.hits + generated.hits);
    }
    for object in [a, b, c] {
        let ((calls, bytes), (dim, phase)) =
            measure(|| (cat.dim(object), cat.frobenius_schur_phase(object)));
        println!("{name} dim/FS: {calls}/{bytes}");
        assert_eq!((calls, bytes), (0, 0), "{name}: warm dimension and phase");
        assert_eq!(
            (dim, phase),
            (cat.dim(object), cat.frobenius_schur_phase(object))
        );
        assert_eq!(measure(|| cat.dual(object)).0, (0, 0), "{name}: dual");
    }
    assert_eq!(measure(|| cat.unit()).0, (0, 0), "{name}: unit");
    elements
}

// The serial guard isolates the cache lifecycle checks.
#[test]
fn warm_ab_allocate_only_leaf_blocks_and_one_buffer_per_product_level() {
    let _serial = crate::common::counting_alloc::serial();
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let oct = su3.irrep(&[1, 1]).unwrap();
    let n = check("SU3", &su3, &oct, &oct, &oct, &[]);
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = su2.irrep(&[1]).unwrap();
    let one = su2.irrep(&[2]).unwrap();
    check("SU2", &su2, &half, &one, &half, &[]);
    let sp4 = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
    let v = sp4.irrep(&[1, 0]).unwrap();
    let adj = sp4.irrep(&[2, 0]).unwrap();
    check("Sp4", &sp4, &v, &adj, &v, &[]);

    let p = Product::new((su3.clone(), su3.clone()));
    let x = (oct.clone(), oct.clone());
    // Two leaf blocks and the one output buffer.
    let nn = check("SU3xSU3", &p, &x, &x, &x, &[n, n, n * n]);
    let p = Product::new((p, su3));
    let x = (x, oct);
    // Three leaf blocks, the nested factor's own block, and the output buffer.
    check("(SU3xSU3)xSU3", &p, &x, &x, &x, &[n, n, nn, n, nn * n]);

    // Complex SU(N) and odd-D chiral labels must also dualize without rebuilding.
    for (id, labels) in [
        (GroupId::su(3).unwrap(), &[1, 0][..]),
        (GroupId::su(4).unwrap(), &[1, 0, 0][..]),
        (GroupId::spin(5).unwrap(), &[0, 1][..]),
        (GroupId::spin(6).unwrap(), &[0, 0, 1][..]),
        (GroupId::so(6).unwrap(), &[0, 2, 0][..]),
    ] {
        let cat = LieGroup::new(id).unwrap();
        let a = cat.irrep(labels).unwrap();
        let dual = cat.dual(&a);
        cat.dim(&a);
        cat.frobenius_schur_phase(&a);
        assert_eq!(measure(|| cat.dual(&a)).0, (0, 0));
        let (allocation, back) = measure(|| cat.dual(&dual));
        assert_eq!(allocation, (0, 0));
        assert_eq!(back, a);
        assert_eq!(
            measure(|| (cat.dim(&a), cat.frobenius_schur_phase(&a))).0,
            (0, 0)
        );
    }

    // Label-owned duals survive cache trim and reset without populating a tier.
    let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let a = cat.irrep(&[1, 0]).unwrap();
    cache::reset();
    let dual = cat.dual(&a);
    let trimmed = cache::trim_to(cache::CoefficientCacheTier::Product, 0);
    assert_eq!(trimmed.removed_entries, 0);
    assert_eq!(trimmed.remaining_charged_bytes, 0);
    assert_eq!(cat.dual(&a), dual);
    assert_eq!(measure(|| cat.dual(&a)).0, (0, 0));
    cache::reset();
    assert_eq!(
        cache::tier_stats(cache::CoefficientCacheTier::Product).entries,
        0
    );
    assert_eq!(cat.dual(&a), dual);
    assert_eq!(measure(|| cat.dual(&a)).0, (0, 0));
}
