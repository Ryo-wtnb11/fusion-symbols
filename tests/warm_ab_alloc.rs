//! Purpose: warm leaf and product A/B allocate nothing (counted in allocations).
#![cfg(feature = "lie-generated")]

use crate::common::counting_alloc;
use fusion_symbols::category::SymbolBlock;
use fusion_symbols::{
    cache,
    category::FusionCategory,
    lie::{GroupId, LieGroup},
    product::Product,
};

fn measure<T>(f: impl FnOnce() -> T) -> ((usize, usize), T) {
    let (calls, bytes, _, result) = counting_alloc::measure(f);
    ((calls, bytes), result)
}

fn check<C: FusionCategory<Scalar = f64>>(
    name: &str,
    cat: &C,
    a: &C::Object,
    b: &C::Object,
    c: &C::Object,
    buffers: usize,
) {
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
        let base = crate::common::stats_sum(&crate::common::BASE_TIERS);
        let generated = crate::common::stats_sum(&crate::common::PRODUCT_AND_R);
        let (allocation, warm) = measure(query);
        assert_eq!(
            allocation.0,
            buffers,
            "{name} {}",
            if is_a { "A" } else { "B" }
        );
        assert_eq!(warm, cold);
        assert!(warm
            .as_slice()
            .iter()
            .zip(cold.as_slice())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        let base_warm = crate::common::stats_sum(&crate::common::BASE_TIERS);
        let generated_warm = crate::common::stats_sum(&crate::common::PRODUCT_AND_R);
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
            measure(|| (cat.dim(object), cat.frobenius_schur_phase(object).unwrap()));
        assert_eq!((calls, bytes), (0, 0), "{name}: warm dimension and phase");
        assert_eq!(
            (dim, phase),
            (cat.dim(object), cat.frobenius_schur_phase(object).unwrap())
        );
        assert_eq!(measure(|| cat.dual(object)).0, (0, 0), "{name}: dual");
    }
    assert_eq!(measure(|| cat.unit()).0, (0, 0), "{name}: unit");
}

// Purpose: warm leaf A/B are cached views (shared or inline, no allocation),
// and so are products of exact leaves (one composite view). Byte totals depend on std
// layouts and are not asserted. The serial guard isolates the cache
// lifecycle checks.
#[test]
fn warm_leaf_and_product_ab_allocate_nothing() {
    let _serial = crate::common::counting_alloc::serial();
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let oct = su3.irrep(&[1, 1]).unwrap();
    check("SU3", &su3, &oct, &oct, &oct, 0);
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = su2.irrep(&[1]).unwrap();
    let one = su2.irrep(&[2]).unwrap();
    check("SU2", &su2, &half, &one, &half, 0);
    let sp4 = LieGroup::new(GroupId::sp(4).unwrap()).unwrap();
    let v = sp4.irrep(&[1, 0]).unwrap();
    let adj = sp4.irrep(&[2, 0]).unwrap();
    check("Sp4", &sp4, &v, &adj, &v, 0);

    let p = Product::new((su3.clone(), su3.clone()));
    let x = (oct.clone(), oct.clone());
    // A warm composite view hit with unit +1: no buffer.
    check("SU3xSU3", &p, &x, &x, &x, 0);
    let p = Product::new((p, su3));
    let x = (x, oct);
    // Nested or flat, one composite view of the three leaves.
    check("(SU3xSU3)xSU3", &p, &x, &x, &x, 0);

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
        cat.frobenius_schur_phase(&a).unwrap();
        assert_eq!(measure(|| cat.dual(&a)).0, (0, 0));
        let (allocation, back) = measure(|| cat.dual(&dual));
        assert_eq!(allocation, (0, 0));
        assert_eq!(back, a);
        assert_eq!(
            measure(|| (cat.dim(&a), cat.frobenius_schur_phase(&a).unwrap())).0,
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
