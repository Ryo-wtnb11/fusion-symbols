use crate::common::counting_alloc;
use fusion_symbols::category::SymbolBlock;
use fusion_symbols::{
    category::FusionCategory,
    lie::{GroupId, LieGroup},
};

#[test]
fn warm_product_composition_allocates_nothing() {
    let _serial = crate::common::counting_alloc::serial();
    use fusion_symbols::{
        fermion::{FermionParity, FermionParityIrrep as Parity},
        product::Product,
    };
    let cat = Product::new((
        LieGroup::new(GroupId::su(2).unwrap()).unwrap(),
        FermionParity,
    ));
    let half = (cat.factors().0.irrep(&[1]).unwrap(), Parity::ODD);
    let zero = cat.unit();
    let _ = cat
        .f_symbol(&half, &half, &half, &half, &zero, &zero)
        .unwrap();
    let stats =
        fusion_symbols::cache::tier_stats(fusion_symbols::cache::CoefficientCacheTier::FSymbol);
    let before = counting_alloc::calls();
    let f = cat
        .f_symbol(&half, &half, &half, &half, &zero, &zero)
        .unwrap();
    let r = cat.r_symbol(&half, &half, &zero).unwrap();
    let empty = cat.r_symbol(&half, &zero, &zero).unwrap();
    let outcomes = cat.fusion(&half, &half).unwrap().count();
    let after = counting_alloc::calls();
    assert_eq!(after, before);
    assert_eq!(f.get([0, 0, 0, 0]), -0.5);
    assert_eq!(r.get([0, 0]), 1.0);
    assert!(empty.as_slice().is_empty());
    assert_eq!(outcomes, 2);
    let warm =
        fusion_symbols::cache::tier_stats(fusion_symbols::cache::CoefficientCacheTier::FSymbol);
    assert_eq!(warm.misses, stats.misses);
    assert_eq!(warm.hits, stats.hits + 1);
}

#[test]
fn warm_d2_queries_allocate_only_as_much_as_product() {
    fn counts<C: FusionCategory>(cat: &C, a: &C::Object, zero: &C::Object) -> [usize; 3] {
        let f = || cat.f_symbol(a, a, a, a, zero, zero).unwrap();
        let n = || cat.nsymbol(a, a, zero).unwrap();
        let dim = || cat.dim(a);
        let _ = (f(), n(), dim());
        [
            counting_alloc::measure(f).0,
            counting_alloc::measure(n).0,
            counting_alloc::measure(dim).0,
        ]
    }

    let _serial = counting_alloc::serial();
    let cat = LieGroup::new(GroupId::spin(4).unwrap()).unwrap();
    let a = cat.irrep(&[1, 1]).unwrap();
    let spin = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = spin.irrep(&[1]).unwrap();
    let product = fusion_symbols::product::Product::new(vec![spin.clone(), spin]);
    let parts = vec![half.clone(), half];
    assert_eq!(
        counts(&cat, &a, &cat.unit()),
        counts(&product, &parts, &product.unit())
    );
}

fn warm<T>(query: impl Fn() -> T) -> (usize, usize) {
    query();
    let (calls, bytes, _, _) = counting_alloc::measure(query);
    (calls, bytes)
}

fn warm_frab<C: FusionCategory>(
    cat: &C,
    [a, b, c]: [&C::Object; 3],
    [fa, fb, fc, fd, fe, ff]: [&C::Object; 6],
) -> [(usize, usize); 4] {
    [
        warm(|| cat.f_symbol(fa, fb, fc, fd, fe, ff).unwrap()),
        warm(|| cat.r_symbol(a, b, c).unwrap()),
        warm(|| cat.a_symbol(a, b, c).unwrap()),
        warm(|| cat.b_symbol(a, b, c).unwrap()),
    ]
}

// Every level of these products has a one-element block, so composition folds
// scalars and the returned blocks stay inline.
#[test]
fn warm_multiplicity_free_products_allocate_nothing() {
    use fusion_symbols::{
        fermion::{FermionParity, FermionParityIrrep as Parity},
        finite::{ZNIrrep, ZN},
        lie::{U1Irrep, U1},
        product::Product,
    };
    let _serial = counting_alloc::serial();
    let none = [(0, 0); 4];
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let h = su2.irrep(&[1]).unwrap();
    let (p1, m1) = (U1Irrep::new(1).unwrap(), U1Irrep::new(-1).unwrap());

    let u2 = LieGroup::new(GroupId::u(2).unwrap()).unwrap();
    let (a, b, one) = (
        u2.irrep_with_charges(&[1], &[1]).unwrap(),
        u2.irrep_with_charges(&[1], &[-1]).unwrap(),
        u2.unit(),
    );
    // One Radical leaf (SU(2)): its own warm view, with no N symbol and so
    // no dual of the `Vec` product object.
    assert_eq!(
        warm_frab(&u2, [&a, &b, &one], [&a, &b, &a, &a, &one, &one]).map(|(calls, _)| calls),
        [0, 0, 0, 0],
        "U(2)"
    );

    let p = Product::new((su2.clone(), U1));
    let (a, b, one) = ((h.clone(), p1), (h.clone(), m1), p.unit());
    assert_eq!(
        warm_frab(&p, [&a, &b, &one], [&a, &b, &a, &a, &one, &one]),
        none,
        "SU2xU1"
    );

    let p = Product::new((ZN::new(3).unwrap(), su2.clone()));
    let [z1, z2] = [1, 2].map(|k| ZNIrrep::new(k, 3).unwrap());
    let (a, b, one) = ((z1, h.clone()), (z2, h.clone()), p.unit());
    assert_eq!(
        warm_frab(&p, [&a, &b, &one], [&a, &b, &a, &a, &one, &one]),
        none,
        "Z3xSU2"
    );

    let p = Product::new((FermionParity, U1, su2.clone()));
    let (a, b, one) = ((Parity::ODD, p1, h.clone()), (Parity::ODD, m1, h), p.unit());
    assert_eq!(
        warm_frab(&p, [&a, &b, &one], [&a, &b, &a, &a, &one, &one]),
        none,
        "FPxU1xSU2"
    );
}

// Purpose: a warm product symbol of exact leaves is one composite view hit
// and allocates nothing, however nested; with a float (table) factor it
// allocates its leaves' own blocks plus one output buffer per product level
// whose block has more than one element. The allocation count is derived
// from that structure; byte totals depend on std layouts and are not
// asserted.
#[cfg(feature = "lie-generated")]
#[test]
fn warm_product_symbols_allocate_by_route() {
    use fusion_symbols::{cache, category::SymbolBlock, product::Product, table::TableCategory};
    // A product level with a multi-element block owns one output buffer.
    fn buffer(len: usize) -> usize {
        usize::from(len > 1)
    }
    // Warm F and R without any cache miss. A product of `k` exact leaves
    // (`composite`) hits its composite view once in the Product tier and no
    // leaf tier; otherwise each SU(3) leaf hits its own tier once.
    #[allow(clippy::too_many_arguments)]
    fn check<C: FusionCategory>(
        name: &str,
        cat: &C,
        x: &C::Object,
        composite: bool,
        k: u64,
        f: usize,
        r: usize,
        // The complex path of a real prefix keeps one transient f64 buffer
        // that is wasted work, so its count is only bounded from below.
        exact: bool,
    ) {
        let matches = |calls: usize, expected: usize| {
            if exact {
                calls == expected
            } else {
                calls >= expected
            }
        };
        use crate::common::{stats_sum, BASE_TIERS, PRODUCT_AND_R};
        use cache::{tier_stats, CoefficientCacheTier as Tier};
        let query_f = || cat.f_symbol(x, x, x, x, x, x).unwrap();
        query_f();
        let leaf_hits = if composite { 0 } else { k };
        let (base, generated, hits, views) = (
            stats_sum(&BASE_TIERS),
            stats_sum(&PRODUCT_AND_R),
            tier_stats(Tier::FSymbol).hits,
            tier_stats(Tier::Product).hits,
        );
        let (calls, _, _, _) = counting_alloc::measure(query_f);
        assert!(matches(calls, f), "{name} F: {calls} calls, expected {f}");
        assert_eq!(
            tier_stats(Tier::FSymbol).hits,
            hits + leaf_hits,
            "{name}: factor hits"
        );
        // The float path's N symbols also hit the Product tier.
        if composite {
            assert_eq!(
                tier_stats(Tier::Product).hits,
                views + 1,
                "{name}: composite hits"
            );
        }
        assert_eq!(stats_sum(&PRODUCT_AND_R).misses, generated.misses, "{name}");
        assert_eq!(stats_sum(&BASE_TIERS).misses, base.misses, "{name}");
        let query_r = || cat.r_symbol(x, x, x).unwrap();
        query_r();
        let (generated, hits, views) = (
            stats_sum(&PRODUCT_AND_R),
            tier_stats(Tier::RSymbol).hits,
            tier_stats(Tier::Product).hits,
        );
        let (calls, _, _, _) = counting_alloc::measure(query_r);
        assert!(matches(calls, r), "{name} R: {calls} calls, expected {r}");
        assert_eq!(
            tier_stats(Tier::RSymbol).hits,
            hits + leaf_hits,
            "{name}: R factor hits"
        );
        // The float path's N symbols also hit the Product tier.
        if composite {
            assert_eq!(
                tier_stats(Tier::Product).hits,
                views + 1,
                "{name}: R composite hits"
            );
        }
        assert_eq!(
            stats_sum(&PRODUCT_AND_R).misses,
            generated.misses,
            "{name} R"
        );
    }

    let _serial = counting_alloc::serial();
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let o = su3.irrep(&[1, 1]).unwrap();
    let fib = TableCategory::fibonacci();
    let tau = fib.object_at(1).unwrap();
    let (n, m) = (
        su3.f_symbol(&o, &o, &o, &o, &o, &o)
            .unwrap()
            .as_slice()
            .len(),
        su3.r_symbol(&o, &o, &o).unwrap().as_slice().len(),
    );
    let (leaf_f, leaf_r) = (
        warm(|| su3.f_symbol(&o, &o, &o, &o, &o, &o).unwrap()).0,
        warm(|| su3.r_symbol(&o, &o, &o).unwrap()).0,
    );
    // A warm leaf R hit shares the cached 2×2 block: no allocation.
    assert_eq!(leaf_r, 0, "SU(3) octet R");
    let (fib_f, fib_r) = (
        warm(|| fib.f_symbol(&tau, &tau, &tau, &tau, &tau, &tau).unwrap()).0,
        warm(|| fib.r_symbol(&tau, &tau, &tau).unwrap()).0,
    );

    let p = Product::new((su3.clone(), su3.clone()));
    let x = (o.clone(), o.clone());
    check("SU3xSU3", &p, &x, true, 2, 0, 0, true);
    let left = Product::new((p.clone(), su3.clone()));
    let y = (x.clone(), o.clone());
    check("(SU3xSU3)xSU3", &left, &y, true, 3, 0, 0, true);
    let right = Product::new((su3.clone(), p));
    let y = (o.clone(), x);
    check("SU3x(SU3xSU3)", &right, &y, true, 3, 0, 0, true);
    let flat = Product::new(vec![su3.clone(); 3]);
    check(
        "Vec[SU3;3]",
        &flat,
        &vec![o.clone(); 3],
        true,
        3,
        0,
        0,
        true,
    );
    // Three Radical leaves fit inline: no leaf buffer at any nesting level.
    let nested = Product::new((Product::new(vec![su3.clone(); 2]), su3.clone()));
    check(
        "Vec[SU3;2]xSU3",
        &nested,
        &(vec![o.clone(); 2], o.clone()),
        true,
        3,
        0,
        0,
        true,
    );
    let real_first = Product::new((su3.clone(), fib.clone()));
    check(
        "SU3xFib",
        &real_first,
        &(o.clone(), tau),
        false,
        1,
        leaf_f + fib_f + buffer(n),
        leaf_r + fib_r + buffer(m),
        false,
    );
    let complex_first = Product::new((fib, su3));
    check(
        "FibxSU3",
        &complex_first,
        &(tau, o),
        false,
        1,
        leaf_f + fib_f + buffer(n),
        leaf_r + fib_r + buffer(m),
        true,
    );
}

// Purpose: preparing a product's largest multiplicity enumerates no nested
// Cartesian stream. Constructing and dropping
// `Product<(Product<Vec<SU(3)>>, SU(3))>::fusion` with four inner factors
// makes the same allocation calls for 2⁴ and for 5⁴ inner channels, and that
// count follows from the constructors: the inner `initial` collect (C(m),
// measured by a collect of the same items), its `active` clone and `current`
// collect (1 + 1), the outer clone of the nested iterator (3 buffers) and the
// first nested output at cursor initialization (1).
#[cfg(feature = "lie-generated")]
#[test]
fn nested_fusion_construction_allocates_independently_of_channel_count() {
    let _serial = counting_alloc::serial();
    use fusion_symbols::{lie::RootSystem, product::Product};
    const M: usize = 4;
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let cat = Product::new((Product::new(vec![su3.clone(); M]), su3.clone()));
    let measure = |label: &[i64]| {
        let a = su3.irrep(label).unwrap();
        let x = (vec![a.clone(); M], a.clone());
        let _ = cat.fusion(&x, &x).unwrap().count();
        let (calls, _, _, ()) = counting_alloc::measure(|| drop(cat.fusion(&x, &x).unwrap()));
        let (collect, _, _, ()) = counting_alloc::measure(|| {
            drop(
                cat.factors()
                    .0
                    .factors()
                    .iter()
                    .map(|g| g.fusion(&a, &a).map(|f| (f, 0usize)))
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap(),
            )
        });
        (calls, collect)
    };
    let (small, collect) = measure(&[1, 0]);
    assert_eq!(small, collect + 6);
    assert_eq!(measure(&[1, 1]).0, small);
    // A composite factor: the same relation, without a closed count.
    let composite =
        LieGroup::new(GroupId::cover(&[RootSystem::A(2), RootSystem::A(2)], 0).unwrap()).unwrap();
    let cat = Product::new((Product::new(vec![su3.clone(); M]), composite.clone()));
    let measure = |label: &[i64]| {
        let a = su3.irrep(label).unwrap();
        let both: Vec<i64> = label.iter().chain(label).copied().collect();
        let x = (vec![a; M], composite.irrep(&both).unwrap());
        let _ = cat.fusion(&x, &x).unwrap().count();
        counting_alloc::measure(|| drop(cat.fusion(&x, &x).unwrap())).0
    };
    assert_eq!(measure(&[1, 0]), measure(&[1, 1]));
}

// Purpose: a composite Lie group used as a factor contributes its leaves to
// the enclosing product's one composite view (here SU(2) and SU(2)), so warm
// unit-label F, R, A and B allocate nothing: no product dual is formed.
#[test]
fn warm_composite_factor_queries_allocate_nothing() {
    use fusion_symbols::product::Product;
    let _serial = counting_alloc::serial();
    let u2 = LieGroup::new(GroupId::u(2).unwrap()).unwrap();
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let p = Product::new((u2, su2));
    let one = p.unit();
    assert_eq!(
        warm_frab(&p, [&one; 3], [&one; 6]).map(|(calls, _)| calls),
        [0, 0, 0, 0]
    );
}

fn tuple12<T: Clone>(t: &T) -> (T, T, T, T, T, T, T, T, T, T, T, T) {
    let c = || t.clone();
    (c(), c(), c(), c(), c(), c(), c(), c(), c(), c(), c(), c())
}

// Purpose: a warm product CGC allocates nothing: two or more Radical leaves
// are one Product-tier hit and an `Arc` clone for any leaf count (the
// largest tuple arity, 12, and a nesting past it), one leaf is that leaf's
// own shared block, and a unit or absent product is a scalar.
#[test]
fn warm_product_cgc_allocates_nothing() {
    use fusion_symbols::category::GroupRepresentations;
    use fusion_symbols::finite::{ZNIrrep, ZN};
    use fusion_symbols::product::Product;
    let _serial = counting_alloc::serial();
    let g = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let x = |j| g.irrep(&[j]).unwrap();
    let z = ZN::new(3).unwrap();
    let q = |c| ZNIrrep::new(c, 3).unwrap();
    let two = Product::new((g.clone(), g.clone()));
    let [a, b, c] = [(x(1), x(2)), (x(1), x(1)), (x(2), x(1))];
    let three = Product::new(vec![g.clone(); 3]);
    let [va, vb, vc] = [[1, 2, 1], [1, 1, 2], [2, 1, 1]].map(|l| l.map(x).to_vec());
    let one = Product::new((g.clone(), z));
    let [oa, ob, oc] = [(x(1), q(1)), (x(1), q(1)), (x(2), q(2))];
    let absent = (x(2), q(0));
    let five = Product::new((g.clone(), g.clone(), g.clone(), g.clone(), g.clone()));
    let h5 = (x(1), x(1), x(1), x(1), x(1));
    let o5 = (x(0), x(2), x(0), x(2), x(0));
    let twelve = Product::new(tuple12(&g));
    let h12 = tuple12(&x(1));
    let o12 = twelve.unit();
    let nested = Product::new((twelve.clone(), g.clone()));
    let (hn, on) = ((h12.clone(), x(1)), (o12.clone(), x(0)));
    assert_eq!(
        [
            warm(|| two.cgc(&a, &b, &c).unwrap()).0,
            warm(|| three.cgc(&va, &vb, &vc).unwrap()).0,
            warm(|| one.cgc(&oa, &ob, &oc).unwrap()).0,
            warm(|| one.cgc(&oa, &ob, &absent).unwrap()).0,
            warm(|| five.cgc(&h5, &h5, &o5).unwrap()).0,
            warm(|| twelve.cgc(&h12, &h12, &o12).unwrap()).0,
            warm(|| nested.cgc(&hn, &hn, &on).unwrap()).0,
        ],
        [0; 7]
    );
}
