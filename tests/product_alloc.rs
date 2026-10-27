use crate::common::counting_alloc;
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
    let half = (cat.factors().0.irrep(&[1], &[]).unwrap(), Parity::ODD);
    let zero = cat.unit();
    let _ = cat
        .f_symbol(&half, &half, &half, &half, &zero, &zero)
        .unwrap();
    let stats =
        fusion_symbols::cache::tier_stats(fusion_symbols::cache::CoefficientCacheTier::DerivedF);
    let before = counting_alloc::calls();
    let f = cat
        .f_symbol(&half, &half, &half, &half, &zero, &zero)
        .unwrap();
    let r = cat.r_symbol(&half, &half, &zero).unwrap();
    let empty = cat.r_symbol(&half, &zero, &zero).unwrap();
    let outcomes = cat.fusion(&half, &half).unwrap().count();
    let after = counting_alloc::calls();
    assert_eq!(after, before);
    assert_eq!(f.at(0, 0, 0, 0), -0.5);
    assert_eq!(r.at(0, 0), 1.0);
    assert!(empty.data().is_empty());
    assert_eq!(outcomes, 2);
    let warm =
        fusion_symbols::cache::tier_stats(fusion_symbols::cache::CoefficientCacheTier::DerivedF);
    assert_eq!(warm.misses, stats.misses);
    assert_eq!(warm.hits, stats.hits + 1);
}

#[test]
fn warm_d2_queries_allocate_only_as_much_as_product() {
    fn counts<C: FusionCategory>(cat: &C, a: &C::Object, zero: &C::Object) -> [usize; 3] {
        let f = || cat.f_symbol(a, a, a, a, zero, zero).unwrap();
        let n = || cat.nsymbol(a, a, zero);
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
    let a = cat.irrep(&[1, 1], &[]).unwrap();
    let spin = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = spin.irrep(&[1], &[]).unwrap();
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
        lie::{LieIrrep, U1Irrep, U1},
        product::Product,
    };
    let _serial = counting_alloc::serial();
    let none = [(0, 0); 4];
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let h = su2.irrep(&[1], &[]).unwrap();
    let (p1, m1) = (U1Irrep::new(1).unwrap(), U1Irrep::new(-1).unwrap());

    let u2 = LieGroup::new(GroupId::u(2).unwrap()).unwrap();
    let (a, b, one) = (
        u2.irrep(&[1], &[1]).unwrap(),
        u2.irrep(&[1], &[-1]).unwrap(),
        u2.unit(),
    );
    // Not composition: A and B take the N symbol of a dual object, and the dual
    // of a `Vec` product object owns a new Vec of its k factor labels.
    let dual = (1, 2 * std::mem::size_of::<LieIrrep>());
    assert_eq!(
        warm_frab(&u2, [&a, &b, &one], [&a, &b, &a, &a, &one, &one]),
        [(0, 0), (0, 0), dual, dual],
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

// Expected allocations are leaf-owned allocations plus one buffer per product
// level whose block has more than one element, plus the k-block holder of a
// `Vec` product nested as a factor, plus one transient f64 buffer when a real
// prefix of more than one element meets a complex factor.
#[cfg(feature = "lie-generated")]
#[test]
fn warm_product_symbols_allocate_one_buffer_per_product_level() {
    use fusion_symbols::{
        cache,
        category::{FBlock, RBlock, SymbolBlock},
        product::Product,
        table::TableCategory,
    };
    use num_complex::Complex64;
    use std::{mem::size_of, sync::atomic::AtomicUsize};

    // A non-inline Arc<[S]> owns two reference counts and its coefficients.
    fn buffer<S>(len: usize) -> (usize, usize) {
        if len > 1 {
            (1, size_of::<[AtomicUsize; 2]>() + len * size_of::<S>())
        } else {
            (0, 0)
        }
    }
    fn holder<B>(k: usize) -> (usize, usize) {
        (1, k * size_of::<B>())
    }
    fn sum<const N: usize>(parts: [(usize, usize); N]) -> (usize, usize) {
        parts
            .iter()
            .fold((0, 0), |(c, b), &(dc, db)| (c + dc, b + db))
    }
    fn times(k: usize, (c, b): (usize, usize)) -> (usize, usize) {
        (k * c, k * b)
    }
    // Warm F and R, and that each hits its SU(3) tier once per SU(3) leaf
    // without any cache miss.
    fn check<C: FusionCategory>(
        name: &str,
        cat: &C,
        x: &C::Object,
        k: u64,
        f: (usize, usize),
        r: (usize, usize),
    ) {
        use crate::common::{generated_tiers, stats_sum, BASE_TIERS};
        use cache::{tier_stats, CoefficientCacheTier as Tier};
        let query_f = || cat.f_symbol(x, x, x, x, x, x).unwrap();
        query_f();
        let (base, generated, hits) = (
            stats_sum(&BASE_TIERS),
            stats_sum(generated_tiers()),
            tier_stats(Tier::SunF).hits,
        );
        let (calls, bytes, _, _) = counting_alloc::measure(query_f);
        println!("{name} F: {calls}/{bytes}");
        assert_eq!((calls, bytes), f, "{name} F");
        assert_eq!(tier_stats(Tier::SunF).hits, hits + k, "{name}: factor hits");
        assert_eq!(
            stats_sum(generated_tiers()).misses,
            generated.misses,
            "{name}"
        );
        assert_eq!(stats_sum(&BASE_TIERS).misses, base.misses, "{name}");
        let query_r = || cat.r_symbol(x, x, x).unwrap();
        query_r();
        let (generated, hits) = (stats_sum(generated_tiers()), tier_stats(Tier::SunR).hits);
        let r_alloc = counting_alloc::measure(query_r);
        let r_alloc = (r_alloc.0, r_alloc.1);
        println!("{name} R: {}/{}", r_alloc.0, r_alloc.1);
        assert_eq!(r_alloc, r, "{name} R");
        assert_eq!(
            tier_stats(Tier::SunR).hits,
            hits + k,
            "{name}: R factor hits"
        );
        assert_eq!(
            stats_sum(generated_tiers()).misses,
            generated.misses,
            "{name} R"
        );
    }

    let _serial = counting_alloc::serial();
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let o = su3.irrep(&[1, 1], &[]).unwrap();
    let fib = TableCategory::fibonacci();
    let tau = fib.object(2).unwrap();
    let (n, m) = (
        su3.f_symbol(&o, &o, &o, &o, &o, &o)
            .unwrap()
            .as_slice()
            .len(),
        su3.r_symbol(&o, &o, &o).unwrap().as_slice().len(),
    );
    let (leaf_f, leaf_r) = (
        warm(|| su3.f_symbol(&o, &o, &o, &o, &o, &o).unwrap()),
        warm(|| su3.r_symbol(&o, &o, &o).unwrap()),
    );
    // A warm leaf R hit shares the cached 2×2 block: no allocation.
    assert_eq!(leaf_r, (0, 0), "SU(3) octet R");
    let (fib_f, fib_r) = (
        warm(|| fib.f_symbol(&tau, &tau, &tau, &tau, &tau, &tau).unwrap()),
        warm(|| fib.r_symbol(&tau, &tau, &tau).unwrap()),
    );

    let p = Product::new((su3.clone(), su3.clone()));
    let x = (o.clone(), o.clone());
    check(
        "SU3xSU3",
        &p,
        &x,
        2,
        sum([times(2, leaf_f), buffer::<f64>(n * n)]),
        sum([times(2, leaf_r), buffer::<f64>(m * m)]),
    );
    let left = Product::new((p.clone(), su3.clone()));
    let y = (x.clone(), o.clone());
    check(
        "(SU3xSU3)xSU3",
        &left,
        &y,
        3,
        sum([
            times(3, leaf_f),
            buffer::<f64>(n * n),
            buffer::<f64>(n * n * n),
        ]),
        sum([
            times(3, leaf_r),
            buffer::<f64>(m * m),
            buffer::<f64>(m * m * m),
        ]),
    );
    let right = Product::new((su3.clone(), p));
    let y = (o.clone(), x);
    check(
        "SU3x(SU3xSU3)",
        &right,
        &y,
        3,
        sum([
            times(3, leaf_f),
            buffer::<f64>(n * n),
            buffer::<f64>(n * n * n),
        ]),
        sum([
            times(3, leaf_r),
            buffer::<f64>(m * m),
            buffer::<f64>(m * m * m),
        ]),
    );
    let flat = Product::new(vec![su3.clone(); 3]);
    check(
        "Vec[SU3;3]",
        &flat,
        &vec![o.clone(); 3],
        3,
        sum([times(3, leaf_f), buffer::<f64>(n * n * n)]),
        sum([times(3, leaf_r), buffer::<f64>(m * m * m)]),
    );
    let nested = Product::new((Product::new(vec![su3.clone(); 2]), su3.clone()));
    check(
        "Vec[SU3;2]xSU3",
        &nested,
        &(vec![o.clone(); 2], o.clone()),
        3,
        sum([
            times(3, leaf_f),
            holder::<FBlock<f64>>(2),
            buffer::<f64>(n * n),
            buffer::<f64>(n * n * n),
        ]),
        sum([
            times(3, leaf_r),
            holder::<RBlock<f64>>(2),
            buffer::<f64>(m * m),
            buffer::<f64>(m * m * m),
        ]),
    );
    let real_first = Product::new((su3.clone(), fib.clone()));
    check(
        "SU3xFib",
        &real_first,
        &(o.clone(), tau),
        1,
        sum([leaf_f, fib_f, buffer::<f64>(n), buffer::<Complex64>(n)]),
        sum([leaf_r, fib_r, buffer::<f64>(m), buffer::<Complex64>(m)]),
    );
    let complex_first = Product::new((fib, su3));
    check(
        "FibxSU3",
        &complex_first,
        &(tau, o),
        1,
        sum([leaf_f, fib_f, buffer::<Complex64>(n)]),
        sum([leaf_r, fib_r, buffer::<Complex64>(m)]),
    );
}
