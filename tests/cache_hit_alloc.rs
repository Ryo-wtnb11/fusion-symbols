//! Every coefficient tier: warm lookups neither allocate nor regenerate.
use crate::common::counting_alloc;
use fusion_symbols::{
    cache::{self, CoefficientCacheTier as Tier},
    category::FusionCategory,
    lie::{su2, GroupId, LieGroup},
};

fn check<T>(tier: Tier, query: impl Fn() -> T) {
    std::hint::black_box(query());
    let before = cache::tier_stats(tier);
    let (calls, bytes, _, value) = counting_alloc::measure(query);
    std::hint::black_box(value);
    assert_eq!((calls, bytes), (0, 0), "{tier}: warm allocation");
    let after = cache::tier_stats(tier);
    assert_eq!(after.misses, before.misses, "{tier}: warm regeneration");
    assert!(after.hits > before.hits, "{tier}: cache hit required");
}

#[test]
fn every_coefficient_tier_has_zero_allocation_warm_hits() {
    let _serial = counting_alloc::serial();
    // Large radicands exercise shared big integers, not just inline limbs.
    check(Tier::Wigner3j, || su2::wigner_3j(60, 60, 60, 0, 0, 0));
    check(Tier::Wigner6j, || su2::wigner_6j(60, 60, 60, 60, 60, 60));
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = su2.irrep(&[1]).unwrap();
    let unit = su2.unit();
    check(Tier::FSymbol, || {
        su2.f_symbol(&half, &half, &half, &half, &unit, &unit)
            .unwrap()
    });

    #[cfg(feature = "lie-generated")]
    {
        use fusion_symbols::category::GroupRepresentations;
        let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        let octet = su3.irrep(&[1, 1]).unwrap();
        check(Tier::Product, || {
            su3.fusion(&octet, &octet).unwrap().count()
        });
        check(Tier::Product, || {
            su3.fusion(&octet, &octet)
                .unwrap()
                .map(|(_, n)| n)
                .sum::<u32>()
        });
        check(Tier::Cgc, || su3.cgc(&octet, &octet, &octet).unwrap());
        check(Tier::FSymbol, || {
            su3.f_symbol(&octet, &octet, &octet, &octet, &octet, &octet)
                .unwrap()
        });
        // The 2×2 multiplicity block is shared, not copied.
        check(Tier::RSymbol, || {
            su3.r_symbol(&octet, &octet, &octet).unwrap()
        });
        let b2 = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
        let unit = b2.unit();
        let vector = b2.irrep(&[1, 0]).unwrap();
        let adjoint = b2.irrep(&[0, 2]).unwrap();
        // Exact products use the same tier for every root system.
        check(Tier::Product, || {
            b2.fusion(&vector, &vector).unwrap().count()
        });
        check(Tier::Product, || {
            b2.fusion(&vector, &vector)
                .unwrap()
                .map(|(_, n)| n)
                .sum::<u32>()
        });
        check(Tier::Cgc, || b2.cgc(&vector, &vector, &adjoint).unwrap());
        check(Tier::FSymbol, || {
            b2.f_symbol(&unit, &vector, &vector, &adjoint, &vector, &adjoint)
                .unwrap()
        });
        check(Tier::RSymbol, || {
            b2.r_symbol(&vector, &vector, &adjoint).unwrap()
        });
        for (id, label) in [
            (GroupId::g2(), vec![1, 0]),
            (GroupId::e6(), vec![1, 0, 0, 0, 0, 0]),
            (GroupId::e8(), vec![0, 0, 0, 0, 0, 0, 0, 1]),
        ] {
            let g = LieGroup::new(id).unwrap();
            let a = g.irrep(&label).unwrap();
            let b = g.dual(&a);
            let one = g.unit();
            check(Tier::Product, || g.nsymbol(&a, &b, &one).unwrap());
            check(Tier::Product, || g.fusion(&a, &b).unwrap().count());
            check(Tier::Product, || {
                g.fusion(&a, &b).unwrap().map(|(_, n)| n).sum::<u32>()
            });
        }
        for id in [
            GroupId::su(6).unwrap(),
            GroupId::spin(12).unwrap(),
            GroupId::sp(8).unwrap(),
            GroupId::e8(),
        ] {
            let g = LieGroup::new(id).unwrap();
            let a = g
                .irrep(&vec![1000; g.group_id().semisimple_rank()])
                .unwrap();
            assert!(u128::try_from(&a.dim_exact()).is_err());
            assert_eq!(counting_alloc::measure(|| g.dim(&a)).0, 0);
        }
    }
}

#[test]
fn warm_hit_on_a_new_thread_allocates_only_on_its_first_query() {
    let _serial = counting_alloc::serial();
    std::hint::black_box(su2::wigner_6j(40, 40, 40, 40, 40, 40));
    std::thread::spawn(|| {
        // First query on this thread may register its reader-stripe and counter indices.
        std::hint::black_box(su2::wigner_6j(40, 40, 40, 40, 40, 40));
        let (calls, bytes, _, value) =
            counting_alloc::measure(|| su2::wigner_6j(40, 40, 40, 40, 40, 40));
        std::hint::black_box(value);
        assert_eq!((calls, bytes), (0, 0), "warm hit after thread setup");
    })
    .join()
    .unwrap();
}

/// Reading every channel's multiplicity from a cache-free category or a tuple
/// product of them allocates nothing.
#[test]
fn cache_free_fusion_multiplicities_allocate_nothing() {
    use fusion_symbols::{
        finite::{ZNIrrep, ZN},
        lie::{U1Irrep, U1},
        product::Product,
        table::TableCategory,
    };
    let _serial = counting_alloc::serial();
    fn zero<C: FusionCategory>(cat: &C, a: &C::Object, b: &C::Object) {
        let sum = || cat.fusion(a, b).unwrap().map(|(_, n)| n).sum::<u32>();
        std::hint::black_box(sum());
        let (calls, _, _, total) = counting_alloc::measure(sum);
        assert_eq!(calls, 0);
        assert!(total >= 1);
    }
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let three_halves = su2.irrep(&[3]).unwrap();
    zero(&su2, &three_halves, &three_halves);
    zero(&U1, &U1Irrep::new(2).unwrap(), &U1Irrep::new(-5).unwrap());
    let z3 = ZN::new(3).unwrap();
    let one = ZNIrrep::new(1, 3).unwrap();
    zero(&z3, &one, &one);
    let fib = TableCategory::fibonacci();
    let tau = fib.object_at(1).unwrap();
    zero(&fib, &tau, &tau);
    let torus = crate::common::lie::torus();
    let q = torus.irrep_with_charges(&[], &[3]).unwrap();
    zero(&torus, &q, &q);
    let tuple = Product::new((su2, U1, fib));
    let x = (three_halves, U1Irrep::new(1).unwrap(), tau);
    zero(&tuple, &x, &x);
}
