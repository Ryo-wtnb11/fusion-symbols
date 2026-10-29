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
    check(Tier::ThreeJ, || su2::wigner_3j(60, 60, 60, 0, 0, 0));
    check(Tier::SixJ, || su2::wigner_6j(60, 60, 60, 60, 60, 60));
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = su2.irrep(&[1]).unwrap();
    let unit = su2.unit();
    check(Tier::DerivedF, || {
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
        check(Tier::SunCgc, || su3.cgc(&octet, &octet, &octet).unwrap());
        check(Tier::SunF, || {
            su3.f_symbol(&octet, &octet, &octet, &octet, &octet, &octet)
                .unwrap()
        });
        // The 2×2 multiplicity block is shared, not copied.
        check(Tier::SunR, || su3.r_symbol(&octet, &octet, &octet).unwrap());
        let b2 = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
        let unit = b2.unit();
        let vector = b2.irrep(&[1, 0]).unwrap();
        let adjoint = b2.irrep(&[0, 2]).unwrap();
        // Exact products use the same tier for every root system.
        check(Tier::Product, || {
            b2.fusion(&vector, &vector).unwrap().count()
        });
        check(Tier::BcdCgc, || b2.cgc(&vector, &vector, &adjoint).unwrap());
        check(Tier::BcdF, || {
            b2.f_symbol(&unit, &vector, &vector, &adjoint, &vector, &adjoint)
                .unwrap()
        });
        check(Tier::BcdR, || {
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
            check(Tier::Product, || g.try_nsymbol(&a, &b, &one).unwrap());
            check(Tier::Product, || g.fusion(&a, &b).unwrap().count());
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
