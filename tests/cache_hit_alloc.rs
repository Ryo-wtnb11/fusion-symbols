//! Every coefficient tier: warm lookups neither allocate nor regenerate.
use crate::common::counting_alloc;
use fusion_symbols::{
    cache::{self, CoefficientCacheTier as Tier, TierStats},
    category::FusionCategory,
    lie::{su2, GroupId, LieGroup},
};

fn stats(tier: Tier) -> TierStats {
    let base = cache::base_cache_stats();
    match tier {
        Tier::ThreeJ => base.three_j,
        Tier::SixJ => base.six_j,
        Tier::DerivedF => base.derived_f,
        #[cfg(feature = "lie-generated")]
        Tier::SunProduct => cache::generated_cache_stats().sun_product,
        #[cfg(feature = "lie-generated")]
        Tier::SunCgc => cache::generated_cache_stats().sun_cgc,
        #[cfg(feature = "lie-generated")]
        Tier::SunF => cache::generated_cache_stats().sun_f,
        #[cfg(feature = "lie-generated")]
        Tier::BcdCgc => cache::generated_cache_stats().bcd_cgc,
        #[cfg(feature = "lie-generated")]
        Tier::BcdF => cache::generated_cache_stats().bcd_f,
        _ => unreachable!(),
    }
}

fn check<T>(tier: Tier, query: impl Fn() -> T) {
    std::hint::black_box(query());
    let before = stats(tier);
    let (calls, bytes, _, value) = counting_alloc::measure(query);
    std::hint::black_box(value);
    assert_eq!((calls, bytes), (0, 0), "{tier}: warm allocation");
    let after = stats(tier);
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
    let half = su2.irrep(&[1], &[]).unwrap();
    let unit = su2.unit();
    check(Tier::DerivedF, || {
        su2.f_symbol(&half, &half, &half, &half, &unit, &unit)
            .unwrap()
    });

    #[cfg(feature = "lie-generated")]
    {
        use fusion_symbols::category::GroupRepresentations;
        let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        let octet = su3.irrep(&[1, 1], &[]).unwrap();
        check(Tier::SunProduct, || {
            su3.fusion(&octet, &octet).unwrap().count()
        });
        check(Tier::SunCgc, || su3.cgc(&octet, &octet, &octet).unwrap());
        check(Tier::SunF, || {
            su3.f_symbol(&octet, &octet, &octet, &octet, &octet, &octet)
                .unwrap()
        });
        let b2 = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
        let unit = b2.unit();
        let vector = b2.irrep(&[1, 0], &[]).unwrap();
        let adjoint = b2.irrep(&[0, 2], &[]).unwrap();
        // The B/C/D tier contains both exact products and CGC.
        check(Tier::BcdCgc, || {
            b2.fusion(&vector, &vector).unwrap().count()
        });
        check(Tier::BcdCgc, || b2.cgc(&vector, &vector, &adjoint).unwrap());
        check(Tier::BcdF, || {
            b2.f_symbol(&unit, &vector, &vector, &adjoint, &vector, &adjoint)
                .unwrap()
        });
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
