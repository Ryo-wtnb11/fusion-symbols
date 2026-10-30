//! Twist oracle: every family's `twist()` must equal the
//! TensorKitSectors generic definition
//! `src/sectors.jl:twist_from_Rsymbol`,
//! θ_a = Σ_{c ∈ a⊗a} (d_c / d_a) tr R^{aa}_c,
//! evaluated here from racah's own R-symbols and fusion rules. The R-symbol
//! sum is independent of the `twist()` implementations, which return a
//! constant without generating any R-symbol.

use crate::common::a1;
#[cfg(feature = "lie-generated")]
use crate::common::lie as lie_test;
use a1::{exact_r, twist};
use fusion_symbols::category::FusionCategory;
#[cfg(feature = "lie-generated")]
use fusion_symbols::category::SymbolBlock;

#[cfg(feature = "lie-generated")]
const TOL: f64 = 1e-10;

#[test]
fn twist_matches_r_symbol_formula_up_to_j5() {
    for dj in 0..=10u32 {
        let a = a1::ir(dj);
        let sum: f64 = a1::group()
            .fusion(&a, &a)
            .unwrap()
            .map(|c| a1::group().dim(&c) * exact_r(dj, dj, c.dynkin().next().unwrap() as u32))
            .sum();
        let theta = sum / a1::group().dim(&a);
        // Integer-valued sum: exact, no tolerance needed.
        assert_eq!(theta, 1.0, "R-formula twist for dj = {dj}");
        assert_eq!(twist(dj), theta, "twist({dj})");
    }
}

#[cfg(feature = "lie-generated")]
fn to_f64(d: malachite_nz::natural::Natural) -> f64 {
    use malachite_base::{
        num::conversion::traits::RoundingFrom, rounding_modes::RoundingMode::Nearest,
    };
    f64::rounding_from(&d, Nearest).0
}

#[cfg(feature = "lie-generated")]
#[test]
fn sun_twist_matches_r_symbol_formula() {
    use lie_test::{product as directproduct, r_symbol};

    // 3, 3̄, 6, 8 (outer multiplicity 2 in 8⊗8→8), 15 = (2,1); SU(4) 4 and 6.
    for d in [
        &[1, 0][..],
        &[0, 1],
        &[2, 0],
        &[1, 1],
        &[2, 1],
        &[1, 0, 0],
        &[0, 1, 0],
    ] {
        let a = lie_test::irrep(d).unwrap();
        let mut sum = 0.0;
        for (c, n) in directproduct(&a, &a).unwrap() {
            let r = r_symbol(&a, &a, &c).unwrap();
            assert_eq!(r.dims()[0], n as usize);
            let tr: f64 = (0..r.dims()[0]).map(|mu| r.get([mu, mu])).sum();
            sum += to_f64(c.dim_exact()) * tr;
        }
        let theta = sum / to_f64(a.dim_exact());
        assert!((theta - 1.0).abs() < TOL, "{d:?}: θ = {theta}");
        assert_eq!(lie_test::group(&a).twist(&a), 1.0, "{d:?}");
    }
}

#[cfg(feature = "lie-generated")]
#[test]
fn bcd_twist_matches_r_symbol_formula() {
    use crate::common::bcd as bcd_test;
    use bcd_test::Series;
    use fusion_symbols::lie::GroupId;
    use fusion_symbols::lie::LieGroup;

    // Spin(5): vector 5 and spinor 4 (quaternionic); Sp(4): defining 4
    // (quaternionic) and 5; Spin(6): vector 6 and the non-self-dual spinor 4.
    for (series, rank, n, d, dim, spinor) in [
        (Series::B, 2, 5, &[1, 0][..], 5, false),
        (Series::B, 2, 5, &[0, 1], 4, true),
        (Series::C, 2, 0, &[1, 0], 4, false),
        (Series::C, 2, 0, &[0, 1], 5, false),
        (Series::D, 3, 6, &[1, 0, 0], 6, false),
        (Series::D, 3, 6, &[0, 1, 0], 4, true),
    ] {
        let id = if n == 0 {
            GroupId::sp(4).unwrap()
        } else {
            GroupId::spin(n).unwrap()
        };
        let cat = LieGroup::new(id).unwrap();
        let a = cat.irrep(d).unwrap();
        assert_eq!(
            (
                to_f64(a.dim_exact()),
                n != 0 && !GroupId::so(n).unwrap().admits(d, &[])
            ),
            (dim as f64, spinor)
        );
        let mut sum = 0.0;
        for (c, n) in bcd_test::product(&cat, &a, &a).unwrap() {
            let r = cat.r_symbol(&a, &a, &c).unwrap();
            assert_eq!(r.dims()[0], n as usize);
            let tr: f64 = (0..r.dims()[0]).map(|mu| r.get([mu, mu])).sum();
            sum += to_f64(c.dim_exact()) * tr;
        }
        let theta = sum / to_f64(a.dim_exact());
        assert!(
            (theta - 1.0).abs() < TOL,
            "{series:?}{rank} {d:?}: θ = {theta}"
        );
        assert_eq!(cat.twist(&a), 1.0, "{series:?}{rank} {d:?}");
    }
}
