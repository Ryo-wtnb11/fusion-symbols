//! The Cartan engine (`docs/gauge_cartan.md`): one exact highest-weight
//! nullspace and intertwiner pipeline for every simple root system, whose only
//! per-type input is the irrep basis of §3. Crate-private.

pub(crate) mod cgc;
mod gt;
pub(crate) mod module;
mod rational;
pub(crate) mod roots;

use std::sync::LazyLock;

#[cfg(test)]
pub(crate) use self::cgc::observe_contraction;
pub(crate) use self::cgc::{GeneralFamily, RequestModules};
use self::roots::CartanData;
use crate::lie::RootSystem;

pub(crate) use roots::RootIrrep as Irrep;

/// The `A_1` root data shared by every rank-one presentation (SU(2), SO(3),
/// Sp(2), PSU(2), `B_1`, `C_1` and each factor of `D_2`, §1), built on the
/// first block-CGC request.
pub(crate) fn a1() -> &'static GeneralFamily {
    static A1: LazyLock<GeneralFamily> =
        LazyLock::new(|| GeneralFamily::with_data(CartanData::new(RootSystem::A(1))));
    &A1
}

#[cfg(test)]
mod tests {
    use super::cgc::BUILDS;
    use super::*;
    use crate::category::SymbolBlock;

    fn builds() -> [usize; 2] {
        BUILDS.with(std::cell::Cell::get)
    }

    /// A hot completed channel survives a working set of many absent R
    /// queries; warm calls build nothing on this thread and the tier stays
    /// within its process-wide limit. Retention never changes a bit.
    #[test]
    fn pressure_retains_completed_channels() {
        // Resetting tests hold this lock; warm-build checks must not overlap them.
        let _guard = crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let fam = GeneralFamily::new(RootSystem::G2);
        let (a, b, one): (&[i64], &[i64], &[i64]) = (&[1, 0], &[0, 1], &[0, 0]);
        let bits = || {
            fam.cgc_entries(a, a, b)
                .unwrap()
                .iter()
                .map(|e| e.value.to_bits())
                .collect::<Vec<_>>()
        };
        let expected = bits();
        let warm = builds();
        assert_eq!(bits(), expected);
        assert_eq!(builds(), warm, "warm calls must not regenerate");
        for n in 1..=600 {
            // Absent R channels request no CGC and build no module.
            assert!(fam
                .r_symbol([one, &[n, 0], one])
                .unwrap()
                .as_slice()
                .is_empty());
        }
        assert_eq!(builds(), warm, "absent channels must not build modules");
        let limit = crate::cache::cache_budgets().limit(crate::cache::CoefficientCacheTier::Cgc);
        assert!(crate::cache::tier_stats(crate::cache::CoefficientCacheTier::Cgc).bytes <= limit);
        assert_eq!(bits(), expected);
        let symbols = || {
            let f = fam.f_symbol([a, a, a, a, b, b]).unwrap();
            let r = fam.r_symbol([a, a, b]).unwrap();
            f.as_slice()
                .iter()
                .chain(r.as_slice())
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        let expected = symbols();
        let warm = builds();
        assert_eq!(symbols(), expected);
        assert_eq!(builds(), warm, "warm F/R must not rebuild channels");
    }
}

#[cfg(all(test, feature = "lie-generated"))]
mod product_contract_tests {
    use super::*;
    use crate::{
        cache,
        category::{FusionCategory, SymbolBlock},
        lie::{GroupId, LieError, LieGroup},
    };
    use std::sync::atomic::Ordering;

    #[test]
    #[ignore = "needs an isolated global producer counter; medium; CI test_suites default job runs it by exact name, single thread"]
    fn exceptional_product_cache_counted() {
        let builds = || roots::PRODUCT_BUILDS.load(Ordering::SeqCst);
        for (id, label) in [
            (GroupId::g2(), vec![1, 0]),
            (GroupId::f4(), vec![0, 0, 0, 1]),
            (GroupId::e6(), vec![1, 0, 0, 0, 0, 0]),
            (GroupId::e7(), vec![0, 0, 0, 0, 0, 0, 1]),
            (GroupId::e8(), vec![0, 0, 0, 0, 0, 0, 0, 1]),
        ] {
            let root = id.root_systems()[0];
            let g = LieGroup::new(id.clone()).unwrap();
            let h = LieGroup::new(id).unwrap();
            let a = g.irrep(&label).unwrap();
            let b = g.dual(&a);
            let one = g.unit();
            let mut absent = label.clone();
            absent[0] += 3;
            let absent = g.irrep(&absent).unwrap();
            cache::reset();
            let before = builds();
            assert_eq!(g.nsymbol(&a, &b, &one).unwrap(), 1);
            assert_eq!(builds() - before, 1);
            assert_eq!(g.nsymbol(&a, &b, &absent).unwrap(), 0);
            assert_eq!(h.nsymbol(&b, &a, &one).unwrap(), 1);
            let channels = g.fusion(&a, &b).unwrap().collect::<Vec<_>>();
            assert!(channels.contains(&one));
            let fam = GeneralFamily::new(root);
            let zero = vec![0; label.len()];
            assert_eq!(
                fam.mult(&label, &b.dynkin().collect::<Vec<_>>(), &zero)
                    .unwrap(),
                1
            );
            assert_eq!(
                builds() - before,
                1,
                "N, fusion, equal groups and the engine must share Product"
            );
            assert_eq!(
                cache::tier_stats(cache::CoefficientCacheTier::Cgc).entries,
                0
            );
            cache::reset();
            let before = builds();
            // Four validated N prechecks contain two distinct unordered pairs: (a,b), (one,a).
            let f = g.f_symbol(&a, &b, &a, &a, &one, &one).unwrap();
            assert!(!f.as_slice().is_empty());
            assert_eq!(builds() - before, 2);
            assert_eq!(g.f_symbol(&a, &b, &a, &a, &one, &one).unwrap(), f);
            assert_eq!(
                builds() - before,
                2,
                "warm validated F must not rebuild exact products"
            );
        }
    }

    #[test]
    fn n_does_not_inherit_coefficient_output_limit() {
        let _guard = cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        cache::reset();
        let g = LieGroup::new(GroupId::e6()).unwrap();
        let data = CartanData::new(RootSystem::E6);
        let big = data.irrep(&[20, 0, 0, 0, 0, 0]).unwrap();
        assert!(u32::try_from(big.dim()).is_err());
        assert!(!big.is_self_dual());
        assert!(!big.public_coefficient_label_ok());
        let one = data.irrep(&[0; 6]).unwrap();
        assert_eq!(data.multiplicity(&one, &big, &big).unwrap(), 1);
        assert_eq!(g.irrep(&[20, 0, 0, 0, 0, 0]), Err(LieError::LabelOverflow));
        cache::reset();
    }
}
