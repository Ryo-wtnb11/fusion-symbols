//! Constructor regressions for #184, #187, and malformed public root data.
use fusion_symbols::lie::{CenterSubgroup, GlobalForm, GroupId, LieError, LieGroup, RootSystem};

#[test]
fn quotient_construction_rejects_degenerate_roots() {
    use CenterSubgroup::*;
    // #184: the last Dynkin label does not exist at B(0).
    assert!(!GroupId {
        root_system: RootSystem::B(0),
        form: GlobalForm::Quotient(Z2),
    }
    .admits(&[]));
    for sub in [Z2, Z4, DVector, DHalfSpinPlus, DHalfSpinMinus, DFull]
        .into_iter()
        .chain((0..=10).map(Zk))
    {
        for root_system in [
            RootSystem::B(0),
            RootSystem::C(0),
            RootSystem::D(0),
            RootSystem::D(1),
            RootSystem::A(0),
        ] {
            let group = GroupId {
                root_system,
                form: GlobalForm::Quotient(sub),
            };
            // Even a malformed label or a nonsensical datum must be safe to query.
            group.admits(&[]);
            group.admits(&[i64::MAX]);
            group.admits(&[-1]);
            for result in [
                LieGroup::new(group),
                LieGroup::with_catalog_budget(group, 0),
            ] {
                // Only these two empty-sum quotient formulas admit the zero label.
                if matches!(
                    (root_system, sub),
                    (RootSystem::C(0), Z2) | (RootSystem::A(0), Zk(1))
                ) {
                    assert!(
                        matches!(result, Err(LieError::Unsupported { .. })),
                        "{group:?}: {result:?}"
                    );
                } else {
                    assert!(
                        matches!(result, Err(LieError::NotAdmissible)),
                        "{group:?}: {result:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn unaddressable_su_weight_returns_a_typed_error() {
    // Only scalars are constructed: the internal weight would need N i64s.
    let group = GroupId::su(isize::MAX as usize / 8 + 1).unwrap();
    for result in [
        LieGroup::new(group),
        LieGroup::with_catalog_budget(group, usize::MAX),
    ] {
        #[cfg(feature = "lie-generated")]
        assert!(matches!(result, Err(LieError::LabelOverflow)), "{result:?}");
        #[cfg(not(feature = "lie-generated"))]
        assert!(
            matches!(
                result,
                Err(LieError::Unsupported {
                    requires: "lie-generated"
                })
            ),
            "{result:?}"
        );
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn huge_bcd_seeds_return_a_typed_error() {
    let r = 1usize << 40;
    for group in [
        GroupId::spin(2 * r + 1),
        GroupId::so(2 * r + 1),
        GroupId::sp(2 * r),
        GroupId::spin(2 * r),
        GroupId::so(2 * r),
    ] {
        let group = group.unwrap();
        for result in [
            LieGroup::new(group),
            LieGroup::with_catalog_budget(group, usize::MAX),
        ] {
            #[cfg(feature = "lie-generated")]
            assert!(
                matches!(result, Err(LieError::Generation(_))),
                "{group:?}: {result:?}"
            );
            #[cfg(not(feature = "lie-generated"))]
            assert!(
                matches!(
                    result,
                    Err(LieError::Unsupported {
                        requires: "lie-generated"
                    })
                ),
                "{group:?}: {result:?}"
            );
        }
    }
}
