use super::{
    check_commutators, defining_seed, spinor_seeds, BcdError, CanonicalCatalog, Irrep, Series,
};
use crate::lie::GroupId;
const FAMILIES: &[(Series, usize, usize)] = &[
    (Series::B, 2, 5),
    (Series::B, 3, 7),
    (Series::D, 3, 6),
    (Series::D, 4, 8),
];
fn ir(n: usize, d: &[i64]) -> Irrep {
    Irrep::from_dynkin_in(&GroupId::spin(n).unwrap(), d).unwrap()
}
/// Every spinor seed passes the same exact commutator self-check as the
/// defining seeds, as every base case must.
#[test]
fn spinor_seed_commutators_close() {
    for &(series, r, _) in FAMILIES {
        for (label, seed) in spinor_seeds(series, r).unwrap() {
            check_commutators(&seed)
                .unwrap_or_else(|e| panic!("{series:?}_{r} spinor {label:?}: {e}"));
        }
    }
}

/// The derived structure constants are properties of the **algebra**, not of
/// the representation: the spinor seed must report exactly the `f_{i,k}` and
/// `d_{i,j}` of the defining seed. This is the sharpest available check that
/// the spinor generators are the *same* abstract generators the bootstrap
/// composes tensor products of — a wrong root assignment, a wrong node order,
/// or a wrong ladder normalization all move these.
#[test]
fn spinor_seed_structure_constants_match_the_defining_seed() {
    for &(series, r, _) in FAMILIES {
        let reference = check_commutators(&defining_seed(series, r).unwrap()).unwrap();
        for (label, seed) in spinor_seeds(series, r).unwrap() {
            let got = check_commutators(&seed).unwrap();
            assert_eq!(
                got.cartan_coeffs, reference.cartan_coeffs,
                "{series:?}_{r} spinor {label:?}: [Sp,Sp†] coefficients"
            );
            assert_eq!(
                got.root_weights, reference.root_weights,
                "{series:?}_{r} spinor {label:?}: root components"
            );
        }
    }
}

/// `B_r` has one spinor base case (`ω_r`), `D_r` two (the half-spinors), `C_r`
/// none — and the carrier dimensions are `2^r` and `2^{r-1}`.
#[test]
fn spinor_seed_inventory_is_one_for_b_two_for_d_none_for_c() {
    assert!(spinor_seeds(Series::C, 3).unwrap().is_empty());
    for &(series, r, _) in FAMILIES {
        let seeds = spinor_seeds(series, r).unwrap();
        let (want_count, want_dim) = match series {
            Series::B => (1, 1usize << r),
            _ => (2, 1usize << (r - 1)),
        };
        assert_eq!(seeds.len(), want_count, "{series:?}_{r} seed count");
        for (_, seed) in &seeds {
            assert_eq!(seed.dim(), want_dim, "{series:?}_{r} carrier dimension");
        }
    }
}

/// The excluded low ranks are excluded for the spinor seeds too, with the
/// simply-connected redirect (`Spin(3) ≅ SU(2)`, `Spin(4) ≅ SU(2)×SU(2)`).
#[test]
fn spinor_seeds_reject_the_excluded_low_ranks() {
    assert!(matches!(
        spinor_seeds(Series::B, 1),
        Err(BcdError::ExcludedRank { redirect, .. }) if redirect.contains("SU(2)")
    ));
    assert!(matches!(
        spinor_seeds(Series::D, 2),
        Err(BcdError::ExcludedRank { redirect, .. }) if redirect.contains("SU(2)×SU(2)")
    ));
}

/// The doubled ε-basis weight is what makes a spinor representable at all: its
/// `2λ` is all-odd, the tensor irreps' is all-even, and `partition()` — which
/// promises an integer partition — is `None` exactly on the spinors.
#[test]
fn spinor_weights_are_half_integers_in_the_doubled_encoding() {
    let s = ir(7, &[0, 0, 1]);
    assert_eq!(s.two_partition(), &[1, 1, 1]);
    assert!(s.is_spinor());
    assert_eq!(s.partition(), None);
    assert_eq!(s.weight_multiplicities(), None);

    let v = ir(7, &[1, 0, 0]);
    assert_eq!(v.two_partition(), &[2, 0, 0]);
    assert!(!v.is_spinor());
    assert_eq!(v.partition(), Some(vec![1, 0, 0]));
}

/// A spinor that is **not** a base case is reached by the canonical-parent
/// recursion (§14.4's spinor branch: `λ = μ + ω_r`, remove a box from `μ`).
/// `B_2`'s `(1,1)` is `λ = (3/2, 1/2)`, dimension 16.
#[test]
fn the_recursion_reaches_a_non_base_spinor() {
    let x = ir(5, &[1, 1]);
    assert_eq!(x.dim(), 16u32.into());
    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    assert_eq!(cat.generators(&x).unwrap().dim(), 16);
}

/// A catalog that is only ever asked for tensor irreps materializes **no**
/// spinor generators: the class-indexed candidate-set restriction keeps the spinor sector entirely out of the tensor bootstrap,
/// which is what preserves every shipped `SO(N)`/`Sp(2N)` coefficient.
#[test]
fn the_tensor_bootstrap_never_touches_the_spinor_sector() {
    let mut cat = CanonicalCatalog::new(Series::B, 2).unwrap();
    let before = cat.len();
    let adjoint = Irrep::from_dynkin(Series::B, &[0, 2]).unwrap();
    cat.generators(&adjoint).unwrap();
    assert!(cat.len() > before, "the adjoint must materialize something");
    // Nothing in the catalog is a spinor: with a spinor parent the SO(5)
    // adjoint's frame would have moved (the `so5_canonical_parent_hand_example`
    // diagnostic in `tests/global_form.rs`).
    let spinor = ir(5, &[0, 1]);
    let mut probe = CanonicalCatalog::new(Series::B, 2).unwrap();
    probe.generators(&adjoint).unwrap();
    assert_eq!(
        probe.len(),
        cat.len(),
        "materialization must be deterministic"
    );
    assert!(
        probe.cgc(&spinor, &spinor, &adjoint).is_ok(),
        "the spinor product is still available on demand"
    );
}
