//! Tests for the exact B/C/D combinatorics. All anchors are independent of the
//! code under test: dimension/product values are hand-verified against the
//! representation theory of the classical groups (Fulton–Harris tables), never
//! read back from `fusion-symbols`'s own output.

use super::*;
use malachite_base::num::conversion::traits::ExactFrom;
use std::collections::HashSet;

fn irr(series: Series, dynkin: &[i64]) -> Irrep {
    Irrep::from_dynkin(series, dynkin).unwrap()
}

// ---- labels / normalization ----------------------------------------------

#[test]
fn dynkin_partition_round_trip() {
    for (series, d) in [
        (Series::B, vec![1, 0]),
        (Series::B, vec![2, 2, 4]),
        (Series::C, vec![1, 0, 1]),
        (Series::C, vec![3, 1]),
        (Series::D, vec![1, 0, 0, 0]),
        (Series::D, vec![0, 1, 1]), // D_3, a2+a3 even
        (Series::D, vec![2, 1, 1, 1]),
    ] {
        let s = irr(series, &d);
        assert_eq!(s.dynkin(), d, "round trip {series:?} {d:?}");
        assert_eq!(s.rank(), d.len());
    }
}

#[test]
fn partition_is_nonincreasing_and_dominant() {
    // B_3 vector rep (1,0,0) → partition (1,0,0).
    assert_eq!(
        irr(Series::B, &[1, 0, 0]).partition().unwrap(),
        vec![1, 0, 0]
    );
    // C_2 fundamental (1,0) → (1,0); adjoint (2,0) → (2,0).
    assert_eq!(irr(Series::C, &[1, 0]).partition().unwrap(), vec![1, 0]);
    assert_eq!(irr(Series::C, &[2, 0]).partition().unwrap(), vec![2, 0]);
    // D_3 chiral (0,2,0) vs (0,0,2): opposite last-coordinate signs (chirality).
    let p = irr(Series::D, &[0, 2, 0]).partition().unwrap();
    let q = irr(Series::D, &[0, 0, 2]).partition().unwrap();
    assert_eq!(p, vec![1, 1, -1]);
    assert_eq!(q, vec![1, 1, 1]);
}

/// `from_dynkin_in` rejects a group it does not implement, and a label whose
/// length disagrees with the group's rank — the two ill-posed inputs the
/// form-aware constructor adds.
#[test]
fn from_dynkin_in_rejects_a_foreign_group_and_a_wrong_length_label() {
    use crate::lie::{GroupId, RootSystem};
    let a3 = GroupId::su(4).unwrap();
    assert_eq!(
        Irrep::from_dynkin_in(&a3, &[1, 0, 0]),
        Err(BcdError::UnsupportedRootSystem {
            root_system: RootSystem::A(3)
        })
    );
    assert_eq!(
        Irrep::from_dynkin_in(&GroupId::spin(7).unwrap(), &[1, 0]),
        Err(BcdError::RankMismatch {
            expected: 3,
            got: 2
        })
    );
    assert_eq!(
        Irrep::from_dynkin_in(&GroupId::spin(7).unwrap(), &[]),
        Err(BcdError::EmptyLabel)
    );
}

/// The `redirect` of an excluded low rank names the right SU(2) statement for
/// the **requested form**: the cover is plain SU(2), the
/// quotient is its integer-`j` (resp. `j₁+j₂ ∈ ℤ`) sublattice.
#[test]
fn low_rank_redirect_is_form_aware() {
    use crate::lie::GroupId;
    let redirect = |g, d: &[i64]| match Irrep::from_dynkin_in(&g, d) {
        Err(BcdError::ExcludedRank { redirect, .. }) => redirect,
        other => panic!("expected ExcludedRank, got {other:?}"),
    };
    assert_eq!(
        redirect(GroupId::spin(3).unwrap(), &[1]),
        "use SU(2) instead"
    );
    assert_eq!(
        redirect(GroupId::so(3).unwrap(), &[2]),
        "use SU(2) with integer j only instead"
    );
    assert_eq!(
        redirect(GroupId::spin(4).unwrap(), &[0, 0]),
        "use SU(2)×SU(2) instead"
    );
    assert_eq!(
        redirect(GroupId::so(4).unwrap(), &[0, 0]),
        "use SU(2)×SU(2) with j₁+j₂ integer instead"
    );
}

// ---- guards ---------------------------------------------------------------

#[test]
fn excluded_low_ranks_are_typed_errors_with_redirection() {
    // B_1 = SO(3) ≅ SU(2).
    assert!(matches!(
        Irrep::from_dynkin(Series::B, &[2]),
        Err(BcdError::ExcludedRank {
            series: Series::B,
            rank: 1,
            redirect
        }) if redirect.contains("SU(2)")
    ));
    // C_1 = Sp(2) ≅ SU(2).
    assert!(matches!(
        Irrep::from_dynkin(Series::C, &[1]),
        Err(BcdError::ExcludedRank {
            series: Series::C,
            rank: 1,
            redirect
        }) if redirect.contains("SU(2)")
    ));
    // D_2 = SO(4) ≅ SU(2)×SU(2).
    assert!(matches!(
        Irrep::from_dynkin(Series::D, &[0, 0]),
        Err(BcdError::ExcludedRank {
            series: Series::D,
            rank: 2,
            redirect
        }) if redirect.contains("SU(2)×SU(2)")
    ));
}

#[test]
fn malformed_and_spinor_labels_are_typed_errors() {
    assert_eq!(
        Irrep::from_dynkin(Series::B, &[]),
        Err(BcdError::EmptyLabel)
    );
    assert!(matches!(
        Irrep::from_dynkin(Series::C, &[1, -1]),
        Err(BcdError::NegativeDynkin { .. })
    ));
    // B_2 spinor: a_2 odd.
    assert!(matches!(
        Irrep::from_dynkin(Series::B, &[0, 1]),
        Err(BcdError::NotAdmissible { .. })
    ));
    // D_3 tensor vs spinor by parity of a_2 + a_3.
    assert!(Irrep::from_dynkin(Series::D, &[1, 0, 0]).is_ok()); // 0+0 even (vector)
    assert!(Irrep::from_dynkin(Series::D, &[0, 1, 1]).is_ok()); // 1+1 even
    assert!(matches!(
        Irrep::from_dynkin(Series::D, &[0, 1, 2]),
        Err(BcdError::NotAdmissible { .. }) // 1+2 = 3 odd
    ));
    // C accepts every non-negative integer label (Sp is simply connected).
    assert!(Irrep::from_dynkin(Series::C, &[1, 1]).is_ok());
}

// ---- Weyl dimension anchors ----------------------------------------------

#[test]
fn dim_anchors_b2_so5() {
    // SO(5) = B_2. Known dims: 1, 5 (vector), 10 (adjoint), 14 (sym traceless),
    // 35, 16.
    let d = |a: &[i64]| irr(Series::B, a).dim();
    assert_eq!(d(&[0, 0]), Natural::from(1u32));
    assert_eq!(d(&[1, 0]), Natural::from(5u32)); // vector
    assert_eq!(d(&[0, 2]), Natural::from(10u32)); // adjoint (a_2 even)
    assert_eq!(d(&[2, 0]), Natural::from(14u32)); // sym traceless
    assert_eq!(d(&[1, 2]), Natural::from(35u32)); // partition (2,1)
    assert_eq!(d(&[2, 2]), Natural::from(81u32)); // partition (3,1)
}

#[test]
fn dim_anchors_c2_sp4() {
    // Sp(4) = C_2. Known dims: 1, 4 (fundamental), 5, 10 (adjoint), 16.
    let d = |a: &[i64]| irr(Series::C, a).dim();
    assert_eq!(d(&[0, 0]), Natural::from(1u32));
    assert_eq!(d(&[1, 0]), Natural::from(4u32)); // fundamental
    assert_eq!(d(&[0, 1]), Natural::from(5u32));
    assert_eq!(d(&[2, 0]), Natural::from(10u32)); // adjoint
    assert_eq!(d(&[1, 1]), Natural::from(16u32));
}

#[test]
fn dim_anchors_d_series() {
    // SO(8) = D_4 vector 8v dim 8; adjoint 28; sym traceless 35v.
    assert_eq!(irr(Series::D, &[1, 0, 0, 0]).dim(), Natural::from(8u32));
    assert_eq!(irr(Series::D, &[0, 1, 0, 0]).dim(), Natural::from(28u32)); // adjoint
    assert_eq!(irr(Series::D, &[2, 0, 0, 0]).dim(), Natural::from(35u32)); // 35v
                                                                           // SO(6) = D_3 ≅ SU(4): vector 6, adjoint 15, 20'.
    assert_eq!(irr(Series::D, &[1, 0, 0]).dim(), Natural::from(6u32));
    assert_eq!(irr(Series::D, &[0, 1, 1]).dim(), Natural::from(15u32)); // adjoint
    assert_eq!(irr(Series::D, &[2, 0, 0]).dim(), Natural::from(20u32));
}

// ---- dual --------------------------------------------------------------

#[test]
fn dual_is_involution_and_preserves_dim() {
    for (series, d) in [
        (Series::B, vec![1, 2]),
        (Series::B, vec![2, 0, 2]),
        (Series::C, vec![1, 1]),
        (Series::C, vec![2, 0, 1]),
        (Series::D, vec![1, 0, 0]),
        (Series::D, vec![1, 1, 3]),    // D_3 (a2+a3 even)
        (Series::D, vec![1, 0, 1, 1]), // D_4 (a3+a4 even)
    ] {
        let s = irr(series, &d);
        assert_eq!(s.dual().dual(), s, "involution {series:?} {d:?}");
        assert_eq!(s.dim(), s.dual().dim(), "dim {series:?} {d:?}");
    }
}

#[test]
fn dual_bc_and_d_even_self_dual() {
    // B, C, and D_even are self-dual.
    for (series, d) in [
        (Series::B, vec![2, 4]),
        (Series::C, vec![1, 3]),
        (Series::D, vec![1, 1, 0, 2]), // D_4 (even)
    ] {
        let s = irr(series, &d);
        assert_eq!(s.dual(), s, "self-dual {series:?} {d:?}");
    }
}

#[test]
fn dual_d_odd_swaps_last_two_dynkin() {
    // D_3 (r odd): duality swaps a_2 ↔ a_3.
    let s = irr(Series::D, &[0, 1, 3]); // 1+3 even
    assert_eq!(s.dual().dynkin(), vec![0, 3, 1]);
    // Vector (1,0,0) is self-dual (swap of the two zeros).
    assert_eq!(irr(Series::D, &[1, 0, 0]).dual().dynkin(), vec![1, 0, 0]);
    // Non-self-dual chiral pair.
    assert_ne!(irr(Series::D, &[0, 2, 0]), irr(Series::D, &[0, 0, 2]));
    assert_eq!(
        irr(Series::D, &[0, 2, 0]).dual(),
        irr(Series::D, &[0, 0, 2])
    );
}

// ---- Frobenius–Schur -----------------------------------------------------

#[test]
fn frobenius_schur_by_series() {
    // B: always +1 (SO(odd) tensor irreps are real).
    for d in [vec![0, 0], vec![1, 0], vec![2, 2], vec![1, 2]] {
        assert_eq!(irr(Series::B, &d).frobenius_schur(), 1, "B {d:?}");
    }
    // C = Sp: quaternionic iff sum of odd-position Dynkin labels is odd.
    // Vector (1,0): a_1 = 1 odd → -1 (quaternionic).
    assert_eq!(irr(Series::C, &[1, 0]).frobenius_schur(), -1);
    // Adjoint (2,0): a_1 = 2 even → +1 (real).
    assert_eq!(irr(Series::C, &[2, 0]).frobenius_schur(), 1);
    // (0,1): odd positions {a_1} = 0 even → +1 (the 5 of Sp(4), real).
    assert_eq!(irr(Series::C, &[0, 1]).frobenius_schur(), 1);
    // C_3 (1,0,0): odd positions a_1 + a_3 = 1 → -1.
    assert_eq!(irr(Series::C, &[1, 0, 0]).frobenius_schur(), -1);
    // C_3 (0,0,1): odd positions a_1 + a_3 = 1 → -1.
    assert_eq!(irr(Series::C, &[0, 0, 1]).frobenius_schur(), -1);
    // D_even: self-dual → +1.
    assert_eq!(irr(Series::D, &[1, 0, 0, 0]).frobenius_schur(), 1);
    // D_odd non-self-dual chiral → 0.
    assert_eq!(irr(Series::D, &[0, 2, 0]).frobenius_schur(), 0);
    // D_odd self-dual (vector) → +1.
    assert_eq!(irr(Series::D, &[1, 0, 0]).frobenius_schur(), 1);
}

// ---- weight multiplicities (Freudenthal) ---------------------------------

#[test]
fn weight_multiplicities_sum_to_dim() {
    // Σ_μ m(μ) · |Weyl orbit of μ| == dim.
    for (series, d) in [
        (Series::B, vec![1, 0]),
        (Series::B, vec![2, 2]), // adjoint of SO(5) has a zero weight of mult 2
        (Series::C, vec![1, 0]),
        (Series::C, vec![2, 0]),
        (Series::D, vec![1, 0, 0]),
        (Series::D, vec![0, 1, 1]),
        (Series::D, vec![1, 0, 0, 0]),
    ] {
        let s = irr(series, &d);
        let total: u64 = s
            .weight_multiplicities()
            .expect("tensor irrep")
            .iter()
            .map(|(mu, &m)| m * weyl_orbit(series, mu).len() as u64)
            .sum();
        assert_eq!(
            Natural::exact_from(total),
            s.dim(),
            "weight count {series:?} {d:?}"
        );
    }
}

#[test]
fn adjoint_has_rank_zero_weight() {
    // The adjoint of SO(5) (B_2 (0,2)) has the zero weight with multiplicity =
    // rank = 2.
    let s = irr(Series::B, &[0, 2]);
    assert_eq!(
        s.weight_multiplicities().unwrap().get(&vec![0, 0]),
        Some(&2)
    );
}

// ---- fundamental products (independent anchors) --------------------------

fn decomp(a: &Irrep, b: &Irrep) -> Vec<(Vec<i64>, u32)> {
    let mut v: Vec<(Vec<i64>, u32)> = directproduct(a, b)
        .unwrap()
        .into_iter()
        .map(|(c, n)| (c.dynkin(), n))
        .collect();
    v.sort();
    v
}

#[test]
fn so5_vector_squared() {
    // SO(5): 5 ⊗ 5 = 1 + 10 + 14.
    let v = irr(Series::B, &[1, 0]);
    let got = decomp(&v, &v);
    // dynkin: 1=(0,0), 14=(2,0), 10=(0,2).
    assert_eq!(got, vec![(vec![0, 0], 1), (vec![0, 2], 1), (vec![2, 0], 1)]);
}

#[test]
fn sp4_fundamental_squared() {
    // Sp(4): 4 ⊗ 4 = 1 + 5 + 10.
    let v = irr(Series::C, &[1, 0]);
    let got = decomp(&v, &v);
    // dynkin: 1=(0,0), 5=(0,1), 10=(2,0).
    assert_eq!(got, vec![(vec![0, 0], 1), (vec![0, 1], 1), (vec![2, 0], 1)]);
}

#[test]
fn so8_vector_squared() {
    // SO(8): 8v ⊗ 8v = 1 + 28 + 35v.
    let v = irr(Series::D, &[1, 0, 0, 0]);
    let got = decomp(&v, &v);
    // 1=(0,0,0,0), 28=(0,1,0,0), 35v=(2,0,0,0).
    assert_eq!(
        got,
        vec![
            (vec![0, 0, 0, 0], 1),
            (vec![0, 1, 0, 0], 1),
            (vec![2, 0, 0, 0], 1),
        ]
    );
}

#[test]
fn so6_vector_squared() {
    // SO(6): 6 ⊗ 6 = 1 + 15 + 20'.
    let v = irr(Series::D, &[1, 0, 0]);
    let got = decomp(&v, &v);
    assert_eq!(
        got,
        vec![(vec![0, 0, 0], 1), (vec![0, 1, 1], 1), (vec![2, 0, 0], 1)]
    );
}

// ---- exact property tests (tool-independent) -----------------------------

fn dim_sum_rule(a: &Irrep, b: &Irrep) {
    let lhs = a.dim() * b.dim();
    let rhs: Natural = directproduct(a, b)
        .unwrap()
        .into_iter()
        .map(|(c, n)| c.dim() * Natural::from(n))
        .sum();
    assert_eq!(lhs, rhs, "dim-sum rule {:?} ⊗ {:?}", a.dynkin(), b.dynkin());
}

fn product_symmetry(a: &Irrep, b: &Irrep) {
    assert_eq!(
        directproduct(a, b).unwrap(),
        directproduct(b, a).unwrap(),
        "N^c_ab == N^c_ba for {:?}, {:?}",
        a.dynkin(),
        b.dynkin()
    );
}

fn dual_twist(a: &Irrep, b: &Irrep) {
    // N^c_ab == N^{c̄}_{ā b̄}.
    let dp = directproduct(a, b).unwrap();
    let dpd = directproduct(&a.dual(), &b.dual()).unwrap();
    let twisted: BTreeMap<Irrep, u32> = dp.into_iter().map(|(c, n)| (c.dual(), n)).collect();
    assert_eq!(
        dpd,
        twisted,
        "dual twist {:?}, {:?}",
        a.dynkin(),
        b.dynkin()
    );
}

#[test]
fn group_mismatch_is_typed_error() {
    let b2 = irr(Series::B, &[1, 0]);
    let c2 = irr(Series::C, &[1, 0]);
    let b3 = irr(Series::B, &[1, 0, 0]);
    assert!(matches!(
        directproduct(&b2, &c2),
        Err(BcdError::GroupMismatch { .. })
    ));
    assert!(matches!(
        directproduct(&b2, &b3),
        Err(BcdError::GroupMismatch { .. })
    ));
}

#[test]
fn randomized_property_sweep() {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x0BC0_DE19_5150_4E37);
    // Keep labels small so weight systems (hence Freudenthal + orbits) stay
    // cheap — this is a property check, not a large-irrep stress test.
    let max_label = |r: usize| -> i64 {
        match r {
            2 => 3,
            3 => 2,
            _ => 1,
        }
    };
    for series in [Series::B, Series::C, Series::D] {
        let min_r = series.min_rank();
        for r in min_r..=4usize {
            let hi = max_label(r);
            let rand_irrep = |rng: &mut rand_chacha::ChaCha8Rng| -> Irrep {
                loop {
                    let d: Vec<i64> = (0..r).map(|_| rng.gen_range(0..=hi)).collect();
                    if let Ok(s) = Irrep::from_dynkin(series, &d) {
                        return s; // reject spinor labels by resampling
                    }
                }
            };
            for _ in 0..30 {
                let a = rand_irrep(&mut rng);
                let b = rand_irrep(&mut rng);
                dim_sum_rule(&a, &b);
                product_symmetry(&a, &b);
                dual_twist(&a, &b);
                assert_eq!(a.dual().dual(), a);
            }
        }
    }
}

// ---- independent closed forms: Weyl dimension, Freudenthal, Brauer–Klimyk --

/// Weights in the orthogonal basis `ε_1..ε_r`, doubled so every coordinate is
/// an integer (B spinor-free labels and `ρ_B` have half-integer coordinates).
mod eps {
    use super::Series;

    /// Highest weight of the Dynkin label, doubled. `ω_i = ε_1 + … + ε_i`
    /// except `ω_r = ½(ε_1 + … + ε_r)` (B), `ω_{r−1} = ½(ε_1 + … + ε_{r−1} − ε_r)`
    /// and `ω_r = ½(ε_1 + … + ε_r)` (D).
    pub fn from_dynkin(series: Series, a: &[i64]) -> Vec<i64> {
        let r = a.len();
        let mut l = vec![0; r];
        for (i, &ai) in a.iter().enumerate() {
            // Doubled fundamental weight `2ω_{i+1}`.
            let omega: Vec<i64> = (0..r)
                .map(|k| match series {
                    Series::B if i == r - 1 => 1,
                    Series::D if i == r - 1 => 1,
                    Series::D if i == r - 2 => {
                        if k == r - 1 {
                            -1
                        } else {
                            1
                        }
                    }
                    _ if k <= i => 2,
                    _ => 0,
                })
                .collect();
            for (lk, w) in l.iter_mut().zip(omega) {
                *lk += ai * w;
            }
        }
        l
    }

    pub fn to_dynkin(series: Series, l: &[i64]) -> Vec<i64> {
        let r = l.len();
        let mut a: Vec<i64> = (0..r - 1).map(|i| (l[i] - l[i + 1]) / 2).collect();
        a.push(match series {
            Series::B => l[r - 1],
            Series::C => l[r - 1] / 2,
            Series::D => (l[r - 2] + l[r - 1]) / 2,
        });
        a
    }

    /// `2ρ`: `2(r−k)+1` (B), `2(r−k+1)` (C), `2(r−k)` (D), `k = 1..r`.
    pub fn rho(series: Series, r: usize) -> Vec<i64> {
        (1..=r as i64)
            .map(|k| {
                let r = r as i64;
                match series {
                    Series::B => 2 * (r - k) + 1,
                    Series::C => 2 * (r - k + 1),
                    Series::D => 2 * (r - k),
                }
            })
            .collect()
    }

    /// Positive roots, doubled: `ε_i ± ε_j` (`i < j`), plus `ε_i` (B) or `2ε_i` (C).
    pub fn positive_roots(series: Series, r: usize) -> Vec<Vec<i64>> {
        let mut out = Vec::new();
        for i in 0..r {
            for j in i + 1..r {
                for s in [-2, 2] {
                    let mut v = vec![0; r];
                    v[i] = 2;
                    v[j] = s;
                    out.push(v);
                }
            }
            let mut v = vec![0; r];
            match series {
                Series::B => v[i] = 2,
                Series::C => v[i] = 4,
                Series::D => continue,
            }
            out.push(v);
        }
        out
    }

    /// Simple roots, doubled: `ε_i − ε_{i+1}`, then `ε_r` (B), `2ε_r` (C),
    /// `ε_{r−1} + ε_r` (D).
    pub fn simple_roots(series: Series, r: usize) -> Vec<Vec<i64>> {
        let mut out: Vec<Vec<i64>> = (0..r - 1)
            .map(|i| {
                let mut v = vec![0; r];
                v[i] = 2;
                v[i + 1] = -2;
                v
            })
            .collect();
        let mut v = vec![0; r];
        match series {
            Series::B => v[r - 1] = 2,
            Series::C => v[r - 1] = 4,
            Series::D => {
                v[r - 2] = 2;
                v[r - 1] = 2;
            }
        }
        out.push(v);
        out
    }

    pub fn dot(a: &[i64], b: &[i64]) -> i64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// The Weyl-group image of `v` in the dominant chamber and the
    /// determinant of the signed permutation reaching it, or `None` when `v`
    /// lies on a wall (W = signed permutations; for D, an even number of
    /// sign changes).
    pub fn dominant(series: Series, v: &[i64]) -> Option<(Vec<i64>, i64)> {
        let r = v.len();
        let mut idx: Vec<usize> = (0..r).collect();
        idx.sort_by_key(|&i| std::cmp::Reverse(v[i].abs()));
        let sorted: Vec<i64> = idx.iter().map(|&i| v[i].abs()).collect();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return None;
        }
        let mut sign = 1;
        for i in 0..r {
            for j in i + 1..r {
                if idx[i] > idx[j] {
                    sign = -sign;
                }
            }
        }
        let negatives = v.iter().filter(|&&x| x < 0).count();
        let mut out = sorted;
        match series {
            Series::B | Series::C => {
                if out[r - 1] == 0 {
                    return None;
                }
                if negatives % 2 == 1 {
                    sign = -sign;
                }
            }
            Series::D => {
                if negatives % 2 == 1 && out[r - 1] != 0 {
                    out[r - 1] = -out[r - 1];
                }
            }
        }
        Some((out, sign))
    }
}

/// `Π_{α>0} ⟨λ+ρ, α⟩ / ⟨ρ, α⟩`.
fn weyl_dimension(series: Series, dynkin: &[i64]) -> Natural {
    let r = dynkin.len();
    let rho = eps::rho(series, r);
    let lr: Vec<i64> = eps::from_dynkin(series, dynkin)
        .iter()
        .zip(&rho)
        .map(|(l, p)| l + p)
        .collect();
    let (mut num, mut den) = (Natural::from(1u32), Natural::from(1u32));
    for alpha in eps::positive_roots(series, r) {
        num *= Natural::exact_from(eps::dot(&lr, &alpha));
        den *= Natural::exact_from(eps::dot(&rho, &alpha));
    }
    assert_eq!(&num % &den, Natural::from(0u32));
    num / den
}

/// Weight multiplicities of the irrep (doubled ε coordinates), by
/// Freudenthal's recursion
/// `m(μ) = 2 Σ_{α>0} Σ_{k≥1} m(μ+kα) ⟨μ+kα, α⟩ / (|λ+ρ|² − |μ+ρ|²)`,
/// visiting weights by depth below `λ` (each step subtracts a simple root).
fn weight_multiplicities(series: Series, dynkin: &[i64]) -> BTreeMap<Vec<i64>, i64> {
    let r = dynkin.len();
    let lambda = eps::from_dynkin(series, dynkin);
    let rho = eps::rho(series, r);
    let shifted = |m: &[i64]| -> Vec<i64> { m.iter().zip(&rho).map(|(a, b)| a + b).collect() };
    let norm_top = {
        let s = shifted(&lambda);
        eps::dot(&s, &s)
    };
    let roots = eps::positive_roots(series, r);
    let simple = eps::simple_roots(series, r);
    let mut mult: BTreeMap<Vec<i64>, i64> = BTreeMap::new();
    mult.insert(lambda.clone(), 1);
    let mut layer = vec![lambda];
    while !layer.is_empty() {
        let mut next: Vec<Vec<i64>> = Vec::new();
        for mu in &layer {
            for s in &simple {
                let cand: Vec<i64> = mu.iter().zip(s).map(|(a, b)| a - b).collect();
                if !next.contains(&cand) && !mult.contains_key(&cand) {
                    next.push(cand);
                }
            }
        }
        let mut kept = Vec::new();
        for mu in next {
            let mut sum = 0;
            for alpha in &roots {
                let mut up: Vec<i64> = mu.iter().zip(alpha).map(|(a, b)| a + b).collect();
                while let Some(&m) = mult.get(&up) {
                    sum += m * eps::dot(&up, alpha);
                    up = up.iter().zip(alpha).map(|(a, b)| a + b).collect();
                }
            }
            let gap = norm_top - {
                let s = shifted(&mu);
                eps::dot(&s, &s)
            };
            if sum == 0 {
                continue;
            }
            assert!(gap > 0 && (2 * sum) % gap == 0, "Freudenthal {mu:?}");
            mult.insert(mu.clone(), 2 * sum / gap);
            kept.push(mu);
        }
        layer = kept;
    }
    mult
}

/// `N^c_{ab} = Σ_μ m_a(μ) det(w)` over `w(λ_b + μ + ρ) − ρ = λ_c` dominant
/// (Brauer–Klimyk).
fn klimyk(series: Series, a: &[i64], b: &[i64]) -> BTreeMap<Vec<i64>, u32> {
    let r = a.len();
    let rho = eps::rho(series, r);
    let lb = eps::from_dynkin(series, b);
    let mut acc: BTreeMap<Vec<i64>, i64> = BTreeMap::new();
    for (mu, m) in weight_multiplicities(series, a) {
        let v: Vec<i64> = (0..r).map(|k| lb[k] + mu[k] + rho[k]).collect();
        if let Some((dom, sign)) = eps::dominant(series, &v) {
            let c: Vec<i64> = dom.iter().zip(&rho).map(|(x, p)| x - p).collect();
            *acc.entry(eps::to_dynkin(series, &c)).or_default() += sign * m;
        }
    }
    acc.into_iter()
        .filter(|&(_, n)| n != 0)
        .map(|(c, n)| (c, u32::try_from(n).expect("nonnegative multiplicity")))
        .collect()
}

#[test]
fn dimensions_are_the_weyl_dimension_formula() {
    let mut checked = 0;
    for series in [Series::B, Series::C, Series::D] {
        for r in series.min_rank()..=4usize {
            let labels = (0..3i64.pow(r as u32)).map(|code| {
                (0..r)
                    .map(|i| code / 3i64.pow(i as u32) % 3)
                    .collect::<Vec<_>>()
            });
            for d in labels {
                let Ok(irrep) = Irrep::from_dynkin(series, &d) else {
                    continue; // spinor label
                };
                assert_eq!(irrep.dim(), weyl_dimension(series, &d), "{series:?} {d:?}");
                let total: i64 = weight_multiplicities(series, &d).values().sum();
                assert_eq!(
                    Natural::exact_from(total),
                    irrep.dim(),
                    "Freudenthal {series:?} {d:?}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 100, "{checked}");
    // Oscar 1.8.0 dimensions for labels with an entry 3, outside the sweep.
    for (series, d, want) in [(Series::B, [3, 2], 154), (Series::C, [3, 2], 140)] {
        assert_eq!(
            weyl_dimension(series, &d),
            Natural::exact_from(want),
            "{series:?} {d:?}"
        );
        assert_eq!(
            irr(series, &d).dim(),
            Natural::exact_from(want),
            "{series:?} {d:?}"
        );
    }
}

/// B/C/D rank 2-4 pairs, up to B3 `[2,2,2] ⊗ [1,2,0]` (19 683 × 693) with
/// multiplicities up to 22.
const KLIMYK_PAIRS: &[(Series, &[i64], &[i64])] = &[
    (Series::B, &[0, 0, 1, 0], &[0, 0, 1, 0]),
    (Series::B, &[0, 0, 1, 0], &[1, 0, 0, 0]),
    (Series::B, &[0, 1, 0, 0], &[0, 0, 1, 0]),
    (Series::B, &[0, 1, 0, 0], &[1, 1, 0, 0]),
    (Series::B, &[0, 1, 0, 0], &[1, 1, 1, 0]),
    (Series::B, &[0, 1, 0], &[1, 1, 0]),
    (Series::B, &[0, 2], &[0, 2]),
    (Series::B, &[0, 2], &[1, 2]),
    (Series::B, &[0, 2], &[3, 0]),
    (Series::B, &[1, 0, 0], &[1, 0, 0]),
    (Series::B, &[1, 0], &[1, 2]),
    (Series::B, &[1, 0], &[3, 2]),
    (Series::B, &[1, 1, 0, 0], &[0, 0, 0, 0]),
    (Series::B, &[1, 1, 0, 0], &[0, 0, 1, 0]),
    (Series::B, &[1, 1, 0, 0], &[0, 1, 0, 0]),
    (Series::B, &[1, 1, 2], &[1, 0, 0]),
    (Series::B, &[1, 2, 0], &[0, 1, 0]),
    (Series::B, &[1, 2], &[0, 0]),
    (Series::B, &[1, 2], &[3, 2]),
    (Series::B, &[2, 0, 2], &[0, 2, 0]),
    (Series::B, &[2, 1, 2], &[1, 2, 0]),
    (Series::B, &[2, 2, 2], &[0, 1, 0]),
    (Series::B, &[2, 2, 2], &[1, 2, 0]),
    (Series::C, &[0, 0, 0, 1], &[1, 1, 1, 1]),
    (Series::C, &[0, 0, 1, 0], &[0, 1, 0, 0]),
    (Series::C, &[0, 0, 1, 1], &[1, 0, 0, 1]),
    (Series::C, &[0, 1, 0, 0], &[0, 0, 1, 1]),
    (Series::C, &[0, 1, 1, 0], &[0, 1, 0, 1]),
    (Series::C, &[0, 1, 1, 1], &[1, 1, 0, 0]),
    (Series::C, &[0, 2, 1], &[2, 2, 2]),
    (Series::C, &[0, 2], &[1, 1]),
    (Series::C, &[1, 0, 0, 1], &[1, 0, 1, 0]),
    (Series::C, &[1, 0, 0], &[1, 2, 1]),
    (Series::C, &[1, 0, 1, 1], &[0, 0, 0, 1]),
    (Series::C, &[1, 2], &[3, 1]),
    (Series::C, &[2, 0], &[1, 1]),
    (Series::C, &[2, 1, 0], &[2, 2, 1]),
    (Series::C, &[2, 1, 2], &[2, 1, 1]),
    (Series::C, &[2, 1], &[1, 0]),
    (Series::C, &[2, 2, 0], &[0, 0, 0]),
    (Series::C, &[2, 2, 0], &[0, 1, 2]),
    (Series::C, &[2, 2, 1], &[1, 1, 0]),
    (Series::C, &[2, 2, 2], &[2, 0, 1]),
    (Series::C, &[2, 3], &[0, 1]),
    (Series::C, &[2, 3], &[1, 0]),
    (Series::C, &[2, 3], &[1, 3]),
    (Series::C, &[3, 0], &[3, 2]),
    (Series::D, &[0, 0, 0, 0], &[0, 1, 0, 0]),
    (Series::D, &[0, 0, 1, 1], &[0, 1, 1, 1]),
    (Series::D, &[0, 0, 1, 1], &[1, 0, 0, 0]),
    (Series::D, &[0, 0, 1, 1], &[1, 0, 1, 1]),
    (Series::D, &[0, 1, 1, 1], &[0, 0, 0, 0]),
    (Series::D, &[0, 1, 1, 1], &[1, 0, 0, 0]),
    (Series::D, &[1, 0, 0, 0], &[0, 1, 0, 0]),
];

#[test]
fn products_are_the_brauer_klimyk_decomposition() {
    for &(series, a, b) in KLIMYK_PAIRS {
        let (ia, ib) = (irr(series, a), irr(series, b));
        // Klimyk sums over the weights of the smaller factor.
        let (x, y) = if ia.dim() <= ib.dim() { (a, b) } else { (b, a) };
        let got: BTreeMap<Vec<i64>, u32> = directproduct(&ia, &ib)
            .unwrap()
            .into_iter()
            .map(|(c, n)| (c.dynkin(), n))
            .collect();
        assert_eq!(got, klimyk(series, x, y), "{series:?} {a:?} ⊗ {b:?}");
    }
}

#[test]
fn products_checked_against_oscar() {
    // Checked against Oscar 1.8.0 (`tensor_product_decomposition`).
    type Row<'a> = (Series, &'a [i64], &'a [i64], &'a [(&'a [i64], u32)]);
    let rows: [Row; 4] = [
        (
            Series::B,
            &[0, 2],
            &[1, 2],
            &[
                (&[0, 2], 1),
                (&[0, 4], 1),
                (&[1, 0], 1),
                (&[1, 2], 2),
                (&[1, 4], 1),
                (&[2, 0], 1),
                (&[2, 2], 1),
                (&[3, 0], 1),
            ],
        ),
        (
            Series::C,
            &[2, 0],
            &[1, 1],
            &[
                (&[1, 0], 1),
                (&[1, 1], 2),
                (&[1, 2], 1),
                (&[3, 0], 1),
                (&[3, 1], 1),
            ],
        ),
        (
            Series::B,
            &[1, 0, 0],
            &[1, 0, 0],
            &[(&[0, 0, 0], 1), (&[0, 1, 0], 1), (&[2, 0, 0], 1)],
        ),
        (
            Series::D,
            &[0, 0, 1, 1],
            &[1, 0, 0, 0],
            &[
                (&[0, 0, 0, 2], 1),
                (&[0, 0, 2, 0], 1),
                (&[0, 1, 0, 0], 1),
                (&[1, 0, 1, 1], 1),
            ],
        ),
    ];
    for (series, a, b, want) in rows {
        let got: BTreeMap<Vec<i64>, u32> = directproduct(&irr(series, a), &irr(series, b))
            .unwrap()
            .into_iter()
            .map(|(c, n)| (c.dynkin(), n))
            .collect();
        let want: BTreeMap<Vec<i64>, u32> = want.iter().map(|&(c, n)| (c.to_vec(), n)).collect();
        assert_eq!(got, want, "{series:?} {a:?} ⊗ {b:?}");
    }
}

#[test]
fn series_root_system_bridge() {
    assert_eq!(Series::B.root_system(3), RootSystem::B(3));
    assert_eq!(Series::C.root_system(3), RootSystem::C(3));
    assert_eq!(Series::D.root_system(4), RootSystem::D(4));
    for (series, r) in [(Series::B, 2), (Series::C, 2), (Series::D, 3)] {
        assert_eq!(
            series.root_system(r),
            series.published_group(r).root_systems()[0]
        );
    }
}

#[test]
fn cover_and_quotient_share_private_labels() {
    use crate::lie::GroupId;
    use std::hash::{Hash, Hasher};
    for (n, labels) in [
        (5, vec![vec![1, 0], vec![0, 2]]),
        (6, vec![vec![1, 0, 0], vec![0, 1, 1]]),
    ] {
        let cover = labels
            .iter()
            .map(|d| Irrep::from_dynkin_in(&GroupId::spin(n).unwrap(), d).unwrap())
            .collect::<Vec<_>>();
        let quotient = labels
            .iter()
            .map(|d| Irrep::from_dynkin_in(&GroupId::so(n).unwrap(), d).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(cover, quotient);
        let hash = |v: &Vec<Irrep>| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(&cover), hash(&quotient));
    }
}

#[test]
fn checked_dim_matches_ratio_path_on_small_labels() {
    let mut checked = 0;
    for series in [Series::B, Series::C, Series::D] {
        for r in series.min_rank().max(2)..=4usize {
            for code in 0..5i64.pow(r as u32) {
                let d: Vec<i64> = (0..r).map(|i| code / 5i64.pow(i as u32) % 5).collect();
                let Ok(x) = Irrep::from_dynkin(series, &d) else {
                    continue; // spinor label
                };
                let exact = x.ratio_dim();
                assert_eq!(exact, weyl_dimension(series, &d), "{series:?} {d:?}");
                assert_eq!(
                    x.checked_dim().map(Natural::from),
                    Some(exact.clone()),
                    "{series:?} {d:?}"
                );
                assert_eq!(x.dim(), exact, "{series:?} {d:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 500, "{checked}");
}

#[test]
fn checked_dim_overflow_falls_back_exactly() {
    for series in [Series::B, Series::C, Series::D] {
        let x = irr(series, &[i64::MAX / 16 - 1; 4]);
        assert_eq!(x.checked_dim(), None, "{series:?}");
        assert_eq!(x.dim(), x.ratio_dim(), "{series:?}");
    }
}

// Pre-optimization enumeration routes, retained only as exact regression oracles.
fn old_weyl_orbit(series: Series, mu: &[i64]) -> Result<Vec<Vec<i64>>, BcdError> {
    let r = mu.len();
    let mut set: HashSet<Vec<i64>> = HashSet::new();
    for signs in 0u32..weyl_sign_count(r)? {
        let flips = signs.count_ones();
        if series == Series::D && !flips.is_multiple_of(2) {
            continue;
        }
        let signed: Vec<i64> = (0..r)
            .map(|i| if signs & (1 << i) != 0 { -mu[i] } else { mu[i] })
            .collect();
        old_permute_into(&signed, &mut set);
    }
    Ok(set.into_iter().collect())
}

/// Insert every permutation of `v` into `set`.
fn old_permute_into(v: &[i64], set: &mut HashSet<Vec<i64>>) {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    old_permute_rec(v, &mut idx, 0, set);
}

fn old_permute_rec(v: &[i64], idx: &mut Vec<usize>, k: usize, set: &mut HashSet<Vec<i64>>) {
    let n = idx.len();
    if k == n {
        set.insert(idx.iter().map(|&i| v[i]).collect());
        return;
    }
    for i in k..n {
        idx.swap(k, i);
        old_permute_rec(v, idx, k + 1, set);
        idx.swap(k, i);
    }
}

fn old_right_product(a: &Irrep, b: &Irrep) -> Result<BTreeMap<Irrep, u32>, BcdError> {
    let series = a.series;
    let r = a.rank();
    let two_rho = two_rho(series, r);
    // 2(a + ρ), constant across the μ loop (the stored weight is already 2a).
    let two_a_rho: Vec<i64> = a
        .two_weight
        .iter()
        .zip(&two_rho)
        .map(|(&av, &tr)| add_weight(av, tr))
        .collect::<Result<_, _>>()?;
    // Reject the largest possible sum before enumerating b's weights.
    for (&ar, &w) in two_a_rho.iter().zip(b.two_weight.iter()) {
        add_weight(ar, w)?;
    }

    let mut acc: BTreeMap<Vec<i64>, i64> = BTreeMap::new();
    for (mu, &m) in &b.try_two_weight_multiplicities()? {
        let m = signed_multiplicity(m)?;
        for omega in old_weyl_orbit(series, mu)? {
            // 2ξ = 2(a+ρ) + 2μ (the orbit is already at the doubled scale).
            let two_xi: Vec<i64> = two_a_rho
                .iter()
                .zip(&omega)
                .map(|(&ar, &w)| add_weight(ar, w))
                .collect::<Result<_, _>>()?;
            for &x in &two_xi {
                x.checked_abs().ok_or(BcdError::LabelOverflow)?;
            }
            if let Some((dom, sign)) = dominant_conjugate_signed(series, &two_xi) {
                // 2c = dom - 2ρ.
                let c = sub(&dom, &two_rho)?;
                let count = acc.entry(c).or_insert(0);
                *count = add_signed_multiplicity(
                    *count,
                    // det(w) is ±1; checked narrowing made m nonnegative and <= i64::MAX.
                    if sign < 0 { -m } else { m },
                )?;
            }
        }
    }

    let mut result: BTreeMap<Irrep, u32> = BTreeMap::new();
    for (c, n) in acc {
        debug_assert!(n >= 0, "Racah–Speiser multiplicity must be non-negative");
        if n > 0 {
            result.insert(
                Irrep {
                    series,
                    two_weight: c.into(),
                },
                multiplicity(n)?,
            );
        }
    }
    Ok(result)
}

#[test]
#[ignore = "manual release timing; run alone with --nocapture"]
fn orbit_product_timing() {
    use std::{hint::black_box, time::Instant};
    let _guard = crate::cache::TEST_CACHE_LOCK.lock().unwrap();
    let a = irr(Series::B, &[1, 0, 0, 0]);
    let b = irr(Series::B, &[6, 0, 0, 0]);
    println!("SO(9) dimensions: {} x {}", a.dim(), b.dim());
    for old in [true, false] {
        let mut samples = Vec::new();
        for _ in 0..7 {
            crate::cache::reset();
            let start = Instant::now();
            black_box(if old {
                old_right_product(&a, &b)
            } else {
                directproduct(&a, &b)
            })
            .unwrap();
            samples.push(start.elapsed());
        }
        samples.sort();
        println!("product old={old}: {:?}", samples[samples.len() / 2]);
        let start = Instant::now();
        for _ in 0..1000 {
            black_box(if old {
                old_weyl_orbit(Series::D, black_box(&[2, 2, 0, 0, 0, 0]))
            } else {
                try_weyl_orbit(Series::D, black_box(&[2, 2, 0, 0, 0, 0]))
            })
            .unwrap();
        }
        println!("D6 orbit old={old}: {:?}/call", start.elapsed() / 1000);
    }
}

#[test]
fn distinct_orbits_match_old_enumeration() {
    for series in [Series::B, Series::C, Series::D] {
        // The orbit helper also covers D2 although the irrep engine redirects it.
        for rank in 2..=5 {
            let mut mixed = vec![0; rank];
            mixed[0] = 4;
            mixed[1] = 2;
            let distinct = (1..=rank as i64).rev().collect::<Vec<_>>();
            let mut weights = vec![vec![0; rank], vec![2; rank], mixed, distinct];
            if series == Series::D {
                let mut chiral = vec![2; rank];
                chiral[rank - 1] = -2;
                weights.push(chiral);
            }
            for weight in weights {
                let expected = old_weyl_orbit(series, &weight)
                    .unwrap()
                    .into_iter()
                    .collect::<HashSet<_>>();
                let candidates = try_weyl_orbit(series, &weight).unwrap();
                assert_eq!(
                    candidates.len(),
                    expected.len(),
                    "{series:?} {weight:?}: one candidate per orbit weight"
                );
                assert_eq!(
                    candidates.into_iter().collect::<HashSet<_>>(),
                    expected,
                    "{series:?} {weight:?}"
                );
            }
        }
        assert_eq!(
            try_weyl_orbit(series, &[0; 32]),
            Err(BcdError::LabelOverflow)
        );
        assert_eq!(
            try_weyl_orbit(series, &[i64::MIN, 0]),
            Err(BcdError::LabelOverflow)
        );
    }
}

#[test]
fn smaller_factor_products_match_old_right_expansion() {
    let _guard = crate::cache::TEST_CACHE_LOCK.lock().unwrap();
    crate::cache::reset();
    for series in [Series::B, Series::C, Series::D] {
        for rank in series.min_rank()..=5 {
            let mut label = vec![0; rank];
            label[0] = 1;
            let small = irr(series, &label);
            label[0] = 3;
            let large = irr(series, &label);
            let one = Irrep::trivial(series, rank).unwrap();
            for (a, b) in [
                (&small, &large),
                (&large, &small),
                (&one, &large),
                (&large, &one),
            ] {
                let expected = old_right_product(a, b).unwrap();
                // Test both cold routes, independently of the unordered cache key.
                assert_eq!(directproduct_uncached(a, b).unwrap(), expected);
                assert_eq!(directproduct(a, b).unwrap(), expected);
                let first = shared_directproduct(a, b).unwrap();
                let second = shared_directproduct(b, a).unwrap();
                assert!(
                    Arc::ptr_eq(&first, &second),
                    "warm swapped product must reuse the cached map"
                );
            }
            // Equal dimensions and chiral/spinor inputs exercise the tie rule too.
            let cover = series.cover_group(rank);
            label.fill(0);
            label[rank - 1] = 1;
            let spinor = Irrep::from_dynkin_in(&cover, &label).unwrap();
            for (a, b) in [(&spinor, &spinor.dual()), (&small, &spinor)] {
                assert_eq!(
                    directproduct_uncached(a, b).unwrap(),
                    old_right_product(a, b).unwrap()
                );
            }
        }
    }
}
