//! Independent oracles (`docs/gauge_exceptional.md` §9.3, §9.5–§9.8): the
//! Brauer–Klimyk counts of §9.3, exact isometry and intertwining checked with
//! a separately written coproduct, CGC entries derived by hand, the
//! pentagon/hexagon/unitarity gates through `frcore`, and `κ_a` against the
//! sign of `F^{a ā a}_a[𝟙, 𝟙]`.

use super::*;
use crate::category::{
    f_unitarity_residual, hexagon_residual, pentagon_residual, BraidingStyle, FBlock,
    FusionCategory, FusionStyle, RBlock, SymbolBlock,
};
use crate::frcore::{f_block_raw, r_block_raw};
use num_traits::Signed;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Instant;

const G7: [i64; 2] = [1, 0];
const G14: [i64; 2] = [0, 1];
const G27: [i64; 2] = [2, 0];
const G64: [i64; 2] = [1, 1];
const G77: [i64; 2] = [0, 2];
const G77P: [i64; 2] = [3, 0];

fn unit(rank: usize, i: usize) -> Vec<i64> {
    let mut l = vec![0; rank];
    l[i] = 1;
    l
}

fn q(n: i64, d: i64) -> Q {
    Q::new(n.into(), d.into())
}

/// `x ↦ X x` with `X` from L2's stored `F` (a vector keyed by `(m_a, m_b)`).
type Vector = BTreeMap<(usize, usize), Q>;

fn add(v: &mut Vector, k: (usize, usize), x: Q) {
    let e = v.entry(k).or_insert_with(Q::zero);
    *e += x;
    if e.is_zero() {
        v.remove(&k);
    }
}

/// `Δ(x) = x ⊗ 1 + 1 ⊗ x`, written independently of the engine's coproduct.
fn delta(op_a: &Op, op_b: &Op, v: &Vector) -> Vector {
    let mut out = Vector::new();
    for (&(p, r), x) in v {
        for (p2, y) in &op_a[p] {
            add(&mut out, (*p2, r), x * y);
        }
        for (r2, y) in &op_b[r] {
            add(&mut out, (p, *r2), x * y);
        }
    }
    out
}

fn as_vector(ch: &Channel, c: &Ladders, mc: usize, mu: usize) -> Vector {
    let mut v = Vector::new();
    for (&pq, x) in ch.spaces[c.space[mc]].pairs.iter().zip(&ch.phi[mc][mu]) {
        if !x.is_zero() {
            v.insert(pq, x.clone());
        }
    }
    v
}

fn weight(m: &Ladders, i: usize) -> &[i64] {
    &m.module.weights[m.space[i]].mu
}

/// §9.5 for one channel: in the rational coordinates,
/// `⟨Φ'_μ b_m, Φ'_ν b_m′⟩ = δ_μν δ_mm′ N_μ n^c_m` with `Φ' = √N Φ`,
/// `Δ(x) Φ' = Φ' x` for every `e_j` and `f_j`, and the weight support rule.
fn check_channel(fam: &mut GeneralFamily, a: &[i64], b: &[i64], c: &[i64], n: usize) {
    let (ch, ma, mb, mc) = fam.channel(a, b, c).unwrap();
    assert_eq!(ch.mult, n, "§5.2 N gate for {a:?}⊗{b:?}→{c:?}");
    let phi: Vec<Vec<Vector>> = (0..mc.dim())
        .map(|m| (0..n).map(|mu| as_vector(&ch, mc, m, mu)).collect())
        .collect();
    for (m, per_mu) in phi.iter().enumerate() {
        // Weight support.
        for v in per_mu {
            for &(p, r) in v.keys() {
                let sum: Vec<i64> = weight(ma, p)
                    .iter()
                    .zip(weight(mb, r))
                    .map(|(x, y)| x + y)
                    .collect();
                assert_eq!(sum, weight(mc, m));
            }
        }
        // Isometry: different weights have disjoint support, so compare within one.
        for (m2, per_mu2) in phi.iter().enumerate() {
            if mc.space[m] != mc.space[m2] {
                continue;
            }
            for (mu, x) in per_mu.iter().enumerate() {
                for (nu, y) in per_mu2.iter().enumerate() {
                    let got: Q = x
                        .iter()
                        .filter_map(|(k, u)| {
                            y.get(k).map(|w| u * w * &ma.norm[k.0] * &mb.norm[k.1])
                        })
                        .sum();
                    let want = if m == m2 && mu == nu {
                        &ch.seed_norms[mu] * &mc.norm[m]
                    } else {
                        Q::zero()
                    };
                    assert_eq!(got, want);
                }
            }
        }
        // Intertwining for every generator, including candidates §4.2 dropped.
        for (ops_a, ops_b, ops_c) in [
            (&ma.lower, &mb.lower, &mc.lower),
            (&ma.raise, &mb.raise, &mc.raise),
        ] {
            for j in 0..ops_c.len() {
                for (mu, v) in per_mu.iter().enumerate() {
                    let lhs = delta(&ops_a[j], &ops_b[j], v);
                    let mut rhs = Vector::new();
                    for (m2, x) in &ops_c[j][m] {
                        for (k, y) in &phi[*m2][mu] {
                            add(&mut rhs, *k, x * y);
                        }
                    }
                    assert_eq!(lhs, rhs, "{a:?}⊗{b:?}→{c:?} gen {j} m {m} μ {mu}");
                }
            }
        }
    }
}

fn check_product(root: RootSystem, a: &[i64], b: &[i64]) {
    let mut fam = GeneralFamily::new(root);
    let outcomes: Vec<(Vec<i64>, u32)> = fam
        .product(a, b)
        .unwrap()
        .iter()
        .map(|(c, n)| (c.clone(), *n))
        .collect();
    for (c, n) in outcomes {
        check_channel(&mut fam, a, b, &c, n as usize);
    }
}

/// §9.3, literal decompositions with dimensions from the spec.
fn decomposition(root: RootSystem, a: &[i64], b: &[i64]) -> Vec<(u64, u32)> {
    let mut fam = GeneralFamily::new(root);
    let product: Vec<(Vec<i64>, u32)> = fam
        .product(a, b)
        .unwrap()
        .iter()
        .map(|(c, n)| (c.clone(), *n))
        .collect();
    let mut out: Vec<(u64, u32)> = product
        .into_iter()
        .map(|(c, n)| {
            let d = u64::try_from(&fam.data().weyl_dimension(&c).unwrap()).unwrap();
            (d, n)
        })
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn brauer_klimyk_counts_match_the_spec() {
    use RootSystem::G2;
    assert_eq!(
        decomposition(G2, &G7, &G7),
        [(1, 1), (7, 1), (14, 1), (27, 1)]
    );
    assert_eq!(decomposition(G2, &G7, &G14), [(7, 1), (27, 1), (64, 1)]);
    assert_eq!(
        decomposition(G2, &G14, &G14),
        [(1, 1), (14, 1), (27, 1), (77, 1), (77, 1)]
    );
    assert_eq!(
        decomposition(G2, &G27, &G27),
        [
            (1, 1),
            (7, 1),
            (14, 1),
            (27, 2),
            (64, 2),
            (77, 1),
            (77, 1),
            (182, 1),
            (189, 1)
        ]
    );
}

/// §9.5 for every channel of G2 7⊗7, 7⊗14 and 14⊗14 (the N gate runs inside
/// `Channel::new`).
#[test]
fn g2_small_products_are_exact_isometric_intertwiners() {
    check_product(RootSystem::G2, &G7, &G7);
    check_product(RootSystem::G2, &G7, &G14);
    check_product(RootSystem::G2, &G14, &G14);
}

/// §9.5 for the multiplicity-2 channels G2 27⊗27 → 27, 64.
#[test]
fn g2_27x27_multiplicity_two_is_exact() {
    let mut fam = GeneralFamily::new(RootSystem::G2);
    for c in [G27, G64] {
        check_channel(&mut fam, &G27, &G27, &c, 2);
    }
}

/// §5.3 example: in G2 27⊗27 → 27 the pivots are `[0, 1]`, `Ψ_μ[p_{μ+1}] > 0`,
/// `Ψ_0[p_2] = 0` and `Ψ_1[p_1] ≠ 0`.
#[test]
fn g2_27x27_to_27_pivot_rule() {
    let mut fam = GeneralFamily::new(RootSystem::G2);
    let (ch, ..) = fam.channel(&G27, &G27, &G27).unwrap();
    assert_eq!(ch.pivots, [0, 1]);
    let psi = &ch.phi[0];
    assert!(psi[0][0].is_positive() && psi[1][1].is_positive());
    assert!(psi[0][1].is_zero());
    assert!(!psi[1][0].is_zero());
}

/// §9.5 on every channel of E6 `27 ⊗ 27̄ = 1 + 78 + 650`.
#[test]
fn e6_27x27bar_is_exact() {
    let (a, b) = (unit(6, 0), unit(6, 5));
    assert_eq!(
        decomposition(RootSystem::E6, &a, &b),
        [(1, 1), (78, 1), (650, 1)]
    );
    check_product(RootSystem::E6, &a, &b);
}

/// §9.5 on every channel of E6 `27 ⊗ 27 = 27̄ + 351 + 351′`.
#[test]
fn e6_27x27_is_exact() {
    let v = unit(6, 0);
    assert_eq!(
        decomposition(RootSystem::E6, &v, &v),
        [(27, 1), (351, 1), (351, 1)]
    );
    check_product(RootSystem::E6, &v, &v);
}

/// §9.5 on E6 `27 ⊗ 78 → 27`.
#[test]
fn e6_27x78_to_27_is_exact() {
    let mut fam = GeneralFamily::new(RootSystem::E6);
    check_channel(&mut fam, &unit(6, 0), &unit(6, 1), &unit(6, 0), 1);
}

/// §9.5 on the non-square F4 channel `26 ⊗ 52 → 26`.
#[test]
fn f4_26x52_to_26_is_exact() {
    let mut fam = GeneralFamily::new(RootSystem::F4);
    check_channel(&mut fam, &unit(4, 3), &unit(4, 0), &unit(4, 3), 1);
}

/// §9.5 on every channel of F4 `26 ⊗ 26 = 1 + 26 + 52 + 273 + 324`.
#[test]
fn f4_26x26_is_exact() {
    let v = unit(4, 3);
    assert_eq!(
        decomposition(RootSystem::F4, &v, &v),
        [(1, 1), (26, 1), (52, 1), (273, 1), (324, 1)]
    );
    check_product(RootSystem::F4, &v, &v);
}

/// §9.5 on every channel of E7 `56 ⊗ 56 = 1 + 133 + 1463 + 1539`.
#[test]
fn e7_56x56_is_exact() {
    let v = unit(7, 6);
    assert_eq!(
        decomposition(RootSystem::E7, &v, &v),
        [(1, 1), (133, 1), (1463, 1), (1539, 1)]
    );
    check_product(RootSystem::E7, &v, &v);
}

/// §9.5 on the cheap channels of E8 `248 ⊗ 248`: `→ 1` and `→ 248`.
#[test]
fn e8_248x248_to_1_and_248_is_exact() {
    let v = unit(8, 7);
    let mut fam = GeneralFamily::new(RootSystem::E8);
    check_channel(&mut fam, &v, &v, &[0; 8], 1);
    check_channel(&mut fam, &v, &v, &v, 1);
}

/// §9.5 on E8 `248 ⊗ 248 → 3875`.
#[test]
#[ignore = "release 1.2 s, debug 9.2 s"]
fn e8_248x248_to_3875_is_exact() {
    let v = unit(8, 7);
    let mut fam = GeneralFamily::new(RootSystem::E8);
    check_channel(&mut fam, &v, &v, &unit(8, 0), 1);
}

#[test]
fn nullity_gate_rejects_a_wrong_multiplicity() {
    let data = CartanData::new(RootSystem::G2);
    let m7 = Ladders::new(&data, &G7).unwrap();
    let m14 = Ladders::new(&data, &G14).unwrap();
    let err = Channel::new(&data, &m7, &m7, &m14, 2).err().unwrap();
    assert_eq!(
        err,
        LieError::from(CgcError::NullityMismatch {
            a: G7.to_vec(),
            b: G7.to_vec(),
            c: G14.to_vec(),
            nullity: 1,
            multiplicity: 2
        })
    );
}

/// G2 7 (§3–§4 by hand, Cartan rows α₀ = (2,−1), α₁ = (−3,2)): weights
/// m0 (1,0), m1 (−1,1), m2 (2,−1), m3 (0,0), m4 (−2,1), m5 (1,−1), m6 (−1,0),
/// one each, with norms 1, 1, 1, 2, 4, 4, 4. In the orthonormal basis
/// `f₀: u0→u1 (1), u2→u3 (√2), u3→u4 (√2), u5→u6 (1)` and
/// `f₁: u1→u2 (1), u4→u5 (1)`.
///
/// 7⊗7 → 7: P = [(3,0), (2,1), (1,2), (0,3)] (composites 3, 9, 15, 21).
/// `Δ(e₀) x = 0` gives `√2 x₁ + x₂ = 0`, `x₃ + √2 x₄ = 0`; `Δ(e₁) x = 0`
/// gives `x₂ + x₃ = 0`. So `Ψ = (1, −√2, √2, −1)/√6`, pivot row 0.
/// Lowering by `Δ(f₀)` (m_c = 1 is `f₀ u0` with entry 1):
/// `(√2 u4⊗u0 − u3⊗u1 + u1⊗u3 − √2 u0⊗u4)/√6`.
#[test]
fn g2_7x7_to_7_hand_entries_and_order() {
    let mut fam = GeneralFamily::new(RootSystem::G2);
    let (ch, ma, mb, mc) = fam.channel(&G7, &G7, &G7).unwrap();
    let got: Vec<_> = ch
        .exact_entries(ma, mb, mc)
        .into_iter()
        .filter(|e| e.2 <= 1)
        .map(|(p, r, m, mu, x)| (p, r, m, mu, x.signed_square()))
        .collect();
    let want = vec![
        (3, 0, 0, 0, q(1, 6)),
        (2, 1, 0, 0, q(-1, 3)),
        (1, 2, 0, 0, q(1, 3)),
        (0, 3, 0, 0, q(-1, 6)),
        (4, 0, 1, 0, q(1, 3)),
        (3, 1, 1, 0, q(-1, 6)),
        (1, 3, 1, 0, q(1, 6)),
        (0, 4, 1, 0, q(-1, 3)),
    ];
    assert_eq!(got, want);
}

/// §5.6 order on a multiplicity-2 channel: `μ`, then `m_c`, then composite.
#[test]
fn entry_order_is_mu_then_mc_then_composite() {
    let mut fam = GeneralFamily::new(RootSystem::G2);
    let e = fam
        .cgc_entries(&G14.to_vec(), &G14.to_vec(), &G27.to_vec())
        .unwrap();
    let d = 14u32;
    let keys: Vec<_> = e.iter().map(|x| (x.mu, x.m3, x.m1 + d * x.m2)).collect();
    assert!(keys.windows(2).all(|w| w[0] < w[1]));
    assert!(e.iter().all(|x| x.value != 0.0));
}

/// §5.5: `1 ⊗ s → s` and `s ⊗ 1 → s` are exactly the identity.
#[test]
fn trivial_couplings_are_the_identity() {
    let mut fam = GeneralFamily::new(RootSystem::G2);
    let one = [0, 0];
    for s in [G7, G14, G27] {
        for (a, b) in [(&one[..], &s[..]), (&s[..], &one[..])] {
            let (ch, ma, mb, mc) = fam.channel(a, b, &s).unwrap();
            let got: Vec<_> = ch.exact_entries(ma, mb, mc);
            assert_eq!(got.len(), mc.dim());
            for (m, (p, r, mcx, mu, x)) in got.into_iter().enumerate() {
                assert_eq!((p + r, mcx, mu), (m, m, 0));
                assert_eq!(x.signed_square(), Q::one());
            }
        }
    }
}

/// §9.8 within one process: two independent families give the same bits.
#[test]
fn regeneration_is_byte_identical() {
    let bits = || {
        let mut fam = GeneralFamily::new(RootSystem::G2);
        let mut out = Vec::new();
        for c in [G7, G14, G27, [0, 0]] {
            for e in fam
                .cgc_entries(&G7.to_vec(), &G7.to_vec(), &c.to_vec())
                .unwrap()
            {
                out.push((e.m1, e.m2, e.m3, e.mu, e.value.to_bits()));
            }
        }
        out
    };
    assert_eq!(bits(), bits());
}

/// Entry-level drift check against the pre-optimization exact formula (§5.4).
fn check_radical_formation(root: RootSystem, a: &[i64], b: &[i64], c: &[i64]) {
    let mut fam = GeneralFamily::new(root);
    let (ch, ma, mb, mc) = fam.channel(a, b, c).unwrap();
    let got = ch.exact_entries(ma, mb, mc);
    let mut entries = got.iter();
    for mu in 0..ch.mult {
        for (m, per_mu) in ch.phi.iter().enumerate() {
            let scale = &ch.seed_norms[mu] * &mc.norm[m];
            for (&(pa, pb), x) in ch.spaces[mc.space[m]].pairs.iter().zip(&per_mu[mu]) {
                if x.is_zero() {
                    continue;
                }
                let old = SignedSqrtRational::from_prefactor_radical(
                    x.clone(),
                    &ma.norm[pa] * &mb.norm[pb] / &scale,
                );
                let (ga, gb, gc, gm, value) = entries.next().unwrap();
                assert_eq!((*ga, *gb, *gc, *gm), (pa, pb, m, mu));
                assert_eq!(*value, old);
                assert_eq!(value.to_f64().to_bits(), old.to_f64().to_bits());
            }
        }
    }
    assert!(entries.next().is_none());
    eprintln!(
        "{root:?} {a:?}⊗{b:?}→{c:?}: {} exact entries agree",
        got.len()
    );
}

#[test]
fn radical_formation_preserves_exact_entries() {
    for c in [G27, G64] {
        check_radical_formation(RootSystem::G2, &G27, &G27, &c);
    }
    for (root, a, c) in [
        (RootSystem::F4, unit(4, 3), unit(4, 2)),
        (RootSystem::E7, unit(7, 6), unit(7, 0)),
        (RootSystem::E8, unit(8, 7), unit(8, 7)),
    ] {
        check_radical_formation(root, &a, &a, &c);
    }
    check_radical_formation(RootSystem::E6, &unit(6, 0), &unit(6, 5), &[0; 6]);
}

#[test]
#[ignore = "large E8 channels; run in release"]
fn radical_formation_preserves_large_e8_entries() {
    let a = unit(8, 7);
    for b in [vec![0, 0, 0, 0, 0, 0, 0, 2], unit(8, 6)] {
        check_radical_formation(RootSystem::E8, &a, &b, &a);
    }
}

/// §10 nonzero counts for the multiplicity-2 channels, from the exact Python
/// prototype. §10 also bounds the radicands by 5-digit numerators and 7-digit
/// denominators; the extremes over both channels are pinned: numerator 33856
/// (`33856/268191`, in → 64) and denominator 7051968 (`18769/7051968`, in → 27).
/// The numerator 33856 is a tie (`33856/268191` and `33856/446985`);
/// `max_by_key` keeps the last maximum in entry order, which is the one pinned.
#[test]
fn g2_27x27_golden_counts() {
    let mut fam = GeneralFamily::new(RootSystem::G2);
    let mut radicands = Vec::new();
    for (c, count) in [(G27, 1354), (G64, 2904)] {
        let (ch, ma, mb, mc) = fam.channel(&G27, &G27, &c).unwrap();
        let e = ch.exact_entries(ma, mb, mc);
        assert_eq!(e.len(), count);
        radicands.extend(e.into_iter().map(|x| x.4.radicand().clone()));
    }
    let max_numer = radicands.iter().max_by_key(|r| r.numer().clone()).unwrap();
    let max_denom = radicands.iter().max_by_key(|r| r.denom().clone()).unwrap();
    assert!(radicands.contains(&q(33856, 446985)));
    assert_eq!(*max_numer, q(33856, 268191));
    assert_eq!(*max_denom, q(18769, 7051968));
}

// ---------------------------------------------------------------------------
// F/R gates and κ through frcore.
// ---------------------------------------------------------------------------

/// A test-only category over [`GeneralFamily`], so the generic residual
/// helpers of `category` can drive `frcore` (not a public surface).
#[derive(Clone)]
struct Cat {
    root: RootSystem,
    fam: Rc<RefCell<GeneralFamily>>,
}
impl Cat {
    fn new(root: RootSystem) -> Self {
        Self {
            root,
            fam: Rc::new(RefCell::new(GeneralFamily::new(root))),
        }
    }
}
impl PartialEq for Cat {
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root
    }
}
impl Eq for Cat {}
impl std::hash::Hash for Cat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.root.hash(state);
    }
}
impl std::fmt::Debug for Cat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cat({:?})", self.root)
    }
}
impl FusionCategory for Cat {
    type Object = Vec<i64>;
    type Scalar = f64;
    type FSymbol = FBlock<f64>;
    type RSymbol = RBlock<f64>;
    type Fusion<'a> = std::vec::IntoIter<Vec<i64>>;
    type Error = LieError;
    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Generic
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> Vec<i64> {
        vec![0; self.root.rank()]
    }
    fn dual(&self, a: &Vec<i64>) -> Vec<i64> {
        self.fam.borrow().data().dual(a).unwrap()
    }
    fn dim(&self, a: &Vec<i64>) -> f64 {
        u64::try_from(&self.fam.borrow().data().weyl_dimension(a).unwrap()).unwrap() as f64
    }
    fn frobenius_schur_phase(&self, a: &Vec<i64>) -> f64 {
        f64::from(self.fam.borrow_mut().frobenius_schur_phase(a).unwrap())
    }
    fn frobenius_schur_indicator(&self, a: &Vec<i64>) -> f64 {
        if &self.dual(a) == a {
            self.frobenius_schur_phase(a)
        } else {
            0.0
        }
    }
    fn twist(&self, _: &Vec<i64>) -> f64 {
        1.0
    }
    fn fusion<'a>(&'a self, a: &Vec<i64>, b: &Vec<i64>) -> Result<Self::Fusion<'a>, LieError> {
        let mut fam = self.fam.borrow_mut();
        Ok(fam
            .product(a, b)?
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter())
    }
    fn nsymbol(&self, a: &Vec<i64>, b: &Vec<i64>, c: &Vec<i64>) -> u32 {
        self.fam.borrow_mut().mult(a, b, c).unwrap() as u32
    }
    fn f_symbol(
        &self,
        a: &Vec<i64>,
        b: &Vec<i64>,
        c: &Vec<i64>,
        d: &Vec<i64>,
        e: &Vec<i64>,
        f: &Vec<i64>,
    ) -> Result<FBlock<f64>, LieError> {
        f_block_raw(&mut *self.fam.borrow_mut(), a, b, c, d, e, f)
    }
    fn r_symbol(&self, a: &Vec<i64>, b: &Vec<i64>, c: &Vec<i64>) -> Result<RBlock<f64>, LieError> {
        r_block_raw(&mut *self.fam.borrow_mut(), a, b, c)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"exceptional-cgc-test"
    }
}

const TOL: f64 = 1e-12;

fn timed(gate: &str, run: impl FnOnce() -> Result<f64, LieError>) -> f64 {
    let t = Instant::now();
    let r = run().unwrap();
    eprintln!("{gate} residual {r:e} in {:?}", t.elapsed());
    r
}

#[test]
fn g2_gates_through_frcore() {
    let cat = Cat::new(RootSystem::G2);
    let (v, adj) = (G7.to_vec(), G14.to_vec());
    assert!(timed("P", || pentagon_residual(&cat, &v, &v, &v, &v)) <= TOL);
    assert!(timed("H", || hexagon_residual(&cat, &v, &v, &v)) <= TOL);
    assert!(timed("H", || hexagon_residual(&cat, &v, &v, &adj)) <= TOL);
    assert!(timed("F", || f_unitarity_residual(&cat, &v, &v, &v, &v)) <= TOL);
    assert!(timed("F", || f_unitarity_residual(&cat, &v, &adj, &v, &adj)) <= TOL);
}

/// Gates through a multiplicity-2 vertex `27 ⊗ 27 → 27`.
#[test]
#[ignore = "release 14 s, debug 122 s"]
fn g2_multiplicity_two_gates_through_frcore() {
    let cat = Cat::new(RootSystem::G2);
    let (v, x) = (G7.to_vec(), G27.to_vec());
    assert_eq!(cat.nsymbol(&x, &x, &x), 2);
    assert!(timed("F", || f_unitarity_residual(&cat, &v, &x, &x, &x)) <= TOL);
    assert!(timed("H", || hexagon_residual(&cat, &x, &x, &v)) <= TOL);
    assert!(timed("P", || pentagon_residual(&cat, &v, &v, &x, &x)) <= TOL);
}

#[test]
#[ignore = "release 9 s"]
fn f4_26_unitarity_and_hexagon_through_frcore() {
    let f4 = Cat::new(RootSystem::F4);
    let v = unit(4, 3);
    assert!(timed("F", || f_unitarity_residual(&f4, &v, &v, &v, &v)) <= TOL);
    assert!(timed("H", || hexagon_residual(&f4, &v, &v, &v)) <= TOL);
}

#[test]
#[ignore = "release 18 s"]
fn e6_27_unitarity_and_hexagon_through_frcore() {
    let e6 = Cat::new(RootSystem::E6);
    let (v, w) = (unit(6, 0), unit(6, 5));
    assert!(timed("F", || f_unitarity_residual(&e6, &v, &w, &v, &v)) <= TOL);
    assert!(timed("H", || hexagon_residual(&e6, &v, &v, &w)) <= TOL);
}

/// `sign(d_a F^{a ā a}_a[𝟙, 𝟙])` through `frcore`, the §9.7 oracle for `κ_a`.
fn kappa_from_f(cat: &Cat, a: &[i64]) -> f64 {
    let a = a.to_vec();
    let dual = cat.dual(&a);
    let one = cat.unit();
    let f = cat.f_symbol(&a, &dual, &a, &a, &one, &one).unwrap();
    let x = f.as_slice()[0] * cat.dim(&a);
    assert!((x.abs() - 1.0).abs() <= 1e-12, "|d_a F| = 1, got {x}");
    x.signum()
}

/// §9.7 literals: G2 irreps real (+1); E7 56 pseudo-real (−1); E6 27, 78 and
/// both 351 pairs +1 with their duals.
#[test]
fn frobenius_schur_phase_literals() {
    let mut g2 = GeneralFamily::new(RootSystem::G2);
    for l in [G7, G14, G27, G64, G77, G77P] {
        assert_eq!(g2.frobenius_schur_phase(&l).unwrap(), 1, "{l:?}");
    }
    let mut e7 = GeneralFamily::new(RootSystem::E7);
    assert_eq!(e7.frobenius_schur_phase(&unit(7, 6)).unwrap(), -1);
    let mut e6 = GeneralFamily::new(RootSystem::E6);
    for l in [
        unit(6, 0),
        unit(6, 5),
        unit(6, 1),
        unit(6, 2),
        unit(6, 4),
        vec![2, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 2],
    ] {
        assert_eq!(e6.frobenius_schur_phase(&l).unwrap(), 1, "{l:?}");
    }
}

#[test]
fn frobenius_schur_phase_matches_f_sign() {
    let g2 = Cat::new(RootSystem::G2);
    for l in [G7, G14] {
        assert_eq!(kappa_from_f(&g2, &l), g2.frobenius_schur_phase(&l.to_vec()));
    }
    let e6 = Cat::new(RootSystem::E6);
    for l in [unit(6, 0), unit(6, 5)] {
        assert_eq!(kappa_from_f(&e6, &l), e6.frobenius_schur_phase(&l));
        assert_eq!(e6.frobenius_schur_indicator(&l), 0.0);
    }
}

#[test]
fn frobenius_schur_phase_matches_f_sign_e6_351_and_e7_56() {
    let e6 = Cat::new(RootSystem::E6);
    for l in [
        unit(6, 2),
        unit(6, 4),
        vec![2, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 2],
    ] {
        assert_eq!(kappa_from_f(&e6, &l), 1.0, "{l:?}");
    }
    let e7 = Cat::new(RootSystem::E7);
    assert_eq!(kappa_from_f(&e7, &unit(7, 6)), -1.0);
}

/// §10 resource report: build time and nonzero count per channel.
#[test]
#[ignore = "timing report; run in release with --nocapture"]
fn resource_report() {
    type Case = (RootSystem, Vec<i64>, Vec<i64>, Vec<Vec<i64>>);
    let cases: Vec<Case> = vec![
        (RootSystem::G2, G27.to_vec(), G27.to_vec(), Vec::new()),
        (RootSystem::F4, unit(4, 3), unit(4, 3), Vec::new()),
        (
            RootSystem::E7,
            unit(7, 6),
            unit(7, 6),
            vec![vec![0, 0, 0, 0, 0, 0, 2], unit(7, 5)],
        ),
    ];
    for (root, a, b, only) in cases {
        let mut fam = GeneralFamily::new(root);
        let t = Instant::now();
        for l in [&a, &b] {
            fam.module(l).unwrap();
        }
        println!("{root:?} {a:?} modules: {:?}", t.elapsed());
        let outcomes: Vec<Vec<i64>> = fam.product(&a, &b).unwrap().keys().cloned().collect();
        for c in outcomes {
            if !only.is_empty() && !only.contains(&c) {
                continue;
            }
            let t = Instant::now();
            fam.module(&c).unwrap();
            let tm = t.elapsed();
            let t = Instant::now();
            let (ch, ma, mb, mc) = fam.channel(&a, &b, &c).unwrap();
            let tc = t.elapsed();
            let entries = ch.exact_entries(ma, mb, mc);
            let max_den = entries
                .iter()
                .map(|e| e.4.radicand().clone())
                .max_by_key(|r| r.denom().clone());
            println!(
                "{root:?} {a:?}⊗{b:?}→{c:?} (dim {}, N {}): module {tm:?}, channel {tc:?}, nonzeros {}, max-denominator radicand {}",
                mc.dim(),
                ch.mult,
                entries.len(),
                max_den.map_or(String::new(), |r| r.to_string()),
            );
        }
    }
}

/// Charges follow retained entries (including policy metadata), and both
/// limits hold even when a single entry or dual pair cannot fit.
#[test]
fn memo_eviction_bounds_and_phase_pairs() {
    let mut fam = GeneralFamily::new(RootSystem::E6);
    let a = unit(6, 0);
    let dual = fam.data.dual(&a).unwrap();
    assert_eq!(fam.frobenius_schur_phase(&a).unwrap(), 1);
    let builds = fam.builds;
    // Drop modules, preserving the two completed signs.
    fam.trim(2, usize::MAX);
    assert!(fam.modules.is_empty());
    assert_eq!(fam.phases.len(), 1);
    let bytes = fam.memo_bytes();
    assert_eq!(bytes, fam.phases.values().map(|e| e.bytes).sum::<usize>());
    fam.trim(2, bytes);
    for l in [&a, &dual] {
        assert_eq!(fam.frobenius_schur_phase(l).unwrap(), 1);
    }
    assert_eq!(fam.builds, builds);
    // A byte short cannot retain just half of a pair.
    fam.trim(2, bytes - 1);
    assert_eq!((fam.memo_len(), fam.memo_bytes()), (0, 0));
    assert_eq!(fam.frobenius_schur_phase(&dual).unwrap(), 1);
    assert_eq!(fam.frobenius_schur_phase(&a).unwrap(), 1);
    assert_eq!(fam.builds, [2 * builds[0], 0]);
    fam.trim(1, usize::MAX);
    assert_eq!((fam.memo_len(), fam.memo_bytes()), (0, 0));

    let mut fam = GeneralFamily::new(RootSystem::G2);
    let (a, b) = (G7.to_vec(), G14.to_vec());
    let bits = |entries: Vec<MEntry>| {
        entries
            .into_iter()
            .map(|e| (e.m1, e.m2, e.m3, e.mu, e.value.to_bits()))
            .collect::<Vec<_>>()
    };
    let expected = bits(fam.cgc_entries(&a, &a, &b).unwrap());
    // A completed channel owns its data after all modules/products are gone.
    fam.trim(1, usize::MAX);
    assert_eq!(fam.channels.len(), 1);
    assert!(fam.modules.is_empty() && fam.products.is_empty());
    let entry = fam.channels.values().next().unwrap();
    assert_eq!(
        entry.bytes,
        3 * labels_bytes(&a)
            + entry.value.capacity() * std::mem::size_of::<MEntry>()
            + std::mem::size_of::<MemoEntry<Vec<MEntry>>>()
    );
    let bytes = fam.memo_bytes();
    assert_eq!(bytes, entry.bytes);
    fam.trim(1, bytes);
    let builds = fam.builds;
    assert_eq!(bits(fam.cgc_entries(&a, &a, &b).unwrap()), expected);
    assert_eq!(fam.builds, builds, "warm channel must not regenerate");
    fam.trim(1, bytes - 1);
    assert_eq!((fam.memo_len(), fam.memo_bytes()), (0, 0));
    assert_eq!(bits(fam.cgc_entries(&a, &a, &b).unwrap()), expected);
    fam.trim(0, 0);
    assert_eq!((fam.memo_len(), fam.memo_bytes()), (0, 0));
}

// Frozen pre-sparse algorithm: drift reference, not a mathematical oracle.
fn dense_echelon_nullspace(mut m: Vec<Vec<Q>>, cols: usize) -> (Vec<usize>, Vec<Vec<Q>>) {
    let mut pivot_of_row: Vec<usize> = Vec::new();
    let mut is_pivot = vec![false; cols];
    for col in (0..cols).rev() {
        let r0 = pivot_of_row.len();
        let Some(r) = (r0..m.len()).find(|&r| !m[r][col].is_zero()) else {
            continue;
        };
        m.swap(r0, r);
        let inv = m[r0][col].recip();
        for x in m[r0].iter_mut().filter(|x| !x.is_zero()) {
            *x *= &inv;
        }
        let pivot = m[r0].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r == r0 || row[col].is_zero() {
                continue;
            }
            let scale = row[col].clone();
            for (x, y) in row.iter_mut().zip(&pivot) {
                if !y.is_zero() {
                    *x -= &scale * y;
                }
            }
        }
        pivot_of_row.push(col);
        is_pivot[col] = true;
    }
    let free: Vec<usize> = (0..cols).filter(|&f| !is_pivot[f]).collect();
    let basis = free
        .iter()
        .map(|&f| {
            let mut v = vec![Q::zero(); cols];
            v[f] = Q::one();
            for (row, &pc) in m.iter().zip(&pivot_of_row) {
                if !row[f].is_zero() {
                    v[pc] = -row[f].clone();
                }
            }
            v
        })
        .collect();
    (free, basis)
}

fn compare_nullspace(data: &CartanData, a: &Ladders, b: &Ladders, top: &[i64], n: usize) {
    let p = ProductSpace::new(a, b, top);
    let mut rows: Vec<Vec<Q>> = Vec::new();
    for i in 0..top.len() {
        let nu: Vec<i64> = (0..top.len()).map(|k| top[k] + data.c(i, k)).collect();
        let target = ProductSpace::new(a, b, &nu);
        if target.pairs.is_empty() {
            continue;
        }
        let base = rows.len();
        rows.resize(base + target.len(), vec![Q::zero(); p.len()]);
        for (col, &(pa, pb)) in p.pairs.iter().enumerate() {
            for (p2, y) in &a.raise[i][pa] {
                rows[base + target.at((*p2, pb))][col] += y;
            }
            for (q2, y) in &b.raise[i][pb] {
                rows[base + target.at((pa, *q2))][col] += y;
            }
        }
    }
    let sparse = raising_rows(data, a, b, &p, top);
    let expected_rows: Vec<BTreeMap<usize, Q>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(col, x)| (col, x.clone()))
                .collect()
        })
        .collect();
    assert_eq!(sparse, expected_rows, "raising assembly at {top:?}");
    let expected = dense_echelon_nullspace(rows, p.len());
    assert_eq!(expected.1.len(), n);
    assert_eq!(
        echelon_nullspace(sparse, p.len()),
        expected,
        "rational basis at {top:?}"
    );
}

#[test]
fn sparse_nullspace_matches_dense_channels() {
    let mut count = 0;
    for (root, a, b) in [
        (RootSystem::G2, G7.to_vec(), G7.to_vec()),
        (RootSystem::G2, G7.to_vec(), G14.to_vec()),
        (RootSystem::G2, G14.to_vec(), G14.to_vec()),
        (RootSystem::G2, G27.to_vec(), G27.to_vec()),
        (RootSystem::F4, unit(4, 3), unit(4, 3)),
        (RootSystem::F4, unit(4, 3), unit(4, 0)),
        (RootSystem::E6, unit(6, 0), unit(6, 0)),
        (RootSystem::E6, unit(6, 0), unit(6, 5)),
        (RootSystem::E6, unit(6, 0), unit(6, 1)),
        (RootSystem::E7, unit(7, 6), unit(7, 6)),
        (RootSystem::E7, unit(7, 6), unit(7, 0)),
    ] {
        let mut fam = GeneralFamily::new(root);
        let outcomes = fam.product(&a, &b).unwrap().clone();
        fam.module(&a).unwrap();
        fam.module(&b).unwrap();
        for (c, n) in outcomes {
            compare_nullspace(
                &fam.data,
                &fam.modules[&a].value,
                &fam.modules[&b].value,
                &c,
                n as usize,
            );
            count += 1;
        }
    }
    eprintln!("Compared {count} exceptional channels exactly");
}

#[test]
#[ignore = "large E8 dense-reference comparisons; run in release"]
fn sparse_nullspace_matches_dense_e8_channels() {
    let data = CartanData::new(RootSystem::E8);
    let a = Ladders::new(&data, &unit(8, 7)).unwrap();
    for top in [vec![0; 8], unit(8, 7), unit(8, 0)] {
        compare_nullspace(&data, &a, &a, &top, 1);
    }
    for b in [vec![0, 0, 0, 0, 0, 0, 0, 2], unit(8, 6)] {
        let b = Ladders::new(&data, &b).unwrap();
        compare_nullspace(&data, &a, &b, &unit(8, 7), 1);
    }
}

#[test]
fn sparse_nullspace_degenerate_matrices() {
    for (rows, cols) in [(0, 0), (0, 3), (3, 0), (3, 4), (4, 3)] {
        for seed in 0..32 {
            let dense: Vec<Vec<Q>> = (0..rows)
                .map(|i| {
                    (0..cols)
                        .map(|j| q(((i * 13 + j * 7 + seed) % 5) as i64 - 2, (j + 1) as i64))
                        .collect()
                })
                .collect();
            let sparse = dense
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .filter(|(_, x)| !x.is_zero())
                        .map(|(j, x)| (j, x.clone()))
                        .collect()
                })
                .collect();
            assert_eq!(
                echelon_nullspace(sparse, cols),
                dense_echelon_nullspace(dense, cols)
            );
        }
    }
}

#[test]
#[ignore = "large E8 channel timing; run in release with --nocapture"]
fn performance_large_e8_channels() {
    let a = unit(8, 7);
    let samples: usize = std::env::var("FS_PERF_SAMPLES")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .unwrap();
    for b in [vec![0, 0, 0, 0, 0, 0, 0, 2], unit(8, 6)] {
        for _ in 0..samples {
            let mut fam = GeneralFamily::new(RootSystem::E8);
            let t = Instant::now();
            let entries = fam.cgc_entries(&a, &b, &a).unwrap();
            eprintln!(
                "CHANNEL {b:?}: {} entries in {:.9} s",
                entries.len(),
                t.elapsed().as_secs_f64()
            );
        }
    }
}

// The c236d39 product-space scan and descent are exact drift references.
fn old_product_space(a: &Ladders, b: &Ladders, nu: &[i64]) -> ProductSpace {
    let mut pairs = Vec::new();
    for wb in &b.module.weights {
        let mu_a: Vec<i64> = nu.iter().zip(&wb.mu).map(|(x, y)| x - y).collect();
        let Some(wa) = a.module.weight_index(&mu_a) else {
            continue;
        };
        let wa = &a.module.weights[wa];
        for q in wb.offset..wb.offset + wb.norms.len() {
            pairs.extend((wa.offset..wa.offset + wa.norms.len()).map(|p| (p, q)));
        }
    }
    let index = pairs.iter().enumerate().map(|(t, &pq)| (pq, t)).collect();
    ProductSpace { pairs, index }
}

fn compare_old_descent(a: &Ladders, b: &Ladders, c: &Ladders, channel: &Channel) {
    for (space, product) in c.module.weights.iter().zip(&channel.spaces) {
        let old = old_product_space(a, b, &space.mu);
        assert_eq!(product.pairs, old.pairs);
        assert_eq!(product.index, old.index);
    }
    let mut spaces: Vec<ProductSpace> = vec![old_product_space(a, b, &c.module.weights[0].mu)];
    let mut phi: Vec<Vec<Vec<Q>>> = Vec::with_capacity(c.dim());
    phi.push(channel.phi[0].clone());
    for space in &c.module.weights[1..] {
        let to = old_product_space(a, b, &space.mu);
        let lowered: Vec<Vec<Vec<Q>>> = space
            .kept
            .iter()
            .map(|&(j, q)| {
                let (src, _) = space.lower[j]
                    .as_ref()
                    .expect("a kept candidate's source weight space exists (§4.1)");
                let from = &spaces[*src];
                let m = c.module.weights[*src].offset + q;
                phi[m]
                    .iter()
                    .map(|v| coproduct(&a.lower[j], &b.lower[j], from, &to, v))
                    .collect()
            })
            .collect();
        for coeffs in &space.gs {
            let per_mu = (0..channel.mult)
                .map(|mu| {
                    let mut v = vec![Q::zero(); to.len()];
                    for (r, cand) in coeffs.iter().zip(&lowered) {
                        if r.is_zero() {
                            continue;
                        }
                        for (x, y) in v.iter_mut().zip(&cand[mu]) {
                            if !y.is_zero() {
                                *x += r * y;
                            }
                        }
                    }
                    v
                })
                .collect();
            phi.push(per_mu);
        }
        spaces.push(to);
    }
    assert_eq!(channel.phi, phi);
}

#[test]
fn descent_matches_old_path() {
    let data = CartanData::new(RootSystem::G2);
    let a = Ladders::new(&data, &G7).unwrap();
    let b = Ladders::new(&data, &G27).unwrap();
    let channel = Channel::new(&data, &b, &b, &b, 2).unwrap();
    compare_old_descent(&b, &b, &b, &channel);
    for (a, b) in [(&a, &b), (&b, &a), (&b, &b)] {
        for w in &b.module.weights {
            let new = ProductSpace::new(a, b, &w.mu);
            let old = old_product_space(a, b, &w.mu);
            assert_eq!(new.pairs, old.pairs);
            assert_eq!(new.index, old.index);
        }
    }
}

#[test]
#[ignore = "large E8 exact descent comparisons; run in release"]
fn large_descent_matches_old_path() {
    let data = CartanData::new(RootSystem::E8);
    let adjoint = Ladders::new(&data, &unit(8, 7)).unwrap();
    for label in [vec![0, 0, 0, 0, 0, 0, 0, 2], unit(8, 6)] {
        let large = Ladders::new(&data, &label).unwrap();
        for (a, b) in [(&adjoint, &large), (&large, &adjoint)] {
            let channel = Channel::new(&data, a, b, &adjoint, 1).unwrap();
            compare_old_descent(a, b, &adjoint, &channel);
        }
    }
}
