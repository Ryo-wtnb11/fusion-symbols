//! Independent oracles (`docs/gauge_cartan.md` §8, oracles 3, 5, 7–9): the
//! Brauer–Klimyk counts of oracle 3, exact isometry and intertwining checked with
//! a separately written coproduct, CGC entries derived by hand, the
//! pentagon/hexagon/unitarity gates through `frcore`, and `κ_a` against the
//! sign of `F^{a ā a}_a[𝟙, 𝟙]`.

use super::*;
use crate::category::{
    f_unitarity_residual, hexagon_residual, pentagon_residual, BraidingStyle, FBlock,
    FusionCategory, FusionStyle, RBlock, SymbolBlock,
};
use crate::frcore::{f_block_raw, r_block_raw};
use std::collections::BTreeMap;
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
    Q::from_integers(n.into(), d.into())
}

/// `x ↦ X x` with `X` from L2's stored `F` (a vector keyed by `(m_a, m_b)`).
type Vector = BTreeMap<(usize, usize), Q>;

fn add(v: &mut Vector, k: (usize, usize), x: Q) {
    let e = v.entry(k).or_insert_with(|| Q::from(0));
    *e += x;
    if e.numerator_ref() == &0 {
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
        if x.numerator_ref() != &0 {
            v.insert(pq, x.clone());
        }
    }
    v
}

fn weight(m: &Ladders, i: usize) -> &[i64] {
    &m.module.weights[m.space[i]].mu
}

/// §8 oracle 5 for one channel: in the rational coordinates,
/// `⟨Φ'_μ b_m, Φ'_ν b_m′⟩ = δ_μν δ_mm′ N_μ n^c_m` with `Φ' = √N Φ`,
/// `Δ(x) Φ' = Φ' x` for every `e_j` and `f_j`, and the weight support rule.
fn check_channel(fam: &GeneralFamily, a: &[i64], b: &[i64], c: &[i64], n: usize) {
    let (ch, ma, mb, mc) = fam.channel(a, b, c).unwrap();
    assert_eq!(ch.mult, n, "§4.2 N gate for {a:?}⊗{b:?}→{c:?}");
    check_exact(&ch, &ma, &mb, &mc);
}

/// The exact identities of [`check_channel`] on a built channel, for any
/// basis of the three modules.
pub(crate) fn check_exact(ch: &Channel, ma: &Ladders, mb: &Ladders, mc: &Ladders) {
    let n = ch.mult;
    let (a, b, c) = (
        &ma.module.weights[0].mu,
        &mb.module.weights[0].mu,
        &mc.module.weights[0].mu,
    );
    let phi: Vec<Vec<Vector>> = (0..mc.dim())
        .map(|m| (0..n).map(|mu| as_vector(ch, mc, m, mu)).collect())
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
                        .fold(Q::from(0), |acc, x| acc + x);
                    let want = if m == m2 && mu == nu {
                        &ch.seed_norms[mu] * &mc.norm[m]
                    } else {
                        Q::from(0)
                    };
                    assert_eq!(got, want);
                }
            }
        }
        // Intertwining for every generator, including candidates §3.2.3 dropped.
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
    let fam = GeneralFamily::new(root);
    let outcomes: Vec<(Vec<i64>, u32)> = fam
        .product(a, b)
        .unwrap()
        .iter()
        .map(|(c, n)| (c.dynkin().to_vec(), *n))
        .collect();
    for (c, n) in outcomes {
        check_channel(&fam, a, b, &c, n as usize);
    }
}

/// §8 oracle 3, literal decompositions with dimensions from the spec.
fn decomposition(root: RootSystem, a: &[i64], b: &[i64]) -> Vec<(u64, u32)> {
    let fam = GeneralFamily::new(root);
    let product: Vec<(Vec<i64>, u32)> = fam
        .product(a, b)
        .unwrap()
        .iter()
        .map(|(c, n)| (c.dynkin().to_vec(), *n))
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

/// §8 oracle 5 for every channel of G2 7⊗7, 7⊗14 and 14⊗14 (the N gate runs inside
/// `Channel::new`).
#[test]
fn g2_small_products_are_exact_isometric_intertwiners() {
    check_product(RootSystem::G2, &G7, &G7);
    check_product(RootSystem::G2, &G7, &G14);
    check_product(RootSystem::G2, &G14, &G14);
}

/// §8 oracle 5 for the multiplicity-2 channels G2 27⊗27 → 27, 64.
#[test]
fn g2_27x27_multiplicity_two_is_exact() {
    let fam = GeneralFamily::new(RootSystem::G2);
    for c in [G27, G64] {
        check_channel(&fam, &G27, &G27, &c, 2);
    }
}

/// §4.3 example: in G2 27⊗27 → 27 the pivots are `[0, 1]`, `Ψ_μ[p_{μ+1}] > 0`,
/// `Ψ_0[p_2] = 0` and `Ψ_1[p_1] ≠ 0`.
#[test]
fn g2_27x27_to_27_pivot_rule() {
    let fam = GeneralFamily::new(RootSystem::G2);
    let (ch, ..) = fam.channel(&G27, &G27, &G27).unwrap();
    assert_eq!(ch.pivots, [0, 1]);
    let psi = &ch.phi[0];
    assert!(psi[0][0] > 0 && psi[1][1] > 0);
    assert!(psi[0][1].numerator_ref() == &0);
    assert!(psi[1][0].numerator_ref() != &0);
}

/// §8 oracle 5 on every channel of E6 `27 ⊗ 27̄ = 1 + 78 + 650`.
#[test]
fn e6_27x27bar_is_exact() {
    let (a, b) = (unit(6, 0), unit(6, 5));
    assert_eq!(
        decomposition(RootSystem::E6, &a, &b),
        [(1, 1), (78, 1), (650, 1)]
    );
    check_product(RootSystem::E6, &a, &b);
}

/// §8 oracle 5 on every channel of E6 `27 ⊗ 27 = 27̄ + 351 + 351′`.
#[test]
fn e6_27x27_is_exact() {
    let v = unit(6, 0);
    assert_eq!(
        decomposition(RootSystem::E6, &v, &v),
        [(27, 1), (351, 1), (351, 1)]
    );
    check_product(RootSystem::E6, &v, &v);
}

/// §8 oracle 5 on E6 `27 ⊗ 78 → 27`.
#[test]
fn e6_27x78_to_27_is_exact() {
    let fam = GeneralFamily::new(RootSystem::E6);
    check_channel(&fam, &unit(6, 0), &unit(6, 1), &unit(6, 0), 1);
}

/// §8 oracle 5 on the non-square F4 channel `26 ⊗ 52 → 26`.
#[test]
fn f4_26x52_to_26_is_exact() {
    let fam = GeneralFamily::new(RootSystem::F4);
    check_channel(&fam, &unit(4, 3), &unit(4, 0), &unit(4, 3), 1);
}

/// §8 oracle 5 on every channel of F4 `26 ⊗ 26 = 1 + 26 + 52 + 273 + 324`.
#[test]
fn f4_26x26_is_exact() {
    let v = unit(4, 3);
    assert_eq!(
        decomposition(RootSystem::F4, &v, &v),
        [(1, 1), (26, 1), (52, 1), (273, 1), (324, 1)]
    );
    check_product(RootSystem::F4, &v, &v);
}

/// §8 oracle 5 on every channel of E7 `56 ⊗ 56 = 1 + 133 + 1463 + 1539`.
#[test]
fn e7_56x56_is_exact() {
    let v = unit(7, 6);
    assert_eq!(
        decomposition(RootSystem::E7, &v, &v),
        [(1, 1), (133, 1), (1463, 1), (1539, 1)]
    );
    check_product(RootSystem::E7, &v, &v);
}

/// §8 oracle 5 on the cheap channels of E8 `248 ⊗ 248`: `→ 1` and `→ 248`.
#[test]
fn e8_248x248_to_1_and_248_is_exact() {
    let v = unit(8, 7);
    let fam = GeneralFamily::new(RootSystem::E8);
    check_channel(&fam, &v, &v, &[0; 8], 1);
    check_channel(&fam, &v, &v, &v, 1);
}

/// §8 oracle 5 on E8 `248 ⊗ 248 → 3875`.
#[test]
#[ignore = "release 1.2 s, debug 9.2 s"]
fn e8_248x248_to_3875_is_exact() {
    let v = unit(8, 7);
    let fam = GeneralFamily::new(RootSystem::E8);
    check_channel(&fam, &v, &v, &unit(8, 0), 1);
}

/// Gate 2 in both directions: a declared multiplicity above or below the
/// exact nullity, including a declared `N = 0` for a present channel and
/// `N = 1` for an absent one (G2 has `7 ⊗ 7 ↛ 64`).
#[test]
fn nullity_gate_rejects_a_wrong_multiplicity() {
    let data = CartanData::new(RootSystem::G2);
    let m7 = Ladders::new(&data, &G7).unwrap();
    let m14 = Ladders::new(&data, &G14).unwrap();
    let m64 = Ladders::new(&data, &G64).unwrap();
    for (c, l, nullity, multiplicity) in [(&m14, G14, 1, 2), (&m14, G14, 1, 0), (&m64, G64, 0, 1)] {
        let err = Channel::new(&data, &m7, &m7, c, multiplicity)
            .err()
            .unwrap();
        assert_eq!(
            err,
            LieError::from(CgcError::NullityMismatch {
                a: G7.to_vec(),
                b: G7.to_vec(),
                c: l.to_vec(),
                nullity,
                multiplicity
            })
        );
    }
}

/// Gate 3: a vanishing principal ladder is a typed failure, not a sign.
#[test]
fn principal_ladder_gate_rejects_a_zero_ladder() {
    let data = CartanData::new(RootSystem::E6);
    let lambda = unit(6, 0);
    let mut l = Ladders::new(&data, &lambda).unwrap();
    assert_eq!(l.principal_sign(&lambda), Ok(1));
    let low = l.dim() - 1;
    for op in &mut l.lower {
        for col in op.iter_mut() {
            col.retain(|&(t, _)| t != low);
        }
    }
    assert_eq!(
        l.principal_sign(&lambda),
        Err(CgcError::ZeroPrincipalLadder { lambda }.into())
    );
}

/// The pairs of `a ⊗ b` with weight sum `nu`, enumerated over public indices
/// in a-slow order and returned at engine indices.
fn public_fibre(a: &Ladders, b: &Ladders, nu: &[i64]) -> Vec<(usize, usize)> {
    let (ea, eb) = (a.engine(), b.engine());
    let mut want = Vec::new();
    for &p in &ea {
        for &q in &eb {
            let sum: Vec<i64> = weight(a, p)
                .iter()
                .zip(weight(b, q))
                .map(|(x, y)| x + y)
                .collect();
            if sum == nu {
                want.push((p, q));
            }
        }
    }
    want
}

/// Every weight of `a ⊗ b` reached from `a`'s highest weight, then every
/// weight of `b` shifted down from `b`'s: both repeated and simple fibres.
fn fibre_weights(a: &Ladders, b: &Ladders) -> Vec<Vec<i64>> {
    let add = |x: &[i64], y: &[i64]| -> Vec<i64> { x.iter().zip(y).map(|(u, v)| u + v).collect() };
    let mut out: Vec<Vec<i64>> = (b.module.weights.iter())
        .map(|w| add(&w.mu, &a.module.weights[0].mu))
        .collect();
    out.extend((a.module.weights.iter()).map(|w| add(&w.mu, &b.module.weights.last().unwrap().mu)));
    out
}

/// The product states of a weight are exactly the pairs of that weight sum
/// in a-slow public order, and row arithmetic finds each at its position,
/// for the Gram–Schmidt (G2, B2) and Gelfand–Tsetlin (A2, A3) bases, unequal
/// and swapped factors, and weight spaces of multiplicity above one.
#[test]
fn product_space_is_the_weight_fibre_in_a_slow_order() {
    let cases: [(RootSystem, &[i64], &[i64]); 4] = [
        (RootSystem::G2, &G7, &G27),
        (RootSystem::B(2), &[1, 1], &[0, 2]),
        (RootSystem::A(2), &[2, 1], &[1, 1]),
        (RootSystem::A(3), &[1, 0, 1], &[0, 1, 1]),
    ];
    for (root, x, y) in cases {
        let data = CartanData::new(root);
        let mx = Ladders::new(&data, x).unwrap();
        let my = Ladders::new(&data, y).unwrap();
        let repeated = |m: &Ladders| m.module.weights.iter().any(|w| w.norms.len() > 1);
        assert!(repeated(&mx) || repeated(&my), "{root:?}");
        // GT public indices are not the engine (depth) order, so the a-slow
        // sort and the public ranks are exercised; GS public is the identity.
        let identity = |m: &Ladders| m.public.iter().enumerate().all(|(i, &p)| p as usize == i);
        let gt = matches!(root, RootSystem::A(_));
        assert_eq!(identity(&mx) && identity(&my), !gt, "{root:?}");
        for (a, b) in [(&mx, &my), (&my, &mx), (&my, &my)] {
            let mut scratch = RowScratch::new(a);
            for nu in fibre_weights(a, b) {
                let want = public_fibre(a, b, &nu);
                let got = ProductSpace::new(a, b, &nu);
                assert_eq!(got.pairs, want, "{root:?} {nu:?}");
                let rows = scratch.rows(&got, b);
                for (t, &pq) in want.iter().enumerate() {
                    assert_eq!(rows.at(pq), t, "{root:?} {nu:?}");
                }
            }
        }
    }
}

/// A pair outside the bound target fails loudly even when its left state
/// keeps a stale, in-range row start from an earlier target.
#[test]
#[should_panic(expected = "is not a coupling pair of the target weight")]
fn row_arithmetic_rejects_an_off_weight_pair() {
    let data = CartanData::new(RootSystem::A(2));
    let a = Ladders::new(&data, &[2, 1]).unwrap();
    let mut scratch = RowScratch::new(&a);
    let source = ProductSpace::new(&a, &a, &[4, 2]);
    let stale = (0, 0);
    assert_eq!(source.pairs.first(), Some(&stale));
    let _ = scratch.rows(&source, &a);
    let target = ProductSpace::new(&a, &a, &[2, -3]);
    // Left state 0 is absent, so its row start is not rewritten, and the
    // stale index lands inside the target on a different pair.
    assert!(target.pairs.iter().all(|&(p, _)| p != stale.0));
    let t = scratch.0[stale.0] + a.rank[stale.1] as usize;
    assert_eq!((t, target.len(), target.pairs[t]), (0, 10, (14, 1)));
    let _ = scratch.rows(&target, &a).at(stale);
}

/// A channel addresses its product spaces by row arithmetic alone: one
/// row-start write per left state of each bound target (the raising targets
/// of §4.2 and every descent space of §4.4), never a scan of all `d_a` slots.
#[test]
fn channel_writes_one_row_start_per_left_state() {
    let distinct = |s: &ProductSpace| s.pairs.chunk_by(|x, y| x.0 == y.0).count();
    for (root, x, y, z, n) in [
        (RootSystem::A(2), &[1, 1][..], &[1, 1][..], &[1, 1][..], 2),
        (RootSystem::G2, &G27[..], &G27[..], &G27[..], 2),
    ] {
        let data = CartanData::new(root);
        let (a, b, c) = (
            Ladders::new(&data, x).unwrap(),
            Ladders::new(&data, y).unwrap(),
            Ladders::new(&data, z).unwrap(),
        );
        ROW_WRITES.with(|w| w.set(0));
        let ch = Channel::new(&data, &a, &b, &c, n).unwrap();
        let top = &c.module.weights[0].mu;
        let raising: usize = (0..top.len())
            .map(|i| {
                let nu: Vec<i64> = (0..top.len()).map(|k| top[k] + data.c(i, k)).collect();
                distinct(&ProductSpace::new(&a, &b, &nu))
            })
            .sum();
        let descent: usize = ch.spaces[1..].iter().map(distinct).sum();
        assert_eq!(ROW_WRITES.with(|w| w.get()), raising + descent, "{root:?}");
        assert!(ch.spaces.iter().all(|s| distinct(s) <= s.len()));
    }
}

/// G2 7 (§3.2 by hand, Cartan rows α₀ = (2,−1), α₁ = (−3,2)): weights
/// m0 (1,0), m1 (−1,1), m2 (2,−1), m3 (0,0), m4 (−2,1), m5 (1,−1), m6 (−1,0),
/// one each, with norms 1, 1, 1, 2, 4, 4, 4. In the orthonormal basis
/// `f₀: u0→u1 (1), u2→u3 (√2), u3→u4 (√2), u5→u6 (1)` and
/// `f₁: u1→u2 (1), u4→u5 (1)`.
///
/// 7⊗7 → 7: P = [(0,3), (1,2), (2,1), (3,0)] (a-slow, §4.1).
/// `Δ(e₀) x = 0` gives `x₁ + √2 x₂ = 0`, `√2 x₃ + x₄ = 0`; `Δ(e₁) x = 0`
/// gives `x₂ + x₃ = 0`. So `Ψ = (1, −√2, √2, −1)/√6`, pivot row 0.
/// Lowering by `Δ(f₀)` (m_c = 1 is `f₀ u0` with entry 1):
/// `(√2 u0⊗u4 − u1⊗u3 + u3⊗u1 − √2 u4⊗u0)/√6`.
#[test]
fn g2_7x7_to_7_hand_entries_and_order() {
    let fam = GeneralFamily::new(RootSystem::G2);
    let (ch, ma, mb, mc) = fam.channel(&G7, &G7, &G7).unwrap();
    let got: Vec<_> = ch
        .exact_entries(&ma, &mb, &mc)
        .into_iter()
        .filter(|e| e.2 <= 1)
        .map(|(p, r, m, mu, x)| (p, r, m, mu, x.signed_square()))
        .collect();
    let want = vec![
        (0, 3, 0, 0, q(1, 6)),
        (1, 2, 0, 0, q(-1, 3)),
        (2, 1, 0, 0, q(1, 3)),
        (3, 0, 0, 0, q(-1, 6)),
        (0, 4, 1, 0, q(1, 3)),
        (1, 3, 1, 0, q(-1, 6)),
        (3, 1, 1, 0, q(1, 6)),
        (4, 0, 1, 0, q(-1, 3)),
    ];
    assert_eq!(got, want);
}

/// §4.7 order on a multiplicity-2 channel: `μ`, then `m_c`, then `(m_a, m_b)`.
#[test]
fn entry_order_is_mu_then_mc_then_a_slow() {
    let fam = GeneralFamily::new(RootSystem::G2);
    let e = fam.cgc_entries(&G14, &G14, &G27).unwrap();
    let keys: Vec<_> = e.iter().map(|x| (x.mu, x.m3, x.m1, x.m2)).collect();
    assert!(keys.windows(2).all(|w| w[0] < w[1]));
    assert!(e.iter().all(|x| x.value != 0.0));
}

/// §4.5: `1 ⊗ s → s` and `s ⊗ 1 → s` are exactly the identity.
#[test]
fn trivial_couplings_are_the_identity() {
    let fam = GeneralFamily::new(RootSystem::G2);
    let one = [0, 0];
    for s in [G7, G14, G27] {
        for (a, b) in [(&one[..], &s[..]), (&s[..], &one[..])] {
            let (ch, ma, mb, mc) = fam.channel(a, b, &s).unwrap();
            let got: Vec<_> = ch.exact_entries(&ma, &mb, &mc);
            assert_eq!(got.len(), mc.dim());
            for (m, (p, r, mcx, mu, x)) in got.into_iter().enumerate() {
                assert_eq!((p + r, mcx, mu), (m, m, 0));
                assert_eq!(x.signed_square(), Q::from(1));
            }
        }
    }
}

/// §8 oracle 9 within one process: two independent families give the same bits.
#[test]
fn regeneration_is_byte_identical() {
    let bits = || {
        let fam = GeneralFamily::new(RootSystem::G2);
        let mut out = Vec::new();
        for c in [G7, G14, G27, [0, 0]] {
            for e in fam.cgc_entries(&G7, &G7, &c).unwrap().iter() {
                out.push((e.m1, e.m2, e.m3, e.mu, e.value.to_bits()));
            }
        }
        out
    };
    assert_eq!(bits(), bits());
}

/// Entry-level drift check against the canonical-rational formula (§4.4,
/// §4.7) through the public exact type: every ordered `(m_a, m_b, m_c, μ)`
/// tuple, its exact value, and the `to_bits()` of the published rounded value,
/// including the `A_1` vertex sign applied once (§4.6).
fn check_radical_formation(root: RootSystem, a: &[i64], b: &[i64], c: &[i64]) {
    let fam = GeneralFamily::new(root);
    let (ch, ma, mb, mc) = fam.channel(a, b, c).unwrap();
    let mut want = Vec::new();
    for mu in 0..ch.mult {
        for m in mc.engine() {
            let scale = &ch.seed_norms[mu] * &mc.norm[m];
            for (&(pa, pb), x) in ch.spaces[mc.space[m]].pairs.iter().zip(&ch.phi[m][mu]) {
                if x.numerator_ref() == &0 {
                    continue;
                }
                let old = SignedSqrtRational::from_prefactor_radical(
                    x.clone(),
                    &ma.norm[pa] * &mb.norm[pb] / &scale,
                );
                let (pa, pb, pc) = (ma.public[pa], mb.public[pb], mc.public[m]);
                want.push((pa as usize, pb as usize, pc as usize, mu, old));
            }
        }
    }
    assert!(ch.exact_entries(&ma, &mb, &mc) == want);

    let odd = root == RootSystem::A(1) && (a[0] + b[0] - c[0]) / 2 % 2 != 0;
    let want: Vec<_> = want
        .iter()
        .map(|(pa, pb, pc, mu, old)| {
            let v = old.to_f64();
            let v = if odd { -v } else { v };
            (*pa as u32, *pb as u32, *pc as u32, *mu as u32, v.to_bits())
        })
        .collect();
    let got: Vec<_> = fam
        .cgc_entries(a, b, c)
        .unwrap()
        .iter()
        .map(|e| (e.m1, e.m2, e.m3, e.mu, e.value.to_bits()))
        .collect();
    assert!(got == want);
    eprintln!(
        "{root:?} {a:?}⊗{b:?}→{c:?}: N = {}, {} entries agree",
        ch.mult,
        want.len()
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
    // GT bases (A_r), with N = 2 at A_2 (1,1)³ and A_3 [1,0,1]³.
    for (a, b, c) in [
        ([1, 1], [1, 1], [1, 1]),
        ([2, 0], [0, 2], [1, 1]),
        ([2, 1], [1, 2], [2, 2]),
    ] {
        check_radical_formation(RootSystem::A(2), &a, &b, &c);
    }
    let adj = [1, 0, 1];
    check_radical_formation(RootSystem::A(3), &adj, &adj, &adj);
    // A_1: odd and even vertex signs.
    for (a, b, c) in [(1, 1, 0), (2, 2, 2), (2, 1, 1), (1, 1, 2), (3, 2, 3)] {
        check_radical_formation(RootSystem::A(1), &[a], &[b], &[c]);
    }
}

/// Round-only formation saves exactly the canonical temporary's one
/// allocation per nonzero entry, with the arithmetic and the bits unchanged
/// (GS and GT bases, N = 2).
#[cfg(feature = "lie-generated")]
#[test]
fn round_only_formation_skips_the_exact_temporary() {
    use super::super::rational::{radical, rounded_radical};
    let calls = crate::audit_alloc::thread_allocation_calls;
    for (root, a, b, c) in [
        (RootSystem::G2, G27.to_vec(), G27.to_vec(), G64.to_vec()),
        (RootSystem::A(2), vec![2, 1], vec![1, 2], vec![2, 2]),
    ] {
        let fam = GeneralFamily::new(root);
        let (ch, ma, mb, mc) = fam.channel(&a, &b, &c).unwrap();
        let mut nonzero = 0;
        for mu in 0..ch.mult {
            for (m, per_mu) in ch.phi.iter().enumerate() {
                let scale = &ch.seed_norms[mu] * &mc.norm[m];
                for (&(pa, pb), x) in ch.spaces[mc.space[m]].pairs.iter().zip(&per_mu[mu]) {
                    if x.numerator_ref() == &0 {
                        continue;
                    }
                    let (na, nb) = (&ma.norm[pa], &mb.norm[pb]);
                    let start = calls();
                    let old = radical(x, na, nb, &scale).to_f64();
                    let mid = calls();
                    let new = rounded_radical(x, na, nb, &scale);
                    let end = calls();
                    assert_eq!(old.to_bits(), new.to_bits());
                    assert_eq!(mid - start, end - mid + 1);
                    nonzero += 1;
                }
            }
        }
        assert!(nonzero > 0);
    }
}

#[test]
#[ignore = "large E8 channels; run in release"]
fn radical_formation_preserves_large_e8_entries() {
    let a = unit(8, 7);
    for b in [vec![0, 0, 0, 0, 0, 0, 0, 2], unit(8, 6)] {
        check_radical_formation(RootSystem::E8, &a, &b, &a);
    }
}

/// §9 nonzero counts for the multiplicity-2 channels, from the exact Python
/// prototype. §9 also bounds the radicands by 5-digit numerators and 7-digit
/// denominators; the extremes over both channels are pinned: numerator 33856
/// (`33856/268191`, in → 64) and denominator 7051968 (`18769/7051968`, in → 27).
/// The numerator 33856 is a tie (`33856/268191` and `33856/446985`);
/// `max_by_key` keeps the last maximum in entry order, which is the one pinned.
#[test]
fn g2_27x27_golden_counts() {
    let fam = GeneralFamily::new(RootSystem::G2);
    let mut radicands = Vec::new();
    for (c, count) in [(G27, 1354), (G64, 2904)] {
        let (ch, ma, mb, mc) = fam.channel(&G27, &G27, &c).unwrap();
        let e = ch.exact_entries(&ma, &mb, &mc);
        assert_eq!(e.len(), count);
        radicands.extend(e.into_iter().map(|x| x.4.radicand().clone()));
    }
    let max_numer = radicands
        .iter()
        .max_by_key(|r| r.numerator_ref().clone())
        .unwrap();
    let max_denom = radicands
        .iter()
        .max_by_key(|r| r.denominator_ref().clone())
        .unwrap();
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
    fam: GeneralFamily,
}
impl Cat {
    fn new(root: RootSystem) -> Self {
        Self {
            root,
            fam: GeneralFamily::new(root),
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
        self.fam.data().dual(a).unwrap()
    }
    fn dim(&self, a: &Vec<i64>) -> f64 {
        u64::try_from(&self.fam.data().weyl_dimension(a).unwrap()).unwrap() as f64
    }
    fn frobenius_schur_phase(&self, a: &Vec<i64>) -> Result<f64, LieError> {
        self.fam
            .frobenius_schur_phase(a, &self.dual(a))
            .map(f64::from)
    }
    fn frobenius_schur_indicator(&self, a: &Vec<i64>) -> f64 {
        if &self.dual(a) == a {
            self.frobenius_schur_phase(a).unwrap()
        } else {
            0.0
        }
    }
    fn twist(&self, _: &Vec<i64>) -> f64 {
        1.0
    }
    fn fusion<'a>(&'a self, a: &Vec<i64>, b: &Vec<i64>) -> Result<Self::Fusion<'a>, LieError> {
        Ok(self
            .fam
            .product(a, b)?
            .iter()
            .map(|(c, _)| c.dynkin().to_vec())
            .collect::<Vec<_>>()
            .into_iter())
    }
    fn nsymbol(&self, a: &Vec<i64>, b: &Vec<i64>, c: &Vec<i64>) -> Result<u32, LieError> {
        Ok(self.fam.mult(a, b, c)? as u32)
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
        f_block_raw(&self.fam, [a, b, c, d, e, f].map(|x| x.as_slice()))
    }
    fn r_symbol(&self, a: &Vec<i64>, b: &Vec<i64>, c: &Vec<i64>) -> Result<RBlock<f64>, LieError> {
        r_block_raw(&self.fam, [a.as_slice(), b, c])
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
    assert_eq!(cat.nsymbol(&x, &x, &x).unwrap(), 2);
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

/// `sign(d_a F^{a ā a}_a[𝟙, 𝟙])` through `frcore`, the §8 oracle 8 oracle for `κ_a`.
fn kappa_from_f(cat: &Cat, a: &[i64]) -> f64 {
    let a = a.to_vec();
    let dual = cat.dual(&a);
    let one = cat.unit();
    let f = cat.f_symbol(&a, &dual, &a, &a, &one, &one).unwrap();
    let x = f.as_slice()[0] * cat.dim(&a);
    assert!((x.abs() - 1.0).abs() <= 1e-12, "|d_a F| = 1, got {x}");
    x.signum()
}

/// §8 oracle 8 literals: G2 irreps real (+1); E7 56 pseudo-real (−1); E6 27, 78 and
/// both 351 pairs +1 with their duals.
fn kappa(fam: &GeneralFamily, l: &[i64]) -> i8 {
    fam.frobenius_schur_phase(l, &fam.data().dual(l).unwrap())
        .unwrap()
}

#[test]
fn frobenius_schur_phase_literals() {
    let g2 = GeneralFamily::new(RootSystem::G2);
    for l in [G7, G14, G27, G64, G77, G77P] {
        assert_eq!(kappa(&g2, &l), 1, "{l:?}");
    }
    let e7 = GeneralFamily::new(RootSystem::E7);
    assert_eq!(kappa(&e7, &unit(7, 6)), -1);
    let e6 = GeneralFamily::new(RootSystem::E6);
    for l in [
        unit(6, 0),
        unit(6, 5),
        unit(6, 1),
        unit(6, 2),
        unit(6, 4),
        vec![2, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 2],
    ] {
        assert_eq!(kappa(&e6, &l), 1, "{l:?}");
    }
}

#[test]
fn frobenius_schur_phase_matches_f_sign() {
    let g2 = Cat::new(RootSystem::G2);
    for l in [G7, G14] {
        assert_eq!(
            kappa_from_f(&g2, &l),
            g2.frobenius_schur_phase(&l.to_vec()).unwrap()
        );
    }
    let e6 = Cat::new(RootSystem::E6);
    for l in [unit(6, 0), unit(6, 5)] {
        assert_eq!(kappa_from_f(&e6, &l), e6.frobenius_schur_phase(&l).unwrap());
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

/// §9 resource report: build time and nonzero count per channel.
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
        let fam = GeneralFamily::new(root);
        let t = Instant::now();
        for l in [&a, &b] {
            fam.module(l).unwrap();
        }
        println!("{root:?} {a:?} modules: {:?}", t.elapsed());
        let outcomes: Vec<Vec<i64>> = fam
            .product(&a, &b)
            .unwrap()
            .iter()
            .map(|(c, _)| c.dynkin().to_vec())
            .collect();
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
            let entries = ch.exact_entries(&ma, &mb, &mc);
            let max_den = entries
                .iter()
                .map(|e| e.4.radicand().clone())
                .max_by_key(|r| r.denominator_ref().clone());
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

// Frozen pre-sparse algorithm: drift reference, not a mathematical oracle.
fn dense_echelon_nullspace(mut m: Vec<Vec<Q>>, cols: usize) -> (Vec<usize>, Vec<Vec<Q>>) {
    let mut pivot_of_row: Vec<usize> = Vec::new();
    let mut is_pivot = vec![false; cols];
    for col in (0..cols).rev() {
        let r0 = pivot_of_row.len();
        let Some(r) = (r0..m.len()).find(|&r| m[r][col].numerator_ref() != &0) else {
            continue;
        };
        m.swap(r0, r);
        let inv = Q::from(1) / &m[r0][col];
        for x in m[r0].iter_mut().filter(|x| x.numerator_ref() != &0) {
            *x *= &inv;
        }
        let pivot = m[r0].clone();
        for (r, row) in m.iter_mut().enumerate() {
            if r == r0 || row[col].numerator_ref() == &0 {
                continue;
            }
            let scale = row[col].clone();
            for (x, y) in row.iter_mut().zip(&pivot) {
                if y.numerator_ref() != &0 {
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
            let mut v = vec![Q::from(0); cols];
            v[f] = Q::from(1);
            for (row, &pc) in m.iter().zip(&pivot_of_row) {
                if row[f].numerator_ref() != &0 {
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
        rows.resize(base + target.len(), vec![Q::from(0); p.len()]);
        // A linear search, independent of the engine's row arithmetic.
        let at = |pq: (usize, usize)| target.pairs.iter().position(|&x| x == pq).unwrap();
        for (col, &(pa, pb)) in p.pairs.iter().enumerate() {
            for (p2, y) in &a.raise[i][pa] {
                rows[base + at((*p2, pb))][col] += y;
            }
            for (q2, y) in &b.raise[i][pb] {
                rows[base + at((pa, *q2))][col] += y;
            }
        }
    }
    let sparse = raising_rows(data, a, b, &p, top, &mut RowScratch::new(a));
    let expected_rows: Vec<BTreeMap<usize, Q>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|(_, x)| x.numerator_ref() != &0)
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
        let fam = GeneralFamily::new(root);
        let outcomes = fam.product(&a, &b).unwrap().clone();
        let (ma, mb) = (fam.module(&a).unwrap(), fam.module(&b).unwrap());
        for (c, n) in outcomes.iter() {
            compare_nullspace(&fam.data, &ma, &mb, c.dynkin(), *n as usize);
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
                        .filter(|(_, x)| x.numerator_ref() != &0)
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
            let fam = GeneralFamily::new(RootSystem::E8);
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

/// §8 oracle 5 on large E8 channels `248 ⊗ λ → 248` and `λ ⊗ 248 → 248` (release).
#[test]
#[ignore = "large E8 exact channels; run in release"]
fn large_e8_channels_are_exact() {
    let data = CartanData::new(RootSystem::E8);
    let adjoint = Ladders::new(&data, &unit(8, 7)).unwrap();
    for label in [vec![0, 0, 0, 0, 0, 0, 0, 2], unit(8, 6)] {
        let large = Ladders::new(&data, &label).unwrap();
        for (a, b) in [(&adjoint, &large), (&large, &adjoint)] {
            check_exact(
                &Channel::new(&data, a, b, &adjoint, 1).unwrap(),
                a,
                b,
                &adjoint,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Rounded gates.
// ---------------------------------------------------------------------------

/// A channel's rounded entries before the gates, with its modules.
fn rounded(
    root: RootSystem,
    a: &[i64],
    b: &[i64],
    c: &[i64],
) -> (Vec<MEntry>, [Ladders; 3], usize) {
    let data = CartanData::new(root);
    let [ma, mb, mc] = [a, b, c].map(|l| Ladders::new(&data, l).unwrap());
    let n = data.multiplicities(a, b, None).unwrap().multiplicity(c) as usize;
    let ch = Channel::new(&data, &ma, &mb, &mc, n).unwrap();
    let entries = ch.rounded_entries(&ma, &mb, &mc).unwrap();
    (entries, [ma, mb, mc], n)
}

fn gate(root: RootSystem, l: &[Ladders; 3], n: usize, e: &[MEntry]) -> Result<(), GateError> {
    check_rounded(&CartanData::new(root), &l[0], &l[1], &l[2], n, e)
}

/// Every channel of the exactly verified products passes both rounded gates.
#[test]
fn rounded_gates_pass_on_exact_channels() {
    for (root, a, b) in [
        (RootSystem::G2, G7.to_vec(), G7.to_vec()),
        (RootSystem::G2, G7.to_vec(), G14.to_vec()),
        (RootSystem::G2, G27.to_vec(), G27.to_vec()),
        (RootSystem::F4, unit(4, 3), unit(4, 3)),
        (RootSystem::E6, unit(6, 0), unit(6, 5)),
    ] {
        let fam = GeneralFamily::new(root);
        for (c, _) in fam.product(&a, &b).unwrap().iter() {
            let (e, l, n) = rounded(root, &a, &b, c.dynkin());
            gate(root, &l, n, &e).unwrap();
        }
    }
}

/// The isometry gate rejects a rescaled column, a missing column, two
/// parallel columns within one weight block, an entry off its weight and
/// nonfinite entries.
#[test]
fn isometry_gate_rejects_broken_channels() {
    let (good, l, n) = rounded(RootSystem::G2, &G27, &G27, &G27);
    assert_eq!(n, 2);
    let isometry = |e: &[MEntry]| match gate(RootSystem::G2, &l, n, e) {
        Err(GateError::Residual {
            gate: "isometry",
            residual,
        }) => residual,
        other => panic!("{other:?}"),
    };
    let m = 3u32;
    // A column norm of 1.01² instead of 1.
    let mut e = good.clone();
    e.iter_mut()
        .filter(|x| (x.m3, x.mu) == (m, 0))
        .for_each(|x| x.value *= 1.01);
    assert!((isometry(&e) - 0.0201).abs() < 1e-6);
    // A whole column absent: its diagonal Gram entry is 0.
    let e: Vec<_> = good
        .iter()
        .copied()
        .filter(|x| (x.m3, x.mu) != (m, 1))
        .collect();
    assert!((isometry(&e) - 1.0).abs() < 1e-12);
    // Column (m, 1) replaced by column (m, 0): same weight block, overlap 1.
    let mut e: Vec<_> = good
        .iter()
        .copied()
        .filter(|x| (x.m3, x.mu) != (m, 1))
        .collect();
    e.extend(
        good.iter()
            .filter(|x| (x.m3, x.mu) == (m, 0))
            .map(|x| MEntry { mu: 1, ..*x }),
    );
    assert!((isometry(&e) - 1.0).abs() < 1e-12);
    // An entry moved onto a pair of the wrong weight.
    let mut e = good.clone();
    e[0].m1 = if e[0].m1 == 0 { 1 } else { 0 };
    assert_eq!(gate(RootSystem::G2, &l, n, &e), Err(GateError::Support));
    // An index out of range.
    let mut e = good.clone();
    e[0].mu = 2;
    assert_eq!(gate(RootSystem::G2, &l, n, &e), Err(GateError::Support));
    // Finite entries whose Gram accumulation overflows.
    let mut e = good.clone();
    e[5].value = 1e200;
    assert_eq!(
        gate(RootSystem::G2, &l, n, &e),
        Err(GateError::NonFinite { gate: "isometry" })
    );
    for bad in [f64::NAN, f64::INFINITY] {
        let mut e = good.clone();
        e[5].value = bad;
        assert_eq!(
            gate(RootSystem::G2, &l, n, &e),
            Err(GateError::NonFinite { gate: "isometry" })
        );
    }
}

/// The ladder gate checks every applicable generator: flipping the sign of
/// the coupled state `f_j u_top` keeps the channel isometric but breaks
/// `Δ(f_j) C_top = Σ_m ⟨u_m, f_j u_top⟩ C_m`, for `j = 0` and for `j = 1`.
#[test]
fn ladder_gate_rejects_a_broken_first_descent() {
    let (good, l, n) = rounded(RootSystem::G2, &G7, &G14, &G64);
    gate(RootSystem::G2, &l, n, &good).unwrap();
    let top = &l[2].module.weights[0].mu;
    for j in 0..2 {
        let nu: Vec<i64> = (0..2)
            .map(|i| top[i] - CartanData::new(RootSystem::G2).c(j, i))
            .collect();
        let w = l[2].module.weight_index(&nu).unwrap();
        let m = l[2].module.weights[w].offset as u32;
        let mut e = good.clone();
        e.iter_mut()
            .filter(|x| x.m3 == m)
            .for_each(|x| x.value = -x.value);
        match gate(RootSystem::G2, &l, n, &e) {
            Err(GateError::Residual {
                gate: "ladder",
                residual,
            }) => assert!(residual > 0.1),
            other => panic!("j = {j}: {other:?}"),
        }
    }
}

/// A channel failing a gate is never retained, its error is a typed
/// generation failure, and a valid retry builds and retains it once. The
/// channel is used by no other test, so no concurrent test publishes it.
#[test]
fn failed_gate_is_not_retained_and_retry_succeeds() {
    // Resetting tests hold this lock; warm-build checks must not overlap them.
    let _guard = crate::cache::TEST_CACHE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fam = GeneralFamily::new(RootSystem::G2);
    let (a, b, c) = (G7.to_vec(), G77P.to_vec(), vec![2, 1]);
    assert_eq!(fam.mult(&a, &b, &c).unwrap(), 1);
    let resident = || crate::cache::cgc_channel_is_resident(RootSystem::G2, [&a, &b, &c]);
    CORRUPT.with(|x| x.set(Some(|e| e[0].value = f64::NAN)));
    let err = fam.cgc_entries(&a, &b, &c).err();
    CORRUPT.with(|x| x.set(None));
    let err = err.unwrap();
    assert_eq!(
        err,
        LieError::from(GateError::NonFinite { gate: "isometry" })
    );
    assert!(matches!(err, LieError::Generation(_)));
    assert!(!resident());
    let good = fam.cgc_entries(&a, &b, &c).unwrap();
    assert!(resident());
    let builds = BUILDS.with(std::cell::Cell::get);
    let again = fam.cgc_entries(&a, &b, &c).unwrap();
    assert_eq!(
        BUILDS.with(std::cell::Cell::get),
        builds,
        "warm channel must not regenerate"
    );
    assert!(Arc::ptr_eq(&again, &good));
}

/// Gates 6 and 7 run on every completed F and R block before it is retained.
/// The labels are used by no other test.
#[test]
fn failed_block_gates_are_not_retained() {
    let fam = GeneralFamily::new(RootSystem::C(3));
    let v: &[i64] = &[1, 0, 0];
    let sym: &[i64] = &[2, 0, 0];
    let f = [v, v, v, v, sym, sym];
    let r = [v, v, sym];
    CORRUPT_BLOCK.with(|x| x.set(Some(|x| x * 2.0)));
    let bad_r = fam.r_symbol(r);
    CORRUPT_BLOCK.with(|x| x.set(Some(|_| f64::INFINITY)));
    let bad_f_inf = fam.f_symbol(f);
    CORRUPT_BLOCK.with(|x| x.set(None));
    assert!(matches!(
        bad_r,
        Err(LieError::Generation(ref e)) if e.to_string().contains("R gate: residual")
    ));
    assert_eq!(
        bad_f_inf,
        Err(LieError::from(GateError::NonFinite { gate: "F" }))
    );
    let good_r = fam.r_symbol(r).unwrap();
    assert_eq!(good_r.as_slice(), [1.0]);
    let good_f = fam.f_symbol(f).unwrap();
    assert!(good_f
        .as_slice()
        .iter()
        .all(|x| x.is_finite() && x.abs() <= 1.0));
}

/// §8 oracle 5 for the Gelfand–Tsetlin basis and the classical
/// Gram–Schmidt bases: every channel is an exact isometric intertwiner with
/// weight support, including A(1), the B spinor and vector, the D3 adjoint
/// (`N = 2`) and the D4 half-spinor.
#[test]
fn gt_and_classical_products_are_exact_isometric_intertwiners() {
    use RootSystem::{A, B, C, D};
    for (root, a, b) in [
        (A(1), &[2][..], &[3][..]),
        (A(2), &[1, 1], &[1, 1]),
        (A(3), &[1, 0, 1], &[0, 1, 0]),
        (B(2), &[0, 1], &[0, 1]),
        (B(2), &[1, 0], &[1, 0]),
        (C(2), &[1, 1], &[1, 1]),
        (D(3), &[0, 1, 1], &[0, 1, 1]),
        (D(4), &[0, 0, 0, 1], &[0, 0, 0, 1]),
    ] {
        check_product(root, a, b);
    }
}

/// §8 oracle 5 on the `N = 3` channel Sp(4) `35 ⊗ 35 → 35`.
#[test]
fn c2_35x35_multiplicity_three_is_exact() {
    let fam = GeneralFamily::new(RootSystem::C(2));
    check_channel(&fam, &[2, 1], &[2, 1], &[2, 1], 3);
}

/// One channel request builds each distinct module label at most once, even
/// when the CGC tier retains nothing (a reset before every load stands in for
/// a zero budget): `λ ⊗ 𝟙 → λ`, `a ⊗ a → c`, and a zero-multiplicity channel
/// whose weight occurs, so the exact nullity check and the channel share the
/// modules. Concurrent tests can only turn a build into a hit, hence `≤`.
/// The bits equal those of a request with retention.
#[test]
fn channel_request_builds_each_module_label_once() {
    let _guard = crate::cache::TEST_CACHE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fam = GeneralFamily::new(RootSystem::G2);
    let bits = |a: &[i64], b: &[i64], c: &[i64]| {
        let ch = fam.cgc_entries(a, b, c).unwrap();
        (
            ch.mult,
            ch.iter().map(|e| e.value.to_bits()).collect::<Vec<_>>(),
        )
    };
    let builds = || BUILDS.with(std::cell::Cell::get);
    for ([a, b, c], distinct, mult) in [
        ([G27, [0, 0], G27], 2, 1),
        ([G7, G7, G14], 2, 1),
        ([G7, [0, 0], [0, 0]], 2, 0),
    ] {
        crate::cache::reset();
        BEFORE_LOAD.with(|h| h.set(Some(crate::cache::reset)));
        let before = builds();
        let rejected = bits(&a, &b, &c);
        let after = builds();
        BEFORE_LOAD.with(|h| h.set(None));
        assert!(after[0] - before[0] <= distinct, "{a:?} {b:?} {c:?}");
        crate::cache::reset();
        let retained = bits(&a, &b, &c);
        assert_eq!(retained.0, mult);
        assert_eq!(rejected, retained);
    }
    crate::cache::reset();
}
