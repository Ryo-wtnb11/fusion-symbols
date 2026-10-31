//! Golden gauge values: committed bit patterns of CGC, F, R, A and B for every
//! simple root family (`docs/gauge_cartan.md` §8 oracle 9; SU(2) closed forms
//! `docs/gauge.md` §1), compared **bitwise**.
//!
//! Each family pins at least one channel with outer multiplicity `N ≥ 2`
//! (A(1) is multiplicity free). Release-only cases are `#[ignore]`d and run by
//! the CI job `bit_identity`, except the E8 `N ≥ 2` golden
//! (`e8_n2_golden`), a manual pre-release gate on a machine with at least
//! 48 GB.
//!
//! Golden values are not an independent oracle of the mathematics: the exact
//! identities are checked in `src/lie/general/cgc/tests.rs`, and the external
//! references (SUNRepresentations.jl 0.4.0, the closed-form SU(2) CGC) are
//! compared within a tolerance in the oracle tests below. A golden change is a
//! specification change (`docs/gauge.md`, "Status"). Print the current tables
//! with
//!
//! ```text
//! cargo run --release --all-features --example print_golden
//! ```

// The golden values are literals on purpose.
#![allow(clippy::approx_constant)]

use fusion_symbols::category::{CgcBlock, FusionCategory, GroupRepresentations, SymbolBlock};
use fusion_symbols::lie::{GroupId, LieGroup, LieIrrep};

/// One pinned case: the group, the channel `a ⊗ b → c` whose CGC is pinned,
/// and how many nonzero entries of each multiplicity copy (in §4.7 entry
/// order, over the first two coupled states) are pinned.
pub struct Case {
    pub name: &'static str,
    group: fn() -> GroupId,
    a: &'static [i64],
    b: &'static [i64],
    c: &'static [i64],
    cgc_entries: usize,
}

/// `(key, bits)`: `C μ,m_c,m_a,m_b` for a CGC entry, `C μ nnz` for the
/// nonzero count of copy `μ` over `m_c < 2`, `F i`, `R i`, `A i`, `B i` for the row-major block entries.
type Golden = &'static [(&'static str, u64)];

/// The pinned CGC rows of `a ⊗ b → c`: for every multiplicity copy `μ`, the
/// nonzero count over the first two coupled states (the slice `m_c = 0` that
/// F and R read, and one lowered state) and its first `k` nonzero entries.
fn cgc_rows(
    g: &LieGroup,
    a: &LieIrrep,
    b: &LieIrrep,
    c: &LieIrrep,
    k: usize,
) -> Vec<(String, u64)> {
    let cgc = g.cgc(a, b, c).unwrap();
    let [d1, d2, d3, n] = cgc.dims();
    assert!(n >= 1, "absent channel");
    let mut out = Vec::new();
    for mu in 0..n {
        let mut nonzero = Vec::new();
        for m3 in 0..d3.min(2) {
            for m1 in 0..d1 {
                for m2 in 0..d2 {
                    let x = cgc.get([m1, m2, m3, mu]);
                    if x != 0.0 {
                        nonzero.push((format!("C {mu},{m3},{m1},{m2}"), x.to_bits()));
                    }
                }
            }
        }
        out.push((format!("C {mu} nnz"), nonzero.len() as u64));
        out.extend(nonzero.into_iter().take(k));
    }
    out
}

/// Appends the row-major entries of one block as `name i`.
fn block_rows(out: &mut Vec<(String, u64)>, name: &str, values: &[f64]) {
    for (i, x) in values.iter().enumerate() {
        out.push((format!("{name} {i}"), x.to_bits()));
    }
}

/// The rows of `case`: its CGC entries, the F blocks of the B- and A-symbol
/// sextets, `F^{a b b̄}_a[c, 𝟙]` and `F^{ā a b}_b[𝟙, c]`, and `R^{ab}_c`,
/// `A^{ab}_c` and `B^{ab}_c`.
pub fn rows(case: &Case) -> Vec<(String, u64)> {
    let g = LieGroup::new((case.group)()).unwrap();
    let [a, b, c] = [case.a, case.b, case.c].map(|l| g.irrep(l).unwrap());
    let one = g.unit();
    let mut out = cgc_rows(&g, &a, &b, &c, case.cgc_entries);
    let bar = |x| g.dual(x);
    let f1 = g.f_symbol(&a, &b, &bar(&b), &a, &c, &one).unwrap();
    block_rows(&mut out, "F", f1.as_slice());
    let f2 = g.f_symbol(&bar(&a), &a, &b, &b, &one, &c).unwrap();
    block_rows(&mut out, "F", f2.as_slice());
    block_rows(&mut out, "R", g.r_symbol(&a, &b, &c).unwrap().as_slice());
    block_rows(&mut out, "A", g.a_symbol(&a, &b, &c).unwrap().as_slice());
    block_rows(&mut out, "B", g.b_symbol(&a, &b, &c).unwrap().as_slice());
    out
}

#[track_caller]
fn check(case: &Case, golden: Golden) {
    compare(case.name, &rows(case), golden);
}

#[track_caller]
fn compare(name: &str, got: &[(String, u64)], golden: Golden) {
    let want: Vec<(String, u64)> = golden.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    if got != want.as_slice() {
        // The whole table, so a deliberate gauge change can be reviewed.
        for (k, v) in got {
            eprintln!("    (\"{k}\", 0x{v:016x}),");
        }
    }
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(
            g,
            w,
            "gauge drift in {name}: got {:e}, want {:e}. A value change needs a \
             docs/gauge*.md edit and a CHANGELOG breaking-change entry in the \
             same PR (docs/gauge.md, Status).",
            f64::from_bits(g.1),
            f64::from_bits(w.1),
        );
    }
    assert_eq!(got.len(), want.len(), "{name}: row count");
}

macro_rules! cases {
    ($($test:ident $(#[$attr:meta])* : $name:literal, $group:expr, $a:expr, $b:expr, $c:expr, $k:expr, $golden:ident;)+) => {
        pub const CASES: &[Case] = &[$(Case {
            name: $name,
            group: $group,
            a: &$a,
            b: &$b,
            c: &$c,
            cgc_entries: $k,
        }),+];
        $(
            #[test]
            $(#[$attr])*
            fn $test() {
                let case = CASES.iter().find(|c| c.name == $name).unwrap();
                check(case, $golden);
            }
        )+
    };
}

fn su2() -> GroupId {
    GroupId::su(2).unwrap()
}

#[cfg(feature = "lie-generated")]
mod generated {
    use fusion_symbols::lie::GroupId;
    pub fn su3() -> GroupId {
        GroupId::su(3).unwrap()
    }
    pub fn spin5() -> GroupId {
        GroupId::spin(5).unwrap()
    }
    pub fn sp4() -> GroupId {
        GroupId::sp(4).unwrap()
    }
    pub fn so6() -> GroupId {
        GroupId::so(6).unwrap()
    }
    pub fn g2() -> GroupId {
        GroupId::g2()
    }
    pub fn f4() -> GroupId {
        GroupId::f4()
    }
    pub fn e6() -> GroupId {
        GroupId::e6()
    }
    pub fn e7() -> GroupId {
        GroupId::e7()
    }
    pub fn e8() -> GroupId {
        GroupId::e8()
    }
}
#[cfg(feature = "lie-generated")]
use generated::*;

#[cfg(not(feature = "lie-generated"))]
cases! {
    a1_golden: "A1 2x2->2", su2, [2], [2], [2], 8, A1;
}

#[cfg(feature = "lie-generated")]
cases! {
    a1_golden: "A1 2x2->2", su2, [2], [2], [2], 8, A1;
    a2_golden: "A2 8x8->8 N=2", su3, [1, 1], [1, 1], [1, 1], 8, A2;
    b2_golden: "B2 16x16->10 N=2", spin5, [1, 1], [1, 1], [0, 2], 8, B2;
    c2_golden: "C2 16x16->10 N=2", sp4, [1, 1], [1, 1], [2, 0], 8, C2;
    d3_golden: "D3 15x15->15 N=2", so6, [0, 1, 1], [0, 1, 1], [0, 1, 1], 8, D3;
    g2_golden: "G2 27x27->27 N=2", g2, [2, 0], [2, 0], [2, 0], 8, G2;
    f4_golden: "F4 26x26->26", f4, [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], 8, F4;
    f4_n2_golden #[ignore = "release golden"]: "F4 273x273->273 N=2", f4, [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], 8, F4_N2;
    e6_golden: "E6 27x27bar->78", e6, [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], 8, E6;
    e6_n2_golden #[ignore = "release golden"]: "E6 78x650->650 N=2", e6, [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], 8, E6_N2;
    e7_golden: "E7 56x56->133", e7, [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0], 8, E7;
    e7_n2_golden #[ignore = "release golden"]: "E7 1539x1539->1539 N=2", e7, [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0], 8, E7_N2;
    e8_golden #[ignore = "release golden"]: "E8 248x248->248", e8, [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 1], 8, E8;
}

/// The multiplicity of every `N ≥ 2` golden channel, certified twice: the
/// literal below must equal the Brauer–Klimyk N, which satisfies the sum rule
/// `Σ_c N d_c = d_a d_b`, and generating the golden CGC checks it against the
/// exact highest-weight nullity (gate 2).
#[cfg(feature = "lie-generated")]
#[test]
fn n2_golden_channels_are_certified() {
    for (name, n) in [
        ("A2 8x8->8 N=2", 2),
        ("B2 16x16->10 N=2", 2),
        ("C2 16x16->10 N=2", 2),
        ("D3 15x15->15 N=2", 2),
        ("G2 27x27->27 N=2", 2),
        ("F4 273x273->273 N=2", 2),
        ("E6 78x650->650 N=2", 2),
        ("E7 1539x1539->1539 N=2", 2),
    ] {
        let case = CASES.iter().find(|c| c.name == name).unwrap();
        let g = LieGroup::new((case.group)()).unwrap();
        let [a, b, c] = [case.a, case.b, case.c].map(|l| g.irrep(l).unwrap());
        assert_eq!(g.nsymbol(&a, &b, &c).unwrap(), n, "{name}");
        let total: f64 = g
            .fusion(&a, &b)
            .unwrap()
            .map(|x| f64::from(g.nsymbol(&a, &b, &x).unwrap()) * g.dim(&x))
            .sum();
        assert_eq!(total, g.dim(&a) * g.dim(&b), "{name}: sum rule");
    }
}

/// E8 `3875 ⊗ 30380 → 30380` (`[1,0,0,0,0,0,0,0] ⊗ [0,0,0,0,0,0,1,0]`,
/// `N = 2`), the smallest E8 channel with `N ≥ 2`: the first four nonzero
/// entries of both multiplicity copies and the `2 × 2` block `R^{ab}_c`, which
/// mixes the two copies.
///
/// A manual pre-release gate on a machine with at least 48 GB, not run by CI
/// (`docs/gauge_cartan.md` §8, oracle 9): R builds both orders of the channel,
/// about 19 minutes and 31 GB peak RSS in total on Apple M-series. The literals are
/// the values of the gated engine (generation gates 1–7 of §7 pass on both
/// channels and on the R block), recorded once; like every golden they pin the
/// gauge rather than check it against an independent oracle.
#[cfg(feature = "lie-generated")]
#[test]
#[ignore = "manual pre-release gate: about 19 minutes and 31 GB peak RSS"]
fn e8_n2_golden() {
    let g = LieGroup::new(GroupId::e8()).unwrap();
    let [a, b] = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]].map(|l| g.irrep(&l).unwrap());
    assert_eq!(g.nsymbol(&a, &b, &b).unwrap(), 2);
    let mut got = cgc_rows(&g, &a, &b, &b, 4);
    block_rows(&mut got, "R", g.r_symbol(&a, &b, &b).unwrap().as_slice());
    compare("E8 3875x30380->30380 N=2", &got, E8_N2);
}

#[cfg(feature = "lie-generated")]
const E8_N2: Golden = &[
    ("C 0 nnz", 0x00000000000008d8),
    ("C 0,0,0,8445", 0x3fa74cba458bf1e9),
    ("C 0,0,0,8447", 0xbf89ad22204e93e3),
    ("C 0,0,0,8448", 0xbfa4011e21112d80),
    ("C 0,0,1,7760", 0xbfa74cba458bf1e9),
    ("C 1 nnz", 0x00000000000009c5),
    ("C 1,0,0,8445", 0xbf61fbf67cc02003),
    ("C 1,0,0,8446", 0x3faa30f1497fed73),
    ("C 1,0,0,8447", 0xbf9f6e8cd1e5ae3d),
    ("C 1,0,0,8448", 0x3f992eb4b01144c4),
    ("R 0", 0x3fee6d4931b08a14),
    ("R 1", 0xbfd3d1cf8ee4965b),
    ("R 2", 0x3fd3d1cf8ee49628),
    ("R 3", 0x3fee6d4931b08a31),
];

// ---------------------------------------------------------------------------
// A(1): Condon–Shortley signs and order (§4.6), from exact signed squares.
// ---------------------------------------------------------------------------

/// Literal Condon–Shortley CGC at `[m_a, m_b, m_c, μ]`, axis index `m + j`:
/// `½ ⊗ ½ → 0` is `(|↑↓⟩ − |↓↑⟩)/√2`, `½ ⊗ ½ → 1` at `m = 1` is `|↑↑⟩`, and
/// `1 ⊗ 1 → 1` at `m = 1` is `(|1,0⟩ − |0,1⟩)/√2`.
#[test]
fn a1_condon_shortley() {
    let g = LieGroup::new(su2()).unwrap();
    let ir = |dj| g.irrep(&[dj]).unwrap();
    let half = 0.5f64.sqrt();
    let singlet = g.cgc(&ir(1), &ir(1), &ir(0)).unwrap();
    assert_eq!(singlet.get([0, 1, 0, 0]).to_bits(), (-half).to_bits());
    assert_eq!(singlet.get([1, 0, 0, 0]).to_bits(), half.to_bits());
    assert_eq!(singlet.get([0, 0, 0, 0]), 0.0);
    let triplet = g.cgc(&ir(1), &ir(1), &ir(2)).unwrap();
    assert_eq!(triplet.get([1, 1, 2, 0]), 1.0);
    let one = g.cgc(&ir(2), &ir(2), &ir(2)).unwrap();
    assert_eq!(one.get([1, 2, 2, 0]).to_bits(), (-half).to_bits());
    assert_eq!(one.get([2, 1, 2, 0]).to_bits(), half.to_bits());
}

// ---------------------------------------------------------------------------
// SU(N): external-reference oracles within 1e-12 (§8 oracle 6).
// ---------------------------------------------------------------------------

#[cfg(feature = "lie-generated")]
mod sun_oracle {
    use super::*;

    const TOL: f64 = 1e-12;

    #[track_caller]
    fn close(what: &str, got: f64, want: f64) {
        let err = (got - want).abs();
        assert!(
            err <= TOL,
            "{what}: got={got:.17e} want={want:.17e} err={err:e}"
        );
    }

    /// `(m1, m2, m3, mu, value)` at GT pattern indices.
    type Row = (usize, usize, usize, usize, f64);

    fn check_rows(n: usize, a: &[i64], b: &[i64], c: &[i64], rows: &[Row]) {
        let g = LieGroup::new(GroupId::su(n).unwrap()).unwrap();
        let [a, b, c] = [a, b, c].map(|l| g.irrep(l).unwrap());
        let cgc = g.cgc(&a, &b, &c).unwrap();
        for &(m1, m2, m3, mu, want) in rows {
            close(
                &format!(
                    "SU({n}) {:?} [{m1},{m2},{m3},{mu}]",
                    c.dynkin().collect::<Vec<_>>()
                ),
                cgc.get([m1, m2, m3, mu]),
                want,
            );
        }
    }

    /// SU(3) `8 ⊗ 8 → 8` and `3 ⊗ 3̄ → 8`, checked against
    /// SUNRepresentations.jl 0.4.0, including the `μ` order of the
    /// multiplicity-2 vertex.
    #[test]
    fn su3_cgc_matches_sunrepresentations() {
        check_rows(
            3,
            &[1, 1],
            &[1, 1],
            &[1, 1],
            &[
                (0, 2, 0, 0, 0.534_522_483_824_848_6),
                (0, 2, 0, 1, -0.119_522_860_933_439_46),
                (0, 4, 0, 1, 0.483_045_891_539_647_8),
                (0, 5, 1, 1, 0.683_130_051_063_973_6),
                (0, 6, 3, 0, 0.654_653_670_707_976_9),
                (0, 7, 2, 0, -0.267_261_241_912_424_4),
            ],
        );
        check_rows(
            3,
            &[1, 0],
            &[0, 1],
            &[1, 1],
            &[
                (0, 0, 0, 0, 0.999_999_999_999_999_8),
                (0, 2, 2, 0, 0.816_496_580_927_726_1),
                (1, 1, 2, 0, 0.408_248_290_463_863_1),
                (1, 1, 4, 0, 0.707_106_781_186_547_4),
            ],
        );
    }

    /// `F^{3,3,3}_1[3̄, 3̄]`, checked against SUNRepresentations.jl 0.4.0.
    #[test]
    fn su3_f_symbol_matches_sunrepresentations() {
        let g = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        let [three, bar, one] = [[1, 0], [0, 1], [0, 0]].map(|l| g.irrep(&l).unwrap());
        let block = g
            .f_symbol(&three, &three, &three, &one, &bar, &bar)
            .unwrap();
        assert_eq!(block.dims(), [1, 1, 1, 1]);
        close(
            "SU(3) F 3,3,3->1",
            block.get([0, 0, 0, 0]),
            0.999_999_999_999_999_9,
        );
    }

    /// SU(4) `15 ⊗ 15 → 15` and SU(6) `35 ⊗ 35 → 35` (both `N = 2`),
    /// checked against SUNRepresentations.jl 0.4.0 `CGC(Float64, a, a, a)`
    /// (Julia 1.11.6) for the partitions `(2,1,1,0)` and `(2,1,1,1,1,0)`;
    /// its 1-based `[m1, m2, m3, μ]` is these indices plus one.
    #[test]
    fn su4_su6_cgc_match_sunrepresentations() {
        check_rows(
            4,
            &[1, 0, 1],
            &[1, 0, 1],
            &[1, 0, 1],
            &[
                (0, 3, 0, 1, -0.089_087_080_637_474_64),
                (0, 6, 0, 1, 0.220_479_275_922_049_6),
                (0, 8, 0, 1, -0.381_881_307_912_985_8),
                (0, 3, 0, 0, 0.462_910_049_886_275_5),
                (0, 12, 4, 0, 0.534_522_483_824_848_6),
                (0, 13, 7, 0, 0.534_522_483_824_848_9),
            ],
        );
        check_rows(
            6,
            &[1, 0, 0, 0, 1],
            &[1, 0, 0, 0, 1],
            &[1, 0, 0, 0, 1],
            &[
                (0, 5, 0, 1, -0.054_232_614_454_663_035),
                (0, 10, 0, 1, 0.094_096_581_588_635_82),
                (0, 14, 0, 1, -0.121_478_164_475_943_77),
                (0, 5, 0, 0, 0.383_482_494_423_684_76),
                (0, 30, 6, 0, 0.420_084_025_208_402_6),
                (0, 31, 11, 0, 0.420_084_025_208_402_66),
            ],
        );
    }
}

#[rustfmt::skip]
const A1: Golden = &[
    ("C 0 nnz", 0x0000000000000004),
    ("C 0,0,0,1", 0xbfe6a09e667f3bcd),
    ("C 0,0,1,0", 0x3fe6a09e667f3bcd),
    ("C 0,1,0,2", 0xbfe6a09e667f3bcd),
    ("C 0,1,2,0", 0x3fe6a09e667f3bcd),
    ("F 0", 0xbfe279a74590331c),
    ("F 0", 0xbfe279a74590331c),
    ("R 0", 0xbff0000000000000),
    ("A 0", 0xbfefffffffffffff),
    ("B 0", 0xbfefffffffffffff),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const A2: Golden = &[
    ("C 0 nnz", 0x0000000000000008),
    ("C 0,0,0,2", 0x3fe11acee560242a),
    ("C 0,0,2,0", 0xbfd11acee560242a),
    ("C 0,0,3,1", 0xbfe4f2ec413cb52b),
    ("C 0,0,4,0", 0x3fdda05179501504),
    ("C 0,1,1,2", 0x3fe11acee560242a),
    ("C 0,1,2,1", 0xbfd11acee560242a),
    ("C 0,1,4,1", 0xbfdda05179501504),
    ("C 0,1,5,0", 0x3fe4f2ec413cb52b),
    ("C 1 nnz", 0x000000000000000c),
    ("C 1,0,0,2", 0xbfbe990cdad55ed2),
    ("C 1,0,0,4", 0x3fdeea3950a8511e),
    ("C 1,0,1,3", 0xbfe5dc338d8781a3),
    ("C 1,0,2,0", 0x3fde990cdad55ed2),
    ("C 1,0,3,1", 0xbfc8fba8a1bf6f96),
    ("C 1,0,4,0", 0x3fc1aa69e4f2777f),
    ("C 1,1,0,5", 0x3fe5dc338d8781a3),
    ("C 1,1,1,2", 0xbfbe990cdad55ed2),
    ("F 0", 0x3fd6a09e667f3bce),
    ("F 1", 0x0000000000000000),
    ("F 2", 0x0000000000000000),
    ("F 3", 0x3fd6a09e667f3bcd),
    ("F 0", 0x3fd6a09e667f3bce),
    ("F 1", 0xbc50000000000000),
    ("F 2", 0xbc50000000000000),
    ("F 3", 0x3fd6a09e667f3bcd),
    ("R 0", 0xbfd2492492492493),
    ("R 1", 0x3feeaa83e629111f),
    ("R 2", 0x3feeaa83e6291120),
    ("R 3", 0x3fd2492492492492),
    ("A 0", 0x3ff0000000000002),
    ("A 1", 0xbc66a09e667f3bce),
    ("A 2", 0xbc66a09e667f3bce),
    ("A 3", 0x3ff0000000000001),
    ("B 0", 0x3ff0000000000002),
    ("B 1", 0x0000000000000000),
    ("B 2", 0x0000000000000000),
    ("B 3", 0x3ff0000000000001),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const B2: Golden = &[
    ("C 0 nnz", 0x0000000000000016),
    ("C 0,0,0,9", 0x3fd186f174f88472),
    ("C 0,0,1,7", 0xbfde5b9d136c6d96),
    ("C 0,0,2,5", 0xbfd186f174f88472),
    ("C 0,0,3,3", 0x3fde5b9d136c6d96),
    ("C 0,0,4,4", 0x3fc43d136248490f),
    ("C 0,0,5,2", 0xbfd186f174f88472),
    ("C 0,0,7,1", 0xbfde5b9d136c6d96),
    ("C 0,0,9,0", 0x3fd186f174f88472),
    ("C 1 nnz", 0x000000000000001e),
    ("C 1,0,0,9", 0xbf94660891e3948c),
    ("C 1,0,0,10", 0x3fdc82037e4ae6f0),
    ("C 1,0,1,7", 0x3fa1aa69e4f2777f),
    ("C 1,0,2,5", 0x3f94660891e3948c),
    ("C 1,0,2,6", 0xbfdc82037e4ae6f0),
    ("C 1,0,3,3", 0xbfa1aa69e4f2777f),
    ("C 1,0,4,4", 0x3fdcb4ec140a022e),
    ("C 1,0,5,2", 0x3f94660891e3948c),
    ("F 0", 0x3fc52a7fa9d2f8ea),
    ("F 1", 0x3fbbb67ae8584cab),
    ("F 2", 0x3fbbb67ae8584caa),
    ("F 3", 0xbfc52a7fa9d2f8e9),
    ("F 0", 0x3fc86d2360c579ad),
    ("F 1", 0xbfaa5677365c8e78),
    ("F 2", 0x3faa5677365c8e77),
    ("F 3", 0x3fc86d2360c579ad),
    ("R 0", 0x3feffffffffffffe),
    ("R 1", 0x3c6c000000000000),
    ("R 2", 0x3c6c000000000000),
    ("R 3", 0x3fefffffffffffff),
    ("A 0", 0xbfeee5a9e77c9f3b),
    ("A 1", 0x3fd0a84f49748d86),
    ("A 2", 0xbfd0a84f49748d85),
    ("A 3", 0xbfeee5a9e77c9f3b),
    ("B 0", 0x3feac5eb3f7ab2f8),
    ("B 1", 0x3fe186f174f88473),
    ("B 2", 0x3fe186f174f88472),
    ("B 3", 0xbfeac5eb3f7ab2f7),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const C2: Golden = &[
    ("C 0 nnz", 0x000000000000001a),
    ("C 0,0,0,8", 0x3fd186f174f88472),
    ("C 0,0,1,6", 0xbfc75e9746a0b098),
    ("C 0,0,1,7", 0xbfca20bd700c2c3e),
    ("C 0,0,2,5", 0xbfde5b9d136c6d96),
    ("C 0,0,3,3", 0x3fcafc19d8606169),
    ("C 0,0,3,4", 0x3fbe2b7dddfefa66),
    ("C 0,0,4,3", 0x3fbe2b7dddfefa66),
    ("C 0,0,4,4", 0x3fdafc19d8606169),
    ("C 1 nnz", 0x0000000000000020),
    ("C 1,0,0,8", 0xbf94660891e3948c),
    ("C 1,0,0,9", 0x3fdc82037e4ae6f0),
    ("C 1,0,1,6", 0xbfd4660891e3948c),
    ("C 1,0,1,7", 0x3fd3f49c0b9ad4db),
    ("C 1,0,2,5", 0x3fa1aa69e4f2777f),
    ("C 1,0,3,3", 0x3fd78de286989f54),
    ("C 1,0,3,4", 0xbfc70aea090565ae),
    ("C 1,0,4,3", 0xbfc70aea090565ae),
    ("F 0", 0x3fc94c583ada5b52),
    ("F 1", 0xbc60000000000000),
    ("F 2", 0xbc60000000000000),
    ("F 3", 0x3fc94c583ada5b52),
    ("F 0", 0x3fc1b9a50c98c8c7),
    ("F 1", 0x3fc20ce944b98c20),
    ("F 2", 0x3fc20ce944b98c20),
    ("F 3", 0xbfc1b9a50c98c8c7),
    ("R 0", 0x3fefffffffffffff),
    ("R 1", 0x3c64000000000000),
    ("R 2", 0x3c64000000000000),
    ("R 3", 0x3fefffffffffffff),
    ("A 0", 0xbfe66bb78e5427af),
    ("A 1", 0xbfe6d50aad739575),
    ("A 2", 0xbfe6d50aad739575),
    ("A 3", 0x3fe66bb78e5427af),
    ("B 0", 0x3ff0000000000000),
    ("B 1", 0xbc843d136248490f),
    ("B 2", 0xbc843d136248490f),
    ("B 3", 0x3ff0000000000000),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const D3: Golden = &[
    ("C 0 nnz", 0x0000000000000011),
    ("C 0,0,0,7", 0x3fdbb67ae8584caa),
    ("C 0,0,1,5", 0x3fc6a09e667f3bcd),
    ("C 0,0,2,4", 0xbfe0f876ccdf6cd9),
    ("C 0,0,4,2", 0xbfc6a09e667f3bcd),
    ("C 0,0,5,1", 0x3fe0f876ccdf6cd9),
    ("C 0,0,7,0", 0x3fc279a74590331c),
    ("C 0,0,8,0", 0xbfda20bd700c2c3e),
    ("C 0,1,0,9", 0xbfc6a09e667f3bcd),
    ("C 1 nnz", 0x0000000000000012),
    ("C 1,0,0,7", 0x3fb5555555555555),
    ("C 1,0,0,8", 0x3fde2b7dddfefa66),
    ("C 1,0,1,5", 0xbfe0547666079ba7),
    ("C 1,0,2,4", 0xbfba20bd700c2c3e),
    ("C 1,0,4,2", 0x3fe0547666079ba7),
    ("C 1,0,5,1", 0x3fba20bd700c2c3e),
    ("C 1,0,7,0", 0xbfdaaaaaaaaaaaab),
    ("C 1,0,8,0", 0xbfce2b7dddfefa66),
    ("F 0", 0xbfd08654a2d4f6da),
    ("F 1", 0x0000000000000000),
    ("F 2", 0x0000000000000000),
    ("F 3", 0xbfd08654a2d4f6db),
    ("F 0", 0xbfd08654a2d4f6da),
    ("F 1", 0x0000000000000000),
    ("F 2", 0x0000000000000000),
    ("F 3", 0xbfd08654a2d4f6db),
    ("R 0", 0x3fe0000000000000),
    ("R 1", 0xbfebb67ae8584caa),
    ("R 2", 0xbfebb67ae8584caa),
    ("R 3", 0xbfdfffffffffffff),
    ("A 0", 0xbfefffffffffffff),
    ("A 1", 0x0000000000000000),
    ("A 2", 0x0000000000000000),
    ("A 3", 0xbff0000000000001),
    ("B 0", 0xbfefffffffffffff),
    ("B 1", 0x0000000000000000),
    ("B 2", 0x0000000000000000),
    ("B 3", 0xbff0000000000001),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const G2: Golden = &[
    ("C 0 nnz", 0x0000000000000022),
    ("C 0,0,0,12", 0x3fc5edc59c0f503a),
    ("C 0,0,0,14", 0xbfb7715fe69cf7e1),
    ("C 0,0,1,9", 0xbfd2fdab494e40d8),
    ("C 0,0,2,7", 0x3fcf031287273992),
    ("C 0,0,2,8", 0x3fc5edc59c0f503a),
    ("C 0,0,3,6", 0x3fdadb70384c4403),
    ("C 0,0,4,4", 0xbfd1e7a025882d58),
    ("C 0,0,4,5", 0xbfb95239b7130120),
    ("C 1 nnz", 0x000000000000002a),
    ("C 1,0,0,12", 0xbf94c7a66821d93e),
    ("C 1,0,0,13", 0x3fd6fde754adaa80),
    ("C 1,0,0,14", 0xbfccb705d47590c3),
    ("C 1,0,1,9", 0x3fa1fef3ba580a17),
    ("C 1,0,1,10", 0xbfd6fde754adaa80),
    ("C 1,0,2,7", 0x3fc6e010e1cb4c97),
    ("C 1,0,2,8", 0xbfd4124c5ec39a90),
    ("C 1,0,3,6", 0xbfa97336ce6fbb1e),
    ("F 0", 0x3fc8a2345cc04426),
    ("F 1", 0x3c40000000000000),
    ("F 2", 0x3c40000000000000),
    ("F 3", 0x3fc8a2345cc04425),
    ("F 0", 0x3fc8a2345cc04426),
    ("F 1", 0x3c40000000000000),
    ("F 2", 0x3c40000000000000),
    ("F 3", 0x3fc8a2345cc04425),
    ("R 0", 0x3ff0000000000000),
    ("R 1", 0xbc50000000000000),
    ("R 2", 0xbc50000000000000),
    ("R 3", 0x3ff0000000000000),
    ("A 0", 0x3ff0000000000000),
    ("A 1", 0x3c64c8dc2e423980),
    ("A 2", 0x3c64c8dc2e423980),
    ("A 3", 0x3fefffffffffffff),
    ("B 0", 0x3ff0000000000000),
    ("B 1", 0x3c64c8dc2e423980),
    ("B 2", 0x3c64c8dc2e423980),
    ("B 3", 0x3fefffffffffffff),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const F4: Golden = &[
    ("C 0 nnz", 0x0000000000000016),
    ("C 0,0,0,13", 0x3fd11acee560242a),
    ("C 0,0,1,11", 0xbfd4f2ec413cb52b),
    ("C 0,0,2,9", 0x3fd4f2ec413cb52b),
    ("C 0,0,3,7", 0xbfd4f2ec413cb52b),
    ("C 0,0,4,5", 0x3fd4f2ec413cb52b),
    ("C 0,0,5,4", 0x3fd4f2ec413cb52b),
    ("C 0,0,7,3", 0xbfd4f2ec413cb52b),
    ("C 0,0,9,2", 0x3fd4f2ec413cb52b),
    ("F 0", 0x3fc91a556151761d),
    ("F 0", 0x3fc91a556151761d),
    ("R 0", 0x3ff0000000000002),
    ("A 0", 0x3ff0000000000001),
    ("B 0", 0x3ff0000000000001),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const F4_N2: Golden = &[
    ("C 0 nnz", 0x0000000000000124),
    ("C 0,0,0,136", 0x3fb71d34986da701),
    ("C 0,0,0,138", 0xbfa42cf5da0b1da6),
    ("C 0,0,0,139", 0xbfb8b5c23ace4264),
    ("C 0,0,0,140", 0xbfb71d34986da701),
    ("C 0,0,1,118", 0xbfbab099ae8f539a),
    ("C 0,0,1,120", 0x3fac885b638e90b0),
    ("C 0,0,1,121", 0x3fc178fca42871ee),
    ("C 0,0,2,108", 0x3fb178fca42871ee),
    ("C 1 nnz", 0x0000000000000152),
    ("C 1,0,0,136", 0xbfae93ba1f46c761),
    ("C 1,0,0,137", 0x3fbd00a1696ffa9f),
    ("C 1,0,0,138", 0xbf912862cba54807),
    ("C 1,0,0,139", 0xbfa5038d584bfcdb),
    ("C 1,0,0,140", 0x3fae93ba1f46c761),
    ("C 1,0,1,118", 0x3fb1a7571ec9b9f1),
    ("C 1,0,1,119", 0xbfb7ae2f785affa0),
    ("C 1,0,1,120", 0x3f9843c590b1395d),
    ("F 0", 0xbfaefcd69be3e7cb),
    ("F 1", 0xbc24000000000000),
    ("F 2", 0xbc32000000000000),
    ("F 3", 0xbfaefcd69be3e7cb),
    ("F 0", 0xbfaefcd69be3e7d2),
    ("F 1", 0xbc14000000000000),
    ("F 2", 0xbc33000000000000),
    ("F 3", 0xbfaefcd69be3e7c9),
    ("R 0", 0xbff0000000000003),
    ("R 1", 0xbc72000000000000),
    ("R 2", 0xbc72000000000000),
    ("R 3", 0xbff0000000000001),
    ("A 0", 0xbff0000000000003),
    ("A 1", 0xbc54a74489a6c65d),
    ("A 2", 0xbc739ee782c4d60b),
    ("A 3", 0xbfeffffffffffffe),
    ("B 0", 0xbff0000000000000),
    ("B 1", 0xbc64a74489a6c65d),
    ("B 2", 0xbc72968a7be2e5ba),
    ("B 3", 0xbff0000000000000),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const E6: Golden = &[
    ("C 0 nnz", 0x000000000000000c),
    ("C 0,0,0,7", 0x3fda20bd700c2c3e),
    ("C 0,0,1,4", 0xbfda20bd700c2c3e),
    ("C 0,0,2,3", 0x3fda20bd700c2c3e),
    ("C 0,0,3,2", 0xbfda20bd700c2c3e),
    ("C 0,0,4,1", 0x3fda20bd700c2c3e),
    ("C 0,0,6,0", 0xbfda20bd700c2c3e),
    ("C 0,1,0,9", 0x3fda20bd700c2c3e),
    ("C 0,1,1,6", 0xbfda20bd700c2c3e),
    ("F 0", 0xbfd4ef3e4b2cf7b9),
    ("F 0", 0xbfd4ef3e4b2cf7ba),
    ("R 0", 0xbfefffffffffffff),
    ("A 0", 0xbff0000000000001),
    ("B 0", 0xbff0000000000000),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const E6_N2: Golden = &[
    ("C 0 nnz", 0x000000000000005e),
    ("C 0,0,0,136", 0x3fc6bdcd076dfc6d),
    ("C 0,0,0,137", 0xbfc19d8c2e54c7a8),
    ("C 0,0,1,105", 0xbfc6bdcd076dfc6d),
    ("C 0,0,1,106", 0x3fc19d8c2e54c7a8),
    ("C 0,0,2,82", 0x3fc6bdcd076dfc6d),
    ("C 0,0,2,84", 0xbfc19d8c2e54c7a8),
    ("C 0,0,3,64", 0xbfc6bdcd076dfc6d),
    ("C 0,0,3,65", 0x3fc014a2883c8175),
    ("C 1 nnz", 0x0000000000000066),
    ("C 1,0,0,136", 0x3f8b816374e9fa18),
    ("C 1,0,0,137", 0xbf854e3c6fce94e8),
    ("C 1,0,0,138", 0x3fcd4b9319bc0cbf),
    ("C 1,0,1,105", 0xbf8b816374e9fa18),
    ("C 1,0,1,106", 0x3f854e3c6fce94e8),
    ("C 1,0,1,107", 0xbfcd4b9319bc0cbf),
    ("C 1,0,2,82", 0x3f8b816374e9fa18),
    ("C 1,0,2,84", 0xbf854e3c6fce94e8),
    ("F 0", 0xbfbcfc7da32a91fb),
    ("F 1", 0xbc29000000000000),
    ("F 2", 0xbc12000000000000),
    ("F 3", 0xbfbcfc7da32a920a),
    ("F 0", 0xbfbcfc7da32a9214),
    ("F 1", 0x3c30000000000000),
    ("F 2", 0x3c30000000000000),
    ("F 3", 0xbfbcfc7da32a9216),
    ("R 0", 0x3fe10e59c5927d50),
    ("R 1", 0xbfeb135de716522e),
    ("R 2", 0xbfeb135de7165231),
    ("R 3", 0xbfe10e59c5927d53),
    ("A 0", 0xbff0000000000001),
    ("A 1", 0x3c61a9dc8f6df104),
    ("A 2", 0x3c61a9dc8f6df104),
    ("A 3", 0xbff0000000000002),
    ("B 0", 0xbfefffffffffffe7),
    ("B 1", 0xbc5b9968a01bc896),
    ("B 2", 0xbc43df18215baf24),
    ("B 3", 0xbfeffffffffffff7),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const E7: Golden = &[
    ("C 0 nnz", 0x0000000000000018),
    ("C 0,0,0,16", 0x3fd279a74590331c),
    ("C 0,0,1,13", 0xbfd279a74590331c),
    ("C 0,0,2,11", 0x3fd279a74590331c),
    ("C 0,0,3,9", 0xbfd279a74590331c),
    ("C 0,0,4,7", 0x3fd279a74590331c),
    ("C 0,0,5,6", 0xbfd279a74590331c),
    ("C 0,0,6,5", 0xbfd279a74590331c),
    ("C 0,0,7,4", 0x3fd279a74590331c),
    ("F 0", 0x3fca5c325622b4ce),
    ("F 0", 0x3fca5c325622b4cf),
    ("R 0", 0x3ff0000000000000),
    ("A 0", 0xbff0000000000002),
    ("B 0", 0x3ff0000000000001),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const E7_N2: Golden = &[
    ("C 0 nnz", 0x000000000000028e),
    ("C 0,0,0,780", 0x3fb5a437679aa08a),
    ("C 0,0,0,782", 0xbfb24a56654fa95f),
    ("C 0,0,1,719", 0xbfb6f44972950fbf),
    ("C 0,0,2,667", 0xbfa2bdf7d9bd3b04),
    ("C 0,0,2,668", 0x3fb4f44a9edfef4f),
    ("C 0,0,3,660", 0x3fb6f44972950fbf),
    ("C 0,0,4,607", 0x3fa2bdf7d9bd3b04),
    ("C 0,0,4,608", 0xbfb4f44a9edfef4f),
    ("C 1 nnz", 0x00000000000002d4),
    ("C 1,0,0,780", 0x3f92880a3481e3d2),
    ("C 1,0,0,781", 0x3fb85e1cea8fec93),
    ("C 1,0,0,782", 0xbfae6417d517d86c),
    ("C 1,0,1,719", 0xbf93a7d06b444c99),
    ("C 1,0,1,720", 0xbfb51a5d421b4f48),
    ("C 1,0,2,667", 0x3fb1421a9b1c54d5),
    ("C 1,0,2,668", 0x3faa33b5dc70d0f4),
    ("C 1,0,3,660", 0x3f93a7d06b444c99),
    ("F 0", 0x3f9a1a37b2bd1671),
    ("F 1", 0x3c08000000000000),
    ("F 2", 0x3c08000000000000),
    ("F 3", 0x3f9a1a37b2bd1674),
    ("F 0", 0x3f9a1a37b2bd1671),
    ("F 1", 0x3c08000000000000),
    ("F 2", 0x3c08000000000000),
    ("F 3", 0x3f9a1a37b2bd1674),
    ("R 0", 0x3feffffffffffffe),
    ("R 1", 0xbca1800000000000),
    ("R 2", 0xbca1800000000000),
    ("R 3", 0x3ff0000000000002),
    ("A 0", 0x3ff0000000000005),
    ("A 1", 0x3c5d6c2d68694394),
    ("A 2", 0x3c5d6c2d68694394),
    ("A 3", 0x3ff0000000000007),
    ("B 0", 0x3ff0000000000005),
    ("B 1", 0x3c5d6c2d68694394),
    ("B 2", 0x3c5d6c2d68694394),
    ("B 3", 0x3ff0000000000007),
];
#[cfg(feature = "lie-generated")]
#[rustfmt::skip]
const E8: Golden = &[
    ("C 0 nnz", 0x0000000000000076),
    ("C 0,0,0,127", 0x3fc75e9746a0b098),
    ("C 0,0,1,119", 0xbfc08654a2d4f6db),
    ("C 0,0,2,111", 0x3fc08654a2d4f6db),
    ("C 0,0,3,104", 0xbfc08654a2d4f6db),
    ("C 0,0,4,97", 0x3fc08654a2d4f6db),
    ("C 0,0,5,90", 0xbfc08654a2d4f6db),
    ("C 0,0,6,83", 0x3fc08654a2d4f6db),
    ("C 0,0,7,82", 0x3fc08654a2d4f6db),
    ("F 0", 0xbfb0418a4806de7e),
    ("F 0", 0xbfb0418a4806de7e),
    ("R 0", 0xbff0000000000006),
    ("A 0", 0xbff0000000000001),
    ("B 0", 0xbff0000000000001),
];
