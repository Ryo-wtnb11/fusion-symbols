//! Value identity of the `cu1` module with two recorded sources.
//!
//! - `fixtures/symtn-2a6c1523-cu1-values.txt`: every value symtn's
//!   `CU1FusionRule` exposes at commit 2a6c152315bb271dc5e56b5f1f034ca818ef999d
//!   (`symtn-sectors/src/cu1.rs`, the implementation this module was ported
//!   from), printed by `fixtures/extract_symtn_cu1.rs` on macOS aarch64,
//!   rustc 1.96.0. Unary data, fusion and N cover `0+`, `0-`, `2j = 1..=5`
//!   and the twice-charge extremes; R and F cover `0+`, `0-`, `2j = 1..=5`
//!   on every tuple (tuples whose value is `+0.0` are omitted).
//! - `fixtures/tks-0.3.9-cu1-values.txt`: the independent oracle,
//!   TensorKitSectors 0.3.9 (git `67d342a1`, tree `9f536263`) `CU1Irrep`
//!   measured with Julia 1.11.6 by `fixtures/tks_cu1_oracle.jl`, on `0+`,
//!   `0-`, `2j = 1..=5`: dual, dim, FS phase and indicator, twist, fusion
//!   order, and every nonzero N, R and F.
//!
//! Labels are `twice_charge:s`; values are IEEE-754 bit patterns. Nothing
//! here runs symtn or Julia.

use std::collections::HashMap;

use fusion_symbols::cu1::{
    cu1_f_symbol, cu1_nsymbol, cu1_r_symbol, CU1Irrep, CU1_MAX_TWICE_CHARGE,
};

const SYMTN: &str = include_str!("fixtures/symtn-2a6c1523-cu1-values.txt");
const TKS: &str = include_str!("fixtures/tks-0.3.9-cu1-values.txt");

fn label(s: &str) -> CU1Irrep {
    let (q, s) = s.split_once(':').expect("twice_charge:s");
    CU1Irrep::new(q.parse().unwrap(), s.parse().unwrap()).expect("valid label")
}

fn name(a: CU1Irrep) -> String {
    format!("{}:{}", a.twice_charge(), a.s())
}

fn bits(s: &str) -> u64 {
    u64::from_str_radix(s, 16).expect("hex bits")
}

/// `0+`, `0-`, `2j = 1..=5`: the labels both sources record R and F on.
fn small() -> Vec<CU1Irrep> {
    [CU1Irrep::VACUUM, CU1Irrep::PSEUDOSCALAR]
        .into_iter()
        .chain((1..=5).map(|q| CU1Irrep::from_twice_charge(q).unwrap()))
        .collect()
}

#[derive(Default, Debug, PartialEq)]
struct Tally {
    /// Lines compared bit for bit or exactly.
    identical: usize,
    /// N lines where the source's checked N failed (fusion not
    /// representable) and this crate returns the exact `N`.
    overflow_n: usize,
    /// Of those, how many are `N = 1` (the `0±` or difference channel).
    overflow_n_one: usize,
}

/// Checks every recorded line; R/F lines go to `symbols` for the
/// all-tuple sweep, keyed `"R a b c"` / `"F a b c d e f"`.
fn check(source: &str, symbols: &mut HashMap<String, u64>) -> Tally {
    let mut t = Tally::default();
    for line in source.lines().filter(|l| !l.starts_with('#')) {
        let w: Vec<&str> = line.split_whitespace().collect();
        let last = *w.last().unwrap();
        let a = || label(w[1]);
        let same = |got: f64, want: &str| assert_eq!(got.to_bits(), bits(want), "{line}");
        match w[0] {
            "max_twice_charge" => assert_eq!(last.parse::<u32>().unwrap(), CU1_MAX_TWICE_CHARGE),
            "vacuum" => assert_eq!(label(w[1]), CU1Irrep::VACUUM, "{line}"),
            "dual" => assert_eq!(a().dual(), label(w[2]), "{line}"),
            "dim" => same(a().dim(), last),
            "inv_dim" => same(1.0 / a().dim(), last),
            "sqrt_dim" => same(a().dim().sqrt(), last),
            "inv_sqrt_dim" => same(1.0 / a().dim().sqrt(), last),
            "twist" => same(a().twist(), last),
            "fs" => {
                same(a().frobenius_schur(), w[2]);
                if w.len() == 4 {
                    // TensorKitSectors' group-theoretic indicator.
                    assert_eq!(
                        f64::from(w[3].parse::<i8>().unwrap()),
                        a().frobenius_schur()
                    );
                }
            }
            "fusion" => {
                let got = a()
                    .fusion(label(w[2]))
                    .ok()
                    .map(|ch| ch.map(name).collect::<Vec<_>>().join(","));
                assert_eq!(got.as_deref(), (last != "err").then_some(last), "{line}");
            }
            "N" => {
                let got = cu1_nsymbol(a(), label(w[2]), label(w[3]));
                if last == "err" {
                    t.overflow_n += 1;
                    t.overflow_n_one += got as usize;
                    continue;
                }
                assert_eq!(got, last.parse::<u32>().unwrap(), "{line}");
            }
            "R" | "F" => {
                symbols.insert(w[..w.len() - 1].join(" "), bits(last));
                continue;
            }
            other => panic!("unknown quantity {other}"),
        }
        t.identical += 1;
    }
    t
}

/// Every R and F tuple over [`small`]: the recorded bits, or `+0.0` when
/// the tuple is not recorded. Returns the number of recorded tuples.
fn sweep_symbols(symbols: &HashMap<String, u64>) -> usize {
    let all = small();
    let mut seen = 0;
    let mut cmp = |key: String, got: f64| {
        let want = symbols.get(&key).copied().unwrap_or(0);
        seen += usize::from(symbols.contains_key(&key));
        assert_eq!(got.to_bits(), want, "{key}");
    };
    for &a in &all {
        for &b in &all {
            for &c in &all {
                cmp(
                    format!("R {} {} {}", name(a), name(b), name(c)),
                    cu1_r_symbol(a, b, c),
                );
                for &d in &all {
                    for &e in &all {
                        for &f in &all {
                            let key = [a, b, c, d, e, f].map(name).join(" ");
                            cmp(format!("F {key}"), cu1_f_symbol(a, b, c, d, e, f));
                        }
                    }
                }
            }
        }
    }
    assert_eq!(seen, symbols.len(), "every recorded tuple is in the sweep");
    seen
}

#[test]
fn every_value_matches_symtn_2a6c1523() {
    let mut symbols = HashMap::new();
    let t = check(SYMTN, &mut symbols);
    // 11 window labels: 1 bound + 1 vacuum + 7·11 unary + 11² fusion lines
    // (33 of them `err` in both: `2(j + k)` above the bound), and 11³ = 1331
    // N, of which symtn's checked N fails on the 33 · 11 with such a pair.
    // Exactly, 14 of those are N = 1: `0+` and `0-` for the 3 equal pairs,
    // and the difference channel `|j - k| = 1/2` or `2147483647/2` for the
    // ordered pairs (1, max), (2147483647, 2147483648), (2147483647, max),
    // (max - 1, max) and their mirrors. 64 nonzero R and 628 nonzero F over
    // the 7 small labels.
    assert_eq!(
        t,
        Tally {
            identical: 1 + 1 + 77 + 121 + (1331 - 363),
            overflow_n: 363,
            overflow_n_one: 14,
        }
    );
    assert_eq!(sweep_symbols(&symbols), 64 + 628);
}

#[test]
fn every_value_matches_tensorkitsectors_0_3_9() {
    let mut symbols = HashMap::new();
    let t = check(TKS, &mut symbols);
    // 7 labels: 4·7 unary + 7² fusion + 64 nonzero N.
    assert_eq!(
        t,
        Tally {
            identical: 28 + 49 + 64,
            ..Tally::default()
        }
    );
    assert_eq!(sweep_symbols(&symbols), 64 + 628);
    // Every admissible N not recorded would be a missing channel: the sweep
    // of N over the same labels has exactly the 64 recorded triples.
    let all = small();
    let mut ones = 0;
    for &a in &all {
        for &b in &all {
            for &c in &all {
                ones += cu1_nsymbol(a, b, c);
            }
        }
    }
    assert_eq!(ones, 64);
}
