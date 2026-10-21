//! Tensor-product channels and multiplicities across SU(N) and B/C/D, checked
//! against GroupMath 1.1.3 (`ReduceRepProduct`, arXiv:2011.01764).
//!
//! Contract: [`fusion_symbols::category::FusionCategory::fusion`] /
//! [`fusion_symbols::category::FusionCategory::fusion`] reproduce the rows below exactly
//! (GroupMath and this crate share the Dynkin node order, so labels compare
//! with no relabeling), and every decomposition satisfies the dimension sum
//! rule `Σ_c N^c_{ab} dim c = dim a · dim b`. CGC values are a different gauge
//! and are not compared.
#![cfg(feature = "lie-generated")]

#[path = "common/lie.rs"]
mod lie_test;
use std::collections::BTreeMap;

use num_bigint::BigInt;

#[path = "common/bcd.rs"]
mod bcd_test;
use bcd_test::Series;
use lie_test::product as sdp;

const FIXTURES: &str = "\
SU2 | 2 | 2 | 3 3 | 4:1 0:1 2:1\n\
SU2 | 1 | 1 | 2 2 | 2:1 0:1\n\
SU2 | 2 | 3 | 3 4 | 5:1 1:1 3:1\n\
SU3 | 1,0 | 0,1 | 3 3 | 1,1:1 0,0:1\n\
SU3 | 1,1 | 1,1 | 8 8 | 2,2:1 0,3:1 3,0:1 0,0:1 1,1:2\n\
SU3 | 2,0 | 1,1 | 6 8 | 3,1:1 1,2:1 2,0:1 0,1:1\n\
SU4 | 1,0,0 | 0,0,1 | 4 4 | 1,0,1:1 0,0,0:1\n\
SU4 | 1,0,1 | 1,0,1 | 15 15 | 2,0,2:1 0,1,2:1 2,1,0:1 0,2,0:1 1,0,1:2 0,0,0:1\n\
SU5 | 1,0,0,0 | 0,0,0,1 | 5 5 | 1,0,0,1:1 0,0,0,0:1\n\
SU5 | 1,0,0,1 | 1,0,0,1 | 24 24 | 2,0,0,2:1 0,1,0,2:1 2,0,1,0:1 0,1,1,0:1 1,0,0,1:2 0,0,0,0:1\n\
SO5 | 1,0 | 1,0 | 5 5 | 2,0:1 0,2:1 0,0:1\n\
SO5 | 1,0 | 0,2 | 5 10 | 1,2:1 1,0:1 0,2:1\n\
SO5 | 2,0 | 2,0 | 14 14 | 4,0:1 0,4:1 0,0:1 2,2:1 0,2:1 2,0:1\n\
Sp4 | 1,0 | 1,0 | 4 4 | 2,0:1 0,1:1 0,0:1\n\
Sp4 | 1,0 | 0,1 | 4 5 | 1,1:1 1,0:1\n\
Sp4 | 2,0 | 1,1 | 10 16 | 3,1:1 1,2:1 3,0:1 1,0:1 1,1:2\n\
SO6 | 0,1,1 | 0,1,1 | 15 15 | 0,2,2:1 1,0,2:1 1,2,0:1 2,0,0:1 0,1,1:2 0,0,0:1\n\
SO6 | 1,0,0 | 1,0,0 | 6 6 | 2,0,0:1 0,1,1:1 0,0,0:1\n\
SO7 | 1,0,0 | 1,0,0 | 7 7 | 2,0,0:1 0,1,0:1 0,0,0:1\n\
SO7 | 0,1,0 | 0,1,0 | 21 21 | 0,2,0:1 1,0,2:1 2,0,0:1 0,1,0:1 0,0,0:1 0,0,2:1\n\
Sp6 | 1,0,0 | 1,0,0 | 6 6 | 2,0,0:1 0,1,0:1 0,0,0:1\n\
";

/// A decomposition keyed by the summand's Dynkin label.
type Channels = BTreeMap<Vec<i64>, u32>;

fn csv_i64(s: &str) -> Vec<i64> {
    s.split(',').map(|x| x.trim().parse().unwrap()).collect()
}

/// Parse the `c:n c:n ...` channel field into a label→multiplicity map.
fn parse_channels(s: &str) -> Channels {
    s.split_whitespace()
        .map(|tok| {
            let (label, mult) = tok.rsplit_once(':').unwrap();
            (csv_i64(label), mult.parse().unwrap())
        })
        .collect()
}

/// racah dimensions and product channels for a group token, so both the SU(N)
/// and B/C/D pipelines feed one comparison. Returns `(dim_a, dim_b, channels)`.
fn racah_side(group: &str, a: &[i64], b: &[i64]) -> (BigInt, BigInt, Channels) {
    // B/C/D groups: (series, expected rank is implied by the label length).
    let bcd_series = match group {
        "SO5" | "SO7" => Some(Series::B),
        "SO6" => Some(Series::D),
        "Sp4" | "Sp6" => Some(Series::C),
        _ => None,
    };
    if let Some(series) = bcd_series {
        let ia = bcd_test::irrep(series, a).unwrap();
        let ib = bcd_test::irrep(series, b).unwrap();
        let ch = bcd_test::product(&bcd_test::group(series, a.len()).unwrap(), &ia, &ib)
            .unwrap()
            .into_iter()
            .map(|(k, v)| (k.dynkin(), v))
            .collect();
        return (ia.dim_exact(), ib.dim_exact(), ch);
    }
    // SU(N): the token is "SU" followed by N.
    assert!(group.starts_with("SU"), "unknown group token {group}");
    let _n: usize = group[2..].parse().unwrap();
    let ia = lie_test::irrep(a).unwrap();
    let ib = lie_test::irrep(b).unwrap();
    let ch = sdp(&ia, &ib)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k.dynkin(), v))
        .collect();
    (ia.dim_exact(), ib.dim_exact(), ch)
}

struct Row {
    group: String,
    a: Vec<i64>,
    b: Vec<i64>,
    dim_a: BigInt,
    dim_b: BigInt,
    channels: Channels,
}

fn parse() -> Vec<Row> {
    let mut rows = Vec::new();
    for line in FIXTURES.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('|').map(str::trim).collect();
        assert_eq!(f.len(), 5, "malformed fixture line: {line}");
        let dims: Vec<&str> = f[3].split_whitespace().collect();
        rows.push(Row {
            group: f[0].to_string(),
            a: csv_i64(f[1]),
            b: csv_i64(f[2]),
            dim_a: dims[0].parse().unwrap(),
            dim_b: dims[1].parse().unwrap(),
            channels: parse_channels(f[4]),
        });
    }
    rows
}

#[test]
fn rows_cover_every_family() {
    let rows = parse();
    let groups: std::collections::BTreeSet<_> = rows.iter().map(|r| r.group.as_str()).collect();
    for g in [
        "SU2", "SU3", "SU4", "SU5", "SO5", "SO6", "SO7", "Sp4", "Sp6",
    ] {
        assert!(groups.contains(g), "missing group {g}");
    }
}

#[test]
fn racah_reproduces_groupmath_products() {
    for r in parse() {
        let (dim_a, dim_b, channels) = racah_side(&r.group, &r.a, &r.b);
        assert_eq!(dim_a, r.dim_a, "{} {:?} dim_a", r.group, r.a);
        assert_eq!(dim_b, r.dim_b, "{} {:?} dim_b", r.group, r.b);
        assert_eq!(
            channels, r.channels,
            "{} {:?} x {:?} channels disagree with GroupMath",
            r.group, r.a, r.b
        );
        let sum: BigInt = channels
            .iter()
            .map(|(c, &n)| racah_side(&r.group, c, &r.b).0 * n)
            .sum();
        assert_eq!(sum, &dim_a * &dim_b, "{} dimension sum rule", r.group);
    }
}
