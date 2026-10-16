//! Value identity with the implementation the `abelian` module was ported
//! from.
//!
//! `fixtures/symtn-49d09dc4-abelian-values.txt` holds every value symtn's
//! abelian providers expose at commit
//! 49d09dc4fc0825b29d547e63a70014221c0c5406 (`symtn-sectors/src/abelian.rs`:
//! `Z2FusionRule` "z2", `FermionParityFusionRule` "fz2", `ZNFusionRule` for
//! N = 1..=4 "zn1".."zn4", `U1FusionRule` "u1"), as IEEE-754 bit patterns, keyed
//! by semantic labels (parity, charge mod N, U(1) charge) rather than symtn's
//! sector ids. It was printed by `fixtures/extract_symtn_abelian.rs` (manifest
//! beside it), built against a `git archive` of that commit with symtn's own
//! `Cargo.lock`, on macOS aarch64, rustc 1.96.0. Nothing here runs symtn.
//!
//! symtn's `inv_dim_scalar`, `sqrt_dim_scalar` and `inv_sqrt_dim_scalar` map
//! to `1.0 / dim`, `dim.sqrt()`, `1.0 / dim.sqrt()`.
//!
//! Every value matches bit for bit, with two deliberate differences, both
//! counted exactly below:
//!
//! - symtn's `f_symbol_scalar` / `r_symbol_scalar` do not check admissibility
//!   (its trait only calls them on admissible vertices) and return `±1.0` on
//!   any tuple; here an inadmissible F or R is `0.0`, as in TensorKitSectors.
//! - symtn's checked `try_nsymbol` fails when the U(1) sum `a + b` leaves the
//!   label range; here `N = 0`, the exact answer. The checked fusion fails in
//!   both, on the same pairs.

use fusion_symbols::abelian::{
    fermion_parity_f_symbol, fermion_parity_nsymbol, fermion_parity_r_symbol, u1_f_symbol,
    u1_nsymbol, u1_r_symbol, z2_f_symbol, z2_nsymbol, z2_r_symbol, zn_f_symbol, zn_nsymbol,
    zn_r_symbol, FermionParity, U1Irrep, Z2Irrep, ZNIrrep,
};

const VALUES: &str = include_str!("fixtures/symtn-49d09dc4-abelian-values.txt");

/// The values one family exposes, keyed by semantic labels.
trait Provider {
    fn vacuum(&self) -> i64;
    fn dual(&self, a: i64) -> i64;
    fn dim(&self, a: i64) -> f64;
    fn fs(&self, a: i64) -> f64;
    fn twist(&self, a: i64) -> f64;
    /// `None` when the fusion is not a label.
    fn fusion(&self, a: i64, b: i64) -> Option<i64>;
    fn n(&self, a: i64, b: i64, c: i64) -> u32;
    fn r(&self, a: i64, b: i64, c: i64) -> f64;
    fn f(&self, l: [i64; 6]) -> f64;
}

fn z2(a: i64) -> Z2Irrep {
    Z2Irrep::new(a == 1)
}

fn z2_id(a: Z2Irrep) -> i64 {
    i64::from(a.is_odd())
}

struct Z2;

impl Provider for Z2 {
    fn vacuum(&self) -> i64 {
        z2_id(Z2Irrep::EVEN)
    }
    fn dual(&self, a: i64) -> i64 {
        z2_id(z2(a).dual())
    }
    fn dim(&self, a: i64) -> f64 {
        z2(a).dim()
    }
    fn fs(&self, a: i64) -> f64 {
        z2(a).frobenius_schur()
    }
    fn twist(&self, a: i64) -> f64 {
        z2(a).twist()
    }
    fn fusion(&self, a: i64, b: i64) -> Option<i64> {
        Some(z2_id(z2(a).fusion(z2(b))))
    }
    fn n(&self, a: i64, b: i64, c: i64) -> u32 {
        z2_nsymbol(z2(a), z2(b), z2(c))
    }
    fn r(&self, a: i64, b: i64, c: i64) -> f64 {
        z2_r_symbol(z2(a), z2(b), z2(c))
    }
    fn f(&self, l: [i64; 6]) -> f64 {
        let [a, b, c, d, e, f] = l.map(z2);
        z2_f_symbol(a, b, c, d, e, f)
    }
}

fn fz2(a: i64) -> FermionParity {
    FermionParity::new(a == 1)
}

fn fz2_id(a: FermionParity) -> i64 {
    i64::from(a.is_odd())
}

struct Fz2;

impl Provider for Fz2 {
    fn vacuum(&self) -> i64 {
        fz2_id(FermionParity::EVEN)
    }
    fn dual(&self, a: i64) -> i64 {
        fz2_id(fz2(a).dual())
    }
    fn dim(&self, a: i64) -> f64 {
        fz2(a).dim()
    }
    fn fs(&self, a: i64) -> f64 {
        fz2(a).frobenius_schur()
    }
    fn twist(&self, a: i64) -> f64 {
        fz2(a).twist()
    }
    fn fusion(&self, a: i64, b: i64) -> Option<i64> {
        Some(fz2_id(fz2(a).fusion(fz2(b))))
    }
    fn n(&self, a: i64, b: i64, c: i64) -> u32 {
        fermion_parity_nsymbol(fz2(a), fz2(b), fz2(c))
    }
    fn r(&self, a: i64, b: i64, c: i64) -> f64 {
        fermion_parity_r_symbol(fz2(a), fz2(b), fz2(c))
    }
    fn f(&self, l: [i64; 6]) -> f64 {
        let [a, b, c, d, e, f] = l.map(fz2);
        fermion_parity_f_symbol(a, b, c, d, e, f)
    }
}

struct Zn(u32);

impl Zn {
    fn irrep(&self, a: i64) -> ZNIrrep {
        ZNIrrep::new(a, self.0).expect("nonzero modulus")
    }
}

impl Provider for Zn {
    fn vacuum(&self) -> i64 {
        i64::from(ZNIrrep::unit(self.0).unwrap().charge())
    }
    fn dual(&self, a: i64) -> i64 {
        i64::from(self.irrep(a).dual().charge())
    }
    fn dim(&self, a: i64) -> f64 {
        self.irrep(a).dim()
    }
    fn fs(&self, a: i64) -> f64 {
        self.irrep(a).frobenius_schur()
    }
    fn twist(&self, a: i64) -> f64 {
        self.irrep(a).twist()
    }
    fn fusion(&self, a: i64, b: i64) -> Option<i64> {
        let c = self.irrep(a).fusion(self.irrep(b)).expect("one modulus");
        Some(i64::from(c.charge()))
    }
    fn n(&self, a: i64, b: i64, c: i64) -> u32 {
        zn_nsymbol(self.irrep(a), self.irrep(b), self.irrep(c))
    }
    fn r(&self, a: i64, b: i64, c: i64) -> f64 {
        zn_r_symbol(self.irrep(a), self.irrep(b), self.irrep(c))
    }
    fn f(&self, l: [i64; 6]) -> f64 {
        let [a, b, c, d, e, f] = l.map(|q| self.irrep(q));
        zn_f_symbol(a, b, c, d, e, f)
    }
}

fn u1(a: i64) -> U1Irrep {
    U1Irrep::new(i32::try_from(a).expect("i32 charge")).expect("label")
}

struct U1;

impl Provider for U1 {
    fn vacuum(&self) -> i64 {
        i64::from(U1Irrep::UNIT.charge())
    }
    fn dual(&self, a: i64) -> i64 {
        i64::from(u1(a).dual().charge())
    }
    fn dim(&self, a: i64) -> f64 {
        u1(a).dim()
    }
    fn fs(&self, a: i64) -> f64 {
        u1(a).frobenius_schur()
    }
    fn twist(&self, a: i64) -> f64 {
        u1(a).twist()
    }
    fn fusion(&self, a: i64, b: i64) -> Option<i64> {
        u1(a).fusion(u1(b)).ok().map(|c| i64::from(c.charge()))
    }
    fn n(&self, a: i64, b: i64, c: i64) -> u32 {
        u1_nsymbol(u1(a), u1(b), u1(c))
    }
    fn r(&self, a: i64, b: i64, c: i64) -> f64 {
        u1_r_symbol(u1(a), u1(b), u1(c))
    }
    fn f(&self, l: [i64; 6]) -> f64 {
        let [a, b, c, d, e, f] = l.map(u1);
        u1_f_symbol(a, b, c, d, e, f)
    }
}

fn recorded(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("hex bits"))
}

#[derive(Default, Debug, PartialEq)]
struct Tally {
    /// Lines compared bit for bit or exactly.
    identical: usize,
    /// Inadmissible F/R where symtn records `±1.0` and this crate `0.0`.
    inadmissible_symbol: usize,
    /// U(1) N where symtn's checked N failed and this crate returns `0`.
    u1_overflow_n: usize,
}

#[test]
fn every_value_matches_symtn_49d09dc4() {
    let mut tally = std::collections::BTreeMap::<String, Tally>::new();
    for line in VALUES.lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let tag = fields[0];
        let p: Box<dyn Provider> = match tag {
            "z2" => Box::new(Z2),
            "fz2" => Box::new(Fz2),
            "u1" => Box::new(U1),
            zn => Box::new(Zn(zn[2..].parse().expect("zn<N>"))),
        };
        let t = tally.entry(tag.to_string()).or_default();
        let what = fields[1];
        let label = |i: usize| -> i64 { fields[2 + i].parse().expect("label") };
        let last = fields[fields.len() - 1];
        let same_bits = |got: f64| assert_eq!(got.to_bits(), recorded(last).to_bits(), "{line}");
        match what {
            "vacuum" => assert_eq!(p.vacuum(), label(0), "{line}"),
            "dual" => assert_eq!(p.dual(label(0)), label(1), "{line}"),
            "dim" => same_bits(p.dim(label(0))),
            "inv_dim" => same_bits(1.0 / p.dim(label(0))),
            "sqrt_dim" => same_bits(p.dim(label(0)).sqrt()),
            "inv_sqrt_dim" => same_bits(1.0 / p.dim(label(0)).sqrt()),
            "fs" => same_bits(p.fs(label(0))),
            "twist" => same_bits(p.twist(label(0))),
            "fusion" => {
                let want = (last != "err").then(|| last.parse::<i64>().expect("one channel"));
                assert_eq!(p.fusion(label(0), label(1)), want, "{line}");
            }
            "N" => {
                let got = p.n(label(0), label(1), label(2));
                if last == "err" {
                    assert_eq!(tag, "u1", "{line}");
                    assert_eq!(got, 0, "{line}");
                    t.u1_overflow_n += 1;
                    continue;
                }
                assert_eq!(got, last.parse::<u32>().unwrap(), "{line}");
            }
            "R" | "F" => {
                let labels: Vec<i64> = (0..if what == "R" { 3 } else { 6 }).map(label).collect();
                let (got, admissible) = if what == "R" {
                    let [a, b, c] = [labels[0], labels[1], labels[2]];
                    (p.r(a, b, c), p.n(a, b, c) == 1)
                } else {
                    let l: [i64; 6] = labels.clone().try_into().unwrap();
                    let [a, b, c, d, e, f] = l;
                    let all = p.n(a, b, e) * p.n(e, c, d) * p.n(b, c, f) * p.n(a, f, d);
                    (p.f(l), all == 1)
                };
                if !admissible {
                    assert_eq!(got.to_bits(), 0.0f64.to_bits(), "{line}");
                    assert_eq!(recorded(last).abs(), 1.0, "{line}");
                    t.inadmissible_symbol += 1;
                    continue;
                }
                same_bits(got);
            }
            other => panic!("unknown quantity {other}"),
        }
        t.identical += 1;
    }

    // Per group of order n over all labels: 1 vacuum + 7n unary + n^2
    // fusion + n^3 N + n^3 R + n^6 F lines; n^2 admissible R and n^3
    // admissible F. U(1): 8 window charges (1 + 56 + 64 + 512 N lines) and
    // R/F on {-1, 0, 1}: 7 admissible R of 27, 15 admissible F of 729.
    // Of the 512 window N, 120 have `a + b` outside the label range.
    let group = |n: usize| Tally {
        identical: 1 + 7 * n + n * n + n.pow(3) + n * n + n.pow(3),
        inadmissible_symbol: (n.pow(3) - n * n) + (n.pow(6) - n.pow(3)),
        u1_overflow_n: 0,
    };
    let mut expected = std::collections::BTreeMap::new();
    for (tag, n) in [
        ("zn1", 1),
        ("zn2", 2),
        ("zn3", 3),
        ("zn4", 4),
        ("z2", 2),
        ("fz2", 2),
    ] {
        expected.insert(tag.to_string(), group(n));
    }
    expected.insert(
        "u1".to_string(),
        Tally {
            identical: 1 + 7 * 8 + 64 + (512 - 120) + 7 + 15,
            inadmissible_symbol: (27 - 7) + (729 - 15),
            u1_overflow_n: 120,
        },
    );
    assert_eq!(tally, expected);
}
