//! Value identity with the implementation this crate was ported from.
//!
//! `fixtures/tenet-957ccc05-values.txt` holds every value symtn's
//! two Fibonacci providers expose at commit
//! 957ccc05737f0841b6a98047880aba201bee63a1 — `tenet_sectors::FibonacciFusionRule`
//! ("closed") and `tenet_category_data::CategoryDataFibonacci` ("table") — as
//! IEEE-754 bit patterns, with zero-based sector ids. It was printed by
//! `fixtures/extract_tenet_values.rs` (manifest beside it), built against a
//! `git archive` of that commit with symtn's own `Cargo.lock`, on macOS
//! aarch64, rustc 1.96.0. Nothing here runs symtn.
//!
//! symtn's quantities map onto this crate as follows; the consumer adapter
//! uses exactly these expressions:
//!
//! | symtn | here |
//! |---|---|
//! | `dim_scalar` | `Complex64::new(dim, 0.0)` |
//! | `inv_dim_scalar` | `1.0 / dim` |
//! | `sqrt_dim_scalar` | `dim.sqrt()` |
//! | `inv_sqrt_dim_scalar` | `1.0 / dim.sqrt()` |
//! | closed-form `f_symbol_scalar`, FS phase | real `f64`, imaginary part `+0.0` |
//!
//! The crate's own closed-form Fibonacci was removed (fusion-symbols #134);
//! [`TableCategory::fibonacci`] is the single Fibonacci path, and both
//! recorded providers are checked against it. The table rows must match bit
//! for bit. The closed-form rows differ by pinned ulp distances, by design:
//! symtn evaluated `1/φ` in f64 (one ulp below the correctly rounded
//! `F^{τττ}_τ[𝟙, 𝟙]`, so also `dim` and `inv_dim`) and the phases
//! `R^{ττ}_𝟙`, `R^{ττ}_τ`, `θ_τ` with `Complex64::from_polar`, whose bits
//! depend on the platform libm (the recorded macOS values). The test pins
//! exactly those distances, so any other change still fails.

use fusion_symbols::tables::{TableCategory, TableObject};
use num_complex::Complex64;

const VALUES: &str = include_str!("fixtures/tenet-957ccc05-values.txt");

/// The values one provider exposes, keyed by symtn's zero-based ids.
trait Provider {
    fn dual(&self, a: usize) -> usize;
    fn dim(&self, a: usize) -> f64;
    fn fs(&self, a: usize) -> Complex64;
    fn twist(&self, a: usize) -> Complex64;
    fn fusion(&self, a: usize, b: usize) -> Vec<usize>;
    fn n(&self, a: usize, b: usize, c: usize) -> u32;
    fn r(&self, a: usize, b: usize, c: usize) -> Complex64;
    fn f(&self, l: [usize; 6]) -> Complex64;
}

fn obj(id: usize) -> TableObject {
    TableObject(id as u16 + 1)
}

fn obj_id(a: TableObject) -> usize {
    usize::from(a.label()) - 1
}

impl Provider for TableCategory {
    fn dual(&self, a: usize) -> usize {
        obj_id(TableCategory::dual(self, obj(a)))
    }
    fn dim(&self, a: usize) -> f64 {
        TableCategory::dim(self, obj(a))
    }
    fn fs(&self, a: usize) -> Complex64 {
        self.frobenius_schur(obj(a))
    }
    fn twist(&self, a: usize) -> Complex64 {
        TableCategory::twist(self, obj(a))
    }
    fn fusion(&self, a: usize, b: usize) -> Vec<usize> {
        TableCategory::fusion(self, obj(a), obj(b))
            .map(obj_id)
            .collect()
    }
    fn n(&self, a: usize, b: usize, c: usize) -> u32 {
        self.nsymbol(obj(a), obj(b), obj(c))
    }
    fn r(&self, a: usize, b: usize, c: usize) -> Complex64 {
        self.r_symbol(obj(a), obj(b), obj(c))
    }
    fn f(&self, l: [usize; 6]) -> Complex64 {
        let [a, b, c, d, e, f] = l.map(obj);
        self.f_symbol(a, b, c, d, e, f)
    }
}

fn recorded(re: &str, im: &str) -> Complex64 {
    let bits = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).expect("hex bits"));
    Complex64::new(bits(re), bits(im))
}

fn ulps(a: f64, b: f64) -> u64 {
    let key = |x: f64| {
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    };
    key(a).abs_diff(key(b))
}

/// Exact `(re, im)` ulp distance of the table from symtn's recorded value.
/// The table rows are bit-identical. symtn's closed form computed `1/φ` in
/// f64 (one ulp below the correctly rounded F entries, hence `dim` and
/// `inv_dim`) and its phases with libm `from_polar`; the table's values are
/// correctly rounded. Every other value has distance `(0, 0)`.
fn known_distance(provider: &str, what: &str, labels: &[usize]) -> (u64, u64) {
    match (provider, what, labels) {
        ("closed", "R", [1, 1, 0]) => (1, 1),
        ("closed", "R", [1, 1, 1]) => (2, 1),
        ("closed", "twist", [1]) => (2, 2),
        ("closed", "dim" | "inv_dim", [1]) => (1, 0),
        ("closed", "F", [1, 1, 1, 1, 0, 0] | [1, 1, 1, 1, 1, 1]) => (1, 0),
        _ => (0, 0),
    }
}

fn assert_value(provider: &str, what: &str, labels: &[usize], got: Complex64, want: Complex64) {
    assert_eq!(
        (ulps(got.re, want.re), ulps(got.im, want.im)),
        known_distance(provider, what, labels),
        "{provider} {what}{labels:?}: {got} vs recorded {want}"
    );
}

#[test]
fn every_value_matches_tenet_957ccc05() {
    let table = TableCategory::fibonacci();
    let mut checked = 0usize;
    for line in VALUES.lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let p: &dyn Provider = match fields[0] {
            // The closed form was removed; its recorded values stay as an
            // oracle for the table, at the pinned distances.
            "closed" | "table" => &table,
            other => panic!("unknown provider {other}"),
        };
        let what = fields[1];
        let ids = |n: usize| -> Vec<usize> {
            fields[2..2 + n]
                .iter()
                .map(|s| s.parse().unwrap())
                .collect()
        };
        let value = || recorded(fields[fields.len() - 2], fields[fields.len() - 1]);
        let unary = |f: fn(f64) -> f64| {
            let a = ids(1);
            Complex64::new(f(p.dim(a[0])), 0.0)
        };
        let (labels, got): (Vec<usize>, Option<Complex64>) = match what {
            "dual" => {
                let a = ids(1);
                assert_eq!(p.dual(a[0]), fields[3].parse::<usize>().unwrap());
                (a, None)
            }
            "fusion" => {
                let a = ids(2);
                let want: Vec<usize> = fields[4].split(',').map(|s| s.parse().unwrap()).collect();
                assert_eq!(p.fusion(a[0], a[1]), want, "{} fusion{a:?}", fields[0]);
                (a, None)
            }
            "N" => {
                let a = ids(3);
                assert_eq!(p.n(a[0], a[1], a[2]), fields[5].parse::<u32>().unwrap());
                (a, None)
            }
            "dim" => (ids(1), Some(unary(|d| d))),
            "inv_dim" => (ids(1), Some(unary(|d| 1.0 / d))),
            "sqrt_dim" => (ids(1), Some(unary(f64::sqrt))),
            "inv_sqrt_dim" => (ids(1), Some(unary(|d| 1.0 / d.sqrt()))),
            "fs" => (ids(1), Some(p.fs(ids(1)[0]))),
            "twist" => (ids(1), Some(p.twist(ids(1)[0]))),
            "R" => {
                let a = ids(3);
                let r = p.r(a[0], a[1], a[2]);
                (a, Some(r))
            }
            "F" => {
                let a = ids(6);
                let f = p.f([a[0], a[1], a[2], a[3], a[4], a[5]]);
                (a, Some(f))
            }
            other => panic!("unknown quantity {other}"),
        };
        if let Some(got) = got {
            assert_value(fields[0], what, &labels, got, value());
        }
        checked += 1;
    }
    // Per provider: 2 x (dual, dim, inv_dim, sqrt_dim, inv_sqrt_dim, fs,
    // twist) + 4 fusion + 8 N + 8 R + 64 F = 98.
    assert_eq!(checked, 2 * 98);
}
