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
//! Every entry must match bit for bit, with one portability exception: the
//! three closed-form values computed through `cos`/`sin` (`R^{ττ}_𝟙`,
//! `R^{ττ}_τ`, `θ_τ`) depend on the platform libm. For those the test
//! asserts bit identity with symtn's exact expression
//! `Complex64::from_polar(1.0, PI * k / 5.0)` evaluated on the running
//! platform, and at most one ulp from the recorded macOS bits.

use fusion_symbols::category_data::{CategoryDataFibonacci, CategoryObject};
use fusion_symbols::fibonacci::{
    fibonacci_f_symbol, fibonacci_nsymbol, fibonacci_r_symbol, FibonacciObject,
};
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

fn fib(id: usize) -> FibonacciObject {
    [FibonacciObject::Vacuum, FibonacciObject::Tau][id]
}

fn fib_id(a: FibonacciObject) -> usize {
    match a {
        FibonacciObject::Vacuum => 0,
        FibonacciObject::Tau => 1,
    }
}

struct Closed;

impl Provider for Closed {
    fn dual(&self, a: usize) -> usize {
        fib_id(fib(a).dual())
    }
    fn dim(&self, a: usize) -> f64 {
        fib(a).dim()
    }
    fn fs(&self, a: usize) -> Complex64 {
        fib(a).frobenius_schur().into()
    }
    fn twist(&self, a: usize) -> Complex64 {
        fib(a).twist()
    }
    fn fusion(&self, a: usize, b: usize) -> Vec<usize> {
        fib(a).fusion(fib(b)).iter().map(|&c| fib_id(c)).collect()
    }
    fn n(&self, a: usize, b: usize, c: usize) -> u32 {
        fibonacci_nsymbol(fib(a), fib(b), fib(c))
    }
    fn r(&self, a: usize, b: usize, c: usize) -> Complex64 {
        fibonacci_r_symbol(fib(a), fib(b), fib(c))
    }
    fn f(&self, l: [usize; 6]) -> Complex64 {
        let [a, b, c, d, e, f] = l.map(fib);
        fibonacci_f_symbol(a, b, c, d, e, f).into()
    }
}

fn obj(id: usize) -> CategoryObject {
    CategoryObject(id as u16 + 1)
}

fn obj_id(a: CategoryObject) -> usize {
    usize::from(a.label()) - 1
}

impl Provider for CategoryDataFibonacci {
    fn dual(&self, a: usize) -> usize {
        obj_id(CategoryDataFibonacci::dual(self, obj(a)))
    }
    fn dim(&self, a: usize) -> f64 {
        CategoryDataFibonacci::dim(self, obj(a))
    }
    fn fs(&self, a: usize) -> Complex64 {
        self.frobenius_schur(obj(a))
    }
    fn twist(&self, a: usize) -> Complex64 {
        CategoryDataFibonacci::twist(self, obj(a))
    }
    fn fusion(&self, a: usize, b: usize) -> Vec<usize> {
        CategoryDataFibonacci::fusion(self, obj(a), obj(b))
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

/// symtn's expression for a libm-dependent closed-form value, if `what` is one.
fn libm_expression(provider: &str, what: &str, labels: &[usize]) -> Option<Complex64> {
    let polar = |k: f64| Complex64::from_polar(1.0, std::f64::consts::PI * k / 5.0);
    match (provider, what, labels) {
        ("closed", "R", [1, 1, 0]) => Some(polar(4.0)),
        ("closed", "R", [1, 1, 1]) => Some(polar(-3.0)),
        ("closed", "twist", [1]) => Some(polar(-4.0)),
        _ => None,
    }
}

fn assert_value(provider: &str, what: &str, labels: &[usize], got: Complex64, want: Complex64) {
    let bits = |z: Complex64| (z.re.to_bits(), z.im.to_bits());
    if let Some(expression) = libm_expression(provider, what, labels) {
        assert_eq!(bits(got), bits(expression), "{provider} {what}{labels:?}");
        assert!(
            ulps(got.re, want.re) <= 1 && ulps(got.im, want.im) <= 1,
            "{provider} {what}{labels:?}: {got} vs recorded {want}"
        );
    } else {
        assert_eq!(
            bits(got),
            bits(want),
            "{provider} {what}{labels:?}: {got} vs {want}"
        );
    }
}

#[test]
fn every_value_is_bit_identical_to_tenet_957ccc05() {
    let table = CategoryDataFibonacci::try_new().expect("tables load");
    let mut checked = 0usize;
    for line in VALUES.lines().filter(|l| !l.starts_with('#')) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let p: &dyn Provider = match fields[0] {
            "closed" => &Closed,
            "table" => &table,
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
