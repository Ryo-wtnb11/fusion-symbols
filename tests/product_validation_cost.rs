//! Product queries validate each leaf object a constant number of times, so the
//! number of leaf calls grows linearly with the number of leaf factors, for flat
//! and nested products alike.
use fusion_symbols::{
    category::{BraidingStyle, FBlock, FusionCategory, FusionStyle, RBlock, SymbolBlock},
    product,
};
use std::cell::Cell;
use std::convert::Infallible;

// Leaf calls made by the product: dim, nsymbol, braiding_style, then the queried method.
thread_local! { static CALLS: [Cell<usize>; 4] = const { [const { Cell::new(0) }; 4] }; }
fn bump(i: usize) {
    CALLS.with(|c| c[i].set(c[i].get() + 1));
}
fn take() -> [usize; 4] {
    CALLS.with(|c| c.each_ref().map(|x| x.replace(0)))
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Counter;
// Validation inside the leaf is inline so that only the product's calls are counted.
fn check(a: &u8) {
    assert_eq!(*a, 0, "foreign counter object");
}
impl FusionCategory for Counter {
    type Object = u8;
    type Scalar = f64;
    type FSymbol = FBlock<f64>;
    type RSymbol = RBlock<f64>;
    type Fusion<'a> = std::iter::Once<u8>;
    type Error = Infallible;
    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Unique
    }
    fn braiding_style(&self) -> BraidingStyle {
        bump(2);
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> u8 {
        0
    }
    fn dual(&self, a: &u8) -> u8 {
        check(a);
        bump(3);
        0
    }
    fn dim(&self, a: &u8) -> f64 {
        check(a);
        bump(0);
        1.0
    }
    fn frobenius_schur_phase(&self, a: &u8) -> f64 {
        check(a);
        bump(3);
        1.0
    }
    fn frobenius_schur_indicator(&self, a: &u8) -> f64 {
        check(a);
        bump(3);
        1.0
    }
    fn twist(&self, a: &u8) -> f64 {
        check(a);
        bump(3);
        1.0
    }
    fn fusion<'a>(&'a self, a: &u8, b: &u8) -> Result<Self::Fusion<'a>, Infallible> {
        [a, b].into_iter().for_each(check);
        bump(3);
        Ok(std::iter::once(0))
    }
    fn nsymbol(&self, a: &u8, b: &u8, c: &u8) -> u32 {
        [a, b, c].into_iter().for_each(check);
        bump(1);
        1
    }
    fn f_symbol(
        &self,
        a: &u8,
        b: &u8,
        c: &u8,
        d: &u8,
        e: &u8,
        f: &u8,
    ) -> Result<FBlock<f64>, Infallible> {
        [a, b, c, d, e, f].into_iter().for_each(check);
        bump(3);
        Ok(FBlock::one(1.0))
    }
    fn r_symbol(&self, a: &u8, b: &u8, c: &u8) -> Result<RBlock<f64>, Infallible> {
        [a, b, c].into_iter().for_each(check);
        bump(3);
        Ok(RBlock::one(1.0))
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"counter"
    }
}

/// Leaf calls `[dim, nsymbol, braiding_style, queried]` for each object query.
fn costs<C: FusionCategory<Scalar = f64>>(cat: &C) -> [[usize; 4]; 9] {
    let u = cat.unit();
    take();
    let mut out = [[0; 4]; 9];
    let mut record = |i: usize, value: f64| {
        assert_eq!(value, 1.0);
        out[i] = take();
    };
    record(0, cat.dim(&u));
    record(1, (cat.dual(&u) == u) as u8 as f64);
    record(2, cat.frobenius_schur_phase(&u));
    record(3, cat.frobenius_schur_indicator(&u));
    record(4, cat.twist(&u));
    record(5, cat.fusion(&u, &u).unwrap().count() as f64);
    record(6, cat.nsymbol(&u, &u, &u) as f64);
    let f = cat.f_symbol(&u, &u, &u, &u, &u, &u).unwrap();
    record(7, f.as_slice()[0]);
    let r = cat.r_symbol(&u, &u, &u).unwrap();
    record(8, r.as_slice()[0]);
    out
}

/// Exact leaf-call counts for a product of `k` leaf factors. F and R query N
/// twice per leaf: once for the product shape and once to check that leaf's
/// block against its own multiplicities.
fn linear(k: usize) -> [[usize; 4]; 9] {
    [
        [k, 0, 0, 0],         // dim
        [k, 0, 0, k],         // dual
        [k, 0, 0, k],         // frobenius_schur_phase
        [k, 0, 0, k],         // frobenius_schur_indicator
        [k, 0, k, k],         // twist
        [2 * k, 0, 0, k],     // fusion
        [3 * k, k, 0, 0],     // nsymbol
        [6 * k, 8 * k, 0, k], // f_symbol
        [3 * k, 4 * k, k, k], // r_symbol
    ]
}

#[test]
fn flat_product_leaf_calls_are_linear() {
    let c = Counter;
    assert_eq!(costs(&product(c.clone(), c.clone())), linear(2));
    assert_eq!(costs(&product![c.clone(), c.clone(), c.clone()]), linear(3));
    assert_eq!(
        costs(&product![c.clone(), c.clone(), c.clone(), c.clone()]),
        linear(4)
    );
    assert_eq!(
        costs(&product![
            c.clone(),
            c.clone(),
            c.clone(),
            c.clone(),
            c.clone()
        ]),
        linear(5)
    );
    assert_eq!(
        costs(&product![
            c.clone(),
            c.clone(),
            c.clone(),
            c.clone(),
            c.clone(),
            c.clone()
        ]),
        linear(6)
    );
}

#[test]
fn nested_product_leaf_calls_are_linear() {
    let c = Counter;
    let p2 = product(c.clone(), c.clone());
    let p3 = product(p2.clone(), c.clone());
    let p4 = product(p3.clone(), c.clone());
    let p5 = product(p4.clone(), c.clone());
    let p6 = product(p5.clone(), c.clone());
    assert_eq!(costs(&p2), linear(2));
    assert_eq!(costs(&p3), linear(3));
    assert_eq!(costs(&p4), linear(4));
    assert_eq!(costs(&p5), linear(5));
    assert_eq!(costs(&p6), linear(6));
    // Right-nested and balanced trees follow the same leaf count.
    let right = product(c.clone(), product(c.clone(), product(c.clone(), c.clone())));
    assert_eq!(costs(&right), linear(4));
    let balanced = product![p2.clone(), product![c.clone(), p2.clone(), c.clone()], p2];
    assert_eq!(costs(&balanced), linear(8));
}

#[test]
fn nested_foreign_objects_still_panic() {
    let c = Counter;
    let cat = product(product(c.clone(), c.clone()), c);
    let u = cat.unit();
    let bad = ((0, 1), 0);
    assert!(std::panic::catch_unwind(|| cat.dual(&bad)).is_err());
    assert!(std::panic::catch_unwind(|| cat.twist(&bad)).is_err());
    assert!(std::panic::catch_unwind(|| cat.fusion(&u, &bad).map(|_| ())).is_err());
    assert!(std::panic::catch_unwind(|| cat.nsymbol(&u, &u, &bad)).is_err());
    assert!(
        std::panic::catch_unwind(|| cat.f_symbol(&u, &u, &u, &u, &u, &bad).map(|_| ())).is_err()
    );
    assert!(std::panic::catch_unwind(|| cat.r_symbol(&u, &u, &bad).map(|_| ())).is_err());
}

/// Reports `N = 2` everywhere but returns 1x1 blocks.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Malformed;
impl FusionCategory for Malformed {
    type Object = u8;
    type Scalar = f64;
    type FSymbol = FBlock<f64>;
    type RSymbol = RBlock<f64>;
    type Fusion<'a> = std::iter::Once<u8>;
    type Error = Infallible;
    fn fusion_style(&self) -> FusionStyle {
        FusionStyle::Generic
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> u8 {
        0
    }
    fn dual(&self, _: &u8) -> u8 {
        0
    }
    fn dim(&self, a: &u8) -> f64 {
        check(a);
        1.0
    }
    fn frobenius_schur_phase(&self, _: &u8) -> f64 {
        1.0
    }
    fn frobenius_schur_indicator(&self, _: &u8) -> f64 {
        1.0
    }
    fn twist(&self, _: &u8) -> f64 {
        1.0
    }
    fn fusion<'a>(&'a self, _: &u8, _: &u8) -> Result<Self::Fusion<'a>, Infallible> {
        Ok(std::iter::once(0))
    }
    fn nsymbol(&self, _: &u8, _: &u8, _: &u8) -> u32 {
        2
    }
    fn f_symbol(
        &self,
        _: &u8,
        _: &u8,
        _: &u8,
        _: &u8,
        _: &u8,
        _: &u8,
    ) -> Result<FBlock<f64>, Infallible> {
        Ok(FBlock::one(1.0))
    }
    fn r_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<RBlock<f64>, Infallible> {
        Ok(RBlock::one(1.0))
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"malformed"
    }
}

fn panic_message(result: std::thread::Result<()>) -> String {
    let payload = result.expect_err("malformed leaf block must panic");
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

// Not a debug assertion: a leaf block whose shape disagrees with its N symbols
// would otherwise be combined into a wrong-shaped product block.
#[test]
fn leaf_blocks_disagreeing_with_n_panic_in_every_build() {
    for nested in [false, true] {
        let flat = product(Counter, Malformed);
        let deep = product(product(Counter, Malformed), Counter);
        let f = |nested: bool| {
            if nested {
                let u = deep.unit();
                deep.f_symbol(&u, &u, &u, &u, &u, &u).map(|_| ()).unwrap()
            } else {
                let u = flat.unit();
                flat.f_symbol(&u, &u, &u, &u, &u, &u).map(|_| ()).unwrap()
            }
        };
        let r = |nested: bool| {
            if nested {
                let u = deep.unit();
                deep.r_symbol(&u, &u, &u).map(|_| ()).unwrap()
            } else {
                let u = flat.unit();
                flat.r_symbol(&u, &u, &u).map(|_| ()).unwrap()
            }
        };
        let message = panic_message(std::panic::catch_unwind(|| f(nested)));
        assert!(message.contains("F block shape disagrees"), "{message}");
        let message = panic_message(std::panic::catch_unwind(|| r(nested)));
        assert!(message.contains("R block shape disagrees"), "{message}");
    }
}
