//! Product queries validate each leaf object a constant number of times, so the
//! number of leaf calls grows linearly with the number of leaf factors, for flat
//! and nested products alike.
use fusion_symbols::{
    category::{
        BraidingStyle, EnumerationError, FBlock, FusionCategory, FusionStyle, RBlock, SymbolBlock,
    },
    product,
    product::ProductError,
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
thread_local! { static CONTAINS: Cell<usize> = const { Cell::new(0) }; }
thread_local! { static CERTIFICATES: Cell<usize> = const { Cell::new(0) }; }
// Leaf enumeration and derived-dimension calls: object_count, object_at,
// object_index, then sqrt_dim, inv_sqrt_dim and inv_dim together.
thread_local! { static ENUMERATION: [Cell<usize>; 4] = const { [const { Cell::new(0) }; 4] }; }
fn bump_enumeration(i: usize) {
    ENUMERATION.with(|c| c[i].set(c[i].get() + 1));
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
    fn has_trivial_associator_gauge(&self) -> bool {
        CERTIFICATES.with(|c| c.set(c.get() + 1));
        true
    }
    fn has_canonical_unit(&self) -> bool {
        CERTIFICATES.with(|c| c.set(c.get() + 1));
        true
    }
    fn contains(&self, a: &u8) -> bool {
        CONTAINS.with(|c| c.set(c.get() + 1));
        *a == 0
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
    // The derived dimensions keep the default's single `dim` call, so the A/B
    // defaults still count three dimensions.
    fn sqrt_dim(&self, a: &u8) -> f64 {
        bump_enumeration(3);
        self.dim(a)
    }
    fn inv_sqrt_dim(&self, a: &u8) -> f64 {
        bump_enumeration(3);
        self.dim(a)
    }
    fn inv_dim(&self, a: &u8) -> f64 {
        bump_enumeration(3);
        self.dim(a)
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        bump_enumeration(0);
        Ok(Some(1))
    }
    fn object_at(&self, index: u64) -> Result<u8, EnumerationError> {
        bump_enumeration(1);
        if index == 0 {
            Ok(0)
        } else {
            Err(EnumerationError::OutOfRange)
        }
    }
    fn object_index(&self, a: &u8) -> Result<u64, EnumerationError> {
        check(a);
        bump_enumeration(2);
        Ok(0)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"counter"
    }
}

/// Leaf calls `[dim, nsymbol, braiding_style, queried]` for each object query.
fn costs<C: FusionCategory<Scalar = f64>>(cat: &C) -> [[usize; 4]; 11] {
    let u = cat.unit();
    take();
    let mut out = [[0; 4]; 11];
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
    let a = cat.a_symbol(&u, &u, &u).unwrap();
    record(9, a.as_slice()[0]);
    let b = cat.b_symbol(&u, &u, &u).unwrap();
    record(10, b.as_slice()[0]);
    out
}

/// Exact leaf-call counts for a product of `k` leaf factors. F, R, A and B
/// query N twice per leaf: once for the product shape and once to check that
/// leaf's block against its own multiplicities. A and B also count the leaf's
/// default derivation (one N, one F, three dimensions, duals).
fn linear(k: usize) -> [[usize; 4]; 11] {
    [
        [k, 0, 0, 0],             // dim
        [k, 0, 0, k],             // dual
        [k, 0, 0, k],             // frobenius_schur_phase
        [k, 0, 0, k],             // frobenius_schur_indicator
        [k, 0, k, k],             // twist
        [2 * k, 0, 0, k],         // fusion
        [3 * k, k, 0, 0],         // nsymbol
        [6 * k, 8 * k, 0, k],     // f_symbol
        [3 * k, 4 * k, k, k],     // r_symbol
        [6 * k, 5 * k, 0, 5 * k], // a_symbol (leaf default: dual, F, FS phase)
        [6 * k, 5 * k, 0, 4 * k], // b_symbol (leaf default: dual, F)
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
    assert!(std::panic::catch_unwind(|| cat.a_symbol(&bad, &u, &u).map(|_| ())).is_err());
    assert!(std::panic::catch_unwind(|| cat.b_symbol(&u, &bad, &u).map(|_| ())).is_err());
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
    fn a_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<RBlock<f64>, Infallible> {
        Ok(RBlock::one(1.0))
    }
    fn b_symbol(&self, _: &u8, _: &u8, _: &u8) -> Result<RBlock<f64>, Infallible> {
        Ok(RBlock::one(1.0))
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"malformed"
    }
}

// A malformed external leaf reports its immediate position through nesting.
#[test]
fn leaf_blocks_disagreeing_with_n_return_typed_errors() {
    for symbol in ['F', 'R', 'A', 'B'] {
        fn query<C: FusionCategory>(cat: &C, symbol: char) -> Result<(), C::Error> {
            let u = cat.unit();
            match symbol {
                'F' => cat.f_symbol(&u, &u, &u, &u, &u, &u).map(|_| ()),
                'R' => cat.r_symbol(&u, &u, &u).map(|_| ()),
                'A' => cat.a_symbol(&u, &u, &u).map(|_| ()),
                'B' => cat.b_symbol(&u, &u, &u).map(|_| ()),
                _ => unreachable!(),
            }
        }
        let expected = ProductError::MalformedSymbolData {
            factor: 1,
            expected: vec![2; if symbol == 'F' { 4 } else { 2 }],
            actual: vec![1; if symbol == 'F' { 4 } else { 2 }],
            storage_len: 1,
        };
        assert_eq!(
            query(&product(Counter, Malformed), symbol),
            Err(expected.clone())
        );
        assert_eq!(
            query(&product(product(Counter, Malformed), Counter), symbol),
            Err(ProductError::Factor0(expected))
        );
    }
}

/// Leaf `contains` calls and every other leaf call made by one product `contains`.
fn contains_cost<C: FusionCategory>(cat: &C, a: &C::Object) -> (bool, usize, [usize; 4]) {
    take();
    CONTAINS.with(|c| c.set(0));
    let result = cat.contains(a);
    (result, CONTAINS.with(|c| c.replace(0)), take())
}

#[test]
fn product_contains_checks_each_leaf_once_without_other_queries() {
    let c = Counter;
    let flat = product![c.clone(), c.clone(), c.clone(), c.clone()];
    assert_eq!(contains_cost(&flat, &flat.unit()), (true, 4, [0; 4]));
    let p2 = product(c.clone(), c.clone());
    let p4 = product(product(p2.clone(), c.clone()), c.clone());
    assert_eq!(contains_cost(&p4, &p4.unit()), (true, 4, [0; 4]));
    let balanced = product![p2.clone(), product![c.clone(), p2.clone(), c.clone()], p2];
    assert_eq!(
        contains_cost(&balanced, &balanced.unit()),
        (true, 8, [0; 4])
    );
    // A foreign leaf is rejected without a panic or any other leaf query.
    let (found, _, others) = contains_cost(&p4, &(((0, 1), 0), 0));
    assert!(!found);
    assert_eq!(others, [0; 4]);
}

/// Leaf certificate calls and every other leaf call made by both product
/// certificates.
fn certificate_cost<C: FusionCategory>(cat: &C) -> (bool, usize, [usize; 4], usize) {
    take();
    CONTAINS.with(|c| c.set(0));
    CERTIFICATES.with(|c| c.set(0));
    let result = cat.has_trivial_associator_gauge() && cat.has_canonical_unit();
    (
        result,
        CERTIFICATES.with(|c| c.replace(0)),
        take(),
        CONTAINS.with(|c| c.replace(0)),
    )
}

#[test]
fn product_certificates_ask_each_leaf_once_without_symbol_queries() {
    let c = Counter;
    let flat = product![c.clone(), c.clone(), c.clone(), c.clone()];
    assert_eq!(certificate_cost(&flat), (true, 8, [0; 4], 0));
    let p2 = product(c.clone(), c.clone());
    let p4 = product(product(p2.clone(), c.clone()), c.clone());
    assert_eq!(certificate_cost(&p4), (true, 8, [0; 4], 0));
    let balanced = product![p2.clone(), product![c.clone(), p2.clone(), c.clone()], p2];
    assert_eq!(certificate_cost(&balanced), (true, 16, [0; 4], 0));
}

#[test]
fn certificates_default_to_false() {
    assert!(!Malformed.has_trivial_associator_gauge());
    assert!(!Malformed.has_canonical_unit());
    // A non-certifying leaf makes the conjunction false.
    let cat = product(Counter, Malformed);
    assert!(!cat.has_trivial_associator_gauge() && !cat.has_canonical_unit());
}

/// Leaf calls `[[object_count, object_at, object_index, derived dimension],
/// [dim, nsymbol, braiding_style, other], contains]` for each enumeration and
/// derived-dimension query.
fn enumeration_costs<C: FusionCategory<Scalar = f64>>(
    cat: &C,
) -> [([usize; 4], [usize; 4], usize); 6] {
    let u = cat.unit();
    let mut out = [([0; 4], [0; 4], 0); 6];
    let reset = || {
        take();
        CONTAINS.with(|c| c.set(0));
        ENUMERATION.with(|c| c.each_ref().map(|x| x.replace(0)));
    };
    let read = || {
        (
            ENUMERATION.with(|c| c.each_ref().map(|x| x.replace(0))),
            take(),
            CONTAINS.with(|c| c.replace(0)),
        )
    };
    reset();
    assert_eq!(cat.object_count(), Ok(Some(1)));
    out[0] = read();
    assert_eq!(cat.object_at(0).as_ref(), Ok(&u));
    out[1] = read();
    assert_eq!(cat.object_index(&u), Ok(0));
    out[2] = read();
    assert_eq!(cat.sqrt_dim(&u), 1.0);
    out[3] = read();
    assert_eq!(cat.inv_sqrt_dim(&u), 1.0);
    out[4] = read();
    assert_eq!(cat.inv_dim(&u), 1.0);
    out[5] = read();
    out
}

/// Each query asks every leaf once: a product validates the whole object with
/// one `contains` per leaf, and a nested product neither re-validates nor
/// recomputes its factors' counts.
fn enumeration_linear(k: usize) -> [([usize; 4], [usize; 4], usize); 6] {
    [
        ([k, 0, 0, 0], [0; 4], 0),       // object_count
        ([k, k, 0, 0], [0; 4], 0),       // object_at
        ([k, 0, k, 0], [0; 4], k),       // object_index
        ([0, 0, 0, k], [k, 0, 0, 0], 0), // sqrt_dim
        ([0, 0, 0, k], [k, 0, 0, 0], 0), // inv_sqrt_dim
        ([0, 0, 0, k], [k, 0, 0, 0], 0), // inv_dim
    ]
}

#[test]
fn enumeration_and_derived_dimension_leaf_calls_are_linear() {
    let c = Counter;
    assert_eq!(
        enumeration_costs(&product(c.clone(), c.clone())),
        enumeration_linear(2)
    );
    assert_eq!(
        enumeration_costs(&product![c.clone(), c.clone(), c.clone(), c.clone()]),
        enumeration_linear(4)
    );
    let p2 = product(c.clone(), c.clone());
    let p3 = product(p2.clone(), c.clone());
    let p4 = product(p3.clone(), c.clone());
    let p5 = product(p4.clone(), c.clone());
    let p6 = product(p5.clone(), c.clone());
    assert_eq!(enumeration_costs(&p3), enumeration_linear(3));
    assert_eq!(enumeration_costs(&p6), enumeration_linear(6));
    let right = product(c.clone(), product(c.clone(), product(c.clone(), c.clone())));
    assert_eq!(enumeration_costs(&right), enumeration_linear(4));
    let balanced = product![p2.clone(), product![c.clone(), p2.clone(), c.clone()], p2];
    assert_eq!(enumeration_costs(&balanced), enumeration_linear(8));
}

#[test]
fn nested_foreign_objects_panic_in_enumeration_and_derived_dimensions() {
    let c = Counter;
    let cat = product(product(c.clone(), c.clone()), c);
    let bad = ((0, 1), 0);
    assert!(std::panic::catch_unwind(|| cat.object_index(&bad)).is_err());
    assert!(std::panic::catch_unwind(|| cat.sqrt_dim(&bad)).is_err());
    assert!(std::panic::catch_unwind(|| cat.inv_sqrt_dim(&bad)).is_err());
    assert!(std::panic::catch_unwind(|| cat.inv_dim(&bad)).is_err());
}
