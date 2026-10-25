//! Object enumeration and derived dimensions. Expected sequences are literals
//! hand-evaluated from TensorKitSectors 0.3.10 (`sectors.jl`, `product.jl`,
//! `auxiliary.jl::to_manhattan_index`) and SUNRepresentations 0.4.0
//! (`sector.jl::getindex`), restricted to fusion-symbols' label domains.
use crate::common::lie::{torus, u1_irrep};
use fusion_symbols::{
    category::{EnumerationError, FusionCategory},
    fermion::{FermionParity, FermionParityIrrep as Parity},
    finite::{ZNIrrep, ZN},
    lie::{CU1Irrep, GroupId, LieGroup, CU1},
    product,
    product::Product,
    table::TableCategory,
};
use EnumerationError::{NotEnumerable, OutOfRange, Overflow};

fn first<C: FusionCategory>(cat: &C, n: u64) -> Vec<C::Object> {
    (0..n).map(|k| cat.object_at(k).unwrap()).collect()
}
fn round_trip<C: FusionCategory>(cat: &C, n: u64) {
    for k in 0..n {
        let a = cat.object_at(k).unwrap();
        assert!(cat.contains(&a));
        assert_eq!(cat.object_index(&a), Ok(k), "{a:?}");
    }
}
fn zn(charge: u32, modulus: u32) -> ZNIrrep {
    ZNIrrep::from_canonical(charge, modulus).unwrap()
}
fn q(charge: i32) -> fusion_symbols::lie::LieIrrep {
    u1_irrep(charge).unwrap()
}
fn lie(group: GroupId) -> LieGroup {
    LieGroup::new(group).unwrap()
}
fn panics(f: impl FnOnce() + std::panic::UnwindSafe) -> bool {
    std::panic::catch_unwind(f).is_err()
}

#[test]
fn cyclic_and_parity_orders() {
    let z1 = ZN::new(1).unwrap();
    assert_eq!(z1.object_count(), Ok(Some(1)));
    assert_eq!(first(&z1, 1), [zn(0, 1)]);
    assert_eq!(z1.object_at(1), Err(OutOfRange));
    let z3 = ZN::new(3).unwrap();
    assert_eq!(z3.object_count(), Ok(Some(3)));
    assert_eq!(first(&z3, 3), [zn(0, 3), zn(1, 3), zn(2, 3)]);
    assert_eq!(z3.object_at(3), Err(OutOfRange));
    assert_eq!(z3.object_at(u64::MAX), Err(OutOfRange));
    round_trip(&z3, 3);

    assert_eq!(FermionParity.object_count(), Ok(Some(2)));
    assert_eq!(first(&FermionParity, 2), [Parity::EVEN, Parity::ODD]);
    assert_eq!(FermionParity.object_at(2), Err(OutOfRange));
    round_trip(&FermionParity, 2);
}

#[test]
fn u1_order_and_label_range() {
    assert_eq!(torus().object_count(), Ok(None));
    assert_eq!(first(&torus(), 5), [q(0), q(1), q(-1), q(2), q(-2)]);
    assert_eq!(torus().object_index(&q(i32::MAX)), Ok(4_294_967_293));
    assert_eq!(torus().object_index(&q(-i32::MAX)), Ok(4_294_967_294));
    assert_eq!(torus().object_at(4_294_967_294), Ok(q(-i32::MAX)));
    assert_eq!(torus().object_at(4_294_967_295), Err(Overflow));
    assert_eq!(torus().object_at(u64::MAX), Err(Overflow));
    round_trip(&torus(), 100);
}

#[test]
fn cu1_order_and_label_range() {
    let c = |t, s| CU1Irrep::new(t, s).unwrap();
    assert_eq!(CU1.object_count(), Ok(None));
    assert_eq!(
        first(&CU1, 5),
        [c(0, 0), c(0, 1), c(1, 2), c(2, 2), c(3, 2)]
    );
    assert_eq!(CU1.object_index(&c(u32::MAX, 2)), Ok(4_294_967_296));
    assert_eq!(CU1.object_at(4_294_967_296), Ok(c(u32::MAX, 2)));
    assert_eq!(CU1.object_at(4_294_967_297), Err(Overflow));
    round_trip(&CU1, 100);
}

#[test]
fn rank_one_lie_groups_enumerate_admitted_doubled_spins() {
    let dj = |g: &LieGroup, n| -> Vec<i64> { first(g, n).iter().map(|a| a.dynkin()[0]).collect() };
    for group in [
        GroupId::su(2).unwrap(),
        GroupId::spin(3).unwrap(),
        GroupId::sp(2).unwrap(),
    ] {
        let g = lie(group);
        assert_eq!(g.object_count(), Ok(None));
        assert_eq!(dj(&g, 4), [0, 1, 2, 3]);
        assert_eq!(
            g.object_index(&g.irrep(&[u32::MAX.into()], &[]).unwrap()),
            Ok(4_294_967_295)
        );
        assert_eq!(g.object_at(4_294_967_296), Err(Overflow));
        round_trip(&g, 50);
    }
    for group in [GroupId::so(3).unwrap(), GroupId::su_quotient(2, 2).unwrap()] {
        let g = lie(group);
        assert_eq!(g.object_count(), Ok(None));
        assert_eq!(dj(&g, 3), [0, 2, 4]);
        assert_eq!(g.object_index(&g.irrep(&[4], &[]).unwrap()), Ok(2));
        assert_eq!(
            g.object_at(2_147_483_647),
            g.irrep(&[4_294_967_294], &[]).map_err(|_| Overflow)
        );
        assert_eq!(g.object_at(2_147_483_648), Err(Overflow));
        round_trip(&g, 50);
    }
}

#[cfg(feature = "lie-generated")]
#[test]
fn su_n_uses_total_grade_then_lexicographic_dynkin_order() {
    let su3 = lie(GroupId::su(3).unwrap());
    assert_eq!(su3.object_count(), Ok(None));
    let labels: Vec<_> = first(&su3, 6).iter().map(|a| a.dynkin()).collect();
    assert_eq!(
        labels,
        [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]].map(Vec::from)
    );
    let far = su3.irrep(&[0, 1_000_000], &[]).unwrap();
    assert_eq!(su3.object_index(&far), Ok(500_000_500_000));
    assert_eq!(su3.object_at(500_000_500_000), Ok(far));
    round_trip(&su3, 60);
    // SU(4): shell 1 is (0,0,1), (0,1,0), (1,0,0).
    let su4 = lie(GroupId::su(4).unwrap());
    let labels: Vec<_> = first(&su4, 5).iter().map(|a| a.dynkin()).collect();
    assert_eq!(
        labels,
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 2]].map(Vec::from)
    );
    round_trip(&su4, 60);
    // SU(3)/Z_1 is SU(3).
    let trivial_quotient = lie(GroupId::su_quotient(3, 1).unwrap());
    assert_eq!(trivial_quotient.object_at(4).unwrap().dynkin(), vec![1, 1]);
}

#[cfg(feature = "lie-generated")]
#[test]
fn other_global_forms_and_bcd_are_not_enumerable() {
    for group in [
        GroupId::psu(3).unwrap(),
        GroupId::so(5).unwrap(),
        GroupId::spin(5).unwrap(),
        GroupId::sp(4).unwrap(),
        GroupId::so(6).unwrap(),
        GroupId::spin(7).unwrap(),
    ] {
        let g = lie(group.clone());
        assert_eq!(g.object_count(), Err(NotEnumerable), "{group:?}");
        assert_eq!(g.object_at(0), Err(NotEnumerable));
        assert_eq!(g.object_index(&g.unit()), Err(NotEnumerable));
    }
}

#[cfg(feature = "lie-generated")]
#[test]
fn foreign_objects_panic_before_the_capability_check() {
    let psu3 = lie(GroupId::psu(3).unwrap());
    let su3 = lie(GroupId::su(3).unwrap());
    let foreign = su3.unit();
    assert!(panics(|| {
        let _ = psu3.object_index(&foreign);
    }));
    let p = product(ZN::new(2).unwrap(), psu3);
    let bad = (zn(0, 3), p.factors().1.unit());
    assert!(panics(|| {
        let _ = p.object_index(&bad);
    }));
    assert_eq!(p.object_index(&p.unit()), Err(NotEnumerable));
}

#[test]
fn foreign_leaf_objects_panic() {
    let z2 = ZN::new(2).unwrap();
    assert!(panics(|| {
        let _ = z2.object_index(&zn(0, 3));
    }));
    let p = product(product(z2, torus()), z2);
    assert!(panics(|| {
        let _ = p.object_index(&((zn(0, 2), q(0)), zn(1, 3)));
    }));
}

#[test]
fn table_source_order() {
    let fib = TableCategory::fibonacci();
    assert_eq!(fib.object_count(), Ok(Some(2)));
    assert_eq!(
        first(&fib, 2),
        [fib.object(1).unwrap(), fib.object(2).unwrap()]
    );
    assert!((fib.dim(&fib.object_at(1).unwrap()) - (1.0 + 5f64.sqrt()) / 2.0).abs() < 1e-15);
    assert_eq!(fib.object_at(2), Err(OutOfRange));
    // fusion-symbols' Ising source order is (I, ψ, σ); TensorKitSectors' is (I, σ, ψ).
    let ising = TableCategory::ising();
    assert_eq!(ising.object_count(), Ok(Some(3)));
    let dims: Vec<f64> = first(&ising, 3).iter().map(|a| ising.dim(a)).collect();
    assert_eq!(dims[..2], [1.0, 1.0]);
    assert!((dims[2] - 2f64.sqrt()).abs() < 1e-15);
    assert_eq!(ising.object_index(&ising.object(3).unwrap()), Ok(2));
    assert_eq!(ising.object_at(3), Err(OutOfRange));
    round_trip(&ising, 3);
}

#[test]
fn bounded_products_follow_manhattan_order() {
    let z = |n| ZN::new(n).unwrap();
    // Bounds (2, 3).
    let p = product(z(2), z(3));
    assert_eq!(p.object_count(), Ok(Some(6)));
    let want = [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)];
    assert_eq!(first(&p, 6), want.map(|(a, b)| (zn(a, 2), zn(b, 3))));
    assert_eq!(p.object_at(6), Err(OutOfRange));
    round_trip(&p, 6);
    // Bounds (2, ∞): U(1) component indices 0, 1, 2, 3 are charges 0, 1, -1, 2.
    let p = product(z(2), torus());
    assert_eq!(p.object_count(), Ok(None));
    let want = [(0, 0), (0, 1), (1, 0), (0, -1), (1, 1), (0, 2), (1, -1)];
    assert_eq!(first(&p, 7), want.map(|(a, b)| (zn(a, 2), q(b))));
    round_trip(&p, 200);
    // Three unbounded factors.
    let p = product![torus(), torus(), torus()];
    assert_eq!(p.object_count(), Ok(None));
    let want = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (0, 0, -1),
        (0, 1, 1),
    ];
    assert_eq!(first(&p, 6), want.map(|(a, b, c)| (q(a), q(b), q(c))));
    round_trip(&p, 200);
}

#[test]
fn explicit_nesting_changes_the_index() {
    assert_eq!(
        product![torus(), torus(), torus()].object_index(&(q(1), q(0), q(0))),
        Ok(3)
    );
    let nested = product(product(torus(), torus()), torus());
    assert_eq!(nested.object_index(&((q(1), q(0)), q(0))), Ok(5));
    assert_eq!(nested.object_at(5), Ok(((q(1), q(0)), q(0))));
    round_trip(&nested, 200);
    let right = product(FermionParity, product(ZN::new(3).unwrap(), torus()));
    round_trip(&right, 200);
}

macro_rules! first_shell {
    ($($i:tt),+) => {{
        let z = ZN::new(2).unwrap();
        let cat = Product::new(($({ let _ = $i; z },)+));
        let n = [$($i),+].len();
        let unit = cat.object_at(0).unwrap();
        assert_eq!([$(unit.$i.charge()),+], [0; [$($i),+].len()]);
        for j in 0..n {
            let a = cat.object_at(1 + j as u64).unwrap();
            let mut want = [0; [$($i),+].len()];
            want[n - 1 - j] = 1;
            assert_eq!([$(a.$i.charge()),+], want, "arity {n}, index {}", 1 + j);
            assert_eq!(cat.object_index(&a), Ok(1 + j as u64));
        }
        cat
    }};
}

#[test]
fn first_shell_for_every_arity() {
    first_shell!(0, 1);
    first_shell!(0, 1, 2);
    first_shell!(0, 1, 2, 3);
    first_shell!(0, 1, 2, 3, 4);
    first_shell!(0, 1, 2, 3, 4, 5);
    first_shell!(0, 1, 2, 3, 4, 5, 6);
    first_shell!(0, 1, 2, 3, 4, 5, 6, 7);
    first_shell!(0, 1, 2, 3, 4, 5, 6, 7, 8);
    first_shell!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
    first_shell!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    let twelve = first_shell!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    assert_eq!(twelve.object_count(), Ok(Some(4096)));
    let one = zn(1, 2);
    let ones = (one, one, one, one, one, one, one, one, one, one, one, one);
    assert_eq!(twelve.object_index(&ones), Ok(4095));
    assert_eq!(twelve.object_at(4095), Ok(ones));
    assert_eq!(twelve.object_at(4096), Err(OutOfRange));
}

#[test]
fn counts_above_u64_keep_indices_addressable() {
    let z = ZN::new(65536).unwrap();
    let p = product![z, z, z, z];
    assert_eq!(p.object_count(), Err(Overflow));
    let top = zn(65535, 65536);
    assert_eq!(p.object_index(&(top, top, top, top)), Ok(u64::MAX));
    assert_eq!(p.object_at(u64::MAX), Ok((top, top, top, top)));
    // Nested inside another product, the exact count 2^64 still bounds the
    // first coordinate: tuple (1, 1) of bounds (2^64, 2) has index 3.
    let outer = product(p, ZN::new(2).unwrap());
    assert_eq!(outer.object_count(), Err(Overflow));
    let o = zn(0, 65536);
    let a = ((o, o, o, zn(1, 65536)), zn(1, 2));
    assert_eq!(outer.object_index(&a), Ok(3));
    assert_eq!(outer.object_at(3), Ok(a));
}

#[test]
fn products_with_a_non_enumerable_factor_are_not_enumerable() {
    #[cfg(feature = "lie-generated")]
    {
        let p = product(torus(), lie(GroupId::psu(3).unwrap()));
        assert_eq!(p.object_count(), Err(NotEnumerable));
        assert_eq!(p.object_at(0), Err(NotEnumerable));
        assert_eq!(p.object_index(&p.unit()), Err(NotEnumerable));
    }
    // NotEnumerable wins over a factor's label overflow.
    #[cfg(feature = "lie-generated")]
    {
        let p = product(torus(), lie(GroupId::psu(3).unwrap()));
        assert_eq!(p.object_at(u64::MAX), Err(NotEnumerable));
    }
    // (0, 2^32 - 1) opens shell 2^32 - 1, at index C(2^32, 2); its second
    // component is past U(1)'s labels.
    let p = product(torus(), torus());
    assert_eq!(p.object_at(9_223_372_034_707_292_160), Err(Overflow));
    assert_eq!(
        p.object_at(9_223_372_034_707_292_159),
        Ok((q(-i32::MAX), q(0)))
    );
}

#[test]
fn derived_dimensions_multiply_factorwise() {
    let su2 = lie(GroupId::su(2).unwrap());
    let fib = TableCategory::fibonacci();
    let p = product(su2.clone(), fib.clone());
    let a = (su2.irrep(&[1], &[]).unwrap(), fib.object(2).unwrap());
    assert_eq!(p.sqrt_dim(&a), 1.7989074399478675);
    assert_eq!(p.inv_sqrt_dim(&a), 0.5558929702514211);
    assert_eq!(p.inv_dim(&a), 0.30901699437494745);
    // Leaf defaults.
    assert_eq!(su2.sqrt_dim(&a.0), 2f64.sqrt());
    assert_eq!(su2.inv_sqrt_dim(&a.0), 0.7071067811865475);
    assert_eq!(su2.inv_dim(&a.0), 0.5);
    // Left to right through explicit nesting.
    let nested = product(product(su2.clone(), fib.clone()), CU1);
    let b = (a.clone(), CU1Irrep::from_twice_charge(3));
    assert_eq!(nested.sqrt_dim(&b), 1.7989074399478675 * 2f64.sqrt());
    assert_eq!(nested.inv_dim(&b), 0.30901699437494745 * 0.5);
}

#[test]
fn enumeration_error_is_a_plain_error() {
    fn traits<T: Copy + Eq + std::fmt::Debug + std::fmt::Display + std::error::Error>() {}
    traits::<EnumerationError>();
    for e in [NotEnumerable, OutOfRange, Overflow] {
        assert!(!e.to_string().is_empty());
    }
}

/// A downstream category forwarding to another, which can report a count above
/// `u64::MAX` only as `Overflow`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Forward<C>(C);
impl<C: FusionCategory> FusionCategory for Forward<C> {
    type Object = C::Object;
    type Scalar = C::Scalar;
    type FSymbol = C::FSymbol;
    type RSymbol = C::RSymbol;
    type Error = C::Error;
    type Fusion<'a>
        = C::Fusion<'a>
    where
        C: 'a;
    fn fusion_style(&self) -> fusion_symbols::category::FusionStyle {
        self.0.fusion_style()
    }
    fn braiding_style(&self) -> fusion_symbols::category::BraidingStyle {
        self.0.braiding_style()
    }
    fn unit(&self) -> C::Object {
        self.0.unit()
    }
    fn contains(&self, a: &C::Object) -> bool {
        self.0.contains(a)
    }
    fn dual(&self, a: &C::Object) -> C::Object {
        self.0.dual(a)
    }
    fn dim(&self, a: &C::Object) -> f64 {
        self.0.dim(a)
    }
    fn frobenius_schur_phase(&self, a: &C::Object) -> C::Scalar {
        self.0.frobenius_schur_phase(a)
    }
    fn frobenius_schur_indicator(&self, a: &C::Object) -> C::Scalar {
        self.0.frobenius_schur_indicator(a)
    }
    fn twist(&self, a: &C::Object) -> C::Scalar {
        self.0.twist(a)
    }
    fn fusion<'a>(&'a self, a: &C::Object, b: &C::Object) -> Result<C::Fusion<'a>, C::Error> {
        self.0.fusion(a, b)
    }
    fn nsymbol(&self, a: &C::Object, b: &C::Object, c: &C::Object) -> u32 {
        self.0.nsymbol(a, b, c)
    }
    fn f_symbol(
        &self,
        a: &C::Object,
        b: &C::Object,
        c: &C::Object,
        d: &C::Object,
        e: &C::Object,
        f: &C::Object,
    ) -> Result<C::FSymbol, C::Error> {
        self.0.f_symbol(a, b, c, d, e, f)
    }
    fn r_symbol(
        &self,
        a: &C::Object,
        b: &C::Object,
        c: &C::Object,
    ) -> Result<C::RSymbol, C::Error> {
        self.0.r_symbol(a, b, c)
    }
    fn authority_fingerprint(&self) -> &[u8] {
        self.0.authority_fingerprint()
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        self.0.object_count()
    }
    fn object_at(&self, index: u64) -> Result<C::Object, EnumerationError> {
        self.0.object_at(index)
    }
    fn object_index(&self, a: &C::Object) -> Result<u64, EnumerationError> {
        self.0.object_index(a)
    }
}

#[test]
fn external_counts_above_u64_keep_product_indices_addressable() {
    let z = ZN::new(65536).unwrap();
    let inner = product![z, z, z, z];
    let wrapped = Forward(inner.clone());
    assert_eq!(wrapped.object_count(), Err(Overflow));
    let direct = product(inner, ZN::new(2).unwrap());
    let outer = product(wrapped.clone(), ZN::new(2).unwrap());
    assert_eq!(outer.object_count(), Err(Overflow));
    // Same literals as the exact nested count: tuple (1, 1) of bounds (2^64, 2).
    let o = zn(0, 65536);
    let a = ((o, o, o, zn(1, 65536)), zn(1, 2));
    assert_eq!(outer.object_index(&a), Ok(3));
    assert_eq!(outer.object_at(3), Ok(a));
    assert_eq!(outer.object_at(0), Ok(outer.unit()));
    for k in [0, 1, 2, 5, 1 << 40, u64::MAX - 1, u64::MAX] {
        let b = direct.object_at(k).unwrap();
        assert_eq!(outer.object_at(k).as_ref(), Ok(&b), "{k}");
        assert_eq!(outer.object_index(&b), Ok(k));
    }
    // Shell 2^64: the index is past u64 whatever the exact bound.
    let top = zn(65535, 65536);
    let past = ((top, top, top, top), zn(1, 2));
    assert_eq!(direct.object_index(&past), Err(Overflow));
    assert_eq!(outer.object_index(&past), Err(Overflow));
    // An infinite factor dominates an overflowing count.
    assert_eq!(product(wrapped.clone(), torus()).object_count(), Ok(None));
    assert_eq!(
        product![torus(), wrapped.clone(), ZN::new(2).unwrap()].object_count(),
        Ok(None)
    );
    let p = product(wrapped, torus());
    let b = p.object_at(1 << 40).unwrap();
    assert_eq!(p.object_index(&b), Ok(1 << 40));
}
