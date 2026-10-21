use fusion_symbols::{
    abelian::{U1Irrep, U1, ZN},
    category::{BraidingStyle, FBlock, FusionCategory, FusionStyle, RBlock, SymbolBlock},
    fermion::{FermionParity, FermionParityIrrep as Parity},
    lie::{GroupId, LieGroup},
    product,
    product::{Product, ProductError},
    table::TableCategory,
};
use num_complex::Complex64;
use std::{
    cell::Cell,
    convert::Infallible,
    hash::{DefaultHasher, Hash, Hasher},
};

fn su2() -> LieGroup {
    LieGroup::new(GroupId::su(2).unwrap()).unwrap()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}
fn complex_close(a: Complex64, b: Complex64) {
    assert!((a - b).norm() < 1e-12, "{a} != {b}");
}

#[test]
fn fermionic_closed_forms_and_complex_promotion() {
    let fu1 = product(U1, FermionParity);
    let a = (U1Irrep::new(1).unwrap(), Parity::ODD);
    let b = (U1Irrep::new(-1).unwrap(), Parity::ODD);
    assert_eq!(fu1.dual(&a), b);
    assert_eq!(
        fu1.fusion(&a, &b).unwrap().collect::<Vec<_>>(),
        [fu1.unit()]
    );
    assert_eq!(fu1.r_symbol(&a, &b, &fu1.unit()).unwrap().at(0, 0), -1.0);
    assert_eq!(fu1.twist(&a), -1.0);
    assert_eq!(fu1.frobenius_schur_indicator(&a), 0.0);
    let fsu2 = product(su2(), FermionParity);
    let half = (fsu2.factors().0.irrep(&[1]).unwrap(), Parity::ODD);
    let triplet = (fsu2.factors().0.irrep(&[2]).unwrap(), Parity::EVEN);
    // SU(2): R^{1/2,1/2}_{j}=(-1)^{1-j}; parity supplies one more minus sign.
    assert_eq!(
        fsu2.r_symbol(&half, &half, &fsu2.unit()).unwrap().at(0, 0),
        1.0
    );
    assert_eq!(
        fsu2.r_symbol(&half, &half, &triplet).unwrap().at(0, 0),
        -1.0
    );
    assert_eq!(fsu2.frobenius_schur_phase(&half), -1.0);
    assert_eq!(fsu2.frobenius_schur_indicator(&half), -1.0);
    assert_eq!(fsu2.twist(&half), -1.0);
    close(
        fsu2.f_symbol(&half, &half, &half, &half, &fsu2.unit(), &fsu2.unit())
            .unwrap()
            .at(0, 0, 0, 0),
        -0.5,
    );
    let ff = product(FermionParity, FermionParity);
    assert_eq!(
        ff.r_symbol(
            &(Parity::ODD, Parity::EVEN),
            &(Parity::EVEN, Parity::ODD),
            &(Parity::ODD, Parity::ODD)
        )
        .unwrap()
        .at(0, 0),
        1.0
    );

    let fibsu2 = product(TableCategory::fibonacci(), su2());
    let tau = fibsu2.factors().0.object(2).unwrap();
    let h = fibsu2.factors().1.irrep(&[1]).unwrap();
    let x = (tau, h);
    // Fibonacci: R^{ττ}_1 = exp(+4πi/5), θ_τ = exp(-4πi/5), F[1,1]=1/φ.
    complex_close(
        fibsu2.r_symbol(&x, &x, &fibsu2.unit()).unwrap().at(0, 0),
        -Complex64::from_polar(1.0, 4.0 * std::f64::consts::PI / 5.0),
    );
    complex_close(
        fibsu2.twist(&x),
        Complex64::from_polar(1.0, -4.0 * std::f64::consts::PI / 5.0),
    );
    complex_close(
        fibsu2
            .f_symbol(&x, &x, &x, &x, &fibsu2.unit(), &fibsu2.unit())
            .unwrap()
            .at(0, 0, 0, 0),
        Complex64::new(-1.0 / (1.0 + 5.0_f64.sqrt()), 0.0),
    );
    // Both promotion directions, and a promoted value followed by a real factor.
    let reverse = product(su2(), TableCategory::fibonacci());
    complex_close(
        reverse.twist(&(reverse.factors().0.unit(), tau)),
        fibsu2.twist(&x),
    );
    let three = product![U1, TableCategory::fibonacci(), FermionParity];
    complex_close(
        three.twist(&(U1.unit(), tau, Parity::ODD)),
        -fibsu2.twist(&x),
    );
}

#[test]
fn fusion_first_factor_fastest_and_reassociation() {
    let s = su2();
    let h = s.irrep(&[1]).unwrap();
    let one = s.irrep(&[2]).unwrap();
    let cat = product(s.clone(), s.clone());
    let x = (h.clone(), one.clone());
    let z = s.unit();
    let two = s.irrep(&[4]).unwrap();
    assert_eq!(
        cat.fusion(&x, &x).unwrap().collect::<Vec<_>>(),
        [
            (z.clone(), z.clone()),
            (one.clone(), z.clone()),
            (z.clone(), one.clone()),
            (one.clone(), one.clone()),
            (z.clone(), two.clone()),
            (one.clone(), two),
        ]
    );
    let flat = product![s.clone(), FermionParity, TableCategory::fibonacci()];
    let left = product(
        product(s.clone(), FermionParity),
        TableCategory::fibonacci(),
    );
    let right = product(s, product(FermionParity, TableCategory::fibonacci()));
    let t = flat.factors().2.object(2).unwrap();
    let a = (h.clone(), Parity::ODD, t);
    let l = ((h.clone(), Parity::ODD), t);
    let r = (h, (Parity::ODD, t));
    complex_close(flat.twist(&a), left.twist(&l));
    complex_close(flat.twist(&a), right.twist(&r));
    let f = flat
        .f_symbol(&a, &a, &a, &a, &flat.unit(), &flat.unit())
        .unwrap();
    complex_close(
        f.at(0, 0, 0, 0),
        left.f_symbol(&l, &l, &l, &l, &left.unit(), &left.unit())
            .unwrap()
            .at(0, 0, 0, 0),
    );
    complex_close(
        f.at(0, 0, 0, 0),
        right
            .f_symbol(&r, &r, &r, &r, &right.unit(), &right.unit())
            .unwrap()
            .at(0, 0, 0, 0),
    );
    complex_close(
        flat.r_symbol(&a, &a, &flat.unit()).unwrap().at(0, 0),
        left.r_symbol(&l, &l, &left.unit()).unwrap().at(0, 0),
    );
    let flattened: Vec<_> = left
        .fusion(&l, &l)
        .unwrap()
        .map(|((a, b), c)| (a, b, c))
        .collect();
    assert_eq!(flat.fusion(&a, &a).unwrap().collect::<Vec<_>>(), flattened);
    assert_ne!(flat.authority_fingerprint(), left.authority_fingerprint());
}

fn hash(x: &impl Hash) -> u64 {
    let mut h = DefaultHasher::new();
    x.hash(&mut h);
    h.finish()
}
#[test]
fn identity_framing_arity_and_macro_evaluation() {
    let a = product(ZN::new(2).unwrap(), ZN::new(3).unwrap());
    let b = a.clone();
    let before = hash(&a);
    let debug = format!("{a:?}");
    let mut expected =
        b"fusion-symbols:product:epoch=1:axes=last-fastest:fusion=first-fastest;".to_vec();
    expected.extend(2u32.to_le_bytes());
    for c in [&a.factors().0, &a.factors().1] {
        let fp = c.authority_fingerprint();
        expected.extend((fp.len() as u64).to_le_bytes());
        expected.extend(fp);
    }
    assert_eq!(a.authority_fingerprint(), expected);
    assert_eq!(a, b);
    assert_eq!(hash(&a), before);
    assert_eq!(format!("{a:?}"), debug);
    assert_eq!(a, a.clone());
    let swapped = product(ZN::new(3).unwrap(), ZN::new(2).unwrap());
    assert_ne!(a, swapped);
    assert_eq!(a.authority_fingerprint(), swapped.authority_fingerprint());
    assert_ne!(
        product(U1, FermionParity).authority_fingerprint(),
        product(FermionParity, U1).authority_fingerprint()
    );
    let max = product![U1, U1, U1, U1, U1, U1, U1, U1, U1, U1, U1, U1,];
    assert_eq!(max.fusion(&max.unit(), &max.unit()).unwrap().count(), 1);
    assert_eq!(max.nsymbol(&max.unit(), &max.unit(), &max.unit()), 1);
    assert_eq!(
        max.f_symbol(
            &max.unit(),
            &max.unit(),
            &max.unit(),
            &max.unit(),
            &max.unit(),
            &max.unit()
        )
        .unwrap(),
        FBlock::one(1.0)
    );
    assert_eq!(
        max.r_symbol(&max.unit(), &max.unit(), &max.unit()).unwrap(),
        RBlock::one(1.0)
    );
    assert_eq!(max.clone(), max);
    assert_eq!(hash(&max), hash(&max.clone()));
    let mut count = 0;
    let mut next = || {
        count += 1;
        U1
    };
    let _ = product![next(), next(), next(),];
    assert_eq!(count, 3);
    let repeated = Product::new((U1, U1));
    assert_eq!(repeated.into_factors(), (U1, U1));
    // Trivial factors change category identity but preserve projected mathematical queries.
    let trivial = product(su2(), ZN::new(1).unwrap());
    let h = trivial.factors().0.irrep(&[1]).unwrap();
    let a = (h, trivial.factors().1.unit());
    assert_eq!(trivial.dim(&a), 2.0);
    assert_eq!(trivial.frobenius_schur_indicator(&a), -1.0);
}

#[test]
fn generic_blocks_and_complex_symbolblock() {
    let f = FBlock::one(0.5);
    let _: fusion_symbols::lie::FBlock = f;
    let r = RBlock::one(-1.0);
    let _: fusion_symbols::lie::RBlock = r;
    let z = Complex64::new(2.0, 3.0);
    let f = FBlock::one(z);
    let r = RBlock::one(z);
    assert_eq!(<FBlock<Complex64> as SymbolBlock<4>>::get(&f, [0; 4]), z);
    assert_eq!(<RBlock<Complex64> as SymbolBlock<2>>::get(&r, [0; 2]), z);
    assert_eq!(f.as_slice(), &[z]);
    assert_eq!(r.dims(), [1; 2]);
}

// Deliberately synthetic blocks isolate axis order, overflow and call precedence.
// These are not mathematical-category oracles; real-category identities are tested separately.
#[derive(Clone, Debug)]
struct Probe {
    n: [u32; 4],
    f: &'static [f64],
    r: &'static [f64],
    fusion: FusionStyle,
    braiding: BraidingStyle,
    fail: bool,
}
impl PartialEq for Probe {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n
            && self
                .f
                .iter()
                .map(|x| x.to_bits())
                .eq(other.f.iter().map(|x| x.to_bits()))
            && self
                .r
                .iter()
                .map(|x| x.to_bits())
                .eq(other.r.iter().map(|x| x.to_bits()))
            && self.fusion == other.fusion
            && self.braiding == other.braiding
            && self.fail == other.fail
    }
}
impl Eq for Probe {}
impl Hash for Probe {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.n.hash(h);
        self.fusion.hash(h);
        self.braiding.hash(h);
        self.fail.hash(h);
        for x in self.f.iter().chain(self.r) {
            x.to_bits().hash(h);
        }
    }
}
impl Default for Probe {
    fn default() -> Self {
        Self {
            n: [1; 4],
            f: &[2.0],
            r: &[3.0],
            fusion: FusionStyle::Unique,
            braiding: BraidingStyle::Bosonic,
            fail: false,
        }
    }
}
#[derive(Clone, Debug)]
struct Dense<const R: usize> {
    dims: [usize; R],
    data: &'static [f64],
}
impl<const R: usize> SymbolBlock<R> for Dense<R> {
    type Scalar = f64;
    fn dims(&self) -> [usize; R] {
        self.dims
    }
    fn get(&self, idx: [usize; R]) -> f64 {
        self.data[idx
            .into_iter()
            .zip(self.dims)
            .fold(0, |p, (i, n)| p * n + i)]
    }
    fn as_slice(&self) -> &[f64] {
        self.data
    }
}
thread_local! { static FUSIONS:Cell<usize> = const { Cell::new(0) }; }
impl FusionCategory for Probe {
    type Object = u8;
    type Scalar = f64;
    type FSymbol = Dense<4>;
    type RSymbol = Dense<2>;
    type Fusion<'a> = std::iter::Take<std::array::IntoIter<u8, 2>>;
    type Error = std::io::Error;
    fn fusion_style(&self) -> FusionStyle {
        self.fusion
    }
    fn braiding_style(&self) -> BraidingStyle {
        self.braiding
    }
    fn unit(&self) -> u8 {
        0
    }
    fn dim(&self, a: &u8) -> f64 {
        assert!(*a < 6, "foreign probe object");
        1.0
    }
    fn dual(&self, a: &u8) -> u8 {
        self.dim(a);
        *a
    }
    fn frobenius_schur_phase(&self, a: &u8) -> f64 {
        self.dim(a);
        1.0
    }
    fn frobenius_schur_indicator(&self, a: &u8) -> f64 {
        self.dim(a);
        1.0
    }
    fn twist(&self, a: &u8) -> f64 {
        self.dim(a);
        assert_ne!(self.braiding, BraidingStyle::None);
        1.0
    }
    fn fusion<'a>(&'a self, a: &u8, b: &u8) -> Result<Self::Fusion<'a>, Self::Error> {
        self.dim(a);
        self.dim(b);
        FUSIONS.set(FUSIONS.get() + 1);
        if self.fail {
            Err(std::io::Error::other("probe failure"))
        } else {
            Ok([0, 1].into_iter().take(if self.n[0] == 0 { 0 } else { 2 }))
        }
    }
    fn nsymbol(&self, a: &u8, b: &u8, c: &u8) -> u32 {
        for x in [a, b, c] {
            self.dim(x);
        }
        self.n[match (*a, *b, *c) {
            (4, 2, 3) => 1,
            (1, 2, 5) => 2,
            (0, 5, 3) => 3,
            _ => 0,
        }]
    }
    fn f_symbol(
        &self,
        a: &u8,
        b: &u8,
        c: &u8,
        d: &u8,
        e: &u8,
        f: &u8,
    ) -> Result<Self::FSymbol, Self::Error> {
        for x in [a, b, c, d, e, f] {
            self.dim(x);
        }
        if self.fail {
            Err(std::io::Error::other("probe failure"))
        } else {
            Ok(Dense {
                dims: self.n.map(|n| n as usize),
                data: self.f,
            })
        }
    }
    fn r_symbol(&self, a: &u8, b: &u8, c: &u8) -> Result<Self::RSymbol, Self::Error> {
        for x in [a, b, c] {
            self.dim(x);
        }
        if self.fail {
            Err(std::io::Error::other("probe failure"))
        } else {
            Ok(Dense {
                dims: [self.n[0] as usize; 2],
                data: self.r,
            })
        }
    }
    fn authority_fingerprint(&self) -> &[u8] {
        b"probe"
    }
}

#[test]
fn multiplicity_axes_are_associative() {
    const REASSOCIATION_TOLERANCE: f64 = 1e-12;
    let p = Probe {
        n: [2, 1, 2, 1],
        f: &[0.2, 0.3, 0.5, 0.7],
        r: &[0.2, 0.3, 0.5, 0.7],
        ..Probe::default()
    };
    let q = Probe {
        n: [3, 3, 1, 1],
        f: &[1.1, 1.3, 1.7, 1.9, 2.3, 2.9, 3.1, 3.7, 4.1],
        r: &[1.1, 1.3, 1.7, 1.9, 2.3, 2.9, 3.1, 3.7, 4.1],
        ..Probe::default()
    };
    let r = Probe {
        n: [2, 1, 1, 2],
        f: &[4.3, 4.7, 5.3, 5.9],
        r: &[4.3, 4.7, 5.3, 5.9],
        ..Probe::default()
    };
    let flat = product![p.clone(), q.clone(), r.clone()];
    let left = product(product(p.clone(), q.clone()), r.clone());
    let right = product(p, product(q, r));
    let a = [0, 1, 2, 3, 4, 5].map(|x| (x, x, x));
    let l = a.map(|(x, y, z)| ((x, y), z));
    let r = a.map(|(x, y, z)| (x, (y, z)));
    let flat_f = flat
        .f_symbol(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5])
        .unwrap();
    let left_f = left
        .f_symbol(&l[0], &l[1], &l[2], &l[3], &l[4], &l[5])
        .unwrap();
    let right_f = right
        .f_symbol(&r[0], &r[1], &r[2], &r[3], &r[4], &r[5])
        .unwrap();
    for block in [&flat_f, &left_f, &right_f] {
        assert_eq!(block.dims(), [12, 3, 2, 2]);
    }
    let flat_r = flat.r_symbol(&a[0], &a[1], &a[4]).unwrap();
    let left_r = left.r_symbol(&l[0], &l[1], &l[4]).unwrap();
    let right_r = right.r_symbol(&r[0], &r[1], &r[4]).unwrap();
    for block in [&flat_r, &left_r, &right_r] {
        assert_eq!(block.dims(), [12; 2]);
    }
    // Row-major slices compare every multiplicity index after tuple relabeling.
    for (flat, left, right) in [
        (flat_f.data(), left_f.data(), right_f.data()),
        (flat_r.data(), left_r.data(), right_r.data()),
    ] {
        assert_eq!(flat.len(), 144);
        assert_eq!(left.len(), flat.len());
        assert_eq!(right.len(), flat.len());
        for (i, ((&a, &b), &c)) in flat.iter().zip(left).zip(right).enumerate() {
            assert!(
                (a - b).abs() < REASSOCIATION_TOLERANCE && (a - c).abs() < REASSOCIATION_TOLERANCE,
                "entry {i}: flat={a}, left={b}, right={c}"
            );
        }
    }
}

#[test]
fn per_axis_kron_has_last_factor_fastest() {
    let a = Probe {
        n: [2, 1, 2, 1],
        f: &[2., 3., 5., 7.],
        r: &[2., 3., 5., 7.],
        ..Probe::default()
    };
    let b = Probe {
        n: [3, 2, 1, 2],
        f: &[11., 13., 17., 19., 23., 29., 31., 37., 41., 43., 47., 53.],
        r: &[11.],
        ..Probe::default()
    };
    let cat = product(a, b);
    let args = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)];
    let f = cat
        .f_symbol(&args[0], &args[1], &args[2], &args[3], &args[4], &args[5])
        .unwrap();
    assert_eq!(f.dims(), [6, 2, 2, 2]);
    // Independent indexed formula; using flat kron on the two buffers gives a different answer.
    for i in 0..6 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    assert_eq!(
                        f.at(i, j, k, l),
                        [2., 3., 5., 7.][(i / 3) * 2 + k]
                            * [11., 13., 17., 19., 23., 29., 31., 37., 41., 43., 47., 53.]
                                [(i % 3) * 4 + j * 2 + l]
                    );
                }
            }
        }
    }
    let cat = product(
        Probe {
            n: [2; 4],
            r: &[2., 3., 5., 7.],
            ..Probe::default()
        },
        Probe {
            n: [3; 4],
            r: &[11., 13., 17., 19., 23., 29., 31., 37., 41.],
            ..Probe::default()
        },
    );
    let r = cat.r_symbol(&(0, 0), &(0, 0), &(0, 0)).unwrap();
    assert_eq!(r.dim(), 6);
    for i in 0..6 {
        for j in 0..6 {
            assert_eq!(
                r.at(i, j),
                [2., 3., 5., 7.][(i / 3) * 2 + j / 3]
                    * [11., 13., 17., 19., 23., 29., 31., 37., 41.][(i % 3) * 3 + j % 3]
            );
        }
    }
}

#[test]
fn style_tables_and_iterators_reset_without_requery() {
    use BraidingStyle::*;
    use FusionStyle::*;
    for (i, a) in [Unique, Simple, Generic].into_iter().enumerate() {
        for (j, b) in [Unique, Simple, Generic].into_iter().enumerate() {
            assert_eq!(
                product(
                    Probe {
                        fusion: a,
                        ..Probe::default()
                    },
                    Probe {
                        fusion: b,
                        ..Probe::default()
                    }
                )
                .fusion_style(),
                [Unique, Simple, Generic][i.max(j)]
            );
        }
    }
    for (i, a) in [Bosonic, Fermionic, Anyonic, None].into_iter().enumerate() {
        for (j, b) in [Bosonic, Fermionic, Anyonic, None].into_iter().enumerate() {
            assert_eq!(
                product(
                    Probe {
                        braiding: a,
                        ..Probe::default()
                    },
                    Probe {
                        braiding: b,
                        ..Probe::default()
                    }
                )
                .braiding_style(),
                [Bosonic, Fermionic, Anyonic, None][i.max(j)]
            );
        }
    }
    FUSIONS.set(0);
    let cat = product(Probe::default(), Probe::default());
    let mut iter = cat.fusion(&(0, 0), &(0, 0)).unwrap();
    assert_eq!(iter.next(), Some((0, 0)));
    let rest = iter.clone().collect::<Vec<_>>();
    assert_eq!(rest, [(1, 0), (0, 1), (1, 1)]);
    assert_eq!(iter.collect::<Vec<_>>(), rest);
    assert_eq!(FUSIONS.get(), 2);
}

// Even the twelfth position must obey the same preflight, zero, and error rules.
#[test]
fn every_factor_position_obeys_validation_and_overflow_precedence() {
    for position in 0..12 {
        let mut factors = std::array::from_fn::<_, 12, _>(|_| Probe::default());
        factors[position].n = [0; 4];
        macro_rules! cat {
            () => {
                Product::new((
                    factors[0].clone(),
                    factors[1].clone(),
                    factors[2].clone(),
                    factors[3].clone(),
                    factors[4].clone(),
                    factors[5].clone(),
                    factors[6].clone(),
                    factors[7].clone(),
                    factors[8].clone(),
                    factors[9].clone(),
                    factors[10].clone(),
                    factors[11].clone(),
                ))
            };
        }
        macro_rules! obj {
            ($v:expr) => {
                (
                    $v[0], $v[1], $v[2], $v[3], $v[4], $v[5], $v[6], $v[7], $v[8], $v[9], $v[10],
                    $v[11],
                )
            };
        }
        let zero = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let cat = cat!();
        assert_eq!(cat.nsymbol(&zero, &zero, &zero), 0);
        let mut empty = cat.fusion(&zero, &zero).unwrap();
        assert_eq!(empty.next(), None);
        assert_eq!(empty.next(), None);
        assert!(cat
            .f_symbol(&zero, &zero, &zero, &zero, &zero, &zero)
            .unwrap()
            .data()
            .is_empty());
        assert!(cat.r_symbol(&zero, &zero, &zero).unwrap().data().is_empty());
        factors[position].braiding = BraidingStyle::None;
        let cat = cat!();
        assert!(
            matches!(cat.r_symbol(&zero,&zero,&zero),Err(ProductError::NoBraiding{factor}) if factor==position)
        );
        // A later zero must also beat multiplication overflow in a nonzero prefix.
        factors.iter_mut().for_each(|c| c.n = [u32::MAX; 4]);
        factors[position].n = [0; 4];
        let cat = cat!();
        assert_eq!(cat.nsymbol(&zero, &zero, &zero), 0);
        assert!(cat
            .f_symbol(&zero, &zero, &zero, &zero, &zero, &zero)
            .unwrap()
            .data()
            .is_empty());
        // Every typed factor error retains its immediate index on each fallible query.
        factors.iter_mut().for_each(|c| {
            c.n = [1; 4];
            c.braiding = BraidingStyle::Bosonic;
        });
        factors[position].fail = true;
        let cat = cat!();
        let prefix = format!("Factor{position}(");
        assert!(format!("{:?}", cat.fusion(&zero, &zero).unwrap_err()).starts_with(&prefix));
        assert!(format!(
            "{:?}",
            cat.f_symbol(&zero, &zero, &zero, &zero, &zero, &zero)
                .unwrap_err()
        )
        .starts_with(&prefix));
        assert!(
            format!("{:?}", cat.r_symbol(&zero, &zero, &zero).unwrap_err()).starts_with(&prefix)
        );
        factors.iter_mut().for_each(|c| {
            c.n = [1; 4];
            c.fail = false;
            c.braiding = BraidingStyle::Bosonic;
        });
        factors[position].n = [u32::MAX; 4];
        factors[(position + 1) % 12].n = [2; 4];
        let overflowing = cat!();
        assert!(std::panic::catch_unwind(|| overflowing.nsymbol(&zero, &zero, &zero)).is_err());
        assert!(matches!(
            overflowing.f_symbol(&zero, &zero, &zero, &zero, &zero, &zero),
            Err(ProductError::Overflow)
        ));
        assert!(matches!(
            overflowing.r_symbol(&zero, &zero, &zero),
            Err(ProductError::Overflow)
        ));
        factors[position].n = [0; 4];
        factors[position].braiding = BraidingStyle::None;
        // All foreign-object preflights beat zero, capability failure and earlier value errors.
        factors[0].fail = true;
        let cat = cat!();
        let mut foreign = [0; 12];
        foreign[position] = 255;
        let bad = obj!(foreign);
        for query in 0..21 {
            assert!(std::panic::catch_unwind(|| match query {
                0 => {
                    cat.dim(&bad);
                }
                1 => {
                    cat.dual(&bad);
                }
                2 => {
                    cat.frobenius_schur_phase(&bad);
                }
                3 => {
                    cat.frobenius_schur_indicator(&bad);
                }
                4 => {
                    cat.twist(&bad);
                }
                5 => {
                    let _ = cat.fusion(&zero, &bad);
                }
                6 => {
                    cat.nsymbol(&zero, &zero, &bad);
                }
                7..=12 => {
                    let mut args = [&zero; 6];
                    args[query - 7] = &bad;
                    let _ = cat.f_symbol(args[0], args[1], args[2], args[3], args[4], args[5]);
                }
                _ => {
                    let _ = cat.r_symbol(&zero, &zero, &bad);
                }
            })
            .is_err());
        }
    }
    let huge = Probe {
        n: [u32::MAX; 4],
        ..Probe::default()
    };
    let cat = product(huge.clone(), huge);
    assert!(std::panic::catch_unwind(|| cat.nsymbol(&(0, 0), &(0, 0), &(0, 0))).is_err());
    assert!(matches!(
        cat.f_symbol(&(0, 0), &(0, 0), &(0, 0), &(0, 0), &(0, 0), &(0, 0)),
        Err(ProductError::Overflow)
    ));
    assert!(matches!(
        cat.r_symbol(&(0, 0), &(0, 0), &(0, 0)),
        Err(ProductError::Overflow)
    ));
    // Axes fit u32, but the element count / allocation layout does not.
    let large = product(
        Probe {
            n: [65536; 4],
            ..Probe::default()
        },
        Probe::default(),
    );
    assert!(matches!(
        large.f_symbol(&(0, 0), &(0, 0), &(0, 0), &(0, 0), &(0, 0), &(0, 0)),
        Err(ProductError::Overflow)
    ));
    let bytes = product(
        Probe {
            n: [u32::MAX; 4],
            ..Probe::default()
        },
        Probe::default(),
    );
    assert!(matches!(
        bytes.r_symbol(&(0, 0), &(0, 0), &(0, 0)),
        Err(ProductError::Overflow)
    ));
}

#[test]
fn empty_logical_axes_skip_errors_and_reject_real_foreign_objects() {
    let cat = product(
        U1,
        Probe {
            fail: true,
            ..Probe::default()
        },
    );
    let a = (U1.unit(), 0);
    let c = (U1Irrep::new(1).unwrap(), 0);
    assert!(cat
        .f_symbol(&a, &a, &a, &a, &c, &a)
        .unwrap()
        .data()
        .is_empty());
    assert!(cat.r_symbol(&a, &a, &c).unwrap().data().is_empty());
    let reverse = product(
        Probe {
            fail: true,
            ..Probe::default()
        },
        U1,
    );
    assert!(reverse
        .r_symbol(&(0, U1.unit()), &(0, U1.unit()), &(0, c.0))
        .unwrap()
        .data()
        .is_empty());
    let cat = product(U1, ZN::new(2).unwrap());
    let foreign = (U1.unit(), ZN::new(3).unwrap().unit());
    assert!(std::panic::catch_unwind(|| cat.f_symbol(
        &cat.unit(),
        &cat.unit(),
        &cat.unit(),
        &cat.unit(),
        &cat.unit(),
        &foreign
    ))
    .is_err());
    // Public binary errors need no unused type arguments.
    let _: Option<ProductError<Infallible, Infallible>> = None;
}

#[test]
fn left_fold_and_unbraided_table_preflight() {
    let cat = product![
        Probe {
            f: &[1e308],
            ..Probe::default()
        },
        Probe {
            f: &[1e-308],
            ..Probe::default()
        },
        Probe {
            f: &[1e-308],
            ..Probe::default()
        }
    ];
    let z = (0, 0, 0);
    assert_eq!(
        cat.f_symbol(&z, &z, &z, &z, &z, &z).unwrap().at(0, 0, 0, 0),
        (1e308 * 1e-308) * 1e-308
    );
    let fib = TableCategory::fibonacci();
    let unbraided = TableCategory::shipped_uuids()
        .map(|id| TableCategory::by_uuid(id).unwrap())
        .find(|cat| !cat.is_braided())
        .unwrap();
    let cat = product(fib.clone(), unbraided.clone());
    let a = cat.unit();
    let c = (fib.object(2).unwrap(), unbraided.unit());
    assert!(matches!(
        cat.r_symbol(&a, &a, &c),
        Err(ProductError::NoBraiding { factor: 1 })
    ));
    let foreign = (fib.unit(), fib.object(2).unwrap());
    assert!(std::panic::catch_unwind(|| cat.r_symbol(&a, &a, &foreign)).is_err());
    assert!(std::panic::catch_unwind(|| cat.twist(&foreign)).is_err());
}

#[cfg(feature = "cgc-gen")]
#[test]
fn su3_octet_multiplicity_uses_independent_gauge_literals() {
    // Signed SUNRepresentations.jl 0.4.0 literals, as recorded in
    // tests/sun_fr_su3_values.rs::f_symbols_checked_against_sunrepresentations
    // and ::r_symbols_checked_against_sunrepresentations (not computed from this product).
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let o = su3.irrep(&[1, 1]).unwrap();
    let literal = Probe {
        n: [2; 4],
        f: &[
            2., 3., 5., 7., 11., 13., 17., 19., 23., 29., 31., 37., 41., 43., 47., 53.,
        ],
        r: &[2., 3., 5., 7.],
        ..Probe::default()
    };
    let cat = product(su3, literal);
    let a = (o, 0);
    assert_eq!(cat.nsymbol(&a, &a, &a), 4);
    let f = cat.f_symbol(&a, &a, &a, &a, &a, &a).unwrap();
    assert_eq!(f.dims(), [4; 4]);
    for (idx, value) in [
        ([0, 0, 0, 0], 0.857_142_857_142_856),
        ([1, 0, 0, 0], 0.0),
        ([1, 1, 0, 0], -0.142_857_142_857_142_8),
        ([1, 0, 1, 0], -0.142_857_142_857_142_8),
        ([0, 1, 1, 0], -0.142_857_142_857_142_8),
        ([1, 1, 1, 0], -0.383_325_938_999_963_3),
        ([0, 1, 1, 1], -0.383_325_938_999_963_3),
        ([1, 1, 1, 1], 0.628_571_428_571_427_4),
    ] {
        for (digit, right_value) in [
            ([0, 0, 0, 0], 2.),
            ([0, 1, 0, 1], 13.),
            ([1, 0, 1, 0], 31.),
            ([1, 1, 1, 1], 53.),
        ] {
            let index = std::array::from_fn::<_, 4, _>(|k| idx[k] * 2 + digit[k]);
            close(f.get(index), value * right_value);
        }
    }
    let r = cat.r_symbol(&a, &a, &a).unwrap();
    assert_eq!(r.dim(), 4);
    for (idx, value) in [
        ([0, 0], -0.285_714_285_714_285_3),
        ([1, 0], 0.958_314_847_499_908_9),
        ([0, 1], 0.958_314_847_499_908_8),
        ([1, 1], 0.285_714_285_714_285_5),
    ] {
        for i in 0..2 {
            for j in 0..2 {
                close(
                    r.at(idx[0] * 2 + i, idx[1] * 2 + j),
                    value * [2., 3., 5., 7.][i * 2 + j],
                );
            }
        }
    }
}

#[test]
fn product_error_accepts_borrowed_payloads() {
    #[derive(Debug)]
    struct Borrowed<'a>(&'a str);
    impl std::fmt::Display for Borrowed<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }
    impl std::error::Error for Borrowed<'_> {}
    fn accepts_error(_: &impl std::error::Error) {}
    let text = String::from("borrowed");
    let error: ProductError<Borrowed<'_>> = ProductError::Factor0(Borrowed(&text));
    accepts_error(&error);
    assert_eq!(error.to_string(), "product factor 0: borrowed");
}

#[test]
fn partial_zero_axes_use_logical_shapes_before_values() {
    let cat = product(
        Probe {
            n: [0, 1, 2, 1],
            fail: true,
            ..Probe::default()
        },
        Probe {
            n: [1, 3, 1, 2],
            ..Probe::default()
        },
    );
    let args = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)];
    let f = cat
        .f_symbol(&args[0], &args[1], &args[2], &args[3], &args[4], &args[5])
        .unwrap();
    assert_eq!(f.dims(), [0, 3, 2, 2]);
    assert!(f.data().is_empty());
}

#[test]
fn u1_overflow_in_every_immediate_position() {
    let cat = product![U1, U1, U1, U1, U1, U1, U1, U1, U1, U1, U1, U1];
    for position in 0..12 {
        let mut a = [U1.unit(); 12];
        let mut b = a;
        a[position] = U1Irrep::new(i32::MAX).unwrap();
        b[position] = U1Irrep::new(1).unwrap();
        macro_rules! obj {
            ($v:expr) => {
                (
                    $v[0], $v[1], $v[2], $v[3], $v[4], $v[5], $v[6], $v[7], $v[8], $v[9], $v[10],
                    $v[11],
                )
            };
        }
        assert!(format!("{:?}", cat.fusion(&obj!(a), &obj!(b)).unwrap_err())
            .starts_with(&format!("Factor{position}(")));
    }
}
