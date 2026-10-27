use fusion_symbols::{
    category::{BraidingStyle, FBlock, FusionCategory, FusionStyle, RBlock, SymbolBlock},
    fermion::{FermionParity, FermionParityIrrep as Parity},
    finite::ZN,
    lie::{GroupId, LieGroup},
    lie::{U1Irrep, U1},
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
    let fu1 = Product::new((U1, FermionParity));
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
    let fsu2 = Product::new((su2(), FermionParity));
    let half = (fsu2.factors().0.irrep(&[1], &[]).unwrap(), Parity::ODD);
    let triplet = (fsu2.factors().0.irrep(&[2], &[]).unwrap(), Parity::EVEN);
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
    let ff = Product::new((FermionParity, FermionParity));
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

    let fibsu2 = Product::new((TableCategory::fibonacci(), su2()));
    let tau = fibsu2.factors().0.object(2).unwrap();
    let h = fibsu2.factors().1.irrep(&[1], &[]).unwrap();
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
    let reverse = Product::new((su2(), TableCategory::fibonacci()));
    complex_close(
        reverse.twist(&(reverse.factors().0.unit(), tau)),
        fibsu2.twist(&x),
    );
    let three = Product::new((U1, TableCategory::fibonacci(), FermionParity));
    complex_close(
        three.twist(&(U1.unit(), tau, Parity::ODD)),
        -fibsu2.twist(&x),
    );
}

#[test]
fn fusion_first_factor_fastest_and_reassociation() {
    let s = su2();
    let h = s.irrep(&[1], &[]).unwrap();
    let one = s.irrep(&[2], &[]).unwrap();
    let cat = Product::new((s.clone(), s.clone()));
    let x = (h.clone(), one.clone());
    let z = s.unit();
    let two = s.irrep(&[4], &[]).unwrap();
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
    let flat = Product::new((s.clone(), FermionParity, TableCategory::fibonacci()));
    let left = Product::new((
        Product::new((s.clone(), FermionParity)),
        TableCategory::fibonacci(),
    ));
    let right = Product::new((s, Product::new((FermionParity, TableCategory::fibonacci()))));
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
fn identity_framing_and_arity() {
    let a = Product::new((ZN::new(2).unwrap(), ZN::new(3).unwrap()));
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
    let swapped = Product::new((ZN::new(3).unwrap(), ZN::new(2).unwrap()));
    assert_ne!(a, swapped);
    assert_eq!(a.authority_fingerprint(), swapped.authority_fingerprint());
    assert_ne!(
        Product::new((U1, FermionParity)).authority_fingerprint(),
        Product::new((FermionParity, U1)).authority_fingerprint()
    );
    let max = Product::new((U1, U1, U1, U1, U1, U1, U1, U1, U1, U1, U1, U1));
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
    let repeated = Product::new((U1, U1));
    assert_eq!(repeated.into_factors(), (U1, U1));
    // Trivial factors change category identity but preserve projected mathematical queries.
    let trivial = Product::new((su2(), ZN::new(1).unwrap()));
    let h = trivial.factors().0.irrep(&[1], &[]).unwrap();
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
    fn contains(&self, a: &u8) -> bool {
        *a < 6
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
    let flat = Product::new((p.clone(), q.clone(), r.clone()));
    let left = Product::new((Product::new((p.clone(), q.clone())), r.clone()));
    let right = Product::new((p, Product::new((q, r))));
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
    let cat = Product::new((a, b));
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
    let cat = Product::new((
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
    ));
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
                Product::new((
                    Probe {
                        fusion: a,
                        ..Probe::default()
                    },
                    Probe {
                        fusion: b,
                        ..Probe::default()
                    }
                ))
                .fusion_style(),
                [Unique, Simple, Generic][i.max(j)]
            );
        }
    }
    for (i, a) in [Bosonic, Fermionic, Anyonic, None].into_iter().enumerate() {
        for (j, b) in [Bosonic, Fermionic, Anyonic, None].into_iter().enumerate() {
            assert_eq!(
                Product::new((
                    Probe {
                        braiding: a,
                        ..Probe::default()
                    },
                    Probe {
                        braiding: b,
                        ..Probe::default()
                    }
                ))
                .braiding_style(),
                [Bosonic, Fermionic, Anyonic, None][i.max(j)]
            );
        }
    }
    FUSIONS.set(0);
    let cat = Product::new((Probe::default(), Probe::default()));
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
    let cat = Product::new((huge.clone(), huge));
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
    let large = Product::new((
        Probe {
            n: [65536; 4],
            ..Probe::default()
        },
        Probe::default(),
    ));
    assert!(matches!(
        large.f_symbol(&(0, 0), &(0, 0), &(0, 0), &(0, 0), &(0, 0), &(0, 0)),
        Err(ProductError::Overflow)
    ));
    let bytes = Product::new((
        Probe {
            n: [u32::MAX; 4],
            ..Probe::default()
        },
        Probe::default(),
    ));
    assert!(matches!(
        bytes.r_symbol(&(0, 0), &(0, 0), &(0, 0)),
        Err(ProductError::Overflow)
    ));
}

#[test]
fn empty_logical_axes_skip_errors_and_reject_real_foreign_objects() {
    let cat = Product::new((
        U1,
        Probe {
            fail: true,
            ..Probe::default()
        },
    ));
    let a = (U1.unit(), 0);
    let c = (U1Irrep::new(1).unwrap(), 0);
    assert!(cat
        .f_symbol(&a, &a, &a, &a, &c, &a)
        .unwrap()
        .data()
        .is_empty());
    assert!(cat.r_symbol(&a, &a, &c).unwrap().data().is_empty());
    let reverse = Product::new((
        Probe {
            fail: true,
            ..Probe::default()
        },
        U1,
    ));
    assert!(reverse
        .r_symbol(&(0, U1.unit()), &(0, U1.unit()), &(0, c.0))
        .unwrap()
        .data()
        .is_empty());
    let cat = Product::new((U1, ZN::new(2).unwrap()));
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
    let cat = Product::new((
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
        },
    ));
    let z = (0, 0, 0);
    assert_eq!(
        cat.f_symbol(&z, &z, &z, &z, &z, &z).unwrap().at(0, 0, 0, 0),
        (1e308 * 1e-308) * 1e-308
    );
    let fib = TableCategory::fibonacci();
    let unbraided = TableCategory::shipped_uuids()
        .map(|id| TableCategory::by_uuid(id).unwrap())
        .find(|cat| cat.braiding_style() == BraidingStyle::None)
        .unwrap();
    let cat = Product::new((fib.clone(), unbraided.clone()));
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

#[cfg(feature = "lie-generated")]
#[test]
fn su3_octet_multiplicity_uses_independent_gauge_literals() {
    // Signed SUNRepresentations.jl 0.4.0 literals, as recorded in
    // tests/sun_fr_su3_values.rs::f_symbols_checked_against_sunrepresentations
    // and ::r_symbols_checked_against_sunrepresentations (not computed from this product).
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let o = su3.irrep(&[1, 1], &[]).unwrap();
    let literal = Probe {
        n: [2; 4],
        f: &[
            2., 3., 5., 7., 11., 13., 17., 19., 23., 29., 31., 37., 41., 43., 47., 53.,
        ],
        r: &[2., 3., 5., 7.],
        ..Probe::default()
    };
    let cat = Product::new((su3, literal));
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
    let cat = Product::new((
        Probe {
            n: [0, 1, 2, 1],
            fail: true,
            ..Probe::default()
        },
        Probe {
            n: [1, 3, 1, 2],
            ..Probe::default()
        },
    ));
    let args = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)];
    let f = cat
        .f_symbol(&args[0], &args[1], &args[2], &args[3], &args[4], &args[5])
        .unwrap();
    assert_eq!(f.dims(), [0, 3, 2, 2]);
    assert!(f.data().is_empty());
}

#[test]
fn u1_overflow_in_every_immediate_position() {
    let cat = Product::new((U1, U1, U1, U1, U1, U1, U1, U1, U1, U1, U1, U1));
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

#[test]
fn try_nsymbol_reports_overflow_and_zero_beats_it() {
    let probe = |n| Probe {
        n: [n; 4],
        ..Probe::default()
    };
    let z2 = (0, 0);
    let z3 = (0, 0, 0);
    // Flat: u32::MAX * 2 overflows, unless a later factor is zero.
    let flat = Product::new((probe(u32::MAX), probe(2)));
    assert!(matches!(
        flat.try_nsymbol(&z2, &z2, &z2),
        Err(ProductError::Overflow)
    ));
    let flat_zero = Product::new((probe(u32::MAX), probe(2), probe(0)));
    assert_eq!(flat_zero.try_nsymbol(&z3, &z3, &z3).unwrap(), 0);
    // Nested: the inner overflow keeps its immediate position, and every
    // F/R shape query returns it instead of panicking.
    let nested = Product::new((Product::new((probe(u32::MAX), probe(2))), probe(1)));
    let z = ((0, 0), 0);
    let inner_overflow = |e| matches!(e, ProductError::Factor0(ProductError::Overflow));
    assert!(inner_overflow(nested.try_nsymbol(&z, &z, &z).unwrap_err()));
    assert!(inner_overflow(
        nested.f_symbol(&z, &z, &z, &z, &z, &z).unwrap_err()
    ));
    assert!(inner_overflow(nested.r_symbol(&z, &z, &z).unwrap_err()));
    let right = Product::new((probe(1), Product::new((probe(2), probe(u32::MAX)))));
    let z = (0, (0, 0));
    assert!(matches!(
        right.try_nsymbol(&z, &z, &z),
        Err(ProductError::Factor1(ProductError::Overflow))
    ));
    assert!(matches!(
        right.f_symbol(&z, &z, &z, &z, &z, &z),
        Err(ProductError::Factor1(ProductError::Overflow))
    ));
    // A zero after a nested overflow fixes N = 0 and empty F/R blocks.
    let nested_zero = Product::new((Product::new((probe(u32::MAX), probe(2))), probe(0)));
    let z = ((0, 0), 0);
    assert_eq!(nested_zero.try_nsymbol(&z, &z, &z).unwrap(), 0);
    assert!(nested_zero
        .f_symbol(&z, &z, &z, &z, &z, &z)
        .unwrap()
        .data()
        .is_empty());
    assert!(nested_zero.r_symbol(&z, &z, &z).unwrap().data().is_empty());
    // Representable values equal nsymbol: 3 * 5 * 7 = 105, also nested.
    let fits = Product::new((Product::new((probe(3), probe(5))), probe(7)));
    assert_eq!(fits.try_nsymbol(&z, &z, &z).unwrap(), 105);
    assert_eq!(fits.nsymbol(&z, &z, &z), 105);
    let su2 = su2();
    let s = su2.irrep(&[1], &[]).unwrap();
    let cat = Product::new((su2.clone(), U1));
    let a = (s.clone(), U1Irrep::new(1).unwrap());
    let c = (su2.irrep(&[2], &[]).unwrap(), U1Irrep::new(2).unwrap());
    assert_eq!(cat.try_nsymbol(&a, &a, &c).unwrap(), 1);
    assert_eq!(cat.try_nsymbol(&a, &a, &cat.unit()).unwrap(), 0);
    assert_eq!(su2.try_nsymbol(&s, &s, &su2.unit()).unwrap(), 1);
    // Foreign objects still panic.
    assert!(std::panic::catch_unwind(|| fits.try_nsymbol(&z, &z, &((0, 0), 255))).is_err());
}

#[test]
fn runtime_checked_composition_and_initialized_iteration() {
    let probe = |n| Probe {
        n: [n; 4],
        ..Probe::default()
    };
    let factors = vec![
        Product::new((probe(65536), probe(1))),
        Product::new((probe(65536), probe(1))),
        Product::new((probe(u32::MAX), probe(2))),
    ];
    let a = vec![(0, 0); 3];
    let cat = Product::new(factors.clone());
    // A factor failure wins over overflow in the preceding nonzero product.
    assert!(matches!(
        cat.try_nsymbol(&a, &a, &a),
        Err(ProductError::Factor {
            index: 2,
            source: ProductError::Overflow
        })
    ));
    let zero = Product::new([factors, vec![Product::new((probe(0), probe(1)))]].concat());
    let z = vec![(0, 0); 4];
    assert_eq!(zero.try_nsymbol(&z, &z, &z).unwrap(), 0);
    assert!(zero
        .f_symbol(&z, &z, &z, &z, &z, &z)
        .unwrap()
        .data()
        .is_empty());
    // Like tuple products, infallible N evaluates every factor: a factor's
    // loud panic is not a typed failure that can be suppressed by a zero.
    let cat = Product::new(vec![
        Product::new((probe(0), probe(1))),
        Product::new((probe(u32::MAX), probe(2))),
    ]);
    let z = vec![(0, 0); 2];
    assert_eq!(cat.try_nsymbol(&z, &z, &z).unwrap(), 0);
    assert!(std::panic::catch_unwind(|| cat.nsymbol(&z, &z, &z)).is_err());
    let overflowing = Product::new(vec![probe(u32::MAX), probe(2)]);
    let z = vec![0; 2];
    assert!(matches!(
        overflowing.try_nsymbol(&z, &z, &z),
        Err(ProductError::Overflow)
    ));
    assert!(matches!(
        overflowing.r_symbol(&z, &z, &z),
        Err(ProductError::Overflow)
    ));
    // Runtime factor positions are not bounded by tuple arity.
    let mut factors = vec![Probe::default(); 14];
    factors[13].fail = true;
    let z = vec![0; 14];
    assert!(matches!(
        Product::new(factors).r_symbol(&z, &z, &z),
        Err(ProductError::Factor { index: 13, .. })
    ));
    let bad = Probe {
        f: &[],
        r: &[],
        ..Probe::default()
    };
    let malformed = Product::new(vec![Probe::default(), bad]);
    let z = vec![0; 2];
    assert!(matches!(
        malformed.f_symbol(&z, &z, &z, &z, &z, &z),
        Err(ProductError::MalformedSymbolData { factor: 1, .. })
    ));
    assert!(matches!(
        malformed.r_symbol(&z, &z, &z),
        Err(ProductError::MalformedSymbolData { factor: 1, .. })
    ));
    let cat = Product::new(vec![Probe::default(); 3]);
    let z = vec![0; 3];
    assert_eq!(
        cat.object_count(),
        Err(fusion_symbols::category::EnumerationError::NotEnumerable)
    );
    FUSIONS.set(0);
    let iter = cat.fusion(&z, &z).unwrap();
    assert_eq!(FUSIONS.get(), 3);
    assert_eq!(iter.clone().rev().count(), 8);
    assert_eq!(iter.count(), 8);
    assert_eq!(FUSIONS.get(), 3, "iteration must reuse initialized factors");
    let cat = Product::new(vec![Probe::default(); usize::BITS as usize]);
    let z = cat.unit();
    assert!(matches!(cat.fusion(&z, &z), Err(ProductError::Overflow)));
    let mut factors = cat.into_factors();
    factors.push(probe(0));
    let zero = Product::new(factors);
    let z = zero.unit();
    assert_eq!(zero.fusion(&z, &z).unwrap().len(), 0);
}

#[test]
fn runtime_complex_scalars_and_enumeration_overflow() {
    let fib = TableCategory::fibonacci();
    let tau = fib.object(2).unwrap();
    let runtime = Product::new(vec![fib.clone(), fib.clone()]);
    let tuple = Product::new((fib.clone(), fib));
    let x = vec![tau, tau];
    assert_eq!(
        runtime.frobenius_schur_phase(&x),
        tuple.frobenius_schur_phase(&(tau, tau))
    );
    assert_eq!(
        runtime.f_symbol(&x, &x, &x, &x, &x, &x).unwrap().as_slice(),
        tuple
            .f_symbol(
                &(tau, tau),
                &(tau, tau),
                &(tau, tau),
                &(tau, tau),
                &(tau, tau),
                &(tau, tau)
            )
            .unwrap()
            .as_slice()
    );
    assert_eq!(
        runtime.r_symbol(&x, &x, &x).unwrap().as_slice(),
        tuple
            .r_symbol(&(tau, tau), &(tau, tau), &(tau, tau))
            .unwrap()
            .as_slice()
    );
    let large = Product::new(vec![ZN::new(u32::MAX).unwrap(); 3]);
    assert_eq!(
        large.object_count(),
        Err(fusion_symbols::category::EnumerationError::Overflow)
    );
    for i in [0, 1, 42, u64::MAX] {
        let a = large.object_at(i).unwrap();
        assert_eq!(large.object_index(&a), Ok(i));
    }
}

type Flat<const R: usize, V = u64> = ([usize; R], Vec<V>);
type Frab<V = u64> = (Flat<4, V>, [Flat<2, V>; 3]);
fn values<const R: usize, S: Copy>(b: &impl SymbolBlock<R, Scalar = S>) -> Flat<R, S> {
    (b.dims(), b.as_slice().to_vec())
}
/// F, then R, A and B at fixed objects, as storage shapes and values.
fn frab_values<C: FusionCategory>(
    cat: &C,
    [a, b, c]: [&C::Object; 3],
    [fa, fb, fc, fd, fe, ff]: [&C::Object; 6],
) -> Frab<C::Scalar> {
    (
        values(&cat.f_symbol(fa, fb, fc, fd, fe, ff).unwrap()),
        [
            values(&cat.r_symbol(a, b, c).unwrap()),
            values(&cat.a_symbol(a, b, c).unwrap()),
            values(&cat.b_symbol(a, b, c).unwrap()),
        ],
    )
}
fn map_frab<V, W>((f, rab): Frab<V>, g: impl Fn(V) -> W) -> Frab<W> {
    fn map<const R: usize, V, W>((d, v): Flat<R, V>, g: &impl Fn(V) -> W) -> Flat<R, W> {
        (d, v.into_iter().map(g).collect())
    }
    (map(f, &g), rab.map(|x| map(x, &g)))
}
/// As [`frab_values`], as value bits.
fn frab<C: FusionCategory<Scalar = f64>>(
    cat: &C,
    objects: [&C::Object; 3],
    f_objects: [&C::Object; 6],
) -> Frab {
    map_frab(frab_values(cat, objects, f_objects), f64::to_bits)
}
fn complex_bits(z: Complex64) -> (u64, u64) {
    (z.re.to_bits(), z.im.to_bits())
}
/// Row-major Kronecker product, decoding each output element independently:
/// every axis index is `left_index * right_dim + right_index`.
fn kron_by<const R: usize, L: Copy, M: Copy, O>(
    (ld, l): &Flat<R, L>,
    (rd, r): &Flat<R, M>,
    multiply: impl Fn(L, M) -> O,
) -> Flat<R, O> {
    let dims: [usize; R] = std::array::from_fn(|i| ld[i] * rd[i]);
    let offset = |idx: [usize; R], dims: [usize; R]| (0..R).fold(0, |o, i| o * dims[i] + idx[i]);
    let data = (0..dims.iter().product())
        .map(|mut k| {
            let mut idx = [0; R];
            for i in (0..R).rev() {
                idx[i] = k % dims[i];
                k /= dims[i];
            }
            let x = l[offset(std::array::from_fn(|i| idx[i] / rd[i]), *ld)];
            let y = r[offset(std::array::from_fn(|i| idx[i] % rd[i]), *rd)];
            multiply(x, y)
        })
        .collect();
    (dims, data)
}
/// F/R/A/B Kronecker products of two factors, multiplying as `multiply`.
fn kron_frab<L: Copy, M: Copy, O>(
    (f, rab): &Frab<L>,
    (g, sab): &Frab<M>,
    multiply: impl Fn(L, M) -> O + Copy,
) -> Frab<O> {
    (
        kron_by(f, g, multiply),
        std::array::from_fn(|i| kron_by(&rab[i], &sab[i], multiply)),
    )
}
fn real_kron(l: &Frab, r: &Frab) -> Frab {
    kron_frab(l, r, |x, y| {
        (f64::from_bits(x) * f64::from_bits(y)).to_bits()
    })
}
/// Left fold of the leaf factors' F/R/A/B, as the product promises.
fn naive_product(leaves: &[Frab]) -> Frab {
    let (first, rest) = leaves.split_first().unwrap();
    rest.iter()
        .fold(first.clone(), |acc, leaf| real_kron(&acc, leaf))
}

// Validation must not perturb values: product F/R/A/B equal, bit for bit, the
// left-folded Kronecker product of the leaves' public symbols.
// SU(3) octets carry fusion multiplicity two, so every axis is nontrivial.
#[cfg(feature = "lie-generated")]
#[test]
fn product_symbols_are_bitwise_leaf_kronecker_products() {
    use fusion_symbols::finite::ZNIrrep;
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let o = su3.irrep(&[1, 1], &[]).unwrap();
    let leaf = frab(&su3, [&o; 3], [&o; 6]);

    let p = Product::new((su3.clone(), su3.clone()));
    let x = (o.clone(), o.clone());
    assert_eq!(
        frab(&p, [&x; 3], [&x; 6]),
        naive_product(&[leaf.clone(), leaf.clone()])
    );

    let q = Product::new((p, su3.clone()));
    let y = (x, o.clone());
    assert_eq!(
        frab(&q, [&y; 3], [&y; 6]),
        naive_product(&[leaf.clone(), leaf.clone(), leaf.clone()])
    );

    let z3 = ZN::new(3).unwrap();
    let [z0, z1, z2] = [0, 1, 2].map(|k| ZNIrrep::new(k, 3).unwrap());
    let w = Product::new((z3, su3.clone()));
    let [a, b, c] = [&z1, &z2, &z0].map(|z| (*z, o.clone()));
    assert_eq!(
        frab(&w, [&a, &b, &c], [&a, &b, &a, &a, &c, &c]),
        naive_product(&[
            frab(&z3, [&z1, &z2, &z0], [&z1, &z2, &z1, &z1, &z0, &z0]),
            leaf,
        ])
    );
}

#[test]
fn multiplicity_free_product_symbols_are_bitwise_leaf_kronecker_products() {
    let s = su2();
    let h = s.irrep(&[1], &[]).unwrap();
    let (p1, m1, u0) = (
        U1Irrep::new(1).unwrap(),
        U1Irrep::new(-1).unwrap(),
        U1.unit(),
    );
    let mf = Product::new((s.clone(), U1));
    let a = (h.clone(), p1);
    let b = (h.clone(), m1);
    let one = mf.unit();
    let z = s.unit();
    assert_eq!(
        frab(&mf, [&a, &b, &one], [&a, &b, &a, &a, &one, &one]),
        naive_product(&[
            frab(&s, [&h, &h, &z], [&h, &h, &h, &h, &z, &z]),
            frab(&U1, [&p1, &m1, &u0], [&p1, &m1, &p1, &p1, &u0, &u0]),
        ])
    );
}

// Composition order must not perturb values either: a right-nested product
// multiplies each element as `l * (m * r)`, a `Vec` product (also nested as a
// factor) folds left like a tuple.
#[cfg(feature = "lie-generated")]
#[test]
fn nested_and_vec_product_symbols_are_bitwise_leaf_kronecker_products() {
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let o = su3.irrep(&[1, 1], &[]).unwrap();
    let leaf = frab(&su3, [&o; 3], [&o; 6]);
    let pair = real_kron(&leaf, &leaf);

    let right = Product::new((su3.clone(), Product::new((su3.clone(), su3.clone()))));
    let x = (o.clone(), (o.clone(), o.clone()));
    assert_eq!(frab(&right, [&x; 3], [&x; 6]), real_kron(&leaf, &pair));

    let flat3 = Product::new(vec![su3.clone(); 3]);
    let x = vec![o.clone(); 3];
    assert_eq!(frab(&flat3, [&x; 3], [&x; 6]), real_kron(&pair, &leaf));

    let held = Product::new((Product::new(vec![su3.clone(); 2]), su3.clone()));
    let x = (vec![o.clone(); 2], o.clone());
    assert_eq!(frab(&held, [&x; 3], [&x; 6]), real_kron(&pair, &leaf));
    let held = Product::new((su3.clone(), Product::new(vec![su3.clone(); 2])));
    let x = (o.clone(), vec![o.clone(); 2]);
    assert_eq!(frab(&held, [&x; 3], [&x; 6]), real_kron(&leaf, &pair));
}

/// Real-by-complex F/R/A/B products, promoting each real entry as `Promote` does.
fn real_complex(l: &Frab<f64>, r: &Frab<Complex64>) -> Frab<(u64, u64)> {
    map_frab(kron_frab(l, r, |x, y| Complex64::from(x) * y), complex_bits)
}
#[cfg(feature = "lie-generated")]
fn complex_real(l: &Frab<Complex64>, r: &Frab<f64>) -> Frab<(u64, u64)> {
    map_frab(kron_frab(l, r, |x, y| x * Complex64::from(y)), complex_bits)
}

// A real factor met by a complex one is promoted entry by entry, after the real
// prefix has been multiplied in f64, exactly as a per-element left fold would.
#[test]
fn real_complex_product_symbols_are_bitwise_promoted_kronecker_products() {
    let fib = TableCategory::fibonacci();
    let tau = fib.object(2).unwrap();
    let (p1, m1, u0) = (
        U1Irrep::new(1).unwrap(),
        U1Irrep::new(-1).unwrap(),
        U1.unit(),
    );
    let p = Product::new((U1, fib.clone()));
    let [a, b, c] = [(p1, tau), (m1, tau), (u0, tau)];
    let cat = |x: &Frab<Complex64>| map_frab(x.clone(), complex_bits);
    assert_eq!(
        cat(&frab_values(&p, [&a, &b, &c], [&a, &b, &a, &a, &c, &c])),
        real_complex(
            &frab_values(&U1, [&p1, &m1, &u0], [&p1, &m1, &p1, &p1, &u0, &u0]),
            &frab_values(&fib, [&tau; 3], [&tau; 6]),
        )
    );
}

#[cfg(feature = "lie-generated")]
#[test]
fn multiplicity_real_complex_product_symbols_are_bitwise_promoted_kronecker_products() {
    let fib = TableCategory::fibonacci();
    let tau = fib.object(2).unwrap();
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let o = su3.irrep(&[1, 1], &[]).unwrap();
    let real = frab_values(&su3, [&o; 3], [&o; 6]);
    let complex = frab_values(&fib, [&tau; 3], [&tau; 6]);
    let pair = kron_frab(&real, &real, |x, y| x * y);
    let bits = |x: Frab<Complex64>| map_frab(x, complex_bits);

    // A 16-element real prefix leaves its f64 buffer for a complex one.
    let p = Product::new((su3.clone(), fib.clone()));
    let x = (o.clone(), tau);
    assert_eq!(
        bits(frab_values(&p, [&x; 3], [&x; 6])),
        real_complex(&real, &complex)
    );
    let p = Product::new((su3.clone(), su3.clone(), fib.clone()));
    let x = (o.clone(), o.clone(), tau);
    assert_eq!(
        bits(frab_values(&p, [&x; 3], [&x; 6])),
        real_complex(&pair, &complex)
    );
    // A complex prefix promotes each later real entry instead.
    let p = Product::new((fib.clone(), su3.clone()));
    let x = (tau, o.clone());
    assert_eq!(
        bits(frab_values(&p, [&x; 3], [&x; 6])),
        complex_real(&complex, &real)
    );
    let p = Product::new((su3.clone(), fib.clone(), su3.clone()));
    let x = (o.clone(), tau, o.clone());
    let left = kron_frab(&real, &complex, |x, y| Complex64::from(x) * y);
    assert_eq!(
        bits(frab_values(&p, [&x; 3], [&x; 6])),
        complex_real(&left, &real)
    );
}
