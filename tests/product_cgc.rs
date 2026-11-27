//! CGC isometries and the overlap of the two three-object coupling bases.
use fusion_symbols::{
    category::{CgcBlock, FusionCategory, GroupRepresentations, SymbolBlock},
    finite::{ZNIrrep, ZN},
    lie::{GroupId, LieGroup},
    product::Product,
};

fn su2() -> LieGroup {
    LieGroup::new(GroupId::su(2).unwrap()).unwrap()
}

fn assert_orthonormal(block: &impl CgcBlock) {
    let [da, db, dc, n] = block.dims();
    for c in 0..dc {
        for mu in 0..n {
            for other_c in 0..dc {
                for other_mu in 0..n {
                    let mut dot = 0.0;
                    for a in 0..da {
                        for b in 0..db {
                            dot += block.get([a, b, c, mu]) * block.get([a, b, other_c, other_mu]);
                        }
                    }
                    let want = f64::from(c == other_c && mu == other_mu);
                    assert!(
                        (dot - want).abs() < 1e-10,
                        "Gram {c},{mu};{other_c},{other_mu}: {dot}"
                    );
                }
            }
        }
    }
}

fn assert_f_from_cgc<C: GroupRepresentations<Scalar = f64>>(cat: &C, labels: [&C::Object; 6]) {
    let [a, b, c, d, e, f] = labels;
    let ab = cat.cgc(a, b, e).unwrap();
    let ec = cat.cgc(e, c, d).unwrap();
    let bc = cat.cgc(b, c, f).unwrap();
    let af = cat.cgc(a, f, d).unwrap();
    let expected = cat.f_symbol(a, b, c, d, e, f).unwrap();
    let [da, db, de, nmu] = ab.dims();
    let [_, dc, _, nnu] = ec.dims();
    let [_, _, df, nkappa] = bc.dims();
    let nlambda = af.dims()[3];
    // Schur's lemma: the coupling-basis overlap is F times identity on d.
    // Fix md = 0 and sum over the three uncoupled magnetic indices.
    let mut overlap = vec![0.0; nmu * nnu * nkappa * nlambda];
    for ma in 0..da {
        for mb in 0..db {
            for mc in 0..dc {
                let left: Vec<f64> = (0..nmu * nnu)
                    .map(|i| {
                        (0..de)
                            .map(|me| ab.get([ma, mb, me, i / nnu]) * ec.get([me, mc, 0, i % nnu]))
                            .sum()
                    })
                    .collect();
                let right: Vec<f64> = (0..nkappa * nlambda)
                    .map(|i| {
                        (0..df)
                            .map(|mf| {
                                bc.get([mb, mc, mf, i / nlambda]) * af.get([ma, mf, 0, i % nlambda])
                            })
                            .sum()
                    })
                    .collect();
                for (i, l) in left.iter().enumerate() {
                    for (j, r) in right.iter().enumerate() {
                        overlap[i * right.len() + j] += l * r;
                    }
                }
            }
        }
    }
    for mu in 0..nmu {
        for nu in 0..nnu {
            for kappa in 0..nkappa {
                for lambda in 0..nlambda {
                    let index = [mu, nu, kappa, lambda];
                    let got = overlap[((mu * nnu + nu) * nkappa + kappa) * nlambda + lambda];
                    let want = expected.get(index);
                    assert!(
                        (got - want).abs() < 1e-10,
                        "F {index:?}: {got}, expected {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn su2_su2_isometry_and_f() {
    let spin = su2();
    let one = spin.irrep(&[2]).unwrap();
    let cat = Product::new((spin.clone(), spin));
    let a = (one.clone(), one);
    assert_orthonormal(&cat.cgc(&a, &a, &a).unwrap());
    assert_f_from_cgc(&cat, [&a; 6]);
}

#[test]
fn su2_zn_isometry_and_f() {
    let spin = su2();
    let one = spin.irrep(&[2]).unwrap();
    let zn = ZN::new(3).unwrap();
    let z = |q| ZNIrrep::new(q, 3).unwrap();
    let cat = Product::new((spin, zn));
    let a = (one.clone(), z(1));
    let e = (one.clone(), z(2));
    let d = (one, z(0));
    assert_orthonormal(&cat.cgc(&a, &a, &e).unwrap());
    assert_f_from_cgc(&cat, [&a, &a, &a, &d, &e, &e]);
    let absent = cat.cgc(&a, &a, &a).unwrap();
    assert_eq!(absent.dims()[3], 0);
    assert!(std::panic::catch_unwind(|| absent.get([0; 4])).is_err());
}

#[cfg(feature = "lie-generated")]
#[test]
fn su3_su2_outer_multiplicity_isometry_and_f() {
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let eight = su3.irrep(&[1, 1]).unwrap();
    let spin = su2();
    let half = spin.irrep(&[1]).unwrap();
    let one = spin.irrep(&[2]).unwrap();
    let cat = Product::new((su3, spin));
    let a = (eight.clone(), half);
    let e = (eight, one);
    let block = cat.cgc(&a, &a, &e).unwrap();
    assert_eq!(block.dims(), [16, 16, 24, 2]);
    assert_orthonormal(&block);
    assert_f_from_cgc(&cat, [&a, &a, &a, &a, &e, &e]);
}

#[test]
fn last_factor_fastest_and_nested_products() {
    let spin = su2();
    let half = spin.irrep(&[1]).unwrap();
    let one = spin.irrep(&[2]).unwrap();
    let unit = spin.unit();
    let cat = Product::new((spin.clone(), spin.clone()));
    let a = (half.clone(), one.clone());
    let c = (one.clone(), unit.clone());
    let block = cat.cgc(&a, &a, &c).unwrap();
    let half_half_to_one = spin.cgc(&half, &half, &one).unwrap();
    let one_one_to_unit = spin.cgc(&one, &one, &unit).unwrap();
    let combined = [3, 2, 1, 0]; // [1*3+0, 0*3+2, 1*1+0, 0*1+0]
    let want = half_half_to_one.get([1, 0, 1, 0]) * one_one_to_unit.get([0, 2, 0, 0]);
    assert!(want.abs() > 0.1);
    assert_eq!(block.get(combined), want);
    assert!(std::panic::catch_unwind(|| block.get([6, 0, 0, 0])).is_err());
    let nested = Product::new((cat, ZN::new(3).unwrap()));
    let charge = ZNIrrep::new(0, 3).unwrap();
    let nested_block = nested
        .cgc(&(a.clone(), charge), &(a, charge), &(c, charge))
        .unwrap();
    assert_eq!(nested_block.dims(), block.dims());
    assert_eq!(nested_block.get(combined), want);
    // The arity-12 implementation remains lazy, including explicitly nested tuples.
    let z = ZN::new(3).unwrap();
    let twelve = Product::new((z, z, z, z, z, z, z, z, z, z, z, z));
    assert_eq!(
        twelve
            .cgc(&twelve.unit(), &twelve.unit(), &twelve.unit())
            .unwrap()
            .get([0; 4]),
        1.0
    );
}

#[cfg(feature = "lie-generated")]
#[test]
fn su3_su3_multiplicity_last_factor_fastest() {
    let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
    let eight = su3.irrep(&[1, 1]).unwrap();
    let c = su3.cgc(&eight, &eight, &eight).unwrap();
    let cat = Product::new((su3.clone(), su3));
    let a = (eight.clone(), eight);
    let block = cat.cgc(&a, &a, &a).unwrap();
    assert_eq!(block.dims(), [64, 64, 64, 4]);
    let want_10 = c.get([0, 2, 0, 1]) * c.get([2, 0, 0, 0]);
    let want_01 = c.get([0, 2, 0, 0]) * c.get([2, 0, 0, 1]);
    assert!((want_10 - want_01).abs() > 0.1);
    // Last factor fastest: magnetic [0*8+2, 2*8+0, 0*8+0], mu = mu0*2+mu1.
    let [i, j, k] = [2, 16, 0];
    assert_eq!(block.get([i, j, k, 2]), want_10); // 1*2+0
    assert_eq!(block.get([i, j, k, 1]), want_01); // 0*2+1
}

#[test]
fn zn_scalar_channels_and_bounds() {
    let cat = ZN::new(3).unwrap();
    let z = |q| ZNIrrep::new(q, 3).unwrap();
    let present = cat.cgc(&z(1), &z(1), &z(2)).unwrap();
    let absent = cat.cgc(&z(1), &z(1), &z(0)).unwrap();
    assert_eq!(present, 1.0);
    assert_eq!(absent, 0.0);
    assert_eq!(CgcBlock::dims(&present), [1; 4]);
    assert_eq!(CgcBlock::dims(&absent), [1, 1, 1, 0]);
    // `f64` is also a scalar `SymbolBlock`; name the CGC reader explicitly.
    assert_eq!(CgcBlock::get(&present, [0; 4]), 1.0);
    assert!(std::panic::catch_unwind(|| CgcBlock::get(&absent, [0; 4])).is_err());
    for axis in 0..4 {
        let mut index = [0; 4];
        index[axis] = 1;
        assert!(std::panic::catch_unwind(|| CgcBlock::get(&present, index)).is_err());
    }
    assert!(
        std::panic::catch_unwind(|| cat.cgc(&z(1), &z(1), &ZNIrrep::new(2, 4).unwrap())).is_err()
    );
}

#[test]
fn axis_overflow_factor_errors_and_foreign_objects() {
    use fusion_symbols::{lie::LieError, product::ProductError};
    let spin = su2();
    let huge = spin.irrep(&[i64::from(u32::MAX)]).unwrap();
    let unit = spin.unit();
    let cat = Product::new((spin.clone(), spin));
    let h = (huge.clone(), huge.clone());
    let u = (unit.clone(), unit.clone());
    // Absent channels need no dense allocation, but each magnetic axis must fit.
    for labels in [[&h, &u, &u], [&u, &h, &u], [&u, &u, &h]] {
        assert!(matches!(
            cat.cgc(labels[0], labels[1], labels[2]),
            Err(ProductError::Overflow)
        ));
    }
    assert!(matches!(
        cat.cgc(&h, &h, &u),
        Err(ProductError::Factor0(LieError::Unsupported { .. }))
    ));
    let right_huge = (unit, huge);
    assert!(matches!(
        cat.cgc(&right_huge, &right_huge, &u),
        Err(ProductError::Factor1(LieError::Unsupported { .. }))
    ));
    let zn = ZN::new(3).unwrap();
    let zero = zn.unit();
    let mixed = Product::new((cat, zn));
    let foreign = ZNIrrep::new(0, 4).unwrap();
    // The foreign last label must panic before the first factor can return an error.
    assert!(std::panic::catch_unwind(|| mixed.cgc(
        &(h.clone(), zero),
        &(h.clone(), zero),
        &(u.clone(), foreign)
    ))
    .is_err());
}

#[test]
fn d2_isometry_and_f() {
    let cat = LieGroup::new(GroupId::spin(4).unwrap()).unwrap();
    let a = cat.irrep(&[2, 2]).unwrap();
    assert_orthonormal(&cat.cgc(&a, &a, &a).unwrap());
    assert_f_from_cgc(&cat, [&a; 6]);
}

#[cfg(feature = "lie-generated")]
#[test]
fn a1_a2_isometry_and_f() {
    use fusion_symbols::lie::RootSystem::A;
    let cat = LieGroup::new(GroupId::cover(&[A(1), A(2)], 0).unwrap()).unwrap();
    let a = cat.irrep(&[1, 1, 1]).unwrap();
    let e = cat.irrep(&[2, 1, 1]).unwrap();
    assert_orthonormal(&cat.cgc(&a, &a, &e).unwrap());
    assert_f_from_cgc(&cat, [&a, &a, &a, &a, &e, &e]);
}

// Coordinate-coded synthetic block: the value names its own index.
struct Coded([usize; 4]);
impl CgcBlock for Coded {
    fn dims(&self) -> [usize; 4] {
        self.0
    }
    fn get(&self, index: [usize; 4]) -> f64 {
        assert!(index.iter().zip(self.0).all(|(&i, n)| i < n));
        let [a, b, c, mu] = index.map(|i| i as f64);
        1000.0 * a + 100.0 * b + 10.0 * c + mu
    }
}

// A block whose reads must never happen.
struct Unread([usize; 4]);
impl CgcBlock for Unread {
    fn dims(&self) -> [usize; 4] {
        self.0
    }
    fn get(&self, _: [usize; 4]) -> f64 {
        panic!("read")
    }
}

fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let payload = std::panic::catch_unwind(f).unwrap_err();
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap()
}

#[test]
fn to_dense_is_row_major_multiplicity_fastest() {
    let dense = Coded([2, 3, 2, 2]).to_dense();
    assert_eq!(dense.len(), 24);
    assert_eq!(&dense[..5], &[0.0, 1.0, 10.0, 11.0, 100.0]);
    assert_eq!(dense[23], 1211.0);
    // Decode each flat position independently: mu fastest, then mc, mb, ma.
    for (flat, &value) in dense.iter().enumerate() {
        let (mu, rest) = (flat % 2, flat / 2);
        let (mc, rest) = (rest % 2, rest / 2);
        let (mb, ma) = (rest % 3, rest / 3);
        assert_eq!(value, (1000 * ma + 100 * mb + 10 * mc + mu) as f64);
    }
}

#[test]
fn to_dense_capacity_and_empty_shapes() {
    // A zero axis returns before multiplying the other axes or reading.
    for axis in 0..4 {
        let mut dims = [usize::MAX; 4];
        dims[axis] = 0;
        assert!(Unread(dims).to_dense().is_empty());
    }
    // Unrepresentable element or byte counts panic before any read or allocation.
    let elements = panic_message(|| drop(Unread([usize::MAX, 2, 1, 1]).to_dense()));
    assert!(elements.starts_with("dense CGC size overflow"));
    let bytes = panic_message(|| drop(Unread([usize::MAX / 8 + 1, 1, 1, 1]).to_dense()));
    assert!(bytes.starts_with("dense CGC byte size overflow"));
    // Scalars: a present abelian channel is [1.0], an absent one is empty;
    // the empty vector product is the trivial one-entry CGC.
    assert_eq!(CgcBlock::to_dense(&1.0f64), [1.0]);
    assert!(CgcBlock::to_dense(&0.0f64).is_empty());
    let empty = Product::new(Vec::<ZN>::new());
    let unit = empty.unit();
    assert_eq!(empty.cgc(&unit, &unit, &unit).unwrap().to_dense(), [1.0]);
}

#[test]
fn to_dense_products_expand_factors_last_fastest() {
    let spin = su2();
    let half = spin.irrep(&[1]).unwrap();
    let one = spin.irrep(&[2]).unwrap();
    let c0 = spin.cgc(&half, &half, &one).unwrap(); // [2, 2, 3, 1]
    let c1 = spin.cgc(&one, &half, &half).unwrap(); // [3, 2, 2, 1]
                                                    // Independent Kronecker expansion over factor indices.
    let [a0, b0, e0, n0] = c0.dims();
    let [a1, b1, e1, n1] = c1.dims();
    let (db, de, dn) = (b0 * b1, e0 * e1, n0 * n1);
    let mut expected = vec![f64::NAN; a0 * a1 * db * de * dn];
    for i0 in 0..a0 {
        for j0 in 0..b0 {
            for k0 in 0..e0 {
                for i1 in 0..a1 {
                    for j1 in 0..b1 {
                        for k1 in 0..e1 {
                            let (ma, mb, mc) = (i0 * a1 + i1, j0 * b1 + j1, k0 * e1 + k1);
                            expected[((ma * db + mb) * de + mc) * dn] =
                                c0.get([i0, j0, k0, 0]) * c1.get([i1, j1, k1, 0]);
                        }
                    }
                }
            }
        }
    }
    let cat = Product::new((spin.clone(), spin.clone()));
    let (a, b, c) = (
        (half.clone(), one.clone()),
        (half.clone(), half.clone()),
        (one.clone(), half.clone()),
    );
    let tuple = cat.cgc(&a, &b, &c).unwrap().to_dense();
    // Each entry is the once-rounded exact product, within an ulp of the
    // product of the rounded factors (bits: `su2_su2_block_is_…`).
    assert_eq!(tuple.len(), expected.len());
    for (x, y) in tuple.iter().zip(&expected) {
        assert!((x - y).abs() <= f64::EPSILON * y.abs(), "{x} {y}");
    }
    let vector = Product::new(vec![spin.clone(); 2])
        .cgc(
            &vec![half.clone(), one.clone()],
            &vec![half.clone(), half.clone()],
            &vec![one.clone(), half.clone()],
        )
        .unwrap();
    assert_eq!(vector.to_dense(), tuple);
    // A trivial ZN factor nested on the right changes no value.
    let z = ZN::new(3).unwrap();
    let nested = Product::new((cat, z));
    let q = ZNIrrep::new(0, 3).unwrap();
    let block = nested.cgc(&(a, q), &(b, q), &(c, q)).unwrap();
    let mut first = block.to_dense();
    assert_eq!(first, tuple);
    // Each call returns independent owned storage.
    first[0] = 7.0;
    assert_eq!(block.to_dense(), tuple);
}

/// `⟨j₁ m₁; j₂ m₂ | J M⟩²` by Racah's formula, every argument doubled.
fn racah_squared(j1: i64, m1: i64, j2: i64, m2: i64, j: i64, m: i64) -> malachite_q::Rational {
    use malachite_nz::natural::Natural;
    use malachite_q::Rational;
    let half = |x: i64| (x >= 0 && x % 2 == 0).then_some(x / 2);
    let fact = |n: i64| (1..=n as u64).map(Natural::from).product::<Natural>();
    let parts = [
        j + j1 - j2,
        j - j1 + j2,
        j1 + j2 - j,
        j1 + j2 + j,
        j + m,
        j - m,
        j1 - m1,
        j1 + m1,
        j2 - m2,
        j2 + m2,
    ]
    .map(half);
    let (true, Some([a, b, c, s, jp, jm, am, ap, bm, bp])) = (
        m1 + m2 == m,
        parts
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .map(|v| <[i64; 10]>::try_from(v).unwrap()),
    ) else {
        return Rational::from(0);
    };
    let pre = Rational::from(j + 1)
        * Rational::from_naturals(fact(a) * fact(b) * fact(c), fact(s + 1))
        * Rational::from(fact(jp) * fact(jm) * fact(am) * fact(ap) * fact(bm) * fact(bp));
    let mut sum = Rational::from(0);
    for k in 0..=c {
        let terms = [
            k,
            c - k,
            am - k,
            bp - k,
            (j - j2 + m1) / 2 + k,
            (j - j1 - m2) / 2 + k,
        ];
        if terms.iter().any(|&t| t < 0) {
            continue;
        }
        let den = terms.iter().map(|&t| fact(t)).product::<Natural>();
        let term = Rational::from_naturals(Natural::from(1u32), den);
        if k % 2 == 0 {
            sum += term;
        } else {
            sum -= term;
        }
    }
    pre * &sum * &sum
}

/// The `f64` nearest to `√r` for a positive rational, by exact midpoint
/// comparisons (no tie occurs for an irrational root or an exact one).
fn nearest_sqrt(r: &malachite_q::Rational) -> f64 {
    use malachite_base::num::conversion::traits::{ExactFrom, RoundingFrom};
    use malachite_base::rounding_modes::RoundingMode;
    use malachite_q::Rational;
    let next = |y: f64| f64::from_bits(y.to_bits() + 1);
    let prev = |y: f64| f64::from_bits(y.to_bits() - 1);
    let mid_squared = |a: f64, b: f64| {
        let m = (Rational::exact_from(a) + Rational::exact_from(b)) / Rational::from(2);
        &m * &m
    };
    let mut y = f64::rounding_from(r, RoundingMode::Nearest).0.sqrt();
    loop {
        let (up, down) = (mid_squared(y, next(y)), mid_squared(prev(y), y));
        assert!(*r != up && *r != down);
        if *r > up {
            y = next(y);
        } else if *r < down {
            y = prev(y);
        } else {
            return y;
        }
    }
}

// Purpose: every entry of a product of two SU(2) CGCs is the exact product
// of the two Racah coefficients, rounded once; the oracle composes the
// Racah squares rationally (signs from the factors' own blocks). An exactly
// zero entry is +0.
#[test]
fn su2_su2_block_is_the_rounded_exact_product() {
    let g = su2();
    let cat = Product::new((g.clone(), g.clone()));
    let pair = |x, y| (g.irrep(&[x]).unwrap(), g.irrep(&[y]).unwrap());
    for [ja0, ja1, jb0, jb1, jc0, jc1] in
        [[1, 2, 1, 1, 2, 1], [3, 2, 2, 2, 3, 2], [4, 3, 2, 3, 2, 4]]
    {
        let [a, b, c] = [(ja0, ja1), (jb0, jb1), (jc0, jc1)].map(|(x, y)| pair(x, y));
        let block = cat.cgc(&a, &b, &c).unwrap();
        let l0 = g.cgc(&a.0, &b.0, &c.0).unwrap();
        let l1 = g.cgc(&a.1, &b.1, &c.1).unwrap();
        let [da1, db1, dc1] = [ja1, jb1, jc1].map(|j| j as usize + 1);
        let [da, db, dc, n] = block.dims();
        assert_eq!(n, 1);
        let m = |i: usize, tj: i64| 2 * i as i64 - tj;
        for i in 0..da {
            for j in 0..db {
                for k in 0..dc {
                    let [x0, x1] = [
                        [i / da1, j / db1, k / dc1, 0],
                        [i % da1, j % db1, k % dc1, 0],
                    ];
                    let r =
                        racah_squared(ja0, m(x0[0], ja0), jb0, m(x0[1], jb0), jc0, m(x0[2], jc0))
                            * racah_squared(
                                ja1,
                                m(x1[0], ja1),
                                jb1,
                                m(x1[1], jb1),
                                jc1,
                                m(x1[2], jc1),
                            );
                    let sign = l0.get(x0).signum() * l1.get(x1).signum();
                    let want = if r == 0 { 0.0 } else { sign * nearest_sqrt(&r) };
                    assert_eq!(block.get([i, j, k, 0]).to_bits(), want.to_bits());
                }
            }
        }
    }
    // Witness: -√(1/3) = ⟨½ -½; ½ ½ | 1 0⟩ · ⟨1 -1; ½ ½ | ½ -½⟩ rounds to
    // ...331c once, while the product of the rounded factors is ...331d.
    let [a, b, c] = [pair(1, 2), pair(1, 1), pair(2, 1)];
    let exact = cat.cgc(&a, &b, &c).unwrap().get([0, 3, 2, 0]);
    let float = g.cgc(&a.0, &b.0, &c.0).unwrap().get([0, 1, 1, 0])
        * g.cgc(&a.1, &b.1, &c.1).unwrap().get([0, 1, 0, 0]);
    assert_eq!(exact.to_bits(), 0xbfe2_79a7_4590_331c);
    assert_eq!(float.to_bits(), 0xbfe2_79a7_4590_331d);
}

// Purpose: a product's CGC depends only on its ordered Radical leaves: a flat
// tuple, nested tuples with a unit factor, a vector and a composite Lie factor
// give the same bits.
#[test]
fn flat_nested_vec_and_composite_products_agree_bitwise() {
    let g = su2();
    let z = ZN::new(3).unwrap();
    let q = ZNIrrep::new(0, 3).unwrap();
    let spin4 = LieGroup::new(GroupId::spin(4).unwrap()).unwrap();
    let x = |j| g.irrep(&[j]).unwrap();
    let labels = [[1, 2, 1], [1, 1, 2], [2, 1, 1]];
    let bits = |block: fusion_symbols::lie::LieCgc| {
        block
            .to_dense()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    let [a, b, c] = labels.map(|[p, r, s]| (x(p), x(r), x(s)));
    let flat = Product::new((g.clone(), g.clone(), g.clone()));
    let want = bits(flat.cgc(&a, &b, &c).unwrap());
    assert!(want.iter().any(|&v| v != 0));
    let nested = Product::new((g.clone(), Product::new((g.clone(), z, g.clone()))));
    let [a, b, c] = labels.map(|[p, r, s]| (x(p), (x(r), q, x(s))));
    assert_eq!(bits(nested.cgc(&a, &b, &c).unwrap()), want);
    let vector = Product::new(vec![g.clone(); 3]);
    let [a, b, c] = labels.map(|l| l.map(x).to_vec());
    assert_eq!(bits(vector.cgc(&a, &b, &c).unwrap()), want);
    let composite = Product::new((g.clone(), spin4.clone()));
    let [a, b, c] = labels.map(|[p, r, s]| (x(p), spin4.irrep(&[r, s]).unwrap()));
    assert_eq!(bits(composite.cgc(&a, &b, &c).unwrap()), want);
}

/// A downstream group forwarding its CGCs, outside the exact protocol.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Forward(LieGroup);
impl FusionCategory for Forward {
    type Object = fusion_symbols::lie::LieIrrep;
    type Scalar = f64;
    type FSymbol = <LieGroup as FusionCategory>::FSymbol;
    type RSymbol = <LieGroup as FusionCategory>::RSymbol;
    type Error = fusion_symbols::lie::LieError;
    type Fusion<'a> = <LieGroup as FusionCategory>::Fusion<'a>;
    fn fusion_style(&self) -> fusion_symbols::category::FusionStyle {
        self.0.fusion_style()
    }
    fn braiding_style(&self) -> fusion_symbols::category::BraidingStyle {
        self.0.braiding_style()
    }
    fn unit(&self) -> Self::Object {
        self.0.unit()
    }
    fn contains(&self, a: &Self::Object) -> bool {
        self.0.contains(a)
    }
    fn dual(&self, a: &Self::Object) -> Self::Object {
        self.0.dual(a)
    }
    fn dim(&self, a: &Self::Object) -> f64 {
        self.0.dim(a)
    }
    fn frobenius_schur_phase(&self, a: &Self::Object) -> Result<f64, Self::Error> {
        self.0.frobenius_schur_phase(a)
    }
    fn frobenius_schur_indicator(&self, a: &Self::Object) -> f64 {
        self.0.frobenius_schur_indicator(a)
    }
    fn twist(&self, a: &Self::Object) -> Result<f64, Self::Error> {
        self.0.twist(a)
    }
    fn fusion<'a>(
        &'a self,
        a: &Self::Object,
        b: &Self::Object,
    ) -> Result<Self::Fusion<'a>, Self::Error> {
        self.0.fusion(a, b)
    }
    fn nsymbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<u32, Self::Error> {
        self.0.nsymbol(a, b, c)
    }
    fn f_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
        d: &Self::Object,
        e: &Self::Object,
        f: &Self::Object,
    ) -> Result<Self::FSymbol, Self::Error> {
        self.0.f_symbol(a, b, c, d, e, f)
    }
    fn r_symbol(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<Self::RSymbol, Self::Error> {
        self.0.r_symbol(a, b, c)
    }
    fn object_count(&self) -> Result<Option<u64>, fusion_symbols::category::EnumerationError> {
        self.0.object_count()
    }
    fn object_at(
        &self,
        index: u64,
    ) -> Result<Self::Object, fusion_symbols::category::EnumerationError> {
        self.0.object_at(index)
    }
    fn object_index(
        &self,
        a: &Self::Object,
    ) -> Result<u64, fusion_symbols::category::EnumerationError> {
        self.0.object_index(a)
    }
}
impl GroupRepresentations for Forward {
    type Cgc = fusion_symbols::lie::LieCgc;
    fn cgc(
        &self,
        a: &Self::Object,
        b: &Self::Object,
        c: &Self::Object,
    ) -> Result<Self::Cgc, Self::Error> {
        self.0.cgc(a, b, c)
    }
}

// Purpose: a product with a factor outside the exact protocol keeps the one
// float path: each entry is the product of the factors' rounded entries, so
// the witness of `su2_su2_block_is_the_rounded_exact_product` keeps its
// float bits there.
#[test]
fn a_downstream_factor_keeps_the_float_product() {
    let g = su2();
    let x = |j| g.irrep(&[j]).unwrap();
    let cat = Product::new((Forward(g.clone()), g.clone()));
    let [a, b, c] = [(x(1), x(2)), (x(1), x(1)), (x(2), x(1))];
    let block = cat.cgc(&a, &b, &c).unwrap();
    assert_eq!(block.dims(), [6, 4, 6, 1]);
    assert_eq!(block.get([0, 3, 2, 0]).to_bits(), 0xbfe2_79a7_4590_331d);
    let (l0, l1) = (
        g.cgc(&a.0, &b.0, &c.0).unwrap(),
        g.cgc(&a.1, &b.1, &c.1).unwrap(),
    );
    let dense = block.to_dense();
    let mut flat = 0;
    for i in 0..6 {
        for j in 0..4 {
            for k in 0..6 {
                let want = l0.get([i / 3, j / 2, k / 2, 0]) * l1.get([i % 3, j % 2, k % 2, 0]);
                // An exactly zero entry has no stored value and reads +0.
                assert_eq!(dense[flat].to_bits(), (want + 0.0).to_bits());
                flat += 1;
            }
        }
    }
}

// Purpose: a composite whose axes fit `u32` but whose sparse entries
// (61^10 here, 1.7e19 bytes) cannot be allocated is an overflow error on
// both the exact and the float path, not an allocation panic.
#[test]
fn unallocatable_composite_cgc_is_overflow() {
    let g = su2();
    let x = g.irrep(&[8]).unwrap();
    let a = vec![x; 10];
    let exact = Product::new(vec![g.clone(); 10]);
    assert!(matches!(
        exact.cgc(&a, &a, &a),
        Err(fusion_symbols::product::ProductError::Overflow)
    ));
    let float = Product::new(vec![Forward(g); 10]);
    assert!(matches!(
        float.cgc(&a, &a, &a),
        Err(fusion_symbols::product::ProductError::Overflow)
    ));
}
