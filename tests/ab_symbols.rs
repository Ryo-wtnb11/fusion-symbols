//! A- and B-symbols (`FusionCategory::a_symbol`, `b_symbol`).
//!
//! Contract: the TensorKitSectors 0.3.9 `Asymbol` / `Bsymbol` values
//! (`src/sectors.jl::Asymbol_from_Fsymbol`, `::Bsymbol_from_Fsymbol`), product
//! blocks as `src/product.jl::Asymbol` / `::Bsymbol` (factor Kronecker, last
//! factor fastest), empty `0 × 0` blocks for absent channels, and no braiding
//! requirement. Literals were printed by TensorKitSectors 0.3.9 and
//! SUNRepresentations.jl 0.4.0 on Julia 1.11.6, entry by entry with `μ` outer
//! and `ν` inner; where the reference prints `±1.0000000000000002` the exact
//! value `±1` is written. Unitarity of A and B is checked separately as a
//! formula-independent structural property.

use fusion_symbols::{
    category::{FusionCategory, RBlock, Scalar},
    finite::{ZNIrrep, ZN},
    lie::{CU1Irrep, GroupId, LieGroup, U1Irrep, CU1, U1},
    product::product,
    table::TableCategory,
};
use num_complex::Complex64;

const TOL: f64 = 1e-12;

fn su2() -> LieGroup {
    LieGroup::new(GroupId::su(2).unwrap()).unwrap()
}

/// The single entry of a `1 × 1` block.
fn entry<S: Scalar>(block: RBlock<S>) -> S {
    assert_eq!(block.dim(), 1);
    block.at(0, 0)
}

/// Worst `|M M† − I|` over a row-major `n × n` block.
fn unitarity<S: Scalar>(n: usize, m: &[S]) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut dot = S::zero();
            for k in 0..n {
                dot = dot + m[i * n + k] * m[j * n + k].conj();
            }
            let target = if i == j { S::one() } else { S::zero() };
            worst = worst.max((dot - target).abs());
        }
    }
    worst
}

/// Every admissible A and B of `cat` over `objects` is unitary, and every
/// absent channel is the empty `0 × 0` block.
fn assert_unitary<C: FusionCategory>(cat: &C, objects: &[C::Object]) {
    for a in objects {
        for b in objects {
            for c in objects {
                let n = cat.nsymbol(a, b, c) as usize;
                for block in [
                    cat.a_symbol(a, b, c).unwrap(),
                    cat.b_symbol(a, b, c).unwrap(),
                ] {
                    assert_eq!(block.dim(), n, "{a:?} {b:?} {c:?}");
                    if n == 0 {
                        assert!(block.data().is_empty(), "{a:?} {b:?} {c:?}");
                        continue;
                    }
                    assert!(
                        unitarity(n, block.data()) < TOL,
                        "{a:?} {b:?} {c:?}: not unitary"
                    );
                }
            }
        }
    }
}

#[test]
fn su2_matches_tensorkitsectors_including_negative_frobenius_schur() {
    let cat = su2();
    let j = |twice: i64| cat.irrep(&[twice], &[]).unwrap();
    assert_eq!(cat.frobenius_schur_phase(&j(1)), -1.0);
    // (2a, 2b, 2c, A, B).
    for (a, b, c, want_a, want_b) in [
        (1, 1, 0, 1.0, -1.0),
        (1, 1, 2, -1.0, 1.0),
        (1, 2, 1, 1.0, -1.0),
        (2, 1, 1, -1.0, -1.0),
        (2, 2, 2, -1.0, -1.0),
        (3, 2, 1, -1.0, 1.0),
    ] {
        let (a, b, c) = (j(a), j(b), j(c));
        let got_a = cat.a_symbol(&a, &b, &c).unwrap();
        let got_b = cat.b_symbol(&a, &b, &c).unwrap();
        assert_eq!((got_a.dim(), got_b.dim()), (1, 1));
        assert!((got_a.at(0, 0) - want_a).abs() < TOL, "A {a:?} {b:?} {c:?}");
        assert!((got_b.at(0, 0) - want_b).abs() < TOL, "B {a:?} {b:?} {c:?}");
    }
    // Absent channel: the empty `0 × 0` block.
    let absent = cat.a_symbol(&j(1), &j(1), &j(4)).unwrap();
    assert_eq!((absent.dim(), absent.data()), (0, &[][..]));
    assert_eq!(cat.b_symbol(&j(1), &j(1), &j(4)).unwrap().dim(), 0);
    assert_unitary(&cat, &(0..=6).map(j).collect::<Vec<_>>());
}

#[test]
fn cu1_charged_sectors_match_tensorkitsectors() {
    let j = CU1Irrep::from_twice_charge;
    let (zp, zm) = (CU1Irrep::VACUUM, CU1Irrep::PSEUDOSCALAR);
    for (a, b, c, want_a, want_b) in [
        (j(1), j(1), zp, 1.0, 1.0),
        (j(1), j(1), zm, -1.0, 1.0),
        (j(1), j(1), j(2), 1.0, 1.0),
        (j(1), j(2), j(1), 1.0, 1.0),
        (j(2), j(1), j(3), 1.0, 1.0),
        (j(1), zm, j(1), -1.0, 1.0),
        (zm, j(1), j(1), 1.0, 1.0),
    ] {
        assert!((entry(CU1.a_symbol(&a, &b, &c).unwrap()) - want_a).abs() < TOL);
        assert!((entry(CU1.b_symbol(&a, &b, &c).unwrap()) - want_b).abs() < TOL);
    }
    assert_eq!(CU1.a_symbol(&j(1), &j(1), &j(4)).unwrap().dim(), 0);
    let objects: Vec<_> = [zp, zm].into_iter().chain((1..=4).map(j)).collect();
    assert_unitary(&CU1, &objects);
}

#[test]
fn abelian_groups_return_n() {
    // TensorKitSectors `irreps.jl` (AbelianIrrep): A = B = N.
    let q = |c| U1Irrep::new(c).unwrap();
    assert_eq!(entry(U1.a_symbol(&q(2), &q(-3), &q(-1)).unwrap()), 1.0);
    assert_eq!(entry(U1.b_symbol(&q(2), &q(-3), &q(-1)).unwrap()), 1.0);
    assert_eq!(U1.a_symbol(&q(2), &q(-3), &q(1)).unwrap().dim(), 0);
    let z3 = ZN::new(3).unwrap();
    let objects: Vec<_> = (0..3).map(|k| ZNIrrep::new(k, 3).unwrap()).collect();
    assert_unitary(&z3, &objects);
}

#[test]
fn fibonacci_matches_tensorkitsectors_in_the_same_gauge() {
    let fib = TableCategory::fibonacci();
    let (one, tau) = (fib.object(1).unwrap(), fib.object(2).unwrap());
    for (a, b, c) in [
        (tau, tau, one),
        (tau, tau, tau),
        (one, tau, tau),
        (tau, one, tau),
    ] {
        // TensorKitSectors prints 0.9999999999999999 for (τ, τ, 𝟙), else 1.0.
        let want = Complex64::new(1.0, 0.0);
        assert!((entry(fib.a_symbol(&a, &b, &c).unwrap()) - want).norm() < TOL);
        assert!((entry(fib.b_symbol(&a, &b, &c).unwrap()) - want).norm() < TOL);
    }
    assert_eq!(fib.a_symbol(&one, &tau, &one).unwrap().dim(), 0);
}

#[test]
fn ising_matches_tensorkitsectors_up_to_the_documented_vertex_gauge() {
    // Here 1, ψ = 2, σ = 3; `TableCategory::ising` documents the vertex gauge
    // u^{σσ}_ψ = −1 relative to TensorKitSectors' `IsingAnyon`, under which
    // A_here = A_TKS · u^{ā a}_𝟙 u^{𝟙 b}_b / (u^{ab}_c u^{ā c}_b) and
    // B_here = B_TKS · u^{ab}_c u^{c b̄}_a / (u^{b b̄}_𝟙 u^{a 𝟙}_a).
    let ising = TableCategory::ising();
    let o = |i| ising.object(i).unwrap();
    let (one, psi, sigma) = (o(1), o(2), o(3));
    let u = |a, b, c| {
        if (a, b, c) == (sigma, sigma, psi) {
            -1.0
        } else {
            1.0
        }
    };
    // (a, b, c, A_TKS, B_TKS); every self-dual. TensorKitSectors prints
    // 0.9999999999999998 for the first two.
    for (a, b, c, a_tks, b_tks) in [
        (sigma, sigma, one, 1.0, 1.0),
        (sigma, sigma, psi, 1.0, 1.0),
        (sigma, psi, sigma, 1.0, 1.0),
        (psi, sigma, sigma, 1.0, 1.0),
        (psi, psi, one, 1.0, 1.0),
    ] {
        let gauge_a = u(a, a, one) * u(one, b, b) / (u(a, b, c) * u(a, c, b));
        let gauge_b = u(a, b, c) * u(c, b, a) / (u(b, b, one) * u(a, one, a));
        let got_a = entry(ising.a_symbol(&a, &b, &c).unwrap());
        let got_b = entry(ising.b_symbol(&a, &b, &c).unwrap());
        assert!(
            (got_a - a_tks * gauge_a).norm() < TOL,
            "A {a:?} {b:?} {c:?}"
        );
        assert!(
            (got_b - b_tks * gauge_b).norm() < TOL,
            "B {a:?} {b:?} {c:?}"
        );
    }
    // The gauge is visible: A^{σσ}_ψ = −1 here.
    assert!((entry(ising.a_symbol(&sigma, &sigma, &psi).unwrap()) + 1.0).norm() < TOL);
}

#[test]
fn every_shipped_table_including_fusion_only_has_unitary_a_and_b() {
    let mut fusion_only = 0;
    for uuid in TableCategory::shipped_uuids() {
        let cat = TableCategory::by_uuid(uuid).unwrap();
        fusion_only += usize::from(!cat.is_braided());
        let objects: Vec<_> = cat.objects().collect();
        assert_unitary(&cat, &objects);
    }
    assert!(fusion_only > 0);
}

#[test]
fn fibonacci_times_su2_promotes_to_complex() {
    // TensorKitSectors `FibonacciAnyon ⊠ SU2Irrep`: A = −1, B = 1 at
    // (τ,½) ⊗ (τ,½) → (τ,1) (the reference prints ±1.0000000000000002).
    let (fib, su2) = (TableCategory::fibonacci(), su2());
    let cat = product(fib.clone(), su2.clone());
    let tau = fib.object(2).unwrap();
    let (half, one) = (su2.irrep(&[1], &[]).unwrap(), su2.irrep(&[2], &[]).unwrap());
    let (x, y) = ((tau, half.clone()), (tau, one));
    let a = cat.a_symbol(&x, &x, &y).unwrap();
    let b = cat.b_symbol(&x, &x, &y).unwrap();
    assert_eq!((a.dim(), b.dim()), (1, 1));
    assert!((a.at(0, 0) - Complex64::new(-1.0, 0.0)).norm() < TOL);
    assert!((b.at(0, 0) - Complex64::new(1.0, 0.0)).norm() < TOL);
}

#[test]
fn absent_channels_and_foreign_objects() {
    let (fib, su2) = (TableCategory::fibonacci(), su2());
    let cat = product(fib.clone(), su2.clone());
    let (one, tau) = (fib.object(1).unwrap(), fib.object(2).unwrap());
    let half = su2.irrep(&[1], &[]).unwrap();
    let x = (tau, half.clone());
    let absent = (one, half.clone());
    assert_eq!(cat.a_symbol(&x, &x, &absent).unwrap().dim(), 0);
    assert!(cat.b_symbol(&x, &x, &absent).unwrap().data().is_empty());

    let ising = TableCategory::ising();
    let foreign = ising.object(3).unwrap();
    assert!(std::panic::catch_unwind(|| fib.a_symbol(&foreign, &tau, &tau)).is_err());
    assert!(std::panic::catch_unwind(|| fib.b_symbol(&tau, &tau, &foreign)).is_err());
    let z4 = ZNIrrep::new(1, 4).unwrap();
    let z3 = ZN::new(3).unwrap();
    let u = z3.unit();
    assert!(std::panic::catch_unwind(|| z3.a_symbol(&z4, &u, &u)).is_err());
    assert!(std::panic::catch_unwind(|| z3.b_symbol(&u, &z4, &u)).is_err());
    let bad = (foreign, half);
    assert!(std::panic::catch_unwind(|| cat.a_symbol(&x, &x, &bad).map(|_| ())).is_err());
}

#[cfg(feature = "lie-generated")]
mod su3 {
    use super::*;
    use fusion_symbols::{
        category::{BraidingStyle, FusionStyle, SymbolBlock},
        lie::{LieError, LieIrrep},
    };

    /// Row-major storage owned by the test, standing in for a downstream
    /// block type.
    #[derive(Clone, Debug)]
    struct Dense<const R: usize> {
        dims: [usize; R],
        data: Vec<f64>,
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
            &self.data
        }
    }

    /// SU(3) in a test-only gauge: the multiplicity basis of the splitting
    /// vertex `27 → 15 ⊗ 15̄` is rotated, `(C'₁, C'₂) = (−C₂, C₁)`, i.e.
    /// `C'_μ = Σ_ρ C_ρ U_{ρμ}` with `U = [[0, 1], [−1, 0]]`.
    ///
    /// A unitary change of basis of one vertex space is a gauge
    /// transformation: N, dimensions, duals and Frobenius–Schur phases are
    /// unchanged, and every F transforms on each axis carrying that vertex as
    /// `F'[.., i, ..] = Σ_ρ U_{ρi} F[.., ρ, ..]` (TensorKitSectors
    /// `Fsymbol_from_fusiontensor` is linear in the left-tree and antilinear
    /// in the right-tree fusion tensors; `U` is real). Every F is rotated
    /// this way, so pentagon and unitarity hold as in SU(3). Braiding is not
    /// rotated, so it is withheld: this is a fusion category only. A then
    /// becomes `Uᵀ A = J` at `(15, 15̄, 27)` (also derived from the rotated
    /// fusion tensors in TensorKitSectors' `Asymbol_from_fusiontensor`).
    ///
    /// Its symbol blocks are a custom type, so it also checks that the A/B
    /// defaults serve a category whose `RSymbol` this crate cannot build.
    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Rotated(LieGroup);
    impl Rotated {
        fn new() -> Self {
            Self(su3())
        }
        fn rotated(&self, x: &LieIrrep, y: &LieIrrep, z: &LieIrrep) -> bool {
            let irr = |d: &[i64]| self.0.irrep(d, &[]).unwrap();
            (x, y, z) == (&irr(&[2, 1]), &irr(&[1, 2]), &irr(&[2, 2]))
        }
    }
    impl FusionCategory for Rotated {
        type Object = LieIrrep;
        type Scalar = f64;
        type FSymbol = Dense<4>;
        type RSymbol = Dense<2>;
        type Fusion<'a> = <LieGroup as FusionCategory>::Fusion<'a>;
        type Error = LieError;
        fn fusion_style(&self) -> FusionStyle {
            self.0.fusion_style()
        }
        fn braiding_style(&self) -> BraidingStyle {
            BraidingStyle::None
        }
        fn unit(&self) -> LieIrrep {
            self.0.unit()
        }
        fn contains(&self, a: &LieIrrep) -> bool {
            self.0.contains(a)
        }
        fn dual(&self, a: &LieIrrep) -> LieIrrep {
            self.0.dual(a)
        }
        fn dim(&self, a: &LieIrrep) -> f64 {
            self.0.dim(a)
        }
        fn frobenius_schur_phase(&self, a: &LieIrrep) -> f64 {
            self.0.frobenius_schur_phase(a)
        }
        fn frobenius_schur_indicator(&self, a: &LieIrrep) -> f64 {
            self.0.frobenius_schur_indicator(a)
        }
        fn twist(&self, a: &LieIrrep) -> f64 {
            self.0.dim(a);
            panic!("no braiding in the rotated gauge")
        }
        fn fusion<'a>(&'a self, a: &LieIrrep, b: &LieIrrep) -> Result<Self::Fusion<'a>, LieError> {
            self.0.fusion(a, b)
        }
        fn nsymbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> u32 {
            self.0.nsymbol(a, b, c)
        }
        fn f_symbol(
            &self,
            a: &LieIrrep,
            b: &LieIrrep,
            c: &LieIrrep,
            d: &LieIrrep,
            e: &LieIrrep,
            f: &LieIrrep,
        ) -> Result<Dense<4>, LieError> {
            let block = self.0.f_symbol(a, b, c, d, e, f)?;
            let dims = [(a, b, e), (e, c, d), (b, c, f), (a, f, d)]
                .map(|(x, y, z)| self.nsymbol(x, y, z) as usize);
            let mut data = if dims.contains(&0) {
                Vec::new()
            } else {
                block.data().to_vec()
            };
            for (axis, (x, y, z)) in [(a, b, e), (e, c, d), (b, c, f), (a, f, d)]
                .into_iter()
                .enumerate()
            {
                if data.is_empty() || !self.rotated(x, y, z) {
                    continue;
                }
                // U = [[0, 1], [−1, 0]]: new[0] = −old[1], new[1] = old[0].
                let inner: usize = dims[axis + 1..].iter().product();
                let old = data.clone();
                for (flat, value) in data.iter_mut().enumerate() {
                    let i = flat / inner % 2;
                    let other = flat + inner * (1 - i) - inner * i;
                    *value = if i == 0 { -old[other] } else { old[other] };
                }
            }
            Ok(Dense { dims, data })
        }
        fn r_symbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<Dense<2>, LieError> {
            self.nsymbol(a, b, c);
            panic!("no braiding in the rotated gauge")
        }
        fn authority_fingerprint(&self) -> &[u8] {
            b"test:su3-rotated-15-15bar-27"
        }
    }

    const J: [f64; 4] = [0.0, -1.0, 1.0, 0.0];

    #[test]
    fn default_derivation_serves_a_custom_block_type_in_a_rotated_gauge() {
        let cat = Rotated::new();
        let irr = |d: &[i64]| cat.0.irrep(d, &[]).unwrap();
        let (f15, f15b, s27) = (irr(&[2, 1]), irr(&[1, 2]), irr(&[2, 2]));
        let a = cat.a_symbol(&f15, &f15b, &s27).unwrap();
        assert_eq!(a.dim(), 2);
        assert_block(a.data(), &J, "A rotated");
        // B reads the rotated vertex on its row axis: B' = Uᵀ B = J · B_15.
        let b = cat.b_symbol(&f15, &f15b, &s27).unwrap();
        let jb = [-B_15[2], -B_15[3], B_15[0], B_15[1]];
        assert_block(b.data(), &jb, "B rotated");
        // The gauge change keeps the rotated F unitary.
        assert!(
            fusion_symbols::category::f_unitarity_residual(&cat, &f15b, &f15, &f15b, &f15b)
                .unwrap()
                < TOL
        );
        // Absent channel (15 ∉ 15 ⊗ 15̄): the empty block, without F.
        assert_eq!(cat.a_symbol(&f15, &f15b, &f15).unwrap().dim(), 0);
        assert!(cat.b_symbol(&f15, &f15b, &f15).unwrap().data().is_empty());
        let foreign = super::su2().irrep(&[1], &[]).unwrap();
        assert!(std::panic::catch_unwind(|| cat.a_symbol(&foreign, &f15b, &s27)).is_err());
    }

    fn su3() -> LieGroup {
        LieGroup::new(GroupId::su(3).unwrap()).unwrap()
    }

    fn assert_block(got: &[f64], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < TOL, "{what}[{i}]: {g} vs {w}");
        }
    }

    // SUNRepresentations.jl 0.4.0 `SU3Irrep` literals; GT pattern (λ₁, λ₂, 0)
    // is Dynkin [λ₁ − λ₂, λ₂]. The octet and 27 blocks print as the identity
    // up to 1e-15 round-off.
    const I2: [f64; 4] = [1.0, 0.0, 0.0, 1.0];
    /// `B^{15, 15̄}_{27}` (`15` = GT (3,1,0)): `[−1/(2√2), √(7/8); −√(7/8), −1/(2√2)]`.
    const B_15: [f64; 4] = [
        -0.353_553_390_593_273_73,
        0.935_414_346_693_485_7,
        -0.935_414_346_693_486_3,
        -0.353_553_390_593_273_9,
    ];

    #[test]
    fn multiplicity_blocks_match_sunrepresentations() {
        let cat = su3();
        let irr = |d: &[i64]| cat.irrep(d, &[]).unwrap();
        let (three, bar, oct) = (irr(&[1, 0]), irr(&[0, 1]), irr(&[1, 1]));
        let (f15, f15b, s27) = (irr(&[2, 1]), irr(&[1, 2]), irr(&[2, 2]));
        for (a, b, c, want_a, want_b) in [
            (&oct, &oct, &oct, I2, I2),
            (&s27, &oct, &s27, I2, I2),
            (&oct, &s27, &s27, I2, I2),
            (&f15, &f15b, &s27, I2, B_15),
        ] {
            let got_a = cat.a_symbol(a, b, c).unwrap();
            let got_b = cat.b_symbol(a, b, c).unwrap();
            assert_eq!((got_a.dim(), got_b.dim()), (2, 2));
            assert_block(got_a.data(), &want_a, &format!("A {a:?} {b:?} {c:?}"));
            assert_block(got_b.data(), &want_b, &format!("B {a:?} {b:?} {c:?}"));
        }
        for (a, b, c, want_a, want_b) in [
            (&three, &three, &bar, -1.0, -1.0),
            (&three, &bar, &oct, 1.0, -1.0),
            (&oct, &three, &three, 1.0, -1.0),
        ] {
            assert_block(cat.a_symbol(a, b, c).unwrap().data(), &[want_a], "A");
            assert_block(cat.b_symbol(a, b, c).unwrap().data(), &[want_b], "B");
        }
        let objects = [&[0, 0], &[1, 0], &[0, 1], &[1, 1], &[2, 0], &[2, 1]]
            .map(|d: &[i64; 2]| irr(d))
            .to_vec();
        assert_unitary(&cat, &objects);
    }

    #[test]
    fn product_blocks_are_last_factor_fastest_kronecker() {
        let cat = su3();
        let irr = |d: &[i64]| cat.irrep(d, &[]).unwrap();
        let (oct, f15, f15b, s27) = (irr(&[1, 1]), irr(&[2, 1]), irr(&[1, 2]), irr(&[2, 2]));
        let p = product(cat.clone(), cat.clone());
        // TensorKitSectors `SU3Irrep ⊠ SU3Irrep` B at (15,8) ⊗ (15̄,8) → (27,8):
        // kron(B_15, I), row-major with μ = 2 μ₁ + μ₂.
        let (r, s) = (B_15[0], B_15[1]);
        let t = B_15[2];
        #[rustfmt::skip]
        let left = [
            r, 0.0, s, 0.0,
            0.0, r, 0.0, s,
            t, 0.0, r, 0.0,
            0.0, t, 0.0, r,
        ];
        #[rustfmt::skip]
        let right = [
            r, s, 0.0, 0.0,
            t, r, 0.0, 0.0,
            0.0, 0.0, r, s,
            0.0, 0.0, t, r,
        ];
        let b = p
            .b_symbol(
                &(f15.clone(), oct.clone()),
                &(f15b.clone(), oct.clone()),
                &(s27.clone(), oct.clone()),
            )
            .unwrap();
        assert_eq!(b.dim(), 4);
        assert_block(b.data(), &left, "B (15,8)");
        // Mirrored factors: kron(I, B_15).
        let b = p
            .b_symbol(
                &(oct.clone(), f15.clone()),
                &(oct.clone(), f15b.clone()),
                &(oct.clone(), s27.clone()),
            )
            .unwrap();
        assert_block(b.data(), &right, "B (8,15)");
        let a = p
            .a_symbol(
                &(f15.clone(), oct.clone()),
                &(f15b.clone(), oct.clone()),
                &(s27.clone(), oct.clone()),
            )
            .unwrap();
        #[rustfmt::skip]
        let id4 = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        assert_block(a.data(), &id4, "A (15,8)");

        // A distinguishes factor order with the rotated gauge, whose
        // A^{15,15̄}_{27} = J, against the octet A^{88}_8 ≈ I₂ (no SU(3) A is
        // non-diagonal in the SUNRepresentations gauge).
        #[rustfmt::skip]
        let j_i = [
            0.0, 0.0, -1.0, 0.0,
            0.0, 0.0, 0.0, -1.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        #[rustfmt::skip]
        let i_j = [
            0.0, -1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0,
            0.0, 0.0, 1.0, 0.0,
        ];
        let rp = product(Rotated::new(), cat.clone());
        let a = rp
            .a_symbol(
                &(f15.clone(), oct.clone()),
                &(f15b.clone(), oct.clone()),
                &(s27.clone(), oct.clone()),
            )
            .unwrap();
        assert_block(a.data(), &j_i, "A (15,8) rotated");
        let pr = product(cat.clone(), Rotated::new());
        let a = pr
            .a_symbol(
                &(oct.clone(), f15.clone()),
                &(oct.clone(), f15b.clone()),
                &(oct.clone(), s27.clone()),
            )
            .unwrap();
        assert_block(a.data(), &i_j, "A (8,15) rotated");

        // SU(3) × U(1): the U(1) factor contributes a 1 × 1 one.
        let q = product(cat.clone(), U1);
        let b = q
            .b_symbol(
                &(f15.clone(), U1Irrep::new(1).unwrap()),
                &(f15b.clone(), U1Irrep::new(-1).unwrap()),
                &(s27.clone(), U1Irrep::new(0).unwrap()),
            )
            .unwrap();
        assert_block(b.data(), &B_15, "B (15,1)");
    }
}
