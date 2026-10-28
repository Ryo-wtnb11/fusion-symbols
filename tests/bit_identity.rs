//! Drift detector, not a mathematical oracle: hashes pinned on release 0.9.2,
//! 854d838e58104a14464fa46d219b83bc4db16111, using the same offline dependencies.
//! Includes shapes, all CGC entries (including zero), F/R/A/B, dimensions,
//! phases, exact multiplicities and authority bytes. The hashes depend on the
//! toolchain (`DefaultHasher`) and on faer's SIMD dispatch, so the fixture is
//! ignored in CI; regenerate it deliberately, and only when values are meant
//! to change.

use fusion_symbols::category::{CgcBlock, FusionCategory, GroupRepresentations, SymbolBlock};
use fusion_symbols::lie::{GroupId, LieGroup};
use std::hash::{DefaultHasher, Hash, Hasher};

fn block<const N: usize>(h: &mut DefaultHasher, x: impl SymbolBlock<N, Scalar = f64>) {
    x.dims().hash(h);
    for v in x.as_slice() {
        v.to_bits().hash(h);
    }
}

#[test]
#[ignore = "local drift check: macOS arm64, Rust 1.99.1, t4a-faer 0.24.9; run with --include-ignored"]
fn pinned_bits() {
    let mut cases = vec![(
        "su2",
        GroupId::su(2).unwrap(),
        vec![12],
        false,
        0xb00ca6af51faeef2u64,
    )];
    #[cfg(feature = "lie-generated")]
    cases.extend([
        (
            "su3",
            GroupId::su(3).unwrap(),
            vec![1, 1],
            true,
            0x44a8ef1883e0f1ecu64,
        ),
        (
            "su4",
            GroupId::su(4).unwrap(),
            vec![1, 0, 1],
            true,
            0x1eb1995d3e0f619bu64,
        ),
        (
            "so5",
            GroupId::so(5).unwrap(),
            vec![1, 0],
            false,
            0x83528d0264626a75u64,
        ),
        (
            "sp4",
            GroupId::sp(4).unwrap(),
            vec![1, 0],
            false,
            0x23d2192c99d67e3du64,
        ),
        (
            "spin7",
            GroupId::spin(7).unwrap(),
            vec![0, 0, 1],
            false,
            0xa0d914218c92512fu64,
        ),
        ("g2", GroupId::g2(), vec![1, 0], true, 0xd65fa0a84e066a1au64),
        ("g2-27", GroupId::g2(), vec![2, 0], true, 0x9468b0e8df9d5a18),
        (
            "f4",
            GroupId::f4(),
            vec![0, 0, 0, 1],
            true,
            0x1a0bd23d46e78666u64,
        ),
        (
            "e7",
            GroupId::e7(),
            vec![0, 0, 0, 0, 0, 0, 1],
            false,
            0xf314abc9518fdbe6u64,
        ),
        (
            "e8",
            GroupId::e8(),
            vec![0, 0, 0, 0, 0, 0, 0, 1],
            false,
            0x1759397a82f701afu64,
        ),
        (
            "e6",
            GroupId::e6(),
            vec![1, 0, 0, 0, 0, 0],
            false,
            0xc2f6d30f24252ba4u64,
        ),
    ]);
    for (name, id, label, adjoint, expected) in cases.drain(..) {
        let g = LieGroup::new(id).unwrap();
        let a = g.irrep(&label).unwrap();
        let b = g.dual(&a);
        let one = g.unit();
        let mut h = DefaultHasher::new();
        g.authority_fingerprint().hash(&mut h);
        for x in [&one, &a, &b] {
            g.dim(x).to_bits().hash(&mut h);
            g.frobenius_schur_phase(x).to_bits().hash(&mut h);
            g.frobenius_schur_indicator(x).to_bits().hash(&mut h);
            block(&mut h, g.a_symbol(x, &g.dual(x), &one).unwrap());
            block(&mut h, g.b_symbol(x, &g.dual(x), &one).unwrap());
        }
        for c in [&one, &a] {
            g.nsymbol(&a, &b, c).hash(&mut h);
            let cg = g.cgc(&a, &b, c).unwrap();
            let [n1, n2, n3, n4] = cg.dims();
            cg.dims().hash(&mut h);
            for i in 0..n1 {
                for j in 0..n2 {
                    for k in 0..n3 {
                        for mu in 0..n4 {
                            cg.at([i, j, k, mu]).to_bits().hash(&mut h);
                        }
                    }
                }
            }
            block(&mut h, g.r_symbol(&a, &b, c).unwrap());
        }
        block(&mut h, g.f_symbol(&a, &b, &a, &a, &one, &one).unwrap());
        if adjoint {
            block(&mut h, g.f_symbol(&a, &a, &a, &a, &a, &a).unwrap());
        }
        println!("{name} {:016x}", h.finish());
        assert_eq!(h.finish(), expected, "{name}");
    }
}

#[cfg(feature = "lie-generated")]
#[test]
fn complex_phase_bits() {
    let g = LieGroup::new(GroupId::e6()).unwrap();
    // Literals checked on 854d838e, including the cold 1728 phase benchmark.
    for label in [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]] {
        let a = g.irrep(&label).unwrap();
        for x in [&a, &g.dual(&a), &a] {
            assert_eq!(g.frobenius_schur_phase(x).to_bits(), 0x3ff0000000000000);
        }
    }
}
