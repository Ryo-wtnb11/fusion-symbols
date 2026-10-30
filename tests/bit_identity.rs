//! Cold-process bit identity (`docs/gauge_cartan.md` §8 oracle 9): a child
//! test writes the canonical little-endian bytes of small CGC, F, R, A and B
//! blocks with their labeled shapes; the parent runs it twice in fresh
//! processes and compares the bytes. Each child also regenerates its bytes
//! after a reset, a zero trim of every tier and in an equal group value, so
//! retention never changes a bit. Readable value pins live in
//! `tests/gauge_golden.rs`; this test only proves reproducibility.

use fusion_symbols::cache::{reset, trim_to, CoefficientCacheTier};
use fusion_symbols::category::{CgcBlock, FusionCategory, GroupRepresentations, SymbolBlock};
use fusion_symbols::lie::{GroupId, LieGroup};

const OUT: &str = "FUSION_SYMBOLS_BIT_IDENTITY_OUT";

fn put(out: &mut Vec<u8>, label: &str, dims: &[usize], values: impl IntoIterator<Item = f64>) {
    out.extend_from_slice(label.as_bytes());
    for d in dims {
        out.extend_from_slice(&(*d as u64).to_le_bytes());
    }
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

fn block<const N: usize>(out: &mut Vec<u8>, label: &str, x: impl SymbolBlock<N, Scalar = f64>) {
    let dims = x.dims();
    put(out, label, &dims, x.as_slice().iter().copied());
}

fn cases() -> Vec<(GroupId, Vec<i64>, bool)> {
    #[cfg_attr(not(feature = "lie-generated"), allow(unused_mut))]
    let mut cases = vec![(GroupId::su(2).unwrap(), vec![12], false)];
    #[cfg(feature = "lie-generated")]
    cases.extend([
        (GroupId::su(3).unwrap(), vec![1, 1], true),
        (GroupId::su(4).unwrap(), vec![1, 0, 1], true),
        (GroupId::so(5).unwrap(), vec![1, 0], false),
        (GroupId::sp(4).unwrap(), vec![1, 0], false),
        (GroupId::spin(7).unwrap(), vec![0, 0, 1], false),
        (GroupId::so(6).unwrap(), vec![0, 1, 1], false),
        (GroupId::g2(), vec![1, 0], true),
        (GroupId::f4(), vec![0, 0, 0, 1], false),
        (GroupId::e6(), vec![1, 0, 0, 0, 0, 0], false),
    ]);
    cases
}

fn emit(fresh_groups: bool) -> Vec<u8> {
    let mut out = Vec::new();
    for (id, label, adjoint) in cases() {
        let g = LieGroup::new(id.clone()).unwrap();
        let g = if fresh_groups { g } else { g.clone() };
        let a = g.irrep(&label).unwrap();
        let b = g.dual(&a);
        let one = g.unit();
        let name = format!("{id:?}");
        for x in [&one, &a, &b] {
            put(
                &mut out,
                &name,
                &[],
                [
                    g.dim(x),
                    g.frobenius_schur_phase(x).unwrap(),
                    g.frobenius_schur_indicator(x),
                ],
            );
            block(&mut out, "A", g.a_symbol(x, &g.dual(x), &one).unwrap());
            block(&mut out, "B", g.b_symbol(x, &g.dual(x), &one).unwrap());
        }
        for c in g.fusion(&a, &b).unwrap() {
            let cg = g.cgc(&a, &b, &c).unwrap();
            let [n1, n2, n3, n4] = cg.dims();
            let values = (0..n1).flat_map(|i| {
                let cg = &cg;
                (0..n2).flat_map(move |j| {
                    (0..n3).flat_map(move |k| (0..n4).map(move |mu| cg.get([i, j, k, mu])))
                })
            });
            put(&mut out, "C", &cg.dims(), values);
            block(&mut out, "R", g.r_symbol(&a, &b, &c).unwrap());
        }
        block(
            &mut out,
            "F",
            g.f_symbol(&a, &b, &a, &a, &one, &one).unwrap(),
        );
        if adjoint {
            block(&mut out, "F", g.f_symbol(&a, &a, &a, &a, &a, &a).unwrap());
        }
    }
    out
}

/// The child process: bytes cold, after a reset, after a zero trim of every
/// tier and from fresh group values must agree before they are written.
#[test]
#[ignore = "child of separate_processes_emit_identical_bytes"]
fn emit_child() {
    let Ok(path) = std::env::var(OUT) else {
        return;
    };
    let cold = emit(false);
    reset();
    assert_eq!(emit(true), cold, "regeneration after reset");
    for &tier in CoefficientCacheTier::ALL {
        trim_to(tier, 0);
    }
    assert_eq!(emit(false), cold, "regeneration after trim");
    assert_eq!(emit(true), cold, "warm equal groups");
    std::fs::write(path, cold).unwrap();
}

#[test]
fn separate_processes_emit_identical_bytes() {
    let exe = std::env::current_exe().unwrap();
    let run = |i: usize| {
        let path = std::env::temp_dir().join(format!("fs-bit-identity-{}-{i}", std::process::id()));
        let status = std::process::Command::new(&exe)
            .args([
                "--exact",
                "bit_identity::emit_child",
                "--include-ignored",
                "--test-threads=1",
                "--quiet",
            ])
            .env(OUT, &path)
            .stdout(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "child {i} failed");
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        bytes
    };
    let first = run(0);
    assert!(!first.is_empty());
    assert_eq!(run(1), first, "two cold processes");
}

/// `κ = +1` for the E6 complex irreps 27, 351 and 1728 (§5.3: every E6
/// coefficient `c_i` is even, so `κ = σ_a σ_ā`, and the minuscule 27 has
/// `σ = +1`); the other two are consistent with the §5.3 conjecture.
#[cfg(feature = "lie-generated")]
#[test]
fn complex_phase_bits() {
    let g = LieGroup::new(GroupId::e6()).unwrap();
    for label in [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]] {
        let a = g.irrep(&label).unwrap();
        for x in [&a, &g.dual(&a), &a] {
            assert_eq!(
                g.frobenius_schur_phase(x).unwrap().to_bits(),
                0x3ff0000000000000
            );
        }
    }
}
