//! Prints every value symtn's CU(1) provider exposes, as IEEE-754 bit
//! patterns, for the cu1 value-identity fixture. Labels are `twice_charge:s`
//! (TensorKitSectors' `twice(j)` and `s`), so the fixture does not depend on
//! symtn's sector-id encoding. Unary data, fusion and N run on a window that
//! includes the twice-charge extremes; R and F on `0+`, `0-`, `j = 1/2..5/2`,
//! over every tuple, printing only those whose bits are not `+0.0` (an absent
//! tuple is `+0.0`). A checked fusion or N that fails is recorded as `err`.
//!
//! Directory layout the manifest `extract_symtn_cu1.Cargo.toml` expects (its dependency
//! is the relative path `../symtn/symtn-sectors`):
//!
//! ```text
//! <dir>/symtn/                    git archive 2a6c152315bb271dc5e56b5f1f034ca818ef999d of symtn-rs
//! <dir>/extract/Cargo.toml        copy of extract_symtn_cu1.Cargo.toml
//! <dir>/extract/extract_symtn_cu1.rs
//! <dir>/extract/Cargo.lock        copy of <dir>/symtn/Cargo.lock
//! ```
//!
//! Then, in `<dir>/extract`: `cargo run --release > symtn-2a6c1523-cu1-values.txt`.
use symtn_sectors::{
    CU1FusionRule, CU1Irrep, CheckedFusionAlgebra, FusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, SectorId, CU1_MAX_TWICE_CHARGE,
};

fn bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

fn label(s: SectorId) -> String {
    let a = CU1Irrep::try_from_sector_id(s).expect("cu1 label");
    format!("{}:{}", a.twice_charge(), a.conjugation_sector())
}

fn main() {
    println!("# symtn 2a6c152315bb271dc5e56b5f1f034ca818ef999d symtn-sectors/src/cu1.rs; labels twice_charge:s; f64 bits");
    let rule = CU1FusionRule;
    let charged = |q: u32| -> SectorId { CU1Irrep::from_twice_charge(q).into() };
    let small: Vec<SectorId> = [CU1Irrep::VACUUM.into(), CU1Irrep::PSEUDOSCALAR.into()]
        .into_iter()
        .chain((1..=5).map(charged))
        .collect();
    let half = CU1_MAX_TWICE_CHARGE / 2;
    let window: Vec<SectorId> = small
        .iter()
        .copied()
        .chain([half, half + 1, CU1_MAX_TWICE_CHARGE - 1, CU1_MAX_TWICE_CHARGE].map(charged))
        .collect();
    println!("max_twice_charge {CU1_MAX_TWICE_CHARGE}");
    println!("vacuum {}", label(rule.vacuum()));
    for &a in &window {
        let i = label(a);
        println!("dual {i} {}", label(rule.dual(a)));
        println!("dim {i} {}", bits(rule.dim_scalar(a)));
        println!("inv_dim {i} {}", bits(rule.inv_dim_scalar(a)));
        println!("sqrt_dim {i} {}", bits(rule.sqrt_dim_scalar(a)));
        println!("inv_sqrt_dim {i} {}", bits(rule.inv_sqrt_dim_scalar(a)));
        println!("fs {i} {}", bits(rule.frobenius_schur_phase_scalar(a)));
        println!("twist {i} {}", bits(rule.twist_scalar(a)));
    }
    for &a in &window {
        for &b in &window {
            match rule.try_fusion_channels(a, b) {
                Ok(ch) => {
                    let ch: Vec<String> = ch.iter().map(|&c| label(c)).collect();
                    println!("fusion {} {} {}", label(a), label(b), ch.join(","));
                }
                Err(_) => println!("fusion {} {} err", label(a), label(b)),
            }
            for &c in &window {
                match rule.try_nsymbol(a, b, c) {
                    Ok(n) => println!("N {} {} {} {n}", label(a), label(b), label(c)),
                    Err(_) => println!("N {} {} {} err", label(a), label(b), label(c)),
                }
            }
        }
    }
    for &a in &small { for &b in &small { for &c in &small {
        let r = rule.r_symbol_scalar(a, b, c);
        if r.to_bits() != 0 { println!("R {} {} {} {}", label(a), label(b), label(c), bits(r)); }
    }}}
    for &a in &small { for &b in &small { for &c in &small { for &d in &small { for &e in &small { for &f in &small {
        let x = rule.f_symbol_scalar(a, b, c, d, e, f);
        if x.to_bits() != 0 {
            println!("F {} {} {} {} {} {} {}", label(a), label(b), label(c), label(d), label(e), label(f), bits(x));
        }
    }}}}}}
}
