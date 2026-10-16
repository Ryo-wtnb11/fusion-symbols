//! Prints every value symtn's four abelian providers expose (Z2, fZ2, Z_N for
//! N = 1..=4 as `zn<N>`, U1 on a charge window plus the i32 extremes), as IEEE-754 bit
//! patterns, for the abelian value-identity fixture. Symbols are recorded on
//! every label tuple, admissible or not, exactly as symtn returns them; a
//! checked fusion that fails is recorded as `err`.
use symtn_sectors::{
    CheckedFusionAlgebra, FermionParityFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, SectorCodec, SectorId, U1FusionRule, U1Irrep, Z2FusionRule, ZNFusionRule,
};

fn bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

/// `label` turns a sector id into the semantic label printed in the fixture
/// (parity, charge mod N, or U(1) charge), so the fixture does not depend on
/// symtn's id encoding.
fn dump<R>(tag: &str, rule: &R, sectors: &[SectorId], label: &dyn Fn(SectorId) -> i64, symbols: &[SectorId])
where
    R: CheckedFusionAlgebra + MultiplicityFreeFusionSymbols<Scalar = f64> + MultiplicityFreeRigidSymbols,
{
    println!("{tag} vacuum {}", label(rule.vacuum()));
    for &a in sectors {
        let i = label(a);
        println!("{tag} dual {i} {}", label(rule.dual(a)));
        println!("{tag} dim {i} {}", bits(rule.dim_scalar(a)));
        println!("{tag} inv_dim {i} {}", bits(rule.inv_dim_scalar(a)));
        println!("{tag} sqrt_dim {i} {}", bits(rule.sqrt_dim_scalar(a)));
        println!("{tag} inv_sqrt_dim {i} {}", bits(rule.inv_sqrt_dim_scalar(a)));
        println!("{tag} fs {i} {}", bits(rule.frobenius_schur_phase_scalar(a)));
        println!("{tag} twist {i} {}", bits(rule.twist_scalar(a)));
    }
    for &a in sectors {
        for &b in sectors {
            match rule.try_fusion_channels(a, b) {
                Ok(ch) => {
                    let ch: Vec<String> = ch.iter().map(|&c| label(c).to_string()).collect();
                    println!("{tag} fusion {} {} {}", label(a), label(b), ch.join(","));
                }
                Err(_) => println!("{tag} fusion {} {} err", label(a), label(b)),
            }
            for &c in sectors {
                match rule.try_nsymbol(a, b, c) {
                    Ok(n) => println!("{tag} N {} {} {} {n}", label(a), label(b), label(c)),
                    Err(_) => println!("{tag} N {} {} {} err", label(a), label(b), label(c)),
                }
            }
        }
    }
    for &a in symbols { for &b in symbols { for &c in symbols {
        println!("{tag} R {} {} {} {}", label(a), label(b), label(c), bits(rule.r_symbol_scalar(a, b, c)));
    }}}
    for &a in symbols { for &b in symbols { for &c in symbols { for &d in symbols { for &e in symbols { for &f in symbols {
        println!("{tag} F {} {} {} {} {} {} {}", label(a), label(b), label(c), label(d), label(e), label(f),
            bits(rule.f_symbol_scalar(a, b, c, d, e, f)));
    }}}}}}
}

fn main() {
    println!("# symtn 49d09dc4fc0825b29d547e63a70014221c0c5406 symtn-sectors/src/abelian.rs; semantic labels; f64 bits");
    let z2: Vec<SectorId> = (0..2).map(SectorId::new).collect();
    let id = |s: SectorId| s.id() as i64;
    dump("z2", &Z2FusionRule, &z2, &id, &z2);
    dump("fz2", &FermionParityFusionRule, &z2, &id, &z2);
    for n in 1..=4u32 {
        let rule = ZNFusionRule::new(n).expect("nonzero modulus");
        let s: Vec<SectorId> = (0..i64::from(n)).map(|q| rule.irrep(q).into()).collect();
        let label = |s: SectorId| i64::from(rule.decode_sector(s).expect("z_n label").charge());
        dump(&format!("zn{n}"), &rule, &s, &label, &s);
    }
    let u1 = |q: i32| -> SectorId { U1Irrep::new(q).into() };
    let label = |s: SectorId| i64::from(U1Irrep::from_sector_id(s).expect("u1 label").charge());
    let window: Vec<SectorId> = [i32::MIN + 1, -2, -1, 0, 1, 2, i32::MAX - 1, i32::MAX].map(u1).to_vec();
    let symbols: Vec<SectorId> = [-1, 0, 1].map(u1).to_vec();
    dump("u1", &U1FusionRule, &window, &label, &symbols);
}
