//! Prints every value the two TeNeT Fibonacci providers expose, as IEEE-754
//! bit patterns, for the fusion-categories value-identity fixture.
use num_complex::Complex64;
use tenet_category_data::CategoryDataFibonacci;
use tenet_sectors::{
    FibonacciFusionRule, FusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    SectorId,
};

fn bits(z: Complex64) -> String {
    format!("{:016x} {:016x}", z.re.to_bits(), z.im.to_bits())
}

fn dump<R>(tag: &str, rule: &R)
where
    R: MultiplicityFreeFusionSymbols<Scalar = Complex64> + MultiplicityFreeRigidSymbols,
{
    let s = [SectorId::new(0), SectorId::new(1)];
    for a in s {
        let i = a.id();
        println!("{tag} dual {i} {}", rule.dual(a).id());
        println!("{tag} dim {i} {}", bits(rule.dim_scalar(a)));
        println!("{tag} inv_dim {i} {}", bits(rule.inv_dim_scalar(a)));
        println!("{tag} sqrt_dim {i} {}", bits(rule.sqrt_dim_scalar(a)));
        println!("{tag} inv_sqrt_dim {i} {}", bits(rule.inv_sqrt_dim_scalar(a)));
        println!("{tag} fs {i} {}", bits(rule.frobenius_schur_phase_scalar(a)));
        println!("{tag} twist {i} {}", bits(rule.twist_scalar(a)));
    }
    for a in s {
        for b in s {
            let ch: Vec<String> = rule.fusion_channels(a, b).iter().map(|c| c.id().to_string()).collect();
            println!("{tag} fusion {} {} {}", a.id(), b.id(), ch.join(","));
            for c in s {
                println!("{tag} N {} {} {} {}", a.id(), b.id(), c.id(), rule.nsymbol(a, b, c));
                println!("{tag} R {} {} {} {}", a.id(), b.id(), c.id(), bits(rule.r_symbol_scalar(a, b, c)));
            }
        }
    }
    for a in s { for b in s { for c in s { for d in s { for e in s { for f in s {
        println!("{tag} F {} {} {} {} {} {} {}", a.id(), b.id(), c.id(), d.id(), e.id(), f.id(),
            bits(rule.f_symbol_scalar(a, b, c, d, e, f)));
    }}}}}}
}

fn main() {
    println!("# TeNeT 957ccc05737f0841b6a98047880aba201bee63a1; zero-based SectorId; values as f64 bits (re im)");
    dump("closed", &FibonacciFusionRule);
    dump("table", &CategoryDataFibonacci::try_new().expect("tables"));
}
