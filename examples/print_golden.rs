//! Prints the current gauge golden tables in the literal form that
//! `tests/gauge_golden.rs` pins, for reviewing a deliberate gauge change.
//!
//! ```text
//! cargo run --release --all-features --example print_golden
//! cargo run --release --all-features --example print_golden e8_n2
//! ```
//!
//! With `e8_n2` it prints only the `R` rows of the heavy `E8_N2` golden
//! (`3875 ⊗ 30380 → 30380`), which need its seeds and three modules but no
//! descent.

#[allow(dead_code, unused_imports)]
#[path = "../tests/gauge_golden.rs"]
mod gauge_golden;

fn main() {
    if std::env::args().nth(1).as_deref() == Some("e8_n2") {
        e8_n2();
        return;
    }
    for case in gauge_golden::CASES {
        println!("// {}", case.name);
        for (k, v) in gauge_golden::rows(case) {
            println!("    (\"{k}\", 0x{v:016x}),");
        }
    }
}

#[cfg(feature = "lie-generated")]
fn e8_n2() {
    use fusion_symbols::category::{FusionCategory, SymbolBlock};
    use fusion_symbols::lie::{GroupId, LieGroup};
    let g = LieGroup::new(GroupId::e8()).unwrap();
    let [a, b] = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]].map(|l| g.irrep(&l).unwrap());
    println!("// E8 3875x30380->30380 N=2 (R only)");
    for (i, x) in g
        .r_symbol(&a, &b, &b)
        .unwrap()
        .as_slice()
        .iter()
        .enumerate()
    {
        println!("    (\"R {i}\", 0x{:016x}),", x.to_bits());
    }
}

#[cfg(not(feature = "lie-generated"))]
fn e8_n2() {
    eprintln!("the e8_n2 mode needs the lie-generated feature");
    std::process::exit(2);
}
