//! Prints the current gauge golden tables in the literal form that
//! `tests/gauge_golden.rs` pins, for reviewing a deliberate gauge change.
//!
//! ```text
//! cargo run --release --all-features --example print_golden
//! ```

#[allow(dead_code, unused_imports)]
#[path = "../tests/gauge_golden.rs"]
mod gauge_golden;

fn main() {
    for case in gauge_golden::CASES {
        println!("// {}", case.name);
        for (k, v) in gauge_golden::rows(case) {
            println!("    (\"{k}\", 0x{v:016x}),");
        }
    }
}
