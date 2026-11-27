//! A first evaluation in a fresh process grows the shared factorial tables to
//! its own largest factorial argument. This binary holds a single test so each
//! call below is the first to need its rows; the second needs exactly the one
//! row past those the first grew.

use fusion_symbols::category::{FusionCategory, SymbolBlock};
use fusion_symbols::lie::{GroupId, LieGroup};

#[test]
fn first_evaluation_grows_tables_to_its_largest_factorial() -> Result<(), Box<dyn std::error::Error>>
{
    // F^{abc}_d[e, f] = (-1)^(a+b+c+d) sqrt((2e+1)(2f+1)) {a b e; c d f}
    // (TensorKitSectors) and {a b e; b a 0} = (-1)^(a+b+e) / sqrt((2a+1)(2b+1))
    // (Edmonds 6.3.2), so F^{aaa}_a[e, 0] = (-1)^(2a+e) sqrt(2e+1) / (2a+1).
    let su2 = LieGroup::new(GroupId::su(2)?)?;
    let f = |da: i64, de: i64| -> Result<f64, Box<dyn std::error::Error>> {
        let (a, e) = (su2.irrep(&[da])?, su2.irrep(&[de])?);
        Ok(su2
            .f_symbol(&a, &a, &a, &a, &e, &su2.unit())?
            .get([0, 0, 0, 0]))
    };
    let close = |got: f64, want: f64| (got - want).abs() <= 2.0 * f64::EPSILON * want.abs();

    // a = e = 100: sign +, value 1/sqrt(201). Largest factorial: 301!.
    let got = f(200, 200)?;
    assert!(close(got, 1.0 / 201f64.sqrt()), "{got}");

    // a = 201/2, e = 100: sign -, value -sqrt(201)/202. Largest factorial: 302!.
    let got = f(201, 200)?;
    assert!(close(got, -(201f64.sqrt()) / 202.0), "{got}");
    Ok(())
}
