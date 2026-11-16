//! A first evaluation in a fresh process grows the shared factorial tables to
//! its own largest factorial argument. This binary holds a single test so each
//! call below is the first to need its rows; the second needs exactly the one
//! row past those the first grew.

use fusion_symbols::lie::su2::{wigner_3j, wigner_6j};
use malachite_q::Rational;

#[test]
fn first_evaluation_grows_tables_to_its_largest_factorial() {
    // {a b c; 0 c b} = (-1)^(a+b+c) / sqrt((2b+1)(2c+1)) (Edmonds 6.3.2);
    // spin 100 throughout, so the sign is +. Largest factorial: 301!.
    let six = wigner_6j(200, 200, 200, 0, 200, 200);
    assert_eq!(
        six.signed_square(),
        Rational::from_unsigneds(1u32, 201 * 201)
    );

    // (j j 0; m -m 0) = (-1)^(j-m) / sqrt(2j+1) (Edmonds 3.7.9); j = 301/2,
    // m = 3/2, so the sign is -. Largest factorial: 302!.
    let three = wigner_3j(301, 301, 0, 3, -3, 0);
    assert_eq!(three.signed_square(), -Rational::from_unsigneds(1u32, 302));
}
