//! SU(2) F- and R-symbols agree with TensorKitSectors `SU2Irrep` values.
//!
//! Contract: a few values checked against TensorKitSectors itself. F is
//! composed exactly and rounded once, so the tolerance covers only the
//! reference's own last-ulp rounding. The closed-form convention over every
//! tuple of doubled spins `≤ 12` is a unit test of the SU(2) kernel, next to
//! the 6j symbol it composes.

use crate::common::a1;
use a1::{exact_f, exact_r};

#[test]
fn values_checked_against_tensorkitsectors() {
    // Checked against TensorKitSectors 0.3.9 (`SU2Irrep`, Julia 1.11.6);
    // exact values `−1/2`, `1/2`, `√3/2`, `2√2/3`, `1/2`, `√(5/6)`, `√5/3`
    // (the reference prints `0.4999999999999999` for the two halves).
    for (d, want) in [
        ([1, 1, 1, 1, 0, 0], -0.5),
        ([1, 1, 1, 1, 2, 2], 0.5),
        ([1, 1, 1, 1, 0, 2], 0.866_025_403_784_438_6),
        ([1, 2, 1, 2, 1, 3], 0.942_809_041_582_063_4),
        ([2, 2, 2, 2, 2, 2], 0.5),
        ([3, 1, 2, 2, 2, 3], 0.912_870_929_175_276_9),
        ([2, 2, 2, 2, 0, 4], 0.745_355_992_499_929_9),
    ] {
        let got = exact_f(d[0], d[1], d[2], d[3], d[4], d[5]);
        assert!((got - want).abs() <= 2.0 * f64::EPSILON, "F {d:?}: {got}");
    }
    for (d, want) in [
        ([1, 1, 0], -1.0),
        ([1, 1, 2], 1.0),
        ([2, 2, 2], -1.0),
        ([2, 1, 1], -1.0),
        ([3, 3, 4], -1.0),
        ([2, 2, 0], 1.0),
    ] {
        assert_eq!(exact_r(d[0], d[1], d[2]), want, "R {d:?}");
    }
}
