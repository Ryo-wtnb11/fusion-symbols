//! A warm F/CGC cache hit hands out the cached coefficients without copying
//! them: `FBlock` and `lie::LieCgc` share their buffers, so no
//! allocation on a hit is as large as the coefficient buffer. Complete
//! zero-allocation tier coverage is in `cache_hit_alloc.rs`.

#![cfg(feature = "lie-generated")]

use crate::common::bcd as bcd_test;
use crate::common::lie as lie_test;
use bcd_test::Series;

use crate::common::counting_alloc;

use counting_alloc::measure;
use fusion_symbols::category::{FusionCategory, GroupRepresentations, SymbolBlock};

/// Measurements count only this thread; the guard isolates shared caches.
#[test]
fn warm_hits_do_not_copy_coefficients() {
    let _serial = crate::common::counting_alloc::serial();
    // SU(3) 8^6: OM = 2 on every vertex, a 2x2x2x2 block (128 bytes).
    let e8 = lie_test::irrep(&[1, 1]).unwrap();
    // Warm lookup excludes construction of a new group-owned root context.
    let group = lie_test::group(&e8);
    let f = || group.f_symbol(&e8, &e8, &e8, &e8, &e8, &e8).unwrap();
    let cold = f();
    let (calls, bytes, largest, warm) = measure(f);
    let buffer = std::mem::size_of_val(warm.as_slice());
    println!("sun F warm hit: {calls} allocs, {bytes} B, largest {largest} B, data {buffer} B");
    assert_eq!(warm, cold);
    assert!(
        largest < buffer,
        "sun F hit copied its data ({largest} >= {buffer})"
    );

    let c = || group.cgc(&e8, &e8, &e8).unwrap();
    let cold = c();
    let (calls, bytes, largest, warm) = measure(c);
    let buffer = std::mem::size_of_val(lie_test::entries(&warm).as_slice());
    println!("sun CGC warm hit: {calls} allocs, {bytes} B, largest {largest} B, data {buffer} B");
    assert_eq!(warm, cold);
    assert!(
        largest < buffer,
        "sun CGC hit copied its entries ({largest} >= {buffer})"
    );

    // Spin(6) adjoint: OM = 2, a 2x2x2x2 block.
    let cat = bcd_test::group(Series::D, 3).unwrap();
    let adj = bcd_test::irrep(Series::D, &[0, 1, 1]).unwrap();
    let f = || bcd_test::f_symbol(&cat, &adj, &adj, &adj, &adj, &adj, &adj).unwrap();
    let cold = f();
    let (calls, bytes, largest, warm) = measure(f);
    let buffer = std::mem::size_of_val(warm.as_slice());
    println!("bcd F warm hit: {calls} allocs, {bytes} B, largest {largest} B, data {buffer} B");
    assert_eq!(warm, cold);
    assert!(
        largest < buffer,
        "bcd F hit copied its data ({largest} >= {buffer})"
    );
}
