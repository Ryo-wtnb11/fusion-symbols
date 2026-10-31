//! Supplemental cold/warm wall-time and allocation report, three samples by default.
//! Run with `cargo bench --offline --bench exceptional`; FS_PERF_CASE selects
//! one row and FS_PERF_SAMPLES changes the repetition count. Module construction
//! is inside each cold row; the E8 rows are the heaviest (select them with
//! FS_PERF_CASE).
#[path = "../tests/common/mod.rs"]
mod common;

use fusion_symbols::category::{CgcBlock, FusionCategory, GroupRepresentations};
use fusion_symbols::lie::{GroupId, LieGroup, RootSystem};
use std::hint::black_box;
use std::time::Instant;

#[global_allocator]
static ALLOCATOR: common::counting_alloc::Counting = common::counting_alloc::Counting;

fn measure(name: &str, mut cold: impl FnMut(), mut warm: impl FnMut()) {
    let t = Instant::now();
    let (allocs, bytes, _, ()) = common::counting_alloc::measure(&mut cold);
    let cold_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let (warm_allocs, warm_bytes, _, ()) = common::counting_alloc::measure(|| {
        for _ in 0..128 {
            warm();
        }
    });
    println!(
        "{name},{cold_s:.9},{:.9},{allocs},{warm_allocs},{bytes},{warm_bytes}",
        t.elapsed().as_secs_f64() / 128.0,
        warm_allocs = warm_allocs / 128,
        warm_bytes = warm_bytes / 128,
    );
}

fn main() {
    println!("case,cold_s,warm_s,cold_allocs,warm_allocs,cold_bytes,warm_bytes");
    let samples: usize = std::env::var("FS_PERF_SAMPLES")
        .unwrap_or_else(|_| "3".into())
        .parse()
        .unwrap();
    for (name, id, a, b, c) in [
        (
            "SU3-R",
            GroupId::su(3).unwrap(),
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
        ),
        (
            "SU4-R",
            GroupId::su(4).unwrap(),
            vec![1, 0, 1],
            vec![1, 0, 1],
            vec![1, 0, 1],
        ),
        (
            "SO5-R",
            GroupId::so(5).unwrap(),
            vec![1, 0],
            vec![1, 0],
            vec![0, 0],
        ),
        (
            "Sp4-R",
            GroupId::sp(4).unwrap(),
            vec![1, 0],
            vec![1, 0],
            vec![0, 0],
        ),
        (
            "G2-27-cgc",
            GroupId::g2(),
            vec![1, 0],
            vec![1, 0],
            vec![2, 0],
        ),
        (
            "G2-64-cgc",
            GroupId::g2(),
            vec![1, 0],
            vec![0, 1],
            vec![1, 1],
        ),
        (
            "F4-273-cgc",
            GroupId::f4(),
            vec![0, 0, 0, 1],
            vec![0, 0, 0, 1],
            vec![0, 0, 1, 0],
        ),
        (
            "F4-324-cgc",
            GroupId::f4(),
            vec![0, 0, 0, 1],
            vec![0, 0, 0, 1],
            vec![0, 0, 0, 2],
        ),
        (
            "E7-912-cgc",
            GroupId::e7(),
            vec![0, 0, 0, 0, 0, 0, 1],
            vec![1, 0, 0, 0, 0, 0, 0],
            vec![0, 1, 0, 0, 0, 0, 0],
        ),
        (
            "E7-1539-cgc",
            GroupId::e7(),
            vec![0, 0, 0, 0, 0, 0, 1],
            vec![0, 0, 0, 0, 0, 0, 1],
            vec![0, 0, 0, 0, 0, 1, 0],
        ),
        (
            "Spin7-R",
            GroupId::spin(7).unwrap(),
            vec![0, 0, 1],
            vec![0, 0, 1],
            vec![0, 0, 0],
        ),
        (
            "Spin7-F",
            GroupId::spin(7).unwrap(),
            vec![0, 0, 1],
            vec![0, 0, 1],
            vec![0, 0, 0],
        ),
        (
            "SU2xSU2-F",
            GroupId::cover(&[RootSystem::A(1), RootSystem::A(1)], 0).unwrap(),
            vec![1, 2],
            vec![1, 2],
            vec![0, 0],
        ),
        (
            "E6-1728-phase",
            GroupId::e6(),
            vec![1, 1, 0, 0, 0, 0],
            vec![0; 6],
            vec![0; 6],
        ),
        (
            "E6-7371-phase",
            GroupId::e6(),
            vec![1, 0, 0, 0, 1, 0],
            vec![0; 6],
            vec![0; 6],
        ),
        ("G2-7-F", GroupId::g2(), vec![1, 0], vec![1, 0], vec![1, 0]),
        ("G2-7-R", GroupId::g2(), vec![1, 0], vec![1, 0], vec![1, 0]),
        ("G2-14-F", GroupId::g2(), vec![0, 1], vec![0, 1], vec![0, 1]),
        ("G2-14-R", GroupId::g2(), vec![0, 1], vec![0, 1], vec![0, 1]),
        ("G2-27-F", GroupId::g2(), vec![2, 0], vec![2, 0], vec![2, 0]),
        ("G2-27-R", GroupId::g2(), vec![2, 0], vec![2, 0], vec![2, 0]),
        (
            "E6-27-phase",
            GroupId::e6(),
            vec![1, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
        ),
        (
            "E6-351-phase",
            GroupId::e6(),
            vec![0, 0, 0, 0, 1, 0],
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
        ),
        (
            "E8-27000-cgc",
            GroupId::e8(),
            vec![0, 0, 0, 0, 0, 0, 0, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 2],
            vec![0, 0, 0, 0, 0, 0, 0, 1],
        ),
        (
            "E8-30380-cgc",
            GroupId::e8(),
            vec![0, 0, 0, 0, 0, 0, 0, 1],
            vec![0, 0, 0, 0, 0, 0, 1, 0],
            vec![0, 0, 0, 0, 0, 0, 0, 1],
        ),
    ] {
        if std::env::var("FS_PERF_CASE").is_ok_and(|filter| filter != name) {
            continue;
        }
        for _ in 0..samples {
            fusion_symbols::cache::reset();
            let g = LieGroup::new(id.clone()).unwrap();
            let (a, b, c) = (
                g.irrep(&a).unwrap(),
                g.irrep(&b).unwrap(),
                g.irrep(&c).unwrap(),
            );
            let query = || {
                if name.ends_with("phase") {
                    assert_eq!(black_box(g.frobenius_schur_phase(&a).unwrap()), 1.0);
                } else if name.ends_with("R") {
                    black_box(g.r_symbol(&a, &b, &c).unwrap());
                } else if name.ends_with("F") {
                    black_box(g.f_symbol(&a, &b, &a, &a, &c, &c).unwrap());
                } else {
                    let cg = black_box(g.cgc(&a, &b, &c).unwrap());
                    assert_eq!(cg.dims()[3], 1);
                }
            };
            measure(name, query, query);
        }
    }
}
