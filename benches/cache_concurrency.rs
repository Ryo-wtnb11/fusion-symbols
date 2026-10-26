//! Warm public-API F throughput; the same file also builds against 9d24600.
//! Reports median/min/max aggregate queries/s from five 500 ms rounds.
use fusion_symbols::{
    category::FusionCategory,
    lie::{GroupId, LieGroup},
};
use std::{
    hint::black_box,
    sync::{
        atomic::{AtomicBool, Ordering},
        Barrier,
    },
    thread,
    time::{Duration, Instant},
};

fn measure<Q: Fn() + Send>(name: &str, setup: impl Fn(usize) -> Q) {
    for threads in [1, 4, 16] {
        let mut rates = Vec::new();
        for _ in 0..5 {
            let queries: Vec<_> = (0..threads).map(&setup).collect();
            for query in &queries {
                query();
            }
            let ready = Barrier::new(threads + 1);
            let start = Barrier::new(threads + 1);
            let stop = AtomicBool::new(false);
            let rate = thread::scope(|scope| {
                let handles: Vec<_> = queries
                    .into_iter()
                    .map(|query| {
                        let (ready, start, stop) = (&ready, &start, &stop);
                        scope.spawn(move || {
                            ready.wait();
                            start.wait();
                            let mut queries = 0u64;
                            while !stop.load(Ordering::Relaxed) {
                                for _ in 0..256 {
                                    query();
                                }
                                queries += 256;
                            }
                            queries
                        })
                    })
                    .collect();
                ready.wait();
                let began = Instant::now();
                start.wait();
                thread::sleep(Duration::from_millis(500));
                stop.store(true, Ordering::Relaxed);
                let count: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
                count as f64 / began.elapsed().as_secs_f64()
            });
            rates.push(rate);
        }
        rates.sort_by(f64::total_cmp);
        println!(
            "{name},{threads},{:.0},{:.0},{:.0}",
            rates[2], rates[0], rates[4]
        );
    }
}

fn main() {
    println!("case,threads,median_qps,min_qps,max_qps");
    let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    for distinct in [false, true] {
        measure(
            if distinct {
                "su2_distinct"
            } else {
                "su2_shared"
            },
            |index| {
                let g = su2.clone();
                let spin = if distinct { 2 * (index as i64 + 1) } else { 2 };
                let a = g.irrep(&[spin], &[]).unwrap();
                move || {
                    black_box(g.f_symbol(black_box(&a), &a, &a, &a, &a, &a).unwrap());
                }
            },
        );
    }
    #[cfg(feature = "lie-generated")]
    {
        let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        let octet = su3.irrep(&[1, 1], &[]).unwrap();
        measure("su3_octet", |_| {
            let (g, a) = (su3.clone(), octet.clone());
            move || {
                black_box(g.f_symbol(black_box(&a), &a, &a, &a, &a, &a).unwrap());
            }
        });
        let b2 = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
        let unit = b2.unit();
        let vector = b2.irrep(&[1, 0], &[]).unwrap();
        let adjoint = b2.irrep(&[0, 2], &[]).unwrap();
        measure("b2", |_| {
            let (g, u, v, a) = (b2.clone(), unit.clone(), vector.clone(), adjoint.clone());
            move || {
                black_box(g.f_symbol(black_box(&u), &v, &v, &a, &v, &a).unwrap());
            }
        });
    }
}
