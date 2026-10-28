use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fusion_symbols::{
    category::FusionCategory,
    lie::{GroupId, LieGroup, U1Irrep, U1},
};
fn torus(c: &mut Criterion) {
    let g = LieGroup::new(GroupId::u(1).unwrap()).unwrap();
    let a = g.irrep_with_charges(&[], &[3]).unwrap();
    let b = g.irrep_with_charges(&[], &[-5]).unwrap();
    let one = g.unit();
    c.bench_function("torus/clone", |b| b.iter(|| black_box(&a).clone()));
    c.bench_function("torus/fusion_collect", |bench| {
        bench.iter(|| {
            g.fusion(black_box(&a), black_box(&b))
                .unwrap()
                .collect::<Vec<_>>()
        })
    });
    let sum = g.irrep_with_charges(&[], &[-2]).unwrap();
    c.bench_function("torus/nsymbol", |bench| {
        bench.iter(|| g.nsymbol(black_box(&a), black_box(&b), black_box(&sum)))
    });
    c.bench_function("torus/f_symbol", |bench| {
        bench.iter(|| {
            g.f_symbol(black_box(&one), &one, &one, &one, &one, &one)
                .unwrap()
        })
    });
    c.bench_function("torus/dim", |b| b.iter(|| g.dim(black_box(&a))));

    let g = U1;
    let a = U1Irrep::new(3).unwrap();
    let b = U1Irrep::new(-5).unwrap();
    let one = g.unit();
    c.bench_function("u1/clone", |b| b.iter(|| *black_box(&a)));
    c.bench_function("u1/fusion_collect", |bench| {
        bench.iter(|| {
            g.fusion(black_box(&a), black_box(&b))
                .unwrap()
                .collect::<Vec<_>>()
        })
    });
    let sum = U1Irrep::new(-2).unwrap();
    c.bench_function("u1/nsymbol", |bench| {
        bench.iter(|| g.nsymbol(black_box(&a), black_box(&b), black_box(&sum)))
    });
    c.bench_function("u1/f_symbol", |bench| {
        bench.iter(|| {
            g.f_symbol(black_box(&one), &one, &one, &one, &one, &one)
                .unwrap()
        })
    });
    c.bench_function("u1/dim", |b| b.iter(|| g.dim(black_box(&a))));

    let g = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = g.irrep(&[1]).unwrap();
    let one = g.unit();
    c.bench_function("su2/clone", |b| b.iter(|| black_box(&half).clone()));
    c.bench_function("su2/nsymbol", |bench| {
        bench.iter(|| g.nsymbol(black_box(&half), black_box(&half), black_box(&one)))
    });
    c.bench_function("su2/f_symbol", |bench| {
        bench.iter(|| {
            g.f_symbol(black_box(&half), &half, &half, &half, &one, &one)
                .unwrap()
        })
    });
    c.bench_function("su2/dim", |b| b.iter(|| g.dim(black_box(&half))));
    c.bench_function("su2/fusion_collect", |bench| {
        bench.iter(|| {
            g.fusion(black_box(&half), black_box(&half))
                .unwrap()
                .collect::<Vec<_>>()
        })
    });
    let six = g.irrep(&[6]).unwrap();
    c.bench_function("su2/f_symbol_6", |bench| {
        bench.iter(|| {
            g.f_symbol(black_box(&six), &six, &six, &six, &six, &six)
                .unwrap()
        })
    });

    let g = LieGroup::new(GroupId::u(2).unwrap()).unwrap();
    let a = g.irrep_with_charges(&[1], &[1]).unwrap();
    let b = g.irrep_with_charges(&[1], &[-1]).unwrap();
    let one = g.unit();
    c.bench_function("u2/clone", |b| b.iter(|| black_box(&a).clone()));
    c.bench_function("u2/fusion_collect", |bench| {
        bench.iter(|| {
            g.fusion(black_box(&a), black_box(&b))
                .unwrap()
                .collect::<Vec<_>>()
        })
    });
    c.bench_function("u2/nsymbol", |bench| {
        bench.iter(|| g.nsymbol(black_box(&a), black_box(&b), black_box(&one)))
    });
    c.bench_function("u2/f_symbol", |bench| {
        bench.iter(|| g.f_symbol(black_box(&a), &b, &a, &a, &one, &one).unwrap())
    });
    c.bench_function("u2/dim", |b| b.iter(|| g.dim(black_box(&a))));

    #[cfg(feature = "lie-generated")]
    {
        // Keep group/label construction outside the timed warm lookup.
        let su3 = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        let octet = su3.irrep(&[1, 1]).unwrap();
        let _ = su3
            .f_symbol(&octet, &octet, &octet, &octet, &octet, &octet)
            .unwrap();
        c.bench_function("su3/f_symbol_octet", |bench| {
            bench.iter(|| {
                su3.f_symbol(black_box(&octet), &octet, &octet, &octet, &octet, &octet)
                    .unwrap()
            })
        });
        let b2 = LieGroup::new(GroupId::so(5).unwrap()).unwrap();
        let unit = b2.unit();
        let vector = b2.irrep(&[1, 0]).unwrap();
        let adjoint = b2.irrep(&[0, 2]).unwrap();
        let _ = b2
            .f_symbol(&unit, &vector, &vector, &adjoint, &vector, &adjoint)
            .unwrap();
        c.bench_function("b2/f_symbol", |bench| {
            bench.iter(|| {
                b2.f_symbol(
                    black_box(&unit),
                    &vector,
                    &vector,
                    &adjoint,
                    &vector,
                    &adjoint,
                )
                .unwrap()
            })
        });
    }
}
criterion_group!(benches, torus);
criterion_main!(benches);
