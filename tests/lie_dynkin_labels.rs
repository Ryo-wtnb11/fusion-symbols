use crate::common::counting_alloc;
use fusion_symbols::{
    category::FusionCategory,
    lie::{GroupId, LieGroup, LieIrrep, RootSystem},
};

// The iterator yields `expected` (including zeros) with an exact length after
// every step, then stays exhausted, without allocating.
fn check_labels(a: &LieIrrep, expected: &[i64]) {
    let before = counting_alloc::calls();
    let mut labels = std::hint::black_box(a).dynkin();
    assert_eq!(labels.len(), expected.len());
    for (i, &coordinate) in expected.iter().enumerate() {
        assert_eq!(labels.next(), Some(coordinate));
        assert_eq!(labels.len(), expected.len() - i - 1);
        assert_eq!(labels.size_hint(), (labels.len(), Some(labels.len())));
    }
    assert_eq!(labels.next(), None);
    assert_eq!(labels.next(), None);
    assert_eq!(labels.len(), 0);
    let after = counting_alloc::calls();
    assert_eq!(after, before);
}

#[cfg(feature = "lie-generated")]
// Textbook duals of a simple factor: A(r) reverses its Dynkin labels; B, C
// and D(even rank) irreps, and rank one, are self-dual.
fn dual_labels(id: &GroupId, label: &[i64]) -> Vec<i64> {
    match id.root_systems() {
        [RootSystem::A(r)] if *r >= 2 => label.iter().rev().copied().collect(),
        [RootSystem::A(_) | RootSystem::B(_) | RootSystem::C(_)] => label.to_vec(),
        [RootSystem::D(r)] if r % 2 == 0 => label.to_vec(),
        other => panic!("no dual oracle for {other:?}"),
    }
}

#[cfg(feature = "lie-generated")]
// A channel's coordinates have no input array: reconstructing the channel
// from them is the independent check.
fn check_channel(cat: &LieGroup, c: &LieIrrep) {
    let expected: Vec<i64> = c.dynkin().collect();
    assert_eq!(&cat.irrep(&expected).unwrap(), c);
    check_labels(c, &expected);
}

#[test]
fn dynkin_matches_input_labels_without_allocation() {
    let _serial = crate::common::counting_alloc::serial();
    // Initialize thread parking before measuring synchronization.
    let _thread = std::thread::current();
    // A worker's allocations must not enter this thread's measurement.
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            barrier.wait(); // Warm thread parking before measuring.
            barrier.wait();
            let measured = counting_alloc::measure(|| std::hint::black_box(vec![0u8; 64]));
            barrier.wait();
            measured
        });
        barrier.wait(); // Warm thread parking before measuring.
        let measured = counting_alloc::measure(|| {
            barrier.wait();
            barrier.wait();
        });
        assert_eq!((measured.0, measured.1, measured.2), (0, 0, 0));
        let measured = worker.join().unwrap();
        assert_eq!((measured.0, measured.1, measured.2), (1, 64, 64));
    });
    // Higher-rank simple centers still admit labels without a scratch allocation.
    for id in [
        GroupId::psu(13).unwrap(),
        GroupId::so(25).unwrap(),
        GroupId::psp(24).unwrap(),
        GroupId::pso(24).unwrap(),
    ] {
        let before = counting_alloc::calls();
        assert!(id.admits(&[2; 12], &[]));
        std::hint::black_box(id.clone());
        assert_eq!(counting_alloc::calls(), before);
    }
    for id in [
        GroupId::su(2).unwrap(),
        GroupId::spin(3).unwrap(),
        GroupId::sp(2).unwrap(),
        GroupId::so(3).unwrap(),
        GroupId::psu(2).unwrap(),
    ] {
        let cat = LieGroup::new(id.clone()).unwrap();
        for label in 0..=200 {
            if !id.admits(&[label], &[]) {
                continue;
            }
            let a = cat.irrep(&[label]).unwrap();
            check_labels(&a, &[label]);
            check_labels(&cat.dual(&a), &[label]);
        }
        let a = cat.irrep(&[4]).unwrap();
        let b = cat.irrep(&[6]).unwrap();
        let expected = cat.fusion(&a, &b).unwrap().count();
        let before = counting_alloc::calls();
        let mut channels = 0;
        for c in cat.fusion(&a, &b).unwrap() {
            for coordinate in c.dynkin() {
                std::hint::black_box(coordinate);
            }
            for coordinate in cat.dual(&c).dynkin() {
                std::hint::black_box(coordinate);
            }
            channels += 1;
        }
        let after = counting_alloc::calls();
        assert_eq!(after, before);
        assert_eq!(channels, expected);
    }

    #[cfg(feature = "lie-generated")]
    for id in [
        GroupId::su(3).unwrap(),
        GroupId::su(4).unwrap(),
        GroupId::psu(3).unwrap(),
        GroupId::spin(5).unwrap(),
        GroupId::so(5).unwrap(),
        GroupId::sp(4).unwrap(),
        GroupId::spin(7).unwrap(),
        GroupId::spin(8).unwrap(),
        GroupId::so(8).unwrap(),
    ] {
        let cat = LieGroup::new(id.clone()).unwrap();
        let rank = id.semisimple_rank();
        // All admitted coordinates in 0..=2, independently checking round trips.
        for mut code in 0..3usize.pow(rank as u32) {
            let label: Vec<_> = (0..rank)
                .map(|_| {
                    let coordinate = (code % 3) as i64;
                    code /= 3;
                    coordinate
                })
                .collect();
            if id.admits(&label, &[]) {
                let a = cat.irrep(&label).unwrap();
                check_labels(&a, &label);
                check_labels(&cat.dual(&a), &dual_labels(&id, &label));
            }
        }
        // Small admitted fundamental seeds (both D-series chiralities included).
        let mut seeds = vec![cat.unit()];
        for i in 0..rank {
            let mut label = vec![0; rank];
            label[i] = 1;
            if id.admits(&label, &[]) {
                seeds.push(cat.irrep(&label).unwrap());
            }
        }
        if id == GroupId::psu(3).unwrap() {
            seeds.push(cat.irrep(&[1, 1]).unwrap());
        }
        for a in &seeds {
            for b in &seeds {
                for c in cat.fusion(a, b).unwrap() {
                    check_channel(&cat, &c);
                    check_channel(&cat, &cat.dual(&c));
                }
            }
        }
    }

    #[cfg(feature = "lie-generated")]
    {
        let cat = LieGroup::new(GroupId::spin(8).unwrap()).unwrap();
        // The last two D coordinates can require an intermediate of twice i64::MAX.
        for label in [[0, 0, i64::MAX, 0], [0, 0, 0, i64::MAX]] {
            let a = cat.irrep(&label).unwrap();
            let before = counting_alloc::calls();
            assert!(std::hint::black_box(&a).dynkin().eq(label));
            let after = counting_alloc::calls();
            assert_eq!(after, before);
        }
    }
}

#[test]
fn torus_and_composite_coordinates() {
    let _serial = crate::common::counting_alloc::serial();
    // Tori and the trivial group have no Dynkin coordinates; charges are separate.
    let u1 = LieGroup::new(GroupId::u(1).unwrap()).unwrap();
    check_labels(&u1.irrep_with_charges(&[], &[3]).unwrap(), &[]);
    let trivial = LieGroup::new(GroupId::new(&[], 0, &[]).unwrap()).unwrap();
    check_labels(&trivial.unit(), &[]);
    let u2 = LieGroup::new(GroupId::u(2).unwrap()).unwrap();
    let x = u2.irrep_with_charges(&[1], &[3]).unwrap();
    check_labels(&x, &[1]);
    assert!(x.charges().eq([3]));
    // D(2) keeps α₁ then α₂, zeros included.
    let spin4 = LieGroup::new(GroupId::spin(4).unwrap()).unwrap();
    check_labels(&spin4.irrep(&[2, 0]).unwrap(), &[2, 0]);
    check_labels(&spin4.irrep(&[0, 3]).unwrap(), &[0, 3]);
    // Mixed covers concatenate factor coordinates in factor order.
    let cover = GroupId::cover(&[RootSystem::A(1), RootSystem::A(1)], 1).unwrap();
    let cat = LieGroup::new(cover).unwrap();
    let y = cat.irrep_with_charges(&[3, 0], &[-2]).unwrap();
    check_labels(&y, &[3, 0]);
    assert!(y.charges().eq([-2]));
    #[cfg(feature = "lie-generated")]
    {
        let cover = GroupId::cover(&[RootSystem::A(1), RootSystem::A(2)], 0).unwrap();
        let cat = LieGroup::new(cover).unwrap();
        check_labels(&cat.irrep(&[1, 0, 2]).unwrap(), &[1, 0, 2]);
        check_labels(&cat.dual(&cat.irrep(&[1, 0, 2]).unwrap()), &[1, 2, 0]);
    }
}
