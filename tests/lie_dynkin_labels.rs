use fusion_symbols::{
    category::FusionCategory,
    lie::{GroupId, LieGroup, LieIrrep},
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
struct Counting;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;

// Compatibility with owned materialization, including length after every step.
// One test keeps this process-wide allocation counter isolated.
fn check_labels(a: &LieIrrep) {
    let expected = a.dynkin();
    let before = ALLOCS.load(Ordering::Relaxed);
    let mut labels = std::hint::black_box(a).dynkin_labels();
    assert_eq!(labels.len(), expected.len());
    for (i, &coordinate) in expected.iter().enumerate() {
        assert_eq!(labels.next(), Some(coordinate));
        assert_eq!(labels.len(), expected.len() - i - 1);
        assert_eq!(labels.size_hint(), (labels.len(), Some(labels.len())));
    }
    assert_eq!(labels.next(), None);
    assert_eq!(labels.next(), None);
    assert_eq!(labels.len(), 0);
    let after = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(after, before);
}

#[test]
fn dynkin_labels_match_owned_labels_without_allocation() {
    for id in [
        GroupId::su(2).unwrap(),
        GroupId::spin(3).unwrap(),
        GroupId::sp(2).unwrap(),
        GroupId::so(3).unwrap(),
        GroupId::psu(2).unwrap(),
    ] {
        let cat = LieGroup::new(id).unwrap();
        for label in 0..=200 {
            if !id.admits(&[label]) {
                continue;
            }
            let a = cat.irrep(&[label]).unwrap();
            assert!(a.dynkin_labels().eq([label]));
            check_labels(&a);
        }
        let a = cat.irrep(&[4]).unwrap();
        let b = cat.irrep(&[6]).unwrap();
        let expected = cat.fusion(&a, &b).unwrap().count();
        let before = ALLOCS.load(Ordering::Relaxed);
        let mut channels = 0;
        for c in cat.fusion(&a, &b).unwrap() {
            for coordinate in c.dynkin_labels() {
                std::hint::black_box(coordinate);
            }
            for coordinate in cat.dual(&c).dynkin_labels() {
                std::hint::black_box(coordinate);
            }
            channels += 1;
        }
        let after = ALLOCS.load(Ordering::Relaxed);
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
        let cat = LieGroup::new(id).unwrap();
        let rank = id.root_system.rank();
        // All admitted coordinates in 0..=2, independently checking round trips.
        for mut code in 0..3usize.pow(rank as u32) {
            let label: Vec<_> = (0..rank)
                .map(|_| {
                    let coordinate = (code % 3) as i64;
                    code /= 3;
                    coordinate
                })
                .collect();
            if id.admits(&label) {
                let a = cat.irrep(&label).unwrap();
                assert!(a.dynkin_labels().eq(label));
                check_labels(&a);
                check_labels(&cat.dual(&a));
            }
        }
        // Small admitted fundamental seeds (both D-series chiralities included).
        let mut seeds = vec![cat.unit()];
        for i in 0..rank {
            let mut label = vec![0; rank];
            label[i] = 1;
            if id.admits(&label) {
                seeds.push(cat.irrep(&label).unwrap());
            }
        }
        if id == GroupId::psu(3).unwrap() {
            seeds.push(cat.irrep(&[1, 1]).unwrap());
        }
        for a in &seeds {
            for b in &seeds {
                for c in cat.fusion(a, b).unwrap() {
                    check_labels(&c);
                    check_labels(&cat.dual(&c));
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
            let before = ALLOCS.load(Ordering::Relaxed);
            assert!(std::hint::black_box(&a).dynkin_labels().eq(label));
            let after = ALLOCS.load(Ordering::Relaxed);
            assert_eq!(after, before);
        }
    }
}
