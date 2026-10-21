//! Generated labels share their weight storage: a warm fusion yields owned
//! channel labels without allocating.
#![cfg(feature = "lie-generated")]
use fusion_symbols::{
    category::FusionCategory,
    lie::{GroupId, LieGroup},
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

// One test keeps this process-wide allocation counter isolated. Warm means:
// the labels are constructed and one identical `fusion(a, b)` has been fully
// consumed, so the decomposition is in the product cache.
#[test]
fn warm_generated_fusion_yields_labels_without_allocation() {
    for (id, a, b, expected) in [
        // SU(3): 8 ⊗ 8 = 1 ⊕ 8 ⊕ 8 ⊕ 10 ⊕ 10̄ ⊕ 27 (distinct outcomes).
        (
            GroupId::su(3).unwrap(),
            vec![1, 1],
            vec![1, 1],
            vec![vec![0, 0], vec![0, 3], vec![1, 1], vec![2, 2], vec![3, 0]],
        ),
        // Sp(4): 4 ⊗ 4 = 1 ⊕ 5 ⊕ 10.
        (
            GroupId::sp(4).unwrap(),
            vec![1, 0],
            vec![1, 0],
            vec![vec![0, 0], vec![0, 1], vec![2, 0]],
        ),
        // Spin(5): 4 ⊗ 5 = 4 ⊕ 16 (spinor ⊗ vector).
        (
            GroupId::spin(5).unwrap(),
            vec![0, 1],
            vec![1, 0],
            vec![vec![0, 1], vec![1, 1]],
        ),
    ] {
        let cat = LieGroup::new(id).unwrap();
        let a = cat.irrep(&a).unwrap();
        let b = cat.irrep(&b).unwrap();
        let mut cold: Vec<Vec<i64>> = cat.fusion(&a, &b).unwrap().map(|c| c.dynkin()).collect();
        cold.sort();
        assert_eq!(cold, expected, "{id:?}");
        let mut sums = [0i64; 8];
        let before = ALLOCS.load(Ordering::Relaxed);
        let mut channels = 0;
        for c in cat.fusion(&a, &b).unwrap() {
            let copy = c.clone();
            for (i, x) in copy.dynkin_labels().enumerate() {
                sums[i] += x;
            }
            channels += 1;
        }
        let after = ALLOCS.load(Ordering::Relaxed);
        assert_eq!(after, before, "{id:?}");
        assert_eq!(channels, expected.len());
        for (i, s) in sums.iter().enumerate() {
            assert_eq!(*s, expected.iter().filter_map(|d| d.get(i)).sum::<i64>());
        }
    }
}
