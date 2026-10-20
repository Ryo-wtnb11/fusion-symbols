use fusion_symbols::{
    category::FusionCategory,
    lie::{FBlock, GroupId, LieGroup, RBlock},
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

#[test]
fn inline_blocks_and_warm_a1_symbols_allocate_nothing() {
    let cat = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
    let half = cat.irrep(&[1]).unwrap();
    let zero = cat.unit();
    let expected = cat
        .f_symbol(&half, &half, &half, &half, &zero, &zero)
        .unwrap();
    let stats = fusion_symbols::cache::base_cache_stats();
    let before = ALLOCS.load(Ordering::Relaxed);
    std::hint::black_box(FBlock::one(0.5));
    std::hint::black_box(RBlock::one(-1.0));
    let f = cat
        .f_symbol(&half, &half, &half, &half, &zero, &zero)
        .unwrap();
    let r = cat.r_symbol(&half, &half, &zero).unwrap();
    let empty_f = cat
        .f_symbol(&half, &zero, &zero, &zero, &zero, &zero)
        .unwrap();
    let empty_r = cat.r_symbol(&half, &zero, &zero).unwrap();
    let fusion = cat.fusion(&half, &half).unwrap();
    let after = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(after, before);
    assert_eq!(f, expected);
    assert_eq!(r.at(0, 0), -1.0);
    assert!(empty_f.data().is_empty());
    assert!(empty_r.data().is_empty());
    assert_eq!(fusion.count(), 2);
    let warm = fusion_symbols::cache::base_cache_stats();
    assert_eq!(warm.derived_f.misses, stats.derived_f.misses);
    assert_eq!(warm.derived_f.hits, stats.derived_f.hits + 1);
}
