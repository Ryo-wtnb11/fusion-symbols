//! Allocation measurements exclude other test/harness threads.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

pub fn serial() -> std::sync::MutexGuard<'static, ()> {
    super::serial()
}

thread_local! {
    static COUNTERS: Cell<(usize, usize, usize)> = const { Cell::new((0, 0, 0)) };
}

fn record(size: usize) {
    // Allocations during thread teardown must also remain safe.
    let _ = COUNTERS.try_with(|c| {
        let (calls, bytes, largest) = c.get();
        c.set((calls + 1, bytes + size, largest.max(size)));
    });
}

pub struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

pub fn calls() -> usize {
    COUNTERS.with(|c| c.get().0)
}

/// `(calls, bytes, largest single allocation, result)` made by this thread.
pub fn measure<T>(f: impl FnOnce() -> T) -> (usize, usize, usize, T) {
    let (c0, b0, _) = COUNTERS.with(|c| {
        let before = c.get();
        c.set((before.0, before.1, 0));
        before
    });
    let result = f();
    let (calls, bytes, largest) = COUNTERS.with(Cell::get);
    (calls - c0, bytes - b0, largest, result)
}
