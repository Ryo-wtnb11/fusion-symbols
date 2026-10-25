//! Allocation budgets share one allocator and serialize cache access.

mod common;

#[global_allocator]
static ALLOCATOR: common::counting_alloc::Counting = common::counting_alloc::Counting;

mod lie_a1_alloc;
mod lie_dynkin_labels;
mod lie_generated_label_alloc;
mod product_alloc;
mod warm_ab_alloc;
mod warm_hit_alloc;

mod torus_alloc;
