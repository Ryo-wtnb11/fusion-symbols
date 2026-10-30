#![allow(dead_code)]
use fusion_symbols::category::SymbolBlock;
use fusion_symbols::{
    category::FusionCategory,
    lie::{GroupId, LieGroup, LieIrrep},
};
pub fn group() -> LieGroup {
    LieGroup::new(GroupId::su(2).unwrap()).unwrap()
}
pub fn ir(dj: u32) -> LieIrrep {
    group().irrep(&[i64::from(dj)]).unwrap()
}
pub fn exact_f(a: u32, b: u32, c: u32, d: u32, e: u32, f: u32) -> f64 {
    let block = group()
        .f_symbol(&ir(a), &ir(b), &ir(c), &ir(d), &ir(e), &ir(f))
        .unwrap();
    block.as_slice().first().copied().unwrap_or(0.0)
}
pub fn exact_r(a: u32, b: u32, c: u32) -> f64 {
    let block = group().r_symbol(&ir(a), &ir(b), &ir(c)).unwrap();
    block.as_slice().first().copied().unwrap_or(0.0)
}
pub fn phase(dj: u32) -> f64 {
    group().frobenius_schur_phase(&ir(dj)).unwrap()
}
pub fn twist(dj: u32) -> f64 {
    group().twist(&ir(dj))
}
