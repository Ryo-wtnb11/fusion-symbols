#![allow(dead_code)]
use fusion_symbols::{
    category::{self, CgcBlock, FusionCategory},
    lie::{GroupId, LieCgc, LieError, LieGroup, LieIrrep},
};
#[derive(Clone, Copy, Debug)]
pub enum Series {
    B,
    C,
    D,
}
pub fn group_id(s: Series, r: usize) -> GroupId {
    match s {
        Series::B => GroupId::so(2 * r + 1).unwrap(),
        Series::C => GroupId::sp(2 * r).unwrap(),
        Series::D => GroupId::so(2 * r).unwrap(),
    }
}
pub fn group(s: Series, r: usize) -> Result<LieGroup, LieError> {
    LieGroup::new(group_id(s, r))
}
pub fn irrep(s: Series, d: &[i64]) -> Result<LieIrrep, LieError> {
    group(s, d.len())?.irrep(d)
}
pub fn irrep_in(g: &GroupId, d: &[i64]) -> Result<LieIrrep, LieError> {
    LieGroup::new(g.clone())?.irrep(d)
}
pub fn trivial(s: Series, r: usize) -> Result<LieIrrep, LieError> {
    irrep(s, &vec![0; r])
}
pub fn product(
    cat: &LieGroup,
    a: &LieIrrep,
    b: &LieIrrep,
) -> Result<std::collections::BTreeMap<LieIrrep, u32>, LieError> {
    Ok(cat
        .fusion(a, b)?
        .map(|c| {
            let n = cat.nsymbol(a, b, &c);
            (c, n)
        })
        .collect())
}
#[allow(clippy::too_many_arguments)]
pub fn f_symbol(
    cat: &LieGroup,
    a: &LieIrrep,
    b: &LieIrrep,
    c: &LieIrrep,
    d: &LieIrrep,
    e: &LieIrrep,
    f: &LieIrrep,
) -> Result<fusion_symbols::category::FBlock<f64>, LieError> {
    cat.f_symbol(a, b, c, d, e, f)
}
pub fn r_symbol(
    cat: &LieGroup,
    a: &LieIrrep,
    b: &LieIrrep,
    c: &LieIrrep,
) -> Result<fusion_symbols::category::RBlock<f64>, LieError> {
    cat.r_symbol(a, b, c)
}
pub fn check_f_unitarity(
    cat: &LieGroup,
    a: &LieIrrep,
    b: &LieIrrep,
    c: &LieIrrep,
    d: &LieIrrep,
) -> Result<(), LieError> {
    assert!(category::f_unitarity_residual(cat, a, b, c, d)? < 1e-9);
    Ok(())
}
pub fn check_pentagon(
    cat: &LieGroup,
    a: &LieIrrep,
    b: &LieIrrep,
    c: &LieIrrep,
    d: &LieIrrep,
) -> Result<(), LieError> {
    assert!(category::pentagon_residual(cat, a, b, c, d)? < 1e-8);
    Ok(())
}
pub fn check_hexagon(
    cat: &LieGroup,
    a: &LieIrrep,
    b: &LieIrrep,
    c: &LieIrrep,
) -> Result<(), LieError> {
    assert!(category::hexagon_residual(cat, a, b, c)? < 1e-8);
    Ok(())
}
/// Original dense catalog order: multiplicity, coupled column, first-factor-fast row.
pub fn copy(c: &LieCgc, mu: usize) -> Vec<f64> {
    let [d1, d2, d3, _] = c.dims();
    (0..d3)
        .flat_map(|m3| (0..d2).flat_map(move |m2| (0..d1).map(move |m1| c.at([m1, m2, m3, mu]))))
        .collect()
}
pub fn data(c: &LieCgc) -> Vec<f64> {
    (0..c.dims()[3]).flat_map(|mu| copy(c, mu)).collect()
}
pub fn fingerprint() -> Vec<u8> {
    group(Series::B, 2)
        .unwrap()
        .authority_fingerprint()
        .to_vec()
}
