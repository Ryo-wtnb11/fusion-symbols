#![allow(dead_code)]
use fusion_symbols::{
    category::{self, CgcBlock, FusionCategory},
    lie::{GroupId, LieError, LieGroup, LieIrrep},
};
pub fn group(a: &LieIrrep) -> LieGroup {
    LieGroup::new(GroupId::su(a.dynkin().len() + 1).unwrap()).unwrap()
}
pub fn irrep(d: &[i64]) -> Result<LieIrrep, LieError> {
    LieGroup::new(GroupId::su(d.len() + 1).unwrap())?.irrep(d)
}
pub fn trivial(n: usize) -> Result<LieIrrep, LieError> {
    irrep(&vec![0; n - 1])
}
pub fn dual(a: &LieIrrep) -> LieIrrep {
    group(a).dual(a)
}
#[allow(clippy::too_many_arguments)]
pub fn f_symbol(
    a: &LieIrrep,
    b: &LieIrrep,
    c: &LieIrrep,
    d: &LieIrrep,
    e: &LieIrrep,
    f: &LieIrrep,
) -> Result<fusion_symbols::category::FBlock<f64>, LieError> {
    group(a).f_symbol(a, b, c, d, e, f)
}
pub fn r_symbol(
    a: &LieIrrep,
    b: &LieIrrep,
    c: &LieIrrep,
) -> Result<fusion_symbols::category::RBlock<f64>, LieError> {
    group(a).r_symbol(a, b, c)
}
pub fn check_f_unitarity(
    a: &LieIrrep,
    b: &LieIrrep,
    c: &LieIrrep,
    d: &LieIrrep,
) -> Result<(), LieError> {
    assert!(category::f_unitarity_residual(&group(a), a, b, c, d)? < 1e-10);
    Ok(())
}
pub fn check_pentagon(
    a: &LieIrrep,
    b: &LieIrrep,
    c: &LieIrrep,
    d: &LieIrrep,
) -> Result<(), LieError> {
    assert!(category::pentagon_residual(&group(a), a, b, c, d)? < 1e-10);
    Ok(())
}
pub fn check_hexagon(a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<(), LieError> {
    assert!(category::hexagon_residual(&group(a), a, b, c)? < 1e-10);
    Ok(())
}
pub fn product(
    a: &LieIrrep,
    b: &LieIrrep,
) -> Result<std::collections::BTreeMap<LieIrrep, u32>, LieError> {
    let cat = group(a);
    Ok(cat
        .fusion(a, b)?
        .map(|c| {
            let n = cat.nsymbol(a, b, &c);
            (c, n)
        })
        .collect())
}
pub fn cgc(
    a: &LieIrrep,
    b: &LieIrrep,
    c: &LieIrrep,
) -> Result<fusion_symbols::lie::LieCgc, LieError> {
    use fusion_symbols::category::GroupRepresentations;
    group(a).cgc(a, b, c)
}

pub fn irrep_in(group: &GroupId, d: &[i64]) -> Result<LieIrrep, LieError> {
    LieGroup::new(group.clone())?.irrep(d)
}
#[derive(Debug, PartialEq)]
pub struct Entry {
    pub m1: u32,
    pub m2: u32,
    pub m3: u32,
    pub mu: u32,
    pub value: f64,
}
pub fn entries(cgc: &fusion_symbols::lie::LieCgc) -> Vec<Entry> {
    let [d1, d2, d3, n] = cgc.dims();
    let mut entries = Vec::new();
    for m1 in 0..d1 {
        for m2 in 0..d2 {
            for m3 in 0..d3 {
                for mu in 0..n {
                    let value = cgc.at([m1, m2, m3, mu]);
                    if value != 0.0 {
                        entries.push(Entry {
                            m1: m1 as u32,
                            m2: m2 as u32,
                            m3: m3 as u32,
                            mu: mu as u32,
                            value,
                        });
                    }
                }
            }
        }
    }
    entries
}

pub fn torus() -> LieGroup {
    LieGroup::new(GroupId::u(1).unwrap()).unwrap()
}
pub fn u1_irrep(q: i32) -> Result<LieIrrep, LieError> {
    torus().irrep_with_charges(&[], &[i64::from(q)])
}
