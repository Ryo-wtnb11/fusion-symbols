//! The F- and R-symbol contractions of the Cartan engine
//! (`docs/gauge_cartan.md` §6) and the generation gates on completed blocks
//! (§7, gates 6 and 7).
//!
//! The F-symbol is the contraction of four Clebsch–Gordan tensors over their
//! magnetic indices, leaving four outer multiplicity indices `[μ, ν, κ, λ]`
//! (SUNRepresentations.jl `sector.jl:_Fsymbol`, `:58-89`; the contraction wiring
//! and axis order match TensorKitSectors `sectors.jl:Fsymbol_from_fusiontensor`,
//! `:406-418`). The CGC are real, so conjugation is the identity and is elided.
//!
//! # Contraction strategy: sparse key-matching accumulation
//!
//! CGC are sparse, and each F/R element is a sum over matched magnetic-index
//! keys, so the contraction is a sequence of sparse joins over shared magnetic
//! indices — data-structure code, not a dense numeric kernel (a densified GEMM
//! would be mostly zeros and pay a densify+GEMM+scatter tax the join avoids).
//! The join order is normative (§6.2): it fixes every rounded product and sum.

use std::collections::HashMap;

/// CGC entries grouped by one magnetic index → `(idx_i, idx_j, mult, value)`.
type GroupBy4 = HashMap<u32, Vec<(u32, u32, u32, f64)>>;

/// A partial-contraction result keyed by the three shared magnetic indices
/// `(ma, mb, mc)` → list of `(mult_p, mult_q, value)`.
type PairGroup = HashMap<(u32, u32, u32), Vec<(u32, u32, f64)>>;

/// One nonzero entry of a sparse CGC channel: `value` at public magnetic
/// indices `(m1, m2, m3)` and outer-multiplicity `mu` (`docs/gauge_cartan.md`
/// §4.7).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MEntry {
    /// 0-based magnetic index in the left factor.
    pub m1: u32,
    /// 0-based magnetic index in the right factor.
    pub m2: u32,
    /// 0-based magnetic index in the coupled irrep.
    pub m3: u32,
    /// 0-based outer-multiplicity index (trailing axis).
    pub mu: u32,
    /// The coefficient, rounded once.
    pub value: f64,
}

use crate::category::{FBlock, RBlock, SymbolBlock};
use crate::lie::general::{GeneralFamily, RequestModules};
use crate::lie::LieError;

/// A rounded CGC, F or R value that failed a generation gate. The gates only
/// verify; a failing value is never repaired or returned.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GateError {
    /// A CGC entry outside its index range or off its weight.
    Support,
    /// A nonfinite entry or residual in `gate`.
    NonFinite { gate: &'static str },
    /// The largest residual of `gate` exceeds its tolerance.
    Residual { gate: &'static str, residual: f64 },
}

impl std::fmt::Display for GateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Support => f.write_str("CGC entry outside its index range or weight"),
            Self::NonFinite { gate } => write!(f, "{gate} gate: nonfinite value"),
            Self::Residual { gate, residual } => {
                write!(f, "{gate} gate: residual {residual:e} exceeds tolerance")
            }
        }
    }
}

impl std::error::Error for GateError {}

/// Gate 7: every entry of a completed F block is finite. A fixed
/// `F^{abc}_d[e, f]` is only a subblock of a change of basis, so it has no
/// unitarity of its own.
pub(crate) fn check_f_block(block: &FBlock<f64>) -> Result<(), GateError> {
    if block.as_slice().iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(GateError::NonFinite { gate: "F" })
    }
}

/// Gate 6: a completed R block is a change of orthonormal multiplicity basis:
/// every entry finite and `max |RᵀR − I| ≤ 1e-9`. The empty block passes.
pub(crate) fn check_r_block(block: &RBlock<f64>) -> Result<(), GateError> {
    const TOL_UNITARY: f64 = 1e-9;
    let ([n, _], r) = (block.dims(), block.as_slice());
    if !r.iter().all(|x| x.is_finite()) {
        return Err(GateError::NonFinite { gate: "R" });
    }
    let mut worst = 0.0f64;
    for mu in 0..n {
        for nu in mu..n {
            let dot: f64 = (0..n).map(|rho| r[rho * n + mu] * r[rho * n + nu]).sum();
            let residual = (dot - f64::from(u8::from(mu == nu))).abs();
            // An overflowed accumulation is nonfinite, which `max` would hide.
            if !residual.is_finite() {
                return Err(GateError::NonFinite { gate: "R" });
            }
            worst = worst.max(residual);
        }
    }
    if worst > TOL_UNITARY {
        return Err(GateError::Residual {
            gate: "R",
            residual: worst,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// F-symbol raw contraction (sector.jl:_Fsymbol, :58-89).
// ---------------------------------------------------------------------------

/// The F contraction with **reference** empty-vertex semantics (an all-zero
/// block, not an error): the reference pentagon/hexagon loops feed empty blocks
/// through harmlessly, with dimensions given by the vertex multiplicities.
///
/// With `A = CGC(a,b,e)`, `B = CGC(e,c,d)`, `C = CGC(b,c,f)`, `D = CGC(a,f,d)`
/// (the last two conjugated in the reference; real, so elided), and `B`/`D`
/// sliced at the first `d`-basis index (`m_d = 0`; the F-symbol is independent of
/// which `d`-state is fixed — the reference "first diagonal element"):
///
/// ```text
/// F[μ,ν,κ,λ] = Σ  D[ma,mf,λ] · C[mb,mc,mf,κ] · A[ma,mb,me,μ] · B[me,mc,ν]
///            ma,mb,me,mc,mf
/// ```
pub(crate) fn f_block_raw(
    fam: &GeneralFamily,
    [a, b, c, d, e, f]: [&[i64]; 6],
) -> Result<FBlock<f64>, LieError> {
    let n1 = fam.mult(a, b, e)?;
    let n2 = fam.mult(e, c, d)?;
    let n3 = fam.mult(b, c, f)?;
    let n4 = fam.mult(a, f, d)?;
    let dims = [n1, n2, n3, n4];
    if n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0 {
        return Ok(FBlock::zeros(dims));
    }

    let len = crate::lie::checked_size(dims)?;
    // Missed channels share their modules; each is released after the last
    // channel that may need it, so at most five of the six labels' modules
    // are live at once, and none during the contraction.
    let mut modules = RequestModules::default();
    let cab = fam.cgc_entries_in(a, b, e, &mut modules)?; // A[ma,mb,me,μ]
    let cecd = fam.cgc_entries_in(e, c, d, &mut modules)?; // B slice at m_d=0: [me,mc,ν]
    modules.retain(&[b, c, f, a, d]);
    let cbcf = fam.cgc_entries_in(b, c, f, &mut modules)?; // C[mb,mc,mf,κ]
    modules.retain(&[a, f, d]);
    #[cfg(test)]
    modules.observe_before_fourth();
    let cafd = fam.cgc_entries_in(a, f, d, &mut modules)?; // D slice at m_d=0: [ma,mf,λ]
    drop(modules);
    #[cfg(test)]
    crate::lie::general::observe_contraction();

    // Step 1: AB[(ma,mb,mc), (μ,ν)] = Σ_me A[ma,mb,me,μ] · B[me,mc,ν].
    // Group A and B by the shared magnetic index me, then cross each group.
    // B is CGC(e,c,d)[:, :, m_d=0, :]: keep only m3 == 0.
    let mut b_by_me: HashMap<u32, Vec<(u32, u32, f64)>> = HashMap::new();
    for x in cecd.iter() {
        if x.m3 == 0 {
            b_by_me.entry(x.m1).or_default().push((x.m2, x.mu, x.value)); // (mc, ν, vB)
        }
    }
    // Only magnetic groups present in the fixed-d slice can contribute.
    let mut a_by_me: GroupBy4 = HashMap::new();
    for x in cab.iter() {
        if b_by_me.contains_key(&x.m3) {
            a_by_me
                .entry(x.m3)
                .or_default()
                .push((x.m1, x.m2, x.mu, x.value)); // (ma, mb, μ, vA)
        }
    }
    // AB keyed by (ma,mb,mc) -> Vec<(μ, ν, value)>.
    let mut ab: PairGroup = HashMap::new();
    let mut me_keys: Vec<_> = a_by_me.keys().copied().collect();
    me_keys.sort_unstable();
    for me in me_keys {
        let Some(blist) = b_by_me.get(&me) else {
            continue;
        };
        for &(ma, mb, mu, va) in &a_by_me[&me] {
            for &(mc, nu, vb) in blist {
                ab.entry((ma, mb, mc)).or_default().push((mu, nu, va * vb));
            }
        }
    }

    // Step 2: CD[(ma,mb,mc), (κ,λ)] = Σ_mf C[mb,mc,mf,κ] · D[ma,mf,λ].
    // D is CGC(a,f,d)[:, :, m_d=0, :]: keep only m3 == 0.
    let mut d_by_mf: HashMap<u32, Vec<(u32, u32, f64)>> = HashMap::new();
    for x in cafd.iter() {
        if x.m3 == 0 {
            d_by_mf.entry(x.m2).or_default().push((x.m1, x.mu, x.value)); // (ma, λ, vD)
        }
    }
    let mut c_by_mf: GroupBy4 = HashMap::new();
    for x in cbcf.iter() {
        if d_by_mf.contains_key(&x.m3) {
            c_by_mf
                .entry(x.m3)
                .or_default()
                .push((x.m1, x.m2, x.mu, x.value)); // (mb, mc, κ, vC)
        }
    }
    let mut cd: PairGroup = HashMap::new();
    let mut mf_keys: Vec<_> = c_by_mf.keys().copied().collect();
    mf_keys.sort_unstable();
    for mf in mf_keys {
        let Some(dlist) = d_by_mf.get(&mf) else {
            continue;
        };
        for &(mb, mc, kappa, vc) in &c_by_mf[&mf] {
            for &(ma, lambda, vd) in dlist {
                if ab.contains_key(&(ma, mb, mc)) {
                    cd.entry((ma, mb, mc))
                        .or_default()
                        .push((kappa, lambda, vc * vd));
                }
            }
        }
    }

    // Step 3: F[μ,ν,κ,λ] = Σ_{ma,mb,mc} AB[(ma,mb,mc),(μ,ν)] · CD[(ma,mb,mc),(κ,λ)].
    let mut data = vec![0.0; len];
    let mut ab_keys: Vec<_> = ab.keys().copied().collect();
    ab_keys.sort_unstable();
    for key in ab_keys {
        let Some(cdlist) = cd.get(&key) else {
            continue;
        };
        for &(mu, nu, vab) in &ab[&key] {
            for &(kappa, lambda, vcd) in cdlist {
                let idx = FBlock::<f64>::flat(
                    dims,
                    usize::try_from(mu).map_err(|_| LieError::LabelOverflow)?,
                    usize::try_from(nu).map_err(|_| LieError::LabelOverflow)?,
                    usize::try_from(kappa).map_err(|_| LieError::LabelOverflow)?,
                    usize::try_from(lambda).map_err(|_| LieError::LabelOverflow)?,
                );
                data[idx] += vab * vcd;
            }
        }
    }
    Ok(FBlock::from_vec(dims, data))
}

// ---------------------------------------------------------------------------
// R-symbol raw contraction (sector.jl:_Rsymbol, :91-110).
// ---------------------------------------------------------------------------

/// The R contraction with reference empty-vertex semantics (zeros, not an
/// error); the gates feed empty blocks through harmlessly.
///
/// With `A = CGC(a,b,c)`, `B = CGC(b,a,c)` sliced at `m_c = 0`:
///
/// ```text
/// R[μ, ν] = Σ  A[ma, mb, μ] · B[mb, ma, ν]      (B conjugated; real, elided)
///          ma,mb
/// ```
pub(crate) fn r_block_raw(
    fam: &GeneralFamily,
    [a, b, c]: [&[i64]; 3],
) -> Result<RBlock<f64>, LieError> {
    let n1 = fam.mult(a, b, c)?; // rows μ
    let n2 = fam.mult(b, a, c)?; // cols ν  (== n1: fusion multiplicities are symmetric)
    if n1 == 0 || n2 == 0 {
        return Ok(RBlock::zeros(n1.max(n2)));
    }
    debug_assert_eq!(n1, n2, "N^c_ab == N^c_ba");

    let len = crate::lie::checked_size([n1, n1])?;
    // Both channels need the same modules, so sharing them adds none to the
    // live set; they are released before the contraction.
    let mut modules = RequestModules::default();
    let cab = fam.cgc_entries_in(a, b, c, &mut modules)?; // A[ma,mb,mc,μ]
    let cba = fam.cgc_entries_in(b, a, c, &mut modules)?; // B[mb,ma,mc,ν]
    drop(modules);
    #[cfg(test)]
    crate::lie::general::observe_contraction();

    // A slice at m_c = 0, keyed by (ma, mb) -> Vec<(μ, value)>.
    let mut a_map: HashMap<(u32, u32), Vec<(u32, f64)>> = HashMap::new();
    for x in cab.iter() {
        if x.m3 == 0 {
            a_map.entry((x.m1, x.m2)).or_default().push((x.mu, x.value));
        }
    }
    // B slice at m_c = 0, keyed by (ma, mb) = (B.m2, B.m1) -> Vec<(ν, value)>.
    let mut b_map: HashMap<(u32, u32), Vec<(u32, f64)>> = HashMap::new();
    for x in cba.iter() {
        if x.m3 == 0 {
            b_map.entry((x.m2, x.m1)).or_default().push((x.mu, x.value));
        }
    }

    let mut data = vec![0.0; len];
    let mut keys: Vec<_> = a_map.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        let Some(blist) = b_map.get(&key) else {
            continue;
        };
        for &(mu, va) in &a_map[&key] {
            for &(nu, vb) in blist {
                data[usize::try_from(mu).map_err(|_| LieError::LabelOverflow)? * n1
                    + usize::try_from(nu).map_err(|_| LieError::LabelOverflow)?] += va * vb;
            }
        }
    }
    Ok(RBlock::from_vec(n1, data))
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    fn r(n: usize, data: &[f64]) -> Result<(), GateError> {
        check_r_block(&RBlock::from_vec(n, data.to_vec()))
    }

    /// R must be a finite orthogonal change of basis; F only finite.
    #[test]
    fn r_and_f_gates() {
        let (c, s) = (0.6, 0.8);
        assert_eq!(r(0, &[]), Ok(()));
        assert_eq!(r(1, &[1.0]), Ok(()));
        assert_eq!(r(1, &[-1.0]), Ok(()));
        assert_eq!(r(2, &[c, -s, s, c]), Ok(()));
        assert!(matches!(
            r(1, &[0.5]),
            Err(GateError::Residual { gate: "R", residual }) if residual == 0.75
        ));
        // Unit columns that are not orthogonal.
        assert!(matches!(
            r(2, &[1.0, c, 0.0, s]),
            Err(GateError::Residual { gate: "R", .. })
        ));
        for bad in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                r(2, &[c, bad, s, c]),
                Err(GateError::NonFinite { gate: "R" })
            );
            let f = FBlock::from_vec([1, 1, 1, 2], vec![0.25, bad]);
            assert_eq!(check_f_block(&f), Err(GateError::NonFinite { gate: "F" }));
        }
        // Finite entries whose Gram accumulation overflows.
        assert_eq!(r(1, &[1e200]), Err(GateError::NonFinite { gate: "R" }));
        // A fixed F[e, f] subblock need not be unitary.
        assert_eq!(
            check_f_block(&FBlock::from_vec([1, 1, 1, 1], vec![0.25])),
            Ok(())
        );
        assert_eq!(check_f_block(&FBlock::zeros([0, 1, 1, 1])), Ok(()));
    }
}
