//! Family-generic F- and R-symbol core: the four-CGC contraction and the
//! pentagon/hexagon/F-unitarity gates, shared by the SU(N) (`crate::lie::sun::fr`)
//! and B/C/D (`crate::lie::bcd::fr`) surfaces.
//!
//! # Why one core, two callers
//!
//! The F-symbol is the contraction of four Clebsch–Gordan tensors over their
//! magnetic indices, leaving four outer multiplicity indices `[μ, ν, κ, λ]`
//! (SUNRepresentations.jl `sector.jl:_Fsymbol`, `:58-89`; the contraction wiring
//! and axis order match TensorKitSectors `sectors.jl:Fsymbol_from_fusiontensor`,
//! `:406-418`). This logic depends on nothing
//! SU(N)-specific: only on an irrep label type, a fusion-multiplicity source
//! `N^c_{ab}`, a CGC provider returning **sparse real** `(m1, m2, m3, mu, value)`
//! entries, and `f64` scalars. Real-valuedness holds for B/C/D CGC too (the sweep
//! produces real orthogonal isometries), so conjugation is the identity and is
//! elided on both surfaces — value-identical to the reference. This module is
//! that logic, parameterized by the [`Family`] seam.
//!
//! # The `&mut self` seam (why it does not force interior mutability on SU(N))
//!
//! [`Family`] takes `&mut self`. That is the superset both callers satisfy: the
//! B/C/D provider is a `&mut CanonicalCatalog` (append-only generator
//! materialization is genuinely mutating), while the SU(N) provider is a
//! stateless zero-sized type whose methods delegate to the process-global CGC
//! cache via free functions and simply ignore the `&mut`. A `&mut` to a local ZST
//! is always available, so SU(N) pays no lock and grows no interior-mutability
//! cell — the mutability is real for one impl and vacuous for the other.
//!
//! # Contraction strategy: sparse key-matching accumulation
//!
//! CGC are sparse, and each F/R element is a sum over matched magnetic-index
//! keys, so the contraction is a sequence of sparse joins over shared magnetic
//! indices — data-structure code, not a dense numeric kernel (a densified GEMM
//! would be mostly zeros and pay a densify+GEMM+scatter tax the join avoids).

use std::collections::HashMap;
use std::hash::Hash;

/// CGC entries grouped by one magnetic index → `(idx_i, idx_j, mult, value)`.
type GroupBy4 = HashMap<u32, Vec<(u32, u32, u32, f64)>>;

/// A partial-contraction result keyed by the three shared magnetic indices
/// `(ma, mb, mc)` → list of `(mult_p, mult_q, value)`.
type PairGroup = HashMap<(u32, u32, u32), Vec<(u32, u32, f64)>>;

/// One nonzero entry of a sparse m-basis CGC tensor: `value` at magnetic indices
/// `(m1, m2, m3)` and outer-multiplicity `mu`. The family-neutral form the core
/// consumes; each [`Family`] adapts its native CGC representation to this.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MEntry {
    /// 0-based magnetic index in the left factor.
    pub m1: u32,
    /// 0-based magnetic index in the right factor.
    pub m2: u32,
    /// 0-based magnetic index in the coupled irrep.
    pub m3: u32,
    /// 0-based outer-multiplicity index (trailing axis).
    pub mu: u32,
    /// The (real, standard-gauge) coefficient.
    pub value: f64,
}

/// The seam between the generic core and a representation family (SU(N) or
/// B/C/D). Exactly two implementors; not a public extension point.
pub(crate) trait Family {
    /// The irrep label type (used as a magnetic-block cache/memo key).
    type Irrep: Clone + Eq + Hash;
    /// The family's error type, surfaced unchanged by the core.
    type Error;
    fn overflow() -> Self::Error;

    /// `N^c_{ab}`, the fusion multiplicity of `a ⊗ b → c`. Errors on an
    /// ill-posed pairing (mixed rank/series/group) per the family's contract.
    fn mult(
        &mut self,
        a: &Self::Irrep,
        b: &Self::Irrep,
        c: &Self::Irrep,
    ) -> Result<usize, Self::Error>;

    /// The sparse real CGC entries coupling `a ⊗ b → c`. Called only for a
    /// non-empty channel (`mult > 0`), which the core checks first.
    fn cgc_entries(
        &mut self,
        a: &Self::Irrep,
        b: &Self::Irrep,
        c: &Self::Irrep,
    ) -> Result<Vec<MEntry>, Self::Error>;
}

// ---------------------------------------------------------------------------
// Dense F/R blocks (family-neutral: just dims + row-major f64 data).
// ---------------------------------------------------------------------------

use crate::category::{FBlock, RBlock};

// ---------------------------------------------------------------------------
// F-symbol raw contraction (sector.jl:_Fsymbol, :58-89).
// ---------------------------------------------------------------------------

/// The raw F contraction with **reference** empty-vertex semantics (an all-zero
/// block, not an error): the reference pentagon/hexagon loops feed empty blocks
/// through harmlessly. The per-family `f_symbol` wrappers preserve these zero
/// blocks, with dimensions given by the vertex multiplicities.
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn f_block_raw<F: Family>(
    fam: &mut F,
    a: &F::Irrep,
    b: &F::Irrep,
    c: &F::Irrep,
    d: &F::Irrep,
    e: &F::Irrep,
    f: &F::Irrep,
) -> Result<FBlock<f64>, F::Error> {
    let n1 = fam.mult(a, b, e)?;
    let n2 = fam.mult(e, c, d)?;
    let n3 = fam.mult(b, c, f)?;
    let n4 = fam.mult(a, f, d)?;
    let dims = [n1, n2, n3, n4];
    if n1 == 0 || n2 == 0 || n3 == 0 || n4 == 0 {
        return Ok(FBlock::zeros(dims));
    }

    let len = crate::lie::checked_size(dims).map_err(|_| F::overflow())?;
    let cab = fam.cgc_entries(a, b, e)?; // A[ma,mb,me,μ]
    let cecd = fam.cgc_entries(e, c, d)?; // B slice at m_d=0: [me,mc,ν]
    let cbcf = fam.cgc_entries(b, c, f)?; // C[mb,mc,mf,κ]
    let cafd = fam.cgc_entries(a, f, d)?; // D slice at m_d=0: [ma,mf,λ]

    // Step 1: AB[(ma,mb,mc), (μ,ν)] = Σ_me A[ma,mb,me,μ] · B[me,mc,ν].
    // Group A and B by the shared magnetic index me, then cross each group.
    let mut a_by_me: GroupBy4 = HashMap::new();
    for x in &cab {
        a_by_me
            .entry(x.m3)
            .or_default()
            .push((x.m1, x.m2, x.mu, x.value)); // (ma, mb, μ, vA)
    }
    // B is CGC(e,c,d)[:, :, m_d=0, :]: keep only m3 == 0.
    let mut b_by_me: HashMap<u32, Vec<(u32, u32, f64)>> = HashMap::new();
    for x in &cecd {
        if x.m3 == 0 {
            b_by_me.entry(x.m1).or_default().push((x.m2, x.mu, x.value)); // (mc, ν, vB)
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
    let mut c_by_mf: GroupBy4 = HashMap::new();
    for x in &cbcf {
        c_by_mf
            .entry(x.m3)
            .or_default()
            .push((x.m1, x.m2, x.mu, x.value)); // (mb, mc, κ, vC)
    }
    // D is CGC(a,f,d)[:, :, m_d=0, :]: keep only m3 == 0.
    let mut d_by_mf: HashMap<u32, Vec<(u32, u32, f64)>> = HashMap::new();
    for x in &cafd {
        if x.m3 == 0 {
            d_by_mf.entry(x.m2).or_default().push((x.m1, x.mu, x.value)); // (ma, λ, vD)
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
                cd.entry((ma, mb, mc))
                    .or_default()
                    .push((kappa, lambda, vc * vd));
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
                    usize::try_from(mu).map_err(|_| F::overflow())?,
                    usize::try_from(nu).map_err(|_| F::overflow())?,
                    usize::try_from(kappa).map_err(|_| F::overflow())?,
                    usize::try_from(lambda).map_err(|_| F::overflow())?,
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

/// The raw R contraction with reference empty-vertex semantics (zeros, not an
/// error); the gates feed empty blocks through harmlessly.
///
/// With `A = CGC(a,b,c)`, `B = CGC(b,a,c)` sliced at `m_c = 0`:
///
/// ```text
/// R[μ, ν] = Σ  A[ma, mb, μ] · B[mb, ma, ν]      (B conjugated; real, elided)
///          ma,mb
/// ```
pub(crate) fn r_block_raw<F: Family>(
    fam: &mut F,
    a: &F::Irrep,
    b: &F::Irrep,
    c: &F::Irrep,
) -> Result<RBlock<f64>, F::Error> {
    let n1 = fam.mult(a, b, c)?; // rows μ
    let n2 = fam.mult(b, a, c)?; // cols ν  (== n1: fusion multiplicities are symmetric)
    if n1 == 0 || n2 == 0 {
        return Ok(RBlock::zeros(n1.max(n2)));
    }
    debug_assert_eq!(n1, n2, "N^c_ab == N^c_ba");

    let len = crate::lie::checked_size([n1, n1]).map_err(|_| F::overflow())?;
    let cab = fam.cgc_entries(a, b, c)?; // A[ma,mb,mc,μ]
    let cba = fam.cgc_entries(b, a, c)?; // B[mb,ma,mc,ν]

    // A slice at m_c = 0, keyed by (ma, mb) -> Vec<(μ, value)>.
    let mut a_map: HashMap<(u32, u32), Vec<(u32, f64)>> = HashMap::new();
    for x in &cab {
        if x.m3 == 0 {
            a_map.entry((x.m1, x.m2)).or_default().push((x.mu, x.value));
        }
    }
    // B slice at m_c = 0, keyed by (ma, mb) = (B.m2, B.m1) -> Vec<(ν, value)>.
    let mut b_map: HashMap<(u32, u32), Vec<(u32, f64)>> = HashMap::new();
    for x in &cba {
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
                data[usize::try_from(mu).map_err(|_| F::overflow())? * n1
                    + usize::try_from(nu).map_err(|_| F::overflow())?] += va * vb;
            }
        }
    }
    Ok(RBlock::from_vec(n1, data))
}
