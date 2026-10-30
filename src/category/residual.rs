//! Structural identity residuals with explicit multiplicity sums.
use super::{FusionCategory, Scalar, SymbolBlock};
use std::collections::HashMap;

fn outcomes<C: FusionCategory>(
    cat: &C,
    a: &C::Object,
    b: &C::Object,
) -> Result<Vec<C::Object>, C::Error> {
    Ok(cat.fusion(a, b)?.collect())
}
fn multiplicity<C: FusionCategory>(
    cat: &C,
    a: &C::Object,
    b: &C::Object,
    c: &C::Object,
) -> Result<usize, C::Error> {
    Ok(cat.nsymbol(a, b, c) as usize)
}
fn intersect_products<C: FusionCategory>(
    cat: &C,
    a: &C::Object,
    b: &C::Object,
    c: &C::Object,
    d: &C::Object,
) -> Result<Vec<C::Object>, C::Error> {
    let left = outcomes(cat, a, b)?;
    let right = outcomes(cat, c, d)?;
    Ok(left.into_iter().filter(|x| right.contains(x)).collect())
}
#[derive(Clone)]
struct LogicalBlock<S, const R: usize> {
    dims: [usize; R],
    data: std::sync::Arc<[S]>,
}
impl<S: Scalar, const R: usize> LogicalBlock<S, R> {
    fn new<B: SymbolBlock<R, Scalar = S>>(block: B, dims: [usize; R]) -> Self {
        Self {
            dims,
            data: if dims.contains(&0) {
                Vec::new().into()
            } else {
                block.as_slice().into()
            },
        }
    }
    fn dims(&self) -> [usize; R] {
        self.dims
    }
    fn dim(&self) -> usize {
        self.dims[0]
    }
    fn get(&self, index: [usize; R]) -> S {
        let mut flat = 0;
        for (i, n) in index.into_iter().zip(self.dims) {
            assert!(i < n);
            flat = flat * n + i;
        }
        self.data[flat]
    }
}
type FKey<C> = [<C as FusionCategory>::Object; 6];
type RKey<C> = [<C as FusionCategory>::Object; 3];
struct BlockMemo<C: FusionCategory> {
    f: HashMap<FKey<C>, LogicalBlock<C::Scalar, 4>>,
    r: HashMap<RKey<C>, LogicalBlock<C::Scalar, 2>>,
}
impl<C: FusionCategory> Default for BlockMemo<C> {
    fn default() -> Self {
        Self {
            f: HashMap::new(),
            r: HashMap::new(),
        }
    }
}
impl<C: FusionCategory> BlockMemo<C> {
    #[allow(clippy::too_many_arguments)]
    fn f_block(
        &mut self,
        cat: &C,
        a: &C::Object,
        b: &C::Object,
        c: &C::Object,
        d: &C::Object,
        e: &C::Object,
        f: &C::Object,
    ) -> Result<LogicalBlock<C::Scalar, 4>, C::Error> {
        let key = [a, b, c, d, e, f].map(Clone::clone);
        if let Some(block) = self.f.get(&key) {
            return Ok(block.clone());
        }
        let dims = [(a, b, e), (e, c, d), (b, c, f), (a, f, d)]
            .map(|(x, y, z)| cat.nsymbol(x, y, z) as usize);
        let block = LogicalBlock::new(cat.f_symbol(a, b, c, d, e, f)?, dims);
        self.f.insert(key, block.clone());
        Ok(block)
    }
    fn r_block(
        &mut self,
        cat: &C,
        a: &C::Object,
        b: &C::Object,
        c: &C::Object,
    ) -> Result<LogicalBlock<C::Scalar, 2>, C::Error> {
        let key = [a, b, c].map(Clone::clone);
        if let Some(block) = self.r.get(&key) {
            return Ok(block.clone());
        }
        let dims = [cat.nsymbol(a, b, c) as usize, cat.nsymbol(b, a, c) as usize];
        let block = LogicalBlock::new(cat.r_symbol(a, b, c)?, dims);
        self.r.insert(key, block.clone());
        Ok(block)
    }
}
/// The worst F-move unitarity residual for fixed outer labels `(a, b, c, d)`.
///
/// The F-symbols form a square matrix `M` with rows `(e, μ, ν)`
/// (`e ∈ a⊗b`, `μ ∈ [0,N^e_{ab})`, `ν ∈ [0,N^d_{ec})`) and columns `(f, κ, λ)`
/// (`f ∈ b⊗c`, `κ ∈ [0,N^f_{bc})`, `λ ∈ [0,N^d_{af})`). The two associativity
/// bases are orthonormal, so `M` is unitary (`M M† = I`); this returns
/// the worst `|(M M† - I)_{ij}|`. The caller chooses its tolerance.
pub fn f_unitarity_residual<C: FusionCategory>(
    fam: &C,
    a: &C::Object,
    b: &C::Object,
    c: &C::Object,
    d: &C::Object,
) -> Result<f64, C::Error> {
    // Rows: (e, μ, ν). Columns: (f, κ, λ).
    let mut rows: Vec<(C::Object, usize, usize)> = Vec::new();
    for e in outcomes(fam, a, b)? {
        let n_ab_e = multiplicity(fam, a, b, &e)?;
        let n_ec_d = multiplicity(fam, &e, c, d)?;
        for mu in 0..n_ab_e {
            for nu in 0..n_ec_d {
                rows.push((e.clone(), mu, nu));
            }
        }
    }
    let mut cols: Vec<(C::Object, usize, usize)> = Vec::new();
    for f in outcomes(fam, b, c)? {
        let n_bc_f = multiplicity(fam, b, c, &f)?;
        let n_af_d = multiplicity(fam, a, &f, d)?;
        for kappa in 0..n_bc_f {
            for lambda in 0..n_af_d {
                cols.push((f.clone(), kappa, lambda));
            }
        }
    }

    // M[row, col].
    let nr = rows.len();
    let nc = cols.len();
    let mut m = vec![C::Scalar::zero(); nr * nc];
    let mut memo = BlockMemo::<C>::default();
    for (ri, (e, mu, nu)) in rows.iter().enumerate() {
        for (ci, (f, kappa, lambda)) in cols.iter().enumerate() {
            let block = memo.f_block(fam, a, b, c, d, e, f)?;
            m[ri * nc + ci] = block.get([*mu, *nu, *kappa, *lambda]);
        }
    }

    // worst |(M Mᵀ - I)_{ij}|.
    let mut worst = 0.0f64;
    for i in 0..nr {
        for j in 0..nr {
            let mut dot = C::Scalar::zero();
            for k in 0..nc {
                dot = dot + m[i * nc + k] * m[j * nc + k].conj();
            }
            let target = if i == j {
                C::Scalar::one()
            } else {
                C::Scalar::zero()
            };
            worst = worst.max((dot - target).abs());
        }
    }
    Ok(worst)
}

// ---------------------------------------------------------------------------
// Gate 2: pentagon (TensorKitSectors sectors.jl:pentagon_equation, :786-819).
// ---------------------------------------------------------------------------

/// The worst pentagon residual for the quadruple `(a, b, c, d)` (GenericFusion
/// branch of `TensorKitSectors/sectors.jl:pentagon_equation`). For every
/// `f ∈ a⊗b`, `h ∈ c⊗d`, `g ∈ f⊗c`, `i ∈ b⊗h`, `e ∈ (g⊗d) ∩ (a⊗i)`:
///
/// ```text
/// p1[λμν κρσ] = Σ_τ F(f,c,d,e,g,h)[λ,μ,ν,τ] · F(a,b,h,e,f,i)[κ,τ,ρ,σ]
/// p2[λμν κρσ] = Σ_{j∈b⊗c, α,β,τ}
///                 F(a,b,c,g,f,j)[κ,λ,α,β] · F(a,j,d,e,g,i)[β,μ,τ,σ]
///                 · F(b,c,d,i,j,h)[α,τ,ν,ρ]
/// ```
///
/// returns `max |p1 - p2|`.
pub fn pentagon_residual<C: FusionCategory>(
    fam: &C,
    a: &C::Object,
    b: &C::Object,
    c: &C::Object,
    d: &C::Object,
) -> Result<f64, C::Error> {
    let mut worst = 0.0f64;
    let mut memo = BlockMemo::<C>::default();

    for f in outcomes(fam, a, b)? {
        for h in outcomes(fam, c, d)? {
            for g in outcomes(fam, &f, c)? {
                for i in outcomes(fam, b, &h)? {
                    for e in intersect_products(fam, &g, d, a, &i)? {
                        // Free-index dims: λ=N_fcg, μ=N_gde, ν=N_cdh, κ=N_abf,
                        // ρ=N_bhi, σ=N_aie.
                        let n_lambda = multiplicity(fam, &f, c, &g)?;
                        let n_mu = multiplicity(fam, &g, d, &e)?;
                        let n_nu = multiplicity(fam, c, d, &h)?;
                        let n_kappa = multiplicity(fam, a, b, &f)?;
                        let n_rho = multiplicity(fam, b, &h, &i)?;
                        let n_sigma = multiplicity(fam, a, &i, &e)?;
                        if [n_lambda, n_mu, n_nu, n_kappa, n_rho, n_sigma].contains(&0) {
                            continue; // empty output family -> vacuous
                        }

                        // p1: F1[λ,μ,ν,τ] · F2[κ,τ,ρ,σ], sum over τ (= N_fhe).
                        let f1 = memo.f_block(fam, &f, c, d, &e, &g, &h)?; // [λ,μ,ν,τ]
                        let f2 = memo.f_block(fam, a, b, &h, &e, &f, &i)?; // [κ,τ,ρ,σ]
                        let n_tau = f1.dims()[3];

                        // p2 factors, summed over j ∈ b⊗c and α,β,τ'.
                        let mut p2_terms = Vec::new();
                        for j in outcomes(fam, b, c)? {
                            let g1 = memo.f_block(fam, a, b, c, &g, &f, &j)?; // [κ,λ,α,β]
                            let g2 = memo.f_block(fam, a, &j, d, &e, &g, &i)?; // [β,μ,τ',σ]
                            let g3 = memo.f_block(fam, b, c, d, &i, &j, &h)?; // [α,τ',ν,ρ]
                            p2_terms.push((g1, g2, g3));
                        }

                        for lambda in 0..n_lambda {
                            for mu in 0..n_mu {
                                for nu in 0..n_nu {
                                    for kappa in 0..n_kappa {
                                        for rho in 0..n_rho {
                                            for sigma in 0..n_sigma {
                                                let mut p1 = C::Scalar::zero();
                                                for tau in 0..n_tau {
                                                    p1 = p1
                                                        + f1.get([lambda, mu, nu, tau])
                                                            * f2.get([kappa, tau, rho, sigma]);
                                                }
                                                let mut p2 = C::Scalar::zero();
                                                for (g1, g2, g3) in &p2_terms {
                                                    // dims: α=g1[2], β=g1[3], τ'=g2[2]
                                                    let n_alpha = g1.dims()[2];
                                                    let n_beta = g1.dims()[3];
                                                    let n_taup = g2.dims()[2];
                                                    for alpha in 0..n_alpha {
                                                        for beta in 0..n_beta {
                                                            for taup in 0..n_taup {
                                                                p2 = p2
                                                                    + g1.get([
                                                                        kappa, lambda, alpha, beta,
                                                                    ]) * g2.get([
                                                                        beta, mu, taup, sigma,
                                                                    ]) * g3.get([
                                                                        alpha, taup, nu, rho,
                                                                    ]);
                                                            }
                                                        }
                                                    }
                                                }
                                                worst = worst.max((p1 - p2).abs());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(worst)
}

// ---------------------------------------------------------------------------
// Gate 3: hexagon (TensorKitSectors sectors.jl:hexagon_equation, :834-871).
// ---------------------------------------------------------------------------

/// The worst residual of both hexagon identities for the triple `(a, b, c)`
/// (GenericFusion branch of `TensorKitSectors/sectors.jl:hexagon_equation`).
/// For every `e ∈ c⊗a`, `f ∈ c⊗b`, `d ∈ (e⊗b) ∩ (a⊗f)`, with
/// `F ≡ F(a,c,b,d,e,f)[λ,β,γ,ν]`:
///
/// ```text
/// RFR1[α,β,μ,ν] = Σ_{λ,γ} R(c,a,e)[α,λ] · F[λ,β,γ,ν] · R(c,b,f)[γ,μ]
/// RFR2[α,β,μ,ν] = Σ_{λ,γ} R(a,c,e)†[α,λ] · F[λ,β,γ,ν] · R(b,c,f)†[γ,μ]
/// FRF1[α,β,μ,ν] = Σ_{g∈a⊗b, δ,σ,ψ}
///                   F(c,a,b,d,e,g)[α,β,δ,σ] · R(c,g,d)[σ,ψ] · F(a,b,c,d,g,f)[δ,ψ,μ,ν]
/// FRF2[α,β,μ,ν] = Σ ... R(g,c,d)†[σ,ψ] ...
/// ```
///
/// with `R(x,y,z)†[σ,ψ] = conj(R(x,y,z)[ψ,σ])`, and returns
/// `max(|RFR1 - FRF1|, |RFR2 - FRF2|)`. The second hexagon uses the inverse
/// braiding, the adjoint of the R block. TensorKitSectors writes it as the
/// elementwise `conj(R[σ, ψ])`, which pairs a `(y,x,z)` multiplicity index
/// with an `(x,y,z)` one; that is the adjoint only when the block is
/// symmetric, so it fails for `N ≥ 2` channels in general multiplicity bases
/// (for example SU(3) `27`, where `R(27, 8, 27)` is not symmetric).
pub fn hexagon_residual<C: FusionCategory>(
    fam: &C,
    a: &C::Object,
    b: &C::Object,
    c: &C::Object,
) -> Result<f64, C::Error> {
    let mut worst = 0.0f64;
    let mut memo = BlockMemo::<C>::default();

    for e in outcomes(fam, c, a)? {
        let rcae = memo.r_block(fam, c, a, &e)?; // [α,λ]
        let race = memo.r_block(fam, a, c, &e)?; // [α,λ]
        for f in outcomes(fam, c, b)? {
            let rcbf = memo.r_block(fam, c, b, &f)?; // [γ,μ]
            let rbcf = memo.r_block(fam, b, c, &f)?; // [γ,μ]
            for d in intersect_products(fam, &e, b, a, &f)? {
                // free dims: α=N_cae, β=N_ebd, μ=N_bcf, ν=N_afd.
                let n_alpha = multiplicity(fam, c, a, &e)?;
                let n_beta = multiplicity(fam, &e, b, &d)?;
                let n_mu = multiplicity(fam, b, c, &f)?;
                let n_nu = multiplicity(fam, a, &f, &d)?;
                if [n_alpha, n_beta, n_mu, n_nu].contains(&0) {
                    continue;
                }
                let facb = memo.f_block(fam, a, c, b, &d, &e, &f)?; // [λ,β,γ,ν]
                let n_lam = facb.dims()[0]; // N_ace = N_cae
                let n_gam = facb.dims()[2]; // N_cbf

                // FRF factors over g ∈ a⊗b.
                let mut frf_terms = Vec::new();
                for g in outcomes(fam, a, b)? {
                    let rcgd = memo.r_block(fam, c, &g, &d)?;
                    let rgcd = memo.r_block(fam, &g, c, &d)?;
                    let fcab = memo.f_block(fam, c, a, b, &d, &e, &g)?; // [α,β,δ,σ]
                    let fabc = memo.f_block(fam, a, b, c, &d, &g, &f)?; // [δ,ψ,μ,ν]
                    frf_terms.push((fcab, rcgd, rgcd, fabc));
                }

                for alpha in 0..n_alpha {
                    for beta in 0..n_beta {
                        for mu in 0..n_mu {
                            for nu in 0..n_nu {
                                // RFR1 / RFR2.
                                let mut rfr1 = C::Scalar::zero();
                                let mut rfr2 = C::Scalar::zero();
                                for lam in 0..n_lam {
                                    for gam in 0..n_gam {
                                        let fv = facb.get([lam, beta, gam, nu]);
                                        rfr1 = rfr1
                                            + rcae.get([alpha, lam]) * fv * rcbf.get([gam, mu]);
                                        rfr2 = rfr2
                                            + race.get([lam, alpha]).conj()
                                                * fv
                                                * rbcf.get([mu, gam]).conj();
                                    }
                                }
                                // FRF1 / FRF2.
                                let mut frf1 = C::Scalar::zero();
                                let mut frf2 = C::Scalar::zero();
                                for (fcab, rcgd, rgcd, fabc) in &frf_terms {
                                    let n_delta = fcab.dims()[2]; // N_abg
                                    let n_sigma = fcab.dims()[3]; // N_cgd
                                    let n_psi = rcgd.dim(); // N_gcd (= N_cgd)
                                    for delta in 0..n_delta {
                                        for sigma in 0..n_sigma {
                                            let fc = fcab.get([alpha, beta, delta, sigma]);
                                            for psi in 0..n_psi {
                                                let fa = fabc.get([delta, psi, mu, nu]);
                                                frf1 = frf1 + fc * rcgd.get([sigma, psi]) * fa;
                                                frf2 =
                                                    frf2 + fc * rgcd.get([psi, sigma]).conj() * fa;
                                            }
                                        }
                                    }
                                }
                                worst = worst.max((rfr1 - frf1).abs());
                                worst = worst.max((rfr2 - frf2).abs());
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(worst)
}
