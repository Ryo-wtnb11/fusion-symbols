//! The Fibonacci category in closed form: objects `𝟙` and `τ` with
//! `τ ⊗ τ = 𝟙 ⊕ τ`.
//!
//! Every value is a verbatim port of TensorKitSectors.jl 0.3.9
//! `src/anyons.jl` (`FibonacciAnyon`: `dim`, `Nsymbol`, `Fsymbol`, `Rsymbol`)
//! and of its generic `src/sectors.jl` derivations for the quantities
//! `FibonacciAnyon` does not override (`frobenius_schur_phase`, `twist`). The
//! expressions are also exactly those of symtn's
//! `tenet-sectors/src/fibonacci.rs`, so the two agree bit for bit
//! (`tests/tenet_values.rs`). Never "simplify" a sign or phase here without
//! rereading those sources.
//!
//! Scalar types follow TensorKitSectors: `fusionscalartype = Float64` (so F,
//! dimensions and the Frobenius–Schur phase are `f64`) and
//! `braidingscalartype = ComplexF64` (R and twist are [`Complex64`]).

use num_complex::Complex64;

/// A simple object of the Fibonacci category.
///
/// Ordered as TensorKitSectors' `values(FibonacciAnyon)`: the unit first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FibonacciObject {
    /// The tensor unit `𝟙`.
    Vacuum,
    /// The nontrivial anyon `τ`.
    Tau,
}

use FibonacciObject::{Tau, Vacuum};

/// The golden ratio `φ = (1 + √5)/2`, TensorKitSectors'
/// `Float64(MathConstants.golden)`.
fn golden() -> f64 {
    (1.0 + 5.0_f64.sqrt()) / 2.0
}

impl FibonacciObject {
    /// Quantum dimension: `1` for `𝟙`, `φ = (1 + √5)/2` for `τ`
    /// (TensorKitSectors `anyons.jl:dim`).
    pub fn dim(self) -> f64 {
        match self {
            Vacuum => 1.0,
            Tau => golden(),
        }
    }

    /// The dual object. Both objects are self-dual (`anyons.jl:dual`).
    pub fn dual(self) -> Self {
        self
    }

    /// Frobenius–Schur phase `κ_a = sign F^{a ā a}_a[𝟙, 𝟙]`, which is `+1` for
    /// both objects.
    ///
    /// TensorKitSectors has no `FibonacciAnyon` override, so it evaluates the
    /// generic `sectors.jl:frobenius_schur_phase_from_Fsymbol`; for `τ` that
    /// is `sign(F^{τττ}_τ[𝟙, 𝟙]) = sign(1/φ) = +1`.
    pub fn frobenius_schur(self) -> f64 {
        1.0
    }

    /// Twist `θ_𝟙 = 1`, `θ_τ = e^{-4πi/5}`.
    ///
    /// TensorKitSectors has no `FibonacciAnyon` override, so it evaluates the
    /// generic `sectors.jl:twist_from_Rsymbol`,
    /// `θ_a = Σ_{b ∈ a⊗a} d_b/d_a · R^{aa}_b`. For `τ` that is
    /// `(1/φ)·e^{4πi/5} + e^{-3πi/5} = e^{-4πi/5}`.
    pub fn twist(self) -> Complex64 {
        match self {
            Vacuum => Complex64::new(1.0, 0.0),
            Tau => Complex64::from_polar(1.0, std::f64::consts::PI * -4.0 / 5.0),
        }
    }

    /// Fusion channels of `self ⊗ other`, each with multiplicity one:
    /// `𝟙 ⊗ a = a ⊗ 𝟙 = a`, `τ ⊗ τ = 𝟙 ⊕ τ`, unit first as in
    /// TensorKitSectors' `FibonacciAnyonProdIterator`.
    pub fn fusion(self, other: Self) -> &'static [FibonacciObject] {
        match (self, other) {
            (Vacuum, Vacuum) => &[Vacuum],
            (Vacuum, Tau) | (Tau, Vacuum) => &[Tau],
            (Tau, Tau) => &[Vacuum, Tau],
        }
    }
}

/// Fusion multiplicity `N^{ab}_c ∈ {0, 1}`: zero exactly when one label is `τ`
/// and the other two are `𝟙` (`anyons.jl:Nsymbol`).
pub fn fibonacci_nsymbol(a: FibonacciObject, b: FibonacciObject, c: FibonacciObject) -> u32 {
    let units = [a, b, c].iter().filter(|&&x| x == Vacuum).count();
    u32::from(units != 2)
}

/// An inadmissible Fibonacci request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FibonacciError {
    /// The vertex `left ⊗ right → coupled` has `N = 0`. For an F-symbol this
    /// is the first failing vertex in the order `(a, b, e)`, `(e, c, d)`,
    /// `(b, c, f)`, `(a, f, d)`.
    NotAdmissible {
        /// First object of the vertex.
        left: FibonacciObject,
        /// Second object of the vertex.
        right: FibonacciObject,
        /// Coupled object of the vertex.
        coupled: FibonacciObject,
    },
}

impl std::fmt::Display for FibonacciError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FibonacciError::NotAdmissible {
                left,
                right,
                coupled,
            } => write!(
                f,
                "inadmissible Fibonacci vertex: {left:?} ⊗ {right:?} does not contain {coupled:?}"
            ),
        }
    }
}

impl std::error::Error for FibonacciError {}

fn check_vertex(
    left: FibonacciObject,
    right: FibonacciObject,
    coupled: FibonacciObject,
) -> Result<(), FibonacciError> {
    if fibonacci_nsymbol(left, right, coupled) == 0 {
        Err(FibonacciError::NotAdmissible {
            left,
            right,
            coupled,
        })
    } else {
        Ok(())
    }
}

/// F-symbol `F^{abc}_d[e, f]`; exact `0.0` when a vertex is inadmissible.
///
/// Verbatim `anyons.jl:Fsymbol`: four `Nsymbol` gates, then the one
/// nontrivial block `F^{τττ}_τ = [1/φ, 1/√φ; 1/√φ, -1/φ]` (rows `e`, columns
/// `f`, `𝟙` before `τ`); every other admissible entry is `1`.
pub fn fibonacci_f_symbol(
    a: FibonacciObject,
    b: FibonacciObject,
    c: FibonacciObject,
    d: FibonacciObject,
    e: FibonacciObject,
    f: FibonacciObject,
) -> f64 {
    fibonacci_f_symbol_checked(a, b, c, d, e, f).unwrap_or(0.0)
}

/// Checked [`fibonacci_f_symbol`]: [`FibonacciError::NotAdmissible`] instead
/// of a zero for an inadmissible request.
pub fn fibonacci_f_symbol_checked(
    a: FibonacciObject,
    b: FibonacciObject,
    c: FibonacciObject,
    d: FibonacciObject,
    e: FibonacciObject,
    f: FibonacciObject,
) -> Result<f64, FibonacciError> {
    check_vertex(a, b, e)?;
    check_vertex(e, c, d)?;
    check_vertex(b, c, f)?;
    check_vertex(a, f, d)?;
    Ok(if [a, b, c, d] == [Tau; 4] {
        let phi = golden();
        match (e, f) {
            (Vacuum, Vacuum) => 1.0 / phi,
            (Tau, Tau) => -1.0 / phi,
            _ => 1.0 / phi.sqrt(),
        }
    } else {
        1.0
    })
}

/// R-symbol `R^{ab}_c`; exact zero when `N^{ab}_c = 0`.
///
/// Verbatim `anyons.jl:Rsymbol`: `1` when either input is `𝟙`,
/// `R^{ττ}_𝟙 = e^{4πi/5}`, `R^{ττ}_τ = e^{-3πi/5}`.
pub fn fibonacci_r_symbol(a: FibonacciObject, b: FibonacciObject, c: FibonacciObject) -> Complex64 {
    fibonacci_r_symbol_checked(a, b, c).unwrap_or(Complex64::new(0.0, 0.0))
}

/// Checked [`fibonacci_r_symbol`]: [`FibonacciError::NotAdmissible`] instead
/// of a zero for an inadmissible request.
pub fn fibonacci_r_symbol_checked(
    a: FibonacciObject,
    b: FibonacciObject,
    c: FibonacciObject,
) -> Result<Complex64, FibonacciError> {
    check_vertex(a, b, c)?;
    Ok(match (a, b, c) {
        (Tau, Tau, Vacuum) => Complex64::from_polar(1.0, std::f64::consts::PI * 4.0 / 5.0),
        (Tau, Tau, Tau) => Complex64::from_polar(1.0, std::f64::consts::PI * -3.0 / 5.0),
        _ => Complex64::new(1.0, 0.0),
    })
}

/// Opaque authority fingerprint of the closed-form Fibonacci provider: the
/// version of the convention set its values are computed in.
///
/// Same contract as [`crate::su2_authority_fingerprint`]: compare by equality
/// only, never parse; the trailing epoch is bumped by hand exactly when a
/// returned value, normalization or convention changes. It deliberately
/// differs from [`crate::category_data::CategoryDataFibonacci::authority_fingerprint`]:
/// the two providers return different floating-point numbers.
pub fn fibonacci_authority_fingerprint() -> &'static [u8] {
    b"fusion-symbols:fibonacci-closed-form:n=tks-anyons:f=tks-anyons:r=tks-anyons:dim=tks-golden:fs=tks-fsymbol-sign:twist=tks-rsymbol-sum:epoch=1"
}

#[cfg(test)]
mod tests {
    use super::*;

    const OBJECTS: [FibonacciObject; 2] = [Vacuum, Tau];

    fn cispi(x: f64) -> Complex64 {
        Complex64::from_polar(1.0, std::f64::consts::PI * x)
    }

    /// The closed-form irrationals admit more than one evaluation order, so
    /// they meet the TensorKitSectors closed forms within a few ulp, not bit
    /// for bit. Bit identity with the ported implementation is
    /// `tests/tenet_values.rs`.
    fn assert_close(what: &str, actual: Complex64, expected: Complex64) {
        let error = (actual - expected).norm();
        assert!(
            error <= 4.0 * f64::EPSILON,
            "{what}: {actual} vs {expected}"
        );
    }

    #[test]
    fn matches_tensorkitsectors_closed_forms() {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        assert_eq!(Vacuum.fusion(Tau), &[Tau]);
        assert_eq!(Tau.fusion(Vacuum), &[Tau]);
        assert_eq!(Tau.fusion(Tau), &[Vacuum, Tau]);
        assert_eq!(fibonacci_nsymbol(Vacuum, Vacuum, Tau), 0);
        for (e, f, expected) in [
            (Vacuum, Vacuum, 1.0 / phi),
            (Vacuum, Tau, 1.0 / phi.sqrt()),
            (Tau, Vacuum, 1.0 / phi.sqrt()),
            (Tau, Tau, -1.0 / phi),
        ] {
            assert_close(
                "F(τ,τ,τ,τ)",
                fibonacci_f_symbol(Tau, Tau, Tau, Tau, e, f).into(),
                expected.into(),
            );
        }
        assert_eq!(fibonacci_f_symbol(Vacuum, Tau, Tau, Tau, Tau, Tau), 1.0);
        assert_eq!(fibonacci_f_symbol(Tau, Vacuum, Tau, Tau, Tau, Tau), 1.0);
        assert_eq!(
            fibonacci_f_symbol(Vacuum, Vacuum, Vacuum, Tau, Vacuum, Vacuum),
            0.0
        );
        assert_close("R(τ,τ,1)", fibonacci_r_symbol(Tau, Tau, Vacuum), cispi(0.8));
        assert_close("R(τ,τ,τ)", fibonacci_r_symbol(Tau, Tau, Tau), cispi(-0.6));
        assert_close("dim(τ)", Tau.dim().into(), phi.into());
        assert_close("twist(τ)", Tau.twist(), cispi(-0.8));
        assert_eq!(Vacuum.frobenius_schur(), 1.0);
        assert_eq!(Tau.frobenius_schur(), 1.0);
    }

    #[test]
    fn derived_quantities_follow_the_generic_tensorkitsectors_formulas() {
        // What: `frobenius_schur` and `twist` are the TKS generic
        // derivations evaluated on this provider's own F and R.
        for a in OBJECTS {
            let unit_f = fibonacci_f_symbol(a, a.dual(), a, a, Vacuum, Vacuum);
            assert_eq!(a.frobenius_schur(), unit_f.signum());
            assert_close("dim", a.dim().into(), (1.0 / unit_f).abs().into());
            let twist: Complex64 = a
                .fusion(a)
                .iter()
                .map(|&b| fibonacci_r_symbol(a, a, b) * (b.dim() / a.dim()))
                .sum();
            assert_close("twist", a.twist(), twist);
        }
    }

    #[test]
    fn unit_channels_and_associators_are_canonical() {
        // What: the vacuum is a unique self-dual unit, and the unitor
        // associators are exactly one.
        assert_eq!(Vacuum.dual(), Vacuum);
        assert_eq!(fibonacci_nsymbol(Vacuum, Tau, Tau), 1);
        assert_eq!(fibonacci_nsymbol(Tau, Vacuum, Tau), 1);
        assert_eq!(fibonacci_nsymbol(Vacuum, Tau, Vacuum), 0);
        for &fused in Tau.fusion(Tau) {
            assert_eq!(fibonacci_f_symbol(Vacuum, Tau, Tau, fused, Tau, fused), 1.0);
            assert_eq!(fibonacci_f_symbol(Tau, Vacuum, Tau, fused, Tau, Tau), 1.0);
            assert_eq!(fibonacci_f_symbol(Tau, Tau, Vacuum, fused, fused, Tau), 1.0);
        }
    }

    #[test]
    fn checked_symbols_name_the_first_failing_vertex() {
        assert_eq!(
            fibonacci_r_symbol_checked(Vacuum, Vacuum, Tau),
            Err(FibonacciError::NotAdmissible {
                left: Vacuum,
                right: Vacuum,
                coupled: Tau
            })
        );
        // (a, b, e) = (τ, τ, τ) is fine; (e, c, d) = (τ, 𝟙, 𝟙) is not.
        assert_eq!(
            fibonacci_f_symbol_checked(Tau, Tau, Vacuum, Vacuum, Tau, Tau),
            Err(FibonacciError::NotAdmissible {
                left: Tau,
                right: Vacuum,
                coupled: Vacuum
            })
        );
        for a in OBJECTS {
            for b in OBJECTS {
                for c in OBJECTS {
                    let checked = fibonacci_r_symbol_checked(a, b, c);
                    assert_eq!(checked.is_ok(), fibonacci_nsymbol(a, b, c) == 1);
                    assert_eq!(
                        checked.unwrap_or_default(),
                        fibonacci_r_symbol(a, b, c),
                        "checked and infallible R agree"
                    );
                }
            }
        }
    }
}
