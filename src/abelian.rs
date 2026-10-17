//! Abelian groups in closed form: `Z2`, the cyclic groups `Z_N`, `U(1)`, and
//! fermion parity `fZ2`.
//!
//! Every value follows TensorKitSectors.jl 0.3.9 (git `67d342a1`):
//! `src/irreps/irreps.jl` (`AbelianIrrep`: `Nsymbol`, `Fsymbol`, `Rsymbol`,
//! `frobenius_schur_phase`), `src/irreps/znirrep.jl` (`ZNIrrep`),
//! `src/irreps/u1irrep.jl` (`U1Irrep`) and `src/fermions.jl`
//! (`FermionParity`), with the generic `src/sectors.jl` `dim` and `twist`
//! derivations for what those types do not override.
//!
//! All four categories have unique fusion and invertible objects of
//! dimension 1. F is the product of the four vertex multiplicities, so `1`
//! when admissible and `0` otherwise; R equals `N^{ab}_c`, except for fermion
//! parity, where `R^{ab}_c = -1` when `a` and `b` are both odd. The twist is
//! `1`, except `θ = -1` for odd fermion parity. The Frobenius–Schur phase is
//! `1` throughout.
//!
//! Scalars are `f64` (exact `±1` and `0`). TensorKitSectors declares `Int`
//! here; `f64` is the scalar type of the providers these were ported from
//! (symtn `symtn-sectors/src/abelian.rs`), whose values they reproduce bit for
//! bit on every admissible label tuple (`tests/symtn_abelian_values.rs`).
//! Unlike that port, an inadmissible F or R is `0.0`, as in TensorKitSectors
//! and [`crate::tables`].
//!
//! Label domains: [`ZNIrrep`] carries its modulus `N ≥ 1` (`u32`) at runtime;
//! [`U1Irrep`] holds integer charges in `i32` except `i32::MIN`, whose dual
//! is not representable. TensorKitSectors also admits half-integer U(1)
//! charges; those are not representable here.

use std::num::NonZeroU32;

/// An inadmissible or ill-posed abelian request.
///
/// `L` is the label type of the family. [`AbelianError::ModulusMismatch`] and
/// [`AbelianError::ZeroModulus`] occur only for [`ZNIrrep`],
/// [`AbelianError::U1ChargeOutOfRange`] only for [`U1Irrep`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum AbelianError<L> {
    /// The vertex `left ⊗ right → coupled` has `N = 0`. For an F-symbol this
    /// is the first failing vertex in the order `(a, b, e)`, `(e, c, d)`,
    /// `(b, c, f)`, `(a, f, d)`.
    NotAdmissible {
        /// First label of the vertex.
        left: L,
        /// Second label of the vertex.
        right: L,
        /// Coupled label of the vertex.
        coupled: L,
    },
    /// Two `Z_N` labels of different moduli in one request.
    ModulusMismatch {
        /// Modulus of the first label of the failing pair.
        left: u32,
        /// Modulus of the second label of the failing pair.
        right: u32,
    },
    /// A `Z_N` modulus of zero.
    ZeroModulus,
    /// A U(1) charge outside `i32::MIN + 1 ..= i32::MAX`, either requested
    /// directly or produced by [`U1Irrep::fusion`].
    U1ChargeOutOfRange {
        /// The exact charge.
        charge: i64,
    },
}

impl<L: std::fmt::Debug> std::fmt::Display for AbelianError<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbelianError::NotAdmissible {
                left,
                right,
                coupled,
            } => write!(
                f,
                "inadmissible abelian vertex: {left:?} ⊗ {right:?} does not contain {coupled:?}"
            ),
            AbelianError::ModulusMismatch { left, right } => {
                write!(f, "Z_N labels of different moduli: {left} and {right}")
            }
            AbelianError::ZeroModulus => write!(f, "Z_N modulus must be nonzero"),
            AbelianError::U1ChargeOutOfRange { charge } => write!(
                f,
                "U(1) charge {charge} is outside i32::MIN + 1 ..= i32::MAX"
            ),
        }
    }
}

impl<L: std::fmt::Debug> std::error::Error for AbelianError<L> {}

/// Shared vertex test, so the four families run one F/R gate. Private: the
/// public surface stays per family (crate root `ponytail` note).
trait Vertex: Copy {
    /// `N^{left right}_coupled == 1`, or a typed error for an ill-posed tuple.
    fn admits(left: Self, right: Self, coupled: Self) -> Result<bool, AbelianError<Self>>;
}

fn check_vertex<L: Vertex>(left: L, right: L, coupled: L) -> Result<(), AbelianError<L>> {
    if L::admits(left, right, coupled)? {
        Ok(())
    } else {
        Err(AbelianError::NotAdmissible {
            left,
            right,
            coupled,
        })
    }
}

fn nsymbol<L: Vertex>(a: L, b: L, c: L) -> u32 {
    u32::from(L::admits(a, b, c).unwrap_or(false))
}

fn f_symbol_checked<L: Vertex>(a: L, b: L, c: L, d: L, e: L, f: L) -> Result<f64, AbelianError<L>> {
    check_vertex(a, b, e)?;
    check_vertex(e, c, d)?;
    check_vertex(b, c, f)?;
    check_vertex(a, f, d)?;
    Ok(1.0)
}

fn r_symbol_checked<L: Vertex>(a: L, b: L, c: L) -> Result<f64, AbelianError<L>> {
    check_vertex(a, b, c)?;
    Ok(1.0)
}

// ---------------------------------------------------------------- Z2 -----

/// An irrep of `Z2`: even (`0`, the unit) or odd (`1`).
///
/// TensorKitSectors `Z2Irrep = ZNIrrep{2}`; ordered unit first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Z2Irrep {
    odd: bool,
}

impl Z2Irrep {
    /// The unit, parity `0`.
    pub const EVEN: Self = Self { odd: false };
    /// Parity `1`.
    pub const ODD: Self = Self { odd: true };

    /// The irrep of the given parity.
    pub const fn new(odd: bool) -> Self {
        Self { odd }
    }

    /// Whether this is the odd irrep.
    pub const fn is_odd(self) -> bool {
        self.odd
    }

    /// Quantum dimension, `1`.
    pub fn dim(self) -> f64 {
        1.0
    }

    /// The dual irrep: every `Z2` irrep is self-dual.
    pub fn dual(self) -> Self {
        self
    }

    /// Frobenius–Schur phase, `1` (`irreps.jl:frobenius_schur_phase`).
    pub fn frobenius_schur(self) -> f64 {
        1.0
    }

    /// Twist, `1` (bosonic).
    pub fn twist(self) -> f64 {
        1.0
    }

    /// The unique fusion channel: parities add mod 2.
    pub fn fusion(self, other: Self) -> Self {
        Self::new(self.odd ^ other.odd)
    }
}

impl Vertex for Z2Irrep {
    fn admits(a: Self, b: Self, c: Self) -> Result<bool, AbelianError<Self>> {
        Ok(a.fusion(b) == c)
    }
}

/// Fusion multiplicity `N^{ab}_c ∈ {0, 1}`.
pub fn z2_nsymbol(a: Z2Irrep, b: Z2Irrep, c: Z2Irrep) -> u32 {
    nsymbol(a, b, c)
}

/// F-symbol `F^{abc}_d[e, f]`: `1.0` when admissible, exact `0.0` otherwise.
pub fn z2_f_symbol(a: Z2Irrep, b: Z2Irrep, c: Z2Irrep, d: Z2Irrep, e: Z2Irrep, f: Z2Irrep) -> f64 {
    z2_f_symbol_checked(a, b, c, d, e, f).unwrap_or(0.0)
}

/// Checked [`z2_f_symbol`]: [`AbelianError::NotAdmissible`] instead of a zero.
pub fn z2_f_symbol_checked(
    a: Z2Irrep,
    b: Z2Irrep,
    c: Z2Irrep,
    d: Z2Irrep,
    e: Z2Irrep,
    f: Z2Irrep,
) -> Result<f64, AbelianError<Z2Irrep>> {
    f_symbol_checked(a, b, c, d, e, f)
}

/// R-symbol `R^{ab}_c`: `1.0` when admissible, exact `0.0` otherwise.
pub fn z2_r_symbol(a: Z2Irrep, b: Z2Irrep, c: Z2Irrep) -> f64 {
    z2_r_symbol_checked(a, b, c).unwrap_or(0.0)
}

/// Checked [`z2_r_symbol`]: [`AbelianError::NotAdmissible`] instead of a zero.
pub fn z2_r_symbol_checked(
    a: Z2Irrep,
    b: Z2Irrep,
    c: Z2Irrep,
) -> Result<f64, AbelianError<Z2Irrep>> {
    r_symbol_checked(a, b, c)
}

/// Opaque authority fingerprint of the `Z2` provider. Same contract as
/// [`crate::su2_authority_fingerprint`]: compare by equality only; the epoch
/// is bumped exactly when a returned value or convention changes.
pub fn z2_authority_fingerprint() -> &'static [u8] {
    b"fusion-symbols:abelian-z2:n=tks-irreps:f=tks-irreps:r=tks-irreps:dim=one:fs=tks-irreps-one:twist=one:scalar=f64:epoch=1"
}

// ---------------------------------------------------------------- Z_N ----

/// An irrep of the cyclic group `Z_N`: a charge `n mod N`, with the modulus
/// `N ≥ 1` carried by the label.
///
/// TensorKitSectors `ZNIrrep{N}` / `LargeZNIrrep{N}` fix `N` in the type; here
/// it is a runtime value, so mixing moduli in one request is the typed error
/// [`AbelianError::ModulusMismatch`]. `Ord` compares the modulus, then the
/// charge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ZNIrrep {
    modulus: NonZeroU32,
    charge: u32,
}

impl ZNIrrep {
    /// The irrep of charge `charge mod modulus` (any integer, reduced as
    /// TensorKitSectors `ZNIrrep{N}(n) = mod(n, N)`).
    ///
    /// # Errors
    ///
    /// [`AbelianError::ZeroModulus`] for `modulus == 0`.
    pub fn new(charge: i64, modulus: u32) -> Result<Self, AbelianError<ZNIrrep>> {
        let modulus = NonZeroU32::new(modulus).ok_or(AbelianError::ZeroModulus)?;
        // rem_euclid of an i64 by a positive u32 lies in 0..modulus.
        let charge = charge.rem_euclid(i64::from(modulus.get())) as u32;
        Ok(Self { modulus, charge })
    }

    /// The unit, charge `0`, of `Z_modulus`.
    ///
    /// # Errors
    ///
    /// [`AbelianError::ZeroModulus`] for `modulus == 0`.
    pub fn unit(modulus: u32) -> Result<Self, AbelianError<ZNIrrep>> {
        Self::new(0, modulus)
    }

    /// The charge, in `0..modulus`.
    pub const fn charge(self) -> u32 {
        self.charge
    }

    /// The modulus `N`.
    pub const fn modulus(self) -> u32 {
        self.modulus.get()
    }

    fn with_charge(self, charge: u64) -> Self {
        Self {
            modulus: self.modulus,
            // Below the u32 modulus.
            charge: (charge % u64::from(self.modulus.get())) as u32,
        }
    }

    /// Quantum dimension, `1`.
    pub fn dim(self) -> f64 {
        1.0
    }

    /// The dual irrep, charge `(N - n) mod N` (`znirrep.jl:dual`).
    pub fn dual(self) -> Self {
        self.with_charge(u64::from(self.modulus.get()) - u64::from(self.charge))
    }

    /// Frobenius–Schur phase, `1` (`irreps.jl:frobenius_schur_phase`).
    pub fn frobenius_schur(self) -> f64 {
        1.0
    }

    /// Twist, `1` (bosonic).
    pub fn twist(self) -> f64 {
        1.0
    }

    /// The unique fusion channel, charge `(a + b) mod N`.
    ///
    /// # Errors
    ///
    /// [`AbelianError::ModulusMismatch`] when the moduli differ.
    pub fn fusion(self, other: Self) -> Result<Self, AbelianError<ZNIrrep>> {
        same_modulus(self, other)?;
        Ok(self.with_charge(u64::from(self.charge) + u64::from(other.charge)))
    }
}

fn same_modulus(left: ZNIrrep, right: ZNIrrep) -> Result<(), AbelianError<ZNIrrep>> {
    if left.modulus == right.modulus {
        Ok(())
    } else {
        Err(AbelianError::ModulusMismatch {
            left: left.modulus(),
            right: right.modulus(),
        })
    }
}

impl Vertex for ZNIrrep {
    fn admits(a: Self, b: Self, c: Self) -> Result<bool, AbelianError<Self>> {
        let fused = a.fusion(b)?;
        same_modulus(a, c)?;
        Ok(fused == c)
    }
}

/// Fusion multiplicity `N^{ab}_c ∈ {0, 1}`; `0` also for mixed moduli.
pub fn zn_nsymbol(a: ZNIrrep, b: ZNIrrep, c: ZNIrrep) -> u32 {
    nsymbol(a, b, c)
}

/// F-symbol `F^{abc}_d[e, f]`: `1.0` when admissible, exact `0.0` otherwise
/// (including mixed moduli).
pub fn zn_f_symbol(a: ZNIrrep, b: ZNIrrep, c: ZNIrrep, d: ZNIrrep, e: ZNIrrep, f: ZNIrrep) -> f64 {
    zn_f_symbol_checked(a, b, c, d, e, f).unwrap_or(0.0)
}

/// Checked [`zn_f_symbol`]: [`AbelianError::NotAdmissible`] or
/// [`AbelianError::ModulusMismatch`] instead of a zero, for the first failing
/// vertex.
pub fn zn_f_symbol_checked(
    a: ZNIrrep,
    b: ZNIrrep,
    c: ZNIrrep,
    d: ZNIrrep,
    e: ZNIrrep,
    f: ZNIrrep,
) -> Result<f64, AbelianError<ZNIrrep>> {
    f_symbol_checked(a, b, c, d, e, f)
}

/// R-symbol `R^{ab}_c`: `1.0` when admissible, exact `0.0` otherwise.
pub fn zn_r_symbol(a: ZNIrrep, b: ZNIrrep, c: ZNIrrep) -> f64 {
    zn_r_symbol_checked(a, b, c).unwrap_or(0.0)
}

/// Checked [`zn_r_symbol`]: [`AbelianError::NotAdmissible`] or
/// [`AbelianError::ModulusMismatch`] instead of a zero.
pub fn zn_r_symbol_checked(
    a: ZNIrrep,
    b: ZNIrrep,
    c: ZNIrrep,
) -> Result<f64, AbelianError<ZNIrrep>> {
    r_symbol_checked(a, b, c)
}

/// Opaque authority fingerprint of the `Z_N` provider (contract as
/// [`z2_authority_fingerprint`]).
pub fn zn_authority_fingerprint() -> &'static [u8] {
    b"fusion-symbols:abelian-zn:labels=u32-runtime-modulus:n=tks-irreps:f=tks-irreps:r=tks-irreps:dim=one:fs=tks-irreps-one:twist=one:scalar=f64:epoch=1"
}

// ---------------------------------------------------------------- U(1) ---

/// An irrep of `U(1)`: an integer charge in `i32::MIN + 1 ..= i32::MAX`.
///
/// `i32::MIN` is excluded so that every label has a representable dual.
/// TensorKitSectors' `U1Irrep` stores a `HalfInt`; half-integer charges are
/// not representable here. `Ord` is numeric charge order, not
/// TensorKitSectors' `isless` (which orders by `|q|` first).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct U1Irrep {
    charge: i32,
}

impl U1Irrep {
    /// The unit, charge `0`.
    pub const UNIT: Self = Self { charge: 0 };

    /// The irrep of the given charge.
    ///
    /// # Errors
    ///
    /// [`AbelianError::U1ChargeOutOfRange`] for `i32::MIN`.
    pub fn new(charge: i32) -> Result<Self, AbelianError<U1Irrep>> {
        Self::from_exact(i64::from(charge))
    }

    fn from_exact(charge: i64) -> Result<Self, AbelianError<U1Irrep>> {
        match i32::try_from(charge) {
            Ok(charge) if charge != i32::MIN => Ok(Self { charge }),
            _ => Err(AbelianError::U1ChargeOutOfRange { charge }),
        }
    }

    /// The charge.
    pub const fn charge(self) -> i32 {
        self.charge
    }

    /// Quantum dimension, `1`.
    pub fn dim(self) -> f64 {
        1.0
    }

    /// The dual irrep, charge `-q` (`u1irrep.jl:dual`); representable because
    /// `i32::MIN` is not a label.
    pub fn dual(self) -> Self {
        Self {
            charge: -self.charge,
        }
    }

    /// Frobenius–Schur phase, `1` (`irreps.jl:frobenius_schur_phase`).
    pub fn frobenius_schur(self) -> f64 {
        1.0
    }

    /// Twist, `1` (bosonic).
    pub fn twist(self) -> f64 {
        1.0
    }

    /// The unique fusion channel, charge `a + b`.
    ///
    /// # Errors
    ///
    /// [`AbelianError::U1ChargeOutOfRange`] when the sum is not a label.
    pub fn fusion(self, other: Self) -> Result<Self, AbelianError<U1Irrep>> {
        Self::from_exact(i64::from(self.charge) + i64::from(other.charge))
    }
}

impl Vertex for U1Irrep {
    // Exact in i64: when `a + b` leaves the label range, no label equals it,
    // so `N = 0` is the answer rather than an overflow.
    fn admits(a: Self, b: Self, c: Self) -> Result<bool, AbelianError<Self>> {
        Ok(i64::from(a.charge) + i64::from(b.charge) == i64::from(c.charge))
    }
}

/// Fusion multiplicity `N^{ab}_c ∈ {0, 1}`, exact for every label triple.
pub fn u1_nsymbol(a: U1Irrep, b: U1Irrep, c: U1Irrep) -> u32 {
    nsymbol(a, b, c)
}

/// F-symbol `F^{abc}_d[e, f]`: `1.0` when admissible, exact `0.0` otherwise.
pub fn u1_f_symbol(a: U1Irrep, b: U1Irrep, c: U1Irrep, d: U1Irrep, e: U1Irrep, f: U1Irrep) -> f64 {
    u1_f_symbol_checked(a, b, c, d, e, f).unwrap_or(0.0)
}

/// Checked [`u1_f_symbol`]: [`AbelianError::NotAdmissible`] instead of a zero.
pub fn u1_f_symbol_checked(
    a: U1Irrep,
    b: U1Irrep,
    c: U1Irrep,
    d: U1Irrep,
    e: U1Irrep,
    f: U1Irrep,
) -> Result<f64, AbelianError<U1Irrep>> {
    f_symbol_checked(a, b, c, d, e, f)
}

/// R-symbol `R^{ab}_c`: `1.0` when admissible, exact `0.0` otherwise.
pub fn u1_r_symbol(a: U1Irrep, b: U1Irrep, c: U1Irrep) -> f64 {
    u1_r_symbol_checked(a, b, c).unwrap_or(0.0)
}

/// Checked [`u1_r_symbol`]: [`AbelianError::NotAdmissible`] instead of a zero.
pub fn u1_r_symbol_checked(
    a: U1Irrep,
    b: U1Irrep,
    c: U1Irrep,
) -> Result<f64, AbelianError<U1Irrep>> {
    r_symbol_checked(a, b, c)
}

/// Opaque authority fingerprint of the `U(1)` provider (contract as
/// [`z2_authority_fingerprint`]).
pub fn u1_authority_fingerprint() -> &'static [u8] {
    b"fusion-symbols:abelian-u1:labels=i32-integer-no-min:n=exact:f=tks-irreps:r=tks-irreps:dim=one:fs=tks-irreps-one:twist=one:scalar=f64:epoch=1"
}

// ---------------------------------------------------------------- fZ2 ----

/// Fermion parity, TensorKitSectors `FermionParity` (`fℤ₂`): even (the unit)
/// or odd, with fermionic braiding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FermionParity {
    odd: bool,
}

impl FermionParity {
    /// The unit, even parity.
    pub const EVEN: Self = Self { odd: false };
    /// Odd parity.
    pub const ODD: Self = Self { odd: true };

    /// The sector of the given parity.
    pub const fn new(odd: bool) -> Self {
        Self { odd }
    }

    /// Whether the parity is odd.
    pub const fn is_odd(self) -> bool {
        self.odd
    }

    /// Quantum dimension, `1` (`fermions.jl:dim`).
    pub fn dim(self) -> f64 {
        1.0
    }

    /// The dual sector: self-dual (`fermions.jl:dual`).
    pub fn dual(self) -> Self {
        self
    }

    /// Frobenius–Schur phase, `1`.
    ///
    /// No `FermionParity` override; the generic
    /// `sectors.jl:frobenius_schur_phase_from_Fsymbol` gives
    /// `sign(F^{aaa}_a[𝟙, 𝟙]) = 1`.
    pub fn frobenius_schur(self) -> f64 {
        1.0
    }

    /// Twist `θ = -1` for odd parity, `1` for even (`fermions.jl:twist`).
    pub fn twist(self) -> f64 {
        if self.odd {
            -1.0
        } else {
            1.0
        }
    }

    /// The unique fusion channel: parities add mod 2.
    pub fn fusion(self, other: Self) -> Self {
        Self::new(self.odd ^ other.odd)
    }
}

impl Vertex for FermionParity {
    fn admits(a: Self, b: Self, c: Self) -> Result<bool, AbelianError<Self>> {
        Ok(a.fusion(b) == c)
    }
}

/// Fusion multiplicity `N^{ab}_c ∈ {0, 1}` (`fermions.jl:Nsymbol`).
pub fn fermion_parity_nsymbol(a: FermionParity, b: FermionParity, c: FermionParity) -> u32 {
    nsymbol(a, b, c)
}

/// F-symbol `F^{abc}_d[e, f]`: `1.0` when admissible, exact `0.0` otherwise
/// (`fermions.jl:Fsymbol`).
pub fn fermion_parity_f_symbol(
    a: FermionParity,
    b: FermionParity,
    c: FermionParity,
    d: FermionParity,
    e: FermionParity,
    f: FermionParity,
) -> f64 {
    fermion_parity_f_symbol_checked(a, b, c, d, e, f).unwrap_or(0.0)
}

/// Checked [`fermion_parity_f_symbol`]: [`AbelianError::NotAdmissible`]
/// instead of a zero.
pub fn fermion_parity_f_symbol_checked(
    a: FermionParity,
    b: FermionParity,
    c: FermionParity,
    d: FermionParity,
    e: FermionParity,
    f: FermionParity,
) -> Result<f64, AbelianError<FermionParity>> {
    f_symbol_checked(a, b, c, d, e, f)
}

/// R-symbol `R^{ab}_c`: `-1.0` when admissible with `a`, `b` both odd, `1.0`
/// otherwise when admissible, exact `0.0` when not (`fermions.jl:Rsymbol`).
pub fn fermion_parity_r_symbol(a: FermionParity, b: FermionParity, c: FermionParity) -> f64 {
    fermion_parity_r_symbol_checked(a, b, c).unwrap_or(0.0)
}

/// Checked [`fermion_parity_r_symbol`]: [`AbelianError::NotAdmissible`]
/// instead of a zero.
pub fn fermion_parity_r_symbol_checked(
    a: FermionParity,
    b: FermionParity,
    c: FermionParity,
) -> Result<f64, AbelianError<FermionParity>> {
    let r = r_symbol_checked(a, b, c)?;
    Ok(if a.odd && b.odd { -r } else { r })
}

/// Opaque authority fingerprint of the fermion-parity provider (contract as
/// [`z2_authority_fingerprint`]).
pub fn fermion_parity_authority_fingerprint() -> &'static [u8] {
    b"fusion-symbols:abelian-fz2:n=tks-fermions:f=tks-fermions:r=tks-fermions-odd-odd-minus:dim=one:fs=tks-fsymbol-sign:twist=tks-fermions:scalar=f64:epoch=1"
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARITIES: [bool; 2] = [false, true];

    fn zn(charge: i64, modulus: u32) -> ZNIrrep {
        ZNIrrep::new(charge, modulus).expect("nonzero modulus")
    }

    fn u1(charge: i32) -> U1Irrep {
        U1Irrep::new(charge).expect("representable charge")
    }

    #[test]
    fn zero_modulus_is_a_typed_error() {
        assert_eq!(ZNIrrep::new(3, 0), Err(AbelianError::ZeroModulus));
        assert_eq!(ZNIrrep::unit(0), Err(AbelianError::ZeroModulus));
    }

    #[test]
    fn zn_reduces_any_integer_charge_mod_n() {
        // What: construction is `mod(n, N)` over the whole i64 range, and the
        // u32 extremes fuse without overflow.
        assert_eq!(zn(-1, 3).charge(), 2);
        assert_eq!(zn(i64::MIN, 7).charge(), i64::MIN.rem_euclid(7) as u32);
        assert_eq!(
            zn(i64::MAX, u32::MAX).charge(),
            (i64::MAX % i64::from(u32::MAX)) as u32
        );
        let edge = zn(i64::from(u32::MAX) - 1, u32::MAX);
        assert_eq!(edge.fusion(edge), Ok(zn(i64::from(u32::MAX) - 2, u32::MAX)));
        assert_eq!(edge.dual(), zn(1, u32::MAX));
        assert_eq!(zn(0, 5).dual(), zn(0, 5));
        assert_eq!(zn(5, 1), ZNIrrep::unit(1).unwrap());
    }

    #[test]
    fn mixed_moduli_are_a_typed_error() {
        let (a, b) = (zn(1, 3), zn(1, 4));
        let mismatch = AbelianError::ModulusMismatch { left: 3, right: 4 };
        assert_eq!(a.fusion(b), Err(mismatch));
        assert_eq!(zn_r_symbol_checked(a, b, zn(2, 3)), Err(mismatch));
        // Coupled label of another modulus: compared against the first label.
        assert_eq!(
            zn_r_symbol_checked(a, zn(1, 3), zn(2, 4)),
            Err(AbelianError::ModulusMismatch { left: 3, right: 4 })
        );
        assert_eq!(zn_nsymbol(a, b, zn(2, 3)), 0);
        assert_eq!(zn_r_symbol(a, b, zn(2, 3)), 0.0);
        let u = zn(0, 3);
        assert_eq!(zn_f_symbol(u, u, u, u, u, zn(0, 4)), 0.0);
        assert_eq!(
            zn_f_symbol_checked(u, u, u, u, u, zn(0, 4)),
            Err(AbelianError::ModulusMismatch { left: 3, right: 4 })
        );
    }

    #[test]
    fn u1_rejects_only_i32_min() {
        assert_eq!(
            U1Irrep::new(i32::MIN),
            Err(AbelianError::U1ChargeOutOfRange {
                charge: i64::from(i32::MIN)
            })
        );
        for q in [i32::MIN + 1, i32::MAX] {
            assert_eq!(u1(q).dual().dual(), u1(q));
            assert_eq!(u1(q).dual().charge(), -q);
        }
    }

    #[test]
    fn u1_fusion_overflow_is_a_typed_error_and_symbols_stay_exact() {
        // What: fusion that must produce a label fails typed exactly where the
        // sum leaves the label range; N/F/R are exact (zero) there.
        let (max, min) = (u1(i32::MAX), u1(i32::MIN + 1));
        assert_eq!(
            max.fusion(u1(1)),
            Err(AbelianError::U1ChargeOutOfRange {
                charge: i64::from(i32::MAX) + 1
            })
        );
        assert_eq!(
            min.fusion(u1(-1)),
            Err(AbelianError::U1ChargeOutOfRange {
                charge: i64::from(i32::MIN)
            })
        );
        assert_eq!(max.fusion(min), Ok(U1Irrep::UNIT));
        assert_eq!(max.fusion(U1Irrep::UNIT), Ok(max));
        for c in [min, U1Irrep::UNIT, max] {
            assert_eq!(u1_nsymbol(max, max, c), 0);
            assert_eq!(u1_r_symbol(max, max, c), 0.0);
            assert_eq!(
                u1_r_symbol_checked(max, max, c),
                Err(AbelianError::NotAdmissible {
                    left: max,
                    right: max,
                    coupled: c
                })
            );
        }
        assert_eq!(u1_nsymbol(max, min, U1Irrep::UNIT), 1);
    }

    #[test]
    fn checked_symbols_name_the_first_failing_vertex() {
        let (e, o) = (Z2Irrep::EVEN, Z2Irrep::ODD);
        assert_eq!(
            z2_r_symbol_checked(o, o, o),
            Err(AbelianError::NotAdmissible {
                left: o,
                right: o,
                coupled: o
            })
        );
        // (a, b, e) = (o, o, e) is fine; (e, c, d) = (e, o, e) is not.
        assert_eq!(
            z2_f_symbol_checked(o, o, o, e, e, e),
            Err(AbelianError::NotAdmissible {
                left: e,
                right: o,
                coupled: e
            })
        );
        // Z_3, (a, b, c, d, e) = (1, 1, 1, 0, 2): the first two vertices hold;
        // f = 1 makes the third, (b, c, f) = (1, 1, 1), fail.
        let z = |q| zn(q, 3);
        assert_eq!(
            zn_f_symbol_checked(z(1), z(1), z(1), z(0), z(2), z(1)),
            Err(AbelianError::NotAdmissible {
                left: z(1),
                right: z(1),
                coupled: z(1)
            })
        );
        assert_eq!(
            zn_f_symbol_checked(z(1), z(1), z(1), z(0), z(2), z(2)),
            Ok(1.0)
        );
        assert_eq!(
            u1_f_symbol_checked(u1(1), u1(1), u1(-1), u1(1), u1(2), u1(0)),
            Ok(1.0)
        );
        assert_eq!(
            u1_f_symbol_checked(u1(1), u1(1), u1(-1), u1(1), u1(2), u1(1)),
            Err(AbelianError::NotAdmissible {
                left: u1(1),
                right: u1(-1),
                coupled: u1(1)
            })
        );
    }

    #[test]
    fn infallible_symbols_are_the_checked_ones_or_zero() {
        for a in PARITIES {
            for b in PARITIES {
                for c in PARITIES {
                    let (x, y, z) = (Z2Irrep::new(a), Z2Irrep::new(b), Z2Irrep::new(c));
                    let checked = z2_r_symbol_checked(x, y, z);
                    assert_eq!(checked.is_ok(), z2_nsymbol(x, y, z) == 1);
                    assert_eq!(checked.unwrap_or(0.0), z2_r_symbol(x, y, z));
                    let (x, y, z) = (
                        FermionParity::new(a),
                        FermionParity::new(b),
                        FermionParity::new(c),
                    );
                    let checked = fermion_parity_r_symbol_checked(x, y, z);
                    assert_eq!(checked.is_ok(), fermion_parity_nsymbol(x, y, z) == 1);
                    assert_eq!(checked.unwrap_or(0.0), fermion_parity_r_symbol(x, y, z));
                }
            }
        }
    }

    #[test]
    fn fermion_twist_is_the_generic_rsymbol_sum() {
        // What: `twist` equals TKS' generic `twist_from_Rsymbol`,
        // `θ_a = Σ_{b ∈ a⊗a} d_b/d_a R^{aa}_b`, on this provider's own R.
        for odd in PARITIES {
            let a = FermionParity::new(odd);
            let b = a.fusion(a);
            assert_eq!(
                a.twist(),
                b.dim() / a.dim() * fermion_parity_r_symbol(a, a, b)
            );
            assert_eq!(
                a.frobenius_schur(),
                fermion_parity_f_symbol(
                    a,
                    a.dual(),
                    a,
                    a,
                    FermionParity::EVEN,
                    FermionParity::EVEN
                )
                .signum()
            );
        }
    }

    /// TensorKitSectors 0.3.9 (git `67d342a1`, Julia 1.11.6), measured with
    /// `tests/fixtures/tks_abelian_oracle.jl`: duals, `dim`,
    /// `frobenius_schur_phase` and `twist` of every Z2, Z3, Z4 and
    /// FermionParity label; every `Rsymbol` that differs from `Nsymbol`
    /// (only fZ2 `R(1, 1, 0) = -1`); the count of nonzero `Fsymbol`s over all
    /// label tuples; and U(1) duals, fusion with charge 1, and two F/R values.
    #[test]
    fn matches_measured_tensorkitsectors_values() {
        // (N, duals by charge, nonzero F count); dim, FS phase, twist are 1.
        let zn_tks: [(u32, &[u32], usize); 3] =
            [(2, &[0, 1], 8), (3, &[0, 2, 1], 27), (4, &[0, 3, 2, 1], 64)];
        for (n, duals, nonzero_f) in zn_tks {
            let labels: Vec<ZNIrrep> = (0..i64::from(n)).map(|q| zn(q, n)).collect();
            for (a, &dual) in labels.iter().zip(duals) {
                assert_eq!(a.dual().charge(), dual);
                assert_eq!((a.dim(), a.frobenius_schur(), a.twist()), (1.0, 1.0, 1.0));
            }
            let mut count = 0;
            for &a in &labels {
                for &b in &labels {
                    for &c in &labels {
                        assert_eq!(zn_r_symbol(a, b, c), f64::from(zn_nsymbol(a, b, c)));
                        for &d in &labels {
                            for &e in &labels {
                                for &f in &labels {
                                    count += usize::from(zn_f_symbol(a, b, c, d, e, f) != 0.0);
                                }
                            }
                        }
                    }
                }
            }
            assert_eq!(count, nonzero_f, "Z{n}");
        }
        let (e, o) = (FermionParity::EVEN, FermionParity::ODD);
        assert_eq!((e.dual(), o.dual()), (e, o));
        assert_eq!((e.twist(), o.twist()), (1.0, -1.0));
        assert_eq!((e.frobenius_schur(), o.frobenius_schur()), (1.0, 1.0));
        let mut count = 0;
        for a in [e, o] {
            for b in [e, o] {
                for c in [e, o] {
                    let n = f64::from(fermion_parity_nsymbol(a, b, c));
                    let tks = if (a, b, c) == (o, o, e) { -1.0 } else { n };
                    assert_eq!(fermion_parity_r_symbol(a, b, c), tks);
                    for d in [e, o] {
                        for x in [e, o] {
                            for y in [e, o] {
                                count +=
                                    usize::from(fermion_parity_f_symbol(a, b, c, d, x, y) != 0.0);
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(count, 8);
        for q in -2..=2 {
            assert_eq!(u1(q).dual(), u1(-q));
            assert_eq!(u1(q).fusion(u1(1)), Ok(u1(q + 1)));
            assert_eq!(
                (u1(q).dim(), u1(q).frobenius_schur(), u1(q).twist()),
                (1.0, 1.0, 1.0)
            );
        }
        assert_eq!(u1_f_symbol(u1(1), u1(1), u1(-1), u1(1), u1(2), u1(0)), 1.0);
        assert_eq!(u1_f_symbol(u1(1), u1(1), u1(-1), u1(1), u1(2), u1(1)), 0.0);
        assert_eq!(u1_r_symbol(u1(1), u1(2), u1(3)), 1.0);
        assert_eq!(u1_r_symbol(u1(1), u1(2), u1(2)), 0.0);
    }

    #[test]
    fn fingerprints_are_pinned_and_distinct() {
        let all = [
            z2_authority_fingerprint(),
            zn_authority_fingerprint(),
            u1_authority_fingerprint(),
            fermion_parity_authority_fingerprint(),
        ];
        assert_eq!(
            all,
            [
                &b"fusion-symbols:abelian-z2:n=tks-irreps:f=tks-irreps:r=tks-irreps:dim=one:fs=tks-irreps-one:twist=one:scalar=f64:epoch=1"[..],
                &b"fusion-symbols:abelian-zn:labels=u32-runtime-modulus:n=tks-irreps:f=tks-irreps:r=tks-irreps:dim=one:fs=tks-irreps-one:twist=one:scalar=f64:epoch=1"[..],
                &b"fusion-symbols:abelian-u1:labels=i32-integer-no-min:n=exact:f=tks-irreps:r=tks-irreps:dim=one:fs=tks-irreps-one:twist=one:scalar=f64:epoch=1"[..],
                &b"fusion-symbols:abelian-fz2:n=tks-fermions:f=tks-fermions:r=tks-fermions-odd-odd-minus:dim=one:fs=tks-fsymbol-sign:twist=tks-fermions:scalar=f64:epoch=1"[..],
            ]
        );
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b);
            }
            assert_ne!(
                *a,
                crate::tables::TableCategory::fibonacci().authority_fingerprint()
            );
        }
    }
}
