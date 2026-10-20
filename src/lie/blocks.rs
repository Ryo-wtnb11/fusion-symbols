//! Dense symbol storage, including inline multiplicity-free coefficients.
use crate::category::SymbolBlock;
use std::sync::Arc;

#[derive(Clone, Debug)]
enum Storage {
    One(f64),
    Shared(Arc<[f64]>),
}
impl Storage {
    fn as_slice(&self) -> &[f64] {
        match self {
            Self::One(x) => std::slice::from_ref(x),
            Self::Shared(x) => x,
        }
    }
}

/// A dense F-symbol block $F^{abc}_d[e, f]$, a rank-4 array over the outer
/// multiplicity indices $[\mu, \nu, \kappa, \lambda]$ in **row-major** order.
///
/// One axis per vertex of the recoupling move:
///
/// | Axis | Index | Vertex | Length |
/// |---|---|---|---|
/// | 0 | $\mu$ | $a \otimes b \to e$ | $N^e_{ab}$ |
/// | 1 | $\nu$ | $e \otimes c \to d$ | $N^d_{ec}$ |
/// | 2 | $\kappa$ | $b \otimes c \to f$ | $N^f_{bc}$ |
/// | 3 | $\lambda$ | $a \otimes f \to d$ | $N^d_{af}$ |
///
/// The axis order matches the TensorKitSectors `GenericFusion` convention
/// (`sectors.jl:Fsymbol_from_fusiontensor`), so a block can be handed to a
/// consumer expecting that layout without a permutation. Axes 0–1 belong to the
/// left coupling tree and 2–3 to the right one; they are *not* interchangeable.
/// For a multiplicity-free family (e.g. SU(2)) every axis is length 1, so the
/// block holds the single scalar at `at(0, 0, 0, 0)`.
///
/// Row-major means the flat index of $(\mu,\nu,\kappa,\lambda)$ is
/// $((\mu \cdot d_1 + \nu) \cdot d_2 + \kappa) \cdot d_3 + \lambda$ for
/// `dims = [d_0, d_1, d_2, d_3]`.
///
/// ```
/// # #[cfg(feature = "cgc-gen")] {
/// use fusion_symbols::{category::FusionCategory, lie::{LieGroup, GroupId}};
/// let cat = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
///
/// let three = cat.irrep(&[1, 0]).unwrap();
/// let eight = cat.irrep(&[1, 1]).unwrap();
/// let block = cat.f_symbol(&three, &cat.dual(&three), &three, &three, &eight, &eight).unwrap();
///
/// assert_eq!(block.dims(), [1, 1, 1, 1]);
/// assert_eq!(block.data().len(), 1);
/// assert_eq!(block.at(0, 0, 0, 0), block.data()[0]);
/// # }
/// ```
///
/// Cloning is cheap: the coefficient buffer is shared (`Arc<[f64]>`), so a
/// cache hit hands out the cached block without copying its data.
#[derive(Clone, Debug)]
pub struct FBlock {
    dims: [usize; 4],
    /// Row-major over `[μ, ν, κ, λ]`.
    data: Storage,
}

impl FBlock {
    /// A single coefficient, stored inline without allocation.
    pub fn one(value: f64) -> Self {
        Self {
            dims: [1; 4],
            data: Storage::One(value),
        }
    }

    pub(crate) fn zeros(dims: [usize; 4]) -> Self {
        if dims.contains(&0) {
            Self {
                dims,
                data: Storage::One(0.0),
            }
        } else {
            Self::from_vec(dims, vec![0.0; dims[0] * dims[1] * dims[2] * dims[3]])
        }
    }

    pub(crate) fn from_vec(dims: [usize; 4], data: Vec<f64>) -> Self {
        FBlock {
            dims,
            data: Storage::Shared(data.into()),
        }
    }

    #[inline]
    pub(crate) fn flat(
        dims: [usize; 4],
        mu: usize,
        nu: usize,
        kappa: usize,
        lambda: usize,
    ) -> usize {
        ((mu * dims[1] + nu) * dims[2] + kappa) * dims[3] + lambda
    }

    /// The axis lengths $[N^e_{ab}, N^d_{ec}, N^f_{bc}, N^d_{af}]$ ($\mu,\nu,\kappa,\lambda$).
    pub fn dims(&self) -> [usize; 4] {
        self.dims
    }

    /// The row-major `[μ, ν, κ, λ]` coefficient data.
    pub fn data(&self) -> &[f64] {
        if self.dims.contains(&0) {
            &[]
        } else {
            self.data.as_slice()
        }
    }

    /// The coefficient at multiplicity indices `(μ, ν, κ, λ)`.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of range for [`FBlock::dims`].
    pub fn at(&self, mu: usize, nu: usize, kappa: usize, lambda: usize) -> f64 {
        assert!(
            mu < self.dims[0] && nu < self.dims[1] && kappa < self.dims[2] && lambda < self.dims[3],
            "FBlock index out of range"
        );
        self.data.as_slice()[Self::flat(self.dims, mu, nu, kappa, lambda)]
    }
}

/// A dense R-symbol block $R^{ab}_c$, an $N^c_{ab} \times N^c_{ba}$ matrix in
/// **row-major** order (rows = $\mu$, cols = $\nu$).
///
/// $N^c_{ab} = N^c_{ba}$ (fusion multiplicities are symmetric), so the matrix
/// is square. For a multiplicity-free family it is the single braiding phase.
#[derive(Clone, Debug)]
pub struct RBlock {
    n: usize,
    /// Row-major `n × n`.
    data: Storage,
}

impl RBlock {
    /// A single coefficient, stored inline without allocation.
    pub fn one(value: f64) -> Self {
        Self {
            n: 1,
            data: Storage::One(value),
        }
    }

    #[cfg(feature = "cgc-gen")]
    pub(crate) fn from_vec(n: usize, data: Vec<f64>) -> Self {
        Self {
            n,
            data: Storage::Shared(data.into()),
        }
    }

    pub(crate) fn zeros(n: usize) -> Self {
        RBlock {
            n,
            data: if n == 0 {
                Storage::One(0.0)
            } else {
                Storage::Shared(vec![0.0; n * n].into())
            },
        }
    }

    /// The multiplicity `N^c_{ab}` (both the row and column count).
    pub fn dim(&self) -> usize {
        self.n
    }

    /// The row-major coefficient data (`N × N`, rows `μ`, cols `ν`).
    pub fn data(&self) -> &[f64] {
        if self.n == 0 {
            &[]
        } else {
            self.data.as_slice()
        }
    }

    /// The braiding coefficient at `(μ, ν)`.
    ///
    /// # Panics
    ///
    /// Panics if `mu` or `nu` is `>= dim()`.
    pub fn at(&self, mu: usize, nu: usize) -> f64 {
        assert!(mu < self.n && nu < self.n, "RBlock index out of range");
        self.data.as_slice()[mu * self.n + nu]
    }
}

impl SymbolBlock<4> for FBlock {
    type Scalar = f64;
    fn dims(&self) -> [usize; 4] {
        self.dims()
    }
    fn get(&self, [mu, nu, kappa, lambda]: [usize; 4]) -> f64 {
        self.at(mu, nu, kappa, lambda)
    }
    fn as_slice(&self) -> &[f64] {
        self.data()
    }
}
impl SymbolBlock<2> for RBlock {
    type Scalar = f64;
    fn dims(&self) -> [usize; 2] {
        [self.dim(); 2]
    }
    fn get(&self, [mu, nu]: [usize; 2]) -> f64 {
        self.at(mu, nu)
    }
    fn as_slice(&self) -> &[f64] {
        self.data()
    }
}

impl PartialEq for FBlock {
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims && self.data() == other.data()
    }
}
impl PartialEq for RBlock {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.data() == other.data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equality_compares_values_not_storage() {
        assert_eq!(FBlock::one(0.5), FBlock::from_vec([1; 4], vec![0.5]));
        assert_eq!(
            FBlock::zeros([0, 1, 1, 1]),
            FBlock::from_vec([0, 1, 1, 1], vec![])
        );
        #[cfg(feature = "cgc-gen")]
        assert_eq!(RBlock::one(-1.0), RBlock::from_vec(1, vec![-1.0]));
    }
}
