//! Dense symbol storage, including inline multiplicity-free coefficients.
use super::{Scalar, SymbolBlock};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) enum Storage<S> {
    Empty,
    One(S),
    Shared(Arc<[S]>),
}
impl<S> Storage<S> {
    pub(crate) fn as_slice(&self) -> &[S] {
        match self {
            Self::Empty => &[],
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
/// # #[cfg(feature = "lie-generated")] {
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
/// Cloning is cheap: the coefficient buffer is shared (`Arc<[S]>`), so a
/// cache hit hands out the cached block without copying its data.
#[derive(Clone, Debug)]
pub struct FBlock<S = f64> {
    data: FStorage<S>,
}

#[derive(Clone, Debug)]
enum FStorage<S> {
    One(S),
    Empty([usize; 4]),
    // Nonempty row-major data determines the last axis; no extra allocation.
    Shared([usize; 3], Arc<[S]>),
}

impl<S: Scalar> FBlock<S> {
    /// A single coefficient, stored inline without allocation.
    pub fn one(value: S) -> Self {
        Self {
            data: FStorage::One(value),
        }
    }

    pub(crate) fn zeros(dims: [usize; 4]) -> Self {
        if dims.contains(&0) {
            Self {
                data: FStorage::Empty(dims),
            }
        } else {
            Self::from_vec(dims, vec![S::zero(); dims.iter().product()])
        }
    }

    pub(crate) fn from_vec(dims: [usize; 4], data: Vec<S>) -> Self {
        let data = match data.len() {
            0 => Storage::Empty,
            1 => Storage::One(data[0]),
            _ => Storage::Shared(data.into()),
        };
        Self::from_storage(dims, data)
    }

    pub(crate) fn from_storage(dims: [usize; 4], data: Storage<S>) -> Self {
        debug_assert_eq!(
            data.as_slice().len(),
            if dims.contains(&0) {
                0
            } else {
                dims.iter().product()
            }
        );
        Self {
            data: match data {
                Storage::Empty => FStorage::Empty(dims),
                Storage::One(value) => FStorage::One(value),
                Storage::Shared(data) => FStorage::Shared([dims[0], dims[1], dims[2]], data),
            },
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
        match &self.data {
            FStorage::One(_) => [1; 4],
            FStorage::Empty(dims) => *dims,
            FStorage::Shared([a, b, c], data) => [*a, *b, *c, data.len() / (a * b * c)],
        }
    }

    /// The row-major `[μ, ν, κ, λ]` coefficient data.
    pub fn data(&self) -> &[S] {
        match &self.data {
            FStorage::One(value) => std::slice::from_ref(value),
            FStorage::Empty(_) => &[],
            FStorage::Shared(_, data) => data,
        }
    }

    /// The coefficient at multiplicity indices `(μ, ν, κ, λ)`.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of range for [`FBlock::dims`].
    pub fn at(&self, mu: usize, nu: usize, kappa: usize, lambda: usize) -> S {
        let dims = self.dims();
        assert!(
            mu < dims[0] && nu < dims[1] && kappa < dims[2] && lambda < dims[3],
            "FBlock index out of range"
        );
        self.data()[Self::flat(dims, mu, nu, kappa, lambda)]
    }
}

/// A dense R-symbol block $R^{ab}_c$, an $N^c_{ab} \times N^c_{ba}$ matrix in
/// **row-major** order (rows = $\mu$, cols = $\nu$).
///
/// $N^c_{ab} = N^c_{ba}$ (fusion multiplicities are symmetric), so the matrix
/// is square. For a multiplicity-free family it is the single braiding phase.
#[derive(Clone, Debug)]
pub struct RBlock<S = f64> {
    data: RStorage<S>,
}

#[derive(Clone, Debug)]
enum RStorage<S> {
    One(S),
    Empty(usize),
    Shared(usize, Arc<[S]>),
}

impl<S: Scalar> RBlock<S> {
    /// A single coefficient, stored inline without allocation.
    pub fn one(value: S) -> Self {
        Self {
            data: RStorage::One(value),
        }
    }

    pub(crate) fn from_vec(n: usize, data: Vec<S>) -> Self {
        Self::from_storage(
            n,
            match data.len() {
                0 => Storage::Empty,
                1 => Storage::One(data[0]),
                _ => Storage::Shared(data.into()),
            },
        )
    }

    pub(crate) fn from_storage(n: usize, data: Storage<S>) -> Self {
        Self {
            data: match data {
                Storage::Empty => RStorage::Empty(n),
                Storage::One(value) => {
                    debug_assert_eq!(n, 1);
                    RStorage::One(value)
                }
                Storage::Shared(data) => RStorage::Shared(n, data),
            },
        }
    }

    /// `n × n` row-major data; a single entry is stored inline.
    pub(crate) fn from_row_major(n: usize, mut data: impl ExactSizeIterator<Item = S>) -> Self {
        debug_assert_eq!(data.len(), n * n);
        let data = match data.len() {
            0 => Storage::Empty,
            1 => Storage::One(data.next().expect("one entry")),
            _ => Storage::Shared(data.collect()),
        };
        Self::from_storage(n, data)
    }

    pub(crate) fn zeros(n: usize) -> Self {
        Self::from_vec(n, vec![S::zero(); n * n])
    }

    /// The multiplicity `N^c_{ab}` (both the row and column count).
    pub fn dim(&self) -> usize {
        match &self.data {
            RStorage::One(_) => 1,
            RStorage::Empty(n) | RStorage::Shared(n, _) => *n,
        }
    }

    /// The row-major coefficient data (`N × N`, rows `μ`, cols `ν`).
    pub fn data(&self) -> &[S] {
        match &self.data {
            RStorage::One(value) => std::slice::from_ref(value),
            RStorage::Empty(_) | RStorage::Shared(0, _) => &[],
            RStorage::Shared(_, data) => data,
        }
    }

    /// The braiding coefficient at `(μ, ν)`.
    ///
    /// # Panics
    ///
    /// Panics if `mu` or `nu` is `>= dim()`.
    pub fn at(&self, mu: usize, nu: usize) -> S {
        let n = self.dim();
        assert!(mu < n && nu < n, "RBlock index out of range");
        self.data()[mu * n + nu]
    }
}

impl<S: Scalar> SymbolBlock<4> for FBlock<S> {
    type Scalar = S;
    fn dims(&self) -> [usize; 4] {
        self.dims()
    }
    fn get(&self, [mu, nu, kappa, lambda]: [usize; 4]) -> S {
        self.at(mu, nu, kappa, lambda)
    }
    fn as_slice(&self) -> &[S] {
        self.data()
    }
}
impl<S: Scalar> SymbolBlock<2> for RBlock<S> {
    type Scalar = S;
    fn dims(&self) -> [usize; 2] {
        [self.dim(); 2]
    }
    fn get(&self, [mu, nu]: [usize; 2]) -> S {
        self.at(mu, nu)
    }
    fn as_slice(&self) -> &[S] {
        self.data()
    }
}

impl<S: Scalar> PartialEq for FBlock<S> {
    fn eq(&self, other: &Self) -> bool {
        self.dims() == other.dims() && self.data() == other.data()
    }
}
impl<S: Scalar> PartialEq for RBlock<S> {
    fn eq(&self, other: &Self) -> bool {
        self.dim() == other.dim() && self.data() == other.data()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_shapes_preserve_axes_and_entries() {
        for axis in 0..4 {
            let mut dims = [usize::MAX; 4];
            dims[axis] = 0;
            let block = FBlock::<f64>::zeros(dims);
            assert_eq!(block.clone().dims(), dims);
            assert!(block.data().is_empty());
        }
        let block = FBlock::from_vec([2, 1, 3, 4], (0..24).map(f64::from).collect());
        assert_eq!(block.clone().dims(), [2, 1, 3, 4]);
        assert_eq!(block.at(1, 0, 2, 3), 23.0);
        let block = RBlock::from_vec(3, (0..9).map(f64::from).collect());
        assert_eq!(block.clone().dim(), 3);
        assert_eq!(block.at(2, 1), 7.0);
    }

    #[test]
    fn equality_compares_values_not_storage() {
        assert_eq!(
            FBlock::one(0.5),
            FBlock::from_storage([1; 4], Storage::Shared(vec![0.5].into()))
        );
        assert!(matches!(
            FBlock::from_vec([1; 4], vec![0.5]).data,
            FStorage::One(0.5)
        ));
        assert!(matches!(
            RBlock::from_vec(1, vec![-1.0]).data,
            RStorage::One(-1.0)
        ));
        let empty = FBlock::<num_complex::Complex64>::zeros([0, 2, 1, 1]);
        assert!(matches!(empty.data, FStorage::Empty(_)));
        assert!(empty.data().is_empty());
        let empty = RBlock::<num_complex::Complex64>::zeros(0);
        assert!(matches!(empty.data, RStorage::Empty(0)));
        assert!(empty.data().is_empty());
        assert_eq!(
            FBlock::<f64>::zeros([0, 1, 1, 1]),
            FBlock::from_vec([0, 1, 1, 1], vec![])
        );
        assert_eq!(RBlock::one(-1.0), RBlock::from_vec(1, vec![-1.0]));
        // A/B blocks: a single entry stays inline, an absent channel is empty.
        assert!(matches!(
            RBlock::from_row_major(1, std::iter::once(-1.0)).data,
            RStorage::One(-1.0)
        ));
        assert!(matches!(
            RBlock::<f64>::from_row_major(0, std::iter::empty()).data,
            RStorage::Empty(0)
        ));
    }
}
