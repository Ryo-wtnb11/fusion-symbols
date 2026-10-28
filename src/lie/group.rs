//! Validated ordered root data and their exact character lattices.
//!
//! Coordinates are the Bourbaki Dynkin labels of each root block, followed by
//! integer torus charges. The character lattice contains every root; restricting
//! to it changes admission, never coefficients on surviving representations.
//! Identity preserves the ordered presentation, including B1/C1/A1 and D3/A3.
//! D2 denotes two ordered A1 coordinates within one presentation block.

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::{fmt, sync::Arc};

/// Root-system presentation of a compact semisimple Lie algebra.
///
/// The rank is the number of Dynkin labels. This fixes the weight lattice
/// and central-character congruences. `LieGroup` dispatches to the applicable
/// coefficient generator internally.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum RootSystem {
    /// `A_r`, center `Z_{r+1}`. Simply connected form `SU(r+1)`.
    A(usize),
    /// `B_r`, center `Z2`. Simply connected form `Spin(2r+1)`.
    B(usize),
    /// `C_r`, center `Z2`. Simply connected form `Sp(2r)`.
    C(usize),
    /// `D_r`, center `Z2×Z2` (`r` even) or `Z4` (`r` odd). Simply connected
    /// form `Spin(2r)`. D(2) is the ordered two-A1 presentation; r ≥ 2.
    D(usize),
    /// `G_2`, trivial center.
    G2,
    /// `F_4`, trivial center.
    F4,
    /// `E_6`, center `Z3`.
    E6,
    /// `E_7`, center `Z2`.
    E7,
    /// `E_8`, trivial center.
    E8,
}

impl RootSystem {
    /// The rank — the number of Dynkin labels a weight of this root system
    /// carries.
    pub fn rank(self) -> usize {
        match self {
            RootSystem::A(r) | RootSystem::B(r) | RootSystem::C(r) | RootSystem::D(r) => r,
            RootSystem::G2 => 2,
            RootSystem::F4 => 4,
            RootSystem::E6 => 6,
            RootSystem::E7 => 7,
            RootSystem::E8 => 8,
        }
    }
}

impl fmt::Display for RootSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RootSystem::A(r) => write!(f, "A{r}"),
            RootSystem::B(r) => write!(f, "B{r}"),
            RootSystem::C(r) => write!(f, "C{r}"),
            RootSystem::D(r) => write!(f, "D{r}"),
            RootSystem::G2 => write!(f, "G2"),
            RootSystem::F4 => write!(f, "F4"),
            RootSystem::E6 => write!(f, "E6"),
            RootSystem::E7 => write!(f, "E7"),
            RootSystem::E8 => write!(f, "E8"),
        }
    }
}

/// Immutable canonical datum. Rows generate the character lattice.
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Datum {
    roots: Box<[RootSystem]>,
    torus_rank: usize,
    // Row-major entries differing from the identity matrix.
    hnf: Box<[(usize, usize, i64)]>,
}

/// A connected compact group in a fixed ordered root/torus presentation.
///
/// The full-rank character lattice lies between the root lattice and the
/// ambient integral weight/charge lattice. Its row Hermite normal form (HNF)
/// fixes equality and hashing independently of the supplied basis. This is
/// presentation identity, not a test for abstract group isomorphism.
/// Classification does not imply an available [`super::LieGroup`] engine.
// The pointer shortcut is equivalent to content equality, matching derived Hash.
#[allow(clippy::derived_hash_with_manual_eq)]
#[derive(Clone, Debug, Eq, Hash, PartialOrd, Ord)]
pub struct GroupId(Arc<Datum>);

impl PartialEq for GroupId {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0 == other.0
    }
}

/// Error from a [`GroupId`] constructor.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GroupError {
    /// A rank or matrix dimension is outside its family's range.
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupError, GroupId};
    /// for result in [GroupId::so(2), GroupId::sp(3), GroupId::half_spin_plus(10)] {
    ///     assert!(matches!(result, Err(GroupError::UnsupportedRank { .. })));
    /// }
    /// ```
    UnsupportedRank {
        /// Constructor or root family.
        family: &'static str,
        /// Rejected argument.
        value: usize,
    },
    /// `k` is zero or does not divide `n`.
    NotACenterSubgroup {
        /// Matrix dimension of SU(n).
        n: usize,
        /// Requested subgroup order.
        k: usize,
    },
    /// The basis has the wrong shape, is singular, or omits a root.
    InvalidLattice {
        /// Failed invariant.
        reason: &'static str,
    },
    /// A matrix size is unaddressable or a canonical entry does not fit i64.
    SizeOverflow,
}

impl fmt::Display for GroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRank { family, value } => write!(
                f,
                "{family}({value}) is not a supported rank for this family"
            ),
            Self::NotACenterSubgroup { n, k } => write!(
                f,
                "Z{k} is not a subgroup of the SU({n}) center Z{n}: {k} does not divide {n}"
            ),
            Self::InvalidLattice { reason } => write!(f, "invalid character lattice: {reason}"),
            Self::SizeOverflow => {
                f.write_str("character lattice size or entry exceeds its integer range")
            }
        }
    }
}
impl std::error::Error for GroupError {}

impl GroupId {
    /// Validate a square integer row basis and normalize it exactly.
    ///
    /// HNF is upper triangular with positive diagonal and
    /// `0 <= H[i,j] < H[j,j]` for `i < j`. Every Cartan row (a root in
    /// Dynkin coordinates) must belong to the lattice. The empty datum is valid.
    /// Invalid ranks return [`GroupError::UnsupportedRank`]; malformed,
    /// singular, or root-excluding bases return [`GroupError::InvalidLattice`].
    pub fn new(
        roots: &[RootSystem],
        torus_rank: usize,
        generators: &[Vec<i64>],
    ) -> Result<Self, GroupError> {
        let n = lattice_size(roots, torus_rank)?;
        let size = n.checked_mul(n).ok_or(GroupError::SizeOverflow)?;
        if generators.len() != n || generators.iter().any(|g| g.len() != n) {
            return Err(GroupError::InvalidLattice {
                reason: "expected a square rank-sized basis",
            });
        }
        let mut h = Vec::new();
        h.try_reserve_exact(size)
            .map_err(|_| GroupError::SizeOverflow)?;
        h.extend(generators.iter().flatten().map(|&x| BigInt::from(x)));
        // Unimodular Euclidean row operations preserve exactly the input lattice.
        for j in 0..n {
            for i in j + 1..n {
                while !h[i * n + j].is_zero() {
                    let q = &h[j * n + j] / &h[i * n + j];
                    for k in j..n {
                        let sub = &q * &h[i * n + k];
                        h[j * n + k] -= sub;
                        h.swap(j * n + k, i * n + k);
                    }
                }
            }
            if h[j * n + j].is_zero() {
                return Err(GroupError::InvalidLattice {
                    reason: "singular basis",
                });
            }
            if h[j * n + j].is_negative() {
                for k in j..n {
                    h[j * n + k] = -&h[j * n + k];
                }
            }
            for i in 0..j {
                let pivot = &h[j * n + j];
                let mut q = &h[i * n + j] / pivot;
                if (&h[i * n + j] % pivot).is_negative() {
                    q -= 1;
                }
                if !q.is_zero() {
                    for k in j..n {
                        let sub = &q * &h[j * n + k];
                        h[i * n + k] -= sub;
                    }
                }
            }
        }
        let mut hnf = Vec::new();
        let differs = |k: usize, x: &BigInt| {
            if k / n == k % n {
                !x.is_one()
            } else {
                !x.is_zero()
            }
        };
        hnf.try_reserve_exact(h.iter().enumerate().filter(|(k, x)| differs(*k, x)).count())
            .map_err(|_| GroupError::SizeOverflow)?;
        for (k, x) in h.iter().enumerate() {
            if differs(k, x) {
                hnf.push((k / n, k % n, x.to_i64().ok_or(GroupError::SizeOverflow)?));
            }
        }
        let group = Self(Arc::new(Datum {
            roots: roots.into(),
            torus_rank,
            hnf: hnf.into(),
        }));
        if !group.contains_roots() {
            return Err(GroupError::InvalidLattice {
                reason: "character lattice omits a root",
            });
        }
        Ok(group)
    }

    fn contains_roots(&self) -> bool {
        let n = self.semisimple_rank() + self.torus_rank();
        let mut root = vec![0; n];
        let mut offset = 0;
        for &system in self.root_systems() {
            for i in 0..system.rank() {
                root.fill(0);
                for j in 0..system.rank() {
                    root[offset + j] = cartan(system, i, j);
                }
                if !self.contains_coordinates(&root, &[]) {
                    return false;
                }
            }
            offset += system.rank();
        }
        true
    }

    /// The full weight/charge lattice `P ⊕ Z^t` in the given presentation.
    /// Root containment and full rank are automatic for the identity basis.
    pub fn cover(roots: &[RootSystem], torus_rank: usize) -> Result<Self, GroupError> {
        lattice_size(roots, torus_rank)?;
        Ok(Self(Arc::new(Datum {
            roots: roots.into(),
            torus_rank,
            hnf: Box::new([]),
        })))
    }

    /// Standard rank-`rank` torus, with integer charges. Requires rank ≥ 1.
    /// Executes through [`super::LieGroup`] with charges in `-i32::MAX..=i32::MAX`.
    /// Standard tori enumerate in Product order; nonstandard torus lattices do not.
    pub fn torus(rank: usize) -> Result<Self, GroupError> {
        if rank == 0 {
            return Err(GroupError::UnsupportedRank {
                family: "torus",
                value: rank,
            });
        }
        Self::cover(&[], rank)
    }

    /// U(n), with `q ≡ Σ i·aᵢ (mod n)` under `(S,z) ↦ zS`.
    /// Fundamental charge is +1, determinant charge is n; U(1) is a torus.
    /// Executes through [`super::LieGroup`]: U(1) and U(2) in the base build,
    /// N ≥ 3 with `lie-generated`. U(1) enumerates 0,1,−1,…; N ≥ 2 returns
    /// [`EnumerationError::NotEnumerable`](crate::category::EnumerationError::NotEnumerable).
    pub fn u(n: usize) -> Result<Self, GroupError> {
        if n == 0 {
            return Err(GroupError::UnsupportedRank {
                family: "u",
                value: n,
            });
        }
        if n == 1 {
            return Self::torus(1);
        }
        Self::congruence(
            &[RootSystem::A(n - 1)],
            1,
            n,
            (0..n).map(|i| (i, if i + 1 == n { 1 } else { -(i as i128 + 1) })),
        )
    }

    /// Ordered root-system blocks; D(2) retains its two-coordinate presentation.
    pub fn root_systems(&self) -> &[RootSystem] {
        &self.0.roots
    }
    /// Total number of Dynkin coordinates.
    pub fn semisimple_rank(&self) -> usize {
        self.0.roots.iter().map(|r| r.rank()).sum()
    }
    /// Number of integer torus charges.
    pub fn torus_rank(&self) -> usize {
        self.0.torus_rank
    }
    /// Canonical row-HNF basis, in Dynkin-then-charge coordinate order.
    /// Allocates an n×n dense matrix, where n is the total rank.
    pub fn character_lattice(&self) -> Vec<Vec<i64>> {
        let n = self.semisimple_rank() + self.torus_rank();
        let mut rows = vec![vec![0; n]; n];
        for (i, row) in rows.iter_mut().enumerate() {
            row[i] = 1;
        }
        for &(i, j, value) in &self.0.hnf {
            rows[i][j] = value;
        }
        rows
    }

    /// Exact lattice admission. Wrong lengths or negative Dynkin labels return
    /// false; signed torus charges are allowed. This predicate never panics.
    ///
    /// ```
    /// use fusion_symbols::lie::GroupId;
    /// assert!(GroupId::psu(3).unwrap().admits(&[1, 1], &[]));
    /// assert!(!GroupId::psu(3).unwrap().admits(&[1, 0], &[]));
    /// assert!(GroupId::u(3).unwrap().admits(&[0, 1], &[-1]));
    /// ```
    pub fn admits(&self, dynkin: &[i64], charges: &[i64]) -> bool {
        if dynkin.len() != self.semisimple_rank()
            || charges.len() != self.torus_rank()
            || dynkin.iter().any(|&a| a < 0)
        {
            return false;
        }
        self.contains_coordinates(dynkin, charges)
    }

    // HNF unit-pivot columns have no entries above the diagonal, so their
    // solution coefficient is the input coordinate. Simple centers need at most
    // two nonunit pivots: keep those on the stack for allocation-free labels.
    fn contains_coordinates(&self, dynkin: &[i64], charges: &[i64]) -> bool {
        let n = dynkin.len() + charges.len();
        let coordinate = |i: usize| {
            if i < dynkin.len() {
                dynkin[i]
            } else {
                charges[i - dynkin.len()]
            }
        };
        let fast = || -> Option<bool> {
            let mut pivots = [(0usize, 0i128); 2];
            let mut used = 0;
            for &(i, j, _) in &self.0.hnf {
                if i == j {
                    if used == pivots.len() {
                        return None;
                    }
                    pivots[used] = (i, i128::from(coordinate(i)));
                    used += 1;
                }
            }
            let mut entries = self.0.hnf.iter().peekable();
            for i in 0..n {
                let mut coefficient = i128::from(coordinate(i));
                if let Some(&&(row, col, pivot)) = entries.peek() {
                    if row == i && col == i {
                        let value = pivots[..used].iter().find(|&&(j, _)| j == i)?.1;
                        if value % i128::from(pivot) != 0 {
                            return Some(false);
                        }
                        coefficient = value / i128::from(pivot);
                        entries.next();
                    }
                }
                while let Some(&&(row, col, value)) = entries.peek() {
                    if row != i {
                        break;
                    }
                    let remainder = &mut pivots[..used].iter_mut().find(|(j, _)| *j == col)?.1;
                    *remainder =
                        remainder.checked_sub(coefficient.checked_mul(i128::from(value))?)?;
                    entries.next();
                }
            }
            Some(true)
        };
        fast().unwrap_or_else(|| {
            // Arbitrarily coupled lattices can require larger exact intermediates.
            let mut remainder: Vec<BigInt> = (0..n).map(|i| BigInt::from(coordinate(i))).collect();
            let mut entries = self.0.hnf.iter().peekable();
            for i in 0..n {
                let pivot = if let Some(&&(row, col, value)) = entries.peek() {
                    if row == i && col == i {
                        entries.next();
                        value
                    } else {
                        1
                    }
                } else {
                    1
                };
                if !(&remainder[i] % pivot).is_zero() {
                    return false;
                }
                let coefficient = &remainder[i] / pivot;
                while let Some(&&(row, col, value)) = entries.peek() {
                    if row != i {
                        break;
                    }
                    remainder[col] -= &coefficient * value;
                    entries.next();
                }
            }
            true
        })
    }
    pub(super) fn is_cover(&self) -> bool {
        self.0.hnf.is_empty()
    }

    /// SU(n), the A(n−1) cover. Requires n ≥ 2.
    pub fn su(n: usize) -> Result<Self, GroupError> {
        if n < 2 {
            return Err(GroupError::UnsupportedRank {
                family: "su",
                value: n,
            });
        }
        Self::cover(&[RootSystem::A(n - 1)], 0)
    }
    /// SU(n)/Z_k; k must divide n. Admission is `Σ i·aᵢ ≡ 0 (mod k)`.
    /// The trivial quotient k=1 equals SU(n); k=n gives PSU(n).
    pub fn su_quotient(n: usize, k: usize) -> Result<Self, GroupError> {
        if n < 2 {
            return Err(GroupError::UnsupportedRank {
                family: "su",
                value: n,
            });
        }
        if k == 0 || !n.is_multiple_of(k) {
            return Err(GroupError::NotACenterSubgroup { n, k });
        }
        Self::congruence(
            &[RootSystem::A(n - 1)],
            0,
            k,
            (0..n - 1).map(|i| (i, i as i128 + 1)),
        )
    }
    /// PSU(n), the adjoint form SU(n)/Z_n.
    pub fn psu(n: usize) -> Result<Self, GroupError> {
        Self::su_quotient(n, n)
    }
    /// Sp(m), the C(m/2) cover. Matrix dimension m must be even and ≥ 2.
    pub fn sp(m: usize) -> Result<Self, GroupError> {
        Self::cover(&[RootSystem::C(sp_rank(m)?)], 0)
    }
    /// PSp(m); admission is `Σ_(i odd) aᵢ ≡ 0 (mod 2)`.
    pub fn psp(m: usize) -> Result<Self, GroupError> {
        let r = sp_rank(m)?;
        Self::congruence(&[RootSystem::C(r)], 0, 2, (0..r).step_by(2).map(|i| (i, 1)))
    }
    /// Spin(n), the B((n−1)/2) or D(n/2) cover. Matrix dimension n ≥ 3.
    pub fn spin(n: usize) -> Result<Self, GroupError> {
        Self::cover(&[orthogonal(n, "spin")?], 0)
    }
    /// SO(n), n ≥ 3. B_r admits `a_r` even; D_r admits
    /// `a_(r−1)+a_r` even (the retained tensor classes `{0,v}`).
    pub fn so(n: usize) -> Result<Self, GroupError> {
        let root = orthogonal(n, "so")?;
        let r = root.rank();
        let start = if matches!(root, RootSystem::B(_)) {
            r - 1
        } else {
            r - 2
        };
        Self::congruence(&[root], 0, 2, (start..r).map(|i| (i, 1)))
    }

    /// PSO(n), the adjoint form. Odd n already gives SO(n).
    ///
    /// For D_r, odd r, admission is κ=0 mod 4 with
    /// `κ = a_r − a_(r−1) + 2 Σ_(i odd ≤ r−2) aᵢ` and generator `[ω_r]=+1`.
    /// Indeed Q is the even-sum sublattice of Z^r and `4ω_r=(2,…,2)∈Q`;
    /// the class of an integral weight is twice its coordinate sum. Slansky
    /// Table 41 (SO(10)) gives `45↦0, 16↦1, 16bar↦−1, 10↦2`.
    /// Slansky's uniform D formula is +κ for r=1 mod 4 and −κ for r=3 mod 4;
    /// our r-uniform generator agrees with duality's `λ_r↦−λ_r`.
    /// For even r, both `p=a_(r−1)+t` and `q=a_r+t` must be even,
    /// where `t=Σ_(i odd ≤ r−2) aᵢ`.
    pub fn pso(n: usize) -> Result<Self, GroupError> {
        let root = orthogonal(n, "pso")?;
        if matches!(root, RootSystem::B(_)) {
            return Self::so(n);
        }
        let r = root.rank();
        if r % 2 == 1 {
            return Self::congruence(
                &[root],
                0,
                4,
                (0..r - 2)
                    .step_by(2)
                    .map(|i| (i, 2))
                    .chain([(r - 2, -1), (r - 1, 1)]),
            );
        }
        let mut hnf = Vec::new();
        hnf.try_reserve_exact(r)
            .map_err(|_| GroupError::SizeOverflow)?;
        for i in (0..r - 2).step_by(2) {
            hnf.push((i, r - 2, 1));
            hnf.push((i, r - 1, 1));
        }
        hnf.push((r - 2, r - 2, 2));
        hnf.push((r - 1, r - 1, 2));
        Ok(Self::named(&[root], 0, hnf))
    }

    /// G₂, the simply connected (and adjoint) form: the center is trivial.
    /// Infallible, like the other exceptional constructors: the rank is fixed.
    /// G₂ coefficients require `lie-generated`.
    pub fn g2() -> Self {
        Self::named(&[RootSystem::G2], 0, Vec::new())
    }
    /// F₄, the simply connected (and adjoint) form: the center is trivial.
    /// Coefficients require `lie-generated`.
    pub fn f4() -> Self {
        Self::named(&[RootSystem::F4], 0, Vec::new())
    }
    /// Simply connected E₆ (center `Z3`); quotients come from [`Self::new`].
    /// Coefficients require `lie-generated`.
    pub fn e6() -> Self {
        Self::named(&[RootSystem::E6], 0, Vec::new())
    }
    /// Simply connected E₇ (center `Z2`); quotients come from [`Self::new`].
    /// Coefficients require `lie-generated`.
    pub fn e7() -> Self {
        Self::named(&[RootSystem::E7], 0, Vec::new())
    }
    /// E₈, the simply connected (and adjoint) form: the center is trivial.
    /// Coefficients require `lie-generated`.
    pub fn e8() -> Self {
        Self::named(&[RootSystem::E8], 0, Vec::new())
    }

    /// Half-spin form retaining `[ω_r]` (`λ_r=+1/2`), Slansky's Ss(n).
    /// Requires n ≥ 8 divisible by 4. With `t=Σ_(i odd ≤ r−2) aᵢ`,
    /// admission is `p=a_(r−1)+t` even; the vector class is excluded.
    ///
    /// Naming is by the retained class: killing `z_(ω_r)` retains ω_r for
    /// r=0 mod 4 but ω_(r−1) for r=2 mod 4, since `(ω_r,ω_r)=r/4`.
    /// Slansky Table 36 labels `[ω_4]` sub-s, consistent with Table 41.
    pub fn half_spin_plus(n: usize) -> Result<Self, GroupError> {
        Self::half_spin(n, false)
    }
    /// Half-spin form retaining `[ω_(r−1)]` (`λ_r=−1/2`), Slansky's Sc(n).
    /// Admission is `q=a_r+Σ_(i odd ≤ r−2) aᵢ` even. Same range and retained-class
    /// naming rationale as [`Self::half_spin_plus`].
    pub fn half_spin_minus(n: usize) -> Result<Self, GroupError> {
        Self::half_spin(n, true)
    }
    fn half_spin(n: usize, minus: bool) -> Result<Self, GroupError> {
        let root = half_spin_root_system(n)?;
        let r = root.rank();
        Self::congruence(
            &[root],
            0,
            2,
            (0..r - 2)
                .step_by(2)
                .chain([r - 2 + usize::from(minus)])
                .map(|i| (i, 1)),
        )
    }

    // Nonzero central-character coefficients are ordered by coordinate. The
    // final one is ±1 modulo m, so eliminating it gives row-HNF directly.
    fn congruence(
        roots: &[RootSystem],
        torus: usize,
        modulus: usize,
        coefficients: impl DoubleEndedIterator<Item = (usize, i128)> + Clone,
    ) -> Result<Self, GroupError> {
        let n = lattice_size(roots, torus)?;
        if modulus == 1 {
            return Self::cover(roots, torus);
        }
        // Bound the entry storage before traversing the coefficients.
        if n.checked_add(1).ok_or(GroupError::SizeOverflow)?
            > isize::MAX as usize / std::mem::size_of::<(usize, usize, i64)>()
        {
            return Err(GroupError::SizeOverflow);
        }
        let m = i64::try_from(modulus).map_err(|_| GroupError::SizeOverflow)?;
        let Some((p, c)) = coefficients
            .clone()
            .rev()
            .find(|&(_, c)| c.rem_euclid(i128::from(m)) != 0)
        else {
            return Self::cover(roots, torus);
        };
        let inverse = if c.rem_euclid(i128::from(m)) == 1 {
            1
        } else {
            -1
        };
        debug_assert_eq!((c * inverse).rem_euclid(i128::from(m)), 1);
        let entries = coefficients
            .map(|(i, c)| (i, p, (-c * inverse).rem_euclid(i128::from(m)) as i64))
            .filter(|&(i, _, value)| i != p && value != 0);
        let mut hnf = Vec::new();
        hnf.try_reserve_exact(
            entries
                .clone()
                .count()
                .checked_add(1)
                .ok_or(GroupError::SizeOverflow)?,
        )
        .map_err(|_| GroupError::SizeOverflow)?;
        hnf.extend(entries);
        hnf.push((p, p, m));
        Ok(Self::named(roots, torus, hnf))
    }

    fn named(roots: &[RootSystem], torus_rank: usize, hnf: Vec<(usize, usize, i64)>) -> Self {
        let group = Self(Arc::new(Datum {
            roots: roots.into(),
            torus_rank,
            hnf: hnf.into(),
        }));
        // Roots have trivial central character; check the construction at small ranks.
        debug_assert!(group.semisimple_rank() + torus_rank > 64 || group.contains_roots());
        group
    }
}

// Validate root domains and the rank sum; dense sizes are checked only by new.
fn lattice_size(roots: &[RootSystem], torus: usize) -> Result<usize, GroupError> {
    let mut n = torus;
    for &root in roots {
        let (family, valid) = match root {
            RootSystem::A(r) => ("A", r >= 1),
            RootSystem::B(r) => ("B", r >= 1),
            RootSystem::C(r) => ("C", r >= 1),
            RootSystem::D(r) => ("D", r >= 2),
            _ => ("exceptional", true),
        };
        if !valid {
            return Err(GroupError::UnsupportedRank {
                family,
                value: root.rank(),
            });
        }
        n = n.checked_add(root.rank()).ok_or(GroupError::SizeOverflow)?;
    }
    Ok(n)
}

// Row i is α_i expressed in fundamental weights: <α_i, α_j∨>.
// Bourbaki B has its short root last, C its long root last; D branches at r−2.
// E has chain 1–3–4–5–… and node 2 attached to 4; F's first two roots are long,
// and G's first root is short. D2 consists of two disconnected A1 roots.
pub(super) fn cartan(root: RootSystem, i: usize, j: usize) -> i64 {
    if i == j {
        return 2;
    }
    let r = root.rank();
    let adjacent = match root {
        RootSystem::D(_) => {
            (i < r - 2 && j == i + 1)
                || (j < r - 2 && i == j + 1)
                || (r >= 3 && ((i == r - 3 && j == r - 1) || (j == r - 3 && i == r - 1)))
        }
        RootSystem::E6 | RootSystem::E7 | RootSystem::E8 => {
            matches!((i, j), (0, 2) | (2, 0) | (1, 3) | (3, 1))
                || (i >= 2 && j >= 2 && i.abs_diff(j) == 1)
        }
        _ => i.abs_diff(j) == 1,
    };
    if !adjacent {
        return 0;
    }
    match root {
        RootSystem::B(_) if i + 2 == r && j + 1 == r => -2,
        RootSystem::C(_) if i + 1 == r && j + 2 == r => -2,
        RootSystem::F4 if i == 1 && j == 2 => -2,
        RootSystem::G2 if i == 1 && j == 0 => -3,
        _ => -1,
    }
}

fn sp_rank(m: usize) -> Result<usize, GroupError> {
    if m < 2 || !m.is_multiple_of(2) {
        return Err(GroupError::UnsupportedRank {
            family: "sp",
            value: m,
        });
    }
    Ok(m / 2)
}

/// `B((n-1)/2)` or `D(n/2)`, from the matrix dimension `n ≥ 3`.
fn orthogonal(n: usize, family: &'static str) -> Result<RootSystem, GroupError> {
    match n {
        n if n < 3 => Err(GroupError::UnsupportedRank { family, value: n }),
        n if n % 2 == 1 => Ok(RootSystem::B((n - 1) / 2)),
        n => Ok(RootSystem::D(n / 2)),
    }
}

fn half_spin_root_system(n: usize) -> Result<RootSystem, GroupError> {
    // n = 2r with r even ⇔ n ≡ 0 (mod 4); r ≥ 4 keeps D_r semisimple and the
    // two chiralities distinct.
    if n < 8 || !n.is_multiple_of(4) {
        return Err(GroupError::UnsupportedRank {
            family: "half_spin",
            value: n,
        });
    }
    Ok(RootSystem::D(n / 2))
}
