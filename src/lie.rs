//! Positive-dimensional compact Lie groups: root systems and global forms via
//! [`LieGroup`] (including tori), and closed-form [`U1`] and [`CU1`].

mod cu1;
mod u1;
pub use cu1::{CU1Error, CU1Fusion, CU1Irrep, CU1};
pub use u1::{U1Error, U1Irrep, U1};

pub(crate) mod general;
mod gt;
pub use gt::GtPattern;
mod group;
pub(crate) mod su2;
pub use group::{GroupError, GroupId, RootSystem};
mod half_int;
pub use half_int::{HalfInt, HalfIntError};
mod prefill;
pub use prefill::{prefill, PrefillRequest};

use crate::category::{
    sealed::Validated, BraidingStyle, CgcBlock, EnumerationError, FBlock, FusionCategory,
    FusionStyle, GroupRepresentations, RBlock,
};
use crate::product::{Product, ProductError};
use crate::rounding::{round_dim, DimValue, Factor};
use malachite_base::{
    num::conversion::traits::RoundingFrom, rounding_modes::RoundingMode::Nearest,
};
use malachite_nz::{integer::Integer, natural::Natural};
use std::collections::BTreeMap;
use std::sync::Arc;
use threadteam::Exec;

/// A Lie group with its global form. A(1), B(1), and C(1) share the exact
/// SU(2) gauge; their single Dynkin label is doubled spin. Every simple factor
/// computes its CGC with the exact Cartan engine (`docs/gauge_cartan.md`): the
/// Gelfand–Tsetlin basis for the A series, the contravariant Gram–Schmidt
/// basis for B, C, D, G2, F4, E6, E7 and E8. SU(2) F and R stay closed forms.
/// Groups other than A(1), B(1), C(1) and D(2) require `lie-generated`.
/// Retained coefficients live in the process-wide tiers of [`crate::cache`],
/// shared by every group value.
/// Admissibility uses the caller's global form.
/// Objects retain the full `GroupId` tag, including for isomorphic groups.
/// [`U1`] is the same integer-charge U(1) as `LieGroup::new(GroupId::u(1)?)`,
/// as a fixed category with `Copy` labels and scalar symbols.
///
/// Enumeration (frozen), with an infinite count:
///
/// - rank one (A(1), B(1), C(1)): admitted doubled spins `dj` in increasing
///   order, so index `dj` when every weight is admitted (SU(2), Spin(3),
///   Sp(2)) and `dj / 2` for SO(3) (`dj = 0, 2, 4, …`); labels past `u32` are
///   [`EnumerationError::Overflow`];
/// - SU(N), `N ≥ 3`: Dynkin labels by increasing total `Σ aᵢ`, then
///   ascending lexicographic order (SUNRepresentations.jl's order), so SU(3)
///   starts `(0,0), (0,1), (1,0), (0,2), (1,1), (2,0)`;
/// - U(1): charges 0, 1, −1, 2, −2, …; positive q has index 2q−1,
///   nonpositive q index −2q; charges beyond `i32::MAX` in magnitude overflow;
/// - the trivial group: one empty label at index zero;
/// - composite covers: the Product's Manhattan order over ordered simple
///   factors then torus coordinates (D(2) expands to A(1), A(1)); every factor must be enumerable;
/// - every composite quotient, other higher A-series global forms, G2, F4,
///   E6, E7, E8, and
///   B/C/D simple factors of rank at least two: [`EnumerationError::NotEnumerable`].
///
/// # Panics
/// Every object-taking category method except `contains` panics on an object
/// from another group.
#[derive(Clone)]
pub struct LieGroup {
    group: GroupId,
    composite: Option<Product<Vec<LieGroup>>>,
    // At most one retained rank-sized unit label per group instance.
    unit: std::sync::OnceLock<LieIrrep>,
    // The root data of a simple factor of rank at least two, built when the
    // first label needs it and shared by clones.
    roots: Option<Arc<std::sync::OnceLock<general::GeneralFamily>>>,
}

impl std::fmt::Debug for LieGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LieGroup")
            .field("group", &self.group)
            .finish()
    }
}

impl PartialEq for LieGroup {
    fn eq(&self, other: &Self) -> bool {
        self.group == other.group
    }
}
impl Eq for LieGroup {}
impl std::hash::Hash for LieGroup {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.group.hash(state);
    }
}

/// A validated representation, including its group's identity. Rank-one
/// labels are stored inline, without allocation. Higher-rank clones share their
/// exact payload, including dimension and conjugate coordinates; duals need no cache.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LieIrrep {
    group: GroupId,
    payload: LiePayload,
}

// A group fixes its engine, so only values within a variant are compared after
// GroupId. Within a variant the ordering is field order, including signed charges.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
// Avoid niche-discriminant decoding on the inline symbol paths.
#[repr(u8)]
enum LiePayload {
    RankOne(u32),
    Torus(U1Irrep),
    Simple(general::Irrep),
    Composite(Arc<Vec<LieIrrep>>),
}

struct DynkinLabels<'a> {
    irrep: &'a LieIrrep,
    indices: std::ops::Range<usize>,
}

impl Iterator for DynkinLabels<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<i64> {
        let mut i = self.indices.next()?;
        if let Some(factors) = self.irrep.factors() {
            for factor in factors.iter() {
                let rank = factor.group.semisimple_rank();
                if i < rank {
                    return DynkinLabels {
                        irrep: factor,
                        indices: i..i + 1,
                    }
                    .next();
                }
                i -= rank;
            }
            unreachable!("composite rank equals the sum of leaf ranks");
        }
        if let Some(a) = self.irrep.simple() {
            return Some(a.dynkin()[i]);
        }
        Some(i64::from(self.irrep.dj()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl ExactSizeIterator for DynkinLabels<'_> {}

enum LieEnumeration {
    RankOne { stride: u64 },
    Dynkin,
}

/// Invalid labels, excluded representations, arithmetic failure, or absent capability.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LieError {
    /// A coefficient generation failed.
    Generation(GenerationError),
    /// Wrong Dynkin/charge lengths, a negative Dynkin label, or a spin on a
    /// group other than SU(2) and SO(3).
    InvalidLabel,
    /// The group's global form excludes this weight.
    NotAdmissible,
    /// A label, weight, dimension or multiplicity exceeds the integer range.
    LabelOverflow,
    /// The named capability is unavailable, for example a feature
    /// (`"lie-generated"`).
    Unsupported {
        /// Missing capability.
        requires: &'static str,
    },
}
/// Opaque diagnostic for a coefficient-generation failure.
#[derive(Clone, Debug)]
pub struct GenerationError(std::sync::Arc<dyn std::error::Error + Send + Sync>);
impl PartialEq for GenerationError {
    fn eq(&self, other: &Self) -> bool {
        self.to_string() == other.to_string()
    }
}
impl std::fmt::Display for GenerationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for GenerationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}
impl std::fmt::Display for LieError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generation(error) => error.fmt(f),
            Self::InvalidLabel => {
                f.write_str("expected rank-many nonnegative Dynkin labels and torus-many charges")
            }
            Self::NotAdmissible => f.write_str("weight is not admitted by this global form"),
            Self::LabelOverflow => {
                f.write_str("a label, weight, dimension or multiplicity exceeds the integer range")
            }
            Self::Unsupported { requires } => write!(f, "unsupported capability: {requires}"),
        }
    }
}
impl std::error::Error for LieError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Generation(error) => Some(error),
            _ => None,
        }
    }
}

/// A Weyl dimension as `f64`: the checked machine value when present, else
/// the exact big-integer one. Both conversions are correctly rounded, so the
/// bits do not depend on which path ran.
fn weyl_dim_f64(checked: Option<u128>, exact: impl FnOnce() -> Natural) -> f64 {
    checked.map_or_else(|| f64::rounding_from(&exact(), Nearest).0, |d| d as f64)
}
fn checked_dimension(value: impl TryInto<usize>) -> Result<usize, LieError> {
    value.try_into().map_err(|_| LieError::LabelOverflow)
}
pub(crate) fn checked_size<const N: usize>(dims: [usize; N]) -> Result<usize, LieError> {
    if dims.contains(&0) {
        return Ok(0);
    }
    let size = dims
        .into_iter()
        .try_fold(1usize, |n, d| n.checked_mul(d))
        .ok_or(LieError::LabelOverflow)?;
    if size > isize::MAX as usize / std::mem::size_of::<f64>() {
        return Err(LieError::LabelOverflow);
    }
    Ok(size)
}

fn checked_count(a: usize, b: usize) -> Result<usize, LieError> {
    a.checked_add(b).ok_or(LieError::LabelOverflow)
}
fn a1_dimension(dj: u32) -> Result<usize, LieError> {
    checked_count(checked_dimension(dj)?, 1)
}

/// A simple-factor label whose coefficients the public API can serve: a
/// complex D(r) (r odd) or E6 label above `u32::MAX` dimensions cannot
/// generate its phase in `u32` magnetic indices.
fn simple_label(x: general::Irrep) -> Result<general::Irrep, LieError> {
    if x.public_coefficient_label_ok() {
        Ok(x)
    } else {
        Err(LieError::LabelOverflow)
    }
}

fn coefficient_roots(group: &GroupId) -> Result<(), LieError> {
    #[cfg(not(feature = "lie-generated"))]
    if group.root_systems().iter().any(|root| {
        !matches!(
            root,
            RootSystem::A(1) | RootSystem::B(1) | RootSystem::C(1) | RootSystem::D(2)
        )
    }) {
        return Err(LieError::Unsupported {
            requires: "lie-generated",
        });
    }
    // With `lie-generated` every root system has coefficients.
    let _ = group;
    Ok(())
}

// Product errors allow borrowed generic sources. Keep source forwarding local
// to the owned Lie error so composition does not tighten Product's public bounds.
#[derive(Debug)]
struct CompositeError(ProductError<LieError>);
impl std::fmt::Display for CompositeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}
impl std::error::Error for CompositeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.0 {
            ProductError::Factor { source, .. } | ProductError::Factor0(source) => Some(source),
            _ => None,
        }
    }
}

/// A composite group's product error as the group's own error.
fn composite_error(error: ProductError<LieError>) -> LieError {
    match error {
        ProductError::Overflow => LieError::LabelOverflow,
        // Keep matchable leaf capability/overflow variants; generation and
        // malformed-block diagnostics retain the complete indexed source path.
        ProductError::Factor {
            source: source @ (LieError::LabelOverflow | LieError::Unsupported { .. }),
            ..
        } => source,
        error => LieError::Generation(GenerationError(Arc::new(CompositeError(error)))),
    }
}

impl LieGroup {
    /// The complete validated classification datum.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupId, LieGroup};
    ///
    /// let u2 = LieGroup::new(GroupId::u(2)?)?;
    /// assert_eq!(u2.group_id(), &GroupId::u(2)?);
    /// assert_eq!((u2.group_id().semisimple_rank(), u2.group_id().torus_rank()), (1, 1));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn group_id(&self) -> &GroupId {
        &self.group
    }
    /// Construct a connected compact group with executable classical factors and any
    /// admitted character lattice. A(1), B(1), C(1), and D(2) need no feature;
    /// A/B/C ranks ≥ 2 and D ranks ≥ 3 require `lie-generated`.
    /// D(2) expands into α₁ then α₂, each in the exact A(1) gauge.
    /// Tori, U(N), semisimple×torus quotients, and the trivial group are supported.
    /// U(2) needs no feature; U(N) for N ≥ 3 requires `lie-generated`.
    /// G2, F4, E6, E7 and E8 require `lie-generated`.
    /// Single factors retain their evaluator and gauge. Composites use ordered
    /// [`Product`] cover symbols and only restrict objects at admission.
    /// Construction generates nothing: root data are built for the first label.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieGroup}};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?; // no feature needed
    /// assert_eq!(su2.dim(&su2.irrep(&[1])?), 2.0);
    ///
    /// let su3 = LieGroup::new(GroupId::su(3)?);
    /// #[cfg(feature = "lie-generated")]
    /// assert!(su3.is_ok());
    /// #[cfg(not(feature = "lie-generated"))]
    /// assert!(matches!(
    ///     su3,
    ///     Err(fusion_symbols::lie::LieError::Unsupported { requires: "lie-generated" })
    /// ));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(group: GroupId) -> Result<Self, LieError> {
        coefficient_roots(&group)?;
        let plain = |group, roots| Self {
            group,
            composite: None,
            unit: std::sync::OnceLock::new(),
            roots,
        };
        if group.root_systems().is_empty() && group.torus_rank() == 1 && group.is_cover() {
            return Ok(plain(group, None));
        }
        if group.torus_rank() != 0
            || group.root_systems().len() != 1
            || group.root_systems()[0] == RootSystem::D(2)
        {
            return Self::composite(group);
        }
        let root = group.root_systems()[0];
        if root.rank() >= 2 {
            // Root tables hold rank-squared entries and A-series weights rank + 1.
            if root.rank() >= isize::MAX as usize / std::mem::size_of::<i64>() {
                return Err(LieError::LabelOverflow);
            }
            return Ok(plain(group, Some(Arc::new(std::sync::OnceLock::new()))));
        }
        if !matches!(root, RootSystem::A(1) | RootSystem::B(1) | RootSystem::C(1)) {
            return Err(LieError::Unsupported {
                requires: "higher-rank Lie representations",
            });
        }
        Ok(plain(group, None))
    }
    fn composite(group: GroupId) -> Result<Self, LieError> {
        let roots: Vec<_> = group
            .root_systems()
            .iter()
            .flat_map(|&root| {
                if root == RootSystem::D(2) {
                    vec![RootSystem::A(1); 2]
                } else {
                    vec![root]
                }
            })
            .collect();
        let mut factors = Vec::new();
        factors
            .try_reserve_exact(checked_count(roots.len(), group.torus_rank())?)
            .map_err(|_| LieError::LabelOverflow)?;
        for root in roots {
            let cover = GroupId::cover(&[root], 0).expect("validated factor has a valid cover");
            factors.push(Self::new(cover)?);
        }
        for _ in 0..group.torus_rank() {
            factors.push(Self::new(
                GroupId::torus(1).expect("rank-one torus is valid"),
            )?);
        }
        Ok(Self {
            group,
            composite: Some(Product::new(factors)),
            unit: std::sync::OnceLock::new(),
            roots: None,
        })
    }
    fn composite_label(group: &GroupId, factors: Vec<LieIrrep>) -> LieIrrep {
        let label = LieIrrep {
            group: group.clone(),
            payload: LiePayload::Composite(Arc::new(factors)),
        };
        debug_assert!(group.admits(
            &label.dynkin().collect::<Vec<_>>(),
            &label.charges().collect::<Vec<_>>()
        ));
        label
    }
    // Callers have checked the outward group; factor tags are private and immutable.
    fn factor_labels<'a>(&self, a: &'a LieIrrep) -> &'a Vec<LieIrrep> {
        a.factors().expect("composite object has factor labels")
    }
    /// Construct an irrep of a group without torus charges (torus rank zero)
    /// from the Dynkin labels of its highest weight: `dynkin[i]` is the
    /// coefficient of the fundamental weight ωᵢ₊₁ in Bourbaki numbering, with
    /// the labels of product factors concatenated in factor order. For SU(3),
    /// `[1, 0]` is the 3, `[0, 1]` the 3̄ and `[1, 1]` the adjoint 8; see the
    /// [Dynkin labels guide](https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/user-guide/labels.md).
    /// A group with torus rank above zero returns [`LieError::InvalidLabel`];
    /// use [`Self::irrep_with_charges`] there. Otherwise identical to
    /// `irrep_with_charges(dynkin, &[])`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieError, LieGroup}};
    ///
    /// let spin4 = LieGroup::new(GroupId::spin(4)?)?; // SU(2) × SU(2): two labels
    /// assert_eq!(spin4.dim(&spin4.irrep(&[1, 1])?), 4.0);
    /// assert_eq!(spin4.irrep(&[1]), Err(LieError::InvalidLabel));
    /// let u2 = LieGroup::new(GroupId::u(2)?)?;
    /// assert_eq!(u2.irrep(&[1]), Err(LieError::InvalidLabel)); // needs a charge
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn irrep(&self, dynkin: &[i64]) -> Result<LieIrrep, LieError> {
        if self.group.torus_rank() != 0 {
            return Err(LieError::InvalidLabel);
        }
        self.irrep_with_charges(dynkin, &[])
    }
    /// The irrep of spin `j` of SU(2) or SO(3), in any rank-one presentation
    /// (root systems exactly `[A(1)]`, `[B(1)]` or `[C(1)]`, no torus), whose
    /// single Dynkin label is `2j`. Equal to `irrep(&[2j])`, so SO(3) rejects
    /// half-integer `j` as [`LieError::NotAdmissible`]. Every other group
    /// returns [`LieError::InvalidLabel`].
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupId, HalfInt, LieError, LieGroup};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// assert_eq!(su2.irrep_spin(HalfInt::new(1, 1)?)?, su2.irrep(&[2])?); // spin 1
    /// let so3 = LieGroup::new(GroupId::so(3)?)?;
    /// assert_eq!(so3.irrep_spin(HalfInt::new(1, 2)?), Err(LieError::NotAdmissible));
    /// let u2 = LieGroup::new(GroupId::u(2)?)?;
    /// assert_eq!(u2.irrep_spin(HalfInt::new(1, 2)?), Err(LieError::InvalidLabel));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn irrep_spin(&self, j: HalfInt) -> Result<LieIrrep, LieError> {
        if !is_spin_group(&self.group) {
            return Err(LieError::InvalidLabel);
        }
        self.irrep(&[i64::from(j.doubled())])
    }
    /// Construct an irrep from Bourbaki Dynkin labels (see [`Self::irrep`])
    /// followed by integer torus charges. For U(N), the fundamental has
    /// charge +1 and `(dynkin, [q])` is admitted when `q ≡ Σ i·aᵢ (mod N)`.
    ///
    /// Validate label lengths and nonnegative Dynkin labels, then lattice
    /// admission, then engine representability. Torus charges lie in
    /// `-i32::MAX..=i32::MAX`; out-of-range charges return [`LieError::LabelOverflow`].
    /// A D(r) (r odd) or E6 irrep that is not self-dual, with dimension above
    /// `u32::MAX`, returns [`LieError::LabelOverflow`] (so does its dual), here
    /// and from `fusion` when an outcome is such a label: its Frobenius–Schur
    /// phase is generated in the module basis, whose magnetic index is `u32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieError, LieGroup}};
    ///
    /// let u2 = LieGroup::new(GroupId::u(2)?)?;
    /// let fundamental = u2.irrep_with_charges(&[1], &[1])?;
    /// let det = u2.irrep_with_charges(&[0], &[2])?;
    /// assert_eq!(u2.nsymbol(&fundamental, &fundamental, &det)?, 1); // Λ²(2) = det
    /// assert_eq!(u2.irrep_with_charges(&[1], &[0]), Err(LieError::NotAdmissible));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn irrep_with_charges(
        &self,
        dynkin: &[i64],
        charges: &[i64],
    ) -> Result<LieIrrep, LieError> {
        if dynkin.len() != self.group.semisimple_rank()
            || charges.len() != self.group.torus_rank()
            || dynkin.iter().any(|&x| x < 0)
        {
            return Err(LieError::InvalidLabel);
        }
        if !self.group.admits(dynkin, charges) {
            return Err(LieError::NotAdmissible);
        }
        if let Some(product) = &self.composite {
            let mut offset = 0;
            let mut charge_offset = 0;
            let factors = product
                .factors()
                .iter()
                .map(|factor| {
                    let end = offset + factor.group.semisimple_rank();
                    let charge_end = charge_offset + factor.group.torus_rank();
                    let label = factor.irrep_with_charges(
                        &dynkin[offset..end],
                        &charges[charge_offset..charge_end],
                    );
                    charge_offset = charge_end;
                    offset = end;
                    label
                })
                .collect::<Result<_, _>>()?;
            return Ok(Self::composite_label(&self.group, factors));
        }
        if self.is_torus() {
            let q = U1Irrep::from_exact(charges[0]).map_err(|_| LieError::LabelOverflow)?;
            return Ok(self.torus_label(q));
        }
        if dynkin
            .iter()
            .try_fold(0i64, |sum, &x| sum.checked_add(x))
            .is_none()
        {
            return Err(LieError::LabelOverflow);
        }
        if self.roots.is_some() {
            return Ok(LieIrrep {
                group: self.group.clone(),
                payload: LiePayload::Simple(simple_label(self.roots().irrep(dynkin)?)?),
            });
        }
        let [dj] = dynkin else {
            return Err(LieError::InvalidLabel);
        };
        let dj = u32::try_from(*dj).map_err(|_| LieError::LabelOverflow)?;
        Ok(LieIrrep {
            group: self.group.clone(),
            payload: LiePayload::RankOne(dj),
        })
    }
    fn roots(&self) -> &Arc<general::roots::CartanData> {
        self.engine().data()
    }
    fn engine(&self) -> &general::GeneralFamily {
        // Large group identities stay cheap until a label needs exact root mathematics.
        self.roots
            .as_ref()
            .expect("simple objects are constructed only with a root context")
            .get_or_init(|| {
                general::GeneralFamily::with_data(general::roots::CartanData::new(
                    self.group.root_systems()[0],
                ))
            })
    }
    fn is_torus(&self) -> bool {
        self.composite.is_none() && self.group.root_systems().is_empty()
    }
    fn torus_label(&self, q: U1Irrep) -> LieIrrep {
        LieIrrep {
            group: self.group.clone(),
            payload: LiePayload::Torus(q),
        }
    }
    fn enumeration(&self) -> Result<LieEnumeration, EnumerationError> {
        match self.group.root_systems()[0] {
            // The rank-one center is Z2: a form admits either every doubled
            // spin or exactly the even ones.
            RootSystem::A(1) | RootSystem::B(1) | RootSystem::C(1) => Ok(LieEnumeration::RankOne {
                stride: if self.group.admits(&[1], &[]) { 1 } else { 2 },
            }),
            // SU(N)/Z_1 admits every weight, as SU(N) does.
            RootSystem::A(_) if self.group.is_cover() => Ok(LieEnumeration::Dynkin),
            _ => Err(EnumerationError::NotEnumerable),
        }
    }
    fn enumerable_product(&self) -> Result<Option<&Product<Vec<LieGroup>>>, EnumerationError> {
        if self.composite.is_some() && !self.group.is_cover() {
            return Err(EnumerationError::NotEnumerable);
        }
        Ok(self.composite.as_ref())
    }
    #[inline]
    fn label(&self, a: &LieIrrep) -> u32 {
        assert_eq!(self.group, a.group, "foreign Lie object");
        a.dj()
    }
}
impl LieIrrep {
    #[inline]
    fn dj(&self) -> u32 {
        match self.payload {
            LiePayload::RankOne(dj) => dj,
            _ => 0,
        }
    }
    fn charge(&self) -> i32 {
        self.torus_label().charge()
    }
    fn factors(&self) -> Option<&Vec<LieIrrep>> {
        match &self.payload {
            LiePayload::Composite(factors) => Some(factors),
            _ => None,
        }
    }
    fn simple(&self) -> Option<&general::Irrep> {
        match &self.payload {
            LiePayload::Simple(a) => Some(a),
            _ => None,
        }
    }

    #[inline]
    fn torus_label(&self) -> U1Irrep {
        match self.payload {
            LiePayload::Torus(q) => q,
            _ => U1Irrep::UNIT,
        }
    }
    /// The spin `j` of an irrep of SU(2) or SO(3), in any rank-one
    /// presentation (as for [`LieGroup::irrep_spin`]); `None` for every other group.
    /// The label is `u32`, checked when the irrep was admitted.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupId, HalfInt, LieGroup};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// assert_eq!(su2.irrep(&[3])?.spin(), Some(HalfInt::new(3, 2)?)); // spin 3/2
    /// let u1 = LieGroup::new(GroupId::u(1)?)?;
    /// assert_eq!(u1.irrep_with_charges(&[], &[1])?.spin(), None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn spin(&self) -> Option<HalfInt> {
        match self.payload {
            LiePayload::RankOne(dj) if is_spin_group(&self.group) => {
                Some(HalfInt::from_doubled(dj))
            }
            _ => None,
        }
    }
    /// Integer torus charges, in the datum's charge-coordinate order.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupId, LieGroup};
    ///
    /// let t2 = LieGroup::new(GroupId::torus(2)?)?;
    /// let q = t2.irrep_with_charges(&[], &[3, -1])?;
    /// assert_eq!(q.charges().collect::<Vec<_>>(), [3, -1]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn charges(&self) -> impl ExactSizeIterator<Item = i64> + '_ {
        (0..self.group.torus_rank()).map(|i| {
            self.factors().map_or(i64::from(self.charge()), |factors| {
                i64::from(factors[factors.len() - self.group.torus_rank() + i].charge())
            })
        })
    }
    /// GT basis in the frozen gauge order; absent for rank-one, B/C/D, and composite groups.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupId, LieGroup};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// assert_eq!(su2.irrep(&[1])?.gt_patterns(), None);
    /// # #[cfg(feature = "lie-generated")] {
    /// let su3 = LieGroup::new(GroupId::su(3)?)?;
    /// let patterns = su3.irrep(&[1, 1])?.gt_patterns().unwrap();
    /// assert_eq!(patterns.len(), 8); // one pattern per basis vector of the 8
    /// # }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn gt_patterns(&self) -> Option<Vec<GtPattern>> {
        let a = self
            .simple()
            .filter(|a| matches!(a.root(), RootSystem::A(_)))?;
        Some(
            a.roots()
                .gt_patterns(a.dynkin())
                .expect("an admitted A-series label has representable partition rows"),
        )
    }

    /// Iterate over every highest-weight Dynkin coordinate, including zeros.
    /// Construction and consumption allocate nothing. Composite coordinates
    /// follow factor order; torus and trivial labels yield none, and charges
    /// are separate ([`Self::charges`]). `len()` reports the number of
    /// remaining coordinates. Collect only where an owned label is needed.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupId, LieGroup};
    ///
    /// let u2 = LieGroup::new(GroupId::u(2)?)?;
    /// let x = u2.irrep_with_charges(&[2], &[4])?;
    /// assert!(x.dynkin().eq([2])); // charges are separate
    /// let spin4 = LieGroup::new(GroupId::spin(4)?)?;
    /// let y = spin4.irrep(&[2, 0])?;
    /// assert_eq!(y.dynkin().len(), 2);
    /// assert_eq!(y.dynkin().collect::<Vec<_>>(), [2, 0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn dynkin(&self) -> impl ExactSizeIterator<Item = i64> + '_ {
        DynkinLabels {
            irrep: self,
            indices: 0..self.group.semisimple_rank(),
        }
    }

    /// Exact dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupId, LieGroup};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// assert_eq!(su2.irrep(&[3])?.dim_exact(), 4u32); // 2j + 1
    /// let spin4 = LieGroup::new(GroupId::spin(4)?)?;
    /// assert_eq!(spin4.irrep(&[1, 2])?.dim_exact(), 6u32); // 2 · 3
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn dim_exact(&self) -> Natural {
        if let Some(factors) = self.factors() {
            return factors.iter().map(Self::dim_exact).product();
        }
        if let Some(a) = self.simple() {
            return a.dim().clone();
        }
        Natural::from(u64::from(self.dj()) + 1)
    }
    /// Weight multiplicities in Dynkin-then-charge coordinates, in ascending order.
    /// Panics if an intermediate weight or multiplicity is unrepresentable.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupId, LieGroup};
    /// use std::collections::BTreeMap;
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// let spin_one = su2.irrep(&[2])?;
    /// // Weights 2m for m = -1, 0, 1, each once.
    /// let expected = BTreeMap::from([(vec![-2], 1), (vec![0], 1), (vec![2], 1)]);
    /// assert_eq!(spin_one.weight_multiplicities(), expected);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn weight_multiplicities(&self) -> BTreeMap<Vec<i64>, u64> {
        if let Some(factors) = self.factors() {
            let mut weights = BTreeMap::from([(Vec::new(), 1u64)]);
            for factor in factors.iter() {
                let next = factor.weight_multiplicities();
                weights = weights
                    .into_iter()
                    .flat_map(|(prefix, n)| {
                        next.iter().map(move |(suffix, m)| {
                            let mut key = prefix.clone();
                            key.extend(suffix);
                            (
                                key,
                                n.checked_mul(*m)
                                    .expect("composite weight multiplicity exceeds u64"),
                            )
                        })
                    })
                    .collect();
            }
            return weights;
        }
        if let Some(a) = self.simple() {
            return a
                .weight_multiplicities(&a.roots())
                .expect("infallible weight_multiplicities failed: a weight exceeds i64");
        }

        if self.group.semisimple_rank() == 0 {
            return BTreeMap::from([(vec![i64::from(self.charge())], 1)]);
        }
        (0..=self.dj())
            .map(|i| (vec![i64::from(self.dj()) - 2 * i64::from(i)], 1))
            .collect()
    }
}

/// Clebsch–Gordan data read through [`CgcBlock`], with axes `[dim(a), dim(b), dim(c), N(a,b,c)]`.
/// Every simple factor indexes magnetic states as in `docs/gauge_cartan.md` §3:
/// A(1) by `m + j` (lowest weight first), higher A-series by the frozen GT
/// pattern order of [`LieIrrep::gt_patterns`], and B, C, D, G2, F4, E6, E7 and
/// E8 by depth order (highest weight first). Composite CGCs use Product's
/// last-factor-fastest decoding on every axis; D(2) factors are α₁ then α₂ in
/// the exact A(1) gauge. A block shares the retained sparse channel; reading
/// an entry is a binary search. [`crate::product::Product`] returns this
/// block too: each entry of a product of exact factors is the exact product
/// of the factors' exact entries, rounded once.
#[derive(Clone, Debug, PartialEq)]
pub struct LieCgc {
    dims: [usize; 4],
    // The exactly nonzero entries in §4.7 order, which is ascending
    // `(μ, m_c, m_a, m_b)`; `None` for a scalar (`1` when present).
    entries: Option<Arc<general::cgc::CgcChannel>>,
}
impl LieCgc {
    /// The scalar block `[1, 1, 1, n]` (or another absent shape): `1` when
    /// present.
    pub(crate) fn scalar(dims: [usize; 4]) -> Self {
        Self {
            dims,
            entries: None,
        }
    }
    /// The block of a shared sparse channel.
    pub(crate) fn sparse(dims: [usize; 3], channel: Arc<general::cgc::CgcChannel>) -> Self {
        Self {
            dims: [dims[0], dims[1], dims[2], channel.mult],
            entries: Some(channel),
        }
    }
}
impl CgcBlock for LieCgc {
    /// Axis lengths, with multiplicity last (zero for an absent channel).
    fn dims(&self) -> [usize; 4] {
        self.dims
    }
    /// Read one coefficient.
    ///
    /// # Panics
    /// Panics if any index is outside its axis.
    fn get(&self, [m1, m2, m3, mu]: [usize; 4]) -> f64 {
        assert!(
            [m1, m2, m3, mu].iter().zip(self.dims).all(|(&i, n)| i < n),
            "CGC index out of range"
        );
        let Some(channel) = &self.entries else {
            // A scalar uses no coefficient allocation.
            return 1.0;
        };
        // In range, so every index fits the u32 storage.
        let key = [mu, m3, m1, m2].map(|i| i as u32);
        let entries = &channel.entries;
        entries
            .binary_search_by_key(&key, |e| [e.mu, e.m3, e.m1, e.m2])
            .map_or(0.0, |i| entries[i].value)
    }
    fn __fusion_symbols_stored(&self, each: &mut dyn FnMut([usize; 4], f64)) {
        let Some(channel) = &self.entries else {
            return crate::category::read_stored(self, each);
        };
        for e in &channel.entries {
            each([e.m1, e.m2, e.m3, e.mu].map(|i| i as usize), e.value);
        }
    }
}
impl FusionCategory for LieGroup {
    type Object = LieIrrep;
    type Scalar = f64;
    type FSymbol = FBlock<f64>;
    type RSymbol = RBlock<f64>;
    type Fusion<'a> = LieFusion;
    type Error = LieError;
    fn fusion_style(&self) -> FusionStyle {
        if let Some(product) = &self.composite {
            return product.fusion_style();
        }
        if self.is_torus() {
            FusionStyle::Unique
        } else if self.group.semisimple_rank() >= 2 {
            FusionStyle::Generic
        } else {
            FusionStyle::Simple
        }
    }
    fn braiding_style(&self) -> BraidingStyle {
        if let Some(product) = &self.composite {
            return product.braiding_style();
        }
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> LieIrrep {
        if let Some(product) = &self.composite {
            return Self::composite_label(&self.group, product.unit());
        }
        if self.roots.is_some() {
            return self
                .unit
                .get_or_init(|| {
                    self.irrep(&vec![0; self.group.semisimple_rank()])
                        .expect("zero Dynkin labels have zero weight and are admitted by the constructed group")
                })
                .clone();
        }
        LieIrrep {
            group: self.group.clone(),
            payload: if self.is_torus() {
                LiePayload::Torus(U1Irrep::UNIT)
            } else {
                LiePayload::RankOne(0)
            },
        }
    }
    // The group is the whole identity; labels are validated at construction.
    fn contains(&self, a: &LieIrrep) -> bool {
        a.group == self.group
    }
    fn dual(&self, a: &LieIrrep) -> LieIrrep {
        self.label(a);
        if let Some(product) = &self.composite {
            return Self::composite_label(&self.group, product.dual(self.factor_labels(a)));
        }
        if self.is_torus() {
            return self.torus_label(U1.dual(&a.torus_label()));
        }
        if let Some(x) = a.simple() {
            return LieIrrep {
                group: self.group.clone(),
                payload: LiePayload::Simple(x.dual()),
            };
        }
        a.clone()
    }
    #[inline(always)]
    fn dim(&self, a: &LieIrrep) -> f64 {
        self.label(a);
        match &a.payload {
            LiePayload::RankOne(dj) => f64::from(*dj) + 1.0,
            LiePayload::Torus(_) => 1.0,
            LiePayload::Composite(factors) => self
                .composite
                .as_ref()
                .expect("composite object has a product")
                .dim(factors),
            LiePayload::Simple(x) => x.dim_f64(),
        }
    }
    // Lie dimensions are integers: no numeric fallback.
    fn sqrt_dim(&self, a: &LieIrrep) -> f64 {
        self.derived_dim(a, DimValue::Sqrt)
    }
    fn inv_sqrt_dim(&self, a: &LieIrrep) -> f64 {
        self.derived_dim(a, DimValue::InvSqrt)
    }
    fn inv_dim(&self, a: &LieIrrep) -> f64 {
        self.derived_dim(a, DimValue::Inv)
    }
    /// A complex A-series label has the closed form; a complex D(r) (r odd)
    /// or E6 label generates σ_a σ_ā (`docs/gauge_cartan.md` §5.3), and a
    /// failure there is the engine's typed [`LieError`], never a substituted
    /// phase.
    fn frobenius_schur_phase(&self, a: &LieIrrep) -> Result<f64, LieError> {
        self.label(a);
        if let Some(product) = &self.composite {
            return product
                .frobenius_schur_phase(self.factor_labels(a))
                .map_err(composite_error);
        }
        if let Some(x) = a.simple() {
            if x.is_self_dual() {
                return Ok(f64::from(x.sign()));
            }
            return self
                .engine()
                .frobenius_schur_phase(x.label(), x.dual().label())
                .map(f64::from);
        }
        Ok(su2::fs_phase(a.dj()))
    }
    // Reads the stored self-dual sign (or the A1 formula) directly: a complex
    // irrep's indicator is zero without generating its phase.
    fn frobenius_schur_indicator(&self, a: &LieIrrep) -> f64 {
        self.label(a);
        if let Some(product) = &self.composite {
            return product.frobenius_schur_indicator(self.factor_labels(a));
        }
        if self.is_torus() {
            return U1.frobenius_schur_indicator(&a.torus_label());
        }
        if let Some(x) = a.simple() {
            return if x.is_self_dual() {
                f64::from(x.sign())
            } else {
                0.0
            };
        }
        su2::fs_phase(a.dj())
    }
    fn twist(&self, a: &LieIrrep) -> Result<f64, LieError> {
        self.label(a);
        if let Some(product) = &self.composite {
            return product
                .twist(self.factor_labels(a))
                .map_err(composite_error);
        }
        // Bosonic, twist one (`docs/gauge_cartan.md` §6.1).
        if a.simple().is_some() {
            return Ok(1.0);
        }
        Ok(su2::twist(a.dj()))
    }
    fn fusion<'a>(&'a self, a: &LieIrrep, b: &LieIrrep) -> Result<Self::Fusion<'a>, LieError> {
        self.label(a);
        self.label(b);
        if let Some(product) = &self.composite {
            return Ok(LieFusion {
                group: Some(self.group.clone()),
                inner: FusionInner::Composite(
                    product
                        .fusion(self.factor_labels(a), self.factor_labels(b))
                        .map_err(composite_error)?
                        .into_cartesian(),
                ),
            });
        }
        if self.is_torus() {
            let q = U1
                .fusion(&a.torus_label(), &b.torus_label())
                .map_err(|_| LieError::LabelOverflow)?
                .next()
                .expect("torus fusion has one outcome")
                .0;
            return Ok(LieFusion {
                group: Some(self.group.clone()),
                inner: FusionInner::Torus(Some(q)),
            });
        }
        if let (Some(a), Some(b)) = (a.simple(), b.simple()) {
            let product = self.roots().product(a, b)?;
            let mut max = 0;
            for (c, n) in product.iter() {
                simple_label(c.clone())?;
                max = max.max(*n);
            }
            let back = product.len();
            return Ok(LieFusion {
                group: Some(self.group.clone()),
                inner: FusionInner::Simple {
                    product,
                    max,
                    front: 0,
                    back,
                },
            });
        }
        let (a, b) = (a.dj(), b.dj());
        let inner = su2::A1Irrep::new(a).fusion(su2::A1Irrep::new(b))?;
        Ok(LieFusion {
            group: Some(self.group.clone()),
            inner: FusionInner::A1(inner),
        })
    }
    fn __fusion_symbols_max_multiplicity(&self, fusion: &LieFusion, _: Validated) -> Option<u32> {
        Some(fusion.max_multiplicity())
    }
    /// Generated fusion checks label/weight sums and multiplicities, returning
    /// [`LieError::LabelOverflow`] on overflow. Torus N is exact even when the
    /// sum is not a representable label. Rank-one N is always zero or one.
    #[inline]
    fn nsymbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<u32, LieError> {
        self.label(a);
        self.label(b);
        self.label(c);
        self.__fusion_symbols_validated_nsymbol(a, b, c, Validated(()))
    }
    #[inline(always)]
    fn __fusion_symbols_validated_nsymbol(
        &self,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
        _: Validated,
    ) -> Result<u32, LieError> {
        if let Some(product) = &self.composite {
            return product
                .__fusion_symbols_validated_nsymbol(
                    self.factor_labels(a),
                    self.factor_labels(b),
                    self.factor_labels(c),
                    Validated(()),
                )
                .map_err(composite_error);
        }
        match (&a.payload, &b.payload, &c.payload) {
            (LiePayload::Torus(a), LiePayload::Torus(b), LiePayload::Torus(c)) => {
                return Ok(u1::conserves(a, b, c))
            }
            (LiePayload::RankOne(a), LiePayload::RankOne(b), LiePayload::RankOne(c)) => {
                return Ok(u32::from(su2::triangle_ok(*a, *b, *c)))
            }
            _ => {}
        }
        if let (Some(a), Some(b), Some(c)) = (a.simple(), b.simple(), c.simple()) {
            return self.roots().multiplicity(a, b, c);
        }
        unreachable!("higher-rank labels have a generated payload")
    }
    fn f_symbol(
        &self,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
        d: &LieIrrep,
        e: &LieIrrep,
        f: &LieIrrep,
    ) -> Result<FBlock<f64>, LieError> {
        self.f_symbol_in([a, b, c, d, e, f], None)
    }
    fn r_symbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<RBlock<f64>, LieError> {
        self.label(a);
        self.label(b);
        self.label(c);
        if let Some(product) = &self.composite {
            return product
                .r_symbol(
                    self.factor_labels(a),
                    self.factor_labels(b),
                    self.factor_labels(c),
                )
                .map_err(composite_error);
        }
        if matches!(a.payload, LiePayload::Torus(_)) {
            return Ok(
                if self.__fusion_symbols_validated_nsymbol(a, b, c, Validated(()))? == 0 {
                    RBlock::zeros(0)
                } else {
                    RBlock::one(1.0)
                },
            );
        }
        if matches!(a.payload, LiePayload::RankOne(_)) {
            return Ok(if su2::triangle_ok(a.dj(), b.dj(), c.dj()) {
                RBlock::one(su2::r_symbol(a.dj(), b.dj(), c.dj()))
            } else {
                RBlock::zeros(0)
            });
        }
        if let (Some(a), Some(b), Some(c)) = (a.simple(), b.simple(), c.simple()) {
            return self.engine().r_symbol([a.label(), b.label(), c.label()]);
        }
        unreachable!("higher-rank labels have a generated payload")
    }
    #[doc(hidden)]
    fn __fusion_symbols_exact_symbol(
        &self,
        op: crate::category::sealed::ExactOp<'_, '_, LieIrrep>,
        t: Validated,
    ) -> Result<crate::category::sealed::ExactAnswer, LieError> {
        if let Some(product) = &self.composite {
            return product
                .__fusion_symbols_exact_symbol(op.map(|x| self.factor_labels(x)), t)
                .map_err(composite_error);
        }
        if self.is_torus() {
            return crate::category::unit_symbol(self, op);
        }
        self.exact_leaf(op)
    }
    #[doc(hidden)]
    #[inline(always)]
    fn __fusion_symbols_exact_dim(
        &self,
        a: &LieIrrep,
        d: &mut crate::category::sealed::DimAcc,
        t: Validated,
    ) -> bool {
        self.label(a);
        match &a.payload {
            LiePayload::RankOne(dj) => d.mul_word(u64::from(*dj) + 1),
            LiePayload::Torus(_) => {}
            LiePayload::Simple(x) => match u64::try_from(x.dim()) {
                Ok(w) => d.mul_word(w),
                Err(_) => d.mul(Factor::Natural(x.dim())),
            },
            LiePayload::Composite(factors) => return self.composite_dim(factors, d, t),
        }
        true
    }
    fn a_symbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<RBlock<f64>, LieError> {
        self.ab_symbol(false, a, b, c)
    }
    fn b_symbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<RBlock<f64>, LieError> {
        self.ab_symbol(true, a, b, c)
    }
    /// Tori and the trivial group certify a trivial associator. Semisimple
    /// factors have an irrep of dimension d > 1 with `|F^{a ā a}_a[𝟙,𝟙]| = 1/d`.
    fn has_trivial_associator_gauge(&self) -> bool {
        if let Some(product) = &self.composite {
            return product.has_trivial_associator_gauge();
        }
        self.is_torus()
    }
    // `true` for every group. The unit is the trivial irrep: self-dual, and
    // `𝟙 ⊗ a = a ⊗ 𝟙 = a` once. SU(2): the exact 6j closed form gives
    // `F^{𝟙bc}_d = F^{a𝟙c}_d = F^{ab𝟙}_d = (-1)^{2(j_1+j_2+j_3)} = 1` (integer
    // triangle sum), rounded exactly. Higher-rank groups contract four CGC
    // exactly (`frcore::f_exact`); with a unit leg two of them are the exact
    // identity embeddings of `𝟙 ⊗ s → s` / `s ⊗ 𝟙 → s`
    // (`docs/gauge_cartan.md` §4.5) and the other two are the same isometry
    // `x ⊗ y → z`, so the block is its column orthonormality
    // `Σ C_κ C_ν = δ_{κν}`: the identity, exactly.
    fn has_canonical_unit(&self) -> bool {
        if let Some(product) = &self.composite {
            return product.has_canonical_unit();
        }
        true
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        if let Some(product) = self.enumerable_product()? {
            return product.object_count();
        }
        if !self.is_torus() {
            self.enumeration()?;
        }
        Ok(None)
    }
    fn object_at(&self, index: u64) -> Result<LieIrrep, EnumerationError> {
        if let Some(product) = self.enumerable_product()? {
            return product
                .object_at(index)
                .map(|factors| Self::composite_label(&self.group, factors));
        }
        if self.is_torus() {
            return U1.object_at(index).map(|q| self.torus_label(q));
        }
        let labels = match self.enumeration()? {
            LieEnumeration::RankOne { stride } => {
                let dj = index
                    .checked_mul(stride)
                    .and_then(|dj| u32::try_from(dj).ok())
                    .ok_or(EnumerationError::Overflow)?;
                vec![i64::from(dj)]
            }
            LieEnumeration::Dynkin => {
                let bounds = vec![None; self.group.semisimple_rank()];
                crate::category::unrank(&bounds, &Integer::from(index))
                    .ok_or(EnumerationError::OutOfRange)?
                    .into_iter()
                    .map(|a| i64::try_from(&a).map_err(|_| EnumerationError::Overflow))
                    .collect::<Result<_, _>>()?
            }
        };
        // The labels are nonnegative, of the right rank and admitted, so only
        // `LieError::LabelOverflow` remains.
        self.irrep(&labels).map_err(|_| EnumerationError::Overflow)
    }
    fn object_index(&self, a: &LieIrrep) -> Result<u64, EnumerationError> {
        let dj = self.label(a);
        if let Some(product) = self.enumerable_product()? {
            return product.object_index(self.factor_labels(a));
        }
        if self.is_torus() {
            return U1.object_index(&a.torus_label());
        }
        match self.enumeration()? {
            LieEnumeration::RankOne { stride } => Ok(u64::from(dj) / stride),
            LieEnumeration::Dynkin => {
                let bounds = vec![None; self.group.semisimple_rank()];
                let x: Vec<Integer> = a.dynkin().map(Integer::from).collect();
                u64::try_from(&crate::category::rank(&bounds, &x))
                    .map_err(|_| EnumerationError::Overflow)
            }
        }
    }
}
/// The bodies of the serial entries. A cold channel they reach runs its
/// descent and isometry gate on `exec`; the entries pass `None`, the serial
/// token, so a warm query never builds a token.
impl LieGroup {
    /// `A^{ab}_c` (`is_b = false`) or `B^{ab}_c` of a simple or rank-one
    /// group (`docs/gauge_cartan.md` §6.1): the rounded view of
    /// `√(d_a d_b/d_c)·κ_a·F^{ā a b}_b[𝟙, c]` (A) or
    /// `√(d_a d_b/d_c)·F^{a b b̄}_a[c, 𝟙]` (B), formed exactly from the exact F
    /// of the unit-leg sextet and rounded once. A torus has the closed form
    /// `1` (or `0 × 0`); a composite group is its product's.
    fn ab_symbol(
        &self,
        is_b: bool,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
    ) -> Result<RBlock<f64>, LieError> {
        // Validates every object before the lookup: a hit must not bypass it.
        let words = [a, b, c].map(|x| [i64::from(self.label(x))]);
        if let Some(product) = &self.composite {
            let [a, b, c] = [a, b, c].map(|x| self.factor_labels(x));
            let block = if is_b {
                product.b_symbol(a, b, c)
            } else {
                product.a_symbol(a, b, c)
            };
            return block.map_err(composite_error);
        }
        let labels = match [a, b, c].map(|x| x.simple()) {
            [Some(a), Some(b), Some(c)] => [a.label(), b.label(), c.label()],
            _ if matches!(a.payload, LiePayload::RankOne(_)) => {
                words.each_ref().map(|w| crate::cache::Label::new(w))
            }
            _ => {
                let n = self.__fusion_symbols_validated_nsymbol(a, b, c, Validated(()))?;
                return Ok(if n == 0 {
                    RBlock::zeros(0)
                } else {
                    RBlock::one(1.0)
                });
            }
        };
        let root = self.group.root_systems()[0];
        crate::cache::ab_block(is_b, root, labels, || {
            let block = self.ab_exact(is_b, a, b, c)?;
            Ok(RBlock::from_vec(
                block.dims()[0],
                crate::rounding::blocks::round_block(&block),
            ))
        })
    }

    /// The exact A (`is_b = false`) or B of a simple or rank-one group, from
    /// the exact F of its unit-leg sextet; `0 × 0` when absent.
    fn ab_exact(
        &self,
        is_b: bool,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
    ) -> Result<crate::exact::radical::RadicalBlock<2>, LieError> {
        use crate::exact::radical::RadicalBlock;
        let n = checked_dimension(self.nsymbol(a, b, c)?)?;
        checked_size([n, n])?;
        if n == 0 {
            return RadicalBlock::zeros([0, 0]).ok_or(LieError::LabelOverflow);
        }
        let (f, dims) = match [a, b, c].map(|x| x.simple()) {
            [Some(xa), Some(xb), Some(xc)] => {
                let fam = self.engine();
                let (unit, bar) = (self.unit(), self.dual(if is_b { b } else { a }));
                let [unit, bar] = [&unit, &bar].map(|x| x.simple().map(general::Irrep::label));
                let (Some(unit), Some(bar)) = (unit, bar) else {
                    unreachable!("a simple group's unit and duals are simple")
                };
                let [la, lb, lc] = [xa.label(), xb.label(), xc.label()];
                let sextet = if is_b {
                    [la, lb, bar, la, lc, unit]
                } else {
                    [bar, la, lb, lb, unit, lc]
                };
                let [da, db, dc] = [la, lb, lc].map(|l| fam.dimension(l));
                (fam.f_exact(sextet)?, [da?, db?, dc?])
            }
            _ => {
                let [ja, jb, jc] = [a, b, c].map(LieIrrep::dj);
                // Rank-one irreps are self-dual and the unit is spin 0.
                let x = if is_b {
                    su2::f_symbol_exact(ja, jb, jb, ja, jc, 0)
                } else {
                    su2::f_symbol_exact(ja, ja, jb, jb, 0, jc)
                };
                let dims = [ja, jb, jc].map(|j| j as usize + 1);
                (Arc::new(RadicalBlock::from_signed_sqrt(&x)), dims)
            }
        };
        let negate = !is_b && self.frobenius_schur_phase(a)? < 0.0;
        let ratio = crate::exact::radical::Radicand::new(
            Natural::from(dims[0] as u64) * Natural::from(dims[1] as u64),
            Natural::from(dims[2] as u64),
        )
        .ok_or(LieError::LabelOverflow)?;
        let (rows, cols) = if is_b { (0, 1) } else { (2, 3) };
        f.scaled(negate, &ratio)
            .matrix(rows, cols)
            .filter(|m| m.dims() == [n, n])
            .ok_or(LieError::LabelOverflow)
    }

    /// The exact dimension of a composite object, out of line so that a
    /// simple or rank-one factor's hook stays small enough to inline.
    #[inline(never)]
    fn composite_dim(
        &self,
        factors: &Vec<LieIrrep>,
        d: &mut crate::category::sealed::DimAcc,
        t: Validated,
    ) -> bool {
        self.composite
            .as_ref()
            .expect("composite object has a product")
            .__fusion_symbols_exact_dim(factors, d, t)
    }

    /// One load for a simple irrep (rounded when it is made), IEEE single
    /// operations or [`crate::rounding::round_dim`]'s word path for rank one.
    fn derived_dim(&self, a: &LieIrrep, value: DimValue) -> f64 {
        self.label(a);
        match &a.payload {
            LiePayload::RankOne(dj) => round_dim(Factor::Words(&[u64::from(*dj) + 1]), value),
            LiePayload::Torus(_) => 1.0,
            LiePayload::Simple(x) => x.derived_dim(value),
            LiePayload::Composite(factors) => {
                let product = self
                    .composite
                    .as_ref()
                    .expect("composite object has a product");
                crate::category::exact_dim(product, factors, value)
                    .expect("a Lie dimension is exact")
            }
        }
    }

    /// The exact product hook of a simple or rank-one group (a torus is
    /// unit-class; a composite delegates to its product).
    fn exact_leaf<'q>(
        &self,
        op: crate::category::sealed::ExactOp<'q, '_, LieIrrep>,
    ) -> Result<crate::category::sealed::ExactAnswer, LieError> {
        use crate::category::sealed::{ExactAnswer, ExactOp, Query, SymbolKind, Unit, View};
        let simple = self.roots.is_some();
        if !simple
            && matches!(
                op,
                ExactOp::Class(SymbolKind::R) | ExactOp::Describe(Query::R(_), _)
            )
        {
            // Rank-one R is `±1` (`su2::r_symbol`), or absent.
            return crate::category::unit_symbol(self, op);
        }
        let leaf = |xs: &[&'q LieIrrep]| self.leaf_ref(xs);
        match op {
            ExactOp::Class(_) => Ok(ExactAnswer::RadicalClass { leaves: 1 }),
            ExactOp::Describe(q, buf) => {
                buf.0.push(match q {
                    Query::F(x) => leaf(&x),
                    Query::R(x) | Query::A(x) | Query::B(x) => leaf(&x),
                });
                Ok(ExactAnswer::Unit(Unit::PlusOne))
            }
            ExactOp::Fetch(q, buf) => {
                match q {
                    Query::F([a, b, c, d, e, f]) => {
                        buf.f.push(match [a, b, c, d, e, f].map(|x| x.simple()) {
                            [Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)] => self
                                .engine()
                                .f_exact([a, b, c, d, e, f].map(general::Irrep::label))?,
                            _ => {
                                let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(LieIrrep::dj);
                                let x = su2::f_symbol_exact(a, b, c, d, e, f);
                                Arc::new(crate::exact::radical::RadicalBlock::from_signed_sqrt(&x))
                            }
                        })
                    }
                    Query::R([a, b, c]) => match [a, b, c].map(|x| x.simple()) {
                        [Some(a), Some(b), Some(c)] => buf.r2.push(
                            self.engine()
                                .r_exact([a, b, c].map(general::Irrep::label))?,
                        ),
                        _ => unreachable!("rank-one R is unit-class"),
                    },
                    Query::A([a, b, c]) => buf.r2.push(Arc::new(self.ab_exact(false, a, b, c)?)),
                    Query::B([a, b, c]) => buf.r2.push(Arc::new(self.ab_exact(true, a, b, c)?)),
                }
                Ok(ExactAnswer::Done)
            }
            ExactOp::View(q, out) => {
                *out = Some(match q {
                    Query::F([a, b, c, d, e, f]) => View::F(self.f_symbol(a, b, c, d, e, f)?),
                    Query::R([a, b, c]) => View::R(self.r_symbol(a, b, c)?),
                    Query::A([a, b, c]) => View::R(self.a_symbol(a, b, c)?),
                    Query::B([a, b, c]) => View::R(self.b_symbol(a, b, c)?),
                });
                Ok(ExactAnswer::Done)
            }
        }
    }

    #[inline]
    pub(crate) fn f_symbol_in(
        &self,
        labels: [&LieIrrep; 6],
        exec: Option<&mut Exec<'_>>,
    ) -> Result<FBlock<f64>, LieError> {
        let [a, b, c, d, e, f] = labels;
        for x in [a, b, c, d, e, f] {
            self.label(x);
        }
        if let Some(product) = &self.composite {
            return product
                .f_symbol(
                    self.factor_labels(a),
                    self.factor_labels(b),
                    self.factor_labels(c),
                    self.factor_labels(d),
                    self.factor_labels(e),
                    self.factor_labels(f),
                )
                .map_err(composite_error);
        }
        if matches!(a.payload, LiePayload::Torus(_)) {
            let mut dims = [0; 4];
            for (n, (a, b, c)) in dims
                .iter_mut()
                .zip([(a, b, e), (e, c, d), (b, c, f), (a, f, d)])
            {
                *n = u1::conserves(&a.torus_label(), &b.torus_label(), &c.torus_label()) as usize;
            }
            return Ok(if dims.contains(&0) {
                FBlock::zeros(dims)
            } else {
                FBlock::one(1.0)
            });
        }
        if matches!(a.payload, LiePayload::RankOne(_)) {
            let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|x| x.dj());
            let dims = [(a, b, e), (e, c, d), (b, c, f), (a, f, d)]
                .map(|(a, b, c)| usize::from(su2::triangle_ok(a, b, c)));
            if dims.contains(&0) {
                return Ok(FBlock::zeros(dims));
            }
            return Ok(FBlock::one(su2::f_symbol(a, b, c, d, e, f)?));
        }
        if let [Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)] =
            [a, b, c, d, e, f].map(|x| x.simple())
        {
            return self
                .engine()
                .f_symbol([a, b, c, d, e, f].map(|x| x.label()), exec);
        }
        unreachable!("higher-rank labels have a generated payload")
    }

    #[inline]
    pub(crate) fn cgc_in(
        &self,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
        exec: Option<&mut Exec<'_>>,
    ) -> Result<LieCgc, LieError> {
        self.label(a);
        self.label(b);
        self.label(c);
        if let Some(product) = &self.composite {
            return product
                .cgc(
                    self.factor_labels(a),
                    self.factor_labels(b),
                    self.factor_labels(c),
                )
                .map_err(composite_error);
        }
        if self.is_torus() {
            let n = self.__fusion_symbols_validated_nsymbol(a, b, c, Validated(()))?;
            return Ok(LieCgc::scalar([1, 1, 1, n as usize]));
        }
        let dims = self.leaf_cgc_dims(a, b, c)?;
        let channel = self.with_cgc_labels(a, b, c, |engine, [a, b, c]| {
            engine.cgc_entries(a, b, c, exec)
        })?;
        Ok(LieCgc::sparse(dims, channel))
    }

    /// The magnetic dimensions of a simple or rank-one CGC query; rank-one
    /// triangle-admissible labels beyond `i32` are an unsupported capability
    /// rather than an unbounded generation.
    fn leaf_cgc_dims(
        &self,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
    ) -> Result<[usize; 3], LieError> {
        if let [Some(a), Some(b), Some(c)] = [a, b, c].map(LieIrrep::simple) {
            let [da, db, dc] = [a, b, c].map(|x| checked_dimension(x.dim()));
            return Ok([da?, db?, dc?]);
        }
        let (a, b, c) = (a.dj(), b.dj(), c.dj());
        if su2::triangle_ok(a, b, c) && [a, b, c].iter().any(|&dj| dj > i32::MAX as u32) {
            return Err(LieError::Unsupported {
                requires: "CGC projections beyond i32",
            });
        }
        let [da, db, dc] = [a, b, c].map(a1_dimension);
        Ok([da?, db?, dc?])
    }

    /// `f` on the Cartan engine and labels of a simple or rank-one CGC query.
    fn with_cgc_labels<R>(
        &self,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
        f: impl FnOnce(&general::GeneralFamily, [crate::cache::Label<'_>; 3]) -> Result<R, LieError>,
    ) -> Result<R, LieError> {
        if let [Some(a), Some(b), Some(c)] = [a, b, c].map(LieIrrep::simple) {
            return f(self.engine(), [a.label(), b.label(), c.label()]);
        }
        let [a, b, c] = [a, b, c].map(|x| [i64::from(x.dj())]);
        f(
            general::a1(),
            [&a, &b, &c].map(|x| crate::cache::Label::new(x)),
        )
    }

    /// The cache key of a simple or rank-one factor's labels in a product
    /// query, read in place.
    fn leaf_ref<'q>(&self, xs: &[&'q LieIrrep]) -> crate::category::sealed::LeafRef<'q> {
        use crate::category::sealed::{LeafLabel, LeafRef, LEAF_A1, LEAF_SIMPLE};
        let family = if self.roots.is_some() {
            LEAF_SIMPLE
        } else {
            LEAF_A1
        };
        LeafRef::new(
            family,
            self.group.root_systems()[0],
            xs.iter().map(|x| match x.simple() {
                Some(l) => {
                    let l = l.label();
                    LeafLabel::Words(l.words, l.hash)
                }
                None => LeafLabel::inline(i64::from(x.dj())),
            }),
        )
    }
}

impl GroupRepresentations for LieGroup {
    type Cgc = LieCgc;
    fn cgc(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<LieCgc, LieError> {
        self.cgc_in(a, b, c, None)
    }
    #[doc(hidden)]
    fn __fusion_symbols_exact_cgc<'q>(
        &self,
        [a, b, c]: [&'q LieIrrep; 3],
        op: crate::category::sealed::CgcOp<'q, '_>,
        t: Validated,
    ) -> Result<bool, LieError> {
        use crate::category::sealed::CgcOp;
        if let Some(product) = &self.composite {
            return product
                .__fusion_symbols_exact_cgc([a, b, c].map(|x| self.factor_labels(x)), op, t)
                .map_err(composite_error);
        }
        if self.is_torus() {
            let n = self.__fusion_symbols_validated_nsymbol(a, b, c, t)?;
            return Ok(crate::category::unit_cgc(n, op));
        }
        match op {
            CgcOp::Describe(leaves) => {
                leaves.times(self.leaf_cgc_dims(a, b, c)?);
                leaves.leaves.0.push(self.leaf_ref(&[a, b, c]));
            }
            CgcOp::Fetch(buf) => {
                buf.0.push(
                    self.with_cgc_labels(a, b, c, |engine, [a, b, c]| engine.cgc_exact(a, b, c))?,
                )
            }
            CgcOp::View(out) => *out = Some(self.cgc_in(a, b, c, None)?),
        }
        Ok(true)
    }
}

/// Deterministic fusion outcomes with their multiplicities, read from the
/// cached product for higher rank.
#[derive(Clone, Debug)]
pub struct LieFusion {
    // Taken by the torus's sole outcome; other iterators retain their group.
    group: Option<GroupId>,
    inner: FusionInner,
}
// The product's fusion state without the borrow of the factor list, which
// `LieGroup`'s owned fusion does not hold.
type CompositeFusion = crate::product::sealed::Cartesian<
    Vec<(LieFusion, usize)>,
    (usize, usize, Vec<(LieIrrep, u32)>),
>;

#[derive(Clone, Debug)]
enum FusionInner {
    Torus(Option<U1Irrep>),
    Composite(CompositeFusion),
    A1(su2::A1Fusion),
    Simple {
        product: general::roots::Product,
        // Largest multiplicity of `product`, for an enclosing product's bound.
        max: u32,
        front: usize,
        back: usize,
    },
}
impl LieFusion {
    fn max_multiplicity(&self) -> u32 {
        match &self.inner {
            FusionInner::Simple { max, .. } => *max,
            FusionInner::Composite(inner) => inner.max_multiplicity(),
            FusionInner::Torus(_) | FusionInner::A1(_) => 1,
        }
    }
    fn next_label(&mut self, reverse: bool) -> Option<(LieIrrep, u32)> {
        let group = self.group.as_ref();
        match &mut self.inner {
            FusionInner::Torus(label) => label
                .take()
                .map(|q| LieIrrep {
                    group: self
                        .group
                        .take()
                        .expect("torus fusion yields its group once"),
                    payload: LiePayload::Torus(q),
                })
                .map(|c| (c, 1)),
            FusionInner::Composite(inner) => (if reverse {
                inner.next_back()
            } else {
                inner.next()
            })
            .map(|(factors, n)| {
                (
                    LieGroup::composite_label(group.expect("fusion retains its group"), factors),
                    n,
                )
            }),
            FusionInner::Simple {
                product,
                front,
                back,
                ..
            } => {
                if *front == *back {
                    return None;
                }
                let i = if reverse {
                    *back -= 1;
                    *back
                } else {
                    let i = *front;
                    *front += 1;
                    i
                };
                Some((
                    LieIrrep {
                        group: group.expect("fusion retains its group").clone(),
                        payload: LiePayload::Simple(product[i].0.clone()),
                    },
                    product[i].1,
                ))
            }
            FusionInner::A1(inner) => (if reverse {
                inner.next_back()
            } else {
                inner.next()
            })
            .map(|a| {
                (
                    LieIrrep {
                        group: group.expect("fusion retains its group").clone(),
                        payload: LiePayload::RankOne(a.dj()),
                    },
                    1,
                )
            }),
        }
    }
}
impl Iterator for LieFusion {
    type Item = (LieIrrep, u32);
    fn next(&mut self) -> Option<(LieIrrep, u32)> {
        self.next_label(false)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match &self.inner {
            FusionInner::Torus(label) => usize::from(label.is_some()),
            FusionInner::Composite(inner) => inner.len(),
            FusionInner::A1(inner) => inner.len(),
            FusionInner::Simple { front, back, .. } => back - front,
        };
        (n, Some(n))
    }
}
impl DoubleEndedIterator for LieFusion {
    fn next_back(&mut self) -> Option<(LieIrrep, u32)> {
        self.next_label(true)
    }
}
impl ExactSizeIterator for LieFusion {}

fn is_spin_group(group: &GroupId) -> bool {
    matches!(
        group.root_systems(),
        [RootSystem::A(1) | RootSystem::B(1) | RootSystem::C(1)]
    ) && group.torus_rank() == 0
}

#[cfg(all(test, feature = "lie-generated"))]
mod tests {
    use super::*;
    use crate::category::SymbolBlock;
    use malachite_q::Rational;
    use std::error::Error;

    #[test]
    fn u128_cast_rounds_like_bigint_to_f64() {
        let mut values = vec![u128::MAX, (1 << 53) - 1];
        for e in 53..128 {
            let p = 1u128 << e;
            // Exact, ties (to even both ways), just below/above ties.
            let half = 1u128 << (e - 53);
            for v in [p, p + half, p + 3 * half, p + half - 1, p + half + 1, p - 1] {
                values.push(v);
            }
        }
        values.extend(
            (0..2000u128).map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835)),
        );
        for v in values {
            let big = f64::rounding_from(&Natural::from(v), Nearest).0;
            assert_eq!((v as f64).to_bits(), big.to_bits(), "{v}");
        }
    }

    /// Whether `y > 0` is the nearest `f64` to the positive real `x`, given
    /// whether `x > m` for a rational `m`: `x` lies strictly between the
    /// midpoints to `y`'s neighbours (no tie occurs for the values below).
    fn nearest(y: f64, above: impl Fn(&Rational) -> bool) -> bool {
        use malachite_base::num::conversion::traits::ExactFrom;
        let q = |k: i64| Rational::exact_from(f64::from_bits(y.to_bits().wrapping_add_signed(k)));
        let two = Rational::from(2u32);
        above(&((q(-1) + q(0)) / &two)) && !above(&((q(0) + q(1)) / &two))
    }

    /// `dim`, `√d`, `1/√d` and `1/d` of `a` are the nearest `f64` to the
    /// exact values of its integer dimension `d`, by rational comparisons.
    fn assert_dims_nearest<C: FusionCategory>(cat: &C, a: &C::Object, d: &Natural) {
        let d = Rational::from(d);
        assert_eq!(cat.dim(a), f64::rounding_from(&d, Nearest).0, "{d}");
        assert!(nearest(cat.sqrt_dim(a), |m| d > m * m), "√{d}");
        assert!(
            nearest(cat.inv_sqrt_dim(a), |m| m * m * &d < 1u32),
            "1/√{d}"
        );
        assert!(nearest(cat.inv_dim(a), |m| m * &d < 1u32), "1/{d}");
    }

    #[test]
    fn derived_and_product_dims_round_once() {
        let su2 = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
        let j = |dj: i64| su2.irrep(&[dj]).unwrap();
        // 1/√2 rounded once; the reciprocal of the rounded √2 is 0x…BCC.
        assert_eq!(su2.inv_sqrt_dim(&j(1)).to_bits(), 0x3FE6_A09E_667F_3BCD);
        // Rank one takes the word path (`inv_sqrt_small` below 2^53).
        for dj in (0..2_000).chain(u32::MAX - 2_000..=u32::MAX) {
            assert_dims_nearest(&su2, &j(i64::from(dj)), &Natural::from(u64::from(dj) + 1));
        }
        let two = crate::lie::CU1Irrep::new(HalfInt::new(1, 2).unwrap(), 2).unwrap();
        assert_eq!(
            crate::lie::CU1.inv_sqrt_dim(&two).to_bits(),
            0x3FE6_A09E_667F_3BCD
        );
        // √14 and 1/√14 of dims 2·7; the products of the factor values are
        // 0x…3F4A and 0x…2429.
        let p2 = Product::new((su2.clone(), su2.clone()));
        let a = (j(1), j(6));
        assert_eq!(p2.sqrt_dim(&a).to_bits(), 0x400D_EEEA_1168_3F49);
        assert_eq!(p2.inv_sqrt_dim(&a).to_bits(), 0x3FD1_1ACE_E560_242A);
        assert_dims_nearest(&p2, &a, &Natural::from(14u32));
        // 424010773·279972654·491870359 > 2^53; the float product of the
        // three dimensions is 0x…D9A6.
        let p3 = Product::new((su2.clone(), su2.clone(), su2.clone()));
        let dims = [424_010_773u64, 279_972_654, 491_870_359];
        let a = (j(424_010_772), j(279_972_653), j(491_870_358));
        assert_eq!(p3.dim(&a).to_bits(), 0x4548_2659_2EF4_D9A7);
        let d: Natural = dims.iter().map(|&x| Natural::from(x)).product();
        assert_dims_nearest(&p3, &a, &d);
        let pv = Product::new(vec![su2.clone(); 3]);
        assert_dims_nearest(&pv, &vec![a.0, a.1, a.2], &d);
        // A composite group's dimensions: Spin(4) = SU(2) × SU(2).
        let spin4 = LieGroup::new(GroupId::spin(4).unwrap()).unwrap();
        let b = spin4.irrep(&[1, 6]).unwrap();
        assert_dims_nearest(&spin4, &b, &Natural::from(14u32));
    }

    #[test]
    fn generated_dim_bits_match_bigint_path() {
        let mut checked = 0;
        let mut above_2_53 = 0;
        let su = |n| GroupId::su(n).unwrap();
        let groups = [
            (
                su(3),
                8_i64,
                vec![(1 << 20) + 1, 6_981_463_658_330, 6_981_463_658_331],
            ),
            (su(4), 4, vec![123_456_789]),
            (su(5), 3, vec![98_765]),
            (su(6), 2, vec![4_321]),
            (
                GroupId::spin(5).unwrap(),
                6,
                vec![(1 << 30) + 1, i64::MAX / 4],
            ),
            (GroupId::spin(7).unwrap(), 4, vec![77_777]),
            (GroupId::sp(6).unwrap(), 4, vec![55_555]),
            (GroupId::spin(8).unwrap(), 3, vec![3_333]),
        ];
        for (id, max, large) in groups {
            let cat = LieGroup::new(id.clone()).unwrap();
            let r = id.semisimple_rank();
            let small = (0..(max + 1).pow(r as u32)).map(|code| {
                (0..r)
                    .map(|i| code / (max + 1).pow(i as u32) % (max + 1))
                    .collect::<Vec<i64>>()
            });
            let big = large.iter().map(|&x| vec![x; r]);
            for d in small.chain(big) {
                let Ok(a) = cat.irrep(&d) else {
                    continue; // spinor label of a non-spin group
                };
                let exact = a.simple().unwrap().dim().clone();
                assert_dims_nearest(&cat, &a, &exact);
                checked += 1;
                above_2_53 += usize::from(exact > (1u64 << 53));
            }
        }
        assert!(checked > 1000 && above_2_53 > 10, "{checked} {above_2_53}");
    }

    #[test]
    fn torus_composites_warm_calls_do_not_generate() {
        let _guard = crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::cache::reset();
        let cover = LieGroup::new(GroupId::cover(&[RootSystem::A(2)], 1).unwrap()).unwrap();
        let quotient = LieGroup::new(GroupId::u(3).unwrap()).unwrap();
        for cat in [&cover, &quotient] {
            let a = cat.irrep_with_charges(&[1, 0], &[1]).unwrap();
            let b = cat.dual(&a);
            let one = cat.unit();
            let query = || {
                cat.cgc(&a, &b, &one).unwrap();
                cat.r_symbol(&a, &b, &one).unwrap();
                cat.f_symbol(&a, &b, &a, &a, &one, &one).unwrap();
            };
            query();
            let before = general::cgc::BUILDS.with(std::cell::Cell::get);
            query();
            assert_eq!(general::cgc::BUILDS.with(std::cell::Cell::get), before);
        }
    }

    #[test]
    fn composite_warm_calls_reuse_cover_caches() {
        let _guard = crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::cache::reset();
        // Cover and quotient share every leaf key of the process-wide tiers.
        let roots = [RootSystem::A(1), RootSystem::A(2), RootSystem::C(2)];
        let cover = LieGroup::new(GroupId::cover(&roots, 0).unwrap()).unwrap();
        let quotient = LieGroup::new(
            GroupId::new(
                &roots,
                0,
                &[
                    vec![2, 0, 0, 0, 0],
                    vec![0, 1, 0, 0, 0],
                    vec![0, 0, 1, 0, 0],
                    vec![0, 0, 0, 1, 0],
                    vec![0, 0, 0, 0, 1],
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let query = |cat: &LieGroup| {
            let a = cat.irrep(&[2, 1, 1, 1, 0]).unwrap();
            let e = cat.irrep(&[2, 1, 1, 0, 0]).unwrap();
            cat.cgc(&a, &a, &e).unwrap();
            cat.f_symbol(&a, &a, &a, &a, &e, &e).unwrap();
            cat.r_symbol(&a, &a, &e).unwrap();
            cat.a_symbol(&a, &a, &e).unwrap();
            cat.b_symbol(&a, &a, &e).unwrap();
        };
        query(&cover);
        let builds = general::cgc::BUILDS.with(std::cell::Cell::get);
        query(&cover);
        query(&quotient);
        assert_eq!(general::cgc::BUILDS.with(std::cell::Cell::get), builds);
    }

    #[test]
    fn composite_error_mapping_retains_factor_path() {
        let error = composite_error(ProductError::Factor {
            index: 2,
            source: LieError::Generation(GenerationError(Arc::new(std::io::Error::other(
                "test failure",
            )))),
        });
        let source = error.source().unwrap().source().unwrap();
        assert!(matches!(
            source.downcast_ref::<CompositeError>(),
            Some(CompositeError(ProductError::Factor { index: 2, .. }))
        ));
        assert!(source
            .source()
            .unwrap()
            .to_string()
            .contains("test failure"));
        assert_eq!(
            composite_error(ProductError::Factor {
                index: 2,
                source: LieError::LabelOverflow
            }),
            LieError::LabelOverflow
        );
        assert_eq!(
            composite_error(ProductError::Overflow),
            LieError::LabelOverflow
        );
        let unsupported = LieError::Unsupported {
            requires: "test capability",
        };
        assert_eq!(
            composite_error(ProductError::Factor {
                index: 2,
                source: unsupported.clone()
            }),
            unsupported
        );
        assert!(matches!(
            composite_error(ProductError::MalformedSymbolData {
                factor: 2,
                expected: vec![1, 1],
                actual: vec![2, 2],
                storage_len: 4,
            }),
            LieError::Generation(_)
        ));
    }

    /// G2 values through `LieGroup` are the engine's values, bit for bit.
    #[test]
    fn exceptional_values_are_the_engine_values() {
        use crate::cache::Label;
        let g = LieGroup::new(GroupId::g2()).unwrap();
        let fam = general::GeneralFamily::new(RootSystem::G2);
        let (v, adj) = (vec![1, 0], vec![0, 1]);
        let [lv, ladj] = [&v, &adj].map(|x| g.irrep(x).unwrap());
        let got = g.f_symbol(&lv, &lv, &lv, &lv, &ladj, &ladj).unwrap();
        let (v, adj) = (Label::new(&v), Label::new(&adj));
        let want = fam.f_exact([v, v, v, v, adj, adj]).unwrap();
        assert_eq!(got.dims(), want.dims());
        let bits = |x: &[f64]| x.iter().map(|y| y.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(got.as_slice()),
            bits(&crate::rounding::blocks::round_block(&want))
        );
        let cgc = g.cgc(&lv, &lv, &ladj).unwrap();
        let entries = fam.cgc_entries(v, v, adj, None).unwrap();
        for e in entries.iter() {
            let at = [e.m1, e.m2, e.m3, e.mu].map(|i| i as usize);
            assert_eq!(cgc.get(at).to_bits(), e.value.to_bits());
        }
        let [d1, d2, d3, n] = cgc.dims();
        let nonzero = (0..d1 * d2 * d3 * n)
            .filter(|i| {
                let idx = [i / (d2 * d3 * n), i / (d3 * n) % d2, i / n % d3, i % n];
                cgc.get(idx) != 0.0
            })
            .count();
        assert_eq!(nonzero, entries.len());
    }

    /// `cache::reset` drops the retained G2 modules and channels of every
    /// group value, including composite factors, without changing values.
    ///
    /// Runs in a child process: unlocked coefficient tests may publish the
    /// same G2 entries into the process-global cache right after a reset.
    #[test]
    fn cache_reset_drops_cartan_construction_data() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "lie::tests::cache_reset_drops_cartan_construction_data_isolated",
                "--ignored",
                "--test-threads=1",
            ])
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "isolated child failed:\n{stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    #[ignore = "child process of cache_reset_drops_cartan_construction_data; light; run by that test in every CI test job"]
    fn cache_reset_drops_cartan_construction_data_isolated() {
        let _guard = crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let g = LieGroup::new(GroupId::g2()).unwrap();
        let x = g.irrep(&[2, 0]).unwrap();
        let f = |g: &LieGroup| {
            let block = g.f_symbol(&x, &x, &x, &x, &x, &x).unwrap();
            block
                .as_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        };
        let before = f(&g);
        assert!(crate::cache::cgc_module_is_resident(
            RootSystem::G2,
            &[2, 0]
        ));
        crate::cache::reset();
        assert!(!crate::cache::cgc_module_is_resident(
            RootSystem::G2,
            &[2, 0]
        ));
        assert_eq!(f(&g.clone()), before);

        let composite =
            LieGroup::new(GroupId::cover(&[RootSystem::G2, RootSystem::A(1)], 0).unwrap()).unwrap();
        let a = composite.irrep(&[1, 0, 1]).unwrap();
        composite.cgc(&a, &a, &composite.unit()).unwrap();
        // A composite reads its factors' exact channels, not their views.
        let resident = || {
            crate::cache::cgc_exact_channel_is_resident(RootSystem::G2, [&[1, 0], &[1, 0], &[0, 0]])
        };
        assert!(resident());
        crate::cache::reset();
        assert!(!resident());
    }

    /// Admitting a large complex E6 label builds nothing; its indicator needs
    /// no module. The first phase query builds at most `27` and `27̄`; later
    /// queries on either label build nothing.
    #[test]
    fn e6_complex_phase_is_generated_once_on_first_use() {
        // Resetting tests hold this lock; warm-build checks must not overlap them.
        let _guard = crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let builds = || general::cgc::BUILDS.with(std::cell::Cell::get);
        let g = LieGroup::new(GroupId::e6()).unwrap();
        let start = builds();
        let big = g.irrep(&[1, 0, 0, 0, 1, 0]).unwrap(); // 7371
        assert_eq!(big.dim_exact(), Natural::from(7371u32));
        assert_eq!(g.frobenius_schur_indicator(&big), 0.0);
        assert_eq!(builds(), start);
        let v = g.irrep(&[1, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(g.frobenius_schur_phase(&v).unwrap(), 1.0);
        let after = builds();
        assert!(after[0] - start[0] <= 2 && after[1] == start[1]);
        assert_eq!(g.frobenius_schur_phase(&v).unwrap(), 1.0);
        assert_eq!(g.frobenius_schur_phase(&g.dual(&v)).unwrap(), 1.0);
        assert_eq!(builds(), after);
    }

    /// A reset during a Cartan generation discards everything that
    /// generation creates afterwards: a reset after the parent's guard but
    /// before a child's leaves neither the parent nor the child retained, for
    /// a channel (child modules), a phase (child modules) and an R block
    /// (child channels). The labels, except G2 `[1,0]`, are used by no other
    /// test.
    #[test]
    fn reset_during_generation_retains_no_child() {
        use crate::cache::{
            cgc_channel_is_resident, cgc_module_is_resident, cgc_phase_is_resident,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        static RESET_AT: AtomicUsize = AtomicUsize::new(0);
        fn hook() {
            if CALLS.fetch_add(1, Ordering::SeqCst) == RESET_AT.load(Ordering::SeqCst) {
                crate::cache::reset();
            }
        }
        let _guard = crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let run = |reset_at: usize, query: &dyn Fn()| {
            crate::cache::reset();
            CALLS.store(0, Ordering::SeqCst);
            RESET_AT.store(reset_at, Ordering::SeqCst);
            general::cgc::BEFORE_LOAD.with(|h| h.set(Some(hook)));
            query();
            general::cgc::BEFORE_LOAD.with(|h| h.set(None));
            assert!(reset_at == usize::MAX || CALLS.load(Ordering::SeqCst) > reset_at);
        };
        let g2 = LieGroup::new(GroupId::g2()).unwrap();
        let (a, b, c) = ([3i64, 1], [1i64, 0], [4i64, 1]);
        let [la, lb, lc] = [a, b, c].map(|l| g2.irrep(&l).unwrap());
        assert_eq!(g2.nsymbol(&la, &lb, &lc).unwrap(), 1);
        // Call 0 is the channel's own lookup; call 1 is its first module.
        run(1, &|| {
            g2.cgc(&la, &lb, &lc).unwrap();
        });
        assert!(!cgc_channel_is_resident(RootSystem::G2, [&a, &b, &c]));
        // The 7 = [1,0] is loaded by concurrent tests too; check the others.
        for l in [&a[..], &c] {
            assert!(!cgc_module_is_resident(RootSystem::G2, l), "{l:?}");
        }
        // Call 0 is the first channel of the R block.
        run(0, &|| {
            g2.r_symbol(&la, &lb, &lc).unwrap();
        });
        assert!(!cgc_channel_is_resident(RootSystem::G2, [&a, &b, &c]));
        assert!(!cgc_channel_is_resident(RootSystem::G2, [&b, &a, &c]));
        // A complex D5 phase: call 0 is its first module.
        let d5 = LieGroup::new(GroupId::spin(10).unwrap()).unwrap();
        let x = d5.irrep(&[2, 0, 0, 1, 0]).unwrap();
        // Owned cache keys; the dual is a temporary.
        let dx: Vec<i64> = x.dynkin().collect();
        let dd: Vec<i64> = d5.dual(&x).dynkin().collect();
        run(0, &|| {
            d5.frobenius_schur_phase(&x).unwrap();
        });
        assert!(!cgc_phase_is_resident(
            RootSystem::D(5),
            dx.clone().min(dd.clone()).as_slice()
        ));
        assert!(!cgc_module_is_resident(RootSystem::D(5), &dx));
        assert!(!cgc_module_is_resident(RootSystem::D(5), &dd));
        // Without a reset the same queries retain their keys.
        run(usize::MAX, &|| {
            g2.cgc(&la, &lb, &lc).unwrap();
            d5.frobenius_schur_phase(&x).unwrap();
        });
        assert!(cgc_channel_is_resident(RootSystem::G2, [&a, &b, &c]));
        assert!(cgc_phase_is_resident(
            RootSystem::D(5),
            dx.clone().min(dd).as_slice()
        ));
    }

    /// A reset that completes on another thread after an outer CGC, F or R
    /// placeholder is acquired but before its generation starts leaves none
    /// of that generation's modules, channels or products retained. The
    /// channel `(a, b, e)` each query builds first is its rounded view (CGC),
    /// its exact channel (F) or its seeds (R).
    #[test]
    fn reset_between_guard_and_generation_retains_no_child() {
        use crate::cache::{
            cgc_channel_is_resident, cgc_exact_channel_is_resident, cgc_module_is_resident,
            cgc_seeds_are_resident, product_multiplicities_are_resident, AFTER_GUARD,
        };
        fn hook() {
            std::thread::spawn(crate::cache::reset).join().unwrap();
        }
        let _guard = crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let g2 = LieGroup::new(GroupId::g2()).unwrap();
        let (a, b, e) = ([3i64, 1], [1i64, 0], [4i64, 1]);
        let [la, lb, le] = [a, b, e].map(|l| g2.irrep(&l).unwrap());
        let lf = g2.fusion(&lb, &lb).unwrap().next_back().unwrap().0;
        let ld = g2
            .fusion(&le, &lb)
            .unwrap()
            .map(|(d, _)| d)
            .find(|d| g2.nsymbol(&la, &lf, d).unwrap() > 0)
            .unwrap();
        type Probe = fn(RootSystem, [&[i64]; 3]) -> bool;
        let resident = |channel: Probe| {
            [
                channel(RootSystem::G2, [&a, &b, &e]),
                cgc_module_is_resident(RootSystem::G2, &a),
                cgc_module_is_resident(RootSystem::G2, &e),
                product_multiplicities_are_resident(RootSystem::G2, &a, &b),
            ]
        };
        let queries: [(&str, Probe, &dyn Fn()); 3] = [
            ("cgc", cgc_channel_is_resident, &|| {
                g2.cgc(&la, &lb, &le).unwrap();
            }),
            ("f", cgc_exact_channel_is_resident, &|| {
                g2.f_symbol(&la, &lb, &lb, &ld, &le, &lf).unwrap();
            }),
            ("r", cgc_seeds_are_resident, &|| {
                g2.r_symbol(&la, &lb, &le).unwrap();
            }),
        ];
        for (name, channel, query) in queries {
            crate::cache::reset();
            AFTER_GUARD.with(|h| h.set(Some(hook)));
            query();
            assert!(AFTER_GUARD.with(std::cell::Cell::get).is_none(), "{name}");
            assert_eq!(resident(channel), [false; 4], "{name}");
            // Without the reset the same query retains them.
            crate::cache::reset();
            query();
            assert_eq!(resident(channel), [true; 4], "{name}: control");
        }
    }

    /// Two concurrent resets do not overlap: a query that runs after one
    /// reset has cleared the Product and CGC tiers, while a second reset is
    /// requested, retains nothing once both have returned.
    #[test]
    fn overlapping_resets_retain_no_query() {
        use crate::cache::{
            cgc_channel_is_resident, cgc_module_is_resident, product_multiplicities_are_resident,
            RESET_PAUSED, RESET_RELEASE,
        };
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant};
        let _guard = crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::cache::reset();
        RESET_PAUSED.store(0, Ordering::SeqCst);
        RESET_RELEASE.store(false, Ordering::SeqCst);
        let spawn = || {
            std::thread::Builder::new()
                .name("overlap-reset".into())
                .spawn(crate::cache::reset)
                .unwrap()
        };
        let wait_paused = |n: usize, limit: Duration| {
            let start = Instant::now();
            while RESET_PAUSED.load(Ordering::SeqCst) < n && start.elapsed() < limit {
                std::thread::sleep(Duration::from_millis(1));
            }
        };
        let first = spawn();
        wait_paused(1, Duration::from_secs(60));
        assert_eq!(RESET_PAUSED.load(Ordering::SeqCst), 1);
        // A second reset must wait for the first; give an overlapping one
        // the chance to reach the same point.
        let second = spawn();
        wait_paused(2, Duration::from_millis(500));
        let g2 = LieGroup::new(GroupId::g2()).unwrap();
        let (a, b, c) = ([3i64, 1], [1i64, 0], [4i64, 1]);
        let [la, lb, lc] = [a, b, c].map(|l| g2.irrep(&l).unwrap());
        g2.cgc(&la, &lb, &lc).unwrap();
        RESET_RELEASE.store(true, Ordering::SeqCst);
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(
            [
                cgc_channel_is_resident(RootSystem::G2, [&a, &b, &c]),
                cgc_module_is_resident(RootSystem::G2, &a),
                cgc_module_is_resident(RootSystem::G2, &c),
                product_multiplicities_are_resident(RootSystem::G2, &a, &b),
            ],
            [false; 4]
        );
        RESET_RELEASE.store(false, Ordering::SeqCst);
        // Without a reset in progress the same query retains them.
        g2.cgc(&la, &lb, &lc).unwrap();
        assert!(cgc_channel_is_resident(RootSystem::G2, [&a, &b, &c]));
    }

    /// A resident channel carries its multiplicity: a CGC hit after only the
    /// Product tier was trimmed computes no tensor product and builds nothing.
    #[test]
    fn cgc_hit_needs_no_product() {
        let _guard = crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let products = || general::roots::THREAD_PRODUCT_BUILDS.with(std::cell::Cell::get);
        let builds = || general::cgc::BUILDS.with(std::cell::Cell::get);
        let g = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        let eight = g.irrep(&[1, 1]).unwrap();
        let first = g.cgc(&eight, &eight, &eight).unwrap();
        crate::cache::trim_to(crate::cache::CoefficientCacheTier::Product, 0);
        let (p, b) = (products(), builds());
        assert_eq!(g.cgc(&eight, &eight, &eight).unwrap(), first);
        assert_eq!((products(), builds()), (p, b));
        // Dense materialization reads the generated channel: no generation
        // (thread-local counters, so parallel tests cannot interfere) and
        // fresh owned storage per call.
        let dense = first.to_dense();
        let again = first.to_dense();
        assert_eq!(dense.len(), 8 * 8 * 8 * 2);
        assert_eq!(dense, again);
        assert_ne!(dense.as_ptr(), again.as_ptr());
        assert_eq!(dense[2 * 8 * 8 * 2 + 1], first.get([2, 0, 0, 1])); // ma = 2, mu = 1
        assert_eq!((products(), builds()), (p, b));
    }

    /// A failure while generating a complex irrep's phase reaches the public
    /// phase and the Lie A override as the same typed `Generation` error.
    #[test]
    fn phase_generation_failure_propagates_to_phase_and_a() {
        fn injected() -> LieError {
            LieError::Generation(GenerationError(Arc::new(std::io::Error::other(
                "injected phase failure",
            ))))
        }
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                general::cgc::FAIL_PHASE.with(|f| f.set(None));
            }
        }
        let g = LieGroup::new(GroupId::spin(10).unwrap()).unwrap();
        let s = g.irrep(&[0, 0, 0, 1, 0]).unwrap(); // complex half-spinor 16
        let dual = g.dual(&s);
        assert_ne!(s, dual);
        let _reset = Reset;
        general::cgc::FAIL_PHASE.with(|f| f.set(Some(injected)));
        assert_eq!(g.frobenius_schur_phase(&s), Err(injected()));
        assert_eq!(g.a_symbol(&s, &dual, &g.unit()), Err(injected()));
        // The indicator reads stored data; B has no phase multiplier.
        assert_eq!(g.frobenius_schur_indicator(&s), 0.0);
        assert!(g.b_symbol(&s, &dual, &g.unit()).is_ok());
    }

    /// `κ` of every A-series irrep is the closed form (§5.3): a complex
    /// SU(N) label builds no module.
    #[test]
    fn a_series_phase_builds_nothing() {
        let builds = || general::cgc::BUILDS.with(std::cell::Cell::get);
        let g = LieGroup::new(GroupId::su(5).unwrap()).unwrap();
        let start = builds();
        for d in [[1, 0, 0, 0], [0, 1, 0, 0], [2, 1, 0, 0]] {
            let a = g.irrep(&d).unwrap();
            assert_ne!(a, g.dual(&a));
            g.frobenius_schur_phase(&a).unwrap();
        }
        assert_eq!(builds(), start);
    }

    /// Isomorphic presentations are different gauges (§1): B2/C2 and D3/A3
    /// keep separate CGC, F and R keys even for equal label tuples.
    #[test]
    fn isomorphic_presentations_do_not_share_keys() {
        let nonzero = |id: GroupId, a: &[i64], c: &[i64]| {
            let g = LieGroup::new(id).unwrap();
            let [a, c] = [a, c].map(|l| g.irrep(l).unwrap());
            let cgc = g.cgc(&a, &a, &c).unwrap();
            let [d1, d2, d3, n] = cgc.dims();
            let mut count = 0;
            for i in 0..d1 * d2 * d3 * n {
                let idx = [i / (d2 * d3 * n), i / (d3 * n) % d2, i / n % d3, i % n];
                count += usize::from(cgc.get(idx) != 0.0);
            }
            (cgc.dims(), count)
        };
        // 5 ⊗ 5 → 1 and 4 ⊗ 4 → 1 share the labels [1,0] ⊗ [1,0] → [0,0].
        assert_eq!(
            nonzero(GroupId::spin(5).unwrap(), &[1, 0], &[0, 0]),
            ([5, 5, 1, 1], 5)
        );
        assert_eq!(
            nonzero(GroupId::sp(4).unwrap(), &[1, 0], &[0, 0]),
            ([4, 4, 1, 1], 4)
        );
        // D3 [1,0,0] is the 6, A3 [1,0,0] the 4.
        assert_eq!(
            nonzero(GroupId::spin(6).unwrap(), &[1, 0, 0], &[0, 0, 0]),
            ([6, 6, 1, 1], 6)
        );
        assert_eq!(
            nonzero(GroupId::su(4).unwrap(), &[1, 0, 0], &[2, 0, 0]),
            ([4, 4, 10, 1], 16)
        );
        let r = |id: GroupId| {
            let g = LieGroup::new(id).unwrap();
            let v = g.irrep(&[1, 0]).unwrap();
            g.r_symbol(&v, &v, &g.unit()).unwrap().as_slice()[0]
        };
        // The B2 vector is real (R = +1); the C2 defining irrep is pseudo-real (R = −1).
        assert!((r(GroupId::spin(5).unwrap()) - 1.0).abs() < 1e-15);
        assert!((r(GroupId::sp(4).unwrap()) + 1.0).abs() < 1e-15);
    }

    #[test]
    fn generation_error_keeps_its_reason() {
        let error = LieError::from(crate::frcore::GateError::NonFinite { gate: "R" });
        assert!(matches!(error, LieError::Generation(_)));
        let gate = error.source().unwrap().source().unwrap();
        assert!(matches!(
            gate.downcast_ref::<crate::frcore::GateError>(),
            Some(crate::frcore::GateError::NonFinite { gate: "R" })
        ));
        assert_eq!(error.to_string(), "R gate: nonfinite value");
    }
}

#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    #[test]
    fn dimension_and_buffer_bounds() {
        assert_eq!(
            checked_dimension(usize::MAX as u128 + 1),
            Err(LieError::LabelOverflow)
        );
        assert_eq!(checked_count(usize::MAX, 1), Err(LieError::LabelOverflow));
        assert_eq!(checked_dimension(-1i64), Err(LieError::LabelOverflow));
        assert_eq!(checked_size([usize::MAX, 2]), Err(LieError::LabelOverflow));
        assert_eq!(
            checked_size([isize::MAX as usize / 8 + 1]),
            Err(LieError::LabelOverflow)
        );
        assert_eq!(checked_size([usize::MAX, 2, 0]), Ok(0));
        assert_eq!(
            a1_dimension(u32::MAX),
            checked_dimension(u64::from(u32::MAX) + 1)
        );
    }

    #[test]
    fn rank_one_n_stays_exact_when_fusion_labels_overflow() {
        let cat = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
        let a = cat.irrep(&[i64::from(u32::MAX)]).unwrap();
        let b = cat.irrep(&[1]).unwrap();
        let c = cat.irrep(&[i64::from(u32::MAX) - 1]).unwrap();
        assert_eq!(cat.nsymbol(&a, &b, &c), Ok(1));
        assert!(matches!(cat.fusion(&a, &b), Err(LieError::LabelOverflow)));
        let a = cat.irrep(&[i64::from(i32::MAX)]).unwrap();
        assert!(matches!(
            cat.cgc(&a, &a, &cat.unit()),
            Err(LieError::LabelOverflow)
        ));
    }
}

#[cfg(all(test, feature = "lie-generated"))]
mod root_payload_tests {
    use super::*;
    #[test]
    fn dim_is_label_owned_and_takes_no_global_lock() {
        use std::{sync::mpsc, time::Duration};
        let _serial = crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let groups = [
            GroupId::su(6).unwrap(),
            GroupId::spin(9).unwrap(),
            GroupId::sp(8).unwrap(),
            GroupId::spin(10).unwrap(),
            GroupId::g2(),
            GroupId::e6(),
        ]
        .map(|id| LieGroup::new(id).unwrap());
        let mut cases = Vec::new();
        for g in &groups {
            let r = g.group.semisimple_rank();
            for label in [vec![1; r], vec![1000; r]] {
                // Complex E6 retains its existing public coefficient-index bound.
                let a = g.irrep(&label).unwrap();
                cases.push((g.clone(), a));
            }
        }
        let expected = cases
            .iter()
            .map(|(g, a)| g.dim(a).to_bits())
            .collect::<Vec<_>>();
        crate::cache::reset();
        general::roots::EVALUATIONS.with(|n| n.set((0, 0)));
        for _ in 0..32 {
            for ((g, a), bits) in cases.iter().zip(&expected) {
                assert_eq!(g.dim(a).to_bits(), *bits);
            }
        }
        general::roots::EVALUATIONS.with(|n| assert_eq!(n.get(), (0, 0)));
        let (tx, rx) = mpsc::channel();
        let (worker, result) = crate::cache::with_publication_locks(|| {
            let worker = std::thread::spawn(move || {
                let bits = cases
                    .iter()
                    .map(|(g, a)| g.dim(a).to_bits())
                    .collect::<Vec<_>>();
                tx.send(bits).unwrap();
            });
            let result = rx.recv_timeout(Duration::from_secs(5));
            (worker, result)
        });
        worker.join().unwrap();
        assert_eq!(
            result.unwrap(),
            expected,
            "dim must finish before any cache publication lock is released"
        );
    }
}

/// The exact-composition oracle of products (design §g.4, case 2): every
/// product entry is the correct rounding of the product of its leaves' exact
/// entries, composed here by rational signed squares in mixed radix (last
/// factor fastest on every axis), never by the production composer.
#[cfg(test)]
mod product_oracle {
    use super::*;
    use crate::category::sealed::{ExactBuf, ExactOp, Query, Validated};
    use crate::category::SymbolBlock;
    use crate::exact::SignedSqrtRational;
    use malachite_q::Rational;

    type Exact<const R: usize> = ([usize; R], Vec<Rational>);

    fn fetch(g: &LieGroup, q: Query<'_, LieIrrep>) -> ExactBuf {
        let mut buf = ExactBuf::with_capacity(1);
        g.__fusion_symbols_exact_symbol(ExactOp::Fetch(q, &mut buf), Validated(()))
            .unwrap();
        buf
    }

    fn squares<const R: usize>(x: &crate::exact::radical::RadicalBlock<R>) -> Exact<R> {
        let len = x.dims().iter().product();
        (
            x.dims(),
            (0..len).map(|i| x.entry(i).signed_square()).collect(),
        )
    }

    fn kron<const R: usize>((ld, l): &Exact<R>, (rd, r): &Exact<R>) -> Exact<R> {
        let dims: [usize; R] = std::array::from_fn(|k| ld[k] * rd[k]);
        let flat = |d: &[usize; R], i: [usize; R]| (0..R).fold(0, |n, k| n * d[k] + i[k]);
        let len = dims.iter().product();
        let out = (0..len)
            .map(|mut i| {
                let mut digits = [0; R];
                for k in (0..R).rev() {
                    digits[k] = i % dims[k];
                    i /= dims[k];
                }
                let li = flat(ld, std::array::from_fn(|k| digits[k] / rd[k]));
                let ri = flat(rd, std::array::from_fn(|k| digits[k] % rd[k]));
                &l[li] * &r[ri]
            })
            .collect();
        (dims, out)
    }

    fn rounded(square: &Rational) -> u64 {
        let sign = if *square < 0 {
            -1
        } else {
            i8::from(*square > 0)
        };
        let magnitude = if *square < 0 { -square } else { square.clone() };
        SignedSqrtRational::from_radicand(sign, magnitude)
            .to_f64()
            .to_bits()
    }

    fn bits(x: &[f64]) -> Vec<u64> {
        x.iter().map(|x| x.to_bits()).collect()
    }

    /// Product F/R/A/B of two copies of `g` against the oracle, at `f` (six
    /// labels per factor) and `r` (three per factor, for R, A and B).
    fn check(g: &LieGroup, f: [[&LieIrrep; 6]; 2], r: [[&LieIrrep; 3]; 2], with_r: bool) {
        let p = crate::product::Product::new((g.clone(), g.clone()));
        let pf: [_; 6] = std::array::from_fn(|i| (f[0][i].clone(), f[1][i].clone()));
        let pr: [_; 3] = std::array::from_fn(|i| (r[0][i].clone(), r[1][i].clone()));
        let leaves = f.map(|q| squares(fetch(g, Query::F(q)).f.iter().next().unwrap()));
        let [a, b, c, d, e, x] = pf.each_ref();
        let (dims, expected) = kron(&leaves[0], &leaves[1]);
        let got = p.f_symbol(a, b, c, d, e, x).unwrap();
        assert_eq!(got.dims(), dims);
        assert_eq!(
            bits(got.as_slice()),
            expected.iter().map(rounded).collect::<Vec<_>>()
        );
        let [a, b, c] = pr.each_ref();
        use crate::category::sealed::SymbolKind as K;
        let kinds: &[K] = if with_r {
            &[K::R, K::A, K::B]
        } else {
            &[K::A, K::B]
        };
        for &k in kinds {
            let query = |q| match k {
                K::R => Query::R(q),
                K::A => Query::A(q),
                _ => Query::B(q),
            };
            let leaves = r.map(|q| squares(fetch(g, query(q)).r2.iter().next().unwrap()));
            let (dims, expected) = kron(&leaves[0], &leaves[1]);
            let got = match k {
                K::R => p.r_symbol(a, b, c),
                K::A => p.a_symbol(a, b, c),
                _ => p.b_symbol(a, b, c),
            }
            .unwrap();
            assert_eq!(got.dims(), dims, "{k:?}");
            assert_eq!(
                bits(got.as_slice()),
                expected.iter().map(rounded).collect::<Vec<_>>(),
                "{k:?}"
            );
        }
    }

    /// SU(2)×SU(2) (R is a unit for SU(2), so only F, A and B compose).
    #[test]
    fn su2_products_round_the_exact_composition() {
        let g = LieGroup::new(GroupId::su(2).unwrap()).unwrap();
        let [s0, h, s1, t] = [0, 1, 2, 3].map(|j| g.irrep(&[j]).unwrap());
        check(
            &g,
            [[&h, &h, &h, &h, &s0, &s1]; 2],
            [[&h, &h, &s1]; 2],
            false,
        );
        check(
            &g,
            [[&h, &s1, &h, &s1, &h, &h], [&t, &h, &h, &t, &s1, &s0]],
            [[&h, &s1, &t], [&t, &h, &s1]],
            false,
        );
    }

    /// SU(3)×SU(3) octets: every axis has multiplicity two, so R composes too.
    #[cfg(feature = "lie-generated")]
    #[test]
    fn su3_products_round_the_exact_composition() {
        let g = LieGroup::new(GroupId::su(3).unwrap()).unwrap();
        let [o, t, tb] = [[1, 1], [1, 0], [0, 1]].map(|l| g.irrep(&l).unwrap());
        check(&g, [[&o; 6]; 2], [[&o; 3]; 2], true);
        let one = g.unit();
        check(
            &g,
            [[&t, &tb, &t, &t, &one, &one], [&o; 6]],
            [[&t, &t, &tb], [&o; 3]],
            true,
        );
    }
}
