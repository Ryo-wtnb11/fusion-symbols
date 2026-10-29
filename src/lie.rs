//! Positive-dimensional compact Lie groups: root systems and global forms via
//! [`LieGroup`] (including tori), and closed-form [`U1`] and [`CU1`].

mod cu1;
mod u1;
pub use cu1::{CU1Error, CU1Fusion, CU1Irrep, CU1};
pub use u1::{U1Error, U1Irrep, U1};

#[cfg(feature = "lie-generated")]
pub(crate) mod bcd;
#[cfg(feature = "lie-generated")]
pub(crate) mod general;
mod gt;
#[cfg(feature = "lie-generated")]
pub(crate) mod sun;
pub use gt::GtPattern;
mod group;
pub mod su2;
pub use group::{GroupError, GroupId, RootSystem};
mod half_int;
pub use half_int::HalfInt;

use crate::category::{
    sealed::Validated, BraidingStyle, CgcBlock, EnumerationError, FBlock, FusionCategory,
    FusionStyle, GroupRepresentations, RBlock,
};
use crate::product::{Product, ProductError};
#[cfg(feature = "lie-generated")]
use malachite_base::{
    num::conversion::traits::RoundingFrom, rounding_modes::RoundingMode::Nearest,
};
use malachite_nz::{integer::Integer, natural::Natural};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A Lie group with its global form. A(1), B(1), and C(1) share the exact
/// SU(2) gauge and fingerprint; their single Dynkin label is doubled spin.
/// With `lie-generated`, higher A-series groups use the GT gauge, B/C/D groups
/// use the canonical generator bootstrap, and G2, F4, E6, E7 and E8 use the exact
/// exceptional engine (`docs/gauge_exceptional.md`). Clones share retained
/// generators and the exceptional memo.
/// Admissibility uses the caller's global form.
/// Objects retain the full `GroupId` tag, including for isomorphic groups.
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
    #[cfg(feature = "lie-generated")]
    catalog: Option<std::sync::Arc<std::sync::Mutex<bcd::CanonicalCatalog>>>,
    // At most one retained rank-sized unit label per group instance.
    #[cfg(feature = "lie-generated")]
    unit: std::sync::OnceLock<LieIrrep>,
    // The exceptional engine and its bounded memo; shared by clones.
    #[cfg(feature = "lie-generated")]
    general: Option<Arc<general::Engine>>,
    #[cfg(feature = "lie-generated")]
    roots: Option<Arc<std::sync::OnceLock<Arc<general::roots::CartanData>>>>,
}

/// Snapshot of a [`LieGroup`]'s retained B/C/D generator catalogs, summed over
/// every catalog of a composite group. All zero for a group without a catalog.
/// See [`LieGroup::catalog_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CatalogStats {
    /// Generator sets currently retained, including the seeded base cases.
    pub entries: usize,
    /// Charged bytes of the retained generators (the quantity the budget bounds).
    pub bytes: usize,
    /// Retained-byte budget.
    pub budget: usize,
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
// GroupId. Their ordering is the former field ordering, including signed charges.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
// Avoid niche-discriminant decoding on the inline symbol paths.
#[repr(u8)]
enum LiePayload {
    RankOne(u32),
    Torus(U1Irrep),
    #[cfg(feature = "lie-generated")]
    Generated(sun::Irrep),
    #[cfg(feature = "lie-generated")]
    Bcd(bcd::Irrep),
    #[cfg(feature = "lie-generated")]
    Exceptional(general::Irrep),
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
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.irrep.exceptional() {
            return Some(a.dynkin()[i]);
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.irrep.generated() {
            let w = a.weight();
            return Some(w[i] - w[i + 1]);
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.irrep.bcd() {
            let w = a.two_partition();
            return Some(if i + 1 < w.len() {
                bcd::half_sum(w[i], -w[i + 1])
            } else {
                match a.series() {
                    bcd::Series::B => w[i],
                    bcd::Series::C => w[i] / 2,
                    bcd::Series::D => bcd::half_sum(w[i - 1], w[i]),
                }
            });
        }
        let _ = i;
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
    /// (`"lie-generated"`) or a B/C/D catalog (`"B/C/D catalog"`).
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
#[cfg(feature = "lie-generated")]
impl From<sun::SunError> for LieError {
    fn from(error: sun::SunError) -> Self {
        if error == sun::SunError::LabelOverflow {
            return Self::LabelOverflow;
        }
        Self::Generation(GenerationError(std::sync::Arc::new(error)))
    }
}
#[cfg(feature = "lie-generated")]
impl From<bcd::BcdError> for LieError {
    fn from(error: bcd::BcdError) -> Self {
        if error == bcd::BcdError::LabelOverflow {
            return Self::LabelOverflow;
        }
        Self::Generation(GenerationError(std::sync::Arc::new(error)))
    }
}
#[cfg(feature = "lie-generated")]
impl From<bcd::CatalogError> for LieError {
    fn from(error: bcd::CatalogError) -> Self {
        match &error {
            bcd::CatalogError::Label(bcd::BcdError::LabelOverflow)
            | bcd::CatalogError::Sweep(bcd::sweep::SweepError::LabelOverflow) => {
                return Self::LabelOverflow
            }
            _ => {}
        }
        Self::Generation(GenerationError(std::sync::Arc::new(error)))
    }
}
#[cfg(feature = "lie-generated")]
fn poisoned_catalog() -> LieError {
    LieError::Generation(GenerationError(std::sync::Arc::new(std::io::Error::other(
        "poisoned Lie catalog",
    ))))
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
#[cfg(feature = "lie-generated")]
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
fn a1_projection(m: i64, dj: i64) -> Result<i64, LieError> {
    m.checked_mul(2)
        .and_then(|m| m.checked_sub(dj))
        .ok_or(LieError::LabelOverflow)
}

fn checked_projection(value: i64) -> Result<i32, LieError> {
    i32::try_from(value).map_err(|_| LieError::LabelOverflow)
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

impl From<ProductError<LieError>> for LieError {
    fn from(error: ProductError<LieError>) -> Self {
        match error {
            ProductError::Overflow => Self::LabelOverflow,
            // Keep matchable leaf capability/overflow variants; generation and
            // malformed-block diagnostics retain the complete indexed source path.
            ProductError::Factor {
                source: source @ (Self::LabelOverflow | Self::Unsupported { .. }),
                ..
            } => source,
            error => Self::Generation(GenerationError(Arc::new(CompositeError(error)))),
        }
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
    /// B/C/D seed charges exceeding the default aggregate catalog budget return
    /// [`LieError::Generation`]. No zero-weight label is allocated at setup.
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
        if group.root_systems().is_empty() && group.torus_rank() == 1 && group.is_cover() {
            return Ok(Self {
                #[cfg(feature = "lie-generated")]
                roots: None,
                group,
                composite: None,
                #[cfg(feature = "lie-generated")]
                catalog: None,
                #[cfg(feature = "lie-generated")]
                unit: std::sync::OnceLock::new(),
                #[cfg(feature = "lie-generated")]
                general: None,
            });
        }
        if group.torus_rank() != 0
            || group.root_systems().len() != 1
            || group.root_systems()[0] == RootSystem::D(2)
        {
            #[cfg(feature = "lie-generated")]
            let budget = bcd::catalog::DEFAULT_MAX_BYTES;
            #[cfg(not(feature = "lie-generated"))]
            let budget = 0;
            return Self::composite(group, budget);
        }
        let root = group.root_systems()[0];
        if matches!(root, RootSystem::B(r) | RootSystem::C(r) | RootSystem::D(r) if r >= 2) {
            #[cfg(feature = "lie-generated")]
            {
                let series = bcd_series(root)
                    .expect("the enclosing match restricts the root system to B/C/D");
                let catalog = bcd::CanonicalCatalog::new(series, root.rank())?;
                return Ok(Self {
                    roots: Some(Arc::new(std::sync::OnceLock::from(catalog.roots.clone()))),
                    group,
                    composite: None,
                    unit: std::sync::OnceLock::new(),
                    general: None,
                    catalog: Some(Arc::new(std::sync::Mutex::new(catalog))),
                });
            }
            #[cfg(not(feature = "lie-generated"))]
            return Err(LieError::Unsupported {
                requires: "lie-generated",
            });
        }
        // `coefficient_roots` rejected every exceptional root without
        // `lie-generated`.
        #[cfg(feature = "lie-generated")]
        if general::serves(root) {
            return Ok(Self {
                #[cfg(feature = "lie-generated")]
                roots: None,
                group,
                composite: None,
                unit: std::sync::OnceLock::new(),
                catalog: None,
                general: Some(Arc::new(general::Engine::new(root))),
            });
        }
        if matches!(root, RootSystem::A(r) if r >= 2) {
            #[cfg(feature = "lie-generated")]
            {
                // The internal SU(N) weight has rank + 1 i64 entries.
                if root.rank() >= isize::MAX as usize / std::mem::size_of::<i64>() {
                    return Err(LieError::LabelOverflow);
                }
                return Ok(Self {
                    roots: Some(Arc::new(std::sync::OnceLock::new())),
                    group,
                    composite: None,
                    unit: std::sync::OnceLock::new(),
                    catalog: None,
                    general: None,
                });
            }
            #[cfg(not(feature = "lie-generated"))]
            return Err(LieError::Unsupported {
                requires: "lie-generated",
            });
        }
        if !matches!(root, RootSystem::A(1) | RootSystem::B(1) | RootSystem::C(1)) {
            return Err(LieError::Unsupported {
                requires: "higher-rank Lie representations",
            });
        }
        Ok(Self {
            #[cfg(feature = "lie-generated")]
            roots: None,
            group,
            composite: None,
            #[cfg(feature = "lie-generated")]
            catalog: None,
            #[cfg(feature = "lie-generated")]
            unit: std::sync::OnceLock::new(),
            #[cfg(feature = "lie-generated")]
            general: None,
        })
    }
    /// Set the retained-generator budget for a group with B/C/D catalogs.
    /// For composites, preflight the sum of all catalog seed charges, then
    /// split the remaining bytes equally among catalogs in factor order,
    /// giving the remainder to earlier factors. Each cap is its seed plus its
    /// share. Without any catalog this returns [`LieError::Unsupported`].
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupId, LieError, LieGroup};
    ///
    /// # #[cfg(feature = "lie-generated")] {
    /// let so5 = LieGroup::with_catalog_budget(GroupId::so(5)?, 64 << 20)?;
    /// assert_eq!(so5.catalog_stats()?.budget, 64 << 20);
    /// # }
    /// // SU(2) has no B/C/D catalog to budget.
    /// let su2 = LieGroup::with_catalog_budget(GroupId::su(2)?, 64 << 20);
    /// assert!(matches!(su2, Err(LieError::Unsupported { .. })));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_catalog_budget(group: GroupId, max_bytes: usize) -> Result<Self, LieError> {
        coefficient_roots(&group)?;
        if group.torus_rank() != 0
            || group.root_systems().len() != 1
            || group.root_systems()[0] == RootSystem::D(2)
        {
            let result = Self::composite(group, max_bytes)?;
            if result.catalog_stats()?.budget == 0 {
                return Err(LieError::Unsupported {
                    requires: "B/C/D catalog",
                });
            }
            return Ok(result);
        }
        let root = group.root_systems()[0];
        #[cfg(feature = "lie-generated")]
        if let Some(series) = bcd_series(root) {
            if root != RootSystem::D(2) && root.rank() >= 2 {
                let catalog = bcd::CanonicalCatalog::with_budget(series, root.rank(), max_bytes)?;
                return Ok(Self {
                    roots: Some(Arc::new(std::sync::OnceLock::from(catalog.roots.clone()))),
                    group,
                    composite: None,
                    unit: std::sync::OnceLock::new(),
                    general: None,
                    catalog: Some(Arc::new(std::sync::Mutex::new(catalog))),
                });
            }
        }
        let _ = (root, max_bytes);
        Self::new(group)?;
        Err(LieError::Unsupported {
            requires: "B/C/D catalog",
        })
    }
    fn composite(group: GroupId, max_bytes: usize) -> Result<Self, LieError> {
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
        #[cfg(feature = "lie-generated")]
        let seeds = roots
            .iter()
            .map(
                |&root| match bcd_series(root).filter(|_| root.rank() >= 2) {
                    Some(series) => {
                        bcd::CanonicalCatalog::seed_requirement(series, root.rank(), usize::MAX)
                            .map(Some)
                            .ok()
                    }
                    None => Some(None),
                },
            )
            .collect::<Option<Vec<_>>>();
        #[cfg(feature = "lie-generated")]
        let (remaining, count, seeds) = {
            let total = seeds.as_ref().and_then(|seeds| {
                seeds
                    .iter()
                    .flatten()
                    .try_fold(0usize, |n, seed| n.checked_add(*seed))
            });
            let total =
                total
                    .filter(|&n| n <= max_bytes)
                    .ok_or(bcd::CatalogError::BudgetExceeded {
                        limit: max_bytes,
                        needed: total.unwrap_or(usize::MAX),
                    })?;
            let seeds = seeds.expect("aggregate preflight validated every seed");
            (max_bytes - total, seeds.iter().flatten().count(), seeds)
        };
        let mut factors = Vec::new();
        factors
            .try_reserve_exact(checked_count(roots.len(), group.torus_rank())?)
            .map_err(|_| LieError::LabelOverflow)?;
        #[cfg(feature = "lie-generated")]
        let mut catalog_index = 0;
        for (i, root) in roots.into_iter().enumerate() {
            let cover = GroupId::cover(&[root], 0).expect("validated factor has a valid cover");
            #[cfg(feature = "lie-generated")]
            if let Some(seed) = seeds[i] {
                let cap = seed + remaining / count + usize::from(catalog_index < remaining % count);
                factors.push(Self::with_catalog_budget(cover, cap)?);
                catalog_index += 1;
                continue;
            }
            let _ = (i, max_bytes);
            factors.push(Self::new(cover)?);
        }
        for _ in 0..group.torus_rank() {
            factors.push(Self::new(
                GroupId::torus(1).expect("rank-one torus is valid"),
            )?);
        }
        Ok(Self {
            #[cfg(feature = "lie-generated")]
            roots: None,
            group,
            composite: Some(Product::new(factors)),
            #[cfg(feature = "lie-generated")]
            catalog: None,
            #[cfg(feature = "lie-generated")]
            unit: std::sync::OnceLock::new(),
            #[cfg(feature = "lie-generated")]
            general: None,
        })
    }
    fn composite_label(group: &GroupId, factors: Vec<LieIrrep>) -> LieIrrep {
        let label = LieIrrep {
            group: group.clone(),
            payload: LiePayload::Composite(Arc::new(factors)),
        };
        debug_assert!(group.admits(&label.dynkin(), &label.charges().collect::<Vec<_>>()));
        label
    }
    // Callers have checked the outward group; factor tags are private and immutable.
    fn factor_labels<'a>(&self, a: &'a LieIrrep) -> &'a Vec<LieIrrep> {
        a.factors().expect("composite object has factor labels")
    }
    /// Retained generator catalog statistics; all zero without a catalog.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{CatalogStats, GroupId, LieGroup};
    ///
    /// let su2 = LieGroup::new(GroupId::su(2)?)?;
    /// assert_eq!(su2.catalog_stats()?, CatalogStats::default()); // no catalog
    /// # #[cfg(feature = "lie-generated")] {
    /// let so5 = LieGroup::new(GroupId::so(5)?)?;
    /// let stats = so5.catalog_stats()?;
    /// assert!(stats.bytes <= stats.budget);
    /// # }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn catalog_stats(&self) -> Result<CatalogStats, LieError> {
        if let Some(product) = &self.composite {
            return product
                .factors()
                .iter()
                .try_fold(CatalogStats::default(), |sum, factor| {
                    let f = factor.catalog_stats()?;
                    Ok(CatalogStats {
                        entries: checked_count(sum.entries, f.entries)?,
                        bytes: checked_count(sum.bytes, f.bytes)?,
                        budget: checked_count(sum.budget, f.budget)?,
                    })
                });
        }
        #[cfg(feature = "lie-generated")]
        if let Some(cat) = &self.catalog {
            let cat = cat.lock().map_err(|_| poisoned_catalog())?;
            return Ok(CatalogStats {
                entries: cat.len(),
                bytes: cat.bytes(),
                budget: cat.budget(),
            });
        }
        Ok(CatalogStats::default())
    }
    /// Reset retained generators to the base seeds and drop the retained
    /// exceptional coefficient memo; no-op for other groups. Shared by every clone.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "lie-generated")] {
    /// use fusion_symbols::{category::GroupRepresentations, lie::{GroupId, LieGroup}};
    ///
    /// let so5 = LieGroup::new(GroupId::so(5)?)?;
    /// let seeded = so5.catalog_stats()?;
    /// let v = so5.irrep(&[1, 0])?;
    /// so5.cgc(&v, &v, &so5.irrep(&[2, 0])?)?;
    /// so5.reset_catalog()?;
    /// assert_eq!(so5.catalog_stats()?, seeded);
    /// # }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn reset_catalog(&self) -> Result<(), LieError> {
        if let Some(product) = &self.composite {
            for factor in product.factors() {
                factor.reset_catalog()?;
            }
            return Ok(());
        }
        #[cfg(feature = "lie-generated")]
        if let Some(cat) = &self.catalog {
            cat.lock().map_err(|_| poisoned_catalog())?.try_reset()?;
        }
        #[cfg(feature = "lie-generated")]
        if let Some(engine) = &self.general {
            engine.reset();
        }
        Ok(())
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
    /// assert_eq!(su2.irrep_spin(HalfInt::half(2))?, su2.irrep(&[2])?); // spin 1
    /// let so3 = LieGroup::new(GroupId::so(3)?)?;
    /// assert_eq!(so3.irrep_spin(HalfInt::half(1)), Err(LieError::NotAdmissible));
    /// let u2 = LieGroup::new(GroupId::u(2)?)?;
    /// assert_eq!(u2.irrep_spin(HalfInt::half(1)), Err(LieError::InvalidLabel));
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
    /// An E6 irrep that is not self-dual, with dimension above `u32::MAX`,
    /// returns [`LieError::LabelOverflow`] (so does its dual), here and from
    /// `fusion` when an outcome is such a label: its Frobenius–Schur phase is
    /// generated in the module basis, whose magnetic index is `u32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::{category::FusionCategory, lie::{GroupId, LieError, LieGroup}};
    ///
    /// let u2 = LieGroup::new(GroupId::u(2)?)?;
    /// let fundamental = u2.irrep_with_charges(&[1], &[1])?;
    /// let det = u2.irrep_with_charges(&[0], &[2])?;
    /// assert_eq!(u2.nsymbol(&fundamental, &fundamental, &det), 1); // Λ²(2) = det
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
        #[cfg(feature = "lie-generated")]
        if matches!(self.group.root_systems()[0], RootSystem::A(r) if r >= 2) {
            return Ok(LieIrrep {
                group: self.group.clone(),
                payload: LiePayload::Generated(sun::Irrep(self.roots().irrep(dynkin)?)),
            });
        }
        #[cfg(feature = "lie-generated")]
        if let Some(engine) = &self.general {
            return Ok(LieIrrep {
                group: self.group.clone(),
                payload: LiePayload::Exceptional(engine.irrep(dynkin)?),
            });
        }
        #[cfg(feature = "lie-generated")]
        if self.catalog.is_some() {
            return Ok(LieIrrep {
                group: self.group.clone(),
                payload: LiePayload::Bcd(bcd::Irrep(self.roots().irrep(dynkin)?)),
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
    #[cfg(feature = "lie-generated")]
    fn roots(&self) -> &Arc<general::roots::CartanData> {
        // Large A-series group identities stay cheap until a label needs exact root mathematics.
        self.roots
            .as_ref()
            .expect("generated root context")
            .get_or_init(|| general::roots::CartanData::new(self.group.root_systems()[0]))
    }
    #[cfg(feature = "lie-generated")]
    fn engine(&self) -> &general::Engine {
        self.general
            .as_ref()
            .expect("exceptional objects are constructed only with an exceptional engine")
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
    #[cfg(feature = "lie-generated")]
    fn generated(&self) -> Option<&sun::Irrep> {
        match &self.payload {
            LiePayload::Generated(a) => Some(a),
            _ => None,
        }
    }
    #[cfg(feature = "lie-generated")]
    fn bcd(&self) -> Option<&bcd::Irrep> {
        match &self.payload {
            LiePayload::Bcd(a) => Some(a),
            _ => None,
        }
    }
    #[cfg(feature = "lie-generated")]
    fn exceptional(&self) -> Option<&general::Irrep> {
        match &self.payload {
            LiePayload::Exceptional(a) => Some(a),
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
    /// assert_eq!(su2.irrep(&[3])?.spin(), Some(HalfInt::half(3))); // spin 3/2
    /// let u1 = LieGroup::new(GroupId::u(1)?)?;
    /// assert_eq!(u1.irrep_with_charges(&[], &[1])?.spin(), None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn spin(&self) -> Option<HalfInt> {
        match self.payload {
            LiePayload::RankOne(dj) if is_spin_group(&self.group) => Some(HalfInt::half(dj)),
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
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.generated() {
            return Some(a.patterns());
        }
        None
    }

    /// Highest weight in Dynkin coordinates, materialized in an owned allocation.
    /// Use [`Self::dynkin_labels`] for allocation-free label reading on hot paths.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupId, LieGroup};
    ///
    /// let u2 = LieGroup::new(GroupId::u(2)?)?;
    /// let x = u2.irrep_with_charges(&[2], &[4])?;
    /// assert_eq!(x.dynkin(), [2]); // charges are separate
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn dynkin(&self) -> Vec<i64> {
        if self.factors().is_some() {
            return self.dynkin_labels().collect();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.bcd() {
            return a.dynkin();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.exceptional() {
            return a.dynkin().to_vec();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.generated() {
            return a.dynkin();
        }
        self.dynkin_labels().collect()
    }

    /// Iterate over every highest-weight Dynkin coordinate, including zeros,
    /// in the same order as [`Self::dynkin`]. Construction and consumption
    /// allocate nothing; composite coordinates follow factor order. `len()` reports
    /// the number of remaining coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use fusion_symbols::lie::{GroupId, LieGroup};
    ///
    /// let spin4 = LieGroup::new(GroupId::spin(4)?)?;
    /// let x = spin4.irrep(&[2, 0])?;
    /// assert_eq!(x.dynkin_labels().len(), 2);
    /// assert!(x.dynkin_labels().eq([2, 0]));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn dynkin_labels(&self) -> impl ExactSizeIterator<Item = i64> + '_ {
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
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.bcd() {
            return a.dim();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.exceptional() {
            return a.dim().clone();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.generated() {
            return a.dim();
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
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.bcd() {
            return a.dynkin_weight_multiplicities();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.exceptional() {
            return a
                .weight_multiplicities(&a.roots())
                .expect("infallible weight_multiplicities failed: a weight exceeds i64");
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = self.generated() {
            return a
                .0
                .weight_multiplicities(&a.0.roots())
                .expect("infallible weight multiplicities exceed range");
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
/// A1 magnetic index zero is the lowest weight, followed by raising in steps
/// of two; higher A-series magnetic indices follow the frozen GT basis order.
/// B/C/D indices follow the canonical bootstrap gauge in `docs/gauge_soN.md`;
/// G2, F4, E6, E7 and E8 indices follow the basis order of `docs/gauge_exceptional.md` §3–§4.
/// Composite CGCs use Product's last-factor-fastest decoding on every axis;
/// D(2) factors are α₁ then α₂ in the exact A(1) gauge.
/// A1 materialization uses the cached exact CG evaluator. Projection labels larger
/// than `i32::MAX` are an unsupported capability of that evaluator.
#[derive(Clone, Debug, PartialEq)]
pub struct LieCgc {
    composite: Option<Vec<LieCgc>>,
    dims: [usize; 4],
    data: Vec<f64>,
    #[cfg(feature = "lie-generated")]
    generated: Option<sun::Cgc>,
    #[cfg(feature = "lie-generated")]
    bcd: Option<std::sync::Arc<bcd::CatalogCgc>>,
}
impl CgcBlock for LieCgc {
    /// Axis lengths, with multiplicity last (zero for an absent channel).
    fn dims(&self) -> [usize; 4] {
        if let Some(blocks) = &self.composite {
            return blocks.dims();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(c) = &self.generated {
            return c.dims();
        }
        self.dims
    }
    /// Read one coefficient.
    ///
    /// # Panics
    /// Panics if any index is outside its axis.
    fn at(&self, [m1, m2, m3, mu]: [usize; 4]) -> f64 {
        if let Some(blocks) = &self.composite {
            return blocks.at([m1, m2, m3, mu]);
        }
        #[cfg(feature = "lie-generated")]
        if let Some(c) = &self.bcd {
            let dims = self.dims();
            assert!(
                [m1, m2, m3, mu].iter().zip(dims).all(|(&i, n)| i < n),
                "CGC index out of range"
            );
            return c.copy(mu)[m3 * c.copy_shape().0 + m1 + dims[0] * m2];
        }
        #[cfg(feature = "lie-generated")]
        if let Some(c) = &self.generated {
            assert!(
                [m1, m2, m3, mu].iter().zip(c.dims()).all(|(&i, n)| i < n),
                "CGC index out of range"
            );
            let key = [m1, m2, m3, mu];
            return c
                .entries()
                .binary_search_by_key(&key, |e| {
                    [e.m1, e.m2, e.m3, e.mu].map(|i| usize::try_from(i)
                        .expect("generated sparse indices came from this allocation's usize basis indices"))
                })
                .map_or(0.0, |i| c.entries()[i].value);
        }
        assert!(
            [m1, m2, m3, mu].iter().zip(self.dims).all(|(&i, n)| i < n),
            "CGC index out of range"
        );
        // The torus scalar uses no coefficient allocation.
        if self.data.is_empty() && self.dims == [1; 4] {
            return 1.0;
        }
        self.data[((m1 * self.dims[1] + m2) * self.dims[2] + m3) * self.dims[3] + mu]
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
        #[cfg(feature = "lie-generated")]
        if self.group.semisimple_rank() >= 2 {
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
    // The group is the whole identity: a B/C/D catalog is a cache that
    // `LieGroup` equality ignores, and labels are validated at construction.
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
        #[cfg(feature = "lie-generated")]
        if let Some(x) = a.bcd() {
            return LieIrrep {
                group: self.group.clone(),
                payload: LiePayload::Bcd(x.dual()),
            };
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = a.generated() {
            return LieIrrep {
                group: self.group.clone(),
                payload: LiePayload::Generated(x.dual()),
            };
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = a.exceptional() {
            return LieIrrep {
                group: self.group.clone(),
                payload: LiePayload::Exceptional(x.dual()),
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
            #[cfg(feature = "lie-generated")]
            LiePayload::Bcd(x) => x.dim_f64(),
            #[cfg(feature = "lie-generated")]
            LiePayload::Generated(x) => x.dim_f64(),
            #[cfg(feature = "lie-generated")]
            LiePayload::Exceptional(x) => x.dim_f64(),
        }
    }
    fn sqrt_dim(&self, a: &LieIrrep) -> f64 {
        if let Some(product) = &self.composite {
            self.label(a);
            return product.sqrt_dim(self.factor_labels(a));
        }
        self.dim(a).sqrt()
    }
    fn inv_sqrt_dim(&self, a: &LieIrrep) -> f64 {
        if let Some(product) = &self.composite {
            self.label(a);
            return product.inv_sqrt_dim(self.factor_labels(a));
        }
        self.sqrt_dim(a).recip()
    }
    fn inv_dim(&self, a: &LieIrrep) -> f64 {
        if let Some(product) = &self.composite {
            self.label(a);
            return product.inv_dim(self.factor_labels(a));
        }
        self.dim(a).recip()
    }
    fn frobenius_schur_phase(&self, a: &LieIrrep) -> f64 {
        self.label(a);
        if let Some(product) = &self.composite {
            return product.frobenius_schur_phase(self.factor_labels(a));
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = a.bcd() {
            return x.frobenius_schur_phase();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = a.generated() {
            return x.frobenius_schur_phase();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = a.exceptional() {
            // Self-dual labels return the stored closed form. An E6 complex
            // label generates σ_a σ_ā (§6.3); admission bounds its dimension
            // (and its dual's) by `u32::MAX`, so building both modules cannot
            // overflow, and §6.3 proves the principal ladder nonzero. What is
            // left is an internal-invariant `Generation` failure.
            return f64::from(self.engine().frobenius_schur_phase(x).expect(
                "E6 complex-irrep κ_a: admission bounds the dimension by u32::MAX and \
                 §6.3 proves ⟨low|L^h|high⟩ ≠ 0; only an internal invariant can fail",
            ));
        }
        su2::fs_phase(a.dj())
    }
    fn frobenius_schur_indicator(&self, a: &LieIrrep) -> f64 {
        self.label(a);
        if let Some(product) = &self.composite {
            return product.frobenius_schur_indicator(self.factor_labels(a));
        }
        if self.is_torus() {
            return U1.frobenius_schur_indicator(&a.torus_label());
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = a.bcd() {
            return f64::from(x.frobenius_schur());
        }
        if *a == self.dual(a) {
            self.frobenius_schur_phase(a)
        } else {
            0.0
        }
    }
    fn twist(&self, a: &LieIrrep) -> f64 {
        self.label(a);
        if let Some(product) = &self.composite {
            return product.twist(self.factor_labels(a));
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = a.bcd() {
            return x.twist();
        }
        // Bosonic, twist one (`docs/gauge_exceptional.md` §7).
        #[cfg(feature = "lie-generated")]
        if a.exceptional().is_some() {
            return 1.0;
        }
        su2::twist(a.dj())
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
                        .map_err(LieError::from)?,
                ),
            });
        }
        if self.is_torus() {
            let q = U1
                .fusion(&a.torus_label(), &b.torus_label())
                .map_err(|_| LieError::LabelOverflow)?
                .next()
                .expect("torus fusion has one outcome");
            return Ok(LieFusion {
                group: Some(self.group.clone()),
                inner: FusionInner::Torus(Some(q)),
            });
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b)) = (a.generated(), b.generated()) {
            let product = sun::Product(self.roots().product(&a.0, &b.0)?);
            let back = product.0.len();
            return Ok(LieFusion {
                group: Some(self.group.clone()),
                inner: FusionInner::Generated {
                    product,
                    front: 0,
                    back,
                },
            });
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b)) = (a.bcd(), b.bcd()) {
            let product = self.roots().product(&a.0, &b.0)?;
            let back = product.len();
            return Ok(LieFusion {
                group: Some(self.group.clone()),
                inner: FusionInner::Bcd {
                    product,
                    front: 0,
                    back,
                },
            });
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b)) = (a.exceptional(), b.exceptional()) {
            let product = self.engine().product(a, b)?;
            let back = product.len();
            return Ok(LieFusion {
                group: Some(self.group.clone()),
                inner: FusionInner::Exceptional {
                    product,
                    front: 0,
                    back,
                },
            });
        }
        let (a, b) = (a.dj(), b.dj());
        let inner = su2::A1Irrep::new(a)
            .fusion(su2::A1Irrep::new(b))
            .map_err(|_| LieError::LabelOverflow)?;
        Ok(LieFusion {
            group: Some(self.group.clone()),
            inner: FusionInner::A1(inner),
        })
    }
    /// Panics if fusion arithmetic overflows or generation fails.
    /// Use [`Self::try_nsymbol`] to receive the typed error.
    #[inline]
    fn nsymbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> u32 {
        self.try_nsymbol(a, b, c)
            .expect("Lie fusion arithmetic failed; use try_nsymbol for a typed error")
    }
    /// Generated fusion checks label/weight sums and multiplicities, returning
    /// [`LieError::LabelOverflow`] on overflow. Torus N is exact even when the
    /// sum is not a representable label. Rank-one N is always zero or one.
    #[inline]
    fn try_nsymbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<u32, LieError> {
        self.label(a);
        self.label(b);
        self.label(c);
        self.__fusion_symbols_validated_try_nsymbol(a, b, c, Validated(()))
    }
    #[inline(always)]
    fn __fusion_symbols_validated_try_nsymbol(
        &self,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
        _: Validated,
    ) -> Result<u32, LieError> {
        if let Some(product) = &self.composite {
            return product
                .__fusion_symbols_validated_try_nsymbol(
                    self.factor_labels(a),
                    self.factor_labels(b),
                    self.factor_labels(c),
                    Validated(()),
                )
                .map_err(LieError::from);
        }
        match (&a.payload, &b.payload, &c.payload) {
            (LiePayload::Torus(a), LiePayload::Torus(b), LiePayload::Torus(c)) => {
                return Ok(U1.nsymbol(a, b, c))
            }
            (LiePayload::RankOne(a), LiePayload::RankOne(b), LiePayload::RankOne(c)) => {
                return Ok(u32::from(su2::triangle_ok(*a, *b, *c)))
            }
            _ => {}
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (a.generated(), b.generated(), c.generated()) {
            return self.roots().multiplicity(&a.0, &b.0, &c.0);
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (a.bcd(), b.bcd(), c.bcd()) {
            return self.roots().multiplicity(&a.0, &b.0, &c.0);
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (a.exceptional(), b.exceptional(), c.exceptional()) {
            return self.engine().nsymbol(a, b, c);
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
                .map_err(LieError::from);
        }
        if matches!(a.payload, LiePayload::Torus(_)) {
            let mut dims = [0; 4];
            for (n, (a, b, c)) in dims
                .iter_mut()
                .zip([(a, b, e), (e, c, d), (b, c, f), (a, f, d)])
            {
                *n = U1.nsymbol(&a.torus_label(), &b.torus_label(), &c.torus_label()) as usize;
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
            return Ok(if dims.contains(&0) {
                FBlock::zeros(dims)
            } else {
                FBlock::one(su2::f_symbol(a, b, c, d, e, f))
            });
        }
        #[cfg(feature = "lie-generated")]
        if let [Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)] =
            [a, b, c, d, e, f].map(|x| x.generated())
        {
            return Ok(sun::f_symbol(a, b, c, d, e, f)?);
        }
        #[cfg(feature = "lie-generated")]
        if let [Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)] =
            [a, b, c, d, e, f].map(|x| x.bcd())
        {
            return bcd::f_symbol(
                self.catalog
                    .as_ref()
                    .expect("B/C/D objects are constructed only with a B/C/D catalog"),
                a,
                b,
                c,
                d,
                e,
                f,
            );
        }
        #[cfg(feature = "lie-generated")]
        if let [Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)] =
            [a, b, c, d, e, f].map(|x| x.exceptional())
        {
            return self.engine().f_symbol([a, b, c, d, e, f]);
        }
        unreachable!("higher-rank labels have a generated payload")
    }
    fn __fusion_symbols_validated_f_symbol(
        &self,
        a: &LieIrrep,
        b: &LieIrrep,
        c: &LieIrrep,
        d: &LieIrrep,
        e: &LieIrrep,
        f: &LieIrrep,
        t: Validated,
    ) -> Result<crate::category::sealed::SymbolWithAxes<FBlock<f64>, 4>, LieError> {
        let block = self.f_symbol(a, b, c, d, e, f)?;
        // Keep Product's independent N-axis gate without validating each label again.
        let n = |x, y, z| self.__fusion_symbols_validated_try_nsymbol(x, y, z, t);
        Ok((block, [n(a, b, e)?, n(e, c, d)?, n(b, c, f)?, n(a, f, d)?]))
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
                .map_err(LieError::from);
        }
        if matches!(a.payload, LiePayload::Torus(_)) {
            return Ok(
                if self.__fusion_symbols_validated_try_nsymbol(a, b, c, Validated(()))? == 0 {
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
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (a.generated(), b.generated(), c.generated()) {
            return Ok(sun::r_symbol(a, b, c)?);
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (a.bcd(), b.bcd(), c.bcd()) {
            return bcd::r_symbol(
                self.catalog
                    .as_ref()
                    .expect("B/C/D objects are constructed only with a B/C/D catalog"),
                a,
                b,
                c,
            );
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (a.exceptional(), b.exceptional(), c.exceptional()) {
            return self.engine().r_symbol(a, b, c);
        }
        unreachable!("higher-rank labels have a generated payload")
    }
    fn a_symbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<RBlock<f64>, LieError> {
        if let Some(product) = &self.composite {
            let [a, b, c] = [a, b, c].map(|x| {
                self.label(x);
                self.factor_labels(x)
            });
            return product.a_symbol(a, b, c).map_err(LieError::from);
        }
        let n = checked_dimension(self.try_nsymbol(a, b, c)?)?;
        checked_size([n, n])?;
        if n == 0 {
            return Ok(RBlock::zeros(0));
        }
        let f = self.f_symbol(&self.dual(a), a, b, b, &self.unit(), c)?;
        let kappa = self.frobenius_schur_phase(a);
        let scale = self.sqrt_dim(a) * self.sqrt_dim(b) / self.sqrt_dim(c);
        Ok(crate::category::rank2_block(
            n,
            f.data().iter().map(|&x| scale * (kappa * x)),
        ))
    }
    fn b_symbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<RBlock<f64>, LieError> {
        if let Some(product) = &self.composite {
            let [a, b, c] = [a, b, c].map(|x| {
                self.label(x);
                self.factor_labels(x)
            });
            return product.b_symbol(a, b, c).map_err(LieError::from);
        }
        let n = checked_dimension(self.try_nsymbol(a, b, c)?)?;
        checked_size([n, n])?;
        if n == 0 {
            return Ok(RBlock::zeros(0));
        }
        let f = self.f_symbol(a, b, &self.dual(b), a, c, &self.unit())?;
        let scale = self.sqrt_dim(a) * self.sqrt_dim(b) / self.sqrt_dim(c);
        Ok(crate::category::rank2_block(
            n,
            f.data().iter().map(|&x| scale * x),
        ))
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
    // triangle sum), rounded exactly. Generated groups contract four CGC
    // (`frcore::f_block_raw`); with a unit leg two of them are the
    // identity embeddings of `𝟙 ⊗ s → s` / `s ⊗ 𝟙 → s` (`docs/gauge.md` §7;
    // B/C/D: the unit's frame is unique, `docs/gauge_soN.md`) and the other
    // two are the same isometry `x ⊗ y → z`, so the block is its column
    // orthonormality `Σ C_κ C_ν = δ_{κν}`: the identity, up to rounding.
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
                let x: Vec<Integer> = a.dynkin_labels().map(Integer::from).collect();
                u64::try_from(&crate::category::rank(&bounds, &x))
                    .map_err(|_| EnumerationError::Overflow)
            }
        }
    }
    fn authority_fingerprint(&self) -> &[u8] {
        if let Some(product) = &self.composite {
            return product.authority_fingerprint();
        }
        if self.is_torus() {
            return U1.authority_fingerprint();
        }
        #[cfg(feature = "lie-generated")]
        if self.catalog.is_some() {
            return bcd::bcd_authority_fingerprint();
        }
        #[cfg(feature = "lie-generated")]
        if self.general.is_some() {
            return general::fingerprint();
        }
        #[cfg(feature = "lie-generated")]
        if matches!(self.group.root_systems()[0], RootSystem::A(r) if r >= 2) {
            return sun::fingerprint();
        }
        su2::fingerprint()
    }
}
impl GroupRepresentations for LieGroup {
    type Cgc = LieCgc;
    fn cgc(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<LieCgc, LieError> {
        self.label(a);
        self.label(b);
        self.label(c);
        if let Some(product) = &self.composite {
            return Ok(LieCgc {
                composite: Some(
                    product
                        .cgc(
                            self.factor_labels(a),
                            self.factor_labels(b),
                            self.factor_labels(c),
                        )
                        .map_err(LieError::from)?,
                ),
                dims: [0; 4],
                data: Vec::new(),
                #[cfg(feature = "lie-generated")]
                generated: None,
                #[cfg(feature = "lie-generated")]
                bcd: None,
            });
        }
        if self.is_torus() {
            return Ok(LieCgc {
                composite: None,
                dims: [
                    1,
                    1,
                    1,
                    self.__fusion_symbols_validated_try_nsymbol(a, b, c, Validated(()))? as usize,
                ],
                data: Vec::new(),
                #[cfg(feature = "lie-generated")]
                generated: None,
                #[cfg(feature = "lie-generated")]
                bcd: None,
            });
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (a.generated(), b.generated(), c.generated()) {
            return Ok(LieCgc {
                composite: None,
                dims: [0; 4],
                data: Vec::new(),
                generated: Some(sun::cgc(a, b, c)?),
                bcd: None,
            });
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (a.bcd(), b.bcd(), c.bcd()) {
            let dims = [
                checked_dimension(&a.dim())?,
                checked_dimension(&b.dim())?,
                checked_dimension(&c.dim())?,
                bcd::fr::mult(a, b, c)?,
            ];
            return Ok(LieCgc {
                composite: None,
                dims,
                data: Vec::new(),
                generated: None,
                bcd: if dims[3] == 0 {
                    None
                } else {
                    Some(bcd::fr::cached_cgc(
                        self.catalog
                            .as_ref()
                            .expect("B/C/D objects are constructed only with a B/C/D catalog"),
                        a,
                        b,
                        c,
                    )?)
                },
            });
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (a.exceptional(), b.exceptional(), c.exceptional()) {
            let (dims, data) = self.engine().cgc(a, b, c)?;
            return Ok(LieCgc {
                composite: None,
                dims,
                data,
                generated: None,
                bcd: None,
            });
        }
        let (a, b, c) = (a.dj(), b.dj(), c.dj());
        // Exact evaluators take i32 projections; reject rather than truncate.
        if su2::triangle_ok(a, b, c) && [a, b, c].iter().any(|&dj| dj > i32::MAX as u32) {
            return Err(LieError::Unsupported {
                requires: "CGC projections beyond i32",
            });
        }
        let dims = [
            a1_dimension(a)?,
            a1_dimension(b)?,
            a1_dimension(c)?,
            usize::from(su2::triangle_ok(a, b, c)),
        ];
        let mut data = Vec::new();
        if dims[3] != 0 {
            data.resize(checked_size(dims)?, 0.0);
            for m1 in 0..=a {
                let dm1 = a1_projection(i64::from(m1), i64::from(a))?;
                for m2 in 0..=b {
                    let dm2 = a1_projection(i64::from(m2), i64::from(b))?;
                    let dm3 = dm1.checked_add(dm2).ok_or(LieError::LabelOverflow)?;
                    // Triangle parity makes (dm3 + c) even; only this projection can be nonzero.
                    if dm3.abs() <= i64::from(c) {
                        let m3 = checked_dimension(
                            dm3.checked_add(i64::from(c))
                                .ok_or(LieError::LabelOverflow)?
                                / 2,
                        )?;
                        data[(checked_dimension(m1)? * dims[1] + checked_dimension(m2)?)
                            * dims[2]
                            + m3] = su2::clebsch_gordan(
                            a,
                            checked_projection(dm1)?,
                            b,
                            checked_projection(dm2)?,
                            c,
                            checked_projection(dm3)?,
                        )
                        .to_f64();
                    }
                }
            }
        }
        Ok(LieCgc {
            composite: None,
            dims,
            data,
            #[cfg(feature = "lie-generated")]
            generated: None,
            #[cfg(feature = "lie-generated")]
            bcd: None,
        })
    }
}

/// Deterministic fusion outcomes, sharing cached products for higher rank.
#[derive(Clone, Debug)]
pub struct LieFusion {
    // Taken by the torus's sole outcome; other iterators retain their group.
    group: Option<GroupId>,
    inner: FusionInner,
}
type CompositeFusion = <Product<Vec<LieGroup>> as FusionCategory>::Fusion<'static>;

#[derive(Clone, Debug)]
enum FusionInner {
    Torus(Option<U1Irrep>),
    Composite(CompositeFusion),
    #[cfg(feature = "lie-generated")]
    Bcd {
        product: general::roots::Product,
        front: usize,
        back: usize,
    },
    A1(su2::A1Fusion),
    #[cfg(feature = "lie-generated")]
    Generated {
        product: sun::Product,
        front: usize,
        back: usize,
    },
    #[cfg(feature = "lie-generated")]
    Exceptional {
        product: general::roots::Product,
        front: usize,
        back: usize,
    },
}
impl LieFusion {
    fn next_label(&mut self, reverse: bool) -> Option<LieIrrep> {
        let group = self.group.as_ref();
        match &mut self.inner {
            FusionInner::Torus(label) => label.take().map(|q| LieIrrep {
                group: self
                    .group
                    .take()
                    .expect("torus fusion yields its group once"),
                payload: LiePayload::Torus(q),
            }),
            FusionInner::Composite(inner) => (if reverse {
                inner.next_back()
            } else {
                inner.next()
            })
            .map(|factors| {
                LieGroup::composite_label(group.expect("fusion retains its group"), factors)
            }),
            #[cfg(feature = "lie-generated")]
            FusionInner::Exceptional {
                product,
                front,
                back,
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
                Some(LieIrrep {
                    group: group.expect("fusion retains its group").clone(),
                    payload: LiePayload::Exceptional(product[i].0.clone()),
                })
            }
            #[cfg(feature = "lie-generated")]
            FusionInner::Bcd {
                product,
                front,
                back,
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
                Some(LieIrrep {
                    group: group.expect("fusion retains its group").clone(),
                    payload: LiePayload::Bcd(bcd::Irrep(product[i].0.clone())),
                })
            }
            FusionInner::A1(inner) => (if reverse {
                inner.next_back()
            } else {
                inner.next()
            })
            .map(|a| LieIrrep {
                group: group.expect("fusion retains its group").clone(),
                payload: LiePayload::RankOne(a.dj()),
            }),
            #[cfg(feature = "lie-generated")]
            FusionInner::Generated {
                product,
                front,
                back,
            } => {
                if *front == *back {
                    return None;
                }
                let index = if reverse {
                    *back -= 1;
                    *back
                } else {
                    let i = *front;
                    *front += 1;
                    i
                };
                Some(LieIrrep {
                    group: group.expect("fusion retains its group").clone(),
                    payload: LiePayload::Generated(product.channel(index)),
                })
            }
        }
    }
}
impl Iterator for LieFusion {
    type Item = LieIrrep;
    fn next(&mut self) -> Option<LieIrrep> {
        self.next_label(false)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match &self.inner {
            FusionInner::Torus(label) => usize::from(label.is_some()),
            FusionInner::Composite(inner) => inner.len(),
            FusionInner::A1(inner) => inner.len(),
            #[cfg(feature = "lie-generated")]
            FusionInner::Generated { front, back, .. }
            | FusionInner::Bcd { front, back, .. }
            | FusionInner::Exceptional { front, back, .. } => back - front,
        };
        (n, Some(n))
    }
}
impl DoubleEndedIterator for LieFusion {
    fn next_back(&mut self) -> Option<LieIrrep> {
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
                let exact = match (a.generated(), a.bcd()) {
                    (Some(x), None) => x.ratio_dim(),
                    (None, Some(x)) => x.ratio_dim(),
                    _ => unreachable!("{id:?} {d:?}"),
                };
                let want = f64::rounding_from(&exact, Nearest).0;
                assert_eq!(cat.dim(&a).to_bits(), want.to_bits(), "{id:?} {d:?}");
                assert_eq!(
                    cat.sqrt_dim(&a).to_bits(),
                    want.sqrt().to_bits(),
                    "{id:?} {d:?}"
                );
                checked += 1;
                above_2_53 += usize::from(exact > (1u64 << 53));
            }
        }
        assert!(checked > 1000 && above_2_53 > 10, "{checked} {above_2_53}");
    }

    #[test]
    fn torus_composites_warm_calls_do_not_generate() {
        let _guard = crate::cache::TEST_CACHE_LOCK.lock().unwrap();
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
            let before = sun::cgc::GENERATIONS.with(|n| n.get());
            assert!(before > 0);
            query();
            assert_eq!(sun::cgc::GENERATIONS.with(|n| n.get()), before);
        }
    }

    #[test]
    fn composite_seed_overflow_reports_caller_budget() {
        let group = GroupId::cover(&[RootSystem::B(usize::MAX / 4), RootSystem::A(1)], 0).unwrap();
        let error = LieGroup::with_catalog_budget(group, 1024).unwrap_err();
        assert!(matches!(
            error
                .source()
                .unwrap()
                .source()
                .unwrap()
                .downcast_ref::<bcd::CatalogError>(),
            Some(bcd::CatalogError::BudgetExceeded {
                limit: 1024,
                needed: usize::MAX,
            })
        ));
    }

    #[test]
    fn composite_catalog_budget_split_and_reset() {
        let _guard = crate::cache::TEST_CACHE_LOCK.lock().unwrap();
        use RootSystem::{A, B, C, D};
        let byte_budget = |s: CatalogStats| (s.bytes, s.budget);
        let roots = [A(1), B(2), D(2), C(2)];
        let group = GroupId::cover(&roots, 0).unwrap();
        let seeds: Vec<_> = [(bcd::Series::B, 2), (bcd::Series::C, 2)]
            .into_iter()
            .map(|(s, r)| bcd::CanonicalCatalog::seed_requirement(s, r, usize::MAX).unwrap())
            .collect();
        let total: usize = seeds.iter().sum();
        assert!(matches!(
            LieGroup::with_catalog_budget(group.clone(), total - 1),
            Err(LieError::Generation(_))
        ));
        for extra in [0, 5] {
            let cat = LieGroup::with_catalog_budget(group.clone(), total + extra).unwrap();
            let leaves = cat.composite.as_ref().unwrap().factors();
            assert_eq!(
                byte_budget(leaves[1].catalog_stats().unwrap()),
                (seeds[0], seeds[0] + extra / 2 + extra % 2)
            );
            assert_eq!(
                byte_budget(leaves[4].catalog_stats().unwrap()),
                (seeds[1], seeds[1] + extra / 2)
            );
            assert_eq!(
                byte_budget(cat.catalog_stats().unwrap()),
                (total, total + extra)
            );
            // Each catalog seeds exactly its trivial and defining generator sets.
            assert_eq!(cat.catalog_stats().unwrap().entries, 2 * 2);
            cat.reset_catalog().unwrap();
            assert_eq!(
                byte_budget(cat.catalog_stats().unwrap()),
                (total, total + extra)
            );
            assert_eq!(cat.catalog_stats().unwrap().entries, 2 * 2);
        }
        let cat = LieGroup::new(group).unwrap();
        assert_eq!(
            byte_budget(cat.catalog_stats().unwrap()),
            (total, bcd::catalog::DEFAULT_MAX_BYTES)
        );
        // Generation grows the B(2) catalog; reset returns to the seed count.
        let b2 = &cat.composite.as_ref().unwrap().factors()[1];
        let s = b2.irrep(&[0, 1]).unwrap();
        let adj = b2.irrep(&[0, 2]).unwrap();
        b2.cgc(&s, &s, &adj).unwrap();
        assert!(cat.catalog_stats().unwrap().entries > 2 * 2);
        cat.reset_catalog().unwrap();
        assert_eq!(cat.catalog_stats().unwrap().entries, 2 * 2);
    }

    #[test]
    fn composite_warm_calls_reuse_cover_caches() {
        use std::sync::atomic::Ordering;
        let _guard = crate::cache::TEST_CACHE_LOCK.lock().unwrap();
        crate::cache::reset();
        // Cover and quotient share all generated leaf keys; D2's A1 leaves
        // have exact closed forms and no numerical generation sweep.
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
        let sun = sun::cgc::GENERATIONS.with(std::cell::Cell::get);
        let bcd = bcd::fr::CGC_SWEEPS.load(Ordering::Relaxed);
        assert!(sun > 0);
        assert!(bcd > 0);
        query(&cover);
        query(&quotient);
        assert_eq!(sun::cgc::GENERATIONS.with(std::cell::Cell::get), sun);
        assert_eq!(bcd::fr::CGC_SWEEPS.load(Ordering::Relaxed), bcd);
    }

    #[test]
    fn composite_error_mapping_retains_factor_path() {
        let error = LieError::from(ProductError::Factor {
            index: 2,
            source: LieError::from(sun::SunError::Linalg("test failure".into())),
        });
        assert!(error.to_string().contains("factor 2"));
        let source = error.source().unwrap().source().unwrap();
        assert!(source.downcast_ref::<CompositeError>().is_some());
        assert!(source
            .source()
            .unwrap()
            .to_string()
            .contains("test failure"));
        assert_eq!(
            LieError::from(ProductError::Factor {
                index: 2,
                source: LieError::LabelOverflow
            }),
            LieError::LabelOverflow
        );
        assert_eq!(
            LieError::from(ProductError::Overflow),
            LieError::LabelOverflow
        );
        let unsupported = LieError::Unsupported {
            requires: "test capability",
        };
        assert_eq!(
            LieError::from(ProductError::Factor {
                index: 2,
                source: unsupported.clone()
            }),
            unsupported
        );
        assert!(matches!(
            LieError::from(ProductError::MalformedSymbolData {
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
        use crate::frcore::{f_block_raw, Family};
        let g = LieGroup::new(GroupId::g2()).unwrap();
        let mut fam = general::cgc::GeneralFamily::new(RootSystem::G2);
        let (v, adj) = (vec![1, 0], vec![0, 1]);
        let [lv, ladj] = [&v, &adj].map(|x| g.irrep(x).unwrap());
        let got = g.f_symbol(&lv, &lv, &lv, &lv, &ladj, &ladj).unwrap();
        let want = f_block_raw(&mut fam, &v, &v, &v, &v, &adj, &adj).unwrap();
        assert_eq!(got.dims(), want.dims());
        let bits = |x: &[f64]| x.iter().map(|y| y.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(got.data()), bits(want.data()));
        let cgc = g.cgc(&lv, &lv, &ladj).unwrap();
        let entries = fam.cgc_entries(&v, &v, &adj).unwrap();
        for e in &entries {
            let at = [e.m1, e.m2, e.m3, e.mu].map(|i| i as usize);
            assert_eq!(cgc.at(at).to_bits(), e.value.to_bits());
        }
        let nonzero = cgc.data.iter().filter(|x| **x != 0.0).count();
        assert_eq!(nonzero, entries.len());
    }

    /// `reset_catalog` on any clone drops the shared G2 memo without
    /// changing values or catalog statistics.
    #[test]
    fn reset_catalog_drops_the_g2_memo() {
        let g = LieGroup::new(GroupId::g2()).unwrap();
        let clone = g.clone();
        let x = g.irrep(&[2, 0]).unwrap();
        let f = |g: &LieGroup| {
            let block = g.f_symbol(&x, &x, &x, &x, &x, &x).unwrap();
            block.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        };
        let before = f(&g);
        let engine = g.general.as_ref().unwrap();
        let (entries, bytes) = engine.memo_stats();
        assert!(entries > 0 && bytes > 0);
        clone.reset_catalog().unwrap();
        assert_eq!(engine.memo_stats(), (0, 0));
        assert_eq!(g.catalog_stats().unwrap(), CatalogStats::default());
        assert_eq!(f(&clone), before);

        let composite =
            LieGroup::new(GroupId::cover(&[RootSystem::G2, RootSystem::A(1)], 0).unwrap()).unwrap();
        let a = composite.irrep(&[1, 0, 1]).unwrap();
        composite.r_symbol(&a, &a, &composite.unit()).unwrap();
        let factor = &composite.composite.as_ref().unwrap().factors()[0];
        assert!(factor.general.as_ref().unwrap().memo_stats().0 > 0);
        composite.reset_catalog().unwrap();
        assert_eq!(factor.general.as_ref().unwrap().memo_stats(), (0, 0));
    }

    /// Admitting a large complex E6 label builds nothing; its indicator needs
    /// no module. The first phase query builds `27` and `27̄` and memoizes
    /// `κ` for both; later queries on either label are memo hits.
    #[test]
    fn e6_complex_phase_is_generated_once_on_first_use() {
        let g = LieGroup::new(GroupId::e6()).unwrap();
        let engine = g.general.as_ref().unwrap();
        let big = g.irrep(&[1, 0, 0, 0, 1, 0]).unwrap(); // 7371
        assert_eq!(big.dim_exact(), Natural::from(7371u32));
        assert_eq!(g.frobenius_schur_indicator(&big), 0.0);
        assert_eq!(engine.memo_stats(), (0, 0));
        let v = g.irrep(&[1, 0, 0, 0, 0, 0]).unwrap();
        assert_eq!(g.frobenius_schur_phase(&v), 1.0);
        let after = engine.memo_stats();
        assert_eq!(after.0, 4, "modules 27 and 27̄, phases of both");
        assert!(after.1 > 0);
        assert_eq!(g.frobenius_schur_phase(&v), 1.0);
        assert_eq!(g.frobenius_schur_phase(&g.dual(&v)), 1.0);
        assert_eq!(engine.memo_stats(), after);
    }

    #[test]
    fn generation_error_keeps_its_reason() {
        let error = LieError::from(sun::SunError::Linalg("test failure".into()));
        assert_eq!(
            error.to_string(),
            "dense factorization failed: test failure"
        );
        assert_eq!(
            error.source().unwrap().source().unwrap().to_string(),
            error.to_string()
        );
    }
}

#[cfg(feature = "lie-generated")]
fn bcd_series(root: RootSystem) -> Option<bcd::Series> {
    match root {
        RootSystem::B(_) => Some(bcd::Series::B),
        RootSystem::C(_) => Some(bcd::Series::C),
        RootSystem::D(_) => Some(bcd::Series::D),
        _ => None,
    }
}

#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    #[test]
    fn payload_order_matches_former_field_order() {
        fn former_cmp(a: &LieIrrep, b: &LieIrrep) -> std::cmp::Ordering {
            let order = a
                .group
                .cmp(&b.group)
                .then_with(|| a.factors().cmp(&b.factors()))
                .then_with(|| a.dj().cmp(&b.dj()))
                .then_with(|| a.charge().cmp(&b.charge()));
            #[cfg(feature = "lie-generated")]
            let order = order
                .then_with(|| a.generated().cmp(&b.generated()))
                .then_with(|| a.bcd().cmp(&b.bcd()));
            order
        }
        let groups = [
            GroupId::u(1),
            GroupId::su(2),
            GroupId::spin(3),
            GroupId::sp(2),
            GroupId::u(2),
            GroupId::spin(4),
        ];
        let mut groups: Vec<_> = groups.into_iter().map(Result::unwrap).collect();
        #[cfg(feature = "lie-generated")]
        groups.extend([
            GroupId::su(3).unwrap(),
            GroupId::sp(4).unwrap(),
            GroupId::spin(5).unwrap(),
        ]);
        let mut labels = Vec::new();
        for group in groups.drain(..) {
            let cat = LieGroup::new(group).unwrap();
            for bits in 0..1 << cat.group.semisimple_rank() {
                let dynkin: Vec<_> = (0..cat.group.semisimple_rank())
                    .map(|i| (bits >> i) & 1)
                    .collect();
                for q in -2..=2 {
                    let charges = vec![q; cat.group.torus_rank()];
                    if let Ok(label) = cat.irrep_with_charges(&dynkin, &charges) {
                        labels.push(label);
                    }
                }
            }
        }
        for a in &labels {
            for b in &labels {
                assert_eq!(a.cmp(b), former_cmp(a, b));
                assert_eq!(a == b, former_cmp(a, b).is_eq());
            }
        }
    }

    #[test]
    fn dimension_and_buffer_bounds() {
        assert_eq!(
            checked_dimension(usize::MAX as u128 + 1),
            Err(LieError::LabelOverflow)
        );
        assert_eq!(checked_count(usize::MAX, 1), Err(LieError::LabelOverflow));
        assert_eq!(
            checked_projection(i64::from(i32::MAX) + 1),
            Err(LieError::LabelOverflow)
        );
        assert_eq!(
            checked_projection(i64::from(i32::MIN) - 1),
            Err(LieError::LabelOverflow)
        );
        assert_eq!(
            a1_projection(i64::MAX / 2 + 1, 0),
            Err(LieError::LabelOverflow)
        );
        assert_eq!(a1_projection(0, i64::MIN), Err(LieError::LabelOverflow));
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
        assert_eq!(cat.try_nsymbol(&a, &b, &c), Ok(1));
        assert!(matches!(cat.fusion(&a, &b), Err(LieError::LabelOverflow)));
        let a = cat.irrep(&[i64::from(i32::MAX)]).unwrap();
        assert!(matches!(
            cat.cgc(&a, &a, &cat.unit()),
            Err(LieError::LabelOverflow)
        ));
    }

    #[cfg(feature = "lie-generated")]
    #[test]
    fn nested_generation_overflow_is_not_wrapped() {
        assert_eq!(
            LieError::from(sun::SunError::LabelOverflow),
            LieError::LabelOverflow
        );
        assert_eq!(
            LieError::from(bcd::BcdError::LabelOverflow),
            LieError::LabelOverflow
        );
        assert_eq!(
            LieError::from(bcd::CatalogError::Label(bcd::BcdError::LabelOverflow)),
            LieError::LabelOverflow
        );
        assert_eq!(
            LieError::from(bcd::CatalogError::Sweep(
                bcd::sweep::SweepError::LabelOverflow
            )),
            LieError::LabelOverflow
        );
    }
}

#[cfg(all(test, feature = "lie-generated"))]
mod root_payload_tests {
    use super::*;
    #[test]
    fn dim_is_label_owned_and_takes_no_global_lock() {
        use std::{sync::mpsc, time::Duration};
        let _serial = crate::cache::TEST_CACHE_LOCK.lock().unwrap();
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
        let catalog_locks = groups
            .iter()
            .filter_map(|g| g.catalog.as_ref())
            .map(|c| c.lock().unwrap())
            .collect::<Vec<_>>();
        let family_locks = groups
            .iter()
            .filter_map(|g| g.general.as_ref())
            .map(|e| e.family())
            .collect::<Vec<_>>();
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
        drop(family_locks);
        drop(catalog_locks);
        worker.join().unwrap();
        assert_eq!(
            result.unwrap(),
            expected,
            "dim must finish before any global/family/catalog lock is released"
        );
    }
}
