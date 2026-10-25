//! Positive-dimensional compact Lie groups: root systems and global forms via
//! [`LieGroup`] (including tori), and closed-form [`CU1`].

mod cu1;
mod u1;
pub use cu1::{CU1Error, CU1Fusion, CU1Irrep, CU1};
use u1::{U1Irrep, U1};

#[cfg(feature = "lie-generated")]
pub(crate) mod bcd;
mod gt;
#[cfg(feature = "lie-generated")]
pub(crate) mod sun;
pub use gt::GtPattern;
mod group;
pub mod su2;
/// Real F-symbol storage (compatibility alias).
pub type FBlock = crate::category::FBlock<f64>;
/// Real R-symbol storage (compatibility alias).
pub type RBlock = crate::category::RBlock<f64>;
pub use group::*;

use crate::category::{
    sealed::Validated, BraidingStyle, CgcBlock, EnumerationError, FusionCategory, FusionStyle,
    GroupRepresentations,
};
use crate::product::{Product, ProductError};
use num_bigint::BigInt;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A Lie group with its global form. A(1), B(1), and C(1) share the exact
/// SU(2) gauge and fingerprint; their single Dynkin label is doubled spin.
/// With `lie-generated`, higher A-series groups use the GT gauge and B/C/D groups
/// use the canonical generator bootstrap. Clones share retained generators.
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
/// - every composite quotient, other higher A-series global forms, and
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
/// weight buffer; warm duals reuse a budgeted group-side lookup.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LieIrrep {
    group: GroupId,
    factors: Option<Arc<Vec<LieIrrep>>>,
    dj: u32,
    charge: i32,
    #[cfg(feature = "lie-generated")]
    generated: Option<sun::Irrep>,
    #[cfg(feature = "lie-generated")]
    bcd: Option<bcd::Irrep>,
}

struct DynkinLabels<'a> {
    irrep: &'a LieIrrep,
    indices: std::ops::Range<usize>,
}

impl Iterator for DynkinLabels<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<i64> {
        let mut i = self.indices.next()?;
        if let Some(factors) = &self.irrep.factors {
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
        if let Some(a) = &self.irrep.generated {
            let w = a.weight();
            return Some(w[i] - w[i + 1]);
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = &self.irrep.bcd {
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
        Some(i64::from(self.irrep.dj))
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
    /// Wrong Dynkin/charge lengths or a negative Dynkin label.
    InvalidLabel,
    /// The group's global form excludes this weight.
    NotAdmissible,
    /// A label, weight, dimension or multiplicity exceeds the integer range.
    LabelOverflow,
    /// The named capability is unavailable.
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
    for root in group.root_systems() {
        let requires = match root {
            RootSystem::E6 => "E6 coefficient engine",
            RootSystem::E7 => "E7 coefficient engine",
            RootSystem::E8 => "E8 coefficient engine",
            RootSystem::F4 => "F4 coefficient engine",
            RootSystem::G2 => "G2 coefficient engine",
            _ => continue,
        };
        return Err(LieError::Unsupported { requires });
    }
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
    pub fn group_id(&self) -> &GroupId {
        &self.group
    }
    /// Construct a connected compact group with executable classical factors and any
    /// admitted character lattice. A(1), B(1), C(1), and D(2) need no feature;
    /// A/B/C ranks ≥ 2 and D ranks ≥ 3 require `lie-generated`.
    /// D(2) expands into α₁ then α₂, each in the exact A(1) gauge.
    /// Tori, U(N), semisimple×torus quotients, and the trivial group are supported.
    /// U(2) needs no feature; U(N) for N ≥ 3 requires `lie-generated`.
    /// Exceptional blocks return [`LieError::Unsupported`].
    /// Single factors retain their evaluator and gauge. Composites use ordered
    /// [`Product`] cover symbols and only restrict objects at admission.
    /// B/C/D seed charges exceeding the default aggregate catalog budget return
    /// [`LieError::Generation`]. No zero-weight label is allocated at setup.
    pub fn new(group: GroupId) -> Result<Self, LieError> {
        coefficient_roots(&group)?;
        if group.root_systems().is_empty() && group.torus_rank() == 1 && group.is_cover() {
            return Ok(Self {
                group,
                composite: None,
                #[cfg(feature = "lie-generated")]
                catalog: None,
                #[cfg(feature = "lie-generated")]
                unit: std::sync::OnceLock::new(),
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
                return Ok(Self {
                    group,
                    composite: None,
                    unit: std::sync::OnceLock::new(),
                    catalog: Some(std::sync::Arc::new(std::sync::Mutex::new(
                        bcd::CanonicalCatalog::new(series, root.rank())?,
                    ))),
                });
            }
            #[cfg(not(feature = "lie-generated"))]
            return Err(LieError::Unsupported {
                requires: "lie-generated",
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
                    group,
                    composite: None,
                    unit: std::sync::OnceLock::new(),
                    catalog: None,
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
            group,
            composite: None,
            #[cfg(feature = "lie-generated")]
            catalog: None,
            #[cfg(feature = "lie-generated")]
            unit: std::sync::OnceLock::new(),
        })
    }
    /// Set the retained-generator budget for a group with B/C/D catalogs.
    /// For composites, preflight the sum of all catalog seed charges, then
    /// split the remaining bytes equally among catalogs in factor order,
    /// giving the remainder to earlier factors. Each cap is its seed plus its
    /// share. Without any catalog this returns [`LieError::Unsupported`].
    pub fn with_catalog_budget(group: GroupId, max_bytes: usize) -> Result<Self, LieError> {
        coefficient_roots(&group)?;
        if group.torus_rank() != 0
            || group.root_systems().len() != 1
            || group.root_systems()[0] == RootSystem::D(2)
        {
            let result = Self::composite(group, max_bytes)?;
            if result.catalog_stats()?.1 == 0 {
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
                return Ok(Self {
                    group,
                    composite: None,
                    unit: std::sync::OnceLock::new(),
                    catalog: Some(std::sync::Arc::new(std::sync::Mutex::new(
                        bcd::CanonicalCatalog::with_budget(series, root.rank(), max_bytes)?,
                    ))),
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
            group,
            composite: Some(Product::new(factors)),
            #[cfg(feature = "lie-generated")]
            catalog: None,
            #[cfg(feature = "lie-generated")]
            unit: std::sync::OnceLock::new(),
        })
    }
    fn composite_label(group: &GroupId, factors: Vec<LieIrrep>) -> LieIrrep {
        let label = LieIrrep {
            group: group.clone(),
            factors: Some(Arc::new(factors)),
            dj: 0,
            charge: 0,
            #[cfg(feature = "lie-generated")]
            generated: None,
            #[cfg(feature = "lie-generated")]
            bcd: None,
        };
        debug_assert!(group.admits(&label.dynkin(), &label.charges().collect::<Vec<_>>()));
        label
    }
    fn factor_labels<'a>(&self, a: &'a LieIrrep) -> &'a Vec<LieIrrep> {
        self.label(a);
        a.factors
            .as_ref()
            .expect("composite object has factor labels")
    }
    /// Retained generator bytes and byte budget, or `(0, 0)` without a catalog.
    pub fn catalog_stats(&self) -> Result<(usize, usize), LieError> {
        if let Some(product) = &self.composite {
            return product
                .factors()
                .iter()
                .try_fold((0, 0), |(bytes, cap), factor| {
                    let (b, c) = factor.catalog_stats()?;
                    Ok((checked_count(bytes, b)?, checked_count(cap, c)?))
                });
        }
        #[cfg(feature = "lie-generated")]
        if let Some(cat) = &self.catalog {
            let cat = cat.lock().map_err(|_| poisoned_catalog())?;
            return Ok((cat.bytes(), cat.budget()));
        }
        Ok((0, 0))
    }
    /// Reset retained generators to the base seeds; no-op without a catalog.
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
        Ok(())
    }
    /// Validate label lengths and nonnegative Dynkin labels, then lattice
    /// admission, then engine representability. Torus charges lie in
    /// `-i32::MAX..=i32::MAX`; out-of-range charges return [`LieError::LabelOverflow`].
    pub fn irrep(&self, dynkin: &[i64], charges: &[i64]) -> Result<LieIrrep, LieError> {
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
                    let label =
                        factor.irrep(&dynkin[offset..end], &charges[charge_offset..charge_end]);
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
                factors: None,
                dj: 0,
                charge: 0,
                generated: Some(sun::Irrep::from_dynkin(dynkin)?),
                bcd: None,
            });
        }
        #[cfg(feature = "lie-generated")]
        if self.catalog.is_some() {
            return Ok(LieIrrep {
                group: self.group.clone(),
                factors: None,
                dj: 0,
                charge: 0,
                generated: None,
                bcd: Some(bcd::Irrep::from_dynkin_in(&self.group, dynkin)?),
            });
        }
        let [dj] = dynkin else {
            return Err(LieError::InvalidLabel);
        };
        let dj = u32::try_from(*dj).map_err(|_| LieError::LabelOverflow)?;
        Ok(LieIrrep {
            group: self.group.clone(),
            factors: None,
            dj,
            charge: 0,
            #[cfg(feature = "lie-generated")]
            generated: None,
            #[cfg(feature = "lie-generated")]
            bcd: None,
        })
    }
    fn is_torus(&self) -> bool {
        self.composite.is_none() && self.group.root_systems().is_empty()
    }
    fn torus_label(&self, q: U1Irrep) -> LieIrrep {
        let mut label = self.unit();
        label.charge = q.charge();
        label
    }
    #[inline]
    fn is_rank_one(&self) -> bool {
        matches!(
            self.group.root_systems(),
            [RootSystem::A(1) | RootSystem::B(1) | RootSystem::C(1)]
        )
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
        a.dj
    }
}
impl LieIrrep {
    #[inline]
    fn torus_label(&self) -> U1Irrep {
        U1Irrep::new(self.charge).expect("validated torus charge has a representable dual")
    }
    /// Integer torus charges, in the datum's charge-coordinate order.
    pub fn charges(&self) -> impl ExactSizeIterator<Item = i64> + '_ {
        (0..self.group.torus_rank()).map(|i| {
            self.factors
                .as_ref()
                .map_or(i64::from(self.charge), |factors| {
                    i64::from(factors[factors.len() - self.group.torus_rank() + i].charge)
                })
        })
    }
    /// GT basis in the frozen gauge order; absent for rank-one, B/C/D, and composite groups.
    pub fn gt_patterns(&self) -> Option<Vec<GtPattern>> {
        #[cfg(feature = "lie-generated")]
        if let Some(a) = &self.generated {
            return Some(a.patterns());
        }
        None
    }

    /// Highest weight in Dynkin coordinates, materialized in an owned allocation.
    /// Use [`Self::dynkin_labels`] for allocation-free label reading on hot paths.
    pub fn dynkin(&self) -> Vec<i64> {
        if self.factors.is_some() {
            return self.dynkin_labels().collect();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = &self.bcd {
            return a.dynkin();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = &self.generated {
            return a.dynkin();
        }
        self.dynkin_labels().collect()
    }

    /// Iterate over every highest-weight Dynkin coordinate, including zeros,
    /// in the same order as [`Self::dynkin`]. Construction and consumption
    /// allocate nothing; composite coordinates follow factor order. `len()` reports
    /// the number of remaining coordinates.
    pub fn dynkin_labels(&self) -> impl ExactSizeIterator<Item = i64> + '_ {
        DynkinLabels {
            irrep: self,
            indices: 0..self.group.semisimple_rank(),
        }
    }

    /// Exact dimension.
    pub fn dim_exact(&self) -> BigInt {
        if let Some(factors) = &self.factors {
            return factors.iter().map(Self::dim_exact).product();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = &self.bcd {
            return a.dim();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = &self.generated {
            return a.dim();
        }
        BigInt::from(u64::from(self.dj) + 1)
    }
    /// Weight multiplicities in Dynkin-then-charge coordinates, in ascending order.
    /// Panics if an intermediate weight or multiplicity is unrepresentable.
    pub fn weight_multiplicities(&self) -> BTreeMap<Vec<i64>, u64> {
        if let Some(factors) = &self.factors {
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
        if let Some(a) = &self.bcd {
            return a.dynkin_weight_multiplicities();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = &self.generated {
            let mut weights = BTreeMap::new();
            for p in a.patterns() {
                let w = sun::cgc::pattern_weight(&p, p.rank()).expect(
                    "infallible weight_multiplicities failed: GT row sum or weight exceeds i64",
                );
                let count = weights
                    .entry(w.windows(2).map(|x| x[0] - x[1]).collect())
                    .or_insert(0u64);
                *count = count
                    .checked_add(1)
                    .expect("weight counts are bounded by the allocated GT pattern count");
            }
            return weights;
        }
        if self.group.semisimple_rank() == 0 {
            return BTreeMap::from([(vec![i64::from(self.charge)], 1)]);
        }
        (0..=self.dj)
            .map(|i| (vec![i64::from(self.dj) - 2 * i64::from(i)], 1))
            .collect()
    }
}

/// Clebsch–Gordan data read through [`CgcBlock`], with axes `[dim(a), dim(b), dim(c), N(a,b,c)]`.
/// A1 magnetic index zero is the lowest weight, followed by raising in steps
/// of two; higher A-series magnetic indices follow the frozen GT basis order.
/// B/C/D indices follow the canonical bootstrap gauge in `docs/gauge_soN.md`.
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
    type FSymbol = FBlock;
    type RSymbol = RBlock;
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
                    self.irrep(&vec![0; self.group.semisimple_rank()], &[])
                        .expect("zero Dynkin labels have zero weight and are admitted by the constructed group")
                })
                .clone();
        }
        LieIrrep {
            group: self.group.clone(),
            factors: None,
            dj: 0,
            charge: 0,
            #[cfg(feature = "lie-generated")]
            generated: None,
            #[cfg(feature = "lie-generated")]
            bcd: None,
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
        // Dual keys store one label in the tiers' existing shared slice layout.
        #[cfg(feature = "lie-generated")]
        use crate::cache::{
            bcd_cgc_cache::{BcdKey, BcdValue},
            cache_bcd_cgc, cache_sun_product,
        };
        #[cfg(feature = "lie-generated")]
        if let Some(x) = &a.bcd {
            let tier = cache_bcd_cgc();
            let key = BcdKey::Dual(x.clone());
            let value = tier.get(&key).unwrap_or_else(|| {
                let dual = x.dual();
                tier.insert(
                    BcdKey::Dual(dual.clone()),
                    BcdValue::Product(std::sync::Arc::from([(x.clone(), 1)])),
                );
                tier.insert(key, BcdValue::Product(std::sync::Arc::from([(dual, 1)])))
            });
            let BcdValue::Product(dual) = value else {
                unreachable!("dual key stores a dual")
            };
            return LieIrrep {
                group: self.group.clone(),
                factors: None,
                dj: 0,
                charge: 0,
                generated: None,
                bcd: Some(dual[0].0.clone()),
            };
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = &a.generated {
            let tier = cache_sun_product();
            let key = sun::SunProductKey::Dual(x.clone());
            let value = tier.get(&key).unwrap_or_else(|| {
                let dual = x.dual();
                tier.insert(
                    sun::SunProductKey::Dual(dual.clone()),
                    sun::SunProduct::from_map(BTreeMap::from([(x.clone(), 1)])),
                );
                tier.insert(key, sun::SunProduct::from_map(BTreeMap::from([(dual, 1)])))
            });
            return LieIrrep {
                group: self.group.clone(),
                factors: None,
                dj: 0,
                charge: 0,
                generated: Some(value.channel(0).clone()),
                bcd: None,
            };
        }
        a.clone()
    }
    fn dim(&self, a: &LieIrrep) -> f64 {
        self.label(a);
        if let Some(product) = &self.composite {
            return product.dim(self.factor_labels(a));
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = &a.bcd {
            use num_traits::ToPrimitive;
            return x.dim().to_f64().unwrap_or(f64::INFINITY);
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = &a.generated {
            use num_traits::ToPrimitive;
            return x.dim().to_f64().unwrap_or(f64::INFINITY);
        }
        f64::from(a.dj) + 1.0
    }
    fn sqrt_dim(&self, a: &LieIrrep) -> f64 {
        if let Some(product) = &self.composite {
            return product.sqrt_dim(self.factor_labels(a));
        }
        self.dim(a).sqrt()
    }
    fn inv_sqrt_dim(&self, a: &LieIrrep) -> f64 {
        if let Some(product) = &self.composite {
            return product.inv_sqrt_dim(self.factor_labels(a));
        }
        self.sqrt_dim(a).recip()
    }
    fn inv_dim(&self, a: &LieIrrep) -> f64 {
        if let Some(product) = &self.composite {
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
        if let Some(x) = &a.bcd {
            return x.frobenius_schur_phase();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = &a.generated {
            return x.frobenius_schur_phase();
        }
        su2::fs_phase(a.dj)
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
        if let Some(x) = &a.bcd {
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
        if let Some(x) = &a.bcd {
            return x.twist();
        }
        su2::twist(a.dj)
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
        if let (Some(a), Some(b)) = (&a.generated, &b.generated) {
            let product = sun::shared_directproduct(a, b)?;
            let back = product.iter().count();
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
        if let (Some(a), Some(b)) = (&a.bcd, &b.bcd) {
            let product = bcd::shared_directproduct(a, b)?;
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
        let (a, b) = (a.dj, b.dj);
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
    /// Fusion checks label/weight sums and multiplicities, returning
    /// [`LieError::LabelOverflow`] on overflow. Rank-one N is always zero or one.
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
                .try_nsymbol(
                    self.factor_labels(a),
                    self.factor_labels(b),
                    self.factor_labels(c),
                )
                .map_err(LieError::from);
        }
        if self.is_torus() {
            return U1
                .try_nsymbol(&a.torus_label(), &b.torus_label(), &c.torus_label())
                .map_err(|_| LieError::LabelOverflow);
        }
        if self.is_rank_one() {
            return Ok(u32::from(su2::triangle_ok(a.dj, b.dj, c.dj)));
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (&a.generated, &b.generated, &c.generated) {
            return Ok(sun::shared_directproduct(a, b)?.multiplicity(c));
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (&a.bcd, &b.bcd, &c.bcd) {
            return Ok(bcd::shared_directproduct(a, b)?
                .iter()
                .find(|(x, _)| x == c)
                .map_or(0, |(_, n)| *n));
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
    ) -> Result<FBlock, LieError> {
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
        if self.is_torus() {
            let mut dims = [0; 4];
            for (n, (a, b, c)) in dims
                .iter_mut()
                .zip([(a, b, e), (e, c, d), (b, c, f), (a, f, d)])
            {
                *n = U1
                    .try_nsymbol(&a.torus_label(), &b.torus_label(), &c.torus_label())
                    .map_err(|_| LieError::LabelOverflow)? as usize;
            }
            return Ok(if dims.contains(&0) {
                FBlock::zeros(dims)
            } else {
                FBlock::one(1.0)
            });
        }
        if self.is_rank_one() {
            let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|x| x.dj);
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
            [a, b, c, d, e, f].map(|x| x.generated.as_ref())
        {
            return Ok(sun::f_symbol(a, b, c, d, e, f)?);
        }
        #[cfg(feature = "lie-generated")]
        if let [Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)] =
            [a, b, c, d, e, f].map(|x| x.bcd.as_ref())
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
        unreachable!("higher-rank labels have a generated payload")
    }
    fn r_symbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<RBlock, LieError> {
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
        if self.is_torus() {
            return Ok(
                if self.__fusion_symbols_validated_try_nsymbol(a, b, c, Validated(()))? == 0 {
                    RBlock::zeros(0)
                } else {
                    RBlock::one(1.0)
                },
            );
        }
        if self.is_rank_one() {
            return Ok(if su2::triangle_ok(a.dj, b.dj, c.dj) {
                RBlock::one(su2::r_symbol(a.dj, b.dj, c.dj))
            } else {
                RBlock::zeros(0)
            });
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (&a.generated, &b.generated, &c.generated) {
            return Ok(sun::r_symbol(a, b, c)?);
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (&a.bcd, &b.bcd, &c.bcd) {
            return bcd::r_symbol(
                self.catalog
                    .as_ref()
                    .expect("B/C/D objects are constructed only with a B/C/D catalog"),
                a,
                b,
                c,
            );
        }
        unreachable!("higher-rank labels have a generated payload")
    }
    fn a_symbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<RBlock, LieError> {
        if let Some(product) = &self.composite {
            let [a, b, c] = [a, b, c].map(|x| self.factor_labels(x));
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
    fn b_symbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<RBlock, LieError> {
        if let Some(product) = &self.composite {
            let [a, b, c] = [a, b, c].map(|x| self.factor_labels(x));
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
                crate::category::unrank(&bounds, &BigInt::from(index))
                    .ok_or(EnumerationError::OutOfRange)?
                    .into_iter()
                    .map(|a| i64::try_from(a).map_err(|_| EnumerationError::Overflow))
                    .collect::<Result<_, _>>()?
            }
        };
        // The labels are nonnegative, of the right rank and admitted, so only
        // `LieError::LabelOverflow` remains.
        self.irrep(&labels, &[])
            .map_err(|_| EnumerationError::Overflow)
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
                let x: Vec<BigInt> = a.dynkin_labels().map(BigInt::from).collect();
                u64::try_from(crate::category::rank(&bounds, &x))
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
        if let (Some(a), Some(b), Some(c)) = (&a.generated, &b.generated, &c.generated) {
            return Ok(LieCgc {
                composite: None,
                dims: [0; 4],
                data: Vec::new(),
                generated: Some(sun::cgc(a, b, c)?),
                bcd: None,
            });
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (&a.bcd, &b.bcd, &c.bcd) {
            let dims = [
                checked_dimension(a.dim())?,
                checked_dimension(b.dim())?,
                checked_dimension(c.dim())?,
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
        let (a, b, c) = (self.label(a), self.label(b), self.label(c));
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
        product: std::sync::Arc<[(bcd::Irrep, u32)]>,
        front: usize,
        back: usize,
    },
    A1(su2::A1Fusion),
    #[cfg(feature = "lie-generated")]
    Generated {
        product: sun::SunProduct,
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
                factors: None,
                dj: 0,
                charge: q.charge(),
                #[cfg(feature = "lie-generated")]
                generated: None,
                #[cfg(feature = "lie-generated")]
                bcd: None,
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
                    factors: None,
                    dj: 0,
                    charge: 0,
                    generated: None,
                    bcd: Some(product[i].0.clone()),
                })
            }
            FusionInner::A1(inner) => (if reverse {
                inner.next_back()
            } else {
                inner.next()
            })
            .map(|a| LieIrrep {
                group: group.expect("fusion retains its group").clone(),
                factors: None,
                dj: a.dj(),
                charge: 0,
                #[cfg(feature = "lie-generated")]
                generated: None,
                #[cfg(feature = "lie-generated")]
                bcd: None,
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
                    factors: None,
                    dj: 0,
                    charge: 0,
                    generated: Some(product.channel(index).clone()),
                    bcd: None,
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
            FusionInner::Generated { front, back, .. } | FusionInner::Bcd { front, back, .. } => {
                back - front
            }
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

#[cfg(all(test, feature = "lie-generated"))]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn torus_composites_warm_calls_do_not_generate() {
        let _guard = crate::cache::TEST_CACHE_LOCK.lock().unwrap();
        crate::cache::reset();
        let cover = LieGroup::new(GroupId::cover(&[RootSystem::A(2)], 1).unwrap()).unwrap();
        let quotient = LieGroup::new(GroupId::u(3).unwrap()).unwrap();
        for cat in [&cover, &quotient] {
            let a = cat.irrep(&[1, 0], &[1]).unwrap();
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
        use RootSystem::{A, B, C, D};
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
                leaves[1].catalog_stats().unwrap(),
                (seeds[0], seeds[0] + extra / 2 + extra % 2)
            );
            assert_eq!(
                leaves[4].catalog_stats().unwrap(),
                (seeds[1], seeds[1] + extra / 2)
            );
            assert_eq!(cat.catalog_stats().unwrap(), (total, total + extra));
            cat.reset_catalog().unwrap();
            assert_eq!(cat.catalog_stats().unwrap(), (total, total + extra));
        }
        assert_eq!(
            LieGroup::new(group).unwrap().catalog_stats().unwrap(),
            (total, bcd::catalog::DEFAULT_MAX_BYTES)
        );
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
            let a = cat.irrep(&[2, 1, 1, 1, 0], &[]).unwrap();
            let e = cat.irrep(&[2, 1, 1, 0, 0], &[]).unwrap();
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
        let a = cat.irrep(&[i64::from(u32::MAX)], &[]).unwrap();
        let b = cat.irrep(&[1], &[]).unwrap();
        let c = cat.irrep(&[i64::from(u32::MAX) - 1], &[]).unwrap();
        assert_eq!(cat.try_nsymbol(&a, &b, &c), Ok(1));
        assert!(matches!(cat.fusion(&a, &b), Err(LieError::LabelOverflow)));
        let a = cat.irrep(&[i64::from(i32::MAX)], &[]).unwrap();
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
