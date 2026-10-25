//! Positive-dimensional compact Lie groups: root systems and global forms via
//! [`LieGroup`], and closed-form [`U1`] and [`CU1`].

mod cu1;
mod u1;
pub use cu1::{CU1Error, CU1Fusion, CU1Irrep, CU1};
pub use u1::{U1Error, U1Irrep, U1};

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
    BraidingStyle, CgcBlock, EnumerationError, FusionCategory, FusionStyle, GroupRepresentations,
};
use num_bigint::BigInt;
use std::collections::BTreeMap;

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
/// - every other global form and the B, C, D series of rank at least two:
///   [`EnumerationError::NotEnumerable`].
///
/// # Panics
/// Every object-taking category method except `contains` panics on an object
/// from another group.
#[derive(Clone)]
pub struct LieGroup {
    group: GroupId,
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
    dj: u32,
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
        let i = self.indices.next()?;
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

fn coefficient_root(group: &GroupId) -> Result<RootSystem, LieError> {
    let requires = if group.torus_rank() != 0 {
        "torus"
    } else {
        match group.root_systems() {
            [] => "trivial group",
            [RootSystem::E6] => "E6 coefficient engine",
            [RootSystem::E7] => "E7 coefficient engine",
            [RootSystem::E8] => "E8 coefficient engine",
            [RootSystem::F4] => "F4 coefficient engine",
            [RootSystem::G2] => "G2 coefficient engine",
            [root] => return Ok(*root),
            _ => "product of root systems",
        }
    };
    Err(LieError::Unsupported { requires })
}

impl LieGroup {
    /// The complete validated classification datum.
    pub fn group_id(&self) -> &GroupId {
        &self.group
    }
    /// Construct a supported single-block classical group, with no torus charges.
    /// Other classified data return [`LieError::Unsupported`]; higher ranks
    /// require `lie-generated`. No zero-weight label is allocated at setup.
    /// B/C/D base seeds exceeding the catalog budget return [`LieError::Generation`].
    pub fn new(group: GroupId) -> Result<Self, LieError> {
        let root = coefficient_root(&group)?;
        if root == RootSystem::D(2) {
            return Err(LieError::Unsupported {
                requires: "D2 presentation",
            });
        }
        if matches!(root, RootSystem::B(r) | RootSystem::C(r) | RootSystem::D(r) if r >= 2) {
            #[cfg(feature = "lie-generated")]
            {
                let series = bcd_series(root)
                    .expect("the enclosing match restricts the root system to B/C/D");
                return Ok(Self {
                    group,
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
            #[cfg(feature = "lie-generated")]
            catalog: None,
            #[cfg(feature = "lie-generated")]
            unit: std::sync::OnceLock::new(),
        })
    }
    /// Construct a B/C/D group with a byte budget for retained generators.
    pub fn with_catalog_budget(group: GroupId, max_bytes: usize) -> Result<Self, LieError> {
        let root = coefficient_root(&group)?;
        #[cfg(feature = "lie-generated")]
        if let Some(series) = bcd_series(root) {
            if root != RootSystem::D(2) && root.rank() >= 2 {
                return Ok(Self {
                    group,
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
    /// Retained generator bytes and byte budget, or `(0, 0)` without a catalog.
    pub fn catalog_stats(&self) -> Result<(usize, usize), LieError> {
        #[cfg(feature = "lie-generated")]
        if let Some(cat) = &self.catalog {
            let cat = cat.lock().map_err(|_| poisoned_catalog())?;
            return Ok((cat.bytes(), cat.budget()));
        }
        Ok((0, 0))
    }
    /// Reset retained generators to the base seeds; no-op without a catalog.
    pub fn reset_catalog(&self) -> Result<(), LieError> {
        #[cfg(feature = "lie-generated")]
        if let Some(cat) = &self.catalog {
            cat.lock().map_err(|_| poisoned_catalog())?.try_reset()?;
        }
        Ok(())
    }
    /// Validate label lengths and nonnegative Dynkin labels, then lattice
    /// admission, then engine representability. Current engines require `charges = &[]`.
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
                dj: 0,
                generated: Some(sun::Irrep::from_dynkin(dynkin)?),
                bcd: None,
            });
        }
        #[cfg(feature = "lie-generated")]
        if self.catalog.is_some() {
            return Ok(LieIrrep {
                group: self.group.clone(),
                dj: 0,
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
            dj,
            #[cfg(feature = "lie-generated")]
            generated: None,
            #[cfg(feature = "lie-generated")]
            bcd: None,
        })
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
    fn label(&self, a: &LieIrrep) -> u32 {
        assert_eq!(self.group, a.group, "foreign Lie object");
        a.dj
    }
}
impl LieIrrep {
    /// Integer torus charges (empty for every supported engine).
    pub fn charges(&self) -> impl ExactSizeIterator<Item = i64> + '_ {
        std::iter::empty()
    }
    /// GT basis in the frozen gauge order; absent for rank-one and B/C/D groups.
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
        #[cfg(feature = "lie-generated")]
        if let Some(a) = &self.bcd {
            return a.dynkin();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = &self.generated {
            return a.dynkin();
        }
        vec![i64::from(self.dj)]
    }

    /// Iterate over every highest-weight Dynkin coordinate, including zeros,
    /// in the same order as [`Self::dynkin`]. Construction and consumption
    /// allocate nothing; each coordinate takes O(1) time. `len()` reports
    /// the number of remaining coordinates.
    pub fn dynkin_labels(&self) -> impl ExactSizeIterator<Item = i64> + '_ {
        DynkinLabels {
            irrep: self,
            indices: 0..self.group.semisimple_rank(),
        }
    }

    /// Exact dimension.
    pub fn dim_exact(&self) -> BigInt {
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
    /// Weight multiplicities in Dynkin coordinates, in ascending coordinate order.
    /// Panics if an intermediate weight or multiplicity is unrepresentable.
    pub fn weight_multiplicities(&self) -> BTreeMap<Vec<i64>, u64> {
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
        (0..=self.dj)
            .map(|i| (vec![i64::from(self.dj) - 2 * i64::from(i)], 1))
            .collect()
    }
}

/// Clebsch–Gordan data read through [`CgcBlock`], with axes `[dim(a), dim(b), dim(c), N(a,b,c)]`.
/// A1 magnetic index zero is the lowest weight, followed by raising in steps
/// of two; higher A-series magnetic indices follow the frozen GT basis order.
/// B/C/D indices follow the canonical bootstrap gauge in `docs/gauge_soN.md`.
/// A1 materialization uses the cached exact CG evaluator. Projection labels larger
/// than `i32::MAX` are an unsupported capability of that evaluator.
#[derive(Clone, Debug, PartialEq)]
pub struct LieCgc {
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
        if self.group.semisimple_rank() >= 2 {
            FusionStyle::Generic
        } else {
            FusionStyle::Simple
        }
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> LieIrrep {
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
            dj: 0,
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
                dj: 0,
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
                dj: 0,
                generated: Some(value.channel(0).clone()),
                bcd: None,
            };
        }
        a.clone()
    }
    fn dim(&self, a: &LieIrrep) -> f64 {
        self.label(a);
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
    fn frobenius_schur_phase(&self, a: &LieIrrep) -> f64 {
        self.label(a);
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
        #[cfg(feature = "lie-generated")]
        if let Some(x) = &a.bcd {
            return x.twist();
        }
        su2::twist(a.dj)
    }
    fn fusion<'a>(&'a self, a: &LieIrrep, b: &LieIrrep) -> Result<Self::Fusion<'a>, LieError> {
        self.label(a);
        self.label(b);
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b)) = (&a.generated, &b.generated) {
            let product = sun::shared_directproduct(a, b)?;
            let back = product.iter().count();
            return Ok(LieFusion {
                group: self.group.clone(),
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
                group: self.group.clone(),
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
            group: self.group.clone(),
            inner: FusionInner::A1(inner),
        })
    }
    /// Panics if generated fusion arithmetic overflows or generation fails.
    /// Use [`Self::try_nsymbol`] to receive the typed error.
    fn nsymbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> u32 {
        self.try_nsymbol(a, b, c)
            .expect("Lie fusion arithmetic failed; use try_nsymbol for a typed error")
    }
    /// Generated fusion checks label/weight sums and multiplicities, returning
    /// [`LieError::LabelOverflow`] on overflow. Rank-one N is always zero or one.
    fn try_nsymbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<u32, LieError> {
        self.label(a);
        self.label(b);
        self.label(c);
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
        let (a, b, c) = (a.dj, b.dj, c.dj);
        Ok(u32::from(su2::triangle_ok(a, b, c)))
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
        let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|x| self.label(x));
        let dims = [(a, b, e), (e, c, d), (b, c, f), (a, f, d)]
            .map(|(a, b, c)| usize::from(su2::triangle_ok(a, b, c)));
        Ok(if dims.contains(&0) {
            FBlock::zeros(dims)
        } else {
            FBlock::one(su2::f_symbol(a, b, c, d, e, f))
        })
    }
    fn r_symbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<RBlock, LieError> {
        self.label(a);
        self.label(b);
        self.label(c);
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
        let (a, b, c) = (self.label(a), self.label(b), self.label(c));
        Ok(if su2::triangle_ok(a, b, c) {
            RBlock::one(su2::r_symbol(a, b, c))
        } else {
            RBlock::zeros(0)
        })
    }
    fn a_symbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<RBlock, LieError> {
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
    /// `false`: every group has an irrep of dimension `d > 1`, whose
    /// `|F^{a ā a}_a[𝟙, 𝟙]| = 1/d` (SU(2): `F^{½½½}_½[0, 0] = -1/2`).
    fn has_trivial_associator_gauge(&self) -> bool {
        false
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
        true
    }
    fn object_count(&self) -> Result<Option<u64>, EnumerationError> {
        self.enumeration()?;
        Ok(None)
    }
    fn object_at(&self, index: u64) -> Result<LieIrrep, EnumerationError> {
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
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (&a.generated, &b.generated, &c.generated) {
            return Ok(LieCgc {
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
    group: GroupId,
    inner: FusionInner,
}
#[derive(Clone, Debug)]
enum FusionInner {
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
        match &mut self.inner {
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
                    group: self.group.clone(),
                    dj: 0,
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
                group: self.group.clone(),
                dj: a.dj(),
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
                    group: self.group.clone(),
                    dj: 0,
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
