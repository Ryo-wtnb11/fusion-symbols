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

use crate::category::{BraidingStyle, FusionCategory, FusionStyle, GroupRepresentations};
use num_bigint::BigInt;
use std::collections::BTreeMap;

/// A Lie group with its global form. A(1), B(1), and C(1) share the exact
/// SU(2) gauge and fingerprint; their single Dynkin label is doubled spin.
/// With `lie-generated`, higher A-series groups use the GT gauge and B/C/D groups
/// use the canonical generator bootstrap. Clones share retained generators.
/// Admissibility uses the caller's global form.
/// Objects retain the full `GroupId` tag, including for isomorphic groups.
///
/// # Panics
/// Every object-taking category method except `contains` panics on an object
/// from another group.
#[derive(Clone)]
pub struct LieGroup {
    group: GroupId,
    #[cfg(feature = "lie-generated")]
    catalog: Option<std::sync::Arc<std::sync::Mutex<bcd::CanonicalCatalog>>>,
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
/// labels are stored inline, without allocation.
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
                ((i128::from(w[i]) - i128::from(w[i + 1])) / 2) as i64
            } else {
                match a.series() {
                    bcd::Series::B => w[i],
                    bcd::Series::C => w[i] / 2,
                    bcd::Series::D => ((i128::from(w[i - 1]) + i128::from(w[i])) / 2) as i64,
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

/// Invalid labels, excluded representations, arithmetic failure, or absent capability.
#[derive(Clone, Debug, PartialEq)]
pub enum LieError {
    /// A coefficient generation failed.
    Generation(GenerationError),
    /// Wrong rank or negative Dynkin label.
    InvalidLabel,
    /// The group's global form excludes this weight.
    NotAdmissible,
    /// A label or its internal weight encoding exceeds the integer range.
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
            Self::InvalidLabel => f.write_str("expected rank-many nonnegative Dynkin labels"),
            Self::NotAdmissible => f.write_str("weight is not admitted by this global form"),
            Self::LabelOverflow => f.write_str("label or weight exceeds the integer range"),
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

impl LieGroup {
    /// Construct a supported compact Lie group.
    pub fn new(group: GroupId) -> Result<Self, LieError> {
        if !group.admits(&vec![0; group.root_system.rank()]) {
            return Err(LieError::NotAdmissible);
        }
        if group.root_system == RootSystem::D(2) {
            return Err(LieError::Unsupported {
                requires: "Spin(4) = SU(2) × SU(2) product",
            });
        }
        if matches!(group.root_system, RootSystem::B(r) | RootSystem::C(r) | RootSystem::D(r) if r >= 2)
        {
            #[cfg(feature = "lie-generated")]
            {
                let series = bcd_series(group.root_system).expect("B/C/D root system");
                return Ok(Self {
                    group,
                    catalog: Some(std::sync::Arc::new(std::sync::Mutex::new(
                        bcd::CanonicalCatalog::new(series, group.root_system.rank())?,
                    ))),
                });
            }
            #[cfg(not(feature = "lie-generated"))]
            return Err(LieError::Unsupported {
                requires: "lie-generated",
            });
        }
        if matches!(group.root_system, RootSystem::A(r) if r >= 2) {
            #[cfg(feature = "lie-generated")]
            return Ok(Self {
                group,
                catalog: None,
            });
            #[cfg(not(feature = "lie-generated"))]
            return Err(LieError::Unsupported {
                requires: "lie-generated",
            });
        }
        if !matches!(
            group.root_system,
            RootSystem::A(1) | RootSystem::B(1) | RootSystem::C(1)
        ) {
            return Err(LieError::Unsupported {
                requires: "higher-rank Lie representations",
            });
        }
        Ok(Self {
            group,
            #[cfg(feature = "lie-generated")]
            catalog: None,
        })
    }
    /// Construct a B/C/D group with a byte budget for retained generators.
    pub fn with_catalog_budget(group: GroupId, max_bytes: usize) -> Result<Self, LieError> {
        if !group.admits(&vec![0; group.root_system.rank()]) {
            return Err(LieError::NotAdmissible);
        }
        #[cfg(feature = "lie-generated")]
        if let Some(series) = bcd_series(group.root_system) {
            if group.root_system != RootSystem::D(2) && group.root_system.rank() >= 2 {
                return Ok(Self {
                    group,
                    catalog: Some(std::sync::Arc::new(std::sync::Mutex::new(
                        bcd::CanonicalCatalog::with_budget(
                            series,
                            group.root_system.rank(),
                            max_bytes,
                        )?,
                    ))),
                });
            }
        }
        let _ = max_bytes;
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
            cat.lock().map_err(|_| poisoned_catalog())?.reset();
        }
        Ok(())
    }
    /// Validate Dynkin labels and the group's global-form admissibility.
    pub fn irrep(&self, dynkin: &[i64]) -> Result<LieIrrep, LieError> {
        if dynkin.len() != self.group.root_system.rank() || dynkin.iter().any(|&x| x < 0) {
            return Err(LieError::InvalidLabel);
        }
        if dynkin
            .iter()
            .try_fold(0i64, |sum, &x| sum.checked_add(x))
            .is_none()
        {
            return Err(LieError::LabelOverflow);
        }
        if !self.group.admits(dynkin) {
            return Err(LieError::NotAdmissible);
        }
        #[cfg(feature = "lie-generated")]
        if matches!(self.group.root_system, RootSystem::A(r) if r >= 2) {
            return Ok(LieIrrep {
                group: self.group,
                dj: 0,
                generated: Some(sun::Irrep::from_dynkin(dynkin)?),
                bcd: None,
            });
        }
        #[cfg(feature = "lie-generated")]
        if self.catalog.is_some() {
            return Ok(LieIrrep {
                group: self.group,
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
            group: self.group,
            dj,
            #[cfg(feature = "lie-generated")]
            generated: None,
            #[cfg(feature = "lie-generated")]
            bcd: None,
        })
    }
    fn label(&self, a: &LieIrrep) -> u32 {
        assert_eq!(self.group, a.group, "foreign Lie object");
        a.dj
    }
}
impl LieIrrep {
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
            indices: 0..self.group.root_system.rank(),
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
    pub fn weight_multiplicities(&self) -> BTreeMap<Vec<i64>, u64> {
        #[cfg(feature = "lie-generated")]
        if let Some(a) = &self.bcd {
            return a.dynkin_weight_multiplicities();
        }
        #[cfg(feature = "lie-generated")]
        if let Some(a) = &self.generated {
            let mut weights = BTreeMap::new();
            for p in a.patterns() {
                let rowsum = |l| (1..=l).map(|k| p.get(k, l)).sum::<i64>();
                let w: Vec<_> = (1..=p.rank()).map(|l| rowsum(l) - rowsum(l - 1)).collect();
                *weights
                    .entry(w.windows(2).map(|x| x[0] - x[1]).collect())
                    .or_insert(0) += 1;
            }
            return weights;
        }
        (0..=self.dj)
            .map(|i| (vec![i64::from(self.dj) - 2 * i64::from(i)], 1))
            .collect()
    }
}

/// Clebsch–Gordan data with axes `[dim(a), dim(b), dim(c), N(a,b,c)]`.
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
impl LieCgc {
    /// Axis lengths, with multiplicity last (zero for an absent channel).
    pub fn dims(&self) -> [usize; 4] {
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
    pub fn at(&self, m1: usize, m2: usize, m3: usize, mu: usize) -> f64 {
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
                    [e.m1 as usize, e.m2 as usize, e.m3 as usize, e.mu as usize]
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
        if self.group.root_system.rank() >= 2 {
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
        if self.group.root_system.rank() >= 2 {
            return self
                .irrep(&vec![0; self.group.root_system.rank()])
                .expect("unit is admissible");
        }
        LieIrrep {
            group: self.group,
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
        #[cfg(feature = "lie-generated")]
        if let Some(x) = &a.bcd {
            return LieIrrep {
                group: self.group,
                dj: 0,
                generated: None,
                bcd: Some(x.dual()),
            };
        }
        #[cfg(feature = "lie-generated")]
        if let Some(x) = &a.generated {
            return LieIrrep {
                group: self.group,
                dj: 0,
                generated: Some(x.dual()),
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
                group: self.group,
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
                group: self.group,
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
            group: self.group,
            inner: FusionInner::A1(inner),
        })
    }
    fn nsymbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> u32 {
        self.label(a);
        self.label(b);
        self.label(c);
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (&a.generated, &b.generated, &c.generated) {
            return sun::shared_directproduct(a, b)
                .expect("validated product labels")
                .multiplicity(c);
        }
        #[cfg(feature = "lie-generated")]
        if let (Some(a), Some(b), Some(c)) = (&a.bcd, &b.bcd, &c.bcd) {
            return bcd::shared_directproduct(a, b)
                .expect("validated product labels")
                .iter()
                .find(|(x, _)| x == c)
                .map_or(0, |(_, n)| *n);
        }
        let (a, b, c) = (a.dj, b.dj, c.dj);
        u32::from(su2::triangle_ok(a, b, c))
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
                self.catalog.as_ref().expect("BCD catalog"),
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
            return bcd::r_symbol(self.catalog.as_ref().expect("BCD catalog"), a, b, c);
        }
        let (a, b, c) = (self.label(a), self.label(b), self.label(c));
        Ok(if su2::triangle_ok(a, b, c) {
            RBlock::one(su2::r_symbol(a, b, c))
        } else {
            RBlock::zeros(0)
        })
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
    fn authority_fingerprint(&self) -> &[u8] {
        #[cfg(feature = "lie-generated")]
        if self.catalog.is_some() {
            return bcd::bcd_authority_fingerprint();
        }
        #[cfg(feature = "lie-generated")]
        if matches!(self.group.root_system, RootSystem::A(r) if r >= 2) {
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
                usize::try_from(a.dim()).expect("tractable dimension"),
                usize::try_from(b.dim()).expect("tractable dimension"),
                usize::try_from(c.dim()).expect("tractable dimension"),
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
                        self.catalog.as_ref().expect("BCD catalog"),
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
            a as usize + 1,
            b as usize + 1,
            c as usize + 1,
            usize::from(su2::triangle_ok(a, b, c)),
        ];
        let mut data = Vec::new();
        if dims[3] != 0 {
            data.resize(dims.iter().product(), 0.0);
            for m1 in 0..=a {
                let dm1 = 2 * i64::from(m1) - i64::from(a);
                for m2 in 0..=b {
                    let dm2 = 2 * i64::from(m2) - i64::from(b);
                    let dm3 = dm1 + dm2;
                    // Triangle parity makes (dm3 + c) even; only this projection can be nonzero.
                    if dm3.abs() <= i64::from(c) {
                        let m3 = ((dm3 + i64::from(c)) / 2) as usize;
                        data[(m1 as usize * dims[1] + m2 as usize) * dims[2] + m3] =
                            su2::clebsch_gordan(a, dm1 as i32, b, dm2 as i32, c, dm3 as i32)
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
                    group: self.group,
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
                group: self.group,
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
                    group: self.group,
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
