//! Lie-group representations: exact rank-one symbols and generated A-series data.

mod gt;
#[cfg(feature = "cgc-gen")]
pub(crate) mod sun;
pub use gt::GtPattern;
mod blocks;
mod group;
pub mod su2;
pub use blocks::{FBlock, RBlock};
pub use group::*;

use crate::category::{BraidingStyle, FusionCategory, FusionStyle, GroupRepresentations};
use num_bigint::BigInt;
use std::collections::BTreeMap;

/// A Lie group with its global form. A(1), B(1), and C(1) share the exact
/// SU(2) gauge and fingerprint; their single Dynkin label is doubled spin.
/// With `cgc-gen`, A(r) for r >= 2 uses the generated GT gauge. In every
/// case admissibility uses the caller's global form.
/// Objects retain the full `GroupId` tag, including for isomorphic groups.
///
/// # Panics
/// Every object-taking category method panics on an object from another group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LieGroup {
    group: GroupId,
}

/// A validated representation, including its group's identity. Rank-one
/// labels are stored inline, without allocation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LieIrrep {
    group: GroupId,
    dj: u32,
    #[cfg(feature = "cgc-gen")]
    generated: Option<sun::Irrep>,
}

/// Invalid labels, excluded representations, arithmetic failure, or absent capability.
#[derive(Clone, Debug, PartialEq)]
pub enum LieError {
    /// A coefficient generation failed.
    Generation(GenerationError),
    /// Wrong rank or negative Dynkin label.
    InvalidLabel,
    /// The group's global form excludes this weight.
    NotAdmissible,
    /// A label cannot be represented by `u32`.
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
#[cfg(feature = "cgc-gen")]
impl From<sun::SunError> for LieError {
    fn from(error: sun::SunError) -> Self {
        Self::Generation(GenerationError(std::sync::Arc::new(error)))
    }
}
impl std::fmt::Display for LieError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generation(error) => error.fmt(f),
            Self::InvalidLabel => f.write_str("expected rank-many nonnegative Dynkin labels"),
            Self::NotAdmissible => f.write_str("weight is not admitted by this global form"),
            Self::LabelOverflow => f.write_str("Dynkin label exceeds u32"),
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
    /// Construct a rank-one group or, with `cgc-gen`, an A-series group.
    pub fn new(group: GroupId) -> Result<Self, LieError> {
        if !group.admits(&vec![0; group.root_system.rank()]) {
            return Err(LieError::NotAdmissible);
        }
        if matches!(group.root_system, RootSystem::A(r) if r >= 2) {
            #[cfg(feature = "cgc-gen")]
            return Ok(Self { group });
            #[cfg(not(feature = "cgc-gen"))]
            return Err(LieError::Unsupported {
                requires: "cgc-gen",
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
        Ok(Self { group })
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
        #[cfg(feature = "cgc-gen")]
        if matches!(self.group.root_system, RootSystem::A(r) if r >= 2) {
            return Ok(LieIrrep {
                group: self.group,
                dj: 0,
                generated: Some(sun::Irrep::from_dynkin(dynkin)?),
            });
        }
        let [dj] = dynkin else {
            return Err(LieError::InvalidLabel);
        };
        let dj = u32::try_from(*dj).map_err(|_| LieError::LabelOverflow)?;
        Ok(LieIrrep {
            group: self.group,
            dj,
            #[cfg(feature = "cgc-gen")]
            generated: None,
        })
    }
    fn label(&self, a: &LieIrrep) -> u32 {
        assert_eq!(self.group, a.group, "foreign Lie object");
        a.dj
    }
}
impl LieIrrep {
    /// GT basis in the frozen gauge order; absent for rank-one groups.
    pub fn gt_patterns(&self) -> Option<Vec<GtPattern>> {
        #[cfg(feature = "cgc-gen")]
        if let Some(a) = &self.generated {
            return Some(a.patterns());
        }
        None
    }

    /// Highest weight in Dynkin coordinates.
    pub fn dynkin(&self) -> Vec<i64> {
        #[cfg(feature = "cgc-gen")]
        if let Some(a) = &self.generated {
            return a.dynkin();
        }
        vec![i64::from(self.dj)]
    }
    /// Exact dimension.
    pub fn dim_exact(&self) -> BigInt {
        #[cfg(feature = "cgc-gen")]
        if let Some(a) = &self.generated {
            return a.dim();
        }
        BigInt::from(u64::from(self.dj) + 1)
    }
    /// Weight multiplicities in Dynkin coordinates, in ascending coordinate order.
    pub fn weight_multiplicities(&self) -> BTreeMap<Vec<i64>, u64> {
        #[cfg(feature = "cgc-gen")]
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
/// Materialization uses the cached exact CG evaluator. Projection labels larger
/// than `i32::MAX` are an unsupported capability of that evaluator.
#[derive(Clone, Debug, PartialEq)]
pub struct LieCgc {
    dims: [usize; 4],
    data: Vec<f64>,
    #[cfg(feature = "cgc-gen")]
    generated: Option<sun::Cgc>,
}
impl LieCgc {
    /// Axis lengths, with multiplicity last (zero for an absent channel).
    pub fn dims(&self) -> [usize; 4] {
        #[cfg(feature = "cgc-gen")]
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
        #[cfg(feature = "cgc-gen")]
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
        if matches!(self.group.root_system, RootSystem::A(r) if r >= 2) {
            FusionStyle::Generic
        } else {
            FusionStyle::Simple
        }
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> LieIrrep {
        #[cfg(feature = "cgc-gen")]
        if matches!(self.group.root_system, RootSystem::A(r) if r >= 2) {
            return self
                .irrep(&vec![0; self.group.root_system.rank()])
                .expect("unit is admissible");
        }
        LieIrrep {
            group: self.group,
            dj: 0,
            #[cfg(feature = "cgc-gen")]
            generated: None,
        }
    }
    fn dual(&self, a: &LieIrrep) -> LieIrrep {
        self.label(a);
        #[cfg(feature = "cgc-gen")]
        if let Some(x) = &a.generated {
            return LieIrrep {
                group: self.group,
                dj: 0,
                generated: Some(x.dual()),
            };
        }
        a.clone()
    }
    fn dim(&self, a: &LieIrrep) -> f64 {
        self.label(a);
        #[cfg(feature = "cgc-gen")]
        if let Some(x) = &a.generated {
            use num_traits::ToPrimitive;
            return x.dim().to_f64().unwrap_or(f64::INFINITY);
        }
        f64::from(a.dj) + 1.0
    }
    fn frobenius_schur_phase(&self, a: &LieIrrep) -> f64 {
        self.label(a);
        #[cfg(feature = "cgc-gen")]
        if let Some(x) = &a.generated {
            return x.frobenius_schur_phase();
        }
        su2::fs_phase(a.dj)
    }
    fn frobenius_schur_indicator(&self, a: &LieIrrep) -> f64 {
        self.label(a);
        if *a == self.dual(a) {
            self.frobenius_schur_phase(a)
        } else {
            0.0
        }
    }
    fn twist(&self, a: &LieIrrep) -> f64 {
        self.label(a);
        su2::twist(a.dj)
    }
    fn fusion<'a>(&'a self, a: &LieIrrep, b: &LieIrrep) -> Result<Self::Fusion<'a>, LieError> {
        self.label(a);
        self.label(b);
        #[cfg(feature = "cgc-gen")]
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
        #[cfg(feature = "cgc-gen")]
        if let (Some(a), Some(b), Some(c)) = (&a.generated, &b.generated, &c.generated) {
            return sun::shared_directproduct(a, b)
                .expect("validated product labels")
                .multiplicity(c);
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
        #[cfg(feature = "cgc-gen")]
        if let [Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)] =
            [a, b, c, d, e, f].map(|x| x.generated.as_ref())
        {
            return Ok(sun::f_symbol(a, b, c, d, e, f)?);
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
        #[cfg(feature = "cgc-gen")]
        if let (Some(a), Some(b), Some(c)) = (&a.generated, &b.generated, &c.generated) {
            return Ok(sun::r_symbol(a, b, c)?);
        }
        let (a, b, c) = (self.label(a), self.label(b), self.label(c));
        Ok(if su2::triangle_ok(a, b, c) {
            RBlock::one(su2::r_symbol(a, b, c))
        } else {
            RBlock::zeros(0)
        })
    }
    fn authority_fingerprint(&self) -> &[u8] {
        #[cfg(feature = "cgc-gen")]
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
        #[cfg(feature = "cgc-gen")]
        if let (Some(a), Some(b), Some(c)) = (&a.generated, &b.generated, &c.generated) {
            return Ok(LieCgc {
                dims: [0; 4],
                data: Vec::new(),
                generated: Some(sun::cgc(a, b, c)?),
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
            for m1 in 0..=a {
                for m2 in 0..=b {
                    for m3 in 0..=c {
                        data.push(
                            su2::clebsch_gordan(
                                a,
                                (2 * i64::from(m1) - i64::from(a)) as i32,
                                b,
                                (2 * i64::from(m2) - i64::from(b)) as i32,
                                c,
                                (2 * i64::from(m3) - i64::from(c)) as i32,
                            )
                            .to_f64(),
                        );
                    }
                }
            }
        }
        Ok(LieCgc {
            dims,
            data,
            #[cfg(feature = "cgc-gen")]
            generated: None,
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
    A1(su2::A1Fusion),
    #[cfg(feature = "cgc-gen")]
    Generated {
        product: sun::SunProduct,
        front: usize,
        back: usize,
    },
}
impl LieFusion {
    fn next_label(&mut self, reverse: bool) -> Option<LieIrrep> {
        match &mut self.inner {
            FusionInner::A1(inner) => (if reverse {
                inner.next_back()
            } else {
                inner.next()
            })
            .map(|a| LieIrrep {
                group: self.group,
                dj: a.dj(),
                #[cfg(feature = "cgc-gen")]
                generated: None,
            }),
            #[cfg(feature = "cgc-gen")]
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
            #[cfg(feature = "cgc-gen")]
            FusionInner::Generated { front, back, .. } => back - front,
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

#[cfg(all(test, feature = "cgc-gen"))]
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
