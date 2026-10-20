//! Lie-group representations and exact rank-one symbols.

mod blocks;
mod group;
pub mod su2;
pub use blocks::{FBlock, RBlock};
pub use group::*;

use crate::category::{BraidingStyle, FusionCategory, FusionStyle, GroupRepresentations};
use num_bigint::BigInt;
use std::collections::BTreeMap;

/// A Lie group with its global form. This slice supports A(1), B(1), and
/// C(1) through the same exact SU(2) gauge and fingerprint. The single Dynkin
/// label is doubled spin in each case; admissibility uses the caller's form.
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LieIrrep {
    group: GroupId,
    dj: u32,
}

/// Invalid labels, excluded representations, arithmetic failure, or absent capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LieError {
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
impl std::fmt::Display for LieError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLabel => f.write_str("expected one nonnegative Dynkin label"),
            Self::NotAdmissible => f.write_str("weight is not admitted by this global form"),
            Self::LabelOverflow => f.write_str("Dynkin label exceeds u32"),
            Self::Unsupported { requires } => write!(f, "unsupported capability: {requires}"),
        }
    }
}
impl std::error::Error for LieError {}

impl LieGroup {
    /// Construct a supported Lie group; higher-rank generation arrives separately.
    pub fn new(group: GroupId) -> Result<Self, LieError> {
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
        let [dj] = dynkin else {
            return Err(LieError::InvalidLabel);
        };
        if *dj < 0 {
            return Err(LieError::InvalidLabel);
        }
        let dj = u32::try_from(*dj).map_err(|_| LieError::LabelOverflow)?;
        if !self.group.admits(dynkin) {
            return Err(LieError::NotAdmissible);
        }
        Ok(LieIrrep {
            group: self.group,
            dj,
        })
    }
    fn label(&self, a: &LieIrrep) -> u32 {
        assert_eq!(self.group, a.group, "foreign Lie object");
        a.dj
    }
}
impl LieIrrep {
    /// Highest weight in Dynkin coordinates.
    pub fn dynkin(&self) -> Vec<i64> {
        vec![i64::from(self.dj)]
    }
    /// Exact dimension.
    pub fn dim_exact(&self) -> BigInt {
        BigInt::from(u64::from(self.dj) + 1)
    }
    /// Weight multiplicities in Dynkin coordinates, in ascending coordinate order.
    pub fn weight_multiplicities(&self) -> BTreeMap<Vec<i64>, u64> {
        (0..=self.dj)
            .map(|i| (vec![i64::from(self.dj) - 2 * i64::from(i)], 1))
            .collect()
    }
}

/// Dense rank-one Clebsch–Gordan data. Row-major axes are
/// `[dim(a), dim(b), dim(c), N(a,b,c)]`, as in `sun::Cgc`; magnetic
/// index zero is the lowest weight, followed by raising in steps of two.
/// Materialization uses the cached exact CG evaluator. Projection labels larger
/// than `i32::MAX` are an unsupported capability of that evaluator.
#[derive(Clone, Debug)]
pub struct LieCgc {
    dims: [usize; 4],
    data: Vec<f64>,
}
impl LieCgc {
    /// Axis lengths, with multiplicity last (zero for an absent channel).
    pub fn dims(&self) -> [usize; 4] {
        self.dims
    }
    /// Read one coefficient.
    ///
    /// # Panics
    /// Panics if any index is outside its axis.
    pub fn at(&self, m1: usize, m2: usize, m3: usize, mu: usize) -> f64 {
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
        FusionStyle::Simple
    }
    fn braiding_style(&self) -> BraidingStyle {
        BraidingStyle::Bosonic
    }
    fn unit(&self) -> LieIrrep {
        LieIrrep {
            group: self.group,
            dj: 0,
        }
    }
    fn dual(&self, a: &LieIrrep) -> LieIrrep {
        self.label(a);
        *a
    }
    fn dim(&self, a: &LieIrrep) -> f64 {
        f64::from(self.label(a)) + 1.0
    }
    fn frobenius_schur_phase(&self, a: &LieIrrep) -> f64 {
        su2::fs_phase(self.label(a))
    }
    fn frobenius_schur_indicator(&self, a: &LieIrrep) -> f64 {
        self.frobenius_schur_phase(a)
    }
    fn twist(&self, a: &LieIrrep) -> f64 {
        su2::twist(self.label(a))
    }
    fn fusion<'a>(&'a self, a: &LieIrrep, b: &LieIrrep) -> Result<Self::Fusion<'a>, LieError> {
        let (a, b) = (self.label(a), self.label(b));
        let inner = su2::A1Irrep::new(a)
            .fusion(su2::A1Irrep::new(b))
            .map_err(|_| LieError::LabelOverflow)?;
        Ok(LieFusion {
            group: self.group,
            inner,
        })
    }
    fn nsymbol(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> u32 {
        let (a, b, c) = (self.label(a), self.label(b), self.label(c));
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
        let (a, b, c) = (self.label(a), self.label(b), self.label(c));
        Ok(if su2::triangle_ok(a, b, c) {
            RBlock::one(su2::r_symbol(a, b, c))
        } else {
            RBlock::zeros(0)
        })
    }
    fn authority_fingerprint(&self) -> &[u8] {
        su2::fingerprint()
    }
}
impl GroupRepresentations for LieGroup {
    type Cgc = LieCgc;
    fn cgc(&self, a: &LieIrrep, b: &LieIrrep, c: &LieIrrep) -> Result<LieCgc, LieError> {
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
        Ok(LieCgc { dims, data })
    }
}

/// Allocation-free, ascending rank-one fusion outcomes.
#[derive(Clone, Copy, Debug)]
pub struct LieFusion {
    group: GroupId,
    inner: su2::A1Fusion,
}
impl Iterator for LieFusion {
    type Item = LieIrrep;
    fn next(&mut self) -> Option<LieIrrep> {
        self.inner.next().map(|a| LieIrrep {
            group: self.group,
            dj: a.dj(),
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
impl DoubleEndedIterator for LieFusion {
    fn next_back(&mut self) -> Option<LieIrrep> {
        self.inner.next_back().map(|a| LieIrrep {
            group: self.group,
            dj: a.dj(),
        })
    }
}
impl ExactSizeIterator for LieFusion {}
