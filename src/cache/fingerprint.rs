//! Label fingerprints and the hasher of the label tiers (F, R, CGC, Product).
//!
//! A label's fingerprint is computed once, when its irrep is constructed, and
//! stored beside its words, so a warm query combines one `u64` per label rather
//! than hashing every word. Owned keys recompute the same fingerprints from
//! their words; they are hashed only on a miss, an eviction or a resize. Keys
//! still compare every word, so a collision can never select another key's
//! value.
//!
//! This file depends only on [`RootSystem`] so that `benches/warm_hits.rs` can
//! include it by path and report the distribution of this exact pipeline.

use crate::lie::RootSystem;
use std::hash::{BuildHasher, Hasher};

const P0: u64 = 0xa076_1d64_78bd_642f;
const P1: u64 = 0xe703_7ed1_a0b4_28db;
const P2: u64 = 0x8ebc_6af0_9c88_c6e3;
const P3: u64 = 0x5899_65cc_7537_4cc3;

/// The folded 64×64→128 multiply of wyhash and foldhash.
#[inline(always)]
fn fold(a: u64, b: u64) -> u64 {
    let x = u128::from(a) * u128::from(b);
    x as u64 ^ (x >> 64) as u64
}

/// One Dynkin label of a key with its fingerprint.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Label<'a> {
    pub(crate) words: &'a [i64],
    pub(crate) hash: u64,
}

impl<'a> Label<'a> {
    /// Fingerprints `words`. Warm queries from a constructed irrep take its
    /// stored fingerprint instead.
    pub(crate) fn new(words: &'a [i64]) -> Self {
        Self {
            words,
            hash: label_hash(words),
        }
    }
}

/// The fingerprint of one label: its length, then each word, in order.
#[inline]
pub(crate) fn label_hash(words: &[i64]) -> u64 {
    words
        .iter()
        .fold(P0 ^ words.len() as u64, |h, &w| fold(h ^ w as u64, P1))
}

/// A key's hash: the key form `tag`, the presented root, then each label
/// fingerprint in key order. The combine is sequential, so label
/// permutations do not collapse.
#[inline]
pub(crate) fn key_hash(tag: u64, root: RootSystem, parts: impl IntoIterator<Item = u64>) -> u64 {
    parts
        .into_iter()
        .fold(fold(tag ^ P2, root_code(root) ^ P3), |h, f| fold(h ^ f, P1))
}

/// `family << 32 | rank`: distinct for distinct presented roots.
fn root_code(root: RootSystem) -> u64 {
    let (family, rank) = match root {
        RootSystem::A(r) => (1, r),
        RootSystem::B(r) => (2, r),
        RootSystem::C(r) => (3, r),
        RootSystem::D(r) => (4, r),
        RootSystem::G2 => (5, 2),
        RootSystem::F4 => (6, 4),
        RootSystem::E6 => (7, 6),
        RootSystem::E7 => (8, 7),
        RootSystem::E8 => (9, 8),
        // `RootSystem` is non-exhaustive for the benchmark that includes this
        // file (compiled with `cfg(test)`); library builds keep the match
        // exhaustive. A shared code would cost lookup time only, never a value.
        #[cfg(test)]
        #[allow(unreachable_patterns)]
        _ => (0, root.rank()),
    };
    (family << 32) | rank as u64
}

/// The label tiers' hasher. Their keys write one already-mixed `u64`, so one
/// folded multiply finishes it; a key that wrote more would still hash
/// consistently.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FoldBuild;

pub(crate) struct Fold(u64);

impl BuildHasher for FoldBuild {
    type Hasher = Fold;
    fn build_hasher(&self) -> Fold {
        Fold(P0)
    }
}

impl Hasher for Fold {
    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.0 = fold(self.0 ^ x, P3);
    }
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}
