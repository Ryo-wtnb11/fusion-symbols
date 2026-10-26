//! B/C/D (SO(2r+1)/Sp(2r)/SO(2r)) F- and R-symbols from catalog-driven CGC.
//!
//! The four-CGC contraction and the pentagon/hexagon/F-unitarity gates are the
//! **family-generic** [`crate::frcore`] core, shared with `crate::lie::sun::fr`; this
//! module is the B/C/D binding of that core. The provider is [`BcdFamily`],
//! wrapping the per-group `Mutex<CanonicalCatalog>`, locked only after the global
//! F/CGC caches miss. It materializes canonical generator bases
//! on demand and adapts each dense [`CatalogCgc`] isometry into the sparse
//! `(m1, m2, m3, mu, value)` entries the core consumes.
//!
//! # Magnetic-index decomposition (the Kronecker gauge)
//!
//! A [`CatalogCgc`] copy is a dense column-major `d1·d2 × d3` isometry. Its row
//! index is the product-basis index `m1 + d1·m2` — the **first factor fast** —
//! fixed by `Generators::product` (QSpace `wbsparray::setRec_kron`, and pinned in
//! `docs/gauge_soN.md`); the column is the coupled index `m3`. So a row `row`
//! decomposes as `m1 = row % d1`, `m2 = row / d1`, with `d1 = dim(s1)`. Because
//! the catalog stores exactly one canonical generator set per irrep, a factor's
//! magnetic index means the same basis state across every CGC it appears in — the
//! shared-index joins in the core (`ma` in `CGC(a,b,e)` vs `CGC(a,f,d)`, etc.) are
//! therefore consistent.
//!
//! # Real-valued CGC
//!
//! The sweep produces real orthogonal isometries, so the reference conjugations
//! are the identity and elided (as for SU(N)); see [`crate::frcore`].

use super::{CanonicalCatalog, CatalogCgc, Irrep};
use crate::cache::bcd_cgc_cache::{BcdKey, BcdRef, BcdValue};
use crate::frcore::{f_block_raw, r_block_raw, Family, MEntry};
use crate::lie::{poisoned_catalog, LieError};
pub use crate::lie::{FBlock, RBlock};
use std::sync::{Arc, Mutex};

pub(crate) static CGC_SWEEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct BcdFamily<'a> {
    cat: &'a Mutex<CanonicalCatalog>,
}
impl Family for BcdFamily<'_> {
    type Irrep = Irrep;
    type Error = LieError;
    fn overflow() -> LieError {
        LieError::LabelOverflow
    }
    fn mult(&mut self, a: &Irrep, b: &Irrep, c: &Irrep) -> Result<usize, LieError> {
        mult(a, b, c)
    }

    fn cgc_entries(&mut self, a: &Irrep, b: &Irrep, c: &Irrep) -> Result<Vec<MEntry>, LieError> {
        sparse_entries(a, cached_product_cgc(self.cat, a, b, c)?.as_ref())
    }
}
pub(crate) fn mult(a: &Irrep, b: &Irrep, c: &Irrep) -> Result<usize, LieError> {
    super::super::checked_dimension(
        super::shared_directproduct(a, b)?
            .iter()
            .find(|(x, _)| x == c)
            .map_or(0, |(_, n)| *n),
    )
}
pub(crate) fn cached_cgc(
    cat: &Mutex<CanonicalCatalog>,
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
) -> Result<Arc<CatalogCgc>, LieError> {
    let tier = crate::cache::cache_bcd_cgc();
    if let Some(BcdValue::Cgc(hit)) = tier.get(&BcdRef::Cgc(a, b, c)) {
        return Ok(hit);
    }
    let value = tier.get_or_try_compute(
        &BcdRef::Cgc(a, b, c),
        || BcdKey::Cgc(a.clone(), b.clone(), c.clone()),
        || {
            let mut cat = cat.lock().map_err(|_| poisoned_catalog())?;
            if let Some(hit) = tier.peek(&BcdRef::Cgc(a, b, c)) {
                return Ok(hit);
            }
            CGC_SWEEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok::<_, LieError>(BcdValue::Cgc(Arc::new(cat.cgc(a, b, c)?)))
        },
    )?;
    let BcdValue::Cgc(value) = value else {
        unreachable!("CGC key stores a CGC")
    };
    Ok(value)
}

fn cached_product_cgc(
    cat: &Mutex<CanonicalCatalog>,
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
) -> Result<Arc<CatalogCgc>, LieError> {
    let tier = crate::cache::cache_bcd_cgc();
    if let Some(BcdValue::Cgc(hit)) = tier.get(&BcdRef::Cgc(a, b, c)) {
        return Ok(hit);
    }
    let value = tier.get_or_try_compute(
        &BcdRef::Cgc(a, b, c),
        || BcdKey::Cgc(a.clone(), b.clone(), c.clone()),
        || {
            let mut cat = cat.lock().map_err(|_| poisoned_catalog())?;
            if let Some(hit) = tier.peek(&BcdRef::Cgc(a, b, c)) {
                return Ok(hit);
            }
            CGC_SWEEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let epoch = tier.epoch();
            let channels = cat.cgc_product(a, b)?;
            let mut wanted = None;
            for ch in channels {
                if ch.s3() == c {
                    wanted = Some(BcdValue::Cgc(Arc::new(ch)));
                } else {
                    tier.insert(
                        epoch,
                        BcdKey::Cgc(a.clone(), b.clone(), ch.s3().clone()),
                        BcdValue::Cgc(Arc::new(ch)),
                    );
                }
            }
            Ok::<_, LieError>(
                wanted.expect("the sweep passed its exact support gate for this nonempty channel"),
            )
        },
    )?;
    let BcdValue::Cgc(value) = value else {
        unreachable!("CGC key stores a CGC")
    };
    Ok(value)
}

fn magnetic_index(value: usize) -> Result<u32, LieError> {
    u32::try_from(value).map_err(|_| LieError::LabelOverflow)
}

fn sparse_entries(s1: &Irrep, cgc: &CatalogCgc) -> Result<Vec<MEntry>, LieError> {
    let (rows, d3) = cgc.copy_shape();
    // d1 = dim(s1); the row index is m1 + d1·m2 (first factor fast).
    let d1 = super::super::checked_dimension(s1.dim())?;
    let mult = cgc.multiplicity();
    let mut out = Vec::new();
    for mu in 0..mult {
        let copy = cgc.copy(mu);
        for col in 0..d3 {
            for row in 0..rows {
                // ponytail: keep every exact-nonzero coefficient; no purge
                // threshold, so no arbitrary cutoff can drop a small-but-real
                // entry. The dense buffer is small for the ranks in scope.
                let v = copy[col * rows + row];
                if v != 0.0 {
                    out.push(MEntry {
                        m1: magnetic_index(row % d1)?,
                        m2: magnetic_index(row / d1)?,
                        m3: magnetic_index(col)?,
                        mu: magnetic_index(mu)?,
                        value: v,
                    });
                }
            }
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub fn f_symbol(
    cat: &Mutex<CanonicalCatalog>,
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
    d: &Irrep,
    e: &Irrep,
    f: &Irrep,
) -> Result<FBlock, LieError> {
    let cache = crate::cache::cache_bcd_f();
    cache.get_or_try_compute(
        &crate::cache::Sextuple(a, b, c, d, e, f),
        || {
            (
                a.clone(),
                b.clone(),
                c.clone(),
                d.clone(),
                e.clone(),
                f.clone(),
            )
        },
        || f_block_raw(&mut BcdFamily { cat }, a, b, c, d, e, f),
    )
}
pub fn r_symbol(
    cat: &Mutex<CanonicalCatalog>,
    a: &Irrep,
    b: &Irrep,
    c: &Irrep,
) -> Result<RBlock, LieError> {
    r_block_raw(&mut BcdFamily { cat }, a, b, c)
}
#[cfg(test)]
mod tests;

#[cfg(test)]
mod arithmetic_tests {
    use super::*;
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn sparse_index_bound() {
        assert_eq!(
            magnetic_index(u32::MAX as usize + 1),
            Err(LieError::LabelOverflow)
        );
    }
}
