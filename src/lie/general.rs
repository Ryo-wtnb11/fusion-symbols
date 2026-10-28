//! The exceptional Lie engine (`docs/gauge_exceptional.md`): generic
//! Cartan-matrix machinery and its binding to `LieGroup`. Crate-private.

pub(crate) mod cgc;
pub(crate) mod module;
pub(crate) mod roots;

use std::sync::{Arc, Mutex, MutexGuard};

use num_bigint::BigInt;

use self::cgc::GeneralFamily;
use self::roots::CartanData;
use crate::category::{FBlock, RBlock};
use crate::frcore::{f_block_raw, r_block_raw, Family};
use crate::lie::{checked_dimension, checked_size, weyl_dim_f64, LieError, RootSystem};

/// The exceptional root systems whose coefficients this engine serves: those
/// with `−w₀ = 1`, where every irrep is self-dual and `κ_a` is the §6.3 closed
/// form. E6 is excluded until its gauge-derived complex-irrep phase is served.
pub(crate) fn serves(root: RootSystem) -> bool {
    matches!(
        root,
        RootSystem::G2 | RootSystem::F4 | RootSystem::E7 | RootSystem::E8
    )
}

/// `docs/gauge_exceptional.md` §8. Bump the trailing `epoch=N` (and the
/// literal in `tests/exceptional_fingerprint.rs`) only when that document is
/// corrected under the four-step rule of `docs/gauge.md`, "Status".
pub(crate) fn fingerprint() -> &'static [u8] {
    b"fusion-symbols:lie-exceptional:ref=gauge-exceptional-spec:input=bourbaki-cartan:order=depth-rootlex-asc:basis=greedy-rank-gs-own-positive:kron=a-fast:mult=rcef-pivot-gs:descent=intertwiner-own-basis:sign=own-candidate-and-pivot-positive:round=once-correct:fs=gauge-derived:fr=frcore:twist=one:scalar=f64:epoch=1"
}

/// Memo entries (modules + products + channels) a group retains before the
/// next query starts from an empty memo. One query adds at most four
/// channels, four products and six modules beyond this bound. A pentagon
/// through the G2 vertex `27 ⊗ 27 → 27` fits; at 256 it ran twice as long.
const MEMO_ENTRIES: usize = 512;

/// Estimated retained memo bytes (see [`GeneralFamily::memo_bytes`]) past
/// which the next query starts from an empty memo. Every exceptional group value and
/// composite factor owns its own memo, so this is per factor. 256 MiB is the
/// default per-group B/C/D catalog budget, and it holds the whole working set
/// of a pentagon through the G2 `N = 2` vertex `27 ⊗ 27 → 27` (about 150 MiB
/// estimated); at 64 MiB that pentagon reset repeatedly and ran 3.5× longer.
/// One entry for an irrep of a few hundred dimensions can be tens of MiB, and
/// one query may exceed the cap transiently: it is enforced between queries.
const MEMO_BYTES: usize = 256 << 20;

/// One exceptional simple factor: its root data and the memo of exact
/// modules, products and channels, owned by the group value and bounded by
/// [`MEMO_ENTRIES`] and [`MEMO_BYTES`]. Values are exact, so the memo state
/// never changes a bit. Fusion and N use only the root data and never take
/// the memo lock.
pub(crate) struct Engine {
    root: RootSystem,
    data: CartanData,
    family: Mutex<GeneralFamily>,
}

impl Engine {
    pub(crate) fn new(root: RootSystem) -> Self {
        Self {
            root,
            data: CartanData::new(root),
            family: Mutex::new(GeneralFamily::new(root)),
        }
    }

    // ponytail: wholesale reset at the bound; an LRU only if F-heavy loops thrash it.
    fn family(&self) -> MutexGuard<'_, GeneralFamily> {
        // A panic mid-query can leave a memo without its partner entries;
        // starting over is always valid.
        let mut fam = self.family.lock().unwrap_or_else(|poisoned| {
            self.family.clear_poison();
            let mut fam = poisoned.into_inner();
            *fam = GeneralFamily::new(self.root);
            fam
        });
        if fam.memo_len() >= MEMO_ENTRIES || fam.memo_bytes() >= MEMO_BYTES {
            *fam = GeneralFamily::new(self.root);
        }
        fam
    }

    /// `(entries, estimated bytes)` currently retained.
    #[cfg(test)]
    pub(crate) fn memo_stats(&self) -> (usize, usize) {
        let fam = self.family.lock().unwrap();
        (fam.memo_len(), fam.memo_bytes())
    }

    /// Drop every memoized module, product and channel.
    pub(crate) fn reset(&self) {
        // Poison recovery already starts over; only the healthy case needs it.
        *self.family() = GeneralFamily::new(self.root);
    }

    /// A validated label: nonnegative, of this rank, with an exact dimension.
    pub(crate) fn irrep(&self, dynkin: &[i64]) -> Result<Irrep, LieError> {
        let dim = self.data.weyl_dimension(dynkin)?;
        let dual = self.data.dual(dynkin)?;
        let sign = self.data.two_rho_vee_sign(dynkin)?;
        let dim_f64 = weyl_dim_f64(u128::try_from(&dim).ok(), || dim.clone());
        Ok(Irrep(Arc::new(Data {
            dynkin: dynkin.to_vec(),
            dual,
            dim,
            dim_f64,
            sign,
        })))
    }

    /// `a ⊗ b` in ascending Dynkin-label order, with multiplicities.
    pub(crate) fn product(&self, a: &Irrep, b: &Irrep) -> Result<Arc<[(Irrep, u32)]>, LieError> {
        self.data
            .tensor_product(&a.0.dynkin, &b.0.dynkin)?
            .into_iter()
            .map(|(c, n)| Ok((self.irrep(&c)?, n)))
            .collect()
    }

    pub(crate) fn nsymbol(&self, a: &Irrep, b: &Irrep, c: &Irrep) -> Result<u32, LieError> {
        Ok(self
            .data
            .tensor_product(&a.0.dynkin, &b.0.dynkin)?
            .get(&c.0.dynkin)
            .copied()
            .unwrap_or(0))
    }

    /// `κ_a` (§6.3): the stored closed form for a self-dual irrep, otherwise
    /// the gauge-derived `σ_a σ_ā`.
    pub(crate) fn frobenius_schur_phase(&self, a: &Irrep) -> Result<i8, LieError> {
        if a.is_self_dual() {
            return Ok(a.0.sign);
        }
        self.family().frobenius_schur_phase(&a.0.dynkin)
    }

    pub(crate) fn f_symbol(
        &self,
        [a, b, c, d, e, f]: [&Irrep; 6],
    ) -> Result<FBlock<f64>, LieError> {
        let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|x| &x.0.dynkin);
        f_block_raw(&mut *self.family(), a, b, c, d, e, f)
    }

    pub(crate) fn r_symbol(
        &self,
        a: &Irrep,
        b: &Irrep,
        c: &Irrep,
    ) -> Result<RBlock<f64>, LieError> {
        r_block_raw(&mut *self.family(), &a.0.dynkin, &b.0.dynkin, &c.0.dynkin)
    }

    /// The CGC as dims `[d_a, d_b, d_c, N]` and a dense row-major buffer
    /// (empty for an absent channel).
    pub(crate) fn cgc(
        &self,
        a: &Irrep,
        b: &Irrep,
        c: &Irrep,
    ) -> Result<([usize; 4], Vec<f64>), LieError> {
        let mut fam = self.family();
        let [da, db, dc] = [a, b, c].map(|x| checked_dimension(&x.0.dim));
        let dims = [
            da?,
            db?,
            dc?,
            fam.mult(&a.0.dynkin, &b.0.dynkin, &c.0.dynkin)?,
        ];
        if dims[3] == 0 {
            return Ok((dims, Vec::new()));
        }
        let mut data = vec![0.0; checked_size(dims)?];
        for e in fam.cgc_entries(&a.0.dynkin, &b.0.dynkin, &c.0.dynkin)? {
            let [m1, m2, m3, mu] = [e.m1, e.m2, e.m3, e.mu].map(|i| i as usize);
            data[((m1 * dims[1] + m2) * dims[2] + m3) * dims[3] + mu] = e.value;
        }
        Ok((dims, data))
    }
}

/// An exceptional irrep label with its exact dimension, dual label and the
/// closed-form sign `(−1)^{⟨λ, 2ρ∨⟩}`. Clones share one allocation; identity
/// is the Dynkin label (the group is compared by `LieIrrep`).
#[derive(Clone, Debug)]
pub(crate) struct Irrep(Arc<Data>);

#[derive(Debug)]
struct Data {
    dynkin: Vec<i64>,
    dual: Vec<i64>,
    dim: BigInt,
    dim_f64: f64,
    /// `⟨−w₀λ, 2ρ∨⟩ = ⟨λ, 2ρ∨⟩`, so the dual carries the same sign.
    sign: i8,
}

impl Irrep {
    pub(crate) fn dynkin(&self) -> &[i64] {
        &self.0.dynkin
    }

    pub(crate) fn dim(&self) -> &BigInt {
        &self.0.dim
    }

    pub(crate) fn dim_f64(&self) -> f64 {
        self.0.dim_f64
    }

    pub(crate) fn is_self_dual(&self) -> bool {
        self.0.dynkin == self.0.dual
    }

    pub(crate) fn dual(&self) -> Self {
        if self.is_self_dual() {
            return self.clone();
        }
        Self(Arc::new(Data {
            dynkin: self.0.dual.clone(),
            dual: self.0.dynkin.clone(),
            dim: self.0.dim.clone(),
            dim_f64: self.0.dim_f64,
            sign: self.0.sign,
        }))
    }

    /// Every weight with its multiplicity, in Dynkin coordinates.
    pub(crate) fn weight_multiplicities(
        &self,
        root: RootSystem,
    ) -> Result<std::collections::BTreeMap<Vec<i64>, u64>, LieError> {
        CartanData::new(root).weight_multiplicities(&self.0.dynkin)
    }
}

impl PartialEq for Irrep {
    fn eq(&self, other: &Self) -> bool {
        self.0.dynkin == other.0.dynkin
    }
}
impl Eq for Irrep {}
impl PartialOrd for Irrep {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Irrep {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.dynkin.cmp(&other.0.dynkin)
    }
}
impl std::hash::Hash for Irrep {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.dynkin.hash(state);
    }
}
