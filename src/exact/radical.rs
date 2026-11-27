//! Exact building blocks of stored radicals: a positive rational radicand
//! ([`Radicand`]) and a slab of signed integers at one limb stride
//! ([`SignedInts`]), with their byte codecs. No floating point.
//!
//! Every stored value of the exact layer is assembled from these: an entry
//! `sign(S)·√(S²·ρ)` keeps `S` in a slab and `ρ` as radicands, and the
//! rounding layer reads both in place.

use malachite_base::num::basic::traits::{One, Zero};
use malachite_base::num::logic::traits::SignificantBits;
#[cfg(test)]
use malachite_nz::integer::Integer;
use malachite_nz::natural::Natural;
use malachite_nz::platform::Limb;
#[cfg(test)]
use malachite_q::Rational;

use super::limbs::{mul_into, trimmed_words};
use super::SignedSqrtRational;

/// A positive rational `num/den`, unreduced unless its producer states
/// otherwise; both parts are at least 1 and compacted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Radicand {
    num: Natural,
    den: Natural,
}

impl Radicand {
    /// `num/den`, if both parts are positive.
    pub(crate) fn new(num: Natural, den: Natural) -> Option<Self> {
        if num == Natural::ZERO || den == Natural::ZERO {
            return None;
        }
        Some(Self {
            num: compact(num),
            den: compact(den),
        })
    }

    fn one() -> Self {
        Self {
            num: Natural::ONE,
            den: Natural::ONE,
        }
    }

    /// The unreduced product `self·other`.
    pub(crate) fn mul(&self, other: &Self) -> Self {
        Self {
            num: compact(&self.num * &other.num),
            den: compact(&self.den * &other.den),
        }
    }

    #[cfg(test)]
    fn rational(&self) -> Rational {
        Rational::from_naturals(self.num.clone(), self.den.clone())
    }

    pub(crate) fn num(&self) -> &Natural {
        &self.num
    }

    pub(crate) fn den(&self) -> &Natural {
        &self.den
    }

    /// Heap and inline bytes of both parts.
    pub(crate) fn bytes(&self) -> usize {
        natural_bytes(&self.num).saturating_add(natural_bytes(&self.den))
    }

    /// Encoded bytes, or `None` when a part has more words than a `u32`
    /// counts.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn encoded_len(&self) -> Option<usize> {
        natural_encoded_len(&self.num)?.checked_add(natural_encoded_len(&self.den)?)
    }

    /// Appends `num` then `den`, each as [`encode_natural`].
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        encode_natural(&self.num, out);
        encode_natural(&self.den, out);
    }

    /// The radicand at `*at`, advancing it, if both parts are canonical and
    /// positive.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn decode(bytes: &[u8], at: &mut usize) -> Option<Self> {
        let num = decode_natural(bytes, at)?;
        let den = decode_natural(bytes, at)?;
        Self::new(num, den)
    }
}

/// Signed integers at one fixed limb stride: entry `i` is `±mag[i·stride..
/// (i+1)·stride]` (little-endian 64-bit words), negative when bit `i` of
/// `neg` is set.
///
/// Canonical form: `stride` is the word count of the widest `|entry|`, so it
/// is zero exactly when every entry is zero (the empty slab included); a sign
/// bit is set only on a nonzero entry, and the bits of `neg` past `len` are
/// zero. Equal values therefore have equal representations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SignedInts {
    len: usize,
    stride: usize,
    neg: Box<[u64]>,
    mag: Box<[u64]>,
}

impl SignedInts {
    /// The slab of `values`, `(negative, |x|)` per entry, when its words can
    /// be counted.
    #[cfg(test)]
    pub(crate) fn from_words(values: &[(bool, u64)]) -> Option<Self> {
        let len = values.len();
        let stride = usize::from(values.iter().any(|&(_, m)| m != 0));
        let mut neg = vec![0u64; len.div_ceil(64)];
        let mut mag = Vec::with_capacity(len.checked_mul(stride)?);
        for (i, &(n, m)) in values.iter().enumerate() {
            if n && m != 0 {
                neg[i / 64] |= 1 << (i % 64);
            }
            if stride == 1 {
                mag.push(m);
            }
        }
        Some(Self {
            len,
            stride,
            neg: neg.into_boxed_slice(),
            mag: mag.into_boxed_slice(),
        })
    }

    /// The slab of `values`, `(negative, |x|)` per entry, when its words can
    /// be counted.
    ///
    /// The stride is found first, so the limbs are written straight into one
    /// slab allocation with no per-entry buffer.
    pub(crate) fn from_naturals(values: &[(bool, Natural)]) -> Option<Self> {
        let len = values.len();
        let stride = values
            .iter()
            .map(|(_, x)| x.significant_bits().div_ceil(64))
            .max()
            .map_or(Some(0), |w| usize::try_from(w).ok())?;
        let mut neg = vec![0u64; len.div_ceil(64)];
        let mut mag = vec![0u64; len.checked_mul(stride)?];
        for (i, (n, x)) in values.iter().enumerate() {
            if *n && *x != Natural::ZERO {
                neg[i / 64] |= 1 << (i % 64);
            }
            write_natural(x, &mut mag[i * stride..(i + 1) * stride]);
        }
        Some(Self {
            len,
            stride,
            neg: neg.into_boxed_slice(),
            mag: mag.into_boxed_slice(),
        })
    }

    /// The slab of `len` entries at `stride` words whose sign words `neg`
    /// and magnitudes `mag` its producer wrote in canonical form (checked in
    /// debug builds): the integralization fills both in place, range by range.
    pub(crate) fn from_raw(len: usize, stride: usize, neg: Vec<u64>, mag: Vec<u64>) -> Self {
        let slab = Self {
            len,
            stride,
            neg: neg.into_boxed_slice(),
            mag: mag.into_boxed_slice(),
        };
        debug_assert!(
            slab.neg.len() == len.div_ceil(64)
                && Some(slab.mag.len()) == len.checked_mul(stride)
                && slab.is_canonical()
        );
        slab
    }

    /// The canonical slab of `len` entries written at `cap ≥` their widest
    /// words each (`mag.len() == len·cap`) with sign words `neg`: the stride
    /// becomes the widest actual magnitude, every entry moving forward in
    /// place, and the sign of a zero entry is cleared.
    pub(crate) fn compacted(len: usize, cap: usize, mut neg: Vec<u64>, mut mag: Vec<u64>) -> Self {
        let mut widest = 0;
        for i in 0..len {
            let used = trimmed_words(&mag[i * cap..(i + 1) * cap]).len();
            if used == 0 {
                neg[i / 64] &= !(1 << (i % 64));
            }
            widest = widest.max(used);
        }
        // Entry `i` moves from `i·cap` to `i·widest ≤ i·cap`: in ascending
        // order no unread word is overwritten.
        for i in 0..len {
            mag.copy_within(i * cap..i * cap + widest, i * widest);
        }
        mag.truncate(len * widest);
        Self::from_raw(len, widest, neg, mag)
    }

    /// Every entry negated.
    fn negated(&self) -> Self {
        let mut neg = self.neg.clone();
        for i in (0..self.len).filter(|&i| !self.magnitude(i).is_empty()) {
            neg[i / 64] ^= 1 << (i % 64);
        }
        Self {
            neg,
            ..self.clone()
        }
    }

    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Words per entry.
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// The magnitude words of entries `i..i + n`, at the slab's stride (high
    /// zero words included).
    pub(crate) fn words(&self, i: usize, n: usize) -> &[u64] {
        &self.mag[i * self.stride..(i + n) * self.stride]
    }

    /// The sign words: bit `i` is set for a negative entry `i`.
    pub(crate) fn sign_words(&self) -> &[u64] {
        &self.neg
    }

    pub(crate) fn is_negative(&self, i: usize) -> bool {
        (self.neg[i / 64] >> (i % 64)) & 1 == 1
    }

    /// `|entry i|` as little-endian words, without high zero words.
    pub(crate) fn magnitude(&self, i: usize) -> &[u64] {
        let w = &self.mag[i * self.stride..(i + 1) * self.stride];
        &w[..w.len() - w.iter().rev().take_while(|&&x| x == 0).count()]
    }

    /// Significant bits of `|entry i|`.
    pub(crate) fn bits(&self, i: usize) -> u64 {
        let w = self.magnitude(i);
        w.last().map_or(0, |&top| {
            64 * w.len() as u64 - u64::from(top.leading_zeros())
        })
    }

    /// The largest [`Self::bits`] over all entries.
    pub(crate) fn max_bits(&self) -> u64 {
        (0..self.len).map(|i| self.bits(i)).max().unwrap_or(0)
    }

    /// Whether every entry is nonzero.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn all_nonzero(&self) -> bool {
        (0..self.len).all(|i| !self.magnitude(i).is_empty())
    }

    /// Entry `i` as a signed big integer.
    #[cfg(test)]
    pub(crate) fn integer(&self, i: usize) -> Integer {
        let m = Integer::from(words_natural(self.magnitude(i)));
        if self.is_negative(i) {
            -m
        } else {
            m
        }
    }

    /// Heap bytes of the sign bitset and the magnitudes.
    pub(crate) fn heap_bytes(&self) -> usize {
        (self.neg.len() + self.mag.len()).saturating_mul(std::mem::size_of::<u64>())
    }

    /// Encoded bytes, or `None` when `len` or `stride` does not fit its
    /// `u32` field.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn encoded_len(&self) -> Option<usize> {
        u32::try_from(self.len).ok()?;
        u32::try_from(self.stride).ok()?;
        (self.neg.len().checked_add(self.mag.len())?)
            .checked_mul(8)?
            .checked_add(8)
    }

    /// `u32 len`, `u32 stride`, then the `⌈len/64⌉` sign words and the
    /// `len·stride` magnitude words, little-endian.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        put_u32(out, self.len);
        put_u32(out, self.stride);
        for w in self.neg.iter().chain(self.mag.iter()) {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }

    /// The slab at `*at`, advancing it, if it is canonical. Every length is
    /// checked against the remaining bytes before anything is allocated.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn decode(bytes: &[u8], at: &mut usize) -> Option<Self> {
        let len = read_u32(bytes, at)? as usize;
        let stride = read_u32(bytes, at)? as usize;
        let signs = len.div_ceil(64);
        let words = len.checked_mul(stride)?;
        let neg = read_words(bytes, at, signs)?;
        let mag = read_words(bytes, at, words)?;
        let slab = Self {
            len,
            stride,
            neg,
            mag,
        };
        slab.is_canonical().then_some(slab)
    }

    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    fn is_canonical(&self) -> bool {
        if !self.len.is_multiple_of(64)
            && self.neg.last().is_some_and(|&w| w >> (self.len % 64) != 0)
        {
            return false;
        }
        let mut widest = false;
        for i in 0..self.len {
            let w = &self.mag[i * self.stride..(i + 1) * self.stride];
            if self.is_negative(i) && w.iter().all(|&x| x == 0) {
                return false;
            }
            widest |= w.last().is_some_and(|&top| top != 0);
        }
        self.stride == 0 || widest
    }
}

/// The little-endian 64-bit words of `x`, without high zero words. Reads
/// Malachite's limbs in place, at either limb width.
pub(crate) fn natural_words(x: &Natural) -> Vec<u64> {
    let mut words = vec![0u64; x.significant_bits().div_ceil(64) as usize];
    for (i, w) in x.limbs().enumerate() {
        let bit = i * Limb::BITS as usize;
        // A Malachite limb is 64 bits wide, or 32 under its `32_bit_limbs`
        // feature, where this conversion is not the identity.
        #[allow(clippy::useless_conversion)]
        let w = u64::from(w);
        words[bit / 64] |= w << (bit % 64);
    }
    words
}

/// ORs the little-endian 64-bit words of `x` into `words`, which hold at
/// least its significant words. Reads Malachite's limbs in place, at either
/// limb width.
pub(crate) fn write_natural(x: &Natural, words: &mut [u64]) {
    for (j, w) in x.limbs().enumerate() {
        let bit = j * Limb::BITS as usize;
        // A Malachite limb is 64 bits wide, or 32 under its `32_bit_limbs`
        // feature, where this conversion is not the identity.
        #[allow(clippy::useless_conversion)]
        let w = u64::from(w);
        words[bit / 64] |= w << (bit % 64);
    }
}

/// The value of little-endian 64-bit words.
pub(crate) fn words_natural(words: &[u64]) -> Natural {
    words
        .iter()
        .rev()
        .fold(Natural::ZERO, |acc, &w| (acc << 64u32) + Natural::from(w))
}

/// `x` re-allocated at exactly its length, so [`natural_bytes`] bounds what
/// a retained value owns: Malachite keeps a vector's capacity when a result
/// shrinks.
pub(crate) fn compact(x: Natural) -> Natural {
    let mut limbs = x.to_limbs_asc();
    limbs.shrink_to_fit();
    Natural::from_owned_limbs_asc(limbs)
}

/// The inline slot and the limbs of a compacted `x` (inline limbs charged
/// too, conservatively).
pub(crate) fn natural_bytes(x: &Natural) -> usize {
    let limbs = usize::try_from(x.limb_count()).unwrap_or(usize::MAX);
    std::mem::size_of::<Natural>().saturating_add(limbs.saturating_mul(std::mem::size_of::<Limb>()))
}

/// Encoded bytes of `x`: a `u32` word count and the words.
#[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
pub(crate) fn natural_encoded_len(x: &Natural) -> Option<usize> {
    let words = x.significant_bits().div_ceil(64);
    u32::try_from(words).ok()?;
    usize::try_from(words).ok()?.checked_mul(8)?.checked_add(4)
}

/// Appends `u32 count`, then the `count` little-endian words of `x` (none
/// for zero).
#[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
pub(crate) fn encode_natural(x: &Natural, out: &mut Vec<u8>) {
    let words = natural_words(x);
    put_u32(out, words.len());
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
}

/// The natural at `*at`, advancing it, if its top word is nonzero.
#[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
pub(crate) fn decode_natural(bytes: &[u8], at: &mut usize) -> Option<Natural> {
    let count = read_u32(bytes, at)? as usize;
    let words = read_words(bytes, at, count)?;
    if words.last() == Some(&0) {
        return None;
    }
    Some(compact(words_natural(&words)))
}

/// Appends `x` as a little-endian `u32`; the caller checked that it fits.
#[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
pub(crate) fn put_u32(out: &mut Vec<u8>, x: usize) {
    out.extend_from_slice(&(x as u32).to_le_bytes());
}

/// The little-endian `u32` at `*at`, advancing it.
#[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
pub(crate) fn read_u32(bytes: &[u8], at: &mut usize) -> Option<u32> {
    let b = bytes.get(*at..at.checked_add(4)?)?;
    *at += 4;
    Some(u32::from_le_bytes(b.try_into().ok()?))
}

/// `count` little-endian `u64` at `*at`, advancing it; the length is checked
/// against the remaining bytes before the words are allocated.
#[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
fn read_words(bytes: &[u8], at: &mut usize, count: usize) -> Option<Box<[u64]>> {
    let end = at.checked_add(count.checked_mul(8)?)?;
    let b = bytes.get(*at..end)?;
    *at = end;
    Some(
        b.as_chunks::<8>()
            .0
            .iter()
            .map(|w| u64::from_le_bytes(*w))
            .collect(),
    )
}

/// A block of exact radicals, row-major over `dims`: entry `i` at axis
/// indices `(i_0, …, i_{R−1})` is
/// `sign(S_i)·√(S_i²·common·Π_{k stored} axis[k][i_k])`.
///
/// Axis `k` is stored when `dims[k] > 1`, with one radicand per index; a
/// singleton (or empty) axis stores nothing, its radicand folded into
/// `common`. `pub` so the sealed product hook can name it; the module is
/// crate-private.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadicalBlock<const R: usize> {
    dims: [usize; R],
    nums: SignedInts,
    axis: [Box<[Radicand]>; R],
    common: Radicand,
}

impl<const R: usize> RadicalBlock<R> {
    /// The block of these parts, if they fit: `Π dims` integers, and `axis[k]`
    /// of `dims[k]` radicands when `dims[k] > 1`, else empty.
    pub(crate) fn new(
        dims: [usize; R],
        nums: SignedInts,
        axis: [Box<[Radicand]>; R],
        common: Radicand,
    ) -> Option<Self> {
        let len = dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d))?;
        let fits = dims
            .iter()
            .zip(&axis)
            .all(|(&d, a)| a.len() == if d > 1 { d } else { 0 });
        (nums.len() == len && fits).then_some(Self {
            dims,
            nums,
            axis,
            common,
        })
    }

    /// The block of `dims` whose leg `k` holds one radicand per index of
    /// axis `k`; the radicand of a singleton axis is folded into `common`.
    pub(crate) fn from_legs(
        dims: [usize; R],
        nums: SignedInts,
        legs: [Vec<Radicand>; R],
        mut common: Radicand,
    ) -> Option<Self> {
        let mut axis: [Box<[Radicand]>; R] = std::array::from_fn(|_| Box::default());
        for (k, leg) in legs.into_iter().enumerate() {
            if dims[k] > 1 {
                axis[k] = leg.into_boxed_slice();
            } else if dims[k] == 1 {
                match leg.as_slice() {
                    [r] => common = common.mul(r),
                    _ => return None,
                }
            }
        }
        Self::new(dims, nums, axis, common)
    }

    /// The integers `S`, row-major.
    pub(crate) fn nums(&self) -> &SignedInts {
        &self.nums
    }

    /// The radicands of axis `k`: empty for a singleton axis.
    pub(crate) fn axis(&self, k: usize) -> &[Radicand] {
        &self.axis[k]
    }

    pub(crate) fn common(&self) -> &Radicand {
        &self.common
    }

    /// The zero block of `dims`, with unit radicands.
    pub(crate) fn zeros(dims: [usize; R]) -> Option<Self> {
        let len = dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d))?;
        let nums = SignedInts::from_raw(len, 0, vec![0; len.div_ceil(64)], Vec::new());
        let axis = dims.map(|d| {
            if d > 1 {
                vec![Radicand::one(); d].into_boxed_slice()
            } else {
                Box::default()
            }
        });
        Self::new(dims, nums, axis, Radicand::one())
    }

    /// The single-entry block of `x`.
    pub(crate) fn from_signed_sqrt(x: &SignedSqrtRational) -> Self {
        let r = x.radicand();
        let (nums, common) =
            match Radicand::new(r.numerator_ref().clone(), r.denominator_ref().clone()) {
                Some(common) => (
                    SignedInts::from_raw(1, 1, vec![u64::from(x.sign() < 0)], vec![1]),
                    common,
                ),
                None => (
                    SignedInts::from_raw(1, 0, vec![0], Vec::new()),
                    Radicand::one(),
                ),
            };
        Self {
            dims: [1; R],
            nums,
            axis: std::array::from_fn(|_| Box::default()),
            common,
        }
    }

    pub(crate) fn dims(&self) -> [usize; R] {
        self.dims
    }

    /// The Kronecker product `self ⊗ other`: each axis index is
    /// `l·other.dims[k] + r`, entry `S_l·S_r`, radicands multiplied
    /// (unreduced). `None` when a dimension overflows.
    pub(crate) fn kron(&self, other: &Self) -> Option<Self> {
        let mut dims = [0; R];
        for (d, (&a, &b)) in dims.iter_mut().zip(self.dims.iter().zip(&other.dims)) {
            *d = a.checked_mul(b)?;
        }
        let len = dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d))?;
        let cap = self.nums.stride() + other.nums.stride();
        let mut neg = vec![0u64; len.div_ceil(64)];
        let mut mag = vec![0u64; len.checked_mul(cap)?];
        let (mut ls, mut rs) = ([1usize; R], [1usize; R]);
        for k in (0..R.saturating_sub(1)).rev() {
            ls[k] = ls[k + 1] * self.dims[k + 1];
            rs[k] = rs[k + 1] * other.dims[k + 1];
        }
        // The axis digits of output `i`, counted up in row-major order.
        let (mut ld, mut rd) = ([0usize; R], [0usize; R]);
        for i in 0..len {
            let l: usize = ld.iter().zip(&ls).map(|(d, s)| d * s).sum();
            let r: usize = rd.iter().zip(&rs).map(|(d, s)| d * s).sum();
            if self.nums.is_negative(l) != other.nums.is_negative(r) {
                neg[i / 64] |= 1 << (i % 64);
            }
            mul_into(
                self.nums.magnitude(l),
                other.nums.magnitude(r),
                &mut mag[i * cap..(i + 1) * cap],
            );
            for k in (0..R).rev() {
                rd[k] += 1;
                if rd[k] < other.dims[k] {
                    break;
                }
                rd[k] = 0;
                ld[k] += 1;
                if ld[k] < self.dims[k] {
                    break;
                }
                ld[k] = 0;
            }
        }
        let axis = std::array::from_fn(|k| {
            let (a, b) = (&self.axis[k], &other.axis[k]);
            match (dims[k] > 1, a.is_empty(), b.is_empty()) {
                (false, ..) => Box::default(),
                (true, false, false) => a.iter().flat_map(|x| b.iter().map(|y| x.mul(y))).collect(),
                (true, false, true) => a.clone(),
                (true, true, _) => b.clone(),
            }
        });
        Self::new(
            dims,
            SignedInts::compacted(len, cap, neg, mag),
            axis,
            self.common.mul(&other.common),
        )
    }

    /// Every entry times `±√r`, negated when `negate`.
    pub(crate) fn scaled(&self, negate: bool, r: &Radicand) -> Self {
        Self {
            dims: self.dims,
            nums: if negate {
                self.nums.negated()
            } else {
                self.nums.clone()
            },
            axis: self.axis.clone(),
            common: self.common.mul(r),
        }
    }

    /// The matrix over axes `rows < cols`, when every other axis is a
    /// singleton: the same entries in the same order.
    pub(crate) fn matrix(self, rows: usize, cols: usize) -> Option<RadicalBlock<2>> {
        let singleton = (0..R).all(|k| k == rows || k == cols || self.dims[k] == 1);
        if rows >= cols || cols >= R || !singleton {
            return None;
        }
        let mut axis = self.axis;
        RadicalBlock::new(
            [self.dims[rows], self.dims[cols]],
            self.nums,
            [
                std::mem::take(&mut axis[rows]),
                std::mem::take(&mut axis[cols]),
            ],
            self.common,
        )
    }

    /// Entry `i` (row-major) exactly: the oracle tests' reader.
    #[cfg(test)]
    pub(crate) fn entry(&self, i: usize) -> SignedSqrtRational {
        let mut rad = self.common.rational();
        let mut rest = i;
        for k in (0..R).rev() {
            let d = self.dims[k];
            if d > 1 {
                rad *= self.axis[k][rest % d].rational();
                rest /= d;
            }
        }
        SignedSqrtRational::from_prefactor_radical(Rational::from(self.nums.integer(i)), rad)
    }

    /// Bytes owned: the struct, the integers and every radicand.
    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.nums.heap_bytes())
            .saturating_add(
                self.axis
                    .iter()
                    .flat_map(|a| a.iter())
                    .map(Radicand::bytes)
                    .sum(),
            )
            .saturating_add(self.common.bytes())
    }

    /// Encoded bytes, or `None` when a count does not fit its `u32` field.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn encoded_len(&self) -> Option<usize> {
        let mut n = R.checked_mul(4)?.checked_add(self.nums.encoded_len()?)?;
        for d in self.dims {
            u32::try_from(d).ok()?;
        }
        for r in self.axis.iter().flat_map(|a| a.iter()) {
            n = n.checked_add(r.encoded_len()?)?;
        }
        n.checked_add(self.common.encoded_len()?)
    }

    /// `R × u32 dims`, the slab `S`, the radicands of every stored axis in
    /// axis order, then `common`.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        for d in self.dims {
            put_u32(out, d);
        }
        self.nums.encode(out);
        for r in self.axis.iter().flat_map(|a| a.iter()) {
            r.encode(out);
        }
        self.common.encode(out);
    }

    /// The block of `payload`, checked before it is used: the slab canonical
    /// with `Π dims` entries, every radicand positive, and no trailing bytes.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn decode(payload: &[u8]) -> Option<Self> {
        let at = &mut 0;
        let mut dims = [0; R];
        for d in &mut dims {
            *d = read_u32(payload, at)? as usize;
        }
        let nums = SignedInts::decode(payload, at)?;
        let mut axis: [Box<[Radicand]>; R] = std::array::from_fn(|_| Box::default());
        for (a, &d) in axis.iter_mut().zip(&dims).filter(|(_, d)| **d > 1) {
            // Grown per decoded radicand, so a short payload allocates no `d` slots.
            let mut v = Vec::new();
            for _ in 0..d {
                v.push(Radicand::decode(payload, at)?);
            }
            *a = v.into_boxed_slice();
        }
        let common = Radicand::decode(payload, at)?;
        if *at != payload.len() {
            return None;
        }
        Self::new(dims, nums, axis, common)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(s: &SignedInts) -> Option<SignedInts> {
        let mut out = Vec::new();
        s.encode(&mut out);
        assert_eq!(out.len(), s.encoded_len().unwrap());
        let mut at = 0;
        let back = SignedInts::decode(&out, &mut at)?;
        (at == out.len()).then_some(back)
    }

    /// The canonical stride is the widest magnitude's word count: zero for
    /// the empty and the all-zero slab, one for words, more for big values,
    /// and the word and big-integer constructors agree on equal values.
    #[test]
    fn stride_is_the_widest_magnitude() {
        let empty = SignedInts::from_words(&[]).unwrap();
        assert_eq!((empty.len(), empty.stride()), (0, 0));
        let zeros = SignedInts::from_words(&[(true, 0), (false, 0)]).unwrap();
        assert_eq!(zeros.stride(), 0);
        assert!(!zeros.is_negative(0), "a zero carries no sign");
        let words = SignedInts::from_words(&[(true, 3), (false, u64::MAX)]).unwrap();
        assert_eq!(words.stride(), 1);
        let naturals = SignedInts::from_naturals(&[
            (true, Natural::from(3u32)),
            (false, Natural::from(u64::MAX)),
        ])
        .unwrap();
        assert_eq!(naturals, words);
        let wide = Natural::ONE << 130u32;
        let big =
            SignedInts::from_naturals(&[(true, Natural::ONE), (false, wide.clone())]).unwrap();
        assert_eq!(big.stride(), 3);
        assert_eq!(big.integer(0), Integer::from(-1));
        assert_eq!(big.integer(1), Integer::from(wide));
        assert_eq!((big.bits(0), big.bits(1), big.max_bits()), (1, 131, 131));
        assert_eq!(big.magnitude(0), &[1]);
    }

    /// Every slab round-trips through its codec, and every noncanonical or
    /// truncated encoding is rejected: a stride wider than the widest entry,
    /// a negative zero, a sign bit past `len`, a nonzero stride over an
    /// all-zero slab, and every proper prefix.
    #[test]
    fn slab_codec_accepts_exactly_the_canonical_form() {
        let wide = Natural::ONE << 70u32;
        let slabs = [
            SignedInts::from_words(&[]).unwrap(),
            SignedInts::from_words(&[(false, 0), (false, 0)]).unwrap(),
            SignedInts::from_words(&(0..70).map(|i| (i % 3 == 0, i + 1)).collect::<Vec<_>>())
                .unwrap(),
            SignedInts::from_naturals(&[(true, wide), (false, Natural::from(5u32))]).unwrap(),
        ];
        for s in &slabs {
            assert_eq!(round_trip(s).as_ref(), Some(s));
        }
        let raw = |len: u32, stride: u32, words: &[u64]| {
            let mut out = Vec::new();
            out.extend_from_slice(&len.to_le_bytes());
            out.extend_from_slice(&stride.to_le_bytes());
            for w in words {
                out.extend_from_slice(&w.to_le_bytes());
            }
            out
        };
        let decodes = |b: &[u8]| SignedInts::decode(b, &mut 0).is_some();
        assert!(decodes(&raw(2, 1, &[0b01, 7, 9])));
        assert!(!decodes(&raw(2, 2, &[0, 7, 0, 9, 0])), "stride too wide");
        assert!(!decodes(&raw(2, 1, &[0b10, 7, 0])), "negative zero");
        assert!(!decodes(&raw(2, 1, &[0b100, 7, 9])), "sign past len");
        assert!(!decodes(&raw(2, 1, &[0, 0, 0])), "stride over zeros");
        assert!(
            !decodes(&raw(u32::MAX, u32::MAX, &[])),
            "lengths beyond the bytes"
        );
        let good = raw(2, 1, &[0b01, 7, 9]);
        for len in 0..good.len() {
            assert!(!decodes(&good[..len]), "prefix {len}");
        }
    }

    /// The big-integer constructor lays out the canonical slab written out
    /// here: the empty slab, an all-word slab, and mixed widths with a zero
    /// entry, whose stride is the widest entry's word count.
    #[test]
    fn natural_slab_layout() {
        let empty = SignedInts::from_naturals(&[]).unwrap();
        assert_eq!(
            (empty.len, empty.stride, &*empty.neg, &*empty.mag),
            (0, 0, &[][..], &[][..])
        );
        let words = SignedInts::from_naturals(&[
            (false, Natural::from(3u32)),
            (true, Natural::from(u64::MAX)),
        ])
        .unwrap();
        assert_eq!(
            (words.len, words.stride, &*words.neg, &*words.mag),
            (2, 1, &[0b10u64][..], &[3, u64::MAX][..])
        );
        let mixed = SignedInts::from_naturals(&[
            (true, (Natural::ONE << 64u32) + Natural::from(5u32)),
            (true, Natural::ZERO),
            (false, Natural::from(7u32)),
        ])
        .unwrap();
        assert_eq!(
            (mixed.len, mixed.stride, &*mixed.neg, &*mixed.mag),
            (3, 2, &[0b001u64][..], &[5, 1, 0, 0, 7, 0][..])
        );
    }

    /// A radicand needs both parts positive and canonical words.
    #[test]
    fn radicand_codec_rejects_zero_and_padded_parts() {
        assert!(Radicand::new(Natural::ZERO, Natural::ONE).is_none());
        assert!(Radicand::new(Natural::ONE, Natural::ZERO).is_none());
        let r = Radicand::new(Natural::from(6u32), Natural::ONE << 64u32).unwrap();
        let mut out = Vec::new();
        r.encode(&mut out);
        assert_eq!(out.len(), r.encoded_len().unwrap());
        let mut at = 0;
        assert_eq!(Radicand::decode(&out, &mut at), Some(r));
        assert_eq!(at, out.len());
        let part = |words: &[u64]| {
            let mut out = (words.len() as u32).to_le_bytes().to_vec();
            for w in words {
                out.extend_from_slice(&w.to_le_bytes());
            }
            out
        };
        let pair = |n: &[u64], d: &[u64]| [part(n), part(d)].concat();
        assert!(Radicand::decode(&pair(&[3], &[2]), &mut 0).is_some());
        assert!(Radicand::decode(&pair(&[], &[2]), &mut 0).is_none(), "zero");
        assert!(
            Radicand::decode(&pair(&[3, 0], &[2]), &mut 0).is_none(),
            "padded"
        );
        assert!(
            Radicand::decode(&pair(&[3], &[2])[..9], &mut 0).is_none(),
            "truncated"
        );
    }

    fn rad(num: u64, den: u64) -> Radicand {
        Radicand::new(Natural::from(num), Natural::from(den)).unwrap()
    }

    fn q(num: i64, den: u64) -> Rational {
        Rational::from_integers(Integer::from(num), Integer::from(den))
    }

    fn signed_squares<const R: usize>(b: &RadicalBlock<R>) -> Vec<Rational> {
        let len = b.dims().iter().product();
        (0..len).map(|i| b.entry(i).signed_square()).collect()
    }

    /// `[2, 1]` with `S = [1, −2]`, axis 0 `[2, 3]`, common `1/5`.
    fn column() -> RadicalBlock<2> {
        let nums = SignedInts::from_words(&[(false, 1), (true, 2)]).unwrap();
        RadicalBlock::new(
            [2, 1],
            nums,
            [vec![rad(2, 1), rad(3, 1)].into(), Box::default()],
            rad(1, 5),
        )
        .unwrap()
    }

    /// `[1, 2]` with `S = [3, −1]`, axis 1 `[1/7, 5]`, common `2`.
    fn row() -> RadicalBlock<2> {
        let nums = SignedInts::from_words(&[(false, 3), (true, 1)]).unwrap();
        RadicalBlock::new(
            [1, 2],
            nums,
            [Box::default(), vec![rad(1, 7), rad(5, 1)].into()],
            rad(2, 1),
        )
        .unwrap()
    }

    /// An entry is `sign(S)·√(S²·common·Π axis)`, and the Kronecker product
    /// takes axis index `l·dims_r + r` with `S_l·S_r` and the radicands
    /// multiplied: by hand, `(0,0)`: `9·2·(1/7)·(2/5) = 36/35`; `(0,1)`:
    /// `−(1·2·5·2/5) = −4`; `(1,0)`: `−(36·3·(1/7)·(2/5)) = −216/35`;
    /// `(1,1)`: `4·3·5·2/5 = 24`.
    #[test]
    fn kron_multiplies_entries_and_radicands() {
        assert_eq!(signed_squares(&column()), [q(2, 5), q(-12, 5)]);
        assert_eq!(signed_squares(&row()), [q(18, 7), q(-10, 1)]);
        let k = column().kron(&row()).unwrap();
        assert_eq!(k.dims(), [2, 2]);
        assert_eq!(
            signed_squares(&k),
            [q(36, 35), q(-4, 1), q(-216, 35), q(24, 1)]
        );
        // A scalar factor is the identity of the shape.
        let one = RadicalBlock::<2>::from_signed_sqrt(&SignedSqrtRational::from_prefactor_radical(
            Rational::ONE,
            Rational::ONE,
        ));
        assert_eq!(signed_squares(&one.kron(&k).unwrap()), signed_squares(&k));
        // Word products carry: (2^64 − 1)² per factor pair.
        let m = SignedInts::from_words(&[(true, u64::MAX)]).unwrap();
        let big = RadicalBlock::<2>::new([1, 1], m, Default::default(), rad(1, 1)).unwrap();
        let sq = big.kron(&big).unwrap();
        let x = Rational::from(u64::MAX);
        assert_eq!(signed_squares(&sq), [&x * &x * &x * &x]);
        assert_eq!(sq.nums.stride(), 2);
    }

    /// Zero entries stay canonical through a product: a product with a zero
    /// block equals the zero block of the product shape.
    #[test]
    fn kron_with_zeros_is_canonical() {
        let z = RadicalBlock::<2>::zeros([1, 2]).unwrap();
        let k = column().kron(&z).unwrap();
        assert_eq!((k.nums.stride(), k.nums.sign_words()), (0, &[0u64][..]));
        assert!(signed_squares(&k).iter().all(|x| *x == 0));
        let zero = RadicalBlock::<2>::from_signed_sqrt(&SignedSqrtRational::zero());
        assert_eq!(signed_squares(&zero), [Rational::ZERO]);
    }

    /// `scaled` multiplies every entry by `±√r`; `matrix` keeps the entries
    /// of the two named axes when every other axis is a singleton, and
    /// refuses otherwise.
    #[test]
    fn scaled_and_matrix_keep_entries() {
        let nums = SignedInts::from_words(&[(false, 1), (true, 2)]).unwrap();
        let b = RadicalBlock::<4>::new(
            [1, 1, 1, 2],
            nums,
            [
                Box::default(),
                Box::default(),
                Box::default(),
                vec![rad(1, 3), rad(1, 1)].into(),
            ],
            rad(1, 1),
        )
        .unwrap();
        let s = b.scaled(true, &rad(3, 2));
        assert_eq!(signed_squares(&s), [q(-1, 2), q(6, 1)]);
        let m = s.clone().matrix(2, 3).unwrap();
        assert_eq!(m.dims(), [1, 2]);
        assert_eq!(signed_squares(&m), [q(-1, 2), q(6, 1)]);
        assert!(s.clone().matrix(3, 2).is_none(), "rows after cols");
        assert!(
            s.clone().matrix(0, 1).is_none(),
            "axis 3 is not a singleton"
        );
        assert!(s.matrix(2, 4).is_none(), "no axis 4");
    }

    /// Parts that disagree with the shape are refused.
    #[test]
    fn new_checks_the_shape() {
        let two = || SignedInts::from_words(&[(false, 1), (false, 1)]).unwrap();
        let ax = || -> Box<[Radicand]> { vec![rad(1, 1), rad(1, 1)].into() };
        assert!(RadicalBlock::<2>::new([2, 1], two(), [ax(), Box::default()], rad(1, 1)).is_some());
        assert!(
            RadicalBlock::<2>::new([2, 1], two(), [Box::default(), Box::default()], rad(1, 1))
                .is_none()
        );
        assert!(
            RadicalBlock::<2>::new([2, 1], two(), [ax(), vec![rad(1, 1)].into()], rad(1, 1))
                .is_none()
        );
        assert!(RadicalBlock::<2>::new([1, 1], two(), Default::default(), rad(1, 1)).is_none());
    }

    /// A block decodes to itself; a truncated payload, a trailing byte, a
    /// zero radicand and a slab of the wrong length are refused.
    #[test]
    fn radical_block_round_trips_and_rejects_malformed() {
        let k = column().kron(&row()).unwrap();
        let mut out = Vec::new();
        k.encode(&mut out);
        assert_eq!(out.len(), k.encoded_len().unwrap());
        assert_eq!(RadicalBlock::<2>::decode(&out), Some(k.clone()));
        assert!(
            RadicalBlock::<2>::decode(&out[..out.len() - 1]).is_none(),
            "truncated"
        );
        let mut long = out.clone();
        long.push(0);
        assert!(RadicalBlock::<2>::decode(&long).is_none(), "trailing byte");
        // The common radicand is last: `num` then `den`, as `u32 words` and
        // the words; a zero denominator (no words) is refused.
        let mut zero = Vec::new();
        k.encode(&mut zero);
        let tail = k.common.encoded_len().unwrap();
        zero.truncate(zero.len() - tail);
        encode_natural(k.common.num(), &mut zero);
        encode_natural(&Natural::ZERO, &mut zero);
        assert!(RadicalBlock::<2>::decode(&zero).is_none(), "zero radicand");
        // Dims of three rows over a slab of four entries.
        let mut wrong = out;
        wrong[..4].copy_from_slice(&3u32.to_le_bytes());
        assert!(
            RadicalBlock::<2>::decode(&wrong).is_none(),
            "dims disagree with the payload"
        );
    }
}
