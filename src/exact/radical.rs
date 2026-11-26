//! Exact building blocks of stored radicals: a positive rational radicand
//! ([`Radicand`]) and a slab of signed integers at one limb stride
//! ([`SignedInts`]), with their byte codecs. No floating point.
//!
//! Every stored value of the exact layer is assembled from these: an entry
//! `sign(S)·√(S²·ρ)` keeps `S` in a slab and `ρ` as radicands, and the
//! rounding layer reads both in place.

use malachite_base::num::basic::traits::Zero;
use malachite_base::num::logic::traits::SignificantBits;
use malachite_nz::natural::Natural;
use malachite_nz::platform::Limb;

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
    pub(crate) fn from_naturals(values: &[(bool, Natural)]) -> Option<Self> {
        let words: Vec<Vec<u64>> = values.iter().map(|(_, x)| natural_words(x)).collect();
        let len = values.len();
        let stride = words.iter().map(Vec::len).max().unwrap_or(0);
        let mut neg = vec![0u64; len.div_ceil(64)];
        let mut mag = vec![0u64; len.checked_mul(stride)?];
        for (i, ((n, _), w)) in values.iter().zip(&words).enumerate() {
            if *n && !w.is_empty() {
                neg[i / 64] |= 1 << (i % 64);
            }
            mag[i * stride..i * stride + w.len()].copy_from_slice(w);
        }
        Some(Self {
            len,
            stride,
            neg: neg.into_boxed_slice(),
            mag: mag.into_boxed_slice(),
        })
    }

    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Words per entry.
    #[cfg(test)]
    pub(crate) fn stride(&self) -> usize {
        self.stride
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
    pub(crate) fn integer(&self, i: usize) -> malachite_nz::integer::Integer {
        let m = malachite_nz::integer::Integer::from(words_natural(self.magnitude(i)));
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

#[cfg(test)]
mod tests {
    use super::*;
    use malachite_base::num::basic::traits::One;
    use malachite_nz::integer::Integer;

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
}
