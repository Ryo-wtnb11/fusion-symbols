//! Fixed-width nonnegative integers on the stack: `Limbs<N>` holds a value
//! below `2^(64·N)` in little-endian 64-bit limbs, with checked products,
//! shifts and comparisons that return `None` instead of overflowing. Pure
//! integer arithmetic, no floating point; the rounding layer's stack
//! conversion builds on it.

use malachite_base::num::basic::traits::Zero;
use malachite_base::num::logic::traits::SignificantBits;
use malachite_nz::natural::Natural;
use malachite_nz::platform::Limb;

/// A nonnegative integer below `2^(64·N)` in little-endian 64-bit limbs.
/// `len` counts the limbs up to the highest nonzero one; the limbs above it
/// are zero, so the representation is unique.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Limbs<const N: usize> {
    limbs: [u64; N],
    len: usize,
}

impl<const N: usize> Limbs<N> {
    pub(crate) const ONE: Self = Self::from_word(1);

    const fn from_word(w: u64) -> Self {
        let mut limbs = [0; N];
        limbs[0] = w;
        Self {
            limbs,
            len: (w != 0) as usize,
        }
    }

    /// The value of `limbs`, of which those from `len` on are zero.
    pub(crate) fn trimmed(limbs: [u64; N], mut len: usize) -> Self {
        while len > 0 && limbs[len - 1] == 0 {
            len -= 1;
        }
        Self { limbs, len }
    }

    /// `x` when it is below `2^(64·N)`; reads Malachite's limbs in place.
    pub(crate) fn from_natural(x: &Natural) -> Option<Self> {
        if let Ok(w) = u64::try_from(x) {
            return Some(Self::from_word(w));
        }
        let bits = x.significant_bits();
        if bits > 64 * N as u64 {
            return None;
        }
        let mut limbs = [0u64; N];
        for (i, w) in x.limbs().enumerate() {
            let bit = i * Limb::BITS as usize;
            // A Malachite limb is 64 bits wide, or 32 under its `32_bit_limbs`
            // feature, where this conversion is not the identity.
            #[allow(clippy::useless_conversion)]
            let w = u64::from(w);
            limbs[bit / 64] |= w << (bit % 64);
        }
        Some(Self {
            limbs,
            len: bits.div_ceil(64) as usize,
        })
    }

    /// The value as a big integer.
    pub(crate) fn to_natural(self) -> Natural {
        self.used()
            .iter()
            .rev()
            .fold(Natural::ZERO, |acc, &w| (acc << 64u32) + Natural::from(w))
    }

    /// The little-endian words up to the highest nonzero one.
    pub(crate) fn used(&self) -> &[u64] {
        &self.limbs[..self.len]
    }

    pub(crate) fn bits(&self) -> u64 {
        self.used().last().map_or(0, |&top| {
            64 * self.len as u64 - u64::from(top.leading_zeros())
        })
    }

    pub(crate) fn word(&self) -> Option<u64> {
        (self.len <= 1).then_some(self.limbs[0])
    }

    /// `(w, s)` with `w` the leading at most 64 bits, `self = w · 2^s + r`,
    /// `r < 2^s`.
    pub(crate) fn leading_word(&self) -> (u64, u64) {
        let s = self.bits().saturating_sub(64);
        let (i, k) = ((s / 64) as usize, (s % 64) as u32);
        let mut w = self.limbs[i] >> k;
        if k > 0 {
            w |= self.limbs[i + 1] << (64 - k);
        }
        (w, s)
    }

    /// `self *= x` when the product fits `N` limbs, `None` (leaving `self`
    /// unspecified) otherwise. A word `x` takes one pass in place.
    pub(crate) fn checked_mul_assign(&mut self, x: &Natural) -> Option<()> {
        let Ok(w) = u64::try_from(x) else {
            *self = self.checked_mul_wide(x)?;
            return Some(());
        };
        self.checked_mul_word(w)
    }

    /// [`Self::checked_mul_assign`] for a factor given as little-endian
    /// 64-bit words, high zero words allowed.
    pub(crate) fn checked_mul_words(&mut self, words: &[u64]) -> Option<()> {
        let used = words.len() - words.iter().rev().take_while(|&&w| w == 0).count();
        match &words[..used] {
            [] => self.checked_mul_word(0),
            [w] => self.checked_mul_word(*w),
            wide => {
                *self = self.checked_mul(&Self::from_words(wide)?)?;
                Some(())
            }
        }
    }

    /// The value of `words` when it has at most `N` of them.
    fn from_words(words: &[u64]) -> Option<Self> {
        let mut limbs = [0u64; N];
        limbs.get_mut(..words.len())?.copy_from_slice(words);
        Some(Self::trimmed(limbs, words.len()))
    }

    /// `self *= w` in one pass in place.
    fn checked_mul_word(&mut self, w: u64) -> Option<()> {
        let mut carry = 0u64;
        for l in &mut self.limbs[..self.len] {
            let t = u128::from(*l) * u128::from(w) + u128::from(carry);
            *l = t as u64;
            carry = (t >> 64) as u64;
        }
        if w == 0 {
            self.len = 0;
        } else if carry != 0 {
            *self.limbs.get_mut(self.len)? = carry;
            self.len += 1;
        }
        Some(())
    }

    // Out of line, so that the word loop above stays small where it inlines.
    #[inline(never)]
    fn checked_mul_wide(&self, x: &Natural) -> Option<Self> {
        self.checked_mul(&Limbs::<N>::from_natural(x)?)
    }

    /// `self · rhs` when the product fits `M` limbs.
    pub(crate) fn checked_mul<const K: usize, const M: usize>(
        &self,
        rhs: &Limbs<K>,
    ) -> Option<Limbs<M>> {
        let (a, b) = (self.used(), rhs.used());
        let mut r = [0u64; M];
        if a.is_empty() || b.is_empty() {
            return Some(Limbs { limbs: r, len: 0 });
        }
        // The product has `a.len() + b.len() - 1` limbs or one more.
        if a.len() + b.len() - 1 > M {
            return None;
        }
        for (i, &x) in a.iter().enumerate() {
            let mut carry = 0u64;
            for (j, &y) in b.iter().enumerate() {
                let t = u128::from(x) * u128::from(y) + u128::from(r[i + j]) + u128::from(carry);
                r[i + j] = t as u64;
                carry = (t >> 64) as u64;
            }
            match r.get_mut(i + b.len()) {
                Some(top) => *top = carry,
                None if carry != 0 => return None,
                None => {}
            }
        }
        Some(Limbs::trimmed(r, (a.len() + b.len()).min(M)))
    }

    /// `self · 2^k` when it fits `M` limbs.
    pub(crate) fn checked_shl<const M: usize>(&self, k: u64) -> Option<Limbs<M>> {
        let mut r = [0u64; M];
        let bits = self.bits();
        if bits == 0 {
            return Some(Limbs { limbs: r, len: 0 });
        }
        if bits + k > 64 * M as u64 {
            return None;
        }
        let (s, b) = ((k / 64) as usize, (k % 64) as u32);
        // Every used limb lands below `M` by the bit count above, and so does
        // its high part whenever that is nonzero.
        for (i, &w) in self.used().iter().enumerate() {
            r[i + s] |= w << b;
            if b > 0 && i + s + 1 < M {
                r[i + s + 1] |= w >> (64 - b);
            }
        }
        Some(Limbs {
            limbs: r,
            len: (bits + k).div_ceil(64) as usize,
        })
    }
}

impl<const N: usize> Ord for Limbs<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.len
            .cmp(&other.len)
            .then_with(|| self.used().iter().rev().cmp(other.used().iter().rev()))
    }
}

impl<const N: usize> PartialOrd for Limbs<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The low 64 bits of a `u128`.
const LOW: u128 = u64::MAX as u128;

/// Adds the split products of `a·b` (little-endian words) into the lazy
/// columns `cols`, one column per word position: each `64 × 64` product adds
/// its low word to column `i + j` and its high word to column `i + j + 1`.
/// `cols` holds at least `a.len() + b.len()` columns.
///
/// One call adds at most `2·min(a.len(), b.len())` summands, each below
/// `2^64`, to any column. The caller bounds the calls per column so that a
/// column plus the settlement carry stays below `2^128` (`docs/gauge_cartan.md`
/// §9 and `frcore::accum`); nothing is checked here.
pub(crate) fn mac_into(a: &[u64], b: &[u64], cols: &mut [u128]) {
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            let p = u128::from(x) * u128::from(y);
            cols[i + j] += p & LOW;
            cols[i + j + 1] += p >> 64;
        }
    }
}

/// `Σ cols[k]·2^(64k)` as little-endian words in `out`, which holds at
/// least `cols.len() + 1` words; every column is cleared. Each column plus
/// the incoming carry is below `2^128` by the caller's bound, so every carry
/// is below `2^64`.
pub(crate) fn settle(cols: &mut [u128], out: &mut [u64]) {
    let mut carry = 0u128;
    for (o, c) in out.iter_mut().zip(cols.iter_mut()) {
        let v = *c + carry;
        *o = v as u64;
        carry = v >> 64;
        *c = 0;
    }
    let (top, rest) = out[cols.len()..].split_at_mut(1);
    top[0] = carry as u64;
    rest.fill(0);
}

/// `a·b` exactly, as little-endian words in `out`, which holds at least
/// `a.len() + b.len()` words; schoolbook, with every partial below `2^128`.
pub(crate) fn mul_into(a: &[u64], b: &[u64], out: &mut [u64]) {
    out.fill(0);
    for (i, &x) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &y) in b.iter().enumerate() {
            let t = u128::from(x) * u128::from(y) + u128::from(out[i + j]) + u128::from(carry);
            out[i + j] = t as u64;
            carry = (t >> 64) as u64;
        }
        out[i + b.len()] = carry;
    }
}

/// `words` without its high zero words.
pub(crate) fn trimmed_words(words: &[u64]) -> &[u64] {
    &words[..words.len() - words.iter().rev().take_while(|&&w| w == 0).count()]
}
