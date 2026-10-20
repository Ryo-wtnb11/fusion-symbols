/// A Gelfand–Tsetlin pattern of an SU(N) irrep.
///
/// Storage mirrors `gtpatterns.jl:GTPattern`: the triangular array flattened
/// top row first (`l = N`, then `l = N-1`, …, `l = 1`), each row `l` occupying
/// `l` contiguous entries. Access via [`GtPattern::get`] with 1-based
/// `(k, l)`, `1 ≤ k ≤ l ≤ N`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GtPattern {
    pub(super) n: usize,
    pub(super) data: Box<[i64]>,
}

impl GtPattern {
    /// The flat pattern data, top row (`l = N`) first (the reference
    /// `m.data` order).
    pub fn data(&self) -> &[i64] {
        &self.data
    }

    /// The rank `N`.
    pub fn rank(&self) -> usize {
        self.n
    }

    /// Entry `m[k, l]`, `1 ≤ k ≤ l ≤ N`. Panics only on an out-of-range index,
    /// which is an internal invariant of the enumeration/ladder code (never
    /// reachable from public input).
    #[inline]
    pub fn get(&self, k: usize, l: usize) -> i64 {
        self.data[Self::flat_index(self.n, k, l)]
    }

    #[inline]
    #[cfg(feature = "cgc-gen")]
    pub(super) fn set(&mut self, k: usize, l: usize, v: i64) {
        let idx = Self::flat_index(self.n, k, l);
        self.data[idx] = v;
    }

    /// 0-based flat index of `m[k, l]`. The reference (1-based) index is
    /// `k + ((l+1+N)(N-l))>>1`; subtracting 1 gives the 0-based form.
    #[inline]
    fn flat_index(n: usize, k: usize, l: usize) -> usize {
        (k - 1) + (((l + 1 + n) * (n - l)) >> 1)
    }
}
