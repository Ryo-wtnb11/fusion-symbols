//! The exact layer of the Cartan CGC (`docs/gauge_cartan.md` §4.7, §9): the
//! stored authority of a channel, of its highest-weight column and of one
//! irrep's norms. No floating point enters this module; the rounded channel
//! that `LieCgc` serves is derived from these by [`crate::rounding::cgc`].
//!
//! An entry of `a ⊗ b → c` is `C = x·√(n^a n^b/(N_μ n^c))` with `x` the
//! rational coordinate of the descent. Per `μ`, every emitted `x` is put on
//! one common denominator `D_μ` with the content `g_μ` divided out: the
//! stored integer is `X = x·D_μ/g_μ` and the vertex factor
//! `ρ_μ = q(N_μ) g_μ²/(D_μ² p(N_μ))`, so that `X² ρ_μ = x²/N_μ` and
//! `C = sign(X)·√(X² ρ_μ n^a n^b/n^c)` is the same real number.

use std::ops::Range;

use malachite_base::num::arithmetic::traits::Gcd;
use malachite_base::num::basic::traits::{One, Zero};
use malachite_nz::natural::Natural;
#[cfg(feature = "disk-cache")]
use malachite_q::Rational;

use super::Ladders;
use crate::exact::radical::{compact, natural_bytes, put_u32, Radicand, SignedInts};
#[cfg(feature = "disk-cache")]
use crate::exact::radical::{decode_natural, encode_natural, natural_encoded_len, read_u32};
use crate::exact::word_gcd;
use crate::lie::general::module::Q;
use crate::lie::LieError;

/// One emitted column of a descent: `(public m_a, public m_b, x)` for every
/// exactly nonzero coordinate, in the a-slow order of §4.1.
pub(crate) type Column = Vec<(u32, u32, Q)>;

/// The integer coordinates of one `μ`: `X` per entry, all nonzero, and the
/// vertex factor `ρ_μ`, reduced once.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct MuCoords {
    /// The largest `|X|` width in bits over the `μ` range (recomputed on
    /// decode): width metadata for exact F and R and the width reports.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) bits: u64,
    pub(crate) x: SignedInts,
    pub(crate) vertex: Radicand,
}

impl MuCoords {
    /// The coordinates `xs` of one `μ` with `N_μ = norm > 0`, integralized
    /// over all of them, `X` negated when `negate` (the `A_1` vertex sign,
    /// §4.6).
    fn new<'a>(
        xs: impl Iterator<Item = &'a Q> + Clone,
        norm: &Q,
        negate: bool,
    ) -> Result<Self, LieError> {
        let Integral { x, den, content } = Integral::new(xs, negate)?;
        let (p, q) = norm.magnitudes();
        let num = &*q * &content * &content;
        let den = &*p * &den * &den;
        let g = (&num).gcd(&den);
        let vertex = Radicand::new(num / &g, den / &g).expect("N_μ, D_μ and g_μ are positive");
        Ok(Self {
            bits: x.max_bits(),
            x,
            vertex,
        })
    }

    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.x.heap_bytes())
            .saturating_add(self.vertex.bytes())
    }

    /// `sign(X)·X²·ρ_μ` of entry `i` of this `μ`.
    #[cfg(test)]
    pub(crate) fn signed_square(&self, i: usize) -> malachite_q::Rational {
        let x = self.x.integer(i);
        let sq = malachite_q::Rational::from(&x * &x)
            * malachite_q::Rational::from_naturals(
                self.vertex.num().clone(),
                self.vertex.den().clone(),
            );
        if self.x.is_negative(i) {
            -sq
        } else {
            sq
        }
    }
}

/// Rationals on their least common denominator with the content divided
/// out: `values[i]·content/den` is the `i`-th input.
struct Integral {
    x: SignedInts,
    den: Natural,
    content: Natural,
}

impl Integral {
    /// In machine words when every denominator, the common one and every
    /// scaled numerator fit, in big integers otherwise; both give the same
    /// integers.
    fn new<'a>(xs: impl Iterator<Item = &'a Q> + Clone, negate: bool) -> Result<Self, LieError> {
        if let Some((values, den, content)) = word_integral(xs.clone(), negate) {
            return Ok(Self {
                x: SignedInts::from_words(&values).ok_or(LieError::LabelOverflow)?,
                den: Natural::from(den),
                content: Natural::from(content),
            });
        }
        let mut den = Natural::ONE;
        for x in xs.clone() {
            let (_, q) = x.magnitudes();
            let g = (&den).gcd(&*q);
            den = den / g * &*q;
        }
        let mut content = Natural::ZERO;
        let mut values = Vec::new();
        for x in xs {
            let (p, q) = x.magnitudes();
            let v = &*p * (&den / &*q);
            content = content.gcd(&v);
            values.push((x.signum().is_lt() != negate, v));
        }
        if content == Natural::ZERO {
            content = Natural::ONE;
        }
        for (_, v) in &mut values {
            *v /= &content;
        }
        Ok(Self {
            x: SignedInts::from_naturals(&values).ok_or(LieError::LabelOverflow)?,
            den: compact(den),
            content: compact(content),
        })
    }
}

/// `(negative, X)` per input, the common denominator `D` and the content `g`.
type WordIntegral = (Vec<(bool, u64)>, u64, u64);

/// [`Integral::new`] in machine words, or `None` when a value does not fit.
fn word_integral<'a>(
    xs: impl Iterator<Item = &'a Q> + Clone,
    negate: bool,
) -> Option<WordIntegral> {
    let mut den = 1u64;
    for x in xs.clone() {
        let (_, _, q) = x.word_parts()?;
        den = (den / word_gcd(den, q)).checked_mul(q)?;
    }
    let mut content = 0u64;
    let mut values = Vec::new();
    for x in xs {
        let (neg, p, q) = x.word_parts()?;
        let v = p.checked_mul(den / q)?;
        content = word_gcd(content, v);
        values.push((neg != negate, v));
    }
    let content = content.max(1);
    for (_, v) in &mut values {
        *v /= content;
    }
    Some((values, den, content))
}

/// The coordinates of a run of `μ` blocks of `width` columns each: the
/// public pairs of every entry, the exclusive end of every column, and the
/// integer coordinates of every `μ`.
#[derive(Debug, PartialEq, Eq)]
struct Coords {
    m_ab: Box<[(u32, u32)]>,
    ends: Box<[usize]>,
    per_mu: Box<[MuCoords]>,
}

impl Coords {
    /// `columns` in `(μ, column)` order, `width` per `μ`, and `N_μ` per `μ`.
    fn new(
        columns: Vec<Column>,
        width: usize,
        norms: &[Q],
        negate: bool,
    ) -> Result<Self, LieError> {
        u32::try_from(norms.len()).map_err(|_| LieError::LabelOverflow)?;
        debug_assert_eq!(columns.len(), norms.len() * width);
        let count = columns.iter().map(Vec::len).sum();
        let mut m_ab = Vec::with_capacity(count);
        let mut ends = Vec::with_capacity(columns.len());
        let mut per_mu = Vec::with_capacity(norms.len());
        let mut rest = columns.into_iter();
        for norm in norms {
            // One μ at a time, so its rationals are dropped once integral.
            let cols: Vec<Column> = rest.by_ref().take(width).collect();
            per_mu.push(MuCoords::new(
                cols.iter().flatten().map(|e| &e.2),
                norm,
                negate,
            )?);
            for col in &cols {
                m_ab.extend(col.iter().map(|e| (e.0, e.1)));
                ends.push(m_ab.len());
            }
        }
        Ok(Self {
            m_ab: m_ab.into_boxed_slice(),
            ends: ends.into_boxed_slice(),
            per_mu: per_mu.into_boxed_slice(),
        })
    }

    fn empty() -> Self {
        Self {
            m_ab: Box::new([]),
            ends: Box::new([]),
            per_mu: Box::new([]),
        }
    }

    /// The entries of column `k`.
    fn column(&self, k: usize) -> Range<usize> {
        let start = if k == 0 { 0 } else { self.ends[k - 1] };
        start..self.ends[k]
    }

    /// The entries of `μ`, whose columns are `μ·width ..= (μ+1)·width − 1`.
    fn mu_range(&self, mu: usize, width: usize) -> Range<usize> {
        self.column(mu * width).start..self.ends[(mu + 1) * width - 1]
    }

    fn bytes(&self) -> usize {
        std::mem::size_of_val(self.m_ab.as_ref())
            .saturating_add(std::mem::size_of_val(self.ends.as_ref()))
            .saturating_add(self.per_mu.iter().map(MuCoords::bytes).sum())
    }

    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    fn encoded_len(&self) -> Option<usize> {
        u32::try_from(self.m_ab.len()).ok()?;
        let mut n = self
            .m_ab
            .len()
            .checked_mul(8)?
            .checked_add(self.ends.len().checked_mul(4)?)?
            .checked_add(4)?;
        for mu in self.per_mu.iter() {
            n = n
                .checked_add(mu.x.encoded_len()?)?
                .checked_add(mu.vertex.encoded_len()?)?;
        }
        Some(n)
    }

    /// `u32 count`, `count × (u32 m_a, u32 m_b)`, the column ends as `u32`,
    /// then per `μ` its slab and its radicand.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    fn encode(&self, out: &mut Vec<u8>) {
        put_u32(out, self.m_ab.len());
        for &(ma, mb) in self.m_ab.iter() {
            out.extend_from_slice(&ma.to_le_bytes());
            out.extend_from_slice(&mb.to_le_bytes());
        }
        for &end in self.ends.iter() {
            put_u32(out, end);
        }
        for mu in self.per_mu.iter() {
            mu.x.encode(out);
            mu.vertex.encode(out);
        }
    }

    /// The coordinates at `*at` of `mult` blocks of `width` columns over
    /// `d_a × d_b`, checked before any arithmetic or indexing: every count
    /// against the remaining bytes before allocating, every pair in range and
    /// strictly ascending within its column, every column nonempty (its end
    /// strictly above the previous one, from an implicit 0) with the last end
    /// at the entry count, and per `μ` a canonical slab of exactly the `μ`
    /// range's length with every entry nonzero and a positive radicand.
    #[cfg(feature = "disk-cache")]
    fn decode(
        bytes: &[u8],
        at: &mut usize,
        mult: usize,
        width: usize,
        [da, db]: [usize; 2],
    ) -> Option<Self> {
        let count = read_u32(bytes, at)? as usize;
        let pairs = bytes.get(*at..at.checked_add(count.checked_mul(8)?)?)?;
        *at += pairs.len();
        let m_ab: Box<[(u32, u32)]> = pairs
            .as_chunks::<4>()
            .0
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[a, b]| (u32::from_le_bytes(*a), u32::from_le_bytes(*b)))
            .collect();
        if m_ab
            .iter()
            .any(|&(ma, mb)| ma as usize >= da || mb as usize >= db)
        {
            return None;
        }
        let columns = mult.checked_mul(width)?;
        let raw = bytes.get(*at..at.checked_add(columns.checked_mul(4)?)?)?;
        *at += raw.len();
        let ends: Box<[usize]> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|e| u32::from_le_bytes(*e) as usize)
            .collect();
        let mut start = 0;
        for &end in ends.iter() {
            if end <= start || end > count {
                return None;
            }
            if !m_ab[start..end].is_sorted_by(|x, y| x < y) {
                return None;
            }
            start = end;
        }
        if start != count {
            return None;
        }
        let mut coords = Self {
            m_ab,
            ends,
            per_mu: Box::new([]),
        };
        let mut per_mu = Vec::new();
        for mu in 0..mult {
            let x = SignedInts::decode(bytes, at)?;
            let vertex = Radicand::decode(bytes, at)?;
            if x.len() != coords.mu_range(mu, width).len() || !x.all_nonzero() {
                return None;
            }
            per_mu.push(MuCoords {
                bits: x.max_bits(),
                x,
                vertex,
            });
        }
        coords.per_mu = per_mu.into_boxed_slice();
        Some(coords)
    }
}

/// The exact channel `a ⊗ b → c`: every exactly nonzero coordinate in the
/// §4.7 order (`μ` outer, public `m_c` ascending, `(m_a, m_b)` a-slow), with
/// one column per `k = μ·d_c + m_c`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ExactCgc {
    mult: usize,
    /// Weyl dimensions `d_a, d_b, d_c`.
    dims: [usize; 3],
    coords: Coords,
}

impl ExactCgc {
    /// The channel of multiplicity zero.
    pub(crate) fn empty(dims: [usize; 3]) -> Self {
        Self {
            mult: 0,
            dims,
            coords: Coords::empty(),
        }
    }

    /// The channel of the emitted `columns` (`μ·d_c + m_c` order) with
    /// `N_μ = norms[μ]`, every `X` negated when `negate`.
    pub(crate) fn from_columns(
        dims: [usize; 3],
        columns: Vec<Column>,
        norms: &[Q],
        negate: bool,
    ) -> Result<Self, LieError> {
        Ok(Self {
            mult: norms.len(),
            dims,
            coords: Coords::new(columns, dims[2], norms, negate)?,
        })
    }

    pub(crate) fn mult(&self) -> usize {
        self.mult
    }

    pub(crate) fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// The number of exactly nonzero entries.
    pub(crate) fn len(&self) -> usize {
        self.coords.m_ab.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.coords.m_ab.is_empty()
    }

    /// The public `(m_a, m_b)` of every entry.
    pub(crate) fn m_ab(&self) -> &[(u32, u32)] {
        &self.coords.m_ab
    }

    /// The exclusive end of every column `μ·d_c + m_c`.
    pub(crate) fn column_ends(&self) -> &[usize] {
        &self.coords.ends
    }

    /// The coordinates of `μ`; its entry `j` is entry
    /// `self.mu_range(mu).start + j` of the channel.
    pub(crate) fn coords(&self, mu: usize) -> &MuCoords {
        &self.coords.per_mu[mu]
    }

    /// The entries of `μ`.
    pub(crate) fn mu_range(&self, mu: usize) -> Range<usize> {
        self.coords.mu_range(mu, self.dims[2])
    }

    /// Bytes owned: slabs, bitsets, column ends, pairs and compacted limbs.
    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(self.coords.bytes())
    }

    /// `(m_a, m_b, m_c, μ, sign(X)·X²·ρ_μ)` per entry, in order.
    #[cfg(test)]
    pub(crate) fn signed_squares(&self) -> Vec<(u32, u32, u32, u32, malachite_q::Rational)> {
        let dc = self.dims[2];
        let mut out = Vec::with_capacity(self.len());
        for k in 0..self.coords.ends.len() {
            let mu = k / dc;
            let start = self.mu_range(mu).start;
            for i in self.coords.column(k) {
                let (ma, mb) = self.coords.m_ab[i];
                let sq = self.coords.per_mu[mu].signed_square(i - start);
                out.push((ma, mb, (k % dc) as u32, mu as u32, sq));
            }
        }
        out
    }

    /// Payload bytes, or `None` when a count does not fit its `u32` field.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn encoded_len(&self) -> Option<usize> {
        for x in [self.mult, self.dims[0], self.dims[1], self.dims[2]] {
            u32::try_from(x).ok()?;
        }
        self.coords.encoded_len()?.checked_add(16)
    }

    /// `u32 mult`, `3 × u32` dims, then the coordinates ([`Coords::encode`]).
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        put_u32(out, self.mult);
        for d in self.dims {
            put_u32(out, d);
        }
        self.coords.encode(out);
    }

    /// The channel of `payload`, if it is exactly a channel of multiplicity
    /// `mult` over `dims` ([`Coords::decode`]) with no trailing bytes.
    #[cfg(feature = "disk-cache")]
    pub(crate) fn decode(payload: &[u8], dims: [usize; 3], mult: usize) -> Option<Self> {
        let at = &mut 0;
        if read_u32(payload, at)? as usize != mult {
            return None;
        }
        for d in dims {
            if read_u32(payload, at)? as usize != d {
                return None;
            }
        }
        // `mu_range` reads the last column of each μ.
        if mult > 0 && dims[2] == 0 {
            return None;
        }
        let coords = Coords::decode(payload, at, mult, dims[2], [dims[0], dims[1]])?;
        (*at == payload.len()).then_some(Self { mult, dims, coords })
    }
}

/// The highest-weight column (engine index 0 of `c`) of `a ⊗ b → c`: per
/// `μ` the coordinates of `√N_μ Φ_μ(v_λ)` over `P`, with their own
/// integral scale.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct HighestColumn {
    mult: usize,
    /// One column per `μ`.
    coords: Coords,
}

// Read by exact F and R; until then only tests and the disk tier.
#[cfg_attr(not(test), allow(dead_code))]
impl HighestColumn {
    pub(crate) fn empty() -> Self {
        Self {
            mult: 0,
            coords: Coords::empty(),
        }
    }

    /// The seed `columns`, one per `μ`, with `N_μ = norms[μ]`.
    pub(crate) fn from_columns(columns: Vec<Column>, norms: &[Q]) -> Result<Self, LieError> {
        Ok(Self {
            mult: norms.len(),
            coords: Coords::new(columns, 1, norms, false)?,
        })
    }

    pub(crate) fn mult(&self) -> usize {
        self.mult
    }

    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(self.coords.bytes())
    }

    /// `(m_a, m_b, μ, sign(X)·X²·ρ_μ)` per entry, in order.
    #[cfg(test)]
    pub(crate) fn signed_squares(&self) -> Vec<(u32, u32, u32, malachite_q::Rational)> {
        let mut out = Vec::new();
        for mu in 0..self.mult {
            let range = self.coords.mu_range(mu, 1);
            for i in range.clone() {
                let (ma, mb) = self.coords.m_ab[i];
                let sq = self.coords.per_mu[mu].signed_square(i - range.start);
                out.push((ma, mb, mu as u32, sq));
            }
        }
        out
    }

    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn encoded_len(&self) -> Option<usize> {
        u32::try_from(self.mult).ok()?;
        self.coords.encoded_len()?.checked_add(4)
    }

    /// `u32 mult`, then the coordinates with one column per `μ`.
    #[cfg_attr(not(feature = "disk-cache"), allow(dead_code))]
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        put_u32(out, self.mult);
        self.coords.encode(out);
    }

    /// The seeds of `payload` over `d_a × d_b`, checked as a channel's
    /// coordinates ([`Coords::decode`]) with no trailing bytes.
    #[cfg(feature = "disk-cache")]
    pub(crate) fn decode(payload: &[u8], dims: [usize; 2], mult: usize) -> Option<Self> {
        let at = &mut 0;
        if read_u32(payload, at)? as usize != mult {
            return None;
        }
        let coords = Coords::decode(payload, at, mult, 1, dims)?;
        (*at == payload.len()).then_some(Self { mult, coords })
    }
}

/// The norm data of one irrep, indexed by **public** magnetic index: the
/// reduced `n_m` (read by the rounding layer), the integral `ν_m` with
/// `n_m = ν_m g/Δ` for the whole-table denominator lcm `Δ` and the content
/// `g` of the scaled norms (read by exact F and R), and each index's
/// position within its weight space.
// `ν`, `Δ`, `g`, ranks, slots and top are read by exact F and R;
// until then only by the disk tier and tests.
#[cfg_attr(not(any(test, feature = "disk-cache")), allow(dead_code))]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NormTable {
    pub(crate) norm: Box<[Q]>,
    pub(crate) nu: SignedInts,
    pub(crate) den: Natural,
    pub(crate) content: Natural,
    pub(crate) rank: Box<[u32]>,
    /// The largest weight multiplicity; every rank is below it.
    pub(crate) slots: u32,
    /// The public index of engine index 0, the highest weight.
    pub(crate) top: u32,
}

impl NormTable {
    /// The table of `l`, whose `public` map is a permutation (checked).
    pub(crate) fn new(l: &Ladders) -> Result<Self, LieError> {
        let dim = l.dim();
        let mut seen = vec![false; dim];
        let mut norm = vec![Q::from(0); dim];
        let mut rank = vec![0u32; dim];
        for (m, &p) in l.public.iter().enumerate() {
            let p = p as usize;
            assert!(
                !std::mem::replace(&mut seen[p], true),
                "the public magnetic indices are a permutation (§3)"
            );
            norm[p] = l.norm[m].clone();
            norm[p].compact();
            rank[p] = l.rank[m];
        }
        let Integral {
            x: nu,
            den,
            content,
        } = Integral::new(norm.iter(), false)?;
        let slots = l
            .module
            .weights
            .iter()
            .map(|s| s.norms.len())
            .max()
            .unwrap_or(0);
        Ok(Self {
            norm: norm.into_boxed_slice(),
            nu,
            den,
            content,
            rank: rank.into_boxed_slice(),
            // A weight multiplicity is at most the dimension, a `u32`.
            slots: slots as u32,
            top: l.public[0],
        })
    }

    /// Bytes owned, with every rational and limb vector compacted.
    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(
                self.norm
                    .iter()
                    .map(|q| std::mem::size_of::<Q>().saturating_add(q.heap_bytes()))
                    .sum(),
            )
            .saturating_add(self.nu.heap_bytes())
            .saturating_add(natural_bytes(&self.den))
            .saturating_add(natural_bytes(&self.content))
            .saturating_add(std::mem::size_of_val(self.rank.as_ref()))
    }

    #[cfg(feature = "disk-cache")]
    pub(crate) fn encoded_len(&self) -> Option<usize> {
        u32::try_from(self.norm.len()).ok()?;
        let mut n = 4usize;
        for q in self.norm.iter() {
            let (p, d) = q.magnitudes();
            n = n
                .checked_add(natural_encoded_len(&p)?)?
                .checked_add(natural_encoded_len(&d)?)?;
        }
        n.checked_add(self.nu.encoded_len()?)?
            .checked_add(natural_encoded_len(&self.den)?)?
            .checked_add(natural_encoded_len(&self.content)?)?
            .checked_add(self.rank.len().checked_mul(4)?)?
            .checked_add(8)
    }

    /// `u32 dim`, `dim × (n_m numerator, denominator)`, the slab `ν`, `Δ`,
    /// `g`, `dim × u32 rank`, `u32 slots`, `u32 top`.
    #[cfg(feature = "disk-cache")]
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        put_u32(out, self.norm.len());
        for q in self.norm.iter() {
            let (p, d) = q.magnitudes();
            encode_natural(&p, out);
            encode_natural(&d, out);
        }
        self.nu.encode(out);
        encode_natural(&self.den, out);
        encode_natural(&self.content, out);
        for &r in self.rank.iter() {
            out.extend_from_slice(&r.to_le_bytes());
        }
        out.extend_from_slice(&self.slots.to_le_bytes());
        out.extend_from_slice(&self.top.to_le_bytes());
    }

    /// The table of `payload` for an irrep of dimension `dim`, checked before
    /// any arithmetic or indexing: every norm positive, `ν` a canonical slab
    /// of `dim` entries `≥ 1`, `Δ, g ≥ 1`, `dim` ranks below `slots ≥ 1`,
    /// `top < dim`, and no trailing bytes. A truncated table fails.
    #[cfg(feature = "disk-cache")]
    pub(crate) fn decode(payload: &[u8], dim: usize) -> Option<Self> {
        let at = &mut 0;
        if read_u32(payload, at)? as usize != dim {
            return None;
        }
        // Grown per decoded entry, so a short payload allocates no `dim` slots.
        let mut norm = Vec::new();
        for _ in 0..dim {
            let p = decode_natural(payload, at)?;
            let d = decode_natural(payload, at)?;
            if p == Natural::ZERO || d == Natural::ZERO {
                return None;
            }
            let mut q = Q::from(Rational::from_naturals(p, d));
            q.compact();
            norm.push(q);
        }
        let nu = SignedInts::decode(payload, at)?;
        if nu.len() != dim || !nu.all_nonzero() || (0..dim).any(|i| nu.is_negative(i)) {
            return None;
        }
        let den = decode_natural(payload, at)?;
        let content = decode_natural(payload, at)?;
        if den == Natural::ZERO || content == Natural::ZERO {
            return None;
        }
        let raw = payload.get(*at..at.checked_add(dim.checked_mul(4)?)?)?;
        *at += raw.len();
        let rank: Box<[u32]> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|r| u32::from_le_bytes(*r))
            .collect();
        let slots = read_u32(payload, at)?;
        let top = read_u32(payload, at)?;
        if slots == 0 || rank.iter().any(|&r| r >= slots) || top as usize >= dim {
            return None;
        }
        (*at == payload.len()).then_some(Self {
            norm: norm.into_boxed_slice(),
            nu,
            den,
            content,
            rank,
            slots,
            top,
        })
    }
}
