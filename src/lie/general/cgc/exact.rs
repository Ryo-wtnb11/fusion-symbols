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
use std::sync::{Mutex, PoisonError};

use malachite_base::num::arithmetic::traits::{DivisibleBy, Gcd};
use malachite_base::num::basic::traits::{One, Zero};
use malachite_base::num::logic::traits::SignificantBits;
use malachite_nz::natural::Natural;
#[cfg(feature = "disk-cache")]
use malachite_q::Rational;
use threadteam::Exec;

use super::Ladders;
use crate::exact::radical::{compact, natural_bytes, put_u32, write_natural, Radicand, SignedInts};
#[cfg(feature = "disk-cache")]
use crate::exact::radical::{decode_natural, encode_natural, natural_encoded_len, read_u32};
use crate::exact::{div as wide_div, gcd as wide_gcd, rem as wide_rem};
use crate::lie::general::module::Q;
use crate::lie::LieError;
use crate::stage::{run_items, split_parts, take_part, StageBuf};

/// One emitted column of a descent: `(public m_a, public m_b, x)` for every
/// exactly nonzero coordinate, in the a-slow order of §4.1, with its
/// [`Scale`], reduced by the item that emits it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Column {
    entries: Vec<(u32, u32, Q)>,
    scale: Scale,
}

impl Column {
    /// The column of `entries` and its scale.
    pub(crate) fn new(entries: Vec<(u32, u32, Q)>) -> Self {
        let scale = Scale::of(entries.iter().map(|e| &e.2));
        Self { entries, scale }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

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

/// `ρ_μ = q(N_μ) g_μ²/(p(N_μ) D_μ²)` for `N_μ = norm > 0`, reduced once.
fn vertex(norm: &Q, den: &Natural, content: &Natural) -> Radicand {
    let (p, q) = norm.magnitudes();
    let num = &*q * content * content;
    let den = &*p * den * den;
    let g = (&num).gcd(&den);
    Radicand::new(num / &g, den / &g).expect("N_μ, D_μ and g_μ are positive")
}

/// Rationals on their least common denominator with the content divided
/// out: `values[i]·content/den` is the `i`-th input.
#[derive(Debug, PartialEq, Eq)]
struct Integral {
    x: SignedInts,
    den: Natural,
    content: Natural,
}

// For reduced `p_i/q_i` and `D = lcm q_i`:
//
// - the content `gcd_i(p_i·D/q_i)` is `gcd_i p_i`. A prime dividing `D`
//   divides some `q_i` to the full power of `D`, and then neither `p_i`
//   (coprime to `q_i`) nor `D/q_i` holds it; a prime not dividing `D` divides
//   each `p_i·D/q_i` exactly as often as `p_i`. So the content is a reduction
//   over the numerators alone, independent of `D`;
// - `|X_i| = |x_i|·D/g` grows with `|x_i|`, so the widest `X` of a run is
//   that of its largest `|x|`, and the canonical stride is known before any
//   `X` is written.

/// [`Integral`] of `xs` on the calling thread.
fn integral<'a>(
    xs: impl Iterator<Item = &'a Q> + Clone,
    negate: bool,
) -> Result<Integral, LieError> {
    integral_in(xs.clone(), negate, Scale::of(xs))
}

/// [`integral`] with the scale of `xs` given: in two words when the scale
/// and the widest `X` fit them, in big integers otherwise; both give the
/// same integers.
fn integral_in<'a>(
    xs: impl Iterator<Item = &'a Q> + Clone,
    negate: bool,
    scale: Scale,
) -> Result<Integral, LieError> {
    let top = scale.wide_top();
    let ints = Integers::new(scale);
    let wide = top.is_some_and(|t| ints.fits_wide(t));
    let len = xs.clone().count();
    let mut mag = vec![
        0u64;
        len.checked_mul(ints.stride)
            .ok_or(LieError::LabelOverflow)?
    ];
    let neg = ints.write(len, xs, negate, wide, &mut mag);
    Ok(Integral {
        x: SignedInts::from_raw(len, ints.stride, neg, mag),
        den: ints.den,
        content: ints.content,
    })
}

/// `(negative, |numerator|, denominator)` of `x` when both fit two words.
fn wide_parts(x: &Q) -> Option<(bool, u128, u128)> {
    if let Some((neg, p, q)) = x.word_parts() {
        return Some((neg, p.into(), q.into()));
    }
    let (p, q) = x.magnitudes();
    Some((
        x.signum().is_lt(),
        u128::try_from(&*p).ok()?,
        u128::try_from(&*q).ok()?,
    ))
}

/// `a·b` as `(high, low)` two-word halves.
fn mul_wide(a: u128, b: u128) -> (u128, u128) {
    const LOW: u128 = u64::MAX as u128;
    if a >> 64 == 0 && b >> 64 == 0 {
        return (0, a * b);
    }
    let (a1, a0, b1, b0) = (a >> 64, a & LOW, b >> 64, b & LOW);
    let (p00, p01, p10, p11) = (a0 * b0, a0 * b1, a1 * b0, a1 * b1);
    let mid = (p00 >> 64) + (p01 & LOW) + (p10 & LOW);
    let low = (p00 & LOW) | ((mid & LOW) << 64);
    let high = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (high, low)
}

/// A run's scale in two words: the lcm `den` of the denominators, the gcd
/// `content` of the numerators (0 while empty) and the `(p, q)` of its
/// largest `|x|`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WideScale {
    den: u128,
    content: u128,
    top: (u128, u128),
}

impl WideScale {
    const EMPTY: Self = Self {
        den: 1,
        content: 0,
        top: (0, 1),
    };

    /// The scale of two runs, or `None` when their lcm leaves two words.
    fn merge(self, o: Self) -> Option<Self> {
        // Divisibility first: the running lcm rarely changes, and a content
        // that reached 1 stays there.
        let den = if wide_rem(self.den, o.den) == 0 {
            self.den
        } else {
            wide_div(self.den, wide_gcd(self.den, o.den)).checked_mul(o.den)?
        };
        let content = if self.content == 1 {
            1
        } else {
            wide_gcd(o.content, self.content)
        };
        let top = if mul_wide(o.top.0, self.top.1) > mul_wide(self.top.0, o.top.1) {
            o.top
        } else {
            self.top
        };
        Some(Self { den, content, top })
    }
}

/// [`WideScale`] in big integers.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BigScale {
    den: Natural,
    content: Natural,
    top: (Natural, Natural),
}

impl BigScale {
    fn single(x: &Q) -> Self {
        let (p, q) = x.magnitudes();
        let (p, q) = (p.into_owned(), q.into_owned());
        Self {
            den: q.clone(),
            content: p.clone(),
            top: (p, q),
        }
    }

    fn merge(self, o: Self) -> Self {
        let den = if (&self.den).divisible_by(&o.den) {
            self.den
        } else {
            let g = (&self.den).gcd(&o.den);
            self.den / g * &o.den
        };
        let content = if self.content == Natural::ONE {
            self.content
        } else {
            self.content.gcd(&o.content)
        };
        let top = if &o.top.0 * &self.top.1 > &self.top.0 * &o.top.1 {
            o.top
        } else {
            self.top
        };
        Self { den, content, top }
    }
}

impl From<WideScale> for BigScale {
    fn from(s: WideScale) -> Self {
        Self {
            den: Natural::from(s.den),
            content: Natural::from(s.content),
            top: (Natural::from(s.top.0), Natural::from(s.top.1)),
        }
    }
}

/// The scale of a run of rationals, in two words while its parts and lcm
/// fit them. Associative and commutative, so a `μ`'s columns reduce in any
/// grouping to the same scale.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Scale {
    Wide(WideScale),
    Big(BigScale),
}

impl Default for Scale {
    fn default() -> Self {
        Self::Wide(WideScale::EMPTY)
    }
}

impl Scale {
    fn of<'a>(xs: impl Iterator<Item = &'a Q>) -> Self {
        let mut s = Self::default();
        for x in xs {
            s = match s {
                Self::Wide(w) => match wide_parts(x).and_then(|(_, p, q)| {
                    w.merge(WideScale {
                        den: q,
                        content: p,
                        top: (p, q),
                    })
                }) {
                    Some(w) => Self::Wide(w),
                    None => Self::Big(BigScale::from(w).merge(BigScale::single(x))),
                },
                Self::Big(b) => Self::Big(b.merge(BigScale::single(x))),
            };
        }
        s
    }

    fn merge(self, o: Self) -> Self {
        match (self, o) {
            (Self::Wide(a), Self::Wide(b)) => match a.merge(b) {
                Some(s) => Self::Wide(s),
                None => Self::Big(BigScale::from(a).merge(BigScale::from(b))),
            },
            (a, b) => Self::Big(a.big().merge(b.big())),
        }
    }

    fn big(self) -> BigScale {
        match self {
            Self::Wide(w) => BigScale::from(w),
            Self::Big(b) => b,
        }
    }

    /// The largest `|x|` when every part of the run fits two words.
    fn wide_top(&self) -> Option<(u128, u128)> {
        match self {
            Self::Wide(w) => Some(w.top),
            Self::Big(_) => None,
        }
    }
}

/// The integers of one run (a `μ`, or a norm table): `D` and `g`, also in
/// two words when the run's scale was, the widest `|X|` in bits and its
/// word count, the canonical stride.
#[derive(Debug)]
struct Integers {
    den: Natural,
    content: Natural,
    wide: Option<(u128, u128)>,
    stride: usize,
    bits: u64,
}

impl Integers {
    fn new(scale: Scale) -> Self {
        let (den, content, wide, bits) = match scale {
            Scale::Wide(s) => {
                let g = s.content.max(1);
                let (p, q) = (wide_div(s.top.0, g), wide_div(s.den, s.top.1));
                let bits = match p.checked_mul(q) {
                    Some(t) => u64::from(u128::BITS - t.leading_zeros()),
                    None => (Natural::from(p) * Natural::from(q)).significant_bits(),
                };
                (
                    Natural::from(s.den),
                    Natural::from(g),
                    Some((s.den, g)),
                    bits,
                )
            }
            Scale::Big(s) => {
                let g = if s.content == Natural::ZERO {
                    Natural::ONE
                } else {
                    s.content
                };
                let bits = (&s.top.0 / &g * (&s.den / &s.top.1)).significant_bits();
                // A column whose own parts and widest `X` fit two words is
                // still written in them.
                let wide = u128::try_from(&s.den).ok().zip(u128::try_from(&g).ok());
                (s.den, g, wide, bits)
            }
        };
        Self {
            den: compact(den),
            content: compact(content),
            wide,
            // A word count of a width in bits.
            stride: bits.div_ceil(64) as usize,
            bits,
        }
    }

    /// Whether every `X` of a run whose parts fit two words and whose
    /// largest `|x|` is `top` fits two words under these integers.
    fn fits_wide(&self, top: (u128, u128)) -> bool {
        self.wide.is_some_and(|(den, g)| {
            wide_div(top.0, g)
                .checked_mul(wide_div(den, top.1))
                .is_some()
        })
    }

    /// Writes `X = (p/g)·(D/q)` of each of the `len` inputs `xs` into `mag`
    /// at the canonical stride, in two words when `wide` (its run passed
    /// [`Self::fits_wide`]) and in big integers otherwise; returns their sign
    /// words (bit `j` for input `j`, set for a nonzero `X` that is negative
    /// after `negate`). In two words it allocates only the sign words.
    fn write<'a>(
        &self,
        len: usize,
        xs: impl Iterator<Item = &'a Q>,
        negate: bool,
        wide: bool,
        mag: &mut [u64],
    ) -> Vec<u64> {
        let mut signs = vec![0u64; len.div_ceil(64)];
        let stride = self.stride;
        for (j, x) in xs.enumerate() {
            let words = &mut mag[j * stride..(j + 1) * stride];
            let (neg, nonzero) = match self.wide.filter(|_| wide) {
                Some((den, g)) => {
                    let (neg, p, q) = wide_parts(x).expect("a two-word run reads in two words");
                    let p = if g == 1 { p } else { wide_div(p, g) };
                    // At most the run's widest `X`, which fits two words.
                    let v = p * wide_div(den, q);
                    for (w, part) in words.iter_mut().zip([v as u64, (v >> 64) as u64]) {
                        *w = part;
                    }
                    (neg, v != 0)
                }
                None => {
                    let (p, q) = x.magnitudes();
                    let v = &*p / &self.content * (&self.den / &*q);
                    write_natural(&v, words);
                    (x.signum().is_lt(), v != Natural::ZERO)
                }
            };
            if nonzero && neg != negate {
                signs[j / 64] |= 1 << (j % 64);
            }
        }
        signs
    }
}

/// ORs `bits` (bit `j` for entry `at + j`) into the bitset `dst`.
fn or_bits(dst: &mut [u64], at: usize, bits: &[u64]) {
    let (w, s) = (at / 64, at % 64);
    for (j, &b) in bits.iter().enumerate() {
        dst[w + j] |= b << s;
        if s > 0 {
            if let Some(next) = dst.get_mut(w + j + 1) {
                *next |= b >> (64 - s);
            }
        }
    }
}

/// One `μ` of a [`Pending`] channel: its integers and vertex.
struct MuPlan {
    ints: Integers,
    vertex: Radicand,
}

/// What every column's writer reads.
struct Meta {
    width: usize,
    negate: bool,
    columns: Vec<Mutex<Column>>,
    ends: Vec<usize>,
    /// The entries of `μ` are `starts[μ]..starts[μ + 1]`.
    starts: Vec<usize>,
    plans: Vec<MuPlan>,
    /// Per column: every `X` of it fits two words.
    wide: Vec<bool>,
}

/// A channel's emitted columns with every `μ`'s integers, vertex and slab
/// sized at its canonical stride: the state between the descent and the one
/// per-column stage that writes the integers ([`Writer::column`]). Built in
/// `O(columns)` from the scales the descent's items reduced.
pub(crate) struct Pending {
    meta: Meta,
    m_ab: Vec<(u32, u32)>,
    mags: Vec<Vec<u64>>,
}

impl Pending {
    /// `columns` in `(μ, column)` order, `width` per `μ`, and `N_μ` per `μ`,
    /// every `X` negated when `negate`.
    pub(crate) fn new(
        columns: Vec<Column>,
        width: usize,
        norms: &[Q],
        negate: bool,
    ) -> Result<Self, LieError> {
        let mult = norms.len();
        u32::try_from(mult).map_err(|_| LieError::LabelOverflow)?;
        debug_assert_eq!(columns.len(), mult * width);
        let mut columns = columns;
        let mut ends = Vec::with_capacity(columns.len());
        let mut count = 0usize;
        for col in &columns {
            count += col.len();
            ends.push(count);
        }
        let mut tops = Vec::with_capacity(columns.len());
        let mut plans = Vec::with_capacity(mult);
        for (mu, norm) in norms.iter().enumerate() {
            let mut scale = Scale::default();
            for col in &mut columns[mu * width..(mu + 1) * width] {
                let s = std::mem::take(&mut col.scale);
                tops.push(s.wide_top());
                scale = scale.merge(s);
            }
            let ints = Integers::new(scale);
            plans.push(MuPlan {
                vertex: vertex(norm, &ints.den, &ints.content),
                ints,
            });
        }
        let wide = tops
            .iter()
            .enumerate()
            .map(|(k, top)| top.is_some_and(|t| plans[k / width].ints.fits_wide(t)))
            .collect();
        let starts: Vec<usize> = (0..=mult)
            .map(|mu| if mu == 0 { 0 } else { ends[mu * width - 1] })
            .collect();
        let mut mags = Vec::with_capacity(mult);
        for (mu, plan) in plans.iter().enumerate() {
            let words = (starts[mu + 1] - starts[mu])
                .checked_mul(plan.ints.stride)
                .ok_or(LieError::LabelOverflow)?;
            mags.push(vec![0u64; words]);
        }
        Ok(Self {
            meta: Meta {
                width,
                negate,
                columns: columns.into_iter().map(Mutex::new).collect(),
                ends,
                starts,
                plans,
                wide,
            },
            m_ab: vec![(0u32, 0u32); count],
            mags,
        })
    }

    pub(crate) fn mult(&self) -> usize {
        self.meta.plans.len()
    }

    /// The number of entries.
    pub(crate) fn len(&self) -> usize {
        self.m_ab.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.m_ab.is_empty()
    }

    /// Every `μ`'s vertex `ρ_μ`, in `μ` order.
    pub(crate) fn vertices(&self) -> impl ExactSizeIterator<Item = &Radicand> + '_ {
        self.meta.plans.iter().map(|p| &p.vertex)
    }

    /// The entries of every column: a column stage's units.
    pub(crate) fn units(&self) -> Vec<u64> {
        crate::stage::part_units(&self.meta.ends)
    }

    /// The writer of every column, each into its own ranges of the pair list
    /// and of its `μ`'s slab.
    pub(crate) fn writer(&mut self) -> Writer<'_> {
        let Self { meta, m_ab, mags } = self;
        let pairs = split_parts(m_ab, meta.ends.iter().copied());
        let mut slabs = Vec::with_capacity(meta.ends.len());
        for (mu, mag) in mags.iter_mut().enumerate() {
            let (at, stride) = (meta.starts[mu], meta.plans[mu].ints.stride);
            slabs.extend(split_parts(
                mag,
                meta.ends[mu * meta.width..(mu + 1) * meta.width]
                    .iter()
                    .map(|&end| (end - at) * stride),
            ));
        }
        Writer { meta, pairs, slabs }
    }

    /// Writes every column as one stage on `exec` (inline without a token)
    /// and assembles the coordinates.
    fn write_all(mut self, exec: Option<&mut Exec<'_>>) -> Coords {
        let units = self.units();
        let signs = {
            let writer = self.writer();
            let mut buf = StageBuf::default();
            let _w = run_items(exec, &units, &mut buf, |k| writer.column(k).into_signs());
            #[cfg(test)]
            count_section(_w);
            buf.drain().collect()
        };
        self.into_coords(signs)
    }

    /// The exact channel over `dims`, every column written on `exec`.
    pub(crate) fn integralize(self, dims: [usize; 3], exec: Option<&mut Exec<'_>>) -> ExactCgc {
        let mult = self.mult();
        ExactCgc {
            mult,
            dims,
            coords: self.write_all(exec),
        }
    }

    /// The exact channel over `dims` once every column was written, with the
    /// columns' sign words in column order ([`WrittenColumn::into_signs`]).
    pub(crate) fn into_channel(self, dims: [usize; 3], signs: Vec<Vec<u64>>) -> ExactCgc {
        let mult = self.mult();
        ExactCgc {
            mult,
            dims,
            coords: self.into_coords(signs),
        }
    }

    /// The coordinates, with the columns' sign words merged per `μ` in
    /// `O(entries / 64)`.
    fn into_coords(self, signs: Vec<Vec<u64>>) -> Coords {
        let Self { meta, m_ab, mags } = self;
        let Meta {
            width,
            ends,
            starts,
            plans,
            ..
        } = meta;
        let mut signs = signs.into_iter();
        let per_mu = plans
            .into_iter()
            .zip(mags)
            .enumerate()
            .map(|(mu, (plan, mag))| {
                let len = starts[mu + 1] - starts[mu];
                let mut neg = vec![0u64; len.div_ceil(64)];
                for k in mu * width..(mu + 1) * width {
                    let at = (if k == 0 { 0 } else { ends[k - 1] }) - starts[mu];
                    if let Some(s) = signs.next() {
                        or_bits(&mut neg, at, &s);
                    }
                }
                MuCoords {
                    bits: plan.ints.bits,
                    x: SignedInts::from_raw(len, plan.ints.stride, neg, mag),
                    vertex: plan.vertex,
                }
            })
            .collect();
        Coords {
            m_ab: m_ab.into_boxed_slice(),
            ends: ends.into_boxed_slice(),
            per_mu,
        }
    }
}

/// Writes the columns of a [`Pending`] channel; every column takes its
/// ranges once, so the writer is shared by a stage's participants.
pub(crate) struct Writer<'p> {
    meta: &'p Meta,
    pairs: Vec<Mutex<&'p mut [(u32, u32)]>>,
    slabs: Vec<Mutex<&'p mut [u64]>>,
}

impl<'p> Writer<'p> {
    /// The exclusive end of every column.
    pub(crate) fn ends(&self) -> &[usize] {
        &self.meta.ends
    }

    /// Writes column `k`'s pairs and integers and releases its rationals.
    pub(crate) fn column(&self, k: usize) -> WrittenColumn<'p> {
        let meta = self.meta;
        let col = std::mem::take(
            &mut *meta.columns[k]
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        );
        let pairs = take_part(&self.pairs, k);
        for (slot, e) in pairs.iter_mut().zip(&col.entries) {
            *slot = (e.0, e.1);
        }
        let mu = k / meta.width;
        let plan = &meta.plans[mu];
        let mag = take_part(&self.slabs, k);
        let signs = plan.ints.write(
            col.len(),
            col.entries.iter().map(|e| &e.2),
            meta.negate,
            meta.wide[k],
            mag,
        );
        WrittenColumn {
            mu,
            mc: k % meta.width,
            pairs,
            mag,
            stride: plan.ints.stride,
            signs,
            vertex: &plan.vertex,
        }
    }
}

/// A column just written by [`Writer::column`], with its sign words.
pub(crate) struct WrittenColumn<'p> {
    mu: usize,
    mc: usize,
    pairs: &'p [(u32, u32)],
    mag: &'p [u64],
    stride: usize,
    signs: Vec<u64>,
    vertex: &'p Radicand,
}

impl WrittenColumn<'_> {
    /// Its coordinates, as the rounding layer reads them.
    pub(crate) fn coords(&self) -> ColumnCoords<'_> {
        ColumnCoords {
            mu: self.mu,
            mc: self.mc,
            pairs: self.pairs,
            mag: self.mag,
            stride: self.stride,
            neg: &self.signs,
            neg_at: 0,
            vertex: self.vertex,
        }
    }

    /// Its sign words, for [`Pending::into_channel`].
    pub(crate) fn into_signs(self) -> Vec<u64> {
        self.signs
    }
}

/// The exact coordinates of one column `(μ, m_c)`, read in place: entry `j`
/// is `±(magnitude j)` at pair `pairs()[j]`, with the `μ`'s vertex.
#[derive(Clone, Copy)]
pub(crate) struct ColumnCoords<'a> {
    mu: usize,
    mc: usize,
    pairs: &'a [(u32, u32)],
    mag: &'a [u64],
    stride: usize,
    neg: &'a [u64],
    neg_at: usize,
    vertex: &'a Radicand,
}

impl<'a> ColumnCoords<'a> {
    pub(crate) fn mu(&self) -> usize {
        self.mu
    }

    /// The public `m_c` of the column.
    pub(crate) fn mc(&self) -> usize {
        self.mc
    }

    /// The public `(m_a, m_b)` of every entry.
    pub(crate) fn pairs(&self) -> &'a [(u32, u32)] {
        self.pairs
    }

    /// `|X|` of entry `j` as little-endian words; high zero words allowed.
    pub(crate) fn magnitude(&self, j: usize) -> &'a [u64] {
        &self.mag[j * self.stride..(j + 1) * self.stride]
    }

    pub(crate) fn is_negative(&self, j: usize) -> bool {
        let b = self.neg_at + j;
        (self.neg[b / 64] >> (b % 64)) & 1 == 1
    }

    pub(crate) fn vertex(&self) -> &'a Radicand {
        self.vertex
    }
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
    /// `columns` in `(μ, column)` order, `width` per `μ`, and `N_μ` per `μ`,
    /// integralized in one stage over the columns on `exec` (inline without
    /// a token, with the same integers).
    fn new(
        columns: Vec<Column>,
        width: usize,
        norms: &[Q],
        negate: bool,
        exec: Option<&mut Exec<'_>>,
    ) -> Result<Self, LieError> {
        Ok(Pending::new(columns, width, norms, negate)?.write_all(exec))
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
    /// `N_μ = norms[μ]`, every `X` negated when `negate`, integralized in
    /// one stage on `exec` ([`Coords::new`]).
    #[cfg(test)]
    pub(crate) fn from_columns(
        dims: [usize; 3],
        columns: Vec<Column>,
        norms: &[Q],
        negate: bool,
        exec: Option<&mut Exec<'_>>,
    ) -> Result<Self, LieError> {
        Ok(Self {
            mult: norms.len(),
            dims,
            coords: Coords::new(columns, dims[2], norms, negate, exec)?,
        })
    }

    pub(crate) fn mult(&self) -> usize {
        self.mult
    }

    /// The number of exactly nonzero entries.
    pub(crate) fn len(&self) -> usize {
        self.coords.m_ab.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.coords.m_ab.is_empty()
    }

    /// The exclusive end of every column `μ·d_c + m_c`.
    pub(crate) fn column_ends(&self) -> &[usize] {
        &self.coords.ends
    }

    /// Every `μ`'s vertex `ρ_μ`, in `μ` order: the values
    /// `column_coords(k).vertex()` returns.
    pub(crate) fn vertices(&self) -> impl ExactSizeIterator<Item = &Radicand> + '_ {
        self.coords.per_mu.iter().map(|m| &m.vertex)
    }

    /// The coordinates of column `k = μ·d_c + m_c`, read in place.
    pub(crate) fn column_coords(&self, k: usize) -> ColumnCoords<'_> {
        let dc = self.dims[2];
        let mu = k / dc;
        let base = self.mu_range(mu).start;
        let range = self.coords.column(k);
        let MuCoords { x, vertex, .. } = &self.coords.per_mu[mu];
        ColumnCoords {
            mu,
            mc: k % dc,
            pairs: &self.coords.m_ab[range.clone()],
            mag: x.words(range.start - base, range.len()),
            stride: x.stride(),
            neg: x.sign_words(),
            neg_at: range.start - base,
            vertex,
        }
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
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct HighestColumn {
    mult: usize,
    /// One column per `μ`.
    coords: Coords,
}

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
            coords: Coords::new(columns, 1, norms, false, None)?,
        })
    }

    pub(crate) fn mult(&self) -> usize {
        self.mult
    }

    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(self.coords.bytes())
    }

    /// The seed column of multiplicity `mu` (engine index 0 of `c`), read in
    /// place: its public `(m_a, m_b)` pairs, `|X|` words, signs and vertex
    /// `ρ_μ`. Its `mc()` is 0, the column's index within the seed set.
    pub(crate) fn column_coords(&self, mu: usize) -> ColumnCoords<'_> {
        let range = self.coords.column(mu);
        let MuCoords { x, vertex, .. } = &self.coords.per_mu[mu];
        ColumnCoords {
            mu,
            mc: 0,
            pairs: &self.coords.m_ab[range.clone()],
            mag: x.words(0, range.len()),
            stride: x.stride(),
            neg: x.sign_words(),
            neg_at: 0,
            vertex,
        }
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
/// reduced `n_m` (read by the rounding layer), and the integral `ν_m` with
/// `n_m = ν_m g/Δ` for the whole-table denominator lcm `Δ` and the content
/// `g` of the scaled norms (read by exact F and R).
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NormTable {
    /// The authority: every other field but the top is derived from it by
    /// [`NormTable::from_parts`].
    pub(crate) norm: Box<[Q]>,
    nu: SignedInts,
    den: Natural,
    content: Natural,
    /// The public index of engine index 0, the highest weight.
    pub(crate) top: u32,
}

impl NormTable {
    /// The table of `l`, whose `public` map is a permutation (checked).
    pub(crate) fn new(l: &Ladders) -> Result<Self, LieError> {
        let dim = l.dim();
        let mut seen = vec![false; dim];
        let mut norm = vec![Q::from(0); dim];
        for (m, &p) in l.public.iter().enumerate() {
            let p = p as usize;
            assert!(
                !std::mem::replace(&mut seen[p], true),
                "the public magnetic indices are a permutation (§3)"
            );
            norm[p] = l.norm[m].clone();
            norm[p].compact();
        }
        Self::from_parts(norm.into(), l.public[0])
    }

    /// The table of the public-indexed `norm` and `top`:
    /// `ν`, `Δ` and `g` derived as `integral(norm)`, the one path every table
    /// is built by, so `n_m = ν_m g/Δ` on every path.
    pub(crate) fn from_parts(norm: Box<[Q]>, top: u32) -> Result<Self, LieError> {
        let Integral {
            x: nu,
            den,
            content,
        } = integral(norm.iter(), false)?;
        let table = Self {
            norm,
            nu,
            den,
            content,
            top,
        };
        #[cfg(test)]
        table.assert_identity();
        Ok(table)
    }

    /// `p_m·Δ == ν_m·g·q_m` for every `m`, on every table the tests build.
    #[cfg(test)]
    fn assert_identity(&self) {
        for (m, q) in self.norm.iter().enumerate() {
            let (p, d) = q.magnitudes();
            let lhs = &*p * &self.den;
            let rhs =
                crate::exact::radical::words_natural(self.nu.magnitude(m)) * &self.content * &*d;
            assert!(lhs == rhs && !self.nu.is_negative(m), "n_{m} = ν_{m} g/Δ");
        }
    }

    /// `ν_m` with `n_m = ν_m g/Δ`, every entry at least 1.
    pub(crate) fn nu(&self) -> &SignedInts {
        &self.nu
    }

    /// `Δ`, the lcm of the norms' denominators.
    pub(crate) fn den(&self) -> &Natural {
        &self.den
    }

    /// `g`, the content of the norms times `Δ`.
    pub(crate) fn content(&self) -> &Natural {
        &self.content
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
        n.checked_add(4)
    }

    /// `u32 dim`, `dim × (n_m numerator, denominator)`, `u32 top`. The
    /// derived `ν`, `Δ` and `g` are not stored.
    #[cfg(feature = "disk-cache")]
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        put_u32(out, self.norm.len());
        for q in self.norm.iter() {
            let (p, d) = q.magnitudes();
            encode_natural(&p, out);
            encode_natural(&d, out);
        }
        out.extend_from_slice(&self.top.to_le_bytes());
    }

    /// The table of `payload` for an irrep of dimension `dim`, checked before
    /// any arithmetic or indexing: every norm positive, `top < dim`, and no
    /// trailing bytes; then built by
    /// [`Self::from_parts`]. A truncated table fails.
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
        let top = read_u32(payload, at)?;
        if top as usize >= dim || *at != payload.len() {
            return None;
        }
        Self::from_parts(norm.into(), top).ok()
    }
}

/// Counts an integralization stage of this thread that ran a section.
#[cfg(test)]
fn count_section(width: std::num::NonZeroUsize) {
    if width.get() > 1 {
        INTEGRAL_SECTIONS.with(|s| s.set(s.get() + 1));
    }
}

#[cfg(test)]
thread_local! {
    /// Integralization-only column stages ([`Pending::integralize`]) on this
    /// thread that ran a section of width two or more.
    pub(crate) static INTEGRAL_SECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests;
