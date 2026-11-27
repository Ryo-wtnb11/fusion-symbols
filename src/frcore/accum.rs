//! Exact integer sums of the F and R contractions (`docs/gauge_cartan.md`
//! §9): a plan chosen from bounds computed before the loop ([`choose`]), the
//! join cells ([`JoinAcc`]) and the entry sums ([`SumAcc`]) on the stack-width
//! path, and malachite integers ([`Plan::Big`]) when a bound fails.
//!
//! Every product of two little-endian operands goes into lazy `u128` columns
//! split by limb (`exact::limbs::mac_into`): a `64 × 64` product adds its low
//! word to one column and its high word to the next, so a column receives
//! summands below `2^64`, and a sign bank of columns stays below `2^128` while
//! the term count times the summands per term is below `2^64`. That
//! inequality is all the fixed path relies on; there is no rounding and no
//! error to bound.
// Read by the exact F and R kernels; until then only tests.
#![cfg_attr(not(test), allow(dead_code))]

use malachite_base::num::arithmetic::traits::UnsignedAbs;
use malachite_nz::integer::Integer;
use malachite_nz::natural::Natural;

use crate::exact::limbs::{mac_into, settle, trimmed_words};
use crate::exact::radical::{natural_words, words_natural, SignedInts};

/// A signed integer read in place: its sign and little-endian magnitude
/// words, high zero words allowed. A zero magnitude carries no meaning in
/// `neg`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Signed<'a> {
    pub(crate) neg: bool,
    pub(crate) mag: &'a [u64],
}

/// One operand side of a sum's products: the join of two legs, whose cell
/// receives at most `terms` products of `bits`-wide entries, or one leg read
/// directly.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Side {
    Joined { bits: [u64; 2], terms: u64 },
    Direct { bits: u64 },
}

/// The bounds of one block's sums, computed before the loop: the two sides
/// of every product, the bits of the weight `w` multiplied into the narrower
/// side, and an upper bound on the products summed into one entry.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bounds {
    pub(crate) sides: [Side; 2],
    pub(crate) w_bits: u64,
    pub(crate) terms: u64,
}

/// How the join cells accumulate: one `i128` per cell, or two sign banks of
/// `2J` lazy columns per cell for operands of at most `J` words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Join {
    Word,
    Wide(usize),
}

/// The arithmetic of one block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Plan {
    /// Joins as `join`; every entry sums products of operands of at most
    /// `limbs` words in two sign banks of `2·limbs` columns; `w` multiplies
    /// the left side when `w_on_left`.
    Fixed {
        join: Join,
        limbs: usize,
        w_on_left: bool,
    },
    /// malachite integers throughout: allocates per term.
    Big,
}

/// The operand widths, in words, of the fixed final sum.
const SUM_LIMBS: [u64; 4] = [1, 4, 8, 16];

/// The plan of a block with bounds `b`; [`Plan::Big`] whenever a check fails
/// or a bound computation overflows.
///
/// 1. A joined side's cell sums at most `terms` products, so every partial
///    sum of it and each sign bank is below `2^width` with `width = bits[0] +
///    bits[1] + bit_len(terms)`. [`Join::Word`] needs every leg within 63 bits
///    and every joined width within 127, so an `i128` cell never overflows.
/// 2. Otherwise [`Join::Wide`] with `J` the words of the widest leg, rounded
///    up to 1, 2 or 4 (legs of more than 256 bits are `Big`), and
///    `terms·2J < 2^64` per joined side: one product adds at most `2J`
///    summands below `2^64` to a column.
/// 3. The final operands: `w` goes on the narrower side, so `op_l = w_bits +
///    min(widths)` and `op_r = max(widths)`; `L` is the smallest of 1, 4, 8,
///    16 with `64L ≥ max(op_l, op_r)`.
/// 4. `terms·2L < 2^64` for the entry sums.
pub(crate) fn choose(b: &Bounds) -> Plan {
    choose_checked(b).unwrap_or(Plan::Big)
}

fn bit_len(x: u64) -> u64 {
    u64::from(u64::BITS - x.leading_zeros())
}

fn choose_checked(b: &Bounds) -> Option<Plan> {
    let mut widths = [0u64; 2];
    let (mut leg, mut joined) = (0u64, 0u64);
    for (w, side) in widths.iter_mut().zip(&b.sides) {
        *w = match *side {
            Side::Joined { bits, terms } => {
                let width = bits[0].checked_add(bits[1])?.checked_add(bit_len(terms))?;
                leg = leg.max(bits[0]).max(bits[1]);
                joined = joined.max(width);
                width
            }
            Side::Direct { bits } => bits,
        };
    }
    let join = if leg <= 63 && joined <= 127 {
        Join::Word
    } else {
        let j: u64 = match leg.div_ceil(64) {
            0 | 1 => 1,
            2 => 2,
            3 | 4 => 4,
            _ => return None,
        };
        for side in &b.sides {
            if let Side::Joined { terms, .. } = *side {
                terms.checked_mul(2 * j)?;
            }
        }
        Join::Wide(usize::try_from(j).ok()?)
    };
    let [l, r] = widths;
    let op_l = b.w_bits.checked_add(l.min(r))?;
    let need = op_l.max(l.max(r));
    let limbs = SUM_LIMBS.into_iter().find(|&n| 64 * n >= need)?;
    b.terms.checked_mul(2 * limbs)?;
    Some(Plan::Fixed {
        join,
        limbs: usize::try_from(limbs).ok()?,
        w_on_left: l <= r,
    })
}

/// `|p − q|` into `p` and whether `q > p`, for little-endian magnitudes of
/// one length.
fn sub_banks(p: &mut [u64], q: &[u64]) -> bool {
    let q_larger = p.iter().rev().cmp(q.iter().rev()) == std::cmp::Ordering::Less;
    let mut borrow = false;
    for (x, &y) in p.iter_mut().zip(q) {
        let (big, small) = if q_larger { (y, *x) } else { (*x, y) };
        let (d, b1) = big.overflowing_sub(small);
        let (d, b2) = d.overflowing_sub(u64::from(borrow));
        *x = d;
        borrow = b1 || b2;
    }
    q_larger
}

/// A signed integer as a malachite integer.
fn integer(x: Signed<'_>) -> Integer {
    let m = Integer::from(words_natural(trimmed_words(x.mag)));
    if x.neg {
        -m
    } else {
        m
    }
}

/// The join cells of one block (`AB` or `CD` values per pair of
/// multiplicity indices): each receives products and is read once.
pub(crate) enum JoinAcc {
    Word(Vec<i128>),
    Wide { words: usize, cols: Vec<u128> },
    Big(Vec<Integer>),
}

impl JoinAcc {
    /// `cells` zero cells for `plan`.
    pub(crate) fn new(plan: &Plan, cells: usize) -> Self {
        match *plan {
            Plan::Fixed {
                join: Join::Word, ..
            } => Self::Word(vec![0; cells]),
            Plan::Fixed {
                join: Join::Wide(j),
                ..
            } => Self::Wide {
                words: j,
                cols: vec![0; cells * 4 * j],
            },
            Plan::Big => Self::Big(vec![Integer::from(0); cells]),
        }
    }

    /// Adds `x·y` to `cell`. On the fixed path both operands have at most
    /// the plan's join words (high zero words trimmed first), and `i64`
    /// magnitudes on [`Join::Word`].
    pub(crate) fn add(&mut self, cell: usize, x: Signed<'_>, y: Signed<'_>) {
        let (xm, ym) = (trimmed_words(x.mag), trimmed_words(y.mag));
        match self {
            Self::Word(cells) => {
                let v = |m: &[u64], neg: bool| {
                    let m = i128::from(m.first().copied().unwrap_or(0));
                    if neg {
                        -m
                    } else {
                        m
                    }
                };
                cells[cell] += v(xm, x.neg) * v(ym, y.neg);
            }
            Self::Wide { words, cols } => {
                let bank = 2 * *words;
                let at = (2 * cell + usize::from(x.neg != y.neg)) * bank;
                mac_into(xm, ym, &mut cols[at..at + bank]);
            }
            Self::Big(cells) => cells[cell] += integer(x) * integer(y),
        }
    }

    /// The value of `cell` as a sign and its magnitude words in `out`
    /// (replaced; no high zero words), clearing the cell.
    pub(crate) fn take(&mut self, cell: usize, out: &mut Vec<u64>) -> bool {
        out.clear();
        match self {
            Self::Word(cells) => {
                let v = std::mem::take(&mut cells[cell]);
                let m = v.unsigned_abs();
                out.extend([m as u64, (m >> 64) as u64]);
                trim(out);
                v < 0
            }
            Self::Wide { words, cols } => {
                let bank = 2 * *words;
                let at = 2 * cell * bank;
                let (pos, neg) = cols[at..at + 2 * bank].split_at_mut(bank);
                // A bank settles into one word more than its columns.
                let mut p = [0u64; 9];
                let mut q = [0u64; 9];
                settle(pos, &mut p[..=bank]);
                settle(neg, &mut q[..=bank]);
                let negative = sub_banks(&mut p[..=bank], &q[..=bank]);
                out.extend_from_slice(&p[..=bank]);
                trim(out);
                negative && !out.is_empty()
            }
            Self::Big(cells) => {
                let v = std::mem::replace(&mut cells[cell], Integer::from(0));
                let negative = v < 0;
                out.extend(natural_words(&v.unsigned_abs()));
                negative
            }
        }
    }
}

/// Drops the high zero words of `v`.
fn trim(v: &mut Vec<u64>) {
    let len = trimmed_words(v).len();
    v.truncate(len);
}

/// The signed sums of a block's entries.
pub(crate) enum SumAcc {
    /// Two sign banks of `2·limbs` lazy columns per entry.
    Fixed {
        limbs: usize,
        cols: Vec<u128>,
    },
    Big(Vec<Integer>),
}

impl SumAcc {
    /// `entries` zero sums for `plan`.
    pub(crate) fn new(plan: &Plan, entries: usize) -> Self {
        match *plan {
            Plan::Fixed { limbs, .. } => Self::Fixed {
                limbs,
                cols: vec![0; entries * 4 * limbs],
            },
            Plan::Big => Self::Big(vec![Integer::from(0); entries]),
        }
    }

    /// Adds `x·y` to entry `i`; on the fixed path both operands have at most
    /// the plan's words (high zero words trimmed first).
    pub(crate) fn add(&mut self, i: usize, x: Signed<'_>, y: Signed<'_>) {
        match self {
            Self::Fixed { limbs, cols } => {
                let bank = 2 * *limbs;
                let at = (2 * i + usize::from(x.neg != y.neg)) * bank;
                mac_into(
                    trimmed_words(x.mag),
                    trimmed_words(y.mag),
                    &mut cols[at..at + bank],
                );
            }
            Self::Big(sums) => sums[i] += integer(x) * integer(y),
        }
    }

    /// The sums as a canonical slab (§f.3a): each entry's banks are settled
    /// into `2·limbs + 1` words, subtracted, and written at that stride; the
    /// widest actual magnitude then fixes the stride, and the entries are
    /// moved forward to it in place. A sum that cancels to zero carries no
    /// sign.
    pub(crate) fn finish(self) -> SignedInts {
        match self {
            Self::Fixed { limbs, mut cols } => {
                let bank = 2 * limbs;
                let cap = bank + 1;
                let len = cols.len() / (2 * bank);
                let mut mag = vec![0u64; len * cap];
                let mut neg = vec![0u64; len.div_ceil(64)];
                let mut q = vec![0u64; cap];
                let mut widest = 0;
                for (i, (entry, out)) in cols
                    .chunks_mut(2 * bank)
                    .zip(mag.chunks_mut(cap))
                    .enumerate()
                {
                    let (pos, negative) = entry.split_at_mut(bank);
                    settle(pos, out);
                    settle(negative, &mut q);
                    let below = sub_banks(out, &q);
                    let used = trimmed_words(out).len();
                    if below && used > 0 {
                        neg[i / 64] |= 1 << (i % 64);
                    }
                    widest = widest.max(used);
                }
                // Entry `i` moves from `i·cap` to `i·widest ≤ i·cap`: in
                // ascending order no unread word is overwritten.
                for i in 0..len {
                    mag.copy_within(i * cap..i * cap + widest, i * widest);
                }
                mag.truncate(len * widest);
                SignedInts::from_raw(len, widest, neg, mag)
            }
            Self::Big(sums) => {
                let values: Vec<(bool, Natural)> = sums
                    .into_iter()
                    .map(|v| (v < 0, v.unsigned_abs()))
                    .collect();
                SignedInts::from_naturals(&values).expect("a block's entry count fits")
            }
        }
    }
}

#[cfg(test)]
mod tests;
