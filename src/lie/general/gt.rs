//! The Gelfand–Tsetlin basis of an `A_r` irrep in the engine's rational
//! coordinates: Molev's unnormalized vectors `ξ_Λ` (A. I. Molev,
//! *Gelfand–Tsetlin bases for classical Lie algebras*, arXiv:math/0211289,
//! Theorem 2.3 and Proposition 2.4).
//!
//! With `λ_{ki} = GtPattern::get(i, k)` (1-based) and `l_{ki} = λ_{ki} − i + 1`,
//! the generators `e_j = E_{j+1, j+2}` and `f_j = E_{j+2, j+1}` act by
//!
//! ```text
//! (2.6) E_{k,k+1} ξ_Λ = Σ_i A_i(Λ) ξ_{Λ+δ_{ki}},  A_i = −Π_{j≤k+1} (l_{ki} − l_{k+1,j}) / Π_{j≠i, j≤k} (l_{ki} − l_{kj})
//! (2.7) E_{k+1,k} ξ_Λ = Σ_i B_i(Λ) ξ_{Λ−δ_{ki}},  B_i =  Π_{j≤k−1} (l_{ki} − l_{k−1,j}) / Π_{j≠i, j≤k} (l_{ki} − l_{kj})
//! ```
//!
//! for `k = j + 1`; a term whose pattern breaks betweenness is absent. The
//! vectors are orthogonal for the contravariant form with `⟨ξ_top, ξ_top⟩ = 1`.
//! Contravariance `⟨E_{k,k+1} ξ_t, ξ_s⟩ = ⟨ξ_t, E_{k+1,k} ξ_s⟩` for
//! `s = t + δ_{ki}` gives `A_i(t) n_s = B_i(s) n_t`, which propagates every
//! norm from the highest weight with no factorials.
//!
//! Weight spaces follow the engine's depth order; inside each, patterns keep
//! their ascending enumeration (public) order. The lowering recipe takes the
//! independent candidates `f_j b_q` in `(j, q)` order and solves for each basis
//! vector exactly; it is generally not triangular.

use malachite_nz::integer::Integer;
use std::collections::{BTreeMap, HashMap};

use super::module::{checked_weyl, DepthOrder, IrrepModule, ModuleError, WeightSpace, Q};
use super::roots::CartanData;
use crate::lie::{GtPattern, LieError};

/// `L(λ)` of `A_r` in the basis `ξ_Λ`, with the public (pattern) index of each
/// engine index. The dimension is checked before the patterns are enumerated,
/// once; their weight counts give the weight order.
pub(super) fn module(
    data: &CartanData,
    lambda: &[i64],
) -> Result<(IrrepModule, Vec<u32>), LieError> {
    let weyl = checked_weyl(data, lambda)?;
    if let [dj] = *lambda {
        return Ok(rank_one(dj, weyl));
    }
    from_enumeration(data, lambda, weyl)
}

/// The module from its enumerated patterns.
fn from_enumeration(
    data: &CartanData,
    lambda: &[i64],
    weyl: usize,
) -> Result<(IrrepModule, Vec<u32>), LieError> {
    let patterns = data.gt_patterns(lambda)?;
    let weights = patterns
        .iter()
        .map(|p| data.gt_weight(p))
        .collect::<Result<Vec<_>, _>>()?;
    let mut mult = BTreeMap::new();
    for mu in &weights {
        *mult.entry(mu.clone()).or_insert(0u64) += 1;
    }
    let order = DepthOrder::new(data, lambda, mult, weyl);
    from_patterns(data, order, &patterns, weights)
}

/// The same module for `A_1`, `λ = (dj)`, without enumerating patterns: the
/// pattern `(dj, 0; dj − t)` is the weight space at depth `t`, of weight
/// `dj − 2t` and public index `dj − t`. Its only lowering candidate is
/// `f ξ_{t−1}`, and (2.7) is `B = 1`, so `ξ_t = f ξ_{t−1}` (recipe 1) and
/// contravariance gives `n_t = t (dj − t + 1) n_{t−1}`, `n_0 = 1`. This is the
/// data `from_patterns` builds, without its maps and eliminations.
fn rank_one(dj: i64, weyl: usize) -> (IrrepModule, Vec<u32>) {
    let one = Q::from(1);
    let mut norm = Q::from(1);
    let mut weights = Vec::with_capacity(weyl);
    let mut index = HashMap::with_capacity(weyl);
    for t in 0..=dj {
        if t > 0 {
            norm *= Q::from(t * (dj - t + 1));
        }
        let first = t == 0;
        let space = WeightSpace {
            mu: vec![dj - 2 * t],
            k: vec![t],
            offset: t as usize,
            norms: vec![norm.clone()],
            kept: if first { Vec::new() } else { vec![(0, 0)] },
            recipe: vec![if first { Vec::new() } else { vec![one.clone()] }],
            lower: vec![(!first).then(|| (t as usize - 1, vec![vec![one.clone()]]))],
        };
        index.insert(space.mu.clone(), t as usize);
        weights.push(space);
    }
    let public = (0..=dj).rev().map(|x| x as u32).collect();
    (IrrepModule::assembled(weights, index, weyl), public)
}

fn l(p: &GtPattern, k: usize, i: usize) -> Result<i64, LieError> {
    // i ≤ k ≤ n is far below i64::MAX.
    p.get(i, k)
        .checked_sub(i as i64 - 1)
        .ok_or(LieError::LabelOverflow)
}

/// `sign · Π_{j ≤ row} (l_{ki} − l_{row,j}) / Π_{j≠i, j≤k} (l_{ki} − l_{kj})`;
/// `None` for a zero denominator.
fn molev(p: &GtPattern, k: usize, i: usize, row: usize, sign: i64) -> Result<Option<Q>, LieError> {
    let lki = l(p, k, i)?;
    let diff = |r: usize, j: usize| -> Result<Integer, LieError> {
        Ok(Integer::from(
            lki.checked_sub(l(p, r, j)?)
                .ok_or(LieError::LabelOverflow)?,
        ))
    };
    let mut num = Integer::from(sign);
    for j in 1..=row {
        num *= diff(row, j)?;
    }
    let mut den = Integer::from(1);
    for j in (1..=k).filter(|&j| j != i) {
        den *= diff(k, j)?;
    }
    Ok((den != 0).then(|| Q::from(malachite_q::Rational::from_integers(num, den))))
}

/// `B_i(Λ)` of (2.7).
fn lower_coefficient(p: &GtPattern, k: usize, i: usize) -> Result<Option<Q>, LieError> {
    molev(p, k, i, k - 1, 1)
}

/// `A_i(Λ)` of (2.6).
fn raise_coefficient(p: &GtPattern, k: usize, i: usize) -> Result<Option<Q>, LieError> {
    molev(p, k, i, k + 1, -1)
}

/// The index of `p` with `λ_{ki}` shifted by `delta`, if that is a pattern.
fn shifted(
    lookup: &HashMap<&[i64], usize>,
    scratch: &mut GtPattern,
    p: &GtPattern,
    k: usize,
    i: usize,
    delta: i64,
) -> Result<Option<usize>, LieError> {
    scratch.data.copy_from_slice(p.data());
    let v = p
        .get(i, k)
        .checked_add(delta)
        .ok_or(LieError::LabelOverflow)?;
    scratch.set(i, k, v);
    Ok(lookup.get(scratch.data()).copied())
}

/// The module on `patterns`, of weights `pattern_weights`, laid out by `order`.
fn from_patterns(
    data: &CartanData,
    order: DepthOrder,
    patterns: &[GtPattern],
    pattern_weights: Vec<Vec<i64>>,
) -> Result<(IrrepModule, Vec<u32>), LieError> {
    let rank = order.spaces[0].1.len();
    let invalid = |weight: &[i64]| -> LieError {
        ModuleError::InvalidLadder {
            weight: weight.to_vec(),
        }
        .into()
    };

    // Each weight space's patterns, in public order; `local[g] = (w, position)`.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); order.spaces.len()];
    let mut local = Vec::with_capacity(patterns.len());
    for (g, mu) in pattern_weights.into_iter().enumerate() {
        let Some(&w) = order.index.get(&mu) else {
            return Err(ModuleError::RankMismatch {
                weight: mu,
                rank: 1,
                multiplicity: 0,
            }
            .into());
        };
        local.push((w, members[w].len()));
        members[w].push(g);
    }
    for (w, (_, mu)) in order.spaces.iter().enumerate() {
        let multiplicity = order.mult.get(mu).copied().unwrap_or(0);
        if u64::try_from(members[w].len()).ok() != Some(multiplicity) {
            return Err(ModuleError::RankMismatch {
                weight: mu.clone(),
                rank: members[w].len(),
                multiplicity,
            }
            .into());
        }
    }
    let lookup: HashMap<&[i64], usize> = patterns
        .iter()
        .enumerate()
        .map(|(g, p)| (p.data(), g))
        .collect();
    let Some(first) = patterns.first() else {
        return Err(ModuleError::DimensionMismatch {
            built: 0,
            weyl: order.weyl,
        }
        .into());
    };
    let mut scratch = first.clone();

    let mut weights = Vec::with_capacity(order.spaces.len());
    let mut offset = 0;
    for (w, (k, mu)) in order.spaces.iter().enumerate() {
        let mut lower = Vec::with_capacity(rank);
        for j in 0..rank {
            let nu = (0..rank)
                .map(|i| mu[i].checked_add(data.c(j, i)))
                .collect::<Option<Vec<i64>>>()
                .ok_or(LieError::LabelOverflow)?;
            lower.push(order.index.get(&nu).map(|&src| {
                let zero = vec![Q::from(0); members[src].len()];
                (src, vec![zero; members[w].len()])
            }));
        }
        weights.push(WeightSpace {
            mu: mu.clone(),
            k: k.clone(),
            offset,
            norms: vec![Q::from(0); members[w].len()],
            kept: Vec::new(),
            recipe: vec![Vec::new(); usize::from(w == 0)],
            lower,
        });
        offset += members[w].len();
    }

    // Norms, highest weight first: every raised neighbour is shallower.
    weights[0].norms[0] = Q::from(1);
    for w in 1..weights.len() {
        for pos in 0..members[w].len() {
            let t = &patterns[members[w][pos]];
            let mut norm = None;
            'raise: for k in 1..=rank {
                for i in 1..=k {
                    if let Some(s) = shifted(&lookup, &mut scratch, t, k, i, 1)? {
                        let a = raise_coefficient(t, k, i)?;
                        let b = lower_coefficient(&patterns[s], k, i)?;
                        let (Some(a), Some(b)) = (a, b) else {
                            return Err(invalid(&weights[w].mu));
                        };
                        if b.is_zero() {
                            return Err(invalid(&weights[w].mu));
                        }
                        let (ws, ps) = local[s];
                        norm = Some(a * &weights[ws].norms[ps] / &b);
                        break 'raise;
                    }
                }
            }
            match norm {
                Some(n) if n.signum().is_gt() => weights[w].norms[pos] = n,
                _ => {
                    return Err(ModuleError::IndefiniteForm {
                        weight: weights[w].mu.clone(),
                    }
                    .into())
                }
            }
        }
    }

    // f_j = E_{j+2, j+1} by (2.7) with k = j + 1.
    for (g, p) in patterns.iter().enumerate() {
        let (ws, bs) = local[g];
        for j in 0..rank {
            let k = j + 1;
            for i in 1..=k {
                let Some(t) = shifted(&lookup, &mut scratch, p, k, i, -1)? else {
                    continue;
                };
                let Some(x) = lower_coefficient(p, k, i)? else {
                    return Err(invalid(&weights[ws].mu));
                };
                if x.is_zero() {
                    continue;
                }
                let (wt, at) = local[t];
                match &mut weights[wt].lower[j] {
                    Some((src, f)) if *src == ws => f[at][bs] = x,
                    _ => return Err(invalid(&weights[wt].mu)),
                }
            }
        }
    }

    for space in weights.iter_mut().skip(1) {
        solve_recipe(space, &members)?;
    }
    let public = members
        .iter()
        .flatten()
        .map(|&g| u32::try_from(g).map_err(|_| LieError::LabelOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((IrrepModule::from_spaces(weights, order)?, public))
}

/// Keeps the independent candidates `c_t = f_j b_q` in `(j, q)` order and
/// writes `b_p = Σ_t recipe[p][t] c_t` by exact Gauss–Jordan elimination of
/// their coordinate vectors, tracking each row's combination of candidates.
fn solve_recipe(space: &mut WeightSpace, members: &[Vec<usize>]) -> Result<(), LieError> {
    let m = space.norms.len();
    let mut rows: Vec<(usize, Vec<Q>, Vec<Q>)> = Vec::with_capacity(m);
    let mut kept = Vec::with_capacity(m);
    'candidates: for (j, slot) in space.lower.iter().enumerate() {
        let Some((src, f)) = slot else { continue };
        for q in 0..members[*src].len() {
            let mut v: Vec<Q> = f.iter().map(|row| row[q].clone()).collect();
            let mut combo = vec![Q::from(0); m];
            combo[kept.len()] = Q::from(1);
            for (p, row, rc) in &rows {
                if !v[*p].is_zero() {
                    let s = v[*p].clone();
                    axpy(&mut v, &s, row);
                    axpy(&mut combo, &s, rc);
                }
            }
            let Some(p) = v.iter().position(|x| !x.is_zero()) else {
                continue;
            };
            let inv = Q::from(1) / &v[p];
            for x in v.iter_mut().chain(combo.iter_mut()) {
                *x *= &inv;
            }
            for (_, row, rc) in &mut rows {
                if !row[p].is_zero() {
                    let s = row[p].clone();
                    axpy(row, &s, &v);
                    axpy(rc, &s, &combo);
                }
            }
            rows.push((p, v, combo));
            kept.push((j, q));
            if kept.len() == m {
                break 'candidates;
            }
        }
    }
    if kept.len() != m {
        return Err(ModuleError::RankMismatch {
            weight: space.mu.clone(),
            rank: kept.len(),
            multiplicity: m as u64,
        }
        .into());
    }
    // Every row is now a unit vector e_p, so its combination is b_p's recipe.
    rows.sort_unstable_by_key(|(p, ..)| *p);
    space.recipe = rows.into_iter().map(|(_, _, combo)| combo).collect();
    space.kept = kept;
    Ok(())
}

/// `x −= s · y` over the nonzero entries of `y`.
fn axpy(x: &mut [Q], s: &Q, y: &[Q]) {
    for (x, y) in x.iter_mut().zip(y) {
        if !y.is_zero() {
            x.sub_mul(s, y);
        }
    }
}

#[cfg(test)]
mod tests;
