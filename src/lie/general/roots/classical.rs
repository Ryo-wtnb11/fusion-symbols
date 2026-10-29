//! Exact signed-coordinate primitives for B/C/D.
use super::*;
use crate::lie::bcd::Series;
pub(crate) fn dynkin_to_two_partition(series: Series, a: &[i64]) -> Result<Vec<i64>, LieError> {
    let r = a.len();
    let mut lam = vec![0i64; r];
    match series {
        Series::B => {
            // 2λ_r = a_r, 2λ_i = 2λ_{i+1} + 2a_i.
            lam[r - 1] = a[r - 1];
            for i in (0..r - 1).rev() {
                lam[i] = a[i]
                    .checked_mul(2)
                    .and_then(|x| lam[i + 1].checked_add(x))
                    .ok_or(LieError::LabelOverflow)?;
            }
        }
        Series::C => {
            // 2λ_r = 2a_r, 2λ_i = 2λ_{i+1} + 2a_i.
            lam[r - 1] = a[r - 1].checked_mul(2).ok_or(LieError::LabelOverflow)?;
            for i in (0..r - 1).rev() {
                lam[i] = a[i]
                    .checked_mul(2)
                    .and_then(|x| lam[i + 1].checked_add(x))
                    .ok_or(LieError::LabelOverflow)?;
            }
        }
        Series::D => {
            // 2λ_{r-1} = a_{r-1}+a_r, 2λ_r = a_r-a_{r-1}, 2λ_i = 2λ_{i+1}+2a_i.
            lam[r - 1] = a[r - 1] - a[r - 2];
            lam[r - 2] = a[r - 1]
                .checked_add(a[r - 2])
                .ok_or(LieError::LabelOverflow)?;
            for i in (0..r - 2).rev() {
                lam[i] = a[i]
                    .checked_mul(2)
                    .and_then(|x| lam[i + 1].checked_add(x))
                    .ok_or(LieError::LabelOverflow)?;
            }
        }
    }
    Ok(lam)
}

/// Dynkin labels `a` from a doubled partition `2λ` (inverse of
/// [`dynkin_to_two_partition`]).
pub(crate) fn two_partition_to_dynkin(series: Series, two_lam: &[i64]) -> Vec<i64> {
    let r = two_lam.len();
    let mut a = vec![0i64; r];
    for i in 0..r - 1 {
        a[i] = half_sum(two_lam[i], -two_lam[i + 1]);
    }
    match series {
        Series::B => a[r - 1] = two_lam[r - 1],
        Series::C => a[r - 1] = two_lam[r - 1] / 2,
        Series::D => {
            a[r - 2] = half_sum(two_lam[r - 2], -two_lam[r - 1]);
            a[r - 1] = half_sum(two_lam[r - 2], two_lam[r - 1]);
        }
    }
    a
}

// ---- root system in the ε-basis ------------------------------------------

pub(crate) fn half_sum(a: i64, b: i64) -> i64 {
    i64::try_from((i128::from(a) + i128::from(b)) / 2)
        .expect("the average of two i64 coordinates fits i64")
}
pub(crate) fn rho_coordinate(series: Series, remaining: usize) -> Result<i64, LieError> {
    let tail = match series {
        Series::B => 1,
        Series::C => 2,
        Series::D => 0,
    };
    i64::try_from(remaining)
        .ok()
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_add(tail))
        .ok_or(LieError::LabelOverflow)
}

/// `2ρ` (twice the Weyl vector, integer-valued) in the ε-basis:
/// `B: 2r-2i+1`, `C: 2r-2i+2`, `D: 2r-2i` (`i = 1..r`).
pub(crate) fn two_rho(series: Series, r: usize) -> Vec<i64> {
    (0..r)
        .map(|i| {
            rho_coordinate(series, r - i - 1)
                .expect("rank comes from an allocated i64 weight, so 2*rank+2 fits i64")
        })
        .collect()
}

/// The positive roots of the series in the ε-basis, as integer coefficient
/// vectors (Fulton–Harris §18). Common to all: `e_i − e_j`, `e_i + e_j`
/// (`i<j`); plus `e_i` for `B`, `2e_i` for `C`, none for `D`.
pub(crate) fn positive_roots(series: Series, r: usize) -> Vec<Vec<i64>> {
    let mut roots = Vec::new();
    for i in 0..r {
        for j in i + 1..r {
            let mut minus = vec![0i64; r];
            minus[i] = 1;
            minus[j] = -1;
            roots.push(minus);
            let mut plus = vec![0i64; r];
            plus[i] = 1;
            plus[j] = 1;
            roots.push(plus);
        }
    }
    match series {
        Series::B => {
            for i in 0..r {
                let mut e = vec![0i64; r];
                e[i] = 1;
                roots.push(e);
            }
        }
        Series::C => {
            for i in 0..r {
                let mut e = vec![0i64; r];
                e[i] = 2;
                roots.push(e);
            }
        }
        Series::D => {}
    }
    roots
}

// ---- Weyl-group dominant conjugation -------------------------------------

/// Sort a vector descending, returning the sorted vector and the sign of the
/// sorting permutation (`+1`/`-1`). For distinct entries this is `sgn(perm)`;
/// callers that need the sign guarantee distinctness first.
pub(crate) fn sort_desc_with_parity(v: &[i64]) -> (Vec<i64>, i32) {
    let n = v.len();
    let mut odd = false;
    for i in 0..n {
        for j in i + 1..n {
            if v[i] < v[j] {
                odd = !odd;
            }
        }
    }
    let mut out = v.to_vec();
    out.sort_unstable_by(|a, b| b.cmp(a));
    (out, if odd { -1 } else { 1 })
}

/// Signed permutations give the dominant image, determinant parity and wall flag.
/// B/C allow every sign change; D allows even changes, and one zero frees parity.
pub(crate) fn dominant_conjugate(series: Series, v: &[i64]) -> (Vec<i64>, bool, bool) {
    let negative = v.iter().filter(|&&x| x < 0).count();
    let abs = v.iter().map(|x| x.abs()).collect::<Vec<_>>();
    let (mut sorted, sign) = sort_desc_with_parity(&abs);
    let wall =
        sorted.windows(2).any(|x| x[0] == x[1]) || (series != Series::D && sorted.contains(&0));
    let odd = (sign < 0) ^ (series != Series::D && negative % 2 == 1);
    if series == Series::D && negative % 2 == 1 {
        let last = sorted.len() - 1;
        sorted[last] = -sorted[last];
    }
    (sorted, odd, wall)
}

// ---- classical dominance enumeration ------------------------------------

/// Coefficients `c` with `d = Σ cᵢ αᵢ` (simple roots, ε-basis) from the
/// **doubled** difference `dd = 2d`, or `None` if `d` is not a non-negative
/// integer combination — which now also rejects the half-integer `d` of a
/// tensor/spinor weight pair (they lie in different classes of `P/Q`, so
/// neither is a weight of the other's irrep). Closed forms from the simple-root
/// structure (Fulton–Harris §18):
/// - `B`/`C`/`D` share `cⱼ = Σ_{i≤j} dᵢ` for the `e_i − e_{i+1}` part;
///   the short/spin root closes the last one(s).
pub(crate) fn simple_root_coeffs(series: Series, dd: &[i64]) -> Result<Option<Vec<i64>>, LieError> {
    let r = dd.len();
    // Prefix sums = coefficients of the e_i - e_{i+1} chain. Each must be an
    // integer: at the doubled scale that is "each prefix of `dd` is even",
    // which holds iff every dᵢ is an integer.
    let mut c = vec![0i64; r];
    let mut acc = 0i64;
    for i in 0..r {
        acc = add_weight(acc, dd[i])?;
        if acc % 2 != 0 {
            return Ok(None);
        }
        c[i] = acc / 2; // provisional; last one(s) fixed per series below
    }
    let total = acc; // = 2·Σ dᵢ
    match series {
        Series::B => {
            // α_r = e_r, c_r = Σ d_i = total/2 (already c[r-1]).
        }
        Series::C => {
            // α_r = 2e_r, c_r = (Σ d_i)/2 = total/4.
            if total % 4 != 0 {
                return Ok(None);
            }
            c[r - 1] = total / 4;
        }
        Series::D => {
            // α_r = e_{r-1}+e_r: c_r = (Σ d_i)/2, c_{r-1} = c_r - d_r.
            if total % 4 != 0 {
                return Ok(None);
            }
            c[r - 1] = total / 4;
            c[r - 2] = (total / 4)
                .checked_sub(dd[r - 1] / 2)
                .ok_or(LieError::LabelOverflow)?;
        }
    }
    if c.iter().all(|&x| x >= 0) {
        Ok(Some(c))
    } else {
        Ok(None)
    }
}

/// All dominant weights `2μ` with `μ ≤ λ` (dominance) and `λ − μ` in the root
/// lattice, by enumerating a bounded box of dominant weights and filtering
/// with [`simple_root_coeffs`]. Doubled throughout; the enumeration steps by
/// `2`, so it stays inside `λ`'s class of `P/Q` (integer or half-integer
/// coordinates, never a mix).
pub(crate) fn enumerate_dominant_below(
    series: Series,
    lambda: &[i64],
) -> Result<Vec<Vec<i64>>, LieError> {
    let r = lambda.len();
    let hi = lambda[0]; // μ ≤ λ ⇒ μ₁ ≤ λ₁; all |μ_i| ≤ λ₁.
    let mut out = Vec::new();
    let mut cur = vec![0i64; r];
    enum_dom_rec(series, lambda, hi, 0, &mut cur, &mut out)?;
    Ok(out)
}

pub(crate) fn enum_dom_rec(
    series: Series,
    lambda: &[i64],
    hi: i64,
    pos: usize,
    cur: &mut Vec<i64>,
    out: &mut Vec<Vec<i64>>,
) -> Result<(), LieError> {
    let r = cur.len();
    if pos == r {
        if simple_root_coeffs(series, &sub(lambda, cur)?)?.is_some() {
            out.push(cur.clone());
        }
        return Ok(());
    }
    // Dominant: μ_pos ≤ μ_{pos-1} (and ≤ hi). For D the last slot also allows
    // negatives down to -μ_{r-2} (chirality); for B/C the floor is 0. Values
    // step by 2 (doubled scale), keeping μ in λ's class — for a spinor λ the
    // floor 0 is therefore never reached, the smallest value is 1.
    let upper = if pos == 0 { hi } else { cur[pos - 1] };
    let lower = if series == Series::D && pos == r - 1 {
        -cur[pos - 1]
    } else {
        0
    };
    let mut v = upper;
    while v >= lower {
        cur[pos] = v;
        enum_dom_rec(series, lambda, hi, pos + 1, cur, out)?;
        let Some(next) = v.checked_sub(2) else {
            break;
        };
        v = next;
    }
    cur[pos] = 0;
    Ok(())
}

// ---- Weyl orbit ----------------------------------------------------------

/// All distinct Weyl-group images of a dominant weight `mu` (ε-basis):
/// signed permutations for `B`/`C`, even-signed permutations for `D`.
#[cfg(test)]
pub(crate) fn weyl_orbit(series: Series, mu: &[i64]) -> Vec<Vec<i64>> {
    try_weyl_orbit(series, mu).unwrap_or_else(|e| panic!("infallible weight orbit failed: {e}"))
}
pub(crate) fn weyl_sign_count(rank: usize) -> Result<u32, LieError> {
    u32::try_from(rank)
        .ok()
        .and_then(|r| 1u32.checked_shl(r))
        .ok_or(LieError::LabelOverflow)
}
pub(crate) fn try_weyl_orbit(series: Series, mu: &[i64]) -> Result<Vec<Vec<i64>>, LieError> {
    weyl_sign_count(mu.len())?;
    let nonzero = mu.iter().filter(|&&x| x != 0).count();
    // A zero frees D parity; otherwise absolute values must retain chirality.
    let constrained = series == Series::D && nonzero == mu.len() && nonzero != 0;
    let sign_count = weyl_sign_count(nonzero - usize::from(constrained))?;
    let negative_count = mu.iter().filter(|&&x| x < 0).count() as u32;
    let mut permutation = mu
        .iter()
        .map(|x| x.checked_abs().ok_or(LieError::LabelOverflow))
        .collect::<Result<Vec<_>, _>>()?;
    permutation.sort_unstable();
    let mut orbit = Vec::new();
    loop {
        for mut signs in 0..sign_count {
            if constrained {
                // Force the last sign instead of generating and discarding half.
                signs |= ((signs.count_ones() + negative_count) % 2) << (nonzero - 1);
            }
            let mut signed = permutation.clone();
            for x in signed.iter_mut().filter(|x| **x != 0) {
                if signs & 1 != 0 {
                    *x = -*x;
                }
                signs >>= 1;
            }
            orbit.push(signed);
        }
        // Next lexicographic multiset permutation: repeated entries never branch.
        let Some(i) = permutation.windows(2).rposition(|w| w[0] < w[1]) else {
            break;
        };
        let j = permutation
            .iter()
            .rposition(|&x| x > permutation[i])
            .expect("the increasing pair supplies a successor");
        permutation.swap(i, j);
        permutation[i + 1..].reverse();
    }
    Ok(orbit)
}
