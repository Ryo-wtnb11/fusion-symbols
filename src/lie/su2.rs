//! Exact SU(2) Wigner 3j, 6j and Clebsch–Gordan coefficients.
//! Category symbols and Frobenius–Schur data use [`super::LieGroup`].
//!
//! All spins are in the doubled ("twice") convention: `dj = 2j` as `u32`,
//! `dm = 2m` as `i32`. So `2` is spin 1 and `1` is spin 1/2; every label stays
//! an exact integer and every admissibility rule is integer arithmetic.
//!
//! # Exact zero for forbidden couplings
//!
//! [`wigner_3j`], [`wigner_6j`] and [`clebsch_gordan`] return the exact zero
//! value for a non-admissible label combination — never an error, never a
//! panic, at every `u32`/`i32` input — matching the reference-crate semantics.
//! Values are computed as big-rational Racah sums and carried as
//! [`SignedSqrtRational`] until a single final rounding to `f64`. A forbidden
//! coupling and an accidental zero of an admissible one both read as zero;
//! the triangle, projection-range, parity and projection-sum rules are the
//! ones documented on each function.
//!
//! See the [User Guide] for the task-oriented version of all of this.
//!
//! [User Guide]: https://github.com/Ryo-wtnb11/fusion-symbols/blob/main/docs/user-guide/README.md

use malachite_q::Rational;

use crate::cache;
pub use crate::exact::SignedSqrtRational;
use crate::primefactor::{factorial as pf_factorial, mul_factorial, sum_series, Pf};

/// Prime-factorized squared triangle coefficient
/// `Delta^2(a,b,c) = num/den` with `num = t1! t2! t3!` and `den = t4!`,
/// `t4 = (a+b+c)/2 + 1`. Callers guarantee admissibility, so every factorial
/// argument is a nonnegative integer. (`WignerSymbols.jl::Delta^2`.)
fn delta_sq_pf(a: u32, b: u32, c: u32) -> (Pf, Pf) {
    let (a, b, c) = (a as i64, b as i64, c as i64);
    let t1 = ((a + b - c) / 2) as u64;
    let t2 = ((a - b + c) / 2) as u64;
    let t3 = ((-a + b + c) / 2) as u64;
    let t4 = ((a + b + c) / 2 + 1) as u64;
    let mut num = pf_factorial(t1);
    mul_factorial(&mut num, t2);
    mul_factorial(&mut num, t3);
    (num, pf_factorial(t4))
}

/// Triangle admissibility for a doubled-spin triple `(a, b, c)`:
/// `|a-b| <= c <= a+b` and `a+b+c` even.
///
/// This is *the* triangle predicate: every admissibility rule in the crate
/// (the 3j/6j engines and the Regge canonicalizers) reduces its triangle
/// conditions to this one function.
pub(super) fn triangle_ok(a: u32, b: u32, c: u32) -> bool {
    let (a, b, c) = (a as i64, b as i64, c as i64);
    (a + b + c) % 2 == 0 && c >= (a - b).abs() && c <= a + b
}

/// Wigner 6j symbol $\{dj_1\, dj_2\, dj_3;\, dj_4\, dj_5\, dj_6\}$ (doubled spins).
///
/// Returns exact zero unless all four triangles
/// `(1,2,3), (1,5,6), (4,2,6), (4,5,3)` are admissible. Uses the Racah
/// single-sum closed form in big-rational arithmetic.
///
/// Transparently served from the process-local [`crate::cache`]: an admissible
/// label set is keyed by its canonical Regge class and the exact value is stored
/// once per class (the 6j symmetries preserve value, so no compensation is
/// needed). A label too large to key or non-admissible falls through to the
/// uncached engine, which is the single source of truth for the value.
///
/// ```
/// use fusion_symbols::lie::su2::wigner_6j;
///
/// // All spins 1 (dj = 2): {1 1 1; 1 1 1} = 1/6.
/// assert!((wigner_6j(2, 2, 2, 2, 2, 2).to_f64() - 1.0 / 6.0).abs() < 1e-14);
///
/// // A violated triangle is exact zero, not an error.
/// assert_eq!(wigner_6j(2, 2, 20, 2, 2, 2).to_f64(), 0.0);
/// ```
pub fn wigner_6j(dj1: u32, dj2: u32, dj3: u32, dj4: u32, dj5: u32, dj6: u32) -> SignedSqrtRational {
    match canonical_regge_6j(dj1, dj2, dj3, dj4, dj5, dj6) {
        Ok(key) => cache::wigner_6j(key, || wigner_6j_uncached(dj1, dj2, dj3, dj4, dj5, dj6)),
        Err(_) => wigner_6j_uncached(dj1, dj2, dj3, dj4, dj5, dj6),
    }
}

/// The uncached Racah single-sum engine behind [`wigner_6j`].
fn wigner_6j_uncached(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dj4: u32,
    dj5: u32,
    dj6: u32,
) -> SignedSqrtRational {
    if !(triangle_ok(dj1, dj2, dj3)
        && triangle_ok(dj1, dj5, dj6)
        && triangle_ok(dj4, dj2, dj6)
        && triangle_ok(dj4, dj5, dj3))
    {
        return SignedSqrtRational::zero();
    }

    // Radical part: the product of the four squared triangle coefficients,
    // assembled in exponent space. splitsquare pulls the perfect-square part
    // out into the rational prefactor `s`, leaving a square-free radicand `r`,
    // so no huge intermediate factorial is ever formed as a big integer.
    // (`WignerSymbols.jl::_wigner6j`.)
    let (n1, d1) = delta_sq_pf(dj1, dj2, dj3);
    let (n2, d2) = delta_sq_pf(dj1, dj5, dj6);
    let (n3, d3) = delta_sq_pf(dj4, dj2, dj6);
    let (n4, d4) = delta_sq_pf(dj4, dj5, dj3);
    let mut num = n1;
    num.mul_assign(&n2);
    num.mul_assign(&n3);
    num.mul_assign(&n4);
    let mut den = d1;
    den.mul_assign(&d2);
    den.mul_assign(&d3);
    den.mul_assign(&d4);
    let (mut snum, mut rnum) = num.splitsquare();
    let (mut sden, mut rden) = den.splitsquare();
    Pf::divgcd(&mut snum, &mut sden);
    Pf::divgcd(&mut rnum, &mut rden);
    let s = Rational::from_integers(snum.to_bigint(), sden.to_bigint());
    let r = Rational::from_integers(rnum.to_bigint(), rden.to_bigint());

    // Racah alternating sum over k (in halved units). Widen to i64 before
    // summing (as delta_sq_pf does): u32 addition of four doubled spins could
    // wrap for absurd labels.
    let (j1, j2, j3, j4, j5, j6) = (
        dj1 as i64, dj2 as i64, dj3 as i64, dj4 as i64, dj5 as i64, dj6 as i64,
    );
    // t1..t4 are the triangle sums; t5..t7 the "square" sums.
    let t1 = (j1 + j2 + j3) / 2;
    let t2 = (j1 + j5 + j6) / 2;
    let t3 = (j4 + j2 + j6) / 2;
    let t4 = (j4 + j5 + j3) / 2;
    let t5 = (j1 + j2 + j4 + j5) / 2;
    let t6 = (j2 + j3 + j5 + j6) / 2;
    let t7 = (j3 + j1 + j6 + j4) / 2;

    let kmin = t1.max(t2).max(t3).max(t4);
    let kmax = t5.min(t6).min(t7);

    // Each term is (-1)^k (k+1)! / [(k-t1)!...(k-t4)! (t5-k)!(t6-k)!(t7-k)!]
    // built as a prime-factorized numerator/denominator pair; sum_series
    // combines them over a common denominator. (`compute6jseries`.)
    let mut terms = Vec::with_capacity((kmax - kmin + 1).max(0) as usize);
    for k in kmin..=kmax {
        let mut nump = pf_factorial((k + 1) as u64);
        if k % 2 != 0 {
            nump = nump.neg();
        }
        let mut denp = pf_factorial((k - t1) as u64);
        mul_factorial(&mut denp, (k - t2) as u64);
        mul_factorial(&mut denp, (k - t3) as u64);
        mul_factorial(&mut denp, (k - t4) as u64);
        mul_factorial(&mut denp, (t5 - k) as u64);
        mul_factorial(&mut denp, (t6 - k) as u64);
        mul_factorial(&mut denp, (t7 - k) as u64);
        Pf::divgcd(&mut nump, &mut denp);
        terms.push((nump, denp));
    }
    let series = sum_series(terms);

    // value = (s * series) * sqrt(r); `r` is nonnegative (square-free part of a
    // factorial product), so the clamp in from_prefactor_radical never fires.
    SignedSqrtRational::from_prefactor_radical(s * series, r)
}

/// Wigner 3j symbol $(dj_1\, dj_2\, dj_3;\, dm_1\, dm_2\, dm_3)$ (doubled spins/projections).
///
/// Returns exact zero unless the labels are admissible: triangle $(1,2,3)$,
/// $|dm_i| \le dj_i$, $dj_i + dm_i$ even for each $i$, and $dm_1+dm_2+dm_3 = 0$.
/// Condon-Shortley phase, matching the standard closed form.
///
/// Transparently served from the process-local [`crate::cache`]. Unlike 6j, the
/// classical 3j symmetries relate orbit members by a sign, so the cache stores
/// one representative per canonical class and a sign compensates on both store
/// and retrieval: the stored value is the representative's, and the class
/// phase moves between it and this input's value. A label too large to key or
/// non-admissible falls through to the uncached engine.
pub fn wigner_3j(dj1: u32, dj2: u32, dj3: u32, dm1: i32, dm2: i32, dm3: i32) -> SignedSqrtRational {
    wigner_3j_wide(dj1, dj2, dj3, dm1.into(), dm2.into(), dm3.into())
}

/// [`wigner_3j`] with projections in `i64`, so that a transformed projection
/// such as the CG's `-dm3` stays exact; admissibility bounds every projection
/// of a nonzero symbol by its `u32` spin.
fn wigner_3j_wide(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dm1: i64,
    dm2: i64,
    dm3: i64,
) -> SignedSqrtRational {
    match canonical_regge_3j(dj1, dj2, dj3, dm1, dm2, dm3) {
        Ok((key, phase)) => {
            // Stored value is the representative's: value(rep) = phase.apply(value(input)).
            // Retrieval undoes it: value(input) = phase.apply(value(rep)).
            let rep = cache::wigner_3j(key, || {
                phase.apply(wigner_3j_uncached(dj1, dj2, dj3, dm1, dm2, dm3))
            });
            phase.apply(rep)
        }
        Err(_) => wigner_3j_uncached(dj1, dj2, dj3, dm1, dm2, dm3),
    }
}

/// The uncached closed-form engine behind [`wigner_3j`].
fn wigner_3j_uncached(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dm1: i64,
    dm2: i64,
    dm3: i64,
) -> SignedSqrtRational {
    if !admissible_3j(dj1, dj2, dj3, dm1, dm2, dm3) {
        return SignedSqrtRational::zero();
    }

    let (j1, j2, j3) = (dj1 as i64, dj2 as i64, dj3 as i64);
    let (m1, m2, m3) = (dm1, dm2, dm3);

    // Radical: Delta^2(j1,j2,j3) * prod_i (j_i+m_i)! (j_i-m_i)!, all halved,
    // assembled in exponent space. splitsquare separates the perfect-square
    // prefactor `s` from the square-free radicand `r`, so the big factorials
    // are never multiplied out as big integers. (`WignerSymbols.jl::_wigner3j`.)
    let (mut num, den) = delta_sq_pf(dj1, dj2, dj3);
    for (dj, dm) in [(j1, m1), (j2, m2), (j3, m3)] {
        mul_factorial(&mut num, ((dj + dm) / 2) as u64);
        mul_factorial(&mut num, ((dj - dm) / 2) as u64);
    }
    let (mut snum, mut rnum) = num.splitsquare();
    let (mut sden, mut rden) = den.splitsquare();
    Pf::divgcd(&mut snum, &mut sden);
    Pf::divgcd(&mut rnum, &mut rden);
    let s = Rational::from_integers(snum.to_bigint(), sden.to_bigint());
    let r = Rational::from_integers(rnum.to_bigint(), rden.to_bigint());

    // Alternating k-sum (halved units). Arguments (all >= 0 within range):
    //   k, (j1+j2-j3)/2 - k, (j1-m1)/2 - k, (j2+m2)/2 - k,
    //   k + (j3-j2+m1)/2, k + (j3-j1-m2)/2
    let a = (j1 + j2 - j3) / 2;
    let b = (j1 - m1) / 2;
    let c = (j2 + m2) / 2;
    let add1 = (j3 - j2 + m1) / 2;
    let add2 = (j3 - j1 - m2) / 2;

    let kmin = 0i64.max(-add1).max(-add2);
    let kmax = a.min(b).min(c);

    // Each term is (-1)^k / [k! (a-k)! (b-k)! (c-k)! (k+add1)! (k+add2)!]:
    // a prime-factorized numerator (+-1) / denominator, combined by sum_series.
    // (`compute3jseries`.)
    let mut terms = Vec::with_capacity((kmax - kmin + 1).max(0) as usize);
    for k in kmin..=kmax {
        let nump = if k % 2 == 0 {
            Pf::one()
        } else {
            Pf::one().neg()
        };
        let mut denp = pf_factorial(k as u64);
        mul_factorial(&mut denp, (a - k) as u64);
        mul_factorial(&mut denp, (b - k) as u64);
        mul_factorial(&mut denp, (c - k) as u64);
        mul_factorial(&mut denp, (k + add1) as u64);
        mul_factorial(&mut denp, (k + add2) as u64);
        terms.push((nump, denp));
    }
    let mut value = s * sum_series(terms);

    // Overall Condon-Shortley phase (-1)^((j1-j2-m3)/2) folds into the sign.
    if phase_is_negative((j1 - j2 - m3) / 2) {
        value = -value;
    }

    // `r` is the square-free part of a factorial product, hence nonnegative, so
    // the clamp in from_prefactor_radical is never exercised here.
    SignedSqrtRational::from_prefactor_radical(value, r)
}

/// Clebsch-Gordan coefficient $\langle dj_1\, dm_1, dj_2\, dm_2 \,|\, dj_3\, dm_3 \rangle$ (doubled spins).
///
/// Composed exactly from [`wigner_3j`] via the standard relation
/// $CG = (-1)^{-j_1+j_2-m_3}\, \sqrt{2 j_3 + 1}\; (j_1\, j_2\, j_3;\, m_1\, m_2\, -m_3)$
/// (multiply the radicand by `dj3+1`, adjust the sign) — no recomputation.
///
/// Argument order interleaves each factor with its projection:
/// `(dj1, dm1, dj2, dm2, dj3, dm3)`, all doubled. Returns exact zero for a
/// non-admissible coupling, including `dm1 + dm2 ≠ dm3`.
///
/// ```
/// use fusion_symbols::lie::su2::clebsch_gordan;
///
/// // <1/2 +1/2, 1/2 -1/2 | 0 0> = 1/sqrt(2).
/// let cg = clebsch_gordan(1, 1, 1, -1, 0, 0);
/// assert!((cg.to_f64() - 0.5f64.sqrt()).abs() < 1e-15);
/// ```
pub fn clebsch_gordan(
    dj1: u32,
    dm1: i32,
    dj2: u32,
    dm2: i32,
    dj3: u32,
    dm3: i32,
) -> SignedSqrtRational {
    // `-dm3` is formed in i64: negating `i32::MIN` is a valid projection, not
    // an inadmissible one.
    let w3 = wigner_3j_wide(dj1, dj2, dj3, dm1.into(), dm2.into(), -i64::from(dm3));
    if w3.sign() == 0 {
        return SignedSqrtRational::zero();
    }
    // sqrt(2 j3 + 1) folds into the radicand; the CS phase folds into the sign.
    let cg = w3.times_sqrt_int(u64::from(dj3) + 1);
    if phase_is_negative(((dj2 as i64) - (dj1 as i64) - (dm3 as i64)) / 2) {
        cg.neg_value()
    } else {
        cg
    }
}

/// SU(2) R-symbol $R^{dj_1,dj_2}_{dj_3}$ as a multiplicity-free scalar.
///
/// $(-1)^{j_1+j_2-j_3}$ on an admissible fusion triangle, exact `0.0` otherwise.
/// In doubled units the exponent is $(dj_1+dj_2-dj_3)/2$, an integer whenever the
/// triangle is admissible (the parity check guarantees it). The zero on a
/// non-admissible triple mirrors `Nsymbol == 0`, so a caller never multiplies a
/// spurious sign into a forbidden fusion channel.
///
/// (TensorKitSectors tugbK `src/irreps/su2irrep.jl:Rsymbol`: `Nsymbol(...) ||
/// return 0; iseven(sa.j+sb.j-sc.j) ? 1 : -1`.)
///
pub(super) fn r_symbol(dj1: u32, dj2: u32, dj3: u32) -> f64 {
    if !triangle_ok(dj1, dj2, dj3) {
        return 0.0;
    }
    if phase_is_negative(((dj1 as i64) + (dj2 as i64) - (dj3 as i64)) / 2) {
        -1.0
    } else {
        1.0
    }
}

/// Frobenius-Schur phase of an SU(2) irrep: $(-1)^{2j}$ as `+-1.0`.
///
/// Every SU(2) irrep is self-dual; the FS indicator is the sign that
/// distinguishes the orthogonal (integer `j`, `+1`) from the symplectic
/// (half-integer `j`, `-1`) self-duality. In doubled units `2j = dj`, so the
/// phase is simply the parity of `dj`. (TensorKitSectors self-dual convention;
/// see the generic `frobeniusschur` and `SU2Irrep` `dual(s) = s`.)
///
pub(super) fn fs_phase(dj: u32) -> f64 {
    if dj.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// Twist $\theta_j$ of an SU(2) irrep: exactly `1.0` for every `dj`.
///
/// Group braiding is the tensor flip $\tau$ on $V_j \otimes V_j$
/// (TensorKitSectors 0.3.9 `src/irreps/irreps.jl`:
/// `BraidingStyle(::Type{<:AbstractIrrep}) = Bosonic()`), and `SU2Irrep` does
/// not override `twist`, so TensorKitSectors evaluates the generic
/// `src/sectors.jl:twist_from_Rsymbol`,
/// $\theta_a = \sum_{c \in a \otimes a} (d_c / d_a)\, \operatorname{tr} R^{aa}_c$.
/// That sum is exactly one, not merely to rounding: $\tau$ acts on the
/// $c$-isotypic part as $R^{aa}_c \otimes \mathrm{id}_{V_c}$, so
/// $\sum_c d_c \operatorname{tr} R^{aa}_c = \operatorname{Tr}\tau = d_a$.
/// With [`r_symbol`] it reads $\sum_c (-1)^{2j-c}(2c+1) = 2j+1$.
///
pub(super) fn twist(dj: u32) -> f64 {
    let _ = dj;
    1.0
}

/// Exact SU(2) F-symbol as a [`SignedSqrtRational`] -- the value authority.
///
/// $F = (-1)^{j_1+j_2+j_3+j_4}\, \sqrt{(dj_5+1)(dj_6+1)}\; \{6j\!: dj_1\, dj_2\, dj_5 / dj_3\, dj_4\, dj_6\}$,
/// composed exactly: the dimension factor folds into the radicand
/// (`times_sqrt_int`), the phase into the sign, with no intermediate rounding.
///
/// Convention and the exact 6j argument order / phase exponent are derived from
/// the reference chain (verified numerically against TensorKitSectors 0.3.6,
/// max abs error 4.4e-16 over the doubled-spin <= 12 grid). TensorKitSectors
/// tugbK `src/irreps/su2irrep.jl:Fsymbol` is `sqrtdim(s5)*sqrtdim(s6) *
/// racahW(T, j1,j2,j4,j3, j5,j6)`, and WignerSymbols.jl v2.0.0
/// `WignerSymbols.jl:racahW(T,j1,j2,J,j3,J12,J23)` is `wigner6j(T,
/// j1,j2,J12,j3,J,J23) * (-1)^(j1+j2+j3+J)`. Substituting racahW's arguments
/// `(j1,j2,J=j4,j3,J12=j5,J23=j6)` gives the 6j `{j1 j2 j5 / j3 j4 j6}` (whose
/// triangle set matches [`wigner_6j`]) and the phase `(-1)^(j1+j2+j3+j4)` --
/// integer-valued exactly when that 6j is admissible, i.e. `(dj1+dj2+dj3+dj4)/2`
/// here.
fn f_symbol_exact(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dj4: u32,
    dj5: u32,
    dj6: u32,
) -> SignedSqrtRational {
    let w = wigner_6j(dj1, dj2, dj5, dj3, dj4, dj6);
    if w.sign() == 0 {
        return SignedSqrtRational::zero();
    }
    let v = w
        .times_sqrt_int((dj5 as u64) + 1)
        .times_sqrt_int((dj6 as u64) + 1);
    if phase_is_negative(((dj1 as i64) + (dj2 as i64) + (dj3 as i64) + (dj4 as i64)) / 2) {
        v.neg_value()
    } else {
        v
    }
}

/// Multiplicity-free SU(2) F-symbol `F^{dj1 dj2 dj3}_{dj4}[dj5, dj6]` as `f64`.
///
/// The consumer-facing presentation of the exact F-symbol. Consumers need an
/// `f64` scalar per recoupling; rounding the exact value on every call would
/// re-run the big-integer `sqrt` in [`SignedSqrtRational::to_f64`] on the hot
/// path. This tier caches the rounded scalar, so a warm hit is a hash lookup
/// returning a `Copy` `f64` -- the only `to_f64` call happens inside the miss
/// closure, which the private `CoefficientCache::get_or_compute` provably skips on
/// a hit (see the f64-tier test in `cache.rs`).
///
/// Layering (the reference splits `@cached Fsymbol` from the coefficient
/// caches for the same reason): the exact 6j tier owns the *value*; this
/// tier owns the *presentation*. The two never disagree because the f64 here is
/// derived from that same exact value, never independently.
///
/// Multiplicity-free: SU(2) has `N^c_ab <= 1`, so the F-symbol is a scalar
/// rather than the rank-4 block the generated families return. Non-admissible
/// labels give exact `0.0`.
///
/// The rounded value passes the F finiteness gate (`docs/gauge_cartan.md` §7,
/// gate 7) before it is cached; `|F| ≤ 1` keeps it finite.
pub(crate) fn f_symbol(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dj4: u32,
    dj5: u32,
    dj6: u32,
) -> Result<f64, crate::frcore::GateError> {
    let finite = |x: f64| {
        if x.is_finite() {
            Ok(x)
        } else {
            Err(crate::frcore::GateError::NonFinite { gate: "F" })
        }
    };
    // Key on the 6j class actually evaluated, {dj1 dj2 dj5 / dj3 dj4 dj6}, plus
    // the two determinants that class does NOT carry (dimension factor, phase).
    // See FKey for the key-completeness argument.
    match canonical_regge_6j(dj1, dj2, dj5, dj3, dj4, dj6) {
        Ok(regge) => {
            let key = FKey::new(
                regge,
                ((dj5 as u64) + 1) * ((dj6 as u64) + 1),
                phase_is_negative(((dj1 as i64) + (dj2 as i64) + (dj3 as i64) + (dj4 as i64)) / 2),
            );
            cache::su2_f(key, || {
                finite(f_symbol_exact(dj1, dj2, dj3, dj4, dj5, dj6).to_f64())
            })
        }
        // Non-admissible 6j (F is exactly 0) or a label too large to key: round
        // the exact value directly. Not worth a cache slot -- zeros are free and
        // overflow-scale labels do not recur in the TN hot loop. Bypassing the
        // tier here also means a truncated/ambiguous key can never be formed.
        Err(_) => finite(f_symbol_exact(dj1, dj2, dj3, dj4, dj5, dj6).to_f64()),
    }
}

/// 3j admissibility: projections sum to zero, each `|dm| <= dj` with
/// `dj + dm` even, and the spin triangle. The single predicate behind the 3j
/// engine, its Regge canonicalizer and [`clebsch_gordan`]. Projections are
/// `i64`, so every sum of `i32`-derived inputs is exact.
fn admissible_3j(dj1: u32, dj2: u32, dj3: u32, dm1: i64, dm2: i64, dm3: i64) -> bool {
    dm1 + dm2 + dm3 == 0
        && [(dj1, dm1), (dj2, dm2), (dj3, dm3)]
            .into_iter()
            .all(|(dj, dm)| dm.abs() <= i64::from(dj) && (i64::from(dj) + dm) % 2 == 0)
        && triangle_ok(dj1, dj2, dj3)
}

/// $(-1)^p < 0$, i.e. `p` odd.
#[inline]
fn phase_is_negative(p: i64) -> bool {
    p.rem_euclid(2) == 1
}

/// Canonical Regge key for a 6j symbol.
///
/// Six nonnegative doubled-integer components (Rasch-Yu canonical form). Every
/// element of a 6j symmetry orbit maps to the same key. Stored losslessly as
/// `u16`; a component exceeding `u16::MAX` yields [`ReggeError::Overflow`]
/// rather than a silent truncation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Regge6j([u16; 6]);

/// Key for the derived-f64 F-symbol tier ([`f_symbol`]).
///
/// # Why these three components, and why they are complete
///
/// `F = phase * sqrt((dj5+1)(dj6+1)) * {6j}`, evaluating the 6j `{dj1 dj2 dj5 /
/// dj3 dj4 dj6}`. Its `f64` value is fixed by exactly three data:
///
/// * `regge` -- the canonical Regge class of that 6j. A 6j is invariant under
///   its full symmetry group, so the class names one exact 6j value (sign
///   included). This is the *only* part shared with the exact 6j tier.
/// * `dim` -- the product `(dj5+1)(dj6+1)` under the square root. F is **not**
///   invariant under the full Regge group of its 6j: two distinct F inputs can
///   share a 6j class yet carry different `(dj5, dj6)`, hence a different
///   radicand factor. Keying on `regge` alone would collide them and hand one
///   F's value to the other. The product -- symmetric in `dj5 <-> dj6`, exactly
///   as the factor is -- restores that missing determinant.
/// * `phase_neg` -- the parity of `(dj1+dj2+dj3+dj4)/2`, the overall sign. Again
///   not carried by the 6j class, and again able to differ between two inputs
///   that share a class.
///
/// Conversely, any two inputs agreeing on all three yield the identical F (same
/// 6j value, same radicand factor, same sign), so the triple is a *complete*
/// key: no collision returns a wrong value and no determinant is omitted.
/// Overflow-scale 6j labels have no Regge class and bypass the tier entirely
/// (computed directly), so a truncated key can never be formed here.
#[derive(Clone, Copy, Debug, Eq)]
pub(crate) struct FKey {
    packed: u128,
    dim: u64,
}

impl std::hash::Hash for FKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&(self.packed, self.dim), state);
    }
}

impl PartialEq for FKey {
    fn eq(&self, other: &Self) -> bool {
        self.packed == other.packed && self.dim == other.dim
    }
}

impl FKey {
    pub(crate) fn new(regge: Regge6j, dim: u64, phase_neg: bool) -> Self {
        let mut packed = u128::from(phase_neg) << 96;
        for (i, label) in regge.0.into_iter().enumerate() {
            packed |= u128::from(label) << (16 * i);
        }
        Self { packed, dim }
    }
}

/// Why a 6j label set has no canonical Regge key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum ReggeError {
    /// The label set is not an admissible 6j (a triangle parity or triangle
    /// inequality is violated), so the symbol is exactly zero and has no
    /// canonical representative. Keying it anyway would collide with a distinct
    /// admissible symbol and hand its nonzero value to a zero symbol.
    NonAdmissible,
    /// A canonical component exceeds `u16::MAX` (doubled spin beyond the
    /// supported range). Reported rather than silently truncated.
    Overflow,
}

impl std::fmt::Display for ReggeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReggeError::NonAdmissible => write!(f, "non-admissible 6j label set has no Regge key"),
            ReggeError::Overflow => write!(f, "Regge key component exceeds u16::MAX"),
        }
    }
}

impl std::error::Error for ReggeError {}

/// Canonical Regge key for `{dj1 dj2 dj3; dj4 dj5 dj6}`.
///
/// Ports the alpha/beta construction of the reference crate (Rasch-Yu 2003):
/// three "column" sums `alpha` and four triangle sums `beta`, then the six
/// nonnegative differences `alpha - beta` in a fixed order. Widened to `u16`
/// with a checked conversion.
///
/// Only admissible 6j symbols have a key: the four triangles are checked first,
/// so a non-admissible (zero-valued) set returns [`ReggeError::NonAdmissible`]
/// instead of a floored-halving collision with a distinct admissible symbol.
pub(crate) fn canonical_regge_6j(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dj4: u32,
    dj5: u32,
    dj6: u32,
) -> Result<Regge6j, ReggeError> {
    // Admissibility gate before any halving: the same four triangles as the 6j.
    // This rejects both odd (parity) sums and triangle-inequality violations,
    // so the alpha/beta differences below are all nonnegative integers.
    if !(triangle_ok(dj1, dj2, dj3)
        && triangle_ok(dj1, dj5, dj6)
        && triangle_ok(dj4, dj2, dj6)
        && triangle_ok(dj4, dj5, dj3))
    {
        return Err(ReggeError::NonAdmissible);
    }

    let (dj1, dj2, dj3, dj4, dj5, dj6) = (
        dj1 as i64, dj2 as i64, dj3 as i64, dj4 as i64, dj5 as i64, dj6 as i64,
    );

    // alpha1 <= alpha2 <= alpha3: the three four-term "square" sums (halved).
    let mut alpha = [
        (dj1 + dj2 + dj4 + dj5) / 2,
        (dj1 + dj3 + dj4 + dj6) / 2,
        (dj2 + dj3 + dj5 + dj6) / 2,
    ];
    alpha.sort_unstable();

    // beta1 >= beta2 >= beta3 >= beta4: the four triangle sums (halved),
    // sorted descending.
    let mut beta = [
        (dj1 + dj2 + dj3) / 2,
        (dj1 + dj5 + dj6) / 2,
        (dj2 + dj4 + dj6) / 2,
        (dj3 + dj4 + dj5) / 2,
    ];
    beta.sort_unstable_by(|a, b| b.cmp(a));

    // s = a1-b1, b = a1-b2, t = a1-b3, x = a1-b4, l = a2-b4, e = a3-b4.
    let raw = [
        alpha[2] - beta[3], // e
        alpha[1] - beta[3], // l
        alpha[0] - beta[3], // x
        alpha[0] - beta[2], // t
        alpha[0] - beta[1], // b
        alpha[0] - beta[0], // s
    ];

    let mut out = [0u16; 6];
    for (slot, &v) in out.iter_mut().zip(raw.iter()) {
        // v >= 0 is guaranteed by the admissibility gate above (Rasch-Yu); the
        // only remaining failure is exceeding the u16 storage width.
        debug_assert!(v >= 0, "admissible 6j produced a negative Regge component");
        if v > u16::MAX as i64 {
            return Err(ReggeError::Overflow);
        }
        *slot = v as u16;
    }
    Ok(Regge6j(out))
}

/// Compensation phase relating a 3j symbol to its canonical Regge
/// representative.
///
/// The *classical* 3j symmetry group has 12 elements: the `3!` column
/// permutations and the simultaneous m-negation. Even column permutations and
/// the identity preserve the symbol's value; odd column permutations and
/// m-negation multiply it by `(-1)^(j1+j2+j3)`. When `j1+j2+j3` is even every
/// orbit member has the identical value, so the phase collapses to `Plus`.
/// [`canonical_regge_3j`] returns the single net sign; the cache layer applies
/// it on both store and retrieval to move between an input and its stored
/// representative. (WignerSymbols.jl v2.0.0 `WignerSymbols.jl:reorder3j`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReggePhase {
    /// The input and its canonical representative have equal value.
    Plus,
    /// The input's value is the representative's, negated.
    Minus,
}

impl ReggePhase {
    /// Apply the compensation to a representative value (`+1` or `-1`).
    pub fn apply(self, v: SignedSqrtRational) -> SignedSqrtRational {
        match self {
            ReggePhase::Plus => v,
            ReggePhase::Minus => v.neg_value(),
        }
    }
}

/// Canonical Regge key for a 3j symbol.
///
/// The six doubled labels — three doubled spins `dj` and three doubled
/// projections `dm` — of the classical-orbit representative (the
/// lexicographically maximal image, see `canonicalize3j`). Every
/// element of the 12-element classical symmetry orbit maps to this same key;
/// the value relation is carried separately as a [`ReggePhase`], so the key
/// alone is the orbit invariant.
///
/// `dj` is stored as `u16` with the same checked-widening discipline as
/// [`Regge6j`] (a doubled spin past `u16::MAX` is [`ReggeError::Overflow`]).
/// `dm` is stored signed in `i32`: once the `dj` gate passes, admissibility
/// bounds `|dm| <= dj <= u16::MAX`, so `i32` holds it losslessly.
///
/// Deviation from the reference: WignerSymbols.jl keys on the derived
/// `(β1,β2,β3,α1,α2)` reparametrization of these same canonical labels; racah
/// stores the labels directly (a 1-1 re-encoding), which the orbit test can
/// read against without reconstructing alphas. (WignerSymbols.jl v2.0.0
/// `WignerSymbols.jl:_wigner3j`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Regge3j {
    dj: [u16; 3],
    dm: [i32; 3],
}

/// Canonicalize a doubled-unit 3j label set over its 12-element classical
/// symmetry group, returning the representative labels and the net sign `eps`
/// with `value(input) == eps * value(representative)`.
///
/// The group is `S3` (column permutations) times `Z2` (simultaneous
/// m-negation). Each generator multiplies the symbol by `(-1)^(j1+j2+j3)` when
/// it is an odd column permutation or an m-negation; even permutations and the
/// identity preserve it — the phase rule ported from `WignerSymbols.jl:
/// reorder3j`. The representative is the lexicographically maximal image over
/// all 12 group elements.
///
/// Deviation from the reference: `reorder3j` sorts only by `j` and leaves the
/// m-order among equal-`j` columns unbroken, so it is *not* a full orbit
/// invariant (two column-permutations of an equal-`j` symbol get distinct
/// keys). racah completes the canonicalization — breaking every tie and
/// tracking the extra permutation's sign — so the acceptance-required
/// "all 12 images share one key" holds, at the cost of enumerating 12 tuples
/// (a fixed, tiny constant) instead of the reference's bubble sort.
fn canonicalize3j(dj: [i64; 3], dm: [i64; 3]) -> ([i64; 3], [i64; 3], i8) {
    // (column index map, whether the permutation is odd).
    const PERMS: [([usize; 3], bool); 6] = [
        ([0, 1, 2], false),
        ([0, 2, 1], true),
        ([1, 0, 2], true),
        ([1, 2, 0], false),
        ([2, 0, 1], false),
        ([2, 1, 0], true),
    ];
    let j_odd = ((dj[0] + dj[1] + dj[2]) / 2) % 2 != 0;

    let mut best: Option<([i64; 3], [i64; 3])> = None;
    let mut best_eps = 1i8;
    for (p, p_odd) in PERMS {
        let pj = [dj[p[0]], dj[p[1]], dj[p[2]]];
        let pm = [dm[p[0]], dm[p[1]], dm[p[2]]];
        for neg in [false, true] {
            let m = if neg { [-pm[0], -pm[1], -pm[2]] } else { pm };
            // value(image)/value(input): (-1)^(j1+j2+j3) once per odd generator
            // (odd permutation and/or m-negation), and +1 whenever J is even.
            let odd_ops = (p_odd ^ neg) as i8;
            let eps: i8 = if j_odd && odd_ops == 1 { -1 } else { 1 };
            let cand = (pj, m);
            if best.as_ref().is_none_or(|b| cand > *b) {
                best = Some(cand);
                best_eps = eps;
            }
        }
    }
    let (bj, bm) = best.expect("the identity image always seeds `best`");
    // eps relates input and representative (±1, self-inverse).
    (bj, bm, best_eps)
}

/// Canonical Regge key and compensation phase for `(dj1 dj2 dj3; dm1 dm2 dm3)`.
///
/// Admissibility is gated first — exactly like [`canonical_regge_6j`] and for
/// the same reason (the `(1,1,1,1,1,1)`-class lesson): a non-admissible 3j is
/// exactly zero and has no representative, so it must never produce a key that
/// could collide with a distinct nonzero symbol. Only then is the label set
/// reordered to its canonical representative and widened losslessly.
///
/// Returns `(key, phase)` such that `value(input) == phase.apply(value(rep))`
/// where `rep` is the symbol named by `key`. (WignerSymbols.jl v2.0.0
/// `WignerSymbols.jl:_wigner3j` / `reorder3j`.)
pub(crate) fn canonical_regge_3j(
    dj1: u32,
    dj2: u32,
    dj3: u32,
    dm1: i64,
    dm2: i64,
    dm3: i64,
) -> Result<(Regge3j, ReggePhase), ReggeError> {
    if !admissible_3j(dj1, dj2, dj3, dm1, dm2, dm3) {
        return Err(ReggeError::NonAdmissible);
    }

    let (dj, dm, sign) = canonicalize3j([dj1 as i64, dj2 as i64, dj3 as i64], [dm1, dm2, dm3]);

    // dj values only reorder under canonicalization, but widen with the same
    // checked u16 discipline as Regge6j. dm is bounded by dj (admissibility), so
    // once the dj gate passes it fits i32 with room to spare; store it signed.
    let mut dju = [0u16; 3];
    for (slot, &v) in dju.iter_mut().zip(dj.iter()) {
        if v > u16::MAX as i64 {
            return Err(ReggeError::Overflow);
        }
        *slot = v as u16;
    }
    let dmi = [dm[0] as i32, dm[1] as i32, dm[2] as i32];

    let phase = if sign < 0 {
        ReggePhase::Minus
    } else {
        ReggePhase::Plus
    };
    Ok((Regge3j { dj: dju, dm: dmi }, phase))
}

/// An SU(2) irreducible representation, labeled by its doubled spin `dj = 2j`.
///
/// Every `u32` is a valid label, so construction is infallible (see [`new`]).
/// Fusion of two irreps can overflow the `u32` label space, which is where the
/// only fallible operation ([`fusion`]) lives.
///
/// [`new`]: A1Irrep::new
/// [`fusion`]: A1Irrep::fusion
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct A1Irrep(u32);

impl A1Irrep {
    /// Construct the irrep with doubled spin `dj = 2j`.
    ///
    /// Infallible by construction: SU(2) has one irrep for every nonnegative
    /// half-integer spin, i.e. exactly one for every `u32` doubled spin, so
    /// there is no invalid label to reject.
    pub fn new(dj: u32) -> Self {
        A1Irrep(dj)
    }

    /// The doubled spin `dj = 2j`.
    pub fn dj(self) -> u32 {
        self.0
    }

    /// The dimension `2j + 1 = dj + 1`.
    ///
    /// Returned as `u64` so it cannot overflow: the maximum `dj` is
    /// `u32::MAX`, and `u32::MAX as u64 + 1` fits `u64` with room to spare.
    #[cfg(test)]
    pub fn dim(self) -> u64 {
        self.0 as u64 + 1
    }

    /// The dual irrep, which for SU(2) is the irrep itself (every SU(2) irrep
    /// is self-dual).
    ///
    /// The self-duality carries a Frobenius–Schur phase distinguishing the
    /// orthogonal (integer `j`) from the symplectic (half-integer `j`) case;
    /// that phase is [`fs_phase`], not folded into this identity.
    #[cfg(test)]
    pub fn dual(self) -> Self {
        self
    }

    /// Fusion decomposition `self ⊗ other`: the irreps `dj` in
    /// `|dj1 − dj2| ..= dj1 + dj2` in steps of 2 (each with multiplicity one).
    ///
    /// Returns [`super::LieError::LabelOverflow`] when `dj1 + dj2` exceeds
    /// `u32` or the channel count is unrepresentable; the returned
    /// [`A1Fusion`] is an allocation-free
    /// [`ExactSizeIterator`]/[`DoubleEndedIterator`].
    ///
    pub fn fusion(self, other: Self) -> Result<A1Fusion, super::LieError> {
        let hi = self
            .0
            .checked_add(other.0)
            .ok_or(super::LieError::LabelOverflow)?;
        let lo = self.0.abs_diff(other.0);
        // lo and hi always share parity (both congruent to dj1+dj2 mod 2), so
        // the half-open count is exact and the last step lands on hi.
        let remaining =
            super::checked_dimension((hi - lo) / 2).and_then(|n| super::checked_count(n, 1))?;
        Ok(A1Fusion {
            front: lo,
            back: hi,
            remaining,
        })
    }
}

/// Allocation-free iterator over the fusion channels of two SU(2) irreps
/// (see [`A1Irrep::fusion`]): doubled spins from `|dj1 − dj2|` to `dj1 + dj2`
/// in steps of 2.
#[derive(Clone, Copy, Debug)]
pub(super) struct A1Fusion {
    /// Next doubled spin yielded from the front (ascending).
    front: u32,
    /// Next doubled spin yielded from the back (descending).
    back: u32,
    /// Channels not yet yielded from either end.
    remaining: usize,
}

impl Iterator for A1Fusion {
    type Item = A1Irrep;

    fn next(&mut self) -> Option<A1Irrep> {
        if self.remaining == 0 {
            return None;
        }
        let dj = self.front;
        self.remaining -= 1;
        // Only advance while channels remain: on the last element `front` may
        // sit at u32::MAX (hi == dj1 + dj2), where `+= 2` would overflow.
        if self.remaining > 0 {
            self.front += 2;
        }
        Some(A1Irrep(dj))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl DoubleEndedIterator for A1Fusion {
    fn next_back(&mut self) -> Option<A1Irrep> {
        if self.remaining == 0 {
            return None;
        }
        let dj = self.back;
        self.remaining -= 1;
        // Only advance while channels remain: on the last element `back` may
        // sit at 0 (lo == 0, the singlet channel), where `-= 2` would underflow.
        if self.remaining > 0 {
            self.back -= 2;
        }
        Some(A1Irrep(dj))
    }
}

impl ExactSizeIterator for A1Fusion {}

#[cfg(test)]
mod tests {
    use super::*;
    use malachite_nz::integer::Integer;

    fn sq(v: &SignedSqrtRational) -> Rational {
        v.signed_square()
    }

    #[test]
    fn triangle_admissibility() {
        assert!(triangle_ok(1, 1, 2)); // 1/2,1/2,1
        assert!(triangle_ok(2, 2, 2)); // 1,1,1
        assert!(!triangle_ok(1, 1, 1)); // parity fails
        assert!(!triangle_ok(2, 2, 6)); // out of range
    }

    /// A stretched coupling, `j3 = j1 + j2` with `m1 = -j1`, `m2 = j2`, is
    /// `sqrt(1 / binomial(2 j3, j3 + m3))`; here `1/binomial(2050, 1015)` lies
    /// below 2^-1022. Before #262 the subnormal result was rounded twice and
    /// returned `0x000f_d1dc_1567_2826`, one subnormal step above the nearest.
    #[test]
    fn subnormal_cgc_rounds_once() {
        let cg = clebsch_gordan(1035, -1035, 1015, 1015, 2050, -20);
        let binom = (1..=1015u32).fold(Integer::from(1), |acc, i| {
            acc * Integer::from(1035 + i) / Integer::from(i)
        });
        assert_eq!(cg.sign(), 1);
        assert_eq!(
            *cg.radicand(),
            Rational::from_integers(Integer::from(1), binom)
        );
        assert_eq!(cg.to_f64().to_bits(), 0x000f_d1dc_1567_2825);
    }

    #[test]
    fn six_j_known_value() {
        // {1/2 1/2 1; 1/2 1/2 1} = 1/6, so signed_square = 1/36.
        let v = wigner_6j(1, 1, 2, 1, 1, 2);
        assert_eq!(
            sq(&v),
            Rational::from_integers(Integer::from(1), Integer::from(36))
        );
        assert!((v.to_f64() - 1.0 / 6.0).abs() < 1e-14);
    }

    #[test]
    fn six_j_all_ones() {
        // {1 1 1; 1 1 1} = 1/6.
        let v = wigner_6j(2, 2, 2, 2, 2, 2);
        assert_eq!(
            sq(&v),
            Rational::from_integers(Integer::from(1), Integer::from(36))
        );
    }

    #[test]
    fn six_j_nonadmissible_is_zero() {
        // triangle (1,1,1) parity violation -> zero.
        let v = wigner_6j(1, 1, 1, 1, 1, 1);
        assert_eq!(v, SignedSqrtRational::zero());
    }

    #[test]
    fn three_j_known_value() {
        // (1/2 1/2 1; 1/2 -1/2 0) = +1/sqrt(6), signed_square = +1/6
        // (checked against WignerSymbols.jl 2.0.0).
        let v = wigner_3j(1, 1, 2, 1, -1, 0);
        assert_eq!(
            sq(&v),
            Rational::from_integers(Integer::from(1), Integer::from(6))
        );
    }

    #[test]
    fn three_j_m_sum_nonzero_is_zero() {
        let v = wigner_3j(1, 1, 2, 1, 1, 0);
        assert_eq!(v, SignedSqrtRational::zero());
    }

    #[test]
    fn cg_known_value() {
        // <1/2 1/2, 1/2 -1/2 | 1 0> = 1/sqrt(2), signed_square = 1/2.
        let v = clebsch_gordan(1, 1, 1, -1, 2, 0);
        assert_eq!(
            sq(&v),
            Rational::from_integers(Integer::from(1), Integer::from(2))
        );
        assert!((v.to_f64() - (0.5f64).sqrt()).abs() < 1e-14);
    }

    #[test]
    fn cg_stretched_is_one() {
        // <1/2 1/2, 1/2 1/2 | 1 1> = 1.
        let v = clebsch_gordan(1, 1, 1, 1, 2, 2);
        assert_eq!(sq(&v), Rational::from(1));
        assert!((v.to_f64() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn regge_key_orbit_invariance_small() {
        // An asymmetric admissible 6j whose three "square" sums are NOT all
        // equal, so the alpha ordering genuinely matters. Doubled labels
        // {1 2 2; 3 2 1}. Swapping columns 1<->2 (j2,j5)<->(j3,j6) permutes the
        // alpha entries; the canonical key must be invariant.
        let base = canonical_regge_6j(2, 4, 4, 6, 4, 2).unwrap();
        let swap12 = canonical_regge_6j(2, 4, 4, 6, 2, 4).unwrap();
        assert_eq!(base, swap12);
        // Sanity: this case is not degenerate (the two label sets differ).
        assert_ne!([2u32, 4, 4, 6, 4, 2], [2u32, 4, 4, 6, 2, 4]);
    }

    #[test]
    fn regge_overflow_reported() {
        // A doubled spin large enough to overflow a component -> typed error.
        let big = 200_000u32;
        assert_eq!(
            canonical_regge_6j(big, big, big, big, big, big),
            Err(ReggeError::Overflow)
        );
    }

    #[test]
    fn regge_nonadmissible_is_error_not_a_key() {
        // Parity-violating {1/2 1/2 1/2; ...}: the exact 6j is zero, while the
        // admissible {1 1 1; 1 1 1} is 1/6. Keying the former (floored halving)
        // would collide with the latter and hand a nonzero value to a zero
        // symbol via the publication cache -- exactly the silent-wrong-answer
        // class the crate excludes. So a non-admissible set has no key.
        assert_eq!(
            canonical_regge_6j(1, 1, 1, 1, 1, 1),
            Err(ReggeError::NonAdmissible)
        );
        let admissible = canonical_regge_6j(2, 2, 2, 2, 2, 2);
        assert!(admissible.is_ok());
        assert_ne!(
            canonical_regge_6j(1, 1, 1, 1, 1, 1).ok(),
            admissible.ok(),
            "non-admissible input must not share a key with an admissible one"
        );
    }

    /// The six column permutations (as index maps) times the m-negation give
    /// the 12-element classical 3j orbit.
    fn orbit_images(dj: [u32; 3], dm: [i32; 3]) -> Vec<([u32; 3], [i32; 3])> {
        const PERMS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut out = Vec::with_capacity(12);
        for p in PERMS {
            let pj = [dj[p[0]], dj[p[1]], dj[p[2]]];
            let pm = [dm[p[0]], dm[p[1]], dm[p[2]]];
            out.push((pj, pm));
            out.push((pj, [-pm[0], -pm[1], -pm[2]]));
        }
        out
    }

    #[test]
    fn regge3j_orbit_same_key_and_phase_compensated_value() {
        // {1 1 1; 1 0 -1} (doubled 2,2,2 / 2,0,-2). J = 3 is ODD, so the phase
        // is nontrivial: odd column permutations and m-negation flip the sign.
        let dj = [2u32, 2, 2];
        let dm = [2i32, 0, -2];

        let mut key0 = None;
        let mut rep_value = None;
        let mut saw_negative_phase = false;
        for (pj, pm) in orbit_images(dj, dm) {
            let (key, phase) = canonical_regge_3j(
                pj[0],
                pj[1],
                pj[2],
                pm[0].into(),
                pm[1].into(),
                pm[2].into(),
            )
            .unwrap();
            // All 12 images share the canonical key.
            match key0 {
                None => key0 = Some(key),
                Some(k) => assert_eq!(k, key, "orbit image produced a different key"),
            }
            // phase.apply(raw value) recovers the single representative value.
            let raw = wigner_3j(pj[0], pj[1], pj[2], pm[0], pm[1], pm[2]);
            let compensated = phase.apply(raw.clone());
            match &rep_value {
                None => rep_value = Some(compensated),
                Some(rv) => assert_eq!(
                    rv, &compensated,
                    "phase-compensated value differs across the orbit"
                ),
            }
            if phase == ReggePhase::Minus {
                saw_negative_phase = true;
                // The phase genuinely acts: raw and compensated differ in sign.
                assert_ne!(raw, phase.apply(raw.clone()));
            }
        }
        assert!(
            saw_negative_phase,
            "J-odd orbit must exercise the Minus phase"
        );
    }

    #[test]
    fn regge3j_even_j_phase_is_always_plus() {
        // {1 1 2; 1 -1 0} (doubled 2,2,4 / 2,-2,0). J = 4 is EVEN, so every
        // orbit member has the identical value and the phase is always Plus.
        let dj = [2u32, 2, 4];
        let dm = [2i32, -2, 0];
        for (pj, pm) in orbit_images(dj, dm) {
            let (_, phase) = canonical_regge_3j(
                pj[0],
                pj[1],
                pj[2],
                pm[0].into(),
                pm[1].into(),
                pm[2].into(),
            )
            .unwrap();
            assert_eq!(phase, ReggePhase::Plus, "even-J phase must be Plus");
        }
    }

    #[test]
    fn regge3j_nonadmissible_is_error_not_a_key() {
        // m-sum nonzero: not an admissible 3j, so no representative and no key.
        assert_eq!(
            canonical_regge_3j(2, 2, 2, 2, 2, 0),
            Err(ReggeError::NonAdmissible)
        );
        // |m| > j.
        assert_eq!(
            canonical_regge_3j(2, 2, 2, 4, -2, -2),
            Err(ReggeError::NonAdmissible)
        );
        // Triangle parity violation.
        assert_eq!(
            canonical_regge_3j(1, 1, 1, 1, -1, 0),
            Err(ReggeError::NonAdmissible)
        );
    }

    #[test]
    fn cached_3j_matches_uncached_over_grid() {
        // Every admissible and non-admissible small 3j: the transparently cached
        // public value must equal the uncached engine, and a repeat call (a
        // cache hit) must still match. This exercises the ReggePhase
        // compensation end-to-end across the whole orbit structure.
        for dj1 in 0..=4u32 {
            for dj2 in 0..=4u32 {
                for dj3 in 0..=4u32 {
                    for dm1 in -4..=4i32 {
                        for dm2 in -4..=4i32 {
                            for dm3 in -4..=4i32 {
                                let raw = wigner_3j_uncached(
                                    dj1,
                                    dj2,
                                    dj3,
                                    dm1.into(),
                                    dm2.into(),
                                    dm3.into(),
                                );
                                let ctx = (dj1, dj2, dj3, dm1, dm2, dm3);
                                assert_eq!(
                                    wigner_3j(dj1, dj2, dj3, dm1, dm2, dm3),
                                    raw,
                                    "cached != uncached at {ctx:?}"
                                );
                                assert_eq!(
                                    wigner_3j(dj1, dj2, dj3, dm1, dm2, dm3),
                                    raw,
                                    "cache-hit != uncached at {ctx:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cached_6j_matches_uncached_over_grid() {
        for dj1 in 0..=4u32 {
            for dj2 in 0..=4u32 {
                for dj3 in 0..=4u32 {
                    for dj4 in 0..=4u32 {
                        for dj5 in 0..=4u32 {
                            for dj6 in 0..=4u32 {
                                let raw = wigner_6j_uncached(dj1, dj2, dj3, dj4, dj5, dj6);
                                let ctx = (dj1, dj2, dj3, dj4, dj5, dj6);
                                assert_eq!(
                                    wigner_6j(dj1, dj2, dj3, dj4, dj5, dj6),
                                    raw,
                                    "cached != uncached at {ctx:?}"
                                );
                                assert_eq!(
                                    wigner_6j(dj1, dj2, dj3, dj4, dj5, dj6),
                                    raw,
                                    "cache-hit != uncached at {ctx:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn public_stats_and_reset_smoke() {
        let _guard = crate::cache::TEST_CACHE_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Exercise the global wrappers; exact counts are covered on local caches
        // in cache::tests because unrelated tests also populate the global tiers.
        let _ = wigner_6j(2, 2, 2, 2, 2, 2);
        let _ = wigner_3j(2, 2, 2, 2, 0, -2);
        let stats = || {
            for &tier in crate::cache::CoefficientCacheTier::ALL {
                let _ = crate::cache::tier_stats(tier);
            }
        };
        stats();
        crate::cache::reset();
        stats();
    }

    #[test]
    fn regge3j_overflow_reported() {
        let big = 200_000u32;
        assert_eq!(
            canonical_regge_3j(big, big, big, 0, 0, 0),
            Err(ReggeError::Overflow)
        );
    }

    #[test]
    fn f_symbol_exact_composition_identity() {
        // Independent exact check of the closed form. Reconstruct signed_square(F)
        // separately from the 6j value, the dimension factor, and the phase --
        // NOT by re-running f_symbol_exact -- and require exact equality:
        //   signed_square(F) == phase * (dj5+1)(dj6+1) * signed_square({6j}).
        // This catches a wrong 6j argument order or a wrong phase exponent (the
        // silent-wrong-answer class) because a mistaken order lands on a
        // different 6j class whose signed_square differs on non-symmetric labels.
        for dj1 in 0..=4u32 {
            for dj2 in 0..=4u32 {
                for dj3 in 0..=4u32 {
                    for dj4 in 0..=4u32 {
                        for dj5 in 0..=4u32 {
                            for dj6 in 0..=4u32 {
                                let f = f_symbol_exact(dj1, dj2, dj3, dj4, dj5, dj6);
                                let w = wigner_6j(dj1, dj2, dj5, dj3, dj4, dj6);
                                let dim = Integer::from(((dj5 + 1) * (dj6 + 1)) as i64);
                                let mut expected = w.signed_square() * Rational::from(dim);
                                let sum = (dj1 as i64) + (dj2 as i64) + (dj3 as i64) + (dj4 as i64);
                                if (sum / 2).rem_euclid(2) == 1 {
                                    expected = -expected;
                                }
                                let ctx = (dj1, dj2, dj3, dj4, dj5, dj6);
                                assert_eq!(f.signed_square(), expected, "F^2 identity at {ctx:?}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn f_symbol_cached_matches_exact_over_grid() {
        // The cached f64 path equals the rounded exact composition, and a repeat
        // (a cache hit) still matches.
        for dj1 in 0..=4u32 {
            for dj2 in 0..=4u32 {
                for dj3 in 0..=4u32 {
                    for dj4 in 0..=4u32 {
                        for dj5 in 0..=4u32 {
                            for dj6 in 0..=4u32 {
                                let want = f_symbol_exact(dj1, dj2, dj3, dj4, dj5, dj6).to_f64();
                                let ctx = (dj1, dj2, dj3, dj4, dj5, dj6);
                                assert_eq!(
                                    f_symbol(dj1, dj2, dj3, dj4, dj5, dj6).unwrap(),
                                    want,
                                    "cached != exact at {ctx:?}"
                                );
                                assert_eq!(
                                    f_symbol(dj1, dj2, dj3, dj4, dj5, dj6).unwrap(),
                                    want,
                                    "cache-hit != exact at {ctx:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn f_symbol_all_trivial_is_one() {
        // All six labels trivial: {0 0 0 / 0 0 0} 6j = 1, dims = 1, phase +.
        assert_eq!(f_symbol(0, 0, 0, 0, 0, 0), Ok(1.0));
    }

    #[test]
    fn f_symbol_nonadmissible_is_zero() {
        // The evaluated 6j {1/2 1/2 1/2 / 1/2 1/2 1/2} is parity-forbidden, so F
        // is exactly zero and takes the Err (bypass) branch.
        assert_eq!(f_symbol(1, 1, 1, 1, 1, 1), Ok(0.0));
    }

    #[test]
    fn r_symbol_matches_convention() {
        // R^{ab}_c = (-1)^(j1+j2-j3) on an admissible triangle, else 0.
        // (Oracle: TensorKitSectors Rsymbol, exact over the full dj<=12 grid.)
        // {1/2 1/2 1}: (1+1-2)/2 = 0 even -> +1.
        assert_eq!(r_symbol(1, 1, 2), 1.0);
        // {1/2 1/2 0}: (1+1-0)/2 = 1 odd -> -1.
        assert_eq!(r_symbol(1, 1, 0), -1.0);
        // {1 1 1}: doubled (2,2,2), (2+2-2)/2 = 1 odd -> -1.
        assert_eq!(r_symbol(2, 2, 2), -1.0);
        // {1 1 2}: doubled (2,2,4), (2+2-4)/2 = 0 even -> +1.
        assert_eq!(r_symbol(2, 2, 4), 1.0);
        // Non-admissible (parity) -> exact zero, no phase.
        assert_eq!(r_symbol(1, 1, 1), 0.0);
        // Non-admissible (triangle inequality) -> exact zero.
        assert_eq!(r_symbol(2, 2, 8), 0.0);
    }

    #[test]
    fn frobenius_schur_is_sign_of_doubled_spin() {
        // (-1)^(2j) as +-1: integer spins +1, half-integer spins -1.
        assert_eq!(fs_phase(0), 1.0); // j=0
        assert_eq!(fs_phase(1), -1.0); // j=1/2
        assert_eq!(fs_phase(2), 1.0); // j=1
        assert_eq!(fs_phase(3), -1.0); // j=3/2
        assert_eq!(fs_phase(4), 1.0); // j=2
    }

    #[test]
    fn regge_triangle_inequality_is_nonadmissible_not_overflow() {
        // (2,2,20) violates the triangle inequality: report non-admissibility,
        // not the u16-overflow variant.
        assert_eq!(
            canonical_regge_6j(2, 2, 20, 2, 2, 2),
            Err(ReggeError::NonAdmissible)
        );
    }
}

#[cfg(test)]
mod checked_tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // -- A1Irrep basics ----------------------------------------------------

    #[test]
    fn irrep_accessors_and_self_dual() {
        let s = A1Irrep::new(3); // spin 3/2
        assert_eq!(s.dj(), 3);
        assert_eq!(s.dim(), 4); // 2j+1
        assert_eq!(s.dual(), s); // SU(2) is self-dual
        assert_eq!(A1Irrep::new(0).dim(), 1);
    }

    #[test]
    fn dim_does_not_overflow_at_max_label() {
        assert_eq!(A1Irrep::new(u32::MAX).dim(), u32::MAX as u64 + 1);
    }

    // -- Fusion range / iterator contract -----------------------------------

    #[test]
    fn fusion_range_matches_direct_triangle_scan() {
        // The fusion channels must be exactly those doubled spins dj3 for which
        // (dj1, dj2, dj3) is an admissible triangle — an independent oracle.
        for dj1 in 0..=8u32 {
            for dj2 in 0..=8u32 {
                let got: Vec<u32> = A1Irrep::new(dj1)
                    .fusion(A1Irrep::new(dj2))
                    .unwrap()
                    .map(|s| s.dj())
                    .collect();
                let want: Vec<u32> = (0..=dj1 + dj2)
                    .filter(|&c| triangle_ok(dj1, dj2, c))
                    .collect();
                assert_eq!(got, want, "fusion {dj1} x {dj2}");
            }
        }
    }

    #[test]
    fn fusion_exact_size_and_double_ended() {
        // len() is exact and consistent with the yielded count; iterating from
        // the back reverses the front order.
        let mut it = A1Irrep::new(4).fusion(A1Irrep::new(2)).unwrap(); // 2..=6 step 2
        assert_eq!(it.len(), 3);
        assert_eq!(it.next().map(|s| s.dj()), Some(2));
        assert_eq!(it.next_back().map(|s| s.dj()), Some(6));
        assert_eq!(it.len(), 1);
        assert_eq!(it.next().map(|s| s.dj()), Some(4));
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
        assert!(it.next_back().is_none());

        let full: Vec<u32> = A1Irrep::new(4)
            .fusion(A1Irrep::new(2))
            .unwrap()
            .rev()
            .map(|s| s.dj())
            .collect();
        assert_eq!(full, vec![6, 4, 2]);
    }

    #[test]
    fn fusion_overflow_at_u32_boundary() {
        // dj1 + dj2 overflows u32 -> typed LabelOverflow, no wraparound.
        let a = A1Irrep::new(u32::MAX);
        assert!(matches!(
            a.fusion(A1Irrep::new(1)),
            Err(super::super::LieError::LabelOverflow)
        ));
        // Exactly on the boundary is fine (sum == u32::MAX).
        assert!(A1Irrep::new(u32::MAX - 1).fusion(A1Irrep::new(1)).is_ok());
    }

    #[test]
    fn fusion_full_back_drain_reaches_singlet_without_underflow() {
        // Draining from the back down to lo = 0 (dj1 == dj2, the singlet
        // channel) must not advance the cursor past the final element:
        // `back -= 2` at back == 0 would underflow u32 and panic in debug.
        let got: Vec<u32> = A1Irrep::new(2)
            .fusion(A1Irrep::new(2))
            .unwrap()
            .rev()
            .map(|s| s.dj())
            .collect();
        assert_eq!(got, vec![4, 2, 0]);
    }

    #[test]
    fn fusion_full_forward_drain_at_u32_max_without_overflow() {
        // Draining forward up to hi == u32::MAX must not advance the cursor
        // past the final element: `front += 2` at front == u32::MAX would
        // overflow and panic in debug. lo = MAX-2, hi = MAX (parity-consistent).
        let got: Vec<u32> = A1Irrep::new(u32::MAX - 1)
            .fusion(A1Irrep::new(1))
            .unwrap()
            .map(|s| s.dj())
            .collect();
        assert_eq!(got, vec![u32::MAX - 2, u32::MAX]);
    }

    #[test]
    fn fusion_mixed_drain_to_singlet_without_underflow() {
        // Alternating next/next_back to exhaustion, ending at lo = 0, must not
        // underflow on either cursor. 2 x 2 = {0, 2, 4}.
        let mut it = A1Irrep::new(2).fusion(A1Irrep::new(2)).unwrap();
        let mut got = vec![
            it.next().unwrap().dj(),      // front: 0
            it.next_back().unwrap().dj(), // back: 4
            it.next_back().unwrap().dj(), // back: 2 (lands cursor at lo = 0)
        ];
        assert!(it.next().is_none());
        assert!(it.next_back().is_none());
        got.sort_unstable();
        assert_eq!(got, vec![0, 2, 4]);
    }

    // -- Guard inventory: each admissibility rule yields exact zero ---------

    fn zero() -> SignedSqrtRational {
        SignedSqrtRational::zero()
    }

    #[test]
    fn guard_6j_triangle() {
        // (1,2,3) triangle inequality violated: dj3 = 20 with dj1=dj2=2.
        assert_eq!(wigner_6j(2, 2, 20, 2, 2, 2), zero());
        // A different triangle of the four: (4,2,6) = (7,3,100) inequality.
        assert_eq!(wigner_6j(2, 3, 3, 7, 2, 100), zero());
        // A triangle-violating near-max 6j is zero cheaply (the predicate
        // widens to i64), never overflowing or summing a huge series.
        assert_eq!(wigner_6j(u32::MAX, 2, 0, 2, 2, 2), zero());
    }

    #[test]
    fn guard_3j_projection_sum() {
        assert_eq!(wigner_3j(2, 2, 2, 2, 2, 2), zero());
        // The sum i32::MAX + 1 overflows i32; it is still a nonzero sum.
        assert_eq!(wigner_3j(0, 0, 0, i32::MAX, 1, 0), zero());
        assert_eq!(wigner_3j(0, 0, 0, i32::MIN, -1, 0), zero());
    }

    #[test]
    fn guard_3j_projection_out_of_range() {
        // |dm1| = 4 > dj1 = 2 (m-sum still zero, so this rule is what fails).
        assert_eq!(wigner_3j(2, 2, 2, 4, -2, -2), zero());
    }

    #[test]
    fn guard_3j_projection_parity() {
        // dj1 + dm1 = 2 + 1 = 3 is odd: projection off the ladder (m-sum zero).
        assert_eq!(wigner_3j(2, 2, 2, 1, 1, -2), zero());
    }

    #[test]
    fn guard_3j_triangle() {
        // Projections all admissible (dm = 0 everywhere, even dj) and sum to
        // zero, but (2,2,8) fails the triangle inequality (8 > 2+2).
        assert_eq!(wigner_3j(2, 2, 8, 0, 0, 0), zero());
    }

    #[test]
    fn guard_cg_inadmissible() {
        // CG couples via the 3j (dj1 dj2 dj3; dm1 dm2 -dm3); m1+m2-m3 != 0.
        assert_eq!(clebsch_gordan(2, 2, 2, 2, 2, 0), zero());
        // -i32::MIN is not an i32; the coupling 0 + 0 -> (dj3 = 0, dm3 = i32::MIN)
        // is forbidden by |dm3| > dj3 and by the projection sum.
        assert_eq!(clebsch_gordan(0, 0, 0, 0, 0, i32::MIN), zero());
    }

    #[test]
    fn cg_projection_beyond_i32_is_admissible_not_zeroed() {
        // CG(J, i32::MIN, 0, 0, J, i32::MIN) with J = 2^31 is the stretched
        // coupling m₁ = -j, j₂ = 0. Its 3j has projection -dm3 = 2^31, which
        // i32 cannot hold; that is not inadmissibility. The key overflows u16
        // instead (the value itself needs factorials of 2^31).
        let j = 1u32 << 31;
        let wide = [i64::from(i32::MIN), 0, -i64::from(i32::MIN)];
        assert!(admissible_3j(j, 0, j, wide[0], wide[1], wide[2]));
        assert_eq!(
            canonical_regge_3j(j, 0, j, wide[0], wide[1], wide[2]),
            Err(ReggeError::Overflow)
        );
    }

    #[test]
    fn guard_f_symbol_triangle() {
        // F evaluates 6j {dj1 dj2 dj5 / dj3 dj4 dj6} = {1/2 1/2 1/2 / ...};
        // (1,1,1) parity-forbidden.
        assert_eq!(f_symbol(1, 1, 1, 1, 1, 1), Ok(0.0));
    }

    #[test]
    fn guard_r_symbol_triangle() {
        assert_eq!(r_symbol(1, 1, 1), 0.0);
        assert_eq!(r_symbol(1, 1, 2), 1.0);
    }

    // -- Accidental zero of an admissible coupling ---------------------------

    #[test]
    fn admissible_but_zero_6j_is_exact_zero() {
        // {2 3 3; 7 6 6} passes all four triangles yet the 6j is exactly zero
        // (an accidental zero), the same value as a forbidden coupling.
        assert!(triangle_ok(2, 3, 3) && triangle_ok(2, 6, 6));
        assert!(triangle_ok(7, 3, 6) && triangle_ok(7, 6, 3));
        assert_eq!(wigner_6j(2, 3, 3, 7, 6, 6), zero());
    }

    // -- Property: forbidden (independent predicate) => exact zero ---------

    #[test]
    fn property_forbidden_is_zero_and_not_admissible() {
        // Independent of the crate predicates: triangle and ladder rules
        // spelled out in i64.
        fn tri(a: u32, b: u32, c: u32) -> bool {
            let (a, b, c) = (i64::from(a), i64::from(b), i64::from(c));
            (a + b + c) % 2 == 0 && (a - b).abs() <= c && c <= a + b
        }
        fn on_ladder(dj: u32, dm: i32) -> bool {
            let (dj, dm) = (i64::from(dj), i64::from(dm));
            dm.abs() <= dj && (dj - dm).rem_euclid(2) == 0
        }
        let mut rng = ChaCha8Rng::seed_from_u64(0xBEEF);
        let mut tested_6j = 0;
        let mut tested_3j = 0;
        for _ in 0..40_000 {
            let d = [(); 6].map(|_| rng.gen_range(0..=8u32));
            let admissible = tri(d[0], d[1], d[2])
                && tri(d[0], d[4], d[5])
                && tri(d[3], d[1], d[5])
                && tri(d[3], d[4], d[2]);
            if !admissible {
                assert_eq!(
                    wigner_6j(d[0], d[1], d[2], d[3], d[4], d[5]),
                    zero(),
                    "structurally forbidden 6j must be exact zero"
                );
                tested_6j += 1;
            }

            let dj = [(); 3].map(|_| rng.gen_range(0..=6u32));
            let dm = [(); 3].map(|_| rng.gen_range(-6..=6i32));
            let admissible = dm.iter().map(|&m| i64::from(m)).sum::<i64>() == 0
                && (0..3).all(|i| on_ladder(dj[i], dm[i]))
                && tri(dj[0], dj[1], dj[2]);
            if !admissible {
                assert_eq!(
                    wigner_3j(dj[0], dj[1], dj[2], dm[0], dm[1], dm[2]),
                    zero(),
                    "structurally forbidden 3j must be exact zero"
                );
                assert_eq!(
                    clebsch_gordan(dj[0], dm[0], dj[1], dm[1], dj[2], -dm[2]),
                    zero(),
                    "forbidden CG must be exact zero"
                );
                tested_3j += 1;
            }
        }
        assert!(
            tested_6j > 100,
            "too few forbidden 6j samples ({tested_6j})"
        );
        assert!(
            tested_3j > 100,
            "too few forbidden 3j samples ({tested_3j})"
        );
    }
}

/// Regge orbit invariance of the crate-private 6j key, checked against an
/// independent generator of the full 144-element symmetry group.
#[cfg(test)]
mod regge_orbit_tests {
    use super::{canonical_regge_3j, canonical_regge_6j, wigner_6j};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn tri(a: u32, b: u32, c: u32) -> bool {
        let (a, b, c) = (a as i64, b as i64, c as i64);
        (a + b + c) % 2 == 0 && c >= (a - b).abs() && c <= a + b
    }

    fn admissible_6j(dj: [u32; 6]) -> bool {
        tri(dj[0], dj[1], dj[2])
            && tri(dj[0], dj[4], dj[5])
            && tri(dj[3], dj[1], dj[5])
            && tri(dj[3], dj[4], dj[2])
    }

    /// The 24 tetrahedral symmetries of a 6j: permute the three columns
    /// `(j1,j4),(j2,j5),(j3,j6)` and flip top/bottom in an even number of columns.
    fn tetrahedral_images(dj: [u32; 6]) -> Vec<[u32; 6]> {
        let cols = [[dj[0], dj[3]], [dj[1], dj[4]], [dj[2], dj[5]]];
        let perms = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        // Row-flip patterns that flip an even number of columns (0 or 2): these are
        // the value-preserving classical symmetries.
        let flips = [
            [false, false, false],
            [true, true, false],
            [true, false, true],
            [false, true, true],
        ];
        let mut out = Vec::with_capacity(24);
        for p in perms {
            for f in flips {
                let mut c = [[0u32; 2]; 3];
                for slot in 0..3 {
                    let src = cols[p[slot]];
                    c[slot] = if f[slot] { [src[1], src[0]] } else { src };
                }
                out.push([c[0][0], c[1][0], c[2][0], c[0][1], c[1][1], c[2][1]]);
            }
        }
        out
    }

    /// The non-classical Regge symmetry generator: fix `(j1,j4)`, map each of
    /// `(j2,j3,j5,j6)` to `rho - j_i` with `rho = (j2+j3+j5+j6)/2`. Together with
    /// the 24 classical images this generates the full 144-element symmetry group.
    /// Returns `None` if an image spin would be negative (never for admissible
    /// input, but guard against underflow rather than panic).
    fn regge_map(dj: [u32; 6]) -> Option<[u32; 6]> {
        let s = (dj[1] as i64 + dj[2] as i64 + dj[4] as i64 + dj[5] as i64) / 2;
        let m = |x: u32| -> Option<u32> {
            let v = s - x as i64;
            (v >= 0).then_some(v as u32)
        };
        Some([dj[0], m(dj[1])?, m(dj[2])?, dj[3], m(dj[4])?, m(dj[5])?])
    }

    /// The full symmetry orbit of a 6j label set: closure under the 24 classical
    /// images and the Regge generator.
    fn full_orbit(dj: [u32; 6]) -> Vec<[u32; 6]> {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![dj];
        seen.insert(dj);
        while let Some(x) = stack.pop() {
            let mut neighbours = tetrahedral_images(x);
            if let Some(r) = regge_map(x) {
                neighbours.push(r);
            }
            for n in neighbours {
                if seen.insert(n) {
                    stack.push(n);
                }
            }
        }
        seen.into_iter().collect()
    }

    #[test]
    fn regge_orbit_key_and_value_invariant() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x8E66_1234);
        let mut tested = 0;
        let mut attempts = 0;
        let mut max_orbit = 0usize;
        while tested < 200 && attempts < 200_000 {
            attempts += 1;
            let dj = [
                rng.gen_range(0..=20),
                rng.gen_range(0..=20),
                rng.gen_range(0..=20),
                rng.gen_range(0..=20),
                rng.gen_range(0..=20),
                rng.gen_range(0..=20),
            ];
            if !admissible_6j(dj) {
                continue;
            }
            let key = canonical_regge_6j(dj[0], dj[1], dj[2], dj[3], dj[4], dj[5]).unwrap();
            let val = wigner_6j(dj[0], dj[1], dj[2], dj[3], dj[4], dj[5]);
            let orbit = full_orbit(dj);
            max_orbit = max_orbit.max(orbit.len());
            for img in &orbit {
                // Every image is itself an admissible 6j, so it has a key.
                let k2 =
                    canonical_regge_6j(img[0], img[1], img[2], img[3], img[4], img[5]).unwrap();
                let v2 = wigner_6j(img[0], img[1], img[2], img[3], img[4], img[5]);
                assert_eq!(
                    key, k2,
                    "Regge key differs on orbit image {img:?} of {dj:?}"
                );
                assert_eq!(
                    val.signed_square(),
                    v2.signed_square(),
                    "6j value differs on orbit image {img:?} of {dj:?}"
                );
            }
            tested += 1;
        }
        assert!(tested >= 50, "only {tested} orbits tested");
        // The Regge generator reaches non-classical images: a generic orbit exceeds
        // the 24 classical symmetries, confirming the extension is exercised.
        assert!(
            max_orbit > 24,
            "orbit never exceeded 24 images; Regge generator not exercised"
        );
    }

    #[test]
    fn regge_keys_never_panic_on_arbitrary_input() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x5EED_4E66);
        for _ in 0..2000 {
            let dj = |r: &mut ChaCha8Rng| r.gen_range(0u32..=40);
            let dm = |r: &mut ChaCha8Rng| r.gen_range(-50i64..=50);
            let (a, b, c, d, e, f) = (
                dj(&mut rng),
                dj(&mut rng),
                dj(&mut rng),
                dj(&mut rng),
                dj(&mut rng),
                dj(&mut rng),
            );
            let _ = canonical_regge_6j(a, b, c, d, e, f);
            let _ = canonical_regge_3j(a, b, c, dm(&mut rng), dm(&mut rng), dm(&mut rng));
        }
    }
}
