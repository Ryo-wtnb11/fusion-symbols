//! The gates defined on completed rounded F and R blocks
//! (`docs/gauge_cartan.md` §7, gates 6 and 7): sanity checks of a rounded
//! block, never certificates of its rounding, and never changing a value.

use crate::category::{FBlock, RBlock, SymbolBlock};
use crate::exact::radical::RadicalBlock;
use crate::frcore::GateError;
use crate::rounding::{round_radical, Factor};

/// Gate 7: every entry of a completed F block is finite. A fixed
/// `F^{abc}_d[e, f]` is only a subblock of a change of basis, so it has no
/// unitarity of its own.
pub(crate) fn check_f_block(block: &FBlock<f64>) -> Result<(), GateError> {
    if block.as_slice().iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(GateError::NonFinite { gate: "F" })
    }
}

/// Gate 6: a completed R block is a change of orthonormal multiplicity basis:
/// every entry finite and `max |RᵀR − I| ≤ 1e-9`. The empty block passes.
pub(crate) fn check_r_block(block: &RBlock<f64>) -> Result<(), GateError> {
    const TOL_UNITARY: f64 = 1e-9;
    let ([n, _], r) = (block.dims(), block.as_slice());
    if !r.iter().all(|x| x.is_finite()) {
        return Err(GateError::NonFinite { gate: "R" });
    }
    let mut worst = 0.0f64;
    for mu in 0..n {
        for nu in mu..n {
            let dot: f64 = (0..n).map(|rho| r[rho * n + mu] * r[rho * n + nu]).sum();
            let residual = (dot - f64::from(u8::from(mu == nu))).abs();
            // An overflowed accumulation is nonfinite, which `max` would hide.
            if !residual.is_finite() {
                return Err(GateError::NonFinite { gate: "R" });
            }
            worst = worst.max(residual);
        }
    }
    if worst > TOL_UNITARY {
        return Err(GateError::Residual {
            gate: "R",
            residual: worst,
        });
    }
    Ok(())
}

/// Every entry of `block`, row-major, correctly rounded once
/// (`docs/gauge_cartan.md` §6.2): entry `i` is
/// `round_radical(sign(S_i), [|S_i|, |S_i|, common.num, axis[k][i_k].num…],
/// [common.den, axis[k][i_k].den…])` over the stored axes `k`. The factor
/// lists are fixed-size, so only the result allocates.
pub(crate) fn round_block<const R: usize>(block: &RadicalBlock<R>) -> Vec<f64> {
    let dims = block.dims();
    let nums = block.nums();
    let len = dims.iter().product::<usize>();
    let mut digits = [0usize; R];
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let s = nums.magnitude(i);
        let sign = if s.is_empty() {
            0
        } else if nums.is_negative(i) {
            -1
        } else {
            1
        };
        let common = block.common();
        let mut num = [Factor::Words(s); 7];
        let mut den = [Factor::Natural(common.den()); 5];
        num[2] = Factor::Natural(common.num());
        let (mut n, mut d) = (3, 1);
        for (k, &digit) in digits.iter().enumerate() {
            if let Some(r) = block.axis(k).get(digit) {
                num[n] = Factor::Natural(r.num());
                den[d] = Factor::Natural(r.den());
                (n, d) = (n + 1, d + 1);
            }
        }
        out.push(round_radical(sign, &num[..n], &den[..d]));
        for k in (0..R).rev() {
            digits[k] += 1;
            if digits[k] < dims[k] {
                break;
            }
            digits[k] = 0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: usize, data: &[f64]) -> Result<(), GateError> {
        check_r_block(&RBlock::from_vec(n, data.to_vec()))
    }

    /// R must be a finite orthogonal change of basis; F only finite.
    #[test]
    fn r_and_f_gates() {
        let (c, s) = (0.6, 0.8);
        assert_eq!(r(0, &[]), Ok(()));
        assert_eq!(r(1, &[1.0]), Ok(()));
        assert_eq!(r(1, &[-1.0]), Ok(()));
        assert_eq!(r(2, &[c, -s, s, c]), Ok(()));
        assert!(matches!(
            r(1, &[0.5]),
            Err(GateError::Residual { gate: "R", residual }) if residual == 0.75
        ));
        // Unit columns that are not orthogonal.
        assert!(matches!(
            r(2, &[1.0, c, 0.0, s]),
            Err(GateError::Residual { gate: "R", .. })
        ));
        for bad in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                r(2, &[c, bad, s, c]),
                Err(GateError::NonFinite { gate: "R" })
            );
            let f = FBlock::from_vec([1, 1, 1, 2], vec![0.25, bad]);
            assert_eq!(check_f_block(&f), Err(GateError::NonFinite { gate: "F" }));
        }
        // Finite entries whose Gram accumulation overflows.
        assert_eq!(r(1, &[1e200]), Err(GateError::NonFinite { gate: "R" }));
        // A fixed F[e, f] subblock need not be unitary.
        assert_eq!(
            check_f_block(&FBlock::from_vec([1, 1, 1, 1], vec![0.25])),
            Ok(())
        );
        assert_eq!(check_f_block(&FBlock::zeros([0, 1, 1, 1])), Ok(()));
    }
}
