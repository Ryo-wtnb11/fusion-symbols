//! Dense SU(N) factorizations on t4a-faer, using explicit sequential paths and
//! scratch workspaces. Plain column-major f64 buffers keep backend types private.
//! SVD with full V supplies the right nullspace; unpivoted QR supplies gauge fixing
//! and least squares. Zero R diagonals are left unflipped (docs/gauge.md §4b).

use super::SunError;
use faer::{
    diag::Diag,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{
        householder,
        qr::no_pivoting::{factor, solve},
        svd::{self, ComputeSvdVectors},
    },
    Conj, Mat as FaerMat, MatRef, Par, ParShape,
};

/// Unpivoted QR, explicitly sequential. The packed lower triangle stores Q's reflectors.
fn qr(a: MatRef<'_, f64>) -> (FaerMat<f64>, FaerMat<f64>) {
    let (m, n) = a.shape();
    let block = factor::recommended_block_size::<f64>(m, n);
    let mut packed = a.to_owned();
    let mut coeff = FaerMat::zeros(block, m.min(n));
    let mut mem = MemBuffer::new(factor::qr_in_place_scratch::<f64>(
        m,
        n,
        block,
        ParShape::Seq,
        Default::default(),
    ));
    factor::qr_in_place(
        packed.as_mut(),
        coeff.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    );
    (packed, coeff)
}

fn positive_q(packed: MatRef<'_, f64>, coeff: MatRef<'_, f64>) -> FaerMat<f64> {
    let (m, n) = packed.shape();
    let k = m.min(n);
    let mut q = FaerMat::identity(m, k);
    let mut mem = MemBuffer::new(
        householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
            m,
            coeff.nrows(),
            k,
        ),
    );
    householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        packed.subcols(0, k),
        coeff,
        Conj::No,
        q.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
    );
    for j in 0..k {
        // An exactly zero R diagonal (including -0.0) is left unflipped.
        if packed[(j, j)] < 0.0 {
            for i in 0..m {
                q[(i, j)] = -q[(i, j)];
            }
        }
    }
    q
}

/// A column-major dense matrix of shape `rows x cols`.
pub(crate) struct Mat {
    pub rows: usize,
    pub cols: usize,
    /// Column-major: element `(i, j)` at `data[i + j * rows]`.
    pub data: Vec<f64>,
}

impl Mat {
    pub fn try_zeros(rows: usize, cols: usize) -> Result<Self, SunError> {
        Ok(Mat {
            rows,
            cols,
            data: vec![
                0.0;
                crate::lie::checked_size([rows, cols])
                    .map_err(|_| SunError::LabelOverflow)?
            ],
        })
    }

    #[inline]
    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        self.data[i + j * self.rows] += v;
    }

    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i + j * self.rows]
    }

    fn view(&self) -> MatRef<'_, f64> {
        MatRef::from_column_major_slice(&self.data, self.rows, self.cols)
    }
}

/// Right nullspace basis of `a` at absolute singular-value tolerance `atol`,
/// returned as the columns of an `a.cols x k` matrix (`k = cols - rank`).
///
/// Ports `_nullspace!(A; atol)`: `rank = #{ s_i > atol }`, and the nullspace is
/// the trailing `Vh` rows (right singular vectors of zero singular value). With
/// `atol > 0` the reference's relative term vanishes (`rtol = … * iszero(atol)`),
/// so the cut is purely `s_i > atol`.
pub(crate) fn nullspace(a: &Mat, atol: f64) -> Result<Mat, SunError> {
    let n = a.cols;
    // Reference `_nullspace!`: an empty system (no rows or no columns) has the
    // whole space as its nullspace -> the n x n identity. Also avoids handing a
    // 0-row matrix to SVD.
    if a.rows == 0 || n == 0 {
        let mut id = Mat::try_zeros(n, n)?;
        for i in 0..n {
            id.data[i + i * n] = 1.0;
        }
        return Ok(id);
    }
    crate::lie::checked_size([n, n]).map_err(|_| SunError::LabelOverflow)?;
    let mut v = FaerMat::zeros(n, n);
    let mut sv = Diag::zeros(a.rows.min(n));
    let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(
        a.rows,
        n,
        ComputeSvdVectors::No,
        ComputeSvdVectors::Full,
        ParShape::Seq,
        Default::default(),
    ));
    svd::svd(
        a.view(),
        sv.as_mut(),
        None,
        Some(v.as_mut()),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .map_err(|e| SunError::Linalg(format!("svd: {e:?}")))?;
    let sv = sv.column_vector();
    let rank = sv.iter().filter(|&&x| x > atol).count();
    let k = n - rank;
    // Nullspace vector alpha is column (rank + alpha) of the full V.
    let mut ns = Mat::try_zeros(n, k)?;
    for alpha in 0..k {
        let t = rank + alpha;
        for j in 0..n {
            ns.data[j + alpha * n] = v[(j, t)];
        }
    }
    Ok(ns)
}

/// Positive-diagonal `Q` of the thin QR of `a` (`qrpos!` returning `first`).
///
/// Ports `qrpos!(C)`: QR with the R-diagonal signs folded into `Q` so each
/// `R[i,i] >= 0`; the gauge-fixing step keeps only `Q` (`gaugefix! = first ∘
/// qrpos! ∘ cref!`). `Q` has shape `a.rows x min(a.rows, a.cols)`.
pub(crate) fn qr_positive_q(a: &Mat) -> Result<Mat, SunError> {
    // Callers supply a nullspace basis transformed by cref, hence full rank.
    // Verify with the same absolute CGS2 residual test as the B/C/D seam.
    let mut basis = FaerMat::<f64>::zeros(a.rows, a.cols);
    for j in 0..a.cols {
        let mut r = a.view().subcols(j, 1).to_owned();
        let q = basis.as_ref().subcols(0, j);
        for _ in 0..2 {
            let mut weights = FaerMat::<f64>::zeros(j, 1);
            faer::linalg::matmul::matmul(
                weights.as_mut(),
                faer::Accum::Replace,
                q.transpose(),
                r.as_ref(),
                1.0,
                Par::Seq,
            );
            faer::linalg::matmul::matmul(
                r.as_mut(),
                faer::Accum::Add,
                q,
                weights.as_ref(),
                -1.0,
                Par::Seq,
            );
        }
        let norm = r.col(0).iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm <= 1e-10 || !norm.is_finite() {
            return Err(SunError::Linalg(
                "qr_positive_q: input is not full column rank".into(),
            ));
        }
        for i in 0..a.rows {
            basis[(i, j)] = r[(i, 0)] / norm;
        }
    }
    let (packed, coeff) = qr(a.view());
    let q = positive_q(packed.as_ref(), coeff.as_ref());
    Ok(Mat {
        rows: a.rows,
        cols: q.ncols(),
        data: q.col_iter().flat_map(|c| c.iter().copied()).collect(),
    })
}

#[cfg(test)]
thread_local! {
    pub(super) static FACTORIZATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Packed unpivoted QR reused by the RHS blocks of one target-weight plan.
pub(crate) struct LeastSquares {
    packed: FaerMat<f64>,
    coeff: FaerMat<f64>,
    mem: MemBuffer,
}

impl LeastSquares {
    pub fn factor(a: &Mat, b: &Mat) -> Result<Self, SunError> {
        if a.rows < a.cols || b.rows != a.rows {
            return Err(SunError::Linalg(
                "lstsq: expected tall/square A and matching RHS rows".into(),
            ));
        }
        #[cfg(test)]
        FACTORIZATIONS.set(FACTORIZATIONS.get() + 1);
        let (packed, coeff) = qr(a.view());
        if (0..a.cols).any(|i| packed[(i, i)] == 0.0) {
            return Err(SunError::Linalg("lstsq: singular R diagonal".into()));
        }
        let mem = MemBuffer::new(solve::solve_lstsq_in_place_scratch::<f64>(
            a.rows,
            a.cols,
            coeff.nrows(),
            b.cols,
            ParShape::Seq,
        ));
        Ok(Self { packed, coeff, mem })
    }

    pub fn solve(&mut self, b: &Mat) -> Result<Mat, SunError> {
        let (m, n) = self.packed.shape();
        if b.rows != m {
            return Err(SunError::Linalg(
                "lstsq: expected tall/square A and matching RHS rows".into(),
            ));
        }
        let mut x = b.view().to_owned();
        let req = solve::solve_lstsq_in_place_scratch::<f64>(
            m,
            n,
            self.coeff.nrows(),
            b.cols,
            ParShape::Seq,
        );
        // RHS widths can differ between alpha columns; retain the largest scratch.
        if !MemStack::new(&mut self.mem).can_hold(req) {
            self.mem = MemBuffer::new(req);
        }
        solve::solve_lstsq_in_place(
            self.packed.as_ref(),
            self.coeff.as_ref(),
            self.packed.as_ref(),
            x.as_mut(),
            Par::Seq,
            MemStack::new(&mut self.mem),
        );
        Ok(Mat {
            rows: n,
            cols: b.cols,
            data: x
                .col_iter()
                .flat_map(|c| c.iter().take(n).copied())
                .collect(),
        })
    }
}

/// Least-squares solution `X` of `A X = B` for tall/square full-column-rank
/// `A` (`ldiv!(qr!(A), B)`). `X` has shape `A.cols x B.cols`.
#[cfg(test)]
pub(super) fn lstsq(a: &Mat, b: &Mat) -> Result<Mat, SunError> {
    if a.rows < a.cols || b.rows != a.rows {
        return Err(SunError::Linalg(
            "lstsq: expected tall/square A and matching RHS rows".into(),
        ));
    }
    let (packed, coeff) = qr(a.view());
    if (0..a.cols).any(|i| packed[(i, i)] == 0.0) {
        return Err(SunError::Linalg("lstsq: singular R diagonal".into()));
    }
    let mut x = b.view().to_owned();
    let mut mem = MemBuffer::new(solve::solve_lstsq_in_place_scratch::<f64>(
        a.rows,
        a.cols,
        coeff.nrows(),
        b.cols,
        ParShape::Seq,
    ));
    solve::solve_lstsq_in_place(
        packed.as_ref(),
        coeff.as_ref(),
        packed.as_ref(),
        x.as_mut(),
        Par::Seq,
        MemStack::new(&mut mem),
    );
    Ok(Mat {
        rows: a.cols,
        cols: b.cols,
        data: x
            .col_iter()
            .flat_map(|c| c.iter().take(a.cols).copied())
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nullspace_of_1x2_recovers_kernel() {
        // A = [1, 1] (1x2). Kernel spanned by (1, -1)/sqrt2. Thin SVD would drop
        // this; full SVD must recover it -- the minimal SU(2)
        // singlet system.
        let a = Mat {
            rows: 1,
            cols: 2,
            data: vec![1.0, 1.0],
        };
        let ns = nullspace(&a, 1e-13).unwrap();
        assert_eq!((ns.rows, ns.cols), (2, 1));
        let (v0, v1) = (ns.at(0, 0), ns.at(1, 0));
        assert!((v0 + v1).abs() < 1e-10, "A v = {}", v0 + v1);
        assert!((v0 * v0 + v1 * v1 - 1.0).abs() < 1e-10, "not unit");
    }

    #[test]
    fn qr_positive_q_is_orthonormal_with_positive_r_diagonal() {
        // Single column (2x1): Q must be that column normalized, oriented so the
        // (only) R diagonal is >= 0, i.e. Q points the same way as the input.
        let a = Mat {
            rows: 2,
            cols: 1,
            data: vec![-3.0, 4.0],
        };
        let q = qr_positive_q(&a).unwrap();
        assert_eq!((q.rows, q.cols), (2, 1));
        // R[0,0] = <q, a> must be >= 0.
        let r00 = q.at(0, 0) * a.at(0, 0) + q.at(1, 0) * a.at(1, 0);
        assert!(r00 >= 0.0, "R diagonal not positive: {r00}");
        assert!((q.at(0, 0).powi(2) + q.at(1, 0).powi(2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn qr_rejects_dependent_columns() {
        for (rows, data) in [(3, vec![1., 2., 3., 2., 4., 6.]), (1, vec![1., 2.])] {
            let a = Mat {
                rows,
                cols: 2,
                data,
            };
            assert!(matches!(qr_positive_q(&a), Err(SunError::Linalg(_))));
        }
    }

    #[test]
    fn lstsq_errors_match_original() {
        let mut a = Mat {
            rows: 2,
            cols: 1,
            data: vec![0.0; 2],
        };
        let mut b = Mat {
            rows: 1,
            cols: 1,
            data: vec![1.0],
        };
        // Bad RHS shape takes precedence over the singular-R error.
        assert_eq!(
            LeastSquares::factor(&a, &b).err(),
            Some(SunError::Linalg(
                "lstsq: expected tall/square A and matching RHS rows".into()
            ))
        );
        b.rows = 2;
        b.data.push(1.0);
        assert_eq!(
            LeastSquares::factor(&a, &b).err(),
            Some(SunError::Linalg("lstsq: singular R diagonal".into()))
        );
        a.data[0] = 1.0;
        let mut qr = LeastSquares::factor(&a, &b).unwrap();
        b.rows = 1;
        assert_eq!(
            qr.solve(&b).err(),
            Some(SunError::Linalg(
                "lstsq: expected tall/square A and matching RHS rows".into()
            ))
        );
        a.rows = 1;
        a.cols = 2;
        assert_eq!(
            LeastSquares::factor(&a, &b).err(),
            Some(SunError::Linalg(
                "lstsq: expected tall/square A and matching RHS rows".into()
            ))
        );
    }

    #[test]
    fn lstsq_recovers_consistent_tall_system() {
        // A (3x2) full column rank, b = A x_true -> lstsq recovers x_true.
        let a = Mat {
            rows: 3,
            cols: 2,
            data: vec![1.0, 1.0, 1.0, 0.0, 1.0, 2.0], // col-major
        };
        let x_true = [2.0, -1.0];
        let mut bdata = vec![0.0; 3];
        for (i, bi) in bdata.iter_mut().enumerate() {
            for (j, xj) in x_true.iter().enumerate() {
                *bi += a.at(i, j) * xj;
            }
        }
        let b = Mat {
            rows: 3,
            cols: 1,
            data: bdata,
        };
        let x = LeastSquares::factor(&a, &b).unwrap().solve(&b).unwrap();
        assert_eq!((x.rows, x.cols), (2, 1));
        assert!((x.at(0, 0) - 2.0).abs() < 1e-9);
        assert!((x.at(1, 0) + 1.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod arithmetic_tests {
    use super::*;
    #[test]
    fn matrix_dimension_bound() {
        assert!(matches!(
            Mat::try_zeros(usize::MAX, 2),
            Err(SunError::LabelOverflow)
        ));
        let a = Mat {
            rows: 0,
            cols: usize::MAX,
            data: vec![],
        };
        assert!(matches!(nullspace(&a, 1e-13), Err(SunError::LabelOverflow)));
    }
}
