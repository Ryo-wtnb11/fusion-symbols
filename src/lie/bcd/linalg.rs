//! Dense B/C/D factorizations and contractions on t4a-faer, with explicit
//! Par::Seq and scratch workspaces. Backend types stay inside this seam.
//! Ordered CGS2 selection precedes full-rank positive-diagonal QR (§4a).

use super::sweep::SweepError;
use faer::{
    diag::Diag,
    dyn_stack::{MemBuffer, MemStack},
    linalg::{
        householder,
        qr::no_pivoting::factor,
        svd::{self, ComputeSvdVectors},
    },
    Conj, Mat as FaerMat, MatRef, Par,
};

/// A column-major dense `rows × cols` matrix of `f64` — the plain buffer the
/// rest of the sweep manipulates so no backend type escapes this module.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Dense {
    pub rows: usize,
    pub cols: usize,
    /// Column-major: element `(i, j)` at `data[i + j * rows]`.
    pub data: Vec<f64>,
}

impl Dense {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Dense {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i + j * self.rows]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i + j * self.rows] = v;
    }

    /// Column `j` as a slice.
    pub fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.rows..(j + 1) * self.rows]
    }

    /// The `k`-th column of the identity, as a `rows × 1` matrix.
    pub fn unit(rows: usize, k: usize) -> Self {
        let mut m = Dense::zeros(rows, 1);
        m.data[k] = 1.0;
        m
    }

    /// Conjugate (real: plain) transpose.
    pub fn transpose(&self) -> Dense {
        let mut t = Dense::zeros(self.cols, self.rows);
        for j in 0..self.cols {
            for i in 0..self.rows {
                t.data[j + i * self.cols] = self.data[i + j * self.rows];
            }
        }
        t
    }

    /// Append the columns of `other` (same `rows`) to the right.
    pub fn cat_cols(&mut self, other: &Dense) {
        debug_assert_eq!(self.rows, other.rows, "cat_cols row mismatch");
        self.data.extend_from_slice(&other.data);
        self.cols += other.cols;
    }

    /// A view of `self` keeping only the columns in `keep` (in order).
    pub fn select_cols(&self, keep: &[usize]) -> Dense {
        let mut out = Dense::zeros(self.rows, keep.len());
        for (jo, &j) in keep.iter().enumerate() {
            out.data[jo * self.rows..(jo + 1) * self.rows].copy_from_slice(self.col(j));
        }
        out
    }

    /// Frobenius norm.
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

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
        Par::Seq,
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

fn view(a: &Dense) -> MatRef<'_, f64> {
    MatRef::from_column_major_slice(&a.data, a.rows, a.cols)
}

fn dense(a: MatRef<'_, f64>) -> Dense {
    Dense {
        rows: a.nrows(),
        cols: a.ncols(),
        data: a.col_iter().flat_map(|c| c.iter().copied()).collect(),
    }
}

/// `A · B` (plain matmul), routed through the t4a-faer sequential kernels.
pub(crate) fn matmul(a: &Dense, b: &Dense) -> Result<Dense, SweepError> {
    debug_assert_eq!(a.cols, b.rows, "matmul inner-dim mismatch");
    if a.rows == 0 || b.cols == 0 || a.cols == 0 {
        return Ok(Dense::zeros(a.rows, b.cols));
    }
    if a.cols != b.rows {
        return Err(SweepError::Linalg(
            "matmul: inner-dimension mismatch".into(),
        ));
    }
    let mut c = FaerMat::zeros(a.rows, b.cols);
    faer::linalg::matmul::matmul(
        c.as_mut(),
        faer::Accum::Replace,
        view(a),
        view(b),
        1.0,
        Par::Seq,
    );
    Ok(dense(c.as_ref()))
}

/// `Aᵀ · B` (`A†B` for real `A`), the projection contraction. One matmul.
pub(crate) fn tmatmul(a: &Dense, b: &Dense) -> Result<Dense, SweepError> {
    matmul(&a.transpose(), b)
}

/// Thin SVD `A = U · diag(s) · Vᵀ`, routed through the t4a-faer sequential kernels.
///
/// Returns `(U, s, Vt)` with `U` (`rows × k`), `s` (length `k`), `Vt` (`k × cols`),
/// `k = min(rows, cols)`. The only consumer is the small per-weight-space
/// orthogonal Procrustes solve in intertwiner alignment: for a square
/// `M`, `W = U · Vt` is the nearest orthogonal matrix to `M`. No hand-rolled
/// kernel — the decision to route factorizations through the backend is `AGENTS.md`.
pub(crate) fn svd(a: &Dense) -> Result<(Dense, Vec<f64>, Dense), SweepError> {
    let k = a.rows.min(a.cols);
    if k == 0 {
        return Ok((Dense::zeros(a.rows, 0), Vec::new(), Dense::zeros(0, a.cols)));
    }
    let mut u = FaerMat::zeros(a.rows, k);
    let mut v = FaerMat::zeros(a.cols, k);
    let mut values = Diag::zeros(k);
    let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(
        a.rows,
        a.cols,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        Par::Seq,
        Default::default(),
    ));
    svd::svd(
        view(a),
        values.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        Par::Seq,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .map_err(|e| SweepError::Linalg(format!("svd: {e:?}")))?;
    Ok((
        dense(u.as_ref()),
        values.column_vector().iter().copied().collect(),
        dense(v.transpose()),
    ))
}

#[cfg(test)]
std::thread_local! {
    static QR_INPUTS: std::cell::RefCell<Option<Vec<Dense>>> = const { std::cell::RefCell::new(None) };
}

/// In-order CGS2 selection, followed by the unique positive-diagonal thin QR
/// of the selected full-rank columns (docs/gauge_soN.md §4a).
pub(crate) fn qr_positive_q(a: &Dense, tol: f64) -> Result<Dense, SweepError> {
    #[cfg(test)]
    QR_INPUTS.with_borrow_mut(|inputs| {
        if let Some(inputs) = inputs {
            inputs.push(a.clone());
        }
    });
    let mut basis = Dense::zeros(a.rows, 0);
    let mut keep = Vec::new();
    for j in 0..a.cols {
        let mut r = a.select_cols(&[j]);
        for _ in 0..2 {
            let projection = matmul(&basis, &tmatmul(&basis, &r)?)?;
            for (x, p) in r.data.iter_mut().zip(projection.data) {
                *x -= p;
            }
        }
        let norm = r.norm();
        if norm > tol {
            keep.push(j);
            for x in &mut r.data {
                *x /= norm;
            }
            basis.cat_cols(&r);
        }
    }
    if keep.is_empty() {
        return Ok(basis);
    }
    let selected = a.select_cols(&keep);
    let (packed, coeff) = qr(view(&selected));
    Ok(dense(positive_q(packed.as_ref(), coeff.as_ref()).as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d3_epoch3_qr_uses_independent_input_columns() {
        use crate::lie::bcd::{CanonicalCatalog, Irrep, Series};
        QR_INPUTS.with_borrow_mut(|inputs| *inputs = Some(Vec::new()));
        let a = Irrep::from_dynkin(Series::D, &[0, 1, 1]).unwrap();
        let c = Irrep::from_dynkin(Series::D, &[0, 2, 2]).unwrap();
        let mut catalog = CanonicalCatalog::new(Series::D, 3).unwrap();
        catalog.cgc(&a, &a, &c).unwrap();
        let inputs = QR_INPUTS.with_borrow_mut(Option::take).unwrap();
        // Actual canonical-parent descent input behind coupled column 31 (#150).
        let input = inputs
            .iter()
            .find(|a| a.rows == 100 && a.cols == 5 && a.at(7, 4) > 1.4 && a.at(70, 0) > 1.4)
            .expect("affected descent block must be exercised");

        // Independent scalar CGS2 oracle selects columns; no frozen coefficients.
        let mut basis: Vec<Vec<f64>> = Vec::new();
        let mut keep = Vec::new();
        for j in 0..input.cols {
            let mut v = input.col(j).to_vec();
            for _ in 0..2 {
                let overlaps: Vec<f64> = basis
                    .iter()
                    .map(|b| b.iter().zip(&v).map(|(x, y)| x * y).sum())
                    .collect();
                for (i, x) in v.iter_mut().enumerate() {
                    *x -= basis
                        .iter()
                        .zip(&overlaps)
                        .map(|(b, p)| b[i] * p)
                        .sum::<f64>();
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                keep.push(j);
                basis.push(v.iter().map(|x| x / norm).collect());
            }
        }
        assert_eq!(keep, [0, 1, 2, 4]);
        let selected = input.select_cols(&keep);
        let (packed, coeff) = qr(view(&selected));
        let expected = dense(positive_q(packed.as_ref(), coeff.as_ref()).as_ref());
        let actual = qr_positive_q(input, 1e-10).unwrap();
        assert_eq!((actual.rows, actual.cols), (100, 4));
        for (x, y) in actual.data.iter().zip(&expected.data) {
            assert!((x - y).abs() < 1e-12);
        }
        // Independently certify positive-diagonal thin QR, including its sign.
        let r = tmatmul(&actual, &selected).unwrap();
        let gram = tmatmul(&actual, &actual).unwrap();
        for i in 0..4 {
            assert!(r.at(i, i) > 1e-10);
            for j in 0..4 {
                assert!((gram.at(i, j) - f64::from(i == j)).abs() < 1e-12);
                if i > j {
                    assert!(r.at(i, j).abs() < 1e-12);
                }
            }
        }
        let reconstructed = matmul(&actual, &r).unwrap();
        for (x, y) in reconstructed.data.iter().zip(&selected.data) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn ordered_qr_skips_zero_and_nontrailing_dependent_columns() {
        let independent = Dense {
            rows: 4,
            cols: 3,
            data: vec![1., 2., 0., 0., 0., 1., 2., 0., 1., 0., 0., 3.],
        };
        let (packed, coeff) = qr(view(&independent));
        let expected = dense(positive_q(packed.as_ref(), coeff.as_ref()).as_ref());
        for columns in [
            vec![
                vec![0.; 4],
                vec![1., 2., 0., 0.],
                vec![2., 4., 0., 0.],
                vec![0., 1., 2., 0.],
                vec![1., 0., 0., 3.],
            ],
            vec![
                vec![1., 2., 0., 0.],
                vec![0., 1., 2., 0.],
                vec![1., 3., 2., 0.],
                vec![1., 0., 0., 3.],
            ],
            vec![
                vec![1., 2., 0., 0.],
                vec![0., 1., 2., 0.],
                vec![1., 0., 0., 3.],
                vec![2., 4., 0., 0.],
                vec![1., 3., 2., 0.],
            ],
        ] {
            let input = Dense {
                rows: 4,
                cols: columns.len(),
                data: columns.concat(),
            };
            let actual = qr_positive_q(&input, 1e-10).unwrap();
            assert_eq!((actual.rows, actual.cols), (4, 3));
            for (a, b) in actual.data.iter().zip(&expected.data) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        let two = independent.select_cols(&[0, 1]);
        let mut dependent = independent.select_cols(&[0]);
        let mut copy = dependent.clone();
        for x in &mut copy.data {
            *x *= 2.0;
        }
        dependent.cat_cols(&copy);
        dependent.cat_cols(&independent.select_cols(&[1]));
        let actual = qr_positive_q(&dependent, 1e-10).unwrap();
        let (packed, coeff) = qr(view(&two));
        let expected_two = dense(positive_q(packed.as_ref(), coeff.as_ref()).as_ref());
        assert_eq!(actual.cols, 2);
        for (a, b) in actual.data.iter().zip(expected_two.data) {
            assert!((a - b).abs() < 1e-12);
        }
        for input in [Dense::zeros(4, 2), Dense::zeros(0, 2), Dense::zeros(4, 0)] {
            assert_eq!(qr_positive_q(&input, 1e-10).unwrap().cols, 0);
        }
        let threshold = Dense {
            rows: 2,
            cols: 2,
            data: vec![1., 0., 0., 1e-10],
        };
        assert_eq!(qr_positive_q(&threshold, 1e-10).unwrap().cols, 1);
    }

    #[test]
    fn svd_reconstructs_and_procrustes_recovers_rotation() {
        // A known 2×2, plus: the nearest-orthogonal of a rotation IS that rotation.
        let m = Dense {
            rows: 2,
            cols: 2,
            data: vec![2.0, 0.0, 0.0, 3.0], // diag(2,3), column-major
        };
        let (u, s, vt) = svd(&m).unwrap();
        // reconstruct U·diag(s)·Vt.
        let mut sd = Dense::zeros(2, 2);
        for (i, &sv) in s.iter().enumerate() {
            sd.set(i, i, sv);
        }
        let recon = matmul(&matmul(&u, &sd).unwrap(), &vt).unwrap();
        for i in 0..4 {
            assert!((recon.data[i] - m.data[i]).abs() < 1e-12);
        }
        // Procrustes on a pure rotation returns the rotation itself (s ≈ [1,1]).
        let c = std::f64::consts::FRAC_1_SQRT_2;
        let rot = Dense {
            rows: 2,
            cols: 2,
            data: vec![c, c, -c, c], // [[c,-c],[c,c]] column-major
        };
        let (u, _s, vt) = svd(&rot).unwrap();
        let w = matmul(&u, &vt).unwrap();
        for i in 0..4 {
            assert!((w.data[i] - rot.data[i]).abs() < 1e-12);
        }
    }
}
