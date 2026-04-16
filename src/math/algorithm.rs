use crate::GenArray;
use crate::math::{Float, Linear, Matrix};

/// Scales a row in the matrix with the specified factor.
fn row_multiplication<R: GenArray, C: GenArray>(
    matrix: &mut Matrix<R, C>,
    row: usize,
    scale: Float,
) {
    for col in 0..C::LEN {
        matrix[(row, col)] *= scale;
    }
}

/// Adds a scaled version of one row to another row.
fn row_addition<R: GenArray, C: GenArray>(
    matrix: &mut Matrix<R, C>,
    source: usize,
    dest: usize,
    scale: Float,
) {
    for col in 0..C::LEN {
        let src = matrix[(source, col)];
        matrix[(dest, col)] += scale * src;
    }
}

/// Computes `inv(lhs)` using Gaussian elimination without pivoting.
///
/// A sufficient criterion for this to work, assuming no rounding errors, is the
/// matrix being block triangular with positive definite matrices on the
/// main diagonal.
///
/// ```
/// # use rmpc::math::{Float, Matrix, inv_no_pivot, matrix};
/// // A positive semidefinite matrix, fulfills the criteria.
/// let good_matrix = matrix([
///     [1.0, 0.5, 0.7],
///     [0.5, 0.5, 0.0],
///     [0.7, 0.0, 1.0],
/// ].map(|r| r.map(Float::from)));
/// let zero = inv_no_pivot(good_matrix) * good_matrix - Matrix::IDENTITY;
/// assert!(zero.0.iter().flatten().all(|elem| elem.abs() <= 1e-10));
///
/// // Does not meet the criteria, which in this case happens to break the
/// // algorithm even though the matrix is invertible.
/// let bad_matrix = matrix([
///     [1.0, 0.5, 1.0],
///     [2.0, 1.0, 2.5],
///     [1.5, 1.0, 0.5],
/// ].map(|r| r.map(Float::from)));
/// let failed_inverse = inv_no_pivot(bad_matrix);
/// let not_zero = failed_inverse * bad_matrix - Matrix::IDENTITY;
/// assert!(not_zero.0.iter().flatten().any(|elem| elem.abs() >= 0.1));
///
/// let inverse = matrix([
///     [ 16.0, -6.0, -2.0],
///     [-22.0,  8.0,  4.0],
///     [ -4.0,  2.0,  0.0],
/// ].map(|r| r.map(Float::from)));
/// let zero = inverse * bad_matrix - Matrix::IDENTITY;
/// assert!(zero.0.iter().flatten().all(|elem| elem.abs() <= 1e-10));
/// ```
pub fn inv_no_pivot<N: GenArray>(matrix: Matrix<N, N>) -> Matrix<N, N> {
    let mut lhs = matrix;
    let mut rhs = Matrix::IDENTITY;

    // To do: This should be doable with less operations as the RHS is known to
    // start out as the identity matrix.
    for pivot in 0..N::LEN {
        // If the diagonal element is nonzero, eliminate it.
        let diag = lhs[(pivot, pivot)];
        let scale = if diag != Float::ZERO {
            1.0 / diag
        } else {
            Float::ZERO
        };

        row_multiplication(&mut lhs, pivot, scale);
        row_multiplication(&mut rhs, pivot, scale);

        for row in 0..N::LEN {
            if row != pivot {
                let scale = -lhs[(row, pivot)];
                row_addition(&mut lhs, pivot, row, scale);
                row_addition(&mut rhs, pivot, row, scale);
            }
        }
    }

    rhs
}
