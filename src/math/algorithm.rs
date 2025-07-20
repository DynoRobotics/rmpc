use crate::GenArray;
use crate::math::Matrix;

/// Scales a row in the matrix with the specified factor.
fn row_multiplication<R: GenArray, C: GenArray>(matrix: &mut Matrix<R, C>, row: usize, scale: f64) {
    for col in 0..C::LEN {
        matrix[(row, col)] *= scale;
    }
}

/// Adds a scaled version of one row to another row.
fn row_addition<R: GenArray, C: GenArray>(
    matrix: &mut Matrix<R, C>,
    source: usize,
    dest: usize,
    scale: f64,
) {
    for col in 0..C::LEN {
        matrix[(dest, col)] += scale * matrix[(source, col)];
    }
}

/// Computes `inv(lhs)` using Gaussian elimination without pivoting.
///
/// A sufficient criterion for this to work, assuming no rounding errors, is the
/// matrix being block triangular with positive definite matrices on the
/// main diagonal.
///
/// ```
/// # use mpc::math::{Matrix, inv_no_pivot, matrix};
/// // A positive semidefinite matrix, fulfills the criteria.
/// let good_matrix = matrix([
///     [1.0, 0.5, 0.7],
///     [0.5, 0.5, 0.0],
///     [0.7, 0.0, 1.0],
/// ]);
/// let zero = inv_no_pivot(good_matrix) * good_matrix - Matrix::IDENTITY;
/// assert!(zero.0.iter().flatten().all(|elem| elem.abs() <= 1e-10));
///
/// // Does not meet the criteria, which in this case happens to break the
/// // algorithm even though the matrix is invertible.
/// let bad_matrix = matrix([
///     [1.0, 0.5, 1.0],
///     [2.0, 1.0, 2.5],
///     [1.5, 1.0, 0.5],
/// ]);
/// let failed_inverse = inv_no_pivot(bad_matrix);
/// assert!(failed_inverse.0.iter().flatten().any(|elem| elem.is_nan()));
///
/// let inverse = matrix([
///     [ 16.0, -6.0, -2.0],
///     [-22.0,  8.0,  4.0],
///     [ -4.0,  2.0,  0.0],
/// ]);
/// let zero = inverse * bad_matrix - Matrix::IDENTITY;
/// assert!(zero.0.iter().flatten().all(|elem| elem.abs() <= 1e-10));
/// ```
pub fn inv_no_pivot<N: GenArray>(matrix: Matrix<N, N>) -> Matrix<N, N> {
    let mut lhs = matrix;
    let mut rhs = Matrix::IDENTITY;

    // To do: This should be doable with less operations as the RHS is known to
    // start out as the identity matrix.
    for pivot in 0..N::LEN {
        let scale = 1.0 / lhs[(pivot, pivot)];
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
