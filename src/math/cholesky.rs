use crate::array::repeat;
use crate::math::{Linear, Matrix, Vector, matrix};
use crate::{Array, ArrayInst, GenArray};

/// A cholesky decomposition of a subset of the rows/columns of a matrix.
#[derive(Clone)]
pub struct Cholesky<N: GenArray> {
    // The lower triangular factor. The inactive rows and columns should be set to
    // zero.
    data: Matrix<N, N>,
    // Which rows and columns are currently included in the decomposition.
    active: Array<N, bool>,
}

impl<N: GenArray> Cholesky<N> {
    /// An instance of [`Cholesky`] where no rows/columns are active.
    pub const EMPTY: Self = {
        Cholesky {
            data: Matrix::ZERO,
            active: repeat(false),
        }
    };

    /// Finds a Cholesky decomposition of the specified part of the matrix.
    ///
    /// This assumes the active part of the matrix is symmetric positive definite.
    pub fn factor(source: &Matrix<N, N>, active: Array<N, bool>) -> Self {
        let mut this = Cholesky {
            data: Matrix::ZERO,
            active,
        };

        let active = active.as_slice();

        for i in (0..N::LEN).filter(|&i| active[i]) {
            for j in (0..i).filter(|&j| active[j]) {
                let mut v = source[(i, j)];
                for k in 0..j {
                    // Note: visiting inactive columns is fine as their values are zero
                    v -= this.data[(i, k)] * this.data[(j, k)];
                }
                this.data[(i, j)] = v / this.data[(j, j)];
            }

            let mut v = source[(i, i)];
            for k in 0..i {
                // Note: visiting inactive columns is fine as their values are zero
                v -= this.data[(i, k)] * this.data[(i, k)];
            }
            this.data[(i, i)] = libm::sqrt(v);
        }

        this
    }

    /// The active rows/columns in `self`.
    pub fn active(&self) -> &Array<N, bool> {
        &self.active
    }

    /// Tests if a specified row/column is active.
    pub fn is_active(&self, index: usize) -> bool {
        self.active.as_slice()[index]
    }

    /// Gets the lower triangular matrix `L`. The inactive rows and columns are set to zero.
    pub fn lower(&self) -> &Matrix<N, N> {
        &self.data
    }

    /// Solves for `A` in the equation `L^T * A = B`, where `B` is replaced with `A`
    /// in place.
    ///
    /// This does not affect the elements in the inactive rows.
    pub fn upper_solve<M: GenArray>(&self, rhs: &mut Matrix<N, M>) {
        // Solve using backward substitution
        for i in (0..N::LEN).rev().filter(|i| self.is_active(*i)) {
            let row = rhs.row(i) / self.data[(i, i)];
            rhs.set_row(i, row);
            for j in 0..i {
                // Note: visiting inactive columns is fine as their values are zero
                rhs.set_row(j, rhs.row(j) - row * self.data[(i, j)]);
            }
        }
    }

    /// Solves for `A` in the equation `L * A = B`, where `B` is replaced with `A`
    /// in place.
    ///
    /// This does not affect the elements in the inactive rows.
    pub fn lower_solve<M: GenArray>(&self, rhs: &mut Matrix<N, M>) {
        // Solve using forward substitution
        for i in (0..N::LEN).filter(|&i| self.is_active(i)) {
            let row = rhs.row(i) / self.data[(i, i)];
            rhs.set_row(i, row);
            for j in i + 1..N::LEN {
                // Note: visiting inactive rows is fine as their values are zero
                rhs.set_row(j, rhs.row(j) - row * self.data[(j, i)]);
            }
        }
    }

    /// Solves for `A` in the equation `L * L^T * A = B`, where `B` is replaced with
    /// `A` in place.
    ///
    /// This does not affect the elements in the inactive rows.
    pub fn solve<M: GenArray>(&self, rhs: &mut Matrix<N, M>) {
        self.lower_solve(rhs);
        self.upper_solve(rhs);
    }

    /// Solves for `x` in the equation `L * L^T * x = b`.
    ///
    /// This does not affect the elements in the inactive rows.
    pub fn solve_vec(&self, rhs: Vector<N>) -> Vector<N> {
        let mut mat = matrix([rhs.0]).transpose();
        self.solve(&mut mat);
        mat.col(0)
    }

    /// Computes `L^T * A`, modifying `A` in place.
    ///
    /// This does not affect the elements in the inactive rows.
    pub fn upper_mul<M: GenArray>(&self, rhs: &mut Matrix<N, M>) {
        for i in (0..N::LEN).filter(|i| self.is_active(*i)) {
            let row = rhs.row(i);
            rhs.set_row(i, row * self.data[(i, i)]);
            for j in 0..i {
                // Note: visiting inactive columns is fine as their values are zero
                rhs.set_row(j, rhs.row(j) + row * self.data[(i, j)]);
            }
        }
    }

    /// Computes `L * A`, modifying `A` in place.
    ///
    /// This does not affect the elements in the inactive rows.
    pub fn lower_mul<M: GenArray>(&self, rhs: &mut Matrix<N, M>) {
        for i in (0..N::LEN).rev().filter(|&i| self.is_active(i)) {
            let row = rhs.row(i);
            rhs.set_row(i, row * self.data[(i, i)]);
            for j in i + 1..N::LEN {
                // Note: visiting inactive rows is fine as their values are zero
                rhs.set_row(j, rhs.row(j) + row * self.data[(j, i)]);
            }
        }
    }

    /// Computes `L * L^T * A`, modifying `A` in place.
    ///
    /// This does not affect the elements in the inactive rows.
    pub fn mul<M: GenArray>(&self, rhs: &mut Matrix<N, M>) {
        self.upper_mul(rhs);
        self.lower_mul(rhs);
    }
}

#[cfg(test)]
mod tests {
    use crate::math::cholesky::Cholesky;
    use crate::math::matrix;

    #[test]
    fn factor() {
        let mat = matrix([
            [7.06, -1.29, 3.09, 2.57, 1.22],
            [-1.29, 3.79, 0.11, -2.78, -1.40],
            [3.09, 0.11, 3.99, 1.72, -0.65],
            [2.57, -2.78, 1.72, 5.18, 0.38],
            [1.22, -1.40, -0.65, 0.38, 1.15],
        ]);
        let n = mat.0.len();

        // Test all possible combinations of active rows.
        for i in 0..1 << n {
            let active = core::array::from_fn(|j| i & (1 << j) != 0);

            let chol = Cholesky::factor(&mat, active);
            assert_eq!(chol.active(), &active);

            let lower = *chol.lower();
            let product = lower * lower.transpose();

            for i in 0..n {
                for j in 0..n {
                    if active[i] && active[j] {
                        // The active part of the product should be equal to the same part of the
                        // original matrix (except for rounding errors).
                        let delta = mat[(i, j)] - product[(i, j)];
                        assert!(delta.abs() <= 1e-10);
                    } else {
                        // Inactive rows/columns should be zero.
                        assert_eq!(lower[(i, j)], 0.0);
                    }

                    if i < j {
                        // The decomposition should be lower triangular.
                        assert_eq!(lower[(i, j)], 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn solve() {
        let mat = matrix([
            [7.34, 5.83, 2.75, 1.48, -1.88],
            [5.83, 6.40, 1.11, -0.34, -2.32],
            [2.75, 1.11, 9.42, -4.20, 1.40],
            [1.48, -0.34, -4.20, 6.46, -0.77],
            [-1.88, -2.32, 1.40, -0.77, 1.62],
        ]);

        let rhs = matrix([
            [1.08, 0.9, -0.07],
            [1.94, -0.51, 0.29],
            [0.76, -1.12, 1.88],
            [0.32, -0.13, -0.5],
            [0.72, -0.82, -0.73],
        ]);

        let n = mat.0.len();
        let c = rhs.0[0].len();

        // Test all possible combinations of active rows.
        for i in 0..1 << n {
            let active = core::array::from_fn(|j| i & (1 << j) != 0);

            let chol = Cholesky::factor(&mat, active);
            let lower = *chol.lower();

            let mut vec = rhs;
            chol.solve(&mut vec);

            let product = lower * (lower.transpose() * vec);
            for i in 0..n {
                for j in 0..c {
                    if active[i] {
                        // The active part of the product should be correct (except for rounding errors).
                        let delta = rhs[(i, j)] - product[(i, j)];
                        assert!(delta.abs() <= 1e-10);
                    } else {
                        // The inactive part should be unaffected.
                        assert_eq!(vec[(i, j)], rhs[(i, j)])
                    }
                }
            }
        }
    }

    #[test]
    fn multiply() {
        let mat = matrix([
            [7.86, 7.32, 1.21, -3.58, 2.92],
            [7.32, 8.45, -0.16, -4.41, 3.96],
            [1.21, -0.16, 2.42, -0.04, 0.21],
            [-3.58, -4.41, -0.04, 3.23, -3.44],
            [2.92, 3.96, 0.21, -3.44, 4.36],
        ]);

        let rhs = matrix([
            [-1.69, -2.23, 0.31],
            [-2.20, 1.32, 0.31],
            [-1.48, -2.17, -1.34],
            [-1.52, -0.85, -0.38],
            [-0.22, 1.06, 0.59],
        ]);

        let n = mat.0.len();
        let c = rhs.0[0].len();

        // Test all possible combinations of active rows.
        for i in 0..1 << n {
            let active = core::array::from_fn(|j| i & (1 << j) != 0);

            let chol = Cholesky::factor(&mat, active);
            let lower = *chol.lower();

            let mut vec = rhs;
            let expected = lower * (lower.transpose() * vec);
            chol.mul(&mut vec);

            for i in 0..n {
                for j in 0..c {
                    if active[i] {
                        // The active part of the product should be correct (except for rounding errors).
                        let delta = vec[(i, j)] - expected[(i, j)];
                        assert!(delta.abs() <= 1e-10);
                    } else {
                        // The inactive part should be unaffected.
                        assert_eq!(vec[(i, j)], rhs[(i, j)])
                    }
                }
            }
        }
    }
}
