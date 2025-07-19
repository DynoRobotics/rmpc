use std::array::from_fn;
use std::iter::zip;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::math::{Linear, Vector};

/// A dense matrix with a fixed size, stored in row major order.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<const R: usize, const C: usize>(pub [[f64; C]; R]);

impl<const R: usize, const C: usize> Matrix<R, C> {
    /// The transpose of `self`.
    pub fn transpose(self) -> Matrix<C, R> {
        Matrix(from_fn(|row| from_fn(|col| self[(col, row)])))
    }

    /// The `i`th row of `self`.
    pub fn row(self, i: usize) -> Vector<C> {
        Vector(self.0[i])
    }

    /// The `i`th row of `self`.
    pub fn col(self, i: usize) -> Vector<R> {
        Vector(self.0.map(|row| row[i]))
    }
}

impl<const N: usize> Matrix<N, N> {
    /// The identity matrix.
    pub const IDENTITY: Self = Self::from_diag([1.0; N]);

    /// A matrix with values on the main diagonals and zeros everywhere else.
    pub const fn from_diag(diag: [f64; N]) -> Self {
        let mut data = [[0.0; N]; N];
        let mut i = 0;
        while i < N {
            data[i][i] = diag[i];
            i += 1;
        }
        Matrix(data)
    }
}

impl<const R: usize, const C: usize> Neg for Matrix<R, C> {
    type Output = Self;

    fn neg(self) -> Self {
        Matrix(self.0.map(|row| row.map(Neg::neg)))
    }
}

impl<const R: usize, const C: usize> Add for Matrix<R, C> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl<const R: usize, const C: usize> AddAssign for Matrix<R, C> {
    fn add_assign(&mut self, rhs: Self) {
        for (dst, src) in zip(&mut self.0, &rhs.0) {
            for (dst, src) in zip(dst, src) {
                *dst += src;
            }
        }
    }
}

impl<const R: usize, const C: usize> Sub for Matrix<R, C> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}

impl<const R: usize, const C: usize> SubAssign for Matrix<R, C> {
    fn sub_assign(&mut self, rhs: Self) {
        for (dst, src) in zip(&mut self.0, &rhs.0) {
            for (dst, src) in zip(dst, src) {
                *dst -= src;
            }
        }
    }
}

impl<const R: usize, const C: usize> Mul<f64> for Matrix<R, C> {
    type Output = Self;

    fn mul(mut self, rhs: f64) -> Self {
        self *= rhs;
        self
    }
}

impl<const R: usize, const C: usize> MulAssign<f64> for Matrix<R, C> {
    fn mul_assign(&mut self, rhs: f64) {
        for dst in &mut self.0 {
            for dst in dst {
                *dst *= rhs;
            }
        }
    }
}

impl<const R: usize, const C: usize> Mul<Matrix<R, C>> for f64 {
    type Output = Matrix<R, C>;

    fn mul(self, mut rhs: Matrix<R, C>) -> Matrix<R, C> {
        rhs *= self;
        rhs
    }
}

impl<const R: usize, const C: usize> Mul<Vector<C>> for Matrix<R, C> {
    type Output = Vector<R>;

    fn mul(self, rhs: Vector<C>) -> Vector<R> {
        Vector(from_fn(|i| self.row(i) * rhs))
    }
}

impl<const R: usize, const C: usize> Mul<Matrix<R, C>> for Vector<R> {
    type Output = Vector<C>;

    fn mul(self, rhs: Matrix<R, C>) -> Vector<C> {
        Vector(from_fn(|i| self * rhs.col(i)))
    }
}

impl<const N: usize> MulAssign<Matrix<N, N>> for Vector<N> {
    fn mul_assign(&mut self, rhs: Matrix<N, N>) {
        *self = *self * rhs;
    }
}

impl<const A: usize, const B: usize, const C: usize> Mul<Matrix<B, C>> for Matrix<A, B> {
    type Output = Matrix<A, C>;

    fn mul(self, rhs: Matrix<B, C>) -> Matrix<A, C> {
        Matrix(from_fn(|i| (self.row(i) * rhs).0))
    }
}

impl<const R: usize, const C: usize> MulAssign<Matrix<C, C>> for Matrix<R, C> {
    fn mul_assign(&mut self, rhs: Matrix<C, C>) {
        *self = *self * rhs;
    }
}

impl<const R: usize, const C: usize> Div<f64> for Matrix<R, C> {
    type Output = Self;

    fn div(mut self, rhs: f64) -> Self {
        self /= rhs;
        self
    }
}

impl<const R: usize, const C: usize> DivAssign<f64> for Matrix<R, C> {
    fn div_assign(&mut self, rhs: f64) {
        for dst in &mut self.0 {
            for dst in dst {
                *dst /= rhs;
            }
        }
    }
}

impl<const R: usize, const C: usize> Linear for Matrix<R, C> {
    const ZERO: Self = Matrix([[0.0; C]; R]);
}

impl<const R: usize, const C: usize> Index<(usize, usize)> for Matrix<R, C> {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &f64 {
        &self.0[index.0][index.1]
    }
}

impl<const R: usize, const C: usize> IndexMut<(usize, usize)> for Matrix<R, C> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f64 {
        &mut self.0[index.0][index.1]
    }
}
