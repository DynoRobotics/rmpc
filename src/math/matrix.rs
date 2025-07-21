use std::iter::zip;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::array::Concat;
use crate::math::{Linear, Vector};
use crate::{Array, ArrayInst, GenArray, array};

/// A dense matrix with a fixed size, stored in row major order.
pub struct Matrix<R: GenArray, C: GenArray>(pub Array<R, Array<C, f64>>);

/// Constructs a matrix. This is equivalent to calling the `Matrix` constructor
/// directly except it allows the type to infer the matrix type from the
/// argument type.
pub fn matrix<R: ArrayInst<Item = C>, C: ArrayInst<Item = f64>>(rows: R) -> Matrix<R::Gen, C::Gen> {
    Matrix(rows)
}

impl<R: GenArray, C: GenArray> Matrix<R, C> {
    /// The transpose of `self`.
    pub fn transpose(self) -> Matrix<C, R> {
        Matrix(C::from_fn(|row| R::from_fn(|col| self[(col, row)])))
    }

    /// The `i`th row of `self`.
    pub fn row(self, r: usize) -> Vector<C> {
        Vector(self.0.as_ref()[r])
    }

    /// The `i`th column of `self`.
    pub fn col(self, j: usize) -> Vector<R> {
        Vector(self.0.map(|row| row.as_ref()[j]))
    }

    /// Sets the `i`th row of `self`.
    pub fn set_row(&mut self, r: usize, row: Vector<C>) {
        for (c, &value) in row.0.as_ref().iter().enumerate() {
            self[(r, c)] = value;
        }
    }

    /// Sets the `i`th column of `self`.
    pub fn set_col(&mut self, c: usize, col: Vector<R>) {
        for (r, &value) in col.0.as_ref().iter().enumerate() {
            self[(r, c)] = value;
        }
    }

    /// Concatenates `self` with another matrix, horizontally.
    pub fn concat_h<D: GenArray>(self, other: Matrix<R, D>) -> Matrix<R, Concat<C, D>> {
        Matrix(array::from_fn(|i| {
            self.0.as_ref()[i].concat(other.0.as_ref()[i])
        }))
    }

    /// Concatenates `self` with another matrix, vertically.
    pub fn concat_v<S: GenArray>(self, other: Matrix<S, C>) -> Matrix<Concat<R, S>, C> {
        Matrix(self.0.concat(other.0))
    }
}

impl<R: GenArray, C: GenArray, D: GenArray> Matrix<R, Concat<C, D>> {
    /// Splits `self` into two matrices, horizontally.
    pub fn split_h(self) -> (Matrix<R, C>, Matrix<R, D>) {
        (
            Matrix(array::from_fn(|i| self.0.as_ref()[i].0)),
            Matrix(array::from_fn(|i| self.0.as_ref()[i].1)),
        )
    }
}

impl<R: GenArray, S: GenArray, C: GenArray> Matrix<Concat<R, S>, C> {
    /// Splits `self` into two matrices, vertically.
    pub fn split_v(self) -> (Matrix<R, C>, Matrix<S, C>) {
        (Matrix(self.0.0), Matrix(self.0.1))
    }
}

impl<R: GenArray, C: GenArray> Copy for Matrix<R, C> {}

impl<R: GenArray, C: GenArray> Clone for Matrix<R, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: GenArray, C: GenArray> std::fmt::Debug for Matrix<R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.0.as_ref().iter().map(|arr| arr.as_ref()))
            .finish()
    }
}

impl<N: GenArray> Matrix<N, N> {
    /// The identity matrix.
    pub const IDENTITY: Self = Self::from_diag(Vector(array::repeat(1.0)));

    /// A matrix with values on the main diagonals and zeros everywhere else.
    pub const fn from_diag(diag: Vector<N>) -> Self {
        let mut data: Self = Matrix(array::repeat(array::repeat(0.0)));
        let mut i = 0;
        while i < N::LEN {
            let row = &mut array::as_mut(&mut data.0)[i];
            array::as_mut(row)[i] = array::as_ref(&diag.0)[i];
            i += 1;
        }
        data
    }
}

impl<R: GenArray, C: GenArray> Neg for Matrix<R, C> {
    type Output = Self;

    fn neg(self) -> Self {
        Matrix(self.0.map(|row| row.map(Neg::neg)))
    }
}

impl<R: GenArray, C: GenArray> Add for Matrix<R, C> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl<R: GenArray, C: GenArray> AddAssign for Matrix<R, C> {
    fn add_assign(&mut self, rhs: Self) {
        for (dst, src) in zip(self.0.as_mut(), rhs.0.as_ref()) {
            for (dst, src) in zip(dst.as_mut(), src.as_ref()) {
                *dst += src;
            }
        }
    }
}

impl<R: GenArray, C: GenArray> Sub for Matrix<R, C> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}

impl<R: GenArray, C: GenArray> SubAssign for Matrix<R, C> {
    fn sub_assign(&mut self, rhs: Self) {
        for (dst, src) in zip(self.0.as_mut(), rhs.0.as_ref()) {
            for (dst, src) in zip(dst.as_mut(), src.as_ref()) {
                *dst -= src;
            }
        }
    }
}

impl<R: GenArray, C: GenArray> Mul<f64> for Matrix<R, C> {
    type Output = Self;

    fn mul(mut self, rhs: f64) -> Self {
        self *= rhs;
        self
    }
}

impl<R: GenArray, C: GenArray> MulAssign<f64> for Matrix<R, C> {
    fn mul_assign(&mut self, rhs: f64) {
        for dst in self.0.as_mut() {
            for dst in dst.as_mut() {
                *dst *= rhs;
            }
        }
    }
}

impl<R: GenArray, C: GenArray> Mul<Matrix<R, C>> for f64 {
    type Output = Matrix<R, C>;

    fn mul(self, mut rhs: Matrix<R, C>) -> Matrix<R, C> {
        rhs *= self;
        rhs
    }
}

impl<R: GenArray, C: GenArray> Mul<Vector<C>> for Matrix<R, C> {
    type Output = Vector<R>;

    fn mul(self, rhs: Vector<C>) -> Vector<R> {
        Vector(R::from_fn(|i| self.row(i) * rhs))
    }
}

impl<R: GenArray, C: GenArray> Mul<Matrix<R, C>> for Vector<R> {
    type Output = Vector<C>;

    fn mul(self, rhs: Matrix<R, C>) -> Vector<C> {
        Vector(C::from_fn(|i| self * rhs.col(i)))
    }
}

impl<N: GenArray> MulAssign<Matrix<N, N>> for Vector<N> {
    fn mul_assign(&mut self, rhs: Matrix<N, N>) {
        *self = *self * rhs;
    }
}

impl<A: GenArray, B: GenArray, C: GenArray> Mul<Matrix<B, C>> for Matrix<A, B> {
    type Output = Matrix<A, C>;

    fn mul(self, rhs: Matrix<B, C>) -> Matrix<A, C> {
        Matrix(A::from_fn(|i| (self.row(i) * rhs).0))
    }
}

impl<R: GenArray, C: GenArray> MulAssign<Matrix<C, C>> for Matrix<R, C> {
    fn mul_assign(&mut self, rhs: Matrix<C, C>) {
        *self = *self * rhs;
    }
}

impl<R: GenArray, C: GenArray> Div<f64> for Matrix<R, C> {
    type Output = Self;

    fn div(mut self, rhs: f64) -> Self {
        self /= rhs;
        self
    }
}

impl<R: GenArray, C: GenArray> DivAssign<f64> for Matrix<R, C> {
    fn div_assign(&mut self, rhs: f64) {
        for dst in self.0.as_mut() {
            for dst in dst.as_mut() {
                *dst /= rhs;
            }
        }
    }
}

impl<R: GenArray, C: GenArray> Linear for Matrix<R, C> {
    const ZERO: Self = Matrix(array::repeat(array::repeat(0.0)));
}

impl<R: GenArray, C: GenArray> Index<(usize, usize)> for Matrix<R, C> {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &f64 {
        &self.0.as_ref()[index.0].as_ref()[index.1]
    }
}

impl<R: GenArray, C: GenArray> IndexMut<(usize, usize)> for Matrix<R, C> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f64 {
        &mut self.0.as_mut()[index.0].as_mut()[index.1]
    }
}
