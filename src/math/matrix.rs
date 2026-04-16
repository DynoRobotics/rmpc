use core::iter::zip;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::array::Concat;
use crate::math::{Float, Linear};
use crate::{Array, ArrayInst, GenArray, array};

/// A dense matrix with a fixed size, stored in row major order.
pub struct Matrix<R: GenArray, C: GenArray>(pub Array<R, Array<C, Float>>);

/// A column vector.
pub type Vector<N> = Matrix<N, [(); 1]>;

/// Constructor for the [`Vector`](tyalias@Vector) type alias.
#[expect(non_snake_case)]
pub const fn Vector<N: GenArray>(elements: Array<N, Float>) -> Vector<N> {
    let mut i = 0;
    let mut v = Matrix::ZERO;
    while i < N::LEN {
        array::as_mut_slice(&mut v.0)[i] = [array::as_slice(&elements)[i]];
        i += 1;
    }
    v
}

/// Constructs a matrix. This is equivalent to calling the `Matrix` constructor
/// directly except it allows the type to infer the matrix type from the
/// argument type.
pub const fn matrix<R: ArrayInst<Item = C>, C: ArrayInst<Item = Float>>(
    rows: R,
) -> Matrix<R::Gen, C::Gen> {
    Matrix(rows)
}

/// Constructs a vector. This is equivalent to calling the `Vector` constructor
/// directly except it allows the type to infer the vector type from the
/// argument type.
pub const fn vector<A: ArrayInst<Item = Float>>(elements: A) -> Vector<A::Gen> {
    Vector(elements)
}

impl<R: GenArray, C: GenArray> Matrix<R, C> {
    /// Constructs a matrix by calling the function for each cell.
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> Float) -> Self {
        Matrix(R::from_fn(|r| C::from_fn(|c| f(r, c))))
    }

    /// The transpose of `self`.
    pub fn transpose(self) -> Matrix<C, R> {
        Matrix(C::from_fn(|row| R::from_fn(|col| self[(col, row)])))
    }

    /// The `i`th row of `self`.
    pub fn row(self, r: usize) -> Vector<C> {
        Matrix(self.0.as_slice()[r].map(|e| [e]))
    }

    /// The `i`th column of `self`.
    pub fn col(self, j: usize) -> Vector<R> {
        Matrix(self.0.map(|row| [row.as_slice()[j]]))
    }

    /// Sets the `i`th row of `self`.
    pub fn set_row(&mut self, r: usize, row: Vector<C>) {
        for (c, &[value]) in row.0.iter().enumerate() {
            self[(r, c)] = value;
        }
    }

    /// Sets the `i`th column of `self`.
    pub fn set_col(&mut self, c: usize, col: Vector<R>) {
        for (r, &[value]) in col.0.iter().enumerate() {
            self[(r, c)] = value;
        }
    }

    /// Concatenates `self` with another matrix, horizontally.
    pub fn concat_h<D: GenArray>(self, other: Matrix<R, D>) -> Matrix<R, Concat<C, D>> {
        Matrix(array::from_fn(|i| {
            self.0.as_slice()[i].concat(other.0.as_slice()[i])
        }))
    }

    /// Concatenates `self` with another matrix, vertically.
    pub fn concat_v<S: GenArray>(self, other: Matrix<S, C>) -> Matrix<Concat<R, S>, C> {
        Matrix(self.0.concat(other.0))
    }
}

impl<N: GenArray> Vector<N> {
    /// Turns `self` into an array of [`Float`].
    pub fn into_array(self) -> Array<N, Float> {
        self.0.map(|[e]| e)
    }
}

impl Matrix<[(); 1], [(); 1]> {
    /// Turns `self` into an array of [`Float`].
    pub fn into_scalar(self) -> Float {
        self.0[0][0]
    }
}

impl<R: GenArray, C: GenArray, D: GenArray> Matrix<R, Concat<C, D>> {
    /// Splits `self` into two matrices, horizontally.
    pub fn split_h(self) -> (Matrix<R, C>, Matrix<R, D>) {
        (
            Matrix(array::from_fn(|i| self.0.as_slice()[i].0)),
            Matrix(array::from_fn(|i| self.0.as_slice()[i].1)),
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

impl<R: GenArray, C: GenArray> core::fmt::Debug for Matrix<R, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list()
            .entries(self.0.iter().map(|arr| arr.as_slice()))
            .finish()
    }
}

impl<N: GenArray> Matrix<N, N> {
    /// The identity matrix.
    pub const IDENTITY: Self = Self::from_diag(Vector(array::repeat(Float::from_f64(1.0))));

    /// A matrix with values on the main diagonal and zeros everywhere else.
    pub const fn from_diag(diag: Vector<N>) -> Self {
        let mut data: Self = Matrix::ZERO;
        let mut i = 0;
        while i < N::LEN {
            let row = &mut array::as_mut_slice(&mut data.0)[i];
            array::as_mut_slice(row)[i] = array::as_slice(&diag.0)[i][0];
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
        for (dst, src) in zip(self.0.iter_mut(), rhs.0.iter()) {
            for (dst, src) in zip(dst.iter_mut(), src.iter()) {
                *dst += *src;
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
        for (dst, src) in zip(self.0.iter_mut(), rhs.0.iter()) {
            for (dst, src) in zip(dst.iter_mut(), src.iter()) {
                *dst -= *src;
            }
        }
    }
}

impl<R: GenArray, C: GenArray> Mul<Float> for Matrix<R, C> {
    type Output = Self;

    fn mul(mut self, rhs: Float) -> Self {
        self *= rhs;
        self
    }
}

impl<R: GenArray, C: GenArray> MulAssign<Float> for Matrix<R, C> {
    fn mul_assign(&mut self, rhs: Float) {
        for dst in self.0.iter_mut() {
            for dst in dst.iter_mut() {
                *dst *= rhs;
            }
        }
    }
}

impl<R: GenArray, C: GenArray> Mul<Matrix<R, C>> for Float {
    type Output = Matrix<R, C>;

    fn mul(self, mut rhs: Matrix<R, C>) -> Matrix<R, C> {
        rhs *= self;
        rhs
    }
}

impl<A: GenArray, B: GenArray, C: GenArray> Mul<Matrix<B, C>> for Matrix<A, B> {
    type Output = Matrix<A, C>;

    fn mul(self, rhs: Matrix<B, C>) -> Matrix<A, C> {
        Matrix(A::from_fn(|i| {
            C::from_fn(|j| (0..B::LEN).map(|k| self[(i, k)] * rhs[(k, j)]).sum())
        }))
    }
}

impl<R: GenArray, C: GenArray> MulAssign<Matrix<C, C>> for Matrix<R, C> {
    fn mul_assign(&mut self, rhs: Matrix<C, C>) {
        *self = *self * rhs;
    }
}

impl<R: GenArray, C: GenArray> Div<Float> for Matrix<R, C> {
    type Output = Self;

    fn div(mut self, rhs: Float) -> Self {
        self /= rhs;
        self
    }
}

impl<R: GenArray, C: GenArray> DivAssign<Float> for Matrix<R, C> {
    fn div_assign(&mut self, rhs: Float) {
        for dst in self.0.iter_mut() {
            for dst in dst.iter_mut() {
                *dst /= rhs;
            }
        }
    }
}

impl<R: GenArray, C: GenArray> Linear for Matrix<R, C> {
    const ZERO: Self = Matrix(array::repeat(array::repeat(Float::ZERO)));
}

impl<R: GenArray, C: GenArray> Index<(usize, usize)> for Matrix<R, C> {
    type Output = Float;

    fn index(&self, index: (usize, usize)) -> &Float {
        &self.0.as_slice()[index.0].as_slice()[index.1]
    }
}

impl<N: GenArray> Index<usize> for Vector<N> {
    type Output = Float;

    fn index(&self, index: usize) -> &Float {
        self.index((index, 0))
    }
}

impl<R: GenArray, C: GenArray> IndexMut<(usize, usize)> for Matrix<R, C> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Float {
        &mut self.0.as_mut_slice()[index.0].as_mut_slice()[index.1]
    }
}

impl<N: GenArray> IndexMut<usize> for Vector<N> {
    fn index_mut(&mut self, index: usize) -> &mut Float {
        self.index_mut((index, 0))
    }
}
