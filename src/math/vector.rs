use std::iter::zip;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::math::Linear;
use crate::{Array, ArrayInst, GenArray, array};

/// A dense vector with a fixed size.
pub struct Vector<A: GenArray>(pub Array<A, f64>);

/// Constructs a vector. This is equivalent to calling the `Vector` constructor
/// directly except it allows the type to infer the vector type from the
/// argument type.
pub fn vector<A: ArrayInst<Item = f64>>(elements: A) -> Vector<A::Gen> {
    Vector(elements)
}

impl<A: GenArray> Copy for Vector<A> {}

impl<A: GenArray> Clone for Vector<A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: GenArray> std::fmt::Debug for Vector<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.0.as_ref(), f)
    }
}

impl<A: GenArray> Neg for Vector<A> {
    type Output = Self;

    fn neg(self) -> Self {
        Vector(self.0.map(Neg::neg))
    }
}

impl<A: GenArray> Add for Vector<A> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Vector(A::from_fn(|i| self[i] + rhs[i]))
    }
}

impl<A: GenArray> AddAssign for Vector<A> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<A: GenArray> Sub for Vector<A> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Vector(A::from_fn(|i| self[i] - rhs[i]))
    }
}

impl<A: GenArray> SubAssign for Vector<A> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<A: GenArray> Mul<f64> for Vector<A> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Vector(self.0.map(|lhs| lhs * rhs))
    }
}

impl<A: GenArray> MulAssign<f64> for Vector<A> {
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl<A: GenArray> Mul<Vector<A>> for f64 {
    type Output = Vector<A>;

    fn mul(self, rhs: Vector<A>) -> Vector<A> {
        rhs * self
    }
}

impl<A: GenArray> Mul for Vector<A> {
    type Output = f64;

    fn mul(self, rhs: Self) -> f64 {
        zip(self.0.as_ref(), rhs.0.as_ref())
            .map(|(l, r)| l * r)
            .sum()
    }
}

impl<A: GenArray> Div<f64> for Vector<A> {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Vector(self.0.map(|lhs| lhs / rhs))
    }
}

impl<A: GenArray> DivAssign<f64> for Vector<A> {
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

impl<A: GenArray> Linear for Vector<A> {
    const ZERO: Self = Vector(array::repeat(0.0));
}

impl<A: GenArray> Index<usize> for Vector<A> {
    type Output = f64;

    #[track_caller]
    fn index(&self, index: usize) -> &f64 {
        self.0.as_ref().index(index)
    }
}

impl<A: GenArray> IndexMut<usize> for Vector<A> {
    #[track_caller]
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        self.0.as_mut().index_mut(index)
    }
}
