use std::array::from_fn;
use std::iter::zip;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::math::Linear;

/// A dense vector with a fixed size.
#[derive(Clone, Copy, Debug)]
pub struct Vector<const N: usize>(pub [f64; N]);

impl<const N: usize> Neg for Vector<N> {
    type Output = Self;

    fn neg(self) -> Self {
        Vector(self.0.map(Neg::neg))
    }
}

impl<const N: usize> Add for Vector<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Vector(from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<const N: usize> AddAssign for Vector<N> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize> Sub for Vector<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Vector(from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<const N: usize> SubAssign for Vector<N> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize> Mul<f64> for Vector<N> {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Vector(self.0.map(|lhs| lhs * rhs))
    }
}

impl<const N: usize> MulAssign<f64> for Vector<N> {
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl<const N: usize> Mul<Vector<N>> for f64 {
    type Output = Vector<N>;

    fn mul(self, rhs: Vector<N>) -> Vector<N> {
        rhs * self
    }
}

impl<const N: usize> Mul for Vector<N> {
    type Output = f64;

    fn mul(self, rhs: Self) -> f64 {
        zip(self.0, rhs.0).map(|(l, r)| l * r).sum()
    }
}

impl<const N: usize> Div<f64> for Vector<N> {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Vector(self.0.map(|lhs| lhs / rhs))
    }
}

impl<const N: usize> DivAssign<f64> for Vector<N> {
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

impl<const N: usize> Linear for Vector<N> {
    const ZERO: Self = Vector([0.0; N]);
}
