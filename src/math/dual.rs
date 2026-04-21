use core::fmt::Debug;

use crate::array::from_fn;
use crate::math::{Float, Linear, Matrix, Vector};
use crate::{Array, ArrayInst, GenArray};

/// Differentiates the function around a certain point, returning the value and
/// Jacobian at that point.
pub fn differentiate<I: GenArray, O: ArrayInst<Item = Dual<Vector<I>>>>(
    point: Vector<I>,
    mut function: impl FnMut(Array<I, Dual<Vector<I>>>) -> O,
) -> (Vector<O::Gen>, Matrix<O::Gen, I>) {
    let input = I::from_fn(|i| Dual {
        value: point[i],
        grad: Vector(from_fn(|j| Float::from_f64((i == j) as u8 as f64))),
    });
    let output = function(input);
    let value = Vector(output.map(|out| out.value));
    let jacobian = Matrix(output.map(|out| out.grad.into_array()));
    (value, jacobian)
}

/// Approximates the function as `A*x + b` around some point.
pub fn linearize<I: GenArray, O: ArrayInst<Item = Dual<Vector<I>>>>(
    point: Vector<I>,
    function: impl FnMut(Array<I, Dual<Vector<I>>>) -> O,
) -> (Matrix<O::Gen, I>, Vector<O::Gen>) {
    let (value, jacobian) = differentiate(point, function);
    (jacobian, value - jacobian * point)
}

/// A value tracking its gradient with respect to some set of variables.
#[derive(Clone, Copy, Debug)]
pub struct Dual<D> {
    value: Float,
    grad: D,
}

impl<D> Dual<D> {
    /// The value at the linearization point, useful for deciding which part of a
    /// piecewise function to use.
    ///
    /// Note: This discards the derivative, so it will be treated as a constant if
    /// used in calculations.
    pub fn value(self) -> f64 {
        self.value.as_f64()
    }

    /// Same as [`value`](Self::value), but returns the value as a [`Float`].
    pub fn float_value(self) -> Float {
        self.value
    }

    /// Gets the sign of `self`. Its derivative is always zero.
    pub fn signum(self) -> f64 {
        self.value.as_f64().signum()
    }

    /// Gets the minimum of `self` and `other`.
    pub fn min(self, other: impl Into<Self>) -> Self {
        let other = other.into();
        if self.value < other.value {
            self
        } else {
            other
        }
    }

    /// Gets the maximum of `self` and `other`.
    pub fn max(self, other: impl Into<Self>) -> Self {
        let other = other.into();
        if self.value < other.value {
            other
        } else {
            self
        }
    }

    /// Clamps `self` to the range `min..=max`. Assumes `min <= max`.
    pub fn clamp(self, min: impl Into<Self>, max: impl Into<Self>) -> Self {
        // Note: They should be this way around, since it should be at least the minimum
        // and at most the maximum.
        self.max(min).min(max)
    }
}

impl<D: Linear> Dual<D> {
    /// Gets the absolute value of `self`.
    pub fn abs(self) -> Self {
        self * self.signum()
    }
}

impl<D: Linear> From<f64> for Dual<D> {
    fn from(value: f64) -> Self {
        Dual::from(Float::from(value))
    }
}

impl<D: Linear> From<Float> for Dual<D> {
    fn from(value: Float) -> Self {
        Dual {
            value,
            grad: D::ZERO,
        }
    }
}

/// Helper to implement operations on dual numbers. The body should contain a
/// number of functions with syntax like
///
/// ```
/// fn sub_mul(x, y, z) {
///     (x - y * z, [1.0, -z, -y])
/// }
/// ```
///
/// The return value at the end of the function should be a tuple with the value
/// and gradient of the function. If the function takes arguments with a type
/// other than `Dual`, then they can be provided after a semicolon. For example.
///
/// ```
/// fn add_powi(a, b; n: i32, m: i32) {
///     (a.powi(n) + b.powi(m), [(n as f64) * a.powi(n - 1), (m as f64) * b.powi(n - 1)])
/// }
/// ```
///
/// Note that there is no derivative with respect to those arguments as only
/// `Dual` values have derivatives.
macro_rules! impl_funcs {
    ($(
        $(#[$attr:meta])*
        fn $name:ident(
            $s:ident $(, $a:ident)*
            $(; $($cname:ident : $cty:ty),*)?
        ) { $($v:tt)* }
    )+) => {
        impl<D: Linear> Dual<D> {
            $(
                $(#[$attr])*
                pub fn $name(self $(, $a: impl Into<Self>)* $($(,$cname: $cty)+)?) -> Self {
                    $( let $a = $a.into(); )*

                    let (value, deriv) = {
                        let $s = self.value;
                        $( let $a = $a.value; )*
                        $($v)*
                    };
                    let grad = core::iter::zip([self.grad $(, $a.grad)*], deriv)
                        .map(|(grad, deriv)| grad * deriv)
                        .reduce(|g1, g2| g1 + g2)
                        .expect("at least 1 gradient to sum");
                    Dual { value, grad }
                }
            )+
        }
    };
}

impl_funcs! {
    /// The square root of the value.
    fn sqrt(v) {
        let s = v.sqrt();
        (s, [0.5 / s])
    }
    /// The exponential (e<sup>x</sup>) of the value.
    fn exp(v) {
        let e = v.exp();
        (e, [e])
    }
    /// The natural logarithm (base e) of the value.
    fn ln(v) {
        (v.ln(), [1.0 / v])
    }
    /// Raises `self` to an integer power.
    fn powi(v; n: i32) {
        let pow_m1 = v.powi(n - 1);
        (v * pow_m1, [(n as f64) * pow_m1])
    }
    /// The sine of an angle in radians.
    fn sin(v) {
        (v.sin(), [v.cos()])
    }
    /// The cosine of an angle in radians.
    fn cos(v) {
        (v.cos(), [-v.sin()])
    }
    /// The tangent of an angle in radians.
    fn tan(v) {
        let c = v.cos();
        (v.tan(), [1.0 / (c * c)])
    }
    /// The arcsine in radians of a value.
    fn asin(v) {
        (v.asin(), [1.0 / (1.0 - v * v).sqrt()])
    }
    /// The arccosine in radians of a value.
    fn acos(v) {
        (v.acos(), [-1.0 / (1.0 - v * v).sqrt()])
    }
    /// The arctangent in radians of a value.
    fn atan(v) {
        (v.atan(), [1.0 / (1.0 + v * v)])
    }
    /// Computes the four quadrant arctangent of `self / other`, in radians.
    fn atan2(y, x) {
        let denom = x * x + y * y;
        (y.atan2(x), [-y / denom, x / denom])
    }
}

impl<D: Linear> core::ops::Neg for Dual<D> {
    type Output = Self;
    fn neg(self) -> Self {
        Dual {
            value: -self.value,
            grad: -self.grad,
        }
    }
}

macro_rules! impl_op {
    ($trait:ident::$method:ident($l:ident, $r:ident) = $v:expr) => {
        impl<D: Linear> core::ops::$trait for Dual<D> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                let $l = self.value;
                let $r = rhs.value;
                let (value, [grad_lhs, grad_rhs]) = $v;
                let grad = self.grad * grad_lhs + rhs.grad * grad_rhs;
                Dual { value, grad }
            }
        }
        impl<D: Linear> core::ops::$trait<f64> for Dual<D> {
            type Output = Self;
            fn $method(self, rhs: f64) -> Self {
                core::ops::$trait::$method(self, Float::from(rhs))
            }
        }
        impl<D: Linear> core::ops::$trait<Dual<D>> for f64 {
            type Output = Dual<D>;
            fn $method(self, rhs: Dual<D>) -> Dual<D> {
                core::ops::$trait::$method(Float::from(self), rhs)
            }
        }
        impl<D: Linear> core::ops::$trait<Float> for Dual<D> {
            type Output = Self;
            fn $method(self, rhs: Float) -> Self {
                let $l = self.value;
                let $r = rhs;
                let (value, [grad_lhs, _grad_rhs]) = $v;
                let grad = self.grad * grad_lhs;
                Dual { value, grad }
            }
        }
        impl<D: Linear> core::ops::$trait<Dual<D>> for Float {
            type Output = Dual<D>;
            fn $method(self, rhs: Dual<D>) -> Dual<D> {
                let $l = self;
                let $r = rhs.value;
                let (value, [_grad_lhs, grad_rhs]) = $v;
                let grad = rhs.grad * grad_rhs;
                Dual { value, grad }
            }
        }
    };
}
impl_op!(Add::add(l, r) = (l + r, [Float::from(1.0), Float::from(1.0)]));
impl_op!(Sub::sub(l, r) = (l - r, [Float::from(1.0), Float::from(-1.0)]));
impl_op!(Mul::mul(l, r) = (l * r, [r, l]));
impl_op!(Div::div(l, r) = (l / r, [Float::from(1.0) / r, -l / (r * r)]));

macro_rules! impl_assign_op {
    ($atrait:ident::$amethod:ident = $trait:ident::$method:ident) => {
        impl<D: Linear> core::ops::$atrait for Dual<D> {
            fn $amethod(&mut self, rhs: Self) {
                *self = core::ops::$trait::$method(*self, rhs);
            }
        }
        impl<D: Linear> core::ops::$atrait<f64> for Dual<D> {
            fn $amethod(&mut self, rhs: f64) {
                *self = core::ops::$trait::$method(*self, rhs);
            }
        }
        impl<D: Linear> core::ops::$atrait<Float> for Dual<D> {
            fn $amethod(&mut self, rhs: Float) {
                *self = core::ops::$trait::$method(*self, rhs);
            }
        }
    };
}
impl_assign_op!(AddAssign::add_assign = Add::add);
impl_assign_op!(SubAssign::sub_assign = Sub::sub);
impl_assign_op!(MulAssign::mul_assign = Mul::mul);
impl_assign_op!(DivAssign::div_assign = Div::div);
