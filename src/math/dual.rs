use std::fmt::Debug;

use crate::math::Linear;

/// A value tracking its gradient with respect to some set of variables.
#[derive(Clone, Copy)]
pub struct Dual<D> {
    value: f64,
    grad: D,
}

impl<D> Dual<D> {
    /// The value at the linearization point, useful for deciding which part of a
    /// piecewise function to use.
    ///
    /// Note: This discards the derivative, so it will be treated as a constant if
    /// used in calculations.
    pub fn value(self) -> f64 {
        self.value
    }
}

impl<D: Debug> Debug for Dual<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dual")
            .field("value", &self.value)
            .field("grad", &self.grad)
            .finish()
    }
}

impl<D: Linear> From<f64> for Dual<D> {
    fn from(value: f64) -> Self {
        Dual {
            value,
            grad: D::ZERO,
        }
    }
}

macro_rules! impl_funcs {
    ($(
        $(#[$attr:meta])*
        fn $name:ident($i:ident) = $v:expr;
    )+) => {
        impl<D: Linear> Dual<D> {
            $(
                $(#[$attr])*
                pub fn $name(self) -> Self {
                    let $i = self.value;
                    let (value, deriv) = $v;
                    let grad = self.grad * deriv;
                    Dual { value, grad }
                }
            )+
        }
    };
}
impl_funcs! {
    /// The square root of the value.
    fn sqrt(v) = (v.sqrt(), 0.5 / v.sqrt());
    /// The sine of an angle in radians.
    fn sin(v) = (v.sin(), v.cos());
    /// The cosine of an angle in radians.
    fn cos(v) = (v.cos(), -v.sin());
    /// The tangent of an angle in radians.
    fn tan(v) = (v.tan(), v.cos().powi(-2));
    /// The arcsine in radians of a value.
    fn asin(v) = (v.asin(), 1.0 / (1.0 - v.powi(2)).sqrt());
    /// The arccosine in radians of a value.
    fn acos(v) = (v.acos(), -1.0 / (1.0 - v.powi(2)).sqrt());
    /// The arctangent in radians of a value.
    fn atan(v) = (v.atan(), 1.0 / (1.0 + v.powi(2)));
    /// The exponential (e<sup>x</sup>) of the value.
    fn exp(v) = (v.exp(), v.exp());
    /// The natural logarithm (base e) of the value.
    fn ln(v) = (v.ln(), 1.0 / v);
}

impl<D: Linear> std::ops::Neg for Dual<D> {
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
        impl<D: Linear> std::ops::$trait for Dual<D> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                let $l = self.value;
                let $r = rhs.value;
                let (value, [grad_lhs, grad_rhs]) = $v;
                let grad = self.grad * grad_lhs + rhs.grad * grad_rhs;
                Dual { value, grad }
            }
        }
        impl<D: Linear> std::ops::$trait<f64> for Dual<D> {
            type Output = Self;
            fn $method(self, rhs: f64) -> Self {
                let $l = self.value;
                let $r = rhs;
                let (value, [grad_lhs, _grad_rhs]) = $v;
                let grad = self.grad * grad_lhs;
                Dual { value, grad }
            }
        }
        impl<D: Linear> std::ops::$trait<Dual<D>> for f64 {
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
impl_op!(Add::add(l, r) = (l + r, [1.0, 1.0]));
impl_op!(Sub::sub(l, r) = (l - r, [1.0, -1.0]));
impl_op!(Mul::mul(l, r) = (l * r, [r, l]));
impl_op!(Div::div(l, r) = (l / r, [1.0 / r, -l / r.powi(2)]));

macro_rules! impl_assign_op {
    ($atrait:ident::$amethod:ident = $trait:ident::$method:ident) => {
        impl<D: Linear> std::ops::$atrait for Dual<D> {
            fn $amethod(&mut self, rhs: Self) {
                *self = std::ops::$trait::$method(*self, rhs);
            }
        }
        impl<D: Linear> std::ops::$atrait<f64> for Dual<D> {
            fn $amethod(&mut self, rhs: f64) {
                *self = std::ops::$trait::$method(*self, rhs);
            }
        }
    };
}
impl_assign_op!(AddAssign::add_assign = Add::add);
impl_assign_op!(SubAssign::sub_assign = Sub::sub);
impl_assign_op!(MulAssign::mul_assign = Mul::mul);
impl_assign_op!(DivAssign::div_assign = Div::div);
