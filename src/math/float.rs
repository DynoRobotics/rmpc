use core::fmt::{self, Debug, Formatter};

use crate::math::Linear;

#[cfg(not(feature = "f64"))]
type Inner = f32;
#[cfg(feature = "f64")]
type Inner = f64;

type Libm = libm::Libm<Inner>;

/// A wrapper that contains a `f32` or `f64` depending on if the `f64` feature
/// is enabled.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Float(Inner);

impl Debug for Float {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Float {
    /// Gets the inner value, as a `f64`.
    #[inline]
    #[allow(clippy::unnecessary_cast)]
    pub const fn as_f64(self) -> f64 {
        self.0 as f64
    }

    /// Turns an `f64` into a [`Float`].
    #[inline]
    #[allow(clippy::unnecessary_cast)]
    pub const fn from_f64(value: f64) -> Float {
        Float(value as Inner)
    }

    /// Returns `true` if `self` is neither infinite nor NaN.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    /// Computes the absolute value of `self`.
    #[inline]
    pub fn abs(self) -> Float {
        Float(self.0.abs())
    }

    /// Clamps `self` to the range `min`..`max`.
    #[inline]
    pub fn clamp(self, min: impl Into<Float>, max: impl Into<Float>) -> Float {
        let min = min.into();
        let max = max.into();
        Float(self.0.clamp(min.0, max.0))
    }

    /// Computes the square root of `self`.
    #[inline]
    pub fn sqrt(self) -> Float {
        Float(Libm::sqrt(self.0))
    }

    /// Computes the exponential function.
    pub fn exp(self) -> Float {
        Float(Libm::exp(self.0))
    }

    /// Computes the natural logarithm.
    pub fn ln(self) -> Float {
        Float(Libm::log(self.0))
    }

    /// Computes `self` to the power of `exp`.
    pub fn powi(mut self, exp: i32) -> Float {
        if exp < 0 {
            self = 1.0 / self;
        }

        let mut exp = exp.unsigned_abs();

        let mut result = Float(1.0);
        while exp > 0 {
            if exp % 2 == 1 {
                result *= self;
            }
            self *= self;
            exp /= 2;
        }

        result
    }

    /// Computes the sine, with the arguments in radians.
    pub fn sin(self) -> Float {
        Float(Libm::sin(self.0))
    }

    /// Computes the cosine, with the arguments in radians.
    pub fn cos(self) -> Float {
        Float(Libm::cos(self.0))
    }

    /// Computes the tangent, with the arguments in radians.
    pub fn tan(self) -> Float {
        Float(Libm::tan(self.0))
    }

    /// Computes the arcsine, in radians.
    pub fn asin(self) -> Float {
        Float(Libm::asin(self.0))
    }

    /// Computes the arccosine, in radians.
    pub fn acos(self) -> Float {
        Float(Libm::acos(self.0))
    }

    /// Computes the arctangent, in radians.
    pub fn atan(self) -> Float {
        Float(Libm::atan(self.0))
    }

    /// Computes the four quadrant arctangent of `self / other`, in radians.
    pub fn atan2(self, other: Float) -> Float {
        Float(Libm::atan2(self.0, other.0))
    }
}

impl From<Float> for f64 {
    #[inline]
    fn from(value: Float) -> f64 {
        value.as_f64()
    }
}

impl From<f64> for Float {
    #[inline]
    fn from(value: f64) -> Float {
        Float::from_f64(value)
    }
}

macro_rules! impl_binop {
    ($($t:ident :: $method:ident;)*) => {
        $(
            impl core::ops::$t for Float {
                type Output = Float;
                #[inline]
                fn $method(self, rhs: Float) -> Float {
                    Float(core::ops::$t::$method(self.0, rhs.0))
                }
            }
            impl core::ops::$t<f64> for Float {
                type Output = Float;
                #[inline]
                fn $method(self, rhs: f64) -> Float {
                    core::ops::$t::$method(self, Float::from(rhs))
                }
            }
            impl core::ops::$t<Float> for f64 {
                type Output = Float;
                #[inline]
                fn $method(self, rhs: Float) -> Float {
                    core::ops::$t::$method(Float::from(self), rhs)
                }
            }
        )*
    };
}
impl_binop! {
    Add::add;
    Sub::sub;
    Mul::mul;
    Div::div;
    Rem::rem;
}

impl PartialEq<f64> for Float {
    #[inline]
    fn eq(&self, other: &f64) -> bool {
        PartialEq::eq(self, &Float::from(*other))
    }
}
impl PartialOrd<f64> for Float {
    #[inline]
    fn partial_cmp(&self, other: &f64) -> Option<core::cmp::Ordering> {
        PartialOrd::partial_cmp(self, &Float::from(*other))
    }
}

macro_rules! impl_assignop {
    ($($t:ident :: $method:ident;)*) => {
        $(
            impl core::ops::$t for Float {
                #[inline]
                fn $method(&mut self, rhs: Float) {
                    core::ops::$t::$method(&mut self.0, rhs.0);
                }
            }
            impl core::ops::$t<f64> for Float {
                #[inline]
                fn $method(&mut self, rhs: f64) {
                    core::ops::$t::$method(self, Float::from(rhs));
                }
            }
        )*
    };
}
impl_assignop! {
    AddAssign::add_assign;
    SubAssign::sub_assign;
    MulAssign::mul_assign;
    DivAssign::div_assign;
    RemAssign::rem_assign;
}

impl core::ops::Neg for Float {
    type Output = Float;

    #[inline]
    fn neg(self) -> Float {
        Float(-self.0)
    }
}

impl core::iter::Sum for Float {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Float(iter.map(|f| f.0).sum())
    }
}

impl Linear for Float {
    const ZERO: Self = Float(0.0);
}
