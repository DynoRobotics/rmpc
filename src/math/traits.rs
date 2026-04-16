use core::ops::{Add, Mul, Neg, Sub};

use crate::math::Float;

/// A copyable type obeying the rules of a vector field.
pub trait Linear:
    Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Float, Output = Self>
    + Neg<Output = Self>
    + Copy
    + Sized
{
    /// The zero vector.
    const ZERO: Self;
}

/// An element in the vector field containing only zero.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Zero;

macro_rules! impl_zero_op {
    ($trait:ident $(<$arg:ty>)? ::$method:ident) => {
        impl $trait $(<$arg>)? for Zero {
            type Output = Zero;
            fn $method(self $(, _rhs: $arg)?) -> Zero {
                Zero
            }
        }
    }
}

impl_zero_op!(Add<Zero>::add);
impl_zero_op!(Sub<Zero>::sub);
impl_zero_op!(Mul<Float>::mul);
impl_zero_op!(Neg::neg);

impl Linear for Zero {
    const ZERO: Zero = Zero;
}
