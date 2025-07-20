use std::ops::{Add, Mul, Neg, Sub};

/// A copyable type obeying the rules of a vector field.
pub trait Linear:
    Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<f64, Output = Self>
    + Neg<Output = Self>
    + Copy
    + Sized
{
    /// The zero vector.
    const ZERO: Self;
}

impl Linear for f64 {
    const ZERO: Self = 0.0;
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
impl_zero_op!(Mul<f64>::mul);
impl_zero_op!(Neg::neg);

impl Linear for Zero {
    const ZERO: Zero = Zero;
}
