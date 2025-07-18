mod dual;
mod matrix;
mod traits;
mod vector;

pub use self::dual::{Dual, linearize};
pub use self::matrix::Matrix;
pub use self::traits::{Linear, Zero};
pub use self::vector::Vector;
