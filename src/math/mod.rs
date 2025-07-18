mod traits;
mod dual;
mod vector;
mod matrix;

pub use self::traits::{Linear, Zero};
pub use self::dual::Dual;
pub use self::vector::Vector;
pub use self::matrix::Matrix;
