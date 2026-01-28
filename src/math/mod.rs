//! Various mathematical types, traits and utilities.

mod algorithm;
mod cholesky;
mod dual;
mod matrix;
mod traits;
mod vector;

pub use self::algorithm::inv_no_pivot;
pub use self::cholesky::Cholesky;
pub use self::dual::{Dual, differentiate, eval_linearized, linearize};
pub use self::matrix::{Matrix, matrix};
pub use self::traits::{Linear, Zero};
pub use self::vector::{Vector, vector};
