//! Various mathematical types, traits and utilities.

mod algorithm;
mod cholesky;
mod dual;
mod float;
mod matrix;
mod traits;

pub use self::algorithm::inv_no_pivot;
pub use self::cholesky::Cholesky;
pub use self::dual::{Dual, differentiate, eval_linearized, linearize};
pub use self::float::Float;
pub use self::matrix::{Matrix, Vector, matrix, vector};
pub use self::traits::{Linear, Zero};
