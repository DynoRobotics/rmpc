//! A work in progress library for ergonomic model predictive control.

pub mod array;
pub mod math;
pub mod model;
pub mod mpc;
mod traits;
pub mod utility;

pub use crate::array::{Array, ArrayInst, GenArray};
pub use crate::traits::FieldNames;
