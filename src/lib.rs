//! A work in progress library for ergonomic model predictive control.

#![no_std]

pub mod array;
pub mod math;
pub mod model;
pub mod mpc;
pub mod riccati;
mod traits;

pub use crate::array::{Array, ArrayInst, GenArray};
pub use crate::traits::FieldNames;
