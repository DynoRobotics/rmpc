//! Mathematical state-space models of systems.

mod continuous;

use crate::math::{Dual, Linear, Zero};
use crate::{Array, ArrayInst, GenArray};

pub use self::continuous::{Continuous, ContinuousDiff, RungeKutta4, RungeKutta4Diff};

/// An explicit non-linear discrete time model on the form
/// `x[k+1] = time_step(x[k], u[k])`.
///
/// Most systems are continuous, so instead of implementing this directly it is
/// recommended to implement [`Continuous`] and use [`discretize`] to turn it
/// into a discrete time model.
///
/// [`discretize`]: Continuous::discretize
pub trait Model {
    /// The state of the system at some time step.
    type State: GenArray;
    /// The input at some time step.
    type Input: GenArray;
    /// A vector whose squared magnitude is the cost at some time step.
    type Cost: GenArray;

    /// Performs a single time step. Uses dual numbers to make it possible to
    /// linearize the model.
    fn time_step<D: Linear>(
        &self,
        time: usize,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::State, Dual<D>>;

    /// The vector whose squared magnitude is the cost at the current time step.
    fn cost_vector<D: Linear>(
        &self,
        time: usize,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Cost, Dual<D>>;

    /// The valid range of each input. The lower bound must be less than or equal to
    /// the upper bound.
    fn input_ranges(&self, time: usize) -> Array<Self::Input, (f64, f64)>;

    /// A convenience method to perform the time step without tracking any gradient.
    fn time_step_f64(
        &self,
        time: usize,
        state: Array<Self::State, f64>,
        input: Array<Self::Input, f64>,
    ) -> Array<Self::State, f64> {
        let state = state.map(Dual::from);
        let input = input.map(Dual::from);
        let next_state = self.time_step::<Zero>(time, state, input);
        next_state.map(Dual::value)
    }

    /// A convenience method to get the cost vector without tracking any gradient.
    fn cost_vector_f64(
        &self,
        time: usize,
        state: Array<Self::State, f64>,
        input: Array<Self::Input, f64>,
    ) -> Array<Self::Cost, f64> {
        let state = state.map(Dual::from);
        let input = input.map(Dual::from);
        let cost = self.cost_vector::<Zero>(time, state, input);
        cost.map(Dual::value)
    }
}
