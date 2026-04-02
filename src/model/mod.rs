//! Mathematical state-space models of systems.

mod continuous;

use core::ops::Range;

use crate::math::{Dual, Linear, Zero};
use crate::{Array, ArrayInst, GenArray};

pub use self::continuous::{Continuous, RungeKutta4};

/// A double-sided soft constraint with constant bounds.
#[derive(Clone, Copy)]
pub struct Bounded<D> {
    /// The value to be constrained.
    pub value: Dual<D>,
    /// The lower bound.
    pub min: f64,
    /// The upper bound.
    pub max: f64,
    /// The cost of a violation.
    pub weight: f64,
}

impl<D> Bounded<D> {
    /// Creates a bound on the form `range.start <= value <= range.end`. If the
    /// constraint is violated, the penalty is the distance outside the range
    /// multiplied by `weight`.
    pub fn new(value: Dual<D>, range: Range<f64>, weight: f64) -> Bounded<D> {
        Bounded {
            value,
            min: range.start,
            max: range.end,
            weight,
        }
    }
}

/// An explicit non-linear discrete time model on the form
/// `x[k+1] = time_step(x[k], u[k], k)`.
///
/// Most systems are continuous, so instead of implementing this directly it is
/// recommended to implement [`Continuous`] and use [`discretize`] to turn it
/// into a discrete time model.
///
/// [`discretize`]: Continuous::discretize
pub trait Discrete {
    /// The state of the system at some time step.
    type State: GenArray;
    /// The input at some time step.
    type Input: GenArray;
    /// A vector whose squared magnitude is the cost at some time step.
    type Cost: GenArray;
    /// The values to limit with soft constraints.
    type Bounds: GenArray;

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

    /// The soft constraints to apply.
    fn bounds<D: Linear>(
        &self,
        time: usize,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Bounds, Bounded<D>>;

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

impl<M: Discrete> Discrete for &M {
    type State = M::State;
    type Input = M::Input;
    type Cost = M::Cost;
    type Bounds = M::Bounds;

    fn time_step<D: Linear>(
        &self,
        time: usize,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::State, Dual<D>> {
        M::time_step(self, time, state, input)
    }

    fn cost_vector<D: Linear>(
        &self,
        time: usize,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Cost, Dual<D>> {
        M::cost_vector(self, time, state, input)
    }

    fn input_ranges(&self, time: usize) -> Array<Self::Input, (f64, f64)> {
        M::input_ranges(self, time)
    }

    fn bounds<D: Linear>(
        &self,
        time: usize,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Bounds, Bounded<D>> {
        M::bounds(self, time, state, input)
    }
}
