//! An implementation of model predictive control using Riccati recursion.

use crate::array::{Concat, repeat};
use crate::math::{self, Dual, Float, Linear, Matrix, Vector, Zero, vector};
use crate::model::{Continuous, Discrete};
use crate::mpc::riccati::{
    backward_recursion_upto, factorize_upto, forward_recursion, update_factorization,
};
use crate::{Array, ArrayInst, GenArray};

pub mod export;
pub mod riccati;
pub mod sqp;

/// Settings for the solver.
#[derive(Clone)]
pub struct Settings<I: GenArray, B: GenArray> {
    /// Tolerance for the inputs.
    ///
    /// This determines how much of a violation the solver will accept when
    /// determining if the solution is feasible.
    pub input_tol: Array<I, f64>,
    /// Tolerance for the soft constraints.
    ///
    /// This determines how much of a violation of the slack variables the solver
    /// will accept when determining if the solution is feasible.
    pub bound_tol: Array<B, f64>,
}

impl<I: GenArray, B: GenArray> Default for Settings<I, B> {
    fn default() -> Self {
        let default_tol = if cfg!(feature = "f64") { 1e-6 } else { 1e-2 };
        Self {
            input_tol: repeat(default_tol),
            bound_tol: repeat(default_tol),
        }
    }
}

/// The bound an input is currently constrained to.
#[allow(missing_docs)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    Upper,
    Lower,
}

/// A change to a constraint.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Change {
    /// The index of the time step.
    time: usize,
    /// The index of the input.
    input: usize,
    /// The previous value of the bound.
    previous: Option<Bound>,
}

/// The state of a single time step in the MPC solver.
#[derive(Clone)]
pub struct MpcStep<S: GenArray, I: GenArray, C: GenArray, B: GenArray> {
    /// The state to linearize the model around.
    pub linearized_state: Array<S, f64>,
    /// The input to linearize the model around.
    pub linearized_input: Array<I, f64>,

    /// The state found by the solver.
    pub optimal_state: Array<S, f64>,
    /// The input found by the solver.
    pub optimal_input: Array<I, f64>,

    /// The allowed range of each input.
    input_ranges: Array<Concat<I, B>, (Float, Float)>,

    /// The current active set.
    active_set: Array<Concat<I, B>, Option<Bound>>,

    /// The `A` matrix in the time step `A*x + B*u + a`.
    state_step: Matrix<S, S>,
    /// The `B` matrix in the time step `A*x + B*u + a`.
    input_step: Matrix<S, Concat<I, B>>,
    /// The `a` vector in the time step `A*x + B*u + a`.
    const_step: Vector<S>,

    /// The `C` matrix in the cost function `|C*x + D*u + c|^2`.
    state_cost: Matrix<Concat<C, B>, S>,
    /// The `D` matrix in the cost function `|C*x + D*u + c|^2`.
    input_cost: Matrix<Concat<C, B>, Concat<I, B>>,
    /// The `c` vector in the cost function `|C*x + D*u + c|^2`.
    const_cost: Vector<Concat<C, B>>,

    /// The `P` matrix in the cost-to-go function `x^T*P*x - psi^T*x + const`.
    p_mat: Matrix<S, S>,
    /// The `psi` vector in the cost-to-go function `x^T*P*x - psi^T*x + const`.
    psi_vec: Vector<S>,

    // Note: The quantities below are for the next time index according to the
    // notation in the paper.
    /// The `H` matrix in the Riccati factorization.
    h_mat: Matrix<S, Concat<I, B>>,
    /// The pseudoinverse of the `G` matrix in the Riccati factorization.
    g_inv: Matrix<Concat<I, B>, Concat<I, B>>,

    /// The `K` matrix in the feedback `u = K*x + k`.
    k_mat: Matrix<Concat<I, B>, S>,
    /// The `k` vector in the feedback `u = K*x + k`.
    k_vec: Vector<Concat<I, B>>,
}

/// A type alias for the step type needed for a specific discrete time model.
pub type MpcStepFor<T> = MpcStep<
    <T as Discrete>::State,
    <T as Discrete>::Input,
    <T as Discrete>::Cost,
    <T as Discrete>::Bounds,
>;

/// Same as [`MpcStepFor`] but expects a continuous time model instead.
pub type MpcStepForCont<T> = MpcStep<
    <T as Continuous>::State,
    <T as Continuous>::Input,
    <T as Continuous>::Cost,
    <T as Continuous>::Bounds,
>;

impl<S: GenArray, I: GenArray, C: GenArray, B: GenArray> MpcStep<S, I, C, B> {
    /// Creates an instance of [`MpcStep`] with all matrices set to zero.
    pub const fn new(linearized_state: Array<S, f64>, linearized_input: Array<I, f64>) -> Self {
        Self {
            linearized_state,
            linearized_input,
            optimal_state: repeat(0.0),
            optimal_input: repeat(0.0),
            input_ranges: repeat((Float::ZERO, Float::ZERO)),
            active_set: repeat(None),
            state_step: Linear::ZERO,
            input_step: Linear::ZERO,
            const_step: Linear::ZERO,
            state_cost: Linear::ZERO,
            input_cost: Linear::ZERO,
            const_cost: Linear::ZERO,
            p_mat: Linear::ZERO,
            psi_vec: Linear::ZERO,
            g_inv: Linear::ZERO,
            h_mat: Linear::ZERO,
            k_mat: Linear::ZERO,
            k_vec: Linear::ZERO,
        }
    }

    /// Linearizes the model around a the state and input in `linearized_state` and `linearized_input`.
    ///
    /// If the model is time-dependent, the parameter `time` should be set to the
    /// index of this time step.
    pub fn linearize<M>(&mut self, model: &M, time: usize)
    where
        M: Discrete<State = S, Input = I, Cost = C, Bounds = B>,
    {
        // The derivative of the cost function is constant with respect to the slack
        // variables, so it doesn't matter what value they have during linearization.
        //
        // To do: Avoid linearizing with respect to the slack variables. The derivatives
        // are trivial so entering them manually would avoid putting them inside the AD
        // gradients, possibly saving a bit of computation time.
        let zeroed_slack: Array<B, Float> = repeat(Float::ZERO);
        let point = vector(
            self.linearized_state
                .map(Float::from)
                .concat(self.linearized_input.map(Float::from).concat(zeroed_slack)),
        );

        let bounds = model.bounds::<Zero>(
            time,
            self.linearized_state.map(Dual::from),
            self.linearized_input.map(Dual::from),
        );
        self.input_ranges = Concat(model.input_ranges(time), bounds.map(|b| (b.min, b.max)))
            .map(|(l, u)| (l.into(), u.into()));

        let (jac_step, const_step) =
            math::linearize(point, |Concat(state, Concat(input, _slack))| {
                model.time_step(time, state, input)
            });
        let (state_step, input_step) = jac_step.split_h();

        self.state_step = state_step;
        self.input_step = input_step;
        self.const_step = const_step;

        let (jac_cost, const_cost) =
            math::linearize(point, |Concat(state, Concat(input, slack))| {
                let cost = model.cost_vector(time, state, input);
                let violation_cost = model
                    .bounds(time, state, input)
                    .zip(slack)
                    .map(|(b, r)| (b.value - r) * b.weight);
                Concat(cost, violation_cost)
            });
        let (state_cost, input_cost) = jac_cost.split_h();

        self.state_cost = state_cost;
        self.input_cost = input_cost;
        self.const_cost = const_cost;
    }
}

/// Convenience function to call [`linearize`][MpcStep::linearize] on every
/// step.
#[allow(clippy::type_complexity)]
pub fn linearize<M: Discrete>(model: &M, steps: &mut [MpcStepFor<M>]) {
    for (i, step) in steps.iter_mut().enumerate() {
        step.linearize(model, i);
    }
}

/// Performs multiple iterations of the QP solver. The first return value is the
/// amount of iterations performed. The second return value is `true` if the QP
/// solver has converged.
///
/// Note that due to rounding errors there is a risk of the solver not detecting
/// convergence if the tolerances are too low.
pub fn iterate<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    initial_state: Array<S, f64>,
    steps: &mut [MpcStep<S, I, C, B>],
    max_iterations: usize,
    settings: &Settings<I, B>,
) -> (usize, bool) {
    // To do: Track the value of the cost function to detect convergence and/or
    // rounding errors from the rank-1 updates

    if max_iterations == 0 {
        return (0, false);
    }

    let mut changed = step(initial_state, steps, settings);
    let mut iterations = 1;

    while changed.is_some() && iterations < max_iterations {
        changed = step_incremental_update(initial_state, steps, changed, settings);
        iterations += 1;
    }

    (iterations, changed.is_none())
}

/// Performs a single iteration of the QP solver.
///
/// If the solution is suboptimal, then a struct representing a constraint that
/// was changed is returned. It can be used with [`step_update`] to make the
/// next iteration more efficient.
///
/// Note that due to rounding errors there is a risk of the solver not detecting
/// convergence if the tolerances are too low.
pub fn step<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    initial_state: Array<S, f64>,
    steps: &mut [MpcStep<S, I, C, B>],
    settings: &Settings<I, B>,
) -> Option<Change> {
    factorize_upto(steps, steps.len() - 1);
    backward_recursion_upto(steps, steps.len() - 1);
    forward_recursion(steps, initial_state, settings)
}

/// Same as [`step_incremental_update`] but recomputes the updated part from
/// scratch.
pub fn step_update<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    initial_state: Array<S, f64>,
    steps: &mut [MpcStep<S, I, C, B>],
    last_change: Option<Change>,
    settings: &Settings<I, B>,
) -> Option<Change> {
    if let Some(last_change) = last_change {
        factorize_upto(steps, last_change.time);
        backward_recursion_upto(steps, last_change.time);
    }
    forward_recursion(steps, initial_state, settings)
}

/// Performs a single iteration of the QP solver. This is similar to [`step`]
/// except more efficient as it performs an incremental update instead of
/// recomputing everything from scratch.
///
/// Note that the list of steps must not have been altered since the last call
/// to [`step`], as that violates the assumptions of this algorithm, likely
/// resulting in nonsensical results.
pub fn step_incremental_update<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    initial_state: Array<S, f64>,
    steps: &mut [MpcStep<S, I, C, B>],
    last_change: Option<Change>,
    settings: &Settings<I, B>,
) -> Option<Change> {
    if let Some(last_change) = last_change {
        update_factorization(steps, last_change);
        backward_recursion_upto(steps, last_change.time);
    }
    forward_recursion(steps, initial_state, settings)
}
