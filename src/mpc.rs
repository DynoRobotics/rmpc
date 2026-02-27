//! Implementation of model predictive control.

use core::iter::zip;

use crate::array::{ArrayInst, Concat, from_fn, repeat};
use crate::math::{Linear, Matrix, Vector, inv_no_pivot, linearize, vector};
use crate::model::Model;
use crate::{Array, GenArray};

/// The bound an input is currently constrained to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bound {
    Upper,
    Lower,
}

/// The state of a single time step in the MPC solver.
pub struct MpcStep<S: GenArray, I: GenArray, C: GenArray> {
    /// The state found by the solver.
    pub optimal_state: Array<S, f64>,
    /// The input found by the solver.
    pub optimal_input: Array<I, f64>,

    /// The allowed range of each input.
    input_ranges: Array<I, (f64, f64)>,

    /// The current active set.
    active_set: Array<I, Option<Bound>>,

    /// The `A` matrix in the time step `A*x + B*u + a`.
    state_step: Matrix<S, S>,
    /// The `B` matrix in the time step `A*x + B*u + a`.
    input_step: Matrix<S, I>,
    /// The `a` vector in the time step `A*x + B*u + a`.
    const_step: Vector<S>,

    /// The `C` matrix in the cost function `|C*x + D*u + c|^2`.
    state_cost: Matrix<C, S>,
    /// The `D` matrix in the cost function `|C*x + D*u + c|^2`.
    input_cost: Matrix<C, I>,
    /// The `c` vector in the cost function `|C*x + D*u + c|^2`.
    const_cost: Vector<C>,

    /// Feedback from the state to a vector with the input values for unconstrained
    /// inputs and the dual variable for constrained inputs.
    state_mod_feedback: Matrix<I, S>,
    /// The constant term for the modified feedback vector.
    const_mod_feedback: Vector<I>,
}

impl<S: GenArray, I: GenArray, C: GenArray> MpcStep<S, I, C> {
    /// Creates an instance of [`MpcStep`] with all matrices set to zero.
    pub const fn new() -> Self {
        Self {
            optimal_state: repeat(0.0),
            optimal_input: repeat(0.0),
            input_ranges: repeat((0.0, 0.0)),
            active_set: repeat(None),
            state_step: Linear::ZERO,
            input_step: Linear::ZERO,
            const_step: Linear::ZERO,
            state_cost: Linear::ZERO,
            input_cost: Linear::ZERO,
            const_cost: Linear::ZERO,
            state_mod_feedback: Linear::ZERO,
            const_mod_feedback: Linear::ZERO,
        }
    }

    /// Linearizes the model around a specified state and input.
    ///
    /// If the model is time-dependent, the parameter `time` should be set to the
    /// index of this time step.
    pub fn linearize<M>(
        &mut self,
        model: &M,
        state: Array<S, f64>,
        input: Array<I, f64>,
        time: usize,
    ) where
        M: Model<State = S, Input = I, Cost = C>,
    {
        let point = vector(state.concat(input));

        self.input_ranges = model.input_ranges(time);

        let (jac_step, const_step) = linearize(point, |Concat(state, input)| {
            model.time_step(time, state, input)
        });
        let (state_step, input_step) = jac_step.split_h();

        self.state_step = state_step;
        self.input_step = input_step;
        self.const_step = const_step;

        let (jac_cost, const_cost) = linearize(point, |Concat(state, input)| {
            model.cost_vector(time, state, input)
        });
        let (state_cost, input_cost) = jac_cost.split_h();

        self.state_cost = state_cost;
        self.input_cost = input_cost;
        self.const_cost = const_cost;
    }
}

impl<S: GenArray, I: GenArray, C: GenArray> Clone for MpcStep<S, I, C> {
    fn clone(&self) -> Self {
        Self {
            optimal_state: self.optimal_state,
            optimal_input: self.optimal_input,
            input_ranges: self.input_ranges,
            active_set: self.active_set,
            state_step: self.state_step,
            input_step: self.input_step,
            const_step: self.const_step,
            state_cost: self.state_cost,
            input_cost: self.input_cost,
            const_cost: self.const_cost,
            state_mod_feedback: self.state_mod_feedback,
            const_mod_feedback: self.const_mod_feedback,
        }
    }
}

impl<S: GenArray, I: GenArray, C: GenArray> Default for MpcStep<S, I, C> {
    fn default() -> Self {
        Self::new()
    }
}

/// Performs a single iteration of the QP solver.
///
/// Returns `true` if convergence has been reached. Note that due to rounding
/// errors there is a risk of the solver not detecting convergence when it has
/// reached the optimum.
pub fn iterate<S: GenArray, I: GenArray, C: GenArray>(
    initial_state: Array<S, f64>,
    steps: &mut [MpcStep<S, I, C>],
) -> bool {
    // The optimal cost given some state is given by the formula
    // ```
    // 1/2 * state * cost_mat * state + cost_vec * state + const
    // ```
    // where `const` is a constant we don't track as it doesn't affect where the
    // optimum is. These are modified as we iterate backwards in the loop below as
    // we don't need them afterwards when the optimal feedback is known.
    let mut cost_mat = Matrix::ZERO;
    let mut cost_vec = Vector::ZERO;

    for step in steps.iter_mut().rev() {
        let upper = vector(step.input_ranges.map(|(_, upper)| upper));
        let lower = vector(step.input_ranges.map(|(lower, _)| lower));

        assert!(
            zip(lower.into_array().iter(), upper.into_array().iter()).all(|(l, u)| l < u),
            "the lower bounds must be less than the upper bounds",
        );

        // Matrices used to find the optimal feedback.
        let mut input_cost_mat = step.input_step.transpose() * cost_mat * step.input_step
            + step.input_cost.transpose() * step.input_cost;

        let cross_cost_mat = step.input_step.transpose() * cost_mat * step.state_step
            + step.input_cost.transpose() * step.state_cost;
        let mut cross_cost_vec = step.input_step.transpose()
            * (cost_mat * step.const_step + cost_vec)
            + step.input_cost.transpose() * step.const_cost;

        // Modify the cost matrix to instead reflect the KKT conditions of the optimum
        // with some inputs constrained. This modified matrix is no longer positive
        // definite but it can be shown that it still meets the conditions for
        // `inv_no_pivot`.
        for (input_i, active) in step.active_set.iter_mut().enumerate() {
            let (sign, fixed) = match active {
                Some(Bound::Lower) => (-1.0, lower[input_i]),
                Some(Bound::Upper) => (1.0, upper[input_i]),
                None => continue,
            };

            // Release infinite bounds from the active set immediately.
            if !fixed.is_finite() {
                *active = None;
                continue;
            }

            cross_cost_vec += fixed * input_cost_mat.col(input_i);
            input_cost_mat.set_col(input_i, Vector::ZERO);
            input_cost_mat[(input_i, input_i)] = -sign;
        }

        // Solve the KKT conditions to get the optimal feedback.
        let inv_input_cost = inv_no_pivot(input_cost_mat);
        let mut state_mod_feedback = -inv_input_cost * cross_cost_mat;
        let mut const_mod_feedback = -inv_input_cost * cross_cost_vec;
        step.state_mod_feedback = state_mod_feedback;
        step.const_mod_feedback = const_mod_feedback;

        // Revert the feedback to use the known inputs instead of dual variables.
        for (input_i, active) in step.active_set.iter().enumerate() {
            let fixed = match active {
                Some(Bound::Lower) => lower[input_i],
                Some(Bound::Upper) => upper[input_i],
                None => continue,
            };

            state_mod_feedback.set_row(input_i, Vector::ZERO);
            const_mod_feedback[input_i] = fixed;
        }

        // Find the state cost function using that feedback.
        let closed_state_step = step.state_step + step.input_step * state_mod_feedback;
        let closed_const_step = step.const_step + step.input_step * const_mod_feedback;
        let closed_state_cost = step.state_cost + step.input_cost * state_mod_feedback;
        let closed_const_cost = step.const_cost + step.input_cost * const_mod_feedback;

        (cost_mat, cost_vec) = (
            closed_state_step.transpose() * cost_mat * closed_state_step
                + closed_state_cost.transpose() * closed_state_cost,
            closed_state_step.transpose() * (cost_mat * closed_const_step + cost_vec)
                + closed_state_cost.transpose() * closed_const_cost,
        );
    }

    // With the optimal feedback known, we can find the actual trajectory and see
    // which inputs should be activated or deactivated.
    let mut state = vector(initial_state);

    // We want to find the constrained input with the largest positive dual variable
    // if there is one, or the unconstrained input farthest outside its bounds
    // otherwise.
    let mut to_change = (f64::NEG_INFINITY, 0, 0, None);

    for (time_i, step) in steps.iter_mut().enumerate() {
        let upper = vector(step.input_ranges.map(|(_, upper)| upper));
        let lower = vector(step.input_ranges.map(|(lower, _)| lower));

        // The free inputs and dual variables at the current time step.
        let mut mod_input = step.state_mod_feedback * state + step.const_mod_feedback;

        // Check all the constraints while replacing the dual with actual input values.
        for (input_i, active) in step.active_set.iter().enumerate() {
            if let Some(active) = active {
                // See if the dual is binding in the wrong direction, and if so, by how much.
                let dual = mod_input[input_i];
                let priority = dual;
                if dual > 0.0 && priority > to_change.0 {
                    to_change = (priority, time_i, input_i, None);
                }

                // Replace the dual variable with the actual input.
                mod_input[input_i] = match active {
                    Bound::Lower => lower[input_i],
                    Bound::Upper => upper[input_i],
                };
            } else {
                // See if we are outside the input bounds, and if so, by how much.
                let (distance, bound) = if mod_input[input_i] > upper[input_i] {
                    (mod_input[input_i] - upper[input_i], Bound::Upper)
                } else if mod_input[input_i] < lower[input_i] {
                    (lower[input_i] - mod_input[input_i], Bound::Lower)
                } else {
                    continue;
                };

                let priority = -1.0 / distance;
                if priority > to_change.0 {
                    to_change = (priority, time_i, input_i, Some(bound));
                }
            }
        }

        step.optimal_state = state.into_array();
        step.optimal_input = from_fn(|i| mod_input[i].clamp(lower[i], upper[i]));

        // Continue the trajectory using this input.
        state = step.state_step * state + step.input_step * mod_input + step.const_step;
    }

    // If we found a problem, the solution is suboptimal and we need to activate or
    // deactivate a constraint to resolve it.
    let optimal = to_change.0.is_infinite();
    if !optimal {
        steps[to_change.1].active_set.as_mut_slice()[to_change.2] = to_change.3;
    }

    optimal
}
