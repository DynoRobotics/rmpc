//! Implementation of model predictive control.

use std::marker::PhantomData;

use crate::array::{ArrayInst, Concat, from_fn, repeat};
use crate::math::{self, Dual, Linear, Matrix, Vector, inv_no_pivot, vector};
use crate::model::Model;
use crate::{Array, GenArray};

/// A linearized version of a model around some state and input.
struct Linearized<M: Model> {
    state_step: Matrix<M::State, M::State>,
    input_step: Matrix<M::State, M::Input>,
    const_step: Vector<M::State>,
    state_cost: Matrix<M::Cost, M::State>,
    input_cost: Matrix<M::Cost, M::Input>,
    const_cost: Vector<M::Cost>,
    model: PhantomData<M>,
}

impl<M: Model> Linearized<M> {
    /// Linearizes the model.
    fn at_point(model: &M, state: Vector<M::State>, input: Vector<M::Input>) -> Self {
        let (state_step, input_step, const_step) =
            linearize_func(state, input, |state, input| model.time_step(state, input));
        let (state_cost, input_cost, const_cost) =
            linearize_func(state, input, |state, input| model.cost_vector(state, input));

        Linearized {
            state_step,
            input_step,
            const_step,
            state_cost,
            input_cost,
            const_cost,
            model: PhantomData,
        }
    }
}

/// Approximates a non-linear function `f(x, u)` as `A*x + B*u + c`. Used by
fn linearize_func<State: GenArray, Input: GenArray, Output: GenArray>(
    state: Vector<State>,
    input: Vector<Input>,
    f: impl FnOnce(
        Array<State, Dual<Vector<Concat<State, Input>>>>,
        Array<Input, Dual<Vector<Concat<State, Input>>>>,
    ) -> Array<Output, Dual<Vector<Concat<State, Input>>>>,
) -> (Matrix<Output, State>, Matrix<Output, Input>, Vector<Output>) {
    let point = state.concat(input);
    let f = |Concat(state, input)| f(state, input);

    let (value, jacobian) = math::linearize::<_, Output>(point, f);
    let constant = value - jacobian * point;

    let (jac_state, jac_input) = jacobian.split_h();
    (jac_state, jac_input, constant)
}

/// The current state of the MPC solver.
pub struct Mpc<M: Model, const N: usize> {
    state_traj: [Array<M::State, f64>; N],
    input_traj: [Array<M::Input, f64>; N],
    bounds: [Array<M::Input, Option<Bound>>; N],
    lower: [Array<M::Input, f64>; N],
    upper: [Array<M::Input, f64>; N],
    model: PhantomData<fn(&M)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bound {
    Upper,
    Lower,
}

impl<M: Model, const N: usize> Mpc<M, N> {
    /// Initializes the MPC.
    pub fn new(
        state_traj: [Array<M::State, f64>; N],
        input_traj: [Array<M::Input, f64>; N],
        lower: [Array<M::Input, f64>; N],
        upper: [Array<M::Input, f64>; N],
    ) -> Self {
        Mpc {
            state_traj,
            input_traj,
            bounds: [repeat(None); N],
            lower,
            upper,
            model: PhantomData,
        }
    }

    /// Shifts the input constraints by a certain number of time steps, reducing the
    /// amount of iterations needed when computing the next time step.
    pub fn shift(&mut self, steps: usize) {
        self.bounds[..steps].fill(repeat(None));
        self.bounds.rotate_left(steps);
    }

    /// Performs a single QP iteration. Returns `true` or `false` depending on if
    /// convergence has been reached, along with the input.
    pub fn iterate(
        &mut self,
        model: &M,
        initial_state: Array<M::State, f64>,
    ) -> (bool, Array<M::Input, f64>) {
        assert!(N > 0, "needs at least one time step before the horizon");

        // Linearized models at each time step
        let models: [_; N] = std::array::from_fn(|time_i| {
            Linearized::at_point(
                model,
                Vector(self.state_traj[time_i]),
                Vector(self.input_traj[time_i]),
            )
        });

        // At each point in time we have a vector `mod_input` consisting of the input
        // value for non-constrained inputs and the dual variables for constrained
        // inputs. This vector is given by the formula
        // ```
        // mod_input = mod_feedback_mat * states + mod_feedback_vec;
        // ```
        // Note that this differs from the L matrix in most literature as there is no
        // minus sign in the formula.
        let mut mod_feedback_mat = [Matrix::ZERO; N];
        let mut mod_feedback_vec = [Vector::ZERO; N];

        // The optimal cost given some state is given by the formula
        // ```
        // state * cost_mat * state + cost_vec * state + const
        // ```
        // where `const` is a constant we don't track as it doesn't affect where the
        // optimum is. These are modified as we iterate backwards in the loop below as
        // we don't need them afterwards when the optimal feedback is known.
        let mut cost_mat = Matrix::ZERO;
        let mut cost_vec = Vector::ZERO;

        for time_i in (0..N).rev() {
            let model = &models[time_i];
            let upper = vector(self.upper[time_i]);
            let lower = vector(self.lower[time_i]);
            let bounds = self.bounds[time_i];

            // Matrices used to find the optimal feedback.
            let mut input_cost = model.input_step.transpose() * cost_mat * model.input_step
                + model.input_cost.transpose() * model.input_cost;

            let cross_cost_mat = model.input_step.transpose() * cost_mat * model.state_step
                + model.input_cost.transpose() * model.state_cost;
            let mut cross_cost_vec = model.input_step.transpose()
                * (cost_mat * model.const_step + cost_vec)
                + model.input_cost.transpose() * model.const_cost;

            // Modify the cost matrix to instead reflect the KKT conditions of the optimum
            // with some inputs constrained. This modified matrix is no longer positive
            // definite but It can be shown that it still meets the conditions for
            // `inv_no_pivot`.
            for (input_i, bound) in bounds.as_ref().iter().enumerate() {
                let (sign, fixed) = match bound {
                    Some(Bound::Lower) => (-1.0, lower[input_i]),
                    Some(Bound::Upper) => (1.0, upper[input_i]),
                    None => continue,
                };

                cross_cost_vec += fixed * input_cost.col(input_i);
                input_cost.set_col(input_i, Vector::ZERO);
                input_cost[(input_i, input_i)] = -sign;
            }

            // Solve the KKT conditions to get the optimal feedback.
            let inv_input_cost = inv_no_pivot(input_cost);
            let mut feedback_mat = -inv_input_cost * cross_cost_mat;
            let mut feedback_vec = -inv_input_cost * cross_cost_vec;
            mod_feedback_mat[time_i] = feedback_mat;
            mod_feedback_vec[time_i] = feedback_vec;

            // Revert the feedback to use the known inputs instead of dual variables.
            for (input_i, bound) in bounds.as_ref().iter().enumerate() {
                let fixed = match bound {
                    Some(Bound::Lower) => lower[input_i],
                    Some(Bound::Upper) => upper[input_i],
                    None => continue,
                };

                feedback_mat.set_row(input_i, Vector::ZERO);
                feedback_vec[input_i] = fixed;
            }

            // Find the state cost function using that feedback.
            let closed_state_step = model.state_step + model.input_step * feedback_mat;
            let closed_const_step = model.const_step + model.input_step * feedback_vec;
            let closed_state_cost = model.state_cost + model.input_cost * feedback_mat;
            let closed_const_cost = model.const_cost + model.input_cost * feedback_vec;

            (cost_mat, cost_vec) = (
                closed_state_step.transpose() * cost_mat * closed_state_step
                    + closed_state_cost.transpose() * closed_state_cost,
                closed_state_step.transpose() * (cost_mat * closed_const_step + cost_vec)
                    + closed_state_cost.transpose() * closed_const_cost,
            );
        }

        // With the optimal feedback known, we can find the actual trajectory and see
        // which inputs should be activated or deactivated.
        let mut state = Vector(initial_state);
        let mut first_input = None;

        // We want to find the constrained input with the largest positive dual variable
        // if there is one, or the unconstrained input farthest outside its bounds
        // otherwise.
        let mut to_change = (f64::NEG_INFINITY, 0, 0, None);

        for time_i in 0..N {
            let model = &models[time_i];
            let upper = vector(self.upper[time_i]);
            let lower = vector(self.lower[time_i]);
            let bounds = self.bounds[time_i];

            // The free inputs and dual variables at the current time step.
            let mut mod_input = mod_feedback_mat[time_i] * state + mod_feedback_vec[time_i];

            // Check all the constraints while replacing the dual
            for (input_i, bound) in bounds.as_ref().iter().enumerate() {
                if let Some(bound) = bound {
                    // See if the dual is binding in the wrong direction, and if so, by how much.
                    let dual = mod_input[input_i];
                    let priority = dual;
                    if dual > 0.0 && priority > to_change.0 {
                        to_change = (priority, time_i, input_i, None);
                    }

                    // Replace the dual variable with the actual input.
                    mod_input[input_i] = match bound {
                        Bound::Lower => lower[input_i],
                        Bound::Upper => upper[input_i],
                    };
                } else {
                    // See if we are outside the input bounds, and if so, by how much.
                    let (near, far, bound) = if mod_input[input_i] > upper[input_i] {
                        (upper[input_i], lower[input_i], Bound::Upper)
                    } else if mod_input[input_i] < lower[input_i] {
                        (lower[input_i], upper[input_i], Bound::Lower)
                    } else {
                        continue;
                    };
                    let relative_distance = (mod_input[input_i] - near) / (near - far);

                    let priority = -1.0 / relative_distance;
                    if priority > to_change.0 {
                        to_change = (priority, time_i, input_i, Some(bound));
                    }
                }
            }

            // Continue the trajectory using this input.
            state = model.state_step * state + model.input_step * mod_input + model.const_step;

            if time_i == 0 {
                first_input = Some(from_fn(|i| mod_input[i].clamp(lower[i], upper[i])));
            }
        }

        // If we found a problem, the solution is suboptimal and we need to activate or
        // deactivate a constraint to resolve it.
        let optimal = to_change.0.is_infinite();
        if !optimal {
            self.bounds[to_change.1].as_mut()[to_change.2] = to_change.3;
        }

        (optimal, first_input.unwrap())
    }
}
