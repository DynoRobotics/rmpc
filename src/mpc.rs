//! Implementation of model predictive control.

use std::marker::PhantomData;

use crate::array::Concat;
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

/// Solves unbounded MPC linearized around some state and input.
pub fn lq<M: Model, const N: usize>(
    model: &M,
    initial: Array<M::State, f64>,
    lin_state: Array<M::State, f64>,
    lin_input: Array<M::Input, f64>,
) -> Array<M::Input, f64> {
    let initial = vector(initial);
    let lin_state = vector(lin_state);
    let lin_input = vector(lin_input);

    let linear = Linearized::at_point(model, lin_state, lin_input);

    let mut cost_mat = Matrix::<M::State, M::State>::ZERO;
    let mut cost_vec = Vector::<M::State>::ZERO;

    assert!(N > 0, "empty horizon");

    for i in (0..N).rev() {
        let input_cost = linear.input_step.transpose() * cost_mat * linear.input_step
            + linear.input_cost.transpose() * linear.input_cost;

        let cross_cost_mat = linear.input_step.transpose() * cost_mat * linear.state_step
            + linear.input_cost.transpose() * linear.state_cost;
        let cross_cost_vec = linear.input_step.transpose()
            * (cost_mat * linear.const_step + cost_vec)
            + linear.input_cost.transpose() * linear.const_cost;

        let inv_input_cost = inv_no_pivot(input_cost);
        let feedback_mat = inv_input_cost * cross_cost_mat;
        let feedback_vec = inv_input_cost * cross_cost_vec;

        if i == 0 {
            return (-feedback_mat * initial - feedback_vec).0;
        }

        let closed_state_step = linear.state_step - linear.input_step * feedback_mat;
        let closed_const_step = linear.const_step - linear.input_step * feedback_vec;
        let closed_state_cost = linear.state_cost - linear.input_cost * feedback_mat;
        let closed_const_cost = linear.const_cost - linear.input_cost * feedback_vec;

        (cost_mat, cost_vec) = (
            closed_state_step.transpose() * cost_mat * closed_state_step
                + closed_state_cost.transpose() * closed_state_cost,
            closed_state_step.transpose() * (cost_mat * closed_const_step + cost_vec)
                + closed_state_cost.transpose() * closed_const_cost,
        );
    }

    unreachable!("the last iteration should have returned from the function");
}
