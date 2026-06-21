//! Implementation of SQP by altering

use core::iter::zip;

use crate::array::Concat;
use crate::math::{Dual, Float, Linear, Zero, differentiate, vector};
use crate::model::Discrete;
use crate::mpc::{MpcStep, MpcStepFor};
use crate::{Array, ArrayInst, GenArray};

const MIN_STEP: Float = Float::from_f64(1e-4);

/// Moves the trajectory used for linearization towards the optimum found by the
/// QP solver, using a trust region to limit the step size.
pub fn trust_step<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &mut [MpcStep<S, I, C, B>],
    state_trust: Array<S, f64>,
    input_trust: Array<I, f64>,
) {
    let trust = Concat(state_trust, input_trust).map(Float::from);

    let mut step_size = Float::from(1.0);

    for step in steps.iter() {
        let previous = Concat(step.linearized_state, step.linearized_input).map(Float::from);
        let target = Concat(step.optimal_state, step.optimal_input).map(Float::from);
        let distance = previous.zip(target).map(|(p, t)| (p - t).abs());
        for (&dist, &trust) in zip(distance.iter(), trust.iter()) {
            if dist * step_size > trust {
                step_size = trust / dist;
            }
        }
    }

    perform_step(steps, step_size);
}

/// A pareto optimum found by `filter_step`.
#[derive(Clone, Copy, Debug)]
pub struct FilterPoint {
    cost: Float,
    violation: Float,
}

impl FilterPoint {
    /// `true` if `self` is strictly better than `other`.
    fn dominates(&self, other: &Self) -> bool {
        self.cost < other.cost && self.violation < other.violation
    }
}

/// Moves the trajectory used for linearization towards the optimum found by the
/// QP solver, using a filter method to determine an appropriate step size.
///
/// This returns a [`FilterPoint`] representing the current progress of the SQP
/// solver. To decrease the risk of losing progress, the points found by
/// previous SQP iterations can be provided in `previous`.
///
/// If `previous` is always empty, then this becomes a Markov Filter
/// (<https://doi.org/10.1023/A:1020533003783>).
pub fn filter_step<M: Discrete>(
    model: &M,
    steps: &mut [MpcStepFor<M>],
    initial_state: Array<M::State, f64>,
    previous: &[FilterPoint],
) -> FilterPoint {
    let (current, deriv) = differentiate(vector([Float::ZERO]), |[step_size]| {
        let (cost, violation) = cost_violation(model, steps, step_size, initial_state);
        [cost, violation]
    });

    // If the direction is an ascent direction then there is no appropriate step
    // length.
    if deriv.into_array().iter().all(|&d| d > 0.0) {
        let [cost, violation] = current.into_array();
        return FilterPoint { cost, violation };
    }

    let mut step_size = Float::from(1.0);

    let point = loop {
        let actual = {
            let (cost, violation) =
                cost_violation::<_, Zero>(model, steps, step_size.into(), initial_state);
            FilterPoint {
                cost: cost.float_value(),
                violation: violation.float_value(),
            }
        };

        let goal = (current + step_size * deriv * Float::from(0.5)).into_array();
        let goal = FilterPoint {
            cost: goal[0],
            violation: goal[1],
        };

        if !goal.dominates(&actual) && !previous.iter().any(|p| p.dominates(&actual)) {
            break actual;
        }

        step_size *= 0.5;

        if step_size < MIN_STEP {
            break actual;
        }
    };

    perform_step(steps, step_size);
    point
}

/// Moves the trajectory used for linearization towards the optimum found by the
/// QP solver, using an L1 penalty function.
pub fn penalty_step<M: Discrete>(
    model: &M,
    steps: &mut [MpcStepFor<M>],
    initial_state: Array<M::State, f64>,
) {
    let mut largest_dual = Float::ZERO;
    for step in steps.iter() {
        let x = vector(step.optimal_state.map(Float::from));
        let lambda = step.p_mat * x - step.psi_vec;
        for &lam in lambda.into_array().iter() {
            largest_dual = largest_dual.max(lam);
        }
    }
    let penalty = 2.0 * largest_dual;

    let (current, deriv) = differentiate(vector([Float::ZERO]), |[step_size]| {
        let (cost, violation) = cost_violation(model, steps, step_size, initial_state);
        [cost + penalty * violation]
    });
    let [current, deriv] = [current, deriv].map(|v| v.into_scalar());

    // If the direction is an ascent direction then there is no appropriate step
    // length.
    if deriv > 0.0 {
        return;
    }

    let mut step_size = Float::from(1.0);
    loop {
        let actual = {
            let (cost, violation) =
                cost_violation::<_, Zero>(model, steps, step_size.into(), initial_state);
            (cost + penalty * violation).float_value()
        };
        let goal = current + step_size * deriv * 0.5;

        if actual < goal {
            break;
        }

        step_size *= 0.5;

        if step_size < MIN_STEP {
            break;
        }
    }

    perform_step(steps, step_size);
}

/// Moves the current trajectory towards the optimum found by the QP solver.
/// `step_size` can be set to a value less than `1` to take a smaller step.
fn perform_step<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &mut [MpcStep<S, I, C, B>],
    step_size: Float,
) {
    for step in steps {
        let previous = Concat(step.linearized_state, step.linearized_input).map(Float::from);
        let target = Concat(step.optimal_state, step.optimal_input).map(Float::from);
        let Concat(state, input) = previous.zip(target).map(|(p, t)| p + (t - p) * step_size);
        step.linearized_state = state.map(f64::from);
        step.linearized_input = input.map(f64::from);
    }
}

/// Computes the cost and total equality violation after taking a step towards
/// the trajectory found by the QP solver. This does not include violations of
/// inequality constraints as they shouldn't be violated.
fn cost_violation<M: Discrete, D: Linear>(
    model: &M,
    steps: &[MpcStepFor<M>],
    step_size: Dual<D>,
    initial_state: Array<M::State, f64>,
) -> (Dual<D>, Dual<D>) {
    let mut total_cost = Dual::from(0.0);
    let mut total_violation = Dual::from(0.0);

    let mut expected_state = initial_state.map(Dual::from);
    for (i, step) in steps.iter().enumerate() {
        let previous = Concat(step.linearized_state, step.linearized_input).map(Float::from);
        let target = Concat(step.optimal_state, step.optimal_input).map(Float::from);
        let Concat(state, input) = previous.zip(target).map(|(p, t)| p + (t - p) * step_size);

        for (&x1, &x2) in zip(state.iter(), expected_state.iter()) {
            total_violation += (x1 - x2).abs();
        }

        total_cost += model.cost(i, state, input);

        expected_state = model.time_step(i, state, input);
    }

    (total_cost, total_violation)
}
