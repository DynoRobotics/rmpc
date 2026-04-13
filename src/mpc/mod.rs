//! An implementation of model predictive control using Riccati recursion.

use core::iter::zip;

use crate::array::{Concat, from_fn, repeat};
use crate::math::{self, Dual, Linear, Matrix, Vector, Zero, inv_no_pivot, vector};
use crate::model::Discrete;
use crate::{Array, ArrayInst, GenArray};

pub mod export;

/// Settings for the solver.
#[derive(Clone)]
pub struct Settings<I: GenArray, B: GenArray> {
    /// Tolerance for the inputs.
    ///
    /// This determines how much of a violation the solver will accept when
    /// determining if the solution is feasible.
    pub input_tol: Array<I, f64>,
    /// Tolerance for the inputs.
    ///
    /// This determines how much of a violation of the slack variables the solver
    /// will accept when determining if the solution is feasible.
    pub bound_tol: Array<B, f64>,
}

impl<I: GenArray, B: GenArray> Default for Settings<I, B> {
    fn default() -> Self {
        Self {
            input_tol: repeat(1e-6),
            bound_tol: repeat(1e-6),
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
    input_ranges: Array<Concat<I, B>, (f64, f64)>,

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

    // Note: These are for the next time step according to the notation in the
    // paper. This simplifies the implementation as those values are almost always
    // the value of interest.
    p_mat: Matrix<S, S>,
    h_mat: Matrix<S, Concat<I, B>>,
    g_inv: Matrix<Concat<I, B>, Concat<I, B>>,
    k_mat: Matrix<Concat<I, B>, S>,
    psi_vec: Vector<S>,
    k_vec: Vector<Concat<I, B>>,
}

impl<S: GenArray, I: GenArray, C: GenArray, B: GenArray> MpcStep<S, I, C, B> {
    /// Creates an instance of [`RiccatiStep`] with all matrices set to zero.
    pub const fn new(linearized_state: Array<S, f64>, linearized_input: Array<I, f64>) -> Self {
        Self {
            linearized_state,
            linearized_input,
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
            p_mat: Linear::ZERO,
            g_inv: Linear::ZERO,
            h_mat: Linear::ZERO,
            k_mat: Linear::ZERO,
            psi_vec: Linear::ZERO,
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
        let zeroed_slack: Array<B, f64> = repeat(0.0);
        let point = vector(
            self.linearized_state
                .concat(self.linearized_input.concat(zeroed_slack)),
        );

        let bounds = model.bounds::<Zero>(
            time,
            self.linearized_state.map(Dual::from),
            self.linearized_input.map(Dual::from),
        );
        self.input_ranges = Concat(model.input_ranges(time), bounds.map(|b| (b.min, b.max)));

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
pub fn linearize<M: Discrete>(
    model: &M,
    steps: &mut [MpcStep<M::State, M::Input, M::Cost, M::Bounds>],
) {
    for (i, step) in steps.iter_mut().enumerate() {
        step.linearize(model, i);
    }
}

/// Moves the trajectory used for linearization towards the optimum found by the
/// QP solver.
pub fn sqp_step<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &mut [MpcStep<S, I, C, B>],
    state_trust: Array<S, f64>,
    input_trust: Array<I, f64>,
) {
    let trust = Concat(state_trust, input_trust);

    let mut maximum_step = 1.0;

    for step in steps.iter() {
        let previous = Concat(step.linearized_state, step.linearized_input);
        let target = Concat(step.optimal_state, step.optimal_input);
        let distance = previous.zip(target).map(|(p, t)| (p - t).abs());
        for (&dist, &trust) in zip(distance.iter(), trust.iter()) {
            if dist * maximum_step > trust {
                maximum_step = trust / dist;
            }
        }
    }

    for step in steps {
        let previous = Concat(step.linearized_state, step.linearized_input);
        let target = Concat(step.optimal_state, step.optimal_input);
        let target = previous
            .zip(target)
            .map(|(p, t)| p + (t - p) * maximum_step);
        step.linearized_state = target.0;
        step.linearized_input = target.1;
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

/// Performs the factorization algorithm. Assumes that the steps after `last`
/// have already been computed.
fn factorize_upto<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &mut [MpcStep<S, I, C, B>],
    last: usize,
) {
    // To do: Possibly support a mayer term
    let mut p_mat = if last == steps.len() - 1 {
        Matrix::ZERO
    } else {
        steps[last].p_mat
    };

    for step in steps[..=last].iter_mut().rev() {
        step.p_mat = p_mat;

        let cost_xx = step.state_cost.transpose() * step.state_cost;
        let cost_xu = step.state_cost.transpose() * step.input_cost;
        let cost_uu = step.input_cost.transpose() * step.input_cost;

        let tmp = step.state_step.transpose() * p_mat;
        let f_mat = cost_xx + tmp * step.state_step;
        step.h_mat = cost_xu + tmp * step.input_step;

        let mut g_mat = cost_uu + step.input_step.transpose() * p_mat * step.input_step;

        for (i, bound) in step.active_set.iter_mut().enumerate() {
            // Sanity check to release infinite bounds immediately
            let (lower, upper) = step.input_ranges.as_slice()[i];
            let value = match bound {
                Some(Bound::Upper) => upper,
                Some(Bound::Lower) => lower,
                None => continue,
            };
            if !value.is_finite() {
                *bound = None;
                continue;
            }

            // Fixed inputs should not be included in `G` or `H`, so they are set to zero.
            step.h_mat.set_col(i, Vector::ZERO);
            g_mat.set_col(i, Vector::ZERO);
            g_mat.set_row(i, Vector::ZERO);
        }

        step.g_inv = inv_no_pivot(g_mat);
        step.k_mat = -step.g_inv * step.h_mat.transpose();

        p_mat = f_mat + step.h_mat * step.k_mat;

        // Due to rounding errors, we can't be sure that the comptuation above yields a
        // symmetric matrix. These rounding errors seem to grow uncontrollably in some
        // cases, so to avoid that we will enforce symmetry.
        p_mat = (p_mat + p_mat.transpose()) * 0.5;
    }
}

/// Uses rank-1 updates to efficiently update the factorization when a single
/// constraint is added or removed.
fn update_factorization<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &mut [MpcStep<S, I, C, B>],
    change: Change,
) {
    let t = change.time;
    let i = change.input;

    let step = &mut steps[t];

    let adding = step.active_set.as_slice()[i].is_some();
    let alpha = if adding { 1.0 } else { -1.0 };
    let pi = Vector(from_fn(|j| if i == j { 1.0 } else { 0.0 }));

    let b = step.input_step.col(i);
    let d = step.input_cost.col(i);

    let g0 = d.transpose() * d + b.transpose() * step.p_mat * b;
    let h = step.state_cost.transpose() * d + step.state_step.transpose() * step.p_mat * b;

    let mut g = step.input_cost.transpose() * d + step.input_step.transpose() * step.p_mat * b;
    for (i, bound) in step.active_set.iter().enumerate() {
        if bound.is_some() {
            g[i] = 0.0;
        }
    }
    if !adding {
        g[i] -= g0.into_scalar();
    }

    if adding {
        // Update G_inv
        let col = step.g_inv.col(change.input);
        let cell = col[change.input];
        step.g_inv -= col / cell * col.transpose();

        // Get rid of any cancellation errors
        step.g_inv.set_col(change.input, Vector::ZERO);
        step.g_inv.set_row(change.input, Vector::ZERO);

        // Update H
        step.h_mat.set_col(change.input, Vector::ZERO);
    }

    let numerator = h - step.h_mat * (step.g_inv * g);
    let denominator = (g0 - g.transpose() * step.g_inv * g).into_scalar();

    let mut v_vec = numerator * (1.0 / libm::sqrt(denominator));
    step.k_mat -= (alpha / denominator) * (step.g_inv * g - pi) * numerator.transpose();

    if adding {
        // Get rid of any cancellation errors
        step.k_mat.set_row(change.input, Vector::ZERO);
    } else {
        // Update G_inv
        let tmp = step.g_inv * g - pi;
        step.g_inv += (1.0 / denominator) * tmp * tmp.transpose();

        // Update H
        step.h_mat.set_col(i, h);
    }

    // Propagate the update to earlier time steps
    for step in steps[..t].iter_mut().rev() {
        step.p_mat += alpha * v_vec * v_vec.transpose();

        let a = step.state_step.transpose() * v_vec;
        let mut b = step.input_step.transpose() * v_vec;

        for (i, bound) in step.active_set.iter().enumerate() {
            if bound.is_some() {
                b[i] = 0.0;
            }
        }

        let tmp = step.g_inv * b;
        step.g_inv -= tmp / (alpha + (b.transpose() * tmp).into_scalar()) * tmp.transpose();

        step.h_mat += alpha * a * b.transpose();

        let tmp = a + step.k_mat.transpose() * b;
        v_vec = libm::sqrt(1.0 - alpha * (b.transpose() * step.g_inv * b).into_scalar()) * tmp;

        step.k_mat -= alpha * (step.g_inv * b) * tmp.transpose();
    }
}

/// Performs the backward recursion algorithm.
fn backward_recursion_upto<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &mut [MpcStep<S, I, C, B>],
    last: usize,
) {
    // To do: Possibly support a mayer term
    let mut psi_vec = if last == steps.len() - 1 {
        Vector::ZERO
    } else {
        steps[last].psi_vec
    };

    for step in steps[..=last].iter_mut().rev() {
        step.psi_vec = psi_vec;

        let const_input = vector(step.active_set.zip(step.input_ranges).map(
            |(bound, (lower, upper))| match bound {
                Some(Bound::Lower) => lower,
                Some(Bound::Upper) => upper,
                None => 0.0,
            },
        ));

        let av = step.input_step * const_input + step.const_step;
        let cv = step.input_cost * const_input + step.const_cost;
        let tmp = step.psi_vec - step.p_mat * av;

        step.k_vec = step.g_inv
            * (step.input_step.transpose() * tmp - step.input_cost.transpose() * step.const_cost);

        psi_vec = step.state_step.transpose() * tmp
            - step.state_cost.transpose() * cv
            - step.h_mat * step.k_vec;
    }
}

/// Performs the forward recursion algorithm and updates a single constraint if
/// the solution is suboptimal or infeasible. Returns a struct containing
/// information about the constraint that was changed.
pub fn forward_recursion<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &mut [MpcStep<S, I, C, B>],
    initial_state: Array<S, f64>,
    settings: &Settings<I, B>,
) -> Option<Change> {
    let mut x = vector(initial_state);
    let tolerance = Concat(settings.input_tol, settings.bound_tol);

    let mut worst_violation = (0.0, 0, 0, Bound::Lower);
    let mut worst_dual = (0.0, 0, 0);

    for (i, step) in steps.iter_mut().enumerate() {
        let const_input = vector(step.active_set.zip(step.input_ranges).map(
            |(bound, (lower, upper))| match bound {
                Some(Bound::Lower) => lower,
                Some(Bound::Upper) => upper,
                None => 0.0,
            },
        ));

        let u = step.k_mat * x + step.k_vec + const_input;

        // Check for violated constraints
        for (j, (&bound, &tol)) in step.active_set.iter().zip(tolerance.iter()).enumerate() {
            if bound.is_none() {
                let value = u[j];
                let (lower, upper) = step.input_ranges.as_slice()[j];

                let (amount, bound) = if value > upper {
                    (value - upper, Bound::Upper)
                } else if value < lower {
                    (lower - value, Bound::Lower)
                } else {
                    continue;
                };

                if amount > worst_violation.0 && amount > tol {
                    worst_violation = (amount, i, j, bound);
                }
            }
        }

        step.optimal_state = x.into_array();
        step.optimal_input = u
            .into_array()
            .zip(step.input_ranges)
            .0
            .map(|(val, (lower, upper))| val.clamp(lower, upper));

        let y = step.state_cost * x + step.input_cost * u + step.const_cost;
        x = step.state_step * x + step.input_step * u + step.const_step;

        // Check for constraints binding in the wrong direction
        let lambda = step.p_mat * x - step.psi_vec;
        let dual = step.input_cost.transpose() * y + step.input_step.transpose() * lambda;
        for (j, bound) in step.active_set.iter().enumerate() {
            let dual = match bound {
                Some(Bound::Upper) => dual[j],
                Some(Bound::Lower) => -dual[j],
                None => continue,
            };

            if dual > worst_dual.0 {
                worst_dual = (dual, i, j);
            }
        }
    }

    // Remove a constraint binding in the wrong direction
    if worst_dual.0 > 0.0 {
        let previous = steps[worst_dual.1].active_set.as_mut_slice()[worst_dual.2].take();
        return Some(Change {
            time: worst_dual.1,
            input: worst_dual.2,
            previous,
        });
    }

    // Add a constraint that has been violated
    if worst_violation.0 > 0.0 {
        steps[worst_violation.1].active_set.as_mut_slice()[worst_violation.2] =
            Some(worst_violation.3);
        return Some(Change {
            time: worst_violation.1,
            input: worst_violation.2,
            previous: None,
        });
    }

    // All KKT conditions are satisified, the solution is optimal.
    None
}
