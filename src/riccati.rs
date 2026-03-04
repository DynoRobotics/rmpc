//! Rewrite of the [`mpc`](crate::mpc) module using a standard Riccati
//! recursion.

use crate::array::{Concat, from_fn, repeat};
use crate::math::{Linear, Matrix, Vector, inv_no_pivot, linearize, vector};
use crate::model::Model;
use crate::{Array, ArrayInst, GenArray};

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
pub struct RiccatiStep<S: GenArray, I: GenArray, C: GenArray> {
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

    // Note: These are for the next time step according to the notation in the
    // paper. This simplifies the implementation as those values are almost always
    // the value of interest.
    p_mat: Matrix<S, S>,
    h_mat: Matrix<S, I>,
    g_inv: Matrix<I, I>,
    k_mat: Matrix<I, S>,
    psi_vec: Vector<S>,
    k_vec: Vector<I>,
}

impl<S: GenArray, I: GenArray, C: GenArray> RiccatiStep<S, I, C> {
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
            p_mat: Linear::ZERO,
            g_inv: Linear::ZERO,
            h_mat: Linear::ZERO,
            k_mat: Linear::ZERO,
            psi_vec: Linear::ZERO,
            k_vec: Linear::ZERO,
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

impl<S: GenArray, I: GenArray, C: GenArray> Default for RiccatiStep<S, I, C> {
    fn default() -> Self {
        Self::new()
    }
}

/// Performs a single iteration of the QP solver.
///
/// If the solution is suboptimal, then a struct representing a constraint that
/// was changed is returned. It can be used with [`step_update`] to make the
/// next iteration more efficient.
///
/// Note that due to rounding errors there is a risk of the solver not detecting
/// convergence when it has reached the optimum.
pub fn step<S: GenArray, I: GenArray, C: GenArray>(
    initial_state: Array<S, f64>,
    steps: &mut [RiccatiStep<S, I, C>],
) -> Option<Change> {
    factorize_upto(steps, steps.len() - 1);
    backward_recursion_upto(steps, steps.len() - 1);
    forward_recursion(steps, initial_state)
}

/// Same as [`step_update`] but recomputes the updated part from scratch.
pub fn step_old_update<S: GenArray, I: GenArray, C: GenArray>(
    initial_state: Array<S, f64>,
    steps: &mut [RiccatiStep<S, I, C>],
    last_change: Change,
) -> Option<Change> {
    factorize_upto(steps, last_change.time);
    backward_recursion_upto(steps, last_change.time);
    forward_recursion(steps, initial_state)
}

/// Performs a single iteration of the QP solver. This is similar to [`step`]
/// except more efficient as it performs an incremental update instead of
/// recomputing everything from scratch.
///
/// Note that the list of steps must not have been altered since the last call
/// to [`step`], as that violates the assumptions of this algorithm, likely
/// resulting in nonsensical results.
pub fn step_update<S: GenArray, I: GenArray, C: GenArray>(
    initial_state: Array<S, f64>,
    steps: &mut [RiccatiStep<S, I, C>],
    last_change: Change,
) -> Option<Change> {
    update_factorization(steps, last_change);
    backward_recursion_upto(steps, last_change.time);
    forward_recursion(steps, initial_state)
}

/// Performs the factorization algorithm. Assumes that the steps after `last`
/// have already been computed.
fn factorize_upto<S: GenArray, I: GenArray, C: GenArray>(
    steps: &mut [RiccatiStep<S, I, C>],
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
    }
}

fn update_factorization<S: GenArray, I: GenArray, C: GenArray>(
    steps: &mut [RiccatiStep<S, I, C>],
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

        step.g_inv.set_col(change.input, Vector::ZERO);
        step.g_inv.set_row(change.input, Vector::ZERO);

        // Update H
        step.h_mat.set_col(change.input, Vector::ZERO);
    }

    let numerator = h - step.h_mat * (step.g_inv * g);
    let denominator = (g0 - g.transpose() * step.g_inv * g).into_scalar();

    let mut v_vec = numerator * (1.0 / libm::sqrt(denominator));
    step.k_mat -= (alpha / denominator) * (step.g_inv * g - pi) * numerator.transpose();

    if !adding {
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
fn backward_recursion_upto<S: GenArray, I: GenArray, C: GenArray>(
    steps: &mut [RiccatiStep<S, I, C>],
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
fn forward_recursion<S: GenArray, I: GenArray, C: GenArray>(
    steps: &mut [RiccatiStep<S, I, C>],
    initial_state: Array<S, f64>,
) -> Option<Change> {
    let mut x = vector(initial_state);

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
        for (j, bound) in step.active_set.iter().enumerate() {
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

                if amount > worst_violation.0 {
                    worst_violation = (amount, i, j, bound);
                }
            }
        }

        step.optimal_state = x.into_array();
        step.optimal_input = u
            .into_array()
            .zip(step.input_ranges)
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
