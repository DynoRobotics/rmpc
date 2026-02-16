//! Rewrite of the [`mpc`](crate::mpc) module using a standard Riccati
//! recursion.

use crate::array::{Concat, repeat};
use crate::math::{Linear, Matrix, Vector, inv_no_pivot, linearize, vector};
use crate::model::Model;
use crate::{Array, ArrayInst, GenArray};

/// The bound an input is currently constrained to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Bound {
    Upper,
    Lower,
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

    /// The dual variables of the input constraints.
    dual_variables: Vector<I>,

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

    p_mat: Matrix<S, S>,
    // Note: These are for the next time step according to the notation in the paper.
    h_mat: Matrix<S, I>,
    g_inv: Matrix<I, I>,
    k_mat: Matrix<I, S>,

    psi_vec: Vector<S>,
    // Note: This is for the next time step according to the notation in the paper.
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
            dual_variables: Linear::ZERO,
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
/// Returns `true` if convergence has been reached. Note that due to rounding
/// errors there is a risk of the solver not detecting convergence when it has
/// reached the optimum.
pub fn iterate<S: GenArray, I: GenArray, C: GenArray>(
    initial_state: Array<S, f64>,
    steps: &mut [RiccatiStep<S, I, C>],
) -> bool {
    factorize(steps);
    backward_recursion(steps);
    forward_recursion(steps, initial_state);
    let changed = update_constraints(steps);
    !changed
}

/// Performs the factorization algorithm.
fn factorize<S: GenArray, I: GenArray, C: GenArray>(steps: &mut [RiccatiStep<S, I, C>]) {
    // To do: Possibly support a mayer term
    let mut next_p = Matrix::ZERO;

    for step in steps.iter_mut().rev() {
        let cost_xx = step.state_cost.transpose() * step.state_cost;
        let cost_xu = step.state_cost.transpose() * step.input_cost;
        let cost_uu = step.input_cost.transpose() * step.input_cost;

        let tmp = step.state_step.transpose() * next_p;
        let f_mat = cost_xx + tmp * step.state_step;
        step.h_mat = cost_xu + tmp * step.input_step;

        let mut g_mat = cost_uu + step.input_step.transpose() * next_p * step.input_step;

        // Only the non-fixed inputs should be included in F and G. Set the rest to zero.
        for (i, bound) in step.active_set.iter().enumerate() {
            if bound.is_some() {
                g_mat.set_row(i, Vector::ZERO);
                g_mat.set_col(i, Vector::ZERO);
                step.h_mat.set_col(i, Vector::ZERO);
            }
        }

        // Note: The implementation of `inv_no_pivot` handles zeroed elements on the
        // diagonal by zeroing their rows during the elimination, which in our case
        // ends up inverting the rest of the matrix.
        step.g_inv = inv_no_pivot(g_mat);

        // step.k_mat = -step.h_mat.transpose();
        // step.g_mat.solve(&mut step.k_mat);
        step.k_mat = -step.g_inv * step.h_mat.transpose();

        step.p_mat = f_mat + step.h_mat * step.k_mat;

        next_p = step.p_mat;
    }
}

/// Performs the backward recursion algorithm.
fn backward_recursion<S: GenArray, I: GenArray, C: GenArray>(steps: &mut [RiccatiStep<S, I, C>]) {
    // To do: Possibly support a mayer term
    let mut next_psi = Vector::ZERO;
    let mut next_p = Matrix::ZERO;

    for step in steps.iter_mut().rev() {
        let const_input = vector(step.active_set.zip(step.input_ranges).map(
            |(bound, (lower, upper))| match bound {
                Some(Bound::Lower) => lower,
                Some(Bound::Upper) => upper,
                None => 0.0,
            },
        ));

        let av = step.input_step * const_input + step.const_step;
        let tmp = next_psi - next_p * av;

        step.k_vec = step.g_inv
            * (step.input_step.transpose() * tmp - step.input_cost.transpose() * step.const_cost);

        step.psi_vec = step.state_step.transpose() * tmp
            - step.h_mat * step.k_vec
            - step.state_cost.transpose() * step.const_cost
            // Where does this term come from? It's not in the paper but appears to be
            // necessary to make the math work out.
            - (step.input_cost * step.k_mat + step.state_cost).transpose()
                * (step.input_cost * const_input);

        next_psi = step.psi_vec;
        next_p = step.p_mat;
    }
}

/// Performs the forward recursion algorithm, including the dual variables.
fn forward_recursion<S: GenArray, I: GenArray, C: GenArray>(
    steps: &mut [RiccatiStep<S, I, C>],
    initial_state: Array<S, f64>,
) {
    let mut x = vector(initial_state);

    for step in steps.iter_mut() {
        let const_input = vector(step.active_set.zip(step.input_ranges).map(
            |(bound, (lower, upper))| match bound {
                Some(Bound::Lower) => lower,
                Some(Bound::Upper) => upper,
                None => 0.0,
            },
        ));

        let u = step.k_mat * x + step.k_vec + const_input;

        step.optimal_state = x.0;
        step.optimal_input = u.0;

        x = step.state_step * x + step.input_step * u + step.const_step;
    }

    let mut next_lambda = Vector::ZERO;
    for step in steps.iter_mut().rev() {
        let x = vector(step.optimal_state);
        let u = vector(step.optimal_input);

        let y = step.state_cost * x + step.input_cost * u + step.const_cost;

        step.dual_variables =
            step.input_cost.transpose() * y + step.input_step.transpose() * next_lambda;
        for (i, bound) in step.active_set.iter().enumerate() {
            if bound.is_none() {
                step.dual_variables[i] = 0.0;
            }
        }

        next_lambda = step.p_mat * x - step.psi_vec;
    }
}

/// Updates the working set by adding or removing a single constraint. Returns
/// `false` if optimality has been reached.
fn update_constraints<S: GenArray, I: GenArray, C: GenArray>(
    steps: &mut [RiccatiStep<S, I, C>],
) -> bool {
    // Remove a constraint binding in the wrong direction
    let mut worst_dual = (0.0, 0, 0);

    for (i, step) in steps.iter().enumerate() {
        for j in 0..I::LEN {
            let dual = match step.active_set.as_slice()[j] {
                Some(Bound::Upper) => step.dual_variables[j],
                Some(Bound::Lower) => -step.dual_variables[j],
                None => continue,
            };

            if dual > worst_dual.0 {
                worst_dual = (dual, i, j);
            }
        }
    }

    if worst_dual.0 > 0.0 {
        steps[worst_dual.1].active_set.as_mut_slice()[worst_dual.2] = None;
        return true;
    }

    // Add a constraint that has been violated
    let mut worst_violation = (0.0, 0, 0, Bound::Lower);

    for (i, step) in steps.iter().enumerate() {
        for j in 0..I::LEN {
            if step.active_set.as_slice()[j].is_none() {
                let value = step.optimal_input.as_slice()[j];
                let (lower, upper) = step.input_ranges.as_slice()[j];

                let (amount, bound) = if value > upper {
                    ((value - upper) / (upper - lower), Bound::Upper)
                } else if value < lower {
                    ((lower - value) / (upper - lower), Bound::Lower)
                } else {
                    continue;
                };

                if amount > worst_violation.0 {
                    worst_violation = (amount, i, j, bound);
                }
            }
        }
    }

    if worst_violation.0 > 0.0 {
        steps[worst_violation.1].active_set.as_mut_slice()[worst_violation.2] =
            Some(worst_violation.3);
        return true;
    }

    // All KKT conditions are satisified, the solution is optimal.
    false
}
