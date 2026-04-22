//! Implementation of the algorithms for computing and updating the Riccati
//! factorization and for finding the trajectory.

use crate::array::{Concat, from_fn};
use crate::math::{Float, Linear, Matrix, Vector, inv_no_pivot, vector};
use crate::mpc::{Bound, Change, MpcStep, Settings};
use crate::{Array, ArrayInst, GenArray};

/// Performs the factorization algorithm. Assumes that the steps after `last`
/// have already been computed.
pub fn factorize_upto<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &mut [MpcStep<S, I, C, B>],
    last: usize,
) {
    // To do: Possibly support a mayer term
    let mayer_p = Matrix::ZERO;

    let (to_recompute, after) = steps.split_at_mut(last + 1);
    let mut next_p = after.first().map_or(&mayer_p, |step| &step.p_mat);

    for step in to_recompute.iter_mut().rev() {
        let cost_xx = step.state_cost.transpose() * step.state_cost;
        let cost_xu = step.state_cost.transpose() * step.input_cost;
        let cost_uu = step.input_cost.transpose() * step.input_cost;

        let tmp = step.state_step.transpose() * *next_p;
        let f_mat = cost_xx + tmp * step.state_step;
        step.h_mat = cost_xu + tmp * step.input_step;

        let mut g_mat = cost_uu + step.input_step.transpose() * *next_p * step.input_step;

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

        step.p_mat = f_mat + step.h_mat * step.k_mat;

        // Due to rounding errors, we can't be sure that the comptuation above yields a
        // symmetric matrix. These rounding errors seem to grow uncontrollably in some
        // cases, so to avoid that we will enforce symmetry.
        step.p_mat = (step.p_mat + step.p_mat.transpose()) * Float::from(0.5);

        next_p = &step.p_mat;
    }
}

/// Uses rank-1 updates to efficiently update the factorization when a single
/// constraint is added or removed.
pub fn update_factorization<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &mut [MpcStep<S, I, C, B>],
    change: Change,
) {
    let t = change.time;
    let i = change.input;

    let mayer_p = Matrix::ZERO;
    let next_p = steps.get(t + 1).map_or(mayer_p, |step| step.p_mat);

    let step = &mut steps[t];

    let adding = step.active_set.as_slice()[i].is_some();
    let alpha = Float::from(if adding { 1.0 } else { -1.0 });
    let pi = Vector(from_fn(|j| Float::from(if i == j { 1.0 } else { 0.0 })));

    let b = step.input_step.col(i);
    let d = step.input_cost.col(i);

    let g0 = d.transpose() * d + b.transpose() * next_p * b;
    let h = step.state_cost.transpose() * d + step.state_step.transpose() * next_p * b;

    let mut g = step.input_cost.transpose() * d + step.input_step.transpose() * next_p * b;
    for (i, bound) in step.active_set.iter().enumerate() {
        if bound.is_some() {
            g[i] = Float::ZERO;
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

    let mut v_vec = numerator * (1.0 / denominator.sqrt());
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

    step.p_mat += alpha * v_vec * v_vec.transpose();

    // Propagate the update to earlier time steps
    for step in steps[..t].iter_mut().rev() {
        let a = step.state_step.transpose() * v_vec;
        let mut b = step.input_step.transpose() * v_vec;

        for (i, bound) in step.active_set.iter().enumerate() {
            if bound.is_some() {
                b[i] = Float::ZERO;
            }
        }

        let tmp = step.g_inv * b;
        step.g_inv -= tmp / (alpha + (b.transpose() * tmp).into_scalar()) * tmp.transpose();

        step.h_mat += alpha * a * b.transpose();

        let tmp = a + step.k_mat.transpose() * b;
        v_vec = (1.0 - alpha * (b.transpose() * step.g_inv * b).into_scalar()).sqrt() * tmp;

        step.k_mat -= alpha * (step.g_inv * b) * tmp.transpose();
        step.p_mat += alpha * v_vec * v_vec.transpose();
    }
}

/// Performs the backward recursion algorithm.
pub fn backward_recursion_upto<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &mut [MpcStep<S, I, C, B>],
    last: usize,
) {
    // To do: Possibly support a mayer term
    let mayer_p = Matrix::ZERO;
    let mayer_psi = Vector::ZERO;

    let (to_recompute, after) = steps.split_at_mut(last + 1);

    let mut next_p = after.first().map_or(&mayer_p, |step| &step.p_mat);
    let mut next_psi = after.first().map_or(&mayer_psi, |step| &step.psi_vec);

    for step in to_recompute.iter_mut().rev() {
        let const_input = vector(step.active_set.zip(step.input_ranges).map(
            |(bound, (lower, upper))| match bound {
                Some(Bound::Lower) => lower,
                Some(Bound::Upper) => upper,
                None => Float::ZERO,
            },
        ));

        let av = step.input_step * const_input + step.const_step;
        let cv = step.input_cost * const_input + step.const_cost;
        let tmp = *next_psi - *next_p * av;

        step.k_vec = step.g_inv
            * (step.input_step.transpose() * tmp - step.input_cost.transpose() * step.const_cost);

        step.psi_vec = step.state_step.transpose() * tmp
            - step.state_cost.transpose() * cv
            - step.h_mat * step.k_vec;

        next_p = &step.p_mat;
        next_psi = &step.psi_vec;
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
    let mut x = vector(initial_state.map(Float::from));
    let tolerance = Concat(settings.input_tol, settings.bound_tol);

    let mut worst_violation = (Float::ZERO, 0, 0, Bound::Lower);
    let mut worst_dual = (Float::ZERO, 0, 0);

    // To do: Possibly support a mayer term
    let mayer_p = Matrix::ZERO;
    let mayer_psi = Vector::ZERO;

    for i in 0..steps.len() {
        let next_p = steps.get(i + 1).map_or(mayer_p, |step| step.p_mat);
        let next_psi = steps.get(i + 1).map_or(mayer_psi, |step| step.psi_vec);
        let step = &mut steps[i];

        let const_input = vector(step.active_set.zip(step.input_ranges).map(
            |(bound, (lower, upper))| match bound {
                Some(Bound::Lower) => lower,
                Some(Bound::Upper) => upper,
                None => Float::ZERO,
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

        step.optimal_state = x.into_array().map(f64::from);
        step.optimal_input = u
            .into_array()
            .zip(step.input_ranges)
            .0
            .map(|(val, (lower, upper))| val.clamp(lower, upper).as_f64());

        let y = step.state_cost * x + step.input_cost * u + step.const_cost;
        x = step.state_step * x + step.input_step * u + step.const_step;

        // Check for constraints binding in the wrong direction
        let lambda = next_p * x - next_psi;
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
