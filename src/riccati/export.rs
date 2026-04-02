//! Methods for exporting the QP to matrices for use in a different
//! solver. The QP is
//!
//! ```text
//! minimize    1/2 X^T Q X + c^T X
//! subject to  bl <= A x <= bh
//! ```
//!
//! Where `X = (x_0, u_0, x_1, u_1, ..., x_(n-1), u_(n-1))`. Note that the cost
//! does not include a constant term, so it is not directly comparable to

use core::iter::once;

use crate::riccati::RiccatiStep;
use crate::{Array, ArrayInst, GenArray};

/// An iterator over the elements in a sparse matrix.
pub struct SparseIter<I> {
    /// The width of the matrix
    pub width: usize,
    /// The height of the matrix
    pub height: usize,
    /// The matrix elements. Each item is a tuple `(row, col, value)`, and they
    /// appear in sorted order.
    pub iter: I,
}

/// The quadratic cost term matrix (`Q`), has the structure
///
/// ```text
///  / C_0^T C_0  C_0^T D_0                                          \
/// |  D_0^T C_0  D_0^T D_0                                           |
/// |                       ...                                       |
/// |                           C_(n-1)^T C_(n-1)  C_(n-1)^T D_(n-1)  |
///  \                          D_(n-1)^T C_(n-1)  D_(n-1)^T D_(n-1) /
/// ```
pub fn cost_mat<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &[RiccatiStep<S, I, C, B>],
) -> SparseIter<impl Iterator<Item = (usize, usize, f64)>> {
    let size = steps.len() * (S::LEN + I::LEN + B::LEN);

    let iter = steps.iter().enumerate().flat_map(|(i, step)| {
        (0..S::LEN + I::LEN + B::LEN).flat_map(move |y| {
            (0..S::LEN + I::LEN + B::LEN).map(move |x| {
                let offset = i * (S::LEN + I::LEN + B::LEN);

                let get_col = |i| {
                    if i < S::LEN {
                        step.state_cost.col(i)
                    } else {
                        step.input_cost.col(i - S::LEN)
                    }
                };
                let x_col = get_col(x);
                let y_col = get_col(y);
                (
                    offset + y,
                    offset + x,
                    (x_col.transpose() * y_col).into_scalar(),
                )
            })
        })
    });

    SparseIter {
        width: size,
        height: size,
        iter,
    }
}

/// The linear cost term vector (`c`), has the structure
///
/// ```text
///  /   C_0^T c_0   \
/// |    D_0^T c_0    |
/// |       ...       |
/// |  C_(n-1)^T c_0  |
///  \ D_(n-1)^T c_0 /
/// ```
pub fn cost_vec<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &[RiccatiStep<S, I, C, B>],
) -> impl Iterator<Item = f64> {
    steps.iter().flat_map(|step| {
        (0..S::LEN + I::LEN + B::LEN).map(move |y| {
            let col = if y < S::LEN {
                step.state_cost.col(y)
            } else {
                step.input_cost.col(y - S::LEN)
            };

            (col.transpose() * step.const_cost).into_scalar()
        })
    })
}

/// The constraint matrix (`A`), has the structure
/// ```text
/// /  -I                                                  \
/// |  A_0  B_0  -I                                        |
/// |            A_1  B_1  -I                              |
/// |                         ...                          |
/// |                               -I                     |
/// |                             A_(n-2)  B_(n-2)  -I     |
/// |        I                                             |
/// |                  I                                   |
/// |                         ...                          |
/// |                                         I            |
///  \                                                  I /
/// ```
pub fn constr_mat<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &[RiccatiStep<S, I, C, B>],
) -> SparseIter<impl Iterator<Item = (usize, usize, f64)>> {
    let size = steps.len() * (S::LEN + I::LEN + B::LEN);

    let first_eq_iter = (0..S::LEN).map(|i| (i, i, -1.0));

    let eq_iter = steps[..steps.len() - 1]
        .iter()
        .enumerate()
        .flat_map(|(i, step)| {
            (0..S::LEN).flat_map(move |y| {
                (0..S::LEN + I::LEN + B::LEN)
                    .map(move |x| {
                        let y_offset = (i + 1) * S::LEN;
                        let x_offset = i * (S::LEN + I::LEN + B::LEN);
                        let value = if x < S::LEN {
                            step.state_step[(y, x)]
                        } else {
                            step.input_step[(y, x - S::LEN)]
                        };
                        (y_offset + y, x_offset + x, value)
                    })
                    .chain(once((
                        (i + 1) * S::LEN + y,
                        (i + 1) * (S::LEN + I::LEN + B::LEN) + y,
                        -1.0,
                    )))
            })
        });

    let ineq_iter = (0..steps.len()).flat_map(move |i| {
        (0..I::LEN + B::LEN).map(move |y| {
            let y_offset = steps.len() * S::LEN + i * (I::LEN + B::LEN);
            let x_offset = i * (S::LEN + I::LEN + B::LEN) + S::LEN;
            (y_offset + y, x_offset + y, 1.0)
        })
    });

    SparseIter {
        width: size,
        height: size,
        iter: first_eq_iter.chain(eq_iter).chain(ineq_iter),
    }
}

/// The constraints bounds, has the structure
///
/// ```text
///  /  -x0_0     -x0_0   \
/// |    -a_0      -a_0    |
/// |    ...       ...     |
/// |  -a_(n-2)  -a_(n-2)  |
/// |    ul_0      uh_0    |
/// |    ...       ...     |
/// \  ul_(n-1)  uh_(n-1) /
/// ```
///
/// where the left column is the lower bounds (`bl`) and the right column the
/// upper bounds (`bh`).
pub fn constr_vec<S: GenArray, I: GenArray, C: GenArray, B: GenArray>(
    steps: &[RiccatiStep<S, I, C, B>],
    initial_state: Array<S, f64>,
) -> impl Iterator<Item = (f64, f64)> {
    let first_eq_iter =
        (0..S::LEN).map(move |i| (-initial_state.as_slice()[i], -initial_state.as_slice()[i]));

    let eq_iter = steps[..steps.len() - 1]
        .iter()
        .flat_map(|step| (0..S::LEN).map(|i| (-step.const_step[i], -step.const_step[i])));

    let ineq_iter = steps
        .iter()
        .flat_map(|step| (0..I::LEN + B::LEN).map(|i| step.input_ranges.as_slice()[i]));

    first_eq_iter.chain(eq_iter).chain(ineq_iter)
}
