use core::iter::zip;

use crate::array::repeat;
use crate::math::{Dual, Linear};
use crate::model::{Bounded, Discrete};
use crate::{Array, ArrayInst, GenArray};

/// An explicit non-linear continuous time model on the form
/// `dx(t)/dt = state_deriv(x(t), u(t), t)`. Can be turned into a continuous
/// time model using [`discretize`].
///
/// All of the methods are given three arguments for determining the current
/// time:
/// - `time`: The current time, as an `f64`. This can be in the range
///   `(idx * dt)..=((idx+1) * dt)`.
/// - `idx`: The index of the current step. This is useful for sampling lookup
///   tables using zero order hold, for example when comparing the current
///   trajectory to a reference sampled at the same frequency.
/// - `dt`: The size of a time step. This can be useful in models where some of
///   the states are discrete time.
///
/// It is possible to treat some of the states as discrete time instead of
/// continuous time. See [`IS_DISCRETE`] for details.
///
/// [`discretize`]: Continuous::discretize
/// [`IS_DISCRETE`]: Continuous::IS_DISCRETE
pub trait Continuous {
    /// The state of the system at some point in time.
    type State: GenArray;
    /// The derivative of the system at some point in time.
    type Input: GenArray;
    /// A vector whose squared magnitude is the cost density at some point in time.
    type Cost: GenArray;
    /// The values to limit with soft constraints.
    type Bounds: GenArray;

    /// Decides which states should be treated as discrete time. For these states,
    /// the value returned by [`update`][Self::update] is used as the value of the
    /// state at the next time step. Defaults to `false` for all states.
    ///
    /// This can be useful for defining costs on derivatives of states or inputs.
    /// For example:
    ///
    /// ```rust
    /// # use rmpc::math::{Dual, Linear};
    /// # use rmpc::GenArray;
    /// # use rmpc::model::Continuous;
    /// #[derive(Clone, Copy, GenArray)]
    /// #[repr(C)]
    /// struct State<T> {
    ///     pos: T,
    ///     last_vel: T,
    /// }
    /// #[derive(Clone, Copy, GenArray)]
    /// #[repr(C)]
    /// struct Input<T> {
    ///     vel: T,
    /// }
    ///
    /// struct Model;
    ///
    /// impl Continuous for Model {
    ///     type State = State<()>;
    ///     type Input = Input<()>;
    ///     type Cost = [(); 2];
    ///
    ///     const IS_DISCRETE: State<bool> = State {
    ///         pos: false,     // Continuous time
    ///         last_vel: true, // Discrete time
    ///     };
    ///
    ///     fn update<D: Linear>(
    ///         &self, _time: f64, _idx: usize, _dt: f64,
    ///         state: State<Dual<D>>, input: Input<Dual<D>>,
    ///     ) -> State<Dual<D>> {
    ///         State {
    ///             pos: input.vel,      // The derivative of the position
    ///             last_vel: input.vel, // The next value of `last_vel`
    ///         }
    ///     }
    ///
    ///     fn cost_vector<D: Linear>(
    ///         &self, _time: f64, _idx: usize, dt: f64,
    ///         state: State<Dual<D>>, input: Input<Dual<D>>,
    ///     ) -> [Dual<D>; 2] {
    ///         // Approximation of the derivative of the input
    ///         let accel = (input.vel - state.last_vel) / dt;
    ///
    ///         [
    ///             state.pos * 10.0,
    ///             accel * 1.0,
    ///         ]
    ///     }
    ///
    ///     fn input_ranges(&self, _time: f64, _idx: usize, _dt: f64) -> Input<(f64, f64)> {
    ///         Input { vel: (-2.0, 2.0) }
    ///     }
    /// }
    /// ```
    const IS_DISCRETE: Array<Self::State, bool> = repeat(false);

    /// Gets the time derivative of the state given some input.
    ///
    /// For discrete states, the returned value is instead used as the value of the
    /// state at the next time step.
    fn update<D: Linear>(
        &self,
        time: f64,
        idx: usize,
        dt: f64,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::State, Dual<D>>;

    /// The vector whose squared magnitude is the cost density at the current point
    /// in time.
    fn cost_vector<D: Linear>(
        &self,
        time: f64,
        idx: usize,
        dt: f64,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Cost, Dual<D>>;

    /// The valid range of each input. The lower bound must be less than or equal to
    /// the upper bound.
    fn input_ranges(&self, time: f64, idx: usize, dt: f64) -> Array<Self::Input, (f64, f64)>;

    /// The soft constraints to apply.
    fn bounds<D: Linear>(
        &self,
        time: f64,
        idx: usize,
        dt: f64,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Bounds, Bounded<D>>;

    /// Turns `self` into a discrete time model using RK4 with zero order hold.
    fn discretize(self, delta_time: f64) -> RungeKutta4<Self>
    where
        Self: Sized,
    {
        RungeKutta4 {
            model: self,
            delta_time,
        }
    }
}

impl<M: Continuous> Continuous for &M {
    type State = M::State;
    type Input = M::Input;
    type Cost = M::Cost;
    type Bounds = M::Bounds;

    const IS_DISCRETE: Array<Self::State, bool> = M::IS_DISCRETE;

    fn update<D: Linear>(
        &self,
        time: f64,
        idx: usize,
        dt: f64,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::State, Dual<D>> {
        M::update(self, time, idx, dt, state, input)
    }

    fn cost_vector<D: Linear>(
        &self,
        time: f64,
        idx: usize,
        dt: f64,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Cost, Dual<D>> {
        M::cost_vector(self, time, idx, dt, state, input)
    }

    fn input_ranges(&self, time: f64, idx: usize, dt: f64) -> Array<Self::Input, (f64, f64)> {
        M::input_ranges(self, time, idx, dt)
    }

    fn bounds<D: Linear>(
        &self,
        time: f64,
        idx: usize,
        dt: f64,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Bounds, Bounded<D>> {
        M::bounds(self, time, idx, dt, state, input)
    }
}

/// A discrete version of a continuous model. Uses the [RK4] method to
/// approximate the discrete solution assuming a zero order hold input.
///
/// [RK4]: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method
pub struct RungeKutta4<M> {
    /// The continuous time model.
    pub model: M,
    /// The size of each time step.
    pub delta_time: f64,
}

impl<M: Continuous> Discrete for RungeKutta4<M> {
    type State = M::State;
    type Input = M::Input;
    type Cost = M::Cost;
    type Bounds = M::Bounds;

    fn time_step<D: Linear>(
        &self,
        time: usize,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::State, Dual<D>> {
        let perturb = |offsets: &[(f64, Array<Self::State, _>)]| {
            let mut state = state;
            for &(scale, delta) in offsets {
                for ((state, &delta), &discrete) in
                    zip(state.iter_mut(), delta.iter()).zip(M::IS_DISCRETE.iter())
                {
                    // Discrete steps should only be updated at the end of the interval, so they are
                    // skipepd here.
                    if !discrete {
                        *state += scale * self.delta_time * delta;
                    }
                }
            }
            state
        };
        let deriv = |step: f64, state: Array<Self::State, _>| {
            let t = (time as f64 + step) * self.delta_time;
            self.model.update(t, time, self.delta_time, state, input)
        };

        let k_1 = deriv(0.0, state);
        let k_2 = deriv(0.5, perturb(&[(0.5, k_1)]));
        let k_3 = deriv(0.5, perturb(&[(0.5, k_2)]));
        let k_4 = deriv(1.0, perturb(&[(1.0, k_3)]));

        let mut next_state = perturb(&[
            (1.0 / 6.0, k_1),
            (2.0 / 6.0, k_2),
            (2.0 / 6.0, k_3),
            (1.0 / 6.0, k_4),
        ]);

        for ((state, new), &discrete) in
            zip(next_state.iter_mut(), k_1.iter()).zip(M::IS_DISCRETE.iter())
        {
            if discrete {
                *state = *new;
            }
        }

        next_state
    }

    fn cost_vector<D: Linear>(
        &self,
        time: usize,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Cost, Dual<D>> {
        // To do: Would it be beneficial to use RK4 here aswell instead of this Riemann
        // integral style summation?

        let scale = libm::sqrt(self.delta_time);

        self.model
            .cost_vector(
                time as f64 * self.delta_time,
                time,
                self.delta_time,
                state,
                input,
            )
            .map(|value| value * scale)
    }

    fn input_ranges(&self, time: usize) -> Array<Self::Input, (f64, f64)> {
        self.model
            .input_ranges(time as f64 * self.delta_time, time, self.delta_time)
    }

    fn bounds<D: Linear>(
        &self,
        time: usize,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Bounds, Bounded<D>> {
        let scale = libm::sqrt(self.delta_time);

        self.model
            .bounds(
                time as f64 * self.delta_time,
                time,
                self.delta_time,
                state,
                input,
            )
            .map(|mut bound| {
                bound.weight *= scale;
                bound
            })
    }
}
