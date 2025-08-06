use std::iter::zip;

use crate::array::Concat;
use crate::math::{Dual, Linear};
use crate::model::Model;
use crate::{Array, ArrayInst, GenArray};

/// An explicit non-linear continuous time model on the form
/// `dx(t)/dt = state_deriv(x(t), u(t))`. Can be turned into a continuous time
/// model using [`discretize`].
///
/// [`discretize`]: Continuous::discretize
pub trait Continuous {
    /// The state of the system at some point in time.
    type State: GenArray;
    /// The derivative of the system at some point in time.
    type Input: GenArray;
    /// A vector whose squared magnitude is the cost density at some point in time.
    type Cost: GenArray;

    /// Gets the time derivative of the state given some input. Uses dual numbers to
    /// make it possible to linearize the model.
    fn state_deriv<D: Linear>(
        &self,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::State, Dual<D>>;

    /// The vector whose squared magnitude is the cost density at the current point
    /// in time.
    fn cost_vector<D: Linear>(
        &self,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Cost, Dual<D>>;

    /// The valid range of each input. The lower bound must be less than or equal to
    /// the upper bound.
    fn input_ranges(&self) -> Array<Self::Input, (f64, f64)>;

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

/// A modified version of [`Continuous`] where the derivative of the input can
/// be used in the cost vector.
pub trait ContinuousDiff {
    /// The state of the system at some point in time.
    type State: GenArray;
    /// The derivative of the system at some point in time.
    type Input: GenArray;
    /// A vector whose squared magnitude is the cost density at some point in time.
    type Cost: GenArray;

    /// Gets the time derivative of the state given some input. Uses dual numbers to
    /// make it possible to linearize the model.
    fn state_deriv<D: Linear>(
        &self,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::State, Dual<D>>;

    /// The vector whose squared magnitude is the cost density at the current point
    /// in time.
    fn cost_vector<D: Linear>(
        &self,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
        input_deriv: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Cost, Dual<D>>;

    /// The valid range of each input. The lower bound must be less than or equal to
    /// the upper bound.
    fn input_ranges(&self) -> Array<Self::Input, (f64, f64)>;

    /// Turns `self` into a discrete time model using RK4 with zero order hold.
    ///
    /// The discretized model includes the last input in the state.
    fn discretize(self, delta_time: f64) -> RungeKutta4Diff<Self>
    where
        Self: Sized,
    {
        RungeKutta4Diff {
            model: self,
            delta_time,
        }
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

impl<M: Continuous> Model for RungeKutta4<M> {
    type State = M::State;
    type Input = M::Input;
    type Cost = M::Cost;

    fn time_step<D: Linear>(
        &self,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::State, Dual<D>> {
        let perturb = |offsets: &[(f64, Array<Self::State, _>)]| {
            let mut state = state;
            for &(scale, delta) in offsets {
                for (state, &delta) in zip(state.as_mut(), delta.as_ref()) {
                    *state += scale * self.delta_time * delta;
                }
            }
            state
        };

        let k_1 = self.model.state_deriv(state, input);
        let k_2 = self.model.state_deriv(perturb(&[(0.5, k_1)]), input);
        let k_3 = self.model.state_deriv(perturb(&[(0.5, k_2)]), input);
        let k_4 = self.model.state_deriv(perturb(&[(1.0, k_3)]), input);

        perturb(&[
            (1.0 / 6.0, k_1),
            (2.0 / 6.0, k_2),
            (2.0 / 6.0, k_3),
            (1.0 / 6.0, k_4),
        ])
    }

    fn cost_vector<D: Linear>(
        &self,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Cost, Dual<D>> {
        // To do: Would it be beneficial to use RK4 here aswell instead of this Riemann
        // integral style summation?

        self.model
            .cost_vector(state, input)
            .map(|value| value * self.delta_time)
    }

    fn input_ranges(&self) -> Array<Self::Input, (f64, f64)> {
        self.model.input_ranges()
    }
}

/// A version of [`RungeKutta4`] for models implementing [`ContinuousDiff`].
pub struct RungeKutta4Diff<M> {
    /// The continuous time model.
    pub model: M,
    /// The size of each time step.
    pub delta_time: f64,
}

impl<M: ContinuousDiff> Model for RungeKutta4Diff<M> {
    type State = Concat<M::State, M::Input>;
    type Input = M::Input;
    type Cost = M::Cost;

    fn time_step<D: Linear>(
        &self,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::State, Dual<D>> {
        let Concat(state, _) = state;

        let perturb = |offsets: &[(f64, Array<M::State, _>)]| {
            let mut state = state;
            for &(scale, delta) in offsets {
                for (state, &delta) in zip(state.as_mut(), delta.as_ref()) {
                    *state += scale * self.delta_time * delta;
                }
            }
            state
        };

        let k_1 = self.model.state_deriv(state, input);
        let k_2 = self.model.state_deriv(perturb(&[(0.5, k_1)]), input);
        let k_3 = self.model.state_deriv(perturb(&[(0.5, k_2)]), input);
        let k_4 = self.model.state_deriv(perturb(&[(1.0, k_3)]), input);

        let deriv = perturb(&[
            (1.0 / 6.0, k_1),
            (2.0 / 6.0, k_2),
            (2.0 / 6.0, k_3),
            (1.0 / 6.0, k_4),
        ]);
        Concat(deriv, input)
    }

    fn cost_vector<D: Linear>(
        &self,
        state: Array<Self::State, Dual<D>>,
        input: Array<Self::Input, Dual<D>>,
    ) -> Array<Self::Cost, Dual<D>> {
        let Concat(state, last_input) = state;
        let deriv = input
            .zip(last_input)
            .map(|(input, last_input)| (input - last_input) / self.delta_time);

        self.model
            .cost_vector(state, input, deriv)
            .map(|value| value * self.delta_time.sqrt())
    }

    fn input_ranges(&self) -> Array<Self::Input, (f64, f64)> {
        self.model.input_ranges()
    }
}
