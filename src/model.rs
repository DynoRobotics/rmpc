use crate::AsVector;
use crate::math::{Dual, Linear, Zero};

/// An explicit non-linear discrete time model on the form
/// `x[k+1] = time_step(x[k], u[k])`.
///
/// Most systems are continuous, so instead of implementing this directly it is
/// recommended to implement [`Continuous`] and use [`discretize`] to turn it
/// into a discrete time model.
///
/// [`discretize`]: Continuous::discretize
pub trait Model<const NX: usize, const NU: usize> {
    type State<T>: AsVector<NX, Item = T>;
    type Input<T>: AsVector<NU, Item = T>;

    /// Performs a single time step. Uses dual numbers to make it possible to
    /// linearize the model.
    fn time_step<D: Linear>(
        &self,
        state: Self::State<Dual<D>>,
        input: Self::Input<Dual<D>>,
    ) -> Self::State<Dual<D>>;

    /// A convenience method to perform the time step without tracking any gradient.
    fn time_step_f64(&self, state: Self::State<f64>, input: Self::Input<f64>) -> Self::State<f64> {
        let state = state.map(Dual::from);
        let input = input.map(Dual::from);
        let next_state = self.time_step::<Zero>(state, input);
        next_state.map(Dual::value)
    }
}

/// An explicit non-linear continuous time model on the form
/// `dx(t)/dt = state_deriv(x(t), u(t))`. Can be turned into a continuous time
/// model using [`discretize`].
///
/// [`discretize`]: Continuous::discretize
pub trait Continuous<const NX: usize, const NU: usize> {
    type State<T>: AsVector<NX, Item = T>;
    type Input<T>: AsVector<NU, Item = T>;

    /// Gets the time derivative of the state given some input. Uses dual numbers to
    /// make it possible to linearize the model.
    fn state_deriv<D: Linear>(
        &self,
        state: Self::State<Dual<D>>,
        input: Self::Input<Dual<D>>,
    ) -> Self::State<Dual<D>>;

    /// A convenience method to get the derivative without tracking any gradient.
    fn state_deriv_f64(
        &self,
        state: Self::State<f64>,
        input: Self::Input<f64>,
    ) -> Self::State<f64> {
        let state = state.map(Dual::from);
        let input = input.map(Dual::from);
        let next_state = self.state_deriv::<Zero>(state, input);
        next_state.map(Dual::value)
    }

    /// Turns `self` into a discrete time model.
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

impl<M, const NX: usize, const NU: usize> Model<NX, NU> for RungeKutta4<M>
where
    M: Continuous<NX, NU>,
{
    type State<T> = <M as Continuous<NX, NU>>::State<T>;
    type Input<T> = <M as Continuous<NX, NU>>::Input<T>;

    fn time_step<D: Linear>(
        &self,
        state: Self::State<Dual<D>>,
        input: Self::Input<Dual<D>>,
    ) -> Self::State<Dual<D>> {
        let state = state.into_vector();
        let input = input.into_vector();

        let state_dot = |state: [Dual<D>; NX]| {
            self.model
                .state_deriv(AsVector::from_vector(state), AsVector::from_vector(input))
                .into_vector()
        };

        let perturb = |offsets: &[(f64, [Dual<D>; NX])]| -> [Dual<D>; NX] {
            let mut state = state;
            for &(scale, delta) in offsets {
                for i in 0..NX {
                    state[i] += scale * self.delta_time * delta[i];
                }
            }
            state
        };

        let k_1 = state_dot(state);
        let k_2 = state_dot(perturb(&[(0.5, k_1)]));
        let k_3 = state_dot(perturb(&[(0.5, k_2)]));
        let k_4 = state_dot(perturb(&[(1.0, k_3)]));

        AsVector::from_vector(perturb(&[
            (1.0 / 6.0, k_1),
            (2.0 / 6.0, k_2),
            (2.0 / 6.0, k_3),
            (1.0 / 6.0, k_4),
        ]))
    }
}
