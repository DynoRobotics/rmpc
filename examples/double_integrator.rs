//! An example using an unstable double integrator.

use rmpc::math::{Dual, Linear};
use rmpc::model::Continuous;
use rmpc::riccati::{self, RiccatiStep};
use rmpc::{ArrayInst, FieldNames, GenArray};

use crate::common::plot::{Plot, PlotType};

mod common;

/// The state of the system.
#[derive(GenArray, FieldNames, Clone, Copy)]
#[repr(C)]
struct State<T> {
    pos: T,
    vel: T,
}

/// The input signals.
#[derive(GenArray, FieldNames, Clone, Copy)]
#[repr(C)]
struct Input<T> {
    acc: T,
}

/// The model of the system. Contains the parameters.
struct DoubleIntegrator {
    target: f64,
}

impl Continuous for DoubleIntegrator {
    type State = State<()>;
    type Input = Input<()>;
    type Cost = [(); 3];

    /// The derivative of the state given the current state and input.
    fn update<D: Linear>(
        &self,
        _time: f64,
        _time_idx: usize,
        _dt: f64,
        state: State<Dual<D>>,
        input: Input<Dual<D>>,
    ) -> State<Dual<D>> {
        State {
            pos: state.vel,
            vel: input.acc,
        }
    }

    /// A vector whose squared norm is used as the cost function. This needs to
    /// include a term for each input signal.
    fn cost_vector<D: Linear>(
        &self,
        _time: f64,
        _time_idx: usize,
        _dt: f64,
        state: State<Dual<D>>,
        input: Input<Dual<D>>,
    ) -> [Dual<D>; 3] {
        [
            (state.pos - self.target) * 10.0_f64.sqrt(),
            state.vel * 1.0_f64.sqrt(),
            input.acc * 0.1_f64.sqrt(),
        ]
    }

    /// Bounds for each input signal.
    fn input_ranges(&self, _time: f64, _time_idx: usize, _dt: f64) -> Input<(f64, f64)> {
        Input { acc: (-1.0, 1.0) }
    }
}

fn main() -> std::io::Result<()> {
    let dt = 1.0;
    let horizon = 30;

    let initial_state = State {
        pos: 25.0,
        vel: 0.0,
    };

    // Turn the continuous time model into a discrete time model.
    let model = DoubleIntegrator { target: 0.0 }.discretize(dt);

    // Linearize the model at each time step. Since the model is linear, it doesn't
    // matter which point is used for the linearization. If the model is nonlinear,
    // it is a good idea to pick a point close to where the optimum is expected to
    // be.
    let mut steps = vec![RiccatiStep::new(); horizon];
    for (i, step) in steps.iter_mut().enumerate() {
        let state = State { pos: 0.0, vel: 0.0 };
        let input = Input { acc: 0.0 };
        step.linearize(&model, state, input, i);
    }

    // Solve the optimization problem to find the optimal solution.
    let mut changed = riccati::step(initial_state, &mut steps);
    let mut iterations = 1;
    while let Some(change) = changed {
        changed = riccati::step_update(initial_state, &mut steps, change);
        iterations += 1;
    }
    println!("Optimum found after {iterations} iterations");

    // Plot the resulting trajectory.
    let mut plot = Plot::with_dt(dt);
    for (i, &name) in State::<()>::FIELD_NAMES.iter().enumerate() {
        let values = steps
            .iter()
            .map(|s| s.optimal_state.as_slice()[i])
            .collect();
        plot = plot.values(PlotType::Lines, name, values);
    }
    for (i, &name) in Input::<()>::FIELD_NAMES.iter().enumerate() {
        let values = steps
            .iter()
            .map(|s| s.optimal_input.as_slice()[i])
            .collect();
        plot = plot.values(PlotType::Stairs, name, values);
    }

    plot.plot_png(format!(
        "{}/examples/{}.png",
        env!("CARGO_MANIFEST_DIR"),
        env!("CARGO_BIN_NAME")
    ))
}
