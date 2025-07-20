//! A simulation of a mass attached to a spring.

use std::io::Result;

use mpc::math::{Dual, Linear};
use mpc::model::{Continuous, Model};
use mpc::utility::{Plot, PlotType};
use mpc::{FieldNames, GenArray};

#[derive(FieldNames, GenArray, Clone, Copy)]
#[repr(C)]
struct SpringState<T> {
    velocity: T,
    position: T,
}

#[derive(FieldNames, GenArray, Clone, Copy)]
#[repr(C)]
struct SpringInput<T> {
    force: T,
}

struct Spring {
    mass: f64,
    spring_const: f64,
}

impl Continuous for Spring {
    type State = SpringState<()>;
    type Input = SpringInput<()>;
    type Cost = [(); 0];

    fn state_deriv<D: Linear>(
        &self,
        state: SpringState<Dual<D>>,
        input: SpringInput<Dual<D>>,
    ) -> SpringState<Dual<D>> {
        SpringState {
            velocity: input.force - state.position * self.spring_const / self.mass,
            position: state.velocity,
        }
    }

    fn cost_vector<D: Linear>(
        &self,
        _state: SpringState<Dual<D>>,
        _input: SpringInput<Dual<D>>,
    ) -> [Dual<D>; 0] {
        []
    }
}

fn main() -> Result<()> {
    let dt = 0.1;
    let mode = Spring {
        spring_const: 0.5,
        mass: 0.1,
    }
    .discretize(dt);

    let mut states = Vec::new();
    let mut inputs = Vec::new();

    let mut state = SpringState {
        velocity: 0.0,
        position: 0.0,
    };
    states.push(state);

    for i in 0..100 {
        let input = SpringInput {
            force: (-dt * i as f64).exp(),
        };
        state = mode.time_step_f64(state, input);
        inputs.push(input);
        states.push(state);
    }

    Plot::with_dt(dt)
        .structs(PlotType::Lines, states)
        .structs(PlotType::Stairs, inputs)
        .plot_png(format!(
            "{}/examples/output/{}.png",
            env!("CARGO_MANIFEST_DIR"),
            env!("CARGO_BIN_NAME")
        ))
}
