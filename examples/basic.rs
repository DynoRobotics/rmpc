//! Code used in the usage guide (`../doc/usage.md`), including a bit of
//! boilerplate to output the results as CSV files.

#![allow(missing_docs)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use rmpc::math::{Dual, Linear};
use rmpc::model::{Bounded, Continuous, Discrete, RungeKutta4};
use rmpc::mpc::{MpcStep, MpcStepForCont, Settings};
use rmpc::{GenArray, mpc};

#[derive(GenArray, Clone, Copy, Debug)]
#[repr(C)]
struct State<T> {
    pos_x: T,
    pos_y: T,
    angle: T,
    vel: T,
}

#[derive(GenArray, Clone, Copy, Debug)]
#[repr(C)]
struct Input<T> {
    accel: T,
    steering: T,
}

#[derive(Clone, Copy, Debug)]
struct Target {
    x: f64,
    y: f64,
}

#[derive(Clone)]
struct Model {
    targets: Vec<Target>,
}

impl Continuous for Model {
    type State = State<()>;
    type Input = Input<()>;
    type Cost = [(); 4];
    type Bounds = [(); 0];

    fn update<D: Linear>(
        &self,
        _time: f64,
        _idx: usize,
        _dt: f64,
        state: State<Dual<D>>,
        input: Input<Dual<D>>,
    ) -> State<Dual<D>> {
        State {
            pos_x: state.vel * state.angle.cos(),
            pos_y: state.vel * state.angle.sin(),
            angle: state.vel * input.steering,
            vel: input.accel,
        }
    }

    fn cost_vector<D: Linear>(
        &self,
        _time: f64,
        idx: usize,
        _dt: f64,
        state: State<Dual<D>>,
        input: Input<Dual<D>>,
    ) -> [Dual<D>; 4] {
        let target = self.targets[idx];

        [
            1.0 * (target.x - state.pos_x),
            1.0 * (target.y - state.pos_y),
            1.0 * input.accel,
            0.2 * input.steering,
        ]
    }

    fn input_ranges(&self, _time: f64, _idx: usize, _dt: f64) -> Input<(f64, f64)> {
        Input {
            accel: (-2.0, 0.5),
            steering: (-1.0, 1.0),
        }
    }

    fn bounds<D: Linear>(
        &self,
        _time: f64,
        _idx: usize,
        _dt: f64,
        _state: State<Dual<D>>,
        _input: Input<Dual<D>>,
    ) -> [Bounded<D>; 0] {
        []
    }
}

fn main() -> std::io::Result<()> {
    let dt = 0.2;
    let length = 30;

    let mut targets = Vec::new();
    for i in 0..length {
        let t = i as f64 * dt;
        targets.push(Target {
            x: t * 0.3,
            y: t * 0.4,
        });
    }

    let mut steps: Vec<MpcStepForCont<Model>> = Vec::new();
    for &target in &targets {
        let lin_state = State {
            pos_x: target.x,
            pos_y: target.y,
            angle: f64::atan2(0.4, 0.3),
            vel: f64::hypot(0.4, 0.3),
        };
        let lin_input = Input::repeat(0.0);

        steps.push(MpcStep::new(lin_state, lin_input))
    }

    let model = Model { targets }.discretize(dt);

    let initial_state = State {
        pos_x: 0.1,
        pos_y: -0.3,
        angle: 0.7,
        vel: 0.1,
    };

    linear_mpc(initial_state, model.clone(), steps.clone())?;
    println!();

    nonlinear_mpc(
        initial_state,
        model.clone(),
        steps.clone(),
        SqpMethod::Penalty,
    )?;
    println!();

    nonlinear_mpc(
        initial_state,
        model.clone(),
        steps.clone(),
        SqpMethod::Filter,
    )?;
    println!();

    nonlinear_mpc(
        initial_state,
        model.clone(),
        steps.clone(),
        SqpMethod::Trust,
    )?;
    println!();

    Ok(())
}

fn linear_mpc(
    initial_state: State<f64>,
    model: RungeKutta4<Model>,
    mut steps: Vec<MpcStepForCont<Model>>,
) -> std::io::Result<()> {
    mpc::linearize(&model, &mut steps);

    let settings = Settings::default();
    let max_iterations = 100;

    let (iterations, finished) = mpc::iterate(initial_state, &mut steps, max_iterations, &settings);
    println!("{iterations} iterations, finished: {finished}");

    save_mpc_data("mpc-linear.csv", initial_state, &model, &steps)?;

    Ok(())
}

enum SqpMethod {
    Penalty,
    Filter,
    Trust,
}

fn nonlinear_mpc(
    initial_state: State<f64>,
    model: RungeKutta4<Model>,
    mut steps: Vec<MpcStepForCont<Model>>,
    method: SqpMethod,
) -> std::io::Result<()> {
    let settings = Settings::default();
    let num_sqp_iter = 5;
    let max_qp_iter = 100;

    let mut filter_history = Vec::new();

    for sqp_iter in 0..num_sqp_iter {
        mpc::linearize(&model, &mut steps);

        let (iterations, finished) =
            mpc::iterate(initial_state, &mut steps, max_qp_iter, &settings);

        println!("SQP iteration {sqp_iter}, {iterations} QP iterations, finished: {finished}");

        // To do: mention that the match is weird to stop LLMS from copying it literally everywhere
        match method {
            SqpMethod::Penalty => {
                mpc::sqp::penalty_step(&model, &mut steps, initial_state);
            }
            SqpMethod::Filter => {
                let point =
                    mpc::sqp::filter_step(&model, &mut steps, initial_state, &filter_history);
                filter_history.push(point);
            }
            SqpMethod::Trust => {
                // We don't need to limit the position or acceleration as they are only used in
                // linear expressions in the model.
                let state_trust = State {
                    pos_x: f64::INFINITY,
                    pos_y: f64::INFINITY,
                    angle: 0.5,
                    vel: 0.5,
                };
                let input_trust = Input {
                    accel: f64::INFINITY,
                    steering: 0.5,
                };
                mpc::sqp::trust_step(&mut steps, state_trust, input_trust);
            }
        }
    }

    let name = match method {
        SqpMethod::Penalty => "mpc-penalty.csv",
        SqpMethod::Filter => "mpc-filter.csv",
        SqpMethod::Trust => "mpc-trust.csv",
    };

    save_mpc_data(name, initial_state, &model, &steps)?;

    Ok(())
}

fn save_mpc_data(
    filename: &str,
    initial_state: State<f64>,
    model: &RungeKutta4<Model>,
    steps: &[MpcStepForCont<Model>],
) -> std::io::Result<()> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("doc/figures")
        .join(filename);
    let mut file = BufWriter::new(File::create(path)?);

    writeln!(file, "target_x,target_y,opt_x,opt_y,sim_x,sim_y")?;

    let mut state = initial_state;
    for (target, step) in std::iter::zip(&model.model.targets, steps) {
        let optimum = step.optimal_state;
        writeln!(
            file,
            "{},{},{},{},{},{}",
            target.x, target.y, optimum.pos_x, optimum.pos_y, state.pos_x, state.pos_y
        )?;

        state = model.time_step_f64(0, state, step.optimal_input);
    }

    Ok(())
}
