//! An example simulating a pendulum with a reaction wheel.

use std::f64::consts::{PI, TAU};

use macroquad::prelude::*;
use rmpc::math::{Dual, Linear};
use rmpc::model::{Bounded, Continuous, Discrete};
use rmpc::mpc::{self, MpcStep};
use rmpc::{FieldNames, GenArray};

#[derive(GenArray, FieldNames, Clone, Copy)]
#[repr(C)]
struct State<T> {
    trolley_pos: T,
    trolley_vel: T,
    load_angle: T,
    load_ang_vel: T,
}

#[derive(GenArray, FieldNames, Clone, Copy)]
#[repr(C)]
struct Input<T> {
    target_vel: T,
}

struct Crane {
    length: f64,
    max_vel: f64,
    time_const: f64,
    gravity: f64,

    target_pos: f64,
}

impl Continuous for Crane {
    type State = State<()>;
    type Input = Input<()>;
    type Cost = [(); 3];
    type Bounds = [(); 1];

    fn update<D: Linear>(
        &self,
        _time: f64,
        _time_idx: usize,
        _dt: f64,
        state: State<Dual<D>>,
        input: Input<Dual<D>>,
    ) -> State<Dual<D>> {
        let trolley_vel = state.trolley_vel;
        let load_angle = state.load_angle;
        let load_ang_vel = state.load_ang_vel;
        let target_vel = input.target_vel;

        let trolley_accel = (target_vel - trolley_vel) / self.time_const;
        let load_accel = self.gravity / self.length * load_angle.sin()
            - trolley_accel / self.length * load_angle.cos();

        State {
            trolley_pos: trolley_vel,
            trolley_vel: trolley_accel,

            load_angle: load_ang_vel,
            load_ang_vel: load_accel,
        }
    }

    fn cost_vector<D: Linear>(
        &self,
        _time: f64,
        _time_idx: usize,
        _dt: f64,
        state: State<Dual<D>>,
        input: Input<Dual<D>>,
    ) -> [Dual<D>; 3] {
        let pos = state.trolley_pos + self.length * state.load_angle.sin();

        [
            (pos - self.target_pos) * 100.0,
            self.length * state.load_ang_vel * 10.0,
            input.target_vel * 10.0,
        ]
    }

    fn input_ranges(&self, _time: f64, _time_idx: usize, _dt: f64) -> Input<(f64, f64)> {
        Input {
            target_vel: (-self.max_vel, self.max_vel),
        }
    }

    fn bounds<D: Linear>(
        &self,
        _time: f64,
        _idx: usize,
        _dt: f64,
        state: State<Dual<D>>,
        _input: Input<Dual<D>>,
    ) -> [Bounded<D>; 1] {
        let max_angle = 10_f64.to_radians();
        [Bounded::new(
            state.load_angle,
            -max_angle..max_angle,
            10_000.0,
        )]
    }
}

fn camera(left: f32, right: f32, top: f32, bottom: f32) -> Camera2D {
    let target = vec2((left + right) / 2.0, (top + bottom) / 2.0);

    let aspect = screen_width() / screen_height();
    let zoom_width = vec2(2.0, 2.0 * aspect) / (right - left);
    let zoom_height = vec2(2.0 / aspect, 2.0) / (bottom - top);

    Camera2D {
        rotation: 0.0,
        zoom: zoom_height.min(zoom_width),
        target,
        offset: vec2(0.0, 0.0),
        render_target: None,
        viewport: None,
    }
}

#[macroquad::main("Crane model")]
async fn main() {
    let model = Crane {
        length: 1.0,
        max_vel: 2.0,
        time_const: 0.5,
        gravity: 9.81,
        target_pos: 0.5,
    };

    let mut state = State {
        trolley_pos: 0.0,
        trolley_vel: 0.0,
        load_angle: 0.0,
        load_ang_vel: 0.0,
    };

    let visible_size = 4.0;

    let traj_point = State {
        trolley_pos: 0.0,
        trolley_vel: 0.0,
        load_angle: 0.0,
        load_ang_vel: 0.0,
    };
    let traj_input = Input { target_vel: 0.0 };

    let mut steps = vec![MpcStep::new(); 50];

    let mut mpc_model = model.discretize(0.2);

    for (i, step) in steps.iter_mut().enumerate() {
        step.linearize(&mpc_model, traj_point, traj_input, i);
    }

    loop {
        // Position the camera so everything is visible
        let model = &mpc_model.model;
        let camera = camera(
            -0.5 * visible_size,
            0.5 * visible_size,
            -1.25 * model.length as f32,
            0.25 * model.length as f32,
        );
        set_camera(&camera);

        // Set the target to where the mouse is.
        let (mouse_x, mouse_y) = mouse_position();
        mpc_model.model.target_pos = camera.screen_to_world(vec2(mouse_x, mouse_y)).x as f64;

        // Re-linearize the model as it has changed
        for (i, step) in steps.iter_mut().enumerate() {
            step.linearize(&mpc_model, traj_point, traj_input, i);
        }

        // Iterate a few times to find a good solution. We need an iteration limit in
        // case the solver doesn't detect convergence. This is acceptable as even if the
        // solver hasn't converged completely, a suboptimal input should be much better
        // than not having a solution in time.
        for _ in 0..10 {
            let done = mpc::iterate(state, &mut steps);
            if done {
                break;
            }
        }
        let input = steps[0].optimal_input;

        // Simulate the model using the input.
        let model = &mpc_model.model;
        let dt = get_frame_time().clamp(0.0, 0.1) as f64;
        state = model.discretize(dt).time_step_f64(0, state, input);

        // Make sure the angle is in the interval [-PI, PI].
        state.load_angle = (state.load_angle + PI).rem_euclid(TAU) - PI;

        clear_background(DARKGRAY);

        // Rail
        draw_rectangle(
            -1.0 / camera.zoom.x,
            -0.05,
            2.0 / camera.zoom.x,
            0.1,
            LIGHTGRAY,
        );

        // Wire
        draw_line(
            state.trolley_pos as f32,
            0.0,
            (state.trolley_pos + model.length * state.load_angle.sin()) as f32,
            (model.length * -state.load_angle.cos()) as f32,
            0.07,
            LIGHTGRAY,
        );

        // Trolley
        draw_rectangle(state.trolley_pos as f32 - 0.2, -0.1, 0.4, 0.2, WHITE);

        // Load
        draw_rectangle_ex(
            (state.trolley_pos + model.length * state.load_angle.sin()) as f32,
            (model.length * -state.load_angle.cos()) as f32,
            0.3,
            0.3,
            DrawRectangleParams {
                offset: vec2(0.5, 0.5),
                rotation: state.load_angle as f32,
                color: WHITE,
            },
        );

        next_frame().await
    }
}
