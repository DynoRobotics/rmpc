# Basic Usage

As this is a Rust crate, the documentation can be generated with the following command

```bash
cargo doc --open
```

## Model Definition

As an example, we will consider a simple vehicle with Double Ackermann steering. The state of the system is given by the following struct:

```rs
#[derive(GenArray, Clone, Copy, Debug)]
#[repr(C)]
struct State<T> {
    pos_x: T,
    pos_y: T,
    angle: T,
    vel: T,
}
```

It has a field for each quantity that we want to track. Note that this struct is generic. This is because we will use it with many different field types later on.

To let the library handle this generic type properly, it derives the `GenArray` trait, which lets the library instantiate it with different types and treat it as an array. The derive macro requires that all fields have type `T` and that the struct is annotated with `#[repr(C)]`. See the documentation for `rmpc::array` for more information.

Similarly, we define a struct for the "input" used to control the system:

```rs
#[derive(GenArray, Clone, Copy, Debug)]
#[repr(C)]
struct Input<T> {
    accel: T,
    steering: T,
}
```

It also derives `GenArray` for the same reason as for the state.

With this we can define the actual model of the system. The parameters to the model are stored in a struct, which we will call `Model`, that implements the `Continuous` trait. This trait is used for al continuous-time models.

```rs
#[derive(Clone)]
struct Model {
    /* ... */
}

impl Continuous for Model {
    /* ... */
}
```

There are a few things we need to provide in this implementation. The first is the type of involved quantities. These are:

- `State`: The state of the system, set to the `State<()>` struct we defined earlier.
- `Input`: The input of the system, set to the `Input<()>` struct we defined earlier.
- `Cost`: The quantity we want to minimize. In our case, we will need four elements, so we set this to an array of length 4: `[(); 4]`
- `Bounds`: Used for soft constraints. We will omit these for now, so we set them to an empty array: `[(); 0]`.

```rs
impl Continuous for Model {
    type State = State<()>;
    type Input = Input<()>;
    type Cost = [(); 4];
    type Bounds = [(); 0];

    /* ... */
}
```

These types can be anything that implements the `GenArray` trait, which is either a struct that derives it or a built in array. The unit type `()` is used as a placeholder for the generic.

Now, for the dynamics of the system. As this is a continuous time model, we describe the dynamics as a differential equation where the derivative of the state is a function of the state and input. For this vehicle example, it looks like

```rs
impl Continuous for Model {
    /* ... */

    fn update<D: Linear>(
        &self,
        _time: f64, _idx: usize, _dt: f64,
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

    /* ... */
}
```

The type `Dual<D>` is used for the solver to be able to track derivatives using a variant of [dual numbers](https://en.wikipedia.org/wiki/Dual_number).

For time-dependent models, the `time` argument can be used to get the current point in time as a float. `idx` and `dt` are the index of the step and the delta time, respectively. The dynamics of this model are not time-dependent, so we ignore them with `_`.

_Note: As the RK4 method calls the method on points in time between time steps, `time == idx * dt` doesn't always hold._

For this model, we will use the distance to a reference trajectory as the cost function, so we will add a field `targets` to `Model` that stores this

```rs
#[derive(Clone)]
struct Model {
    targets: Vec<Target>,
}

#[derive(Clone, Copy, Debug)]
struct Target {
    x: f64,
    y: f64,
}
```

The cost is done using a nonlinear least squares function, defined as the squared magnitude of `cost_vector`. For this example we will use

```rs
impl Continuous for Model {
    /* ... */

    fn cost_vector<D: Linear>(
        &self,
        _time: f64, idx: usize, _dt: f64,
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

    /* ... */
}
```

which means that the actual cost function becomes:

```rs
(target.x - state.pos_x)**2
+ (target.y - state.pos_y)**2
+ input.accel**2
+ (0.2 * input.steering)**2
```

_**Warning:** The solver assumes that the derivatives of the cost vector with respect to the different inputs are always linearly independent (or equivalently, the Jacobian with respect to the inputs is a tall matrix with full rank). If this assumption is violated then the solver is likely to produce nonsensical results. Having an component with a scaled version for each input is a sufficient condition._

Finally, we have the constraints for the system. They are given by two parts. The first is the hard constraints for the input signals, defined as an interval for each input:

```rs
impl Continuous for Model {
    /* ... */

    fn input_ranges(&self, _time: f64, _idx: usize, _dt: f64) -> Input<(f64, f64)> {
        Input {
            accel: (-2.0, 0.5),
            steering: (-1.0, 1.0),
        }
    }

    /* ... */
}
```

The second part is the soft constraints. We will skip them for now, so we implement that method by returning an empty array:

```rs
impl Continuous for Model {
    /* ... */

    fn bounds<D: Linear>(
        &self,
        _time: f64, _idx: usize, _dt: f64,
        _state: State<Dual<D>>,
        _input: Input<Dual<D>>,
    ) -> [Bounded<D>; 0] {
        []
    }
}
```

With that, we have defined the entire model. See [usage-guide.rs](../examples/usage-guide.rs) for the complete definition.

### Simulation

To simulate the model, we first need to discretize it:

```rs
let model = Model {
    // we only use the model for simulation, so we don't need to set the targets now.
    targets: Vec::new(),
};
let model = model.discretize();
```

This turns it into `RungeKutta4<Model>`, which implements the `Discrete` trait. You can then simulate the model by calling `time_step_f64`:

```rs
let mut state: State<f64> = /* ... */;

for idx in 0..10 {
    let input: Input<f64> = /* ... */;
    state = model.time_step_f64(idx, state, input);
}
```

The `idx` argument can be set to anything if the model is not time-dependent.

## Linear MPC

The simplest kind of MPC is where we linearize the model around some trajectory. If the model is already linear or if the optimum is close to the trajectory used for linearization, then this typically yields very good results while being cheapter to compute than the nonlinear methods.

To start, let's create a straight path to use as the target.


```rs
// Sample time
let dt = 0.2;
// Number of samples
let length = 30;

let mut targets = Vec::new();
for i in 0..length {
    let t = i as f64 * dt;
    targets.push(Target {
        x: t * 0.3,
        y: t * 0.4,
    });
}
```

_Note: In practice, you should make sure to sample the path the same speed that you want the robot to follow it, accounting for the time it takes to accelerate._

The state of the solver is stored in the `MpcStep` type. It has a lot of generics, but can be instantiated easily using the `MpcStepForCont` type alias. Each of these steps stores a state and input that is used to linearize around. One option, which we will use here, is to use the path to determine them. We could also for example have linearized every step around the current state.

_Note: We don't actually need the type annotation in this particular case, as the compiler will figure out the type based on the usage later. But it's good to know if you need to store this state somewhere._

```rs
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
```

Now, we define the model and use the `linearize` function from the `mpc` module to linearize it at each time step. (The linearized version is stored inside the `MpcStep`.)

```rs
mpc::linearize(&model, &mut steps);
```

The linear solver can then be invoked by calling `mpc::iterate`, which performs a limited number of QP solver iterations. The `Settings` struct can be used to define what tolerances to use. Here we will use the default settings.

```rs
let initial_state = State {
    pos_x: 0.1,
    pos_y: -0.3,
    angle: 0.7,
    vel: 0.1,
};

let settings = Settings::default();
let max_iterations = 100;

let (iterations, finished) = mpc::iterate(initial_state, &mut steps, max_iterations, &settings);
println!("{iterations} iterations, finished: {finished}");
```

This prints

```
7 iterations, finished: true
```

meaning that the solver only needed 7 iterations. If it doesn't finish within the iteration limit, then it will stop at a suboptimal solution. The trajectory found by the solver is stored in the fields `optimal_state` and `optimal_input` for each `MpcStep`.

Note that, as this is uses a linear approximation of the model, the solution won't be exact. The picture below shows the target (gray), trajectory found by the solver (blue), and the path we get if we were to simulate the model with the sequence of inputs it found. As you can see, the simulation drifts away from the optimum.

![Linear MPC trajectory](figures/figure-1.svg)

This is much less of a problem when running the controller in a closed loop, as the feedback will help it correct for this error as long as the solution is close enough.

When running in a closed loop, it is advantageous to reuse the same `MpcStep`s every time the solver is called, as `mpc::iterate` will use the last solution as the initial guess. This is typically very close when running in a closed loop, which will make the solver converge very quickly under normal circumstances.

## Nonlinear MPC

To improve the accuracy when working with nonlinear models, RMPC uses sequential quadratic programming. This works by taking the solution from the linear MPC problem and moving the linearization trajectory closer to it. After solving the problem again, this will result in a better solution. If the initial linearization trajectory is close enough then this should converge to the actual optimum.

Because nonlinear models can be unpredictable, there is a risk of divergence if the steps are too large. RMPC has a few different methods for determining how large the steps can be, which are shown in the sections below.

### Penalty Function

This method works by finding a step size that tries to find a balance between making the solution feasible and making it optimal. This method is very stable, but can be overly cautios for some models due to the Maratos effect. It is used as follows:

```rust
// repeat some number of times
for _ in 0..5 {
    // Linearize around the current trajectory
    mpc::linearize(&model, &mut steps);

    // Solve the linear MPC problem
    let (iterations, _) = mpc::iterate(initial_state, &mut steps, max_qp_iter, &settings);
    println!("SQP iteration {sqp_iter}, {iterations} QP iterations");

    // Move the linearization trajectory closer
    mpc::sqp::penalty_step(&model, &mut steps, initial_state);
}
```

For this particular model, this method only performs slightly better than the linear version because it's too cautious to converge.

![Penalty function trajectory](figures/figure-2.svg)

### Filter Method

This method is similar to the penalty function method, but instead of trying to find a specific balance of the objective and feasibility it instead continues as long as it can find a new Pareto optimum. This makes it more aggressive than the penalty function. It is used as follows:

```rust
let mut filter_history = Vec::new();

// repeat some number of times
for _ in 0..5 {
    // Linearize around the current trajectory
    mpc::linearize(&model, &mut steps);

    // Solve the linear MPC problem
    let (iterations, _) = mpc::iterate(initial_state, &mut steps, max_qp_iter, &settings);
    println!("SQP iteration {sqp_iter}, {iterations} QP iterations");

    // Move the linearization trajectory closer, and make sure to add the new point to the history.
    let point = mpc::sqp::filter_step(&model, &mut steps, initial_state, &filter_history);
    filter_history.push(point);
}
```

This method manages to converge for this model, as seen in the plot below where the yellow and blue paths are directly on top of each other.

![Filter method trajectory](figures/figure-3.svg)

### Trust Region

This method, unlike the other two, tries to take a full step every time, with a limit to how large the step is. It is used as follows:

```rust
let state_trust = State {
    pos_x: f64::INFINITY,
    pos_y: f64::INFINITY,
    angle: 0.5,
    vel: 0.5,
};
let input_trust = Input {
    accel: f64::INFINITY,
    steering: 1.0,
};

// repeat some number of times
for _ in 0..5 {
    // Linearize around the current trajectory
    mpc::linearize(&model, &mut steps);

    // Solve the linear MPC problem
    let (iterations, _) = mpc::iterate(initial_state, &mut steps, max_qp_iter, &settings);
    println!("SQP iteration {sqp_iter}, {iterations} QP iterations");

    // Move the linearization trajectory closer
    mpc::sqp::trust_step(&mut steps, state_trust, input_trust);
}
```

`state_trust` and `input_trust` are the limits for how big the step can be. In the code above, this limit is set so that the `angle` and `vel` states can change by at most `0.5`, and the `steering` input by at most `1.0`. The rest of the components are set to `f64::INFINITY` to make them unrestricted. Making them unrestricted is fine in this case as all of the expressions in the model that use them are linear, so they don't affect the linearization anyways.

For some models, it could be fine to use `f64::INFINITY` for all fields, but for others you might need to tune these limits to get a good result. In general, smaller limits will make the solver more stable and larger limits will make it faster.

This method converges to the right solution for this model, as shown below.

![Trust region trajectory](figures/figure-4.svg)

## Advanced Models

The sections below list some more things that can be done in the model definitions.

### Soft Constraints

Soft constraints are constraints where the solution is allowed to violate them, but at some cost. Unlike the hard input constraints, the soft constraints can be used on arbitrary functions of the states and inputs.

As a simple example, let's say we want to limit the velocity of the vehicle from the example before. First, we change the `Bounds` type to an array with a single item, as we only want to add a single constraint.

```rs
impl Continuous for Model {
    type Bounds = [(); 1];

    /* ... */
}
```

Then, we change the `bounds` method to add a constraint for `state.velocity` with a range from `0.0 <= state.velocity <= 2.0` and a weight of `10.0`.

```rs
impl Continuous for Model {
    /* ... */

    fn bounds<D: Linear>(
        &self,
        _time: f64, _idx: usize, _dt: f64,
        state: State<Dual<D>>,
        _input: Input<Dual<D>>,
    ) -> [Bounded<D>; 1] {
        [
            Bounded::new(state.velocity, 0.0..2.0, 10.0),
        ]
    }
}
```

The weight is used to determine how costly a violation is. This bound, for example, will result in the following extra cost:

```rust
(10 * max(0.0, state.velocity - 2.0))**2   // cost when above 2.0
+ (10 * max(0.0, 0.0 - state.velocity))**2 // cost when below 0.0
```

Note that we are not limited to a single state. We could have used an arbitrary expression instead of `state.velocity`. The bounds and weight however is not allowed to depend on the state or input, and the types of the variables enforces this.

### Discrete Models and Derivatives

In some cases it is easier to describe the system using a discrete-time model. In those cases it can be more convenient to implement the `Discrete` trait instead. It is similar to `Continuous` except that the `update` method returns the next state instead of the state derivative, and all of the methods only take a single `time: usize` argument instead of the three different time arguments used by `Continuous`. It also doesn't need to be discretized as it is already discrete.

It is also possible to have a mix of continuous and non-continuous states in a model. See the documentation for `Continuous::IS_DISCRETE` for details on how to use it, including an example for how to use it to compute a derivative.
