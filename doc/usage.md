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

_Note: As the RK4 method calls the function on points in time between time steps, `time == idx * dt` doesn't always hold._

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

The second part is the soft constraints. We will skip them for now, so we implement that function by returning an empty array:

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

![Linear MPC trajectory](figures/figure-1.svg)

## Nonlinear MPC

### Penalty Function

![Penalty function trajectory](figures/figure-2.svg)

### Filter Method

![Filter method trajectory](figures/figure-3.svg)

### Trust Region

![Trust region trajectory](figures/figure-4.svg)

## Advanced Models

### Soft Constraints

### Discrete Models

### Discrete states

### Derivatives of States and Inputs
