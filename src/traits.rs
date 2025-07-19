pub use mpc_derive::{AsVector, FieldNames};

/// A trait for structs consisting of exactly `N` fields, each of type `T`.
///
/// ## Derivable
///
/// This trait can be used with `#[derive]` on any kind of struct with at least
/// one field where all fields have the same type.
///
/// ```
/// # use mpc::AsVector;
/// #[derive(AsVector, PartialEq, Debug)]
/// struct Point {
///     x: i32,
///     y: i32,
///     z: i32,
/// }
/// assert_eq!(Point::from_vector([3, 1, 4]), Point { x: 3, y: 1, z: 4 });
/// assert_eq!(Point { x: 2, y: 7, z: 1 }.into_vector(), [2, 7, 1]);
/// ```
pub trait AsVector<const N: usize>: Sized {
    /// The type of the fields.
    type Item;

    /// Destructures `self`, returning the field values in the order the fields are
    /// defined.
    fn into_vector(self) -> [Self::Item; N];

    /// Constructs an instance of `Self` from the specified field values, in the
    /// order the fields are defined,
    fn from_vector(vector: [Self::Item; N]) -> Self;

    /// Maps each field using the function.
    ///
    /// Note: This is only intended to be used between the same generic struct, but
    /// enforcing that restriction isn't possible with the type system. The number
    /// of fields however is enforced, which should be enough to prevent accidental
    /// misuse in most cases.
    fn map<V: AsVector<N>>(self, f: impl FnMut(Self::Item) -> V::Item) -> V {
        V::from_vector(self.into_vector().map(f))
    }
}

impl<T, const N: usize> AsVector<N> for [T; N] {
    type Item = T;

    fn into_vector(self) -> [T; N] {
        self
    }

    fn from_vector(vector: [T; N]) -> Self {
        vector
    }
}

/// A trait for getting the field names of structs.
///
/// ## Derivable
///
/// This trait can be used with `#[derive]` on any kind of struct.
///
/// ```
/// # use mpc::FieldNames;
/// #[derive(FieldNames)]
/// struct Complex {
///     real: f32,
///     imaginary: f32,
/// }
/// assert_eq!(Complex::FIELD_NAMES, &["real", "imaginary"]);
/// ```
pub trait FieldNames {
    /// The names of the fields, in the order they are defined.
    const FIELD_NAMES: &[&str];
}
