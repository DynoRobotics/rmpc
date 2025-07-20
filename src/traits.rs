pub use mpc_derive::FieldNames;

/// A trait for getting the field names of structs.
///
/// ## Derivable
///
/// This trait can be used with `#[derive]` on any kind of struct.
///
/// ```
/// use mpc::FieldNames;
///
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
