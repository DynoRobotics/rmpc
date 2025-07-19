mod as_vector;
mod field_names;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// Derive macro for the `AsVector` trait.
#[proc_macro_derive(AsVector)]
pub fn derive_as_vector(input: TokenStream) -> TokenStream {
    as_vector::handle_as_vector(parse_macro_input!(input as DeriveInput))
        .unwrap_or_else(|error| error.into_compile_error())
        .into()
}

/// Derive macro for the `FieldNames` trait.
#[proc_macro_derive(FieldNames)]
pub fn derive_field_names(input: TokenStream) -> TokenStream {
    field_names::handle_field_names(parse_macro_input!(input as DeriveInput))
        .unwrap_or_else(|error| error.into_compile_error())
        .into()
}
