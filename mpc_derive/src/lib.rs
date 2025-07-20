mod field_names;
mod gen_array;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// Derive macro for the `FieldNames` trait.
#[proc_macro_derive(FieldNames)]
pub fn derive_field_names(input: TokenStream) -> TokenStream {
    field_names::handle_field_names(parse_macro_input!(input as DeriveInput))
        .unwrap_or_else(|error| error.into_compile_error())
        .into()
}

/// Derive macro for the `GenArray` and `ArrayInst` traits.
#[proc_macro_derive(GenArray)]
pub fn derive_gen_array(input: TokenStream) -> TokenStream {
    gen_array::handle_as_vector(parse_macro_input!(input as DeriveInput))
        .unwrap_or_else(|error| error.into_compile_error())
        .into()
}
