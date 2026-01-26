use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{DeriveInput, Error, Fields, GenericParam, Member, Result};

fn get_fields(input: &DeriveInput) -> Result<&Fields> {
    match &input.data {
        syn::Data::Struct(data) => Ok(&data.fields),
        syn::Data::Enum(data) => Err(Error::new_spanned(
            data.enum_token,
            "`FieldNames` does not support enums",
        )),
        syn::Data::Union(data) => Err(Error::new_spanned(
            data.union_token,
            "`FieldNames` does not support unions",
        )),
    }
}

pub fn handle_field_names(input: DeriveInput) -> Result<TokenStream> {
    let fields = get_fields(&input)?;

    let field_names = fields.members().map(|field| match field {
        Member::Named(ident) => ident.to_string(),
        Member::Unnamed(index) => index.index.to_string(),
    });

    let name = &input.ident;
    let gen_defs = &input.generics;
    let gen_args = input.generics.params.iter().map(|param| match param {
        GenericParam::Lifetime(param) => param.lifetime.clone().into_token_stream(),
        GenericParam::Type(param) => param.ident.clone().into_token_stream(),
        GenericParam::Const(param) => param.ident.clone().into_token_stream(),
    });
    let wher = &input.generics.where_clause;

    Ok(quote! {
        impl #gen_defs ::rmpc::FieldNames for #name <#(#gen_args),*> #wher {
            const FIELD_NAMES: &'static [&'static str] = &[#(#field_names),*];
        }
    })
}
