use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::{DeriveInput, Error, Fields, GenericParam, Ident, Member, Result};

fn get_fields(input: &DeriveInput) -> Result<&Fields> {
    match &input.data {
        syn::Data::Struct(data_struct) => Ok(&data_struct.fields),
        syn::Data::Enum(data_enum) => Err(Error::new_spanned(
            data_enum.enum_token,
            "`AsVector` does not support enums",
        )),
        syn::Data::Union(data_union) => Err(Error::new_spanned(
            data_union.union_token,
            "`AsVector` does not support unions",
        )),
    }
}

pub fn handle_as_vector(input: DeriveInput) -> Result<TokenStream> {
    let fields = get_fields(&input)?;

    let Some(first_field) = fields.iter().next() else {
        return Err(Error::new_spanned(
            input,
            "`AsVector` needs at least one field",
        ));
    };

    let ty = &first_field.ty;
    let length = fields.len();

    let field_names = fields
        .members()
        .map(|member| match member {
            Member::Named(ident) => ident.to_token_stream(),
            Member::Unnamed(index) => index.to_token_stream(),
        })
        .collect::<Vec<_>>();

    let var_names = fields
        .members()
        .map(|field| match field {
            Member::Named(ident) => ident,
            Member::Unnamed(index) => Ident::new(&format!("_{}", index.index), index.span),
        })
        .collect::<Vec<_>>();

    let name = &input.ident;

    let gen_defs = &input.generics;
    let gen_args = input.generics.params.iter().map(|param| match param {
        GenericParam::Lifetime(param) => param.lifetime.clone().into_token_stream(),
        GenericParam::Type(param) => param.ident.clone().into_token_stream(),
        GenericParam::Const(param) => param.ident.clone().into_token_stream(),
    });
    let wher = &input.generics.where_clause;

    Ok(quote! {
        impl #gen_defs ::mpc::AsVector<#ty, #length> for #name <#(#gen_args),*> #wher {
            fn into_vector(self) -> [#ty; #length] {
                [#(self.#field_names),*]
            }
            fn from_vector(vector: [#ty; #length]) -> Self {
                let [#(#var_names),*] = vector;
                Self { #(#field_names: #var_names),* }
            }
        }
    })
}
