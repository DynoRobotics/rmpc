use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, Error, Fields, Ident, Result, Type};

fn get_fields(input: &DeriveInput) -> Result<&Fields> {
    match &input.data {
        syn::Data::Struct(data_struct) => Ok(&data_struct.fields),
        syn::Data::Enum(data_enum) => Err(Error::new_spanned(
            data_enum.enum_token,
            "`GenArray` does not support enums",
        )),
        syn::Data::Union(data_union) => Err(Error::new_spanned(
            data_union.union_token,
            "`GenArray` does not support unions",
        )),
    }
}

fn get_generic(input: &DeriveInput) -> Result<&Ident> {
    let generics = &input.generics;

    if let Some(param) = generics.const_params().next() {
        return Err(Error::new_spanned(
            param,
            "`GenArray` does not support const generics",
        ));
    }
    if let Some(param) = generics.lifetimes().next() {
        return Err(Error::new_spanned(
            param,
            "`GenArray` does not support lifetimes",
        ));
    }
    let mut params = generics.type_params();
    let Some(param) = params.next() else {
        return Err(Error::new_spanned(
            &input.ident,
            "`GenArray` needs a generic",
        ));
    };
    if let Some(param) = params.next() {
        return Err(Error::new_spanned(
            param,
            "`GenArray` only support one generic",
        ));
    }

    if !param.bounds.is_empty() {
        return Err(Error::new_spanned(
            param,
            "`GenArray` does not support generic bounds",
        ));
    }
    if generics.where_clause.is_some() {
        return Err(Error::new_spanned(
            &generics.where_clause,
            "`GenArray` does not support where clauses",
        ));
    }

    Ok(&param.ident)
}

fn check_repr_c(input: &DeriveInput) -> Result<()> {
    let mut repr_c = false;
    for attr in &input.attrs {
        if attr.path().is_ident("repr") {
            if let Ok(repr) = attr.parse_args::<Ident>()
                && repr.to_string().as_str() == "C"
            {
                repr_c = true;
            } else {
                return Err(Error::new_spanned(attr, "`GenArray` requires `#[repr(C)]`"));
            }
        }
    }
    if !repr_c {
        return Err(Error::new_spanned(
            &input.ident,
            "`GenArray` requires `#[repr(C)]`",
        ));
    }

    Ok(())
}

pub fn handle_as_vector(input: DeriveInput) -> Result<TokenStream> {
    let fields = get_fields(&input)?;
    let generic = get_generic(&input)?;

    for field in fields {
        if let Type::Path(path) = &field.ty
            && path.qself.is_none()
            && path.path.is_ident(generic)
        {
            continue;
        }

        return Err(Error::new_spanned(
            &field.ty,
            format!("`GenArray` requires all fields to have type `{generic}`"),
        ));
    }

    check_repr_c(&input)?;

    let name = &input.ident;
    let len = fields.len();

    Ok(quote! {
        impl ::mpc::GenArray for #name<()> {
            type Arr<#generic: Copy> = #name<#generic>;
            const LEN: usize = #len;
        }
        unsafe impl<#generic: Copy> ::mpc::ArrayInst for #name<#generic> {
            type Gen = #name<()>;
            type Item = #generic;
        }
    })
}
