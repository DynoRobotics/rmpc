//! Traits and helper methods for custom higher kinded array types.
//!
//! The idea is that, while the state and input of a system are vectors in the
//! mathematical sense, using an array of values to represent them gets a bit
//! annoying in practice due to having to remember whick index corresponds to
//! which physical quantity. Due to the currently quite limited support for
//! const generics in the Rust compiler it also isnt't possible to, for example,
//! concatenate two different types of vectors into one larger vector.
//!
//! To solve these problems, this module provides helper traits to treat custom
//! generic types as arrays, for example:
//!
//! ```
//! use mpc::{GenArray, ArrayInst};
//!
//! #[derive(Clone, Copy, GenArray, PartialEq, Debug)]
//! #[repr(C)]
//! struct Pendulum<T> {
//!     angle: T,
//!     velocity: T,
//! }
//!
//! // Get a slice of the fields.
//! let some_state = Pendulum { angle: 0.5, velocity: -1.0 };
//! assert_eq!(some_state.as_ref(), &[0.5, -1.0]);
//!
//! // Map the fields to a different type.
//! let fixed_point_state = some_state.map(|field| (field * 1024.0) as i32);
//! assert_eq!(fixed_point_state, Pendulum { angle: 512, velocity: -1024 });
//! ```
//!
//! Note the `#[repr(C)]` on the struct definition, which is what guarantees
//! that the struct layout is the same as that of an array.
//!
//! ## Traits
//!
//! There are two traits making this possible, [`ArrayInst`] and [`GenArray`],
//! both of which are derived by the same derive macro.
//!
//! Types implementing [`ArrayInst`] have the same representation as `[T; N]`
//! where `T` is the type of the fields and `N` is the number of fields. It
//! provides methods to get a `&[T]` or `&mut [T]` of the fields, making it
//! possible to use it as an array. In the example above, it is implemented for
//! `Pendulum<T: Copy>`.
//!
//! [`GenArray`] represents a generic ("higher-kinded") array and is implemented
//! by a single variant of an array type. In the case of the `Pendulum` struct
//! in the example above, it is implemented for `Pendulum<()>`. This trait is
//! what makes it possible to write code that is generic over different kinds of
//! generic arrays.

#![allow(unsafe_code)]

use std::mem::MaybeUninit;

pub use mpc_derive::GenArray;

/// A generic fixed-size array. See the [module documentation][`self`] for more
/// information.
///
/// ## Derivable
///
/// This trait can be used with `#[derive]`, if the following are true:
/// * The struct has `#[repr(C)]`
/// * The type has exactly one generic, and it doesn't have any bounds.
/// * All fields are of that generic type.
pub trait GenArray {
    /// The actual array type, for some `T`.
    ///
    /// The [`Array`] type alias makes accessing this slightly easier.
    type Arr<T: Copy>: ArrayInst<Gen = Self, Item = T>;

    /// The length of the array.
    const LEN: usize;

    /// Constructs an instance from a function, similar to
    /// [`std::array::from_fn`]. In some cases the equivalend function [`from_fn`]
    /// may provide better type inference.
    fn from_fn<T: Copy>(f: impl FnMut(usize) -> T) -> Self::Arr<T> {
        from_fn(f)
    }
}

/// An instance of a generic fixed-size array. See the
/// [module documentation][`self`] for more information.
///
/// ## Derivable
///
/// This trait is implemented along with [`GenArray`] by its derive macro.
///
/// ## Safety
///
/// When implementing this manually, you must ensure that  `Self` and
/// `[Self::Item; Self::Gen::LEN]` have the same layout and that it is sound to
/// transmute between them.
pub unsafe trait ArrayInst: Copy {
    /// The generic version of this array.
    type Gen: GenArray<Arr<Self::Item> = Self>;

    /// The type of the fields inside `self`.
    type Item: Copy;

    /// Gets a reference to the fields.
    ///
    /// A const version is available as [`as_ref`].
    #[inline(always)]
    fn as_ref(&self) -> &[Self::Item] {
        as_ref(self)
    }

    /// Gets a mutable reference to the fields.
    ///
    /// A const version is available as [`as_mut`].
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [Self::Item] {
        as_mut(self)
    }

    /// Maps the elements, producing an instance with a different field type.
    fn map<T: Copy>(self, mut f: impl FnMut(Self::Item) -> T) -> Array<Self::Gen, T> {
        from_fn(|i| f(self.as_ref()[i]))
    }

    /// Zips `self` with another array that may have a different field type.
    fn zip<T: Copy>(self, other: Array<Self::Gen, T>) -> Array<Self::Gen, (Self::Item, T)> {
        from_fn(|i| (self.as_ref()[i], other.as_ref()[i]))
    }

    /// Concatenates `self` with another array that has the same field type.
    fn concat<A>(self, other: A) -> Concat<Self, A>
    where
        A: ArrayInst<Item = Self::Item>,
    {
        Concat(self, other)
    }
}

/// An instance of a generic array.
pub type Array<A, T> = <A as GenArray>::Arr<T>;

/// Initializes an array with all elements set to the same value.
pub const fn repeat<A: ArrayInst>(item: A::Item) -> A {
    let mut array = MaybeUninit::<A>::uninit();

    let mut i = 0;
    while i < A::Gen::LEN {
        let first = <*mut _>::cast::<A::Item>(&mut array);

        // Safety: The `ArrayInst` implementation guarantees that this points
        // to one of the fields.
        unsafe { first.add(i).write(item) };

        i += 1;
    }

    // Safety: We have initialized all elements.
    unsafe { array.assume_init() }
}

/// Equivalent to [`GenArray::from_fn`] but with a slightly different signature
/// allowing the type inference to deduce the generic array type.
pub fn from_fn<A: ArrayInst>(mut f: impl FnMut(usize) -> A::Item) -> A {
    let mut array = MaybeUninit::<A>::uninit();

    for i in 0..A::Gen::LEN {
        let first = <*mut _>::cast::<A::Item>(&mut array);

        // Safety: The `ArrayInst` implementation guarantees that this points
        // to one of the fields.
        unsafe { first.add(i).write(f(i)) };
    }

    // Safety: We have initialized all elements.
    unsafe { array.assume_init() }
}

/// A `const` version of [`ArrayInst::as_ref`].
#[inline(always)]
pub const fn as_ref<A: ArrayInst>(array: &A) -> &[A::Item] {
    let ptr = (array as *const A).cast::<A::Item>();
    let len = A::Gen::LEN;
    // Safety: Enforced by the implementor.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

/// A `const` version of [`ArrayInst::as_mut`].
#[inline(always)]
pub const fn as_mut<A: ArrayInst>(array: &mut A) -> &mut [A::Item] {
    let ptr = (array as *mut A).cast::<A::Item>();
    let len = A::Gen::LEN;
    // Safety: Enforced by the implementor.
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

impl<const N: usize> GenArray for [(); N] {
    type Arr<T: Copy> = [T; N];
    const LEN: usize = N;
}

// Safety: Obviously meets all requirements.
unsafe impl<T: Copy, const N: usize> ArrayInst for [T; N] {
    type Gen = [(); N];
    type Item = T;
}

/// The concatenation of two arrays. See [`ArrayInst::concat`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Concat<A, B>(pub A, pub B);

impl<A: GenArray, B: GenArray> GenArray for Concat<A, B> {
    type Arr<T: Copy> = Concat<A::Arr<T>, B::Arr<T>>;
    const LEN: usize = A::LEN + B::LEN;
}

// Safety: The layout and transmute validity are guaranteed by `#[repr(C)]` and
// the `ArrayInst` implementations of `A` and `B`.
unsafe impl<A: ArrayInst, B: ArrayInst<Item = A::Item>> ArrayInst for Concat<A, B> {
    type Gen = Concat<A::Gen, B::Gen>;
    type Item = A::Item;
}
