from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, cast

from slate.array import build

if TYPE_CHECKING:
    import numpy as np

    from slate.array import Array
    from slate.array._array import ArrayConversion
    from slate.basis import (
        Basis,
        FundamentalBasis,
        TupleBasis,
        WrappedBasis,
    )
    from slate.basis._basis import ctype
    from slate.metadata import (
        BasisMetadata,
        SimpleMetadata,
        TupleMetadata,
    )


def ctype_variance() -> None:
    a = cast("ctype[np.floating]", {})
    # If it can handle a float, it might not be able to handle all generic types
    _b: ctype[np.generic] = a  # type: ignore should fail
    # If a basis can handle a float, it can also handle a float128
    _c: ctype[np.float128] = a
    # But it cannot handle a complex128
    _d: ctype[np.complexfloating] = a  # type: ignore should fail
    # A generic type however should handle a float128
    b = cast("ctype[np.generic]", {})
    _e: ctype[np.float128] = b

    _f: ctype[Never] = b


def basis_dtype_variance() -> None:
    a = cast("Basis[SimpleMetadata, ctype[np.floating]]", {})
    # If it can handle a float, it might not be able to handle all generic types
    _b: Basis[BasisMetadata, ctype[np.generic]] = a  # type: ignore should fail
    # If a basis can handle a float, it can also handle a float128
    _c: Basis[BasisMetadata, ctype[np.float128]] = a
    # But it cannot handle a complex128
    _d: Basis[BasisMetadata, ctype[np.complexfloating]] = a  # type: ignore should fail
    # A generic type however should handle a float128
    b = cast("Basis[SimpleMetadata, ctype[np.generic]]", {})
    _e: Basis[BasisMetadata, ctype[np.float128]] = b


def tuple_basis_dtype_variance() -> None:
    a = cast(
        "TupleBasis[Any,Any, ctype[np.number[Any,Any]]]",
        {},
    )
    # A number array cannot support all generic types
    _b: TupleBasis[Any, Any, ctype[np.generic]] = a  # type: ignore should fail
    # But it can support a float128 array
    _c: TupleBasis[Any, Any, ctype[np.float128]] = a

    b = cast("TupleBasis[Any,Any, ctype[np.generic]]", {})
    _e: TupleBasis[Any, Any, ctype[np.float128]] = b
    # Check that we can actually upcast the basis
    c = cast(
        "TupleBasis[tuple[Basis[BasisMetadata, ctype[np.float64]], ...],Any, ctype[Never]]",
        {},
    )
    c_upcast = c.upcast()
    _c_upcast_res: TupleBasis[
        tuple[Basis[BasisMetadata, ctype[Never]], ...], Any, ctype[np.float64]
    ] = c_upcast

    d = cast(
        "TupleBasis[tuple[Basis[BasisMetadata, ctype[np.float64]]],Any, ctype[Never]]",
        {},
    )
    d_upcast = d.upcast()
    _d_upcast_res: TupleBasis[
        tuple[Basis[BasisMetadata, ctype[Never]], ...], Any, ctype[np.float64]
    ] = d_upcast


def tuple_basis_extra_variance() -> None:
    a = cast(
        "TupleBasis[Any,np.number, Any]",
        {},
    )
    # Number is a generic
    _b: TupleBasis[Any, np.generic, Any] = a
    # But it is not a float128
    _c: TupleBasis[Any, np.float128, Any] = a  # type: ignore should fail


def tuple_basis_children_variance() -> None:
    a = cast(
        "TupleBasis[tuple[Basis[BasisMetadata, ctype[np.generic]]], Any, Any]",
        {},
    )
    _c: TupleBasis[tuple[Basis[BasisMetadata, ctype[np.float128]]], Any, Any] = a
    # The basis of a is not compatible with the basis of _d
    _d: TupleBasis[tuple[Basis[SimpleMetadata, ctype[np.float128]]], Any, Any] = a  # type: ignore should fail
    # One is valid if we specify any number
    _e: TupleBasis[tuple[Basis[BasisMetadata, ctype[np.generic]], ...], Any, Any] = a
    # But not if we have two
    _f: TupleBasis[
        tuple[
            Basis[BasisMetadata, ctype[np.generic]],
            Basis[BasisMetadata, ctype[np.generic]],
        ],
        Any,
        Any,
    ] = a  # type: ignore should fail


def basis_conversion() -> None:
    basis = cast("Basis[SimpleMetadata, ctype[np.floating]]", {})

    compatible_data = cast("np.ndarray[Any, np.dtype[np.float64]]", {})
    incompatible_data = cast("np.ndarray[Any, np.dtype[np.complexfloating]]", {})

    _converted = basis.__into_fundamental__(compatible_data)
    _converted = basis.__from_fundamental__(compatible_data)
    _converted = basis.__into_fundamental__(incompatible_data).ok()  # type: ignore should fail
    _converted = basis.__from_fundamental__(incompatible_data).ok()  # type: ignore should fail

    compatible_basis = cast("Basis[SimpleMetadata, ctype[np.generic]]", {})
    # Note also these should infer as np.float64
    _converted = basis.__convert_vector_into__(compatible_data, compatible_basis).ok()
    _converted = compatible_basis.__convert_vector_into__(compatible_data, basis).ok()
    incompatible_basis = cast("Basis[SimpleMetadata, ctype[np.complexfloating]]", {})
    _converted = basis.__convert_vector_into__(compatible_data, incompatible_basis).ok()  # type: ignore should fail

    _converted = basis.__convert_vector_into__(incompatible_data, compatible_basis).ok()  # type: ignore should fail
    _converted = compatible_basis.__convert_vector_into__(incompatible_data, basis).ok()  # type: ignore should fail

    wrapped_basis = cast(
        "WrappedBasis[Basis[SimpleMetadata, ctype[np.generic]], ctype[np.float64]]", {}
    )
    _converted = wrapped_basis.__into_fundamental__(compatible_data).ok()
    _converted = wrapped_basis.__from_fundamental__(compatible_data).ok()
    # We should be able to make this basis, but that doesn't guarantee we can convert the data...
    # Note this is not perfect, because the wrapped basis can be assigned to Basis[Any,ctype[np.float64]]
    bad_wrapped_basis = cast(
        "WrappedBasis[Basis[SimpleMetadata, ctype[np.complexfloating]], ctype[np.float64]]",
        {},
    )
    _converted = bad_wrapped_basis.__into_fundamental__(compatible_data).ok()  # type: ignore should fail
    _converted = bad_wrapped_basis.__from_fundamental__(compatible_data).ok()  # type: ignore should fail


def array_basis_variance() -> None:
    basis = cast("Basis[SimpleMetadata, ctype[np.float64]]", {})
    data = cast("np.ndarray[Any, np.dtype[np.float64]]", {})
    a = build(basis, data).ok()
    _as_array_0 = a.as_array()
    _with_basis = a.with_basis(basis).ok()

    basis = cast("Basis[SimpleMetadata, ctype[np.number]]", {})
    data = cast("np.ndarray[Any, np.dtype[np.float64]]", {})
    a = build(basis, data).ok()
    _as_array_1 = a.as_array()
    _with_basis_1 = a.with_basis(basis).ok()
    # It should block an incompatible basis
    incompatible_basis = cast("Basis[SimpleMetadata, ctype[np.float128]]", {})
    _ = a.with_basis(incompatible_basis).ok()  # type: ignore should fail
    # Note here that TupleMetadata is not compatible with SimpleMetadata
    incompatible_basis = cast("Basis[TupleMetadata[Any, Any], ctype[np.generic]]", {})
    _ = a.with_basis(incompatible_basis).ok()  # type: ignore should fail
    # but not a compatible one
    compatible_basis_0 = cast("Basis[SimpleMetadata, ctype[np.generic]]", {})
    _with_basis_2 = a.with_basis(compatible_basis_0).ok()
    compatible_basis_1 = cast("Basis[BasisMetadata, ctype[np.generic]]", {})
    _with_basis_3 = a.with_basis(compatible_basis_1).ok()

    conversion = cast(
        "ArrayConversion[SimpleMetadata, Basis[BasisMetadata, ctype[np.generic]], np.dtype[np.float64]]",
        {},
    )
    conversion.ok()
    _: ArrayConversion[
        BasisMetadata, Basis[BasisMetadata, ctype[np.generic]], np.dtype[np.float64]
    ] = conversion
    conversion = cast(
        "ArrayConversion[BasisMetadata, Basis[SimpleMetadata, ctype[np.generic]], np.dtype[np.float64]]",
        {},
    )
    conversion.ok()

    basis = cast("Basis[SimpleMetadata, ctype[np.complexfloating]]", {})
    data = cast("np.ndarray[Any, np.dtype[np.float64]]", {})
    # We can't create an array with any old basis
    a = build(basis, data).ok()  # type: ignore should fail
    # But we cannot convert the array if incompatible
    _ = a.as_array()  # type: ignore should fail
    _ = a.with_basis(basis).ok()  # type: ignore should fail


def array_dtype_variance() -> None:
    # Check that the Array dtype field has the correct variance
    a = cast(
        "Array[Basis, np.dtype[np.number[Any,Any]]]",
        {},
    )
    # A number array is a generic array
    _b: Array[Basis, np.dtype[np.generic]] = a
    # But it is not necessarily a float128 array
    _c: Array[Basis, np.dtype[np.float128]] = a  # type: ignore should fail
    _d: Array[Basis, np.dtype[np.float64]] = a  # type: ignore should fail
    _e: Array[Basis, np.dtype[np.complex128]] = a  # type: ignore should fail

    b = cast(
        "Array[Basis, np.dtype[np.float64]]",
        {},
    )
    # But a float64 array is not a float128 array
    _f: Array[Basis, np.dtype[np.float128]] = b  # type: ignore should fail
    # But it is a floating array
    _g: Array[Basis, np.dtype[np.floating]] = b


def fundamental_basis_variance() -> None:
    a = cast("FundamentalBasis[SimpleMetadata]", {})
    _b: Basis[BasisMetadata, ctype[np.generic]] = a
    _c: Basis[BasisMetadata, ctype[np.float128]] = a


def wrappeed_basis_variance() -> None:
    a = cast(
        "WrappedBasis[Basis[SimpleMetadata, ctype[np.float64]], ctype[np.float64] ]",
        {},
    )
    # Fails due to not supporting generic
    _b: WrappedBasis[Basis[SimpleMetadata, ctype[np.float64]], ctype[np.generic]] = a  # type: ignore should fail
    _c: WrappedBasis[Basis[SimpleMetadata, ctype[np.float64]], ctype[np.float64]] = a
    # Fails due to not supporting generic
    _d: WrappedBasis[Basis[BasisMetadata, ctype[np.float64]], ctype[np.generic]] = a  # type: ignore should fail
    _e: WrappedBasis[Basis[BasisMetadata, ctype[np.float64]], ctype[np.float64]] = a
    # Fails due to bad basis type
    _f: WrappedBasis[
        Basis[TupleMetadata[Any, Any], ctype[np.float64]], ctype[np.generic]
    ] = a  # type: ignore should fail
    _g: WrappedBasis[
        Basis[TupleMetadata[Any, Any], ctype[np.float64]], ctype[np.float64]
    ] = a  # type: ignore should fail

    b = cast(
        "WrappedBasis[Basis[SimpleMetadata, ctype[np.float64]], ctype[Never] ]",
        {},
    )
    # Upcasting the basis should infer the correct type
    b_upcast = b.upcast()
    _h: WrappedBasis[Basis[SimpleMetadata, ctype[np.float64]], ctype[np.float64]] = (
        b_upcast
    )
