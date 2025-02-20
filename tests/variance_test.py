from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from slate.array import Array

if TYPE_CHECKING:
    import numpy as np

    from slate.basis import (
        Basis,
        DiagonalBasis,
        FundamentalBasis,
        TupleBasis,
        TupleBasis2D,
        WrappedBasis,
    )
    from slate.basis._basis import ctype
    from slate.metadata import (
        BasisMetadata,
        LengthMetadata,
        Metadata2D,
        MetadataND,
        SimpleMetadata,
        TupleMetadata,
        VolumeMetadata,
    )


def ctype_variance() -> None:
    a = cast("ctype[np.floating[Any]]", {})
    # If it can handle a float, it might not be able to handle all generic types
    _b: ctype[np.generic] = a  # type: ignore should fail
    # If a basis can handle a float, it can also handle a float128
    _c: ctype[np.float128] = a
    # But it cannot handle a complex128
    _d: ctype[np.complexfloating[Any, Any]] = a  # type: ignore should fail
    # A generic type however should handle a float128
    b = cast("ctype[np.generic]", {})
    _e: ctype[np.float128] = b


def basis_dtype_variance() -> None:
    a = cast("Basis[SimpleMetadata, ctype[np.floating[Any]]]", {})
    # If it can handle a float, it might not be able to handle all generic types
    _b: Basis[BasisMetadata, ctype[np.generic]] = a  # type: ignore should fail
    # If a basis can handle a float, it can also handle a float128
    _c: Basis[BasisMetadata, ctype[np.float128]] = a
    # But it cannot handle a complex128
    _d: Basis[BasisMetadata, ctype[np.complexfloating[Any, Any]]] = a  # type: ignore should fail
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


def tuple_basis_extra_variance() -> None:
    a = cast(
        "TupleBasis[Any,np.number, Any]",
        {},
    )
    _b: TupleBasis[Any, np.generic, Any] = a


def basis_conversion() -> None:
    basis = cast("Basis[SimpleMetadata, ctype[np.floating[Any]]]", {})

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
    a = Array(basis, data)
    _ = a.as_array()
    _ = a.with_basis(basis)

    basis = cast("Basis[SimpleMetadata, ctype[np.number[Any]]]", {})
    data = cast("np.ndarray[Any, np.dtype[np.float64]]", {})
    a = Array(basis, data)
    _ = a.as_array()
    _ = a.with_basis(basis)
    # It should block an incompatible basis
    incompatible_basis = cast("Basis[SimpleMetadata, ctype[np.float128]]", {})
    _ = a.with_basis(incompatible_basis).unwrap()  # type: ignore should fail
    incompatible_basis = cast("Basis[TupleMetadata[Any, Any], ctype[np.generic]]", {})
    _ = a.with_basis(incompatible_basis).unwrap()  # type: ignore should fail
    # but not a compatible one
    compatible_basis = cast("Basis[SimpleMetadata, ctype[np.generic]]", {})
    _ = a.with_basis(compatible_basis).unwrap()

    basis = cast("Basis[SimpleMetadata, ctype[np.complexfloating]]", {})
    data = cast("np.ndarray[Any, np.dtype[np.float64]]", {})
    # We can happily create an array with any basis
    a = Array(basis, data)
    # But we cannot convert the array if incompatible
    _ = a.as_array()  # type: ignore should fail
    _ = a.with_basis(basis).unwrap()  # type: ignore should fail


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
        "WrappedBasis[SimpleMetadata, np.float64, Basis[SimpleMetadata, np.float64]]",
        {},
    )
    _b: WrappedBasis[SimpleMetadata, np.float64, Basis[SimpleMetadata, np.generic]] = a  # type: ignore should fail
    _c: WrappedBasis[SimpleMetadata, np.float64, Basis[SimpleMetadata, np.float64]] = a
    _d: WrappedBasis[SimpleMetadata, np.float64, Basis[BasisMetadata, np.generic]] = a  # type: ignore should fail
    _e: WrappedBasis[SimpleMetadata, np.float64, Basis[BasisMetadata, np.float64]] = a


def stacked_metadata_variance() -> None:
    a = cast("Metadata2D[SimpleMetadata, SimpleMetadata, SimpleMetadata]", {})
    _b: Metadata2D[BasisMetadata, BasisMetadata, BasisMetadata] = a
    _c: Metadata2D[BasisMetadata, BasisMetadata, LengthMetadata] = a  # type: ignore should fail
    _d: Metadata2D[BasisMetadata, LengthMetadata, BasisMetadata] = a  # type: ignore should fail
    _e: Metadata2D[LengthMetadata, BasisMetadata, BasisMetadata] = a  # type: ignore should fail

    a1 = cast("MetadataND[SimpleMetadata, SimpleMetadata, SimpleMetadata]", {})
    _b1: MetadataND[BasisMetadata, BasisMetadata, BasisMetadata] = a1  # type: ignore i disagree with type checker


def tuple_basis_2d_variance() -> None:
    a1 = cast(
        "TupleBasis2D[np.floating[Any], Basis[LengthMetadata, np.float64], Basis[SimpleMetadata, np.float64], None]",
        {},
    )
    _c1: TupleBasis2D[
        np.float64,
        Basis[SimpleMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a1
    _e1: TupleBasis2D[
        np.float64,
        Basis[BasisMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a1
    _e2: TupleBasis2D[
        np.float64,
        Basis[BasisMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a1
    _e5: TupleBasis[
        BasisMetadata,
        None,
        np.float64,
    ] = a1
    _e4: Metadata2D[LengthMetadata, SimpleMetadata, None] = a1.metadata()
    _e3: TupleBasis2D[
        np.float64,
        Basis[BasisMetadata, np.float64],
        Basis[LengthMetadata, np.float64],
        None,
    ] = a1  # type: ignore should fail

    a2 = cast(
        "TupleBasis2D[np.dtype[np.generic], Basis[VolumeMetadata, Any], Basis[VolumeMetadata, Any], None]",
        {},
    )
    _b2: TupleBasis[Any, None, np.complex128, TupleMetadata[Any, None]] = a2


def diagonal_basis_variance() -> None:
    a = cast(
        "DiagonalBasis[np.floating[Any], Basis[SimpleMetadata, np.float64], Basis[SimpleMetadata, np.float64], None]",
        {},
    )
    _b: DiagonalBasis[
        np.generic,
        Basis[SimpleMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a  # type: ignore should fail
    _c: DiagonalBasis[
        np.float64,
        Basis[SimpleMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a
    _d: DiagonalBasis[
        np.float64,
        Basis[TupleMetadata[Any, Any], np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a  # type: ignore should fail
    _e: DiagonalBasis[
        np.float64,
        Basis[BasisMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a
