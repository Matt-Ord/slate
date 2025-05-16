from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Never, TypeGuard, cast, overload

import numpy as np

from slate_core import basis
from slate_core.array._array import build
from slate_core.array._conversion import (
    as_fundamental_basis,
    as_index_basis,
    as_supports_type_basis,
    cast_basis,
)
from slate_core.basis import Basis, Ctype

if TYPE_CHECKING:
    from slate_core.array._array import Array, ArrayWithMetadata
    from slate_core.basis import TupleBasisLike
    from slate_core.metadata import BasisMetadata
    from slate_core.metadata._tuple import TupleMetadata


@overload
def is_tuple_basis_like[CT: Ctype[Never], DT: np.dtype[np.generic]](
    array: Array[Basis[BasisMetadata, CT], DT], *, n_dim: Literal[1]
) -> TypeGuard[Array[TupleBasisLike[tuple[BasisMetadata], Never, CT], DT]]: ...
@overload
def is_tuple_basis_like[CT: Ctype[Never], DT: np.dtype[np.generic]](
    array: Array[Basis[BasisMetadata, CT], DT], *, n_dim: Literal[2]
) -> TypeGuard[
    Array[TupleBasisLike[tuple[BasisMetadata, BasisMetadata], Never, CT], DT]
]: ...
@overload
def is_tuple_basis_like[CT: Ctype[Never], DT: np.dtype[np.generic]](
    array: Array[Basis[BasisMetadata, CT], DT], *, n_dim: Literal[3]
) -> TypeGuard[
    Array[
        TupleBasisLike[tuple[BasisMetadata, BasisMetadata, BasisMetadata], Never, CT],
        DT,
    ]
]: ...
@overload
def is_tuple_basis_like[CT: Ctype[Never], DT: np.dtype[np.generic]](
    array: Array[Basis[BasisMetadata, CT], DT], *, n_dim: int | None = None
) -> TypeGuard[Array[TupleBasisLike[tuple[BasisMetadata, ...], Never, CT], DT]]: ...


def is_tuple_basis_like[CT: Ctype[Never], DT: np.dtype[np.generic]](
    array: Array[Basis[BasisMetadata, CT], DT], *, n_dim: int | None = None
) -> TypeGuard[Array[TupleBasisLike[tuple[BasisMetadata, ...], Never, CT], DT]]:
    return basis.is_tuple_basis_like(array.basis, n_dim=n_dim)


def real[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.number]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Extract the real part of a slate array."""
    converted = as_supports_type_basis(as_index_basis(array), np.floating)
    return build(converted.basis, np.real(converted.raw_data)).ok()


def imag[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.generic]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Extract the imag part of a slate array."""
    converted = as_supports_type_basis(as_index_basis(array), np.floating)
    return build(converted.basis, np.imag(converted.raw_data)).ok()


def angle[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.complexfloating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Get the phase of data in the array."""
    converted = as_supports_type_basis(as_index_basis(array), np.floating)
    return build(converted.basis, np.angle(converted.raw_data)).ok()


def abs[M: BasisMetadata](  # noqa: A001
    array: ArrayWithMetadata[M, np.dtype[np.generic]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Abs value of data in a slate array."""
    converted = as_supports_type_basis(as_index_basis(array), np.floating)
    return build(converted.basis, np.abs(converted.raw_data)).ok()


def log[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Natural log of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.log(converted.raw_data)).ok()


def log10[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Log base 10 of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.log10(converted.raw_data)).ok()


def log2[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Log base 2 of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.log2(converted.raw_data)).ok()


def exp[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Exponential of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.exp(converted.raw_data)).ok()


def sqrt[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Square root of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.sqrt(converted.raw_data)).ok()


def square[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Square of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.square(converted.raw_data)).ok()


def sin[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Sine of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.sin(converted.raw_data)).ok()


def cos[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Cosine of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.cos(converted.raw_data)).ok()


def tan[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Tangent of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.tan(converted.raw_data)).ok()


def arcsin[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Arcsine of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.arcsin(converted.raw_data)).ok()


def arccos[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Arccosine of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.arccos(converted.raw_data)).ok()


def arctan[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Arctangent of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.arctan(converted.raw_data)).ok()


def sinh[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Hyperbolic sine of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.sinh(converted.raw_data)).ok()


def cosh[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Hyperbolic cosine of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.cosh(converted.raw_data)).ok()


def tanh[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Hyperbolic tangent of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.tanh(converted.raw_data)).ok()


def arcsinh[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Inverse hyperbolic sine of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.arcsinh(converted.raw_data)).ok()


def arccosh[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Inverse hyperbolic cosine of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.arccosh(converted.raw_data)).ok()


def arctanh[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Inverse hyperbolic tangent of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.arctanh(converted.raw_data)).ok()


def mod[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
    val: np.floating | float,
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Modulus of data in a slate array."""
    converted = as_index_basis(array)
    return build(converted.basis, np.mod(converted.raw_data, val)).ok()


def unwrap[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
    discont: float | None = None,
    axis: int = -1,
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Unwrap the phase of data in the array."""
    converted = as_fundamental_basis(array)
    unwrapped = np.unwrap(converted.as_array(), discont, axis)
    converted.raw_data = unwrapped
    return build(converted.basis, converted.raw_data).ok()


def cast_as_dual[B: Basis, DT: np.dtype[np.generic]](
    array: Array[B, DT],
) -> Array[B, DT]:
    """Cast a slate array as an array in dual space."""
    basis_dual = array.basis.dual_basis()
    assert basis.supports_dtype(basis_dual, array.dtype)
    return build(basis_dual, array.raw_data).ok()


def dual_basis[B: Basis, DT: np.dtype[np.generic]](
    array: Array[B, DT],
) -> Array[B, DT]:
    """Cast a slate array as an array in dual space."""
    basis_dual = array.basis.dual_basis()
    assert basis.supports_dtype(basis_dual, array.dtype)
    return array.with_basis(basis_dual).ok()


def supports_type[M: BasisMetadata, T: np.generic, DT: np.dtype[np.generic]](
    array: ArrayWithMetadata[M, DT], ty: type[T]
) -> TypeGuard[Array[Basis[M, Ctype[T]], DT]]:
    """Check if the basis supports the given data type.

    This is a type guard, so it will narrow the type of the basis to
    `Array[Basis[BasisMetadata, Ctype[T]], DT]` if it returns `True`.
    """
    return array.basis.ctype.supports_type(ty)


def supports_dtype[M: BasisMetadata, T: np.generic, DT: np.dtype[np.generic]](
    array: ArrayWithMetadata[M, DT], dtype: np.dtype[T]
) -> TypeGuard[Array[Basis[M, Ctype[T]], DT]]:
    """Check if the basis supports the given data type.

    This is a type guard, so it will narrow the type of the basis to
    `Array[Basis[BasisMetadata, Ctype[T]], DT]` if it returns `True`.
    """
    return array.basis.ctype.supports_dtype(dtype)


def extract_diagonal[M: BasisMetadata, DT: np.dtype[np.generic]](
    array: ArrayWithMetadata[TupleMetadata[tuple[M, M], Any], DT],
) -> ArrayWithMetadata[M, DT]:
    """Extract the diagonal of a 2D slate array.

    Note that this traets the data as a raw 2d array of data, it does not return the eigenstates
    of the array, even if it is already diagonalised.
    """
    as_diagonal = basis.as_diagonal(basis.as_index(array.basis))
    if as_diagonal is not None:
        return cast_basis(
            array.with_basis(as_diagonal).assert_ok(), as_diagonal.inner.children[0]
        ).assert_ok()
    out_basis = basis.from_metadata(array.basis.metadata().children[0])
    data = cast("np.ndarray[Any, DT]", np.diag(array.as_array()))
    return build(out_basis, data).ok()


def max_arg[M: BasisMetadata, DT: np.dtype[np.generic]](
    array: ArrayWithMetadata[M, DT],
) -> int:
    """Get the index of the maximum value in a slate array."""
    converted = as_fundamental_basis(array)
    return np.argmax(converted.raw_data).item()
