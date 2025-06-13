from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Never, TypeGuard, cast, overload

import numpy as np

from slate_core import basis
from slate_core.array._array import Array
from slate_core.array._conversion import (
    as_fundamental_basis,
    as_index_basis,
    as_supports_type_basis,
)
from slate_core.basis import Basis, Ctype
from slate_core.metadata._metadata import SIMPLE_FEATURE

if TYPE_CHECKING:
    from slate_core.array._array import ArrayWithMetadata
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
    return Array(converted.basis, np.real(converted.raw_data))


def imag[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.number]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Extract the imag part of a slate array."""
    converted = as_supports_type_basis(as_index_basis(array), np.floating)
    return Array(converted.basis, np.imag(converted.raw_data))


def angle[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.complexfloating]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Get the phase of data in the array."""
    converted = as_supports_type_basis(as_index_basis(array), np.floating)
    return Array(converted.basis, np.angle(converted.raw_data))


def abs[M: BasisMetadata](  # noqa: A001
    array: ArrayWithMetadata[M, np.dtype[np.number]],
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Abs value of data in a slate array."""
    converted = as_supports_type_basis(as_index_basis(array), np.floating)
    return Array(converted.basis, np.abs(converted.raw_data))


def log[M: BasisMetadata, DT: np.dtype[np.floating]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Natural log of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.log(converted.raw_data))  # type: ignore[return-value]


def log10[M: BasisMetadata, DT: np.dtype[np.floating]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Log base 10 of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.log10(converted.raw_data))  # type: ignore[return-value]


def log2[M: BasisMetadata, DT: np.dtype[np.floating]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Log base 2 of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.log2(converted.raw_data))  # type: ignore[return-value]


def exp[M: BasisMetadata, DT: np.dtype[np.complexfloating]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Exponential of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.exp(converted.raw_data))  # type: ignore[return-value]


def sqrt[M: BasisMetadata, DT: np.dtype[np.floating]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Square root of data in a slate array."""
    converted = as_index_basis(array)
    if array.dtype is np.complexfloating:
        return Array(converted.basis, np.lib.scimath.sqrt(converted.raw_data))  # type: ignore[return-value]
    return Array(converted.basis, np.sqrt(converted.raw_data))  # type: ignore[return-value]


def square[M: BasisMetadata, DT: np.dtype[np.floating]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Square of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.square(converted.raw_data))  # type: ignore[return-value]


def sin[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the sine of all elements in the array.

    Computes the trigonometric sine of each element in the array, element-wise.
    The result has the same shape and basis as the input array.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 126
        :lines: 126-138
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.sin(converted.raw_data))  # type: ignore[return-value]


def cos[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the cosine of all elements in the array.

    Computes the trigonometric cosine of each element in the array, element-wise.
    The result has the same shape and basis as the input array.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 140
        :lines: 140-153
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.cos(converted.raw_data))  # type: ignore[return-value]


def tan[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the tangent of all elements in the array.

    Computes the trigonometric tangent of each element in the array, element-wise.
    The result has the same shape and basis as the input array.

    Note: Be cautious with values near π/2 + nπ where the tangent function has singularities.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 276
        :lines: 276-295
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.tan(converted.raw_data))  # type: ignore[return-value]


def arcsin[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the inverse sine of all elements in the array.

    Computes the element-wise inverse sine of the array. The domain of this function
    is [-1, 1], and the range is [-π/2, π/2].
    The result has the same shape and basis as the input array.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 155
        :lines: 155-168
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.arcsin(converted.raw_data))  # type: ignore[return-value]


def arccos[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the inverse cosine of all elements in the array.

    Computes the element-wise inverse cosine of the array. The domain of this function
    is [-1, 1], and the range is [0, π].
    The result has the same shape and basis as the input array.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 170
        :lines: 170-183
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.arccos(converted.raw_data))  # type: ignore[return-value]


def arctan[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the inverse tangent of all elements in the array.

    Computes the element-wise inverse tangent of the array. The domain is all real numbers,
    and the range is [-π/2, π/2].
    The result has the same shape and basis as the input array.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 185
        :lines: 185-199
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.arctan(converted.raw_data))  # type: ignore[return-value]


def sinh[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the hyperbolic sine of all elements in the array.

    Computes the element-wise hyperbolic sine of the array.
    The domain is all real numbers, and the result has the same shape and basis
    as the input array.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 201
        :lines: 201-213
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.sinh(converted.raw_data))  # type: ignore[return-value]


def cosh[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the hyperbolic cosine of all elements in the array.

    Computes the element-wise hyperbolic cosine of the array.
    The domain is all real numbers, and the range is [1, ∞).
    The result has the same shape and basis as the input array.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 215
        :lines: 215-230
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.cosh(converted.raw_data))  # type: ignore[return-value]


def tanh[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the hyperbolic tangent of all elements in the array.

    Computes the element-wise hyperbolic tangent of the array.
    The domain is all real numbers, and the range is (-1, 1).
    The result has the same shape and basis as the input array.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 297
        :lines: 297-316
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.tanh(converted.raw_data))  # type: ignore[return-value]


def arcsinh[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the inverse hyperbolic sine of all elements in the array.

    Computes the element-wise inverse hyperbolic sine of the array.
    The domain is all real numbers, and the result has the same shape and basis
    as the input array.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 232
        :lines: 232-244
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.arcsinh(converted.raw_data))  # type: ignore[return-value]


def arccosh[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the inverse hyperbolic cosine of all elements in the array.

    Computes the element-wise inverse hyperbolic cosine of the array.
    The domain of this function is [1, ∞), and the result has the same shape and basis
    as the input array.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 246
        :lines: 246-259
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.arccosh(converted.raw_data))  # type: ignore[return-value]


def arctanh[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
) -> ArrayWithMetadata[M, DT]:
    """Calculate the inverse hyperbolic tangent of all elements in the array.

    Computes the element-wise inverse hyperbolic tangent of the array.
    The domain of this function is (-1, 1), and the result has the same shape and basis
    as the input array.

    Examples
    --------
    .. literalinclude:: ../../tests/array/misc_test.py
        :language: python
        :lineno-start: 261
        :lines: 261-274
        :dedent: 4
    """
    converted = as_index_basis(array)
    return Array(converted.basis, np.arctanh(converted.raw_data))  # type: ignore[return-value]


def mod[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[M, DT],
    val: np.number | float,
) -> ArrayWithMetadata[M, DT]:
    """Modulus of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.mod(converted.raw_data, val))  # type: ignore[return-value]


def unwrap[M: BasisMetadata](
    array: ArrayWithMetadata[M, np.dtype[np.floating]],
    discont: float | None = None,
    axis: int = -1,
) -> ArrayWithMetadata[M, np.dtype[np.floating]]:
    """Unwrap the phase of data in the array."""
    converted = as_fundamental_basis(array)
    unwrapped = np.unwrap(converted.as_array(), discont, axis)
    converted.raw_data = unwrapped
    return Array(converted.basis, converted.raw_data)


def cast_as_dual[B: Basis, DT: np.dtype[np.generic]](
    array: Array[B, DT],
) -> Array[B, DT]:
    """Cast a slate array as an array in dual space."""
    basis_dual = array.basis.dual_basis()
    assert basis.supports_dtype(basis_dual, array.dtype)
    return Array(basis_dual, array.raw_data)


def dual_basis[B: Basis, DT: np.dtype[np.generic]](
    array: Array[B, DT],
) -> Array[B, DT]:
    """Cast a slate array as an array in dual space."""
    basis_dual = array.basis.dual_basis()
    assert basis.supports_dtype(basis_dual, array.dtype)
    return array.with_basis(basis_dual)


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
        lhs_basis = as_diagonal.inner.children[0]
        rhs_basis = as_diagonal.inner.children[1]
        # Note if lhs_basis is not 'simple' there is no need
        # to scale here since we would then have to immediately
        # scale it back
        new_components = (
            array.raw_data
            if SIMPLE_FEATURE in rhs_basis.metadata().features
            else cast(
                "Any",
                array.raw_data  # type: ignore[return-value, arg-type]
                * rhs_basis.metadata().basis_weights[rhs_basis.points],
            )
        )

        return Array(lhs_basis, new_components)
    out_basis = basis.from_metadata(array.basis.metadata().children[0])
    data = cast("np.ndarray[Any, DT]", np.diag(array.as_array()))
    return Array.from_array(data, metadata=out_basis.metadata())


def max_arg[M: BasisMetadata, DT: np.dtype[np.generic]](
    array: ArrayWithMetadata[M, DT],
) -> int:
    """Get the index of the maximum value in a slate array."""
    converted = as_fundamental_basis(array)
    return np.argmax(converted.raw_data).item()
