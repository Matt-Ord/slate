from __future__ import annotations

import numpy as np

from slate.array._array import Array
from slate.array._conversion import as_fundamental_basis, as_index_basis
from slate.metadata._metadata import BasisMetadata


def real[M: BasisMetadata, DT: np.generic](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Extract the real part of a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.real(converted.raw_data)).with_basis(array.basis)


def imag[M: BasisMetadata, DT: np.generic](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Extract the imag part of a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.imag(converted.raw_data)).with_basis(array.basis)


def angle[M: BasisMetadata, DT: np.complexfloating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Get the phase of data in the array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.angle(converted.raw_data)).with_basis(array.basis)


def abs[M: BasisMetadata, DT: np.generic](  # noqa: A001
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Abs value of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.abs(converted.raw_data)).with_basis(array.basis)


def log[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Natural log of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.log(converted.raw_data)).with_basis(array.basis)


def log10[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Log base 10 of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.log10(converted.raw_data)).with_basis(array.basis)


def log2[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Log base 2 of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.log2(converted.raw_data)).with_basis(array.basis)


def exp[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Exponential of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.exp(converted.raw_data)).with_basis(array.basis)


def sqrt[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Square root of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.sqrt(converted.raw_data)).with_basis(array.basis)


def square[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Square of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.square(converted.raw_data)).with_basis(array.basis)


def sin[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Sine of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.sin(converted.raw_data)).with_basis(array.basis)


def cos[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Cosine of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.cos(converted.raw_data)).with_basis(array.basis)


def tan[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Tangent of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.tan(converted.raw_data)).with_basis(array.basis)


def arcsin[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Arcsine of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.arcsin(converted.raw_data)).with_basis(array.basis)


def arccos[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Arccosine of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.arccos(converted.raw_data)).with_basis(array.basis)


def arctan[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Arctangent of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.arctan(converted.raw_data)).with_basis(array.basis)


def sinh[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Hyperbolic sine of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.sinh(converted.raw_data)).with_basis(array.basis)


def cosh[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Hyperbolic cosine of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.cosh(converted.raw_data)).with_basis(array.basis)


def tanh[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Hyperbolic tangent of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.tanh(converted.raw_data)).with_basis(array.basis)


def arcsinh[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Inverse hyperbolic sine of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.arcsinh(converted.raw_data)).with_basis(
        array.basis
    )


def arccosh[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Inverse hyperbolic cosine of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.arccosh(converted.raw_data)).with_basis(
        array.basis
    )


def arctanh[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Inverse hyperbolic tangent of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.arctanh(converted.raw_data)).with_basis(
        array.basis
    )


def mod[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT],
    val: np.floating | float,
) -> Array[M, np.floating]:
    """Modulus of data in a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.mod(converted.raw_data, val)).with_basis(
        array.basis
    )


def unwrap[M: BasisMetadata, DT: np.floating](
    array: Array[M, DT], discont: float | None = None, axis: int = -1
) -> Array[M, np.floating]:
    """Unwrap the phase of data in the array."""
    converted = as_fundamental_basis(array)
    unwrapped = np.unwrap(converted.as_array(), discont, axis)
    return Array(converted.basis, unwrapped).with_basis(array.basis)
