from __future__ import annotations

import numpy as np

from slate.array._array import Array, ArrayBuilder
from slate.array._conversion import as_fundamental_basis, as_index_basis
from slate.basis import Basis


def real[B: Basis, DT: np.dtype[np.number]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Extract the real part of a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.real(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def imag[B: Basis, DT: np.dtype[np.generic]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Extract the imag part of a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.imag(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def angle[B: Basis, DT: np.dtype[np.complexfloating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Get the phase of data in the array."""
    converted = as_index_basis(array)
    # TODO: this is a bug - if the basis can support complex floating it may not support float
    return (
        ArrayBuilder(converted.basis, np.angle(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def abs[B: Basis, DT: np.dtype[np.generic]](  # noqa: A001
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Abs value of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.abs(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def log[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Natural log of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.log(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def log10[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Log base 10 of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.log10(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def log2[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Log base 2 of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.log2(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def exp[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Exponential of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.exp(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def sqrt[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Square root of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.sqrt(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def square[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Square of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.square(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def sin[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Sine of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.sin(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def cos[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Cosine of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.cos(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def tan[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Tangent of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.tan(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def arcsin[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Arcsine of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.arcsin(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def arccos[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Arccosine of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.arccos(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def arctan[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Arctangent of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.arctan(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def sinh[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Hyperbolic sine of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.sinh(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def cosh[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Hyperbolic cosine of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.cosh(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def tanh[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Hyperbolic tangent of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.tanh(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def arcsinh[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Inverse hyperbolic sine of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.arcsinh(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def arccosh[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Inverse hyperbolic cosine of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.arccosh(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def arctanh[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
) -> Array[B, np.dtype[np.floating]]:
    """Inverse hyperbolic tangent of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.arctanh(converted.raw_data))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def mod[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
    val: np.floating | float,
) -> Array[B, np.dtype[np.floating]]:
    """Modulus of data in a slate array."""
    converted = as_index_basis(array)
    return (
        ArrayBuilder(converted.basis, np.mod(converted.raw_data, val))
        .ok()
        .with_basis(array.basis)
        .ok()
    )


def unwrap[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT], discont: float | None = None, axis: int = -1
) -> Array[B, np.dtype[np.floating]]:
    """Unwrap the phase of data in the array."""
    converted = as_fundamental_basis(array)
    unwrapped = np.unwrap(converted.as_array(), discont, axis)
    converted.raw_data = unwrapped
    # Since b has the same dtype and metadata as the original basis
    # it is safe to use it in a conversion.
    # Unfortunately is not possible to express this invariant in the type system.
    return converted.with_basis(array.basis).ok()  # type: ignore safe, since outer_recast must support DT
