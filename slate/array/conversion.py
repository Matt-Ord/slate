from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slate.array.array import SlateArray

if TYPE_CHECKING:
    from slate.basis import Basis


def convert_array[B0: Basis[Any, Any], B1: Basis[Any, Any]](
    array: SlateArray[B0],
    basis: B1,
) -> SlateArray[B1]:
    """Convert the array to the given basis.

    Returns
    -------
    SlateArray[M, DT]
    """
    return SlateArray(basis, array.basis.__convert_vector_into__(array.raw_data, basis))
