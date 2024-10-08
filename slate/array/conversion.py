from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slate.array.array import SlateArray

if TYPE_CHECKING:
    from slate.basis.basis import Basis


def convert_array[_B0: Basis[Any, Any], _B1: Basis[Any, Any]](
    array: SlateArray[_B0],
    basis: _B1,
) -> SlateArray[_B1]:
    """Convert the array to the given basis.

    Returns
    -------
    SlateArray[_M, _DT]
    """
    return SlateArray(basis, array.basis.__convert_vector_into__(array.raw_data, basis))
