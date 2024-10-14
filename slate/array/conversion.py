from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array.array import SlateArray

if TYPE_CHECKING:
    from slate.basis import Basis


def convert_array[DT: np.generic, B0: Basis[Any, Any], B1: Basis[Any, Any]](
    array: SlateArray[DT, B0],
    basis: B1,
) -> SlateArray[DT, B1]:
    """Convert the array to the given basis.

    Returns
    -------
    SlateArray[DT, B1]
    """
    return SlateArray(basis, array.basis.__convert_vector_into__(array.raw_data, basis))
