from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from slate.array.array import SlateArray
    from slate.basis import Basis


def convert_array[DT: np.generic, B0: Basis[Any, Any], B1: Basis[Any, Any]](
    array: SlateArray[DT, B0],
    basis: B1,
) -> SlateArray[DT, B1]:
    """Convert the array to the given basis."""
    return array.with_basis(basis)
