from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis import Basis

    from ._array import SlateArray


def convert_array[
    M: BasisMetadata,
    DT: np.generic,
    B1: Basis[Any, Any],
](
    array: SlateArray[M, DT],
    basis: B1,
) -> SlateArray[M, DT, B1]:
    """Convert the array to the given basis."""
    return array.with_basis(basis)
