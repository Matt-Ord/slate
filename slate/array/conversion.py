from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.metadata._metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.array.array import SlateArray
    from slate.basis import Basis


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
