from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.array.array import SlateArray
    from slate.basis import Basis


def einsum[M: BasisMetadata, DT: np.number[Any]](
    idx: str,
    array1: SlateArray[DT, Basis[M, Any]],
    array2: SlateArray[DT, Basis[M, Any]],
) -> SlateArray[DT, Basis[M, Any]]:
    msg = "Not implemented yet."
    raise NotImplementedError(msg)
