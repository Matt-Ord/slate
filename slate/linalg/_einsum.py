from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from slate.array import SlateArray
    from slate.metadata import BasisMetadata


def einsum[DT: np.number[Any]](
    idx: str,
    array1: SlateArray[BasisMetadata, DT],
    array2: SlateArray[BasisMetadata, DT],
) -> SlateArray[BasisMetadata, DT]:
    msg = "Not implemented yet."
    raise NotImplementedError(msg)
