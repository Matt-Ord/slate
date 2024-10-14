from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, Self, cast

import numpy as np

from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.array.array import SlateArray
    from slate.basis._basis import Basis
    from slate.basis.stacked._tuple_basis import VariadicTupleBasis


class ExplicitBasis[M: BasisMetadata, DT: np.generic](WrappedBasis[M, DT]):
    """Represents a truncated basis."""

    def __init__(
        self: Self,
        data: SlateArray[DT, VariadicTupleBasis[Basis[M, DT], Basis[M, DT], Any, DT]],
    ) -> None:
        self._data = data
        super().__init__(data.basis[1])

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self._data.basis[0].size

    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        transformed = np.tensordot(
            cast(np.ndarray[Any, np.dtype[Never]], vectors),
            cast(
                np.ndarray[Any, np.dtype[Never]],
                self._data.raw_data.reshape(self._data.basis.shape),
            ),
            axes=([axis], [0]),
        )
        return np.moveaxis(transformed, -1, axis)

    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        transformed = np.tensordot(
            cast(np.ndarray[Any, np.dtype[Never]], vectors),
            np.linalg.inv(
                cast(
                    np.ndarray[Any, np.dtype[Never]],
                    self._data.raw_data.reshape(self._data.basis.shape),
                )
            ),
            axes=([axis], [0]),
        )
        return np.moveaxis(cast(np.ndarray[Any, np.dtype[DT1]], transformed), -1, axis)
