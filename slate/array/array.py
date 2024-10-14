from __future__ import annotations  # noqa: A005

from typing import Any, Self, cast

import numpy as np

from slate.basis import Basis, FundamentalBasis
from slate.basis.metadata import FundamentalBasisMetadata


class SlateArray[DT: np.generic, B: Basis[Any, Any]]:  # B: Basis[Any, DT]
    """An array with data stored in a given basis."""

    def __init__(self, basis: B, data: np.ndarray[Any, np.dtype[DT]]) -> None:
        assert basis.size == data.size
        self._basis = basis
        self._data = data.ravel()

    @property
    def fundamental_shape(self: Self) -> tuple[int, ...]:
        """Datatype of the data stored in the array."""
        return self._basis.fundamental_shape

    @property
    def dtype(self: Self) -> np.dtype[DT]:
        """Datatype of the data stored in the array."""
        return self._data.dtype

    @property
    def basis(self: Self) -> B:
        """The basis of the Array."""
        return self._basis

    @property
    def raw_data(self: Self) -> np.ndarray[Any, np.dtype[DT]]:
        """The raw data for the array."""
        return self._data

    @raw_data.setter
    def raw_data[DT1: np.generic](  # [DT1: DT]
        self: Self, data: np.ndarray[Any, np.dtype[DT1]]
    ) -> None:
        """Set the raw data for the array."""
        assert self.basis.size == data.size
        self._data = cast(np.ndarray[Any, np.dtype[DT]], data).ravel()

    def as_array(self: Self) -> np.ndarray[Any, np.dtype[DT]]:
        """Get the data as a (full) np.array.

        Parameters
        ----------
        self : Self

        Returns
        -------
        np.ndarray[Any, np.dtype[DT]]
        """
        return self.basis.__convert_vector_into__(
            self._data.ravel(), FundamentalBasis(self.basis.metadata)
        ).reshape(self.basis.fundamental_shape)

    @staticmethod
    def from_array[DT1: np.generic](
        array: np.ndarray[Any, np.dtype[DT1]],
    ) -> SlateArray[DT1, FundamentalBasis[FundamentalBasisMetadata]]:
        """Get a SlateArray from an array.

        Parameters
        ----------
        array : np.ndarray[Any, np.dtype[DT]]

        Returns
        -------
        SlateArray[FundamentalBasisMetadata, DT]
        """
        return SlateArray[DT1, FundamentalBasis[FundamentalBasisMetadata]](
            FundamentalBasis[FundamentalBasisMetadata](
                FundamentalBasisMetadata(array.shape)
            ),
            array,
        )
