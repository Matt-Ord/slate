from __future__ import annotations  # noqa: A005

from typing import Any, Self

import numpy as np

from slate.basis.basis import Basis, FundamentalBasis
from slate.basis.metadata import FundamentalBasisMetadata


class SlateArray[_B: Basis[Any, Any]]:
    """An array with data stored in a given basis."""

    def __init__(self, basis: _B, data: np.ndarray[Any, np.dtype[Any]]) -> None:
        self._basis = basis
        self._data = data

    @property
    def dtype(self: Self) -> np.dtype[Any]:
        """Datatype of the data stored in the array."""
        return self._data.dtype

    @property
    def basis(self: Self) -> _B:
        """The basis of the Array."""
        return self._basis

    @property
    def raw_data(self: Self) -> np.ndarray[Any, np.dtype[Any]]:
        """The raw data for the array."""
        return self._data

    def as_array(self: Self) -> np.ndarray[Any, np.dtype[Any]]:
        """Get the data as a (full) np.array.

        Parameters
        ----------
        self : Self

        Returns
        -------
        np.ndarray[Any, np.dtype[_DT]]
        """
        return self.basis.__convert_vector_into__(
            self._data.ravel(), FundamentalBasis(self.basis.metadata)
        ).reshape(self.basis.fundamental_shape)

    @staticmethod
    def from_array[_DT: np.generic](
        array: np.ndarray[Any, np.dtype[_DT]],
    ) -> SlateArray[FundamentalBasis[FundamentalBasisMetadata, _DT]]:
        """Get a SlateArray from an array.

        Parameters
        ----------
        array : np.ndarray[Any, np.dtype[_DT]]

        Returns
        -------
        SlateArray[FundamentalBasisMetadata, _DT]
        """
        return SlateArray(
            FundamentalBasis(FundamentalBasisMetadata(array.shape)), array
        )
