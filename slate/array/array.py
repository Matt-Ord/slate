from __future__ import annotations  # noqa: A005

from typing import Any, Self

import numpy as np

from slate.basis.basis import Basis, BasisMetadata, FundamentalBasis


class SlateArray[_M: BasisMetadata, _DT: np.generic]:
    """An array with data stored in a given basis."""

    def __init__(
        self, basis: Basis[_M, _DT], data: np.ndarray[Any, np.dtype[_DT]]
    ) -> None:
        self._basis = basis
        self._data = data

    @property
    def basis(self: Self) -> Basis[_M, _DT]:
        """The basis of the Array."""
        return self._basis

    def as_array(self: Self) -> np.ndarray[Any, np.dtype[_DT]]:
        """Get the data as a (full) np.array.

        Parameters
        ----------
        self : Self

        Returns
        -------
        np.ndarray[Any, np.dtype[_DT]]
        """
        return self.basis.__convert_vector_into__(
            self._data.ravel(), FundamentalBasis[_M, _DT](self.basis.metadata)
        ).reshape(self.basis.fundamental_shape)
