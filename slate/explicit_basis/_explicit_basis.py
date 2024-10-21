from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, Self, cast, override

import numpy as np

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis._basis import Basis
from slate.basis.stacked._tuple_basis import tuple_basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate.basis.stacked._tuple_basis import VariadicTupleBasis


class ExplicitBasis[M: BasisMetadata, DT: np.generic](
    WrappedBasis[M, DT, Basis[M, DT]]
):
    """Represents an explicit basis."""

    def __init__(
        self: Self,
        data: SlateArray[DT, VariadicTupleBasis[DT, Basis[M, DT], Basis[M, DT], Any]],
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

    @override
    def with_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](self: Self, inner: Basis[M1, DT1]) -> ExplicitBasis[M1, DT1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](
        self: Self, wrapper: Callable[[Basis[M, DT]], Basis[M1, DT1]]
    ) -> ExplicitBasis[M1, DT1]:
        """Get the wrapped basis after wrapper is applied to inner.

        Returns
        -------
        TruncatedBasis[M, DT]
        """
        new_basis = tuple_basis(
            (self._data.basis[0], wrapper(self.inner)),
            extra_metadata=self._data.basis.metadata.extra,
        )
        return ExplicitBasis(
            cast(SlateArray[DT1, Any], convert_array(self._data, new_basis))
        )


def _assert_unitary[DT: np.generic](vectors: np.ndarray[Any, np.dtype[DT]]) -> None:
    identity_matrix = np.eye(vectors.shape[0])
    result = np.dot(vectors, np.conj(np.transpose(vectors)))

    assert np.testing.assert_allclose(
        result,
        identity_matrix,
        err_msg="The provided axis vectors are not orthonormal.",
    )


class ExplicitUnitaryBasis[M: BasisMetadata, DT: np.generic](ExplicitBasis[M, DT]):
    """Represents a truncated basis."""

    def __init__(
        self: Self,
        data: SlateArray[DT, VariadicTupleBasis[DT, Basis[M, DT], Basis[M, DT], Any]],
        *,
        assert_unitary: bool = False,
    ) -> None:
        if assert_unitary:
            _assert_unitary(self._data.raw_data.reshape(self._data.basis.shape))
        super().__init__(data)

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
            np.conj(
                np.transpose(
                    cast(
                        np.ndarray[Any, np.dtype[Never]],
                        self._data.raw_data.reshape(self._data.basis.shape),
                    )
                )
            ),
            axes=([axis], [0]),
        )
        return np.moveaxis(cast(np.ndarray[Any, np.dtype[DT1]], transformed), -1, axis)
