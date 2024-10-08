from __future__ import annotations

from typing import Any, Iterator, Literal, Protocol, Self, overload, override

import numpy as np

from slate.basis.basis import Basis, FundamentalBasis
from slate.basis.metadata import BasisMetadata


class TupleMetadata[_M: BasisMetadata, _E](BasisMetadata):
    """Metadata built from a tuple of individual metadata entries."""

    def __init__(self: Self, inner: tuple[_M, ...], extra: _E) -> None:
        self._inner = inner
        self._extra = extra

    @property
    def inner(self: Self) -> tuple[_M, ...]:
        """Inner metadata."""
        return self._inner

    @property
    def extra(self: Self) -> _E:
        """Extra metadata."""
        return self._extra

    @property
    def fundamental_shape(self: Self) -> tuple[int, ...]:
        """Shape of the full data."""
        return tuple(np.prod(i.fundamental_shape).item() for i in self._inner)

    def __getitem__(self: Self, index: int) -> _M:
        return self._inner[index]

    def __eq__(self, value: object) -> bool:
        if isinstance(value, TupleMetadata):
            return (self.extra == value.extra) and all(  # type: ignore unknown
                a == b
                for (a, b) in zip(self.inner, value.inner)  # type: ignore unknown
            )
        return False

    def __hash__(self) -> int:
        return hash((self.extra, self.inner))


type StackedBasis[_M: BasisMetadata, _E, _DT: np.generic] = Basis[
    TupleMetadata[_M, _E], _DT
]


def stacked_basis_as_fundamental[_M: BasisMetadata, _E, _DT: np.generic](
    basis: StackedBasis[_M, _E, _DT],
) -> TupleBasis[_M, _E, _DT]:
    """Get the equivalent Tuple of Fundamental Basis.

    Returns
    -------
    TupleBasis[_M, _E, _DT]
    """
    return TupleBasis[_M, _E, _DT](
        tuple(FundamentalBasis(basis.metadata[i]) for i in range(basis.n_dim)),
        basis.metadata.extra,
    )


def _convert_tuple_basis_vector[_M: BasisMetadata, _E, _DT: np.generic](
    vectors: np.ndarray[Any, np.dtype[_DT]],
    initial_basis: TupleBasis[_M, _E, _DT],
    final_basis: TupleBasis[_M, _E, _DT],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[_DT]]:
    """
    Convert a vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    vector : np.ndarray[tuple[int], np.dtype[np.complex_] | np.dtype[np.float_]]
        the vector to convert
    from_config : _B3d0Inv
    to_config : _B3d1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    swapped = vectors.swapaxes(axis, 0)
    stacked = swapped.reshape(*initial_basis.shape, *swapped.shape[1:])
    for ax, (initial, final) in enumerate(zip(initial_basis, final_basis, strict=True)):
        stacked = initial.__convert_vector_into__(stacked, final, axis=ax)

    return stacked.reshape(-1, *swapped.shape[1:]).swapaxes(axis, 0)


class TupleBasis[_M: BasisMetadata, _E, _DT: np.generic](
    Basis[TupleMetadata[_M, _E], _DT]
):
    """Represents a Tuple of independent basis."""

    def __init__(
        self: Self, inner: tuple[Basis[_M, _DT], ...], extra_metadata: _E
    ) -> None:
        self._inner = inner
        self._metadata = TupleMetadata(tuple(i.metadata for i in inner), extra_metadata)

    @property
    def inner(self) -> tuple[Basis[_M, _DT], ...]:
        """Inner basis."""
        return self._inner

    @property
    def size(self) -> int:
        """Number of elements in the basis."""
        return np.prod(self.shape).item()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the inner data."""
        return tuple(x.size for x in self._inner)

    @override
    def __into_fundamental__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        basis = stacked_basis_as_fundamental(self)
        return _convert_tuple_basis_vector(vectors, basis, self, axis)

    @override
    def __from_fundamental__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        basis = stacked_basis_as_fundamental(self)
        return _convert_tuple_basis_vector(vectors, basis, self, axis)

    def __iter__(self) -> Iterator[Basis[_M, _DT]]:
        return self._inner.__iter__()

    def __getitem__(self: Self, index: int) -> Basis[_M, _DT]:
        return self._inner[index]

    def __convert_vector_into__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        basis: Basis[TupleMetadata[_M, _E], _DT],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        assert self.metadata == basis.metadata

        if self == basis:
            return vectors

        if not isinstance(basis, TupleBasis):
            return super().__convert_vector_into__(vectors, basis, axis)

        # We overload __convert_vector_into__, more likely to get the 'happy path'
        return _convert_tuple_basis_vector(vectors, self, basis, axis)  # type: ignore unknown


class TupleBasisLike[*_TS](Protocol):
    """A type for basis like the tuple basis."""

    @property
    def inner(self) -> tuple[*_TS]:
        """Inner basis."""
        ...

    @property
    def size(self) -> int:
        """Number of elements in the basis."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the inner data."""
        ...

    @overload
    def __getitem__[_B: Basis[Any, Any]](
        self: TupleBasisLike[Any, Any, _B, *tuple[Any, ...]], index: Literal[2]
    ) -> _B: ...
    @overload
    def __getitem__[_B: Basis[Any, Any]](
        self: TupleBasisLike[Any, _B, *tuple[Any, ...]], index: Literal[1]
    ) -> _B: ...
    @overload
    def __getitem__[_B: Basis[Any, Any]](
        self: TupleBasisLike[_B, *tuple[Any, ...]], index: Literal[0]
    ) -> _B: ...
    @overload
    def __getitem__(self: Self, index: int) -> Basis[Any, Any]: ...

    def __getitem__(self: Self, index: int) -> Basis[Any, Any]: ...
