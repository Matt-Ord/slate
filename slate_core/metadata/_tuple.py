from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Never,
    TypeGuard,
    cast,
    overload,
    override,
)

from slate_core.metadata._metadata import BasisMetadata, SimpleMetadata
from slate_core.metadata._shape import size_from_nested_shape
from slate_core.util._product import outer_product

if TYPE_CHECKING:
    import numpy as np

    from slate_core.metadata._shape import NestedLength


class TupleMetadata[
    C: tuple[BasisMetadata, ...] = tuple[BasisMetadata, ...],
    E = Any,
](BasisMetadata):
    """Metadata built from a tuple of individual metadata entries."""

    @overload
    def __init__[C_: tuple[BasisMetadata, ...] = tuple[BasisMetadata, ...], E_ = Any](
        self: TupleMetadata[C_, E_], children: C_, extra: E_
    ) -> None: ...
    @overload
    def __init__[C_: tuple[BasisMetadata, ...] = tuple[BasisMetadata, ...]](
        self: TupleMetadata[C_, None], children: C_, extra: None = None
    ) -> None: ...

    def __init__(self, children: C, extra: E | None = None) -> None:
        self._children = children
        self._extra = cast("E", extra)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the metadata."""
        return tuple(size_from_nested_shape(i.fundamental_shape) for i in self.children)

    @property
    def n_dim(self) -> int:
        """Number of dimensions."""
        return len(self.children)

    @property
    def fundamental_size(self) -> int:
        """Size of the full data."""
        return size_from_nested_shape(self.fundamental_shape)

    @property
    def children(self) -> C:
        """Children metadata."""
        return self._children

    @property
    def extra(self) -> E:
        """Extra metadata."""
        return self._extra

    @property
    @override
    def fundamental_shape(self) -> tuple[NestedLength, ...]:
        """Shape of the full data."""
        return tuple(i.fundamental_shape for i in self.children)

    @override
    def __eq__(self, other: object) -> bool:
        if is_tuple_metadata(other):
            return (self.extra == other.extra) and self.children == other.children
        return False

    @override
    def __hash__(self) -> int:
        return hash((self.extra, self.children))

    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int], *, extra: None = None
    ) -> TupleMetadata[tuple[SimpleMetadata], None]: ...
    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int], *, extra: E1
    ) -> TupleMetadata[tuple[SimpleMetadata], E1]: ...

    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, int], *, extra: None = None
    ) -> TupleMetadata[tuple[SimpleMetadata, SimpleMetadata], None]: ...
    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, int], *, extra: E1
    ) -> TupleMetadata[tuple[SimpleMetadata, SimpleMetadata], E1]: ...

    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, int, int], *, extra: None = None
    ) -> TupleMetadata[tuple[SimpleMetadata, SimpleMetadata, SimpleMetadata], None]: ...
    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, int, int], *, extra: E1
    ) -> TupleMetadata[tuple[SimpleMetadata, SimpleMetadata, SimpleMetadata], E1]: ...

    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, ...], *, extra: None = None
    ) -> TupleMetadata[tuple[SimpleMetadata, ...], None]: ...

    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, ...], *, extra: E1
    ) -> TupleMetadata[tuple[SimpleMetadata, ...], E1]: ...

    @overload
    @staticmethod
    def from_shape[E1](shape: int, *, extra: None = None) -> SimpleMetadata: ...

    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[NestedLength, ...], *, extra: None = None
    ) -> TupleMetadata[tuple[BasisMetadata, ...], None]: ...
    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[NestedLength, ...], *, extra: E1
    ) -> TupleMetadata[tuple[BasisMetadata, ...], E1]: ...

    @staticmethod
    def from_shape[E1](
        shape: NestedLength, *, extra: E1 | None = None
    ) -> (
        TupleMetadata[tuple[SimpleMetadata, ...], E1]
        | TupleMetadata[tuple[SimpleMetadata, ...], None]
        | BasisMetadata
    ):
        """Get a basic stacked metadata from a shape and (optional) extra."""
        if isinstance(shape, int):
            return SimpleMetadata(shape)
        return TupleMetadata[tuple[BasisMetadata, ...], Any](
            tuple(TupleMetadata.from_shape(s) for s in shape), extra
        )

    @property
    @override
    def basis_weights(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return outer_product(*(c.basis_weights for c in self.children)).reshape(-1)

    @property
    @override
    def features(self) -> set[str]:
        return set[str].intersection(*(c.features for c in self.children))


type AnyMetadata = BasisMetadata | TupleMetadata[AnyMetadata, Any]


@overload
def is_tuple_metadata(
    metadata: object, *, n_dim: Literal[1]
) -> TypeGuard[TupleMetadata[tuple[BasisMetadata], Never]]: ...
@overload
def is_tuple_metadata(
    metadata: object, *, n_dim: Literal[2]
) -> TypeGuard[TupleMetadata[tuple[BasisMetadata, BasisMetadata], Never]]: ...
@overload
def is_tuple_metadata(
    metadata: object, *, n_dim: Literal[3]
) -> TypeGuard[
    TupleMetadata[tuple[BasisMetadata, BasisMetadata, BasisMetadata], Never]
]: ...
@overload
def is_tuple_metadata(
    metadata: object, *, n_dim: int | None = None
) -> TypeGuard[TupleMetadata[tuple[BasisMetadata, ...], Never]]: ...


def is_tuple_metadata(
    metadata: object, *, n_dim: int | None = None
) -> TypeGuard[TupleMetadata[tuple[BasisMetadata, ...], Never]]:
    return isinstance(metadata, TupleMetadata) and (
        n_dim is None or metadata.n_dim == n_dim
    )
