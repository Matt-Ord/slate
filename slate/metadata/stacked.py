from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload, override

from slate.metadata._metadata import BasisMetadata, SimpleMetadata
from slate.metadata._shape import size_from_nested_shape

if TYPE_CHECKING:
    from slate.metadata._shape import NestedLength


class StackedMetadata[M: BasisMetadata, E](BasisMetadata):
    """Metadata built from a tuple of individual metadata entries."""

    def __init__(self, children: tuple[M, ...], extra: E) -> None:
        self._children = children
        self._extra = extra

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
    def children(self) -> tuple[M, ...]:
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

    def __getitem__(self, index: int) -> M:
        return self.children[index]

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, StackedMetadata):
            return (self.extra == other.extra) and self.children == other.children  # type: ignore unknown
        return False

    @override
    def __hash__(self) -> int:
        return hash((self.extra, self.children))

    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int], *, extra: None = None
    ) -> Metadata1D[SimpleMetadata, None]: ...
    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int], *, extra: E1
    ) -> Metadata1D[SimpleMetadata, E1]: ...

    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, int], *, extra: None = None
    ) -> Metadata2D[SimpleMetadata, SimpleMetadata, None]: ...
    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, int], *, extra: E1
    ) -> Metadata2D[SimpleMetadata, SimpleMetadata, E1]: ...

    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, int, int], *, extra: None = None
    ) -> Metadata3D[SimpleMetadata, SimpleMetadata, SimpleMetadata, None]: ...
    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, int, int], *, extra: E1
    ) -> Metadata3D[SimpleMetadata, SimpleMetadata, SimpleMetadata, E1]: ...

    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, ...], *, extra: None = None
    ) -> StackedMetadata[SimpleMetadata, None]: ...

    @overload
    @staticmethod
    def from_shape[E1](
        shape: tuple[int, ...], *, extra: E1
    ) -> StackedMetadata[SimpleMetadata, E1]: ...

    @overload
    @staticmethod
    def from_shape(shape: NestedLength, *, extra: None = None) -> BasisMetadata: ...

    @staticmethod
    def from_shape[E1](
        shape: NestedLength, *, extra: E1 | None = None
    ) -> (
        StackedMetadata[SimpleMetadata, E1]
        | StackedMetadata[SimpleMetadata, None]
        | BasisMetadata
    ):
        """Get a basic stacked metadata from a shape and (optional) extra."""
        if isinstance(shape, int):
            return SimpleMetadata(shape)
        return StackedMetadata[BasisMetadata, Any](
            tuple(StackedMetadata.from_shape(s) for s in shape), extra
        )


type AnyMetadata = BasisMetadata | StackedMetadata[AnyMetadata, Any]


class MetadataND[*M, E](StackedMetadata[Any, E]):
    """Metadata built from a tuple of three individual metadata entries."""

    def __init__[*M1, _E](
        self: MetadataND[*M1, _E],
        children: tuple[*M1],
        extra: _E,
    ) -> None:
        super().__init__(cast("tuple[BasisMetadata, ...]", children), extra)

    @property
    @override
    def children(self) -> tuple[*M]:  # type: ignore not allowed to put bounds on m
        return cast("tuple[*M,]", super().children)


class Metadata1D[M0: BasisMetadata, E](MetadataND[BasisMetadata, E]):
    """Metadata built from a tuple of two individual metadata entries."""

    def __init__[_M0: BasisMetadata, _E](
        self: Metadata1D[_M0, _E],
        children: tuple[_M0],
        extra: _E,
    ) -> None:
        super().__init__(children, cast("E", extra))

    @property
    @override
    def children(self) -> tuple[M0]:
        return cast("tuple[M0]", super().children)

    @override
    def __getitem__(self, index: int) -> M0:
        return super().__getitem__(index)


class Metadata2D[M0: BasisMetadata, M1: BasisMetadata, E](
    MetadataND[BasisMetadata, BasisMetadata, E]
):
    """Metadata built from a tuple of two individual metadata entries."""

    def __init__[_M0: BasisMetadata, _M1: BasisMetadata, _E](
        self: Metadata2D[_M0, _M1, _E],
        children: tuple[_M0, _M1],
        extra: _E,
    ) -> None:
        super().__init__(children, cast("E", extra))

    @property
    @override
    def children(self) -> tuple[M0, M1]:
        return cast("tuple[M0, M1]", super().children)

    @overload
    def __getitem__(self, index: Literal[0]) -> M0: ...
    @overload
    def __getitem__(self, index: Literal[1]) -> M1: ...
    @overload
    def __getitem__(self, index: int) -> M0 | M1: ...

    @override
    def __getitem__(self, index: int) -> M0 | M1:
        return super().__getitem__(index)


class Metadata3D[M0: BasisMetadata, M1: BasisMetadata, M2: BasisMetadata, E](
    MetadataND[BasisMetadata, BasisMetadata, BasisMetadata, E]
):
    """Metadata built from a tuple of three individual metadata entries."""

    def __init__[_M0: BasisMetadata, _M1: BasisMetadata, _M2: BasisMetadata, _E](
        self: Metadata3D[_M0, _M1, _M2, _E],
        children: tuple[_M0, _M1, _M2],
        extra: _E,
    ) -> None:
        super().__init__(children, cast("E", extra))

    @property
    @override
    def children(self) -> tuple[M0, M1, M2]:
        return cast("tuple[M0, M1, M2]", super().children)

    @overload
    def __getitem__(self, index: Literal[0]) -> M0: ...
    @overload
    def __getitem__(self, index: Literal[1]) -> M1: ...
    @overload
    def __getitem__(self, index: Literal[2]) -> M2: ...
    @overload
    def __getitem__(self, index: int) -> M0 | M1 | M2: ...

    @override
    def __getitem__(self, index: int) -> M0 | M1 | M2:
        return super().__getitem__(index)
