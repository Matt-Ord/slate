from __future__ import annotations

from typing import Any, Self, cast, overload

import numpy as np

from slate.metadata import BasisMetadata, SimpleMetadata


class StackedMetadata[M: BasisMetadata, E](BasisMetadata):
    """Metadata built from a tuple of individual metadata entries."""

    def __init__(self: Self, children: tuple[M, ...], extra: E) -> None:
        self._children = children
        self._extra = extra

    @property
    def children(self: Self) -> tuple[M, ...]:
        """Children metadata."""
        return self._children

    @property
    def extra(self: Self) -> E:
        """Extra metadata."""
        return self._extra

    @property
    def fundamental_shape(self: Self) -> tuple[int, ...]:
        """Shape of the full data."""
        return tuple(np.prod(i.fundamental_shape).item() for i in self.children)

    def __getitem__(self: Self, index: int) -> M:
        return self.children[index]

    def __eq__(self, value: object) -> bool:
        if isinstance(value, StackedMetadata):
            return (self.extra == value.extra) and self.children == value.children  # type: ignore unknown
        return False

    def __hash__(self) -> int:
        return hash((self.extra, self.children))

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

    @staticmethod
    def from_shape[E1](
        shape: tuple[int, ...], *, extra: E1 | None = None
    ) -> StackedMetadata[SimpleMetadata, E1] | StackedMetadata[SimpleMetadata, None]:
        """Get a basic stacked metadata from a shape and (optional) extra."""
        return StackedMetadata[SimpleMetadata, Any](
            tuple(SimpleMetadata((s,)) for s in shape), extra
        )


class MetadataND[*M, E](StackedMetadata[Any, E]):
    """Metadata built from a tuple of three individual metadata entries."""

    def __init__[*M1, _E](
        self: MetadataND[*M1, _E],
        children: tuple[*M1],
        extra: _E,
    ) -> None:
        super().__init__(cast(tuple[BasisMetadata, ...], children), extra)

    @property
    @overload
    def children(self: Self) -> tuple[*M]:  # type: ignore not allowed to put bounds on m
        return cast(tuple[*M], super().children)


class Metadata1D[M0: BasisMetadata, E](MetadataND[BasisMetadata, E]):
    """Metadata built from a tuple of two individual metadata entries."""

    def __init__[_M0: BasisMetadata, _E](
        self: Metadata1D[_M0, _E],
        children: tuple[_M0],
        extra: _E,
    ) -> None:
        super().__init__(children, cast(E, extra))

    @property
    @overload
    def children(self: Self) -> tuple[M0]:
        return cast(tuple[M0], super().children)


class Metadata2D[M0: BasisMetadata, M1: BasisMetadata, E](
    MetadataND[BasisMetadata, BasisMetadata, E]
):
    """Metadata built from a tuple of two individual metadata entries."""

    def __init__[_M0: BasisMetadata, _M1: BasisMetadata, _E](
        self: Metadata2D[_M0, _M1, _E],
        children: tuple[_M0, _M1],
        extra: _E,
    ) -> None:
        super().__init__(children, cast(E, extra))

    @property
    @overload
    def children(self: Self) -> tuple[M0, M1]:
        return cast(tuple[M0, M1], super().children)


class Metadata3D[M0: BasisMetadata, M1: BasisMetadata, M2: BasisMetadata, E](
    MetadataND[BasisMetadata, BasisMetadata, BasisMetadata, E]
):
    """Metadata built from a tuple of three individual metadata entries."""

    def __init__[_M0: BasisMetadata, _M1: BasisMetadata, _M2: BasisMetadata, _E](
        self: Metadata3D[_M0, _M1, _M2, _E],
        children: tuple[_M0, _M1, _M2],
        extra: _E,
    ) -> None:
        super().__init__(children, cast(E, extra))

    @property
    @overload
    def children(self: Self) -> tuple[M0, M1, M2]:
        return cast(tuple[M0, M1, M2], super().children)
