from __future__ import annotations

from typing import Any, Self, overload

import numpy as np

from slate.metadata._metadata import BasisMetadata, SimpleMetadata


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
