from __future__ import annotations

from typing import Any, override

from slate.basis._basis import Basis
from slate.metadata._metadata import SimpleMetadata


class BasisStateMetadata[B: Basis[Any, Any]](SimpleMetadata):
    """Metadata for a list which is derived from some data in another basis."""

    def __init__(self, basis: B) -> None:
        self._basis = basis
        super().__init__(basis.size)

    @property
    def basis(self) -> B:
        """The basis the metadata is derived from."""
        return self._basis

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, BasisStateMetadata) and other.basis == self.basis  # type: ignore unknown

    @override
    def __hash__(self) -> int:
        return hash(self.basis)
