from __future__ import annotations

from typing import TypeGuard, override

from slate_core.basis._basis import Basis
from slate_core.metadata._metadata import SimpleMetadata


class BasisStateMetadata[B: Basis = Basis](SimpleMetadata):
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
        return is_basis_state_metadata(other) and other.basis == self.basis

    @override
    def __hash__(self) -> int:
        return hash(self.basis)


def is_basis_state_metadata(obj: object) -> TypeGuard[BasisStateMetadata]:
    """Check if an object is a basis state metadata."""
    return isinstance(obj, BasisStateMetadata)
