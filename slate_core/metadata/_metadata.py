from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, override

import numpy as np

if TYPE_CHECKING:
    from slate_core.metadata._shape import NestedLength

PERIODIC_FEATUIRE = "Periodic"
"""The periodic feature indicates that the basis functions are periodic, such as in Fourier interpolation."""
SIMPLE_FEATURE = "Simple"
"""The simple feature indicates that all of the basis functions are unitary."""
BARICENTRIC_FEATURE = "Barycentric"
"""This indicates that the basis functions are defined in terms of barycentric coordinates."""


class BasisMetadata(Protocol):
    """The BasisMetadata specifies all information about the basis functions."""

    @property
    def fundamental_shape(self) -> NestedLength:
        """Shape of the full data."""
        ...

    @property
    def basis_weights(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        r"""Weights for the basis functions.

        Data is not always stored in "raw" form, but rather in a form that is
        weighted by the basis functions. This property provides the weights
        that are applied to the data when it is transformed to the basis.

        The fundamental basis represents data discretised into a series
        of orthogonal basis functions U_i at discrete points R_j
        where j = 0, ..., N-1 such that:

        .. math::
            U_i(R_j) = w_k \delta_{ij}

        where :math:`w_k` are the weights. Data represented in this basis
        must therefore be scaled by these weights

        .. math::
            f(R) = f_j U_j(R)
        where

        .. math::
            f_j = f(R_j) / w_j
        """
        ...

    @property
    def features(self) -> set[str]:
        """The features of the basis."""
        ...


@dataclass(frozen=True)
class SimpleMetadata(BasisMetadata):
    """Basic Metadata for a Fundamental Basis."""

    fundamental_size: int

    @property
    @override
    def fundamental_shape(self) -> int:
        """Shape of the full data."""
        return self.fundamental_size

    @property
    @override
    def basis_weights(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return np.ones(self.fundamental_size, dtype=np.float64)

    @property
    @override
    def features(self) -> set[str]:
        """The features of the basis."""
        return {SIMPLE_FEATURE}


def is_simple_metadata(
    metadata: BasisMetadata,
) -> TypeGuard[SimpleMetadata]:
    return isinstance(metadata, SimpleMetadata)


class LabeledMetadata[DT: np.dtype[np.generic]](SimpleMetadata, ABC):
    """A metadata with some data associated to each location."""

    @property
    @abstractmethod
    def values(self) -> np.ndarray[Any, DT]:
        """Shape of the full data."""

    @property
    def unit(self) -> str:
        """Unit value for the metadata."""
        return "a.u."


class ExplicitLabeledMetadata[DT: np.dtype[np.generic]](LabeledMetadata[DT]):
    """A metadata with some data associated to each location."""

    def __init__(self, values: np.ndarray[Any, DT]) -> None:
        self._values = values
        super().__init__(values.shape)

    @property
    @override
    def values(self) -> np.ndarray[Any, DT]:
        """Shape of the full data."""
        return self._values

    @property
    @override
    def basis_weights(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return np.ones(self.fundamental_size, dtype=np.float64)

    @property
    @override
    def features(self) -> set[str]:
        return {SIMPLE_FEATURE}
