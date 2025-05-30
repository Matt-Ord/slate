from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import starmap
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Never,
    Self,
    TypeGuard,
    cast,
    override,
)

import numpy as np

from slate_core.metadata._metadata import BasisMetadata
from slate_core.metadata._shape import NestedLength, size_from_nested_shape

if TYPE_CHECKING:
    from collections.abc import Callable

BasisFeature = Literal["ADD", "MUL", "SUB", "LINEAR_MAP", "INDEX"]
"""
Features of a basis.
This specify certain operations that can be performed on the basis.
- ADD: addition of data
- MUL: multiplication of data
- SUB: subtraction of data
- LINEAR_MAP: the basis can be described as the linear map from the full basis
  ie, data_i = sum_j matrix_ij fundamental_data_j
- INDEX: The basis is a simple truncation of a fundamental basis which can be represented as a CoordinateBasis basis.
"""


type NestedBool = bool | tuple[NestedBool, ...]
type NestedBoolOrNone = bool | tuple[NestedBoolOrNone, ...] | None


def are_dual_shapes(lhs: NestedBool, rhs: NestedBool) -> bool:
    """Check if two bases are dual to each other.

    The two basis must have the same shape, otherwise a `ValueError` will be raised.

    Raises
    ------
    ValueError
        If the two shapes are not dual to each other.
    """
    if isinstance(lhs, tuple) and isinstance(rhs, tuple):
        return all(starmap(are_dual_shapes, zip(lhs, rhs, strict=False)))
    if isinstance(lhs, bool) and isinstance(rhs, bool):
        return lhs != rhs
    msg = "The two basis have different shapes"
    raise ValueError(msg)


def are_dual[M: BasisMetadata](lhs: Basis[M], rhs: Basis[M]) -> bool:
    """Check if two bases are dual to each other.

    The two basis must have the same shape, otherwise a `ValueError` will be raised.
    """
    return lhs.metadata() == rhs.metadata() and are_dual_shapes(
        lhs.is_dual, rhs.is_dual
    )


class Ctype[T: np.generic]:
    """A type which is contravariant in the data type D."""

    def __init__[T_: np.generic](self: Ctype[T_], dtype: type[T_]) -> None:
        self._dtype = cast("type[np.generic]", dtype)

    def supports_dtype[T_: np.generic](
        self, dtype: np.dtype[T_]
    ) -> TypeGuard[Ctype[T_]]:
        """Check if the type supports the given data type."""
        return self.supports_type(dtype.type)

    def supports_type[T_: np.generic](self, ty: type[T_]) -> TypeGuard[Ctype[T_]]:
        """Check if the type supports the given data type."""
        return issubclass(ty, self._dtype)

    def _variance_fn(self, value: T, _private: Never) -> None: ...

    def assert_supports_type[T_: np.generic](self: Ctype[T_], ty: type[T_]) -> None:
        """Assert that the type supports the given data type.

        Raises
        ------
        CtypeError
            If the type does not support the given data type.
        """
        if not self.supports_type(ty):
            raise CtypeError(self, ty)

    def assert_supports_dtype[T_: np.generic](
        self: Ctype[T_], dtype: np.dtype[T_]
    ) -> None:
        """Assert that the type supports the given data type.

        Raises
        ------
        CtypeError
            If the type does not support the given data type.
        """
        if not self.supports_dtype(dtype):
            raise CtypeError(self, dtype.type)


class CtypeError(TypeError):
    """Exception raised when a Ctype does not support a given data type."""

    def __init__(self, ctype: Ctype[Never], ty: type) -> None:
        super().__init__(
            f"{ctype.__class__.__name__} does not support dtype {ty.__name__}"
        )
        self.ctype = ctype
        self.ty = ty


class UnionCtype[T: np.generic](Ctype[T]):
    def __init__[T_: np.generic](
        self: UnionCtype[T_], dtypes: tuple[Ctype[T_], ...]
    ) -> None:
        self._ctypes = cast("tuple[Ctype[np.generic],...]", dtypes)

    @override
    def supports_type[T_: np.generic](self, ty: type[T_]) -> TypeGuard[Ctype[T_]]:
        """Check if the type supports the given data type."""
        return all(t.supports_type(ty) for t in self._ctypes)


class BasisConversion[T1: np.generic, T2: np.generic, T3: np.generic]:
    def __init__(self, fn: Callable[[], np.ndarray[Any, np.dtype[T2]]]) -> None:
        self._fn = fn

    def _variance_fn_1(self, value: T1, _private: Never) -> Never: ...
    def _variance_fn_2(self, value: T2, _private: Never) -> T2: ...
    def _variance_fn_3(self, value: T3, _private: Never) -> Never: ...

    def ok[T: np.generic](
        self: BasisConversion[T, T, T],
    ) -> np.ndarray[Any, np.dtype[T2]]:
        return cast("Any", self)._fn()  # noqa: SLF001


def _convert_vectors_unsafe[DT2: np.generic](
    initial: Basis[Any, Ctype[np.generic]],
    vectors: np.ndarray[Any, np.dtype[DT2]],
    final: Basis[Any, Ctype[np.generic]],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT2]]:
    assert initial.metadata() == final.metadata()

    if initial == final:
        return vectors

    fundamental = initial.__into_fundamental__(vectors, axis).ok()
    return final.__from_fundamental__(fundamental, axis).ok()


class Basis[M: BasisMetadata = BasisMetadata, CT: Ctype[Any] = Ctype[Any]](ABC):
    """Base class for a basis."""

    def __init__(self, metadata: M) -> None:
        self._metadata = metadata

    @property
    @abstractmethod
    def ctype(self) -> CT:
        """The type of data the basis supports."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of elements in the basis."""

    @property
    def fundamental_size(self) -> int:
        """Size of the full data."""
        return size_from_nested_shape(self.fundamental_shape)

    @property
    def fundamental_shape(self) -> NestedLength:
        """Shape of the full data."""
        return self.metadata().fundamental_shape

    def metadata(self) -> M:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return self._metadata

    @property
    def features(self) -> set[BasisFeature]:
        """Features of the basis."""
        return set()

    @abstractmethod
    def __into_fundamental__[T1: np.generic, T2: np.generic](
        self: Basis[Any, Ctype[T1]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, np.generic]:
        """Convert a vector in the non-conjugate basis into the fundamental basis."""

    @abstractmethod
    def __from_fundamental__[T2: np.generic, T3: np.generic](
        self: Basis[Any, Ctype[T3]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[np.generic, T2, T3]:
        """Convert a vector into the non-conjugate basis from the fundamental basis."""

    @property
    @abstractmethod
    def is_dual(self) -> NestedBool: ...

    @abstractmethod
    def dual_basis(self) -> Self:
        """Get the dual basis.

        A dual basis is a basis that is conjugate to the current basis, and
        is the basis that is used in the inner product.
        """
        ...

    def __convert_vector_into__[
        M_: BasisMetadata,
        T1: np.generic,
        T2: np.generic,
        T3: np.generic,
    ](
        self: Basis[M_, Ctype[T1]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        basis: Basis[M_, Ctype[T3]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, T3]:
        return BasisConversion[T1, T2, T3](
            lambda: _convert_vectors_unsafe(self, vectors, basis, axis)  # type: ignore BasisConversion makes this safe
        )

    def add_data[T: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[T]],
        rhs: np.ndarray[Any, np.dtype[T]],
    ) -> np.ndarray[Any, np.dtype[T]]:
        msg = "add_data not implemented for this basis"
        raise NotImplementedError(msg)

    def sub_data[T: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[T]],
        rhs: np.ndarray[Any, np.dtype[T]],
    ) -> np.ndarray[Any, np.dtype[T]]:
        msg = "sub_data not implemented for this basis"
        raise NotImplementedError(msg)

    def mul_data[T: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[T]],
        rhs: complex,
    ) -> np.ndarray[Any, np.dtype[T]]:
        msg = "mul_data not implemented for this basis"
        raise NotImplementedError(msg)

    @property
    def points(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        msg = "points not implemented for this basis, requires the INDEX feature"
        raise NotImplementedError(msg)


def supports_type[M: BasisMetadata, DT: np.generic](
    basis: Basis[M], dtype: type[DT]
) -> TypeGuard[Basis[M, Ctype[DT]]]:
    """Check if the basis supports the given data type.

    This is a type guard, so it will narrow the type of the basis to
    `Basis[M, Ctype[DT]]` if it returns `True`.
    """
    return basis.ctype.supports_type(dtype)


def supports_dtype[M: BasisMetadata, DT: np.generic](
    basis: Basis[M], dtype: np.dtype[DT]
) -> TypeGuard[Basis[M, Ctype[DT]]]:
    """Check if the basis supports the given data type.

    This is a type guard, so it will narrow the type of the basis to
    `Basis[M, Ctype[DT]]` if it returns `True`.
    """
    return basis.ctype.supports_dtype(dtype)
