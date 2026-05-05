from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

import sympy as sp

from slate_core import Ctype
from slate_core.metadata import NestedLength, size_from_nested_shape

RAW_ARRAY_SYMBOL_NAME = "RAW_ARRAY"


@dataclass(frozen=True, kw_only=True)
class InnerArray:
    """Symbol representing the inner array in a transform."""

    size: int


class Transform[CT: Ctype[Any] = Ctype[Any]]:
    """Represents a basis transform."""

    def __init__(self) -> None: ...

    @property
    @abstractmethod
    def ctype(self) -> CT:
        """The type of data the basis supports."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of elements in the transformed basis."""

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

    @classmethod
    def _eval_transform(
        cls, inner: Transform[M, CT], *params: *tuple[Any, ...]
    ) -> Transform[M, CT] | None:
        return None

    @classmethod
    @override
    def eval(cls, *args: Any) -> Any | None:
        inner_cls, *params = args
        if isinstance(inner_cls, Transform):
            return cls._eval_transform(inner_cls, *params)
        if not isinstance(inner_cls, InnerArraySymbol):
            msg = "First argument must be a Transform or a Symbol."
            raise TypeError(msg)
        return None

    def as_inverse_transform(self) -> Transform[M, CT]:
        """Get the inverse transform."""
        msg = "Inverse transform not implemented for this transform."
        raise NotImplementedError(msg)

    def inverse(self, argindex: int = 1) -> Transform[M, CT]:
        """Get the inverse transform to plug into the inverse transform API of sympy."""
        if argindex != 1:
            msg = "Only argindex=1 is supported for inverse transforms."
            raise NotImplementedError(msg)
        return self.as_inverse_transform()


def unwrap[CT: Ctype[Any]](
    transform: Transform[CT],
) -> InnerArray | Transform[CT]:
    """Unwrap the transform."""
    return transform.args[0]


class ContractionTransform(Transform): ...


class SumTransform(Transform):
    ...
    # TODO: is this just a special case of contraction?


class AlongAxisTransform[M, CT: Ctype[Any]](Transform[M, CT]):
    if TYPE_CHECKING:

        def __new__(
            cls, *args: *tuple[Transform[M, CT] | InnerArraySymbol[M], Any, ...]
        ) -> Transform[M, CT]: ...

        @property
        def args(self) -> tuple[Transform[M, CT] | InnerArraySymbol[M], ...]: ...  # noqa: D102

    @property
    def axis_transforms(self) -> tuple[Transform[M, CT] | InnerArraySymbol[M], ...]:
        """The transforms that are applied along each axis."""
        return self.args[1:]

    @override
    def as_inverse_transform(self) -> Transform[M, CT]:
        return AlongAxisTransform(*(arg.as_inverse_transform() for arg in self.args))

    @override
    def _eval_transform(
        self,
        inner: Transform[M, CT],
        *params: *tuple[Transform[M, CT] | InnerArraySymbol[M], ...],
    ) -> Transform[M, CT] | None:
        if isinstance(inner, AlongAxisTransform):
            new_base = unwrap(inner)
            # Directly combine the axis transforms
            # that are applied to each axis.
            combined_axis_transforms = tuple(
                self_axis.subs(InnerArraySymbol(None), inner_axis)
                for self_axis, inner_axis in zip(
                    self.axis_transforms, inner.axis_transforms, strict=True
                )
            )
            return AlongAxisTransform(new_base, combined_axis_transforms)
        return None


class ElementwiseTransform(Transform):
    if TYPE_CHECKING:

        def __new__(  # noqa: D102
            cls, *args: *tuple[Transform | Symbol, Any, Function]
        ) -> Transform: ...

        @property
        def args(self) -> tuple[Transform | Symbol, Function]: ...  # noqa: D102

    @override
    def _eval_transform(
        self, inner: Transform, *params: *tuple[Transform | Symbol, ...]
    ) -> Transform | None:
        if isinstance(inner, ElementwiseTransform):
            msg = "Combine the elementwise functions"
            raise NotImplementedError(msg)
        return None


class ElementMaskTransform(Transform): ...


class UnitaryTransform(Transform): ...


class FourierTransform(UnitaryTransform): ...


a = sp.simplify()


def simplify_transform[CT: Ctype[Any]](transform: Transform[CT]) -> Transform[CT]:
    """Simplify the transform."""
    return transform
