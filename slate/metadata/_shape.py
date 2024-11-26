from __future__ import annotations

from typing import Union

import numpy as np

type NestedLength = Union[int, tuple[NestedLength, ...]]


def size_from_nested_shape(shape: NestedLength) -> int:
    """Get the size from a nested shape."""
    if isinstance(shape, int):
        return shape
    return np.prod([size_from_nested_shape(sub_shape) for sub_shape in shape]).item()


def shallow_shape_from_nested(shape: NestedLength) -> tuple[int, ...]:
    """Get the flat shape from a nested shape."""
    if isinstance(shape, int):
        return (shape,)
    return tuple(size_from_nested_shape(sub_shape) for sub_shape in shape)
