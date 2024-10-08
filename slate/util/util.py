from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import EllipsisType


def slice_along_axis(
    slice_at_axis: slice | int | None, axis: int = -1
) -> tuple[EllipsisType | slice | int | None, ...] | tuple[slice | int | None, ...]:
    """Return a slice such that the 1d slice provided by slice_at_axis, slices along the dimension provided.

    Parameters
    ----------
    slice_at_axis : slice | int | None
    axis : int, optional
        axis, by default -1

    Returns
    -------
    tuple[EllipsisType | slice | int | None, ...] | tuple[slice | int | None, ...]
    """
    from_end = False
    if axis < 0:  # choosing axis at the end
        from_end = True
        axis = -1 - axis
    # Pad the slice with slice(None)
    slice_padding = axis * (slice(None),)
    if from_end:
        return (Ellipsis, slice_at_axis, *slice_padding)

    return (*slice_padding, slice_at_axis)
