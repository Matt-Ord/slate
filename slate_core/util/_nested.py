from __future__ import annotations

type NestedData[T] = T | tuple[NestedData[T], ...]


def flatten_nested[T](nested: NestedData[T]) -> tuple[T, ...]:
    if isinstance(nested, tuple):
        return tuple(item for subtuple in nested for item in flatten_nested(subtuple))  # type: ignore unknown
    return (nested,)
