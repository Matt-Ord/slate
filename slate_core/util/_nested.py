type NestedData[T] = T | tuple[NestedData[T], ...]


# TODO: this is poorly typed, should return only the "base" type T  # noqa: FIX002
def flatten_nested[T](nested: NestedData[T]) -> tuple[T, ...]:
    if isinstance(nested, tuple):
        return tuple(item for subtuple in nested for item in flatten_nested(subtuple))  # type: ignore unknown
    return (nested,)
