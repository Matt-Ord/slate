from __future__ import annotations

from dataclasses import dataclass

type NestedTokens = str | list[NestedTokens]


def _parse_einsum_tokens(tokens: list[str], idx: int = 0) -> tuple[NestedTokens, int]:
    out = list[NestedTokens]()

    while idx < len(tokens):
        token = tokens[idx]
        if token == "(":  # noqa: S105
            # Start of deeper nesting
            inner, idx = _parse_einsum_tokens(tokens, idx + 1)
            out.append(inner)
            continue
        if token == ")":  # noqa: S105
            # End of current nesting
            return (out, idx + 1)

        # Add token to current nesting
        out.append(token)
        idx += 1

    return (out, idx + 1)


@dataclass(frozen=True)
class EinsteinIndex:
    label: str
    is_dual: bool = False


type NestedEinsteinIndex = EinsteinIndex | tuple[NestedEinsteinIndex, ...]


def _process_einsum_tokens(tokens: NestedTokens) -> NestedEinsteinIndex:
    if isinstance(tokens, str):
        if tokens.endswith("'"):
            return EinsteinIndex(tokens[:-1], is_dual=True)
        return EinsteinIndex(tokens, is_dual=False)

    return tuple(_process_einsum_tokens(t) for t in tokens)


def parse_einsum_index(idx: str) -> NestedEinsteinIndex:
    """
    Parse an einsum index string into a nested einstein index.

    An einsum index is a series of labels in a nested structure.

    eg
    (i j)
    (m (i j))
    (m (i j) k)
    (m (i (k l)) (k l))

    Labels are separated by spaces and nested by brackets.
    """
    idx_split = idx.replace("(", " ( ").replace(")", " ) ").split()
    return _process_einsum_tokens(_parse_einsum_tokens(idx_split)[0][0])


@dataclass
class EinsumSpecification:
    parts: tuple[NestedEinsteinIndex, ...]
    result: NestedEinsteinIndex | None


def parse_einsum_specification(specification: str) -> EinsumSpecification:
    parts, result = specification.split("->")
    parts = tuple(parse_einsum_index(p) for p in parts.split(","))
    result = parse_einsum_index(result) if result.strip("") else None
    return EinsumSpecification(parts, result)
