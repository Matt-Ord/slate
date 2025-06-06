"""The linalg module aims to mirror np.linalg, but returns Slate-specific types.

For example, the eig and eigh functions return a diagonal matrix, in the
basis that diagonalizes the array

.. literalinclude:: ../../examples/linalg.py
    :language: python
    :lineno-start: 8
"""

from __future__ import annotations

from slate_core._einsum import einsum
from slate_core.array._transpose import inv, transpose
from slate_core.linalg._eig import (
    get_eigenvalues,
    get_eigenvalues_hermitian,
    into_diagonal,
    into_diagonal_hermitian,
)
from slate_core.linalg._misc import extract_diagonal, norm

__all__ = [
    "einsum",
    "extract_diagonal",
    "get_eigenvalues",
    "get_eigenvalues",
    "get_eigenvalues_hermitian",
    "get_eigenvalues_hermitian",
    "into_diagonal",
    "into_diagonal",
    "into_diagonal",
    "into_diagonal_hermitian",
    "into_diagonal_hermitian",
    "inv",
    "norm",
    "transpose",
]
