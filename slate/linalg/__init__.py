"""The linalg module aims to mirror np.linalg, but returns Slate-specific types.

For example, the eig and eigh functions return a diagonal matrix, in the
basis that diagonalizes the array

.. literalinclude:: ../../examples/linalg.py
    :language: python
    :lineno-start: 8
"""

from __future__ import annotations

from slate.array._transpose import inv, transpose
from slate.linalg._eig import (
    get_eigenvalues,
    get_eigenvalues_hermitian,
    into_diagonal,
    into_diagonal_hermitian,
)
from slate.linalg._einsum import einsum

__all__ = [
    "einsum",
    "get_eigenvalues",
    "get_eigenvalues_hermitian",
    "into_diagonal",
    "into_diagonal",
    "into_diagonal_hermitian",
    "inv",
    "transpose",
]
