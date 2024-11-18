"""The linalg module aims to mirror np.linalg, but returns Slate-specific types.

For example, the eig and eigh functions return a diagonal matrix, in the
basis that diagonalizes the array

.. literalinclude:: ../../examples/linalg.py
    :language: python
    :lineno-start: 8
"""

from __future__ import annotations

from ._eig import eig, eig_vals, eigh, eigh_vals
from ._einsum import einsum

__all__ = ["eig", "eig_vals", "eigh", "eigh_vals", "einsum"]
