"""The linalg module aims to mirror np.linalg, but returns Slate-specific types.

For example, the eig and eigh functions return a diagonal matrix, in the
basis that diagonalizes the array

.. literalinclude:: ../../examples/linalg.py
    :language: python
"""

from __future__ import annotations

from ._eig import eig, eigh

__all__ = ["eig", "eigh"]
