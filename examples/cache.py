from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from slate import Array, BasisMetadata
from slate.util import cached, timed

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as dir_name:
        count = 0

        @cached(Path(dir_name) / "out")
        @timed
        def _my_expensive_function() -> Array[BasisMetadata, np.float64]:
            global count  # noqa: PLW0603
            count += 1
            return Array.from_array(np.zeros((300000, 1000)))

        _my_expensive_function()
        # The second call will be cached
        _my_expensive_function()
        assert count == 1

        # call_uncached will always call the function
        _my_expensive_function.call_uncached()
        assert count == 2

    # Just to compare it to the numpy save and load
    with tempfile.TemporaryDirectory() as dir_name:

        @timed
        def _save_numpy() -> None:
            data = Array.from_array(np.zeros((300000, 1000)))
            np.savez(Path(dir_name) / "out.npz", data.raw_data)

        @timed
        def _load_numpy() -> None:
            np.load(Path(dir_name) / "out.npz")[f"arr_{0}"]

        _save_numpy()
        _load_numpy()
