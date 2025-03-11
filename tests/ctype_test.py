from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from slate_core.basis import Ctype


@pytest.mark.parametrize(
    "dtype",
    [
        np.int32,
        np.complex256,
        np.complex128,
        np.float64,
    ],
)
def test_supports_ctype_generic(dtype: type) -> None:
    generic_ctype = Ctype(np.generic)
    assert generic_ctype.supports_type(dtype)
    assert generic_ctype.supports_dtype(np.dtype(cast("type[object]", dtype)))


@pytest.mark.parametrize(
    ("ctype", "dtype", "expected"),
    [
        (np.number, np.int32, True),
        (np.number, np.complex256, True),
        (np.number, np.complex128, True),
        (np.number, np.float64, True),
        (np.number, np.int64, True),
        (np.number, np.float32, True),
        (np.complexfloating, np.int32, False),
        (np.complexfloating, np.complex256, True),
        (np.complexfloating, np.complex128, True),
        (np.complexfloating, np.float64, False),
        (np.complexfloating, np.int64, False),
        (np.complexfloating, np.float32, False),
    ],
)
def test_supports_ctype_number(
    ctype: type[np.generic], dtype: type, *, expected: bool
) -> None:
    actual_type = Ctype(ctype)
    assert actual_type.supports_type(dtype) == expected
    assert actual_type.supports_dtype(np.dtype(cast("type[object]", dtype))) == expected
