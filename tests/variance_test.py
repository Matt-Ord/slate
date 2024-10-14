from __future__ import annotations

from typing import Any, cast

import numpy as np

from slate.basis import Basis
from slate.basis.metadata import BasisMetadata, FundamentalBasisMetadata

a = cast(Basis[FundamentalBasisMetadata, np.floating[Any]], {})
b: Basis[BasisMetadata, np.generic] = a  # type: ignore should fail
c: Basis[BasisMetadata, np.float128] = a
