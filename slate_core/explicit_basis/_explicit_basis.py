from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, Never, Self, cast, overload, override

import numpy as np

from slate_core import array as _array
from slate_core import basis
from slate_core._einsum import einsum
from slate_core.array import Array, build
from slate_core.array._misc import cast_as_dual
from slate_core.array._transpose import inv, transpose
from slate_core.basis import (
    AsUpcast,
    Basis,
    BasisFeature,
    BasisStateMetadata,
    FundamentalBasis,
    RecastBasis,
    TupleBasis2D,
    WrappedBasis,
)
from slate_core.basis._basis import BasisConversion, Ctype, UnionCtype
from slate_core.basis._diagonal import DiagonalBasis
from slate_core.basis._tuple import TupleBasis, TupleBasisLike
from slate_core.basis._util import as_fundamental
from slate_core.metadata import (
    BasisMetadata,
    SimpleMetadata,
    shallow_shape_from_nested,
)
from slate_core.metadata._tuple import TupleMetadata

if TYPE_CHECKING:
    from slate_core.array._array import ArrayBuilder

type Direction = Literal["forward", "backward"]


class ExplicitBasis[
    Transform: Array[
        Basis[TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None]],
        np.dtype[np.number],
    ],
    CT: Ctype[Never] = Ctype[Never],
](
    WrappedBasis[Basis, CT],
):
    """Represents an explicit basis."""

    @overload
    def __init__[
        Transform_: Array[
            Basis[
                TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None]
            ],
            np.dtype[np.number],
        ],
    ](
        self: ExplicitBasis[Transform_, Ctype[Never]],
        matrix: Transform_,
        *,
        direction: Literal["forward"] = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None: ...

    @overload
    def __init__[
        M_: TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None],
        DT_: Ctype[Never],
    ](
        self: ExplicitBasis[
            Array[Basis[M_, DT_], np.dtype[np.number]],
            Ctype[Never],
        ],
        matrix: Array[Basis[M_, DT_], np.dtype[np.number]],
        *,
        direction: Literal["backward"],
        data_id: uuid.UUID | None = None,
    ) -> None: ...

    def __init__(
        self,
        matrix: Array[
            TupleBasisLike[tuple[BasisMetadata, BasisMetadata], None],
            np.dtype[np.number],
        ],
        *,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None:
        self._matrix = matrix
        self._direction: Direction = direction
        self._data_id = data_id or uuid.uuid4()
        super().__init__(self.transform().basis.metadata().children[1].basis)

    @override
    def upcast[M: BasisMetadata](
        self: ExplicitBasisWithMetadata[M, Any],
    ) -> AsUpcast[ExplicitBasis[Transform, CT], M, CT]:
        return cast("Any", AsUpcast(self, self.metadata()))

    @override
    def metadata[M: BasisMetadata](self: ExplicitBasisWithMetadata[M, Any]) -> M:
        return cast("M", self.inner.metadata())

    @property
    @override
    def ctype(self) -> CT:
        return cast("CT", UnionCtype((self.inner.ctype, Ctype(np.number))))

    @override
    def resolve_ctype[DT_: Ctype[Never]](
        self: ExplicitBasis[
            Array[
                Basis[
                    TupleMetadata[
                        tuple[
                            SimpleMetadata,
                            BasisStateMetadata[Basis[BasisMetadata, DT_]],
                        ],
                        None,
                    ]
                ],
                np.dtype[np.number],
            ],
            Any,
        ],
    ) -> ExplicitBasis[Transform, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("ExplicitBasis[Transform, DT_]", self)

    @property
    def data_id(self) -> uuid.UUID:
        return self._data_id

    @override
    def dual_basis(self) -> Self:
        dual = super().dual_basis()
        dual._direction = "backward" if self.direction == "forward" else "forward"  # noqa: SLF001
        return dual

    def transform(self) -> Transform:
        return cast(
            "Transform",
            (
                self._matrix
                if self.direction == "forward"
                else transpose(inv(self._matrix))
            ),
        )

    def inverse_transform[
        M0_: SimpleMetadata,
        M1_: BasisStateMetadata[Basis],
        DT_: np.dtype[np.number],
        DT1_: Ctype[Never],
    ](
        self: ExplicitBasis[
            Array[Basis[TupleMetadata[tuple[M0_, M1_], None], DT1_], DT_],
            Ctype[Never],
        ],
    ) -> Array[Basis[TupleMetadata[tuple[M1_, M0_], None], DT1_], DT_]:
        return cast(
            "Array[Basis[TupleMetadata[tuple[M1_, M0_], None], DT1_], DT_]",
            (
                inv(self._matrix)
                if self.direction == "forward"
                else transpose(self._matrix)
            ),
        )

    def eigenvectors[
        M1_: SimpleMetadata,
        BInner_: Basis,
        DT_: np.dtype[np.number],
        DT1_: Ctype[Never],
    ](
        self: ExplicitBasis[
            Array[
                Basis[
                    TupleMetadata[tuple[M1_, BasisStateMetadata[BInner_]], None], DT1_
                ],
                DT_,
            ]
        ],
    ) -> ArrayBuilder[
        AsUpcast[
            RecastBasis[
                TupleBasis2D[tuple[Basis[M1_, Ctype[np.generic]], BInner_], None],
                TupleBasisLike[
                    tuple[M1_, BasisStateMetadata[BInner_]], None, Ctype[np.generic]
                ],
                TupleBasisLike[tuple[M1_, BasisStateMetadata[BInner_]], None, DT1_],
            ],
            TupleMetadata[tuple[M1_, BasisMetadata], None],
        ],
        DT_,
    ]:
        transposed = transpose(self.inverse_transform())
        inner_recast = basis.from_metadata(transposed.basis.metadata())

        state_basis = inner_recast.children[1].metadata().basis
        if inner_recast.children[1].is_dual:
            state_basis = state_basis.dual_basis()

        inner = (
            TupleBasis((inner_recast.children[0], state_basis)).resolve_ctype().upcast()
        )
        eigenvectors_basis = RecastBasis(
            inner, inner_recast.upcast(), transposed.basis
        ).upcast()
        return _array.cast_basis(transposed, eigenvectors_basis)

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExplicitBasis):
            return (
                self.size == other.size
                and other.inner == self.inner
                and other.direction == self.direction
                and other._data_id == self._data_id
            )
        return False

    @override
    def __hash__(self) -> int:
        return hash((1, self.inner, self.direction, self._data_id))

    @property
    @override
    def size(self) -> int:
        return shallow_shape_from_nested(
            self.eigenvectors().basis.metadata().fundamental_shape
        )[0]

    @property
    def direction(self) -> Direction:
        """The convention used to select the direction for the forward transform."""
        return self._direction

    @override
    def __into_inner__[DT1: np.generic, DT2: np.generic](
        self: ExplicitBasis[Any, Ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT1]:
        def fn() -> np.ndarray[Any, np.dtype[DT2]]:
            swapped = cast("np.ndarray[Any, np.dtype[Any]]", vectors).swapaxes(axis, -1)
            flat = swapped.reshape(-1, vectors.shape[axis])

            transform = self.transform()
            flat_basis = TupleBasis(
                (
                    FundamentalBasis.from_size(flat.shape[0]),
                    FundamentalBasis(
                        transform.basis.metadata().children[0],
                        is_dual=not cast("tuple[bool,...]", transform.basis.is_dual)[0],
                    ),
                )
            )
            swapped_array = build(flat_basis, flat).ok()

            transformed = einsum("(i j'),(j k)->(i k)", swapped_array, transform)
            return (
                transformed.as_array()
                .reshape(*swapped.shape[:-1], -1)
                .swapaxes(axis, -1)
            )

        return BasisConversion(fn)

    @override
    def __from_inner__[DT2: np.generic, DT3: np.generic](
        self: ExplicitBasis[Any, Ctype[DT3]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT3, DT2, DT3]:
        def fn() -> np.ndarray[Any, np.dtype[DT2]]:
            swapped = cast("np.ndarray[Any, np.dtype[Any]]", vectors).swapaxes(axis, -1)
            flat = swapped.reshape(-1, vectors.shape[axis])

            transform = self.inverse_transform()
            transform.basis.metadata().children[0]
            flat_basis = TupleBasis(
                (
                    FundamentalBasis.from_size(flat.shape[0]),
                    FundamentalBasis(
                        transform.basis.metadata().children[0],
                        is_dual=not cast("tuple[bool,...]", transform.basis.is_dual)[0],
                    ),
                )
            )
            swapped_array = build(flat_basis, flat).ok()

            transform = self.inverse_transform()
            transformed = einsum("(i j'),(j k)->(i k)", swapped_array, transform)
            return (
                transformed.as_array()
                .reshape(*swapped.shape[:-1], -1)
                .swapaxes(axis, -1)
            )

        return BasisConversion(fn)

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "LINEAR_MAP" in self.inner.features:
            out.add("ADD")
            out.add("MUL")
            out.add("SUB")
            out.add("LINEAR_MAP")
        return out

    @override
    def add_data[DT1: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: complex
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)


def _assert_unitary[DT: np.dtype[np.generic]](
    vectors: np.ndarray[Any, DT],
) -> None:
    identity_matrix = np.eye(vectors.shape[0])
    result = np.dot(vectors, np.conj(np.transpose(vectors)))

    assert np.testing.assert_allclose(
        result,
        identity_matrix,
        err_msg="The provided axis vectors are not orthonormal.",
    )


def _dual_unitary_data[
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    DT: np.dtype[np.generic],
](
    array: Array[TupleBasisLike[tuple[M1, M2], E], DT],
) -> Array[TupleBasisLike[tuple[M1, M2], E], DT]:
    conj = _array.conjugate(array)
    return cast_as_dual(conj)


class ExplicitUnitaryBasis[
    Transform: Array[
        Basis[TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None]],
        Any,
    ],
    CT: Ctype[Never] = Ctype[Never],
](ExplicitBasis[Transform, CT]):
    """Represents a truncated basis."""

    @overload
    def __init__[
        Transform_: Array[
            Basis[
                TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None]
            ],
            np.dtype[np.number],
        ],
    ](
        self: ExplicitUnitaryBasis[Transform_, Ctype[Never]],
        matrix: Transform_,
        *,
        direction: Literal["forward"] = "forward",
        data_id: uuid.UUID | None = None,
        assert_unitary: bool = False,
    ) -> None: ...

    @overload
    def __init__[
        M_: TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None],
        DT_: Ctype[Never],
    ](
        self: ExplicitUnitaryBasis[
            Array[Basis[M_, DT_], np.dtype[np.number]], Ctype[Never]
        ],
        matrix: Array[Basis[M_, DT_], np.dtype[np.number]],
        *,
        direction: Literal["backward"],
        data_id: uuid.UUID | None = None,
        assert_unitary: bool = False,
    ) -> None: ...

    def __init__(
        self,
        matrix: Array[
            TupleBasisLike[tuple[BasisMetadata, BasisMetadata], None],
            np.dtype[np.number],
        ],
        *,
        assert_unitary: bool = False,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None:
        super().__init__(cast("Any", matrix), direction=direction, data_id=data_id)
        if assert_unitary:
            states_tuple = (
                self.eigenvectors()
                .ok()
                .with_basis(as_fundamental(self.eigenvectors().basis))
                .ok()
            )
            _assert_unitary(
                states_tuple.raw_data.reshape(states_tuple.basis.metadata().shape)
            )

    @override
    def transform(self) -> Transform:
        return cast(
            "Transform",
            (
                self._matrix
                if self.direction == "forward"
                else _dual_unitary_data(self._matrix)
            ),
        )

    @override
    def inverse_transform[
        M0_: SimpleMetadata,
        M1_: BasisStateMetadata[Basis],
        DT_: np.dtype[np.number],
    ](
        self: ExplicitUnitaryBasis[
            Array[Basis[TupleMetadata[tuple[M0_, M1_], None]], DT_],
            Ctype[Never],
        ],
    ) -> Array[Basis[TupleMetadata[tuple[M1_, M0_], None]], DT_]:
        return cast(
            "Array[Basis[TupleMetadata[tuple[M1_, M0_], None]], DT_]",
            (
                _dual_unitary_data(transpose(self.transform()))
                if self.direction == "forward"
                else transpose(self._matrix)
            ),
        )


type ExplicitBasisWithInner[Inner: Basis] = ExplicitBasis[
    Array[
        Basis[TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Inner]], None]],
        np.dtype[np.number],
    ],
]
type ExplicitBasisWithMetadata[M: BasisMetadata, CT: Ctype[Never] = Ctype[Never]] = (
    ExplicitBasis[
        Array[
            Basis[
                TupleMetadata[
                    tuple[SimpleMetadata, BasisStateMetadata[Basis[M, CT]]], None
                ]
            ],
            np.dtype[np.number],
        ],
        CT,
    ]
)
type UpcastExplicitBasisWithMetadata[
    M: BasisMetadata,
    CT: Ctype[Never] = Ctype[Never],
] = AsUpcast[ExplicitBasisWithMetadata[M, CT], M, CT]

type ExplicitDiagonalBasis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    CT: Ctype[Never],
] = DiagonalBasis[
    TupleBasis[
        tuple[
            UpcastExplicitBasisWithMetadata[M0, CT],
            UpcastExplicitBasisWithMetadata[M1, CT],
        ],
        E,
        CT,
    ],
    CT,
]
