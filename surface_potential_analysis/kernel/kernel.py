from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np

from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
)
from surface_potential_analysis.operator.operator import DiagonalOperator, Operator
from surface_potential_analysis.operator.operator_list import (
    DiagonalOperatorList,
    OperatorList,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisLike,
    )

_B0_co = TypeVar("_B0_co", bound=BasisLike[Any, Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BasisLike[Any, Any], covariant=True)
_B2_co = TypeVar("_B2_co", bound=BasisLike[Any, Any], covariant=True)
_B3_co = TypeVar("_B3_co", bound=BasisLike[Any, Any], covariant=True)

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])


class NoiseKernel(TypedDict, Generic[_B0_co, _B1_co, _B2_co, _B3_co]):
    r"""
    Represents a generic noise kernel in the given basis.

    Note this kernel has an implicit 'noise operator' in this basis

    ```latex
    Z_i,j = \ket{i}\bra{j}
    ```

    which we can diagonalize to get a DiagonalNoiseKernel. This noise kernel
    represents sources of noise which have the same frequency dependance (for example
    a set of noise operators which are all markovian)
    """

    basis: StackedBasisLike[
        StackedBasisLike[_B0_co, _B1_co], StackedBasisLike[_B2_co, _B3_co]
    ]
    """The basis of the underlying noise operator"""
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


SingleBasisNoiseKernel = NoiseKernel[_B0, _B0, _B0, _B0]


class DiagonalNoiseKernel(TypedDict, Generic[_B0_co, _B1_co, _B2_co, _B3_co]):
    r"""
    Represents a noise kernel, written in diagonal form.

    Note we assume that all sources of noise in this kernel have the same time/frequency dependance

    This is useful for noise such as that caused by a local (coulomb) interaction.

    Note this kernel has an implicit 'noise operator' in this basis

    ```latex
    Z_i,j = \ket{i}\bra{j} \delta{i,j}
    ```

    Since the noise operator is zero i!=j, we only care about the diagonal elements of the kernel

    """

    basis: StackedBasisLike[
        StackedBasisLike[_B0_co, _B1_co], StackedBasisLike[_B2_co, _B3_co]
    ]
    """The basis of the underlying noise operator"""
    data: np.ndarray[tuple[int], np.dtype[np.complex128]]


SingleBasisDiagonalNoiseKernel = DiagonalNoiseKernel[_B0, _B0, _B0, _B0]


class EigenOperator(Operator[_B0_co, _B1_co], TypedDict):
    """A State vector which is the eigenvector of some operator."""

    eigenvalue: complex | np.complex128


class EigenOperatorList(
    OperatorList[_B0_co, _B1_co, _B2_co],
    TypedDict,
):
    """Represents a collection of eigen-operators, each with the same basis."""

    eigenvalue: np.ndarray[tuple[int], np.dtype[np.complex128]]


class DiagonalEigenOperatorList(
    DiagonalOperatorList[_B0_co, _B1_co, _B2_co],
    TypedDict,
):
    """Represents a collection of eigen-operators, each with the same basis."""

    eigenvalue: np.ndarray[tuple[int], np.dtype[np.complex128]]


# A noise operator represents the noise in a system.
# Each index in the noise operator is multiplied by a separate
# random operator


NoiseOperator = Operator
SingleBasisNoiseOperator = NoiseOperator[_B0, _B0]

DiagonalNoiseOperator = DiagonalOperator
SingleBasisDiagonalNoiseOperator = DiagonalNoiseOperator[_B0, _B0]

NoiseOperatorList = EigenOperatorList[_B0, _B1, _B2]
DiagonalNoiseOperatorList = DiagonalEigenOperatorList[_B0, _B1, _B2]
DiagonalNoiseOperator = DiagonalOperator


def get_single_factorized_noise_operators(
    kernel: NoiseKernel[_B0, _B1, _B0, _B1],
) -> NoiseOperatorList[FundamentalBasis[int], _B0, _B1]:
    r"""
    Given a noise kernel, find the noise operator which diagonalizes the kernel.

    Note these are the operators `L`

    Parameters
    ----------
    kernel : NoiseKernel[_B0, _B0, _B0, _B0]
        _description_

    Returns
    -------
    NoiseOperatorList[FundamentalBasis[int], _B0, _B0]
        _description_
    """
    data = (
        kernel["data"]
        .reshape(*kernel["basis"][0].shape, *kernel["basis"][1].shape)
        .swapaxes(0, 1)
        .reshape(kernel["basis"][0].n, kernel["basis"][1].n)
    )
    # Find the n^2 operators which are independent
    # I think this is always true
    np.testing.assert_array_almost_equal(data, np.conj(np.transpose(data)))

    res = np.linalg.eigh(data)
    np.testing.assert_array_almost_equal(
        data,
        np.einsum(
            "ak,k,kb->ab",
            res.eigenvectors,
            res.eigenvalues,
            np.conj(np.transpose(res.eigenvectors)),
        ),
    )
    # The original kernel has the noise operators as \ket{i}\bra{j}
    # When we diagonalize we have \hat{Z}'_\beta = U^\dagger_{\beta, \alpha} \hat{Z}_\alpha
    # np.conj(res.eigenvectors) is U^\dagger_{\beta, \alpha}
    return {
        "basis": StackedBasis(
            FundamentalBasis(kernel["basis"][0].n), kernel["basis"][0]
        ),
        "data": np.conj(np.transpose(res.eigenvectors)).reshape(-1),
        "eigenvalue": res.eigenvalues,
    }


def get_noise_kernel(
    operators: NoiseOperatorList[FundamentalBasis[int], _B0, _B1],
) -> NoiseKernel[_B0, _B1, _B0, _B1]:
    operators_data = operators["data"].reshape(-1, *operators["basis"][1].shape)

    data = np.einsum(
        "a,aji,akl->ij kl",
        operators["eigenvalue"],
        np.conj(operators_data),
        operators_data,
    )
    return {
        "basis": StackedBasis(operators["basis"][1], operators["basis"][1]),
        "data": data.reshape(-1),
    }


def get_single_factorized_noise_operators_diagonal(
    kernel: DiagonalNoiseKernel[_B0, _B1, _B0, _B1],
) -> DiagonalNoiseOperatorList[FundamentalBasis[int], _B0, _B1]:
    r"""
    For a diagonal kernel it is possible to find N independent noise sources, each of which is diagonal.

    Each of these will be represented by a particular noise operator
    ```latex
    Z_i \ket{i}\bra{i}
    ```
    Note we return a list of noise operators, rather than a single noise operator,
    as it is not currently possible to represent a sparse StackedBasis (unless it can
    be represented as a StackedBasis of individual sparse Basis)

    Parameters
    ----------
    kernel : DiagonalNoiseKernel[_B0, _B0, _B0, _B0]
        _description_

    Returns
    -------
    DiagonalNoiseOperator[BasisLike[Any, Any], BasisLike[Any, Any]]
        _description_
    """
    data = kernel["data"].reshape(kernel["basis"][0][0].n, -1)
    # Find the n^2 operators which are independent
    # I think this is always true
    np.testing.assert_array_almost_equal(data, np.conj(np.transpose(data)))
    res = np.linalg.eigh(data)

    np.testing.assert_array_almost_equal(
        data,
        np.einsum(
            "k,ak,kb->ab",
            res.eigenvalues,
            res.eigenvectors,
            np.conj(np.transpose(res.eigenvectors)),
        ),
    )
    # The original kernel has the noise operators as \ket{i}\bra{j}
    # When we diagonalize we have \hat{Z}'_\beta = U^\dagger_{\beta, \alpha} \hat{Z}_\alpha
    # np.conj(res.eigenvectors) is U^\dagger_{\beta, \alpha}
    return {
        "basis": StackedBasis(
            FundamentalBasis(kernel["basis"][0][0].n), kernel["basis"][0]
        ),
        "data": np.conj(np.transpose(res.eigenvectors)).reshape(-1),
        "eigenvalue": res.eigenvalues,
    }


def get_diagonal_noise_kernel(
    operators: DiagonalNoiseOperatorList[FundamentalBasis[int], _B0, _B1],
) -> DiagonalNoiseKernel[_B0, _B1, _B0, _B1]:
    operators_data = operators["data"].reshape(operators["basis"][0].n, -1)
    data = np.einsum(
        "a,ai,aj->ij",
        operators["eigenvalue"],
        np.conj(operators_data),
        operators_data,
    )
    return {
        "basis": StackedBasis(operators["basis"][1], operators["basis"][1]),
        "data": data.reshape(-1),
    }


def truncate_diagonal_noise_kernel(
    kernel: DiagonalNoiseKernel[_B0, _B1, _B0, _B1], *, n: int | slice
) -> DiagonalNoiseKernel[_B0, _B1, _B0, _B1]:
    operators = get_single_factorized_noise_operators_diagonal(kernel)

    arg_sort = np.argsort(np.abs(operators["eigenvalue"]))
    args = arg_sort[-n::] if isinstance(n, int) else arg_sort[n]
    return get_diagonal_noise_kernel(
        {
            "basis": StackedBasis(FundamentalBasis(args.size), operators["basis"][1]),
            "data": operators["data"]
            .reshape(operators["basis"][0].n, -1)[args]
            .ravel(),
            "eigenvalue": operators["eigenvalue"][args],
        }
    )


def truncate_noise_kernel(
    kernel: NoiseKernel[_B0, _B1, _B0, _B1], *, n: int
) -> NoiseKernel[_B0, _B1, _B0, _B1]:
    operators = get_single_factorized_noise_operators(kernel)

    arg_sort = np.argsort(np.abs(operators["eigenvalue"]))
    args = arg_sort[-n::]
    return get_noise_kernel(
        {
            "basis": StackedBasis(FundamentalBasis(n), operators["basis"][1]),
            "data": operators["data"]
            .reshape(operators["basis"][0].n, -1)[args]
            .ravel(),
            "eigenvalue": operators["eigenvalue"][args],
        }
    )
