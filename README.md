# Slate

"The world is your slate. You get to write on it, layer by layer."

## Overview

**S**L**ATE** (Sparse Automatic Transformation Environment) is a Python package for representing quantum states and operators, without having to keep track of which basis they are stored in.

In most simulation code a wavefunction is just an array of coefficients, and the basis is left implicit. It is then up to the programmer to remember which conventions are used where, and to write the transformations between them by hand. In Slate, every `Array` carries its `Basis` along with its data. A basis is built up layer by layer from a small set of composable transformations (Fourier transforms, truncation, diagonal sparsity, ...), so Slate always knows how to convert between two representations of the same Hilbert space - and does so automatically.

The result is code which reads like the physics: you write the operation you want, and Slate takes care of getting every array into a compatible basis.

## Features

- **Automatic Basis Conversion**: Any two arrays with the same metadata can be converted into one another, with no transformation matrices built by the user.
- **Composable Primitives**: Diagonal, truncated, cropped, Fourier and trigonometric transformed, block diagonal and coordinate bases can be layered to describe complex representations.
- **Sparsity for Free**: The same primitives used to change basis are used to represent sparse data, so only the coefficients you need are ever stored.
- **Basis Aware `einsum`**: Write tensor contractions in Einstein notation, and Slate converts each array into a compatible basis before contracting, exploiting diagonal sparsity where it can.
- **Physical Metadata**: Arrays know about the space they live in (lengths, lattice vectors, labels), which is used to produce correctly oriented plots.
- **NumPy Underneath**: Coefficients are stored as a plain NumPy array, so Slate works alongside the rest of the scientific Python stack.
- **Fully Typed**: The basis of an array is tracked by the type checker as well as at runtime.

## Installation

Slate requires Python 3.14 or later. You can install SLATE directly via pip:

```bash
pip install slate-core
```

To include the optional plotting utilities (which depend on matplotlib):

```bash
pip install "slate-core[plot]"
```

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/Matt-Ord/slate.git
```

## Usage Examples

### Changing Basis

An `Array` is built from a basis and the coefficients in that basis. Converting to a different basis is a single call to `with_basis`:

```python
import numpy as np

from slate_core import Array, FundamentalBasis, basis
from slate_core.basis import CroppedBasis

x = np.arange(64)
position = FundamentalBasis.from_size(64)
psi = Array(position, np.exp(-((x - 32) ** 2) / 20).astype(np.complex128))

# Convert into a Fourier (momentum) basis
momentum = basis.as_transformed(position)
psi_k = psi.with_basis(momentum)

# Layer a CroppedBasis on top, keeping only the 16 lowest momentum states
psi_cropped = psi_k.with_basis(CroppedBasis(16, momentum).upcast())
print(psi_cropped.raw_data.shape)  # (16,)
```

### Converting Back to a Full NumPy Array

Whatever basis an array is stored in, `as_array` returns the data in the fundamental representation:

```python
np.testing.assert_allclose(psi_k.as_array(), psi.as_array(), atol=1e-12)
print(np.abs(psi_cropped.as_array() - psi.as_array()).max())  # ~1e-2
```

### Mixing Representations

Operators can be stored in whatever basis is most natural. Here the potential $V(x)$ is stored as a diagonal, while the wavefunction is stored in momentum space - `einsum` takes care of the rest:

```python
from slate_core import TupleBasis, linalg
from slate_core.basis import DiagonalBasis

# A potential operator V(x), which only stores its diagonal
potential = Array(
    DiagonalBasis(TupleBasis((position, position.dual_basis())))
    .resolve_ctype()
    .upcast(),
    (0.5 * (x - 32) ** 2).astype(np.complex128),
)

# Apply V to psi_k, without converting anything by hand
v_psi = linalg.einsum("(i j'),j->i", potential, psi_k)
```

More examples can be found in the [examples](./examples) folder, and the full API is documented at [matt-ord.github.io/slate](https://matt-ord.github.io/slate/).

## Built With Slate

- [slate_quantum](https://github.com/Matt-Ord/slate_quantum): Quantum dynamics simulations, including closed and open (stochastic Schrödinger equation) solvers.
- [multiscat](https://github.com/Matt-Ord/multiscat): Elastic atom-surface scattering calculations, where the scattering potential is provided directly as a function of position.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for how to get started.

## License

Slate is released under the [MIT License](./LICENSE).
