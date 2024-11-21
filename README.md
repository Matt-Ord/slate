# Slate

"The world is your slate. You get to write on it, layer by layer."

## Overview

**S**L**ATE** (Sparse Automatic Transformation Environment) is a Python package designed to simplify the representation and manipulation of sparse or compressed matrices through a hierarchical, modular system. By providing composable primitives like diagonal, truncated, and COO (Coordinate List) bases, SLATE allows for flexibility and automatic transformations between different matrix representations.

SLATE aims to be **versatile** and **structured**, making it ideal for users who need a flexible environment to work with matrix and tensor computations without the constraints of optimization or efficiency requirements.

## Features

- **Composable Primitives**: Use diagonal, truncated, transformed, and other bases to build complex matrix representations.
- **Hierarchical Basis**: A modular and hierarchical approach to represent matrices and tensors.
- **Automatic Transformations**: Seamlessly switch between different matrix representations.
- **Sparse Representation**: Handle large, sparse matrices in an intuitive way.
- **Customization**: Combine and expand matrix basis in any configuration you need.

## Installation

You can install SLATE directly via pip:

```bash
pip install slate-core
```

## Usage Examples

### Creating a SlateArray

To create a `SlateArray` with a given basis and data:

```python
import numpy as np
from slate.array import SlateArray
from slate.basis import FundamentalBasis
from slate.metadata import SimpleMetadata

# Create some data
data = np.array([[1, 2], [3, 4]])

# Create a SlateArray
new_slate_array = SlateArray.from_array(data)

print(slate_array.raw_data)
```

### Using Different Bases

SLATE supports various bases like `TruncatedBasis`, `CroppedBasis`, and `TransformedBasis`. Here is an example of using a `CroppedBasis`:

```python
from slate.basis import CroppedBasis
from slate.metadata import BasisMetadata

# Create a truncated basis
truncated_basis = CroppedBasis(10, FundamentalBasis(SimpleMetadata((20,))))

# Create a SlateArray with the truncated basis
truncated_slate_array = SlateArray(truncated_basis, data)
```

### Converting Back to a Full NumPy Array

To convert the `SlateArray` back to a full NumPy array:

```python
full_array = slate_array.as_array()
print(full_array)
```
