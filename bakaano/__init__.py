"""Bakaano package init."""

import numpy as np

# NumPy 2 removed the np.bool8 alias used by some Numba-dependent paths.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

__version__ = "1.3.7"
