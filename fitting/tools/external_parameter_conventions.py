"""Explicit parameter-convention conversions for external baselines."""

from __future__ import annotations

import numpy as np


def ems_shape_to_project(shape) -> np.ndarray:
    """Convert EMS ``[meridional, azimuthal]`` to project storage order."""
    shape = np.asarray(shape, dtype=np.float64).reshape(2)
    if not np.isfinite(shape).all() or np.any(shape <= 0.0):
        raise ValueError("EMS shape parameters must be finite and positive")
    return shape[[1, 0]]
