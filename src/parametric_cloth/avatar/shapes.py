"""Representative SMPL-X body-shape samples.

SMPL-X body shape is controlled by 10 ``beta`` parameters. These samples span a
range of proportions for simulation sampling (Module 3) and training-data
generation (Module 6). All share the SMPL-X topology, so landmark indices stay
valid across every shape.
"""

from __future__ import annotations

import numpy as np

N_BETAS = 10

BODY_SHAPE_SAMPLES: dict[str, np.ndarray] = {
    "average":   np.zeros(N_BETAS),
    "athletic":  np.array([1.5, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    "heavy":     np.array([-1.0, 1.5, 0.8, 0, 0, 0, 0, 0, 0, 0], dtype=float),
    "tall_slim": np.array([0.5, -1.0, -0.5, 1.5, 0, 0, 0, 0, 0, 0], dtype=float),
    "short":     np.array([0.0, 0.5, 0.3, -1.5, 0, 0, 0, 0, 0, 0], dtype=float),
}


def get_body_shape(name: str) -> np.ndarray:
    """Return a copy of a named beta vector."""
    try:
        return BODY_SHAPE_SAMPLES[name].copy()
    except KeyError:
        raise KeyError(
            f"unknown body shape '{name}'; known: {sorted(BODY_SHAPE_SAMPLES)}"
        ) from None
