"""Post-simulation result validation.

Cloth solvers fail in characteristic ways -- vertices flying to infinity, NaNs,
gross interpenetration. These pure-numpy checks detect an exploded/broken result
so the pipeline can retry with more conservative settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationResult:
    ok: bool
    issues: list[str]
    max_distance: float = 0.0

    def __bool__(self) -> bool:
        return self.ok


def validate_simulation_result(
    vertices: np.ndarray,
    avatar_center: np.ndarray,
    *,
    max_distance: float = 2.0,
) -> ValidationResult:
    """Check that a draped mesh is finite and stayed near the avatar.

    Args:
        vertices: simulated garment vertices, (V, 3) world space (meters).
        avatar_center: reference point to measure distance from.
        max_distance: any vertex farther than this is treated as an explosion.
    """
    vertices = np.asarray(vertices, dtype=float)
    center = np.asarray(avatar_center, dtype=float)
    issues: list[str] = []

    if vertices.size == 0:
        return ValidationResult(ok=False, issues=["empty result mesh"])

    if not np.all(np.isfinite(vertices)):
        n_bad = int(np.sum(~np.isfinite(vertices)))
        issues.append(f"{n_bad} non-finite vertex coordinates (NaN/Inf)")
        return ValidationResult(ok=False, issues=issues, max_distance=float("inf"))

    distances = np.linalg.norm(vertices - center, axis=1)
    max_d = float(distances.max())
    if max_d > max_distance:
        n_far = int(np.sum(distances > max_distance))
        issues.append(
            f"{n_far} vertices beyond {max_distance} m from avatar center "
            f"(max {max_d:.2f} m) -- simulation likely exploded"
        )

    return ValidationResult(ok=not issues, issues=issues, max_distance=max_d)


def settle_delta(prev_vertices: np.ndarray, curr_vertices: np.ndarray) -> float:
    """Mean per-vertex movement between two frames (meters).

    Small values indicate the cloth has settled; useful for early-stopping a
    bake loop instead of always running the full frame count.
    """
    prev = np.asarray(prev_vertices, dtype=float)
    curr = np.asarray(curr_vertices, dtype=float)
    if prev.shape != curr.shape or prev.size == 0:
        return float("inf")
    return float(np.linalg.norm(curr - prev, axis=1).mean())
