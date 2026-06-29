"""Evenly-spaced anchor points around the waist for skirt-panel placement.

A skirt's ``panel_i`` pieces anchor to ``waist_segment_i`` (Module 1's
``create_skirt``). This module derives those anchors from the avatar by slicing a
horizontal band of vertices at waist height and distributing ``n_segments``
points evenly around the resulting ring, each with an outward-facing normal.

Pure numpy -- testable against a synthetic body (e.g. a cylinder).
"""

from __future__ import annotations

import numpy as np

from .mesh import AvatarMesh, normalize


def _vertical_band(vertices: np.ndarray, height: float, band: float) -> np.ndarray:
    mask = np.abs(vertices[:, 1] - height) <= band
    return vertices[mask]


def compute_waist_segments(
    mesh: AvatarMesh,
    n_segments: int,
    *,
    waist_height: float | None = None,
    band: float = 0.04,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{"waist_segment_i": (position, outward_normal), ...}``.

    Args:
        mesh: avatar body mesh (meters).
        n_segments: number of evenly-spaced anchors (matches the skirt's panel
            count).
        waist_height: y of the waist plane; if ``None`` it is taken as the mean
            height of the ``waist_front``/``waist_back`` landmarks.
        band: half-thickness (m) of the vertex slice taken around the waist.

    The anchors are ordered counter-clockwise starting from the +x axis, so
    ``waist_segment_0`` is on the body's right-ish side and they wrap around --
    consistent with the wrap-around seams emitted by ``create_skirt``.
    """
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1 (got {n_segments})")

    verts = mesh.vertices
    if waist_height is None:
        waist_height = float(np.mean([
            mesh.landmark_position("waist_front")[1],
            mesh.landmark_position("waist_back")[1],
        ]))

    # Widen the band until we capture enough vertices to form a ring.
    ring = _vertical_band(verts, waist_height, band)
    grow = band
    while len(ring) < max(n_segments * 2, 8) and grow < 1.0:
        grow *= 1.5
        ring = _vertical_band(verts, waist_height, grow)
    if len(ring) == 0:
        raise ValueError("no vertices found near waist height; check waist_height")

    # Vertical axis through the ring centroid (horizontal components only).
    axis = ring.mean(axis=0)
    axis[1] = waist_height

    rel = ring - axis
    angles = np.mod(np.arctan2(rel[:, 2], rel[:, 0]), 2.0 * np.pi)  # [0, 2pi)
    radii = np.hypot(rel[:, 0], rel[:, 2])
    mean_radius = float(np.mean(radii)) if len(radii) else 0.0

    segments: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(n_segments):
        lo = 2.0 * np.pi * i / n_segments
        hi = 2.0 * np.pi * (i + 1) / n_segments
        in_sector = (angles >= lo) & (angles < hi)

        if np.any(in_sector):
            position = ring[in_sector].mean(axis=0)
        else:
            # Empty sector: synthesize a point on the mean-radius circle.
            mid = (lo + hi) / 2.0
            position = axis + np.array(
                [mean_radius * np.cos(mid), 0.0, mean_radius * np.sin(mid)]
            )
            position[1] = ring[:, 1].mean()

        outward = position - axis
        outward[1] = 0.0
        segments[f"waist_segment_{i}"] = (position, normalize(outward))

    return segments
