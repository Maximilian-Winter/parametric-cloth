"""SMPL-X anatomical landmark registry.

Each entry maps a human-readable anchor name (used by ``PlacementHint.anchor``
in Module 1) to a fixed SMPL-X vertex index. Because SMPL-X has a constant
topology of 10475 vertices, a given index refers to the same body location
across all shape/pose parameters -- this is what makes placement deterministic.

.. warning::
   The indices below are taken verbatim from the design document and are
   **provisional**. They have not been verified against the published SMPL-X
   topology. ``LANDMARKS_VERIFIED`` is therefore ``False``. Before relying on
   placement in production, run :func:`verify_landmark_indices` against a real
   SMPL-X mesh and visually confirm each landmark (see
   ``scripts/verify_landmarks.py`` / the Module 2 deliverable checklist).
"""

from __future__ import annotations

import numpy as np

#: Number of vertices in the SMPL-X body mesh (neutral topology).
SMPLX_NUM_VERTICES = 10475

#: Set to True only once the indices below have been confirmed on a real mesh.
LANDMARKS_VERIFIED = False

SMPLX_LANDMARKS: dict[str, int] = {
    # Torso
    "chest_front":      3065,
    "chest_back":       5937,
    "waist_front":      3502,
    "waist_back":       6295,
    "hip_front":        1176,
    "hip_back":         4540,

    # Shoulders and arms
    "left_shoulder":    4432,
    "right_shoulder":   7198,
    "left_upper_arm":   4620,
    "right_upper_arm":  7660,
    "left_elbow":       4800,
    "right_elbow":      7830,
    "left_wrist":       5070,
    "right_wrist":      8100,

    # Legs
    "left_hip":         910,
    "right_hip":        4380,
    "left_knee":        1100,
    "right_knee":       4550,
    "left_ankle":       3330,
    "right_ankle":      6700,

    # Neck
    "neck_front":       3068,
    "neck_back":        6010,
}

#: Landmark names that should be mirror-symmetric across the body's sagittal
#: plane on a neutral (betas=0) mesh -- used by the symmetry sanity check.
LEFT_RIGHT_PAIRS: list[tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_upper_arm", "right_upper_arm"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]


def verify_landmark_indices(
    mesh=None, *, num_vertices: int | None = None
) -> list[str]:
    """Return a list of problems with the landmark registry, empty if none.

    Checks performed without a mesh:
      * every index is a non-negative int within the vertex count,
      * no two distinct landmarks share an index.

    If an ``AvatarMesh`` is supplied, additionally checks that each left/right
    pair is approximately mirror-symmetric in x on the given mesh (only
    meaningful for a neutral, front-facing body).
    """
    issues: list[str] = []

    limit = num_vertices
    if limit is None and mesh is not None:
        limit = mesh.n_vertices
    if limit is None:
        limit = SMPLX_NUM_VERTICES

    for name, idx in SMPLX_LANDMARKS.items():
        if not isinstance(idx, int) or idx < 0:
            issues.append(f"landmark '{name}' has invalid index {idx!r}")
        elif idx >= limit:
            issues.append(
                f"landmark '{name}' index {idx} >= vertex count {limit}"
            )

    seen: dict[int, str] = {}
    for name, idx in SMPLX_LANDMARKS.items():
        if idx in seen:
            issues.append(
                f"landmarks '{seen[idx]}' and '{name}' share index {idx}"
            )
        else:
            seen[idx] = name

    if mesh is not None and not issues:
        issues.extend(_check_symmetry(mesh))

    return issues


def _check_symmetry(mesh, *, tol: float = 0.05) -> list[str]:
    issues: list[str] = []
    for left, right in LEFT_RIGHT_PAIRS:
        pl = mesh.landmark_position(left)
        pr = mesh.landmark_position(right)
        # Expect mirrored x and similar y/z on a neutral body.
        if abs(pl[0] + pr[0]) > tol:
            issues.append(
                f"pair ({left}, {right}) not x-symmetric: "
                f"x={pl[0]:.3f} vs {pr[0]:.3f}"
            )
        if float(np.linalg.norm(pl[1:] - pr[1:])) > tol:
            issues.append(
                f"pair ({left}, {right}) differ in y/z beyond {tol} m"
            )
    return issues
