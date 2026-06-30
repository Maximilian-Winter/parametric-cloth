"""Blender-free drape preview.

A quick, dependency-free way to see how a pattern piece hangs under gravity --
no Blender, no avatar, no GPU. Reuses the differentiable mass-spring solver
built for Module 10 (:mod:`parametric_cloth.fitting.mass_spring`) purely in
forward mode; no gradients are computed here.

This is a sanity-check tool, not a substitute for the production pipeline:

* Force-based mass-spring, not the constraint-based solver Module 3 drives in
  Blender -- expect looser, less "fabric-like" behavior, especially for stiff
  materials.
* Each piece is draped **independently**, with no seams to other pieces and no
  avatar collision -- useful for "does this one panel sag the way I expect"
  while iterating on a template, not for previewing a fully assembled garment.
"""

from __future__ import annotations

import numpy as np

from .fitting.mass_spring import (
    DEFAULT_DAMPING,
    DEFAULT_DT,
    DEFAULT_GRAVITY,
    DifferentiableMassSpring,
    SpringTopology,
)
from .fitting.fitter import CM_TO_M, resolve_pin_mask
from .pattern import GarmentDefinition, PatternPiece
from .simulation.tessellate import tessellate_piece


def _oriented_initial_positions(vertices2d: np.ndarray, pin: str | np.ndarray) -> np.ndarray:
    """Lift flat 2D pattern coordinates into 3D with the pinned edge at the top.

    Garment templates don't agree on which way "down the pattern" points: a
    skirt panel's waist is at min-y and its hem at max-y, while a T-shirt
    panel's hem is at min-y and its shoulder (attachment) at max-y (see
    ``create_skirt``/``create_tshirt`` in ``templates.py``). Mapping pattern y
    to world y unflipped would start the *unpinned* edge above the pinned one
    for one of those conventions -- gravity then has to pull it down past its
    own attachment before anything looks like "hanging," which looks wrong for
    a quick visual preview. Orienting by *which edge is pinned* (not which
    template produced it) handles both conventions automatically.
    """
    x0 = np.zeros((vertices2d.shape[0], 3))
    x0[:, 0] = vertices2d[:, 0] * CM_TO_M
    if isinstance(pin, str) and pin == "min_y":
        reference = vertices2d[:, 1].min()
        x0[:, 1] = (reference - vertices2d[:, 1]) * CM_TO_M
    elif isinstance(pin, str) and pin == "max_y":
        reference = vertices2d[:, 1].max()
        x0[:, 1] = (vertices2d[:, 1] - reference) * CM_TO_M
    else:
        x0[:, 1] = vertices2d[:, 1] * CM_TO_M    # explicit pin mask: orientation is the caller's call
    return x0


def preview_drape(
    piece: PatternPiece,
    *,
    pin: str = "min_y",
    levels: int = 1,
    n_steps: int = 80,
    stiffness: float = 300.0,
    damping: float = 0.95,
    gravity: np.ndarray = DEFAULT_GRAVITY,
    dt: float = DEFAULT_DT,
) -> np.ndarray:
    """Drape a single pattern piece under gravity; returns final vertices (V,3) in meters.

    ``pin`` fixes the attachment edge in place and orients it at the top
    (``"min_y"``/``"max_y"``, or an explicit boolean/index array -- see
    :func:`~parametric_cloth.fitting.fitter.resolve_pin_mask`). The defaults
    favor a stable, visually sensible drape over physical accuracy: a
    force-based mass-spring system gets noticeably stretchy/rubbery at low
    stiffness, so this uses a stiffer spring and more damping than
    :class:`~parametric_cloth.fitting.fitter.DifferentiableClothFitter`'s
    defaults (which are tuned for optimizability, not looks).
    """
    mesh = tessellate_piece(piece, levels=levels)
    pinned = resolve_pin_mask(mesh.vertices, pin)
    topology = SpringTopology.from_faces(mesh.faces, stiffness=stiffness)
    sim = DifferentiableMassSpring(
        topology, pinned, n_steps=n_steps, dt=dt, damping=damping, gravity=gravity,
    )
    x0 = _oriented_initial_positions(mesh.vertices, pin)
    return sim.forward(x0)


def preview_drape_all(garment: GarmentDefinition, **kwargs) -> dict[str, np.ndarray]:
    """Drape every piece of a garment independently; ``{piece_name: vertices}``.

    Each piece is previewed in isolation (its own local origin) -- there is no
    cross-piece seaming or relative placement here, so this won't show the
    assembled silhouette. For that, run the full Module 3 Blender pipeline.
    """
    return {piece.name: preview_drape(piece, **kwargs) for piece in garment.pieces}
