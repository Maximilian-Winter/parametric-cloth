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

from .avatar.placement import AnchorResolver, PlacementTransform
from .avatar.segments import compute_waist_segments
from .avatar.synthetic import SyntheticBody, make_simple_body
from .fitting.mass_spring import (
    DEFAULT_DAMPING,
    DEFAULT_DT,
    DEFAULT_GRAVITY,
    DifferentiableMassSpring,
    SpringTopology,
)
from .fitting.fitter import CM_TO_M, resolve_pin_mask
from .pattern import GarmentDefinition, PatternPiece, PatternVertex
from .simulation.assembly import AssembledGarment, assemble_garment
from .simulation.tessellate import tessellate_piece
from .simulation.weld import weld_vertices


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


class DrapedGarmentPreview:
    """Result of :func:`preview_drape_garment_on_body`."""

    def __init__(self, vertices, faces, vertex_panel, body: SyntheticBody):
        self.vertices = vertices
        self.faces = faces
        self.vertex_panel = vertex_panel
        self.body = body


def _resolver_for_synthetic_body(garment: GarmentDefinition, body: SyntheticBody) -> AnchorResolver:
    """Build an AnchorResolver for a garment on a :class:`SyntheticBody`.

    Deliberately does *not* reuse
    :func:`~parametric_cloth.avatar.placement.build_resolver_for_garment`: that
    helper falls back to ``SMPLX_LANDMARKS`` vertex *indices* for any anchor it
    doesn't recognize as dynamic, which would silently index into the wrong
    mesh here (``SyntheticBody`` has completely different topology). Every
    anchor the built-in templates use must come from ``body.anchors`` or
    ``compute_waist_segments`` instead.
    """
    dynamic = dict(body.anchors)
    n_waist = sum(
        1 for p in garment.pieces
        if p.placement and p.placement.anchor.startswith("waist_segment_")
    )
    if n_waist:
        dynamic.update(compute_waist_segments(body.mesh, n_waist, waist_height=body.hip_height))
    return AnchorResolver(body.mesh, dynamic=dynamic)


def _project_out_of_body(vertices: np.ndarray, body: SyntheticBody, *, margin: float = 0.005) -> np.ndarray:
    """Push any vertex that's sunk inside the body back out to its surface.

    A cheap, non-differentiable collision correction: approximates the body as
    a circular column whose radius tapers with height
    (:meth:`SyntheticBody.radius_at_height`) and radially projects penetrating
    points back out. Good enough for a visual preview, not a real collision
    solver (no friction, no response to garment self-collision).
    """
    out = vertices.copy()
    radial = out[:, [0, 2]]
    dist = np.linalg.norm(radial, axis=1)
    limit = body.radius_at_height(out[:, 1]) + margin
    inside = dist < limit
    if np.any(inside):
        safe_dist = np.where(dist < 1e-9, 1.0, dist)
        scale = limit[inside] / safe_dist[inside]
        out[inside, 0] *= scale
        out[inside, 2] *= scale
    return out


def preview_drape_garment_on_body(
    garment: GarmentDefinition,
    body: SyntheticBody | None = None,
    *,
    pin: str = "min_y",
    levels: int = 1,
    n_steps: int = 150,
    stiffness: float = 300.0,
    damping: float = 0.95,
    gravity: np.ndarray = DEFAULT_GRAVITY,
    dt: float = DEFAULT_DT,
    collide: bool = True,
    collision_passes: int = 3,
) -> DrapedGarmentPreview:
    """Place every piece of a garment on a body, weld the seams, and drape it.

    This is the closest thing to the full pipeline's actual promise --
    parametric pattern -> 3D garment shape -- runnable with nothing but numpy:
    it places each piece using the *real* Module 2 placement math (against a
    :class:`SyntheticBody` instead of SMPL-X), assembles + welds the seams with
    :func:`~parametric_cloth.simulation.weld.weld_vertices`, and settles the
    result under gravity in one pass.

    ``pin`` fixes each piece's own attachment edge (see :func:`preview_drape`)
    -- for the built-in templates, ``"min_y"`` fits ``create_skirt``/
    ``create_cape`` (pinned at the waist/neck) and ``"max_y"`` fits
    ``create_tshirt`` (pinned at the shoulder).

    If ``collide``, a cheap *non-physical* correction runs after the drape
    completes: any vertex that ended up inside the body's approximate radius
    is pushed back out to the surface (``collision_passes`` position-only
    relaxation iterations, no re-simulation). This is deliberately a
    post-process, not real continuous collision -- interleaving it with the
    physics would reset the spring rest lengths to whatever shape each
    correction produced (``DifferentiableMassSpring`` derives rest length from
    whatever positions it's given), silently baking in distortion instead of
    the pattern's real proportions.
    """
    body = body or make_simple_body()
    resolver = _resolver_for_synthetic_body(garment, body)

    # Two placement-semantic gaps to correct before this is spatially sane:
    #
    # 1. A raw PlacementTransform puts the piece's *local origin* at the
    #    anchor. For "chest_front" etc. that's roughly mid-torso, but the
    #    panel's local origin is its hem (y=0) while it's pinned at the
    #    opposite edge (y=length, the shoulder) -- left uncorrected, the
    #    panel's whole height gets added on top of the anchor instead of
    #    hanging below it. Fixed by re-anchoring on the piece's own *pinned*
    #    edge, so the point held fixed during draping is the one placed at
    #    the body.
    # 2. basis_from_normal() always maps local +Y toward world "up," which
    #    matches create_tshirt (pinned at max-y/shoulder, hem at low-y should
    #    hang below) but is backwards for create_skirt/create_cape (pinned at
    #    min-y/waist, hem at high-y should *also* hang below -- not above).
    #    Fixed by reflecting local y around the pin's own level for "min_y"
    #    pieces before embedding, so "away from the pin" maps to "down" for
    #    both conventions. A pure y-reflection is a mirror (not a proper
    #    rotation), but nothing here depends on face winding/handedness --
    #    the mass-spring solver only uses edges (SpringTopology.from_faces),
    #    seam welding matches by declared local indices + nearest 3D point
    #    (not global left/right), and the wireframe plot draws edges, not
    #    shaded/oriented faces.
    transforms = {}
    embed_pieces = {}
    pin_masks = {}
    for piece in garment.pieces:
        if piece.placement is None:
            continue
        piece_mesh = tessellate_piece(piece, levels=levels)
        pin_mask = resolve_pin_mask(piece_mesh.vertices, pin)
        pin_masks[piece.name] = pin_mask

        embed_vertices = piece_mesh.vertices
        if isinstance(pin, str) and pin == "min_y":
            reference = piece_mesh.vertices[pin_mask, 1].mean()
            embed_vertices = piece_mesh.vertices.copy()
            embed_vertices[:, 1] = 2 * reference - piece_mesh.vertices[:, 1]
            embed_pieces[piece.name] = PatternPiece(
                name=piece.name,
                vertices=[PatternVertex(float(x), float(y)) for x, y in embed_vertices[:piece_mesh.n_corners]],
                subdivisions=piece.subdivisions, placement=piece.placement,
                fabric=piece.fabric, seam_allowance=piece.seam_allowance,
            )
        else:
            embed_pieces[piece.name] = piece

        raw = resolver.transform_for(piece.placement)
        pinned_local_2d = embed_vertices[pin_mask].mean(axis=0)
        pinned_local_3d = np.array([pinned_local_2d[0], pinned_local_2d[1], 0.0]) * CM_TO_M
        adjusted_location = raw.location - pinned_local_3d @ raw.rotation.T
        transforms[piece.name] = PlacementTransform(location=adjusted_location, rotation=raw.rotation)

    embed_garment = GarmentDefinition(
        name=garment.name, pieces=[embed_pieces.get(p.name, p) for p in garment.pieces],
        seams=garment.seams, simulation_frames=garment.simulation_frames,
        simulation_substeps=garment.simulation_substeps, gravity=garment.gravity,
    )
    assembled: AssembledGarment = assemble_garment(embed_garment, transforms, levels=levels)

    pinned_pre_weld = np.zeros(assembled.n_vertices, dtype=bool)
    for piece in garment.pieces:
        if piece.name not in assembled.piece_offsets:
            continue
        start, end = assembled.piece_offsets[piece.name]
        pinned_pre_weld[start:end] = pin_masks[piece.name]

    vertices, faces, old_to_new = weld_vertices(
        assembled.vertices, assembled.faces, assembled.seam_pairs,
    )
    n_new = vertices.shape[0]
    pinned = np.zeros(n_new, dtype=bool)
    np.maximum.at(pinned, old_to_new, pinned_pre_weld)
    vertex_panel = np.zeros(n_new, dtype=int)
    vertex_panel[old_to_new] = assembled.vertex_panel

    topology = SpringTopology.from_faces(faces, stiffness=stiffness)
    sim = DifferentiableMassSpring(
        topology, pinned, n_steps=n_steps, dt=dt, damping=damping, gravity=gravity,
    )
    draped = sim.forward(vertices)

    if collide:
        for _ in range(max(collision_passes, 1)):
            draped = _project_out_of_body(draped, body)

    return DrapedGarmentPreview(vertices=draped, faces=faces, vertex_panel=vertex_panel, body=body)
