"""Matplotlib-based visualization helpers (optional: ``pip install -e ".[viz]"``).

Turns "the tests pass" into an actual picture: pattern outlines, draped
wireframes, and fitting loss curves. ``matplotlib`` is imported lazily so the
core package has no plotting dependency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .pattern import GarmentDefinition, PatternPiece


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ModuleNotFoundError(
            "parametric_cloth.viz needs matplotlib; install with "
            "`pip install -e \".[viz]\"` or `pip install matplotlib`"
        ) from None


def plot_pattern_piece(piece: PatternPiece, *, ax=None, show: bool = True, label_vertices: bool = True):
    """Plot one pattern piece's 2D outline (centimeters)."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    xs = [v.x for v in piece.vertices] + [piece.vertices[0].x]
    ys = [v.y for v in piece.vertices] + [piece.vertices[0].y]
    ax.plot(xs, ys, "-o", markersize=4)
    ax.fill(xs, ys, alpha=0.15)
    if label_vertices:
        for i, v in enumerate(piece.vertices):
            ax.annotate(str(i), (v.x, v.y), fontsize=8, color="gray")

    ax.set_title(piece.name)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal")
    if show:
        plt.show()
    return ax


def plot_pattern_pieces(garment: GarmentDefinition, *, show: bool = True):
    """Plot every piece of a garment in its own subplot."""
    plt = _require_matplotlib()
    n = len(garment.pieces)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    for i, piece in enumerate(garment.pieces):
        ax = axes[i // cols][i % cols]
        plot_pattern_piece(piece, ax=ax, show=False)
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis("off")

    fig.suptitle(garment.name)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_draped_wireframe(
    vertices: np.ndarray, faces: np.ndarray, *, ax=None, show: bool = True,
    elev: float = 20, azim: float = -60, title: Optional[str] = None,
):
    """3D wireframe plot of a draped mesh (e.g. from ``preview_drape``)."""
    plt = _require_matplotlib()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401 (registers 3d proj)

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    for face in faces:
        loop = list(face) + [face[0]]
        ax.plot(vertices[loop, 0], vertices[loop, 1], vertices[loop, 2],
                color="steelblue", linewidth=0.6)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return ax


_PANEL_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def plot_garment_on_body(preview, *, ax=None, show: bool = True, elev: float = 15,
                         azim: float = -60, title: Optional[str] = None):
    """3D plot of a :class:`~parametric_cloth.preview.DrapedGarmentPreview`.

    Draws the body as a light gray wireframe and the garment on top, colored
    per source pattern piece (one color per ``preview.vertex_panel`` id) so
    seams between panels are visible.
    """
    plt = _require_matplotlib()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401 (registers 3d proj)

    if ax is None:
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(projection="3d")

    # World coordinates are y-up (gravity is -y), but matplotlib's elev/azim
    # treat *its* z-axis as vertical -- plotted as-is, no elev/azim choice
    # makes world "up" read as screen-up, which makes a tall, roughly-
    # cylindrical garment look like a squashed blob no matter the angle. Swap
    # world y into matplotlib's z slot (and relabel) so the default-style
    # viewing angles behave the way they look like they should.
    def screen(v):
        return v[:, 0], v[:, 2], v[:, 1]

    body_v = preview.body.mesh.vertices
    bx, by, bz = screen(body_v)
    for face in preview.body.mesh.faces[::3]:      # thin out for a lighter body sketch
        loop = list(face) + [face[0]]
        ax.plot(bx[loop], by[loop], bz[loop], color="lightgray", linewidth=0.4, zorder=1)

    vertices, faces, vertex_panel = preview.vertices, preview.faces, preview.vertex_panel
    gx, gy, gz = screen(vertices)
    face_panel = vertex_panel[faces[:, 0]]
    for panel_id in np.unique(face_panel):
        color = _PANEL_PALETTE[int(panel_id) % len(_PANEL_PALETTE)]
        for face in faces[face_panel == panel_id]:
            loop = list(face) + [face[0]]
            ax.plot(gx[loop], gy[loop], gz[loop], color=color, linewidth=0.9, zorder=2)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("height, world y (m)")
    ax.view_init(elev=elev, azim=azim)
    try:
        # True-to-scale aspect (not a fixed ratio): an equal box would make a
        # tall, narrow body look squashed and misleadingly "clumped."
        all_v = np.vstack([vertices, body_v])
        extents = all_v.max(axis=0) - all_v.min(axis=0)
        extents = np.where(extents < 1e-6, 1e-6, extents)
        ax.set_box_aspect((extents[0], extents[2], extents[1]))
    except AttributeError:
        pass
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return ax


def plot_loss_curve(losses: list[float], *, ax=None, show: bool = True, label: Optional[str] = None):
    """Plot a fitting loss curve (e.g. ``FitResult.losses``)."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(losses, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    if label:
        ax.legend()
    if show:
        plt.show()
    return ax
