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
