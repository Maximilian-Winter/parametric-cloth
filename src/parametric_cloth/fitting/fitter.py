"""DifferentiableClothFitter (Module 10): refine a pattern to match a target.

Optimizes a single pattern piece's 2D vertex positions so that, after draping
through the differentiable mass-spring simulator
(:mod:`~parametric_cloth.fitting.mass_spring`), the result matches a target
silhouette or 3D scan. Gradients flow end-to-end -- loss -> renderer/chamfer ->
simulator -> pattern vertices -- via the hand-derived analytic gradients in
this package; no autodiff framework is needed.

Scope: this fits one ``PatternPiece`` at a time (the draped shape of a single
flat panel), not a multi-panel garment with avatar placement. Modeling contact
against a real body surface during fitting would route through Modules 2/3's
(non-differentiable) Blender solver, which is exactly why this module needs its
own lightweight differentiable simulator. Fitting a full garment means calling
this once per piece and reassembling the result (``fit_to_silhouette``/
``fit_to_3d_scan`` do this for a named piece of a ``GarmentDefinition``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..pattern import GarmentDefinition, PatternPiece, PatternVertex
from ..simulation.tessellate import tessellate_piece
from .chamfer import chamfer_distance_and_grad
from .mass_spring import (
    DEFAULT_DAMPING,
    DEFAULT_DT,
    DEFAULT_GRAVITY,
    DifferentiableMassSpring,
    SpringTopology,
)
from .optim import Adam
from .rendering import OrthographicCamera, SoftSplatRenderer

CM_TO_M = 0.01


def resolve_pin_mask(vertices2d: np.ndarray, strategy) -> np.ndarray:
    """Resolve a pin strategy to a boolean mask over ``vertices2d``.

    ``strategy`` is ``"min_y"``/``"max_y"`` (pin the vertices at the pattern's
    lowest/highest y, e.g. the waist edge of a skirt panel vs. the shoulder
    edge of a top), or an explicit boolean array / index list.
    """
    if isinstance(strategy, str):
        if strategy not in ("min_y", "max_y"):
            raise ValueError(f"unknown pin strategy '{strategy}'")
        y = vertices2d[:, 1]
        edge = y.min() if strategy == "min_y" else y.max()
        return np.isclose(y, edge, atol=1e-6)

    idx = np.asarray(strategy)
    if idx.dtype == bool:
        return idx
    mask = np.zeros(len(vertices2d), dtype=bool)
    mask[idx] = True
    return mask


@dataclass
class FitResult:
    piece: PatternPiece
    losses: list[float]
    n_iterations: int


def _piece_with_vertices(piece: PatternPiece, vertices2d: np.ndarray) -> PatternPiece:
    return PatternPiece(
        name=piece.name,
        vertices=[PatternVertex(float(x), float(y)) for x, y in vertices2d],
        subdivisions=piece.subdivisions,
        placement=piece.placement,
        fabric=piece.fabric,
        seam_allowance=piece.seam_allowance,
    )


def _select_piece(garment: GarmentDefinition, piece_name: Optional[str]) -> PatternPiece:
    if piece_name is not None:
        piece = garment.piece(piece_name)
        if piece is None:
            raise KeyError(f"garment '{garment.name}' has no piece '{piece_name}'")
        return piece
    if len(garment.pieces) != 1:
        raise ValueError(
            f"garment '{garment.name}' has {len(garment.pieces)} pieces; "
            f"pass piece_name= to select one"
        )
    return garment.pieces[0]


def _replace_piece(garment: GarmentDefinition, new_piece: PatternPiece) -> GarmentDefinition:
    pieces = [new_piece if p.name == new_piece.name else p for p in garment.pieces]
    return GarmentDefinition(
        name=garment.name, pieces=pieces, seams=garment.seams,
        simulation_frames=garment.simulation_frames,
        simulation_substeps=garment.simulation_substeps, gravity=garment.gravity,
    )


@dataclass
class DifferentiableClothFitter:
    """Fits a pattern piece's vertex positions to a target via gradient descent."""

    levels: int = 0                  # 0 keeps every simulated particle a polygon corner
    n_sim_steps: int = 30
    dt: float = DEFAULT_DT
    damping: float = DEFAULT_DAMPING
    gravity: np.ndarray = field(default_factory=lambda: DEFAULT_GRAVITY.copy())
    stiffness: float = 50.0
    regularization: float = 0.01

    def _build_sim(self, piece: PatternPiece, pin_strategy):
        mesh = tessellate_piece(piece, levels=self.levels)
        pinned = resolve_pin_mask(mesh.vertices, pin_strategy)
        topology = SpringTopology.from_faces(mesh.faces, stiffness=self.stiffness)
        sim = DifferentiableMassSpring(
            topology, pinned, n_steps=self.n_sim_steps, dt=self.dt,
            damping=self.damping, gravity=self.gravity,
        )
        return mesh, sim

    def _regularize(self, params2d: np.ndarray, initial2d: np.ndarray):
        diff = params2d - initial2d
        loss = self.regularization * float(np.mean(diff ** 2))
        grad = self.regularization * 2.0 * diff / diff.size
        return loss, grad

    def _optimize(self, mesh, sim, params, initial, n_iterations, lr, loss_and_grad_fn):
        # Pinned vertices are the pattern's attachment edge (e.g. a waistband or
        # neckline): the simulator already holds them fixed *during* the drape,
        # but the optimizer must also leave their *position* alone, or the
        # anchor itself drifts every step -- not what "pin" should mean here.
        free = ~sim.pinned
        optimizer = Adam(params.shape, lr=lr)
        losses = []
        for _ in range(n_iterations):
            x0 = np.zeros((mesh.n_vertices, 3))
            x0[:, :2] = params * CM_TO_M
            x_final = sim.forward(x0)

            target_loss, dL_dXfinal = loss_and_grad_fn(x_final)
            reg_loss, reg_grad = self._regularize(params, initial)
            losses.append(target_loss + reg_loss)

            dL_dX0 = sim.backward(dL_dXfinal)
            grad = dL_dX0[:, :2] * CM_TO_M + reg_grad
            grad[~free] = 0.0
            params = optimizer.step(params, grad)
            params[~free] = initial[~free]
        return params, losses

    def fit_piece_to_silhouette(
        self, piece: PatternPiece, target_silhouette: np.ndarray,
        *, camera: Optional[OrthographicCamera] = None, sigma_px: float = 1.5,
        pin: str = "min_y", n_iterations: int = 100, lr: float = 0.5,
    ) -> FitResult:
        """``sigma_px`` must match whatever rendered ``target_silhouette`` (the
        Gaussian splat width) -- a mismatch distorts the loss landscape, since
        the optimized mesh's own splats and the target's are different scales.
        """
        mesh, sim = self._build_sim(piece, pin)
        renderer = SoftSplatRenderer(camera or OrthographicCamera(), sigma_px=sigma_px)
        sim_params, losses = self._optimize(
            mesh, sim, mesh.vertices.copy(), mesh.vertices.copy(), n_iterations, lr,
            lambda x_final: renderer.render_and_loss(x_final, target_silhouette),
        )
        return FitResult(
            piece=_piece_with_vertices(piece, sim_params[:mesh.n_corners]),
            losses=losses, n_iterations=n_iterations,
        )

    def fit_piece_to_3d_scan(
        self, piece: PatternPiece, target_mesh: np.ndarray,
        *, pin: str = "min_y", n_iterations: int = 100, lr: float = 0.5,
    ) -> FitResult:
        mesh, sim = self._build_sim(piece, pin)
        sim_params, losses = self._optimize(
            mesh, sim, mesh.vertices.copy(), mesh.vertices.copy(), n_iterations, lr,
            lambda x_final: chamfer_distance_and_grad(x_final, target_mesh),
        )
        return FitResult(
            piece=_piece_with_vertices(piece, sim_params[:mesh.n_corners]),
            losses=losses, n_iterations=n_iterations,
        )

    def fit_to_silhouette(
        self, garment: GarmentDefinition, target_silhouette: np.ndarray,
        *, piece_name: Optional[str] = None, **kwargs,
    ) -> GarmentDefinition:
        """Fit one named piece of a garment (defaults to its only piece)."""
        piece = _select_piece(garment, piece_name)
        result = self.fit_piece_to_silhouette(piece, target_silhouette, **kwargs)
        return _replace_piece(garment, result.piece)

    def fit_to_3d_scan(
        self, garment: GarmentDefinition, target_mesh: np.ndarray,
        *, piece_name: Optional[str] = None, **kwargs,
    ) -> GarmentDefinition:
        piece = _select_piece(garment, piece_name)
        result = self.fit_piece_to_3d_scan(piece, target_mesh, **kwargs)
        return _replace_piece(garment, result.piece)
