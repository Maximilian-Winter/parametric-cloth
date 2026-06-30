"""A simplified differentiable silhouette renderer (Module 10).

Stands in for nvdiffrast / PyTorch3D: a soft point-splat renderer with an
analytic, hand-derived gradient, in plain NumPy. It is not a full
differentiable *rasterizer* (no triangle coverage, occlusion, or shading) --
each 3D point is projected orthographically and splatted as a Gaussian blob,
and per-pixel coverage is the soft-OR of all splats. This is enough to fit a
mesh's rough silhouette and demonstrates the differentiable-rendering half of
the fitting loop with zero GPU/framework dependency; swap in
nvdiffrast/PyTorch3D for production-quality results (triangle coverage,
anti-aliasing, occlusion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_AXES = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}


@dataclass
class OrthographicCamera:
    """Maps 3D points to a 2D pixel grid by dropping one axis (orthographic)."""

    axis: str = "z"                          # the axis to project away (view direction)
    resolution: tuple[int, int] = (64, 64)    # (H, W)
    extent: float = 1.5                       # world-space half-width of the image plane (m)

    def axes(self) -> tuple[int, int]:
        """(a0, a1): a0 maps to columns/W, a1 maps to rows/H."""
        return _AXES[self.axis]

    def project(self, points: np.ndarray) -> np.ndarray:
        """World points (N,3) -> pixel coords (N,2) as (row, col), float."""
        a0, a1 = self.axes()
        h, w = self.resolution
        col = (points[:, a0] + self.extent) / (2 * self.extent) * (w - 1)
        row = (points[:, a1] + self.extent) / (2 * self.extent) * (h - 1)
        return np.stack([row, col], axis=1)

    def jacobian_scale(self) -> tuple[float, float]:
        """(d(row)/d(a1 coord), d(col)/d(a0 coord)) -- constant scale factors."""
        h, w = self.resolution
        return (h - 1) / (2 * self.extent), (w - 1) / (2 * self.extent)


def _pixel_grid(resolution: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = resolution
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return rows.astype(float), cols.astype(float)


@dataclass
class SoftSplatRenderer:
    """Differentiable orthographic soft-point-splat silhouette renderer."""

    camera: OrthographicCamera
    sigma_px: float = 1.5

    def render(self, points: np.ndarray) -> np.ndarray:
        """points (N,3) -> coverage image (H, W) in [0, 1)."""
        occupancy = self._occupancy(points)
        return 1.0 - np.exp(-occupancy)

    def _occupancy(self, points: np.ndarray) -> np.ndarray:
        pix = self.camera.project(points)              # (N,2) = (row,col)
        rows, cols = _pixel_grid(self.camera.resolution)
        dr = rows[..., None] - pix[:, 0]                # (H,W,N)
        dc = cols[..., None] - pix[:, 1]
        r2 = dr * dr + dc * dc
        return np.exp(-r2 / (2 * self.sigma_px ** 2)).sum(axis=2)

    def render_and_loss(self, points: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
        """Forward render + MSE loss + analytic gradient w.r.t. ``points``."""
        points = np.asarray(points, dtype=float)
        h, w = self.camera.resolution
        pix = self.camera.project(points)               # (N,2)
        rows, cols = _pixel_grid(self.camera.resolution)
        dr = rows[..., None] - pix[:, 0]                 # (H,W,N)
        dc = cols[..., None] - pix[:, 1]
        r2 = dr * dr + dc * dc
        weights = np.exp(-r2 / (2 * self.sigma_px ** 2))  # (H,W,N)
        occupancy = weights.sum(axis=2)
        coverage = 1.0 - np.exp(-occupancy)

        diff = coverage - np.asarray(target, dtype=float)
        loss = float(np.mean(diff ** 2))

        n_pixels = h * w
        dL_dcoverage = 2.0 * diff / n_pixels
        dL_doccupancy = dL_dcoverage * np.exp(-occupancy)              # (H,W)
        dL_dr2 = dL_doccupancy[..., None] * weights * (-1.0 / (2 * self.sigma_px ** 2))

        dL_drow = np.sum(dL_dr2 * (-2.0 * dr), axis=(0, 1))            # (N,)
        dL_dcol = np.sum(dL_dr2 * (-2.0 * dc), axis=(0, 1))            # (N,)

        scale_row, scale_col = self.camera.jacobian_scale()
        a0, a1 = self.camera.axes()
        dL_dpoints = np.zeros_like(points)
        dL_dpoints[:, a1] += dL_drow * scale_row
        dL_dpoints[:, a0] += dL_dcol * scale_col
        return loss, dL_dpoints
