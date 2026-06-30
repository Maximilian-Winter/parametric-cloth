"""Module 10: Differentiable Physics Fitting.

Automatically refines pattern parameters to match a target garment appearance
(silhouette or 3D scan) via gradient descent through a draping simulation.

No production differentiable-physics/-rendering framework (Warp, Taichi,
DiffCloth, nvdiffrast, PyTorch3D) is available in this environment, so this
package implements its own small differentiable mass-spring simulator and
soft-splat silhouette renderer in plain NumPy, with hand-derived (not
autodiff-framework-generated) reverse-mode gradients -- verified against
central finite differences in the test suite. Swap in a production backend for
higher fidelity; the ``DifferentiableClothFitter`` orchestration and the
pattern <-> simulation plumbing stay the same.
"""

from __future__ import annotations

from .chamfer import chamfer_distance, chamfer_distance_and_grad
from .fitter import DifferentiableClothFitter, FitResult, resolve_pin_mask
from .mass_spring import DifferentiableMassSpring, SpringTopology
from .optim import Adam
from .rendering import OrthographicCamera, SoftSplatRenderer

__all__ = [
    "DifferentiableMassSpring",
    "SpringTopology",
    "chamfer_distance",
    "chamfer_distance_and_grad",
    "OrthographicCamera",
    "SoftSplatRenderer",
    "Adam",
    "DifferentiableClothFitter",
    "FitResult",
    "resolve_pin_mask",
]
