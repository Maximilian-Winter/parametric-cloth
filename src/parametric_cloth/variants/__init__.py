"""Module 5: Variant System.

Compress a family of garment variants into one PCA basis plus a few coefficients
per variant, giving continuous parameter variation at the storage cost of a
handful of shapes. Pure numpy (PCA via SVD); the only Blender dependency is the
optional blend-shape exporter.
"""

from __future__ import annotations

from .blendshapes import (
    BlendShapeTarget,
    blend_shape_targets,
    coefficients_to_weights,
    export_pca_as_blend_shapes,
)
from .library import VariantLibrary, build_variant_library
from .pca import PCABasis, build_pca_basis
from .sampling import generate_sample_points, latin_hypercube

__all__ = [
    "PCABasis",
    "build_pca_basis",
    "generate_sample_points",
    "latin_hypercube",
    "VariantLibrary",
    "build_variant_library",
    "BlendShapeTarget",
    "blend_shape_targets",
    "coefficients_to_weights",
    "export_pca_as_blend_shapes",
]
