"""Module 7: Game Engine Integration.

Runtime garment customization: two deformation paths (PCA blend shapes or learned
pose-conditioned offsets), body masking for layering, texture customization, and
wardrobe management. Pure numpy -- testable without an engine; the ONNX deformer
is the only piece that needs an external runtime (lazy import).
"""

from __future__ import annotations

from .deformer import (
    DeformState,
    LearnedDeformer,
    ONNXDeformer,
    PCADeformer,
)
from .garment import RuntimeGarment
from .masking import (
    GarmentCoverage,
    covered_regions,
    hidden_garment_regions,
    resolve_visible_regions,
    visible_body_faces,
)
from .profiling import BenchmarkResult, benchmark
from .texture import TextureCustomization, apply_customization
from .wardrobe import SLOT_LAYERS, Wardrobe

__all__ = [
    "DeformState",
    "PCADeformer",
    "LearnedDeformer",
    "ONNXDeformer",
    "RuntimeGarment",
    "GarmentCoverage",
    "visible_body_faces",
    "covered_regions",
    "resolve_visible_regions",
    "hidden_garment_regions",
    "TextureCustomization",
    "apply_customization",
    "Wardrobe",
    "SLOT_LAYERS",
    "benchmark",
    "BenchmarkResult",
]
