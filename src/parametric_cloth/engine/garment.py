"""A runtime garment: a deformer plus its coverage and texture customization.

Ties Module 6/5 deformation to Module 7 rendering concerns so the engine can,
each frame, deform the mesh and know which body regions it occludes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .deformer import DeformState
from .masking import GarmentCoverage
from .texture import TextureCustomization


@dataclass
class RuntimeGarment:
    garment_id: str
    deformer: object                  # PCADeformer | LearnedDeformer | ONNXDeformer
    regions: set[str] = field(default_factory=set)
    layer: int = 0
    texture: TextureCustomization = field(default_factory=TextureCustomization)

    def deform(self, state: DeformState) -> np.ndarray:
        """Deformed garment vertices (V, 3) for the given frame state."""
        return self.deformer.deform(state)

    def coverage(self) -> GarmentCoverage:
        return GarmentCoverage(self.garment_id, set(self.regions), self.layer)
