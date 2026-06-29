"""Real-time texture customization.

Color and pattern changes are independent of the mesh pipeline and apply per
frame. Because UVs come from the 2D sewing pattern (Module 4), textures painted
on the flat pattern wrap correctly on the garment.

Pure numpy image compositing -- testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TextureCustomization:
    base_color: tuple[float, float, float] = (1.0, 1.0, 1.0)   # multiply, 0..1
    pattern: np.ndarray | None = None        # optional overlay image HxWx3, 0..1
    pattern_opacity: float = 0.0             # 0..1
    roughness: float = 0.6
    metallic: float = 0.0
    normal_intensity: float = 1.0

    def to_material_dict(self) -> dict:
        return {
            "base_color": list(self.base_color),
            "roughness": self.roughness,
            "metallic": self.metallic,
            "normal_intensity": self.normal_intensity,
        }


def apply_customization(base: np.ndarray, custom: TextureCustomization) -> np.ndarray:
    """Composite a base texture (H, W, 3 in [0,1]) with color and pattern overlay."""
    base = np.asarray(base, dtype=float)
    out = base * np.asarray(custom.base_color, dtype=float)

    if custom.pattern is not None and custom.pattern_opacity > 0.0:
        pattern = np.asarray(custom.pattern, dtype=float)
        if pattern.shape != base.shape:
            raise ValueError("pattern overlay must match the base texture shape")
        a = float(np.clip(custom.pattern_opacity, 0.0, 1.0))
        out = (1.0 - a) * out + a * pattern

    return np.clip(out, 0.0, 1.0)
