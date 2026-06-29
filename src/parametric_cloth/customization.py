"""Player-facing customization mapping.

Translates human-friendly customization options (slim/regular/loose, crew/v-neck,
...) into the numeric pattern parameters consumed by the template functions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fabric import FabricType
from .pattern import GarmentDefinition
from .templates import create_tshirt


@dataclass
class TShirtCustomization:
    fit: str = "regular"           # slim, regular, loose, oversized
    length: str = "regular"        # cropped, regular, longline
    sleeve_length: str = "short"   # cap, short, three_quarter, long
    neckline: str = "crew"         # crew, v_neck, scoop, boat
    fabric: str = "cotton"

    _EASE = {"slim": 1.02, "regular": 1.1, "loose": 1.25, "oversized": 1.4}
    _LENGTH = {"cropped": 50, "regular": 65, "longline": 80}
    _SLEEVE = {"cap": 8, "short": 20, "three_quarter": 40, "long": 55}
    _NECK = {"crew": 4, "v_neck": 15, "scoop": 10, "boat": 2}

    def to_pattern_params(self) -> dict:
        def _lookup(table: dict, key: str, field: str):
            try:
                return table[key]
            except KeyError:
                raise ValueError(
                    f"unknown {field} '{key}'; expected one of {sorted(table)}"
                ) from None

        return {
            "ease": _lookup(self._EASE, self.fit, "fit"),
            "length": _lookup(self._LENGTH, self.length, "length"),
            "sleeve_length": _lookup(self._SLEEVE, self.sleeve_length, "sleeve_length"),
            "neck_depth_front": _lookup(self._NECK, self.neckline, "neckline"),
            "fabric": FabricType(self.fabric),
        }

    def build(self) -> GarmentDefinition:
        """Produce the garment definition for this customization."""
        return create_tshirt(**self.to_pattern_params())
