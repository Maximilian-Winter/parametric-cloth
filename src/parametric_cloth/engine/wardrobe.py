"""Equipment / wardrobe management: equip, unequip, save and load loadouts.

A simple slot-based model with per-slot layering, enough to drive layered
rendering via :mod:`parametric_cloth.engine.masking`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Default render layer per slot (higher = outer). Outerwear sits over tops, etc.
SLOT_LAYERS: dict[str, int] = {
    "underwear": 0,
    "bottom": 1,
    "top": 2,
    "dress": 2,
    "outerwear": 3,
    "accessory": 4,
}


@dataclass
class Wardrobe:
    """Currently equipped garments, keyed by slot."""

    equipped: dict[str, str] = field(default_factory=dict)

    def equip(self, slot: str, garment_id: str) -> None:
        if slot not in SLOT_LAYERS:
            raise ValueError(f"unknown slot '{slot}'; known: {sorted(SLOT_LAYERS)}")
        self.equipped[slot] = garment_id

    def unequip(self, slot: str) -> str | None:
        return self.equipped.pop(slot, None)

    def layer_of(self, slot: str) -> int:
        return SLOT_LAYERS[slot]

    def ordered_garments(self) -> list[tuple[str, str, int]]:
        """(slot, garment_id, layer) sorted from innermost to outermost."""
        items = [(slot, gid, SLOT_LAYERS[slot]) for slot, gid in self.equipped.items()]
        return sorted(items, key=lambda t: t[2])

    def loadout(self) -> dict[str, str]:
        return dict(self.equipped)

    def apply_loadout(self, loadout: dict[str, str]) -> None:
        for slot in loadout:
            if slot not in SLOT_LAYERS:
                raise ValueError(f"unknown slot '{slot}' in loadout")
        self.equipped = dict(loadout)

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.loadout(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Wardrobe":
        with open(path, encoding="utf-8") as fh:
            loadout = json.load(fh)
        w = cls()
        w.apply_loadout(loadout)
        return w
