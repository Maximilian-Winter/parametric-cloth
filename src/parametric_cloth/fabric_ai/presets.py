"""Extended fabric preset table with fuzzy natural-language lookup.

Module 1's six base presets cover the common case but break down for variations
within a type ("heavy brushed cotton twill" vs "lightweight silk charmeuse").
This table adds named sub-variants and a fuzzy matcher so a free-text fabric
description resolves to the closest known preset.

Pure stdlib (``difflib``) -- no ML dependency for the common case. The learned
regressor in :mod:`parametric_cloth.fabric_ai.predictor` is for descriptions that
don't match anything here.
"""

from __future__ import annotations

import difflib

from ..fabric import FabricProperties

# Extends FABRIC_PRESETS (Module 1) with named sub-variants. Values fill in
# FabricProperties fields not given here from FabricProperties defaults.
FABRIC_PRESETS_EXTENDED: dict[str, dict[str, float]] = {
    # Cotton variants
    "cotton_jersey":        {"mass_per_area": 0.18, "stiffness": 10.0, "bending": 0.3,  "damping": 4.0,  "friction": 0.5, "stretch_limit": 1.10},
    "cotton_twill":         {"mass_per_area": 0.25, "stiffness": 16.0, "bending": 1.2,  "damping": 6.0,  "friction": 0.5, "stretch_limit": 1.04},
    "cotton_canvas":        {"mass_per_area": 0.35, "stiffness": 22.0, "bending": 3.0,  "damping": 8.0,  "friction": 0.6, "stretch_limit": 1.02},
    "cotton_voile":         {"mass_per_area": 0.08, "stiffness": 6.0,  "bending": 0.1,  "damping": 2.0,  "friction": 0.4, "stretch_limit": 1.08},

    # Silk variants
    "silk_charmeuse":       {"mass_per_area": 0.08, "stiffness": 6.0,  "bending": 0.05, "damping": 1.5,  "friction": 0.25, "stretch_limit": 1.10},
    "silk_dupioni":         {"mass_per_area": 0.15, "stiffness": 9.0,  "bending": 0.8,  "damping": 3.0,  "friction": 0.35, "stretch_limit": 1.05},
    "silk_organza":         {"mass_per_area": 0.05, "stiffness": 5.0,  "bending": 0.3,  "damping": 1.0,  "friction": 0.3, "stretch_limit": 1.07},

    # Denim variants
    "denim_lightweight":    {"mass_per_area": 0.30, "stiffness": 24.0, "bending": 2.5,  "damping": 6.0,  "friction": 0.6, "stretch_limit": 1.03},
    "denim_heavyweight":    {"mass_per_area": 0.55, "stiffness": 36.0, "bending": 6.0,  "damping": 10.0, "friction": 0.65, "stretch_limit": 1.01},
    "denim_stretch":        {"mass_per_area": 0.35, "stiffness": 18.0, "bending": 2.0,  "damping": 5.0,  "friction": 0.55, "stretch_limit": 1.15},

    # Wool variants
    "wool_flannel":         {"mass_per_area": 0.25, "stiffness": 16.0, "bending": 1.0,  "damping": 6.0,  "friction": 0.5, "stretch_limit": 1.05},
    "wool_tweed":           {"mass_per_area": 0.40, "stiffness": 26.0, "bending": 3.5,  "damping": 9.0,  "friction": 0.55, "stretch_limit": 1.02},
    "wool_crepe":           {"mass_per_area": 0.20, "stiffness": 14.0, "bending": 0.5,  "damping": 4.0,  "friction": 0.45, "stretch_limit": 1.06},

    # Synthetic
    "polyester_chiffon":    {"mass_per_area": 0.06, "stiffness": 5.0,  "bending": 0.08, "damping": 1.5,  "friction": 0.3, "stretch_limit": 1.10},
    "nylon_ripstop":        {"mass_per_area": 0.07, "stiffness": 7.0,  "bending": 0.2,  "damping": 2.0,  "friction": 0.4, "stretch_limit": 1.08},
    "neoprene":             {"mass_per_area": 0.60, "stiffness": 40.0, "bending": 8.0,  "damping": 12.0, "friction": 0.7, "stretch_limit": 1.20},

    # Historical / specialty
    "linen_heavy":          {"mass_per_area": 0.30, "stiffness": 18.0, "bending": 1.5,  "damping": 6.0,  "friction": 0.45, "stretch_limit": 1.04},
    "velvet":               {"mass_per_area": 0.35, "stiffness": 17.0, "bending": 1.8,  "damping": 7.0,  "friction": 0.65, "stretch_limit": 1.05},
    "burlap":               {"mass_per_area": 0.40, "stiffness": 28.0, "bending": 4.0,  "damping": 9.0,  "friction": 0.7, "stretch_limit": 1.01},
    "chainmail_cloth":      {"mass_per_area": 2.50, "stiffness": 80.0, "bending": 0.1,  "damping": 15.0, "friction": 0.4, "stretch_limit": 1.001},
}

# Free-text aliases that should resolve to a specific sub-variant even when they
# don't fuzzy-match the key itself closely (e.g. "workwear" doesn't look like
# "cotton_canvas" by edit distance).
_ALIASES: dict[str, str] = {
    "workwear": "cotton_canvas",
    "denim": "denim_lightweight",
    "jeans": "denim_lightweight",
    "raw denim": "denim_heavyweight",
    "silk": "silk_charmeuse",
    "chainmail": "chainmail_cloth",
    "chain mail": "chainmail_cloth",
    "tshirt cotton": "cotton_jersey",
    "t-shirt cotton": "cotton_jersey",
}


def fabric_properties_for(name: str, **overrides) -> FabricProperties:
    """Build :class:`FabricProperties` for an extended preset name."""
    try:
        values = FABRIC_PRESETS_EXTENDED[name]
    except KeyError:
        raise KeyError(
            f"unknown extended fabric preset '{name}'; "
            f"known: {sorted(FABRIC_PRESETS_EXTENDED)}"
        ) from None
    merged = {**values, **overrides}
    return FabricProperties(**merged)


def find_closest_preset(description: str, *, cutoff: float = 0.4) -> str | None:
    """Fuzzy-match a free-text fabric description to the closest preset name.

    Tries, in order: an exact alias, an exact preset-name match, then
    ``difflib`` sequence-matching against preset names and their underscore-
    split tokens (so "heavy cotton" matches "cotton_canvas" via "cotton" +
    "heavy"-ish weight cues). Returns ``None`` if nothing clears ``cutoff``.
    """
    text = description.strip().lower()
    if not text:
        return None

    if text in _ALIASES:
        return _ALIASES[text]
    if text in FABRIC_PRESETS_EXTENDED:
        return text

    candidates = list(FABRIC_PRESETS_EXTENDED)
    # Score against both the raw key and a space-joined version, since
    # "cotton_canvas" reads more like "cotton canvas" to a human description.
    spaced = {c: c.replace("_", " ") for c in candidates}

    best_name, best_score = None, 0.0
    for name in candidates:
        for form in (name, spaced[name]):
            score = difflib.SequenceMatcher(None, text, form).ratio()
            if score > best_score:
                best_name, best_score = name, score

    # Token overlap as a tiebreaker / boost: shared words count for a lot.
    text_tokens = set(text.replace("-", " ").split())
    for name in candidates:
        tokens = set(spaced[name].split())
        overlap = len(text_tokens & tokens) / max(len(tokens), 1)
        boosted = 0.5 * overlap + 0.5 * difflib.SequenceMatcher(
            None, text, spaced[name]
        ).ratio()
        if boosted > best_score:
            best_name, best_score = name, boosted

    return best_name if best_score >= cutoff else None


def fabric_from_description(description: str, **overrides) -> FabricProperties:
    """Resolve a free-text fabric description to :class:`FabricProperties`.

    Tries the extended preset table (with fuzzy matching) first; raises
    ``LookupError`` if nothing matches closely enough. For genuinely novel
    descriptions, fall back to :class:`~parametric_cloth.fabric_ai.predictor.FabricPredictor`
    (needs ``sentence-transformers`` + ``torch``).
    """
    match = find_closest_preset(description)
    if match is None:
        raise LookupError(
            f"no preset close enough to '{description}'; try "
            f"FabricPredictor for free-form descriptions"
        )
    return fabric_properties_for(match, **overrides)
