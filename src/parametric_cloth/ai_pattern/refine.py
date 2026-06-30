"""Rule-based pattern refinement from simple text feedback.

The design's ``refine()`` workflow ("make the sleeves wider and shorten the hem
by 5cm") assumes an LLM/VLM backend (ChatGarment) that understands open-ended
feedback. No such backend is available here, so this module implements a
deterministic fallback: a small regex grammar over
``<action> <target> [by <amount>cm]`` directives, applied as geometric scaling
to pattern pieces whose name matches the target keyword.

This is intentionally limited -- it is a fallback for simple, explicit numeric
edits, not a substitute for real natural-language understanding. Unmatched or
ambiguous feedback simply produces no directives (the garment is returned
unchanged), so it never guesses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..pattern import GarmentDefinition, PatternPiece, PatternVertex

# Canonical action -> ((axis, sign), recognized words). widen/narrow scale
# x-extent; lengthen/shorten scale y-extent (patterns are y-up: hem is +y from
# waist). Both verb ("shorten") and adjective ("shorter") forms are recognized,
# since English expresses the same edit either word order.
_ACTION_GROUPS: dict[str, tuple[tuple[str, int], list[str]]] = {
    "widen":    (("x", 1),  ["widen", "broaden", "wider", "broader", "bigger"]),
    "narrow":   (("x", -1), ["narrow", "slim", "narrower", "slimmer", "smaller"]),
    "lengthen": (("y", 1),  ["lengthen", "extend", "longer"]),
    "shorten":  (("y", -1), ["shorten", "shrink", "shorter"]),
}
_ACTION_AXIS = {canon: axis_sign for canon, (axis_sign, _) in _ACTION_GROUPS.items()}
_ACTION_WORD_TO_CANON = {
    word: canon for canon, (_, words) in _ACTION_GROUPS.items() for word in words
}

# Canonical target -> (piece-name filter, recognized words). Filters are
# matched as a substring against piece names, except the two special tokens:
# "BODY" = every piece except sleeves ("the hem" means torso length, not
# sleeve length), "ALL" = literally every piece.
_TARGET_GROUPS: dict[str, tuple[str, list[str]]] = {
    "hem":     ("BODY", ["hem", "length"]),
    "garment": ("ALL", ["garment", "all"]),
    "sleeve":  ("sleeve", ["sleeve", "sleeves"]),
    "panel":   ("panel", ["skirt", "panel", "panels"]),
    "front":   ("front", ["body", "front"]),
    "back":    ("back", ["back"]),
}
_TARGET_FILTER = {canon: filt for canon, (filt, _) in _TARGET_GROUPS.items()}
_TARGET_WORD_TO_CANON = {
    word: canon for canon, (_, words) in _TARGET_GROUPS.items() for word in words
}

_DEFAULT_AMOUNT_CM = 2.0
_ACTIONS = "|".join(_ACTION_WORD_TO_CANON)
_TARGETS = "|".join(_TARGET_WORD_TO_CANON)

# Verb-first: "shorten the hem by 5cm". Adjective-after: "sleeves wider by 5cm".
_PREFIX_RE = re.compile(
    rf"\b(?P<action>{_ACTIONS})\b(?:\s+the)?\s+(?P<target>{_TARGETS})\b"
    rf"(?:\s+by\s+(?P<amount>[\d.]+)\s*cm)?",
    re.IGNORECASE,
)
_POSTFIX_RE = re.compile(
    rf"\b(?:the\s+)?(?P<target>{_TARGETS})\b\s+(?P<action>{_ACTIONS})\b"
    rf"(?:\s+by\s+(?P<amount>[\d.]+)\s*cm)?",
    re.IGNORECASE,
)


@dataclass
class RefineDirective:
    action: str            # canonical: widen | narrow | lengthen | shorten
    target: str             # canonical: hem | sleeve | panel | front | back
    amount_cm: float

    @property
    def axis_and_sign(self) -> tuple[str, int]:
        return _ACTION_AXIS[self.action]

    @property
    def piece_filter(self) -> str:
        return _TARGET_FILTER[self.target]


def parse_feedback(feedback: str) -> list[RefineDirective]:
    """Extract recognized directives from text, in either word order:
    verb-first ("shorten the hem by 5cm") or adjective-after ("sleeves wider").

    Returns an empty list if nothing matches -- callers should treat that as
    "no automatic edit possible" rather than guessing.
    """
    text = feedback.lower()
    matches = list(_PREFIX_RE.finditer(text)) + list(_POSTFIX_RE.finditer(text))
    matches.sort(key=lambda m: m.start())

    directives = []
    claimed: list[tuple[int, int]] = []
    for m in matches:
        if any(m.start() < end and start < m.end() for start, end in claimed):
            continue                                  # overlaps an earlier match
        claimed.append((m.start(), m.end()))
        amount = float(m.group("amount")) if m.group("amount") else _DEFAULT_AMOUNT_CM
        directives.append(RefineDirective(
            action=_ACTION_WORD_TO_CANON[m.group("action")],
            target=_TARGET_WORD_TO_CANON[m.group("target")],
            amount_cm=amount,
        ))
    return directives


def _scale_piece(piece: PatternPiece, axis: str, delta_cm: float) -> PatternPiece:
    """Scale a piece's extent along ``axis`` by adding ``delta_cm`` to its span.

    Vertices are scaled about the piece's minimum on that axis (e.g. the waist
    edge stays put when lengthening a hem), preserving silhouette shape.
    """
    coord = (lambda v: v.x) if axis == "x" else (lambda v: v.y)
    lo = min(coord(v) for v in piece.vertices)
    span = max(coord(v) for v in piece.vertices) - lo
    if span <= 1e-9:
        return piece
    factor = (span + delta_cm) / span

    new_vertices = []
    for v in piece.vertices:
        if axis == "x":
            new_vertices.append(PatternVertex(lo + (v.x - lo) * factor, v.y))
        else:
            new_vertices.append(PatternVertex(v.x, lo + (v.y - lo) * factor))

    return PatternPiece(
        name=piece.name, vertices=new_vertices, subdivisions=piece.subdivisions,
        placement=piece.placement, fabric=piece.fabric,
        seam_allowance=piece.seam_allowance,
    )


def _matches(piece_name: str, target: str) -> bool:
    name = piece_name.lower()
    if target == "ALL":
        return True
    if target == "BODY":
        return "sleeve" not in name
    return target in name


def apply_directive(garment: GarmentDefinition, directive: RefineDirective) -> GarmentDefinition:
    """Apply one directive to every matching piece, returning a new garment."""
    axis, sign = directive.axis_and_sign
    delta = sign * directive.amount_cm
    target = directive.piece_filter

    new_pieces = [
        _scale_piece(p, axis, delta) if _matches(p.name, target) else p
        for p in garment.pieces
    ]
    return GarmentDefinition(
        name=garment.name, pieces=new_pieces, seams=garment.seams,
        simulation_frames=garment.simulation_frames,
        simulation_substeps=garment.simulation_substeps,
        gravity=garment.gravity,
    )


def rule_based_refine(garment: GarmentDefinition, feedback: str) -> GarmentDefinition:
    """Apply every recognized directive in ``feedback`` to ``garment`` in order."""
    garment_out = garment
    for directive in parse_feedback(feedback):
        garment_out = apply_directive(garment_out, directive)
    return garment_out
