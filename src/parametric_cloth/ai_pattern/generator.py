"""Unified interface for AI-assisted pattern generation.

Three entry points -- photo, sketch, text -- each backed by a pretrained model
(SewFormer, ChatGarment, SewingLDM; see the design doc). None of those models or
their weights are available in this environment, so each backend is a lazy
import: calling ``from_photo``/``from_sketch``/``from_text`` without the
dependency installed raises ``ModuleNotFoundError`` with a clear message,
exactly like the other optional-dependency modules in this package.

The orchestration around a backend -- validating its raw output, converting
through :mod:`~parametric_cloth.ai_pattern.garmentcode`, surfacing geometry
errors -- is independent of which backend runs, so ``backend=`` accepts an
injected callable for testing without any model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..pattern import GarmentDefinition
from .garmentcode import garmentcode_to_definition
from .refine import rule_based_refine

# A backend maps raw input (image array / description string) to an evaluated
# GarmentCode-shaped dict (see garmentcode.garmentcode_to_definition).
PhotoBackend = Callable[[np.ndarray, Optional[np.ndarray]], dict]
TextBackend = Callable[[str], dict]


def _sewformer_backend(image: np.ndarray, body_shape: Optional[np.ndarray]) -> dict:
    """Reconstruct a sewing pattern from a photo via SewFormer (needs torch + weights)."""
    import torch  # noqa: F401
    raise ModuleNotFoundError(
        "SewFormer model weights are not available in this environment; "
        "inject a `backend=` callable for testing, or install/configure "
        "SewFormer and its pretrained weights for real inference."
    )


def _chatgarment_text_backend(description: str) -> dict:
    """Generate a pattern spec from text via ChatGarment (needs a VLM backend)."""
    raise ModuleNotFoundError(
        "ChatGarment is not available in this environment; inject a "
        "`backend=` callable for testing, or configure a ChatGarment endpoint."
    )


def _chatgarment_sketch_backend(sketch: np.ndarray, garment_type: Optional[str]) -> dict:
    """Generate a pattern spec from a sketch via ChatGarment/SewingLDM."""
    raise ModuleNotFoundError(
        "ChatGarment/SewingLDM are not available in this environment; inject a "
        "`backend=` callable for testing, or configure a sketch-to-pattern endpoint."
    )


@dataclass
class PatternGenerator:
    """Generates :class:`GarmentDefinition` patterns from photos, sketches, or text.

    Each backend defaults to the real (currently unavailable) model; pass
    ``backend=`` to ``from_photo``/``from_sketch``/``from_text`` to inject a
    stand-in -- e.g. in tests, or to wire up a self-hosted inference endpoint
    without modifying this class.
    """

    default_fabric: str = "cotton"

    def from_photo(
        self, image: np.ndarray, body_shape: Optional[np.ndarray] = None,
        *, backend: PhotoBackend = _sewformer_backend,
    ) -> GarmentDefinition:
        """Reconstruct a sewing pattern from a garment photograph (SewFormer)."""
        gc_json = backend(image, body_shape)
        return self._to_definition(gc_json)

    def from_sketch(
        self, sketch: np.ndarray, garment_type: Optional[str] = None,
        *, backend: Callable[[np.ndarray, Optional[str]], dict] = _chatgarment_sketch_backend,
    ) -> GarmentDefinition:
        """Generate a sewing pattern from a hand-drawn sketch (ChatGarment/SewingLDM)."""
        gc_json = backend(sketch, garment_type)
        return self._to_definition(gc_json)

    def from_text(
        self, description: str, body_shape: Optional[np.ndarray] = None,
        *, backend: TextBackend = _chatgarment_text_backend,
    ) -> GarmentDefinition:
        """Generate a sewing pattern from a text description (ChatGarment)."""
        gc_json = backend(description)
        return self._to_definition(gc_json)

    def refine(self, garment: GarmentDefinition, feedback: str) -> GarmentDefinition:
        """Refine a generated pattern from text feedback.

        Uses the deterministic rule-based fallback
        (:func:`~parametric_cloth.ai_pattern.refine.rule_based_refine`) -- it
        recognizes simple ``<action> <target> [by Xcm]`` directives and leaves
        anything it doesn't recognize unchanged. For open-ended feedback
        understanding, wire up an LLM backend the same way as the other
        ``from_*`` methods.
        """
        return rule_based_refine(garment, feedback)

    def _to_definition(self, gc_json: dict) -> GarmentDefinition:
        garment = garmentcode_to_definition(gc_json)
        issues = garment.validate()
        if issues:
            raise ValueError(
                f"generated pattern '{garment.name}' failed validation: "
                f"{'; '.join(issues)}"
            )
        return garment
