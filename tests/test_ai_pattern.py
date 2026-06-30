import numpy as np
import pytest

from parametric_cloth.ai_pattern import (
    PatternGenerator,
    apply_directive,
    definition_to_garmentcode,
    garmentcode_to_definition,
    parse_feedback,
    rule_based_refine,
)
from parametric_cloth.ai_pattern.refine import RefineDirective
from parametric_cloth.fabric import FabricType
from parametric_cloth.pattern import GarmentDefinition
from parametric_cloth.templates import create_skirt, create_tshirt


# --- GarmentCode adapter -----------------------------------------------

def _square_gc():
    return {
        "name": "square_test",
        "panels": {
            "a": {"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]], "fabric": "denim"},
            "b": {"vertices": [[0, 0], [10, 0], [10, 10], [0, 10]],
                  "anchor": "chest_front", "offset_normal": 3.0},
        },
        "stitches": [
            [{"panel": "a", "edge": [1, 2]}, {"panel": "b", "edge": [3, 0]}],
        ],
    }


def test_garmentcode_to_definition_basic():
    garment = garmentcode_to_definition(_square_gc())
    assert garment.name == "square_test"
    assert {p.name for p in garment.pieces} == {"a", "b"}
    assert garment.piece("a").fabric.type is FabricType.DENIM
    assert len(garment.seams) == 1
    assert garment.is_valid(), garment.validate()


def test_garmentcode_placement_preserved():
    garment = garmentcode_to_definition(_square_gc())
    assert garment.piece("b").placement.anchor == "chest_front"
    assert garment.piece("b").placement.offset_normal == 3.0


def test_garmentcode_missing_panels_raises():
    with pytest.raises(ValueError):
        garmentcode_to_definition({"name": "empty", "panels": {}, "stitches": []})


def test_garmentcode_missing_vertices_raises():
    with pytest.raises(ValueError):
        garmentcode_to_definition({"panels": {"a": {}}, "stitches": []})


def test_garmentcode_bad_stitch_raises():
    gc = {"panels": {"a": {"vertices": [[0, 0], [1, 0], [1, 1]]}},
          "stitches": [[{"panel": "a", "edge": [0, 1]}]]}   # only one side
    with pytest.raises(ValueError):
        garmentcode_to_definition(gc)


def test_definition_to_garmentcode_roundtrip_skirt():
    skirt = create_skirt(panels=4)
    gc = definition_to_garmentcode(skirt)
    restored = garmentcode_to_definition(gc)

    assert restored.name == skirt.name
    assert len(restored.pieces) == len(skirt.pieces)
    assert len(restored.seams) == len(skirt.seams)
    for orig, rest in zip(skirt.pieces, restored.pieces):
        assert orig.name == rest.name
        assert orig.fabric.type is rest.fabric.type
        for ov, rv in zip(orig.vertices, rest.vertices):
            assert ov.x == pytest.approx(rv.x)
            assert ov.y == pytest.approx(rv.y)


def test_definition_to_garmentcode_roundtrip_tshirt():
    shirt = create_tshirt(fabric=FabricType.SILK)
    restored = garmentcode_to_definition(definition_to_garmentcode(shirt))
    assert restored.is_valid(), restored.validate()
    assert restored.piece("front").fabric.type is FabricType.SILK


def test_garmentcode_preserves_placement_roundtrip():
    skirt = create_skirt(panels=4)
    restored = garmentcode_to_definition(definition_to_garmentcode(skirt))
    for orig, rest in zip(skirt.pieces, restored.pieces):
        assert rest.placement.anchor == orig.placement.anchor
        assert rest.placement.offset_normal == pytest.approx(orig.placement.offset_normal)


# --- rule-based refine -------------------------------------------------

def test_parse_feedback_extracts_action_target_amount():
    directives = parse_feedback("Make the sleeves wider and shorten the hem by 5cm")
    kinds = {(d.action, d.target, d.amount_cm) for d in directives}
    assert ("widen", "sleeve", 2.0) in kinds       # no explicit amount -> default
    assert ("shorten", "hem", 5.0) in kinds


def test_parse_feedback_handles_both_word_orders():
    prefix = parse_feedback("shorten the hem by 5cm")
    postfix = parse_feedback("hem shorter by 5cm")
    assert [(d.action, d.target, d.amount_cm) for d in prefix] == \
           [(d.action, d.target, d.amount_cm) for d in postfix]


def test_parse_feedback_no_match_returns_empty():
    assert parse_feedback("make it look more elegant somehow") == []


def test_apply_directive_widens_matching_pieces_only():
    shirt = create_tshirt()
    front_before = shirt.piece("front").area()
    sleeve_before = shirt.piece("left_sleeve").area()

    directive = RefineDirective(action="widen", target="sleeve", amount_cm=4.0)
    out = apply_directive(shirt, directive)

    assert out.piece("left_sleeve").area() > sleeve_before
    assert out.piece("right_sleeve").area() > sleeve_before
    assert out.piece("front").area() == pytest.approx(front_before)  # untouched


def test_apply_directive_shorten_reduces_y_extent():
    skirt = create_skirt(panels=4)
    panel = skirt.piece("panel_0")
    y_extent_before = max(v.y for v in panel.vertices) - min(v.y for v in panel.vertices)

    directive = RefineDirective(action="shorten", target="hem", amount_cm=5.0)
    out = apply_directive(skirt, directive)
    panel_after = out.piece("panel_0")
    y_extent_after = max(v.y for v in panel_after.vertices) - min(v.y for v in panel_after.vertices)

    assert y_extent_after == pytest.approx(y_extent_before - 5.0, abs=1e-6)


def test_apply_directive_preserves_anchor_point():
    # Lengthening should keep the minimum-y (waist) edge fixed in place.
    skirt = create_skirt(panels=4)
    panel = skirt.piece("panel_0")
    waist_y_before = min(v.y for v in panel.vertices)

    directive = RefineDirective(action="lengthen", target="hem", amount_cm=10.0)
    out = apply_directive(skirt, directive)
    waist_y_after = min(v.y for v in out.piece("panel_0").vertices)

    assert waist_y_after == pytest.approx(waist_y_before, abs=1e-6)


def test_rule_based_refine_applies_multiple_directives():
    shirt = create_tshirt()
    out = rule_based_refine(shirt, "widen the sleeve by 3cm and shorten the hem by 2cm")
    assert out.piece("left_sleeve").area() > shirt.piece("left_sleeve").area()
    assert out.is_valid(), out.validate()


def test_rule_based_refine_unrecognized_feedback_is_noop():
    shirt = create_tshirt()
    out = rule_based_refine(shirt, "make it pop")
    for orig, rest in zip(shirt.pieces, out.pieces):
        assert orig.area() == pytest.approx(rest.area())


def test_refine_keeps_garment_valid():
    skirt = create_skirt(panels=6)
    out = rule_based_refine(skirt, "narrow the panel by 1cm")
    assert out.is_valid(), out.validate()


# --- PatternGenerator: orchestration is testable via injected backends -----

def test_from_text_with_injected_backend():
    gen = PatternGenerator()

    def fake_backend(description: str) -> dict:
        return _square_gc()

    garment = gen.from_text("a square test garment", backend=fake_backend)
    assert garment.name == "square_test"


def test_from_photo_with_injected_backend():
    gen = PatternGenerator()

    def fake_backend(image, body_shape):
        return _square_gc()

    garment = gen.from_photo(np.zeros((4, 4, 3)), backend=fake_backend)
    assert garment.name == "square_test"


def test_from_sketch_with_injected_backend():
    gen = PatternGenerator()

    def fake_backend(sketch, garment_type):
        return _square_gc()

    garment = gen.from_sketch(np.zeros((4, 4)), backend=fake_backend)
    assert garment.name == "square_test"


def test_generator_rejects_invalid_backend_output():
    gen = PatternGenerator()

    def bowtie_backend(description: str) -> dict:
        return {
            "name": "bad",
            "panels": {"a": {"vertices": [[0, 0], [10, 10], [10, 0], [0, 10]]}},
            "stitches": [],
        }

    with pytest.raises(ValueError):
        gen.from_text("anything", backend=bowtie_backend)


def test_generator_refine_uses_rule_based_fallback():
    gen = PatternGenerator()
    shirt = create_tshirt()
    out = gen.refine(shirt, "widen the sleeve by 3cm")
    assert out.piece("left_sleeve").area() > shirt.piece("left_sleeve").area()


def test_default_backends_require_real_models():
    gen = PatternGenerator()
    with pytest.raises(ModuleNotFoundError):
        gen.from_text("a garment")
    with pytest.raises(ModuleNotFoundError):
        gen.from_photo(np.zeros((4, 4, 3)))
    with pytest.raises(ModuleNotFoundError):
        gen.from_sketch(np.zeros((4, 4)))
