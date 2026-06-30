import importlib

import numpy as np
import pytest

from parametric_cloth.fabric import FabricProperties, FabricType
from parametric_cloth.fabric_ai import (
    FABRIC_PRESETS_EXTENDED,
    FabricPredictor,
    PARAM_NAMES,
    fabric_from_description,
    fabric_properties_for,
    find_closest_preset,
    training_pairs_from_presets,
)


# --- extended presets --------------------------------------------------

def test_extended_table_has_at_least_20_variants():
    assert len(FABRIC_PRESETS_EXTENDED) >= 20


def test_every_extended_preset_builds_valid_properties():
    for name in FABRIC_PRESETS_EXTENDED:
        props = fabric_properties_for(name)
        assert props.validate() == []


def test_fabric_properties_for_unknown_raises():
    with pytest.raises(KeyError):
        fabric_properties_for("unobtainium")


def test_fabric_properties_for_applies_overrides():
    props = fabric_properties_for("cotton_canvas", friction=0.9)
    assert props.friction == 0.9
    assert props.mass_per_area == FABRIC_PRESETS_EXTENDED["cotton_canvas"]["mass_per_area"]


# --- fuzzy lookup --------------------------------------------------------

def test_exact_name_matches_itself():
    assert find_closest_preset("cotton_canvas") == "cotton_canvas"


def test_spaced_name_matches():
    assert find_closest_preset("cotton canvas") == "cotton_canvas"


def test_alias_resolves():
    assert find_closest_preset("workwear") == "cotton_canvas"
    assert find_closest_preset("raw denim") == "denim_heavyweight"


def test_close_description_matches_reasonable_preset():
    match = find_closest_preset("heavy brushed cotton twill")
    assert match is not None
    assert match.startswith("cotton")


def test_silk_description_matches_silk_variant():
    match = find_closest_preset("lightweight silk charmeuse for a gown")
    assert match == "silk_charmeuse"


def test_nonsense_description_returns_none():
    assert find_closest_preset("zzzqx_totally_unrelated_gibberish_999") is None


def test_empty_description_returns_none():
    assert find_closest_preset("") is None
    assert find_closest_preset("   ") is None


def test_fabric_from_description_builds_properties():
    props = fabric_from_description("heavy cotton canvas")
    assert isinstance(props, FabricProperties)
    assert props.validate() == []


def test_fabric_from_description_raises_on_nonsense():
    with pytest.raises(LookupError):
        fabric_from_description("zzzqx_totally_unrelated_gibberish_999")


# --- FabricProperties.from_description integration -------------------------

def test_fabric_properties_from_description_classmethod():
    props = FabricProperties.from_description("cotton canvas")
    assert isinstance(props, FabricProperties)
    assert props.mass_per_area == FABRIC_PRESETS_EXTENDED["cotton_canvas"]["mass_per_area"]


def test_fabric_properties_from_description_raises_on_nonsense():
    with pytest.raises(LookupError):
        FabricProperties.from_description("zzzqx_totally_unrelated_gibberish_999")


# --- bootstrap training pairs -----------------------------------------------

def test_training_pairs_shapes_match():
    descriptions, targets = training_pairs_from_presets()
    assert len(descriptions) == len(FABRIC_PRESETS_EXTENDED)
    assert targets.shape == (len(FABRIC_PRESETS_EXTENDED), len(PARAM_NAMES))


def test_training_pairs_descriptions_are_spaced():
    descriptions, _ = training_pairs_from_presets()
    assert all("_" not in d for d in descriptions)


# --- FabricPredictor: numpy inference path is testable, encoder is lazy ----

def test_predictor_predict_without_weights_raises():
    predictor = FabricPredictor()
    with pytest.raises(RuntimeError):
        predictor.predict("anything")


def test_predictor_forward_matches_manual_computation():
    rng = np.random.default_rng(0)
    D = 4  # pretend embedding dim for this unit test
    W0, b0 = rng.normal(size=(D, 5)), rng.normal(size=5)
    W1, b1 = rng.normal(size=(5, len(PARAM_NAMES))), rng.normal(size=len(PARAM_NAMES))
    predictor = FabricPredictor(weights=[W0, W1], biases=[b0, b1])

    # Bypass the (lazy, unavailable) sentence encoder by injecting a fixed embedding.
    x = rng.normal(size=D)
    predictor._embed = lambda description: x  # noqa: SLF001 (test stub)

    props = predictor.predict("anything")
    hidden = np.maximum(x @ W0 + b0, 0.0)
    expected = hidden @ W1 + b1
    actual = [getattr(props, name) for name in PARAM_NAMES]
    assert np.allclose(actual, expected)


def test_predictor_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    predictor = FabricPredictor(
        weights=[rng.normal(size=(4, len(PARAM_NAMES)))],
        biases=[rng.normal(size=len(PARAM_NAMES))],
    )
    path = str(tmp_path / "predictor.npz")
    predictor.save(path)
    loaded = FabricPredictor.load(path)
    loaded._embed = lambda description: np.ones(4)  # noqa: SLF001 (test stub)
    predictor._embed = lambda description: np.ones(4)  # noqa: SLF001 (test stub)
    p1 = predictor.predict("x")
    p2 = loaded.predict("x")
    for name in PARAM_NAMES:
        assert getattr(p1, name) == pytest.approx(getattr(p2, name))


def test_predictor_train_requires_torch():
    predictor = FabricPredictor()
    predictor._embed = lambda description: np.zeros(384)  # noqa: SLF001 (test stub)
    with pytest.raises(ModuleNotFoundError):
        predictor.train(["a", "b"], np.zeros((2, len(PARAM_NAMES))), epochs=1)


def test_predictor_embed_requires_sentence_transformers():
    mod = importlib.import_module("parametric_cloth.fabric_ai.predictor")
    predictor = mod.FabricPredictor()
    with pytest.raises(ModuleNotFoundError):
        predictor._embed("hello")  # noqa: SLF001
