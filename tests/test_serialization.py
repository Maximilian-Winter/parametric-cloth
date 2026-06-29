import pytest

from parametric_cloth.fabric import FabricType
from parametric_cloth.pattern import GarmentDefinition
from parametric_cloth.serialization import (
    SCHEMA_VERSION,
    garment_from_dict,
    garment_from_json,
    garment_to_json,
    load_garment,
    save_garment,
)
from parametric_cloth.templates import create_cape, create_skirt, create_tshirt


def _assert_garments_equal(a: GarmentDefinition, b: GarmentDefinition):
    assert a.name == b.name
    assert len(a.pieces) == len(b.pieces)
    assert len(a.seams) == len(b.seams)
    for pa, pb in zip(a.pieces, b.pieces):
        assert pa.name == pb.name
        assert pa.fabric.type is pb.fabric.type
        assert pa.fabric.mass_per_area == pytest.approx(pb.fabric.mass_per_area)
        assert len(pa.vertices) == len(pb.vertices)
        for va, vb in zip(pa.vertices, pb.vertices):
            assert va.x == pytest.approx(vb.x)
            assert va.y == pytest.approx(vb.y)


@pytest.mark.parametrize("factory", [
    lambda: create_tshirt(fabric=FabricType.SILK),
    lambda: create_skirt(panels=6, flare=1.7),
    lambda: create_cape(),
])
def test_json_roundtrip_preserves_garment(factory):
    original = factory()
    restored = garment_from_json(garment_to_json(original))
    _assert_garments_equal(original, restored)


def test_enum_serialized_as_string():
    shirt = create_tshirt(fabric=FabricType.DENIM)
    text = garment_to_json(shirt)
    assert '"denim"' in text


def test_schema_version_present():
    shirt = create_tshirt()
    text = garment_to_json(shirt)
    assert f'"schema_version": {SCHEMA_VERSION}' in text


def test_future_schema_version_rejected():
    with pytest.raises(ValueError):
        garment_from_dict({"schema_version": SCHEMA_VERSION + 1,
                           "name": "x", "pieces": [], "seams": []})


def test_save_and_load_file(tmp_path):
    shirt = create_tshirt(fabric=FabricType.WOOL)
    path = tmp_path / "shirt.json"
    save_garment(shirt, str(path))
    restored = load_garment(str(path))
    _assert_garments_equal(shirt, restored)
