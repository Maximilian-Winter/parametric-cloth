from parametric_cloth.fabric import FABRIC_PRESETS, FabricProperties, FabricType


def test_every_fabric_type_has_a_preset():
    for fabric_type in FabricType:
        assert fabric_type in FABRIC_PRESETS


def test_from_preset_sets_type_and_values():
    props = FabricProperties.from_preset(FabricType.DENIM)
    assert props.type is FabricType.DENIM
    assert props.mass_per_area == FABRIC_PRESETS[FabricType.DENIM]["mass_per_area"]


def test_presets_are_physically_sane():
    for fabric_type in FabricType:
        props = FabricProperties.from_preset(fabric_type)
        assert props.validate() == []


def test_validate_flags_bad_values():
    bad = FabricProperties(mass_per_area=-1, friction=2.0, stretch_limit=0.5)
    issues = bad.validate()
    assert any("mass_per_area" in i for i in issues)
    assert any("friction" in i for i in issues)
    assert any("stretch_limit" in i for i in issues)


def test_heavier_fabrics_weigh_more_than_lighter():
    silk = FabricProperties.from_preset(FabricType.SILK)
    leather = FabricProperties.from_preset(FabricType.LEATHER)
    assert leather.mass_per_area > silk.mass_per_area
