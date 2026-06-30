from parametric_cloth.envcheck import check_environment, main


def test_check_environment_returns_expected_keys(capsys):
    results = check_environment()
    expected = {
        "smplx", "torch", "trimesh", "onnx", "onnxruntime",
        "sentence_transformers", "matplotlib", "pytest", "blender",
    }
    assert expected <= set(results)
    assert all(isinstance(v, bool) for v in results.values())
    capsys.readouterr()  # just confirm it printed without raising


def test_known_installed_packages_report_true():
    results = check_environment(verbose=False)
    assert results["pytest"] is True
    assert results["matplotlib"] is True


def test_quiet_mode_prints_nothing(capsys):
    check_environment(verbose=False)
    assert capsys.readouterr().out == ""


def test_main_returns_zero(capsys):
    assert main() == 0
    capsys.readouterr()
