import json

from parametric_cloth.cli import main


def test_cli_writes_skirt(tmp_path, capsys):
    out = tmp_path / "skirt.json"
    rc = main(["--type", "skirt", "--panels", "6", "--output", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["name"] == "skirt_6panel"
    assert len(data["pieces"]) == 6


def test_cli_tshirt_to_stdout(capsys):
    rc = main(["--type", "tshirt", "--fit", "oversized", "--fabric", "linen"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["name"] == "tshirt"
    assert data["pieces"][0]["fabric"]["type"] == "linen"


def test_cli_cape(tmp_path):
    out = tmp_path / "cape.json"
    rc = main(["--type", "cape", "--output", str(out)])
    assert rc == 0
    assert json.loads(out.read_text())["name"] == "cape"
