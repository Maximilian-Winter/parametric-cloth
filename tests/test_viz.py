import matplotlib

matplotlib.use("Agg")  # headless: no display needed to run these tests

import numpy as np
import pytest

from parametric_cloth import viz
from parametric_cloth.preview import preview_drape
from parametric_cloth.simulation.tessellate import tessellate_piece
from parametric_cloth.templates import create_cape, create_skirt, create_tshirt


def test_plot_pattern_piece_runs():
    cape = create_cape()
    ax = viz.plot_pattern_piece(cape.pieces[0], show=False)
    assert ax is not None


def test_plot_pattern_pieces_runs_for_multi_piece_garment():
    shirt = create_tshirt()
    fig = viz.plot_pattern_pieces(shirt, show=False)
    assert fig is not None


def test_plot_draped_wireframe_runs():
    cape = create_cape()
    mesh = tessellate_piece(cape.pieces[0], levels=1)
    vertices = preview_drape(cape.pieces[0], levels=1)
    ax = viz.plot_draped_wireframe(vertices, mesh.faces, show=False)
    assert ax is not None


def test_plot_loss_curve_runs():
    ax = viz.plot_loss_curve([1.0, 0.5, 0.25, 0.1], show=False, label="demo")
    assert ax is not None


def test_plot_pattern_piece_skirt_panel():
    # Exercise a different (concave-free) piece to catch axis/label issues.
    skirt = create_skirt(panels=4)
    ax = viz.plot_pattern_piece(skirt.pieces[0], show=False)
    assert ax.get_title() == "panel_0"


def test_missing_matplotlib_raises_clear_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    cape = create_cape()
    with pytest.raises(ModuleNotFoundError, match="pip install"):
        viz.plot_pattern_piece(cape.pieces[0], show=False)
