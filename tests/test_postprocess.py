import importlib
import json

import numpy as np
import pytest

from parametric_cloth.templates import create_skirt, create_tshirt
from parametric_cloth.simulation import assemble_garment
from parametric_cloth.postprocess import (
    build_metadata,
    pack_uv_atlas,
    render_uv_layout_svg,
    write_package,
)


def _skirt_assembly(panels=4, levels=1):
    return assemble_garment(create_skirt(panels=panels), levels=levels)


# --- UV atlas --------------------------------------------------------------

def test_uv_atlas_covers_all_vertices_in_unit_square():
    asm = _skirt_assembly()
    atlas = pack_uv_atlas(asm)
    assert atlas.uv.shape == (asm.n_vertices, 2)
    assert atlas.uv.min() >= 0.0 - 1e-9
    assert atlas.uv.max() <= 1.0 + 1e-9


def test_uv_grid_dimensions():
    atlas = pack_uv_atlas(_skirt_assembly(panels=4))
    assert atlas.cols * atlas.rows >= 4
    # 4 panels -> 2x2 grid.
    assert atlas.cols == 2 and atlas.rows == 2


def test_each_panel_stays_within_its_cell():
    asm = _skirt_assembly(panels=4)
    atlas = pack_uv_atlas(asm)
    for p in range(asm.n_panels):
        idx = np.where(asm.vertex_panel == p)[0]
        umin, vmin, umax, vmax = atlas.cell_of_panel(p)
        uvs = atlas.uv[idx]
        assert uvs[:, 0].min() >= umin - 1e-9 and uvs[:, 0].max() <= umax + 1e-9
        assert uvs[:, 1].min() >= vmin - 1e-9 and uvs[:, 1].max() <= vmax + 1e-9


def test_uv_scale_is_uniform_aspect_preserved():
    asm = _skirt_assembly(panels=4)
    atlas = pack_uv_atlas(asm)
    idx = np.where(asm.vertex_panel == 0)[0]
    pts = asm.pattern_uv[idx]
    uvs = atlas.uv[idx]
    ratio_x = np.ptp(uvs[:, 0]) / np.ptp(pts[:, 0])
    ratio_y = np.ptp(uvs[:, 1]) / np.ptp(pts[:, 1])
    assert ratio_x == pytest.approx(ratio_y, rel=1e-6)


def test_panels_do_not_share_cells():
    asm = _skirt_assembly(panels=4)
    atlas = pack_uv_atlas(asm)
    centers = []
    for p in range(asm.n_panels):
        idx = np.where(asm.vertex_panel == p)[0]
        centers.append(tuple(np.round(atlas.uv[idx].mean(axis=0), 3)))
    assert len(set(centers)) == asm.n_panels


# --- UV layout svg ---------------------------------------------------------

def test_render_uv_layout_svg(tmp_path):
    asm = _skirt_assembly(panels=4)
    path = tmp_path / "uv_layout.svg"
    render_uv_layout_svg(asm, str(path))
    text = path.read_text()
    assert "<svg" in text
    assert text.count("<polygon") == asm.n_panels


# --- metadata / package ----------------------------------------------------

def test_build_metadata_fields():
    garment = create_tshirt()
    asm = assemble_garment(garment, levels=1)
    atlas = pack_uv_atlas(asm)
    meta = build_metadata(garment, asm, atlas=atlas,
                          parameters={"fit": "regular"}, poly_count=4200)
    assert meta["name"] == "tshirt"
    assert meta["n_pieces"] == 4
    assert meta["poly_count"] == 4200
    assert meta["parameters"]["fit"] == "regular"
    assert meta["uv_atlas"]["cols"] >= 1
    assert "cotton" in meta["fabrics"]


def test_write_package_creates_artifacts(tmp_path):
    garment = create_skirt(panels=6)
    asm = assemble_garment(garment, levels=1)
    pkg = write_package(str(tmp_path / "skirt"), garment, asm,
                        parameters={"panels": 6})

    meta = json.loads(open(pkg.metadata_path).read())
    assert meta["name"] == "skirt_6panel"
    assert meta["n_pieces"] == 6
    assert meta["files"]["mesh"] == "mesh.fbx"

    assert open(pkg.uv_layout_path).read().count("<polygon") == 6


# --- Blender post stays import-safe without bpy ----------------------------

def test_blender_post_imports_without_bpy():
    mod = importlib.import_module("parametric_cloth.postprocess.blender_post")
    assert hasattr(mod, "export_garment_package")


def test_blender_post_calls_require_bpy():
    mod = importlib.import_module("parametric_cloth.postprocess.blender_post")
    with pytest.raises(ModuleNotFoundError):
        mod.optimize_for_game(object())
