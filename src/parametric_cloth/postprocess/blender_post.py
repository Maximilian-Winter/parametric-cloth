"""Blender-side post-processing (Module 4).

UV assignment, normal-map baking, bone-weight transfer, decimation, and the full
game-ready package export. Requires ``bpy`` (imported lazily); built on the pure
``uv``/``layout``/``metadata`` modules, which are tested without Blender.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from ..pattern import GarmentDefinition
from ..simulation.assembly import AssembledGarment
from .metadata import GarmentPackage, write_package
from .uv import AtlasLayout, pack_uv_atlas


def assign_uv_layer(obj, uv: np.ndarray, *, name: str = "PatternUV") -> None:
    """Write per-vertex UVs onto a mesh as a per-loop UV layer.

    Must run *before* seam welding: ``uv`` is indexed by the assembled
    (un-welded) vertex order. Welding then merges vertices but keeps per-loop
    UVs, producing the desired UV seams at panel boundaries.
    """
    mesh = obj.data
    if name not in mesh.uv_layers:
        mesh.uv_layers.new(name=name)
    layer = mesh.uv_layers[name]
    for loop in mesh.loops:
        u, v = uv[loop.vertex_index]
        layer.data[loop.index].uv = (float(u), float(v))


def optimize_for_game(obj, target_poly_count: int = 5000):
    """Decimate to a poly budget, then clean up doubles and normals."""
    import bpy

    bpy.context.view_layer.objects.active = obj
    current = len(obj.data.polygons)
    if current > target_poly_count:
        bpy.ops.object.modifier_add(type="DECIMATE")
        obj.modifiers["Decimate"].ratio = target_poly_count / current
        bpy.ops.object.modifier_apply(modifier="Decimate")

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def bake_normal_map(high_poly, low_poly, output_path: str, *, resolution: int = 2048):
    """Bake wrinkle detail from the simulated high-poly mesh onto the low-poly."""
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.bake_type = "NORMAL"

    image = bpy.data.images.new("NormalMap", resolution, resolution)

    mat = low_poly.data.materials and low_poly.data.materials[0]
    if not mat:
        mat = bpy.data.materials.new("GarmentMat")
        low_poly.data.materials.append(mat)
    mat.use_nodes = True
    tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex_node.image = image
    mat.node_tree.nodes.active = tex_node

    bpy.ops.object.select_all(action="DESELECT")
    high_poly.select_set(True)
    low_poly.select_set(True)
    bpy.context.view_layer.objects.active = low_poly
    bpy.ops.object.bake(type="NORMAL", use_selected_to_active=True, cage_extrusion=0.01)

    image.filepath_raw = output_path
    image.file_format = "PNG"
    image.save()
    return output_path


def transfer_bone_weights(avatar, garment) -> None:
    """Transfer SMPL-X skeletal weights to the garment via Surface Deform."""
    import bpy

    bpy.context.view_layer.objects.active = garment
    bpy.ops.object.modifier_add(type="SURFACE_DEFORM")
    mod = garment.modifiers["SurfaceDeform"]
    mod.target = avatar
    bpy.ops.object.surfacedeform_bind(modifier="SurfaceDeform")
    bpy.ops.object.modifier_apply(modifier="SurfaceDeform")


def duplicate_object(obj):
    import bpy

    copy = obj.copy()
    copy.data = obj.data.copy()
    bpy.context.collection.objects.link(copy)
    return copy


def export_object(obj, output_path: str) -> None:
    import bpy

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    lower = output_path.lower()
    if lower.endswith(".fbx"):
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True,
                                 mesh_smooth_type="FACE")
    elif lower.endswith((".glb", ".gltf")):
        bpy.ops.export_scene.gltf(filepath=output_path, use_selection=True,
                                  export_format="GLB")
    else:
        raise ValueError(f"unsupported export format: {output_path}")


def export_garment_package(
    high_poly,
    avatar,
    garment: GarmentDefinition,
    assembled: AssembledGarment,
    out_dir: str,
    *,
    atlas: Optional[AtlasLayout] = None,
    parameters: Optional[dict[str, Any]] = None,
    target_poly_count: int = 5000,
    normal_resolution: int = 2048,
) -> GarmentPackage:
    """Produce the full game-ready package from a draped (UV'd) garment.

    ``high_poly`` must already carry the pattern UV layer (assign it before
    welding via :func:`assign_uv_layer`). Builds a decimated low-poly copy, bakes
    its normal map, transfers bone weights, exports the mesh, and writes the
    metadata + UV-layout reference.
    """
    atlas = atlas or pack_uv_atlas(assembled)

    low = duplicate_object(high_poly)
    optimize_for_game(low, target_poly_count)
    bake_normal_map(high_poly, low, os.path.join(out_dir, "normal.png"),
                    resolution=normal_resolution)
    transfer_bone_weights(avatar, low)
    export_object(low, os.path.join(out_dir, "mesh.fbx"))

    return write_package(
        out_dir, garment, assembled,
        atlas=atlas, parameters=parameters,
        poly_count=len(low.data.polygons),
    )
