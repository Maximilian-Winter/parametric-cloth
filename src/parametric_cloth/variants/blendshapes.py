"""Export a PCA basis as blend shapes / morph targets for game engines.

Engines that support morph targets (Unity, Unreal, Godot) can drive garment
variation directly: component *i* becomes shape key ``Variation_i``, and the
PCA coefficients become the runtime blend weights.

The target geometry computation is pure (testable); the Blender shape-key export
is a thin lazy-``bpy`` wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pca import PCABasis


@dataclass
class BlendShapeTarget:
    name: str
    vertices: np.ndarray              # (V, 3) absolute positions for this key


def blend_shape_targets(basis: PCABasis, *, scale: float = 1.0) -> list[BlendShapeTarget]:
    """Absolute vertex positions for each PCA component as a blend-shape target.

    The basis (``Basis`` key) is the mean shape; ``Variation_i`` is
    ``mean + scale * component_i``. At runtime a coefficient of ``c`` on
    ``Variation_i`` reproduces ``c/scale`` units along that component.
    """
    targets = [BlendShapeTarget("Basis", basis.mean_shape.copy())]
    for i in range(basis.n_components):
        targets.append(
            BlendShapeTarget(f"Variation_{i}", basis.mean_shape + scale * basis.components[i])
        )
    return targets


def coefficients_to_weights(coefficients: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    """Convert PCA coefficients into blend-shape weights for the given scale."""
    return np.asarray(coefficients, dtype=float) / scale


def export_pca_as_blend_shapes(
    basis: PCABasis, base_mesh_path: str, output_path: str, *, scale: float = 1.0
) -> str:
    """Import a base mesh, add PCA components as shape keys, and export (Blender)."""
    import bpy
    from mathutils import Vector

    lower = base_mesh_path.lower()
    if lower.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=base_mesh_path)
    elif lower.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=base_mesh_path)
    elif lower.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=base_mesh_path)
    else:
        raise ValueError(f"unsupported base mesh format: {base_mesh_path}")

    obj = bpy.context.selected_objects[0]
    for target in blend_shape_targets(basis, scale=scale):
        key = obj.shape_key_add(name=target.name, from_mix=False)
        for vi, vert in enumerate(key.data):
            vert.co = Vector(tuple(float(c) for c in target.vertices[vi]))

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    if output_path.lower().endswith((".glb", ".gltf")):
        bpy.ops.export_scene.gltf(filepath=output_path, use_selection=True,
                                  export_format="GLB")
    else:
        bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True)
    return output_path
