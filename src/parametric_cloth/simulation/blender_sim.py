"""Headless Blender cloth-simulation driver (Module 3).

Runs inside Blender (``bpy`` available). It loads an avatar, places the pattern
pieces (Module 2), assembles + welds them into one cloth mesh (pure modules
above), runs the solver, validates, and exports.

All Blender-specific work lives here; the geometry/seam/validation logic it calls
is unit-tested separately. Invoke via ``scripts/simulate_garment.py``::

    blender --background --python scripts/simulate_garment.py -- \
        --garment skirt.json --avatar smplx_average.obj --output out/skirt.fbx
"""

from __future__ import annotations

import numpy as np

from ..pattern import GarmentDefinition
from .assembly import AssembledGarment, assemble_garment
from .config import SimulationConfig, cloth_settings_from_fabric
from .validate import validate_simulation_result


# --- scene assembly --------------------------------------------------------

def clear_scene() -> None:
    import bpy
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def load_avatar(filepath: str, config: SimulationConfig):
    """Import an avatar mesh and set it up as a collision object."""
    import bpy

    lower = filepath.lower()
    if lower.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif lower.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif lower.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=filepath)
    else:
        raise ValueError(f"unsupported avatar format: {filepath}")

    avatar = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = avatar
    bpy.ops.object.modifier_add(type="COLLISION")
    avatar.collision.thickness_outer = config.collision_thickness_outer
    avatar.collision.thickness_inner = config.collision_thickness_inner
    avatar.collision.cloth_friction = 5.0
    return avatar


def build_cloth_object(assembled: AssembledGarment):
    """Create a single Blender mesh object from an assembled garment."""
    import bpy

    mesh = bpy.data.meshes.new(f"{assembled.name}_mesh")
    obj = bpy.data.objects.new(assembled.name, mesh)
    bpy.context.collection.objects.link(obj)

    verts = [tuple(float(c) for c in v) for v in assembled.vertices]
    faces = [tuple(int(i) for i in f) for f in assembled.faces]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return obj


def weld_seams(obj, assembled: AssembledGarment, *, merge_distance: float = 1e-4):
    """Merge welded seam pairs by snapping each pair to its midpoint.

    The pure assembler already computed which global vertex pairs correspond
    (Strategy A); here we move them together and let Blender's
    remove-doubles collapse them into shared vertices.
    """
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    for i, j in assembled.seam_pairs:
        midpoint = (bm.verts[i].co + bm.verts[j].co) / 2.0
        bm.verts[i].co = midpoint
        bm.verts[j].co = midpoint
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_distance)
    bm.to_mesh(obj.data)
    bm.free()


def add_cloth_physics(obj, fabric, config: SimulationConfig,
                      *, damping_multiplier: float = 1.0) -> None:
    import bpy

    settings = cloth_settings_from_fabric(
        fabric, quality=config.quality, damping_multiplier=damping_multiplier
    )

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type="CLOTH")
    cloth = obj.modifiers["Cloth"]

    for attr, value in settings.to_blender_dict().items():
        setattr(cloth.settings, attr, value)

    cloth.collision_settings.use_collision = True
    cloth.collision_settings.use_self_collision = config.self_collision
    cloth.collision_settings.self_friction = settings.friction
    cloth.collision_settings.collision_quality = config.collision_quality


def run_simulation(obj, config: SimulationConfig) -> None:
    """Step the scene through every frame so the cloth solver bakes."""
    import bpy

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = config.frames
    for frame in range(1, config.frames + 1):
        scene.frame_set(frame)


def garment_vertices_world(obj) -> np.ndarray:
    """Read back the deformed vertices in world space."""
    obj_eval = obj.evaluated_get(_depsgraph())
    mesh = obj_eval.to_mesh()
    mw = obj.matrix_world
    verts = np.array([list(mw @ v.co) for v in mesh.vertices], dtype=float)
    obj_eval.to_mesh_clear()
    return verts


def _depsgraph():
    import bpy
    return bpy.context.evaluated_depsgraph_get()


def export_garment(obj, output_path: str) -> None:
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


# --- orchestration ---------------------------------------------------------

def simulate_garment(
    garment: GarmentDefinition,
    avatar_path: str,
    output_path: str,
    config: SimulationConfig | None = None,
) -> bool:
    """Full draping run with conservative-retry on simulation explosions.

    Returns True if a valid result was exported, False if every retry exploded.
    """
    from ..avatar.placement import place_garment
    from ..avatar.placement import avatar_mesh_from_bpy

    config = config or SimulationConfig()

    for attempt, damping_mult in enumerate(config.damping_schedule()):
        clear_scene()
        avatar = load_avatar(avatar_path, config)
        avatar_mesh = avatar_mesh_from_bpy(avatar)

        transforms = place_garment(garment, avatar_mesh)
        assembled = assemble_garment(
            garment, transforms, levels=config.subdivide_levels
        )

        obj = build_cloth_object(assembled)
        weld_seams(obj, assembled)

        # All pieces currently share one fabric; use the first piece's.
        fabric = garment.pieces[0].fabric
        add_cloth_physics(obj, fabric, config, damping_multiplier=damping_mult)

        run_simulation(obj, config)

        result = validate_simulation_result(
            garment_vertices_world(obj),
            np.asarray(avatar.location),
            max_distance=config.explosion_distance,
        )
        if result.ok:
            export_garment(obj, output_path)
            print(f"[ok] {garment.name}: exported to {output_path} "
                  f"(attempt {attempt + 1}, damping x{damping_mult})")
            return True

        print(f"[retry] {garment.name}: attempt {attempt + 1} invalid: "
              f"{'; '.join(result.issues)}")

    print(f"[fail] {garment.name}: all {config.max_retries} attempts exploded")
    return False
