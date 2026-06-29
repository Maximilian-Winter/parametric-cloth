# Parametric Cloth Pipeline

## From Sewing Patterns to Real-Time Game Garments

---

## Overview

A pipeline for generating 3D clothing from parametric 2D sewing patterns, using automated cloth simulation and learned runtime deformation. Designed for solo and small-team game development where programming skill is strong but 3D modeling skill is limited.

Garments are defined as code — 2D pattern geometry, construction rules, and fabric properties. The pipeline simulates draping on a standardized parametric body, learns a fast deformation model from the simulation results, and exports game-ready meshes that can be customized through parameter variation in real time.

The system is structured as seven modules, each building on the previous. Each module produces a testable, usable result before the next begins.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Module 1: PATTERN DEFINITION                                    │
│  Parametric 2D geometry, fabric properties, seam rules           │
│  Result: GarmentDefinition data structures                       │
│                                                                  │
│  Module 2: AVATAR SYSTEM                                         │
│  SMPL-X parametric body with known landmarks                     │
│  Result: Deterministic pattern placement on any body shape       │
│                                                                  │
│  Module 3: CLOTH SIMULATION                                      │
│  Automated Blender headless pipeline                             │
│  Result: Draped 3D garment meshes from 2D patterns               │
│                                                                  │
│  Module 4: POST-PROCESSING & EXPORT                              │
│  UV mapping from patterns, normal baking, bone weights           │
│  Result: Game-ready FBX/glTF with textures                       │
│                                                                  │
│  Module 5: VARIANT SYSTEM                                        │
│  Batch generation and PCA compression                            │
│  Result: Compact garment library with continuous variation        │
│                                                                  │
│  Module 6: LEARNED RUNTIME DEFORMATION                           │
│  TailorNet-style pose-conditioned neural network                 │
│  Result: Real-time cloth deformation without physics solver      │
│                                                                  │
│  Module 7: GAME ENGINE INTEGRATION                               │
│  Player customization, ONNX inference, texture system            │
│  Result: In-game clothing customization with real-time draping    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Module 1: Pattern Definition

**Depends on:** Nothing
**Produces:** Data structures representing any garment as parametric 2D geometry
**Development time:** 1 week

### Core Idea

Real sewing patterns are 2D shapes. A basic T-shirt consists of a front panel, back panel, and two sleeve pieces. A skirt is two to eight trapezoidal panels. A cape is a single large arc. All of these are completely describable as parametric geometry — functions that take measurements and return polygons.

Because patterns are code, garment creation becomes programming rather than 3D modeling. And because they are parametric, player customization becomes parameter adjustment.

### Data Model

```python
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class FabricType(Enum):
    COTTON = "cotton"
    DENIM = "denim"
    SILK = "silk"
    LEATHER = "leather"
    WOOL = "wool"
    LINEN = "linen"


# Physical properties from KES-F textile measurements
FABRIC_PRESETS = {
    FabricType.COTTON:  {"mass_per_area": 0.20, "stiffness": 15.0, "bending": 0.8,  "damping": 5.0,  "friction": 0.5, "stretch_limit": 1.05},
    FabricType.DENIM:   {"mass_per_area": 0.45, "stiffness": 30.0, "bending": 4.5,  "damping": 8.0,  "friction": 0.6, "stretch_limit": 1.02},
    FabricType.SILK:    {"mass_per_area": 0.10, "stiffness": 8.0,  "bending": 0.1,  "damping": 2.0,  "friction": 0.3, "stretch_limit": 1.08},
    FabricType.LEATHER: {"mass_per_area": 0.80, "stiffness": 50.0, "bending": 12.0, "damping": 15.0, "friction": 0.7, "stretch_limit": 1.01},
    FabricType.WOOL:    {"mass_per_area": 0.30, "stiffness": 20.0, "bending": 1.5,  "damping": 7.0,  "friction": 0.5, "stretch_limit": 1.04},
    FabricType.LINEN:   {"mass_per_area": 0.18, "stiffness": 12.0, "bending": 0.6,  "damping": 4.0,  "friction": 0.4, "stretch_limit": 1.06},
}


@dataclass
class FabricProperties:
    """Physical properties controlling simulation behavior."""
    type: FabricType = FabricType.COTTON
    mass_per_area: float = 0.20       # kg/m²
    stiffness: float = 15.0           # structural stiffness
    bending: float = 0.8              # bending resistance
    damping: float = 5.0              # motion settling speed
    friction: float = 0.5             # surface friction
    stretch_limit: float = 1.05       # max stretch ratio (1.0 = no stretch)

    @classmethod
    def from_preset(cls, fabric_type: FabricType) -> "FabricProperties":
        return cls(type=fabric_type, **FABRIC_PRESETS[fabric_type])


@dataclass
class PatternVertex:
    """A point on a 2D pattern piece, in centimeters."""
    x: float
    y: float


@dataclass
class SeamEdge:
    """One side of a seam, referencing a range of vertices on a pattern piece."""
    piece_name: str
    vertex_start_index: int
    vertex_end_index: int


@dataclass
class Seam:
    """A connection between two edges on pattern pieces.
    During simulation, these edges are pulled together like sewing."""
    edge_a: SeamEdge
    edge_b: SeamEdge
    stiffness: float = 1.0


@dataclass
class PlacementHint:
    """Where to position a pattern piece relative to the avatar.
    Uses body landmark names resolved by the avatar system (Module 2)."""
    anchor: str               # e.g. "chest_front", "left_upper_arm", "waist"
    offset_normal: float      # distance from body surface in cm
    rotation: float = 0.0     # rotation around the surface normal in degrees


@dataclass
class PatternPiece:
    """A single 2D pattern piece — one component of a garment."""
    name: str
    vertices: List[PatternVertex]
    subdivisions: int = 10
    placement: Optional[PlacementHint] = None
    fabric: FabricProperties = field(default_factory=FabricProperties)
    seam_allowance: float = 1.0       # cm, excluded from simulation


@dataclass
class GarmentDefinition:
    """Complete definition of a garment — patterns, construction, simulation."""
    name: str
    pieces: List[PatternPiece]
    seams: List[Seam]
    simulation_frames: int = 250
    simulation_substeps: int = 15
    gravity: float = -9.81
```

### Garment Template Functions

Each garment type is a Python function that takes measurements and returns a `GarmentDefinition`. This is where the parametric power lives — the same function produces different garments for different parameters.

```python
def create_skirt(
    waist_half: float = 18.0,       # cm, half the waist circumference
    hip_half: float = 24.0,         # cm, half the hip circumference
    length: float = 50.0,           # cm, waist to hem
    panels: int = 4,                # number of panels (2, 4, 6, 8)
    flare: float = 1.3,             # hem-to-hip ratio (1.0 = pencil, 2.0 = full)
    fabric: FabricType = FabricType.COTTON,
) -> GarmentDefinition:
    """Generate a parametric skirt from panel count and measurements."""
    
    panel_waist = waist_half * 2 / panels
    panel_hip = hip_half * 2 / panels
    panel_hem = panel_hip * flare
    hip_drop = 20.0  # cm from waist to hip line
    
    pieces = []
    seams = []
    
    for i in range(panels):
        piece = PatternPiece(
            name=f"panel_{i}",
            vertices=[
                PatternVertex(0, 0),                          # waist left
                PatternVertex(panel_waist, 0),                # waist right
                PatternVertex(panel_hip + (panel_hem - panel_hip) * 0.5, hip_drop),  # hip right
                PatternVertex(panel_hem, length),             # hem right
                PatternVertex(0 - (panel_hem - panel_hip) * 0.5 + (panel_hem - panel_waist) * 0.5, length),  # hem left
            ],
            placement=PlacementHint(
                anchor=f"waist_segment_{i}",
                offset_normal=2.0,
            ),
            fabric=FabricProperties.from_preset(fabric),
            subdivisions=12,
        )
        pieces.append(piece)
        
        # Connect each panel to the next (wrapping around)
        next_i = (i + 1) % panels
        seams.append(Seam(
            edge_a=SeamEdge(f"panel_{i}", 1, 2),      # right edge of this panel
            edge_b=SeamEdge(f"panel_{next_i}", 0, 4),  # left edge of next panel
        ))
    
    return GarmentDefinition(
        name=f"skirt_{panels}panel",
        pieces=pieces,
        seams=seams,
    )


def create_tshirt(
    chest_half_width: float = 25.0,
    length: float = 65.0,
    shoulder_width: float = 20.0,
    neck_width: float = 8.0,
    neck_depth_front: float = 6.0,
    neck_depth_back: float = 2.0,
    sleeve_length: float = 20.0,
    sleeve_upper_width: float = 18.0,
    sleeve_lower_width: float = 16.0,
    ease: float = 1.1,
    fabric: FabricType = FabricType.COTTON,
) -> GarmentDefinition:
    """Generate a parametric T-shirt."""
    
    w = chest_half_width * ease
    fab = FabricProperties.from_preset(fabric)
    
    front = PatternPiece(
        name="front",
        vertices=[
            PatternVertex(0, 0),
            PatternVertex(w, 0),
            PatternVertex(w, length),
            PatternVertex(w - shoulder_width, length),
            PatternVertex(w/2 + neck_width/2, length - neck_depth_front * 0.3),
            PatternVertex(w/2, length - neck_depth_front),
            PatternVertex(w/2 - neck_width/2, length - neck_depth_front * 0.3),
            PatternVertex(shoulder_width, length),
            PatternVertex(0, length),
        ],
        placement=PlacementHint(anchor="chest_front", offset_normal=3.0),
        fabric=fab,
        subdivisions=12,
    )
    
    back = PatternPiece(
        name="back",
        vertices=[
            PatternVertex(0, 0),
            PatternVertex(w, 0),
            PatternVertex(w, length),
            PatternVertex(w - shoulder_width, length),
            PatternVertex(w/2 + neck_width/2, length - neck_depth_back * 0.3),
            PatternVertex(w/2, length - neck_depth_back),
            PatternVertex(w/2 - neck_width/2, length - neck_depth_back * 0.3),
            PatternVertex(shoulder_width, length),
            PatternVertex(0, length),
        ],
        placement=PlacementHint(anchor="chest_back", offset_normal=3.0),
        fabric=fab,
        subdivisions=12,
    )
    
    left_sleeve = PatternPiece(
        name="left_sleeve",
        vertices=[
            PatternVertex(0, 0),
            PatternVertex(sleeve_lower_width, 0),
            PatternVertex(sleeve_upper_width, sleeve_length),
            PatternVertex(0, sleeve_length),
        ],
        placement=PlacementHint(anchor="left_upper_arm", offset_normal=2.0),
        fabric=fab,
        subdivisions=8,
    )
    
    right_sleeve = PatternPiece(
        name="right_sleeve",
        vertices=[
            PatternVertex(0, 0),
            PatternVertex(sleeve_lower_width, 0),
            PatternVertex(sleeve_upper_width, sleeve_length),
            PatternVertex(0, sleeve_length),
        ],
        placement=PlacementHint(anchor="right_upper_arm", offset_normal=2.0),
        fabric=fab,
        subdivisions=8,
    )
    
    seams = [
        Seam(edge_a=SeamEdge("front", 0, 8), edge_b=SeamEdge("back", 0, 8)),
        Seam(edge_a=SeamEdge("front", 1, 2), edge_b=SeamEdge("back", 1, 2)),
        Seam(edge_a=SeamEdge("front", 2, 3), edge_b=SeamEdge("back", 2, 3)),
        Seam(edge_a=SeamEdge("front", 7, 8), edge_b=SeamEdge("back", 7, 8)),
        Seam(edge_a=SeamEdge("left_sleeve", 2, 3), edge_b=SeamEdge("front", 8, 2)),
        Seam(edge_a=SeamEdge("right_sleeve", 2, 3), edge_b=SeamEdge("front", 2, 8)),
    ]
    
    return GarmentDefinition(
        name="tshirt",
        pieces=[front, back, left_sleeve, right_sleeve],
        seams=seams,
    )
```

### Player Customization Mapping

Player-facing options translate to pattern parameters through a mapping layer.

```python
@dataclass
class TShirtCustomization:
    fit: str = "regular"           # slim, regular, loose, oversized
    length: str = "regular"        # cropped, regular, longline
    sleeve_length: str = "short"   # cap, short, three_quarter, long
    neckline: str = "crew"         # crew, v_neck, scoop, boat
    fabric: str = "cotton"

    def to_pattern_params(self) -> dict:
        ease_map = {"slim": 1.02, "regular": 1.1, "loose": 1.25, "oversized": 1.4}
        length_map = {"cropped": 50, "regular": 65, "longline": 80}
        sleeve_map = {"cap": 8, "short": 20, "three_quarter": 40, "long": 55}
        neck_map = {"crew": 4, "v_neck": 15, "scoop": 10, "boat": 2}
        
        return {
            "ease": ease_map[self.fit],
            "length": length_map[self.length],
            "sleeve_length": sleeve_map[self.sleeve_length],
            "neck_depth_front": neck_map[self.neckline],
            "fabric": FabricType(self.fabric),
        }
```

### Garment Template Priority

```
Priority 1 — minimum viable wardrobe:
  Skirt          2-8 panels, 2-8 seams      Easy
  Cape           1-2 panels, 1-2 seams      Easy
  T-Shirt        4 pieces, 6-8 seams        Medium
  Pants          4-6 pieces, 6-10 seams     Hard

Priority 2 — expanded wardrobe:
  Hoodie         6-8 pieces                 Hard
  Dress          bodice + skirt panels      Medium
  Tank Top       2 pieces                   Easy
  Vest           jacket without sleeves     Medium

Priority 3 — specialty:
  Robe           long, open front           Medium
  Kimono wrap    rectangular panels + sash  Medium
  Armor overlay  rigid panels + cloth       Hard
```

Start with skirts and capes. They have the fewest pieces, the simplest seams, and gravity does most of the work. Get the pipeline reliable on simple garments before attempting pants.

### Module 1 Deliverables

- [ ] `GarmentDefinition` data model with all supporting types
- [ ] `FabricProperties` with KES-F presets for 6 fabric types
- [ ] `create_skirt()` template function with parameterization
- [ ] `create_tshirt()` template function with parameterization
- [ ] Serialization to/from JSON for pipeline interchange
- [ ] Unit tests for pattern generation (geometry sanity checks)

---

## Module 2: Avatar System

**Depends on:** Module 1 (PlacementHint anchor names)
**Produces:** A standardized parametric body with deterministic landmark positions
**Development time:** 1 week

### SMPL-X Parametric Body

SMPL-X is a parametric 3D body model with approximately 10,000 vertices, a standard skeleton, and a parameter space covering body shape (10 beta parameters), pose (joint rotations), and facial expression. It provides deterministic vertex indices for anatomical landmarks, eliminating guesswork about body positioning.

### Why SMPL-X

The pattern placement problem — "where exactly is the chest?" — is the most fragile part of any automated cloth pipeline. With a custom avatar, you must either hardcode positions, use bone names that vary between rigs, or mark vertex groups manually. All three approaches break when the avatar changes.

SMPL-X solves this because every body shape variant uses the same vertex topology. Vertex 3065 is always the anterior chest. Vertex 4432 is always the left shoulder. The landmarks are published, consistent, and parametric — they move predictably when body shape parameters change.

### Landmark Registry

```python
SMPLX_LANDMARKS = {
    # Torso
    "chest_front":      3065,
    "chest_back":       5937,
    "waist_front":      3502,
    "waist_back":       6295,
    "hip_front":        1176,
    "hip_back":         4540,
    
    # Shoulders and arms
    "left_shoulder":    4432,
    "right_shoulder":   7198,
    "left_upper_arm":   4620,
    "right_upper_arm":  7660,
    "left_elbow":       4800,
    "right_elbow":      7830,
    "left_wrist":       5070,
    "right_wrist":      8100,
    
    # Legs
    "left_hip":         910,
    "right_hip":        4380,
    "left_knee":        1100,
    "right_knee":       4550,
    "left_ankle":       3330,
    "right_ankle":      6700,
    
    # Neck
    "neck_front":       3068,
    "neck_back":        6010,
}


def compute_waist_segments(avatar_mesh, n_segments: int) -> dict:
    """Compute evenly-spaced anchor points around the waist for skirt panels.
    Uses the waist landmark ring on the SMPL-X body."""
    # Extract the waist vertex ring
    # Distribute n_segments anchor points evenly around the circumference
    # Return {"waist_segment_0": (position, normal), ...}
    pass
```

### Pattern Positioning

```python
import bpy
from mathutils import Vector

def position_piece_on_avatar(
    pattern_obj: bpy.types.Object,
    avatar_obj: bpy.types.Object,
    placement: PlacementHint,
):
    """Position a 2D pattern piece near the avatar body surface.
    
    Uses SMPL-X vertex positions and normals for deterministic placement
    that adapts to any body shape parameterization.
    """
    vertex_idx = SMPLX_LANDMARKS[placement.anchor]
    body_vertex = avatar_obj.data.vertices[vertex_idx]
    
    position = Vector(body_vertex.co)
    normal = Vector(body_vertex.normal)
    offset = placement.offset_normal / 100  # cm to meters
    
    pattern_obj.location = position + normal * offset
    
    # Orient the pattern piece to face the body surface
    track_quat = normal.to_track_quat('Z', 'Y')
    pattern_obj.rotation_euler = track_quat.to_euler()
```

### Body Shape Variation

SMPL-X body shape is controlled by 10 beta parameters. Varying these produces different body proportions while maintaining consistent vertex topology and landmark indices.

```python
import numpy as np

# Representative body shapes for simulation sampling
BODY_SHAPE_SAMPLES = {
    "average":   np.zeros(10),
    "athletic":  np.array([1.5, -0.5, 0.3, 0, 0, 0, 0, 0, 0, 0]),
    "heavy":     np.array([-1.0, 1.5, 0.8, 0, 0, 0, 0, 0, 0, 0]),
    "tall_slim": np.array([0.5, -1.0, -0.5, 1.5, 0, 0, 0, 0, 0, 0]),
    "short":     np.array([0.0, 0.5, 0.3, -1.5, 0, 0, 0, 0, 0, 0]),
}


def generate_smplx_avatar(betas: np.ndarray, pose: np.ndarray = None) -> str:
    """Generate an SMPL-X mesh with given shape parameters.
    Returns path to exported FBX file."""
    import smplx
    
    model = smplx.create(
        model_path="models/smplx",
        model_type="smplx",
        gender="neutral",
        betas=torch.tensor(betas).unsqueeze(0),
        body_pose=torch.tensor(pose).unsqueeze(0) if pose is not None else None,
    )
    
    output = model()
    vertices = output.vertices.detach().numpy().squeeze()
    faces = model.faces
    
    # Export as FBX via trimesh or Blender
    export_path = f"avatars/smplx_{'_'.join(f'{b:.1f}' for b in betas[:3])}.fbx"
    export_mesh(vertices, faces, export_path)
    return export_path
```

### Module 2 Deliverables

- [ ] SMPL-X model integration (Python `smplx` package)
- [ ] Landmark registry with verified vertex indices
- [ ] `position_piece_on_avatar()` function
- [ ] `compute_waist_segments()` for skirt panel placement
- [ ] Body shape sampling with 5 representative shapes
- [ ] Avatar export to FBX for Blender pipeline
- [ ] Tests: verify landmarks are correct across body shape variations

---

## Module 3: Cloth Simulation

**Depends on:** Module 1 (patterns), Module 2 (avatar)
**Produces:** Draped 3D garment meshes from 2D pattern inputs
**Development time:** 2-3 weeks

### Pipeline Overview

The simulation pipeline runs headless via Blender's Python API. It takes a `GarmentDefinition` and an avatar mesh, assembles the scene, runs cloth physics until the garment settles, and exports the result.

```bash
blender --background --python simulate_garment.py -- \
    --garment skirt_4panel.json \
    --avatar smplx_average.fbx \
    --output output/skirt_4panel.fbx \
    --frames 250
```

### Blender Scene Assembly

```python
"""simulate_garment.py — Headless Blender cloth simulation pipeline."""

import bpy
import bmesh
import json
import sys
import math
from mathutils import Vector


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)


def load_avatar(filepath: str) -> bpy.types.Object:
    bpy.ops.import_scene.fbx(filepath=filepath)
    avatar = bpy.context.selected_objects[0]
    
    bpy.context.view_layer.objects.active = avatar
    bpy.ops.object.modifier_add(type='COLLISION')
    avatar.collision.thickness_outer = 0.002
    avatar.collision.thickness_inner = 0.001
    avatar.collision.cloth_friction = 5.0
    
    return avatar


def create_pattern_mesh(piece: dict) -> bpy.types.Object:
    name = piece["name"]
    vertices = piece["vertices"]
    subdivisions = piece.get("subdivisions", 10)
    
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    bm = bmesh.new()
    bm_verts = []
    for v in vertices:
        # Convert cm to meters
        bm_verts.append(bm.verts.new((v["x"] / 100, v["y"] / 100, 0)))
    
    bm.verts.ensure_lookup_table()
    bm.faces.new(bm_verts)
    
    bmesh.ops.subdivide_edges(
        bm, edges=bm.edges[:], cuts=subdivisions, use_grid_fill=True,
    )
    
    bm.to_mesh(mesh)
    bm.free()
    
    return obj


def add_cloth_physics(obj: bpy.types.Object, fabric: dict):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type='CLOTH')
    
    cloth = obj.modifiers["Cloth"]
    settings = cloth.settings
    
    settings.mass = fabric.get("mass_per_area", 0.3)
    settings.tension_stiffness = fabric.get("stiffness", 15.0)
    settings.compression_stiffness = fabric.get("stiffness", 15.0)
    settings.bending_stiffness = fabric.get("bending", 0.5)
    settings.tension_damping = fabric.get("damping", 5.0)
    settings.air_damping = 1.0
    
    cloth.collision_settings.use_collision = True
    cloth.collision_settings.use_self_collision = True
    cloth.collision_settings.self_friction = fabric.get("friction", 0.5)
    cloth.collision_settings.collision_quality = 5
    
    settings.quality = 15


def run_simulation(frame_count: int = 250):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count
    
    for frame in range(1, frame_count + 1):
        scene.frame_set(frame)
    
    scene.frame_set(frame_count)


def apply_simulation(obj: bpy.types.Object):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Cloth")
```

### Seam Handling Strategies

Sewing seams — pulling separate pattern piece edges together during simulation — are the most challenging part of the pipeline. Blender's native sewing spring support is limited compared to dedicated tools like Marvelous Designer. Three approaches, in order of complexity:

**Strategy A: Pre-merge (simplest, recommended for initial development)**

Merge seam vertices before simulation. Position pattern pieces close to the avatar, identify matching edge vertices, merge them, then simulate the resulting connected mesh.

```python
def pre_merge_seams(objects: dict, seams: list):
    """Merge seam edge vertices before simulation.
    
    Produces a single connected mesh that drapes naturally.
    Simpler than sewing springs but less physically accurate.
    """
    # Join all pattern piece objects into one mesh
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects.values():
        obj.select_set(True)
    bpy.context.view_layer.objects.active = list(objects.values())[0]
    bpy.ops.object.join()
    
    merged = bpy.context.active_object
    
    # For each seam, find closest vertex pairs and merge
    bm = bmesh.new()
    bm.from_mesh(merged.data)
    bm.verts.ensure_lookup_table()
    
    for seam in seams:
        # Find vertices along each seam edge
        edge_a_verts = get_edge_vertices(bm, seam["edge_a"])
        edge_b_verts = get_edge_vertices(bm, seam["edge_b"])
        
        # Merge closest pairs
        for va in edge_a_verts:
            closest = min(edge_b_verts, key=lambda vb: (va.co - vb.co).length)
            bmesh.ops.pointmerge(bm, verts=[va, closest], merge_co=((va.co + closest.co) / 2))
    
    bm.to_mesh(merged.data)
    bm.free()
    
    return merged
```

**Strategy B: Two-pass (shrinkwrap then simulate)**

1. Shrinkwrap each pattern piece onto the avatar surface for rough positioning.
2. Merge seam vertices on the shrinkwrapped result.
3. Run cloth simulation from this well-positioned starting state.

This produces better results than raw pre-merge because the cloth starts close to its final position, reducing simulation instability.

**Strategy C: Dedicated solver (HiPhyEngine or custom PBD)**

For production-quality sewing with proper tension along seam lines:

- **HiPhyEngine** is a Blender add-on (2025) from an ex-Walt Disney Animation Studios engineer. GPU-based, intersection-free guarantees, proper sewing spring support. Drop-in replacement for Blender's cloth solver.
- **Custom PBD/XPBD solver** in Python/C++. Position-Based Dynamics is conceptually straightforward — iterative position constraint projection — and can be implemented in a few hundred lines of NumPy. XPBD (Macklin et al., 2016) adds timestep-independent stiffness for stable behavior. A custom solver gives full control over sewing spring behavior.

### Simulation Stability

Cloth simulation can fail in several ways: vertices flying to infinity, interpenetration, oscillation. Mitigations:

```python
def validate_simulation_result(obj: bpy.types.Object, 
                                avatar: bpy.types.Object,
                                max_distance: float = 2.0) -> bool:
    """Check if the simulation produced a reasonable result.
    
    Returns False if any vertex is more than max_distance meters
    from the avatar center, indicating a simulation explosion.
    """
    avatar_center = Vector(avatar.location)
    
    for vert in obj.data.vertices:
        world_pos = obj.matrix_world @ vert.co
        if (world_pos - avatar_center).length > max_distance:
            return False
    
    return True


def simulate_with_retry(garment_def, avatar_path, max_retries=3):
    """Run simulation with progressively more conservative settings on failure."""
    
    damping_multipliers = [1.0, 2.0, 4.0]
    
    for attempt, damping_mult in enumerate(damping_multipliers):
        clear_scene()
        avatar = load_avatar(avatar_path)
        
        # Create pattern meshes with adjusted damping
        for piece in garment_def["pieces"]:
            piece["fabric"]["damping"] *= damping_mult
        
        # ... assemble and simulate ...
        
        if validate_simulation_result(garment_obj, avatar):
            return garment_obj
    
    raise SimulationError(f"Failed after {max_retries} attempts")
```

### Module 3 Deliverables

- [ ] Blender headless simulation script
- [ ] Pattern mesh creation from `GarmentDefinition` JSON
- [ ] Avatar loading with collision setup
- [ ] Pre-merge seam handling (Strategy A)
- [ ] Simulation execution with configurable frame count
- [ ] Result validation (detect explosions)
- [ ] Retry logic with conservative fallback settings
- [ ] End-to-end test: 4-panel skirt on SMPL-X avatar → FBX export

---

## Module 4: Post-Processing & Export

**Depends on:** Module 3 (simulated garment meshes)
**Produces:** Game-ready meshes with UVs, normal maps, and bone weights
**Development time:** 1-2 weeks

### UV Mapping from Patterns

The pattern-based approach provides a significant advantage: the original 2D pattern IS the UV map. No manual UV unwrapping is needed. Each vertex in the simulated mesh corresponds to a vertex in the original 2D pattern, and those 2D coordinates become UV coordinates directly.

This means textures designed on the flat pattern — stripes, prints, logos — wrap correctly around the 3D garment without manual adjustment. This is exactly how real fabric printing works.

```python
def generate_uv_from_patterns(obj: bpy.types.Object, 
                               pieces: list,
                               atlas_margin: float = 0.02):
    """Create UV coordinates from the original 2D pattern layout.
    
    Each panel gets its own UV island, packed into a single atlas
    with margin to prevent texture bleeding at seams.
    """
    mesh = obj.data
    
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="PatternUV")
    
    uv_layer = mesh.uv_layers["PatternUV"]
    
    # Group vertices by their source panel
    panel_vertices = group_vertices_by_panel(mesh, pieces)
    
    # Pack panels into UV atlas with margin
    atlas_layout = pack_panels_into_atlas(panel_vertices, margin=atlas_margin)
    
    # Assign UV coordinates
    for face in mesh.polygons:
        for loop_idx in face.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            uv_layer.data[loop_idx].uv = atlas_layout.get_uv(vert_idx)
```

### Normal Map Baking

Captures wrinkle and fold detail from the high-resolution simulation mesh into a texture that can be applied to a lower-polygon game mesh.

```python
def bake_normal_map(high_poly: bpy.types.Object,
                    low_poly: bpy.types.Object,
                    output_path: str,
                    resolution: int = 2048):
    """Bake normal map from simulated mesh to game-ready mesh."""
    
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.bake_type = 'NORMAL'
    
    image = bpy.data.images.new("NormalMap", resolution, resolution)
    
    # Set up material with image texture node for baking target
    mat = ensure_material(low_poly, "GarmentMat")
    mat.use_nodes = True
    tex_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex_node.image = image
    mat.node_tree.nodes.active = tex_node
    
    # Bake from high-poly to low-poly
    bpy.ops.object.select_all(action='DESELECT')
    high_poly.select_set(True)
    low_poly.select_set(True)
    bpy.context.view_layer.objects.active = low_poly
    
    bpy.ops.object.bake(
        type='NORMAL',
        use_selected_to_active=True,
        cage_extrusion=0.01,
    )
    
    image.filepath_raw = output_path
    image.file_format = 'PNG'
    image.save()
```

### Bone Weight Transfer

Transfers skeletal deformation weights from the SMPL-X avatar to the garment mesh, enabling the garment to animate with the character's skeleton.

```python
def transfer_bone_weights(avatar: bpy.types.Object, 
                          garment: bpy.types.Object):
    """Transfer SMPL-X bone weights to garment via Surface Deform.
    
    Surface Deform is more robust than nearest-vertex data transfer
    for meshes with different topologies.
    """
    bpy.context.view_layer.objects.active = garment
    
    # Add Surface Deform modifier
    bpy.ops.object.modifier_add(type='SURFACE_DEFORM')
    mod = garment.modifiers["SurfaceDeform"]
    mod.target = avatar
    
    # Bind to avatar surface
    bpy.ops.object.surfacedeform_bind(modifier="SurfaceDeform")
    
    # Apply the binding as vertex groups
    bpy.ops.object.modifier_apply(modifier="SurfaceDeform")
```

### Mesh Optimization

```python
def optimize_for_game(obj: bpy.types.Object,
                      target_poly_count: int = 5000):
    """Reduce polygon count while preserving garment shape."""
    
    bpy.context.view_layer.objects.active = obj
    
    # Decimate
    bpy.ops.object.modifier_add(type='DECIMATE')
    mod = obj.modifiers["Decimate"]
    
    current_polys = len(obj.data.polygons)
    if current_polys > target_poly_count:
        mod.ratio = target_poly_count / current_polys
        bpy.ops.object.modifier_apply(modifier="Decimate")
    else:
        bpy.ops.object.modifier_remove(modifier="Decimate")
    
    # Clean up
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
```

### Export

```python
def export_garment(garment_obj: bpy.types.Object,
                   output_path: str,
                   format: str = "fbx"):
    """Export the final game-ready garment."""
    
    bpy.ops.object.select_all(action='DESELECT')
    garment_obj.select_set(True)
    
    if format == "fbx":
        bpy.ops.export_scene.fbx(
            filepath=output_path,
            use_selection=True,
            mesh_smooth_type='FACE',
        )
    elif format == "gltf":
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            use_selection=True,
            export_format='GLB',
        )
```

### Per-Garment Export Package

Each garment produces a standardized export package:

```
garments/
  tshirt_regular_cotton/
    mesh.fbx              # geometry with bone weights
    normal.png            # baked wrinkle detail (2048×2048)
    uv_layout.png         # reference image of the UV/pattern layout
    metadata.json         # pattern parameters, fabric type, poly count
```

### Module 4 Deliverables

- [ ] UV generation from 2D pattern coordinates
- [ ] UV atlas packing for multi-panel garments
- [ ] Normal map baking (high-poly → low-poly)
- [ ] Bone weight transfer via Surface Deform
- [ ] Mesh decimation to target poly budget
- [ ] Geometry cleanup (remove doubles, fix normals)
- [ ] Standardized export package (FBX + textures + metadata)
- [ ] End-to-end test: simulated skirt → complete export package

---

## Module 5: Variant System

**Depends on:** Module 4 (export pipeline)
**Produces:** Compact garment library supporting continuous parameter variation
**Development time:** 2 weeks

### The Variant Explosion Problem

A T-shirt with 3 fit options, 3 lengths, 4 sleeve lengths, and 3 necklines across 5 body shapes produces 3 × 3 × 4 × 3 × 5 = 540 discrete variants. At 50KB per mesh, that's 27MB per garment type — manageable, but the combinatorics grow quickly when adding more garment types and parameters.

### PCA Compression

Instead of storing every variant as a separate mesh, simulate a carefully chosen subset of variants and compute a PCA basis over the vertex positions. At runtime, any variant can be reconstructed from a small number of PCA coefficients.

```python
import numpy as np
from sklearn.decomposition import PCA


def build_variant_basis(simulated_variants: list,
                        n_components: int = 10) -> dict:
    """Compute PCA basis from a set of simulated garment variant meshes.
    
    Args:
        simulated_variants: List of vertex arrays, each (V, 3)
        n_components: Number of PCA components to retain
    
    Returns:
        Basis dictionary with mean shape, components, and explained variance
    """
    # Flatten each variant to a single row vector
    V = simulated_variants[0].shape[0]  # number of vertices
    stacked = np.stack([v.reshape(-1) for v in simulated_variants])  # (N, V*3)
    
    pca = PCA(n_components=n_components)
    pca.fit(stacked)
    
    return {
        "mean_shape": pca.mean_.reshape(V, 3),          # (V, 3)
        "components": pca.components_.reshape(n_components, V, 3),  # (C, V, 3)
        "explained_variance": pca.explained_variance_ratio_,
        "n_vertices": V,
        "n_components": n_components,
    }


def encode_variant(basis: dict, variant_vertices: np.ndarray) -> np.ndarray:
    """Encode a variant mesh as PCA coefficients."""
    flat = variant_vertices.reshape(-1) - basis["mean_shape"].reshape(-1)
    components_flat = basis["components"].reshape(basis["n_components"], -1)
    return components_flat @ flat


def decode_variant(basis: dict, coefficients: np.ndarray) -> np.ndarray:
    """Reconstruct a variant mesh from PCA coefficients."""
    components_flat = basis["components"].reshape(basis["n_components"], -1)
    flat = basis["mean_shape"].reshape(-1) + coefficients @ components_flat
    return flat.reshape(-1, 3)
```

### Sampling Strategy

Not all parameter combinations need simulation. Choose samples that span the extremes of each parameter dimension, plus a few interior points.

```python
def generate_sample_points(param_ranges: dict,
                           samples_per_dim: int = 3) -> list:
    """Generate sample points across the parameter space.
    
    Uses the extreme values and midpoint of each dimension
    in a space-filling pattern (not full factorial).
    """
    import itertools
    
    # For each parameter, take min, mid, max
    dim_samples = {}
    for param, (lo, hi) in param_ranges.items():
        mid = (lo + hi) / 2
        dim_samples[param] = [lo, mid, hi]
    
    # Latin hypercube sampling or sparse grid instead of full factorial
    # to avoid combinatorial explosion
    
    # Simple approach: take all extremes + center
    samples = []
    
    # Center point
    samples.append({p: (lo+hi)/2 for p, (lo, hi) in param_ranges.items()})
    
    # One-at-a-time extremes
    center = {p: (lo+hi)/2 for p, (lo, hi) in param_ranges.items()}
    for param, (lo, hi) in param_ranges.items():
        for val in [lo, hi]:
            sample = dict(center)
            sample[param] = val
            samples.append(sample)
    
    return samples
```

### Storage Format

```
garments/
  tshirt/
    pca_basis.npz           # mean shape + PCA components (~200KB)
    variants/
      slim_cropped.json      # PCA coefficients (~100 bytes)
      regular_regular.json
      oversized_longline.json
      ...
    textures/
      normal.png             # shared across variants
      uv_layout.png
    metadata.json
```

10 PCA components × 5000 vertices × 4 bytes = 200KB per garment type.
With 10 components, you get continuous variation along 10 dimensions
at the storage cost of 10 shapes.

### Blend Shape Export for Game Engine

For engines that support morph targets / blend shapes natively (Unity, Unreal, Godot), export the PCA components as blend shapes:

```python
def export_pca_as_blend_shapes(basis: dict, 
                                base_mesh_path: str,
                                output_path: str):
    """Export PCA basis as FBX with blend shapes / shape keys.
    
    Component 0 → Shape Key "Variation_0"
    Component 1 → Shape Key "Variation_1"
    ...
    
    At runtime, the engine interpolates between shape keys
    using the PCA coefficients as weights.
    """
    # Load mean shape as base mesh
    bpy.ops.import_scene.fbx(filepath=base_mesh_path)
    obj = bpy.context.selected_objects[0]
    
    # Add basis shape key
    obj.shape_key_add(name="Basis", from_mix=False)
    
    # Add each PCA component as a shape key
    for i in range(basis["n_components"]):
        key = obj.shape_key_add(name=f"Variation_{i}", from_mix=False)
        component = basis["components"][i]
        
        for vi, vert in enumerate(key.data):
            vert.co = Vector(basis["mean_shape"][vi] + component[vi])
    
    # Export
    bpy.ops.export_scene.fbx(filepath=output_path, use_selection=True)
```

### Module 5 Deliverables

- [ ] Batch simulation script (run pipeline for multiple parameter combinations)
- [ ] PCA basis computation from simulated variants
- [ ] Variant encoding/decoding (parameters ↔ PCA coefficients)
- [ ] Intelligent sample point generation (avoid full factorial)
- [ ] Blend shape export for game engines
- [ ] Storage format specification
- [ ] Test: encode → decode round-trip preserves mesh quality within tolerance

---

## Module 6: Learned Runtime Deformation

**Depends on:** Module 5 (variant meshes), Module 2 (SMPL-X poses)
**Produces:** A neural network that predicts garment deformation from body pose
**Development time:** 3-4 weeks

### The Problem

Real-time cloth physics is expensive. A high-quality PBD solver requires 15-30ms per garment per frame. For a game with 10-20 visible characters, that's 150-600ms of physics per frame — far over budget.

Bone weight skinning (the standard alternative) deforms cloth as if it were skin. It follows the skeleton but produces no cloth-like behavior: no swinging, no settling, no wrinkle formation.

### The Solution: Learned Deformation

Simulate many garment-pose combinations offline. Train a small neural network that maps (garment parameters, body pose) to vertex offsets from the rest shape. At runtime, the network predicts deformation in under 5ms per garment — orders of magnitude faster than physics, orders of magnitude more realistic than skinning.

This approach is used in production by major studios. The canonical reference is TailorNet (Patel et al., CVPR 2020).

### Training Data Generation

The training data comes from your existing simulation pipeline (Modules 1-4), applied across a range of body poses.

```python
import numpy as np


def generate_training_data(garment_def: dict,
                           pose_sequences: np.ndarray,
                           body_shapes: list,
                           output_dir: str):
    """Generate training pairs: (garment_params, pose) → deformed_mesh.
    
    Args:
        garment_def: Garment definition dict
        pose_sequences: SMPL-X pose parameters, shape (N, 21, 3)
                       Use AMASS dataset for realistic pose distribution
        body_shapes: List of SMPL-X beta parameter arrays
        output_dir: Where to save training samples
    """
    samples = []
    
    for shape_idx, betas in enumerate(body_shapes):
        # Generate avatar for this body shape
        avatar_path = generate_smplx_avatar(betas)
        
        for pose_idx, pose in enumerate(pose_sequences):
            # Generate posed avatar
            posed_avatar_path = generate_smplx_avatar(betas, pose=pose)
            
            # Simulate garment on posed avatar
            result = simulate_garment(garment_def, posed_avatar_path)
            
            if result is not None:
                samples.append({
                    "body_shape": betas,
                    "body_pose": pose.flatten(),
                    "deformed_vertices": result.vertices,
                })
    
    # Save dataset
    np.savez(
        f"{output_dir}/training_data.npz",
        shapes=np.stack([s["body_shape"] for s in samples]),
        poses=np.stack([s["body_pose"] for s in samples]),
        vertices=np.stack([s["deformed_vertices"] for s in samples]),
    )
    
    return len(samples)
```

The AMASS dataset provides approximately 10,000 realistic SMPL body poses captured from motion data. Using 100-500 representative poses across 3-5 body shapes produces 300-2500 training samples — sufficient for a per-garment MLP.

### Network Architecture

```python
import torch
import torch.nn as nn


class GarmentDeformationNet(nn.Module):
    """Predicts per-vertex offsets from garment parameters and body pose.
    
    Input:  garment_params (G) + body_shape (10) + body_pose (63)
    Output: vertex offsets (V × 3)
    
    The network learns: rest_garment + offset = deformed_garment
    """
    
    def __init__(self,
                 n_garment_params: int = 10,
                 n_shape_params: int = 10,
                 n_pose_params: int = 63,    # 21 joints × 3
                 n_vertices: int = 5000,
                 hidden_dim: int = 256,
                 n_hidden_layers: int = 4):
        super().__init__()
        
        input_dim = n_garment_params + n_shape_params + n_pose_params
        output_dim = n_vertices * 3
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.n_vertices = n_vertices
    
    def forward(self, garment_params, shape_params, pose_params):
        x = torch.cat([garment_params, shape_params, pose_params], dim=-1)
        offsets = self.net(x)
        return offsets.view(-1, self.n_vertices, 3)


def train_deformation_model(data_path: str,
                             garment_params: np.ndarray,
                             n_epochs: int = 200,
                             batch_size: int = 32,
                             lr: float = 1e-3) -> nn.Module:
    """Train the deformation network on simulation data."""
    
    data = np.load(data_path)
    rest_vertices = data["vertices"][0]  # Rest pose as reference
    
    # Compute offsets from rest pose
    offsets = data["vertices"] - rest_vertices[np.newaxis]
    
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(np.broadcast_to(garment_params, (len(offsets), len(garment_params))), dtype=torch.float32),
        torch.tensor(data["shapes"], dtype=torch.float32),
        torch.tensor(data["poses"], dtype=torch.float32),
        torch.tensor(offsets.reshape(len(offsets), -1), dtype=torch.float32),
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = GarmentDeformationNet(
        n_garment_params=len(garment_params),
        n_vertices=rest_vertices.shape[0],
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(n_epochs):
        total_loss = 0
        for gp, sp, pp, target in loader:
            pred = model(gp, sp, pp).view(gp.shape[0], -1)
            loss = loss_fn(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={total_loss/len(loader):.6f}")
    
    return model


def export_to_onnx(model: nn.Module, output_path: str,
                   n_garment_params: int = 10,
                   n_shape_params: int = 10,
                   n_pose_params: int = 63):
    """Export trained model to ONNX for game engine deployment."""
    
    model.eval()
    
    dummy_garment = torch.randn(1, n_garment_params)
    dummy_shape = torch.randn(1, n_shape_params)
    dummy_pose = torch.randn(1, n_pose_params)
    
    torch.onnx.export(
        model,
        (dummy_garment, dummy_shape, dummy_pose),
        output_path,
        input_names=["garment_params", "shape_params", "pose_params"],
        output_names=["vertex_offsets"],
        dynamic_axes={
            "garment_params": {0: "batch"},
            "shape_params": {0: "batch"},
            "pose_params": {0: "batch"},
            "vertex_offsets": {0: "batch"},
        },
    )
```

### Runtime Inference

At runtime, the game engine loads the ONNX model and runs inference each frame:

```
Each frame:
  1. Animation system produces SMPL-X pose parameters
  2. Feed (garment_params, body_shape, pose) to ONNX model
  3. Receive vertex offsets
  4. Apply: final_mesh = rest_mesh + offsets
  5. Render
  
Cost: ~1-5ms per garment per frame
Quality: Simulated cloth behavior (wrinkles, swing, settling)
No runtime physics solver required
```

### Module 6 Deliverables

- [ ] Training data generation script (simulation across AMASS poses)
- [ ] AMASS pose dataset integration (download, sample, convert)
- [ ] `GarmentDeformationNet` implementation
- [ ] Training loop with validation
- [ ] ONNX export pipeline
- [ ] Inference benchmark (verify <5ms per garment on target hardware)
- [ ] Visual comparison: network output vs. ground-truth simulation
- [ ] Per-garment model training script

---

## Module 7: Game Engine Integration

**Depends on:** Module 4 (export format), Module 5 or 6 (variant system or learned deformation)
**Produces:** Complete in-game clothing customization system
**Development time:** 2-3 weeks

### Runtime Architecture

```
Player opens clothing customizer:
  1. Select garment type (T-Shirt, Pants, Skirt, etc.)
  2. Adjust sliders:
     - Fit (slim → oversized)
     - Length (cropped → longline)
     - Sleeve length
     - Neckline style
  3. Choose fabric and color
  4. Engine loads:
     - Rest mesh (FBX/glTF)
     - Deformation model (ONNX) — if Module 6 is implemented
     - OR PCA blend shapes — if using Module 5 only
     - Texture set
  5. Each frame:
     - Avatar animates → pose parameters update
     - Garment deforms via ONNX model or blend shapes
     - Texture applied with player customization
  6. Color/pattern changes are instant (texture swap)
     Fit/shape changes load different rest mesh or adjust PCA coefficients
```

### Two Integration Paths

**Path A: PCA Blend Shapes Only (Module 5, simpler)**

Garment shape varies via blend shape weights driven by customization sliders. Pose-dependent deformation relies on standard bone weight skinning. Good enough for stylized games; lacks realistic cloth motion.

**Path B: Learned Deformation (Module 6, recommended)**

Full pose-conditioned cloth behavior. The ONNX model replaces both blend shapes and bone weight skinning for cloth. Requires ONNX Runtime integration in the engine.

Both paths use the same export format (Module 4) and the same player-facing customization UI.

### ONNX Runtime Integration

```
Required in game engine:
  - ONNX Runtime library (~5MB)
  - Model file per garment type (~1-5MB)
  - Rest mesh per garment variant
  
Per-frame cost:
  - 1 ONNX inference call per visible garment
  - 1 mesh upload to GPU if vertices changed
  
Total overhead for 5 visible garments: ~5-25ms
```

### Texture Customization

Color and pattern changes are independent of the mesh pipeline and can be applied in real time:

```
Solid color:      Multiply base texture by chosen color
Pattern overlay:  Blend pattern texture onto base using UV coordinates
Fabric material:  Swap PBR material parameters (roughness, metallic, normal intensity)
```

Because UVs come from the 2D sewing pattern, textures applied to the flat pattern will wrap correctly on the 3D garment. A designer can paint on the pattern layout and see correct results on the body.

### Layering (Simplified)

For the initial implementation, use the standard game approach:

1. Each garment has a **body mask** indicating which body regions it covers.
2. When equipping a garment, hide the body mesh faces under it.
3. When layering (shirt + jacket), hide the shirt faces under the jacket.
4. No inter-garment physics. The outer garment is simulated or deformed as if the inner garment is part of the body.

This is how the majority of shipped games handle clothing layers.

### Module 7 Deliverables

- [ ] Garment loading system (rest mesh + textures)
- [ ] Customization UI (parameter sliders → mesh variation)
- [ ] Texture customization (color, pattern, fabric material)
- [ ] ONNX Runtime integration for learned deformation (Path B)
- [ ] OR blend shape interpolation from PCA coefficients (Path A)
- [ ] Body masking system for layered garments
- [ ] Equipment/wardrobe management (equip, unequip, save loadouts)
- [ ] Performance profiling on target platform

---

## Module 8: AI Pattern Generation

**Depends on:** Module 1 (pattern data model)
**Produces:** Sewing patterns generated from photos, sketches, or text descriptions
**Development time:** 2-3 weeks
**Status:** Optional accelerator — enhances the authoring workflow but is not required for the core pipeline

### Purpose

Manually authoring pattern template functions (Module 1) works well for standard garment types but becomes tedious for unusual designs, historical costumes, or garments copied from visual references. AI pattern generation provides alternative entry points: give the system a photo, a sketch, or a text description, and receive a sewing pattern as output. The result feeds into the same simulation pipeline as hand-authored patterns.

### Three Entry Points

**From a photograph** — a designer uploads a reference image of a garment and receives a reconstructed sewing pattern.

**From a sketch** — a designer draws a rough garment outline (front view, optionally side view) and receives a pattern that approximates the sketch.

**From a text description** — a designer describes a garment in natural language ("A-line midi skirt with six panels and a high waistband") and receives a pattern matching the description.

All three entry points produce output compatible with the `GarmentDefinition` data model from Module 1.

### Relevant Models and Tools

**SewFormer** (Liu et al., SIGGRAPH Asia 2023) reconstructs sewing patterns from single photographs. Trained on approximately one million synthetic image-pattern pairs. Produces 2D panel outlines with seam connectivity. The canonical tool for photo-to-pattern reconstruction.

**Panelformer** (Chen et al., WACV 2024) performs a similar photo-to-pattern task with a different architecture. Worth evaluating alongside SewFormer to determine which produces better results for your garment types.

**ChatGarment** (2025) takes text descriptions or sketches as input and outputs a JSON garment specification through a fine-tuned vision-language model. The output format is based on GarmentCode, which can be adapted to the Module 1 data model. The most production-ready option for text-based generation.

**SewingLDM** (Liu et al., ICCV 2025) uses multimodal latent diffusion to generate patterns from combinations of text, body shape parameters, and garment sketches. Handles conditional generation well — "T-shirt like X but in style Y for body Z."

**GarmentDiffusion** (Li, Yao, Wang, IJCAI 2025) generates centimeter-precise vectorized 3D sewing patterns from text, image, or partial pattern inputs using a multimodal diffusion transformer.

### Integration Architecture

```python
class PatternGenerator:
    """Unified interface for AI-assisted pattern generation.
    
    Each backend produces a GarmentDefinition compatible with Module 1.
    The generated pattern can be refined by adjusting parameters manually
    before feeding into the simulation pipeline.
    """
    
    def from_photo(self, image_path: str, 
                   body_shape: np.ndarray = None) -> GarmentDefinition:
        """Reconstruct a sewing pattern from a garment photograph.
        
        Uses SewFormer or Panelformer backend.
        Optionally takes SMPL-X body shape parameters to scale the pattern.
        """
        # Load and preprocess image
        # Run inference through SewFormer
        # Convert output panels to PatternPiece objects
        # Infer seam connectivity
        # Return GarmentDefinition
        pass
    
    def from_sketch(self, sketch_path: str,
                    garment_type: str = None) -> GarmentDefinition:
        """Generate a sewing pattern from a hand-drawn sketch.
        
        Uses ChatGarment or SewingLDM backend.
        Sketch should show front view; side view optional.
        """
        pass
    
    def from_text(self, description: str,
                  body_shape: np.ndarray = None) -> GarmentDefinition:
        """Generate a sewing pattern from a text description.
        
        Uses ChatGarment backend.
        Example: "Loose-fitting linen tunic with wide sleeves,
                  knee-length, boat neckline"
        """
        pass
    
    def refine(self, garment: GarmentDefinition,
               feedback: str) -> GarmentDefinition:
        """Iteratively refine a generated pattern based on text feedback.
        
        Example: "Make the sleeves wider and shorten the hem by 5cm"
        """
        pass
```

### GarmentCode Interop

ChatGarment and SewingLDM output patterns in GarmentCode format (Korosteleva & Sorkine-Hornung, SIGGRAPH Asia 2023). An adapter converts between GarmentCode and the Module 1 data model.

```python
def garmentcode_to_definition(gc_json: dict) -> GarmentDefinition:
    """Convert a GarmentCode JSON specification to a GarmentDefinition.
    
    GarmentCode represents patterns as parametric programs.
    This adapter evaluates the program and extracts the resulting
    2D panel geometry and seam connectivity.
    """
    from pygarment import ParametricPattern
    
    pattern = ParametricPattern.from_spec(gc_json)
    panels = pattern.evaluate()
    
    pieces = []
    for panel in panels:
        pieces.append(PatternPiece(
            name=panel.name,
            vertices=[PatternVertex(v[0], v[1]) for v in panel.vertices_cm],
            placement=infer_placement_from_panel(panel),
            fabric=FabricProperties.from_preset(FabricType.COTTON),
        ))
    
    seams = []
    for edge_pair in pattern.stitches:
        seams.append(Seam(
            edge_a=SeamEdge(edge_pair[0].panel, edge_pair[0].start, edge_pair[0].end),
            edge_b=SeamEdge(edge_pair[1].panel, edge_pair[1].start, edge_pair[1].end),
        ))
    
    return GarmentDefinition(name=gc_json.get("name", "generated"), pieces=pieces, seams=seams)


def definition_to_garmentcode(garment: GarmentDefinition) -> dict:
    """Convert a GarmentDefinition to GarmentCode JSON for interop
    with research tools and datasets."""
    pass
```

### Access to Training Data

GarmentCodeData (Korosteleva et al., ECCV 2024) provides 115,000+ synthetic garments with sewing patterns, simulation results, and body parameters. This dataset can be used to:

- Fine-tune pattern generation models for specific garment categories.
- Provide reference patterns for validation.
- Bootstrap the variant simulation dataset needed in Module 5.

### Workflow

```
Designer wants a new garment:

Option A (code-native, Module 1):
  Write create_garment() function → GarmentDefinition → simulate

Option B (photo reference, Module 8):
  Upload reference photo → SewFormer → GarmentDefinition → refine params → simulate

Option C (sketch, Module 8):
  Draw rough sketch → ChatGarment → GarmentDefinition → refine params → simulate

Option D (text, Module 8):
  Describe garment → ChatGarment → GarmentDefinition → refine params → simulate

All options converge at GarmentDefinition → same simulation pipeline.
```

### Module 8 Deliverables

- [ ] `PatternGenerator` interface with `from_photo`, `from_sketch`, `from_text` methods
- [ ] SewFormer integration for photo-to-pattern (inference only, using pretrained weights)
- [ ] ChatGarment integration for text/sketch-to-pattern
- [ ] GarmentCode adapter (bidirectional conversion)
- [ ] Validation: compare AI-generated patterns against hand-authored templates
- [ ] Refinement workflow: AI generates first draft, designer adjusts parameters
- [ ] Documentation: which model works best for which garment category

---

## Module 9: AI Fabric Parameter Prediction

**Depends on:** Module 1 (FabricProperties data model)
**Produces:** Automatic mapping from fabric descriptions to physical simulation parameters
**Development time:** 1 week
**Status:** Optional quality-of-life improvement

### Purpose

Module 1 includes a set of hardcoded fabric presets (cotton, denim, silk, leather, wool, linen). This works for common fabrics but breaks down when a designer wants "heavy brushed cotton twill" or "lightweight silk charmeuse" — variations within a fabric type that have meaningfully different physical behavior.

Module 9 learns a mapping from natural language fabric descriptions to the physical parameters used by the cloth simulator.

### Rule-Based Baseline

The simplest approach extends the existing preset table with sub-variants.

```python
FABRIC_PRESETS_EXTENDED = {
    # Cotton variants
    "cotton_jersey":        {"mass_per_area": 0.18, "bending": 0.3,  "damping": 4.0},
    "cotton_twill":         {"mass_per_area": 0.25, "bending": 1.2,  "damping": 6.0},
    "cotton_canvas":        {"mass_per_area": 0.35, "bending": 3.0,  "damping": 8.0},
    "cotton_voile":         {"mass_per_area": 0.08, "bending": 0.1,  "damping": 2.0},
    
    # Silk variants
    "silk_charmeuse":       {"mass_per_area": 0.08, "bending": 0.05, "damping": 1.5},
    "silk_dupioni":         {"mass_per_area": 0.15, "bending": 0.8,  "damping": 3.0},
    "silk_organza":         {"mass_per_area": 0.05, "bending": 0.3,  "damping": 1.0},
    
    # Denim variants
    "denim_lightweight":    {"mass_per_area": 0.30, "bending": 2.5,  "damping": 6.0},
    "denim_heavyweight":    {"mass_per_area": 0.55, "bending": 6.0,  "damping": 10.0},
    "denim_stretch":        {"mass_per_area": 0.35, "bending": 2.0,  "damping": 5.0,
                             "stretch_limit": 1.15},
    
    # Wool variants
    "wool_flannel":         {"mass_per_area": 0.25, "bending": 1.0,  "damping": 6.0},
    "wool_tweed":           {"mass_per_area": 0.40, "bending": 3.5,  "damping": 9.0},
    "wool_crepe":           {"mass_per_area": 0.20, "bending": 0.5,  "damping": 4.0},
    
    # Synthetic
    "polyester_chiffon":    {"mass_per_area": 0.06, "bending": 0.08, "damping": 1.5},
    "nylon_ripstop":        {"mass_per_area": 0.07, "bending": 0.2,  "damping": 2.0},
    "neoprene":             {"mass_per_area": 0.60, "bending": 8.0,  "damping": 12.0},
    
    # Historical / specialty
    "linen_heavy":          {"mass_per_area": 0.30, "bending": 1.5,  "damping": 6.0},
    "velvet":               {"mass_per_area": 0.35, "bending": 1.8,  "damping": 7.0},
    "burlap":               {"mass_per_area": 0.40, "bending": 4.0,  "damping": 9.0},
    "chainmail_cloth":      {"mass_per_area": 2.50, "bending": 0.1,  "damping": 15.0},
}
```

### Learned Model

For open-ended fabric descriptions, a small regression model maps text embeddings to physical parameters.

```python
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class FabricPredictor:
    """Predicts cloth simulation parameters from natural language fabric descriptions.
    
    Training data comes from published KES-F (Kawabata Evaluation System for Fabrics)
    measurements paired with fabric descriptions, supplemented by the CLOTH-FX
    and Garment-Pile datasets.
    """
    
    def __init__(self, model_path: str = None):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.predictor = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # mass, stiffness, bending, damping, friction, stretch
        )
        if model_path:
            self.predictor.load_state_dict(torch.load(model_path))
    
    def predict(self, description: str) -> FabricProperties:
        """Predict fabric properties from a text description.
        
        Example: "Heavy brushed cotton twill, like workwear denim but softer"
        """
        embedding = self.encoder.encode(description, convert_to_tensor=True)
        params = self.predictor(embedding)
        
        return FabricProperties(
            mass_per_area=float(params[0]),
            stiffness=float(params[1]),
            bending=float(params[2]),
            damping=float(params[3]),
            friction=float(params[4]),
            stretch_limit=float(params[5]),
        )
    
    def train(self, descriptions: list, measurements: np.ndarray, epochs: int = 100):
        """Train on paired (description, KES-F measurement) data."""
        embeddings = self.encoder.encode(descriptions, convert_to_tensor=True)
        targets = torch.tensor(measurements, dtype=torch.float32)
        
        optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        
        for epoch in range(epochs):
            pred = self.predictor(embeddings)
            loss = loss_fn(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Training Data Sources

- **KES-F measurements**: the standard textile evaluation system. Published datasets contain physical parameters for hundreds of fabric samples with descriptions.
- **CLOTH-FX dataset**: paired fabric videos with estimated physical parameters.
- **Garment-Pile dataset**: simulated garments with known fabric parameters.
- **Manual curation**: extend the rule-based table above, using each entry as a training pair (description → parameters).

### Practical Recommendation

Start with the extended rule-based table. It covers the most common cases and requires no ML infrastructure. Add the learned model later if designers frequently need unusual fabrics that don't match existing presets.

### Module 9 Deliverables

- [ ] Extended fabric preset table (20+ fabric sub-variants)
- [ ] Lookup function with fuzzy matching ("heavy cotton" → "cotton_canvas")
- [ ] `FabricPredictor` model architecture
- [ ] Training pipeline with KES-F data
- [ ] Validation against known fabric measurements
- [ ] Integration with Module 1 `FabricProperties.from_description()` method

---

## Module 10: Differentiable Physics Fitting

**Depends on:** Module 3 (simulation pipeline), Module 2 (SMPL-X avatar)
**Produces:** Automatic refinement of pattern parameters to match a target garment appearance
**Development time:** 3-4 weeks
**Status:** Advanced research integration — useful for precision fitting but not required for game development

### Purpose

Given a target — a photograph of a garment, a silhouette, or a 3D scan — automatically adjust the sewing pattern parameters so the simulated garment matches the target as closely as possible. This closes the loop between "I want it to look like THIS" and the parametric pattern definition.

The enabling technology is differentiable cloth simulation: a physics solver that supports gradient computation through the simulation, allowing pattern parameters to be optimized via backpropagation.

### How It Works

```
1. Start with an initial pattern (from Module 1 or Module 8)
2. Simulate the garment on an SMPL-X avatar
3. Render or extract the simulated garment's silhouette
4. Compare against the target (photo, silhouette, or 3D scan)
5. Compute a loss (silhouette difference, surface distance, etc.)
6. Backpropagate through the renderer and simulator to get gradients
   with respect to pattern parameters
7. Update pattern parameters
8. Repeat from step 2 until converged
```

### Architecture

```python
import torch


class DifferentiableClothFitter:
    """Fits pattern parameters to match a target garment appearance.
    
    Uses a differentiable cloth simulator to compute gradients of the
    appearance loss with respect to pattern parameters.
    """
    
    def __init__(self, simulator, renderer):
        """
        Args:
            simulator: A differentiable cloth simulator (e.g., DiffCloth,
                      Warp, or a custom XPBD implementation in PyTorch/Taichi)
            renderer: A differentiable renderer (e.g., nvdiffrast, PyTorch3D)
        """
        self.simulator = simulator
        self.renderer = renderer
    
    def fit_to_silhouette(self, 
                          initial_pattern: GarmentDefinition,
                          target_silhouette: torch.Tensor,
                          body_params: dict,
                          n_iterations: int = 100,
                          lr: float = 0.01) -> GarmentDefinition:
        """Optimize pattern parameters to match a target silhouette.
        
        Args:
            initial_pattern: Starting pattern (from Module 1 or 8)
            target_silhouette: Binary mask of target garment (H, W)
            body_params: SMPL-X shape and pose parameters
            n_iterations: Optimization steps
            lr: Learning rate for pattern parameter updates
        """
        # Convert pattern parameters to differentiable tensors
        params = pattern_to_differentiable(initial_pattern)
        params.requires_grad_(True)
        
        optimizer = torch.optim.Adam([params], lr=lr)
        
        for i in range(n_iterations):
            # Generate 2D pattern from current parameters
            pattern_mesh = generate_pattern_mesh(params)
            
            # Simulate draping (differentiable forward pass)
            draped_mesh = self.simulator.simulate(
                pattern_mesh, body_params, n_steps=200
            )
            
            # Render silhouette (differentiable)
            predicted_silhouette = self.renderer.render_silhouette(
                draped_mesh, camera_params
            )
            
            # Compute loss
            loss = silhouette_loss(predicted_silhouette, target_silhouette)
            
            # Optional: regularization to keep parameters reasonable
            loss += 0.01 * parameter_regularization(params, initial_pattern)
            
            # Backpropagate through renderer → simulator → pattern
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Iteration {i}: loss={loss.item():.6f}")
        
        # Convert optimized parameters back to GarmentDefinition
        return differentiable_to_pattern(params)
    
    def fit_to_3d_scan(self,
                       initial_pattern: GarmentDefinition,
                       target_mesh: torch.Tensor,
                       body_params: dict,
                       n_iterations: int = 100,
                       lr: float = 0.01) -> GarmentDefinition:
        """Optimize pattern parameters to match a target 3D scan.
        
        Uses Chamfer distance between simulated and target surfaces
        instead of silhouette comparison. More precise but requires
        a 3D scan as input.
        """
        params = pattern_to_differentiable(initial_pattern)
        params.requires_grad_(True)
        
        optimizer = torch.optim.Adam([params], lr=lr)
        
        for i in range(n_iterations):
            pattern_mesh = generate_pattern_mesh(params)
            draped_mesh = self.simulator.simulate(pattern_mesh, body_params)
            
            loss = chamfer_distance(draped_mesh.vertices, target_mesh.vertices)
            loss += 0.01 * parameter_regularization(params, initial_pattern)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return differentiable_to_pattern(params)
```

### Differentiable Simulation Backends

Several options exist for the differentiable cloth solver, ranging from research prototypes to production tools:

**Warp (NVIDIA)** — a Python framework for differentiable simulation. Includes cloth simulation primitives. Well-maintained, GPU-accelerated, actively developed.

**Taichi / DiffTaichi** — a differentiable programming language for physical simulation. Supports cloth via mass-spring or PBD formulations. Good Python integration.

**DiffCloth** (Li et al.) — a dedicated differentiable cloth simulator. Research code, less maintained, but specifically designed for this use case.

**Custom XPBD in PyTorch** — implement a simple Position-Based Dynamics solver directly in PyTorch. All tensor operations are automatically differentiable. A basic XPBD solver is approximately 200-300 lines of PyTorch code. Given the EvoGraph background, this is a realistic option.

### Differentiable Rendering

For silhouette-based fitting, a differentiable renderer produces gradients of the rendered image with respect to mesh vertex positions:

**nvdiffrast** (NVIDIA) — fast differentiable rasterization. Production-quality, GPU-accelerated.

**PyTorch3D** (Meta) — differentiable rendering and 3D operations in PyTorch. More Pythonic, easier to integrate, slightly slower.

### Relevant Research

**Dress-1-to-3** (Li et al., SIGGRAPH 2025) demonstrates the full pipeline: single photo → pattern reconstruction → differentiable simulation → parameter refinement. This paper validates that the approach works end-to-end for real garment photos.

**DrapeDiff** combines differentiable simulation with diffusion-based generation for garment design optimization.

### Use Cases

This module is most valuable for:

- **Reproducing real garments**: "Make my simulation match this photo exactly."
- **Fine-tuning AI-generated patterns**: Module 8 generates a rough pattern from a photo; Module 10 refines it until the simulation matches the photo precisely.
- **Historical costume reproduction**: given a reference image of a historical garment, recover the pattern that produces the correct drape.
- **Fit optimization**: adjust pattern parameters until the garment achieves a specific fit profile on a given body shape.

For standard game development with designer-authored garments, this module is not necessary. It becomes valuable when precision reproduction of real-world garments is required.

### Module 10 Deliverables

- [ ] Differentiable cloth solver (custom XPBD in PyTorch or Warp integration)
- [ ] Differentiable renderer integration (nvdiffrast or PyTorch3D)
- [ ] `DifferentiableClothFitter` with `fit_to_silhouette` and `fit_to_3d_scan`
- [ ] Pattern-to-differentiable and differentiable-to-pattern converters
- [ ] Silhouette loss and Chamfer distance implementations
- [ ] Parameter regularization to prevent degenerate patterns
- [ ] Validation: fit to known garment photos and compare against ground truth
- [ ] Integration test: Module 8 generates initial pattern → Module 10 refines it

---

## Development Summary

### Core Pipeline (Modules 1-7)

| Module | Depends On | Time Estimate | Key Deliverable |
|--------|-----------|---------------|-----------------|
| 1. Pattern Definition | — | 1 week | Data model + skirt/T-shirt templates |
| 2. Avatar System | Module 1 | 1 week | SMPL-X integration with landmarks |
| 3. Cloth Simulation | Modules 1, 2 | 2-3 weeks | Headless Blender pipeline |
| 4. Post-Processing | Module 3 | 1-2 weeks | Game-ready export with UV + normals |
| 5. Variant System | Module 4 | 2 weeks | PCA-compressed garment library |
| 6. Learned Deformation | Modules 2, 5 | 3-4 weeks | ONNX pose-conditioned model |
| 7. Game Integration | Modules 4, 5 or 6 | 2-3 weeks | In-game customization system |

### AI Extensions (Modules 8-10)

| Module | Depends On | Time Estimate | Key Deliverable |
|--------|-----------|---------------|-----------------|
| 8. AI Pattern Generation | Module 1 | 2-3 weeks | Photo/sketch/text → sewing pattern |
| 9. AI Fabric Prediction | Module 1 | 1 week | Text description → simulation parameters |
| 10. Differentiable Fitting | Modules 2, 3 | 3-4 weeks | Auto-refine patterns to match target |

### Dependency Graph

```
Module 1: Pattern Definition
    │
    ├── Module 2: Avatar System
    │       │
    │       ├── Module 3: Cloth Simulation
    │       │       │
    │       │       ├── Module 4: Post-Processing & Export
    │       │       │       │
    │       │       │       ├── Module 5: Variant System (PCA)
    │       │       │       │       │
    │       │       │       │       └── Module 6: Learned Deformation
    │       │       │       │
    │       │       │       └── Module 7: Game Integration
    │       │       │
    │       │       └── Module 10: Differentiable Fitting [optional]
    │       │
    │       └── Module 10: Differentiable Fitting [optional]
    │
    ├── Module 8: AI Pattern Generation [optional]
    │
    └── Module 9: AI Fabric Prediction [optional]
```

### Development Milestones

**Minimum viable pipeline:** Modules 1-4 (5-7 weeks). Produces static garment meshes from parametric patterns.

**With variant customization:** Add Module 5 (7-9 weeks). Players can adjust fit and style through PCA blend shapes.

**With realistic cloth motion:** Add Module 6 (10-13 weeks). Garments deform realistically with body animation, no runtime physics.

**Full game integration:** Add Module 7 (12-16 weeks total). Complete clothing customization system.

**With AI-assisted authoring:** Add Modules 8 and 9 (15-20 weeks total). Designers can create garments from photos, sketches, or text descriptions.

**With precision fitting:** Add Module 10 (18-24 weeks total). Automatic pattern refinement to match target garment appearance.

---

## Tools & Dependencies

### Required (Modules 1-4)

- Python 3.10+
- Blender 3.6+ (headless scripting, cloth simulation)
- NumPy, SciPy (mesh processing, PCA)
- `smplx` Python package (SMPL-X body model)

### Required for Module 5

- scikit-learn (PCA computation)

### Required for Module 6

- PyTorch (training the deformation model)
- ONNX + ONNX Runtime (model export and deployment)
- AMASS dataset (realistic pose sequences)

### Required for Module 8

- Pretrained model weights (SewFormer, ChatGarment, or SewingLDM)
- `pygarment` Python package (GarmentCode interop)
- GPU with sufficient VRAM for inference (8GB+ recommended)

### Required for Module 9

- sentence-transformers (text embedding for fabric description)
- KES-F measurement data (training pairs)
- PyTorch (small regression model)

### Required for Module 10

- Differentiable simulation backend (Warp, Taichi, or custom PyTorch XPBD)
- Differentiable renderer (nvdiffrast or PyTorch3D)
- GPU with sufficient VRAM for differentiable simulation (12GB+ recommended)

### Optional

- **HiPhyEngine** — Blender add-on, GPU cloth solver with proper sewing support
- **trimesh** — mesh I/O and processing utilities
- **GarmentCode / pygarment** — standardized pattern format for interoperability
- **GarmentCodeData** (ECCV 2024) — 115,000+ synthetic garments for training/validation

### CLI Commands

```bash
# Module 1: Generate a garment definition
python create_garment.py --type tshirt --fit regular --length regular --output tshirt.json

# Module 3: Simulate garment on avatar (headless)
blender --background --python simulate.py -- \
    --garment tshirt.json --avatar smplx_average.fbx --output output/tshirt.fbx

# Module 5: Batch generate variants
blender --background --python batch_simulate.py -- \
    --garment tshirt.json --config variants.json --output-dir output/tshirt/

# Module 5: Compute PCA basis from variants
python build_pca_basis.py --input-dir output/tshirt/ --components 10 --output tshirt_basis.npz

# Module 6: Generate training data for deformation model
python generate_training_data.py \
    --garment tshirt.json --poses data/amass/ --output data/training/tshirt/

# Module 6: Train deformation model
python train_deformation.py \
    --data data/training/tshirt/ --epochs 200 --output models/tshirt.onnx

# Module 8: Generate pattern from photo
python generate_pattern.py --from-photo reference.jpg --output generated_garment.json

# Module 8: Generate pattern from text
python generate_pattern.py --from-text "A-line midi skirt, 6 panels, high waist" --output generated_garment.json

# Module 9: Predict fabric parameters
python predict_fabric.py --description "heavy brushed cotton twill" --output fabric_params.json

# Module 10: Fit pattern to target photo
python fit_pattern.py \
    --initial generated_garment.json --target-photo reference.jpg \
    --avatar smplx_average.fbx --output fitted_garment.json --iterations 100
```
