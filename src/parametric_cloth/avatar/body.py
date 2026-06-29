"""SMPL-X avatar generation and export.

These functions require the ``smplx`` package, ``torch``, and SMPL-X model
weights, and (for FBX) Blender. They are imported lazily so the rest of the
``avatar`` package -- the pure-numpy geometry -- stays usable without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .mesh import AvatarMesh
from .shapes import N_BETAS, get_body_shape


@dataclass
class SMPLXConfig:
    """Configuration for the SMPL-X body model."""

    model_path: str = "models/smplx"
    gender: str = "neutral"          # "neutral" | "male" | "female"
    num_betas: int = N_BETAS


def generate_smplx_avatar(
    betas: np.ndarray | str,
    pose: Optional[np.ndarray] = None,
    *,
    config: SMPLXConfig | None = None,
) -> AvatarMesh:
    """Generate an SMPL-X mesh for the given shape (and optional pose).

    Args:
        betas: a (num_betas,) beta vector, or the name of a sample in
            ``BODY_SHAPE_SAMPLES`` (e.g. ``"athletic"``).
        pose: optional body pose as (21, 3) or (63,) axis-angle joint rotations.
        config: SMPL-X model configuration.

    Returns:
        An :class:`AvatarMesh` with the posed vertices and SMPL-X faces.
    """
    config = config or SMPLXConfig()
    if isinstance(betas, str):
        betas = get_body_shape(betas)
    betas = np.asarray(betas, dtype=float).reshape(-1)

    import torch  # lazy
    import smplx  # lazy

    body_pose = None
    if pose is not None:
        body_pose = torch.tensor(
            np.asarray(pose, dtype=float).reshape(1, -1), dtype=torch.float32
        )

    model = smplx.create(
        model_path=config.model_path,
        model_type="smplx",
        gender=config.gender,
        num_betas=config.num_betas,
        use_pca=False,
    )
    output = model(
        betas=torch.tensor(betas[: config.num_betas], dtype=torch.float32).unsqueeze(0),
        body_pose=body_pose,
    )

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    faces = np.asarray(model.faces, dtype=np.int64)
    name = "smplx_" + "_".join(f"{b:.1f}" for b in betas[:3])
    return AvatarMesh(vertices=vertices, faces=faces, name=name)


def export_avatar(mesh: AvatarMesh, path: str) -> str:
    """Export an avatar mesh to disk.

    ``.obj``, ``.ply`` and ``.glb`` are written via ``trimesh``. ``.fbx`` is not
    supported by ``trimesh``; for FBX, import the ``.obj``/``.glb`` into Blender
    (Module 3 already runs inside Blender) or export from there. Returns ``path``.
    """
    lower = path.lower()
    if lower.endswith(".fbx"):
        raise NotImplementedError(
            "FBX export requires Blender. Export .obj/.glb here and convert "
            "inside the Blender simulation step (Module 3), which imports the "
            "avatar directly."
        )
    import trimesh  # lazy

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    tm.export(path)
    return path
