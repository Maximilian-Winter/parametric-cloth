#!/usr/bin/env python3
"""Generate deformation training data by draping a garment across poses (Blender).

Run inside Blender::

    blender --background --python scripts/generate_training_data.py -- \
        --garment tshirt.json --poses data/amass/sample.npz --n-poses 200 \
        --shapes average athletic heavy --output data/training/tshirt/training_data.npz

For each (body shape, pose) it generates a posed SMPL-X avatar, drapes the
garment, and records the deformed vertices. The first successful sample (zero
pose, first shape) is used as the rest shape. Topology must be constant, so this
varies pose/shape only -- not garment structure.
"""

from __future__ import annotations

import argparse
import os
import sys


def _args_after_double_dash(argv: list[str]) -> list[str]:
    return argv[argv.index("--") + 1:] if "--" in argv else []


def _ensure_package_importable() -> None:
    try:
        import parametric_cloth  # noqa: F401
        return
    except ModuleNotFoundError:
        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(os.path.dirname(here), "src")
        if os.path.isdir(src):
            sys.path.insert(0, src)


def main() -> int:
    _ensure_package_importable()

    import numpy as np
    from parametric_cloth.serialization import load_garment
    from parametric_cloth.simulation import SimulationConfig
    from parametric_cloth.simulation.blender_sim import (
        add_cloth_physics, assemble_garment, build_cloth_object, clear_scene,
        garment_vertices_world, load_avatar, run_simulation, weld_seams,
    )
    from parametric_cloth.avatar.body import SMPLXConfig, generate_smplx_avatar, export_avatar
    from parametric_cloth.avatar.placement import avatar_mesh_from_bpy, place_garment
    from parametric_cloth.avatar.shapes import get_body_shape
    from parametric_cloth.deformation.dataset import DeformationDataset
    from parametric_cloth.deformation.poses import load_amass, zero_pose

    parser = argparse.ArgumentParser(description="Generate deformation training data.")
    parser.add_argument("--garment", required=True)
    parser.add_argument("--poses", required=True, help="AMASS-style .npz")
    parser.add_argument("--n-poses", type=int, default=200)
    parser.add_argument("--shapes", nargs="+", default=["average"])
    parser.add_argument("--garment-params", nargs="*", type=float, default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(_args_after_double_dash(sys.argv))

    garment = load_garment(args.garment)
    poses = load_amass(args.poses, n=args.n_poses)
    poses = np.vstack([zero_pose()[None], poses])      # ensure a rest pose first
    config = SimulationConfig()

    def drape(avatar_path):
        clear_scene()
        avatar = load_avatar(avatar_path, config)
        transforms = place_garment(garment, avatar_mesh_from_bpy(avatar))
        assembled = assemble_garment(garment, transforms, levels=config.subdivide_levels)
        obj = build_cloth_object(assembled)
        weld_seams(obj, assembled)
        add_cloth_physics(obj, garment.pieces[0].fabric, config)
        run_simulation(obj, config)
        return garment_vertices_world(obj)

    samples, rest = [], None
    tmp = os.path.join(os.path.dirname(args.output) or ".", "_avatar.obj")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    for shape_name in args.shapes:
        betas = get_body_shape(shape_name)
        for pose in poses:
            avatar_mesh = generate_smplx_avatar(betas, pose=pose, config=SMPLXConfig())
            export_avatar(avatar_mesh, tmp)
            verts = drape(tmp)
            if rest is None:
                rest = verts
            if verts.shape != rest.shape:
                print(f"[skip] topology drift for shape={shape_name}")
                continue
            samples.append({
                "garment_params": args.garment_params,
                "body_shape": betas,
                "body_pose": pose,
                "vertices": verts,
            })

    if rest is None or len(samples) < 2:
        print("error: not enough samples generated", file=sys.stderr)
        return 1

    DeformationDataset.from_samples(rest, samples).save(args.output)
    print(f"wrote {len(samples)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
