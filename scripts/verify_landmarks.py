#!/usr/bin/env python3
"""Verify SMPL-X landmark indices against real model weights.

The indices in ``parametric_cloth.avatar.landmarks`` are provisional. This
script generates SMPL-X meshes for every body-shape sample, runs the registry
sanity checks, and writes one OBJ per shape with the landmark vertices marked,
so they can be confirmed visually before flipping ``LANDMARKS_VERIFIED`` to True.

Requires ``smplx``, ``torch``, ``trimesh`` and SMPL-X weights::

    python scripts/verify_landmarks.py --model-path models/smplx --out-dir /tmp/landmarks
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from parametric_cloth.avatar import (
    BODY_SHAPE_SAMPLES,
    SMPLX_LANDMARKS,
    SMPLXConfig,
    generate_smplx_avatar,
    verify_landmark_indices,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/smplx")
    parser.add_argument("--gender", default="neutral")
    parser.add_argument("--out-dir", default="landmark_check")
    args = parser.parse_args(argv)

    config = SMPLXConfig(model_path=args.model_path, gender=args.gender)

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    ok = True
    for name in BODY_SHAPE_SAMPLES:
        mesh = generate_smplx_avatar(name, config=config)
        issues = verify_landmark_indices(mesh)
        status = "OK" if not issues else "ISSUES"
        print(f"[{status}] shape '{name}': {mesh.n_vertices} vertices")
        for issue in issues:
            ok = False
            print(f"    - {issue}")

        _write_marked_obj(mesh, f"{args.out_dir}/{name}.obj")

    print()
    if ok:
        print("All checks passed. Inspect the OBJ files to confirm landmarks "
              "visually, then set LANDMARKS_VERIFIED = True.")
    else:
        print("Some checks failed -- fix the indices in avatar/landmarks.py.")
    return 0 if ok else 1


def _write_marked_obj(mesh, path: str) -> None:
    """Write the body plus small markers at each landmark for visual inspection."""
    import trimesh

    body = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    markers = [body]
    for name, idx in SMPLX_LANDMARKS.items():
        sphere = trimesh.creation.uv_sphere(radius=0.01)
        sphere.apply_translation(mesh.vertices[idx])
        markers.append(sphere)
    trimesh.util.concatenate(markers).export(path)
    print(f"    wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
