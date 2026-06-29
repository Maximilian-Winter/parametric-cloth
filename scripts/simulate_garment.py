#!/usr/bin/env python3
"""Headless Blender entry point for cloth simulation (Module 3).

Run inside Blender::

    blender --background --python scripts/simulate_garment.py -- \
        --garment skirt_4panel.json \
        --avatar smplx_average.obj \
        --output output/skirt_4panel.fbx \
        --frames 250

Arguments after ``--`` are parsed here; everything before it belongs to Blender.
"""

from __future__ import annotations

import argparse
import os
import sys


def _args_after_double_dash(argv: list[str]) -> list[str]:
    return argv[argv.index("--") + 1:] if "--" in argv else []


def _ensure_package_importable() -> None:
    """Add the repo's ``src`` to sys.path so this works under Blender's Python."""
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

    from parametric_cloth.serialization import load_garment
    from parametric_cloth.simulation import SimulationConfig
    from parametric_cloth.simulation.blender_sim import simulate_garment

    parser = argparse.ArgumentParser(description="Simulate a garment in Blender.")
    parser.add_argument("--garment", required=True, help="garment JSON path")
    parser.add_argument("--avatar", required=True, help="avatar mesh (obj/glb/fbx)")
    parser.add_argument("--output", required=True, help="output mesh (fbx/glb)")
    parser.add_argument("--package-dir", default=None,
                        help="if set, also write the full Module 4 export package")
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--subdivide-levels", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args(_args_after_double_dash(sys.argv))

    garment = load_garment(args.garment)
    issues = garment.validate()
    if issues:
        print("garment failed validation:", *issues, sep="\n  ", file=sys.stderr)
        return 1

    config = SimulationConfig(
        frames=args.frames,
        subdivide_levels=args.subdivide_levels,
        max_retries=args.max_retries,
    )
    ok = simulate_garment(garment, args.avatar, args.output, config,
                          package_dir=args.package_dir)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
