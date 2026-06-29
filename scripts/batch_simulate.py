#!/usr/bin/env python3
"""Batch-simulate garment variants in Blender, saving a vertex dataset for PCA.

Run inside Blender::

    blender --background --python scripts/batch_simulate.py -- \
        --garment tshirt.json --avatar smplx_average.obj \
        --config variants.json --output-dir output/tshirt

``variants.json`` is a list of parameter dicts (see
``parametric_cloth.variants.sampling``). Each is applied to the garment, draped,
and its deformed vertices collected into ``<output-dir>/variants.npz`` for
``build_pca_basis.py``.

Topology note: PCA requires identical topology across variants, so vary only
*continuous* parameters (length, flare, ease...) and keep structural ones (panel
count, subdivide levels) fixed. Variants whose vertex count differs from the
first successful one are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
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
    from parametric_cloth.avatar.placement import avatar_mesh_from_bpy, place_garment

    parser = argparse.ArgumentParser(description="Batch-simulate garment variants.")
    parser.add_argument("--garment", required=True)
    parser.add_argument("--avatar", required=True)
    parser.add_argument("--config", required=True, help="JSON list of param dicts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--apply", default=None,
                        help="dotted path to a builder fn: module:function "
                             "taking **params -> GarmentDefinition (overrides --garment)")
    args = parser.parse_args(_args_after_double_dash(sys.argv))

    base_garment = load_garment(args.garment)
    samples = json.loads(open(args.config).read())
    config = SimulationConfig()
    os.makedirs(args.output_dir, exist_ok=True)

    collected, names, kept_params = [], [], []
    ref_count = None

    for i, params in enumerate(samples):
        garment = _apply_params(base_garment, params, args.apply)
        clear_scene()
        avatar = load_avatar(args.avatar, config)
        transforms = place_garment(garment, avatar_mesh_from_bpy(avatar))
        assembled = assemble_garment(garment, transforms,
                                     levels=config.subdivide_levels)
        obj = build_cloth_object(assembled)
        weld_seams(obj, assembled)
        add_cloth_physics(obj, garment.pieces[0].fabric, config)
        run_simulation(obj, config)

        verts = garment_vertices_world(obj)
        if ref_count is None:
            ref_count = len(verts)
        if len(verts) != ref_count:
            print(f"[skip] variant {i}: {len(verts)} verts != {ref_count} "
                  f"(topology changed)")
            continue

        collected.append(verts)
        names.append(f"variant_{i}")
        kept_params.append(params)

    if len(collected) < 2:
        print("error: fewer than 2 consistent variants collected", file=sys.stderr)
        return 1

    out = os.path.join(args.output_dir, "variants.npz")
    np.savez(out, vertices=np.stack(collected),
             names=np.array(names),
             parameters=json.dumps(kept_params))
    print(f"wrote {len(collected)} variants to {out}")
    return 0


def _apply_params(base_garment, params, apply_spec):
    if not apply_spec:
        return base_garment
    import importlib
    module_name, func_name = apply_spec.split(":")
    func = getattr(importlib.import_module(module_name), func_name)
    return func(**params)


if __name__ == "__main__":
    sys.exit(main())
