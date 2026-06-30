#!/usr/bin/env python3
"""Predict cloth simulation parameters from a fabric description.

    python scripts/predict_fabric.py --description "heavy brushed cotton twill" \
        --output fabric_params.json

Tries the extended preset table with fuzzy matching first (no ML dependency).
If nothing matches closely and --predictor-weights is given, falls back to the
learned FabricPredictor (needs sentence-transformers + torch).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from parametric_cloth.fabric import FabricProperties
from parametric_cloth.fabric_ai import FabricPredictor, find_closest_preset
from parametric_cloth.fabric_ai.presets import fabric_properties_for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--description", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--predictor-weights", default=None,
                        help="path to trained FabricPredictor weights (.npz)")
    args = parser.parse_args(argv)

    match = find_closest_preset(args.description)
    if match is not None:
        props = fabric_properties_for(match)
        source = f"preset:{match}"
    elif args.predictor_weights:
        predictor = FabricPredictor.load(args.predictor_weights)
        props = predictor.predict(args.description)
        source = "learned"
    else:
        print(f"error: no preset close enough to '{args.description}' "
              f"and no --predictor-weights given", file=sys.stderr)
        return 1

    result = {"source": source, **{
        k: v if not hasattr(v, "value") else v.value
        for k, v in dataclasses.asdict(props).items()
    }}
    text = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"wrote {args.output} ({source})")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
