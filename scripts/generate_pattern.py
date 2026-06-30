#!/usr/bin/env python3
"""Generate a sewing pattern from a photo or text description (Module 8).

    python scripts/generate_pattern.py --from-text "A-line midi skirt, 6 panels, high waist" \
        --output generated_garment.json

Requires a configured backend (SewFormer / ChatGarment); see
``parametric_cloth.ai_pattern.PatternGenerator`` for how to inject one. Without
a backend installed, this will raise a clear ModuleNotFoundError explaining
what's missing.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from parametric_cloth.ai_pattern import PatternGenerator
from parametric_cloth.serialization import save_garment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-photo", help="path to a garment reference photo")
    group.add_argument("--from-text", help="text description of the garment")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    generator = PatternGenerator()
    try:
        if args.from_text:
            garment = generator.from_text(args.from_text)
        else:
            image = _load_image(args.from_photo)
            garment = generator.from_photo(image)
    except ModuleNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    save_garment(garment, args.output)
    print(f"wrote {garment.name} to {args.output}")
    return 0


def _load_image(path: str) -> np.ndarray:
    try:
        import imageio.v3 as iio
        return iio.imread(path)
    except ImportError:
        raise ModuleNotFoundError(
            "reading images needs 'imageio' (or swap in your own loader)"
        ) from None


if __name__ == "__main__":
    sys.exit(main())
