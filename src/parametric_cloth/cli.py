"""Command-line entry point for generating garment definitions.

Examples::

    create-garment --type tshirt --fit regular --length regular --output tshirt.json
    create-garment --type skirt --panels 6 --flare 1.8 --output skirt.json
    create-garment --type cape --output cape.json
"""

from __future__ import annotations

import argparse
import sys

from .customization import TShirtCustomization
from .fabric import FabricType
from .serialization import garment_to_json, save_garment
from .templates import create_cape, create_skirt, create_tshirt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="create-garment",
        description="Generate a parametric garment definition as JSON.",
    )
    parser.add_argument(
        "--type", required=True, choices=["tshirt", "skirt", "cape"],
        help="garment type to generate",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="output JSON path (default: stdout)",
    )
    parser.add_argument(
        "--fabric", default=None,
        choices=[f.value for f in FabricType],
        help="fabric preset",
    )

    # T-shirt customization (player-facing).
    parser.add_argument("--fit", default="regular",
                        choices=["slim", "regular", "loose", "oversized"])
    parser.add_argument("--length", default="regular",
                        choices=["cropped", "regular", "longline"])
    parser.add_argument("--sleeve-length", default="short",
                        choices=["cap", "short", "three_quarter", "long"])
    parser.add_argument("--neckline", default="crew",
                        choices=["crew", "v_neck", "scoop", "boat"])

    # Skirt parameters.
    parser.add_argument("--panels", type=int, default=4)
    parser.add_argument("--flare", type=float, default=1.3)
    parser.add_argument("--length-cm", type=float, default=None,
                        help="skirt/cape length in cm (overrides type default)")

    return parser


def _build_garment(args: argparse.Namespace):
    if args.type == "tshirt":
        cust = TShirtCustomization(
            fit=args.fit,
            length=args.length,
            sleeve_length=args.sleeve_length,
            neckline=args.neckline,
            fabric=args.fabric or "cotton",
        )
        return cust.build()

    if args.type == "skirt":
        kwargs = dict(panels=args.panels, flare=args.flare)
        if args.fabric:
            kwargs["fabric"] = FabricType(args.fabric)
        if args.length_cm is not None:
            kwargs["length"] = args.length_cm
        return create_skirt(**kwargs)

    if args.type == "cape":
        kwargs = {}
        if args.fabric:
            kwargs["fabric"] = FabricType(args.fabric)
        if args.length_cm is not None:
            kwargs["length"] = args.length_cm
        return create_cape(**kwargs)

    raise ValueError(f"unknown garment type {args.type!r}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        garment = _build_garment(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    issues = garment.validate()
    if issues:
        print("error: generated garment failed validation:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    if args.output:
        save_garment(garment, args.output)
        print(f"wrote {garment.name} ({len(garment.pieces)} pieces, "
              f"{len(garment.seams)} seams) to {args.output}")
    else:
        print(garment_to_json(garment))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
