#!/usr/bin/env python3
"""
generate_dome_reference.py -- build the plate-centered dome point cloud.

this produces data/reference/dome_cloud.ply: a synthetic point cloud
representing the hemisphere carved out by the arc carriage (inner
surface) plus the plate disk underneath. the pipeline's dome filter
uses this as a background reference.

run once after assembling the hardware, or whenever the arc geometry
changes. output is saved to data/reference/dome_cloud.ply by default.

usage:
    python scripts/generate_dome_reference.py
    python scripts/generate_dome_reference.py --spacing 0.003
    python scripts/generate_dome_reference.py --no-plate
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from old.dome_filter import build_dome_reference, DOME_CLOUD_PATH


def main():
    p = argparse.ArgumentParser(description="generate dome reference cloud")
    p.add_argument("--radius", type=float, default=0.275,
                   help="inner dome radius in meters (default 0.275 = 275mm)")
    p.add_argument("--center-z", type=float, default=0.050,
                   help="dome center height above plate in meters (default 0.050)")
    p.add_argument("--spacing", type=float, default=0.005,
                   help="point spacing on dome surface (default 0.005 = 5mm)")
    p.add_argument("--no-plate", action="store_true",
                   help="skip generating plate disk points")
    p.add_argument("--plate-radius", type=float, default=0.2159,
                   help="plate disk radius in meters (default 0.2159 = 8.5in)")
    p.add_argument("--output", type=str, default=str(DOME_CLOUD_PATH),
                   help=f"output path (default {DOME_CLOUD_PATH})")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from pathlib import Path
    build_dome_reference(
        radius_m=args.radius,
        center_z_m=args.center_z,
        point_spacing_m=args.spacing,
        include_plate=not args.no_plate,
        plate_radius_m=args.plate_radius,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()