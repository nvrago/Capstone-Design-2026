"""
densify_dome.py -- fix the sparse dome reference by synthesizing
the missing plate-interior points.

the onshape export of dome_cloud.ply has 616 points on the plate-top
plane, but they're all at radius 0.23-0.27m (an outer ring), with zero
coverage of the plate interior. this breaks dome subtraction because
captured plate points at radius 0-0.16m have no nearby dome points
to match against.

this script:
    1. loads the original dome (hemisphere + plate rim ring)
    2. synthesizes a grid of points on the plate-top plane (z=0) 
       at 2mm spacing, covering the full disc interior
    3. merges the two
    4. writes dome_cloud_dense.ply

run once. then point pipeline config at the dense file.

usage:
    python scripts/densify_dome.py
    python scripts/densify_dome.py --spacing 0.001 --output data/reference/dome_cloud_fine.ply
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)


def densify_plate_disc(
    inner_radius_m: float,
    outer_radius_m: float,
    z_m: float,
    spacing_m: float,
) -> np.ndarray:
    """
    synthesize a uniform grid of points on a flat annular disc at height z_m.
    covers radius inner_radius_m to outer_radius_m. inner_radius_m=0 gives
    a full disc; nonzero gives a ring (useful if the original already has
    outer-edge points and only the interior needs filling).
    """
    xs = np.arange(-outer_radius_m, outer_radius_m + spacing_m / 2, spacing_m)
    ys = np.arange(-outer_radius_m, outer_radius_m + spacing_m / 2, spacing_m)
    xx, yy = np.meshgrid(xs, ys)
    r = np.sqrt(xx * xx + yy * yy)
    mask = (r >= inner_radius_m) & (r <= outer_radius_m)
    pts = np.stack([xx[mask], yy[mask], np.full(mask.sum(), z_m)], axis=-1)
    return pts


def parse_args():
    p = argparse.ArgumentParser(
        description="densify dome reference with synthetic plate-interior points"
    )
    p.add_argument("--input", type=str,
                   default="data/reference/dome_cloud.ply",
                   help="path to original sparse dome ply")
    p.add_argument("--output", type=str,
                   default="data/reference/dome_cloud_dense.ply",
                   help="path to write densified dome ply")
    p.add_argument("--spacing", type=float, default=0.002,
                   help="grid spacing for synthesized disc, meters (default 0.002)")
    p.add_argument("--disc-radius", type=float, default=0.230,
                   help="outer radius for synthesized disc interior, meters. "
                        "defaults to 0.230 to meet the existing dome's outer "
                        "ring at 0.23-0.27m.")
    p.add_argument("--plate-z", type=float, default=0.000,
                   help="height of plate-top plane, meters (default 0.000)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        logger.error(f"input dome not found: {in_path}")
        sys.exit(1)

    orig = o3d.io.read_point_cloud(str(in_path))
    orig_pts = np.asarray(orig.points)
    logger.info(f"loaded original dome: {len(orig_pts)} points from {in_path}")

    # synthesize plate-interior points from center out to just inside the
    # existing outer ring. inner radius 0 = full disc coverage.
    disc = densify_plate_disc(
        inner_radius_m=0.0,
        outer_radius_m=args.disc_radius,
        z_m=args.plate_z,
        spacing_m=args.spacing,
    )
    logger.info(f"synthesized plate disc: {len(disc)} points "
                f"(spacing={args.spacing*1000:.1f}mm, "
                f"radius<={args.disc_radius*1000:.0f}mm)")

    # merge
    merged_pts = np.vstack([orig_pts, disc])
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(merged_pts)

    # preserve colors if original had them; give synthesized points a neutral gray
    if orig.has_colors():
        orig_colors = np.asarray(orig.colors)
        disc_colors = np.tile([0.5, 0.5, 0.5], (len(disc), 1))
        merged_colors = np.vstack([orig_colors, disc_colors])
        merged.colors = o3d.utility.Vector3dVector(merged_colors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), merged)
    logger.info(f"wrote densified dome: {len(merged_pts)} points to {out_path}")
    logger.info(f"update pipeline config:")
    logger.info(f"    dome_reference_path: \"{out_path}\"")


if __name__ == "__main__":
    main()