"""
verify_pose.py -- one-time diagnostic for the camera-to-plate transform.

captures a single frame at a chosen arc angle and writes three .ply files:
    camera_frame.ply         raw cloud in the camera's own frame
    plate_frame.ply          same cloud transformed to plate coords
    dome_reference.ply       the onshape dome, copied next to them

open all three in CloudCompare. the plate-frame cloud and the dome
reference should overlap tightly. if they do, the pose math is right
and you can delete this script. if they don't, read the "how to
interpret" section at the bottom.

usage (with d405 connected):
    python scripts/verify_pose.py --angle 90 --arc-radius 0.300

usage (from a .bag file, no hardware needed):
    python scripts/verify_pose.py --bag data/recordings/test.bag --angle 90

flags:
    --angle           arc angle in degrees (default 90, straight overhead)
    --arc-radius      arc radius in meters (default 0.300)
    --arc-center-z    arc center height above plate, meters (default 0.000)
    --dome            path to dome reference ply
                      (default data/reference/dome_cloud.ply)
    --out-dir         where to write the three plys
                      (default data/pose_verify/)

this script is intentionally minimal, decoupled from pipeline.py,
and has no config dependencies. you can run it on a fresh checkout
and it will tell you yes/no on whether the pose transform is correct.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# make src importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scanner.capture import RealSenseCapture, camera_to_plate

logger = logging.getLogger(__name__)


def stats(name: str, pcd: o3d.geometry.PointCloud) -> dict:
    """print bounding box and centroid for a cloud."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        logger.warning(f"{name}: empty cloud")
        return {}
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    cent = pts.mean(axis=0)
    logger.info(
        f"{name}: {len(pts)} pts | "
        f"min=[{mn[0]:+.3f},{mn[1]:+.3f},{mn[2]:+.3f}] | "
        f"max=[{mx[0]:+.3f},{mx[1]:+.3f},{mx[2]:+.3f}] | "
        f"cent=[{cent[0]:+.3f},{cent[1]:+.3f},{cent[2]:+.3f}]"
    )
    return {"min": mn, "max": mx, "cent": cent}


def parse_args():
    p = argparse.ArgumentParser(description="verify camera-to-plate pose")
    p.add_argument("--angle", type=float, default=90.0,
                   help="arc angle for this capture, degrees")
    p.add_argument("--arc-radius", type=float, default=0.300,
                   help="arc radius, meters")
    p.add_argument("--arc-center-z", type=float, default=0.000,
                   help="arc center z (above plate), meters")
    p.add_argument("--bag", type=str, default=None,
                   help="optional .bag file to replay instead of live capture")
    p.add_argument("--frames", type=int, default=15,
                   help="temporal averaging frames")
    p.add_argument("--dome", type=str,
                   default="data/reference/dome_cloud.ply",
                   help="path to onshape dome reference .ply")
    p.add_argument("--out-dir", type=str, default="data/pose_verify",
                   help="output directory for diagnostic files")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. check dome reference exists and read its shape
    dome_path = Path(args.dome)
    if not dome_path.exists():
        logger.error(f"dome reference not found at {dome_path}")
        logger.error("export the onshape dome as .ply and place it there.")
        sys.exit(1)
    dome = o3d.io.read_point_cloud(str(dome_path))
    stats("dome reference (plate frame, from onshape)", dome)

    # 2. log the pose matrix at this angle
    T = camera_to_plate(args.angle, args.arc_radius, args.arc_center_z)
    logger.info(f"pose at {args.angle:.1f} deg:")
    logger.info(f"  camera position in plate coords: {T[:3, 3]}")
    logger.info(f"  camera forward axis: {T[:3, 2]}")
    logger.info(f"  camera right   axis: {T[:3, 0]}")
    logger.info(f"  camera down    axis: {T[:3, 1]}")

    # 3. capture a single frame
    scanner = RealSenseCapture(
        temporal_frames=args.frames,
        bag_file=args.bag,
        arc_radius_m=args.arc_radius,
        arc_center_z_m=args.arc_center_z,
    )
    scanner.start()
    try:
        # capture in camera frame
        cam_cloud = scanner.capture(angle_deg=None)
        stats("camera frame", cam_cloud)
        o3d.io.write_point_cloud(str(out_dir / "camera_frame.ply"), cam_cloud)

        # apply transform manually (equivalent to scanner.capture(angle_deg=...))
        plate_cloud = o3d.geometry.PointCloud(cam_cloud)
        plate_cloud.transform(T)
        stats("plate frame (after transform)", plate_cloud)
        o3d.io.write_point_cloud(str(out_dir / "plate_frame.ply"), plate_cloud)
    finally:
        scanner.stop()

    # 4. copy the dome reference alongside so it's trivial to load all
    # three together in cloudcompare
    shutil.copy(str(dome_path), str(out_dir / "dome_reference.ply"))

    logger.info("=" * 60)
    logger.info(f"wrote three .ply files to {out_dir}/")
    logger.info("  camera_frame.ply      raw capture, camera coords")
    logger.info("  plate_frame.ply       after camera_to_plate transform")
    logger.info("  dome_reference.ply    onshape truth (plate frame)")
    logger.info("")
    logger.info("next step: open all three in cloudcompare and compare")
    logger.info("plate_frame.ply against dome_reference.ply.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()