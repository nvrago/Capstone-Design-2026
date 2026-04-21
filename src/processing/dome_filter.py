"""
dome_filter.py -- plate-centered dome background filter

the arc carriage defines a physical envelope of what the D405 can reach:
a hemisphere of inner radius ~275mm centered 50mm above the plate. any
captured point outside that envelope is, by definition, not the object
we are scanning -- it's either the arc frame, the surrounding room, or
sensor noise.

this module applies that envelope as a background filter. two stages:

  1. clip_outside_dome -- hard geometric cutoff. anything outside the
     hemisphere or below the plate is dropped, no tolerance.

  2. subtract_dome_surface -- for points near the dome surface itself,
     use a pregenerated point cloud of the dome and an existing
     nearest-neighbor subtraction (from zero_mesh.subtract_zero). this
     catches points that fall inside the dome but near the boundary,
     where the arc frame itself shows up in the scan.

both operate in plate coordinates. the pipeline transforms each captured
cloud from camera frame to plate frame, applies the filter, then
transforms back to camera frame for downstream processing.

to regenerate the dome reference cloud after changing dome parameters,
run: python scripts/generate_dome_reference.py

the dome .ply is saved to data/reference/dome_cloud.ply.
"""

import logging
from pathlib import Path

import numpy as np
import open3d as o3d

from processing.o3d_safe import make_pointcloud
from processing.arc_geometry import (
    transform_camera_to_plate,
    transform_plate_to_camera,
    apply_transform,
)
from processing.zero_mesh import subtract_zero

logger = logging.getLogger(__name__)

DOME_CLOUD_PATH = Path("data/reference/dome_cloud.ply")


# dome geometry generation

def generate_dome_points(
    radius_m: float,
    center_z_m: float,
    point_spacing_m: float = 0.005,
    include_plate: bool = True,
    plate_radius_m: float = 0.220,
) -> np.ndarray:
    """
    generate a uniformly-sampled point cloud of the inner hemisphere
    surface + (optionally) the plate disk.

    args:
        radius_m: inner dome radius
        center_z_m: height of dome center above plate (arc geometry)
        point_spacing_m: target distance between adjacent dome points.
            smaller -> denser dome, better surface subtraction, bigger file.
        include_plate: also generate points on the plate disk surface
            (z=0, radius <= plate_radius_m). usually yes, since the plate
            itself is background we want to subtract.
        plate_radius_m: plate disk radius. default 17in / 2 -> 0.2159m.

    returns:
        (N, 3) numpy array of points in plate coordinates.
    """
    points = []

    # hemisphere surface: sample using spherical coordinates with equal
    # area on the sphere. use the "golden spiral" for roughly uniform
    # distribution without pole clumping.
    surface_area = 2 * np.pi * radius_m ** 2
    n_dome = max(int(surface_area / (point_spacing_m ** 2)), 100)

    # fibonacci hemisphere sampling: uniform points on a hemisphere
    golden = (1 + np.sqrt(5)) / 2
    for i in range(n_dome):
        # y goes from 1 (top) to 0 (equator) for the upper hemisphere
        y = 1.0 - (i / max(n_dome - 1, 1))
        r_at_y = np.sqrt(1.0 - y * y)
        theta = 2 * np.pi * i / golden
        px = r_at_y * np.cos(theta) * radius_m
        pz = r_at_y * np.sin(theta) * radius_m
        py = y * radius_m
        # rotate so the dome opens downward toward the plate:
        # we want +z up, hemisphere above z = center_z_m.
        # current (px, py, pz) has +py up. map to plate frame:
        points.append([px, pz, py + center_z_m])

    if include_plate:
        # plate disk: concentric rings at z = 0
        n_rings = max(int(plate_radius_m / point_spacing_m), 1)
        for ring in range(n_rings + 1):
            r = ring * point_spacing_m
            if r > plate_radius_m:
                continue
            circumference = 2 * np.pi * r
            n_pts = max(int(circumference / point_spacing_m), 1)
            for j in range(n_pts):
                phi = 2 * np.pi * j / n_pts
                points.append([r * np.cos(phi), r * np.sin(phi), 0.0])

    return np.ascontiguousarray(points, dtype=np.float64)


def build_dome_reference(
    radius_m: float,
    center_z_m: float,
    point_spacing_m: float = 0.005,
    include_plate: bool = True,
    plate_radius_m: float = 0.2159,
    output_path: Path = DOME_CLOUD_PATH,
) -> o3d.geometry.PointCloud:
    """
    generate + save a dome reference point cloud.
    returns the open3d PointCloud.
    """
    pts = generate_dome_points(
        radius_m=radius_m,
        center_z_m=center_z_m,
        point_spacing_m=point_spacing_m,
        include_plate=include_plate,
        plate_radius_m=plate_radius_m,
    )
    dome = make_pointcloud(pts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), dome)
    logger.info(f"dome reference saved: {len(pts)} points to {output_path}")
    return dome


def load_dome_reference(path: Path = DOME_CLOUD_PATH) -> o3d.geometry.PointCloud:
    """load the pregenerated dome reference cloud."""
    if not path.exists():
        raise FileNotFoundError(
            f"no dome reference at {path}. "
            f"run: python scripts/generate_dome_reference.py"
        )
    dome = o3d.io.read_point_cloud(str(path))
    logger.info(f"loaded dome reference: {len(dome.points)} points")
    return dome


# filter stages -- all operate in plate coordinates

def clip_outside_dome(
    scan_pts_plate: np.ndarray,
    scan_cols: np.ndarray,
    dome_radius_m: float,
    dome_center_z_m: float,
    plate_z_m: float = 0.0,
    tolerance_m: float = 0.00001,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    hard geometric cutoff: keep only points inside the hemisphere
    and above the plate. pure numpy, no open3d.

    args:
        scan_pts_plate: (N, 3) points in plate coordinates
        scan_cols: (N, 3) colors or None
        dome_radius_m: hemisphere radius
        dome_center_z_m: hemisphere center height above plate
        plate_z_m: plate top surface z (below this = dropped)
        tolerance_m: shell margin. points within this much of the
            dome surface are kept (avoids clipping legitimate surface
            detail at the mathematical boundary).

    returns:
        (filtered_points, filtered_colors_or_None)
    """
    if scan_pts_plate.shape[0] == 0:
        return scan_pts_plate, scan_cols

    x = scan_pts_plate[:, 0]
    y = scan_pts_plate[:, 1]
    z = scan_pts_plate[:, 2]

    # distance squared from dome center
    r2 = x * x + y * y + (z - dome_center_z_m) ** 2
    r_max = dome_radius_m + tolerance_m

    inside_dome = r2 < r_max * r_max
    above_plate = z > plate_z_m

    keep = inside_dome & above_plate
    out_pts = np.ascontiguousarray(scan_pts_plate[keep], dtype=np.float64)
    out_cols = None
    if scan_cols is not None:
        out_cols = np.ascontiguousarray(scan_cols[keep], dtype=np.float64)

    return out_pts, out_cols


# top-level entry point

def apply_dome_filter(
    pcd: o3d.geometry.PointCloud,
    angle_deg: float,
    arc_radius_m: float,
    arc_center_z_m: float,
    dome_radius_m: float,
    plate_z_m: float = 0.0,
    tolerance_m: float = 0.00001,
    dome_reference: o3d.geometry.PointCloud | None = None,
    subtract_dome_surface: bool = True,
    dome_distance_threshold_m: float = 0.003,
) -> o3d.geometry.PointCloud:
    """
    plate-centered dome filter for a single position cloud.

    flow:
      1. transform points from camera frame to plate frame (arc geometry)
      2. clip anything outside the hemisphere or below the plate
      3. optionally subtract points near the dome reference surface
      4. transform surviving points back to camera frame
      5. return as a new open3d PointCloud

    args:
        pcd: input cloud in CAMERA coordinates
        angle_deg: the arc angle at which this cloud was captured
        arc_radius_m: inner sweep radius of the arc carriage
        arc_center_z_m: arc center height above plate (plate coords)
        dome_radius_m: hemisphere radius for the filter. usually equal
            to arc_radius_m, but can be smaller for a tighter filter.
        plate_z_m: plate top z in plate coords (default 0)
        tolerance_m: dome shell tolerance
        dome_reference: pregenerated dome point cloud. if None and
            subtract_dome_surface=True, loads from disk.
        subtract_dome_surface: also run nearest-neighbor subtraction
            against the dome reference to catch arc-frame artifacts.
        dome_distance_threshold_m: knn threshold for surface subtraction

    returns:
        new o3d.geometry.PointCloud in CAMERA coordinates
    """
    if len(pcd.points) == 0:
        return pcd

    cam_pts = np.asarray(pcd.points)
    cam_cols = np.asarray(pcd.colors) if pcd.has_colors() else None

    # 1. camera -> plate
    T_cp = transform_camera_to_plate(angle_deg, arc_radius_m, arc_center_z_m)
    plate_pts = apply_transform(cam_pts, T_cp)

    n_before = plate_pts.shape[0]

    # 2. hard geometric cutoff
    plate_pts, cam_cols = clip_outside_dome(
        plate_pts,
        cam_cols,
        dome_radius_m=dome_radius_m,
        dome_center_z_m=arc_center_z_m,
        plate_z_m=plate_z_m,
        tolerance_m=tolerance_m,
    )
    logger.info(
        f"  dome clip at {angle_deg:.1f}deg: {plate_pts.shape[0]}/{n_before} "
        f"points kept inside hemisphere"
    )

    # 3. optional dome-surface subtraction (for arc-frame artifacts)
    if subtract_dome_surface and plate_pts.shape[0] > 0:
        if dome_reference is None:
            try:
                dome_reference = load_dome_reference()
            except FileNotFoundError as e:
                logger.warning(f"dome surface subtraction skipped: {e}")
                dome_reference = None

        if dome_reference is not None:
            # subtract_zero expects o3d PointClouds. build a temporary
            # one in plate coords for the subtraction step.
            scan_plate = make_pointcloud(plate_pts, cam_cols)
            filtered = subtract_zero(
                scan_plate,
                dome_reference,
                distance_threshold=dome_distance_threshold_m,
            )
            plate_pts = np.asarray(filtered.points)
            cam_cols = np.asarray(filtered.colors) if filtered.has_colors() else None

    # 4. plate -> camera
    T_pc = transform_plate_to_camera(angle_deg, arc_radius_m, arc_center_z_m)
    cam_pts_out = apply_transform(plate_pts, T_pc)

    # 5. rebuild in camera frame
    return make_pointcloud(cam_pts_out, cam_cols)