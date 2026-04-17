"""
zero_mesh.py -- background subtraction for scan-to-cnc pipeline

workflow:
    1. capture zero scans (empty plate, all arc positions) -> store as reference
    2. on each real scan, subtract reference point cloud before meshing
    3. clip everything below the plate surface using max Z from reference
    4. what remains is only the object geometry

two subtraction methods applied in sequence:
    kd-tree: removes points spatially near the reference (handles edges, apparatus)
    z-clip: removes everything below the plate surface (handles flat plate cleanly)
"""

import numpy as np
import open3d as o3d
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ZERO_CLOUD_PATH = Path("data/reference/zero_cloud.ply")


# zero capture

def capture_zero(scanner, n_positions: int = 8, frames_per_position: int = 30) -> o3d.geometry.PointCloud:
    """
    capture reference scans of empty plate across all arc positions.
    call this once with no object on the plate.

    args:
        scanner: RealSenseCapture instance (already started)
        n_positions: number of arc positions to capture at
        frames_per_position: frames to average at each position

    returns:
        combined reference point cloud
    """
    clouds = []

    for i in range(n_positions):
        logger.info(f"capturing zero reference at position {i+1}/{n_positions}...")
        input(f"  move to position {i+1} and press Enter...")

        pcd = scanner.capture(n_frames=frames_per_position)
        clouds.append(pcd)
        logger.info(f"  captured {len(pcd.points)} points at position {i+1}")

    combined = o3d.geometry.PointCloud()
    for cloud in clouds:
        combined += cloud

    combined = combined.voxel_down_sample(voxel_size=0.002)
    logger.info(f"zero reference cloud: {len(combined.points)} points total")

    return combined


def save_zero(pcd: o3d.geometry.PointCloud, path: Path = ZERO_CLOUD_PATH) -> None:
    """save reference cloud to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)
    logger.info(f"saved zero reference to {path}")


def load_zero(path: Path = ZERO_CLOUD_PATH) -> o3d.geometry.PointCloud:
    """load reference cloud from disk."""
    if not path.exists():
        raise FileNotFoundError(f"no zero reference found at {path}. run capture_zero first.")
    pcd = o3d.io.read_point_cloud(str(path))
    logger.info(f"loaded zero reference: {len(pcd.points)} points from {path}")
    return pcd


# kd-tree subtraction

def subtract_zero(
    scan: o3d.geometry.PointCloud,
    reference: o3d.geometry.PointCloud,
    distance_threshold: float = 0.003
) -> o3d.geometry.PointCloud:
    """
    remove reference (background) points from a scan.

    for each point in scan, if its nearest neighbor in the reference cloud
    is within distance_threshold, it's considered background and removed.

    args:
        scan: raw scan point cloud (with object)
        reference: zero reference cloud (empty plate)
        distance_threshold: points within this distance (meters) of reference
            are considered background. default 3mm.

    returns:
        point cloud with background removed, object only
    """
    if len(reference.points) == 0:
        logger.warning("reference cloud is empty, skipping subtraction")
        return scan

    ref_tree = o3d.geometry.KDTreeFlann(reference)

    object_indices = []
    for i, point in enumerate(scan.points):
        [k, idx, dist] = ref_tree.search_knn_vector_3d(point, 1)
        nearest_dist = np.sqrt(dist[0])

        if nearest_dist > distance_threshold:
            object_indices.append(i)

    object_cloud = scan.select_by_index(object_indices)
    n_removed = len(scan.points) - len(object_cloud.points)
    logger.info(f"kd-tree subtraction: removed {n_removed} background points, "
                f"{len(object_cloud.points)} object points remain")

    return object_cloud


# z-clip using reference surface

def clip_below_reference(
    scan: o3d.geometry.PointCloud,
    reference: o3d.geometry.PointCloud,
    buffer_m: float = 0.001
) -> o3d.geometry.PointCloud:
    """
    remove all points at or below the plate surface.

    finds the max Z value in the reference cloud (highest point on the
    empty plate), adds a small buffer, and removes everything below that
    threshold from the scan. this cleanly eliminates the flat plate
    surface that kd-tree subtraction might miss.

    args:
        scan: point cloud to clip (usually after kd-tree subtraction)
        reference: zero reference cloud
        buffer_m: buffer above max Z to keep (meters). default 1mm.
            positive = keep more (safer), negative = clip more aggressively.

    returns:
        point cloud with plate surface removed
    """
    if len(reference.points) == 0:
        logger.warning("reference cloud is empty, skipping z-clip")
        return scan

    ref_points = np.asarray(reference.points)
    max_z = np.max(ref_points[:, 2])
    # subtract buffer so we clip slightly into the plate rather than above it
    clip_z = max_z - buffer_m

    scan_points = np.asarray(scan.points)

    # the D405 looks down, so Z increases with distance from camera.
    # the plate is far (high Z), the object sticks up toward camera (lower Z).
    # keep points that are closer to the camera than the plate surface.
    object_mask = scan_points[:, 2] < clip_z
    object_indices = np.where(object_mask)[0].tolist()

    clipped = scan.select_by_index(object_indices)
    n_removed = len(scan.points) - len(clipped.points)

    logger.info(f"z-clip: plate max Z={max_z:.4f}m, threshold={clip_z:.4f}m, "
                f"removed {n_removed} points below plate, {len(clipped.points)} remain")

    return clipped


# convenience: full subtract pipeline step

def apply_zero_subtraction(
    scan: o3d.geometry.PointCloud,
    reference_path: Path = ZERO_CLOUD_PATH,
    distance_threshold: float = 0.003,
    z_clip: bool = True,
    z_clip_buffer: float = 0.002
) -> o3d.geometry.PointCloud:
    """
    load reference and apply both subtraction methods in one call.

    runs kd-tree subtraction first, then z-clip if enabled.

    usage in pipeline.py:
        pcd = capture()
        pcd = apply_zero_subtraction(pcd)
        pcd = process_pointcloud(pcd)
    """
    try:
        reference = load_zero(reference_path)

        # step 1: kd-tree subtraction (handles edges, apparatus)
        result = subtract_zero(scan, reference, distance_threshold)

        # step 2: z-clip (handles flat plate surface)
        if z_clip:
            result = clip_below_reference(result, reference, z_clip_buffer)

        return result

    except FileNotFoundError as e:
        logger.warning(f"zero subtraction skipped: {e}")
        return scan
