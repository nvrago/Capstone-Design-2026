"""
dome_subtract -- static reference subtraction using the onshape dome cloud.

both the captured cloud and the dome reference must be in plate (world)
coordinates. for each captured point, finds the nearest dome reference
point. if the distance is below threshold, the point is treated as rig
geometry (plate surface, arc structure, mount hardware) and dropped.
otherwise it's kept as object geometry.

assumes capture.camera_to_plate has already been applied so the cloud
is in the same coordinate frame as data/reference/dome_cloud.ply.

usage:
    from processing.dome_subtract import subtract_dome
    pcd = subtract_dome(pcd, threshold_m=0.008)
"""

import logging
from pathlib import Path

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

DEFAULT_DOME_PATH = Path("data/reference/dome_cloud.ply")
_cached_dome = None
_cached_dome_tree = None
_cached_path = None


def _load_dome(path: Path):
    """load dome reference once and cache the kd-tree."""
    global _cached_dome, _cached_dome_tree, _cached_path

    if _cached_path == path and _cached_dome is not None:
        return _cached_dome, _cached_dome_tree

    if not path.exists():
        raise FileNotFoundError(
            f"dome reference not found at {path}. export the onshape "
            f"dome as .ply and place it at this path."
        )

    dome = o3d.io.read_point_cloud(str(path))
    if len(dome.points) == 0:
        raise ValueError(f"dome reference at {path} is empty")

    tree = o3d.geometry.KDTreeFlann(dome)
    logger.info(f"loaded dome reference: {len(dome.points)} points from {path}")

    _cached_dome = dome
    _cached_dome_tree = tree
    _cached_path = path
    return dome, tree


def subtract_dome(
    pcd: o3d.geometry.PointCloud,
    threshold_m: float = 0.008,
    dome_path: Path = DEFAULT_DOME_PATH,
) -> o3d.geometry.PointCloud:
    """
    drop points within threshold_m of any dome reference point.

    args:
        pcd: captured cloud in plate frame
        threshold_m: max distance to dome; points closer than this are
                     treated as rig and removed. 8mm matches the dome
                     reference's native ~11mm point spacing.
        dome_path: path to dome reference ply, plate frame.

    returns:
        new pointcloud containing only points further than threshold
        from the dome surface.
    """
    if len(pcd.points) == 0:
        return pcd

    _, tree = _load_dome(dome_path)

    pts = np.asarray(pcd.points)
    keep = np.ones(len(pts), dtype=bool)

    threshold_sq = threshold_m * threshold_m
    for i in range(len(pts)):
        # search for the single nearest dome point; returns squared distance
        _, _, sq_dist = tree.search_knn_vector_3d(pts[i], 1)
        if sq_dist[0] < threshold_sq:
            keep[i] = False

    kept_indices = np.where(keep)[0]
    removed = len(pts) - len(kept_indices)
    logger.info(
        f"dome subtract: removed {removed} rig points "
        f"({100.0 * removed / len(pts):.1f}%), "
        f"{len(kept_indices)} object points remain"
    )

    return pcd.select_by_index(kept_indices.tolist())