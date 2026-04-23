"""
clip.py -- plate-frame filtering for tsdf output.

two operations, in order:
  1. box clip:       keep points inside |x|<ext, |y|<ext, z_min<z<z_max
                     (drops plate surface, arc hardware, background)
  2. largest cluster: dbscan + keep biggest component (drops stragglers)

both take and return o3d.geometry.PointCloud. assumes the input cloud is
already in plate frame (z=0 = plate top, +z = up).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


@dataclass
class ClipConfig:
    xy_extent_m: float = 0.100   # +/- 100mm from plate center
    z_min_m: float = 0.002       # drop plate surface + noise
    z_max_m: float = 0.100       # max object height
    cluster_eps_m: float = 0.005
    cluster_min_points: int = 50


def box_clip(pcd: o3d.geometry.PointCloud,
             cfg: ClipConfig) -> o3d.geometry.PointCloud:
    """axis-aligned box clip in plate frame."""
    if len(pcd.points) == 0:
        logger.warning("box_clip: empty input")
        return pcd

    pts = np.asarray(pcd.points)
    mask = (
        (np.abs(pts[:, 0]) < cfg.xy_extent_m) &
        (np.abs(pts[:, 1]) < cfg.xy_extent_m) &
        (pts[:, 2] > cfg.z_min_m) &
        (pts[:, 2] < cfg.z_max_m)
    )
    kept = np.where(mask)[0]
    out = pcd.select_by_index(kept)
    logger.info("box_clip: %d/%d points kept (%.1f%%)",
                len(kept), len(pts), 100.0 * len(kept) / max(len(pts), 1))
    return out


def keep_largest_cluster(pcd: o3d.geometry.PointCloud,
                         cfg: ClipConfig) -> o3d.geometry.PointCloud:
    """
    dbscan cluster, keep the cluster with the most points.
    noise points (label -1) are always dropped.
    """
    if len(pcd.points) == 0:
        logger.warning("keep_largest_cluster: empty input")
        return pcd

    labels = np.asarray(
        pcd.cluster_dbscan(eps=cfg.cluster_eps_m,
                           min_points=cfg.cluster_min_points,
                           print_progress=False)
    )
    if labels.size == 0 or labels.max() < 0:
        logger.warning("keep_largest_cluster: no clusters found, returning input")
        return pcd

    # pick cluster by highest max-z instead of most points. the object
    # is taller than the plate by definition, so the tallest cluster is
    # always the object. picking by count fails when the plate remnant
    # (after box clip) has more points than the object's top surface.
    pts = np.asarray(pcd.points)
    n_clusters = int(labels.max()) + 1
    max_z_per_cluster = {}
    for lbl in range(n_clusters):
        cluster_pts = pts[labels == lbl]
        if len(cluster_pts) > 0:
            max_z_per_cluster[lbl] = cluster_pts[:, 2].max()

    if not max_z_per_cluster:
        logger.warning("keep_largest_cluster: no non-empty clusters")
        return pcd

    tallest = max(max_z_per_cluster, key=max_z_per_cluster.get)
    kept = np.where(labels == tallest)[0]
    out = pcd.select_by_index(kept)
    logger.info("keep_largest_cluster: %d clusters found, kept label=%d "
                "(max_z=%.1fmm, %d/%d points)",
                n_clusters, tallest, max_z_per_cluster[tallest]*1000,
                len(kept), len(pcd.points))
    return out


def filter_plate_cloud(pcd: o3d.geometry.PointCloud,
                       cfg: Optional[ClipConfig] = None
                       ) -> o3d.geometry.PointCloud:
    """convenience: box clip, then largest cluster."""
    cfg = cfg or ClipConfig()
    pcd = box_clip(pcd, cfg)
    pcd = keep_largest_cluster(pcd, cfg)
    return pcd
