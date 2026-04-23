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

    # count points per non-noise label
    valid = labels[labels >= 0]
    counts = np.bincount(valid)
    largest = int(np.argmax(counts))
    kept = np.where(labels == largest)[0]

    n_clusters = int(labels.max()) + 1
    out = pcd.select_by_index(kept)
    logger.info("keep_largest_cluster: %d clusters found, kept label=%d "
                "(%d/%d points)",
                n_clusters, largest, len(kept), len(pcd.points))
    return out


def filter_plate_cloud(pcd: o3d.geometry.PointCloud,
                       cfg: Optional[ClipConfig] = None
                       ) -> o3d.geometry.PointCloud:
    """convenience: box clip, then largest cluster."""
    cfg = cfg or ClipConfig()
    pcd = box_clip(pcd, cfg)
    pcd = keep_largest_cluster(pcd, cfg)
    return pcd
