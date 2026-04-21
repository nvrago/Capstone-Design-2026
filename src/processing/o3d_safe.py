"""
o3d_safe -- safe wrappers for open3d operations on arm64

the from-source open3d build on arm64 (raspberry pi) segfaults inside
pybind when handed non-contiguous or non-float64 numpy arrays. every
numpy -> open3d crossing needs to force contiguity and dtype first.

use these wrappers anywhere a point cloud is constructed, combined,
transformed, or fed to an open3d algorithm. they are no-ops on x86_64
where open3d handles the conversion internally, but prevent segfaults
on arm64.
"""

import numpy as np
import open3d as o3d


def make_pointcloud(points, colors=None, normals=None):
    """
    construct an o3d PointCloud from numpy arrays, safely.
    forces c-contiguous float64 before handing to Vector3dVector.
    """
    pcd = o3d.geometry.PointCloud()
    pts = np.ascontiguousarray(np.asarray(points), dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        cols = np.ascontiguousarray(np.asarray(colors), dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(cols)
    if normals is not None:
        nrm = np.ascontiguousarray(np.asarray(normals), dtype=np.float64)
        pcd.normals = o3d.utility.Vector3dVector(nrm)
    return pcd


def rebuild_pointcloud(pcd):
    """
    rebuild a pcd by extracting its numpy arrays and reconstructing.
    strips any non-contiguous state that may have accumulated from
    prior operations (masking, downsample, transform, etc.).
    """
    pts = np.asarray(pcd.points) if len(pcd.points) else np.zeros((0, 3))
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    nrm = np.asarray(pcd.normals) if pcd.has_normals() else None
    return make_pointcloud(pts, cols, nrm)


def safe_transform(pcd, matrix):
    """
    apply a 4x4 transform to a pcd with forced contiguity on the matrix.
    returns a new pcd; does not modify the input.
    """
    m = np.ascontiguousarray(np.asarray(matrix), dtype=np.float64)
    out = rebuild_pointcloud(pcd)
    out.transform(m)
    return rebuild_pointcloud(out)


def safe_voxel_downsample(pcd, voxel_size):
    """voxel downsample, with rebuild on both sides."""
    clean = rebuild_pointcloud(pcd)
    down = clean.voxel_down_sample(voxel_size)
    return rebuild_pointcloud(down)


def safe_estimate_normals(pcd, radius, max_nn=30):
    """estimate normals in-place on a rebuilt pcd."""
    clean = rebuild_pointcloud(pcd)
    clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    return rebuild_pointcloud(clean)


def safe_crop_z(pcd, z_min, z_max):
    """
    crop points by z range using numpy masking, avoiding
    AxisAlignedBoundingBox which segfaults on arm64.
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return rebuild_pointcloud(pcd)
    mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    cols = np.asarray(pcd.colors)[mask] if pcd.has_colors() else None
    nrm = np.asarray(pcd.normals)[mask] if pcd.has_normals() else None
    return make_pointcloud(pts[mask], cols, nrm)


def safe_combine(pcds):
    """concatenate multiple pcds into one, rebuilding to keep contiguous."""
    if len(pcds) == 0:
        return o3d.geometry.PointCloud()
    all_pts = []
    all_cols = []
    all_nrm = []
    has_colors = all(p.has_colors() for p in pcds)
    has_normals = all(p.has_normals() for p in pcds)
    for p in pcds:
        all_pts.append(np.asarray(p.points))
        if has_colors:
            all_cols.append(np.asarray(p.colors))
        if has_normals:
            all_nrm.append(np.asarray(p.normals))
    pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    cols = np.concatenate(all_cols, axis=0) if has_colors and all_cols else None
    nrm = np.concatenate(all_nrm, axis=0) if has_normals and all_nrm else None
    return make_pointcloud(pts, cols, nrm)


def safe_icp_point_to_plane(
    source, target, max_distance, initial_transform, max_iterations=50
):
    """
    run point-to-plane ICP with safe inputs. both clouds must have
    normals on the target side. returns the icp result object.
    """
    src = rebuild_pointcloud(source)
    tgt = rebuild_pointcloud(target)
    init = np.ascontiguousarray(np.asarray(initial_transform), dtype=np.float64)
    return o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        max_distance,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )