"""
cad_mask -- apparatus masking using the solidworks assembly stl.

uses the full-apparatus cad model as ground truth for what is and is not
object. two tests:
  1. surface distance: points within threshold_m of any apparatus surface
     are rejected (they are apparatus: plate, arc, control box, frame).
  2. bounding volume: points outside the apparatus bbox + margin are
     rejected (they are background: walls, ceiling, stray reflections).

a point survives only if it is inside the apparatus envelope but not on
any apparatus surface. by construction, that is the object.

assumes the captured cloud has already been transformed to plate frame
(origin at plate center top, z-up). the stl is loaded, scaled, and
translated to match this frame once, then cached.

usage:
    from processing.cad_mask import apply_cad_mask
    pcd = apply_cad_mask(pcd, stl_path=Path("data/reference/apparatus.STL"))
"""

import logging
from pathlib import Path

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

DEFAULT_STL_PATH = Path("data/reference/apparatus.STL")
DEFAULT_SCALE = 0.001
DEFAULT_ORIGIN_OFFSET = (-0.2773, -0.3300, -0.0945)

_cached_scene = None
_cached_bbox_min = None
_cached_bbox_max = None
_cached_key = None


def _load_apparatus(
    stl_path: Path,
    scale: float,
    origin_offset: tuple,
):
    """load stl once, scale + translate to plate frame, cache raycast scene."""
    global _cached_scene, _cached_bbox_min, _cached_bbox_max, _cached_key

    key = (str(stl_path), scale, tuple(origin_offset))
    if _cached_key == key and _cached_scene is not None:
        return _cached_scene, _cached_bbox_min, _cached_bbox_max

    if not stl_path.exists():
        raise FileNotFoundError(
            f"apparatus stl not found at {stl_path}. export the solidworks "
            f"assembly as stl and place it at this path."
        )

    mesh = o3d.io.read_triangle_mesh(str(stl_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"apparatus stl at {stl_path} has no vertices")

    # scale mm -> m (or whatever the stl units require)
    mesh.scale(scale, center=(0.0, 0.0, 0.0))

    # translate so plate-center-top is at origin
    mesh.translate(origin_offset)

    # bbox in plate frame for the bounding-volume test
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_min = np.asarray(bbox.min_bound)
    bbox_max = np.asarray(bbox.max_bound)

    # raycast scene for surface-distance queries
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    logger.info(
        f"loaded apparatus cad: {len(mesh.vertices)} verts, "
        f"{len(mesh.triangles)} tris from {stl_path}"
    )
    logger.info(
        f"apparatus bbox (plate frame): "
        f"x=[{bbox_min[0]:.3f}, {bbox_max[0]:.3f}], "
        f"y=[{bbox_min[1]:.3f}, {bbox_max[1]:.3f}], "
        f"z=[{bbox_min[2]:.3f}, {bbox_max[2]:.3f}]"
    )

    _cached_scene = scene
    _cached_bbox_min = bbox_min
    _cached_bbox_max = bbox_max
    _cached_key = key
    return scene, bbox_min, bbox_max


def apply_cad_mask(
    pcd: o3d.geometry.PointCloud,
    stl_path: Path = DEFAULT_STL_PATH,
    scale: float = DEFAULT_SCALE,
    origin_offset: tuple = DEFAULT_ORIGIN_OFFSET,
    surface_threshold_m: float = 0.003,
    bounding_margin_m: float = 0.01,
    keep_largest_cluster: bool = False,
    cluster_eps_m: float = 0.005,
    cluster_min_points: int = 50,
) -> o3d.geometry.PointCloud:
    """
    mask out apparatus and background points using the cad assembly.

    args:
        pcd: captured cloud in plate frame.
        stl_path: apparatus assembly stl.
        scale: stl -> meters scale factor (0.001 for mm stl).
        origin_offset: translation applied on load to align cad with plate frame.
        surface_threshold_m: max distance to apparatus surface. points closer
                             than this are treated as apparatus and removed.
        bounding_margin_m: slack on the apparatus bbox. points more than this
                           far outside the bbox are treated as background.

    returns:
        new pointcloud containing only points that are inside the apparatus
        envelope and not on apparatus surfaces.
    """
    if len(pcd.points) == 0:
        return pcd

    scene, bbox_min, bbox_max = _load_apparatus(stl_path, scale, origin_offset)

    pts = np.asarray(pcd.points, dtype=np.float32)
    n_total = len(pts)

    # bounding volume test: reject points outside bbox + margin
    expanded_min = bbox_min - bounding_margin_m
    expanded_max = bbox_max + bounding_margin_m
    inside_bbox = np.all(
        (pts >= expanded_min) & (pts <= expanded_max),
        axis=1,
    )
    n_outside_bbox = n_total - inside_bbox.sum()

    # surface distance test: reject points near apparatus surfaces
    # raycast scene expects float32 tensor
    query_pts = o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    distances = scene.compute_distance(query_pts).numpy()
    far_from_surface = distances > surface_threshold_m
    n_on_surface = (~far_from_surface).sum()

    # a point survives both tests
    keep = inside_bbox & far_from_surface
    kept_indices = np.where(keep)[0]

    n_kept = len(kept_indices)
    n_removed = n_total - n_kept
    logger.info(
        f"cad mask: removed {n_removed} points "
        f"({100.0 * n_removed / n_total:.1f}%) "
        f"[{n_outside_bbox} outside bbox, {n_on_surface} on surface], "
        f"{n_kept} object points remain"
    )

    out = pcd.select_by_index(kept_indices.tolist())

    if keep_largest_cluster and len(out.points) > cluster_min_points:
        labels = np.array(out.cluster_dbscan(
            eps=cluster_eps_m, min_points=cluster_min_points, print_progress=False
        ))
        n_clusters = int(labels.max()) + 1 if len(labels) else 0
        if n_clusters > 0:
            sizes = [(labels == l).sum() for l in range(n_clusters)]
            biggest = int(np.argmax(sizes))
            keep_idx = np.where(labels == biggest)[0]
            n_before = len(out.points)
            out = out.select_by_index(keep_idx.tolist())
            logger.info(
                f"cluster filter: kept largest of {n_clusters} clusters, "
                f"{n_before} -> {len(out.points)} points "
                f"(eps={cluster_eps_m*1000:.0f}mm, min_pts={cluster_min_points})"
            )

    return out

def build_depth_mask_from_cad(
    depth_array: np.ndarray,
    intrinsics: "o3d.camera.PinholeCameraIntrinsic",
    camera_pose: np.ndarray,
    depth_scale_m: float,
    stl_path: Path = DEFAULT_STL_PATH,
    scale: float = DEFAULT_SCALE,
    origin_offset: tuple = DEFAULT_ORIGIN_OFFSET,
    match_threshold_m: float = 0.005,
) -> np.ndarray:
    """
    zero out depth pixels that are hitting apparatus surfaces.

    for each pixel, back-projects a ray from the camera through the pixel,
    raycasts it against the apparatus cad, and compares the hit distance to
    the captured depth. if they match within match_threshold_m, the pixel is
    hitting apparatus and gets zeroed. object points (closer than the cad
    surface or missing the cad entirely) are kept.

    args:
        depth_array: uint16 depth image, raw sensor units.
        intrinsics: open3d PinholeCameraIntrinsic matching depth_array shape.
        camera_pose: 4x4 plate_T_camera transform.
        depth_scale_m: meters per sensor unit (e.g. 0.0001 for d405).
        stl_path, scale, origin_offset: same as apply_cad_mask.
        match_threshold_m: pixels within this distance of the cad surface
            are treated as apparatus and zeroed.

    returns:
        masked depth_array (copy, same dtype as input).
    """
    scene, _, _ = _load_apparatus(stl_path, scale, origin_offset)

    h, w = depth_array.shape
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]

    # camera origin in plate frame
    cam_origin = camera_pose[:3, 3].astype(np.float32)
    # camera rotation: columns are x/y/z axes in plate frame
    cam_rot = camera_pose[:3, :3].astype(np.float32)

    # build one ray direction per pixel, in camera frame, then rotate to plate
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    dirs_cam = np.stack([
        (us - cx) / fx,
        (vs - cy) / fy,
        np.ones_like(us, dtype=np.float32),
    ], axis=-1).astype(np.float32)  # (h, w, 3)

    # normalize each direction (raycast expects unit vectors for distance to mean meters)
    norms = np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    dirs_cam = dirs_cam / norms

    # rotate camera-frame directions into plate frame: plate_dir = R @ cam_dir
    dirs_plate = dirs_cam @ cam_rot.T  # (h, w, 3)

    # build ray tensor: (n_rays, 6) = [ox, oy, oz, dx, dy, dz]
    n_rays = h * w
    rays = np.zeros((n_rays, 6), dtype=np.float32)
    rays[:, 0] = cam_origin[0]
    rays[:, 1] = cam_origin[1]
    rays[:, 2] = cam_origin[2]
    rays[:, 3:6] = dirs_plate.reshape(-1, 3)

    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    result = scene.cast_rays(rays_tensor)
    cad_distances = result["t_hit"].numpy().reshape(h, w)  # meters along ray

    # convert captured depth to meters along the ray (depth is z-distance in
    # camera frame; our ray direction already accounts for per-pixel angle
    # because we normalized, so sensor_z -> ray_length = sensor_z * |dir| / dir_z.
    # since we normalized dirs_cam, dir_z is 1/|original|, so:
    captured_m_z = depth_array.astype(np.float32) * depth_scale_m
    # ray length to that z: sensor_z / dir_cam_z_normalized
    # dir_cam_z_normalized = 1.0 / norm (from normalization above)
    captured_m_along_ray = captured_m_z * norms.reshape(h, w)

    # apparatus pixel: |captured - cad| < threshold AND both finite
    valid_captured = depth_array > 0
    valid_cad = np.isfinite(cad_distances)
    hits_apparatus = (
        valid_captured &
        valid_cad &
        (np.abs(captured_m_along_ray - cad_distances) < match_threshold_m)
    )

    masked = depth_array.copy()
    masked[hits_apparatus] = 0

    n_apparatus = hits_apparatus.sum()
    n_valid = valid_captured.sum()
    logger.info(
        f"depth mask: zeroed {n_apparatus}/{n_valid} apparatus pixels "
        f"({100.0 * n_apparatus / max(n_valid, 1):.1f}% of valid depth), "
        f"{n_valid - n_apparatus} object pixels remain"
    )

    return masked
