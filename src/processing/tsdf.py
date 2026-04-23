"""
tsdf.py -- scalable tsdf volume wrapper for multi-frame fusion.

wraps open3d's ScalableTSDFVolume. the caller passes depth+color frames
along with a plate_T_camera pose (natural direction: camera frame in plate
frame). this module inverts the pose internally to produce the extrinsic
that open3d expects (camera_T_world).

defaults are tuned for the d405 at close range:
  voxel_size_m = 0.001   (1mm, matches d405 sub-mm accuracy)
  sdf_trunc_m  = 0.004   (4mm, ~4 voxels of carving distance)
  depth_trunc_m = 0.5    (0.5m, matches d405 ideal range)

d405 depth is in units of 0.0001m by default. open3d's depth_scale is the
divisor applied to raw depth values to yield meters, so for d405 raw uint16
frames depth_scale = 10000.0. if you pass depth already in meters (float32),
use depth_scale = 1.0.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


@dataclass
class TSDFConfig:
    voxel_size_m: float = 0.001
    sdf_trunc_m: float = 0.004
    depth_trunc_m: float = 0.5
    depth_scale: float = 10000.0     # d405 default: 0.0001 m/unit -> divisor 10000
    use_color: bool = True


class TSDFIntegrator:
    """
    scalable tsdf fusion for a fixed-base, moving-camera setup.

    usage:
        tsdf = TSDFIntegrator(intrinsics, cfg)
        for depth_u16, color_rgb, plate_T_camera in frames:
            tsdf.integrate(depth_u16, color_rgb, plate_T_camera)
        pcd  = tsdf.extract_point_cloud()
        mesh = tsdf.extract_mesh()
    """

    def __init__(self, intrinsics: o3d.camera.PinholeCameraIntrinsic,
                 cfg: Optional[TSDFConfig] = None):
        self.intrinsics = intrinsics
        self.cfg = cfg or TSDFConfig()

        color_type = (o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                      if self.cfg.use_color
                      else o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.cfg.voxel_size_m,
            sdf_trunc=self.cfg.sdf_trunc_m,
            color_type=color_type,
        )
        self._n_integrated = 0
        logger.info(
            "tsdf volume: voxel=%.4fm sdf_trunc=%.4fm depth_trunc=%.3fm",
            self.cfg.voxel_size_m, self.cfg.sdf_trunc_m, self.cfg.depth_trunc_m,
        )

    def integrate(self,
                  depth: np.ndarray,
                  color: np.ndarray,
                  plate_T_camera: np.ndarray) -> None:
        """
        integrate a single frame.

        depth:          uint16 (raw d405) or float32 (meters) HxW array
        color:          uint8 HxWx3 rgb array, same HxW as depth
        plate_T_camera: 4x4 pose, camera frame in plate frame
        """
        if depth.ndim != 2:
            raise ValueError(f"depth must be 2D, got shape {depth.shape}")
        if color.shape[:2] != depth.shape:
            raise ValueError(
                f"color shape {color.shape[:2]} != depth shape {depth.shape}"
            )

        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth))
        color_img = o3d.geometry.Image(np.ascontiguousarray(color))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=self.cfg.depth_scale,
            depth_trunc=self.cfg.depth_trunc_m,
            convert_rgb_to_intensity=False,
        )

        extrinsic = np.linalg.inv(np.ascontiguousarray(plate_T_camera, dtype=np.float64))
        self.volume.integrate(rgbd, self.intrinsics, extrinsic)
        self._n_integrated += 1

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        pcd = self.volume.extract_point_cloud()
        logger.info("tsdf extract: %d points from %d frames",
                    len(pcd.points), self._n_integrated)
        return pcd

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        logger.info("tsdf extract: %d verts / %d tris from %d frames",
                    len(mesh.vertices), len(mesh.triangles), self._n_integrated)
        return mesh

    @property
    def n_integrated(self) -> int:
        return self._n_integrated

    def reset(self) -> None:
        self.volume.reset()
        self._n_integrated = 0


def make_pinhole_intrinsic(width: int, height: int,
                           fx: float, fy: float,
                           cx: float, cy: float) -> o3d.camera.PinholeCameraIntrinsic:
    """build an open3d intrinsic from realsense intrinsics dict fields."""
    return o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
    )
