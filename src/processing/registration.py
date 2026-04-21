"""
registration -- multi-view point cloud alignment via ICP

takes point clouds captured at known arc positions and registers
them into a single unified cloud. uses the known arc angles as
initial alignment estimates, then refines with ICP.

the arc carriage rotates the camera around the object, so the
transformation between positions is a rotation about the arc's
center axis. the known angle gives a strong initial guess that
ICP just needs to refine.

all open3d operations go through processing.o3d_safe to avoid
segfaults on the arm64 open3d build when handed non-contiguous
or non-float64 numpy arrays.

usage:
    reg = CloudRegistrator(arc_center=[0.0, 0.0, 0.0], arc_axis=[0.0, 1.0, 0.0])
    reg.add_cloud(pcd_0, angle=0.0)
    reg.add_cloud(pcd_1, angle=30.0)
    reg.add_cloud(pcd_2, angle=60.0)
    combined = reg.register_all()
"""

import numpy as np
import open3d as o3d
import logging
from dataclasses import dataclass, field

from processing.o3d_safe import (
    rebuild_pointcloud,
    safe_voxel_downsample,
    safe_estimate_normals,
    safe_transform,
    safe_combine,
    safe_icp_point_to_plane,
)

logger = logging.getLogger(__name__)


@dataclass
class PositionCloud:
    """a point cloud with its capture metadata."""
    cloud: o3d.geometry.PointCloud
    angle: float
    transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))


class CloudRegistrator:
    def __init__(
        self,
        arc_center: list = None,
        arc_axis: list = None,
        voxel_size: float = 0.002,
        icp_max_distance: float = 0.005,
        icp_max_iterations: int = 50
    ):
        """
        args:
            arc_center: center of rotation [x, y, z] in meters
            arc_axis: rotation axis [x, y, z] (normalized internally)
            voxel_size: downsample resolution for registration (meters)
            icp_max_distance: max correspondence distance for ICP (meters)
            icp_max_iterations: ICP iteration limit
        """
        self.arc_center = np.ascontiguousarray(
            arc_center or [0.0, 0.0, 0.0], dtype=np.float64
        )
        axis = np.ascontiguousarray(
            arc_axis or [0.0, 1.0, 0.0], dtype=np.float64
        )
        self.arc_axis = axis / np.linalg.norm(axis)

        self.voxel_size = voxel_size
        self.icp_max_distance = icp_max_distance
        self.icp_max_iterations = icp_max_iterations

        self.positions: list[PositionCloud] = []

    def add_cloud(self, cloud: o3d.geometry.PointCloud, angle: float):
        """add a captured cloud with its arc angle in degrees."""
        # rebuild on ingest so every downstream op starts from a clean,
        # contiguous pcd regardless of how the caller constructed it.
        clean = rebuild_pointcloud(cloud)
        self.positions.append(PositionCloud(cloud=clean, angle=angle))
        logger.info(f"added cloud at {angle:.1f} degrees: {len(clean.points)} points")

    def _rotation_matrix(self, angle_deg: float) -> np.ndarray:
        """
        build a 4x4 transform that rotates about the arc axis
        through the arc center by angle_deg degrees.
        returns a c-contiguous float64 matrix.
        """
        angle_rad = np.radians(angle_deg)
        axis = self.arc_axis
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        t = 1 - c

        # rodrigues rotation matrix
        R = np.array([
            [t*axis[0]*axis[0] + c, t*axis[0]*axis[1] - s*axis[2], t*axis[0]*axis[2] + s*axis[1]],
            [t*axis[0]*axis[1] + s*axis[2], t*axis[1]*axis[1] + c, t*axis[1]*axis[2] - s*axis[0]],
            [t*axis[0]*axis[2] - s*axis[1], t*axis[1]*axis[2] + s*axis[0], t*axis[2]*axis[2] + c]
        ], dtype=np.float64)

        # build 4x4: translate to origin, rotate, translate back
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = self.arc_center - R @ self.arc_center

        return np.ascontiguousarray(T, dtype=np.float64)

    def _prepare_for_icp(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """downsample and estimate normals for ICP, safely."""
        down = safe_voxel_downsample(cloud, self.voxel_size)
        return safe_estimate_normals(
            down,
            radius=self.voxel_size * 4,
            max_nn=30,
        )

    def register_pair(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        initial_transform: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        run point-to-plane ICP between source and target.
        returns (refined_transform, fitness_score).
        """
        source_down = self._prepare_for_icp(source)
        target_down = self._prepare_for_icp(target)

        result = safe_icp_point_to_plane(
            source_down,
            target_down,
            self.icp_max_distance,
            initial_transform,
            max_iterations=self.icp_max_iterations,
        )

        logger.info(f"ICP fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")
        # the transformation matrix from open3d may not be contiguous;
        # force it before returning so downstream callers are safe.
        refined = np.ascontiguousarray(result.transformation, dtype=np.float64)
        return refined, result.fitness

    def register_all(self, min_fitness: float = 0.3) -> o3d.geometry.PointCloud:
        """
        register all captured clouds into one unified cloud.

        uses position 0 as the reference frame. for each subsequent
        position, computes initial alignment from the known arc angle,
        then refines with ICP.

        args:
            min_fitness: minimum ICP fitness to accept a registration.
                below this, logs a warning but still includes the
                cloud with the geometric-only transform.

        returns:
            combined, registered point cloud
        """
        if len(self.positions) == 0:
            raise ValueError("no clouds to register")

        if len(self.positions) == 1:
            logger.info("single cloud, no registration needed")
            return rebuild_pointcloud(self.positions[0].cloud)

        # sort by angle for sequential registration
        self.positions.sort(key=lambda p: p.angle)

        # first cloud is the reference, identity transform
        reference = self.positions[0]
        reference.transform = np.eye(4, dtype=np.float64)
        logger.info(f"reference cloud at {reference.angle:.1f} degrees")

        # register each subsequent cloud to the reference
        for i in range(1, len(self.positions)):
            pos = self.positions[i]
            delta_angle = pos.angle - reference.angle

            # initial guess from known geometry
            initial = self._rotation_matrix(delta_angle)

            # skip icp for clouds that are too sparse to match usefully.
            # with very few points the kdtree / point-to-plane solver can
            # segfault on arm64. fall back to the geometric transform.
            if len(pos.cloud.points) < 100 or len(reference.cloud.points) < 100:
                logger.warning(
                    f"skipping ICP at {pos.angle:.1f} degrees "
                    f"(source={len(pos.cloud.points)}, "
                    f"target={len(reference.cloud.points)} points), "
                    f"using geometric transform"
                )
                pos.transform = initial
                continue

            # refine with ICP
            try:
                refined, fitness = self.register_pair(
                    pos.cloud, reference.cloud, initial
                )
            except Exception as e:
                logger.warning(
                    f"ICP failed at {pos.angle:.1f} degrees ({e}), "
                    f"using geometric transform"
                )
                pos.transform = initial
                continue

            if fitness < min_fitness:
                logger.warning(
                    f"low ICP fitness ({fitness:.3f}) at {pos.angle:.1f} degrees, "
                    f"using geometric transform only"
                )
                pos.transform = initial
            else:
                pos.transform = refined

        # transform each cloud into the reference frame and combine.
        # safe_transform rebuilds both before and after the transform call,
        # and safe_combine concatenates via numpy rather than open3d's +=.
        transformed_clouds = []
        for pos in self.positions:
            t = safe_transform(pos.cloud, pos.transform)
            transformed_clouds.append(t)
            logger.info(
                f"merged {len(pos.cloud.points)} points from {pos.angle:.1f} degrees"
            )

        combined = safe_combine(transformed_clouds)
        logger.info(f"combined cloud: {len(combined.points)} points from "
                     f"{len(self.positions)} positions")

        return combined

    def clear(self):
        """reset all stored clouds."""
        self.positions.clear()