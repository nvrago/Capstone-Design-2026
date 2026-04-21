"""
registration -- multi-view point cloud refinement via ICP

takes point clouds already expressed in the plate (world) frame
(transformed at capture time by camera_to_plate) and refines
residual mechanical error between positions with ICP.

because every cloud is already in the shared plate frame, the
initial transform for every pair is identity. ICP only corrects
for mechanical slop: arc flex, backlash in the rack-and-pinion,
radius measurement error, camera mount tilt.

all open3d operations go through processing.o3d_safe to avoid
segfaults on the arm64 open3d build when handed non-contiguous
or non-float64 numpy arrays.

usage:
    reg = CloudRegistrator()
    reg.add_cloud(pcd_0, angle=0.0)
    reg.add_cloud(pcd_1, angle=15.0)
    reg.add_cloud(pcd_2, angle=30.0)
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
        icp_max_distance: float = 0.05,
        icp_max_iterations: int = 50,
    ):
        """
        args:
            arc_center: kept for api compat, no longer used. clouds arrive
                        already in plate frame.
            arc_axis: kept for api compat, no longer used.
            voxel_size: downsample resolution for registration (meters)
            icp_max_distance: max correspondence distance for ICP (meters)
            icp_max_iterations: ICP iteration limit
        """
        # kept as attributes in case callers introspect them; no longer
        # drives transform math, since clouds are pre-aligned in plate frame.
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
        """add a captured cloud (already in plate frame) with its arc angle."""
        clean = rebuild_pointcloud(cloud)
        self.positions.append(PositionCloud(cloud=clean, angle=angle))
        logger.info(f"added cloud at {angle:.1f} degrees: {len(clean.points)} points")

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

        # coarse: loose threshold absorbs residual plate-frame misalignment
        coarse = safe_icp_point_to_plane(
            source_down,
            target_down,
            self.icp_max_distance,        # now 0.05
            initial_transform,
            max_iterations=self.icp_max_iterations,
        )

        # fine: tight threshold, seeded from coarse result
        result = safe_icp_point_to_plane(
            source_down,
            target_down,
            self.icp_max_distance * 0.1,  # 0.005
            coarse.transformation,
            max_iterations=self.icp_max_iterations,
        )

        logger.info(f"ICP fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")
        refined = np.ascontiguousarray(result.transformation, dtype=np.float64)
        return refined, result.fitness

    def register_all(self, min_fitness: float = 0.3) -> o3d.geometry.PointCloud:
        """
        refine alignment of all captured clouds with ICP and combine them.

        clouds must already be in plate (world) coordinates (applied by
        capture.camera_to_plate at capture time). ICP starts from identity
        and refines residual mechanical error only.

        args:
            min_fitness: minimum ICP fitness to accept. below this, falls
                         back to identity (trust the geometric pose).

        returns:
            combined point cloud in plate frame
        """
        if len(self.positions) == 0:
            raise ValueError("no clouds to register")

        if len(self.positions) == 1:
            logger.info("single cloud, no registration needed")
            return rebuild_pointcloud(self.positions[0].cloud)

        self.positions.sort(key=lambda p: p.angle)

        reference = self.positions[0]
        reference.transform = np.eye(4, dtype=np.float64)
        logger.info(
            f"reference cloud at {reference.angle:.1f} degrees "
            f"(all clouds already in plate frame)"
        )

        for i in range(1, len(self.positions)):
            pos = self.positions[i]

            # clouds already share the plate frame. ICP initial guess
            # is identity; it only needs to correct mechanical slop.
            initial = np.eye(4, dtype=np.float64)

            if len(pos.cloud.points) < 100 or len(reference.cloud.points) < 100:
                logger.warning(
                    f"skipping ICP at {pos.angle:.1f} degrees "
                    f"(source={len(pos.cloud.points)}, "
                    f"target={len(reference.cloud.points)} points), "
                    f"using identity transform"
                )
                pos.transform = initial
                continue

            try:
                refined, fitness = self.register_pair(
                    pos.cloud, reference.cloud, initial
                )
            except Exception as e:
                logger.warning(
                    f"ICP failed at {pos.angle:.1f} degrees ({e}), "
                    f"using identity transform"
                )
                pos.transform = initial
                continue

            if fitness < min_fitness:
                logger.warning(
                    f"low ICP fitness ({fitness:.3f}) at {pos.angle:.1f} degrees, "
                    f"using identity transform"
                )
                pos.transform = initial
            else:
                pos.transform = refined

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