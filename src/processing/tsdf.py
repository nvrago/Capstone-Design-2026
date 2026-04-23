"""
tsdf -- truncated signed distance field volumetric fusion.

fuses multiple rgbd frames at known camera poses into a single volume,
then extracts a watertight mesh via marching cubes. replaces the
icp-register-then-poisson path when mesh_method = 'tsdf' is selected
in pipeline config.

inputs: depth + color frames (raw, per-pose), intrinsics, plate_T_camera
outputs: triangle mesh, point cloud (both in plate frame)

two integrators are provided:
    TSDFIntegrator: single fused volume (original, kept for compatibility)
    MultiAngleTSDFIntegrator: one sub-volume per arc angle + fused volume,
        enables per-vertex angle voting to reject single-view artifacts.

usage:
    integrator = TSDFIntegrator(voxel_size_m=0.001, sdf_trunc_m=0.004)
    for depth_img, color_img, intrinsics, pose in frames_with_poses:
        integrator.integrate_frame(depth_img, color_img, intrinsics, pose,
                                    depth_scale_m=0.0001)
    mesh = integrator.extract_mesh()
    cloud = integrator.extract_cloud()

pre-integration depth masking via cad (option A) is handled by the
caller by passing pre-masked depth arrays. post-integration cloud
masking (option B) and cluster filtering are called separately on the
extracted cloud.
"""

import logging
from typing import Optional

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


class TSDFIntegrator:
    """accumulates rgbd frames into a scalable tsdf volume."""

    def __init__(
        self,
        voxel_size_m: float = 0.001,
        sdf_trunc_m: float = 0.004,
        depth_trunc_m: float = 0.5,
        color: bool = True,
    ):
        """
        args:
            voxel_size_m: volume voxel edge length (default 1mm).
            sdf_trunc_m: sdf truncation band (default 4mm, ~4x voxel).
            depth_trunc_m: max depth accepted per pixel (default 0.5m).
            color: integrate rgb alongside depth.
        """
        self.voxel_size_m = voxel_size_m
        self.sdf_trunc_m = sdf_trunc_m
        self.depth_trunc_m = depth_trunc_m

        color_type = (
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            if color
            else o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        )
        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size_m,
            sdf_trunc=sdf_trunc_m,
            color_type=color_type,
        )
        self._n_integrated = 0

    def integrate_frame(
        self,
        depth_array: np.ndarray,
        color_array: np.ndarray,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        plate_T_camera: np.ndarray,
        depth_scale_m: float,
    ):
        """integrate one rgbd frame at a known camera pose.

        args:
            depth_array: uint16 depth image, raw sensor units.
            color_array: uint8 rgb image, same h/w as depth (aligned).
            intrinsics: camera intrinsics for the depth stream.
            plate_T_camera: 4x4 camera pose in plate frame.
            depth_scale_m: meters per sensor unit (d405: 0.0001).
        """
        if depth_array.shape[:2] != color_array.shape[:2]:
            raise ValueError(
                f"depth shape {depth_array.shape} != color shape "
                f"{color_array.shape[:2]}; caller must align them"
            )

        depth_o3d = o3d.geometry.Image(depth_array)
        color_o3d = o3d.geometry.Image(color_array)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_o3d,
            depth=depth_o3d,
            depth_scale=1.0 / depth_scale_m,
            depth_trunc=self.depth_trunc_m,
            convert_rgb_to_intensity=False,
        )

        # open3d's integrate expects world_T_camera inverted, i.e. camera_T_world
        extrinsic = np.linalg.inv(plate_T_camera)
        self._volume.integrate(rgbd, intrinsics, extrinsic)
        self._n_integrated += 1

    def integrate_position(
        self,
        depth_frames: list,
        color_frames: list,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        plate_T_camera: np.ndarray,
        depth_scale_m: float,
    ):
        """integrate n frames at the same pose (single arc position).

        redundant frames at a fixed pose reduce per-voxel noise through
        sdf averaging; equivalent to temporal averaging but done inside
        the volume rather than at the pixel level.
        """
        if len(depth_frames) != len(color_frames):
            raise ValueError(
                f"depth/color count mismatch: {len(depth_frames)} vs "
                f"{len(color_frames)}"
            )
        for d, c in zip(depth_frames, color_frames):
            self.integrate_frame(d, c, intrinsics, plate_T_camera, depth_scale_m)

    @property
    def frame_count(self) -> int:
        return self._n_integrated

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        """extract a triangle mesh via marching cubes on the sdf."""
        if self._n_integrated == 0:
            raise RuntimeError("no frames integrated; cannot extract mesh")
        mesh = self._volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        logger.info(
            f"tsdf extract_mesh: {len(mesh.vertices)} verts, "
            f"{len(mesh.triangles)} tris from {self._n_integrated} frames"
        )
        return mesh

    def extract_cloud(self) -> o3d.geometry.PointCloud:
        """extract a point cloud by sampling zero-crossings of the sdf."""
        if self._n_integrated == 0:
            raise RuntimeError("no frames integrated; cannot extract cloud")
        pcd = self._volume.extract_point_cloud()
        logger.info(
            f"tsdf extract_cloud: {len(pcd.points)} points "
            f"from {self._n_integrated} frames"
        )
        return pcd


class MultiAngleTSDFIntegrator:
    """
    integrates rgbd frames into one fused volume plus per-angle sub-volumes.

    the fused volume is used for mesh extraction (same as single-volume
    integrator). the per-angle volumes enable vote-based filtering: a
    mesh vertex is "confirmed" by angle k if a per-angle cloud from angle
    k has a surface point near the vertex. vertices confirmed by fewer
    than min_votes angles are single-view artifacts (V-shape planes,
    apparatus reflections, etc.) and can be dropped.

    usage:
        integrator = MultiAngleTSDFIntegrator(voxel_size_m=0.001, ...)
        for angle, frames, intrinsics, pose, scale in per_angle_data:
            integrator.set_angle(angle)
            for depth, color in frames:
                integrator.integrate_frame(depth, color, intrinsics, pose, scale)
        mesh = integrator.extract_mesh()
        cloud = integrator.extract_cloud()
        votes = integrator.vertex_angle_votes(mesh.vertices, distance_m=0.002)
        # votes[i] is the number of angles that confirm vertex i
    """

    def __init__(
        self,
        voxel_size_m: float = 0.001,
        sdf_trunc_m: float = 0.004,
        depth_trunc_m: float = 0.5,
        color: bool = True,
    ):
        self.voxel_size_m = voxel_size_m
        self.sdf_trunc_m = sdf_trunc_m
        self.depth_trunc_m = depth_trunc_m

        color_type = (
            o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            if color
            else o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        )
        self._color_type = color_type

        # fused volume across all angles
        self._fused = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size_m,
            sdf_trunc=sdf_trunc_m,
            color_type=color_type,
        )
        self._n_integrated = 0

        # per-angle sub-volumes, keyed by angle_deg (rounded to 0.1)
        self._per_angle: dict[float, o3d.pipelines.integration.ScalableTSDFVolume] = {}
        self._current_angle: Optional[float] = None

    def _angle_key(self, angle_deg: float) -> float:
        return round(float(angle_deg), 1)

    def set_angle(self, angle_deg: float):
        """switch the current angle for subsequent integrate_frame calls.

        frames integrated after this call go into both the fused volume
        and the per-angle sub-volume for this angle.
        """
        key = self._angle_key(angle_deg)
        if key not in self._per_angle:
            self._per_angle[key] = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.voxel_size_m,
                sdf_trunc=self.sdf_trunc_m,
                color_type=self._color_type,
            )
        self._current_angle = key

    def integrate_frame(
        self,
        depth_array: np.ndarray,
        color_array: np.ndarray,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        plate_T_camera: np.ndarray,
        depth_scale_m: float,
    ):
        """integrate one rgbd frame into both fused and current-angle volumes."""
        if self._current_angle is None:
            raise RuntimeError(
                "no angle set; call set_angle(angle_deg) before integrate_frame"
            )
        if depth_array.shape[:2] != color_array.shape[:2]:
            raise ValueError(
                f"depth shape {depth_array.shape} != color shape "
                f"{color_array.shape[:2]}"
            )

        depth_o3d = o3d.geometry.Image(depth_array)
        color_o3d = o3d.geometry.Image(color_array)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_o3d,
            depth=depth_o3d,
            depth_scale=1.0 / depth_scale_m,
            depth_trunc=self.depth_trunc_m,
            convert_rgb_to_intensity=False,
        )
        extrinsic = np.linalg.inv(plate_T_camera)

        self._fused.integrate(rgbd, intrinsics, extrinsic)
        self._per_angle[self._current_angle].integrate(rgbd, intrinsics, extrinsic)
        self._n_integrated += 1

    @property
    def frame_count(self) -> int:
        return self._n_integrated

    @property
    def angle_count(self) -> int:
        return len(self._per_angle)

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        """extract mesh from the fused volume."""
        if self._n_integrated == 0:
            raise RuntimeError("no frames integrated; cannot extract mesh")
        mesh = self._fused.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        logger.info(
            f"tsdf extract_mesh (fused): {len(mesh.vertices)} verts, "
            f"{len(mesh.triangles)} tris from {self._n_integrated} frames "
            f"across {len(self._per_angle)} angles"
        )
        return mesh

    def extract_cloud(self) -> o3d.geometry.PointCloud:
        """extract cloud from the fused volume."""
        if self._n_integrated == 0:
            raise RuntimeError("no frames integrated; cannot extract cloud")
        pcd = self._fused.extract_point_cloud()
        logger.info(
            f"tsdf extract_cloud (fused): {len(pcd.points)} points "
            f"from {self._n_integrated} frames"
        )
        return pcd

    def extract_per_angle_clouds(self) -> dict[float, o3d.geometry.PointCloud]:
        """extract one point cloud per angle. used for vote-based filtering."""
        clouds = {}
        for angle, vol in self._per_angle.items():
            pcd = vol.extract_point_cloud()
            clouds[angle] = pcd
            logger.info(
                f"per-angle cloud at {angle:+.1f} deg: {len(pcd.points)} points"
            )
        return clouds

    def vertex_angle_votes(
        self,
        vertices: np.ndarray,
        distance_m: float,
    ) -> np.ndarray:
        """count how many angles confirm each vertex.

        a vertex is confirmed by angle k if there is any point in that
        angle's surface cloud within distance_m of the vertex.

        args:
            vertices: (N, 3) vertex positions in plate frame.
            distance_m: confirmation radius (typically 2x voxel_size_m).

        returns:
            votes: (N,) int array, each entry in [0, n_angles].
        """
        if len(self._per_angle) == 0:
            raise RuntimeError("no per-angle volumes; nothing to vote on")

        verts = np.asarray(vertices)
        votes = np.zeros(len(verts), dtype=np.int32)
        dist_sq = distance_m * distance_m

        for angle, vol in self._per_angle.items():
            pcd = vol.extract_point_cloud()
            if len(pcd.points) == 0:
                continue
            # kdtree on this angle's surface points; any vertex with a
            # neighbor within distance_m counts as "confirmed by this angle".
            kdtree = o3d.geometry.KDTreeFlann(pcd)
            confirmed = np.zeros(len(verts), dtype=bool)
            for i, v in enumerate(verts):
                # search for the single nearest neighbor; faster than radius search
                # and equivalent for our thresholding purpose
                _, _, sq_dists = kdtree.search_knn_vector_3d(v, 1)
                if sq_dists and sq_dists[0] < dist_sq:
                    confirmed[i] = True
            votes += confirmed.astype(np.int32)
            logger.info(
                f"votes from {angle:+.1f} deg: {confirmed.sum()}/{len(verts)} "
                f"vertices confirmed ({100.0 * confirmed.sum() / len(verts):.1f}%)"
            )

        return votes