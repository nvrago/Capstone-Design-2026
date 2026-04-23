"""
stage 1: depth capture via intel realsense d405
captures depth + color frames, applies temporal averaging,
transforms into plate (world) frame using arc geometry,
and outputs a ply point cloud for stage 2 (open3d processing).
no gui, fully automated, headless-compatible.
"""

import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# camera-to-plate pose
#
# plate (world) frame, as defined by dome_cloud.ply from onshape:
#   origin at arc center, on the plate bottom
#   +Z up, perpendicular to plate
#   arc sweeps in the world XZ plane, rotation axis is world +Y
#   angle 0 deg: camera at (+R, 0, Z_arc), looking toward origin
#   angle 90 deg: camera directly overhead (0, 0, Z_arc + R)
#   angle 180 deg: camera at (-R, 0, Z_arc), looking toward origin
#
# camera frame (realsense convention, matches what rs.pointcloud returns):
#   +Z forward (out of the sensor)
#   +Y down (in image)
#   +X right

def camera_to_plate(
    angle_deg: float,
    arc_radius_m: float,
    arc_center_z_m: float,
) -> np.ndarray:
    """
    build the 4x4 transform that takes a point in the camera frame
    at arc angle `angle_deg` and expresses it in the plate (world) frame.

    returns a c-contiguous float64 matrix T such that
        plate_point = T @ camera_point    (homogeneous)
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    # camera position in plate coords
    cam_pos = np.array([
        arc_radius_m * c,
        0.0,
        arc_center_z_m + arc_radius_m * s,
    ], dtype=np.float64)

    # camera forward (+Z) in plate coords = from camera toward arc center.
    # at theta=0: (-1,0,0). at theta=90: (0,0,-1). at theta=180: (1,0,0).
    forward = np.array([-c, 0.0, -s], dtype=np.float64)

    # camera down (+Y) in plate coords. rotates with the arc so the
    # image stays consistently oriented as the camera sweeps overhead.
    # at theta=0: (0,0,-1). at theta=90: (1,0,0). at theta=180: (0,0,1).
    down = np.array([s, 0.0, -c], dtype=np.float64)

    # camera right (+X) = down x forward, right-handed.
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)

    R = np.column_stack([right, down, forward]).astype(np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = cam_pos
    return np.ascontiguousarray(T, dtype=np.float64)


class RealSenseCapture:
    """automated depth capture from intel realsense d405."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        temporal_frames: int = 15,
        decimation_magnitude: int = 2,
        bag_file: str = None,
        arc_radius_m: float = 0.255,
        arc_center_z_m: float = 0.000,
        filter_black_threshold: int = None,
        visual_preset: int = 3,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.temporal_frames = temporal_frames
        self.decimation_magnitude = decimation_magnitude
        self.bag_file = bag_file

        # arc geometry for camera-to-plate transform. the pipeline
        # passes these from config; defaults are sane but should be
        # overridden with measured values.
        self.arc_radius_m = arc_radius_m
        self.arc_center_z_m = arc_center_z_m

        # optional: drop near-black pixels from captured clouds (matte
        # black background cloth, for example). per-channel threshold on
        # 0-255 rgb; a point is removed only if r, g, and b are all
        # below the threshold. None disables the filter entirely.
        self.filter_black_threshold = filter_black_threshold
        self.visual_preset = visual_preset

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.profile = None

    def _configure_streams(self):
        """enable depth and color streams, or load from .bag file."""
        if self.bag_file:
            logger.info(f"configuring playback from: {self.bag_file}")
            self.config.enable_device_from_file(self.bag_file, repeat_playback=False)
        else:
            self.config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )
            self.config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )

    def _build_filter_pipeline(self):
        """hardware post-processing filter chain."""
        # threshold filter runs FIRST to zero out invalid / out-of-range
        # pixels before anything else can propagate them. the d405 marks
        # invalid pixels as z16 max (65535), which at depth scale 1e-4
        # becomes ~6.55m; without this filter, hole-filling treats them
        # as real depth and floods the cloud with spike artifacts.
        #
        # range is tuned to the actual scanning geometry: camera sits
        # ~30cm from arc center, plate surface is ~30cm away, object top
        # is ~22-28cm. 0.20-0.35m keeps only the working volume and
        # drops the room/workbench/rig entirely.
        self.threshold = rs.threshold_filter()
        self.threshold.set_option(rs.option.min_distance, 0.20)   # 20 cm
        self.threshold.set_option(rs.option.max_distance, 0.28)   # 35 cm

        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, self.decimation_magnitude)

        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)

        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)

        self.hole_filling = rs.hole_filling_filter()

    def _apply_filters(self, depth_frame):
        """run depth frame through the filter chain."""
        frame = depth_frame
        frame = self.threshold.process(frame)   # clip invalid / out-of-range first
        frame = self.decimation.process(frame)
        frame = self.spatial.process(frame)
        frame = self.temporal.process(frame)
        frame = self.hole_filling.process(frame)
        return frame

    def start(self):
        """initialize and start the realsense pipeline."""
        self._configure_streams()
        self._build_filter_pipeline()

        logger.info("starting realsense pipeline...")
        self.profile = self.pipeline.start(self.config)

        if self.bag_file:
            playback = self.profile.get_device().as_playback()
            playback.set_real_time(False)
            logger.info("playback mode: real-time disabled, processing at full speed")
        else:
            device = self.profile.get_device()
            depth_sensor = device.first_depth_sensor()

            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, self.visual_preset)
                logger.info(f"set depth sensor visual_preset = {self.visual_preset}")

            depth_scale = depth_sensor.get_depth_scale()
            logger.info(f"depth scale: {depth_scale} (meters per unit)")

        self.align = rs.align(rs.stream.depth)

        if not self.bag_file:
            logger.info("warming up sensor (30 frames)...")
            for _ in range(30):
                self.pipeline.wait_for_frames()

        logger.info("realsense pipeline ready")

    def capture_averaged_frames(self):
        """
        capture multiple frames and let the temporal filter
        build up a stable depth estimate.
        returns aligned (depth_frame, color_frame) after averaging.
        """
        logger.info(f"capturing {self.temporal_frames} frames for temporal averaging...")

        depth_frame = None
        color_frame = None

        for i in range(self.temporal_frames):
            frameset = self.pipeline.wait_for_frames()
            aligned = self.align.process(frameset)

            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()

            if not depth or not color:
                logger.warning(f"frame {i}: missing depth or color, skipping")
                continue

            depth_frame = self._apply_filters(depth)
            color_frame = color

        if depth_frame is None or color_frame is None:
            raise RuntimeError("failed to capture valid frames from realsense")

        logger.info("temporal averaging complete")
        return depth_frame, color_frame

    def frames_to_point_cloud(self, depth_frame, color_frame):
        """
        convert aligned depth + color frames to an open3d point cloud
        in the camera frame. use capture(angle_deg=...) to get the cloud
        already transformed into plate coordinates.
        """
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

        tex_coords = (
            np.asanyarray(points.get_texture_coordinates())
            .view(np.float32)
            .reshape(-1, 2)
        )

        color_h, color_w = self.height, self.width
        u = np.clip((tex_coords[:, 0] * color_w).astype(int), 0, color_w - 1)
        v = np.clip((tex_coords[:, 1] * color_h).astype(int), 0, color_h - 1)

        color_image = np.asanyarray(color_frame.get_data())
        mapped_colors = color_image[v, u][:, ::-1]

        mask = ~np.all(vertices == 0, axis=1)
        vertices = vertices[mask]
        mapped_colors = mapped_colors[mask]

        # optional: drop near-black pixels (e.g. matte black background
        # cloth). per-channel check: a point is removed only if r, g, b
        # are all below the threshold, which protects dark-but-not-black
        # object regions. disabled unless filter_black_threshold is set.
        if self.filter_black_threshold is not None:
            t = self.filter_black_threshold
            not_black = ~np.all(mapped_colors < t, axis=1)
            n_before = len(vertices)
            vertices = vertices[not_black]
            mapped_colors = mapped_colors[not_black]
            n_removed = n_before - len(vertices)
            pct = 100.0 * n_removed / max(n_before, 1)
            logger.info(
                f"black-pixel filter (threshold={t}): removed {n_removed} "
                f"points ({pct:.1f}%), {len(vertices)} remain"
            )

        # belt-and-suspenders: the threshold filter in _apply_filters zeroes
        # out invalid/out-of-range depth, but clip again at the point-cloud
        # level in case a cloud came from elsewhere (bag replay without
        # filters, loaded from disk, etc).
        valid_range = (vertices[:, 2] > 0.20) & (vertices[:, 2] < 0.35)
        vertices = vertices[valid_range]
        mapped_colors = mapped_colors[valid_range]

        logger.info(f"raw point cloud: {vertices.shape[0]} points (camera frame)")

        # arm64 open3d segfaults inside Vector3dVector on non-contiguous
        # or float32 arrays. force contiguous float64 before handing off.
        vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        colors = np.ascontiguousarray(mapped_colors, dtype=np.float64) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def capture_frames(self, angle_deg: float = None):
        """
        capture raw depth + color arrays plus pose and intrinsics.

        returns the data tsdf integration needs without going through
        the rs.pointcloud -> open3d conversion. every n-frame temporal
        sample is captured separately (no temporal averaging) so tsdf
        can do its own volumetric averaging at the voxel level.

        args:
            angle_deg: arc angle for pose computation. if None, pose is
                       identity (camera-frame output, debug use only).

        returns:
            frames: list of (depth_array, color_array) tuples. depth is
                    uint16 raw sensor units; color is uint8 hxwx3 rgb.
                    both aligned + decimated + spatial filtered.
            intrinsics: open3d PinholeCameraIntrinsic matching the
                        decimated depth resolution.
            plate_T_camera: 4x4 float64 matrix. identity if angle_deg is None.
            depth_scale_m: meters per depth sensor unit (d405: 0.0001).
        """
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale_m = depth_sensor.get_depth_scale()

        logger.info(f"capturing {self.temporal_frames} frames for tsdf integration...")

        frames = []
        intrinsics = None

        for i in range(self.temporal_frames):
            frameset = self.pipeline.wait_for_frames()
            aligned = self.align.process(frameset)

            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()

            if not depth or not color:
                logger.warning(f"frame {i}: missing depth or color, skipping")
                continue

            # spatial + decimation filters only; NO temporal (tsdf handles
            # that at the voxel level across frames).
            if not hasattr(self, "_tsdf_decimation"):
                self._tsdf_decimation = rs.decimation_filter()
                self._tsdf_decimation.set_option(rs.option.filter_magnitude, self.decimation_magnitude)
                self._tsdf_spatial = rs.spatial_filter()
                self._tsdf_spatial.set_option(rs.option.filter_magnitude, 2)
                self._tsdf_spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
                self._tsdf_spatial.set_option(rs.option.filter_smooth_delta, 20)
            depth = self._tsdf_decimation.process(depth)
            depth = self._tsdf_spatial.process(depth)

            depth_array = np.asanyarray(depth.get_data())
            color_array = np.asanyarray(color.get_data())[:, :, ::-1].copy()

            if intrinsics is None:
                stream = depth.profile.as_video_stream_profile()
                intr = stream.get_intrinsics()
                intrinsics = o3d.camera.PinholeCameraIntrinsic(
                    width=intr.width,
                    height=intr.height,
                    fx=intr.fx,
                    fy=intr.fy,
                    cx=intr.ppx,
                    cy=intr.ppy,
                )

            # handle shape mismatch between decimated depth and full color
            if depth_array.shape[:2] != color_array.shape[:2]:
                h_target, w_target = depth_array.shape[:2]
                h_src, w_src = color_array.shape[:2]
                ys = (np.arange(h_target) * h_src / h_target).astype(np.int32)
                xs = (np.arange(w_target) * w_src / w_target).astype(np.int32)
                color_array = color_array[ys[:, None], xs[None, :]]

            frames.append((depth_array, color_array))

        if not frames:
            raise RuntimeError("failed to capture valid frames from realsense")

        if angle_deg is not None:
            plate_T_camera = camera_to_plate(
                angle_deg,
                arc_radius_m=self.arc_radius_m,
                arc_center_z_m=self.arc_center_z_m,
            )
        else:
            plate_T_camera = np.eye(4, dtype=np.float64)

        logger.info(
            f"captured {len(frames)} frames at {angle_deg or 0:.1f} deg: "
            f"depth {frames[0][0].shape}, color {frames[0][1].shape}"
        )

        return frames, intrinsics, plate_T_camera, depth_scale_m
    
    def capture(self, angle_deg: float = None, output_path: str = None):
        """
        full single-view capture: frames -> temporal avg -> point cloud.

        args:
            angle_deg: arc angle for this capture in degrees. if provided,
                       the returned cloud is in plate (world) coordinates.
                       if None, cloud is returned in camera coordinates
                       (for raw sensor debugging only, not for pipeline use).
            output_path: optional .ply output path.

        returns: open3d PointCloud. in plate frame if angle_deg was given.
        """
        depth_frame, color_frame = self.capture_averaged_frames()
        pcd = self.frames_to_point_cloud(depth_frame, color_frame)

        if angle_deg is not None:
            T = camera_to_plate(
                angle_deg,
                arc_radius_m=self.arc_radius_m,
                arc_center_z_m=self.arc_center_z_m,
            )
            pcd.transform(T)
            logger.info(
                f"transformed cloud to plate frame at {angle_deg:.1f} deg "
                f"(R={self.arc_radius_m:.3f}m, Zc={self.arc_center_z_m:.3f}m)"
            )

        if output_path:
            o3d.io.write_point_cloud(output_path, pcd)
            logger.info(f"saved point cloud to {output_path}")

        return pcd

    def stop(self):
        """shut down the pipeline."""
        self.pipeline.stop()
        logger.info("realsense pipeline stopped")

    def record_bag(self, output_bag: str, duration_sec: float = 5.0):
        """
        record raw realsense frames to a .bag file.
        run this on the pi with the d405 connected, then
        copy the .bag to your mac for offline development.
        """
        cfg = rs.config()
        cfg.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        cfg.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        cfg.enable_record_to_file(output_bag)

        pipe = rs.pipeline()
        logger.info(f"recording to {output_bag} for {duration_sec}s...")
        pipe.start(cfg)

        start = time.time()
        frame_count = 0
        try:
            while time.time() - start < duration_sec:
                pipe.wait_for_frames()
                frame_count += 1
        finally:
            pipe.stop()

        logger.info(f"recorded {frame_count} frames to {output_bag}")


def main():
    """single-view capture with cli support for live, playback, and recording."""
    import argparse

    parser = argparse.ArgumentParser(description="realsense d405 capture - stage 1")
    parser.add_argument("--bag", type=str, help="path to .bag file for playback")
    parser.add_argument("--record", type=str, help="record live frames to .bag file")
    parser.add_argument("--duration", type=float, default=5.0, help="recording duration in seconds")
    parser.add_argument("-o", "--output", type=str, default="scan_output.ply", help="output ply path")
    parser.add_argument("--frames", type=int, default=15, help="temporal averaging frame count")
    parser.add_argument("--angle", type=float, default=None,
                        help="arc angle in degrees (transforms cloud to plate frame)")
    parser.add_argument("--arc-radius", type=float, default=0.300,
                        help="arc radius in meters (default 0.300)")
    parser.add_argument("--arc-center-z", type=float, default=0.000,
                        help="arc center height above plate in meters (default 0.000)")
    parser.add_argument("--filter-black", type=int, default=None, metavar="THRESHOLD",
                        help="drop near-black pixels (0-255). typical 30-60 for "
                             "matte black cloth. omit to disable.")
    args = parser.parse_args()

    scanner = RealSenseCapture(
        width=640,
        height=480,
        fps=30,
        temporal_frames=args.frames,
        bag_file=args.bag,
        arc_radius_m=args.arc_radius,
        arc_center_z_m=args.arc_center_z,
        filter_black_threshold=args.filter_black,
    )

    if args.record:
        scanner.record_bag(args.record, duration_sec=args.duration)
        return

    try:
        scanner.start()
        pcd = scanner.capture(angle_deg=args.angle, output_path=args.output)
        logger.info(
            f"capture complete: {len(pcd.points)} points, "
            f"saved to {args.output}"
        )
    finally:
        scanner.stop()


if __name__ == "__main__":
    main()