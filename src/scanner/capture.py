"""
stage 1: depth capture via intel realsense d405
captures depth + color frames, applies temporal averaging,
and outputs a ply point cloud for stage 2 (open3d processing).
no gui — fully automated, headless-compatible.
"""

import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.temporal_frames = temporal_frames
        self.decimation_magnitude = decimation_magnitude
        self.bag_file = bag_file

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
            # disable real-time throttling for playback
            playback = self.profile.get_device().as_playback()
            playback.set_real_time(False)
            logger.info("playback mode: real-time disabled, processing at full speed")
        else:
            # live sensor config
            device = self.profile.get_device()
            depth_sensor = device.first_depth_sensor()

            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 3)
                logger.info("set depth sensor to high accuracy preset")

            depth_scale = depth_sensor.get_depth_scale()
            logger.info(f"depth scale: {depth_scale} (meters per unit)")

        self.align = rs.align(rs.stream.depth)

        # let auto-exposure stabilize (skip for .bag)
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

            # each pass through temporal filter refines the estimate
            depth_frame = self._apply_filters(depth)
            color_frame = color

        if depth_frame is None or color_frame is None:
            raise RuntimeError("failed to capture valid frames from realsense")

        logger.info("temporal averaging complete")
        return depth_frame, color_frame

    def frames_to_point_cloud(self, depth_frame, color_frame):
        """
        convert aligned depth + color frames to an open3d point cloud.
        uses realsense's built-in pointcloud computation for proper
        intrinsic handling.
        """
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

        # decimation filter changes resolution so color array size
        # may not match vertex count. use texture coords instead.
        tex_coords = (
            np.asanyarray(points.get_texture_coordinates())
            .view(np.float32)
            .reshape(-1, 2)
        )

        color_h, color_w = self.height, self.width
        u = np.clip((tex_coords[:, 0] * color_w).astype(int), 0, color_w - 1)
        v = np.clip((tex_coords[:, 1] * color_h).astype(int), 0, color_h - 1)

        color_image = np.asanyarray(color_frame.get_data())
        mapped_colors = color_image[v, u][:, ::-1]  # bgr -> rgb

        # filter out zero-depth points
        mask = ~np.all(vertices == 0, axis=1)
        vertices = vertices[mask]
        mapped_colors = mapped_colors[mask]

        logger.info(f"raw point cloud: {vertices.shape[0]} points")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(mapped_colors.astype(np.float64) / 255.0)

        return pcd

    def capture(self, output_path: str = None):
        """
        full single-view capture: frames -> temporal avg -> point cloud.
        optionally saves to ply.
        returns: open3d PointCloud
        """
        depth_frame, color_frame = self.capture_averaged_frames()
        pcd = self.frames_to_point_cloud(depth_frame, color_frame)

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
    args = parser.parse_args()

    scanner = RealSenseCapture(
        width=640,
        height=480,
        fps=30,
        temporal_frames=args.frames,
        bag_file=args.bag,
    )

    # record mode: save .bag and exit
    if args.record:
        scanner.record_bag(args.record, duration_sec=args.duration)
        return

    # capture mode: live or .bag playback
    try:
        scanner.start()
        pcd = scanner.capture(output_path=args.output)
        logger.info(
            f"capture complete: {len(pcd.points)} points, "
            f"saved to {args.output}"
        )
    finally:
        scanner.stop()


if __name__ == "__main__":
    main()