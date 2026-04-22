#!/usr/bin/env python3
"""
test_tsdf_single.py -- standalone TSDF smoke test.

captures one depth+color frame from the d405, builds a one-frame tsdf
volume, extracts a mesh via marching cubes, and saves the result.

purpose: confirm open3d's tsdf api works on the pi, the d405 intrinsics
extract correctly, and a single-frame integration produces a reasonable
mesh. should be run with the camera at a known arc position (default:
overhead, 90 deg) with an object on the plate.

usage:
    python scripts/test_tsdf_single.py
    python scripts/test_tsdf_single.py --angle 90 --voxel 0.001
    python scripts/test_tsdf_single.py --output /tmp/tsdf_test
"""

import sys
import os
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import open3d as o3d
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


def get_camera_intrinsics(pipeline_profile, stream_type=rs.stream.depth):
    """extract intrinsics from an active realsense pipeline profile.

    returns an open3d PinholeCameraIntrinsic configured for the active
    stream. open3d expects width/height/fx/fy/cx/cy in pixels.
    """
    stream = pipeline_profile.get_stream(stream_type).as_video_stream_profile()
    intr = stream.get_intrinsics()
    return o3d.camera.PinholeCameraIntrinsic(
        width=intr.width,
        height=intr.height,
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.ppx,
        cy=intr.ppy,
    )


def arc_pose_matrix(angle_deg, arc_radius_m=0.255, arc_center_z_m=0.0):
    """compute the camera pose in plate frame for a given arc angle.

    matches the math in scanner.capture.transform_to_plate_frame:
    the camera sits on the arc at radius R, sweeping the XZ plane.
    angle 0 = +X side, angle 90 = overhead (+Z), angle 180 = -X side.

    returns a 4x4 transform matrix: plate_T_camera (camera pose in
    plate frame, suitable for open3d's tsdf.integrate which wants the
    camera-to-world / world_T_camera transform).
    """
    angle_rad = np.deg2rad(angle_deg)

    # camera position in plate frame
    cam_x = arc_radius_m * np.cos(angle_rad)
    cam_z = arc_radius_m * np.sin(angle_rad) + arc_center_z_m
    cam_y = 0.0

    # camera orientation: looking AT plate origin (0,0,0) FROM camera position.
    # camera +Z is the optical axis pointing forward into the scene.
    # camera +X is right, camera +Y is down (opencv/realsense convention).
    look_dir = -np.array([cam_x, cam_y, cam_z])
    look_dir /= np.linalg.norm(look_dir)

    # arbitrary up vector. for arc sweeping in XZ, the camera's "up"
    # in plate frame is +Y (since arc never rotates about Z).
    # gram-schmidt to build orthonormal basis.
    world_up = np.array([0.0, 1.0, 0.0])

    # camera +X (right) = up × forward (right-handed)
    cam_x_axis = np.cross(world_up, look_dir)
    cam_x_axis /= np.linalg.norm(cam_x_axis)

    # camera +Y (down) = forward × right
    cam_y_axis = np.cross(look_dir, cam_x_axis)

    # build 4x4: columns are camera axes in plate frame, last col is position
    T = np.eye(4)
    T[:3, 0] = cam_x_axis
    T[:3, 1] = cam_y_axis
    T[:3, 2] = look_dir
    T[:3, 3] = [cam_x, cam_y, cam_z]

    return T


def capture_one_rgbd(width=640, height=480, fps=30, warmup_frames=30,
                      averaging_frames=15, laser_power=None):
    """capture one depth+color frame from the d405 with temporal averaging.

    returns:
        depth_o3d: open3d Image, uint16 millimeters
        color_o3d: open3d Image, uint8 rgb
        intrinsics: open3d PinholeCameraIntrinsic
        depth_scale_m: float, meters per depth unit (for d405 typically 0.0001)
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)

    # set high accuracy preset
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 3)  # high accuracy
    if laser_power is not None:
        # try several option names since names vary by device.
        # d405 typically exposes 'projector_power'; older devices use 'laser_power'.
        candidates = []
        for opt_name in ('projector_power', 'laser_power'):
            opt = getattr(rs.option, opt_name, None)
            if opt is not None and depth_sensor.supports(opt):
                candidates.append((opt_name, opt))
        if not candidates:
            # log every supported option to help diagnose
            supported = [str(o) for o in dir(rs.option) if not o.startswith('_')
                         and depth_sensor.supports(getattr(rs.option, o, None))]
            logger.warning(f"no projector/laser power option found. "
                           f"supported options on this sensor: {supported[:20]}...")
        else:
            opt_name, opt = candidates[0]
            depth_sensor.set_option(opt, laser_power)
            logger.info(f"{opt_name} set to {laser_power}")
    depth_scale_m = depth_sensor.get_depth_scale()
    logger.info(f"depth scale: {depth_scale_m} m/unit")

    # get intrinsics from the depth stream
    intrinsics = get_camera_intrinsics(profile, rs.stream.depth)
    logger.info(f"intrinsics: {intrinsics.width}x{intrinsics.height}, "
                f"fx={intrinsics.intrinsic_matrix[0,0]:.2f}, "
                f"fy={intrinsics.intrinsic_matrix[1,1]:.2f}")

    # post-processing filters (matches main pipeline)
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    align = rs.align(rs.stream.depth)

    # warmup
    logger.info(f"warming up ({warmup_frames} frames)...")
    for _ in range(warmup_frames):
        pipeline.wait_for_frames()

    # temporal averaging
    logger.info(f"capturing {averaging_frames} frames for temporal averaging...")
    depth_frame = None
    color_frame = None
    for i in range(averaging_frames):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        depth = decimation.process(depth)
        depth = spatial.process(depth)
        depth = temporal.process(depth)
        if i == averaging_frames - 1:
            depth_frame = depth
            color_frame = color

    pipeline.stop()

    depth_array = np.asanyarray(depth_frame.get_data())
    color_array = np.asanyarray(color_frame.get_data())[:, :, ::-1].copy()  # bgr -> rgb

    # decimation downsamples depth; resize color to match if needed
    if depth_array.shape[:2] != color_array.shape[:2]:
        logger.warning(
            f"depth shape {depth_array.shape} != color shape {color_array.shape[:2]}, "
            f"adjusting intrinsics to match depth"
        )
        # rebuild intrinsics for decimated depth resolution
        scale_x = depth_array.shape[1] / intrinsics.width
        scale_y = depth_array.shape[0] / intrinsics.height
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=depth_array.shape[1],
            height=depth_array.shape[0],
            fx=intrinsics.intrinsic_matrix[0, 0] * scale_x,
            fy=intrinsics.intrinsic_matrix[1, 1] * scale_y,
            cx=intrinsics.intrinsic_matrix[0, 2] * scale_x,
            cy=intrinsics.intrinsic_matrix[1, 2] * scale_y,
        )
        # color resize: simple nearest-neighbor downsample using numpy indexing.
        # decimation is integer (typically 2x), so this is exact.
        h_target, w_target = depth_array.shape[:2]
        h_src, w_src = color_array.shape[:2]
        ys = (np.arange(h_target) * h_src / h_target).astype(np.int32)
        xs = (np.arange(w_target) * w_src / w_target).astype(np.int32)
        color_array = color_array[ys[:, None], xs[None, :]]

    depth_o3d = o3d.geometry.Image(depth_array)
    color_o3d = o3d.geometry.Image(color_array)

    logger.info(
        f"captured: depth {depth_array.shape} dtype={depth_array.dtype}, "
        f"color {color_array.shape} dtype={color_array.dtype}, "
        f"valid depth pixels: {(depth_array > 0).sum()}/{depth_array.size}"
    )

    return depth_o3d, color_o3d, intrinsics, depth_scale_m


def build_tsdf_one_frame(depth_o3d, color_o3d, intrinsics, depth_scale_m,
                          camera_pose, voxel_size=0.001, sdf_trunc=0.004):
    """integrate a single rgbd frame into a fresh tsdf volume.

    returns the integrated TSDFVolume.
    """
    # use scalable for now; uniform requires committing to volume bounds upfront
    # and we want this test to "just work" without bbox tuning
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # open3d wants depth in meters via depth_scale (units per meter, so 1/depth_scale_m)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        depth_scale=1.0 / depth_scale_m,  # convert depth units to meters
        depth_trunc=0.5,                  # ignore beyond 50cm (d405 working range)
        convert_rgb_to_intensity=False,
    )

    # open3d's integrate expects extrinsic = world_T_camera inverse = camera_T_world
    # camera_pose is plate_T_camera, so extrinsic is its inverse
    extrinsic = np.linalg.inv(camera_pose)

    tsdf.integrate(rgbd, intrinsics, extrinsic)

    logger.info("tsdf integration complete")
    return tsdf


def main():
    p = argparse.ArgumentParser(description="single-frame TSDF smoke test")
    p.add_argument("--angle", type=float, default=90.0,
                   help="arc angle in degrees for camera pose math (default: 90)")
    p.add_argument("--voxel", type=float, default=0.001,
                   help="tsdf voxel size in meters (default: 0.001 = 1mm)")
    p.add_argument("--sdf-trunc", type=float, default=0.004,
                   help="sdf truncation distance in meters (default: 0.004 = 4mm)")
    p.add_argument("--output", type=str, default="/tmp/tsdf_test",
                   help="output directory (default: /tmp/tsdf_test)")
    p.add_argument("--arc-radius", type=float, default=0.255,
                   help="arc radius in meters (default: 0.255)")
    p.add_argument("--arc-center-z", type=float, default=0.0,
                   help="arc center z in meters (default: 0.0)")
    p.add_argument("--laser-power", type=float, default=None,
                   help="d405 IR laser power in mW (default: sensor default 150). "
                        "try lower values (30-90) for shiny/reflective surfaces.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # 1. capture one rgbd frame
    depth_o3d, color_o3d, intrinsics, depth_scale_m = capture_one_rgbd(laser_power=args.laser_power)

    # 2. compute camera pose for the given arc angle
    pose = arc_pose_matrix(args.angle, args.arc_radius, args.arc_center_z)
    logger.info(f"camera pose at {args.angle} deg:")
    logger.info(f"  position (m): {pose[:3, 3]}")
    logger.info(f"  forward (cam +Z in plate frame): {pose[:3, 2]}")

    # 3. integrate into tsdf
    tsdf = build_tsdf_one_frame(
        depth_o3d, color_o3d, intrinsics, depth_scale_m,
        pose, args.voxel, args.sdf_trunc,
    )

    # 4. extract mesh
    logger.info("extracting mesh via marching cubes...")
    mesh = tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    logger.info(f"extracted: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # 5. extract point cloud (alternate output for inspection)
    pcd = tsdf.extract_point_cloud()
    logger.info(f"extracted point cloud: {len(pcd.points)} points")

    # 6. save outputs
    o3d.io.write_triangle_mesh(str(output / "tsdf_mesh.ply"), mesh)
    o3d.io.write_triangle_mesh(str(output / "tsdf_mesh.stl"), mesh)
    o3d.io.write_point_cloud(str(output / "tsdf_cloud.ply"), pcd)
    o3d.io.write_image(str(output / "depth.png"), depth_o3d)
    o3d.io.write_image(str(output / "color.png"), color_o3d)

    logger.info(f"outputs in {output}/")
    logger.info("  tsdf_mesh.ply / tsdf_mesh.stl: marching-cubes mesh")
    logger.info("  tsdf_cloud.ply: point cloud from tsdf")
    logger.info("  depth.png / color.png: input images")

    # 7. report mesh bounds for sanity
    if len(mesh.vertices) > 0:
        verts = np.asarray(mesh.vertices)
        logger.info(f"mesh bounds (m, plate frame):")
        logger.info(f"  x: {verts[:, 0].min():.4f} to {verts[:, 0].max():.4f}")
        logger.info(f"  y: {verts[:, 1].min():.4f} to {verts[:, 1].max():.4f}")
        logger.info(f"  z: {verts[:, 2].min():.4f} to {verts[:, 2].max():.4f}")


if __name__ == "__main__":
    main()