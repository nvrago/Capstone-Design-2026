#!/usr/bin/env python3
"""
test_tsdf_single.py -- standalone TSDF smoke test.

captures n depth+color frames from the d405 at a single arc position,
optionally masks apparatus pixels against the cad mesh (option A),
integrates each frame into a tsdf volume, extracts a mesh via marching
cubes, and optionally runs the extracted cloud through cad mask + poisson
(option B) for comparison.

usage:
    python scripts/test_tsdf_single.py
    python scripts/test_tsdf_single.py --angle 90 --frames 15
    python scripts/test_tsdf_single.py --cad-mask-depth          # option A
    python scripts/test_tsdf_single.py --cad-mask                # option B
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
    """extract intrinsics from an active realsense pipeline profile."""
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
    """compute the camera pose in plate frame for a given arc angle."""
    angle_rad = np.deg2rad(angle_deg)

    cam_x = arc_radius_m * np.cos(angle_rad)
    cam_z = arc_radius_m * np.sin(angle_rad) + arc_center_z_m
    cam_y = 0.0

    look_dir = -np.array([cam_x, cam_y, cam_z])
    look_dir /= np.linalg.norm(look_dir)

    world_up = np.array([0.0, 1.0, 0.0])
    cam_x_axis = np.cross(world_up, look_dir)
    cam_x_axis /= np.linalg.norm(cam_x_axis)
    cam_y_axis = np.cross(look_dir, cam_x_axis)

    T = np.eye(4)
    T[:3, 0] = cam_x_axis
    T[:3, 1] = cam_y_axis
    T[:3, 2] = look_dir
    T[:3, 3] = [cam_x, cam_y, cam_z]

    return T


def capture_rgbd_frames(width=640, height=480, fps=30, warmup_frames=30,
                        n_frames=15):
    """capture n individual depth+color frames for tsdf integration.

    no temporal filter here. tsdf does the equivalent work at the voxel
    level; pre-averaging frames would defeat the point of volumetric fusion.
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 4)  # high density (tsdf averages out noise at voxel level)
    depth_scale_m = depth_sensor.get_depth_scale()
    logger.info(f"depth scale: {depth_scale_m} m/unit")

    intrinsics = get_camera_intrinsics(profile, rs.stream.depth)
    logger.info(f"intrinsics: {intrinsics.width}x{intrinsics.height}, "
                f"fx={intrinsics.intrinsic_matrix[0,0]:.2f}, "
                f"fy={intrinsics.intrinsic_matrix[1,1]:.2f}")

    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)
    spatial = rs.spatial_filter()
    align = rs.align(rs.stream.depth)

    logger.info(f"warming up ({warmup_frames} frames)...")
    for _ in range(warmup_frames):
        pipeline.wait_for_frames()

    logger.info(f"capturing {n_frames} frames for tsdf integration...")
    captured = []
    rebuilt_intrinsics = None
    for i in range(n_frames):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        depth = decimation.process(depth)
        depth = spatial.process(depth)

        depth_array = np.asanyarray(depth.get_data())
        color_array = np.asanyarray(color.get_data())[:, :, ::-1].copy()

        if depth_array.shape[:2] != color_array.shape[:2]:
            if rebuilt_intrinsics is None:
                logger.warning(
                    f"depth shape {depth_array.shape} != color shape "
                    f"{color_array.shape[:2]}, adjusting intrinsics"
                )
                scale_x = depth_array.shape[1] / intrinsics.width
                scale_y = depth_array.shape[0] / intrinsics.height
                rebuilt_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                    width=depth_array.shape[1],
                    height=depth_array.shape[0],
                    fx=intrinsics.intrinsic_matrix[0, 0] * scale_x,
                    fy=intrinsics.intrinsic_matrix[1, 1] * scale_y,
                    cx=intrinsics.intrinsic_matrix[0, 2] * scale_x,
                    cy=intrinsics.intrinsic_matrix[1, 2] * scale_y,
                )
            h_target, w_target = depth_array.shape[:2]
            h_src, w_src = color_array.shape[:2]
            ys = (np.arange(h_target) * h_src / h_target).astype(np.int32)
            xs = (np.arange(w_target) * w_src / w_target).astype(np.int32)
            color_array = color_array[ys[:, None], xs[None, :]]

        captured.append((
            o3d.geometry.Image(depth_array),
            o3d.geometry.Image(color_array),
        ))

    pipeline.stop()

    if rebuilt_intrinsics is not None:
        intrinsics = rebuilt_intrinsics

    last_depth = np.asanyarray(captured[-1][0])
    logger.info(
        f"captured {len(captured)} frames: depth {last_depth.shape}, "
        f"valid pixels last frame: {(last_depth > 0).sum()}/{last_depth.size}"
    )

    return captured, intrinsics, depth_scale_m


def build_tsdf_from_frames(frames, intrinsics, depth_scale_m, camera_pose,
                           voxel_size=0.001, sdf_trunc=0.004,
                           depth_trunc=0.5):
    """integrate n rgbd frames into a fresh tsdf volume at a single pose."""
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    extrinsic = np.linalg.inv(camera_pose)
    depth_scale_units_per_m = 1.0 / depth_scale_m

    for i, (depth_o3d, color_o3d) in enumerate(frames):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_o3d,
            depth=depth_o3d,
            depth_scale=depth_scale_units_per_m,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        tsdf.integrate(rgbd, intrinsics, extrinsic)

    logger.info(f"tsdf integration complete: {len(frames)} frames integrated")
    return tsdf


def mesh_via_poisson(pcd, depth=9, density_trim_quantile=0.05):
    """reconstruct a mesh from a point cloud using poisson."""
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.005, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    densities = np.asarray(densities)
    if density_trim_quantile > 0.0:
        thresh = np.quantile(densities, density_trim_quantile)
        mesh.remove_vertices_by_mask(densities < thresh)
    mesh.compute_vertex_normals()
    return mesh


def main():
    p = argparse.ArgumentParser(description="single-angle tsdf smoke test")
    p.add_argument("--angle", type=float, default=90.0,
                   help="arc angle in degrees (default: 90)")
    p.add_argument("--frames", type=int, default=15,
                   help="number of frames to integrate (default: 15)")
    p.add_argument("--voxel", type=float, default=0.001,
                   help="tsdf voxel size in meters (default: 0.001 = 1mm)")
    p.add_argument("--sdf-trunc", type=float, default=0.004,
                   help="sdf truncation in meters (default: 0.004 = 4mm)")
    p.add_argument("--depth-trunc", type=float, default=0.5,
                   help="max depth in meters (default: 0.5)")
    p.add_argument("--output", type=str, default="/tmp/tsdf_test",
                   help="output directory (default: /tmp/tsdf_test)")
    p.add_argument("--arc-radius", type=float, default=0.255,
                   help="arc radius in meters (default: 0.255)")
    p.add_argument("--arc-center-z", type=float, default=0.0,
                   help="arc center z in meters (default: 0.0)")
    p.add_argument("--cad-mask", action="store_true",
                   help="option B: mask cloud after tsdf, then poisson re-mesh")
    p.add_argument("--cad-mask-depth", action="store_true",
                   help="option A: mask depth frames before tsdf integration")
    p.add_argument("--mask-threshold", type=float, default=0.005,
                   help="depth mask match threshold in meters (default: 0.005 = 5mm)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # 1. capture n frames
    frames, intrinsics, depth_scale_m = capture_rgbd_frames(n_frames=args.frames)

    # 2. compute camera pose for the given arc angle
    pose = arc_pose_matrix(args.angle, args.arc_radius, args.arc_center_z)
    logger.info(f"camera pose at {args.angle} deg:")
    logger.info(f"  position (m): {pose[:3, 3]}")
    logger.info(f"  forward (cam +Z in plate frame): {pose[:3, 2]}")

    # 2b. optional cad-based depth masking (option A: pre-integration)
    if args.cad_mask_depth:
        from processing.cad_mask import build_depth_mask_from_cad
        logger.info("masking apparatus pixels in depth frames (pre-tsdf)...")
        masked_frames = []
        for i, (depth_o3d, color_o3d) in enumerate(frames):
            depth_array = np.asarray(depth_o3d)
            masked_depth = build_depth_mask_from_cad(
                depth_array, intrinsics, pose, depth_scale_m,
                match_threshold_m=args.mask_threshold,
            )
            masked_frames.append((o3d.geometry.Image(masked_depth), color_o3d))
        frames = masked_frames

    # 3. integrate into tsdf
    tsdf = build_tsdf_from_frames(
        frames, intrinsics, depth_scale_m, pose,
        args.voxel, args.sdf_trunc, args.depth_trunc,
    )

    # 4. direct tsdf outputs (mesh + cloud)
    logger.info("extracting direct tsdf mesh via marching cubes...")
    mesh_direct = tsdf.extract_triangle_mesh()
    mesh_direct.compute_vertex_normals()
    logger.info(f"  direct mesh: {len(mesh_direct.vertices)} verts, "
                f"{len(mesh_direct.triangles)} tris")

    pcd_direct = tsdf.extract_point_cloud()
    logger.info(f"  direct cloud: {len(pcd_direct.points)} points")

    o3d.io.write_triangle_mesh(str(output / "tsdf_mesh.ply"), mesh_direct)
    o3d.io.write_triangle_mesh(str(output / "tsdf_mesh.stl"), mesh_direct)
    o3d.io.write_point_cloud(str(output / "tsdf_cloud.ply"), pcd_direct)

    # 5. optional: option B cad-mask + poisson comparison path
    if args.cad_mask:
        from processing.cad_mask import apply_cad_mask
        logger.info("applying cad mask to tsdf cloud (post-integration)...")
        pcd_masked = apply_cad_mask(pcd_direct)
        logger.info(f"  masked cloud: {len(pcd_masked.points)} points")

        if len(pcd_masked.points) > 100:
            logger.info("reconstructing poisson mesh from masked cloud...")
            mesh_masked = mesh_via_poisson(pcd_masked)
            logger.info(f"  masked mesh: {len(mesh_masked.vertices)} verts, "
                        f"{len(mesh_masked.triangles)} tris")
            o3d.io.write_point_cloud(str(output / "tsdf_cloud_masked.ply"), pcd_masked)
            o3d.io.write_triangle_mesh(str(output / "tsdf_mesh_masked.ply"), mesh_masked)
            o3d.io.write_triangle_mesh(str(output / "tsdf_mesh_masked.stl"), mesh_masked)
        else:
            logger.warning("too few points after cad mask, skipping poisson")

    # 6. save one representative rgbd for reference
    if frames:
        o3d.io.write_image(str(output / "depth.png"), frames[-1][0])
        o3d.io.write_image(str(output / "color.png"), frames[-1][1])

    # 7. output summary
    logger.info(f"outputs in {output}/")
    if args.cad_mask_depth:
        logger.info("  tsdf_mesh.{ply,stl}: direct marching-cubes mesh (cad-depth-masked)")
    else:
        logger.info("  tsdf_mesh.{ply,stl}: direct marching-cubes mesh (no mask)")
    logger.info("  tsdf_cloud.ply: point cloud from tsdf")
    if args.cad_mask:
        logger.info("  tsdf_cloud_masked.ply: after post-hoc cad mask")
        logger.info("  tsdf_mesh_masked.{ply,stl}: cad-masked + poisson mesh")
    logger.info("  depth.png / color.png: last input frame")

    # 8. bounds sanity
    if len(mesh_direct.vertices) > 0:
        verts = np.asarray(mesh_direct.vertices)
        logger.info("direct mesh bounds (m, plate frame):")
        logger.info(f"  x: {verts[:, 0].min():.4f} to {verts[:, 0].max():.4f}")
        logger.info(f"  y: {verts[:, 1].min():.4f} to {verts[:, 1].max():.4f}")
        logger.info(f"  z: {verts[:, 2].min():.4f} to {verts[:, 2].max():.4f}")


if __name__ == "__main__":
    main()