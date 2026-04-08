#!/usr/bin/env python3
"""
end-to-end pipeline smoke test.
runs each stage independently so you can see where things break.
optional visualization at each stage for debugging.

usage:
  # live capture from d405
  python scripts/test_pipeline.py

  # from a .bag file
  python scripts/test_pipeline.py --bag data/bags/test.bag

  # with visualization at each stage
  python scripts/test_pipeline.py --visualize

  # skip capture, start from existing ply
  python scripts/test_pipeline.py --input data/scan/pointcloud.ply
"""

import argparse
import logging
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_separator(stage_name: str):
    logger.info("")
    logger.info(f"{'=' * 50}")
    logger.info(f"  {stage_name}")
    logger.info(f"{'=' * 50}")


def print_pointcloud_stats(pcd, label: str):
    """log useful stats about a point cloud."""
    points = np.asarray(pcd.pcd.points)
    if len(points) == 0:
        logger.warning(f"{label}: empty point cloud")
        return

    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    extent = max_bound - min_bound

    logger.info(f"{label}:")
    logger.info(f"  points: {len(points)}")
    logger.info(f"  bounds min: [{min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f}]")
    logger.info(f"  bounds max: [{max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f}]")
    logger.info(f"  extent: [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")
    logger.info(f"  has normals: {pcd.has_normals()}")
    logger.info(f"  has colors: {len(pcd.pcd.colors) > 0}")


def print_mesh_stats(mesh, label: str):
    """log useful stats about a mesh."""
    logger.info(f"{label}:")
    logger.info(f"  vertices: {mesh.vertex_count}")
    logger.info(f"  triangles: {mesh.triangle_count}")
    min_bound, max_bound = mesh.get_bounds()
    extent = max_bound - min_bound
    logger.info(f"  bounds min: [{min_bound[0]:.3f}, {min_bound[1]:.3f}, {min_bound[2]:.3f}]")
    logger.info(f"  bounds max: [{max_bound[0]:.3f}, {max_bound[1]:.3f}, {max_bound[2]:.3f}]")
    logger.info(f"  extent: [{extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f}]")


def visualize(geometry, window_name: str):
    """show an open3d visualization window."""
    try:
        import open3d as o3d
        geo = geometry.pcd if hasattr(geometry, "pcd") else geometry.mesh
        o3d.visualization.draw_geometries([geo], window_name=window_name)
    except Exception as e:
        logger.warning(f"visualization failed: {e}")


def test_stage_1(bag_file: str = None, show: bool = False):
    """test depth capture from realsense d405."""
    print_separator("stage 1: depth capture")

    from scanner.capture import RealSenseCapture

    output_path = "data/test/raw_capture.ply"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    scanner = RealSenseCapture(
        width=640,
        height=480,
        fps=30,
        temporal_frames=15,
        bag_file=bag_file,
    )

    start = time.time()
    try:
        scanner.start()
        o3d_pcd = scanner.capture(output_path=output_path)
    finally:
        scanner.stop()

    elapsed = time.time() - start
    logger.info(f"capture time: {elapsed:.1f}s")

    # convert to wrapper for stats
    from processing.pointcloud import PointCloud
    pcd = PointCloud(np.asarray(o3d_pcd.points))
    if o3d_pcd.has_colors():
        pcd.pcd.colors = o3d_pcd.colors

    print_pointcloud_stats(pcd, "raw capture")

    if show:
        visualize(pcd, "stage 1: raw capture")

    return pcd


def test_stage_2(pcd, show: bool = False):
    """test point cloud processing."""
    print_separator("stage 2: point cloud processing")

    start = time.time()

    # voxel downsample
    pcd_down = pcd.downsample_voxel(0.005)
    print_pointcloud_stats(pcd_down, "after voxel downsample (0.5mm)")

    # outlier removal
    pcd_clean = pcd_down.remove_outliers_statistical(nb_neighbors=20, std_ratio=2.0)
    print_pointcloud_stats(pcd_clean, "after outlier removal")

    # normal estimation
    pcd_clean.estimate_normals(radius=2.0)
    print_pointcloud_stats(pcd_clean, "after normal estimation")

    # save
    output_path = "data/test/processed.ply"
    pcd_clean.save(output_path)

    elapsed = time.time() - start
    logger.info(f"processing time: {elapsed:.1f}s")

    if show:
        visualize(pcd_clean, "stage 2: processed point cloud")

    return pcd_clean


def test_stage_3(pcd, show: bool = False):
    """test mesh reconstruction."""
    print_separator("stage 3: mesh reconstruction")

    from processing.mesh import MeshReconstructor

    start = time.time()

    reconstructor = MeshReconstructor()

    # try poisson first
    try:
        mesh = reconstructor.poisson_with_density_filter(pcd, depth=9, density_threshold=0.1)
        logger.info("used poisson reconstruction")
    except Exception as e:
        logger.warning(f"poisson failed: {e}, falling back to ball pivoting")
        mesh = reconstructor.ball_pivoting(pcd, radii=[0.5, 1.0, 2.0])

    mesh.remove_degenerate()
    mesh.compute_normals()
    print_mesh_stats(mesh, "reconstructed mesh")

    # save
    mesh.save(str(Path("data/test/mesh.stl")))
    mesh.save(str(Path("data/test/mesh.ply")))

    elapsed = time.time() - start
    logger.info(f"reconstruction time: {elapsed:.1f}s")

    if show:
        visualize(mesh, "stage 3: reconstructed mesh")

    return mesh


def test_stage_4(mesh):
    """test toolpath generation and gcode output."""
    print_separator("stage 4: toolpath + gcode")

    try:
        from processing.toolpath import ToolpathGenerator, CutterDef, CutterType
        from gcode.writer import GcodeWriter, GcodeConfig
    except ImportError:
        logger.error("opencamlib not installed, skipping stage 4")
        return None

    start = time.time()

    cutter = CutterDef(
        type=CutterType.CYLINDRICAL,
        diameter=6.0,
        length=25.0,
    )

    generator = ToolpathGenerator(cutter=cutter)
    generator.load_mesh(mesh)

    min_bound, max_bound = mesh.get_bounds()
    logger.info(f"generating toolpath over mesh bounds")

    passes = generator.surface_dropcutter(
        x_min=min_bound[0],
        x_max=max_bound[0],
        y_min=min_bound[1],
        y_max=max_bound[1],
        stepover=2.0,
        direction="x",
    )

    clearance_z = 10.0
    passes = generator.add_lead_in_out(passes, clearance_z)

    logger.info(f"toolpath passes: {len(passes)}")

    # gcode
    writer = GcodeWriter(GcodeConfig(
        feed_rate=500,
        plunge_rate=100,
        spindle_speed=10000,
        dialect="grbl",
    ))

    writer.from_toolpath(passes, clearance_z=clearance_z)

    output_path = "data/test/output.gcode"
    writer.save(output_path)

    est_time = writer.estimate_time()
    logger.info(f"gcode lines: {len(writer.lines)}")
    logger.info(f"estimated machining time: {est_time:.1f} min")

    elapsed = time.time() - start
    logger.info(f"toolpath + gcode time: {elapsed:.1f}s")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="end-to-end pipeline smoke test")
    parser.add_argument("--bag", type=str, help=".bag file for playback")
    parser.add_argument("--input", type=str, help="skip capture, load existing ply")
    parser.add_argument("--visualize", action="store_true", help="show visualization at each stage")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4],
                        help="run only up to this stage")
    args = parser.parse_args()

    max_stage = args.stage or 4

    logger.info("scan-to-cnc end-to-end smoke test")
    logger.info(f"running stages 1-{max_stage}")
    total_start = time.time()

    # stage 1: capture
    if args.input:
        logger.info(f"skipping capture, loading from {args.input}")
        from processing.pointcloud import PointCloud
        if args.input.endswith(".npy"):
            pcd = PointCloud(np.load(args.input))
        else:
            pcd = PointCloud.from_file(args.input)
        print_pointcloud_stats(pcd, "loaded input")
        if args.visualize:
            visualize(pcd, "loaded input")
    else:
        pcd = test_stage_1(bag_file=args.bag, show=args.visualize)

    if max_stage < 2:
        return

    # stage 2: process
    pcd = test_stage_2(pcd, show=args.visualize)

    if max_stage < 3:
        return

    # stage 3: mesh
    mesh = test_stage_3(pcd, show=args.visualize)

    if max_stage < 4:
        return

    # stage 4: toolpath + gcode
    test_stage_4(mesh)

    total_elapsed = time.time() - total_start
    print_separator("summary")
    logger.info(f"total time: {total_elapsed:.1f}s")
    logger.info("outputs saved to data/test/")
    logger.info("  raw_capture.ply")
    logger.info("  processed.ply")
    logger.info("  mesh.stl")
    logger.info("  mesh.ply")
    logger.info("  output.gcode")


if __name__ == "__main__":
    main()