#!/usr/bin/env python3
"""
Test Script: Point Cloud → Mesh Pipeline

Takes a .ply/.pcd/.obj point cloud (e.g. from phone LiDAR via Polycam/Scaniverse)
and runs it through the Open3D processing pipeline to produce a mesh.

Usage:
    python scripts/test_pointcloud_to_mesh.py --input data/my_scan.ply
    python scripts/test_pointcloud_to_mesh.py --input data/my_scan.ply --depth 9 --view pointcloud
    python scripts/test_pointcloud_to_mesh.py --input data/my_scan.ply --no-viewer --output data/output.stl
"""

import sys
import os
import argparse
import logging
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("ERROR: Open3D is required. Install with: pip install open3d")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load point cloud from file."""
    ext = os.path.splitext(path)[1].lower()
    supported = ['.ply', '.pcd', '.xyz', '.pts', '.obj']
    if ext not in supported:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {supported}")

    logger.info(f"Loading point cloud: {path}")
    pcd = o3d.io.read_point_cloud(path)

    if pcd.is_empty():
        raise ValueError(f"Failed to load or empty point cloud: {path}")

    logger.info(f"  Points: {len(pcd.points)}")
    logger.info(f"  Has colors: {pcd.has_colors()}")
    logger.info(f"  Has normals: {pcd.has_normals()}")

    bbox = pcd.get_axis_aligned_bounding_box()
    dims = bbox.max_bound - bbox.min_bound
    logger.info(f"  Bounding box: {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}")

    return pcd


def preprocess(pcd: o3d.geometry.PointCloud, voxel_size: float, nb_neighbors: int, std_ratio: float) -> o3d.geometry.PointCloud:
    """Downsample and remove outliers."""
    original_count = len(pcd.points)

    # Voxel downsample
    logger.info(f"Downsampling with voxel size {voxel_size}...")
    pcd = pcd.voxel_down_sample(voxel_size)
    logger.info(f"  After downsample: {len(pcd.points)} points (was {original_count})")

    # Statistical outlier removal
    logger.info(f"Removing outliers (neighbors={nb_neighbors}, std_ratio={std_ratio})...")
    pcd, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    removed = original_count - len(pcd.points)
    logger.info(f"  After outlier removal: {len(pcd.points)} points ({removed} removed total)")

    return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int) -> o3d.geometry.PointCloud:
    """Estimate and orient normals."""
    logger.info(f"Estimating normals (radius={radius}, max_nn={max_nn})...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    logger.info("Orienting normals consistently...")
    pcd.orient_normals_consistent_tangent_plane(k=15)
    return pcd


def reconstruct_mesh(pcd: o3d.geometry.PointCloud, depth: int, scale: float) -> o3d.geometry.TriangleMesh:
    """Poisson surface reconstruction."""
    logger.info(f"Running Poisson reconstruction (depth={depth}, scale={scale})...")
    t0 = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=False
    )
    elapsed = time.time() - t0
    logger.info(f"  Reconstruction took {elapsed:.2f}s")
    logger.info(f"  Raw mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    # Trim low-density vertices (removes the inflated boundary Poisson creates)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.01)
    logger.info(f"  Trimming low-density vertices (threshold={density_threshold:.4f})...")
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    logger.info(f"  Trimmed mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Clean up mesh artifacts."""
    logger.info("Cleaning mesh...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    logger.info(f"  Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh


def print_mesh_stats(mesh: o3d.geometry.TriangleMesh):
    """Print mesh quality statistics."""
    bbox = mesh.get_axis_aligned_bounding_box()
    dims = bbox.max_bound - bbox.min_bound
    area = mesh.get_surface_area()
    watertight = mesh.is_watertight()

    print("\n" + "=" * 50)
    print("MESH STATISTICS")
    print("=" * 50)
    print(f"  Vertices:    {len(mesh.vertices)}")
    print(f"  Triangles:   {len(mesh.triangles)}")
    print(f"  Dimensions:  {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}")
    print(f"  Surface area: {area:.4f}")
    print(f"  Watertight:  {watertight}")
    print(f"  Has normals: {mesh.has_vertex_normals()}")
    print("=" * 50 + "\n")


def view_geometry(geometry, title="Scan-to-CNC Viewer"):
    """Open interactive 3D viewer."""
    logger.info(f"Opening viewer: {title}")
    o3d.visualization.draw_geometries(
        [geometry],
        window_name=title,
        width=1280,
        height=720,
        point_show_normal=False
    )


def main():
    parser = argparse.ArgumentParser(description="Test point cloud to mesh pipeline")
    parser.add_argument('--input', '-i', required=True, help="Input point cloud file (.ply, .pcd, .obj)")
    parser.add_argument('--output', '-o', default=None, help="Output STL path (default: data/<input_name>_mesh.stl)")
    parser.add_argument('--voxel-size', type=float, default=0.5, help="Voxel downsample size in mm (default: 0.5)")
    parser.add_argument('--nb-neighbors', type=int, default=20, help="SOR neighbor count (default: 20)")
    parser.add_argument('--std-ratio', type=float, default=2.0, help="SOR std ratio (default: 2.0)")
    parser.add_argument('--normal-radius', type=float, default=2.0, help="Normal estimation radius (default: 2.0)")
    parser.add_argument('--normal-max-nn', type=int, default=30, help="Normal estimation max neighbors (default: 30)")
    parser.add_argument('--depth', type=int, default=8, help="Poisson reconstruction depth (default: 8)")
    parser.add_argument('--scale', type=float, default=1.1, help="Poisson scale factor (default: 1.1)")
    parser.add_argument('--view', choices=['mesh', 'pointcloud', 'both'], default='mesh', help="What to visualize")
    parser.add_argument('--no-viewer', action='store_true', help="Skip interactive viewer")

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Set default output path
    if args.output is None:
        os.makedirs('data', exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join('data', f'{base}_mesh.stl')

    # === PIPELINE ===
    pipeline_start = time.time()

    # Stage 1: Load
    pcd = load_point_cloud(args.input)

    # Stage 2: Preprocess
    pcd = preprocess(pcd, args.voxel_size, args.nb_neighbors, args.std_ratio)

    # Stage 3: Normal estimation
    pcd = estimate_normals(pcd, args.normal_radius, args.normal_max_nn)

    # Stage 4: Reconstruct
    mesh = reconstruct_mesh(pcd, args.depth, args.scale)

    # Stage 5: Clean
    mesh = clean_mesh(mesh)

    total_time = time.time() - pipeline_start
    logger.info(f"Total pipeline time: {total_time:.2f}s")

    # Stats
    print_mesh_stats(mesh)

    # Save
    logger.info(f"Saving mesh to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    o3d.io.write_triangle_mesh(args.output, mesh)
    print(f"Saved: {args.output}")

    # Viewer
    if not args.no_viewer:
        if args.view == 'pointcloud':
            view_geometry(pcd, "Point Cloud")
        elif args.view == 'both':
            view_geometry(pcd, "Point Cloud (close to see mesh)")
            view_geometry(mesh, "Reconstructed Mesh")
        else:
            view_geometry(mesh, "Reconstructed Mesh")


if __name__ == '__main__':
    main()