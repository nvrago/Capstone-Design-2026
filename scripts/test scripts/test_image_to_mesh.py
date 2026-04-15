#!/usr/bin/env python3
"""
Test Script: Single Image → Depth Map → Mesh (Stress Test)

Uses MiDaS monocular depth estimation to generate a 2.5D mesh from a single photo.
This is a stress test / limitation test — NOT the production pipeline path.

NOTE: MiDaS produces relative depth (not metric). The output is a 2.5D relief,
not a full 3D model. This is useful for evaluating how the mesh pipeline handles
noisy/approximate input data.

Requirements (install separately):
    pip install torch torchvision timm

Usage:
    python scripts/test_image_to_mesh.py --input data/photo.jpg
    python scripts/test_image_to_mesh.py --input data/photo.jpg --model-type DPT_Large
    python scripts/test_image_to_mesh.py --input data/photo.jpg --view depth
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

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV is required. Install with: pip install opencv-python")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch torchvision")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load and validate image."""
    logger.info(f"Loading image: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    logger.info(f"  Image size: {img.shape[1]}x{img.shape[0]}")
    return img


def estimate_depth(img: np.ndarray, model_type: str) -> np.ndarray:
    """Run MiDaS monocular depth estimation.
    
    Returns a depth map (H x W) with relative depth values.
    Higher values = closer to camera.
    """
    logger.info(f"Loading MiDaS model: {model_type}")
    t0 = time.time()

    # Load model from torch hub
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Using device: {device}")
    midas.to(device)
    midas.eval()

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    load_time = time.time() - t0
    logger.info(f"  Model loaded in {load_time:.2f}s")

    # Run inference
    logger.info("Running depth estimation...")
    t1 = time.time()
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    infer_time = time.time() - t1
    logger.info(f"  Inference took {infer_time:.2f}s")
    logger.info(f"  Depth range: {depth.min():.2f} to {depth.max():.2f}")

    return depth


def depth_to_point_cloud(depth: np.ndarray, img: np.ndarray, depth_scale: float,
                         downsample: int) -> o3d.geometry.PointCloud:
    """Convert depth map + image to colored point cloud.
    
    Since MiDaS gives relative depth, we scale it to a reasonable
    physical range for visualization and mesh testing.
    """
    logger.info("Converting depth map to point cloud...")
    h, w = depth.shape

    # Normalize depth to [0, depth_scale] range
    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min) * depth_scale

    # Generate pixel grid (downsampled)
    ys = np.arange(0, h, downsample)
    xs = np.arange(0, w, downsample)
    xx, yy = np.meshgrid(xs, ys)

    # Sample depth and color at grid points
    z = depth_norm[yy, xx]
    colors = img[yy, xx].astype(np.float64) / 255.0

    # Create 3D points: x = pixel x, y = -pixel y (flip), z = depth
    # Scale x/y to be proportional to depth FOV
    x = xx.astype(np.float64)
    y = -yy.astype(np.float64)  # flip y so image isn't upside down

    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    colors_flat = colors.reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_flat)

    logger.info(f"  Generated {len(pcd.points)} points")
    return pcd


def depth_to_mesh_direct(depth: np.ndarray, img: np.ndarray, depth_scale: float,
                         downsample: int) -> o3d.geometry.TriangleMesh:
    """Create a triangle mesh directly from depth map using grid connectivity.
    
    This avoids Poisson reconstruction artifacts for 2.5D depth maps
    by connecting adjacent pixels as triangles directly.
    """
    logger.info("Building mesh directly from depth grid...")
    h, w = depth.shape

    # Normalize depth
    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min) * depth_scale

    # Downsample
    ys = np.arange(0, h, downsample)
    xs = np.arange(0, w, downsample)
    rows, cols = len(ys), len(xs)

    xx, yy = np.meshgrid(xs, ys)
    z = depth_norm[yy, xx]
    colors = img[yy, xx].astype(np.float64) / 255.0

    # Build vertices
    points = np.stack([
        xx.astype(np.float64).flatten(),
        -yy.astype(np.float64).flatten(),
        z.flatten()
    ], axis=1)

    colors_flat = colors.reshape(-1, 3)

    # Build triangles from grid connectivity
    # Each quad (i,j)→(i+1,j)→(i+1,j+1)→(i,j+1) becomes 2 triangles
    triangles = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            idx00 = r * cols + c
            idx01 = r * cols + (c + 1)
            idx10 = (r + 1) * cols + c
            idx11 = (r + 1) * cols + (c + 1)
            triangles.append([idx00, idx10, idx01])
            triangles.append([idx01, idx10, idx11])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_flat)
    mesh.compute_vertex_normals()

    logger.info(f"  Grid mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    return mesh


def print_mesh_stats(mesh: o3d.geometry.TriangleMesh, label: str = "MESH"):
    """Print mesh quality statistics."""
    bbox = mesh.get_axis_aligned_bounding_box()
    dims = bbox.max_bound - bbox.min_bound
    area = mesh.get_surface_area()

    print("\n" + "=" * 50)
    print(f"{label} STATISTICS")
    print("=" * 50)
    print(f"  Vertices:    {len(mesh.vertices)}")
    print(f"  Triangles:   {len(mesh.triangles)}")
    print(f"  Dimensions:  {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}")
    print(f"  Surface area: {area:.4f}")
    print(f"  Watertight:  {mesh.is_watertight()}")
    print("=" * 50 + "\n")


def save_depth_image(depth: np.ndarray, path: str):
    """Save depth map as a grayscale image for inspection."""
    d_min, d_max = depth.min(), depth.max()
    depth_vis = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imwrite(path, depth_colored)
    logger.info(f"Saved depth visualization: {path}")


def view_geometry(geometry, title="Viewer"):
    """Open interactive 3D viewer."""
    logger.info(f"Opening viewer: {title}")
    o3d.visualization.draw_geometries(
        [geometry],
        window_name=title,
        width=1280,
        height=720
    )


def main():
    parser = argparse.ArgumentParser(description="Test image to mesh pipeline (stress test)")
    parser.add_argument('--input', '-i', required=True, help="Input image file (.jpg, .png)")
    parser.add_argument('--output', '-o', default=None, help="Output STL path")
    parser.add_argument('--model-type', default='DPT_Large',
                        choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                        help="MiDaS model (default: DPT_Large)")
    parser.add_argument('--depth-scale', type=float, default=50.0,
                        help="Scale for depth values in output units (default: 50.0)")
    parser.add_argument('--downsample', type=int, default=4,
                        help="Pixel downsample factor (default: 4, lower = more detail but slower)")
    parser.add_argument('--method', choices=['grid', 'poisson', 'both'], default='grid',
                        help="Meshing method: grid (direct connectivity) or poisson (default: grid)")
    parser.add_argument('--poisson-depth', type=int, default=8,
                        help="Poisson reconstruction depth (default: 8)")
    parser.add_argument('--view', choices=['mesh', 'pointcloud', 'depth', 'all'], default='mesh',
                        help="What to visualize")
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

    # Stage 1: Load image
    img = load_image(args.input)

    # Stage 2: Depth estimation
    depth = estimate_depth(img, args.model_type)

    # Save depth visualization
    depth_img_path = os.path.join('data', os.path.splitext(os.path.basename(args.input))[0] + '_depth.png')
    os.makedirs('data', exist_ok=True)
    save_depth_image(depth, depth_img_path)

    # Stage 3: Generate point cloud
    pcd = depth_to_point_cloud(depth, img, args.depth_scale, args.downsample)

    # Stage 4: Generate mesh
    meshes = {}

    if args.method in ['grid', 'both']:
        grid_mesh = depth_to_mesh_direct(depth, img, args.depth_scale, args.downsample)
        meshes['grid'] = grid_mesh
        print_mesh_stats(grid_mesh, "GRID MESH")

    if args.method in ['poisson', 'both']:
        # For Poisson, we need normals on the point cloud
        logger.info("Estimating normals for Poisson reconstruction...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        logger.info(f"Running Poisson reconstruction (depth={args.poisson_depth})...")
        poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.poisson_depth
        )
        # Trim
        densities = np.asarray(densities)
        thresh = np.quantile(densities, 0.01)
        poisson_mesh.remove_vertices_by_mask(densities < thresh)
        poisson_mesh.compute_vertex_normals()
        meshes['poisson'] = poisson_mesh
        print_mesh_stats(poisson_mesh, "POISSON MESH")

    total_time = time.time() - pipeline_start
    logger.info(f"Total pipeline time: {total_time:.2f}s")

    # Save primary mesh
    primary_mesh = meshes.get('grid') or meshes.get('poisson')
    logger.info(f"Saving mesh to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    o3d.io.write_triangle_mesh(args.output, primary_mesh)
    print(f"Saved: {args.output}")

    # Save both if both were generated
    if args.method == 'both':
        base = os.path.splitext(args.output)[0]
        o3d.io.write_triangle_mesh(f"{base}_grid.stl", meshes['grid'])
        o3d.io.write_triangle_mesh(f"{base}_poisson.stl", meshes['poisson'])
        print(f"Saved both: {base}_grid.stl, {base}_poisson.stl")

    # Viewer
    if not args.no_viewer:
        if args.view == 'depth':
            # Show depth as image via OpenCV window
            d_vis = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
            cv2.imshow("Depth Map (press any key to close)", d_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif args.view == 'pointcloud':
            view_geometry(pcd, "Point Cloud from Depth")
        elif args.view == 'all':
            d_vis = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
            cv2.imshow("Depth Map (press any key)", d_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            view_geometry(pcd, "Point Cloud (close to see mesh)")
            view_geometry(primary_mesh, "Reconstructed Mesh")
        else:
            view_geometry(primary_mesh, "Reconstructed Mesh")


if __name__ == '__main__':
    main()