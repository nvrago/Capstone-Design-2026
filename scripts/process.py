#!/usr/bin/env python3
"""
Processing Script

Converts point cloud to mesh and generates G-code toolpaths.
Can process existing scan data without hardware.
"""

import argparse
import logging
import sys
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from processing.pointcloud import PointCloud
from processing.mesh import Mesh, MeshReconstructor
from processing.toolpath import ToolpathGenerator, CutterDef, CutterType
from gcode.writer import GcodeWriter, GcodeConfig

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load processing configuration."""
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def process_pointcloud(input_path: Path, config: dict) -> PointCloud:
    """Load and process point cloud."""
    logger.info(f"Loading point cloud from {input_path}")
    
    # Load based on extension
    if input_path.suffix == '.npy':
        points = np.load(input_path)
        pc = PointCloud(points)
    else:
        pc = PointCloud.from_file(str(input_path))
        
    logger.info(f"Loaded {len(pc)} points")
    
    # Get processing config
    pc_config = config.get('pointcloud', {})
    
    # Outlier removal
    outlier_cfg = pc_config.get('outlier_removal', {})
    if outlier_cfg:
        pc = pc.remove_outliers_statistical(
            nb_neighbors=outlier_cfg.get('nb_neighbors', 20),
            std_ratio=outlier_cfg.get('std_ratio', 2.0)
        )
        
    # Voxel downsampling
    voxel_size = pc_config.get('voxel_size')
    if voxel_size:
        pc = pc.downsample_voxel(voxel_size)
        
    # Estimate normals
    normal_radius = pc_config.get('normal_radius', 2.0)
    pc.estimate_normals(radius=normal_radius)
    
    logger.info(f"Processed point cloud: {len(pc)} points")
    return pc


def reconstruct_mesh(pointcloud: PointCloud, config: dict) -> Mesh:
    """Reconstruct mesh from point cloud."""
    mesh_config = config.get('mesh', {})
    method = mesh_config.get('method', 'poisson')
    
    reconstructor = MeshReconstructor()
    
    if method == 'poisson':
        poisson_cfg = mesh_config.get('poisson', {})
        mesh = reconstructor.poisson_with_density_filter(
            pointcloud,
            depth=poisson_cfg.get('depth', 9),
            density_threshold=0.1
        )
    elif method == 'ball_pivoting':
        bp_cfg = mesh_config.get('ball_pivoting', {})
        mesh = reconstructor.ball_pivoting(
            pointcloud,
            radii=bp_cfg.get('radii', [0.5, 1.0, 2.0])
        )
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")
        
    # Post-processing
    mesh.remove_degenerate()
    
    smooth_iter = mesh_config.get('smooth_iterations', 0)
    if smooth_iter > 0:
        mesh.smooth_taubin(iterations=smooth_iter)
        
    simplify_target = mesh_config.get('simplify_target')
    if simplify_target:
        mesh = mesh.simplify(target_triangles=simplify_target)
        
    mesh.compute_normals()
    
    logger.info(f"Reconstructed mesh: {mesh.vertex_count} vertices, "
               f"{mesh.triangle_count} triangles")
    return mesh


def generate_toolpath(mesh: Mesh, config: dict):
    """Generate toolpath from mesh."""
    tp_config = config.get('toolpath', {})
    cutter_cfg = tp_config.get('cutter', {})
    
    # Create cutter
    cutter_type = CutterType(cutter_cfg.get('type', 'cylindrical'))
    cutter = CutterDef(
        type=cutter_type,
        diameter=cutter_cfg.get('diameter', 6.0),
        length=cutter_cfg.get('length', 25.0),
        corner_radius=cutter_cfg.get('corner_radius', 0.0)
    )
    
    generator = ToolpathGenerator(cutter)
    generator.load_mesh(mesh)
    
    # Get mesh bounds
    min_bound, max_bound = mesh.get_bounds()
    
    # Add margin for cutter radius
    margin = cutter.diameter / 2 + 1.0
    x_min, y_min, z_min = min_bound - margin
    x_max, y_max, z_max = max_bound + margin
    
    operation = tp_config.get('operation', 'surface')
    all_passes = []
    
    if operation in ('surface', 'both'):
        surface_cfg = tp_config.get('surface', {})
        passes = generator.surface_dropcutter(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            stepover=surface_cfg.get('stepover', 2.0),
            direction=surface_cfg.get('direction', 'x')
        )
        all_passes.extend(passes)
        
    if operation in ('waterline', 'both'):
        wl_cfg = tp_config.get('waterline', {})
        passes = generator.waterline(
            z_min=z_min, z_max=z_max,
            z_step=wl_cfg.get('z_step', 1.0),
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max
        )
        all_passes.extend(passes)
        
    # Add lead-in/lead-out
    clearance = tp_config.get('clearance_height', 10.0)
    all_passes = generator.add_lead_in_out(all_passes, clearance)
    
    logger.info(f"Generated {len(all_passes)} toolpath passes")
    return all_passes, clearance


def generate_gcode(passes, clearance: float, config: dict) -> GcodeWriter:
    """Generate G-code from toolpath."""
    gc_config = config.get('gcode', {})
    
    writer = GcodeWriter(GcodeConfig(
        feed_rate=gc_config.get('feed_rate', 500),
        plunge_rate=gc_config.get('plunge_rate', 100),
        spindle_speed=gc_config.get('spindle_speed', 10000),
        coolant=gc_config.get('coolant', False),
        dialect=gc_config.get('dialect', 'grbl')
    ))
    
    writer.from_toolpath(passes, clearance_z=clearance)
    
    est_time = writer.estimate_time()
    logger.info(f"Generated G-code: {len(writer.lines)} lines, "
               f"estimated time: {est_time:.1f} min")
    return writer


def main():
    parser = argparse.ArgumentParser(
        description='Process point cloud to G-code'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input point cloud (NPY, PLY, PCD)')
    parser.add_argument('--output', '-o', type=str, default='data/output',
                        help='Output directory')
    parser.add_argument('--config', '-c', type=str, default='config/processing.yaml',
                        help='Processing configuration file')
    parser.add_argument('--mesh-only', action='store_true',
                        help='Only generate mesh, skip G-code')
    parser.add_argument('--visualize', action='store_true',
                        help='Show visualization windows')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    config = load_config(Path(args.config))
    
    # Process point cloud
    pointcloud = process_pointcloud(input_path, config)
    
    # Save processed point cloud
    pc_output = output_dir / 'processed_pointcloud.ply'
    pointcloud.save(str(pc_output))
    
    # Visualize point cloud
    if args.visualize:
        try:
            import open3d as o3d
            o3d.visualization.draw_geometries([pointcloud.pcd],
                                              window_name="Point Cloud")
        except Exception as e:
            logger.warning(f"Could not visualize: {e}")
    
    # Reconstruct mesh
    mesh = reconstruct_mesh(pointcloud, config)
    
    # Save mesh
    mesh_output = output_dir / 'mesh.stl'
    mesh.save_stl(str(mesh_output))
    
    # Also save as PLY for visualization
    mesh.save(str(output_dir / 'mesh.ply'))
    
    # Visualize mesh
    if args.visualize:
        try:
            import open3d as o3d
            o3d.visualization.draw_geometries([mesh.mesh],
                                              window_name="Mesh")
        except Exception as e:
            logger.warning(f"Could not visualize: {e}")
    
    if args.mesh_only:
        logger.info("Mesh-only mode, skipping G-code generation")
        return 0
        
    # Generate toolpath and G-code
    try:
        passes, clearance = generate_toolpath(mesh, config)
        writer = generate_gcode(passes, clearance, config)
        
        # Save G-code
        gcode_output = output_dir / 'toolpath.nc'
        writer.save(str(gcode_output))
        
    except ImportError:
        logger.error("OpenCAMLib not available - cannot generate toolpath")
        logger.error("Install with: pip install opencamlib")
        logger.error("Or build from source for ARM64")
        return 1
        
    logger.info(f"\nOutputs saved to {output_dir}:")
    logger.info(f"  - processed_pointcloud.ply")
    logger.info(f"  - mesh.stl")
    logger.info(f"  - mesh.ply")
    logger.info(f"  - toolpath.nc")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())