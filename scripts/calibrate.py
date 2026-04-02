#!/usr/bin/env python3
"""
Basic Scan Script

Performs a raster scan pattern, capturing and triangulating laser points
at each position.
"""

import argparse
import logging
import sys
import time
import yaml
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cnc.grbl import GrblController, GrblState
from scanner.camera import Camera, CameraConfig
from scanner.laser import LaserDetector, LaserConfig
from scanner.triangulation import Triangulator, TriangulationConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_dir: Path) -> dict:
    """Load all configuration files."""
    config = {}
    for name in ['machine', 'scanner', 'processing']:
        path = config_dir / f'{name}.yaml'
        if path.exists():
            with open(path) as f:
                config[name] = yaml.safe_load(f)
    return config


def generate_raster_pattern(x_min: float, x_max: float,
                            y_min: float, y_max: float,
                            step: float) -> list:
    """Generate raster scan positions."""
    positions = []
    y = y_min
    direction = 1
    
    while y <= y_max:
        if direction == 1:
            x_start, x_end = x_min, x_max
        else:
            x_start, x_end = x_max, x_min
            
        x = x_start
        while (direction == 1 and x <= x_end) or (direction == -1 and x >= x_end):
            positions.append((x, y))
            x += step * direction
            
        y += step
        direction *= -1
        
    return positions


def main():
    parser = argparse.ArgumentParser(description='Run a 3D scan')
    parser.add_argument('--output', '-o', type=str, default='data/scan',
                        help='Output directory for scan data')
    parser.add_argument('--config', '-c', type=str, default='config',
                        help='Configuration directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate without hardware')
    parser.add_argument('--area', type=float, nargs=4,
                        metavar=('X_MIN', 'Y_MIN', 'X_MAX', 'Y_MAX'),
                        default=[0, 0, 50, 50],
                        help='Scan area in mm')
    parser.add_argument('--step', type=float, default=5.0,
                        help='Step size in mm')
    args = parser.parse_args()
    
    # Setup paths
    config_dir = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(config_dir)
    machine_cfg = config.get('machine', {})
    scanner_cfg = config.get('scanner', {})
    
    # Initialize components
    logger.info("Initializing components...")
    
    # CNC Controller
    if args.dry_run:
        cnc = None
        logger.info("Dry run - CNC disabled")
    else:
        cnc = GrblController(
            port=machine_cfg.get('serial', {}).get('port', '/dev/ttyUSB0'),
            baud_rate=machine_cfg.get('serial', {}).get('baud_rate', 115200)
        )
        if not cnc.connect():
            logger.error("Failed to connect to CNC")
            return 1
            
    # Camera
    cam_cfg = scanner_cfg.get('camera', {})
    camera = Camera(CameraConfig(
        width=cam_cfg.get('resolution', {}).get('width', 1920),
        height=cam_cfg.get('resolution', {}).get('height', 1080)
    ))
    camera.initialize()
    
    # Laser detector
    laser_cfg = scanner_cfg.get('laser', {})
    detector = LaserDetector(LaserConfig(
        hsv_lower=tuple(laser_cfg.get('hsv_lower', [0, 100, 100])),
        hsv_upper=tuple(laser_cfg.get('hsv_upper', [10, 255, 255])),
        hsv_lower_alt=tuple(laser_cfg.get('hsv_lower_alt', [170, 100, 100])),
        hsv_upper_alt=tuple(laser_cfg.get('hsv_upper_alt', [180, 255, 255]))
    ))
    
    # Triangulator
    tri_cfg = scanner_cfg.get('triangulation', {})
    triangulator = Triangulator(TriangulationConfig(
        baseline=tri_cfg.get('baseline', 100.0),
        laser_angle=tri_cfg.get('laser_angle', 45.0),
        camera_angle=tri_cfg.get('camera_angle', 45.0)
    ))
    
    # Check for calibration
    cal_cfg = scanner_cfg.get('calibration', {})
    if cal_cfg.get('camera_matrix') is not None:
        triangulator.set_camera_calibration(
            np.array(cal_cfg['camera_matrix']),
            np.array(cal_cfg.get('dist_coeffs', [0, 0, 0, 0, 0]))
        )
    else:
        # Use approximate camera matrix
        logger.warning("No camera calibration - using approximate values")
        fx = fy = 1000.0  # approximate focal length
        cx, cy = cam_cfg.get('resolution', {}).get('width', 1920) / 2, \
                 cam_cfg.get('resolution', {}).get('height', 1080) / 2
        triangulator.set_camera_calibration(
            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            np.zeros(5)
        )
        
    triangulator.compute_default_laser_plane()
    
    # Generate scan pattern
    x_min, y_min, x_max, y_max = args.area
    positions = generate_raster_pattern(x_min, x_max, y_min, y_max, args.step)
    logger.info(f"Scan pattern: {len(positions)} positions")
    
    # Collect all 3D points
    all_points = []
    scan_height = machine_cfg.get('scan_motion', {}).get('clearance', 30.0)
    feed_rate = machine_cfg.get('scan_motion', {}).get('feed_rate', 500)
    settle_time = machine_cfg.get('scan_motion', {}).get('settle_time', 0.1)
    
    try:
        # Move to safe height
        if cnc:
            cnc.move_to(z=scan_height, rapid=True)
            cnc.wait_idle()
            
        for i, (x, y) in enumerate(positions):
            logger.info(f"Position {i+1}/{len(positions)}: ({x:.1f}, {y:.1f})")
            
            # Move to position
            if cnc:
                cnc.move_to(x=x, y=y, feed=feed_rate)
                cnc.wait_idle()
                time.sleep(settle_time)
                
            # Capture image
            image = camera.capture()
            if image is None:
                logger.warning(f"Capture failed at ({x}, {y})")
                continue
                
            # Save raw image
            import cv2
            cv2.imwrite(str(output_dir / f'frame_{i:04d}.png'), image)
            
            # Detect laser points
            points_2d = detector.detect(image)
            logger.debug(f"  Detected {len(points_2d)} points")
            
            if len(points_2d) == 0:
                continue
                
            # Triangulate to 3D
            gantry_pos = (x, y, scan_height) if not args.dry_run else (x, y, scan_height)
            points_3d = triangulator.triangulate_frame(points_2d, gantry_pos)
            
            all_points.append(points_3d)
            logger.debug(f"  Triangulated {len(points_3d)} points")
            
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
    finally:
        # Cleanup
        camera.close()
        if cnc:
            cnc.move_to(z=scan_height, rapid=True)
            cnc.wait_idle()
            cnc.disconnect()
            
    # Save point cloud
    if all_points:
        combined = np.vstack(all_points)
        output_path = output_dir / 'pointcloud.npy'
        np.save(output_path, combined)
        logger.info(f"Saved {len(combined)} points to {output_path}")
        
        # Also save as PLY for viewing
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(combined)
            o3d.io.write_point_cloud(str(output_dir / 'pointcloud.ply'), pcd)
            logger.info(f"Saved PLY to {output_dir / 'pointcloud.ply'}")
        except ImportError:
            logger.warning("Open3D not available - skipping PLY export")
    else:
        logger.warning("No points captured")
        
    return 0


if __name__ == '__main__':
    sys.exit(main())