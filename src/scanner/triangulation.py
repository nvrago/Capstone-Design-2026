"""
Triangulation Module

Converts 2D laser points from camera images to 3D world coordinates
using structured light triangulation.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TriangulationConfig:
    # Geometry: both at 45° pointing inward
    baseline: float = 100.0      # mm between laser and camera
    laser_angle: float = 45.0    # degrees from vertical
    camera_angle: float = 45.0   # degrees from vertical


class Triangulator:
    """
    Converts 2D pixel coordinates to 3D world coordinates.
    
    Assumes a calibrated camera and known laser plane geometry.
    """
    
    def __init__(self, config: TriangulationConfig = None):
        self.config = config or TriangulationConfig()
        
        # Camera intrinsics (set via calibrate())
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        
        # Laser plane equation: ax + by + cz + d = 0
        self.laser_plane: Optional[np.ndarray] = None
        
        # Precomputed inverse camera matrix
        self._inv_camera_matrix: Optional[np.ndarray] = None
        
    def set_camera_calibration(self, camera_matrix: np.ndarray, 
                                dist_coeffs: np.ndarray):
        """Set camera intrinsic parameters from calibration."""
        self.camera_matrix = camera_matrix.astype(np.float64)
        self.dist_coeffs = dist_coeffs.astype(np.float64)
        self._inv_camera_matrix = np.linalg.inv(self.camera_matrix)
        logger.info("Camera calibration set")
        
    def set_laser_plane(self, plane_coeffs: np.ndarray):
        """
        Set laser plane equation coefficients.
        
        Args:
            plane_coeffs: [a, b, c, d] where ax + by + cz + d = 0
        """
        self.laser_plane = np.array(plane_coeffs, dtype=np.float64)
        # Normalize
        norm = np.linalg.norm(self.laser_plane[:3])
        self.laser_plane /= norm
        logger.info(f"Laser plane set: {self.laser_plane}")
        
    def compute_default_laser_plane(self):
        """
        Compute laser plane from geometry config.
        
        Assumes laser projects a vertical plane at the configured angle.
        """
        angle_rad = np.radians(self.config.laser_angle)
        # Plane normal points toward camera
        # For 45° laser angle, normal is [sin(45), 0, cos(45)]
        a = np.sin(angle_rad)
        b = 0.0
        c = np.cos(angle_rad)
        # d offset based on baseline
        d = -self.config.baseline * np.sin(angle_rad)
        
        self.laser_plane = np.array([a, b, c, d], dtype=np.float64)
        logger.info(f"Computed default laser plane: {self.laser_plane}")
        
    def pixel_to_3d(self, points_2d: np.ndarray, 
                    gantry_position: Tuple[float, float, float] = (0, 0, 0)
                    ) -> np.ndarray:
        """
        Convert 2D pixel coordinates to 3D world coordinates.
        
        Args:
            points_2d: Nx2 array of (u, v) pixel coordinates
            gantry_position: Current (x, y, z) position of scanner gantry
            
        Returns:
            Nx3 array of (x, y, z) world coordinates in mm
        """
        if self.camera_matrix is None:
            raise ValueError("Camera not calibrated - call set_camera_calibration()")
        if self.laser_plane is None:
            raise ValueError("Laser plane not set - call set_laser_plane()")
            
        if len(points_2d) == 0:
            return np.array([]).reshape(0, 3)
            
        # Undistort points
        points_undist = cv2.undistortPoints(
            points_2d.reshape(-1, 1, 2),
            self.camera_matrix,
            self.dist_coeffs,
            P=self.camera_matrix
        ).reshape(-1, 2)
        
        # Convert to normalized camera coordinates
        # [u, v, 1]^T -> K^-1 * [u, v, 1]^T = [x_n, y_n, 1]^T
        ones = np.ones((len(points_undist), 1))
        points_h = np.hstack([points_undist, ones])
        rays = (self._inv_camera_matrix @ points_h.T).T
        
        # Ray-plane intersection
        # Ray: P = t * ray_dir (camera at origin)
        # Plane: a*x + b*y + c*z + d = 0
        # Solve for t: t = -d / (a*rx + b*ry + c*rz)
        
        a, b, c, d = self.laser_plane
        plane_normal = self.laser_plane[:3]
        
        dots = rays @ plane_normal
        # Avoid division by zero for parallel rays
        valid = np.abs(dots) > 1e-6
        
        t = np.zeros(len(rays))
        t[valid] = -d / dots[valid]
        
        # Compute 3D points
        points_3d = rays * t[:, np.newaxis]
        
        # Mark invalid points with NaN
        points_3d[~valid] = np.nan
        points_3d[t < 0] = np.nan  # Behind camera
        
        # Transform to world coordinates (add gantry offset)
        gx, gy, gz = gantry_position
        points_3d[:, 0] += gx
        points_3d[:, 1] += gy
        points_3d[:, 2] += gz
        
        # Filter out NaN points
        valid_mask = ~np.isnan(points_3d).any(axis=1)
        return points_3d[valid_mask]
        
    def triangulate_frame(self, points_2d: np.ndarray,
                          gantry_position: Tuple[float, float, float]
                          ) -> np.ndarray:
        """
        Convenience method: triangulate a single frame's worth of points.
        
        Same as pixel_to_3d but with clearer naming for scan pipeline.
        """
        return self.pixel_to_3d(points_2d, gantry_position)
        
    @property
    def is_calibrated(self) -> bool:
        """Check if triangulator is ready to use."""
        return (self.camera_matrix is not None and 
                self.laser_plane is not None)
                
    def estimate_accuracy(self, depth: float) -> float:
        """
        Estimate depth measurement accuracy at a given depth.
        
        Returns approximate error in mm based on geometry and
        assumed 0.5 pixel detection accuracy.
        """
        if self.camera_matrix is None:
            return float('inf')
            
        fx = self.camera_matrix[0, 0]
        pixel_error = 0.5  # assumed subpixel accuracy
        
        # Error propagation through triangulation
        angle_rad = np.radians(self.config.laser_angle)
        depth_error = (depth ** 2 * pixel_error) / (fx * self.config.baseline * np.sin(angle_rad))
        
        return depth_error