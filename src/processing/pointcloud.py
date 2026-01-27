"""
Point Cloud Processing Module

Handles point cloud operations using Open3D: filtering, downsampling,
normal estimation, and registration.
"""

import numpy as np
import logging
from typing import Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D not available - point cloud functions limited")


class PointCloud:
    """Wrapper around Open3D point cloud with convenience methods."""
    
    def __init__(self, points: np.ndarray = None):
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D required for point cloud processing")
            
        self.pcd = o3d.geometry.PointCloud()
        if points is not None:
            self.set_points(points)
            
    def set_points(self, points: np.ndarray):
        """Set point cloud from Nx3 numpy array."""
        self.pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        
    def get_points(self) -> np.ndarray:
        """Get points as Nx3 numpy array."""
        return np.asarray(self.pcd.points)
        
    def __len__(self) -> int:
        return len(self.pcd.points)
        
    @classmethod
    def from_file(cls, path: str) -> 'PointCloud':
        """Load point cloud from file (PLY, PCD, XYZ, etc.)."""
        pc = cls()
        pc.pcd = o3d.io.read_point_cloud(str(path))
        logger.info(f"Loaded {len(pc)} points from {path}")
        return pc
        
    def save(self, path: str):
        """Save point cloud to file."""
        o3d.io.write_point_cloud(str(path), self.pcd)
        logger.info(f"Saved {len(self)} points to {path}")
        
    def remove_outliers_statistical(self, nb_neighbors: int = 20, 
                                     std_ratio: float = 2.0) -> 'PointCloud':
        """
        Remove statistical outliers.
        
        Points with mean distance to neighbors greater than std_ratio * std
        are removed.
        """
        clean, idx = self.pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        result = PointCloud()
        result.pcd = clean
        removed = len(self) - len(result)
        logger.info(f"Statistical outlier removal: {removed} points removed")
        return result
        
    def remove_outliers_radius(self, radius: float = 2.0, 
                                min_neighbors: int = 5) -> 'PointCloud':
        """Remove points with fewer than min_neighbors within radius."""
        clean, idx = self.pcd.remove_radius_outlier(
            nb_points=min_neighbors,
            radius=radius
        )
        result = PointCloud()
        result.pcd = clean
        removed = len(self) - len(result)
        logger.info(f"Radius outlier removal: {removed} points removed")
        return result
        
    def downsample_voxel(self, voxel_size: float) -> 'PointCloud':
        """Downsample using voxel grid filter."""
        down = self.pcd.voxel_down_sample(voxel_size=voxel_size)
        result = PointCloud()
        result.pcd = down
        logger.info(f"Voxel downsampling: {len(self)} -> {len(result)} points")
        return result
        
    def estimate_normals(self, radius: float = 2.0, max_nn: int = 30):
        """Estimate point normals."""
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius,
                max_nn=max_nn
            )
        )
        # Orient normals consistently (toward camera/+Z)
        self.pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0, 0, 100])
        )
        logger.info("Normals estimated and oriented")
        
    def has_normals(self) -> bool:
        """Check if normals have been computed."""
        return len(self.pcd.normals) > 0
        
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box as (min_bound, max_bound)."""
        bbox = self.pcd.get_axis_aligned_bounding_box()
        return np.asarray(bbox.min_bound), np.asarray(bbox.max_bound)
        
    def crop_to_bounds(self, min_bound: np.ndarray, 
                       max_bound: np.ndarray) -> 'PointCloud':
        """Crop point cloud to bounding box."""
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound,
            max_bound=max_bound
        )
        cropped = self.pcd.crop(bbox)
        result = PointCloud()
        result.pcd = cropped
        logger.info(f"Cropped: {len(self)} -> {len(result)} points")
        return result
        
    def transform(self, matrix: np.ndarray) -> 'PointCloud':
        """Apply 4x4 transformation matrix."""
        transformed = self.pcd.transform(matrix)
        result = PointCloud()
        result.pcd = transformed
        return result
        
    def merge(self, other: 'PointCloud') -> 'PointCloud':
        """Merge with another point cloud."""
        combined = PointCloud()
        combined.pcd = self.pcd + other.pcd
        return combined
        
    def cluster_dbscan(self, eps: float = 2.0, 
                       min_points: int = 10) -> List['PointCloud']:
        """
        Cluster points using DBSCAN.
        
        Returns list of point clouds, one per cluster.
        """
        labels = np.array(self.pcd.cluster_dbscan(
            eps=eps, min_points=min_points
        ))
        
        clusters = []
        for label in set(labels):
            if label < 0:  # noise
                continue
            mask = labels == label
            cluster = PointCloud()
            cluster.pcd = self.pcd.select_by_index(np.where(mask)[0].tolist())
            clusters.append(cluster)
            
        logger.info(f"DBSCAN clustering: {len(clusters)} clusters found")
        return clusters


class PointCloudAccumulator:
    """Accumulates points from multiple scan frames."""
    
    def __init__(self):
        self.all_points: List[np.ndarray] = []
        self.frame_count = 0
        
    def add_frame(self, points: np.ndarray):
        """Add points from a single scan frame."""
        if len(points) > 0:
            self.all_points.append(points)
            self.frame_count += 1
            
    def get_combined(self) -> PointCloud:
        """Get combined point cloud from all frames."""
        if not self.all_points:
            return PointCloud(np.array([]).reshape(0, 3))
            
        combined = np.vstack(self.all_points)
        logger.info(f"Combined {self.frame_count} frames: {len(combined)} points")
        return PointCloud(combined)
        
    def clear(self):
        """Clear accumulated points."""
        self.all_points = []
        self.frame_count = 0