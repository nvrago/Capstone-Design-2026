"""
Mesh Reconstruction Module

Converts point clouds to triangle meshes using Open3D.
"""

import numpy as np
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    logger.warning("Open3D not available")

from .pointcloud import PointCloud


class Mesh:
    """Wrapper around Open3D triangle mesh."""
    
    def __init__(self):
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D required for mesh processing")
        self.mesh = o3d.geometry.TriangleMesh()
        
    @property
    def vertices(self) -> np.ndarray:
        """Get vertices as Nx3 array."""
        return np.asarray(self.mesh.vertices)
        
    @property
    def triangles(self) -> np.ndarray:
        """Get triangle indices as Mx3 array."""
        return np.asarray(self.mesh.triangles)
        
    @property
    def vertex_count(self) -> int:
        return len(self.mesh.vertices)
        
    @property
    def triangle_count(self) -> int:
        return len(self.mesh.triangles)
        
    @classmethod
    def from_file(cls, path: str) -> 'Mesh':
        """Load mesh from file (STL, OBJ, PLY, etc.)."""
        m = cls()
        m.mesh = o3d.io.read_triangle_mesh(str(path))
        logger.info(f"Loaded mesh: {m.vertex_count} vertices, {m.triangle_count} triangles")
        return m
        
    def save(self, path: str):
        """Save mesh to file."""
        path = str(path)
        o3d.io.write_triangle_mesh(path, self.mesh)
        logger.info(f"Saved mesh to {path}")
        
    def save_stl(self, path: str):
        """Save as STL (ASCII or binary based on extension)."""
        self.save(path)
        
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding box as (min_bound, max_bound)."""
        bbox = self.mesh.get_axis_aligned_bounding_box()
        return np.asarray(bbox.min_bound), np.asarray(bbox.max_bound)
        
    def compute_normals(self):
        """Compute vertex and face normals."""
        self.mesh.compute_vertex_normals()
        self.mesh.compute_triangle_normals()
        
    def smooth_laplacian(self, iterations: int = 1):
        """Apply Laplacian smoothing."""
        self.mesh = self.mesh.filter_smooth_laplacian(
            number_of_iterations=iterations
        )
        logger.info(f"Applied {iterations} iterations of Laplacian smoothing")
        
    def smooth_taubin(self, iterations: int = 10):
        """Apply Taubin smoothing (less shrinkage than Laplacian)."""
        self.mesh = self.mesh.filter_smooth_taubin(
            number_of_iterations=iterations
        )
        logger.info(f"Applied {iterations} iterations of Taubin smoothing")
        
    def simplify(self, target_triangles: int) -> 'Mesh':
        """Simplify mesh to target triangle count."""
        simplified = self.mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles
        )
        result = Mesh()
        result.mesh = simplified
        logger.info(f"Simplified: {self.triangle_count} -> {result.triangle_count} triangles")
        return result
        
    def remove_degenerate(self):
        """Remove degenerate triangles."""
        self.mesh.remove_degenerate_triangles()
        self.mesh.remove_duplicated_triangles()
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_unreferenced_vertices()
        
    def fill_holes(self):
        """Attempt to fill holes in mesh."""
        # Open3D doesn't have direct hole filling, but we can detect holes
        edges = self.mesh.get_non_manifold_edges()
        if len(edges) > 0:
            logger.warning(f"Mesh has {len(edges)} non-manifold edges (holes)")
        return self


class MeshReconstructor:
    """Creates meshes from point clouds."""
    
    def __init__(self):
        self.last_densities = None
        
    def poisson_reconstruction(self, pointcloud: PointCloud,
                                depth: int = 9,
                                width: float = 0,
                                scale: float = 1.1,
                                linear_fit: bool = False) -> Mesh:
        """
        Poisson surface reconstruction.
        
        Requires point cloud with normals.
        
        Args:
            pointcloud: Input point cloud (must have normals)
            depth: Octree depth (higher = more detail, slower)
            width: Target width of finest octree cells
            scale: Ratio between cube diameter and bounding box
            linear_fit: Use linear interpolation for iso-surface
        """
        if not pointcloud.has_normals():
            logger.info("Computing normals for Poisson reconstruction")
            pointcloud.estimate_normals()
            
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pointcloud.pcd,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=linear_fit
        )
        
        self.last_densities = np.asarray(densities)
        
        result = Mesh()
        result.mesh = mesh
        logger.info(f"Poisson reconstruction: {result.triangle_count} triangles")
        return result
        
    def poisson_with_density_filter(self, pointcloud: PointCloud,
                                     depth: int = 9,
                                     density_threshold: float = 0.1) -> Mesh:
        """
        Poisson reconstruction with low-density vertex removal.
        
        Removes vertices in areas with sparse point coverage.
        """
        mesh = self.poisson_reconstruction(pointcloud, depth=depth)
        
        if self.last_densities is not None:
            # Remove low-density vertices
            threshold = np.quantile(self.last_densities, density_threshold)
            vertices_to_remove = self.last_densities < threshold
            mesh.mesh.remove_vertices_by_mask(vertices_to_remove)
            logger.info(f"Removed {vertices_to_remove.sum()} low-density vertices")
            
        return mesh
        
    def ball_pivoting(self, pointcloud: PointCloud,
                      radii: list = [0.5, 1.0, 2.0]) -> Mesh:
        """
        Ball pivoting surface reconstruction.
        
        Args:
            pointcloud: Input point cloud (must have normals)
            radii: List of ball radii to try (in mm)
        """
        if not pointcloud.has_normals():
            logger.info("Computing normals for ball pivoting")
            pointcloud.estimate_normals()
            
        radii_vec = o3d.utility.DoubleVector(radii)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pointcloud.pcd,
            radii_vec
        )
        
        result = Mesh()
        result.mesh = mesh
        logger.info(f"Ball pivoting: {result.triangle_count} triangles")
        return result
        
    def alpha_shape(self, pointcloud: PointCloud, alpha: float = 2.0) -> Mesh:
        """
        Alpha shape surface reconstruction.
        
        Args:
            pointcloud: Input point cloud
            alpha: Alpha value (smaller = tighter fit, may have holes)
        """
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pointcloud.pcd,
            alpha=alpha
        )
        
        result = Mesh()
        result.mesh = mesh
        logger.info(f"Alpha shape (alpha={alpha}): {result.triangle_count} triangles")
        return result
        
    def reconstruct(self, pointcloud: PointCloud, 
                    method: str = 'poisson',
                    **kwargs) -> Mesh:
        """
        Reconstruct mesh using specified method.
        
        Args:
            pointcloud: Input point cloud
            method: 'poisson', 'ball_pivoting', or 'alpha_shape'
            **kwargs: Method-specific parameters
        """
        methods = {
            'poisson': self.poisson_reconstruction,
            'ball_pivoting': self.ball_pivoting,
            'alpha_shape': self.alpha_shape
        }
        
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Use one of {list(methods.keys())}")
            
        return methods[method](pointcloud, **kwargs)