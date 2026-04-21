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

    def remove_small_components(self, min_ratio: float = 0.1):
        """remove disconnected mesh fragments smaller than min_ratio of the largest component."""
        triangle_clusters, cluster_n_triangles, _ = (
            self.mesh.cluster_connected_triangles()
        )
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        largest = cluster_n_triangles.max()
        threshold = int(largest * min_ratio)
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < threshold
        self.mesh.remove_triangles_by_mask(triangles_to_remove)
        self.mesh.remove_unreferenced_vertices()
        n_removed = triangles_to_remove.sum()
        logger.info(f"removed {n_removed} triangles from small components "
                     f"(threshold: {threshold} of {largest})")

    def extrude_to_plate(self, plate_z: float = 0.0) -> 'Mesh':
        """close an open 2.5d mesh into a watertight solid by extruding its
        boundary loop down to a flat plane at plate_z.

        the top surface stays as the real scanned geometry; the walls are
        vertical drops from every boundary vertex to plate_z; the bottom is
        a flat cap. result is a watertight mesh suitable for OCL dropcutter.

        intended for single-angle captures where only the top is scanned.
        call remove_small_components() first so the mesh is one connected
        piece — the fan-triangulation cap assumes a single boundary loop.

        returns self for chaining.
        """
        verts = np.asarray(self.mesh.vertices)
        tris = np.asarray(self.mesh.triangles)

        if len(tris) == 0:
            logger.warning("extrude_to_plate: mesh has no triangles, skipping")
            return self

        # find boundary edges: edges used by exactly one triangle.
        # canonical (min, max) ordering for the count key; remember original
        # directed (a, b) winding so wall quads stitch the right way.
        edge_count = {}
        edge_dir = {}
        for tri in tris:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                key = (min(a, b), max(a, b))
                edge_count[key] = edge_count.get(key, 0) + 1
                if key not in edge_dir:
                    edge_dir[key] = (a, b)

        boundary_edges = [edge_dir[k] for k, c in edge_count.items() if c == 1]

        if not boundary_edges:
            logger.info("extrude_to_plate: mesh is already closed, nothing to do")
            return self

        logger.info(f"extrude_to_plate: found {len(boundary_edges)} boundary edges")

        # check which way the top mesh faces so walls and cap wind correctly
        self.mesh.compute_triangle_normals()
        tri_normals = np.asarray(self.mesh.triangle_normals)
        top_facing_up = tri_normals[:, 2].mean() > 0

        new_verts = list(verts)
        new_tris = list(tris)
        top_to_bottom = {}

        def get_bottom(top_idx: int) -> int:
            if top_idx not in top_to_bottom:
                v = verts[top_idx]
                new_verts.append([v[0], v[1], plate_z])
                top_to_bottom[top_idx] = len(new_verts) - 1
            return top_to_bottom[top_idx]

        # stitch vertical walls from each boundary edge down to plate_z
        for (a, b) in boundary_edges:
            a_bot = get_bottom(a)
            b_bot = get_bottom(b)
            if top_facing_up:
                new_tris.append([a, b, b_bot])
                new_tris.append([a, b_bot, a_bot])
            else:
                new_tris.append([a, b_bot, b])
                new_tris.append([a, a_bot, b_bot])

        # build a flat bottom cap via fan triangulation from the centroid.
        # valid for roughly convex projected outlines; degenerate for
        # severely concave shapes (narrow waists, C-shapes, rings).
        bottom_indices = list(top_to_bottom.values())
        if len(bottom_indices) >= 3:
            bottom_pts = np.array([new_verts[i] for i in bottom_indices])
            centroid_xy = bottom_pts[:, :2].mean(axis=0)
            centroid_idx = len(new_verts)
            new_verts.append([centroid_xy[0], centroid_xy[1], plate_z])

            deltas = bottom_pts[:, :2] - centroid_xy
            angles = np.arctan2(deltas[:, 1], deltas[:, 0])
            order = np.argsort(angles)
            ordered = [bottom_indices[i] for i in order]

            for i in range(len(ordered)):
                v1 = ordered[i]
                v2 = ordered[(i + 1) % len(ordered)]
                if top_facing_up:
                    new_tris.append([centroid_idx, v2, v1])
                else:
                    new_tris.append([centroid_idx, v1, v2])

        # write back and clean up
        self.mesh.vertices = o3d.utility.Vector3dVector(np.array(new_verts))
        self.mesh.triangles = o3d.utility.Vector3iVector(np.array(new_tris))
        self.mesh.remove_duplicated_vertices()
        self.mesh.remove_duplicated_triangles()
        self.mesh.remove_degenerate_triangles()
        self.compute_normals()

        n_added_tris = len(new_tris) - len(tris)
        n_added_verts = len(new_verts) - len(verts)
        logger.info(f"extrude_to_plate: added {n_added_verts} bottom verts + "
                     f"{n_added_tris} triangles. "
                     f"final: {self.vertex_count} verts, {self.triangle_count} tris")

        return self

    def fill_holes(self):
        """Attempt to fill holes in mesh."""
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

        # trim low-density vertices (poisson artifacts at edges)
        threshold = np.quantile(self.last_densities, 0.20)
        vertices_to_remove = self.last_densities < threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        logger.info(f"density trim: removed {vertices_to_remove.sum()} low-confidence vertices")

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