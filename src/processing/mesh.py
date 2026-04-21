"""
Mesh Reconstruction Module

Converts point clouds to triangle meshes using Open3D.
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
    logger.warning("Open3D not available")

try:
    from scipy.spatial import Delaunay
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, extrude_to_plate will fall back to fan triangulation")

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

    # ---------------- extrude_to_plate helpers ----------------

    @staticmethod
    def _find_boundary_edges(tris: np.ndarray) -> List[Tuple[int, int]]:
        """return directed boundary edges (edges used by exactly one triangle).

        winding is preserved from the original triangle that owns each edge,
        which is what the wall-stitching code uses to choose the outward face.
        """
        edge_count = {}
        edge_dir = {}
        for tri in tris:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                key = (min(a, b), max(a, b))
                edge_count[key] = edge_count.get(key, 0) + 1
                if key not in edge_dir:
                    edge_dir[key] = (a, b)
        return [edge_dir[k] for k, c in edge_count.items() if c == 1]

    @staticmethod
    def _walk_boundary_loops(boundary_edges: List[Tuple[int, int]]) -> List[List[int]]:
        """walk the directed boundary edges into ordered vertex loops.

        returns a list of loops; each loop is an ordered list of vertex
        indices forming a closed polygon. one loop = outer boundary; more
        loops = holes or multiple disconnected pieces.

        robust to degenerate cases: branches (a vertex with >2 boundary
        edges) are handled by picking an arbitrary unused edge, and
        unreachable remainders start fresh loops.
        """
        # build a map: from-vertex -> list of (to-vertex) edges still available.
        # list because a boundary vertex might connect to multiple edges in
        # messy meshes.
        adj = {}
        for (a, b) in boundary_edges:
            adj.setdefault(a, []).append(b)

        loops = []
        while adj:
            # pick any remaining starting edge
            start = next(iter(adj))
            loop = [start]
            current = start
            while True:
                if current not in adj or not adj[current]:
                    # dead end (shouldn't happen on closed boundary,
                    # but don't crash on weird inputs)
                    break
                nxt = adj[current].pop()
                if not adj[current]:
                    del adj[current]
                if nxt == start:
                    # loop closed
                    break
                loop.append(nxt)
                current = nxt
            if len(loop) >= 3:
                loops.append(loop)
        return loops

    @staticmethod
    def _triangulate_polygon_delaunay(
        polygon_xy: np.ndarray,
    ) -> np.ndarray:
        """triangulate a 2D polygon via Delaunay, culling triangles whose
        centroid falls outside the polygon.

        polygon_xy: (N, 2) array of ordered boundary vertices.
        returns: (M, 3) array of triangle indices into polygon_xy.

        handles concave shapes correctly because Delaunay-over-the-hull
        includes triangles in concave pockets which we then remove via
        the point-in-polygon test.
        """
        if not SCIPY_AVAILABLE:
            # fall back to fan triangulation from centroid
            return Mesh._triangulate_polygon_fan(polygon_xy)

        if len(polygon_xy) < 3:
            return np.zeros((0, 3), dtype=int)

        try:
            dly = Delaunay(polygon_xy)
        except Exception as e:
            logger.warning(f"Delaunay failed ({e}), falling back to fan triangulation")
            return Mesh._triangulate_polygon_fan(polygon_xy)

        # cull triangles whose centroid is outside the polygon
        tris = dly.simplices  # (M, 3) indices into polygon_xy
        centroids = polygon_xy[tris].mean(axis=1)  # (M, 2)
        inside = Mesh._points_in_polygon(centroids, polygon_xy)
        return tris[inside]

    @staticmethod
    def _triangulate_polygon_fan(polygon_xy: np.ndarray) -> np.ndarray:
        """fan triangulation from the polygon centroid. used as a fallback
        when scipy isn't available. appends a centroid vertex and fans
        triangles out from it; caller must handle the extra vertex.

        returns triangle indices where index N (the last, appended entry)
        refers to the centroid vertex that the caller must add.
        """
        n = len(polygon_xy)
        if n < 3:
            return np.zeros((0, 3), dtype=int)
        centroid_idx = n  # caller must append this vertex
        tris = np.array([
            [centroid_idx, i, (i + 1) % n] for i in range(n)
        ], dtype=int)
        return tris

    @staticmethod
    def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """ray-cast point-in-polygon test. points and polygon are (N, 2)
        and (M, 2) respectively. returns boolean array of length N.

        standard even-odd crossing algorithm; handles concave polygons.
        """
        n = len(polygon)
        inside = np.zeros(len(points), dtype=bool)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            # edge from polygon[j] to polygon[i].
            # a horizontal ray from each point toggles inside-ness on crossing.
            cond = ((yi > points[:, 1]) != (yj > points[:, 1])) & (
                points[:, 0] < (xj - xi) * (points[:, 1] - yi) / (yj - yi + 1e-20) + xi
            )
            inside ^= cond
            j = i
        return inside

    def extrude_to_plate(self, plate_z: float = 0.0) -> 'Mesh':
        """close an open 2.5d mesh into a watertight solid by extruding its
        boundary loop(s) down to a flat plane at plate_z.

        top surface stays as the real scanned geometry; walls are vertical
        drops from every boundary vertex to plate_z; bottom is a proper
        2D Delaunay triangulation of the boundary polygon (concave-aware,
        trimmed to the polygon footprint).

        handles:
        - multiple boundary loops (outer + holes, or multiple components)
        - concave footprints (via Delaunay + point-in-polygon cull)
        - ordered boundary traversal (walls share vertices cleanly)

        returns self for chaining.
        """
        verts = np.asarray(self.mesh.vertices)
        tris = np.asarray(self.mesh.triangles)

        if len(tris) == 0:
            logger.warning("extrude_to_plate: mesh has no triangles, skipping")
            return self

        boundary_edges = self._find_boundary_edges(tris)
        if not boundary_edges:
            logger.info("extrude_to_plate: mesh is already closed, nothing to do")
            return self

        loops = self._walk_boundary_loops(boundary_edges)
        if not loops:
            logger.warning("extrude_to_plate: could not form boundary loops, skipping")
            return self

        logger.info(f"extrude_to_plate: {len(boundary_edges)} boundary edges "
                     f"-> {len(loops)} loop(s)")

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
                new_verts.append([float(v[0]), float(v[1]), float(plate_z)])
                top_to_bottom[top_idx] = len(new_verts) - 1
            return top_to_bottom[top_idx]

        # per-loop: stitch walls and build a bottom cap
        for loop_idx, loop in enumerate(loops):
            # stitch walls from ordered loop. consecutive vertices in the
            # loop are guaranteed to be connected by a boundary edge, so
            # walls share vertices with their neighbors.
            n = len(loop)
            for i in range(n):
                a = loop[i]
                b = loop[(i + 1) % n]
                a_bot = get_bottom(a)
                b_bot = get_bottom(b)
                if top_facing_up:
                    new_tris.append([a, b, b_bot])
                    new_tris.append([a, b_bot, a_bot])
                else:
                    new_tris.append([a, b_bot, b])
                    new_tris.append([a, a_bot, b_bot])

            # build the bottom cap for this loop.
            # we triangulate the loop's XY footprint (at plate_z) via Delaunay
            # and trim anything outside the polygon.
            bottom_indices = [top_to_bottom[v] for v in loop]
            polygon_xy = np.array([
                [new_verts[i][0], new_verts[i][1]]
                for i in bottom_indices
            ])

            if SCIPY_AVAILABLE:
                cap_local = self._triangulate_polygon_delaunay(polygon_xy)
                # cap_local indexes into polygon_xy (local 0..n-1);
                # remap to the global vertex list
                for (i, j, k) in cap_local:
                    v1 = bottom_indices[i]
                    v2 = bottom_indices[j]
                    v3 = bottom_indices[k]
                    if top_facing_up:
                        new_tris.append([v1, v3, v2])
                    else:
                        new_tris.append([v1, v2, v3])
            else:
                # fan fallback: append centroid and fan from it
                centroid_xy = polygon_xy.mean(axis=0)
                centroid_idx = len(new_verts)
                new_verts.append([float(centroid_xy[0]),
                                   float(centroid_xy[1]),
                                   float(plate_z)])
                for i in range(n):
                    v1 = bottom_indices[i]
                    v2 = bottom_indices[(i + 1) % n]
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
                     f"{n_added_tris} triangles "
                     f"({'Delaunay cap' if SCIPY_AVAILABLE else 'fan cap (scipy unavailable)'}). "
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