"""
pipeline.py -- scan-to-cnc orchestrator

stages:
    1. capture: arc sweep, D405 depth frames transformed to plate frame
    2. register: ICP merge position clouds + plate cut + apparatus mask
    3. process: downsample + outlier removal + normals
    4. mesh: poisson reconstruction (or ball_pivoting / alpha_shape)
    5. toolpath: opencamlib + g-code writer
    6. execute: stream g-code to grbl (optional)

each run is saved to data/runs/<timestamp>/ with intermediate plys.
previous runs are never overwritten.

usage:
    pipe = ScanPipeline(PipelineConfig())
    pipe.run()                      # full run
    pipe.run(skip_execute=True)     # stop after g-code generation
"""

import logging
import time
import yaml
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from scanner.capture import RealSenseCapture
from processing.registration import CloudRegistrator
from processing.dome_subtract import subtract_dome
from processing.cad_mask import apply_cad_mask
from processing.pointcloud import PointCloud
from processing.mesh import MeshReconstructor
from processing.toolpath import ToolpathGenerator, CutterDef, CutterType
from gcode.writer import GcodeWriter, GcodeConfig
from cnc.grbl import GrblController
from arc.controller import ArcController, MockArcController

logger = logging.getLogger(__name__)


def _plate_frame_clip(
    pcd: o3d.geometry.PointCloud,
    xy_extent_m: float,
    z_min_m: float,
    z_max_m: float,
) -> o3d.geometry.PointCloud:
    """box clip in plate frame. outermost envelope filter in stage 1."""
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    mask = (
        (np.abs(pts[:, 0]) < xy_extent_m) &
        (np.abs(pts[:, 1]) < xy_extent_m) &
        (pts[:, 2] > z_min_m) &
        (pts[:, 2] < z_max_m)
    )
    return pcd.select_by_index(np.where(mask)[0].tolist())


def _plate_surface_cut(
    pcd: o3d.geometry.PointCloud,
    z_cut_m: float,
) -> o3d.geometry.PointCloud:
    """
    drop all points at or below z_cut_m (plate surface + noise).
    captured plate lands at z ~= 0 with a few mm of noise, so a small
    positive cutoff (e.g. 0.003m) removes it cleanly while keeping
    everything on the object above.

    this is the primary background-removal filter. dome subtraction
    runs after it only to catch hemisphere / arc housing geometry
    that the camera sees at oblique angles.
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    mask = pts[:, 2] > z_cut_m
    kept = np.where(mask)[0]
    removed = len(pts) - len(kept)
    logger.info(
        f"plate surface cut at z={z_cut_m*1000:.1f}mm: "
        f"removed {removed} points ({100.0 * removed / len(pts):.1f}%), "
        f"{len(kept)} remain"
    )
    return pcd.select_by_index(kept.tolist())


def _plate_surface_cut_adaptive(
    pcd: o3d.geometry.PointCloud,
    buffer_m: float,
    bottom_fraction: float = 0.20,
) -> o3d.geometry.PointCloud:
    """
    adaptive plate removal. fits a plane to the lowest bottom_fraction
    of points (guaranteed plate since it sits below any real object),
    then drops everything within buffer_m above the plane.

    adapts to the actual plate height and tilt, so a 1.5mm buffer can
    hug the plate tightly without caring whether the plate sits at
    z=2mm or z=5mm on a given run. works at any arc angle, not just
    overhead, because the plane fit follows the plate rather than
    assuming a fixed z.
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd

    # take the bottom fraction by z; these are the plate
    n = len(pts)
    n_bottom = max(int(n * bottom_fraction), 50)
    bottom_idx = np.argpartition(pts[:, 2], n_bottom)[:n_bottom]
    bottom_pts = pts[bottom_idx]

    # least-squares plane fit: z = a*x + b*y + c
    # gives plane normal [-a, -b, 1] / sqrt(a^2 + b^2 + 1), offset c
    A = np.column_stack([bottom_pts[:, 0], bottom_pts[:, 1], np.ones(n_bottom)])
    coeffs, *_ = np.linalg.lstsq(A, bottom_pts[:, 2], rcond=None)
    a, b, c = coeffs
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    d = -c / np.sqrt(a * a + b * b + 1.0)

    # signed distance from each point to the plane, positive = above
    dist = pts @ normal + d

    mask = dist > buffer_m
    kept = np.where(mask)[0]
    removed = n - len(kept)
    logger.info(
        f"adaptive plate cut: plane tilt={np.degrees(np.arccos(normal[2])):.2f} deg, "
        f"buffer={buffer_m*1000:.1f}mm, "
        f"removed {removed} points ({100.0 * removed / n:.1f}%), "
        f"{len(kept)} remain"
    )
    return pcd.select_by_index(kept.tolist())


def _object_component_vertices(mesh) -> np.ndarray:
    """
    identify the "object" component among all mesh fragments. picks
    the component with the highest maximum z, since the object is
    always taller than plate-fragment residuals (which sit near z=0
    by construction after the plate cut).

    falls back to biggest-by-triangle-count only if something fails.
    """
    labels, counts, _ = mesh.mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    if len(counts) == 0:
        return np.asarray(mesh.mesh.vertices)

    verts = np.asarray(mesh.mesh.vertices)
    tris = np.asarray(mesh.mesh.triangles)

    # compute max z per component
    max_z_per_component = np.full(len(counts), -np.inf)
    for comp_idx in range(len(counts)):
        tri_mask = labels == comp_idx
        if not tri_mask.any():
            continue
        comp_tris = tris[tri_mask]
        comp_vert_idx = np.unique(comp_tris.ravel())
        comp_max_z = verts[comp_vert_idx, 2].max()
        max_z_per_component[comp_idx] = comp_max_z

    best = int(np.argmax(max_z_per_component))
    logger.info(
        f"object component: #{best} of {len(counts)} "
        f"(max_z={max_z_per_component[best]*1000:.1f}mm, "
        f"{counts[best]} triangles)"
    )

    tri_mask = labels == best
    comp_tris = tris[tri_mask]
    comp_vert_idx = np.unique(comp_tris.ravel())
    return verts[comp_vert_idx]

def _xy_hull_with_margin(points: np.ndarray, margin_m: float) -> np.ndarray:
    """
    compute convex hull of the points' XY projection, then inflate
    outward by margin_m. returns an Nx2 array of hull vertices in CCW
    order (the inflated polygon). safe on small/degenerate inputs.
    """
    from scipy.spatial import ConvexHull
    xy = points[:, :2]
    if len(xy) < 3:
        mn = xy.min(axis=0) - margin_m
        mx = xy.max(axis=0) + margin_m
        return np.array([
            [mn[0], mn[1]],
            [mx[0], mn[1]],
            [mx[0], mx[1]],
            [mn[0], mx[1]],
        ])
    try:
        hull = ConvexHull(xy)
    except Exception:
        # scipy complains on collinear or duplicate points; fall back to bbox
        mn = xy.min(axis=0) - margin_m
        mx = xy.max(axis=0) + margin_m
        return np.array([
            [mn[0], mn[1]],
            [mx[0], mn[1]],
            [mx[0], mx[1]],
            [mn[0], mx[1]],
        ])
    hull_pts = xy[hull.vertices]
    # inflate outward: push each hull vertex away from the centroid
    centroid = hull_pts.mean(axis=0)
    directions = hull_pts - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    inflated = hull_pts + directions / norms * margin_m
    return inflated


def _clip_cloud_to_xy_polygon(
    pcd: o3d.geometry.PointCloud,
    polygon_xy: np.ndarray,
) -> o3d.geometry.PointCloud:
    """
    drop points whose (x, y) falls outside polygon_xy. ray-casting
    point-in-polygon test. polygon is assumed closed (last vertex
    connects back to the first).
    """
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    xy = pts[:, :2]
    inside = np.zeros(len(xy), dtype=bool)
    n = len(polygon_xy)
    j = n - 1
    for i in range(n):
        xi, yi = polygon_xy[i]
        xj, yj = polygon_xy[j]
        cond = ((yi > xy[:, 1]) != (yj > xy[:, 1])) & (
            xy[:, 0] < (xj - xi) * (xy[:, 1] - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= cond
        j = i
    kept = np.where(inside)[0]
    removed = len(pts) - len(kept)
    logger.info(
        f"hull clip: removed {removed} points outside object footprint "
        f"({100.0 * removed / len(pts):.1f}%), {len(kept)} remain"
    )
    return pcd.select_by_index(kept.tolist())


@dataclass
class PipelineConfig:
    """all tunable parameters. loaded from yaml via from_yaml()."""

    # arc scan
    arc_start_deg: float = 0.0
    arc_end_deg: float = 180.0
    arc_step_deg: float = 15.0
    arc_host: str = "192.168.1.20"
    arc_modbus_port: int = 502
    arc_slave_id: int = 1
    steps_per_degree: float = 333.33

    # arc geometry (plate frame, from onshape dome)
    # origin at plate-top center, +Z up, arc sweeps XZ plane.
    # verified calibration: plate frame centroid = -0.002m at 90 deg.
    arc_radius_m: float = 0.255
    arc_center_z_m: float = 0.000

    # capture
    capture_width: int = 640
    capture_height: int = 480
    capture_fps: int = 30
    frames_per_position: int = 30
    decimation_magnitude: int = 2

    # plate-frame box clip (outermost envelope filter).
    # sized to match plate footprint and CNC work envelope (both ~30cm),
    # with 15cm max object height.
    plate_xy_extent_m: float = 0.060
    plate_z_min_m: float = 0.002
    plate_z_max_m: float = 0.060

    # plate-surface z cut (primary background removal).
    # captured plate lands at z ~= 0 with ~3mm noise, so cutting at 3mm
    # drops it cleanly while keeping object points above.
    plate_surface_z_cut_m: float = 0.003

    # adaptive plate cut (replaces the static z_cut above when enabled).
    # fits a plane to the lowest 20% of points and cuts within this
    # distance above the plane. tighter than the static cut because
    # it adapts to actual plate height/tilt rather than assuming z~=0.
    # 2.5mm preserves sub-5mm object features while hugging the plate.
    plate_surface_buffer_m: float = 0.0025
    use_adaptive_plate_cut: bool = True

    # object-footprint hull clip (stage 4 cleanup).
    # after the first mesh pass, finds the largest connected component,
    # takes its XY convex hull (with small margin), clips the processed
    # cloud to that footprint, then re-meshes. eliminates plate-fragment
    # residuals cleanly without touching the object.
    use_hull_clip: bool = True
    hull_margin_m: float = 0.003  # 3mm outward inflation, preserves edges

    # dome subtraction (fallback apparatus masking when cad mask disabled).
    # threshold is loose (8mm) so it only catches obvious hemisphere
    # geometry seen at oblique arc angles. at overhead (90 deg) this
    # removes essentially nothing, which is correct - plate cut
    # already did the work. at 0 or 180 deg, dome subtract removes
    # the arc-housing points the camera sees at grazing angles.
    dome_reference_path: str = "data/reference/dome_cloud.ply"
    dome_threshold_m: float = 0.008

    # cad-based apparatus masking (replaces dome subtract when enabled).
    # uses the solidworks assembly stl as ground truth for apparatus
    # geometry. two filters: points near apparatus surfaces are rejected
    # (plate, arc, frame, control box), and points outside the apparatus
    # bbox + margin are rejected (walls, ceiling, stray returns).
    # only points inside the envelope but off all surfaces survive - by
    # construction, that's the object. default ON; pass --no-cad-mask
    # to fall back to dome subtraction.
    use_cad_mask: bool = True
    cad_stl_path: str = "data/reference/apparatus.STL"
    cad_scale: float = 0.001
    cad_origin_offset: tuple = (-0.2773, -0.3300, -0.0945)
    cad_surface_threshold_m: float = 0.003
    cad_bounding_margin_m: float = 0.01

    # icp registration (clouds already in plate frame, so initial is identity)
    icp_voxel_size: float = 0.005
    icp_max_distance: float = 0.05

    # processing. tuned for post-subtraction clouds of a few thousand points.
    # voxel 2mm preserves mm-scale detail; nb_neighbors 20 avoids over-culling
    # small clouds; std_ratio 2.0 keeps object edges.
    voxel_size: float = 0.002
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    normal_radius: float = 0.02

    # mesh
    mesh_method: str = "poisson"
    poisson_depth: int = 7
    poisson_scale: float = 1.1
    # alpha shape reconstruction (better for open surfaces / single-angle captures).
    # smaller alpha = tighter fit. 0.01 = 10mm, tune per object scale.
    alpha_shape_alpha: float = 0.010
    # ball pivoting reconstruction (good for uniformly-dense clouds).
    # radii in meters; smallest should be ~voxel_size, largest ~3-4x.
    ball_pivoting_radii: list = field(default_factory=lambda: [0.003, 0.006, 0.012])
    # single-angle mode: extrude the heightmap mesh into a watertight
    # solid so OCL gets a closed surface for dropcutter. set True when
    # only the top is scanned (one arc angle); leave False for full sweeps.
    extrude_to_plate: bool = False

    # capture-time color filtering.
    # drop near-black pixels (e.g. matte black cloth background). per-channel:
    # a point is removed only if r, g, b are all below this value (0-255).
    # None disables the filter entirely.
    filter_black_threshold: int = None

    # toolpath
    cutter_diameter: float = 6.0
    cutter_length: float = 25.0
    stepover: float = 2.0
    surface_direction: str = "x"
    clearance_height: float = 10.0

    # gcode
    feed_rate: float = 500.0
    plunge_rate: float = 100.0
    spindle_speed: int = 10000

    # cnc / grbl
    cnc_port: str = "/dev/grbl"
    cnc_baud: int = 115200
    cnc_timeout: float = 2.0
    homing_cycle: bool = False

    # output
    data_dir: str = "data"

    # flags
    use_mock_arc: bool = False
    bag_file: str = None

    @classmethod
    def from_yaml(cls, config_dir: str = "config") -> "PipelineConfig":
        """load config from scanner.yaml + processing.yaml + machine.yaml.
        cli args override yaml values where applicable."""
        config_dir = Path(config_dir)
        overrides = {}

        yaml_map = {
            "scanner": {
                "streams.depth.width": "capture_width",
                "streams.depth.height": "capture_height",
                "streams.depth.fps": "capture_fps",
                "filters.temporal.frames": "frames_per_position",
                "filters.decimation.magnitude": "decimation_magnitude",
                "arc.radius_m": "arc_radius_m",
                "arc.center_z_m": "arc_center_z_m",
                "plate.xy_extent_m": "plate_xy_extent_m",
                "plate.z_min_m": "plate_z_min_m",
                "plate.z_max_m": "plate_z_max_m",
                "plate.surface_z_cut_m": "plate_surface_z_cut_m",
                "plate.surface_buffer_m": "plate_surface_buffer_m",
                "plate.use_adaptive_cut": "use_adaptive_plate_cut",
                "dome.reference_path": "dome_reference_path",
                "dome.threshold_m": "dome_threshold_m",
            },
            "processing": {
                "pointcloud.voxel_size": "voxel_size",
                "pointcloud.outlier_removal.nb_neighbors": "outlier_nb_neighbors",
                "pointcloud.outlier_removal.std_ratio": "outlier_std_ratio",
                "pointcloud.normal_radius": "normal_radius",
                "mesh.method": "mesh_method",
                "mesh.poisson.depth": "poisson_depth",
                "mesh.poisson.scale": "poisson_scale",
                "mesh.alpha_shape.alpha": "alpha_shape_alpha",
                "mesh.ball_pivoting.radii": "ball_pivoting_radii",
                "mesh.hull_clip": "use_hull_clip",
                "mesh.hull_margin_m": "hull_margin_m",
                "toolpath.cutter.diameter": "cutter_diameter",
                "toolpath.cutter.length": "cutter_length",
                "toolpath.surface.stepover": "stepover",
                "toolpath.surface.direction": "surface_direction",
                "toolpath.clearance_height": "clearance_height",
                "gcode.feed_rate": "feed_rate",
                "gcode.plunge_rate": "plunge_rate",
                "gcode.spindle_speed": "spindle_speed",
                "icp.voxel_size": "icp_voxel_size",
                "icp.max_distance": "icp_max_distance",
                "cad_mask.enabled": "use_cad_mask",
                "cad_mask.stl_path": "cad_stl_path",
                "cad_mask.scale": "cad_scale",
                "cad_mask.surface_threshold_m": "cad_surface_threshold_m",
                "cad_mask.bounding_margin_m": "cad_bounding_margin_m",
            },
            "machine": {
                "serial.port": "cnc_port",
                "serial.baud_rate": "cnc_baud",
                "serial.timeout": "cnc_timeout",
                "limits.homing_cycle": "homing_cycle",
                "arc.host": "arc_host",
                "arc.modbus_port": "arc_modbus_port",
                "arc.slave_id": "arc_slave_id",
                "arc.steps_per_degree": "steps_per_degree",
            },
        }

        for filename, mappings in yaml_map.items():
            path = config_dir / f"{filename}.yaml"
            if not path.exists():
                logger.warning(f"config not found: {path}")
                continue

            with open(path) as f:
                data = yaml.safe_load(f) or {}

            logger.info(f"loaded {filename} config")

            for dotpath, field_name in mappings.items():
                val = data
                for key in dotpath.split("."):
                    if isinstance(val, dict):
                        val = val.get(key)
                    else:
                        val = None
                        break
                if val is not None:
                    overrides[field_name] = val

        # cad origin offset is nested in yaml, handle separately
        proc_path = config_dir / "processing.yaml"
        if proc_path.exists():
            with open(proc_path) as f:
                proc_data = yaml.safe_load(f) or {}
            cad = proc_data.get("cad_mask", {})
            offset = cad.get("origin_offset")
            if offset is not None and all(k in offset for k in ("x", "y", "z")):
                overrides["cad_origin_offset"] = (
                    float(offset["x"]),
                    float(offset["y"]),
                    float(offset["z"]),
                )

        return cls(**overrides)


class ScanPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.arc = None
        self.scanner = None
        self.run_dir: Path = None

        # state carried between stages
        self.position_clouds: list[tuple[float, o3d.geometry.PointCloud]] = []
        self.combined_cloud: o3d.geometry.PointCloud = None
        self.processed_cloud: o3d.geometry.PointCloud = None
        self.mesh = None
        self.gcode_path: Path = None

    # helpers

    def _create_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = Path(self.config.data_dir) / "runs" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "position_clouds").mkdir(exist_ok=True)
        self.run_dir = run_dir
        logger.info(f"run output directory: {run_dir}")
        return run_dir

    def _save(self, obj, filename: str):
        if self.run_dir is None:
            self._create_run_dir()
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(obj, o3d.geometry.PointCloud):
            o3d.io.write_point_cloud(str(path), obj)
        elif isinstance(obj, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(str(path), obj)
        elif hasattr(obj, 'save'):
            obj.save(str(path))
        else:
            with open(path, "w") as f:
                f.write(str(obj))

        logger.info(f"saved {filename}")
        return path

    def _angle_to_steps(self, angle_deg: float) -> int:
        return int(round(angle_deg * self.config.steps_per_degree))

    def _clip(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        return _plate_frame_clip(
            pcd,
            xy_extent_m=self.config.plate_xy_extent_m,
            z_min_m=self.config.plate_z_min_m,
            z_max_m=self.config.plate_z_max_m,
        )

    # lifecycle

    def setup(self):
        """initialize arc controller + d405."""
        if self.config.use_mock_arc:
            self.arc = MockArcController()
        else:
            self.arc = ArcController(
                host=self.config.arc_host,
                port=self.config.arc_modbus_port,
                slave_id=self.config.arc_slave_id,
            )
        self.arc.connect()

        self.scanner = RealSenseCapture(
            width=self.config.capture_width,
            height=self.config.capture_height,
            fps=self.config.capture_fps,
            temporal_frames=self.config.frames_per_position,
            decimation_magnitude=self.config.decimation_magnitude,
            bag_file=self.config.bag_file,
            arc_radius_m=self.config.arc_radius_m,
            arc_center_z_m=self.config.arc_center_z_m,
            filter_black_threshold=self.config.filter_black_threshold,
        )
        self.scanner.start()
        logger.info("hardware initialized")

    def teardown(self):
        if self.scanner:
            try:
                self.scanner.stop()
            except Exception as e:
                logger.warning(f"scanner stop failed: {e}")
        if self.arc:
            self.arc.disconnect()
        logger.info("hardware released")

    # stage 1: capture

    def stage_1_capture(self):
        """arc sweep, capture depth at each position, transform to plate frame."""
        logger.info("=== stage 1: capture ===")
        start = time.time()

        self.arc.home()
        self.position_clouds.clear()

        positions = np.arange(
            self.config.arc_start_deg,
            self.config.arc_end_deg + self.config.arc_step_deg / 2,
            self.config.arc_step_deg,
        )

        logger.info(f"scanning {len(positions)} positions: "
                    f"{self.config.arc_start_deg} to {self.config.arc_end_deg} "
                    f"in {self.config.arc_step_deg} deg steps")

        for i, angle in enumerate(positions):
            logger.info(f"position {i+1}/{len(positions)}: {angle:.1f} deg")

            self.arc.move_to_steps(self._angle_to_steps(angle))
            time.sleep(0.3)

            pcd = self.scanner.capture(angle_deg=float(angle))

            if pcd is None or len(pcd.points) == 0:
                logger.warning(f"empty capture at {angle:.1f} deg, skipping")
                continue

            before = len(pcd.points)
            pcd = self._clip(pcd)
            logger.info(f"clip: {len(pcd.points)}/{before} points kept")

            if len(pcd.points) == 0:
                logger.warning(f"clip removed all points at {angle:.1f} deg")
                continue

            self._save(pcd, f"position_clouds/pos_{angle:.1f}.ply")
            self.position_clouds.append((float(angle), pcd))

        logger.info(f"stage 1 complete: {len(self.position_clouds)} positions "
                    f"in {time.time() - start:.1f}s")

    # stage 2: register + background removal

    def stage_2_register(self):
        """icp merge position clouds, plate surface z-cut, then apparatus mask."""
        logger.info("=== stage 2: register + background removal ===")
        start = time.time()

        if not self.position_clouds:
            raise RuntimeError("no position clouds, run stage 1 first")

        registrator = CloudRegistrator(
            voxel_size=self.config.icp_voxel_size,
            icp_max_distance=self.config.icp_max_distance,
        )
        for angle, cloud in self.position_clouds:
            registrator.add_cloud(cloud, angle)

        self.combined_cloud = registrator.register_all()
        self._save(self.combined_cloud, "raw_combined.ply")
        logger.info(f"after icp: {len(self.combined_cloud.points)} points")

        # plate surface cut: primary background removal.
        # adaptive mode fits a plane and cuts within buffer_m above it;
        # static mode cuts at a fixed z. adaptive is the default because
        # it hugs the plate tighter without risking object bases.
        if self.config.use_adaptive_plate_cut:
            self.combined_cloud = _plate_surface_cut_adaptive(
                self.combined_cloud,
                buffer_m=self.config.plate_surface_buffer_m,
            )
        else:
            self.combined_cloud = _plate_surface_cut(
                self.combined_cloud,
                z_cut_m=self.config.plate_surface_z_cut_m,
            )
        self._save(self.combined_cloud, "after_plate_cut.ply")

        # apparatus masking: either cad-based (default) or dome subtract (fallback).
        # cad mask uses the full solidworks assembly stl as ground truth - rejects
        # any point near an apparatus surface or outside the apparatus envelope.
        # dome subtract is the legacy path using a sampled dome reference ply.
        if self.config.use_cad_mask:
            self.combined_cloud = apply_cad_mask(
                self.combined_cloud,
                stl_path=Path(self.config.cad_stl_path),
                scale=self.config.cad_scale,
                origin_offset=self.config.cad_origin_offset,
                surface_threshold_m=self.config.cad_surface_threshold_m,
                bounding_margin_m=self.config.cad_bounding_margin_m,
            )
            self._save(self.combined_cloud, "after_cad_mask.ply")
            logger.info(f"after cad mask: {len(self.combined_cloud.points)} points")
        else:
            self.combined_cloud = subtract_dome(
                self.combined_cloud,
                threshold_m=self.config.dome_threshold_m,
                dome_path=Path(self.config.dome_reference_path),
            )
            self._save(self.combined_cloud, "after_dome.ply")
            logger.info(f"after dome subtract: {len(self.combined_cloud.points)} points")

        logger.info(f"stage 2 complete in {time.time() - start:.1f}s")

    # stage 3: process

    def stage_3_process(self):
        """downsample + outlier removal + normals."""
        logger.info("=== stage 3: process ===")
        start = time.time()

        if self.combined_cloud is None:
            raise RuntimeError("no combined cloud, run stage 2 first")

        if len(self.combined_cloud.points) == 0:
            raise RuntimeError(
            "combined cloud is empty. check stage 2 background removal - "
            "the plate cut and apparatus mask may be too aggressive."
        )
        pcd = PointCloud(np.asarray(self.combined_cloud.points))
        if self.combined_cloud.has_colors():
            pcd.pcd.colors = self.combined_cloud.colors

        pcd = pcd.downsample_voxel(self.config.voxel_size)
        logger.info(f"voxel downsample: {len(pcd)} points")

        pcd = pcd.remove_outliers_statistical(
            nb_neighbors=self.config.outlier_nb_neighbors,
            std_ratio=self.config.outlier_std_ratio,
        )
        logger.info(f"outlier removal: {len(pcd)} points")

        pcd.estimate_normals(radius=self.config.normal_radius)
        self._save(pcd, "processed.ply")
        self.processed_cloud = pcd

        logger.info(f"stage 3 complete: {len(pcd)} points in {time.time() - start:.1f}s")

    # stage 4: mesh

    def stage_4_mesh(self):
        """surface reconstruction. method selectable via config.mesh_method.
        with use_hull_clip, does two passes: first mesh finds the object
        footprint, second mesh runs on the XY-clipped cloud to remove
        plate-fragment residuals."""
        logger.info("=== stage 4: mesh ===")
        start = time.time()

        if self.processed_cloud is None:
            raise RuntimeError("no processed cloud, run stage 3 first")

        reconstructor = MeshReconstructor()

        method = self.config.mesh_method
        if method == "poisson":
            kwargs = {
                "depth": self.config.poisson_depth,
                "scale": self.config.poisson_scale,
            }
        elif method == "alpha_shape":
            kwargs = {"alpha": self.config.alpha_shape_alpha}
        elif method == "ball_pivoting":
            kwargs = {"radii": self.config.ball_pivoting_radii}
        else:
            raise ValueError(
                f"unknown mesh_method '{method}'. "
                f"use 'poisson', 'alpha_shape', or 'ball_pivoting'."
            )

        logger.info(f"mesh method: {method} with {kwargs}")

        # first pass: mesh the processed cloud to find the object footprint
        mesh = reconstructor.reconstruct(
            self.processed_cloud,
            method=method,
            **kwargs,
        )
        mesh.remove_degenerate()

        # split pinch-point (non-manifold) vertices so separate surface
        # patches become separate components. no-op on already-manifold
        # output (poisson). for ball_pivoting / alpha_shape, this turns
        # false single-component meshes into the real multi-component
        # topology, which lets remove_small_components actually do its job.
        mesh.split_non_manifold_vertices()

        mesh.remove_small_components()

        # hull clip: find largest component, take XY convex hull with margin,
        # drop any processed-cloud point outside that footprint, re-mesh.
        # this kills plate-fragment residuals that slipped past the adaptive
        # plate cut without risking the object itself.
        if self.config.use_hull_clip:
            logger.info("hull clip: extracting object footprint from first-pass mesh")
            obj_verts = _object_component_vertices(mesh)
            hull_poly = _xy_hull_with_margin(
                obj_verts,
                margin_m=self.config.hull_margin_m,
            )
            logger.info(
                f"hull clip: {len(hull_poly)} hull vertices, "
                f"margin={self.config.hull_margin_m*1000:.1f}mm"
            )

            clipped = _clip_cloud_to_xy_polygon(
                self.processed_cloud.pcd,
                hull_poly,
            )
            self._save(clipped, "after_hull_clip.ply")

            # rebuild a PointCloud wrapper so reconstructor can use it
            clipped_pc = PointCloud(np.asarray(clipped.points))
            if clipped.has_colors():
                clipped_pc.pcd.colors = clipped.colors
            if clipped.has_normals():
                clipped_pc.pcd.normals = clipped.normals
            else:
                clipped_pc.estimate_normals(radius=self.config.normal_radius)

            # second mesh pass on the clipped cloud
            mesh = reconstructor.reconstruct(
                clipped_pc,
                method=method,
                **kwargs,
            )
            mesh.remove_degenerate()
            mesh.split_non_manifold_vertices()
            mesh.remove_small_components()
            logger.info(f"hull clip: re-meshed, {mesh.triangle_count} triangles")

        # close single-angle heightmap into a watertight solid so OCL's
        # dropcutter gets a proper closed surface to sample. runs after
        # the cleanup above so the boundary is one clean loop.
        if self.config.extrude_to_plate:
            logger.info("extruding heightmap to plate (single-angle mode)")
            mesh.extrude_to_plate(plate_z=0.0)

        mesh.compute_normals()

        self._save(mesh, "mesh.stl")
        self._save(mesh, "mesh.ply")
        self.mesh = mesh

        logger.info(f"stage 4 complete: {mesh.triangle_count} triangles "
                    f"in {time.time() - start:.1f}s")

    def stage_5_toolpath(self):
        """opencamlib surface dropcutter + g-code."""
        logger.info("=== stage 5: toolpath ===")
        start = time.time()

        if self.mesh is None:
            raise RuntimeError("no mesh, run stage 4 first")

        cutter = CutterDef(
            type=CutterType("cylindrical"),
            diameter=self.config.cutter_diameter,
            length=self.config.cutter_length,
        )
        generator = ToolpathGenerator(cutter=cutter)

        # load_mesh scales m -> mm and shifts min corner to (0,0), top to Z=0.
        # bounds come back in mm already in the shifted coordinate space.
        bounds = generator.load_mesh(self.mesh)

        passes = generator.surface_dropcutter(
            x_min=bounds['x'][0], x_max=bounds['x'][1],
            y_min=bounds['y'][0], y_max=bounds['y'][1],
            stepover=self.config.stepover,
            direction=self.config.surface_direction,
        )
        passes = generator.add_lead_in_out(passes, self.config.clearance_height)

        writer = GcodeWriter(GcodeConfig(
            feed_rate=self.config.feed_rate,
            plunge_rate=self.config.plunge_rate,
            spindle_speed=self.config.spindle_speed,
            dialect="grbl",
        ))
        writer.from_toolpath(passes, clearance_z=self.config.clearance_height)

        self.gcode_path = self.run_dir / "toolpath.gcode"
        writer.save(str(self.gcode_path))

        logger.info(f"stage 5 complete: {len(writer.lines)} lines, "
                    f"est. {writer.estimate_time():.1f} min in {time.time() - start:.1f}s")


    # stage 6: cnc execution

    def stage_6_execute(self, dry_run: bool = False):
        """stream g-code to grbl."""
        logger.info("=== stage 6: cnc execute ===")

        if self.gcode_path is None:
            raise RuntimeError("no gcode, run stage 5 first")

        if dry_run:
            logger.info("dry run, skipping cnc")
            return

        start = time.time()
        cnc = GrblController(
            port=self.config.cnc_port,
            baud_rate=self.config.cnc_baud,
            timeout=self.config.cnc_timeout,
        )

        try:
            if not cnc.connect():
                raise RuntimeError("failed to connect to CNC")

            if self.config.homing_cycle:
                cnc.home()

            with open(self.gcode_path) as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                cnc.send(line)
                if i % 100 == 0:
                    logger.info(f"executing line {i}/{len(lines)}")

            cnc.wait_idle()
        finally:
            cnc.disconnect()

        logger.info(f"stage 6 complete in {time.time() - start:.1f}s")

    # full run

    def run(
        self,
        start_stage: int = 1,
        end_stage: int = 6,
        dry_run: bool = False,
        skip_execute: bool = False,
    ):
        if skip_execute and end_stage > 5:
            end_stage = 5

        self._create_run_dir()

        file_handler = logging.FileHandler(self.run_dir / "run.log")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logging.getLogger().addHandler(file_handler)

        stages = {
            1: ("capture", self.stage_1_capture),
            2: ("register", self.stage_2_register),
            3: ("process", self.stage_3_process),
            4: ("mesh", self.stage_4_mesh),
            5: ("toolpath", self.stage_5_toolpath),
            6: ("execute", lambda: self.stage_6_execute(dry_run=dry_run)),
        }

        logger.info(f"pipeline: stages {start_stage}-{end_stage}")
        t_start = time.time()

        try:
            if start_stage <= 1:
                self.setup()

            for stage_num in range(start_stage, end_stage + 1):
                name, func = stages[stage_num]
                t0 = time.time()
                func()
                logger.info(f"stage {stage_num} ({name}): {time.time() - t0:.1f}s")

        except KeyboardInterrupt:
            logger.warning("pipeline interrupted")
            raise
        except Exception as e:
            logger.error(f"pipeline failed: {e}", exc_info=True)
            raise
        finally:
            if start_stage <= 1:
                self.teardown()
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

        logger.info(f"pipeline complete in {time.time() - t_start:.1f}s")
        logger.info(f"outputs in {self.run_dir}")