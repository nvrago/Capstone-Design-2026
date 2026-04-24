"""
scan-to-cnc pipeline, tsdf edition.

stages:
    1. capture: arc sweep, d405 depth+color frames, stored both as
               transformed point clouds (for ui/debug) and as raw
               frames+poses (for tsdf integration).
    2. register: no-op. tsdf integrates directly from known poses, so
                 icp isn't needed. kept as a named method because
                 server.py calls it.
    3. process: tsdf integrate all frames -> extract point cloud ->
                plate-frame box clip -> largest connected cluster.
                produces both self.processed_cloud and self.mesh
                (marching cubes extracted from the tsdf volume).
    4. mesh: wrap extracted mesh for stage 5 + save mesh.stl / mesh.ply.
    5. toolpath: opencamlib surface dropcutter + g-code writer.
    6. execute: stream g-code to grbl (optional).

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
from processing.tsdf import TSDFIntegrator, TSDFConfig
from processing.clip import filter_plate_cloud, ClipConfig
from processing.mesh import Mesh
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
    arc_radius_m: float = 0.255
    arc_center_z_m: float = 0.000

    # capture
    capture_width: int = 640
    capture_height: int = 480
    capture_fps: int = 30
    frames_per_position: int = 30
    decimation_magnitude: int = 2

    # plate-frame box clip (stage 1, pre-tsdf envelope).
    # kept loose here; tight filtering happens again in stage 3.
    plate_xy_extent_m: float = 0.150
    plate_z_min_m: float = -0.010
    plate_z_max_m: float = 0.150

    # tsdf integration (stage 3 core).
    # voxel 1mm matches d405 sub-mm accuracy; sdf_trunc ~4 voxels of
    # carving distance; depth_trunc matches d405 ideal range (50cm).
    tsdf_voxel_size_m: float = 0.001
    tsdf_sdf_trunc_m: float = 0.004
    tsdf_depth_trunc_m: float = 0.5
    # depth_scale: meters-per-raw-unit divisor. d405 default is 0.0001,
    # so depth_scale = 10000. overridden at runtime by what the scanner
    # reports, but kept as a config fallback.
    tsdf_depth_scale: float = 10000.0

    # final plate-frame clip + cluster (stage 3 post-tsdf).
    # tighter than stage-1 clip; kills plate surface, arc hardware,
    # and stragglers. sized to plate footprint and ~10cm object height.
    clip_xy_extent_m: float = 0.100
    clip_z_min_m: float = 0.002
    clip_z_max_m: float = 0.100
    cluster_eps_m: float = 0.005
    cluster_min_points: int = 50

    # legacy fields kept for config compatibility. from_yaml loads these
    # if present but the tsdf pipeline ignores them. removing them would
    # break server.py / UI configs that still reference old yaml keys.
    plate_surface_z_cut_m: float = 0.003
    plate_surface_buffer_m: float = 0.0025
    use_adaptive_plate_cut: bool = True
    use_hull_clip: bool = False
    hull_margin_m: float = 0.003
    dome_reference_path: str = "data/reference/dome_cloud.ply"
    dome_threshold_m: float = 0.008
    icp_voxel_size: float = 0.005
    icp_max_distance: float = 0.05
    voxel_size: float = 0.002
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    normal_radius: float = 0.02
    mesh_method: str = "tsdf"
    poisson_depth: int = 7
    poisson_scale: float = 1.1
    alpha_shape_alpha: float = 0.010
    ball_pivoting_radii: list = field(default_factory=lambda: [0.003, 0.006, 0.012])
    extrude_to_plate: bool = True
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
        legacy keys are still read for compat; unknown keys are ignored."""
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
                # legacy scanner keys, ignored by tsdf path but still parsed
                "plate.surface_z_cut_m": "plate_surface_z_cut_m",
                "plate.surface_buffer_m": "plate_surface_buffer_m",
                "plate.use_adaptive_cut": "use_adaptive_plate_cut",
                "dome.reference_path": "dome_reference_path",
                "dome.threshold_m": "dome_threshold_m",
            },
            "processing": {
                # tsdf block (new)
                "tsdf.voxel_size_m": "tsdf_voxel_size_m",
                "tsdf.sdf_trunc_m": "tsdf_sdf_trunc_m",
                "tsdf.depth_trunc_m": "tsdf_depth_trunc_m",
                "tsdf.depth_scale": "tsdf_depth_scale",
                # clip block (new)
                "clip.xy_extent_m": "clip_xy_extent_m",
                "clip.z_min_m": "clip_z_min_m",
                "clip.z_max_m": "clip_z_max_m",
                "clip.cluster_eps_m": "cluster_eps_m",
                "clip.cluster_min_points": "cluster_min_points",
                # toolpath / gcode (unchanged)
                "toolpath.cutter.diameter": "cutter_diameter",
                "toolpath.cutter.length": "cutter_length",
                "toolpath.surface.stepover": "stepover",
                "toolpath.surface.direction": "surface_direction",
                "toolpath.clearance_height": "clearance_height",
                "gcode.feed_rate": "feed_rate",
                "gcode.plunge_rate": "plunge_rate",
                "gcode.spindle_speed": "spindle_speed",
                # legacy processing keys, ignored by tsdf path
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
                "icp.voxel_size": "icp_voxel_size",
                "icp.max_distance": "icp_max_distance",
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

        return cls(**overrides)


class ScanPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.arc = None
        self.scanner = None
        self.run_dir: Path = None

        # state carried between stages.
        # position_clouds: list[(angle_deg, o3d.PointCloud)] — transformed
        #                  point clouds, for ui display and debug saves.
        # position_frames: list[dict] — raw depth/color/pose/intrinsics,
        #                  for tsdf integration in stage 3.
        self.position_clouds: list = []
        self.position_frames: list = []
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
        elif hasattr(obj, "save"):
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
        """arc sweep. at each position, capture both a transformed point
        cloud (for debug / ui) and raw frames+pose (for tsdf integration)."""
        logger.info("=== stage 1: capture ===")
        start = time.time()

        self.arc.home()
        self.position_clouds.clear()
        self.position_frames.clear()

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

            # raw frames for tsdf. does the temporal-average capture once
            # internally, so this is the full "measurement" for the angle.
            try:
                frame = self.scanner.capture_frames(angle_deg=float(angle))
            except Exception as e:
                logger.warning(f"capture_frames failed at {angle:.1f} deg: {e}")
                continue

            self.position_frames.append({"angle_deg": float(angle), **frame})

            # transformed point cloud for debug / ui. cheap to derive
            # separately via scanner.capture() since it reuses the filter
            # chain; we could reproject from the frame dict instead but
            # keeping the existing call path preserves the ply-saving
            # behaviour server.py/UI expects.
            try:
                pcd = self.scanner.capture(angle_deg=float(angle))
            except Exception as e:
                logger.warning(f"capture (pcd) failed at {angle:.1f} deg: {e}")
                pcd = None

            if pcd is not None and len(pcd.points) > 0:
                before = len(pcd.points)
                pcd = self._clip(pcd)
                logger.info(f"clip: {len(pcd.points)}/{before} points kept")
                if len(pcd.points) > 0:
                    self._save(pcd, f"position_clouds/pos_{angle:.1f}.ply")
                    self.position_clouds.append((float(angle), pcd))

        logger.info(f"stage 1 complete: {len(self.position_frames)} frames, "
                    f"{len(self.position_clouds)} clouds in "
                    f"{time.time() - start:.1f}s")

    # stage 2: register (no-op in tsdf pipeline)

    def stage_2_register(self):
        """no-op. tsdf integrates directly from known arc poses, so icp
        registration is unnecessary. kept so server.py's UI button path
        doesn't AttributeError."""
        logger.info("=== stage 2: register ===")
        logger.info("stage 2 (register): no-op in tsdf pipeline "
                    "(poses are known from arc geometry)")

    # stage 3: tsdf integrate + clip + cluster

    def stage_3_process(self):
        """tsdf integrate all captured frames with known poses, then
        plate-frame box clip and largest-cluster filter. also extracts
        the marching-cubes mesh since the tsdf volume is right here."""
        logger.info("=== stage 3: process (tsdf) ===")
        start = time.time()

        if not self.position_frames:
            raise RuntimeError("no captured frames, run stage 1 first")

        # use the intrinsics from the first frame. all frames share the
        # same camera, so intrinsics are constant.
        intr = self.position_frames[0]["intrinsics"]

        tsdf_cfg = TSDFConfig(
            voxel_size_m=self.config.tsdf_voxel_size_m,
            sdf_trunc_m=self.config.tsdf_sdf_trunc_m,
            depth_trunc_m=self.config.tsdf_depth_trunc_m,
            depth_scale=self.config.tsdf_depth_scale,
            use_color=True,
        )
        # if scanner reported a real depth_scale, prefer that over config.
        # d405 default is 0.0001 -> divisor 10000; scanner returns the
        # raw scale (meters/unit), so flip it.
        reported = self.position_frames[0].get("depth_scale_m")
        if reported and reported > 0:
            tsdf_cfg.depth_scale = 1.0 / float(reported)
            logger.info(f"using reported depth_scale divisor: {tsdf_cfg.depth_scale}")

        integrator = TSDFIntegrator(intrinsics=intr, cfg=tsdf_cfg)

        for f in self.position_frames:
            integrator.integrate(
                depth=f["depth"],
                color=f["color"],
                plate_T_camera=f["pose"],
            )

        # fused point cloud + marching-cubes mesh straight from the volume
        fused = integrator.extract_point_cloud()
        self._save(fused, "tsdf_fused.ply")
        logger.info(f"tsdf fused cloud: {len(fused.points)} points")

        raw_mesh = integrator.extract_mesh()
        self._save(raw_mesh, "tsdf_raw.ply")

        # final filtering: box clip in plate frame + largest cluster.
        clip_cfg = ClipConfig(
            xy_extent_m=self.config.clip_xy_extent_m,
            z_min_m=self.config.clip_z_min_m,
            z_max_m=self.config.clip_z_max_m,
            cluster_eps_m=self.config.cluster_eps_m,
            cluster_min_points=self.config.cluster_min_points,
        )
        self.processed_cloud = filter_plate_cloud(fused, clip_cfg)
        self._save(self.processed_cloud, "processed.ply")

        # also keep the raw fused cloud accessible for debug
        self.combined_cloud = fused

        # stash the raw o3d mesh; stage 4 wraps it in the Mesh class
        self._raw_mesh = raw_mesh

        logger.info(f"stage 3 complete: {len(self.processed_cloud.points)} points "
                    f"in {time.time() - start:.1f}s")

    # stage 4: mesh (wrap tsdf output for stage 5)

    def stage_4_mesh(self):
        """wrap the tsdf-extracted mesh in the Mesh class that stage 5
        expects, remove degenerate triangles, save mesh.stl / mesh.ply."""
        logger.info("=== stage 4: mesh ===")
        start = time.time()

        if getattr(self, "_raw_mesh", None) is None:
            raise RuntimeError("no tsdf mesh, run stage 3 first")

        # Mesh wrapper provides .remove_degenerate / .compute_normals /
        # .get_bounds / .triangle_count / .mesh — all methods stage 5
        # and the toolpath generator use.
        # crop the raw tsdf mesh to the bounding box of the cluster-filtered
        # point cloud. this is what removes the plate and surrounding junk
        # from the mesh before it goes to opencamlib. without this, ocl
        # toolpaths the entire plate too.
        import open3d as _o3d
        _raw = self._raw_mesh
        if self.processed_cloud is not None and len(self.processed_cloud.points) > 0:
            bbox = self.processed_cloud.get_axis_aligned_bounding_box()
            # expand a few mm in each direction so the object's side walls
            # and base aren't clipped. z_min drops to the plate clip
            # (z_min_m from config) so the walls extend down to the plate.
            margin = 0.002
            min_b = bbox.min_bound.copy()
            max_b = bbox.max_bound.copy()
            min_b[0] -= margin
            min_b[1] -= margin
            min_b[2] = self.config.clip_z_min_m   # drop to plate clip floor
            max_b[0] += margin
            max_b[1] += margin
            max_b[2] += margin
            crop_bbox = _o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
            before = len(_raw.triangles)
            _raw = _raw.crop(crop_bbox)
            after = len(_raw.triangles)
            logger.info(f"stage_4: cropped mesh to object bbox "
                        f"({before} -> {after} tris, "
                        f"x=[{min_b[0]*1000:.0f},{max_b[0]*1000:.0f}]mm "
                        f"y=[{min_b[1]*1000:.0f},{max_b[1]*1000:.0f}]mm "
                        f"z=[{min_b[2]*1000:.0f},{max_b[2]*1000:.0f}]mm)")
        else:
            logger.warning("stage_4: no processed_cloud, mesh not cropped")

        mesh = Mesh()
        mesh.mesh = _raw
        mesh.remove_degenerate()

        # extra degeneracy sweep: open3d's remove_duplicated_vertices only
        # catches bitwise-identical positions. marching cubes can emit
        # vertices that differ in the last float bits but round to the same
        # position when opencamlib computes edge lengths. drop any triangle
        # with a sub-epsilon edge before handing to ocl (which asserts
        # hard and aborts the process on zero-length edges).
        import numpy as _np
        verts = _np.asarray(mesh.mesh.vertices)
        tris = _np.asarray(mesh.mesh.triangles)
        if len(tris) > 0:
            v0 = verts[tris[:, 0]]
            v1 = verts[tris[:, 1]]
            v2 = verts[tris[:, 2]]
            eps = 1e-9
            good = (
                (_np.linalg.norm(v1 - v0, axis=1) > eps) &
                (_np.linalg.norm(v2 - v1, axis=1) > eps) &
                (_np.linalg.norm(v0 - v2, axis=1) > eps)
            )
            n_bad = int((~good).sum())
            if n_bad > 0:
                logger.info(f"stage_4: dropping {n_bad} near-zero-edge tris")
                import open3d as _o3d
                mesh.mesh.triangles = _o3d.utility.Vector3iVector(tris[good])
                mesh.mesh.remove_unreferenced_vertices()

        mesh.compute_normals()

        # close the open 2.5d mesh into a watertight solid: extrude boundary
        # loops straight down to plate_z and cap with delaunay bottom face.
        # required so ocl dropcutter sees a closed surface and so the output
        # looks like a real block instead of a hollow top.
        if self.config.extrude_to_plate:
            logger.info("stage_4: extruding heightmap to plate (z=0)")
            before_tri = mesh.triangle_count
            mesh.extrude_to_plate(plate_z=0.0)
            logger.info(f"stage_4: extrude added {mesh.triangle_count - before_tri} "
                        f"triangles ({before_tri} -> {mesh.triangle_count})")

        self._save(mesh, "mesh.stl")
        self._save(mesh, "mesh.ply")
        self.mesh = mesh

        logger.info(f"stage 4 complete: {mesh.triangle_count} triangles "
                    f"in {time.time() - start:.1f}s")

    # stage 5: toolpath

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
        generator.load_mesh(self.mesh)

        # mesh is in meters; toolpath generator works in mm. scale bounds.
        min_bound, max_bound = self.mesh.get_bounds()
        min_bound_mm = min_bound * 1000.0
        max_bound_mm = max_bound * 1000.0

        passes = generator.surface_dropcutter(
            x_min=min_bound_mm[0], x_max=max_bound_mm[0],
            y_min=min_bound_mm[1], y_max=max_bound_mm[1],
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
                    f"est. {writer.estimate_time():.1f} min in "
                    f"{time.time() - start:.1f}s")

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

    # server.py compat stubs (no-ops, tsdf pipeline doesn't need them)

    def capture_zero_reference(self):
        """no-op. the tsdf box clip replaces zero-reference subtraction."""
        logger.info("capture_zero_reference: no-op in tsdf pipeline "
                    "(box clip handles background)")

    def capture_frame(self, angle_deg: float, **kwargs):
        """interactive single-frame capture (used by ui). delegates to
        stage_1_capture's per-angle logic for a single angle."""
        if self.scanner is None:
            raise RuntimeError("scanner not started; call setup() first")
        if self.arc is None:
            raise RuntimeError("arc not connected; call setup() first")

        logger.info(f"interactive capture at {angle_deg:.1f} deg")
        self.arc.move_to_steps(self._angle_to_steps(angle_deg))
        time.sleep(0.3)

        frame = self.scanner.capture_frames(angle_deg=float(angle_deg))
        self.position_frames.append({"angle_deg": float(angle_deg), **frame})

        pcd = self.scanner.capture(angle_deg=float(angle_deg))
        if pcd is not None and len(pcd.points) > 0:
            pcd = self._clip(pcd)
            if len(pcd.points) > 0:
                if self.run_dir is None:
                    self._create_run_dir()
                self._save(pcd, f"position_clouds/pos_{angle_deg:.1f}.ply")
                self.position_clouds.append((float(angle_deg), pcd))

        return {
            "angle_deg": float(angle_deg),
            "n_points": 0 if pcd is None else len(pcd.points),
        }

    def end_interactive_capture(self):
        """no-op. setup()/teardown() manage the scanner lifecycle."""
        logger.info("end_interactive_capture: no-op "
                    "(teardown handles scanner cleanup)")

    # full run

    def run(
        self,
        start_stage: int = 1,
        end_stage: int = 6,
        dry_run: bool = False,
        skip_execute: bool = False,
        run_dir: str = None,   # accepted for server.py compat, ignored
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
                logger.info(f"stage {stage_num} ({name}): "
                            f"{time.time() - t0:.1f}s")

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