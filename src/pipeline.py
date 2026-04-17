"""
pipeline.py -- main scan-to-cnc orchestrator

runs the full automated sequence:
    1. home arc, loop through positions capturing depth frames
    2. ICP register all position clouds + subtract zero reference
    3. process combined cloud (downsample, outlier removal, normals)
    4. poisson mesh reconstruction
    5. toolpath generation + gcode output
    6. (optional) stream gcode to CNC via GRBL

every run creates a timestamped directory under data/runs/ with all
intermediate outputs preserved. previous runs and test data are
never overwritten.

data/
    reference/
        zero_cloud.ply
    test/
        (existing test data, untouched)
    runs/
        2026-04-15_143022/
            position_clouds/
                pos_-60.0.ply
                ...
            raw_combined.ply
            after_zero_sub.ply
            processed.ply
            mesh.stl
            mesh.ply
            toolpath.gcode
            run.log

usage:
    from pipeline import ScanPipeline, PipelineConfig
    config = PipelineConfig()
    pipe = ScanPipeline(config)
    pipe.run()
"""

import numpy as np
import open3d as o3d
import logging
import time
import yaml
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

from scanner.capture import RealSenseCapture
from processing.registration import CloudRegistrator
from processing.zero_mesh import apply_zero_subtraction
from processing.pointcloud import PointCloud
from processing.mesh import MeshReconstructor
from processing.toolpath import ToolpathGenerator, CutterDef, CutterType
from gcode.writer import GcodeWriter, GcodeConfig
from cnc.grbl import GrblController
from arc.controller import ArcController, MockArcController

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """all tunable parameters for the pipeline."""

    # arc scan positions
    arc_start_deg: float = -60.0
    arc_end_deg: float = 60.0
    arc_step_deg: float = 15.0
    arc_port: str = "/dev/ttyUSB0"
    arc_baud: int = 115200

    # capture
    capture_width: int = 640
    capture_height: int = 480
    capture_fps: int = 30
    frames_per_position: int = 30
    decimation_magnitude: int = 2
    depth_clip_max_m: float = 0.350

    # registration
    arc_center: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    arc_axis: list = field(default_factory=lambda: [0.0, 1.0, 0.0])
    icp_voxel_size: float = 0.002
    icp_max_distance: float = 0.005

    # zero subtraction
    zero_reference_path: str = "data/reference/zero_cloud.ply"
    zero_distance_threshold: float = 0.003

    # processing
    voxel_size: float = 0.005
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    normal_radius: float = 0.02

    # mesh
    mesh_method: str = "poisson"
    poisson_depth: int = 8
    poisson_width: int = 0
    poisson_scale: float = 1.1
    poisson_linear_fit: bool = False
    smooth_iterations: int = 0
    simplify_target: int = None

    # toolpath
    cutter_type: str = "cylindrical"
    cutter_diameter: float = 6.0
    cutter_length: float = 25.0
    cutter_corner_radius: float = 0.0
    toolpath_operation: str = "surface"
    stepover: float = 2.0
    surface_direction: str = "x"
    waterline_z_step: float = 1.0
    clearance_height: float = 10.0

    # gcode
    feed_rate: float = 500.0
    plunge_rate: float = 100.0
    spindle_speed: int = 10000
    coolant: bool = False
    gcode_dialect: str = "grbl"

    # cnc / grbl
    cnc_port: str = "/dev/ttyUSB1"
    cnc_baud: int = 115200
    cnc_timeout: float = 2.0
    homing_cycle: bool = False

    # output
    data_dir: str = "data"

    # flags
    use_mock_arc: bool = False
    skip_zero_subtraction: bool = False
    bag_file: str = None

    @classmethod
    def from_yaml(cls, config_dir: str = "config") -> "PipelineConfig":
        """load config from yaml files, falling back to defaults."""
        config_dir = Path(config_dir)
        overrides = {}

        yaml_map = {
            "scanner": {
                "streams.depth.width": "capture_width",
                "streams.depth.height": "capture_height",
                "streams.depth.fps": "capture_fps",
                "filters.temporal.frames": "frames_per_position",
                "filters.decimation.magnitude": "decimation_magnitude",
            },
            "processing": {
                "pointcloud.voxel_size": "voxel_size",
                "pointcloud.outlier_removal.nb_neighbors": "outlier_nb_neighbors",
                "pointcloud.outlier_removal.std_ratio": "outlier_std_ratio",
                "pointcloud.normal_radius": "normal_radius",
                "mesh.method": "mesh_method",
                "mesh.poisson.depth": "poisson_depth",
                "mesh.poisson.width": "poisson_width",
                "mesh.poisson.scale": "poisson_scale",
                "mesh.poisson.linear_fit": "poisson_linear_fit",
                "mesh.smooth_iterations": "smooth_iterations",
                "mesh.simplify_target": "simplify_target",
                "toolpath.cutter.type": "cutter_type",
                "toolpath.cutter.diameter": "cutter_diameter",
                "toolpath.cutter.length": "cutter_length",
                "toolpath.cutter.corner_radius": "cutter_corner_radius",
                "toolpath.operation": "toolpath_operation",
                "toolpath.surface.stepover": "stepover",
                "toolpath.surface.direction": "surface_direction",
                "toolpath.waterline.z_step": "waterline_z_step",
                "toolpath.clearance_height": "clearance_height",
                "gcode.feed_rate": "feed_rate",
                "gcode.plunge_rate": "plunge_rate",
                "gcode.spindle_speed": "spindle_speed",
                "gcode.coolant": "coolant",
                "gcode.dialect": "gcode_dialect",
            },
            "machine": {
                "serial.port": "cnc_port",
                "serial.baud_rate": "cnc_baud",
                "serial.timeout": "cnc_timeout",
                "limits.homing_cycle": "homing_cycle",
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

        # run output directory (timestamped, created at run start)
        self.run_dir: Path = None

        # pipeline state, persists between stages
        self.position_clouds: list[tuple[float, o3d.geometry.PointCloud]] = []
        self.combined_cloud: o3d.geometry.PointCloud = None
        self.processed_cloud: o3d.geometry.PointCloud = None
        self.mesh = None
        self.gcode_path: Path = None

    # run directory

    def _create_run_dir(self) -> Path:
        """create a timestamped output directory for this run."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = Path(self.config.data_dir) / "runs" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "position_clouds").mkdir(exist_ok=True)
        self.run_dir = run_dir
        logger.info(f"run output directory: {run_dir}")
        return run_dir

    def _save(self, obj, filename: str):
        """save an intermediate file to the run directory."""
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

        logger.info(f"saved {filename} to {path}")
        return path

    # lifecycle

    def setup(self):
        """initialize hardware connections."""
        if self.config.use_mock_arc:
            self.arc = MockArcController()
        else:
            self.arc = ArcController(
                port=self.config.arc_port,
                baud=self.config.arc_baud
            )
        self.arc.connect()

        self.scanner = RealSenseCapture(
            width=self.config.capture_width,
            height=self.config.capture_height,
            fps=self.config.capture_fps,
            temporal_frames=self.config.frames_per_position,
            decimation_magnitude=self.config.decimation_magnitude,
            bag_file=self.config.bag_file,
        )
        self.scanner.start()

        logger.info("hardware initialized")

    def teardown(self):
        """clean up hardware connections."""
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
        """
        capture point clouds at each arc position.
        saves each position cloud individually to position_clouds/.
        """
        logger.info("=== stage 1: multi-position depth capture ===")
        start = time.time()

        self.arc.home()
        self.position_clouds.clear()

        positions = np.arange(
            self.config.arc_start_deg,
            self.config.arc_end_deg + self.config.arc_step_deg / 2,
            self.config.arc_step_deg
        )

        logger.info(f"scanning {len(positions)} positions: "
                     f"{self.config.arc_start_deg} to {self.config.arc_end_deg} "
                     f"in {self.config.arc_step_deg} degree steps")

        for i, angle in enumerate(positions):
            logger.info(f"position {i+1}/{len(positions)}: {angle:.1f} degrees")

            self.arc.move_to(angle)
            time.sleep(0.3)

            pcd = self.scanner.capture()

            if pcd is None or len(pcd.points) == 0:
                logger.warning(f"empty capture at {angle:.1f} degrees, skipping")
                continue

            # depth clip
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.array([-10.0, -10.0, 0.0]),
                max_bound=np.array([10.0, 10.0, self.config.depth_clip_max_m])
            )
            pcd = pcd.crop(bbox)

            self._save(pcd, f"position_clouds/pos_{angle:.1f}.ply")
            self.position_clouds.append((angle, pcd))
            logger.info(f"captured {len(pcd.points)} points at {angle:.1f} degrees")

        elapsed = time.time() - start
        logger.info(f"stage 1 complete: {len(self.position_clouds)} positions "
                     f"in {elapsed:.1f}s")

    # stage 2: registration + zero subtraction

    def stage_2_register(self):
        """ICP-register all position clouds, subtract zero reference."""
        logger.info("=== stage 2: registration + background subtraction ===")
        start = time.time()

        if not self.position_clouds:
            raise RuntimeError("no position clouds, run stage 1 first")

        registrator = CloudRegistrator(
            arc_center=self.config.arc_center,
            arc_axis=self.config.arc_axis,
            voxel_size=self.config.icp_voxel_size,
            icp_max_distance=self.config.icp_max_distance
        )

        for angle, cloud in self.position_clouds:
            registrator.add_cloud(cloud, angle)

        self.combined_cloud = registrator.register_all()
        self._save(self.combined_cloud, "raw_combined.ply")
        logger.info(f"registered cloud: {len(self.combined_cloud.points)} points")

        if not self.config.skip_zero_subtraction:
            self.combined_cloud = apply_zero_subtraction(
                self.combined_cloud,
                reference_path=Path(self.config.zero_reference_path),
                distance_threshold=self.config.zero_distance_threshold
            )
            self._save(self.combined_cloud, "after_zero_sub.ply")
            logger.info(f"after zero subtraction: "
                         f"{len(self.combined_cloud.points)} points")
        else:
            logger.info("zero subtraction skipped")

        elapsed = time.time() - start
        logger.info(f"stage 2 complete in {elapsed:.1f}s")

    # stage 3: point cloud processing

    def stage_3_process(self):
        """downsample, remove outliers, estimate normals."""
        logger.info("=== stage 3: point cloud processing ===")
        start = time.time()

        if self.combined_cloud is None:
            raise RuntimeError("no combined cloud, run stage 2 first")

        # wrap in PointCloud class for method access
        pcd = PointCloud(np.asarray(self.combined_cloud.points))
        if self.combined_cloud.has_colors():
            pcd.pcd.colors = self.combined_cloud.colors

        if self.config.voxel_size:
            pcd = pcd.downsample_voxel(self.config.voxel_size)
            logger.info(f"after voxel downsample ({self.config.voxel_size}m): "
                         f"{len(pcd)} points")

        pcd = pcd.remove_outliers_statistical(
            nb_neighbors=self.config.outlier_nb_neighbors,
            std_ratio=self.config.outlier_std_ratio,
        )
        logger.info(f"after outlier removal: {len(pcd)} points")

        pcd.estimate_normals(radius=self.config.normal_radius)
        logger.info("normals estimated")

        self._save(pcd, "processed.ply")
        self.processed_cloud = pcd

        elapsed = time.time() - start
        logger.info(f"stage 3 complete: {len(pcd)} points in {elapsed:.1f}s")

    # stage 4: mesh reconstruction

    def stage_4_mesh(self):
        """surface reconstruction to watertight mesh."""
        logger.info("=== stage 4: mesh reconstruction ===")
        start = time.time()

        if self.processed_cloud is None:
            raise RuntimeError("no processed cloud, run stage 3 first")

        reconstructor = MeshReconstructor()

        kwargs = {}
        if self.config.mesh_method == "poisson":
            kwargs = {
                "depth": self.config.poisson_depth,
                "width": self.config.poisson_width,
                "scale": self.config.poisson_scale,
                "linear_fit": self.config.poisson_linear_fit,
            }
        elif self.config.mesh_method == "ball_pivoting":
            kwargs = {"radii": [0.5, 1.0, 2.0]}

        mesh = reconstructor.reconstruct(
            self.processed_cloud,
            method=self.config.mesh_method,
            **kwargs
        )

        mesh.remove_degenerate()
        mesh.remove_small_components(min_ratio=0.1)

        if self.config.smooth_iterations > 0:
            mesh.smooth_laplacian(iterations=self.config.smooth_iterations)

        if self.config.simplify_target:
            mesh = mesh.simplify(target_triangles=self.config.simplify_target)

        mesh.compute_normals()
        self._save(mesh, "mesh.stl")
        self._save(mesh, "mesh.ply")
        self.mesh = mesh

        elapsed = time.time() - start
        logger.info(f"stage 4 complete: {mesh.triangle_count} triangles "
                     f"in {elapsed:.1f}s")

    # stage 5: toolpath + gcode

    def stage_5_toolpath(self):
        """generate toolpath from mesh and write gcode."""
        logger.info("=== stage 5: toolpath + G-code generation ===")
        start = time.time()

        if self.mesh is None:
            raise RuntimeError("no mesh, run stage 4 first")

        cutter = CutterDef(
            type=CutterType(self.config.cutter_type),
            diameter=self.config.cutter_diameter,
            length=self.config.cutter_length,
            corner_radius=self.config.cutter_corner_radius,
        )

        generator = ToolpathGenerator(cutter=cutter)
        generator.load_mesh(self.mesh)

        min_bound, max_bound = self.mesh.get_bounds()

        all_passes = []
        op = self.config.toolpath_operation

        if op in ("surface", "both"):
            passes = generator.surface_dropcutter(
                x_min=min_bound[0], x_max=max_bound[0],
                y_min=min_bound[1], y_max=max_bound[1],
                stepover=self.config.stepover,
                direction=self.config.surface_direction,
            )
            all_passes.extend(passes)

        if op in ("waterline", "both"):
            passes = generator.waterline(
                z_min=min_bound[2], z_max=max_bound[2],
                z_step=self.config.waterline_z_step,
                x_min=min_bound[0], x_max=max_bound[0],
                y_min=min_bound[1], y_max=max_bound[1],
            )
            all_passes.extend(passes)

        all_passes = generator.add_lead_in_out(
            all_passes, self.config.clearance_height
        )

        writer = GcodeWriter(GcodeConfig(
            feed_rate=self.config.feed_rate,
            plunge_rate=self.config.plunge_rate,
            spindle_speed=self.config.spindle_speed,
            coolant=self.config.coolant,
            dialect=self.config.gcode_dialect,
        ))
        writer.from_toolpath(all_passes, clearance_z=self.config.clearance_height)

        gcode_path = self.run_dir / "toolpath.gcode"
        writer.save(str(gcode_path))
        self.gcode_path = gcode_path

        estimated_time = writer.estimate_time()
        elapsed = time.time() - start
        logger.info(f"stage 5 complete: {len(writer.lines)} lines, "
                     f"est. machining {estimated_time:.1f} min, "
                     f"generated in {elapsed:.1f}s")

    # stage 6: cnc execution

    def stage_6_execute(self, dry_run: bool = False):
        """stream gcode to CNC via GRBL."""
        logger.info("=== stage 6: G-code execution ===")

        if self.gcode_path is None:
            raise RuntimeError("no gcode, run stage 5 first")

        if dry_run:
            logger.info("dry run mode, skipping cnc execution")
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
                logger.info("running homing cycle...")
                cnc.home()

            with open(self.gcode_path) as f:
                lines = f.readlines()

            total = len(lines)
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                cnc.send(line)
                if i % 100 == 0:
                    logger.info(f"executing: line {i}/{total}")

            cnc.wait_idle()
        finally:
            cnc.disconnect()

        elapsed = time.time() - start
        logger.info(f"stage 6 complete: cnc execution finished in {elapsed:.1f}s")

    # full run

    def run(
        self,
        start_stage: int = 1,
        end_stage: int = 6,
        dry_run: bool = False,
        skip_execute: bool = False,
    ):
        """
        run pipeline stages sequentially.

        args:
            start_stage: first stage to run (1-6)
            end_stage: last stage to run (1-6)
            dry_run: simulate cnc execution without sending gcode
            skip_execute: stop after gcode generation (same as end_stage=5)
        """
        if skip_execute and end_stage > 5:
            end_stage = 5

        self._create_run_dir()

        # set up file logging to run directory
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

        logger.info(f"pipeline starting: stages {start_stage}-{end_stage}")
        logger.info(f"output directory: {self.run_dir}")
        t_start = time.time()

        try:
            if start_stage <= 1:
                self.setup()

            for stage_num in range(start_stage, end_stage + 1):
                name, func = stages[stage_num]
                t0 = time.time()
                func()
                dt = time.time() - t0
                logger.info(f"stage {stage_num} ({name}) completed in {dt:.1f}s")

        except KeyboardInterrupt:
            logger.warning("pipeline interrupted by user")
            raise
        except Exception as e:
            logger.error(f"pipeline failed: {e}", exc_info=True)
            raise
        finally:
            if start_stage <= 1:
                self.teardown()
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

        total = time.time() - t_start
        logger.info(f"pipeline complete in {total:.1f}s")
        logger.info(f"all outputs in {self.run_dir}")

    # zero reference capture

    def capture_zero_reference(self, n_positions: int = None):
        """capture zero reference scan (empty plate, no object)."""
        logger.info("capturing zero reference (empty plate)")

        self.setup()
        try:
            self.arc.home()

            positions = np.arange(
                self.config.arc_start_deg,
                self.config.arc_end_deg + self.config.arc_step_deg / 2,
                self.config.arc_step_deg
            )
            if n_positions:
                positions = np.linspace(
                    self.config.arc_start_deg,
                    self.config.arc_end_deg,
                    n_positions
                )

            clouds = []
            for angle in positions:
                self.arc.move_to(angle)
                time.sleep(0.3)
                pcd = self.scanner.capture()
                if pcd and len(pcd.points) > 0:
                    clouds.append(pcd)
                    logger.info(f"zero capture at {angle:.1f} degrees: "
                                 f"{len(pcd.points)} points")

            combined = o3d.geometry.PointCloud()
            for c in clouds:
                combined += c
            combined = combined.voxel_down_sample(voxel_size=0.002)

            ref_path = Path(self.config.zero_reference_path)
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(ref_path), combined)
            logger.info(f"zero reference saved: {len(combined.points)} points "
                         f"to {ref_path}")

        finally:
            self.teardown()
