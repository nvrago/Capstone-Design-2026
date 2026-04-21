"""
pipeline.py -- scan-to-cnc orchestrator

stages:
    1. capture: arc sweep, D405 depth frames transformed to plate frame
    2. register: ICP merge position clouds + dome subtract rig
    3. process: downsample + outlier removal + normals
    4. mesh: poisson reconstruction
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
    """box clip in plate frame. safety net on top of dome subtraction."""
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
    # verified calibration: plate frame centroid = -0.002m at 90 deg.
    arc_radius_m: float = 0.250
    arc_center_z_m: float = 0.000

    # capture
    capture_width: int = 640
    capture_height: int = 480
    capture_fps: int = 30
    frames_per_position: int = 30
    decimation_magnitude: int = 2

    # plate-frame box clip (safety net on top of dome subtraction)
    # sized to match plate footprint and CNC work envelope (both ~30cm),
    # with 15cm max object height.
    plate_xy_extent_m: float = 0.155
    plate_z_min_m: float = -0.010
    plate_z_max_m: float = 0.150

    # dome subtraction
    dome_reference_path: str = "data/reference/dome_cloud.ply"
    dome_threshold_m: float = 0.008

    # icp registration (clouds already in plate frame, so initial is identity)
    icp_voxel_size: float = 0.005
    icp_max_distance: float = 0.020

    # processing
    voxel_size: float = 0.005
    outlier_nb_neighbors: int = 50
    outlier_std_ratio: float = 1.0
    normal_radius: float = 0.02

    # mesh
    poisson_depth: int = 7
    poisson_scale: float = 1.1

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
                "dome.reference_path": "dome_reference_path",
                "dome.threshold_m": "dome_threshold_m",
            },
            "processing": {
                "pointcloud.voxel_size": "voxel_size",
                "pointcloud.outlier_removal.nb_neighbors": "outlier_nb_neighbors",
                "pointcloud.outlier_removal.std_ratio": "outlier_std_ratio",
                "pointcloud.normal_radius": "normal_radius",
                "mesh.poisson.depth": "poisson_depth",
                "mesh.poisson.scale": "poisson_scale",
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

    # stage 2: register + dome subtract

    def stage_2_register(self):
        """icp merge position clouds, then dome subtract rig."""
        logger.info("=== stage 2: register + dome subtract ===")
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
        """poisson reconstruction."""
        logger.info("=== stage 4: mesh ===")
        start = time.time()

        if self.processed_cloud is None:
            raise RuntimeError("no processed cloud, run stage 3 first")

        reconstructor = MeshReconstructor()
        mesh = reconstructor.reconstruct(
            self.processed_cloud,
            method="poisson",
            depth=self.config.poisson_depth,
            scale=self.config.poisson_scale,
        )

        mesh.remove_degenerate()
        mesh.remove_small_components(min_ratio=0.1)
        mesh.compute_normals()

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

        min_bound, max_bound = self.mesh.get_bounds()

        passes = generator.surface_dropcutter(
            x_min=min_bound[0], x_max=max_bound[0],
            y_min=min_bound[1], y_max=max_bound[1],
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