#!/usr/bin/env python3
"""
main pipeline orchestrator.
chains all 5 stages: capture -> process -> mesh -> toolpath -> gcode execution.
designed to run fully automated with no manual intervention.
"""

import argparse
import logging
import sys
import time
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scanner.capture import RealSenseCapture
from processing.pointcloud import PointCloud
from processing.mesh import MeshReconstructor
from processing.toolpath import ToolpathGenerator, CutterDef, CutterType
from gcode.writer import GcodeWriter, GcodeConfig
from cnc.grbl import GrblController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Pipeline:
    """automated scan-to-cnc pipeline."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config = self._load_configs()
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_configs(self) -> dict:
        """load all yaml config files."""
        config = {}
        for name in ["machine", "scanner", "processing"]:
            path = self.config_dir / f"{name}.yaml"
            if path.exists():
                with open(path) as f:
                    config[name] = yaml.safe_load(f)
                logger.info(f"loaded {name} config")
            else:
                logger.warning(f"config not found: {path}")
                config[name] = {}
        return config

    def stage_1_capture(self, bag_file: str = None) -> PointCloud:
        """stage 1: depth capture from realsense d405."""
        logger.info("=== stage 1: depth capture ===")
        start = time.time()

        scanner_cfg = self.config.get("scanner", {})
        streams = scanner_cfg.get("streams", {})
        depth_cfg = streams.get("depth", {})
        filters_cfg = scanner_cfg.get("filters", {})
        temporal_cfg = filters_cfg.get("temporal", {})

        scanner = RealSenseCapture(
            width=depth_cfg.get("width", 640),
            height=depth_cfg.get("height", 480),
            fps=depth_cfg.get("fps", 30),
            temporal_frames=temporal_cfg.get("frames", 15),
            decimation_magnitude=filters_cfg.get("decimation", {}).get("magnitude", 2),
            bag_file=bag_file,
        )

        output_path = str(self.data_dir / "raw_capture.ply")

        try:
            scanner.start()
            o3d_pcd = scanner.capture(output_path=output_path)
        finally:
            scanner.stop()

        # convert raw open3d point cloud to our PointCloud wrapper
        pcd = PointCloud(np.asarray(o3d_pcd.points))

        # preserve colors if available
        if o3d_pcd.has_colors():
            pcd.pcd.colors = o3d_pcd.colors

        elapsed = time.time() - start
        logger.info(f"stage 1 complete: {len(pcd)} points in {elapsed:.1f}s")
        return pcd

    def stage_2_process(self, pcd: PointCloud) -> PointCloud:
        """stage 2: point cloud processing (downsample, denoise, normals)."""
        logger.info("=== stage 2: point cloud processing ===")
        start = time.time()

        pc_cfg = self.config.get("processing", {}).get("pointcloud", {})
        outlier_cfg = pc_cfg.get("outlier_removal", {})

        voxel_size = pc_cfg.get("voxel_size")
        if voxel_size:
            pcd = pcd.downsample_voxel(voxel_size)

        pcd = pcd.remove_outliers_statistical(
            nb_neighbors=outlier_cfg.get("nb_neighbors", 20),
            std_ratio=outlier_cfg.get("std_ratio", 2.0),
        )

        normal_radius = pc_cfg.get("normal_radius", 2.0)
        pcd.estimate_normals(radius=normal_radius)

        pcd.save(str(self.data_dir / "processed.ply"))

        elapsed = time.time() - start
        logger.info(f"stage 2 complete: {len(pcd)} points in {elapsed:.1f}s")
        return pcd

    def stage_3_mesh(self, pcd: PointCloud):
        """stage 3: surface reconstruction."""
        logger.info("=== stage 3: mesh reconstruction ===")
        start = time.time()

        mesh_cfg = self.config.get("processing", {}).get("mesh", {})
        method = mesh_cfg.get("method", "poisson")

        reconstructor = MeshReconstructor()

        # build kwargs based on method
        kwargs = {}
        if method == "poisson":
            poisson_cfg = mesh_cfg.get("poisson", {})
            kwargs = {
                "depth": poisson_cfg.get("depth", 9),
                "width": poisson_cfg.get("width", 0),
                "scale": poisson_cfg.get("scale", 1.1),
                "linear_fit": poisson_cfg.get("linear_fit", False),
            }
        elif method == "ball_pivoting":
            bp_cfg = mesh_cfg.get("ball_pivoting", {})
            kwargs = {"radii": bp_cfg.get("radii", [0.5, 1.0, 2.0])}

        mesh = reconstructor.reconstruct(pcd, method=method, **kwargs)

        # post-processing
        mesh.remove_degenerate()

        smooth_iters = mesh_cfg.get("smooth_iterations", 0)
        if smooth_iters > 0:
            mesh.smooth_laplacian(iterations=smooth_iters)

        simplify_target = mesh_cfg.get("simplify_target")
        if simplify_target:
            mesh = mesh.simplify(target_triangles=simplify_target)

        mesh.compute_normals()
        mesh.save(str(self.data_dir / "mesh.stl"))

        elapsed = time.time() - start
        logger.info(f"stage 3 complete: {mesh.triangle_count} triangles in {elapsed:.1f}s")
        return mesh

    def stage_4_toolpath(self, mesh) -> str:
        """stage 4: toolpath generation via opencamlib and gcode output."""
        logger.info("=== stage 4: toolpath generation ===")
        start = time.time()

        tp_cfg = self.config.get("processing", {}).get("toolpath", {})
        gcode_cfg = self.config.get("processing", {}).get("gcode", {})
        cutter_cfg = tp_cfg.get("cutter", {})

        # build cutter definition
        cutter_type_str = cutter_cfg.get("type", "cylindrical")
        cutter_type = CutterType(cutter_type_str)

        cutter = CutterDef(
            type=cutter_type,
            diameter=cutter_cfg.get("diameter", 6.0),
            length=cutter_cfg.get("length", 25.0),
            corner_radius=cutter_cfg.get("corner_radius", 0.0),
        )

        generator = ToolpathGenerator(cutter=cutter)
        generator.load_mesh(mesh)

        # get mesh bounds for toolpath area
        min_bound, max_bound = mesh.get_bounds()
        clearance_z = tp_cfg.get("clearance_height", 10.0)

        # generate passes based on operation type
        operation = tp_cfg.get("operation", "surface")
        all_passes = []

        if operation in ("surface", "both"):
            surface_cfg = tp_cfg.get("surface", {})
            passes = generator.surface_dropcutter(
                x_min=min_bound[0],
                x_max=max_bound[0],
                y_min=min_bound[1],
                y_max=max_bound[1],
                stepover=surface_cfg.get("stepover", 2.0),
                direction=surface_cfg.get("direction", "x"),
            )
            all_passes.extend(passes)

        if operation in ("waterline", "both"):
            wl_cfg = tp_cfg.get("waterline", {})
            passes = generator.waterline(
                z_min=min_bound[2],
                z_max=max_bound[2],
                z_step=wl_cfg.get("z_step", 1.0),
                x_min=min_bound[0],
                x_max=max_bound[0],
                y_min=min_bound[1],
                y_max=max_bound[1],
            )
            all_passes.extend(passes)

        # add lead-in/out rapids and plunges
        all_passes = generator.add_lead_in_out(all_passes, clearance_z)

        # convert to gcode
        writer = GcodeWriter(GcodeConfig(
            feed_rate=gcode_cfg.get("feed_rate", 500),
            plunge_rate=gcode_cfg.get("plunge_rate", 100),
            spindle_speed=gcode_cfg.get("spindle_speed", 10000),
            coolant=gcode_cfg.get("coolant", False),
            dialect=gcode_cfg.get("dialect", "grbl"),
        ))

        writer.from_toolpath(all_passes, clearance_z=clearance_z)

        output_path = str(self.data_dir / "output.gcode")
        writer.save(output_path)

        estimated_time = writer.estimate_time()
        elapsed = time.time() - start
        logger.info(
            f"stage 4 complete: {len(writer.lines)} lines of gcode, "
            f"estimated machining time {estimated_time:.1f} min, "
            f"generated in {elapsed:.1f}s"
        )
        return output_path

    def stage_5_execute(self, gcode_path: str, dry_run: bool = False):
        """stage 5: send gcode to cnc via grbl."""
        logger.info("=== stage 5: gcode execution ===")

        if dry_run:
            logger.info("dry run mode - skipping cnc execution")
            return

        start = time.time()
        machine_cfg = self.config.get("machine", {})
        serial_cfg = machine_cfg.get("serial", {})

        cnc = GrblController(
            port=serial_cfg.get("port", "/dev/ttyUSB0"),
            baud_rate=serial_cfg.get("baud_rate", 115200),
            timeout=serial_cfg.get("timeout", 2.0),
        )

        try:
            if not cnc.connect():
                raise RuntimeError("failed to connect to cnc")

            # home if configured
            limits_cfg = machine_cfg.get("limits", {})
            if limits_cfg.get("homing_cycle", False):
                logger.info("running homing cycle...")
                cnc.home()

            # stream gcode line by line
            with open(gcode_path, "r") as f:
                lines = f.readlines()

            total_lines = len(lines)
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith(";"):
                    continue
                cnc.send(line)

                if i % 100 == 0:
                    logger.info(f"executing: line {i}/{total_lines}")

            cnc.wait_idle()
        finally:
            cnc.disconnect()

        elapsed = time.time() - start
        logger.info(f"stage 5 complete: cnc execution finished in {elapsed:.1f}s")

    def run(self, bag_file: str = None, dry_run: bool = False, skip_execute: bool = False):
        """run the full pipeline end to end."""
        logger.info("starting scan-to-cnc pipeline")
        total_start = time.time()

        pcd = self.stage_1_capture(bag_file=bag_file)
        pcd = self.stage_2_process(pcd)
        mesh = self.stage_3_mesh(pcd)
        gcode_path = self.stage_4_toolpath(mesh)

        if not skip_execute:
            self.stage_5_execute(gcode_path, dry_run=dry_run)

        total_elapsed = time.time() - total_start
        logger.info(f"pipeline complete in {total_elapsed:.1f}s")
        return gcode_path


def main():
    parser = argparse.ArgumentParser(description="scan-to-cnc automated pipeline")
    parser.add_argument("--config", "-c", type=str, default="config",
                        help="configuration directory")
    parser.add_argument("--bag", type=str,
                        help=".bag file for playback instead of live capture")
    parser.add_argument("--dry-run", action="store_true",
                        help="skip cnc execution")
    parser.add_argument("--skip-execute", action="store_true",
                        help="stop after gcode generation")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5],
                        help="run only up to this stage")
    args = parser.parse_args()

    pipeline = Pipeline(config_dir=args.config)

    if args.stage:
        pcd = None
        mesh = None
        gcode_path = None

        if args.stage >= 1:
            pcd = pipeline.stage_1_capture(bag_file=args.bag)
        if args.stage >= 2 and pcd is not None:
            pcd = pipeline.stage_2_process(pcd)
        if args.stage >= 3 and pcd is not None:
            mesh = pipeline.stage_3_mesh(pcd)
        if args.stage >= 4 and mesh is not None:
            gcode_path = pipeline.stage_4_toolpath(mesh)
        if args.stage >= 5 and gcode_path is not None:
            pipeline.stage_5_execute(gcode_path, dry_run=args.dry_run)
    else:
        pipeline.run(
            bag_file=args.bag,
            dry_run=args.dry_run,
            skip_execute=args.skip_execute,
        )


if __name__ == "__main__":
    main()