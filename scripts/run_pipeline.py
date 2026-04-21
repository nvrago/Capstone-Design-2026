#!/usr/bin/env python3
"""
run_pipeline.py -- CLI entry point for the scan-to-cnc pipeline

examples:

full automated run, skip cnc execution:
    python scripts/run_pipeline.py --skip-execute

full run including cnc execution:
    python scripts/run_pipeline.py

single-angle test at 90 deg (carriage stays still, one capture,
auto-selects ball_pivoting and extrudes to a solid):
    python scripts/run_pipeline.py --single-angle 90 --skip-execute --mock

single-angle with near-black pixels filtered (matte cloth background):
    python scripts/run_pipeline.py --single-angle 90 --skip-execute --mock --filter-black 40

single-angle without heightmap extrusion (open-surface mesh, for debug):
    python scripts/run_pipeline.py --single-angle 90 --skip-execute --mock --no-extrude

run with mock arc (no ClearCore hardware):
    python scripts/run_pipeline.py --mock --skip-execute

skip capture, process from saved clouds:
    python scripts/run_pipeline.py --start-stage 2 --skip-execute

run only capture (debug arc movement):
    python scripts/run_pipeline.py --end-stage 1

run specific stages:
    python scripts/run_pipeline.py --start-stage 3 --end-stage 4

override arc positions:
    python scripts/run_pipeline.py --arc-start 0 --arc-end 180 --arc-step 15

use .bag recording instead of live camera:
    python scripts/run_pipeline.py --bag data/recordings/test.bag --mock

dry run (generate gcode but simulate cnc):
    python scripts/run_pipeline.py --dry-run

load config from yaml files:
    python scripts/run_pipeline.py --config config/

verbose logging:
    python scripts/run_pipeline.py -v
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import ScanPipeline, PipelineConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="scan-to-cnc automated pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # config
    p.add_argument("--config", "-c", type=str, default=None,
                   help="config directory with yaml files (optional, "
                        "CLI args override yaml values)")

    # stage control
    p.add_argument("--start-stage", type=int, default=1,
                   choices=[1, 2, 3, 4, 5, 6],
                   help="first pipeline stage to run (default: 1)")
    p.add_argument("--end-stage", type=int, default=6,
                   choices=[1, 2, 3, 4, 5, 6],
                   help="last pipeline stage to run (default: 6)")
    p.add_argument("--skip-execute", action="store_true",
                   help="stop after gcode generation (skip stage 6)")
    p.add_argument("--dry-run", action="store_true",
                   help="run stage 6 in simulation mode (no cnc)")

    # hardware
    p.add_argument("--mock", action="store_true",
                   help="use mock arc controller (no ClearCore hardware). "
                        "still captures live from d405 unless --bag is set.")
    p.add_argument("--arc-host", type=str, default=None,
                   help="ClearCore IP address (default: 192.168.1.20)")
    p.add_argument("--bag", type=str, default=None,
                   help=".bag file for camera playback instead of live D405")

    # arc positions
    p.add_argument("--arc-start", type=float, default=None,
                   help="arc start angle in degrees")
    p.add_argument("--arc-end", type=float, default=None,
                   help="arc end angle in degrees")
    p.add_argument("--arc-step", type=float, default=None,
                   help="arc step increment in degrees")
    p.add_argument("--single-angle", type=float, default=None,
                   help="capture a single angle only (e.g. --single-angle 90 "
                        "for overhead). overrides --arc-start/end/step. useful "
                        "for 1D testing without carriage motion. when set, "
                        "auto-selects ball_pivoting meshing and extrudes the "
                        "heightmap into a closed solid (use --no-extrude to "
                        "skip the extrusion).")

    # capture
    p.add_argument("--frames", type=int, default=None,
                   help="frames to average per position")
    p.add_argument("--filter-black", type=int, default=None, metavar="THRESHOLD",
                   help="drop near-black pixels from capture (0-255). typical "
                        "values 30-60 for matte black cloth backgrounds. omit "
                        "to disable. per-channel: a point is removed only if "
                        "r, g, b are ALL below the threshold, which protects "
                        "dark-but-not-black object regions.")

    # processing
    p.add_argument("--voxel-size", type=float, default=None,
                   help="voxel downsample size in meters")
    p.add_argument("--poisson-depth", type=int, default=None,
                   help="poisson reconstruction depth")
    p.add_argument("--mesh-method", type=str, default=None,
                   choices=["poisson", "alpha_shape", "ball_pivoting"],
                   help="mesh reconstruction method. if omitted: ball_pivoting "
                        "for --single-angle runs, poisson for full arc sweeps. "
                        "explicit value always wins. poisson builds closed "
                        "watertight meshes (best for full multi-angle scans). "
                        "alpha_shape builds open surfaces (tight-fitting, may "
                        "hole). ball_pivoting builds surface strips from "
                        "uniformly-dense clouds.")
    p.add_argument("--alpha", type=float, default=None,
                   help="alpha value for alpha_shape mesh method, meters "
                        "(default: 0.010)")
    p.add_argument("--no-extrude", action="store_true",
                   help="skip heightmap-to-solid extrusion in single-angle "
                        "mode. default: extrude whenever --single-angle is set.")
    p.add_argument("--dome-threshold", type=float, default=None,
                   help="dome subtraction threshold in meters (default: 0.003)")
    p.add_argument("--plate-z-cut", type=float, default=None,
                   help="plate surface z cut in meters (default: 0.003). "
                        "drops all points at or below this height, cleanly "
                        "removing the plate plane.")

    # output
    p.add_argument("--data-dir", type=str, default=None,
                   help="data directory (default: data)")

    # logging
    p.add_argument("-v", "--verbose", action="store_true",
                   help="enable debug logging")

    return p.parse_args()


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    # single-angle shortcut: one capture at the specified angle.
    # sets start = end = angle and step = 1 (any nonzero value works since
    # arange produces a single element when end == start).
    if args.single_angle is not None:
        config.arc_start_deg = args.single_angle
        config.arc_end_deg = args.single_angle
        config.arc_step_deg = 1.0
        logging.info(f"single-angle mode: capturing at {args.single_angle} deg")

    # mesh method auto-selection: ball_pivoting works much better than
    # poisson for single-angle (one-sided) captures, since poisson tries
    # to close the surface and hallucinates the back. explicit --mesh-method
    # always overrides this.
    if args.mesh_method is None:
        if args.single_angle is not None:
            config.mesh_method = "ball_pivoting"
            logging.info("mesh method auto-selected: ball_pivoting (single-angle)")
        # else leave config.mesh_method at its default / yaml value (poisson)

    # heightmap-to-solid extrusion: default ON for single-angle runs,
    # OFF for full sweeps. --no-extrude forces it off.
    config.extrude_to_plate = (
        args.single_angle is not None and not args.no_extrude
    )

    # cli overrides apply only if explicitly set, so yaml/defaults stay intact
    if args.arc_start is not None:
        config.arc_start_deg = args.arc_start
    if args.arc_end is not None:
        config.arc_end_deg = args.arc_end
    if args.arc_step is not None:
        config.arc_step_deg = args.arc_step
    if args.arc_host is not None:
        config.arc_host = args.arc_host
    if args.frames is not None:
        config.frames_per_position = args.frames
    if args.filter_black is not None:
        config.filter_black_threshold = args.filter_black
    if args.voxel_size is not None:
        config.voxel_size = args.voxel_size
    if args.poisson_depth is not None:
        config.poisson_depth = args.poisson_depth
    if args.mesh_method is not None:
        config.mesh_method = args.mesh_method
    if args.alpha is not None:
        config.alpha_shape_alpha = args.alpha
    if args.dome_threshold is not None:
        config.dome_threshold_m = args.dome_threshold
    if args.plate_z_cut is not None:
        config.plate_surface_z_cut_m = args.plate_z_cut
    if args.data_dir is not None:
        config.data_dir = args.data_dir

    config.use_mock_arc = args.mock
    config.bag_file = args.bag

    pipe = ScanPipeline(config)
    pipe.run(
        start_stage=args.start_stage,
        end_stage=args.end_stage,
        dry_run=args.dry_run,
        skip_execute=args.skip_execute,
    )


if __name__ == "__main__":
    main()