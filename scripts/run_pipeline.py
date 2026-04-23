#!/usr/bin/env python3
"""
run_pipeline.py -- CLI entry point for the scan-to-cnc pipeline

examples:

full automated run (config auto-loaded from ./config/), skip cnc execution:
    python scripts/run_pipeline.py --skip-execute

full run including cnc execution:
    python scripts/run_pipeline.py

single-angle test at 90 deg (carriage stays still, one capture):
    python scripts/run_pipeline.py --single-angle 90 --skip-execute --mock

single-angle with near-black pixels filtered (matte cloth background):
    python scripts/run_pipeline.py --single-angle 90 --skip-execute --mock --filter-black 40

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

load config from a different directory:
    python scripts/run_pipeline.py --config other_config/

disable config loading entirely (use dataclass defaults):
    python scripts/run_pipeline.py --no-config

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

    # config. default loads from ./config/. pass --config <dir> to override,
    # or --no-config to skip yaml entirely and use dataclass defaults.
    p.add_argument("--config", "-c", type=str, default="config",
                   help="config directory with yaml files "
                        "(default: ./config/; CLI args override yaml)")
    p.add_argument("--no-config", action="store_true",
                   help="skip yaml loading entirely, use dataclass defaults")

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
                        "for 1D testing without carriage motion.")

    # capture
    p.add_argument("--frames", type=int, default=None,
                   help="frames to average per position")
    p.add_argument("--filter-black", type=int, default=None, metavar="THRESHOLD",
                   help="drop near-black pixels from capture (0-255). typical "
                        "values 30-60 for matte black cloth backgrounds. omit "
                        "to disable. per-channel: a point is removed only if "
                        "r, g, b are ALL below the threshold.")

    # legacy flags. kept for backwards compat with team scripts but
    # flagged as no-ops in the tsdf pipeline. still applied to the
    # (now-ignored) config fields so nothing crashes.
    p.add_argument("--voxel-size", type=float, default=None,
                   help="[legacy] pre-tsdf voxel downsample. no-op.")
    p.add_argument("--poisson-depth", type=int, default=None,
                   help="[legacy] poisson depth. no-op; tsdf uses marching cubes.")
    p.add_argument("--mesh-method", type=str, default=None,
                   choices=["poisson", "alpha_shape", "ball_pivoting", "tsdf"],
                   help="[legacy] mesh method. no-op; tsdf always uses "
                        "marching cubes from the fused volume.")
    p.add_argument("--alpha", type=float, default=None,
                   help="[legacy] alpha_shape alpha. no-op.")
    p.add_argument("--no-extrude", action="store_true",
                   help="[legacy] disable heightmap extrusion. no-op; "
                        "tsdf produces its own mesh topology.")
    p.add_argument("--dome-threshold", type=float, default=None,
                   help="[legacy] dome subtract threshold. no-op; "
                        "tsdf pipeline uses a plate-frame box clip instead.")
    p.add_argument("--plate-z-cut", type=float, default=None,
                   help="[legacy] static plate-surface z cut. no-op; "
                        "see clip.z_min_m in processing.yaml instead.")

    # tsdf-specific tunables (new)
    p.add_argument("--tsdf-voxel", type=float, default=None,
                   help="tsdf voxel size in meters (default: 0.001)")
    p.add_argument("--tsdf-sdf-trunc", type=float, default=None,
                   help="tsdf sdf truncation in meters (default: 0.004)")
    p.add_argument("--tsdf-depth-trunc", type=float, default=None,
                   help="tsdf depth truncation in meters (default: 0.5)")
    p.add_argument("--clip-z-min", type=float, default=None,
                   help="plate-frame z_min clip in meters (default: 0.002). "
                        "drops plate surface + noise.")
    p.add_argument("--clip-z-max", type=float, default=None,
                   help="plate-frame z_max clip in meters (default: 0.100). "
                        "max object height.")

    # output
    p.add_argument("--data-dir", type=str, default=None,
                   help="data directory (default: data)")

    # logging
    p.add_argument("-v", "--verbose", action="store_true",
                   help="enable debug logging")

    return p.parse_args()


def _warn_legacy(log, flag_name: str, value) -> None:
    log.warning(
        f"[legacy flag] {flag_name}={value} is a no-op in the tsdf pipeline"
    )


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("run_pipeline")

    # config auto-load from ./config/ by default. --no-config skips it.
    if args.no_config:
        log.info("skipping yaml, using PipelineConfig dataclass defaults")
        config = PipelineConfig()
    else:
        config_dir = args.config
        log.info(f"loading config from {config_dir}/")
        config = PipelineConfig.from_yaml(config_dir)

    # single-angle shortcut: one capture at the specified angle.
    if args.single_angle is not None:
        config.arc_start_deg = args.single_angle
        config.arc_end_deg = args.single_angle
        config.arc_step_deg = 1.0
        log.info(f"single-angle mode: capturing at {args.single_angle} deg")

    # cli overrides — only applied if explicitly set.
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
    if args.data_dir is not None:
        config.data_dir = args.data_dir

    # tsdf tunables
    if args.tsdf_voxel is not None:
        config.tsdf_voxel_size_m = args.tsdf_voxel
    if args.tsdf_sdf_trunc is not None:
        config.tsdf_sdf_trunc_m = args.tsdf_sdf_trunc
    if args.tsdf_depth_trunc is not None:
        config.tsdf_depth_trunc_m = args.tsdf_depth_trunc
    if args.clip_z_min is not None:
        config.clip_z_min_m = args.clip_z_min
    if args.clip_z_max is not None:
        config.clip_z_max_m = args.clip_z_max

    # legacy flags: still write through to the (ignored) config fields so
    # nothing ever breaks, but warn loudly so the user knows they're no-ops.
    if args.voxel_size is not None:
        _warn_legacy(log, "--voxel-size", args.voxel_size)
        config.voxel_size = args.voxel_size
    if args.poisson_depth is not None:
        _warn_legacy(log, "--poisson-depth", args.poisson_depth)
        config.poisson_depth = args.poisson_depth
    if args.mesh_method is not None:
        _warn_legacy(log, "--mesh-method", args.mesh_method)
        config.mesh_method = args.mesh_method
    if args.alpha is not None:
        _warn_legacy(log, "--alpha", args.alpha)
        config.alpha_shape_alpha = args.alpha
    if args.no_extrude:
        _warn_legacy(log, "--no-extrude", True)
        config.extrude_to_plate = False
    if args.dome_threshold is not None:
        _warn_legacy(log, "--dome-threshold", args.dome_threshold)
        config.dome_threshold_m = args.dome_threshold
    if args.plate_z_cut is not None:
        _warn_legacy(log, "--plate-z-cut", args.plate_z_cut)
        config.plate_surface_z_cut_m = args.plate_z_cut

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