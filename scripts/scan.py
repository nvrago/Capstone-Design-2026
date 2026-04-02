#!/usr/bin/env python3
"""
scan script - captures a point cloud from the realsense d405.
supports live capture, .bag playback, and .bag recording.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scanner.capture import RealSenseCapture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_scanner_config(config_dir: Path) -> dict:
    """load scanner.yaml configuration."""
    path = config_dir / "scanner.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    logger.warning(f"no config found at {path}, using defaults")
    return {}


def main():
    parser = argparse.ArgumentParser(description="run a 3d scan with realsense d405")
    parser.add_argument("-o", "--output", type=str, default="data/scan/pointcloud.ply",
                        help="output ply path")
    parser.add_argument("--config", "-c", type=str, default="config",
                        help="configuration directory")
    parser.add_argument("--bag", type=str, help="path to .bag file for playback")
    parser.add_argument("--record", type=str, help="record live frames to .bag file")
    parser.add_argument("--duration", type=float, default=None,
                        help="recording duration in seconds (overrides config)")
    parser.add_argument("--frames", type=int, default=None,
                        help="temporal averaging frame count (overrides config)")
    args = parser.parse_args()

    config_dir = Path(args.config)
    cfg = load_scanner_config(config_dir)

    # pull values from config, allow cli overrides
    streams = cfg.get("streams", {})
    depth_cfg = streams.get("depth", {})
    filters_cfg = cfg.get("filters", {})
    temporal_cfg = filters_cfg.get("temporal", {})
    recording_cfg = cfg.get("recording", {})

    temporal_frames = args.frames or temporal_cfg.get("frames", 15)
    record_duration = args.duration or recording_cfg.get("duration", 5.0)

    # ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scanner = RealSenseCapture(
        width=depth_cfg.get("width", 640),
        height=depth_cfg.get("height", 480),
        fps=depth_cfg.get("fps", 30),
        temporal_frames=temporal_frames,
        decimation_magnitude=filters_cfg.get("decimation", {}).get("magnitude", 2),
        bag_file=args.bag,
    )

    # record mode
    if args.record:
        bag_dir = Path(recording_cfg.get("output_dir", "data/bags"))
        bag_dir.mkdir(parents=True, exist_ok=True)
        bag_path = bag_dir / args.record if "/" not in args.record else Path(args.record)
        scanner.record_bag(str(bag_path), duration_sec=record_duration)
        return 0

    # capture mode
    try:
        scanner.start()
        pcd = scanner.capture(output_path=str(output_path))
        logger.info(f"scan complete: {len(pcd.points)} points -> {output_path}")
    finally:
        scanner.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())