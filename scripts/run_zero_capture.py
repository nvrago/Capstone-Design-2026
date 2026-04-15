#!/usr/bin/env python3
"""
run_zero_capture.py -- capture zero reference (empty plate)

run this once before scanning any objects. captures the empty
plate and arc apparatus across all positions so background
points can be subtracted from real scans.

usage:
    python scripts/run_zero_capture.py
    python scripts/run_zero_capture.py --mock
    python scripts/run_zero_capture.py --positions 12
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import ScanPipeline, PipelineConfig


def main():
    p = argparse.ArgumentParser(description="capture zero reference scan")
    p.add_argument("--mock", action="store_true", help="use mock arc controller")
    p.add_argument("--arc-port", type=str, default="/dev/ttyUSB0")
    p.add_argument("--positions", type=int, default=None,
                   help="number of arc positions (default: uses arc step config)")
    p.add_argument("--frames", type=int, default=30,
                   help="frames to average per position")
    p.add_argument("--bag", type=str, default=None, help=".bag file for playback")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    config = PipelineConfig(
        arc_port=args.arc_port,
        frames_per_position=args.frames,
        use_mock_arc=args.mock,
        bag_file=args.bag,
    )

    pipe = ScanPipeline(config)
    pipe.capture_zero_reference(n_positions=args.positions)
    logging.getLogger(__name__).info("zero reference capture complete")


if __name__ == "__main__":
    main()