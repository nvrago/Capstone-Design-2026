"""
server.py -- lightweight pipeline server for GUI communication

runs as a background service on the Pi. the touchscreen GUI connects
via TCP and sends JSON commands. the server manages pipeline state,
runs stages in background threads, and streams status/progress back
to the GUI.

protocol (JSON over TCP on localhost:5001):

    GUI to server:
        {"cmd": "scan"}
        {"cmd": "scan", "dry_run": true}
        {"cmd": "zero"}
        {"cmd": "stop"}
        {"cmd": "status"}
        {"cmd": "stage", "stage": 3, "end": 5}
        {"cmd": "config", "key": "arc_step_deg", "value": 10.0}
        {"cmd": "capture_pose", "pose_deg": 0.0, "pose_idx": 1}

    server to GUI:
        {"type": "status", "state": "idle"}
        {"type": "status", "state": "running", "stage": 2, "stage_name": "register"}
        {"type": "status", "state": "complete", "run_dir": "data/runs/..."}
        {"type": "status", "state": "error", "message": "no device connected"}
        {"type": "log", "level": "info", "message": "captured 12000 points"}
        {"type": "capture_complete", "ply_path": "...", "n_points": 12345,
         "pose_idx": 1, "pose_deg": 0.0}

the GUI doesn't need to know about venvs, paths, or Python. it just
opens a socket and sends/receives JSON lines.
"""

import json
import logging
import socket
import threading
import time
from pathlib import Path

# Note: `from pipeline import ScanPipeline, PipelineConfig` is deferred
# into the methods that actually use it. That lets the server start up
# and service `capture_pose` even when the full ScanPipeline's deps
# (OpenCAMLib, Open3D, etc.) aren't installed. `capture_pose` only uses
# pyrealsense2 + numpy and is fully self-contained.
try:
    from pipeline import PipelineConfig
    _SCANPIPELINE_AVAILABLE = True
except ImportError as _scanpipe_err:
    _SCANPIPELINE_AVAILABLE = False
    _SCANPIPE_IMPORT_ERROR = _scanpipe_err

    # Minimal stand-in so the server can still construct a default config
    # and carry it around. `scan` / `zero` / `stage` will fail loudly when
    # invoked, but startup + `capture_pose` work fine.
    class PipelineConfig:  # type: ignore[no-redef]
        def __init__(self):
            self.use_mock_arc = False

        @classmethod
        def from_yaml(cls, path):
            raise RuntimeError(
                "PipelineConfig.from_yaml unavailable: "
                f"ScanPipeline import failed ({_scanpipe_err})"
            )

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 5001
BUFFER_SIZE = 4096


class GUIHandler(logging.Handler):
    """logging handler that forwards log messages to the GUI socket."""

    def __init__(self):
        super().__init__()
        self.client = None

    def set_client(self, client_socket):
        self.client = client_socket

    def emit(self, record):
        if self.client is None:
            return
        try:
            msg = {
                "type": "log",
                "level": record.levelname.lower(),
                "message": self.format(record),
            }
            self._send(msg)
        except Exception:
            pass

    def _send(self, msg):
        if self.client:
            try:
                data = json.dumps(msg) + "\n"
                self.client.sendall(data.encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.client = None


class PipelineServer:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.pipe = ScanPipeline(self.config)
        self.state = "idle"
        self.current_stage = 0
        self.current_stage_name = ""
        self.run_thread = None
        self.client = None
        self.gui_handler = GUIHandler()
        self.gui_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(message)s"
        ))
        self._stop_requested = False

    def _send(self, msg: dict):
        """send a JSON message to the connected GUI client."""
        if self.client:
            try:
                data = json.dumps(msg) + "\n"
                self.client.sendall(data.encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError, OSError):
                logger.warning("GUI client disconnected")
                self.client = None

    def _send_status(self, **kwargs):
        """send a status update to the GUI."""
        msg = {"type": "status", "state": self.state}
        if self.state == "running":
            msg["stage"] = self.current_stage
            msg["stage_name"] = self.current_stage_name
        msg.update(kwargs)
        self._send(msg)

    def _run_pipeline(self, start_stage=1, end_stage=6, dry_run=False, skip_execute=False):
        """run the pipeline in a background thread."""
        self.state = "running"
        self._stop_requested = False

        try:
            from pipeline import ScanPipeline  # deferred to avoid OCL dep at startup
        except ImportError as e:
            self.state = "error"
            self._send_status(message=f"ScanPipeline unavailable: {e}")
            logger.error(f"ScanPipeline import failed: {e}")
            return

        try:
            logging.getLogger().addHandler(self.gui_handler)
            self.gui_handler.set_client(self.client)

            self.pipe = ScanPipeline(self.config)
            self.pipe.run(
                start_stage=start_stage,
                end_stage=end_stage,
                dry_run=dry_run,
                skip_execute=skip_execute,
            )

            self.state = "complete"
            self._send_status(
                run_dir=str(self.pipe.run_dir) if self.pipe.run_dir else None
            )

        except KeyboardInterrupt:
            self.state = "stopped"
            self._send_status(message="pipeline stopped by user")
        except Exception as e:
            self.state = "error"
            self._send_status(message=str(e))
            logger.error(f"pipeline error: {e}", exc_info=True)
        finally:
            logging.getLogger().removeHandler(self.gui_handler)
            self.gui_handler.set_client(None)

    def _run_zero(self):
        """capture zero reference in background thread."""
        self.state = "running"
        self.current_stage_name = "zero_capture"

        try:
            from pipeline import ScanPipeline  # deferred to avoid OCL dep at startup
        except ImportError as e:
            self.state = "error"
            self._send_status(message=f"ScanPipeline unavailable: {e}")
            logger.error(f"ScanPipeline import failed: {e}")
            return

        try:
            logging.getLogger().addHandler(self.gui_handler)
            self.gui_handler.set_client(self.client)

            self.pipe = ScanPipeline(self.config)
            self.pipe.capture_zero_reference()

            self.state = "complete"
            self._send_status(message="zero reference captured")

        except Exception as e:
            self.state = "error"
            self._send_status(message=str(e))
        finally:
            logging.getLogger().removeHandler(self.gui_handler)
            self.gui_handler.set_client(None)

    def _run_capture_pose(self, pose_deg: float, pose_idx: int):
        """single-pose capture for the UI's multi-pose scan flow.

        the UI is driving arc motion via the ClearCore directly (Modbus),
        so we don't move anything here -- we just open the RealSense,
        grab a temporally-averaged depth frame, build a point cloud,
        save a binary PLY, and report the path back over the socket.

        this path deliberately AVOIDS open3d so the server can run under
        python 3.12 (open3d publishes no cp312 aarch64 wheel). the UI,
        which runs under 3.11 with a working open3d install, handles all
        ICP alignment and merging after loading the PLY we produce.
        """
        from pathlib import Path
        from datetime import datetime
        import struct
        # Import here so the rest of server.py still runs on a dev machine
        # without pyrealsense2 installed.
        import pyrealsense2 as rs
        import numpy as np

        self.state = "running"
        self.current_stage_name = f"capture_pose_{pose_idx}"

        pipe = None
        try:
            logging.getLogger().addHandler(self.gui_handler)
            self.gui_handler.set_client(self.client)

            # Pick an output directory: reuse the pipeline's run_dir if one
            # already exists, otherwise drop into data/poses/<timestamp>.
            if self.pipe and getattr(self.pipe, "run_dir", None):
                out_dir = Path(self.pipe.run_dir) / "poses"
            else:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = Path("data") / "poses" / stamp
            out_dir.mkdir(parents=True, exist_ok=True)

            ply_path = out_dir / f"pose_{pose_idx:03d}_{pose_deg:+07.2f}deg.ply"
            logger.info(
                f"capture_pose: idx={pose_idx} deg={pose_deg:+.2f} -> {ply_path}"
            )

            # ── Pipeline setup ────────────────────────────────────────────
            W, H, FPS = 640, 480, 30
            TEMPORAL_FRAMES = 15

            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
            profile = pipe.start(cfg)

            # High-accuracy preset for D405 if supported
            dev = profile.get_device()
            depth_sensor = dev.first_depth_sensor()
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 3)

            # Filter chain: decimation → spatial → temporal → hole-fill
            decimation = rs.decimation_filter()
            decimation.set_option(rs.option.filter_magnitude, 2)
            spatial = rs.spatial_filter()
            spatial.set_option(rs.option.filter_magnitude, 2)
            spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            spatial.set_option(rs.option.filter_smooth_delta, 20)
            temporal = rs.temporal_filter()
            temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
            temporal.set_option(rs.option.filter_smooth_delta, 20)
            hole_fill = rs.hole_filling_filter()

            # Warm-up: auto-exposure stabilize
            for _ in range(30):
                pipe.wait_for_frames()

            # Temporal averaging — each pass refines the estimate
            depth_frame = None
            for _ in range(TEMPORAL_FRAMES):
                frames = pipe.wait_for_frames()
                d = frames.get_depth_frame()
                if not d:
                    continue
                d = decimation.process(d)
                d = spatial.process(d)
                d = temporal.process(d)
                d = hole_fill.process(d)
                depth_frame = d
            if depth_frame is None:
                raise RuntimeError("no valid depth frames captured")

            # Convert to point cloud
            pc = rs.pointcloud()
            points = pc.calculate(depth_frame)
            verts = (np.asanyarray(points.get_vertices())
                       .view(np.float32)
                       .reshape(-1, 3))

            # Drop zero-depth points and anything outside the D405's
            # accurate range. Matches the UI's old clipping (~4cm - 50cm).
            z = verts[:, 2]
            mask = (z > 0.04) & (z < 0.50)
            verts = verts[mask]

            n_points = int(verts.shape[0])
            logger.info(f"capture_pose: {n_points} points after filtering")

            # ── Write binary little-endian PLY ────────────────────────────
            # Open3D reads this format just fine on the UI side.
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {n_points}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
            )
            with open(ply_path, "wb") as f:
                f.write(header.encode("ascii"))
                f.write(verts.astype(np.float32).tobytes())

            self._send({
                "type": "capture_complete",
                "ply_path": str(ply_path),
                "n_points": n_points,
                "pose_idx": pose_idx,
                "pose_deg": pose_deg,
            })
            self.state = "idle"
            self._send_status(message=f"pose {pose_idx} captured")

        except Exception as e:
            self.state = "error"
            logger.error(f"capture_pose error: {e}", exc_info=True)
            self._send_status(message=str(e))
            self._send({
                "type": "capture_failed",
                "pose_idx": pose_idx,
                "message": str(e),
            })
        finally:
            if pipe is not None:
                try:
                    pipe.stop()
                except Exception:
                    pass
            logging.getLogger().removeHandler(self.gui_handler)
            self.gui_handler.set_client(None)

    def handle_command(self, raw: str):
        """parse and execute a command from the GUI."""
        try:
            cmd = json.loads(raw.strip())
        except json.JSONDecodeError:
            self._send({"type": "error", "message": f"invalid JSON: {raw}"})
            return

        action = cmd.get("cmd", "")
        logger.info(f"GUI command: {action}")

        if action == "status":
            self._send_status()

        elif action == "scan":
            if self.state == "running":
                self._send({"type": "error", "message": "pipeline already running"})
                return

            dry_run = cmd.get("dry_run", False)
            skip_execute = cmd.get("skip_execute", True)
            self.run_thread = threading.Thread(
                target=self._run_pipeline,
                kwargs={"dry_run": dry_run, "skip_execute": skip_execute},
                daemon=True,
            )
            self.run_thread.start()
            self._send_status()

        elif action == "zero":
            if self.state == "running":
                self._send({"type": "error", "message": "pipeline already running"})
                return

            self.run_thread = threading.Thread(
                target=self._run_zero, daemon=True
            )
            self.run_thread.start()
            self._send_status()

        elif action == "stage":
            if self.state == "running":
                self._send({"type": "error", "message": "pipeline already running"})
                return

            start = cmd.get("stage", 1)
            end = cmd.get("end", 6)
            dry_run = cmd.get("dry_run", False)

            self.run_thread = threading.Thread(
                target=self._run_pipeline,
                kwargs={
                    "start_stage": start,
                    "end_stage": end,
                    "dry_run": dry_run,
                },
                daemon=True,
            )
            self.run_thread.start()
            self._send_status()

        elif action == "capture_pose":
            if self.state == "running":
                self._send({"type": "error",
                            "message": "pipeline already running"})
                return

            pose_deg = float(cmd.get("pose_deg", 0.0))
            pose_idx = int(cmd.get("pose_idx", 0))

            self.run_thread = threading.Thread(
                target=self._run_capture_pose,
                kwargs={"pose_deg": pose_deg, "pose_idx": pose_idx},
                daemon=True,
            )
            self.run_thread.start()
            self._send_status()

        elif action == "stop":
            logger.warning("stop requested from GUI")
            self._stop_requested = True
            if self.pipe and self.pipe.arc:
                try:
                    self.pipe.arc.stop()
                except Exception:
                    pass
            self._send_status(message="stop requested")

        elif action == "config":
            key = cmd.get("key")
            value = cmd.get("value")
            if key and hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"config updated: {key} = {value}")
                self._send({"type": "config_updated", "key": key, "value": value})
            else:
                self._send({"type": "error", "message": f"unknown config key: {key}"})

        else:
            self._send({"type": "error", "message": f"unknown command: {action}"})

    def serve(self):
        """start the TCP server and listen for GUI connections."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        logger.info(f"pipeline server listening on {HOST}:{PORT}")

        while True:
            try:
                self.client, addr = server.accept()
                logger.info(f"GUI connected from {addr}")
                self._send_status()

                buffer = ""
                while True:
                    data = self.client.recv(BUFFER_SIZE)
                    if not data:
                        break
                    buffer += data.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            self.handle_command(line)

            except (ConnectionResetError, BrokenPipeError):
                logger.info("GUI disconnected")
            except Exception as e:
                logger.error(f"server error: {e}", exc_info=True)
            finally:
                if self.client:
                    try:
                        self.client.close()
                    except Exception:
                        pass
                    self.client = None


def main():
    global PORT
    import argparse

    p = argparse.ArgumentParser(description="pipeline server for GUI")
    p.add_argument("--config", "-c", type=str, default=None,
                   help="yaml config directory")
    p.add_argument("--mock", action="store_true",
                   help="use mock arc controller")
    p.add_argument("--port", type=int, default=PORT,
                   help=f"TCP port (default: {PORT})")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig()

    if args.mock:
        config.use_mock_arc = True

    PORT = args.port

    server = PipelineServer(config)
    server.serve()


if __name__ == "__main__":
    main()
