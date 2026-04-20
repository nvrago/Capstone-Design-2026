"""
server.py -- pipeline server for GUI communication

runs as a background service on the Pi. the touchscreen GUI connects
via TCP and sends JSON commands. the server manages the camera,
captures frames on demand, and can run the full processing pipeline.

supports two scan modes:
    1. GUI-driven (stepped): GUI controls the ClearCore directly via Modbus,
       sends capture_frame at each position, then end_scan_session to trigger
       processing. this is the primary mode.
    2. autonomous: GUI sends "scan" and the server runs the full pipeline
       internally (capture + process + mesh + gcode). used for testing
       without the ClearCore.

protocol (JSON over TCP on localhost:5001):

    GUI to server:
        {"cmd": "start_scan_session"}
        {"cmd": "capture_frame", "request_id": 1, "index": 0, "angle_deg": 45.0, "target_steps": 50000}
        {"cmd": "end_scan_session"}
        {"cmd": "scan"}
        {"cmd": "scan", "skip_execute": true}
        {"cmd": "zero"}
        {"cmd": "stop"}
        {"cmd": "status"}
        {"cmd": "stage", "stage": 3, "end": 5}
        {"cmd": "config", "key": "arc_step_deg", "value": 10.0}

    server to GUI:
        {"type": "status", "state": "idle"}
        {"type": "status", "state": "running", "stage": 2, "stage_name": "register"}
        {"type": "status", "state": "complete", "run_dir": "data/runs/..."}
        {"type": "status", "state": "error", "message": "..."}
        {"type": "capture_result", "request_id": 1, "index": 0, "angle_deg": 45.0, "ok": true, "point_count": 12345}
        {"type": "log", "level": "info", "message": "..."}
"""

import json
import logging
import socket
import threading
import time
import numpy as np
import open3d as o3d
from datetime import datetime
from pathlib import Path

from pipeline import ScanPipeline, PipelineConfig
from scanner.capture import RealSenseCapture
from processing.zero_mesh import apply_zero_subtraction
from processing.registration import CloudRegistrator

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

        # camera instance (shared across scan sessions)
        self.scanner = None

        # scan session state
        self._session_active = False
        self._session_clouds = []  # list of (angle_deg, o3d.geometry.PointCloud)
        self._session_run_dir = None

    # camera lifecycle

    def _ensure_camera(self):
        """start the camera if not already running."""
        if self.scanner is not None:
            return
        try:
            self.scanner = RealSenseCapture(
                width=self.config.capture_width,
                height=self.config.capture_height,
                fps=self.config.capture_fps,
                temporal_frames=self.config.frames_per_position,
                decimation_magnitude=self.config.decimation_magnitude,
                bag_file=self.config.bag_file,
            )
            self.scanner.start()
            logger.info("camera started")
        except Exception as e:
            logger.error(f"camera start failed: {e}")
            self.scanner = None

    def _stop_camera(self):
        """stop the camera if running."""
        if self.scanner:
            try:
                self.scanner.stop()
            except Exception:
                pass
            self.scanner = None

    # message sending

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

    # scan session commands (GUI-driven stepped scan)

    def _handle_start_scan_session(self, cmd: dict):
        """prepare a new scan session. GUI will send capture_frame for each position."""
        if self._session_active:
            self._send({"type": "error", "message": "scan session already active"})
            return

        self._ensure_camera()
        if self.scanner is None:
            self._send({"type": "error", "message": "camera not available"})
            return

        # create timestamped run directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._session_run_dir = Path(self.config.data_dir) / "runs" / timestamp
        self._session_run_dir.mkdir(parents=True, exist_ok=True)
        (self._session_run_dir / "position_clouds").mkdir(exist_ok=True)

        self._session_clouds = []
        self._session_active = True
        self.state = "running"
        self.current_stage_name = "capture"

        logger.info(f"scan session started, output: {self._session_run_dir}")
        self._send({"type": "scan_session_started",
                     "run_dir": str(self._session_run_dir)})
        self._send_status()

    def _handle_capture_frame(self, cmd: dict):
        """capture a single frame at the current position."""
        if not self._session_active:
            self._send({"type": "capture_result",
                         "request_id": cmd.get("request_id"),
                         "ok": False,
                         "error": "no active scan session"})
            return

        request_id = cmd.get("request_id")
        index = cmd.get("index", 0)
        angle_deg = cmd.get("angle_deg", 0.0)
        target_steps = cmd.get("target_steps", 0)

        logger.info(f"capturing frame #{index} at {angle_deg} deg "
                     f"(steps={target_steps})")

        try:
            pcd = self.scanner.capture()
            if pcd is None or len(pcd.points) == 0:
                self._send({"type": "capture_result",
                             "request_id": request_id,
                             "index": index,
                             "angle_deg": angle_deg,
                             "ok": False,
                             "error": "empty capture"})
                return

            # depth clip
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=np.array([-10.0, -10.0, 0.0]),
                max_bound=np.array([10.0, 10.0, self.config.depth_clip_max_m])
            )
            pcd = pcd.crop(bbox)

            # save position cloud
            ply_path = self._session_run_dir / "position_clouds" / f"pos_{angle_deg:.1f}.ply"
            o3d.io.write_point_cloud(str(ply_path), pcd)

            self._session_clouds.append((angle_deg, pcd))
            point_count = len(pcd.points)

            logger.info(f"captured {point_count} points at {angle_deg} deg")

            self._send({"type": "capture_result",
                         "request_id": request_id,
                         "index": index,
                         "angle_deg": angle_deg,
                         "ok": True,
                         "point_count": point_count})

        except Exception as e:
            logger.error(f"capture failed: {e}")
            self._send({"type": "capture_result",
                         "request_id": request_id,
                         "index": index,
                         "angle_deg": angle_deg,
                         "ok": False,
                         "error": str(e)})

    def _handle_end_scan_session(self, cmd: dict):
        """end the scan session and run processing pipeline on captured data."""
        if not self._session_active:
            self._send({"type": "scan_session_ended", "ok": False,
                         "error": "no active session"})
            return

        self._session_active = False
        n_clouds = len(self._session_clouds)
        logger.info(f"scan session ended with {n_clouds} captures")

        if n_clouds == 0:
            self.state = "idle"
            self._send({"type": "scan_session_ended", "ok": False,
                         "error": "no frames captured"})
            self._send_status()
            return

        # run processing in background thread
        self.run_thread = threading.Thread(
            target=self._process_session,
            daemon=True,
        )
        self.run_thread.start()

    def _process_session(self):
        """process captured scan session data through the pipeline."""
        try:
            logging.getLogger().addHandler(self.gui_handler)
            self.gui_handler.set_client(self.client)

            self.state = "running"
            self.current_stage_name = "processing"

            # set up the pipeline with pre-captured data
            pipe = ScanPipeline(self.config)
            pipe.run_dir = self._session_run_dir
            pipe.position_clouds = list(self._session_clouds)

            # run stages 2-5 (skip capture since we already have the clouds)
            # set up file logging
            file_handler = logging.FileHandler(pipe.run_dir / "run.log")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logging.getLogger().addHandler(file_handler)

            t_start = time.time()

            pipe.stage_2_register()
            self._send_status(stage=2, stage_name="register")

            pipe.stage_3_process()
            self._send_status(stage=3, stage_name="process")

            pipe.stage_4_mesh()
            self._send_status(stage=4, stage_name="mesh")

            pipe.stage_5_toolpath()
            self._send_status(stage=5, stage_name="toolpath")

            total = time.time() - t_start
            logger.info(f"processing complete in {total:.1f}s")

            self.state = "complete"
            self._send_status(run_dir=str(self._session_run_dir))
            self._send({"type": "scan_session_ended", "ok": True,
                         "run_dir": str(self._session_run_dir)})

            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

        except Exception as e:
            self.state = "error"
            self._send_status(message=str(e))
            logger.error(f"processing failed: {e}", exc_info=True)
        finally:
            logging.getLogger().removeHandler(self.gui_handler)
            self.gui_handler.set_client(None)

    # autonomous pipeline commands

    def _run_pipeline(self, start_stage=1, end_stage=6, dry_run=False, skip_execute=False):
        """run the full pipeline autonomously in a background thread."""
        self.state = "running"
        self._stop_requested = False

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

    # command dispatch

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

        elif action == "start_scan_session":
            self._handle_start_scan_session(cmd)

        elif action == "capture_frame":
            self._handle_capture_frame(cmd)

        elif action == "end_scan_session":
            self._handle_end_scan_session(cmd)

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

        elif action == "stop":
            logger.warning("stop requested from GUI")
            self._stop_requested = True
            self._session_active = False
            if self.pipe and self.pipe.arc:
                try:
                    self.pipe.arc.stop()
                except Exception:
                    pass
            self.state = "idle"
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
        # start camera early so it's ready when the GUI connects
        self._ensure_camera()

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
                # end any active session on disconnect
                self._session_active = False


def main():
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

    global PORT
    PORT = args.port

    server = PipelineServer(config)
    server.serve()


if __name__ == "__main__":
    main()