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
        {"type": "capture_result", "request_id": 1, "ok": true, "index": 0, "angle_deg": 0.0, "point_count": 4823}
        {"type": "capture_result", "request_id": 1, "ok": false, "index": 0, "angle_deg": 0.0, "error": "..."}
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

    # autonomous pipeline commands

    def _run_pipeline(self, start_stage=1, end_stage=6, dry_run=False, skip_execute=False):
        """run the full pipeline autonomously in a background thread."""
        self.state = "running"
        self._stop_requested = False

        try:
            logging.getLogger().addHandler(self.gui_handler)
            self.gui_handler.set_client(self.client)

            # release any interactive-session scanner before the pipeline
            # claims the camera itself in setup()
            if self.pipe is not None:
                try:
                    self.pipe.end_interactive_capture()
                except Exception:
                    pass

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

    def _process_session(self):
        """process captured scan session data through the pipeline stages 2-5."""
        try:
            logging.getLogger().addHandler(self.gui_handler)
            self.gui_handler.set_client(self.client)

            self.state = "running"
            self.current_stage_name = "processing"

            # set up file logging
            file_handler = logging.FileHandler(self.pipe.run_dir / "run.log")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logging.getLogger().addHandler(file_handler)

            t_start = time.time()

            self.pipe.stage_2_register()
            self._send_status(stage=2, stage_name="register")

            self.pipe.stage_3_process()
            self._send_status(stage=3, stage_name="process")

            self.pipe.stage_4_mesh()
            self._send_status(stage=4, stage_name="mesh")

            self.pipe.stage_5_toolpath()
            self._send_status(stage=5, stage_name="toolpath")

            total = time.time() - t_start
            logger.info(f"processing complete in {total:.1f}s")

            self.state = "complete"
            self._send_status(run_dir=str(self.pipe.run_dir))
            self._send({"type": "scan_session_ended", "ok": True,
                         "run_dir": str(self.pipe.run_dir)})

            logging.getLogger().removeHandler(file_handler)
            file_handler.close()

        except Exception as e:
            self.state = "error"
            self._send_status(message=str(e))
            logger.error(f"processing failed: {e}", exc_info=True)
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
            if self.state == "running":
                self._send({"type": "error",
                             "message": "cannot start scan session while pipeline is running"})
                return
            # tear down old pipeline's scanner if still hot
            if self.pipe is not None:
                try:
                    self.pipe.end_interactive_capture()
                except Exception:
                    pass
            self.pipe = ScanPipeline(self.config)
            logger.info("interactive scan session started (fresh pipeline)")
            self._send({"type": "scan_session_started"})

        elif action == "capture_frame":
            req_id = cmd.get("request_id")
            index = cmd.get("index")
            angle = cmd.get("angle_deg")
            target_steps = cmd.get("target_steps", 0)

            logger.info(f"capture_frame req={req_id} idx={index} angle={angle}")

            if self.pipe is None:
                self._send({
                    "type": "capture_result",
                    "request_id": req_id,
                    "ok": False,
                    "index": index,
                    "angle_deg": angle,
                    "error": "no active pipeline, send start_scan_session first",
                })
                return

            try:
                result = self.pipe.capture_frame(
                    index=index,
                    angle_deg=angle,
                    target_steps=target_steps,
                )
                self._send({
                    "type": "capture_result",
                    "request_id": req_id,
                    "ok": True,
                    "index": index,
                    "angle_deg": angle,
                    **(result or {}),
                })
            except Exception as e:
                logger.error(f"capture_frame failed: {e}", exc_info=True)
                self._send({
                    "type": "capture_result",
                    "request_id": req_id,
                    "ok": False,
                    "index": index,
                    "angle_deg": angle,
                    "error": str(e),
                })

        elif action == "end_scan_session":
            if self.pipe is None or not self.pipe.position_clouds:
                self._send({"type": "scan_session_ended", "ok": False,
                             "error": "no frames captured"})
                return

            # run processing in background thread
            self.run_thread = threading.Thread(
                target=self._process_session,
                daemon=True,
            )
            self.run_thread.start()

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
