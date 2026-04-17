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
        {"cmd": "capture_frame", "request_id": 1, "index": 0, "angle_deg": 0.0, "target_steps": 0}
        
    server to GUI:
        {"type": "status", "state": "idle"}
        {"type": "status", "state": "running", "stage": 2, "stage_name": "register"}
        {"type": "status", "state": "complete", "run_dir": "data/runs/..."}
        {"type": "status", "state": "error", "message": "no device connected"}
        {"type": "log", "level": "info", "message": "captured 12000 points"}
        {"type": "capture_result", "request_id": 1, "ok": true, "index": 0, "angle_deg": 0.0, "point_count": 4823}
        {"type": "capture_result", "request_id": 1, "ok": false, "index": 0, "angle_deg": 0.0, "error": "no device connected"}

the GUI doesn't need to know about venvs, paths, or Python. it just
opens a socket and sends/receives JSON lines.
"""

import json
import logging
import socket
import threading
import time
from pathlib import Path

from pipeline import ScanPipeline, PipelineConfig

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

        elif action == "capture_frame":
        # Called from the UI between motion stops during a stepped scan.
        # Runs synchronously on the accept thread — capture is expected to
        # take under a second, so this doesn't block inbound commands long.
        req_id = cmd.get("request_id")
        index = cmd.get("index")
        angle = cmd.get("angle_deg")
        target_steps = cmd.get("target_steps")

        logger.info(f"capture_frame req={req_id} idx={index} angle={angle}")

        try:
            # self.pipe must have a capture_frame method; see note below.
            # Should return something like {"point_count": N, "path": "..."}
            # or raise on failure.
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
