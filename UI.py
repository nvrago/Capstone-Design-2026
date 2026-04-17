"""
Scan-to-Mill UI
===============
Multi-pose scan workflow:

    UI  ──Modbus TCP──>  ClearCore PLC   (motion: CCMD_MOVE between poses)
    UI  ──JSON  TCP──>  server.py        (camera: capture_pose at each stop)
    UI  ──ICP+merge──>  PyVista viewport (accumulated cloud displayed live)

The UI drives the sequence end-to-end:
    for pose in SCAN_POSES_STEPS:
        CCMD_MOVE -> pose, wait until STATUS_MOVING clears
        settle dwell (~300 ms)
        JSON cmd capture_pose -> receive .ply path
        load, ICP-align against accumulated cloud, merge, render
    done.

The RealSense D405 is owned exclusively by the server process. The UI
never touches pyrealsense2 — that avoids two processes fighting over
the same USB device.

Requires: PyQt6 (or PyQt5 fallback), pyvistaqt, pyvista, numpy, open3d,
          pymodbus 3.x
"""

import sys
import queue
import json
import socket
import numpy as np
import os

# Force Qt to use the X11 (xcb) platform plugin instead of Wayland.
# VTK's rendering code talks directly to X11 and gets BadWindow errors
# when Qt is using the wayland backend on Raspberry Pi OS. xcb works
# correctly on both X11 and Wayland sessions (via XWayland).
os.environ["QT_QPA_PLATFORM"] = "xcb"

import pyvista as pv
from pyvistaqt import QtInteractor

try:
    from pymodbus.client import ModbusTcpClient
    from pymodbus.exceptions import ModbusException
    _PYMODBUS_AVAILABLE = True
except ModuleNotFoundError:
    ModbusTcpClient = None
    ModbusException = Exception
    _PYMODBUS_AVAILABLE = False

try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ModuleNotFoundError:
    o3d = None
    _OPEN3D_AVAILABLE = False

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QProgressBar, QTextEdit, QGroupBox,
        QSizePolicy, QFrame,
    )
    from PyQt6.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
    from PyQt6.QtGui import QTextCursor
except ModuleNotFoundError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QProgressBar, QTextEdit, QGroupBox,
        QSizePolicy, QFrame,
    )
    from PyQt5.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
    from PyQt5.QtGui import QTextCursor


# ── Stylesheet ────────────────────────────────────────────────────────────────
STYLE = """
QMainWindow, QWidget {
    background-color: #0d0f14;
    color: #c8cdd6;
    font-family: 'Courier New', monospace;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #2a2f3d;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 8px;
    font-size: 11px;
    font-weight: bold;
    color: #5a8fa8;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QPushButton {
    background-color: #1a1e28;
    border: 1px solid #2a2f3d;
    border-radius: 3px;
    padding: 7px 14px;
    color: #c8cdd6;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    letter-spacing: 0.5px;
}
QPushButton:hover {
    background-color: #252a38;
    border-color: #4a8fa8;
    color: #ffffff;
}
QPushButton:pressed { background-color: #0d1520; }
QPushButton:disabled {
    color: #3a3f4d;
    border-color: #1e2230;
}
QPushButton#btn_start {
    background-color: #0d2d1a;
    border-color: #1a6b3a;
    color: #2dcc70;
    font-weight: bold;
}
QPushButton#btn_start:hover {
    background-color: #133d22;
    border-color: #2dcc70;
}
QPushButton#btn_stop {
    background-color: #2d0d0d;
    border-color: #6b1a1a;
    color: #cc2d2d;
    font-weight: bold;
}
QPushButton#btn_stop:hover {
    background-color: #3d1313;
    border-color: #cc2d2d;
}
QProgressBar {
    background-color: #111520;
    border: 1px solid #2a2f3d;
    border-radius: 2px;
    height: 16px;
    text-align: center;
    color: #c8cdd6;
    font-size: 10px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a4a6b, stop:1 #2d8fa8);
    border-radius: 1px;
}
QTextEdit {
    background-color: #080a0f;
    border: 1px solid #1e2230;
    border-radius: 3px;
    color: #7a8fa0;
    font-family: 'Courier New', monospace;
    font-size: 10px;
    padding: 4px;
}
QLabel { color: #8a9aaa; }
QLabel#header {
    color: #2daacc;
    font-size: 18px;
    font-weight: bold;
    letter-spacing: 3px;
}
QLabel#subheader {
    color: #4a6a7a;
    font-size: 10px;
    letter-spacing: 2px;
}
QLabel#value_display {
    background-color: #080a0f;
    border: 1px solid #1e2230;
    border-radius: 2px;
    color: #2daacc;
    font-size: 13px;
    font-weight: bold;
    padding: 4px 8px;
    min-width: 80px;
}
QLabel#status_ok   { color: #2dcc70; font-weight: bold; }
QLabel#status_off  { color: #cc4d2d; font-weight: bold; }
"""


# ══════════════════════════════════════════════════════════════════════════════
# ClearCore Modbus TCP Link  (unchanged from prior version)
# ══════════════════════════════════════════════════════════════════════════════
#
# Register map mirrors CLIENT_INFC in the ClearCore sketch.
# 32-bit values cross the wire low-word-first (little-endian ARM).
#
# ──────────────────────────────────────────────────────────────────────────────

MODBUS_DEFAULT_HOST = "192.168.1.20"
MODBUS_DEFAULT_TCP_PORT = 502
MODBUS_DEFAULT_SLAVE_ID = 1
MODBUS_POLL_INTERVAL_MS = 100   # tighter poll so we catch MOVING edges faster

# CCMTR_CMD
CCMD_NONE       = 0
CCMD_ENAB_MTRS  = 1
CCMD_DISAB_MTRS = 2
CCMD_SET_ZERO   = 3
CCMD_MOVE       = 4
CCMD_STOP       = 5
CCMD_RUN_1      = 6
CCMD_ACK        = 7
CCMD_NACK       = 8

# Register offsets (word addresses into CLIENT_INFC)
W_ACC         = 0
W_VEL         = 2
W_TARGET_POSN = 4
W_CMD         = 6
W_CUR_POSN    = 7
W_STATUS      = 9
W_MSG_CNT     = 10
W_MSG         = 11
W_STATE       = 43
W_MSG_LEN     = 32

STATUS_BLOCK_START = W_CUR_POSN
STATUS_BLOCK_LEN   = 4

# Status bits
STATUS_READY    = 1 << 0
STATUS_MOVING   = 1 << 1
STATUS_HOMED    = 1 << 2
STATUS_FAULT    = 1 << 3
STATUS_ESTOP    = 1 << 4
STATUS_SCANNING = 1 << 5
STATUS_HLFB     = 1 << 6


def pack_u32(value: int) -> list:
    v = value & 0xFFFFFFFF
    return [v & 0xFFFF, (v >> 16) & 0xFFFF]

def pack_i32(value: int) -> list:
    return pack_u32(value & 0xFFFFFFFF)

def unpack_i32(regs) -> int:
    raw = ((regs[1] & 0xFFFF) << 16) | (regs[0] & 0xFFFF)
    if raw & 0x80000000:
        raw -= 0x100000000
    return raw


class ClearCoreModbus(QThread):
    """Background worker owning the Modbus TCP link to the ClearCore."""

    connected     = pyqtSignal(bool)
    status_update = pyqtSignal(dict)
    log_message   = pyqtSignal(str)
    error         = pyqtSignal(str)

    def __init__(self, host: str = MODBUS_DEFAULT_HOST,
                 tcp_port: int = MODBUS_DEFAULT_TCP_PORT,
                 slave_id: int = MODBUS_DEFAULT_SLAVE_ID,
                 parent=None):
        super().__init__(parent)
        self._host = host
        self._tcp_port = tcp_port
        self._slave_id = slave_id
        self._client = None
        self._cmd_queue: "queue.Queue[tuple]" = queue.Queue()
        self._stop_flag = False
        self._connected = False
        self._last_msg_cnt = 0

    def send_command(self, cmd_code: int, target_posn: int = 0,
                     velocity: int = 0, accel: int = 0):
        self._cmd_queue.put(("cmd", cmd_code, target_posn, velocity, accel))

    def write_holding(self, address: int, value: int):
        self._cmd_queue.put(("wr", address, value))

    def stop(self):
        self._stop_flag = True

    def run(self):
        if not _PYMODBUS_AVAILABLE:
            self.error.emit("pymodbus not installed — run: pip install pymodbus")
            return

        if not self._open_port():
            return

        self.log_message.emit(
            f"[MODBUS] Link up to ClearCore at {self._host}:{self._tcp_port}, "
            f"slave id {self._slave_id}"
        )

        while not self._stop_flag:
            self._drain_command_queue()
            if self._stop_flag:
                break
            self._poll_status()
            self.msleep(MODBUS_POLL_INTERVAL_MS)

        self._close_port()
        self.log_message.emit("[MODBUS] Link closed.")

    def _open_port(self) -> bool:
        try:
            self._client = ModbusTcpClient(
                host=self._host, port=self._tcp_port, timeout=2.0,
            )
            if not self._client.connect():
                self.error.emit(
                    f"[MODBUS] Failed to reach ClearCore at "
                    f"{self._host}:{self._tcp_port}"
                )
                self.connected.emit(False)
                return False
            self._connected = True
            self.connected.emit(True)
            return True
        except Exception as e:
            self.error.emit(f"[MODBUS] Open error: {e}")
            self.connected.emit(False)
            return False

    def _close_port(self):
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
        self._client = None
        if self._connected:
            self._connected = False
            self.connected.emit(False)

    def _drain_command_queue(self):
        while True:
            try:
                item = self._cmd_queue.get_nowait()
            except queue.Empty:
                return

            kind = item[0]
            try:
                if kind == "cmd":
                    _, cmd_code, target_posn, velocity, accel = item
                    if cmd_code == CCMD_MOVE:
                        values = (pack_u32(accel) +
                                  pack_i32(velocity) +
                                  pack_i32(target_posn) +
                                  [cmd_code])
                        rsp = self._client.write_registers(
                            W_ACC, values, device_id=self._slave_id
                        )
                        label = (f"MOVE target={target_posn} "
                                 f"vel={velocity} acc={accel}")
                    else:
                        rsp = self._client.write_register(
                            W_CMD, cmd_code, device_id=self._slave_id
                        )
                        label = f"CMD {cmd_code}"

                    if rsp.isError():
                        self.error.emit(f"[MODBUS] write err: {rsp}")
                    else:
                        self.log_message.emit(f"[MODBUS] → {label}")

                elif kind == "wr":
                    _, address, value = item
                    rsp = self._client.write_register(
                        address, value, device_id=self._slave_id
                    )
                    if rsp.isError():
                        self.error.emit(f"[MODBUS] write_register err: {rsp}")
            except ModbusException as e:
                self.error.emit(f"[MODBUS] Write exception: {e}")
            except Exception as e:
                self.error.emit(f"[MODBUS] Unexpected write error: {e}")

    def _poll_status(self):
        if self._client is None:
            return
        try:
            rsp = self._client.read_holding_registers(
                STATUS_BLOCK_START, count=STATUS_BLOCK_LEN,
                device_id=self._slave_id
            )
            if rsp.isError():
                self.error.emit(f"[MODBUS] read err: {rsp}")
                return
            regs = rsp.registers
            cur_posn = unpack_i32([regs[0], regs[1]])
            status_word = regs[2] & 0xFFFF
            msg_cnt = regs[3] & 0xFFFF

            parsed = {
                "raw":      status_word,
                "ready":    bool(status_word & STATUS_READY),
                "moving":   bool(status_word & STATUS_MOVING),
                "homed":    bool(status_word & STATUS_HOMED),
                "fault":    bool(status_word & STATUS_FAULT),
                "estop":    bool(status_word & STATUS_ESTOP),
                "scanning": bool(status_word & STATUS_SCANNING),
                "hlfb":     bool(status_word & STATUS_HLFB),
                "cur_posn": cur_posn,
                "msg_cnt":  msg_cnt,
            }
            self.status_update.emit(parsed)

            if msg_cnt != self._last_msg_cnt:
                self._last_msg_cnt = msg_cnt
                self._pull_msg_buffer()

        except ModbusException as e:
            self.error.emit(f"[MODBUS] Read exception: {e}")
        except Exception as e:
            self.error.emit(f"[MODBUS] Unexpected read error: {e}")

    def _pull_msg_buffer(self):
        try:
            rsp = self._client.read_holding_registers(
                W_MSG, count=W_MSG_LEN, device_id=self._slave_id
            )
            if rsp.isError():
                return
            buf = bytearray()
            for r in rsp.registers:
                buf.append(r & 0xFF)
                buf.append((r >> 8) & 0xFF)
            nul = buf.find(0)
            if nul >= 0:
                buf = buf[:nul]
            text = buf.decode("ascii", errors="replace").strip()
            if text:
                self.log_message.emit(f"[CC] {text}")
        except Exception as e:
            self.error.emit(f"[MODBUS] msg read err: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Client  —  JSON over TCP to server.py
# ══════════════════════════════════════════════════════════════════════════════
#
# server.py listens on 127.0.0.1:5001 for newline-delimited JSON commands
# and sends back newline-delimited JSON events. This worker:
#
#   - maintains the socket connection in a background thread
#   - exposes send_json() as the main-thread API
#   - demultiplexes incoming events into Qt signals so the GUI can react
#
# Protocol extensions (server side) we rely on for multi-pose scans:
#   GUI -> server:  {"cmd": "capture_pose", "pose_deg": 0.0, "pose_idx": 1}
#   server -> GUI:  {"type": "capture_complete",
#                    "ply_path": "/path/to/pose_1.ply",
#                    "n_points": 12483,
#                    "pose_idx": 1}
#
# Existing server commands (scan / zero / status / stop / stage / config)
# still work unchanged — we just add capture_pose on top.
#
# ──────────────────────────────────────────────────────────────────────────────

PIPELINE_DEFAULT_HOST = "127.0.0.1"
PIPELINE_DEFAULT_PORT = 5001


class PipelineClient(QThread):
    """Background worker for the JSON/TCP link to server.py."""

    connected        = pyqtSignal(bool)
    log_message      = pyqtSignal(str)
    error            = pyqtSignal(str)
    status_update    = pyqtSignal(dict)        # {"state": ..., "stage": ...}
    capture_complete = pyqtSignal(dict)        # {"ply_path": ..., "n_points": ..., "pose_idx": ...}

    def __init__(self, host: str = PIPELINE_DEFAULT_HOST,
                 port: int = PIPELINE_DEFAULT_PORT, parent=None):
        super().__init__(parent)
        self._host = host
        self._port = port
        self._sock: "socket.socket | None" = None
        self._send_queue: "queue.Queue[dict]" = queue.Queue()
        self._stop_flag = False
        self._connected = False

    # ── Public API ────────────────────────────────────────────────────────────
    def send_json(self, msg: dict):
        """Queue a JSON message for the server. Non-blocking."""
        self._send_queue.put(msg)

    def capture_pose(self, pose_deg: float, pose_idx: int):
        """Convenience wrapper for the multi-pose scan flow."""
        self.send_json({
            "cmd": "capture_pose",
            "pose_deg": float(pose_deg),
            "pose_idx": int(pose_idx),
        })

    def stop(self):
        self._stop_flag = True

    # ── Worker loop ───────────────────────────────────────────────────────────
    def run(self):
        if not self._open_socket():
            return

        self._sock.settimeout(0.1)  # short timeout so we can service sends
        recv_buffer = ""

        while not self._stop_flag:
            # 1. flush pending outgoing messages
            self._drain_send_queue()
            if self._stop_flag:
                break

            # 2. try to read anything the server sent
            try:
                chunk = self._sock.recv(4096)
                if not chunk:
                    self.error.emit("[PIPE] Server closed connection")
                    break
                recv_buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in recv_buffer:
                    line, recv_buffer = recv_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._dispatch(line)
            except socket.timeout:
                pass
            except OSError as e:
                self.error.emit(f"[PIPE] Socket error: {e}")
                break

        self._close_socket()

    def _open_socket(self) -> bool:
        try:
            self._sock = socket.create_connection(
                (self._host, self._port), timeout=2.0,
            )
            self._connected = True
            self.connected.emit(True)
            self.log_message.emit(
                f"[PIPE] Connected to server at {self._host}:{self._port}"
            )
            return True
        except OSError as e:
            self.error.emit(
                f"[PIPE] Could not reach server at "
                f"{self._host}:{self._port}: {e}"
            )
            self.connected.emit(False)
            return False

    def _close_socket(self):
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        if self._connected:
            self._connected = False
            self.connected.emit(False)
        self.log_message.emit("[PIPE] Link closed.")

    def _drain_send_queue(self):
        while True:
            try:
                msg = self._send_queue.get_nowait()
            except queue.Empty:
                return
            try:
                line = (json.dumps(msg) + "\n").encode("utf-8")
                self._sock.sendall(line)
            except OSError as e:
                self.error.emit(f"[PIPE] Send error: {e}")
                self._stop_flag = True
                return

    def _dispatch(self, line: str):
        """Parse one JSON line from the server and emit the right signal."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            self.error.emit(f"[PIPE] Bad JSON from server: {line!r}")
            return

        mtype = msg.get("type", "")
        if mtype == "log":
            level = msg.get("level", "info").upper()
            text = msg.get("message", "")
            self.log_message.emit(f"[SRV/{level}] {text}")
        elif mtype == "status":
            self.status_update.emit(msg)
        elif mtype == "capture_complete":
            self.capture_complete.emit(msg)
        elif mtype == "error":
            self.error.emit(f"[PIPE] Server error: {msg.get('message')}")
        else:
            # Anything else — surface it as a log line so we don't lose signal
            self.log_message.emit(f"[PIPE] {line}")


# ══════════════════════════════════════════════════════════════════════════════
# Scan Orchestrator  —  drives the 3-pose sequence
# ══════════════════════════════════════════════════════════════════════════════
#
# States:
#   IDLE       no scan in progress
#   MOVING     CCMD_MOVE sent, waiting for STATUS_MOVING to drop
#   SETTLING   arrived at pose, dwelling briefly so vibration damps out
#   CAPTURING  capture_pose sent, waiting for capture_complete from server
#   MERGING    loading .ply, running ICP, merging into accumulated cloud
#   DONE       all poses captured
#   STOPPED    user hit STOP or E-stop latched
#
# Flow (per pose):
#   MOVING  --(moving=0)-->  SETTLING  --(timer)-->  CAPTURING
#   CAPTURING  --(capture_complete)-->  MERGING  --(done)-->  next pose or DONE
#
# ──────────────────────────────────────────────────────────────────────────────

# -- Arc geometry and motion tuning ------------------------------------------
# TUNE: steps per degree on your arc carriage. Depends on motor steps/rev
# × microstepping × gearbox ratio × any pulley reduction to the arc.
# Placeholder: 100 steps/° (i.e. 36000 steps for a full revolution).
STEPS_PER_DEG = 100

# 3 hard-coded poses for MVP. Change angles here, steps recompute.
SCAN_POSES_DEG = [-30.0, 0.0, +30.0]
SCAN_POSES_STEPS = [int(round(d * STEPS_PER_DEG)) for d in SCAN_POSES_DEG]

# Default motion profile for pose-to-pose moves.
SCAN_MOVE_VEL   = 20_000     # steps/sec
SCAN_MOVE_ACCEL = 100_000    # steps/sec²

# How long to sit still at each pose before snapping, in ms.
SETTLE_DWELL_MS = 300

# Max time to wait for a single move or capture to complete, in ms.
MOVE_TIMEOUT_MS    = 20_000
CAPTURE_TIMEOUT_MS = 15_000


class ScanOrchestrator(QObject):
    """State machine that sequences motion + captures. Lives on the GUI
    thread and is driven by signals from ClearCoreModbus and PipelineClient.
    """

    # Lifecycle signals for the UI to react to
    progress     = pyqtSignal(int, int)         # (current_pose_idx, total_poses)
    stage_change = pyqtSignal(str)              # human-readable stage name
    log_message  = pyqtSignal(str)
    pose_merged  = pyqtSignal(object)           # merged pv.PolyData (accumulated)
    finished     = pyqtSignal(bool, str)        # (success, reason)

    # States
    S_IDLE      = "IDLE"
    S_MOVING    = "MOVING"
    S_SETTLING  = "SETTLING"
    S_CAPTURING = "CAPTURING"
    S_MERGING   = "MERGING"
    S_DONE      = "DONE"
    S_STOPPED   = "STOPPED"

    def __init__(self, modbus: ClearCoreModbus, pipeline: PipelineClient,
                 parent=None):
        super().__init__(parent)
        self._modbus = modbus
        self._pipeline = pipeline
        self._state = self.S_IDLE

        # Pose bookkeeping
        self._pose_idx = 0           # which pose we're currently working on
        self._poses_deg = list(SCAN_POSES_DEG)
        self._poses_steps = list(SCAN_POSES_STEPS)

        # Accumulated cloud (Open3D + PyVista mirror)
        self._accum_o3d: "o3d.geometry.PointCloud | None" = None
        self._accum_pv: "pv.PolyData | None" = None

        # Timers for state transitions
        self._settle_timer = QTimer(self)
        self._settle_timer.setSingleShot(True)
        self._settle_timer.timeout.connect(self._on_settle_done)

        self._watchdog = QTimer(self)
        self._watchdog.setSingleShot(True)
        self._watchdog.timeout.connect(self._on_watchdog_timeout)

        # Hook up external signals
        self._modbus.status_update.connect(self._on_modbus_status)
        self._pipeline.capture_complete.connect(self._on_capture_complete)

        self._last_modbus_status: dict = {}

    # ── Public API ────────────────────────────────────────────────────────────
    def start(self):
        """Kick off a full 3-pose scan from the IDLE state."""
        if self._state != self.S_IDLE and self._state != self.S_DONE \
                and self._state != self.S_STOPPED:
            self.log_message.emit(
                f"[ORCH] start() rejected — already in {self._state}"
            )
            return

        self._accum_o3d = None
        self._accum_pv = None
        self._pose_idx = 0
        self.log_message.emit(
            f"[ORCH] Starting scan: {len(self._poses_deg)} poses "
            f"{self._poses_deg}"
        )
        self._start_next_pose()

    def stop(self, reason: str = "user requested"):
        """Abort whatever we're doing. Sends CCMD_STOP to the ClearCore."""
        if self._state in (self.S_IDLE, self.S_DONE, self.S_STOPPED):
            return
        self.log_message.emit(f"[ORCH] STOP: {reason}")
        self._modbus.send_command(CCMD_STOP)
        self._settle_timer.stop()
        self._watchdog.stop()
        self._state = self.S_STOPPED
        self.stage_change.emit("STOPPED")
        self.finished.emit(False, reason)

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state not in (self.S_IDLE, self.S_DONE, self.S_STOPPED)

    # ── State transitions ────────────────────────────────────────────────────
    def _start_next_pose(self):
        """Send CCMD_MOVE for the current pose index and enter MOVING."""
        if self._pose_idx >= len(self._poses_steps):
            # All poses done
            self._state = self.S_DONE
            self.stage_change.emit("COMPLETE")
            self.progress.emit(len(self._poses_steps), len(self._poses_steps))
            self.log_message.emit("[ORCH] Scan complete.")
            self.finished.emit(True, "ok")
            return

        target = self._poses_steps[self._pose_idx]
        deg = self._poses_deg[self._pose_idx]
        self.progress.emit(self._pose_idx, len(self._poses_steps))
        self.stage_change.emit(
            f"MOVE → POSE {self._pose_idx + 1}/{len(self._poses_steps)}"
        )
        self.log_message.emit(
            f"[ORCH] Pose {self._pose_idx + 1}/{len(self._poses_steps)}: "
            f"move to {deg:+.1f}° ({target} steps)"
        )

        self._state = self.S_MOVING
        self._watchdog.start(MOVE_TIMEOUT_MS)
        self._modbus.send_command(
            CCMD_MOVE,
            target_posn=target,
            velocity=SCAN_MOVE_VEL,
            accel=SCAN_MOVE_ACCEL,
        )

    def _on_modbus_status(self, status: dict):
        """Watch for MOVING edge-falling in the MOVING state."""
        prev = self._last_modbus_status
        self._last_modbus_status = status

        # E-stop or fault during scan → abort
        if self.is_running and (status.get("estop") or status.get("fault")):
            reason = "E-STOP" if status.get("estop") else "FAULT"
            self.stop(f"ClearCore {reason}")
            return

        if self._state != self.S_MOVING:
            return

        # We issued CCMD_MOVE. Wait for a rising edge into MOVING, then for
        # MOVING to fall back to 0 once the carriage settles on target.
        # If we never saw MOVING go high (very short move), the watchdog
        # will still clear us out when CCMD_MOVE completes synchronously.
        was_moving = prev.get("moving", False)
        now_moving = status.get("moving", False)
        if was_moving and not now_moving:
            self._watchdog.stop()
            self.log_message.emit(
                f"[ORCH] Arrived at pose {self._pose_idx + 1} "
                f"(cur_posn={status.get('cur_posn')})"
            )
            self._state = self.S_SETTLING
            self.stage_change.emit(f"SETTLING POSE {self._pose_idx + 1}")
            self._settle_timer.start(SETTLE_DWELL_MS)

    def _on_settle_done(self):
        if self._state != self.S_SETTLING:
            return
        self._state = self.S_CAPTURING
        self.stage_change.emit(f"CAPTURING POSE {self._pose_idx + 1}")
        self.log_message.emit(
            f"[ORCH] Capturing pose {self._pose_idx + 1}/"
            f"{len(self._poses_steps)} at "
            f"{self._poses_deg[self._pose_idx]:+.1f}°"
        )
        self._watchdog.start(CAPTURE_TIMEOUT_MS)
        self._pipeline.capture_pose(
            pose_deg=self._poses_deg[self._pose_idx],
            pose_idx=self._pose_idx,
        )

    def _on_capture_complete(self, msg: dict):
        """Server delivered a .ply for the pose we requested."""
        if self._state != self.S_CAPTURING:
            # Stale event from a previous run — drop it.
            return

        if msg.get("pose_idx") != self._pose_idx:
            self.log_message.emit(
                f"[ORCH] Ignoring stale capture "
                f"(got idx {msg.get('pose_idx')}, expected {self._pose_idx})"
            )
            return

        self._watchdog.stop()
        ply_path = msg.get("ply_path")
        n_points = msg.get("n_points", 0)
        self.log_message.emit(
            f"[ORCH] Pose {self._pose_idx + 1} capture: "
            f"{n_points} pts ← {ply_path}"
        )

        self._state = self.S_MERGING
        self.stage_change.emit(f"MERGING POSE {self._pose_idx + 1}")

        # Do the merge synchronously on the GUI thread. ICP on a decimated
        # D405 cloud is fast enough (<< 1s) that blocking the UI here is
        # fine; if it becomes an issue we can spin this out into a QThread.
        try:
            merged = self._merge_ply(ply_path)
        except Exception as e:
            self.log_message.emit(f"[ORCH] Merge failed: {e}")
            self.stop(f"merge failed: {e}")
            return

        if merged is not None:
            self.pose_merged.emit(merged)

        # Advance to next pose
        self._pose_idx += 1
        self._start_next_pose()

    def _on_watchdog_timeout(self):
        self.log_message.emit(
            f"[ORCH] WATCHDOG TIMEOUT in state {self._state} "
            f"at pose {self._pose_idx + 1}"
        )
        self.stop("watchdog timeout")

    # ── Point-cloud merge logic  ─────────────────────────────────────────────
    def _merge_ply(self, ply_path: str) -> "pv.PolyData | None":
        """Load a PLY, ICP-align to the accumulated cloud, merge, mirror
        into a pv.PolyData for the viewport. Returns the accumulated cloud
        after merge, or None if the input was empty.
        """
        if not _OPEN3D_AVAILABLE:
            raise RuntimeError("open3d not installed — can't ICP-merge")

        if not ply_path or not os.path.exists(ply_path):
            raise FileNotFoundError(f"PLY not found: {ply_path}")

        new_cloud = o3d.io.read_point_cloud(ply_path)
        if len(new_cloud.points) == 0:
            self.log_message.emit(f"[ORCH] Empty PLY: {ply_path}")
            return self._accum_pv

        # Light downsample to tame D405 density before ICP
        new_cloud = new_cloud.voxel_down_sample(voxel_size=0.002)  # 2 mm

        # Normals are required for point-to-plane ICP
        new_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.01, max_nn=30
            )
        )

        if self._accum_o3d is None:
            # First pose — seed the accumulator, no ICP needed
            self._accum_o3d = new_cloud
            self.log_message.emit(
                f"[ORCH] Seed pose: {len(new_cloud.points)} pts"
            )
        else:
            # Align new_cloud -> accum using point-to-plane ICP.
            # Initial guess is identity because the arc geometry keeps
            # cameras roughly co-located; if poses get wider apart you'll
            # want to seed this with the known arc transform.
            threshold = 0.01  # 1 cm correspondence distance
            reg = o3d.pipelines.registration.registration_icp(
                new_cloud, self._accum_o3d, threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=30
                ),
            )
            new_cloud.transform(reg.transformation)
            self._accum_o3d += new_cloud
            # Keep point count bounded — downsample after every merge
            self._accum_o3d = self._accum_o3d.voxel_down_sample(
                voxel_size=0.002
            )
            self.log_message.emit(
                f"[ORCH] ICP fit={reg.fitness:.3f} "
                f"rmse={reg.inlier_rmse:.4f}  "
                f"accum={len(self._accum_o3d.points)} pts"
            )

        # Mirror into a pv.PolyData for the viewport
        pts = np.asarray(self._accum_o3d.points, dtype=np.float32)
        pv_cloud = pv.PolyData(pts)
        if pts.shape[0] > 0:
            pv_cloud["depth"] = pts[:, 2].copy()
        self._accum_pv = pv_cloud
        return pv_cloud


# ══════════════════════════════════════════════════════════════════════════════
# Main Window
# ══════════════════════════════════════════════════════════════════════════════

def make_empty_pointcloud() -> pv.PolyData:
    cloud = pv.PolyData(np.empty((0, 3), dtype=np.float32))
    cloud["depth"] = np.empty((0,), dtype=np.float32)
    return cloud


class ScanToMillUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCAN-TO-MILL  //  Control Interface")
        self.setMinimumSize(1280, 820)
        self.showMaximized()

        self._point_count = 0
        self._modbus_last_status: dict = {}
        self._estop_active = False

        self._build_ui()
        self._start_clock()
        self._init_modbus()
        self._init_pipeline()
        self._init_orchestrator()

    # ── Hardware/service links ───────────────────────────────────────────────
    def _init_modbus(self):
        self._modbus = ClearCoreModbus(
            host=MODBUS_DEFAULT_HOST,
            tcp_port=MODBUS_DEFAULT_TCP_PORT,
            slave_id=MODBUS_DEFAULT_SLAVE_ID,
        )
        self._modbus.connected.connect(self._on_modbus_connected)
        self._modbus.status_update.connect(self._on_modbus_status)
        self._modbus.log_message.connect(self._log)
        self._modbus.error.connect(self._log)
        self._log(f"[MODBUS] Connecting to ClearCore at "
                  f"{MODBUS_DEFAULT_HOST}:{MODBUS_DEFAULT_TCP_PORT}...")
        self._modbus.start()

    def _init_pipeline(self):
        self._pipeline = PipelineClient(
            host=PIPELINE_DEFAULT_HOST, port=PIPELINE_DEFAULT_PORT,
        )
        self._pipeline.connected.connect(self._on_pipeline_connected)
        self._pipeline.log_message.connect(self._log)
        self._pipeline.error.connect(self._log)
        self._pipeline.status_update.connect(self._on_pipeline_status)
        self._log(f"[PIPE] Connecting to server at "
                  f"{PIPELINE_DEFAULT_HOST}:{PIPELINE_DEFAULT_PORT}...")
        self._pipeline.start()

    def _init_orchestrator(self):
        self._orch = ScanOrchestrator(self._modbus, self._pipeline)
        self._orch.progress.connect(self._on_orch_progress)
        self._orch.stage_change.connect(self._on_orch_stage)
        self._orch.log_message.connect(self._log)
        self._orch.pose_merged.connect(self._on_pose_merged)
        self._orch.finished.connect(self._on_orch_finished)

    # ── Modbus handlers ──────────────────────────────────────────────────────
    def _on_modbus_connected(self, ok: bool):
        if ok:
            self.lbl_cnc_status.setText("● CNC ONLINE")
            self.lbl_cnc_status.setObjectName("status_ok")
            self._log("[MODBUS] ClearCore link ESTABLISHED.")
        else:
            self.lbl_cnc_status.setText("● CNC OFFLINE")
            self.lbl_cnc_status.setObjectName("status_off")
            self._log("[MODBUS] ClearCore link DOWN.")
        # Force stylesheet re-apply since object names changed
        self.lbl_cnc_status.style().unpolish(self.lbl_cnc_status)
        self.lbl_cnc_status.style().polish(self.lbl_cnc_status)

    def _on_modbus_status(self, status: dict):
        prev = self._modbus_last_status
        flag_keys = ("ready", "moving", "homed", "fault", "estop", "scanning")
        changed = [k for k in flag_keys if prev.get(k) != status.get(k)]
        if changed:
            flags = " ".join(
                k.upper() for k in flag_keys if status.get(k)
            ) or "—"
            self._log(
                f"[MODBUS] STATUS: {flags}  pos={status.get('cur_posn', '?')}"
            )

        if status.get("fault") and not prev.get("fault"):
            self._log("[MODBUS] !! FAULT latched")

        # E-stop edge handling
        estop_now = status.get("estop", False)
        if estop_now and not self._estop_active:
            self._estop_active = True
            self._on_estop_engaged()
        elif not estop_now and self._estop_active:
            self._estop_active = False
            self._on_estop_cleared()

        self._modbus_last_status = status

    # ── Pipeline handlers ────────────────────────────────────────────────────
    def _on_pipeline_connected(self, ok: bool):
        if ok:
            self.lbl_camera_status.setText("● CAMERA ONLINE")
            self.lbl_camera_status.setObjectName("status_ok")
            self._log("[PIPE] Pipeline server link ESTABLISHED.")
            # Ask for current state so the log shows something useful
            self._pipeline.send_json({"cmd": "status"})
        else:
            self.lbl_camera_status.setText("● CAMERA OFFLINE")
            self.lbl_camera_status.setObjectName("status_off")
            self._log("[PIPE] Pipeline server link DOWN.")
        self.lbl_camera_status.style().unpolish(self.lbl_camera_status)
        self.lbl_camera_status.style().polish(self.lbl_camera_status)

    def _on_pipeline_status(self, msg: dict):
        state = msg.get("state", "?")
        stage = msg.get("stage_name", "")
        extra = f" ({stage})" if stage else ""
        self._log(f"[PIPE] server state = {state}{extra}")

    # ── Orchestrator handlers ────────────────────────────────────────────────
    def _on_orch_progress(self, idx: int, total: int):
        # idx is 0-based "currently working on pose idx"; treat completion as
        # proportional progress.
        pct = int(100 * idx / max(1, total))
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"POSE {idx + 1}/{total}  —  %p%")
        self.lbl_pose_stat.setText(f"{idx + 1}/{total}")

    def _on_orch_stage(self, stage: str):
        self.lbl_stage.setText(stage)
        self.progress_bar.setFormat(f"{stage}  —  %p%")

    def _on_pose_merged(self, cloud: pv.PolyData):
        """Update the viewport with the new accumulated cloud."""
        if cloud is None or cloud.n_points == 0:
            return
        # Swap the accumulated cloud in place so VTK re-uploads just the
        # vertex buffer — no actor churn, no flicker.
        if hasattr(self, "_cloud") and self._cloud is not None:
            # reconstruct_surface etc. expect a consistent schema
            self._cloud.copy_from(cloud)
        else:
            self._cloud = cloud
        self._point_count = self._cloud.n_points
        self.lbl_pts_stat.setText(str(self._point_count))
        if hasattr(self, "plotter"):
            self.plotter.render()

    def _on_orch_finished(self, success: bool, reason: str):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("COMPLETE")
            self.lbl_stage.setText("COMPLETE")
            self.btn_export.setEnabled(True)
        else:
            self.progress_bar.setFormat(f"STOPPED — {reason}")
            self.lbl_stage.setText("STOPPED")

    # ── E-stop UX ────────────────────────────────────────────────────────────
    def _on_estop_engaged(self):
        self._log("[SYS] !! EMERGENCY STOP ENGAGED !!")
        w = int(self.width() * 0.65)
        h = 120
        x = (self.width() - w) // 2
        y = (self.height() - h) // 2
        self.estop_banner.setGeometry(x, y, w, h)
        self.estop_banner.show()
        self.estop_banner.raise_()
        self._orch.stop("E-stop engaged")
        self.btn_start.setEnabled(False)

    def _on_estop_cleared(self):
        self._log("[SYS] E-stop released. Press START to resume.")
        self.estop_banner.hide()
        self.lbl_stage.setText("IDLE")
        self.progress_bar.setFormat("%p%  —  IDLE")
        # Delay re-enable by 500 ms so the ClearCore's serviceEstop() loop
        # sees the switch cleared before we re-enable drives.
        QTimer.singleShot(
            500, lambda: self._modbus.send_command(CCMD_ENAB_MTRS)
        )
        self.btn_start.setEnabled(True)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        root.addWidget(self._make_header())

        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self._make_left_panel(), stretch=0)
        body.addWidget(self._make_viewport_panel(), stretch=1)
        root.addLayout(body, stretch=1)

        root.addWidget(self._make_bottom_panel())
        self._make_estop_banner()

    def _make_header(self):
        w = QWidget()
        w.setFixedHeight(52)
        w.setStyleSheet("background:#080a0f; border-bottom:1px solid #1e2230;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(12, 4, 12, 4)

        title = QLabel("SCAN-TO-MILL")
        title.setObjectName("header")
        sub = QLabel("3D CAPTURE → CNC MACHINING SYSTEM  //  OSU CAPSTONE")
        sub.setObjectName("subheader")

        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addStretch()

        self.lbl_clock = QLabel("--:--:--")
        self.lbl_clock.setObjectName("value_display")
        self.lbl_camera_status = QLabel("● CAMERA OFFLINE")
        self.lbl_camera_status.setObjectName("status_off")
        self.lbl_cnc_status = QLabel("● CNC OFFLINE")
        self.lbl_cnc_status.setObjectName("status_off")

        for w2 in [self.lbl_camera_status, self.lbl_cnc_status, self.lbl_clock]:
            lay.addWidget(w2)
            lay.addSpacing(14)
        return w

    def _make_left_panel(self):
        w = QWidget()
        w.setFixedWidth(210)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        grp = QGroupBox("Scan Controls")
        g = QVBoxLayout(grp)

        self.btn_start = QPushButton("▶  START SCAN")
        self.btn_start.setObjectName("btn_start")
        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_view_reset = QPushButton("⌂  Reset Camera/Home")
        self.btn_clear_cloud = QPushButton("✕  Clear Cloud")

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_view_reset.clicked.connect(self._reset_camera)
        self.btn_clear_cloud.clicked.connect(self._clear_cloud)

        for b in [self.btn_start, self.btn_stop,
                  self.btn_view_reset, self.btn_clear_cloud]:
            g.addWidget(b)
        lay.addWidget(grp)

        grp2 = QGroupBox("Post-Process")
        g2 = QVBoxLayout(grp2)
        self.btn_export = QPushButton("Export STL")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export_stl)
        g2.addWidget(self.btn_export)
        lay.addWidget(grp2)

        lay.addStretch()
        return w

    def _make_viewport_panel(self):
        grp = QGroupBox("3D Viewport — Accumulated Cloud")
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(4, 12, 4, 4)

        toolbar = QHBoxLayout()
        self.btn_view_cloud = QPushButton("Point Cloud")
        self.btn_view_mesh = QPushButton("Mesh")
        self.btn_view_cnc_preview = QPushButton("CNC Preview")
        self.lbl_render_mode = QLabel("MODE: POINT CLOUD")
        self.lbl_render_mode.setObjectName("value_display")

        self.btn_view_cloud.clicked.connect(lambda: self._set_view_mode("cloud"))
        self.btn_view_mesh.clicked.connect(lambda: self._set_view_mode("mesh"))
        self.btn_view_cnc_preview.clicked.connect(self._on_cnc_preview)

        for w2 in [self.btn_view_cloud, self.btn_view_mesh,
                   self.btn_view_cnc_preview, self.lbl_render_mode]:
            toolbar.addWidget(w2)
        toolbar.addStretch()
        lay.addLayout(toolbar)

        self.vtk_frame = QFrame()
        self.vtk_frame.setStyleSheet(
            "background:#050709; border:1px solid #1e2230;"
        )
        self.vtk_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        lay.addWidget(self.vtk_frame, stretch=1)
        return grp

    def _make_bottom_panel(self):
        w = QWidget()
        w.setFixedHeight(160)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        prog_grp = QGroupBox("Job Progress")
        prog_lay = QVBoxLayout(prog_grp)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%  —  IDLE")
        prog_lay.addWidget(self.progress_bar)

        stats_row = QHBoxLayout()
        for label, attr in [
            ("Elapsed", "lbl_elapsed"),
            ("Pose", "lbl_pose_stat"),
            ("Points", "lbl_pts_stat"),
            ("Stage", "lbl_stage"),
        ]:
            col = QVBoxLayout()
            col.addWidget(QLabel(label))
            lbl = QLabel("—")
            lbl.setObjectName("value_display")
            setattr(self, attr, lbl)
            col.addWidget(lbl)
            stats_row.addLayout(col)
        prog_lay.addLayout(stats_row)
        lay.addWidget(prog_grp, stretch=1)

        log_grp = QGroupBox("System Log")
        log_lay = QVBoxLayout(log_grp)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_lay.addWidget(self.log_output)
        lay.addWidget(log_grp, stretch=1)

        self._log("[SYS] Scan-to-Mill UI initialized.")
        self._log(
            f"[SYS] Poses: {SCAN_POSES_DEG} "
            f"({STEPS_PER_DEG} steps/° → {SCAN_POSES_STEPS})"
        )
        self._log("[SYS] Waiting for hardware connection...")
        return w

    # ── Viewport ──────────────────────────────────────────────────────────────
    def _init_viewport(self):
        vtk_lay = QVBoxLayout(self.vtk_frame)
        vtk_lay.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self.vtk_frame)
        self.plotter.set_background("#050709")
        vtk_lay.addWidget(self.plotter.interactor)

        self._cloud = make_empty_pointcloud()
        self._cloud_actor = self.plotter.add_mesh(
            self._cloud,
            scalars="depth",
            cmap="cool",
            point_size=3,
            render_points_as_spheres=True,
            name="pointcloud",
            show_scalar_bar=False,
        )
        self._mesh_actor = None
        self.plotter.add_axes(color="#4a6a7a")
        self.plotter.camera_position = "iso"
        self._view_mode = "cloud"
        self._log("[VIZ] Viewport ready — awaiting first capture")

    def _set_view_mode(self, mode):
        self._view_mode = mode
        if mode == "cloud":
            self.lbl_render_mode.setText("MODE: POINT CLOUD")
            if self._mesh_actor is not None:
                self.plotter.remove_actor(self._mesh_actor)
                self._mesh_actor = None
            self._cloud_actor.SetVisibility(True)
        else:
            self.lbl_render_mode.setText("MODE: MESH")
            self._cloud_actor.SetVisibility(False)
            if self._cloud.n_points < 50:
                self._log("[VIZ] Not enough points to surface-reconstruct yet")
                self.plotter.render()
                return
            try:
                surf = self._cloud.reconstruct_surface(nbr_sz=10)
            except Exception as e:
                self._log(f"[VIZ] reconstruct_surface failed: {e}")
                self.plotter.render()
                return
            self._mesh_actor = self.plotter.add_mesh(
                surf, color="#2a5a7a", show_edges=False,
                opacity=1.0, name="mesh_actor",
                pbr=False, interpolate_before_map=False,
            )
        self.plotter.render()

    def _reset_camera(self):
        if not hasattr(self, "plotter"):
            return
        self.plotter.camera_position = "iso"
        self.plotter.render()

    def _clear_cloud(self):
        """Wipe the accumulated cloud and reset the viewport."""
        if hasattr(self, "_cloud") and self._cloud is not None:
            empty = make_empty_pointcloud()
            self._cloud.copy_from(empty)
        self._point_count = 0
        self.lbl_pts_stat.setText("0")
        # Tell the orchestrator to drop its accumulator too
        self._orch._accum_o3d = None
        self._orch._accum_pv = None
        if hasattr(self, "plotter"):
            self.plotter.render()
        self._log("[VIZ] Cloud cleared.")

    # ── Scan Actions ──────────────────────────────────────────────────────────
    def _on_start(self):
        if self._orch.is_running:
            self._log("[UI] Scan already in progress.")
            return

        self.progress_bar.setValue(0)
        self.lbl_elapsed.setText("00:00")
        self.lbl_pts_stat.setText("0")
        self.lbl_pose_stat.setText(f"0/{len(SCAN_POSES_DEG)}")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.progress_bar.setFormat("STARTING —  %p%")
        self.lbl_stage.setText("STARTING")

        self._scan_elapsed = 0
        if not hasattr(self, "_scan_timer") or self._scan_timer is None:
            self._scan_timer = QTimer(self)
            self._scan_timer.timeout.connect(self._tick_elapsed)
        self._scan_timer.start(1000)

        # Clear accumulated cloud before starting fresh
        self._clear_cloud()

        # Make sure drives are enabled before we try to move
        self._modbus.send_command(CCMD_ENAB_MTRS)

        # Kick off the orchestrator
        self._orch.start()
        self._log("[SYS] 3-pose scan started.")

    def _on_stop(self):
        self._modbus.send_command(CCMD_STOP)
        self._orch.stop("user pressed STOP")
        if hasattr(self, "_scan_timer") and self._scan_timer is not None:
            self._scan_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setFormat("%p%  —  IDLE")
        self.lbl_stage.setText("IDLE")

    def _tick_elapsed(self):
        self._scan_elapsed += 1
        m, s = divmod(self._scan_elapsed, 60)
        self.lbl_elapsed.setText(f"{m:02d}:{s:02d}")

    def _on_cnc_preview(self):
        self._log("[VIEW] CNC preview not yet implemented.")

    def _on_export_stl(self):
        """Surface-reconstruct the accumulated cloud and save an STL."""
        if not hasattr(self, "_cloud") or self._cloud.n_points < 100:
            self._log("[EXP] Not enough points in cloud to export.")
            return
        try:
            surf = self._cloud.reconstruct_surface(nbr_sz=10)
            out_path = os.path.expanduser("~/scan_output.stl")
            surf.save(out_path)
            self._log(f"[EXP] Saved STL → {out_path}")
        except Exception as e:
            self._log(f"[EXP] STL export failed: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}"
        self.log_output.append(line)
        self.log_output.moveCursor(QTextCursor.MoveOperation.End)

    def _start_clock(self):
        timer = QTimer(self)
        timer.timeout.connect(self._update_clock)
        timer.start(1000)
        self._update_clock()

    def _update_clock(self):
        from datetime import datetime
        self.lbl_clock.setText(datetime.now().strftime("%H:%M:%S"))

    def closeEvent(self, event):
        if hasattr(self, "_modbus") and self._modbus.isRunning():
            self._modbus.send_command(CCMD_STOP)
            self._modbus.send_command(CCMD_DISAB_MTRS)
            self._modbus.stop()
            self._modbus.wait(2000)
        if hasattr(self, "_pipeline") and self._pipeline.isRunning():
            self._pipeline.stop()
            self._pipeline.wait(2000)
        if hasattr(self, "plotter"):
            self.plotter.close()
        event.accept()

    def showEvent(self, event):
        super().showEvent(event)
        if not hasattr(self, "_viewport_initialized"):
            self._viewport_initialized = True
            QTimer.singleShot(0, self._init_viewport_safe)

    def _init_viewport_safe(self):
        try:
            self._init_viewport()
            self._log("[VIZ] _init_viewport completed successfully")
        except Exception as e:
            self._log(f"[VIZ] _init_viewport FAILED: {e}")
            import traceback
            traceback.print_exc()

    def _make_estop_banner(self):
        self.estop_banner = QLabel("⚠  EMERGENCY STOP ENGAGED  ⚠", self)
        self.estop_banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.estop_banner.setStyleSheet("""
            QLabel {
                background-color: #cc2d2d;
                color: #ffffff;
                font-size: 28px;
                font-weight: bold;
                letter-spacing: 4px;
                border: 4px solid #ff5555;
                padding: 24px;
            }
        """)
        self.estop_banner.hide()
        self.estop_banner.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "estop_banner"):
            w = int(self.width() * 0.6)
            h = 120
            x = (self.width() - w) // 2
            y = (self.height() - h) // 2
            self.estop_banner.setGeometry(x, y, w, h)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pv.set_plot_theme("dark")
    pv.global_theme.multi_samples = 1
    pv.global_theme.smooth_shading = False
    pv.global_theme.allow_empty_mesh = True
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = ScanToMillUI()
    window.show()
    sys.exit(app.exec())
