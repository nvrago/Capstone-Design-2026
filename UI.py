"""
Scan-to-Mill UI
Presentation wireframe — dummy data, no hardware required.
Requires: PyQt6, pyvistaqt, pyvista, numpy
"""

import sys
import queue
import numpy as np
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"           # <-- add
os.environ["VTK_DISABLE_OPENGL_RENDER_TIMER"] = "1" # <-- add
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
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QPushButton, QLabel, QProgressBar, QTextEdit,
        QGroupBox, QSlider, QSizePolicy, QFrame, QComboBox, QSpinBox,
        QDoubleSpinBox
    )
    from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
    from PyQt6.QtGui import QFont, QColor, QPalette, QTextCursor
except ModuleNotFoundError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QPushButton, QLabel, QProgressBar, QTextEdit,
        QGroupBox, QSlider, QSizePolicy, QFrame, QComboBox, QSpinBox,
        QDoubleSpinBox
    )
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor

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
QPushButton:pressed {
    background-color: #0d1520;
}
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

QLabel {
    color: #8a9aaa;
}
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
QLabel#status_ok {
    color: #2dcc70;
    font-weight: bold;
}
QLabel#status_off {
    color: #cc4d2d;
    font-weight: bold;
}

QSlider::groove:horizontal {
    background: #1a1e28;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #2d8fa8;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}
QSlider::sub-page:horizontal {
    background: #2d8fa8;
    border-radius: 2px;
}

QComboBox {
    background-color: #1a1e28;
    border: 1px solid #2a2f3d;
    border-radius: 3px;
    padding: 4px 8px;
    color: #c8cdd6;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #1a1e28;
    border: 1px solid #2a2f3d;
    selection-background-color: #252a38;
}

QSpinBox, QDoubleSpinBox {
    background-color: #1a1e28;
    border: 1px solid #2a2f3d;
    border-radius: 3px;
    padding: 4px 6px;
    color: #c8cdd6;
}

QFrame#divider {
    color: #2a2f3d;
}
"""


# ── Empty point cloud helper ──────────────────────────────────────────────────
def make_empty_pointcloud() -> pv.PolyData:
    """Return an empty point cloud placeholder.

    Real point data will be pushed in from the depth camera pipeline once
    the capture backend is wired up.
    """
    cloud = pv.PolyData(np.empty((0, 3), dtype=np.float32))
    cloud["depth"] = np.empty((0,), dtype=np.float32)
    return cloud
# ── Pipeline server (server.py) TCP/JSON client ───────────────────────────────
#
# The Pi also runs a separate process (server.py) that manages the multi-stage
# scan-to-mill pipeline: capture, register, mesh, toolpath, etc. It speaks
# line-delimited JSON over TCP on localhost:5001.
#
# This is independent of the Modbus link to the ClearCore. The Modbus link
# drives the motor directly (Start Scan -> CCMD_RUN_1). The pipeline server
# orchestrates the higher-level workflow. For now they're parallel channels;
# you can use either or both from the UI.
#
# Protocol reference (see server.py docstring for the full list):
#   UI -> server:  {"cmd": "scan"}  {"cmd": "zero"}  {"cmd": "stop"}
#                  {"cmd": "status"}  {"cmd": "stage", "stage": N, "end": M}
#                  {"cmd": "config", "key": "...", "value": ...}
#   server -> UI:  {"type": "status",  "state": "idle"|"running"|"complete"|...}
#                  {"type": "log",     "level": "...", "message": "..."}
#                  {"type": "error",   "message": "..."}

PIPELINE_DEFAULT_HOST = "127.0.0.1"
PIPELINE_DEFAULT_PORT = 5001
PIPELINE_RECONNECT_SEC = 3.0   # backoff when server.py isn't running


class PipelineClient(QThread):
    """Background worker owning the TCP/JSON link to server.py.

    Emits Qt signals for each message category. Main thread calls
    send_cmd() / scan() / etc. to issue commands.
    """

    connected      = pyqtSignal(bool)   # True on open, False on close/error
    pipeline_state = pyqtSignal(dict)   # {"state": "running", "stage": 2, ...}
    log_message    = pyqtSignal(str)    # log lines forwarded from the server
    error          = pyqtSignal(str)

    def __init__(self, host: str = PIPELINE_DEFAULT_HOST,
                 port: int = PIPELINE_DEFAULT_PORT, parent=None):
        super().__init__(parent)
        import socket as _socket
        self._socket_mod = _socket
        self._host = host
        self._port = port
        self._sock = None
        self._cmd_queue: "queue.Queue[dict]" = queue.Queue()
        self._stop_flag = False
        self._connected = False
        self._send_lock = None   # created in run() on the worker thread

    # ── Public API (main thread) ──────────────────────────────────────────────
    def send_cmd(self, **cmd):
        """Queue a command for the server. Non-blocking.

        Usage: self._pipeline.send_cmd(cmd="scan", dry_run=True)
        """
        self._cmd_queue.put(cmd)

    def scan(self, dry_run: bool = False, skip_execute: bool = True):
        self.send_cmd(cmd="scan", dry_run=dry_run, skip_execute=skip_execute)

    def zero(self):
        self.send_cmd(cmd="zero")

    def run_stages(self, start: int, end: int, dry_run: bool = False):
        self.send_cmd(cmd="stage", stage=start, end=end, dry_run=dry_run)

    def stop_pipeline(self):
        self.send_cmd(cmd="stop")

    def request_status(self):
        self.send_cmd(cmd="status")

    def set_config(self, key: str, value):
        self.send_cmd(cmd="config", key=key, value=value)

    def stop(self):
        """Shut down the worker thread (called from the main thread)."""
        self._stop_flag = True

    # ── Worker thread ─────────────────────────────────────────────────────────
    def run(self):
        import threading as _threading
        self._send_lock = _threading.Lock()

        while not self._stop_flag:
            if not self._open_socket():
                # backoff, stay responsive to stop
                for _ in range(int(PIPELINE_RECONNECT_SEC * 10)):
                    if self._stop_flag:
                        return
                    self.msleep(100)
                continue

            self._service_connection()
            self._close_socket()

        self._close_socket()

    def _open_socket(self) -> bool:
        try:
            s = self._socket_mod.socket(self._socket_mod.AF_INET,
                                        self._socket_mod.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((self._host, self._port))
            s.settimeout(0.2)   # short read timeout so we can check stop flag
            self._sock = s
            self._connected = True
            self.connected.emit(True)
            self.log_message.emit(
                f"[PIPE] Connected to server.py at {self._host}:{self._port}"
            )
            return True
        except (OSError, ConnectionRefusedError) as e:
            # Server not running yet — quiet retry. Only log on first failure.
            if self._connected:
                self.error.emit(f"[PIPE] Connection lost: {e}")
            self._connected = False
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

    def _service_connection(self):
        """Interleave command draining and line-delimited JSON reads.

        Uses a short socket recv() timeout so the loop stays responsive to
        the stop flag and doesn't starve the command queue.
        """
        import json as _json
        buffer = ""
        while not self._stop_flag:
            # 1. Drain outbound commands
            self._drain_command_queue()

            # 2. Read any inbound data
            try:
                chunk = self._sock.recv(4096)
                if not chunk:
                    # server closed connection
                    return
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = _json.loads(line)
                    except _json.JSONDecodeError:
                        self.error.emit(f"[PIPE] Bad JSON: {line!r}")
                        continue
                    self._dispatch(msg)
            except self._socket_mod.timeout:
                # normal - just means no data in the last 200ms
                pass
            except (OSError, ConnectionResetError, BrokenPipeError) as e:
                self.error.emit(f"[PIPE] Read error: {e}")
                return

    def _drain_command_queue(self):
        import json as _json
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                return
            try:
                data = (_json.dumps(cmd) + "\n").encode("utf-8")
                with self._send_lock:
                    self._sock.sendall(data)
                self.log_message.emit(f"[PIPE] -> {cmd.get('cmd', '?')}")
            except (OSError, BrokenPipeError) as e:
                self.error.emit(f"[PIPE] Write error: {e}")
                # Drop the socket; outer loop will reconnect
                try:
                    self._sock.close()
                except Exception:
                    pass
                self._sock = None
                return

    def _dispatch(self, msg: dict):
        """Route a parsed JSON message to the appropriate Qt signal."""
        mtype = msg.get("type", "")
        if mtype == "status":
            self.pipeline_state.emit(msg)
        elif mtype == "log":
            level = msg.get("level", "info").upper()
            text = msg.get("message", "")
            self.log_message.emit(f"[PIPE/{level}] {text}")
        elif mtype == "error":
            self.error.emit(f"[PIPE] {msg.get('message', '')}")
        elif mtype == "config_updated":
            key = msg.get("key")
            val = msg.get("value")
            self.log_message.emit(f"[PIPE] config {key} = {val}")
        else:
            # Unknown message type — log it so we don't silently drop things
            self.log_message.emit(f"[PIPE] <{mtype}> {msg}")

# ── ClearCore Modbus TCP Link ─────────────────────────────────────────────────
#
# Ethernet link between the Raspberry Pi (Modbus master / client) and the
# ClearCore PLC (Modbus slave / server) over TCP/IP, port 502. The register
# map mirrors the Teknic "CLIENT_INFC" struct in shared.h from their Modbus
# HMI reference — see ClearCoreModbusTest.ino for the canonical definition.
#
# Wiring: Pi eth0 ──direct cable or switch──> ClearCore Ethernet port.
# Give both sides a static IP on the same /24. Default here assumes
# Pi = 192.168.1.10, ClearCore = 192.168.1.20.
#
# REGISTER MAP (must match CLIENT_INFC in the .ino):
#
#   Command channel — Pi writes, ClearCore reads:
#     word 0-1   acc           uint32_t  steps/sec^2
#     word 2-3   vel           int32_t   steps/sec (magnitude)
#     word 4-5   target_posn   int32_t   steps, absolute
#     word 6     cmd           uint16_t  CCMD_*
#
#   Status channel — ClearCore writes, Pi reads:
#     word 7-8   cur_posn      int32_t   steps
#     word 9     status        uint16_t  CCMTR_STATUS bitfield
#     word 10    msg_cnt       uint16_t  bumped on each new debug msg
#     word 11-42 msg           char[64]  ClearCore-side debug text
#     word 43    state         uint16_t  CCMTR_STATE
#
# Endianness:
#   ClearCore is little-endian ARM Cortex-M4, and its Modbus handler reads
#   the struct via reinterpret_cast<uint16_t*>. 32-bit values therefore
#   cross the wire LOW-WORD-FIRST. All 32-bit values in this module go
#   through pack_i32() / unpack_i32() to stay consistent with that.
#
# ──────────────────────────────────────────────────────────────────────────────

# Default TCP settings — override in ScanToMillUI if needed.
# CHANGE MODBUS_DEFAULT_HOST to match the static IP configured on the
# ClearCore's Ethernet interface.
MODBUS_DEFAULT_HOST = "192.168.1.20"
MODBUS_DEFAULT_TCP_PORT = 502
MODBUS_DEFAULT_SLAVE_ID = 1
MODBUS_POLL_INTERVAL_MS = 200  # how often to poll the status block

# ── Command enum — mirrors CCMTR_CMD in the .ino ──────────────────────────────
CCMD_NONE       = 0
CCMD_ENAB_MTRS  = 1   # enable motor drive
CCMD_DISAB_MTRS = 2   # disable motor drive (used for software E-stop)
CCMD_SET_ZERO   = 3   # zero current position (not implemented in test)
CCMD_MOVE       = 4   # use acc/vel/target_posn for an absolute move
CCMD_STOP       = 5   # abrupt stop, drive stays enabled
CCMD_RUN_1      = 6   # start/restart sequence 1 (back-and-forth)
CCMD_ACK        = 7
CCMD_NACK       = 8

# ── Controller state enum — mirrors CCMTR_STATE in the .ino ──────────────────
CST_INIT    = 0
CST_IDLE    = 1
CST_ENABLED = 2
CST_RUNNING = 3
CST_STOPPED = 4
CST_FAULT   = 5
CST_UNKNOWN = 6

# ── Register offsets (word addresses into CLIENT_INFC) ────────────────────────
# These must exactly match the struct layout in the .ino.
W_ACC         = 0     # 2 words
W_VEL         = 2     # 2 words
W_TARGET_POSN = 4     # 2 words
W_CMD         = 6     # 1 word
W_CUR_POSN    = 7     # 2 words
W_STATUS      = 9     # 1 word
W_MSG_CNT     = 10    # 1 word
W_MSG         = 11    # 32 words (64 bytes)
W_STATE       = 43    # 1 word
W_MSG_LEN     = 32    # msg[] buffer length in registers

# Length of the command-channel write block (words 0..6 inclusive)
CMD_BLOCK_LEN = 7
# Length of the status-block burst read (cur_posn..msg_cnt = words 7..10)
STATUS_BLOCK_START = W_CUR_POSN
STATUS_BLOCK_LEN   = 4   # cur_posn(2) + status(1) + msg_cnt(1)

# ── Status bit masks (CCMTR_STATUS bitfield in the .ino) ─────────────────────
STATUS_READY    = 1 << 0   # drvs_enabled
STATUS_MOVING   = 1 << 1
STATUS_HOMED    = 1 << 2
STATUS_FAULT    = 1 << 3
STATUS_ESTOP    = 1 << 4
STATUS_SCANNING = 1 << 5
STATUS_HLFB     = 1 << 6


# ── 32-bit pack/unpack helpers ────────────────────────────────────────────────
# ClearCore stores 32-bit fields natively (little-endian), so when the
# Modbus handler reads them as two uint16_t the LOW word comes first.
def pack_u32(value: int) -> list:
    """Pack an unsigned 32-bit int into [low_word, high_word]."""
    v = value & 0xFFFFFFFF
    return [v & 0xFFFF, (v >> 16) & 0xFFFF]

def pack_i32(value: int) -> list:
    """Pack a signed 32-bit int into [low_word, high_word]."""
    return pack_u32(value & 0xFFFFFFFF)

def unpack_i32(regs) -> int:
    """Unpack [low_word, high_word] into a signed 32-bit int."""
    raw = ((regs[1] & 0xFFFF) << 16) | (regs[0] & 0xFFFF)
    if raw & 0x80000000:
        raw -= 0x100000000
    return raw


class ClearCoreModbus(QThread):
    """Background worker that owns the Modbus TCP link to the ClearCore.

    The main thread should only interact via send_command() and the emitted
    signals — never touch self._client directly, it's not thread-safe.
    """

    connected     = pyqtSignal(bool)      # True on open, False on close/error
    status_update = pyqtSignal(dict)      # parsed status block
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
        self._client: "ModbusTcpClient | None" = None
        self._cmd_queue: "queue.Queue[tuple]" = queue.Queue()
        self._stop_flag = False
        self._connected = False
        self._last_msg_cnt = 0   # tracks ClearCore debug msg counter

    # ── Public API (call from main thread) ────────────────────────────────────
    def send_command(self, cmd_code: int, target_posn: int = 0,
                     velocity: int = 0, accel: int = 0):
        """Queue a command for the ClearCore. Non-blocking.

        For CCMD_MOVE, fill in target_posn (steps), velocity (steps/sec),
        and accel (steps/sec^2). For simple commands like CCMD_STOP or
        CCMD_RUN_1, the extra args are ignored.
        """
        self._cmd_queue.put(("cmd", cmd_code, target_posn, velocity, accel))

    def write_holding(self, address: int, value: int):
        """Queue a raw single-register write. Use sparingly."""
        self._cmd_queue.put(("wr", address, value))

    def stop(self):
        self._stop_flag = True

    # ── Worker loop ───────────────────────────────────────────────────────────
    def run(self):
        if not _PYMODBUS_AVAILABLE:
            self.error.emit(
                "pymodbus not installed — run: pip install pymodbus"
            )
            return

        if not self._open_port():
            return

        self.log_message.emit(
            f"[MODBUS] Link up to ClearCore at {self._host}:{self._tcp_port}, "
            f"slave id {self._slave_id}"
        )

        while not self._stop_flag:
            # 1. Drain any pending writes from the main thread
            self._drain_command_queue()
            if self._stop_flag:
                break

            # 2. Poll the status block
            self._poll_status()

            # 3. Sleep until next poll (but stay responsive to stop/commands)
            self.msleep(MODBUS_POLL_INTERVAL_MS)

        self._close_port()
        self.log_message.emit("[MODBUS] Link closed.")

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _open_port(self) -> bool:
        try:
            self._client = ModbusTcpClient(
                host=self._host,
                port=self._tcp_port,
                timeout=2.0,
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
        """Process every pending command in the queue.

        For a full command (CCMD_MOVE), writes acc/vel/target_posn/cmd in
        one atomic 7-word write starting at word 0, so the ClearCore sees a
        coherent argument set when handleCommand() dispatches. For simple
        commands like CCMD_STOP or CCMD_RUN_1, we just write the cmd word.
        """
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
                        # Full command block: acc, vel, target_posn, cmd
                        values = (pack_u32(accel) +       # word 0-1
                                  pack_i32(velocity) +    # word 2-3
                                  pack_i32(target_posn) + # word 4-5
                                  [cmd_code])             # word 6
                        rsp = self._client.write_registers(
                            W_ACC, values, device_id=self._slave_id
                        )
                        label = (f"MOVE target={target_posn} "
                                 f"vel={velocity} acc={accel}")
                    else:
                        # Simple command: just the cmd word
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
        """Read cur_posn / status / msg_cnt every tick, and pull the msg
        buffer only when msg_cnt changes (saves bandwidth on a hot loop).
        """
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
            # regs = [cur_posn_lo, cur_posn_hi, status, msg_cnt]
            regs = rsp.registers
            cur_posn = unpack_i32([regs[0], regs[1]])
            status_word = regs[2] & 0xFFFF
            msg_cnt = regs[3] & 0xFFFF

            parsed = {
                "raw":          status_word,
                "ready":        bool(status_word & STATUS_READY),
                "moving":       bool(status_word & STATUS_MOVING),
                "homed":        bool(status_word & STATUS_HOMED),
                "fault":        bool(status_word & STATUS_FAULT),
                "estop":        bool(status_word & STATUS_ESTOP),
                "scanning":     bool(status_word & STATUS_SCANNING),
                "hlfb":         bool(status_word & STATUS_HLFB),
                "cur_posn":     cur_posn,
                "msg_cnt":      msg_cnt,
            }
            self.status_update.emit(parsed)

            # If the ClearCore has a new debug message, burst-read it
            if msg_cnt != self._last_msg_cnt:
                self._last_msg_cnt = msg_cnt
                self._pull_msg_buffer()

        except ModbusException as e:
            self.error.emit(f"[MODBUS] Read exception: {e}")
        except Exception as e:
            self.error.emit(f"[MODBUS] Unexpected read error: {e}")

    def _pull_msg_buffer(self):
        """Read the ClearCore's 64-byte msg[] buffer and emit it as a log
        line. Called only when msg_cnt ticks."""
        try:
            rsp = self._client.read_holding_registers(
                W_MSG, count=W_MSG_LEN, device_id=self._slave_id
            )
            if rsp.isError():
                return
            # Each register is 2 bytes, low byte of the native uint16 is
            # the first char in memory on little-endian ARM. We packed the
            # bytes by reinterpret_cast on the ClearCore side, so for each
            # register the low 8 bits = first char, high 8 bits = second.
            buf = bytearray()
            for r in rsp.registers:
                buf.append(r & 0xFF)
                buf.append((r >> 8) & 0xFF)
            # Trim at first NUL
            nul = buf.find(0)
            if nul >= 0:
                buf = buf[:nul]
            text = buf.decode("ascii", errors="replace").strip()
            if text:
                self.log_message.emit(f"[CC] {text}")
        except Exception as e:
            self.error.emit(f"[MODBUS] msg read err: {e}")

# ── Stepped scan worker ──────────────────────────────────────────────────────
#
# Drives the carriage to a list of angular stops, waiting at each one for
# the camera to capture a frame. The angles are converted to absolute
# step positions using STEPS_PER_DEGREE. Motion-complete detection polls
# the ClearCore's MOVING status bit via the shared last-status dict on
# the main window.

# Fill in after measuring. See the calc above.
STEPS_PER_REV = 6400
STEPS_PER_DEGREE = STEPS_PER_REV / 360   # direct drive: motor shaft = arc pivot

# Velocity/accel for stepped moves (slower = cleaner captures, less ringing)
STEP_VEL_SPS   = 2000    # matches SCAN_VEL_SPS on the ClearCore
STEP_ACCEL_SPSPS = 20000


class SteppedScanWorker(QThread):
    """Drive the carriage to a list of angles, pausing for a capture at each.

    The worker runs on its own thread. It sends CCMD_MOVE via the shared
    ClearCoreModbus client, then polls `get_status_fn()` until the motion
    completes. At each stop it emits capture_requested and waits for
    capture_complete_event to be set by the main thread (or the pipeline).
    """

    capture_requested = pyqtSignal(int, float, int)   # (index, angle_deg, target_steps)
    progress          = pyqtSignal(int, int)          # (completed, total)
    log_message       = pyqtSignal(str)
    finished_ok       = pyqtSignal()
    finished_err      = pyqtSignal(str)

    # Timing knobs
    MOVE_START_GRACE_MS   = 300    # how long to wait for MOVING bit to go high
    MOVE_POLL_INTERVAL_MS = 50     # how often to check motion-complete
    MOVE_TIMEOUT_SEC      = 30     # hard cap on any single move
    SETTLE_MS             = 300    # post-move dwell before capture
    CAPTURE_TIMEOUT_SEC   = 10     # how long to wait for main thread to signal done

    def __init__(self, modbus, get_status_fn, angles_deg,
                 steps_per_degree: float = STEPS_PER_DEGREE,
                 return_home_on_finish: bool = True, parent=None):
        super().__init__(parent)
        self._modbus = modbus
        self._get_status = get_status_fn        # callable returning last status dict
        self._angles = list(angles_deg)
        self._spd = float(steps_per_degree)
        self._return_home = return_home_on_finish
        self._stopped = False

        self.capture_complete_event = threading.Event()

    def stop(self):
        self._stopped = True
        self.capture_complete_event.set()   # unblock any waiting capture

    def run(self):
        try:
            total = len(self._angles)
            self.log_message.emit(
                f"[STEP] Starting stepped scan: {total} stops at "
                f"{self._angles} deg ({self._spd:.2f} steps/deg)"
            )

            for idx, angle in enumerate(self._angles):
                if self._stopped:
                    self.log_message.emit("[STEP] Aborted by user.")
                    return

                target_steps = int(round(angle * self._spd))
                self.log_message.emit(
                    f"[STEP] Stop {idx + 1}/{total}: moving to "
                    f"{angle} deg ({target_steps} steps)"
                )

                # 1. Command the move
                self._modbus.send_command(
                    CCMD_MOVE,
                    target_posn=target_steps,
                    velocity=STEP_VEL_SPS,
                    accel=STEP_ACCEL_SPSPS,
                )

                # 2. Wait for motion to start (MOVING bit goes high) and then
                #    finish (MOVING bit goes low), with a hard timeout.
                if not self._wait_for_move_complete():
                    self.finished_err.emit(
                        f"[STEP] Move to {angle} deg timed out or faulted"
                    )
                    return

                if self._stopped:
                    return

                # 3. Settle time before capture
                self.msleep(self.SETTLE_MS)

                # 4. Request a capture and wait for the main thread to complete it
                self.capture_complete_event.clear()
                self.capture_requested.emit(idx, angle, target_steps)
                self.log_message.emit(
                    f"[STEP] At {angle} deg — awaiting capture..."
                )
                got_it = self.capture_complete_event.wait(
                    timeout=self.CAPTURE_TIMEOUT_SEC
                )
                if self._stopped:
                    return
                if not got_it:
                    self.log_message.emit(
                        f"[STEP] Capture at {angle} deg timed out — continuing"
                    )
                else:
                    self.log_message.emit(f"[STEP] Capture {idx + 1} done.")

                self.progress.emit(idx + 1, total)

            # Optional return to home
            if self._return_home and not self._stopped:
                self.log_message.emit("[STEP] Returning to home (0 deg).")
                self._modbus.send_command(
                    CCMD_MOVE, target_posn=0,
                    velocity=STEP_VEL_SPS, accel=STEP_ACCEL_SPSPS,
                )
                self._wait_for_move_complete()

            self.finished_ok.emit()

        except Exception as e:
            self.finished_err.emit(f"[STEP] Worker crashed: {e}")

    # ── helpers ──────────────────────────────────────────────────────────
    def _wait_for_move_complete(self) -> bool:
        """Block until the ClearCore reports motion complete. Returns False
        on timeout, fault, e-stop, or user abort."""
        import time as _t
        t0 = _t.monotonic()

        # First phase: wait for moving bit to assert (motion actually started).
        # If the bit never asserts within the grace window, assume the move
        # was effectively instantaneous (already at target) and move on.
        phase1_deadline = t0 + (self.MOVE_START_GRACE_MS / 1000.0)
        saw_motion = False
        while _t.monotonic() < phase1_deadline:
            if self._stopped:
                return False
            st = self._get_status() or {}
            if st.get("fault") or st.get("estop"):
                return False
            if st.get("moving"):
                saw_motion = True
                break
            self.msleep(self.MOVE_POLL_INTERVAL_MS)

        if not saw_motion:
            # Target was already the current position — that's fine.
            return True

        # Second phase: wait for moving bit to de-assert (motion complete).
        while True:
            if self._stopped:
                return False
            if _t.monotonic() - t0 > self.MOVE_TIMEOUT_SEC:
                return False
            st = self._get_status() or {}
            if st.get("fault") or st.get("estop"):
                return False
            if not st.get("moving", False):
                return True
            self.msleep(self.MOVE_POLL_INTERVAL_MS)
            
# ── Worker thread for fake scan progress ─────────────────────────────────────
class ScanWorker(QThread):
    progress = pyqtSignal(int)
    point_count = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._paused = False
        self._stopped = False
        
    def stop(self): self._stopped = True; self._paused = False

    def run(self):
        steps = 100
        self.log_message.emit("[SCAN] Initializing depth sensor...")
        self.msleep(400)
        self.log_message.emit("[SCAN] Stream started at 640x480 @ 30fps")
        for i in range(1, steps + 1):
            while self._paused:
                self.msleep(100)
            if self._stopped:
                self.log_message.emit("[SCAN] Scan aborted by user.")
                return
            self.msleep(80)
            self.progress.emit(i)
            self.point_count.emit(i * 47)
            if i == 25:
                self.log_message.emit("[SCAN] 25% — front face captured")
            elif i == 50:
                self.log_message.emit("[SCAN] 50% — rotating to side profile")
            elif i == 75:
                self.log_message.emit("[SCAN] 75% — top surface acquired")
            elif i == 100:
                self.log_message.emit("[SCAN] Complete — 4700 points captured")
        self.finished.emit()


# ── Main Window ───────────────────────────────────────────────────────────────
class ScanToMillUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCAN-TO-MILL  //  Control Interface")
        self.setMinimumSize(1280, 820)
        self.showMaximized()
        self._scan_worker = None
        self._scan_running = False
        self._scan_phase = "idle"
        self._point_count = 0
        self._build_ui()
        self._start_clock()
        self._init_modbus()
        self._init_pipeline()

    # ── Pipeline server link ──────────────────────────────────────────────────
    def _init_pipeline(self):
        self._pipeline = PipelineClient(
            host=PIPELINE_DEFAULT_HOST,
            port=PIPELINE_DEFAULT_PORT,
        )
        self._pipeline.connected.connect(self._on_pipeline_connected)
        self._pipeline.pipeline_state.connect(self._on_pipeline_state)
        self._pipeline.log_message.connect(self._log)
        self._pipeline.error.connect(self._log)
        self._pipeline_last_state = {}
        self._log(f"[PIPE] Connecting to server.py at "
                  f"{PIPELINE_DEFAULT_HOST}:{PIPELINE_DEFAULT_PORT}...")
        self._pipeline.start()

    def _on_pipeline_connected(self, ok: bool):
        if ok:
            self._log("[PIPE] server.py link ESTABLISHED.")
            self.lbl_camera_status.setText("● CAMERA ONLINE")
            self.lbl_camera_status.setObjectName("status_ok")
            # Qt needs a style refresh to pick up the new objectName
            self.lbl_camera_status.style().unpolish(self.lbl_camera_status)
            self.lbl_camera_status.style().polish(self.lbl_camera_status)
            # Ask for current state so UI reflects reality on first connect
            self._pipeline.request_status()
        else:
            self._log("[PIPE] server.py link DOWN.")
            self.lbl_camera_status.setText("● CAMERA OFFLINE")
            self.lbl_camera_status.setObjectName("status_off")
            self.lbl_camera_status.style().unpolish(self.lbl_camera_status)
            self.lbl_camera_status.style().polish(self.lbl_camera_status)

    def _on_pipeline_state(self, state: dict):
        prev = self._pipeline_last_state
        new_state = state.get("state")
        old_state = prev.get("state")
        if new_state != old_state:
            self._log(f"[PIPE] STATE: {new_state}")

        stage = state.get("stage")
        stage_name = state.get("stage_name")
        if stage is not None and stage != prev.get("stage"):
            self._log(f"[PIPE] Stage {stage}: {stage_name or '?'}")

        # On pipeline completion, unlock post-process if we want that behavior
        if new_state == "complete" and old_state != "complete":
            run_dir = state.get("run_dir")
            if run_dir:
                self._log(f"[PIPE] Run complete: {run_dir}")
            self.btn_export.setEnabled(True)
        elif new_state == "error":
            self._log(f"[PIPE] !! ERROR: {state.get('message', '(no detail)')}")

        self._pipeline_last_state = state


    def _transition_to_stepped_scan(self):
        """Called on the HOMED edge. Halts the firmware's own sweep and
        launches the Python-side stepped worker."""
        self._scan_phase = "scanning"

        # Halt whatever the firmware is doing (it was about to drive toward the
        # far limit as part of its built-in sequence)
        self._modbus.send_command(CCMD_STOP)

        # Update progress bar display
        self.progress_bar.setFormat("%p%  —  SCANNING")
        self.lbl_stage.setText("SCANNING")
        self.progress_bar.setValue(0)

        # Launch the stepped worker
        angles = [0, 45, 90, 135, 180]
        self._stepped_worker = SteppedScanWorker(
            modbus=self._modbus,
            get_status_fn=lambda: self._modbus_last_status,
            angles_deg=angles,
        )
        self._stepped_worker.capture_requested.connect(self._on_capture_requested)
        self._stepped_worker.progress.connect(self._on_stepped_progress)
        self._stepped_worker.log_message.connect(self._log)
        self._stepped_worker.finished_ok.connect(self._on_stepped_done)
        self._stepped_worker.finished_err.connect(self._on_stepped_err)
        self._stepped_worker.start()
    # ── Modbus TCP link to ClearCore ──────────────────────────────────────────
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
        self._modbus_last_status = {}
        self._log(f"[MODBUS] Connecting to ClearCore at "
                  f"{MODBUS_DEFAULT_HOST}:{MODBUS_DEFAULT_TCP_PORT}...")
        self._modbus.start()

    def _on_modbus_connected(self, ok: bool):
        if ok:
            self._log("[MODBUS] ClearCore link ESTABLISHED.")
        else:
            self._log("[MODBUS] ClearCore link DOWN.")

    def _on_modbus_status(self, status: dict):
        # Log only on change to avoid flooding the console
        prev = self._modbus_last_status
        flag_keys = ("ready", "moving", "homed", "fault", "estop", "scanning")
        changed = [k for k in flag_keys if prev.get(k) != status.get(k)]
        if changed:
            flags = " ".join(
                k.upper() for k in flag_keys if status.get(k)
            ) or "—"
            self._log(f"[MODBUS] STATUS: {flags}  pos={status.get('cur_posn', '?')}")

        if status.get("fault") and not prev.get("fault"):
            self._log("[MODBUS] !! FAULT latched")

        # Detect home-complete during the HOMING phase of a stepped scan
        if (self._scan_phase == "homing"
            and status.get("homed")
            and not prev.get("homed")):
            self._log("[SYS] Homing complete — starting stepped scan.")
            self._transition_to_stepped_scan()
        # E-stop edge handling
        estop_now = status.get("estop", False)
        if estop_now and not self._estop_active:
            self._estop_active = True
            self._on_estop_engaged()
        elif not estop_now and self._estop_active:
            self._estop_active = False
            self._on_estop_cleared()

        self._modbus_last_status = status

    def _on_estop_engaged(self):
        self._scan_phase = "idle"
        self._log("[SYS] !! EMERGENCY STOP ENGAGED !!")
        # Force geometry in case resizeEvent hasn't sized it yet
        w = int(self.width() * 0.6)
        h = 120
        x = (self.width() - w) // 2
        y = (self.height() - h) // 2
        self.estop_banner.setGeometry(x, y, w, h)
        self.estop_banner.show()
        self.estop_banner.raise_()
        # Also halt any local scan worker
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.stop()
        self._reset_scan_ui()
        self.progress_bar.setFormat("%p%  —  E-STOP")
        self.lbl_stage.setText("E-STOP")
        self.btn_start.setEnabled(False)

    def _on_estop_cleared(self):
        self._log("[SYS] E-stop released. System disarmed — press START to resume.")
        self.estop_banner.hide()
        self.lbl_stage.setText("IDLE")
        self.progress_bar.setFormat("%p%  —  IDLE")
        # Delay the enable command by 500ms (2-3 poll cycles) so the ClearCore's
        # serviceEstop() loop has time to register the switch as cleared before
        # handleCommand() runs. Without this, the command arrives while the ClearCore
        # still sees estopEngaged()==true and silently NACKs it, leaving motionState
        # stuck at MS_DISABLED.
        QTimer.singleShot(500, lambda: self._modbus.send_command(CCMD_ENAB_MTRS))
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
        self._make_estop_banner()   # <-- add this line

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

        # ── Scan Controls ──
        grp = QGroupBox("Scan Controls")
        g = QVBoxLayout(grp)

        self.btn_start = QPushButton("▶  START SCAN")
        self.btn_start.setObjectName("btn_start")
        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_view_reset = QPushButton("⌂  Reset Camera/Home")

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_view_reset.clicked.connect(self._reset_camera)

        for b in [self.btn_start, self.btn_stop, self.btn_view_reset]:
            g.addWidget(b)

        lay.addWidget(grp)

        # ── Post-Processing ──
        grp2 = QGroupBox("Post-Process")
        g2 = QVBoxLayout(grp2)
        self.btn_export = QPushButton("Export STL")
        self.btn_export.setEnabled(False)
        g2.addWidget(self.btn_export)
        lay.addWidget(grp2)

        lay.addStretch()
        return w

    def _make_viewport_panel(self):
        grp = QGroupBox("3D Viewport")
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

        # PyVista interactor placeholder (populated in _init_viewport)
        self.vtk_frame = QFrame()
        self.vtk_frame.setStyleSheet("background:#050709; border:1px solid #1e2230;")
        self.vtk_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.vtk_frame, stretch=1)

        return grp

    def _make_bottom_panel(self):
        w = QWidget()
        w.setFixedHeight(160)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        # Progress section
        prog_grp = QGroupBox("Job Progress")
        prog_lay = QVBoxLayout(prog_grp)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%  —  IDLE")
        prog_lay.addWidget(self.progress_bar)

        stats_row = QHBoxLayout()
        for label, attr in [("Elapsed", "lbl_elapsed"), ("Est. Remaining", "lbl_remaining"),
                             ("Points", "lbl_pts_stat"), ("Stage", "lbl_stage")]:
            col = QVBoxLayout()
            col.addWidget(QLabel(label))
            lbl = QLabel("—")
            lbl.setObjectName("value_display")
            setattr(self, attr, lbl)
            col.addWidget(lbl)
            stats_row.addLayout(col)
        prog_lay.addLayout(stats_row)
        lay.addWidget(prog_grp, stretch=1)

        # Log section
        log_grp = QGroupBox("System Log")
        log_lay = QVBoxLayout(log_grp)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_lay.addWidget(self.log_output)
        lay.addWidget(log_grp, stretch=1)

        self._log("[SYS] Scan-to-Mill UI initialized.")
        self._log("[SYS] Waiting for hardware connection...")
        return w

    # ── Viewport ──────────────────────────────────────────────────────────────
    def _init_viewport(self):

        vtk_lay = QVBoxLayout(self.vtk_frame)
        vtk_lay.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self.vtk_frame)
        self.plotter.set_background("#050709")
        vtk_lay.addWidget(self.plotter.interactor)

        # Empty point cloud placeholder — real data will come from the camera
        self._cloud = make_empty_pointcloud()
        self.plotter.add_axes(color="#4a6a7a")
        self.plotter.camera_position = "iso"
        self._view_mode = "cloud"
        self._log("[VIZ] Viewport ready — awaiting camera stream")

    def _set_view_mode(self, mode):
        if not hasattr(self, "plotter"):
            return  # Viewport not initialized yet
        self._view_mode = mode
        if mode == "cloud":
            self.lbl_render_mode.setText("MODE: POINT CLOUD")
            self.plotter.remove_actor("mesh_actor")
            self.plotter.add_mesh(
                self._cloud, scalars="depth", cmap="cool",
                point_size=3, render_points_as_spheres=True,
                name="pointcloud", show_scalar_bar=False
            )    
        else:
            self.lbl_render_mode.setText("MODE: MESH")
            self.plotter.remove_actor("pointcloud")
            if self._cloud is None or self._cloud.n_points < 4:
                # Need at least a few points for surface reconstruction.
                self._log("[VIZ] Mesh view: not enough points yet.")
                self.plotter.render()
                return
            surf = self._cloud.reconstruct_surface(nbr_sz=10)
            self.plotter.add_mesh(
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

    def _update_pointcloud_live(self, cloud: pv.PolyData):
        """Push a new point cloud frame into the viewport.

        TODO: call this from the camera capture thread with a real
        pv.PolyData built from D405 depth frames.
        """
        if not hasattr(self, "plotter"):
            return
        if cloud is None or cloud.n_points == 0:
            return
        self._cloud = cloud
        self.plotter.remove_actor("pointcloud")
        self.plotter.add_mesh(
            self._cloud, scalars="depth", cmap="cool",
            point_size=3, render_points_as_spheres=True,
            name="pointcloud", show_scalar_bar=False
        )
        self.plotter.render()

    # ── Scan Actions ──────────────────────────────────────────────────────────
    def _on_start(self):
        if hasattr(self, "_stepped_worker") and self._stepped_worker.isRunning():
            self._log("[STEP] Already running.")
            return

        self._log("[SYS] Scan started — homing first, then stepped scan.")

        # Phase: HOMING
        self._scan_phase = "homing"
        self._scan_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)

        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("HOMING...")   # no % while homing
        self.lbl_stage.setText("HOMING")
        self.lbl_elapsed.setText("00:00")
        self.lbl_remaining.setText("—")
        self.lbl_pts_stat.setText("—")

        # Start elapsed timer now (homing time counts as part of the scan)
        self._scan_elapsed = 0
        self._scan_timer = QTimer()
        self._scan_timer.timeout.connect(self._tick_elapsed)
        self._scan_timer.start(1000)

        # Fire the firmware's home→scan→return; we'll interrupt it once HOMED flips high
        self._modbus.send_command(CCMD_RUN_1)

    def _on_stepped_scan(self):
        """Run a stepped scan: 0, 45, 90, 135, 180 degrees, pausing at each."""
        if hasattr(self, "_stepped_worker") and self._stepped_worker.isRunning():
            self._log("[STEP] Already running.")
            return

        angles = [0, 45, 90, 135, 180]
        self._log(f"[STEP] Stepped scan requested: {angles} deg")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setFormat("%p%  —  STEPPED SCAN")
        self.lbl_stage.setText("STEPPED")
        self.progress_bar.setValue(0)

        self._stepped_worker = SteppedScanWorker(
            modbus=self._modbus,
            get_status_fn=lambda: self._modbus_last_status,
            angles_deg=angles,
        )
        self._stepped_worker.capture_requested.connect(self._on_capture_requested)
        self._stepped_worker.progress.connect(self._on_stepped_progress)
        self._stepped_worker.log_message.connect(self._log)
        self._stepped_worker.finished_ok.connect(self._on_stepped_done)
        self._stepped_worker.finished_err.connect(self._on_stepped_err)
        self._stepped_worker.start()

    def _on_capture_requested(self, idx: int, angle: float, target_steps: int):
        """Called on the main thread when the worker reaches a capture point."""
        self._log(f"[CAPTURE] #{idx + 1} at {angle} deg (pos={target_steps} steps)")
        # TODO: trigger camera. For now, simulate with a 500ms fake capture so
        # you can see the full loop work before the camera is wired in.
        QTimer.singleShot(500, lambda: self._stepped_worker.capture_complete_event.set())

        # Eventually, this is where you'd fire a pipeline command:
        # self._pipeline.send_cmd(cmd="capture_frame", index=idx, angle_deg=angle)
        # and the server.py response (or the D405 capture callback) would set
        # the event.

    def _on_stepped_progress(self, done: int, total: int):
        pct = int(round(100 * done / total))
        self.progress_bar.setValue(pct)

    def _on_stepped_done(self):
        self._scan_phase = "complete"
        self._log("[STEP] Stepped scan complete.")
        self._reset_scan_ui()
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("COMPLETE")
        self.lbl_stage.setText("COMPLETE")
        self.btn_export.setEnabled(True)

    def _on_stepped_err(self, msg: str):
        self._scan_phase = "idle"
        self._log(msg)
        self._reset_scan_ui()
        self.progress_bar.setFormat("%p%  —  ERROR")
        self.lbl_stage.setText("ERROR")
        
    def _on_stop(self):
        self._scan_phase = "idle"
        if self._scan_worker:
            self._scan_worker.stop()
        if hasattr(self, "_stepped_worker") and self._stepped_worker.isRunning():
            self._stepped_worker.stop()
        self._modbus.send_command(CCMD_STOP)
        self._reset_scan_ui()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%  —  IDLE")
        self.lbl_stage.setText("IDLE")

    def _on_scan_done(self):
        self._reset_scan_ui()
        self.progress_bar.setFormat("100%  —  COMPLETE")
        self.lbl_stage.setText("COMPLETE")
        self.btn_export.setEnabled(True)
        self._log("[SYS] Post-processing options unlocked.")

    def _reset_scan_ui(self):
        self._scan_running = False
        if hasattr(self, "_scan_timer"):
            self._scan_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_progress(self, val: int):
        self.progress_bar.setValue(val)
        remaining = int((100 - val) * 0.08)
        self.lbl_remaining.setText(f"{remaining}s")

    def _on_points(self, n: int):
        self._point_count = n
        self.lbl_pts_stat.setText(str(n))
        # Viewport updates will be driven by the camera pipeline, not the
        # progress counter. Hook real frames in via _update_pointcloud_live().

    def _tick_elapsed(self):
        self._scan_elapsed += 1
        m, s = divmod(self._scan_elapsed, 60)
        self.lbl_elapsed.setText(f"{m:02d}:{s:02d}")
        
    def _on_cnc_preview(self):
        """ Placeholder — will render the CNC toolpath preview in the viewport."""
        self._log("[VIEW] CNC preview not yet implemented.")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}"
        self.log_output.append(line)
        self.log_output.moveCursor(QTextCursor.MoveOperation.End)

    def _labeled_combo(self, label: str, items: list) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)
        lay.addWidget(QLabel(label))
        cb = QComboBox()
        cb.addItems(items)
        lay.addWidget(cb)
        return w

    def _start_clock(self):
        timer = QTimer(self)
        timer.timeout.connect(self._update_clock)
        timer.start(1000)
        self._update_clock()

    def _update_clock(self):
        from datetime import datetime
        self.lbl_clock.setText(datetime.now().strftime("%H:%M:%S"))

    def closeEvent(self, event):
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.stop()
            self._scan_worker.wait()
        if hasattr(self, "_modbus") and self._modbus.isRunning():
            # Park the ClearCore before tearing down the link
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
        self._estop_active = False
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the banner centered & sized to ~60% width whenever window resizes
        if hasattr(self, "estop_banner"):
            w = int(self.width() * 0.6)
            h = 120
            x = (self.width() - w) // 2
            y = (self.height() - h) // 2
            self.estop_banner.setGeometry(x, y, w, h)
            

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pv.set_plot_theme("dark")
    pv.global_theme.multi_samples = 0   # was 1 — must be 0 on Pi 5 V3D/llvmpipe
    pv.global_theme.smooth_shading = False
    pv.global_theme.allow_empty_mesh = True   # placeholder cloud has 0 points
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = ScanToMillUI()
    window.show()
    sys.exit(app.exec())
