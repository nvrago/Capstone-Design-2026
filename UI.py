"""
Scan-to-Mill UI
Presentation wireframe — dummy data, no hardware required.
Requires: PyQt6, pyvistaqt, pyvista, numpy
"""

import sys
import queue
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
        self._point_count = 0
        self._build_ui()
        self._start_clock()
        self._init_modbus()

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
        self._scan_running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.progress_bar.setFormat("%p%  —  SCANNING")
        self.lbl_stage.setText("SCANNING")
        self._scan_elapsed = 0
        self._scan_timer = QTimer()
        self._scan_timer.timeout.connect(self._tick_elapsed)
        self._scan_timer.start(1000)

        self._scan_worker = ScanWorker()
        self._scan_worker.progress.connect(self._on_progress)
        self._scan_worker.point_count.connect(self._on_points)
        self._scan_worker.log_message.connect(self._log)
        self._scan_worker.finished.connect(self._on_scan_done)
        self._scan_worker.start()

        # Tell the ClearCore to begin its scan motion profile
        self._modbus.send_command(CCMD_RUN_1)


    def _on_stop(self):
        if self._scan_worker:
            self._scan_worker.stop()
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
    pv.global_theme.multi_samples = 1        # <-- add this
    pv.global_theme.smooth_shading = False
    pv.global_theme.allow_empty_mesh = True   # placeholder cloud has 0 points
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = ScanToMillUI()
    window.show()
    sys.exit(app.exec())
