"""
arc controller -- Modbus TCP interface to ClearCore

communicates with the ClearCore motor controller over Modbus TCP
(port 502) to position the D405 camera along the scanning arc.

the register map matches the CLIENT_INFC struct in
ClearCoreModbusTest.ino. 32-bit values are little-endian
(low word first) matching the ClearCore's ARM Cortex-M4.

register map:
    command channel (Pi writes):
        word 0-1   acc           uint32  steps/sec^2
        word 2-3   vel           int32   steps/sec
        word 4-5   target_posn   int32   steps, absolute
        word 6     cmd           uint16  command code

    status channel (ClearCore writes):
        word 7-8   cur_posn      int32   steps
        word 9     status        uint16  bitfield
        word 10    msg_cnt       uint16  debug message counter
        word 11-42 msg           char[64] debug text
        word 43    state         uint16  controller state

usage:
    arc = ArcController("192.168.1.20")
    arc.connect()
    arc.home()
    arc.move_to_steps(50000)
    pos = arc.get_position_steps()
    arc.disconnect()
"""

import logging
import time

logger = logging.getLogger(__name__)

# command codes (must match CCMTR_CMD in .ino)
CCMD_NONE = 0
CCMD_ENAB_MTRS = 1
CCMD_DISAB_MTRS = 2
CCMD_SET_ZERO = 3
CCMD_MOVE = 4
CCMD_STOP = 5
CCMD_RUN_1 = 6
CCMD_ACK = 7
CCMD_NACK = 8
CCMD_NEXT_POINT = 9

# controller states (must match CCMTR_STATE in .ino)
CST_INIT = 0
CST_IDLE = 1
CST_ENABLED = 2
CST_RUNNING = 3
CST_STOPPED = 4
CST_FAULT = 5
CST_UNKNOWN = 6

# status bit masks
STATUS_READY = 1 << 0
STATUS_MOVING = 1 << 1
STATUS_HOMED = 1 << 2
STATUS_FAULT = 1 << 3
STATUS_ESTOP = 1 << 4
STATUS_SCANNING = 1 << 5
STATUS_HLFB = 1 << 6
STATUS_AT_CAPTURE = 1 << 7

# register addresses (word offsets into CLIENT_INFC)
W_ACC = 0
W_VEL = 2
W_TARGET_POSN = 4
W_CMD = 6
W_CUR_POSN = 7
W_STATUS = 9
W_MSG_CNT = 10
W_MSG = 11
W_STATE = 43
W_MSG_LEN = 32

# default motion parameters
DEFAULT_VEL = 2000
DEFAULT_ACCEL = 20000
DEFAULT_HOST = "192.168.1.20"
DEFAULT_PORT = 502
DEFAULT_SLAVE_ID = 1


def pack_u32(value):
    """pack unsigned 32-bit into [low_word, high_word]."""
    v = value & 0xFFFFFFFF
    return [v & 0xFFFF, (v >> 16) & 0xFFFF]


def pack_i32(value):
    """pack signed 32-bit into [low_word, high_word]."""
    return pack_u32(value & 0xFFFFFFFF)


def unpack_i32(regs):
    """unpack [low_word, high_word] into signed 32-bit."""
    raw = ((regs[1] & 0xFFFF) << 16) | (regs[0] & 0xFFFF)
    if raw & 0x80000000:
        raw -= 0x100000000
    return raw


class ArcError(Exception):
    """raised when ClearCore communication fails or returns an error."""
    pass


class ArcController:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 slave_id: int = DEFAULT_SLAVE_ID,
                 velocity: int = DEFAULT_VEL, accel: int = DEFAULT_ACCEL):
        self.host = host
        self.port = port
        self.slave_id = slave_id
        self.velocity = velocity
        self.accel = accel
        self._client = None
        self._position_steps = 0

    def connect(self):
        """open Modbus TCP connection to ClearCore."""
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError:
            raise ImportError("pymodbus required: pip install pymodbus")

        logger.info(f"connecting to ClearCore at {self.host}:{self.port}")
        self._client = ModbusTcpClient(
            host=self.host,
            port=self.port,
            timeout=2.0,
        )
        if not self._client.connect():
            raise ArcError(f"failed to connect to ClearCore at {self.host}:{self.port}")
        logger.info("ClearCore connected via Modbus TCP")

    def disconnect(self):
        """close Modbus TCP connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            logger.info("ClearCore disconnected")

    def _write_cmd(self, cmd_code: int, target_posn: int = 0,
                   velocity: int = 0, accel: int = 0):
        """write a full command block (acc + vel + target + cmd) atomically."""
        if not self._client:
            raise ArcError("not connected to ClearCore")

        vel = velocity or self.velocity
        acc = accel or self.accel

        if cmd_code == CCMD_MOVE:
            values = (pack_u32(acc) +
                      pack_i32(vel) +
                      pack_i32(target_posn) +
                      [cmd_code])
            # pymodbus 3.7+ renamed the slave= kwarg to device_id=
            rsp = self._client.write_registers(W_ACC, values, device_id=self.slave_id)
        else:
            rsp = self._client.write_register(W_CMD, cmd_code, device_id=self.slave_id)

        if rsp.isError():
            raise ArcError(f"Modbus write error: {rsp}")

    def read_status(self) -> dict:
        """read the status block from ClearCore."""
        if not self._client:
            raise ArcError("not connected to ClearCore")

        rsp = self._client.read_holding_registers(
            W_CUR_POSN, count=4, device_id=self.slave_id
        )
        if rsp.isError():
            raise ArcError(f"Modbus read error: {rsp}")

        regs = rsp.registers
        cur_posn = unpack_i32([regs[0], regs[1]])
        status_word = regs[2] & 0xFFFF
        msg_cnt = regs[3] & 0xFFFF

        return {
            "cur_posn": cur_posn,
            "raw": status_word,
            "ready": bool(status_word & STATUS_READY),
            "moving": bool(status_word & STATUS_MOVING),
            "homed": bool(status_word & STATUS_HOMED),
            "fault": bool(status_word & STATUS_FAULT),
            "estop": bool(status_word & STATUS_ESTOP),
            "scanning": bool(status_word & STATUS_SCANNING),
            "at_capture": bool(status_word & STATUS_AT_CAPTURE),
            "msg_cnt": msg_cnt,
        }

    def read_state(self) -> int:
        """read the controller state register."""
        rsp = self._client.read_holding_registers(
            W_STATE, count=1, device_id=self.slave_id
        )
        if rsp.isError():
            return CST_UNKNOWN
        return rsp.registers[0]

    def wait_for_move(self, timeout: float = 30.0, poll_interval: float = 0.05) -> bool:
        """block until motion completes. returns False on timeout/fault."""
        t0 = time.time()

        # wait for moving bit to assert (motion started)
        grace_deadline = t0 + 0.5
        saw_motion = False
        while time.time() < grace_deadline:
            st = self.read_status()
            if st.get("fault") or st.get("estop"):
                return False
            if st.get("moving"):
                saw_motion = True
                break
            time.sleep(poll_interval)

        if not saw_motion:
            return True

        # wait for moving bit to de-assert (motion complete)
        while time.time() - t0 < timeout:
            st = self.read_status()
            if st.get("fault") or st.get("estop"):
                return False
            if not st.get("moving"):
                self._position_steps = st.get("cur_posn", self._position_steps)
                return True
            time.sleep(poll_interval)

        return False

    def wait_for_at_capture(self, timeout: float = 30.0, poll_interval: float = 0.05) -> bool:
        """block until ClearCore reports at_capture (settled, ready for scan)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = self.read_status()
            if st.get("fault") or st.get("estop"):
                return False
            if st.get("at_capture"):
                self._position_steps = st.get("cur_posn", self._position_steps)
                return True
            time.sleep(poll_interval)
        return False

    # motion commands

    def enable(self):
        """enable motor drive."""
        logger.info("enabling motors")
        self._write_cmd(CCMD_ENAB_MTRS)

    def disable(self):
        """disable motor drive."""
        logger.info("disabling motors")
        self._write_cmd(CCMD_DISAB_MTRS)

    def home(self):
        """start homing sequence (RUN_1). blocks until homed or fault."""
        logger.info("starting home sequence (RUN_1)")
        self._write_cmd(CCMD_RUN_1)

        # wait for homed bit
        t0 = time.time()
        while time.time() - t0 < 60.0:
            st = self.read_status()
            if st.get("fault") or st.get("estop"):
                raise ArcError("fault or estop during homing")
            if st.get("at_capture"):
                self._position_steps = 0
                logger.info("homing complete, at first capture point")
                return
            time.sleep(0.1)

        raise ArcError("homing timed out")

    def move_to_steps(self, target_steps: int, velocity: int = 0, accel: int = 0):
        """move to absolute step position. blocks until complete."""
        logger.info(f"moving to {target_steps} steps")
        self._write_cmd(CCMD_MOVE, target_posn=target_steps,
                        velocity=velocity, accel=accel)
        if not self.wait_for_move():
            raise ArcError(f"move to {target_steps} failed or timed out")
        self._position_steps = target_steps

    def next_point(self):
        """tell ClearCore to advance to the next scan point.
        used in the ClearCore-driven scan sequence (RUN_1).
        blocks until at_capture or fault."""
        logger.info("sending NEXT_POINT")
        self._write_cmd(CCMD_NEXT_POINT)
        if not self.wait_for_at_capture():
            raise ArcError("next_point failed or timed out")

    def stop(self):
        """emergency stop."""
        logger.warning("sending STOP to ClearCore")
        try:
            self._write_cmd(CCMD_STOP)
        except ArcError:
            logger.error("failed to send stop command")

    def get_position_steps(self) -> int:
        """query current position from ClearCore."""
        st = self.read_status()
        self._position_steps = st["cur_posn"]
        return self._position_steps

    @property
    def position(self) -> int:
        """last known position in steps (cached)."""
        return self._position_steps

    # context manager

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


class MockArcController(ArcController):
    """
    simulated arc controller for testing without hardware.

    responds to all commands as if ClearCore is connected.
    tracks position in software.

    usage:
        arc = MockArcController()
        arc.connect()
        arc.home()
        arc.move_to_steps(50000)
    """

    def __init__(self, move_delay: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.move_delay = move_delay
        self._mock_status = {
            "cur_posn": 0,
            "raw": 0,
            "ready": True,
            "moving": False,
            "homed": False,
            "fault": False,
            "estop": False,
            "scanning": False,
            "at_capture": False,
            "msg_cnt": 0,
        }

    def connect(self):
        logger.info("[MOCK] arc controller connected (simulated)")

    def disconnect(self):
        logger.info("[MOCK] arc controller disconnected")

    def _write_cmd(self, cmd_code, target_posn=0, velocity=0, accel=0):
        logger.debug(f"[MOCK] cmd={cmd_code} target={target_posn}")

    def read_status(self) -> dict:
        return dict(self._mock_status)

    def read_state(self) -> int:
        return CST_ENABLED

    def wait_for_move(self, timeout=30.0, poll_interval=0.05) -> bool:
        time.sleep(self.move_delay)
        return True

    def wait_for_at_capture(self, timeout=30.0, poll_interval=0.05) -> bool:
        time.sleep(self.move_delay)
        self._mock_status["at_capture"] = True
        return True

    def enable(self):
        logger.info("[MOCK] motors enabled")
        self._mock_status["ready"] = True

    def disable(self):
        logger.info("[MOCK] motors disabled")
        self._mock_status["ready"] = False

    def home(self):
        logger.info("[MOCK] homing...")
        time.sleep(self.move_delay)
        self._position_steps = 0
        self._mock_status["cur_posn"] = 0
        self._mock_status["homed"] = True
        self._mock_status["at_capture"] = True
        logger.info("[MOCK] homed")

    def move_to_steps(self, target_steps, velocity=0, accel=0):
        logger.info(f"[MOCK] moving to {target_steps} steps")
        time.sleep(self.move_delay)
        self._position_steps = target_steps
        self._mock_status["cur_posn"] = target_steps

    def next_point(self):
        logger.info("[MOCK] next point")
        time.sleep(self.move_delay)
        self._mock_status["at_capture"] = True

    def stop(self):
        logger.info("[MOCK] stop")

    def get_position_steps(self) -> int:
        return self._position_steps