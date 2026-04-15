"""
arc controller -- serial command interface to ClearCore

communicates with the ClearCore motor controller over serial to
position the D405 camera along the scanning arc. uses a simple
ASCII protocol with request/response pairs.

protocol:
    Pi sends:   ARC:HOME\n        -> home the carriage
    Pi sends:   ARC:MOVE:45.0\n   -> move to absolute 45 degrees
    Pi sends:   ARC:STEP:15.0\n   -> relative move +15 degrees
    Pi sends:   ARC:POS?\n        -> query current position
    Pi sends:   ARC:STOP\n        -> emergency stop
    ClearCore:  ARC:OK\n          -> success
    ClearCore:  ARC:POS:45.0\n    -> position response
    ClearCore:  ARC:ERR:msg\n     -> error

usage:
    arc = ArcController("/dev/ttyUSB0")
    arc.connect()
    arc.home()
    arc.move_to(45.0)
    pos = arc.get_position()
    arc.disconnect()
"""

import serial
import logging
import time

logger = logging.getLogger(__name__)

ARC_PREFIX = "ARC:"
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT = 10.0
HOME_TIMEOUT = 30.0


class ArcError(Exception):
    """raised when the ClearCore returns an error or communication fails."""
    pass


class ArcController:
    def __init__(self, port: str, baud: int = DEFAULT_BAUD, timeout: float = DEFAULT_TIMEOUT):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self._position = 0.0

    def connect(self):
        """open serial connection to ClearCore."""
        logger.info(f"connecting to ClearCore on {self.port} at {self.baud} baud")
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            timeout=self.timeout,
            write_timeout=self.timeout
        )
        # wait for ClearCore to boot/reset after serial open
        time.sleep(2.0)
        # flush any startup messages
        self.ser.reset_input_buffer()
        logger.info("ClearCore connected")

    def disconnect(self):
        """close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("ClearCore disconnected")

    def _send(self, command: str, timeout: float = None) -> str:
        """
        send a command and wait for response.

        returns the response string with prefix stripped.
        raises ArcError on timeout, communication failure, or ARC:ERR response.
        """
        if not self.ser or not self.ser.is_open:
            raise ArcError("not connected to ClearCore")

        timeout = timeout or self.timeout
        original_timeout = self.ser.timeout
        self.ser.timeout = timeout

        try:
            cmd = f"{command}\n"
            logger.debug(f"TX: {command}")
            self.ser.write(cmd.encode("ascii"))
            self.ser.flush()

            raw = self.ser.readline()
            if not raw:
                raise ArcError(f"timeout waiting for response to: {command}")

            response = raw.decode("ascii").strip()
            logger.debug(f"RX: {response}")

            if not response.startswith(ARC_PREFIX):
                raise ArcError(f"unexpected response format: {response}")

            payload = response[len(ARC_PREFIX):]

            if payload.startswith("ERR:"):
                raise ArcError(f"ClearCore error: {payload[4:]}")

            return payload

        finally:
            self.ser.timeout = original_timeout

    # motion commands

    def home(self):
        """home the arc carriage. blocks until complete."""
        logger.info("homing arc carriage...")
        result = self._send("ARC:HOME", timeout=HOME_TIMEOUT)
        if result != "OK":
            raise ArcError(f"unexpected home response: {result}")
        self._position = 0.0
        logger.info("arc homed to 0.0 degrees")

    def move_to(self, degrees: float):
        """move carriage to absolute position in degrees."""
        logger.info(f"moving arc to {degrees:.1f} degrees")
        result = self._send(f"ARC:MOVE:{degrees:.1f}")
        if result != "OK":
            raise ArcError(f"unexpected move response: {result}")
        self._position = degrees
        logger.info(f"arc at {degrees:.1f} degrees")

    def step(self, degrees: float):
        """relative move by degrees (positive or negative)."""
        logger.info(f"stepping arc by {degrees:+.1f} degrees")
        result = self._send(f"ARC:STEP:{degrees:.1f}")
        if result != "OK":
            raise ArcError(f"unexpected step response: {result}")
        self._position += degrees
        logger.info(f"arc at {self._position:.1f} degrees")

    def stop(self):
        """emergency stop."""
        logger.warning("sending emergency stop to arc")
        try:
            result = self._send("ARC:STOP", timeout=2.0)
        except ArcError:
            logger.error("no response to emergency stop")
            raise

    def get_position(self) -> float:
        """query current arc position from ClearCore."""
        payload = self._send("ARC:POS?")
        if not payload.startswith("POS:"):
            raise ArcError(f"unexpected position response: {payload}")
        try:
            pos = float(payload[4:])
        except ValueError:
            raise ArcError(f"invalid position value: {payload[4:]}")
        self._position = pos
        return pos

    @property
    def position(self) -> float:
        """last known position (cached, does not query hardware)."""
        return self._position

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

    responds to all commands as if the ClearCore is connected,
    tracks position in software. use for pipeline development
    and debugging without the physical arc.

    usage:
        arc = MockArcController()
        arc.connect()
        arc.home()
        arc.move_to(45.0)
    """

    def __init__(self, move_delay: float = 0.5):
        super().__init__(port="MOCK", baud=0)
        self.move_delay = move_delay

    def connect(self):
        logger.info("[MOCK] arc controller connected (simulated)")

    def disconnect(self):
        logger.info("[MOCK] arc controller disconnected")

    def _send(self, command: str, timeout: float = None) -> str:
        logger.debug(f"[MOCK] TX: {command}")
        if "MOVE" in command or "STEP" in command or "HOME" in command:
            time.sleep(self.move_delay)
        return "OK"

    def home(self):
        logger.info("[MOCK] homing arc carriage...")
        self._send("ARC:HOME")
        self._position = 0.0
        logger.info("[MOCK] arc homed to 0.0 degrees")

    def move_to(self, degrees: float):
        logger.info(f"[MOCK] moving arc to {degrees:.1f} degrees")
        self._send(f"ARC:MOVE:{degrees:.1f}")
        self._position = degrees

    def step(self, degrees: float):
        logger.info(f"[MOCK] stepping arc by {degrees:+.1f} degrees")
        self._send(f"ARC:STEP:{degrees:.1f}")
        self._position += degrees

    def stop(self):
        logger.info("[MOCK] emergency stop (simulated)")

    def get_position(self) -> float:
        return self._position