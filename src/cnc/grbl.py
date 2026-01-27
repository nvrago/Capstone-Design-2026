"""
GRBL 1.1 Serial Communication Module

Handles connection, command sending, status polling, and error handling
for GRBL-based CNC controllers.
"""

import serial
import time
import re
import logging
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GrblState(Enum):
    IDLE = "Idle"
    RUN = "Run"
    HOLD = "Hold"
    JOG = "Jog"
    ALARM = "Alarm"
    DOOR = "Door"
    CHECK = "Check"
    HOME = "Home"
    SLEEP = "Sleep"
    UNKNOWN = "Unknown"


@dataclass
class MachineStatus:
    state: GrblState
    position: Tuple[float, float, float]  # x, y, z in mm
    feed_rate: float
    buffer_available: int = 0
    
    
class GrblError(Exception):
    """Raised when GRBL returns an error."""
    pass


class GrblAlarm(Exception):
    """Raised when GRBL enters alarm state."""
    pass


class GrblController:
    """Interface for GRBL 1.1 CNC controller."""
    
    # GRBL response patterns
    OK_PATTERN = re.compile(r'^ok$')
    ERROR_PATTERN = re.compile(r'^error:(\d+)$')
    ALARM_PATTERN = re.compile(r'^ALARM:(\d+)$')
    STATUS_PATTERN = re.compile(
        r'<(\w+)\|MPos:([-\d.]+),([-\d.]+),([-\d.]+)\|.*?>'
    )
    
    def __init__(self, port: str, baud_rate: int = 115200, timeout: float = 2.0):
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial: Optional[serial.Serial] = None
        self._connected = False
        
    def connect(self) -> bool:
        """Establish serial connection to GRBL controller."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            # Wait for GRBL to initialize
            time.sleep(2.0)
            # Clear startup messages
            self.serial.flushInput()
            # Send wake-up
            self.serial.write(b'\r\n\r\n')
            time.sleep(0.5)
            self.serial.flushInput()
            
            self._connected = True
            logger.info(f"Connected to GRBL on {self.port}")
            return True
            
        except serial.SerialException as e:
            logger.error(f"Failed to connect: {e}")
            return False
            
    def disconnect(self):
        """Close serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
        self._connected = False
        logger.info("Disconnected from GRBL")
        
    @property
    def is_connected(self) -> bool:
        return self._connected and self.serial and self.serial.is_open
        
    def send(self, command: str, wait_for_ok: bool = True) -> str:
        """
        Send a command to GRBL and optionally wait for response.
        
        Args:
            command: G-code or GRBL command (without newline)
            wait_for_ok: Block until 'ok' or 'error' received
            
        Returns:
            Response string from GRBL
            
        Raises:
            GrblError: If GRBL returns an error
            GrblAlarm: If GRBL enters alarm state
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to GRBL")
            
        cmd = command.strip() + '\n'
        self.serial.write(cmd.encode())
        logger.debug(f"Sent: {command}")
        
        if not wait_for_ok:
            return ""
            
        response_lines = []
        while True:
            line = self.serial.readline().decode().strip()
            if not line:
                continue
                
            logger.debug(f"Received: {line}")
            
            if self.OK_PATTERN.match(line):
                return '\n'.join(response_lines)
                
            error_match = self.ERROR_PATTERN.match(line)
            if error_match:
                raise GrblError(f"GRBL error {error_match.group(1)}: {command}")
                
            alarm_match = self.ALARM_PATTERN.match(line)
            if alarm_match:
                raise GrblAlarm(f"GRBL alarm {alarm_match.group(1)}")
                
            response_lines.append(line)
            
    def status(self) -> MachineStatus:
        """Query current machine status."""
        if not self.is_connected:
            raise ConnectionError("Not connected to GRBL")
            
        self.serial.write(b'?')
        line = self.serial.readline().decode().strip()
        
        match = self.STATUS_PATTERN.match(line)
        if match:
            state_str = match.group(1)
            try:
                state = GrblState(state_str)
            except ValueError:
                state = GrblState.UNKNOWN
                
            return MachineStatus(
                state=state,
                position=(
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4))
                ),
                feed_rate=0.0  # Could parse from status if needed
            )
        else:
            logger.warning(f"Could not parse status: {line}")
            return MachineStatus(
                state=GrblState.UNKNOWN,
                position=(0.0, 0.0, 0.0),
                feed_rate=0.0
            )
            
    def wait_idle(self, timeout: float = 60.0, poll_interval: float = 0.1):
        """Block until machine reaches idle state."""
        start = time.time()
        while time.time() - start < timeout:
            st = self.status()
            if st.state == GrblState.IDLE:
                return
            if st.state == GrblState.ALARM:
                raise GrblAlarm("Machine in alarm state")
            time.sleep(poll_interval)
        raise TimeoutError("Timed out waiting for idle")
        
    # Convenience methods
    def home(self):
        """Run homing cycle."""
        self.send('$H')
        self.wait_idle()
        
    def unlock(self):
        """Clear alarm lock."""
        self.send('$X')
        
    def reset(self):
        """Soft reset GRBL."""
        self.serial.write(b'\x18')  # Ctrl-X
        time.sleep(1.0)
        self.serial.flushInput()
        
    def move_to(self, x: float = None, y: float = None, z: float = None, 
                feed: float = None, rapid: bool = False):
        """
        Move to position.
        
        Args:
            x, y, z: Target coordinates (None to keep current)
            feed: Feed rate in mm/min (required for G1)
            rapid: Use G0 rapid move instead of G1
        """
        cmd = 'G0' if rapid else 'G1'
        if x is not None:
            cmd += f' X{x:.3f}'
        if y is not None:
            cmd += f' Y{y:.3f}'
        if z is not None:
            cmd += f' Z{z:.3f}'
        if not rapid and feed is not None:
            cmd += f' F{feed:.1f}'
        self.send(cmd)
        
    def get_settings(self) -> Dict[str, str]:
        """Query all GRBL $$ settings."""
        response = self.send('$$')
        settings = {}
        for line in response.split('\n'):
            if line.startswith('$'):
                parts = line.split('=')
                if len(parts) == 2:
                    settings[parts[0]] = parts[1]
        return settings