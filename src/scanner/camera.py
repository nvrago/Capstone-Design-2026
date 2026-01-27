"""
Pi Camera Capture Module

Handles image acquisition from Pi Camera for structured light scanning.
"""

import numpy as np
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Conditional import - allows testing on non-Pi systems
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    logger.warning("picamera2 not available - camera functions will be simulated")


@dataclass
class CameraConfig:
    width: int = 1920
    height: int = 1080
    exposure_time: Optional[int] = None  # microseconds, None for auto


class Camera:
    """Pi Camera interface for scan capture."""
    
    def __init__(self, config: CameraConfig = None):
        self.config = config or CameraConfig()
        self.camera = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize camera hardware."""
        if not PICAMERA_AVAILABLE:
            logger.warning("Running in simulation mode - no real camera")
            self._initialized = True
            return True
            
        try:
            self.camera = Picamera2()
            
            # Configure for still capture
            cam_config = self.camera.create_still_configuration(
                main={"size": (self.config.width, self.config.height)},
                buffer_count=2
            )
            self.camera.configure(cam_config)
            
            # Set exposure if specified
            if self.config.exposure_time:
                self.camera.set_controls({
                    "ExposureTime": self.config.exposure_time,
                    "AeEnable": False
                })
                
            self.camera.start()
            self._initialized = True
            logger.info(f"Camera initialized: {self.config.width}x{self.config.height}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
            
    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a single frame.
        
        Returns:
            BGR image as numpy array, or None on failure
        """
        if not self._initialized:
            logger.error("Camera not initialized")
            return None
            
        if not PICAMERA_AVAILABLE:
            # Return simulated noise image for testing
            return np.random.randint(
                0, 255, 
                (self.config.height, self.config.width, 3), 
                dtype=np.uint8
            )
            
        try:
            # Capture returns RGB, convert to BGR for OpenCV compatibility
            rgb = self.camera.capture_array()
            bgr = rgb[:, :, ::-1].copy()
            return bgr
            
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None
            
    def capture_sequence(self, count: int, delay_ms: float = 100) -> list:
        """
        Capture multiple frames with delay between.
        
        Useful for averaging to reduce noise.
        """
        import time
        frames = []
        for i in range(count):
            frame = self.capture()
            if frame is not None:
                frames.append(frame)
            if i < count - 1:
                time.sleep(delay_ms / 1000.0)
        return frames
        
    def set_exposure(self, microseconds: int):
        """Adjust exposure time."""
        if self.camera and PICAMERA_AVAILABLE:
            self.camera.set_controls({
                "ExposureTime": microseconds,
                "AeEnable": False
            })
            self.config.exposure_time = microseconds
            
    def set_auto_exposure(self, enabled: bool = True):
        """Enable or disable auto exposure."""
        if self.camera and PICAMERA_AVAILABLE:
            self.camera.set_controls({"AeEnable": enabled})
            
    def get_resolution(self) -> Tuple[int, int]:
        """Return current resolution as (width, height)."""
        return (self.config.width, self.config.height)
        
    def close(self):
        """Release camera resources."""
        if self.camera and PICAMERA_AVAILABLE:
            self.camera.stop()
            self.camera.close()
        self._initialized = False
        logger.info("Camera closed")
        
    def __enter__(self):
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()