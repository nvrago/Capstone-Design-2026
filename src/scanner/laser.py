"""
Laser Detection Module

Extracts laser grid points from captured images using HSV filtering
and subpixel refinement.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LaserConfig:
    # Primary HSV range for red laser
    hsv_lower: Tuple[int, int, int] = (0, 100, 100)
    hsv_upper: Tuple[int, int, int] = (10, 255, 255)
    # Secondary range for red hue wraparound
    hsv_lower_alt: Tuple[int, int, int] = (170, 100, 100)
    hsv_upper_alt: Tuple[int, int, int] = (180, 255, 255)
    # Filtering
    min_area: int = 5
    max_area: int = 1000
    # Subpixel refinement
    use_subpixel: bool = True
    subpixel_window: int = 5


class LaserDetector:
    """Detects laser grid points in images."""
    
    def __init__(self, config: LaserConfig = None):
        self.config = config or LaserConfig()
        
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect laser points in an image.
        
        Args:
            image: BGR image from camera
            
        Returns:
            Nx2 array of (x, y) pixel coordinates
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for red laser (handle hue wraparound)
        mask1 = cv2.inRange(
            hsv, 
            np.array(self.config.hsv_lower), 
            np.array(self.config.hsv_upper)
        )
        mask2 = cv2.inRange(
            hsv, 
            np.array(self.config.hsv_lower_alt), 
            np.array(self.config.hsv_upper_alt)
        )
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract centroids
        points = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_area <= area <= self.config.max_area:
                M = cv2.moments(contour)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    points.append([cx, cy])
                    
        if not points:
            return np.array([]).reshape(0, 2)
            
        points = np.array(points, dtype=np.float32)
        
        # Subpixel refinement
        if self.config.use_subpixel and len(points) > 0:
            points = self._refine_subpixel(image, points)
            
        logger.debug(f"Detected {len(points)} laser points")
        return points
        
    def _refine_subpixel(self, image: np.ndarray, 
                         points: np.ndarray) -> np.ndarray:
        """
        Refine point locations to subpixel accuracy using intensity centroid.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        half = self.config.subpixel_window // 2
        
        refined = []
        for px, py in points:
            # Extract window around point
            x0 = max(0, int(px) - half)
            x1 = min(w, int(px) + half + 1)
            y0 = max(0, int(py) - half)
            y1 = min(h, int(py) + half + 1)
            
            window = gray[y0:y1, x0:x1].astype(np.float32)
            
            if window.size == 0:
                refined.append([px, py])
                continue
                
            # Intensity-weighted centroid
            total = window.sum()
            if total > 0:
                yy, xx = np.mgrid[0:window.shape[0], 0:window.shape[1]]
                cx = (xx * window).sum() / total + x0
                cy = (yy * window).sum() / total + y0
                refined.append([cx, cy])
            else:
                refined.append([px, py])
                
        return np.array(refined, dtype=np.float32)
        
    def detect_with_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect laser points and also return the binary mask.
        
        Useful for debugging/visualization.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(
            hsv, 
            np.array(self.config.hsv_lower), 
            np.array(self.config.hsv_upper)
        )
        mask2 = cv2.inRange(
            hsv, 
            np.array(self.config.hsv_lower_alt), 
            np.array(self.config.hsv_upper_alt)
        )
        mask = cv2.bitwise_or(mask1, mask2)
        
        points = self.detect(image)
        return points, mask
        
    def visualize(self, image: np.ndarray, points: np.ndarray, 
                  radius: int = 3, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw detected points on image for visualization.
        """
        vis = image.copy()
        for x, y in points:
            cv2.circle(vis, (int(x), int(y)), radius, color, -1)
        return vis
        
    def tune_thresholds(self, image: np.ndarray) -> dict:
        """
        Analyze image to suggest HSV threshold values.
        
        Useful during initial setup.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Find brightest pixels (likely laser)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = np.percentile(gray, 99)
        bright_mask = gray > threshold
        
        if bright_mask.sum() == 0:
            return {"error": "No bright pixels found"}
            
        # Get HSV values at bright pixels
        h_vals = hsv[:, :, 0][bright_mask]
        s_vals = hsv[:, :, 1][bright_mask]
        v_vals = hsv[:, :, 2][bright_mask]
        
        return {
            "h_range": (int(np.percentile(h_vals, 5)), int(np.percentile(h_vals, 95))),
            "s_range": (int(np.percentile(s_vals, 5)), int(np.percentile(s_vals, 95))),
            "v_range": (int(np.percentile(v_vals, 5)), int(np.percentile(v_vals, 95))),
            "sample_count": int(bright_mask.sum())
        }