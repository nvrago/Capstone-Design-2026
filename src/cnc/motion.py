"""
Motion Planning Module

Generates scan patterns and handles gap-fill path planning.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PatternType(Enum):
    RASTER = "raster"
    SERPENTINE = "serpentine"
    SPIRAL = "spiral"


@dataclass
class ScanRegion:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_height: float = 30.0  # scanning height


class MotionPlanner:
    """Generates scan patterns and manages coverage."""
    
    def __init__(self, work_envelope: Tuple[float, float, float] = (300, 200, 60)):
        self.envelope_x, self.envelope_y, self.envelope_z = work_envelope
        self.visited_positions: List[Tuple[float, float]] = []
        self.coverage_grid: Optional[np.ndarray] = None
        self.grid_resolution: float = 1.0  # mm per cell
        
    def generate_raster(self, region: ScanRegion, step: float,
                        serpentine: bool = True) -> List[Tuple[float, float]]:
        """
        Generate raster scan pattern.
        
        Args:
            region: Area to scan
            step: Distance between scan lines in mm
            serpentine: If True, alternate direction each row (faster)
        """
        positions = []
        y = region.y_min
        direction = 1
        
        while y <= region.y_max:
            if serpentine:
                if direction == 1:
                    x_range = np.arange(region.x_min, region.x_max + step, step)
                else:
                    x_range = np.arange(region.x_max, region.x_min - step, -step)
            else:
                x_range = np.arange(region.x_min, region.x_max + step, step)
                
            for x in x_range:
                if region.x_min <= x <= region.x_max:
                    positions.append((float(x), float(y)))
                    
            y += step
            direction *= -1
            
        logger.info(f"Generated raster pattern: {len(positions)} positions")
        return positions
        
    def generate_spiral(self, region: ScanRegion, step: float) -> List[Tuple[float, float]]:
        """Generate inward spiral pattern."""
        positions = []
        
        x_min, x_max = region.x_min, region.x_max
        y_min, y_max = region.y_min, region.y_max
        
        while x_min <= x_max and y_min <= y_max:
            # Top edge (left to right)
            x = x_min
            while x <= x_max:
                positions.append((x, y_min))
                x += step
                
            y_min += step
            
            # Right edge (top to bottom)
            y = y_min
            while y <= y_max:
                positions.append((x_max, y))
                y += step
                
            x_max -= step
            
            # Bottom edge (right to left)
            if y_min <= y_max:
                x = x_max
                while x >= x_min:
                    positions.append((x, y_max))
                    x -= step
                y_max -= step
                
            # Left edge (bottom to top)
            if x_min <= x_max:
                y = y_max
                while y >= y_min:
                    positions.append((x_min, y))
                    y -= step
                x_min += step
                
        logger.info(f"Generated spiral pattern: {len(positions)} positions")
        return positions
        
    def init_coverage_grid(self, region: ScanRegion, resolution: float = 1.0):
        """Initialize coverage tracking grid."""
        self.grid_resolution = resolution
        width = int((region.x_max - region.x_min) / resolution) + 1
        height = int((region.y_max - region.y_min) / resolution) + 1
        self.coverage_grid = np.zeros((height, width), dtype=np.int32)
        self._grid_origin = (region.x_min, region.y_min)
        logger.info(f"Coverage grid initialized: {width}x{height}")
        
    def update_coverage(self, points_3d: np.ndarray):
        """Update coverage grid with scanned points."""
        if self.coverage_grid is None:
            return
            
        for x, y, z in points_3d:
            gx = int((x - self._grid_origin[0]) / self.grid_resolution)
            gy = int((y - self._grid_origin[1]) / self.grid_resolution)
            
            if 0 <= gx < self.coverage_grid.shape[1] and \
               0 <= gy < self.coverage_grid.shape[0]:
                self.coverage_grid[gy, gx] += 1
                
    def find_gaps(self, min_density: int = 1) -> List[Tuple[float, float]]:
        """
        Find regions with insufficient point density.
        
        Returns list of positions that need additional scanning.
        """
        if self.coverage_grid is None:
            return []
            
        gaps = []
        sparse = self.coverage_grid < min_density
        
        # Find connected components of sparse regions
        from scipy import ndimage
        labeled, num_features = ndimage.label(sparse)
        
        for i in range(1, num_features + 1):
            region = labeled == i
            ys, xs = np.where(region)
            
            if len(xs) > 0:
                # Return centroid of gap region
                cx = xs.mean() * self.grid_resolution + self._grid_origin[0]
                cy = ys.mean() * self.grid_resolution + self._grid_origin[1]
                gaps.append((cx, cy))
                
        logger.info(f"Found {len(gaps)} gap regions")
        return gaps
        
    def generate_gap_fill_pattern(self, gaps: List[Tuple[float, float]], 
                                   radius: float = 5.0,
                                   step: float = 2.0) -> List[Tuple[float, float]]:
        """
        Generate scan positions to fill gaps.
        
        Creates small raster patterns around each gap centroid.
        """
        positions = []
        
        for cx, cy in gaps:
            # Small raster around gap center
            for dy in np.arange(-radius, radius + step, step):
                for dx in np.arange(-radius, radius + step, step):
                    x, y = cx + dx, cy + dy
                    # Check within work envelope
                    if 0 <= x <= self.envelope_x and 0 <= y <= self.envelope_y:
                        positions.append((x, y))
                        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for pos in positions:
            key = (round(pos[0], 1), round(pos[1], 1))
            if key not in seen:
                seen.add(key)
                unique.append(pos)
                
        logger.info(f"Generated gap-fill pattern: {len(unique)} positions")
        return unique
        
    def optimize_path(self, positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Optimize scan order to minimize travel distance.
        
        Uses nearest-neighbor heuristic.
        """
        if len(positions) <= 2:
            return positions
            
        remaining = list(positions)
        optimized = [remaining.pop(0)]
        
        while remaining:
            current = optimized[-1]
            # Find nearest unvisited position
            distances = [np.hypot(p[0] - current[0], p[1] - current[1]) 
                        for p in remaining]
            nearest_idx = np.argmin(distances)
            optimized.append(remaining.pop(nearest_idx))
            
        return optimized
        
    def estimate_scan_time(self, positions: List[Tuple[float, float]],
                           feed_rate: float = 500,
                           settle_time: float = 0.1) -> float:
        """
        Estimate total scan time in seconds.
        
        Args:
            positions: Scan positions
            feed_rate: Travel speed in mm/min
            settle_time: Wait time at each position in seconds
        """
        if len(positions) < 2:
            return 0.0
            
        # Total travel distance
        total_dist = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_dist += np.hypot(dx, dy)
            
        travel_time = (total_dist / feed_rate) * 60  # convert to seconds
        settle_total = len(positions) * settle_time
        
        return travel_time + settle_total