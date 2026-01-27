"""
Toolpath Generation Module

Wraps OpenCAMLib for CAM toolpath generation from meshes.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

try:
    import ocl
    OCL_AVAILABLE = True
except ImportError:
    OCL_AVAILABLE = False
    logger.warning("OpenCAMLib not available - toolpath generation disabled")

from .mesh import Mesh


class CutterType(Enum):
    CYLINDRICAL = "cylindrical"
    BALL = "ball"
    BULL = "bull"  # corner radius


@dataclass
class CutterDef:
    type: CutterType = CutterType.CYLINDRICAL
    diameter: float = 6.0      # mm
    length: float = 25.0       # mm
    corner_radius: float = 0.0  # mm (for bull cutter)


@dataclass 
class ToolpathPoint:
    x: float
    y: float
    z: float
    feed_type: str = 'cut'  # 'rapid', 'cut', 'plunge'


class ToolpathGenerator:
    """Generates CNC toolpaths from meshes using OpenCAMLib."""
    
    def __init__(self, cutter: CutterDef = None):
        if not OCL_AVAILABLE:
            raise ImportError("OpenCAMLib required. Install with: pip install opencamlib")
        self.cutter_def = cutter or CutterDef()
        self._ocl_cutter = None
        self._ocl_surface = None
        self._create_cutter()
        
    def _create_cutter(self):
        """Create OpenCAMLib cutter object."""
        d = self.cutter_def.diameter
        length = self.cutter_def.length
        r = self.cutter_def.corner_radius
        
        if self.cutter_def.type == CutterType.CYLINDRICAL:
            self._ocl_cutter = ocl.CylCutter(d, length)
        elif self.cutter_def.type == CutterType.BALL:
            self._ocl_cutter = ocl.BallCutter(d, length)
        elif self.cutter_def.type == CutterType.BULL:
            self._ocl_cutter = ocl.BullCutter(d, r, length)
        else:
            raise ValueError(f"Unknown cutter type: {self.cutter_def.type}")
            
        logger.info(f"Created {self.cutter_def.type.value} cutter: "
                   f"D={d}mm, L={length}mm")
                   
    def load_mesh(self, mesh: Mesh):
        """Load mesh into OpenCAMLib STL surface."""
        self._ocl_surface = ocl.STLSurf()
        
        vertices = mesh.vertices
        triangles = mesh.triangles
        
        for tri in triangles:
            p1 = ocl.Point(*vertices[tri[0]])
            p2 = ocl.Point(*vertices[tri[1]])
            p3 = ocl.Point(*vertices[tri[2]])
            t = ocl.Triangle(p1, p2, p3)
            self._ocl_surface.addTriangle(t)
            
        logger.info(f"Loaded {len(triangles)} triangles into OCL surface")
        
    def surface_dropcutter(self, x_min: float, x_max: float,
                           y_min: float, y_max: float,
                           stepover: float,
                           direction: str = 'x') -> List[List[ToolpathPoint]]:
        """
        Generate parallel finishing toolpath using drop cutter.
        
        Args:
            x_min, x_max, y_min, y_max: Toolpath bounds
            stepover: Distance between passes
            direction: 'x' for X-parallel, 'y' for Y-parallel, 'both' for crosshatch
            
        Returns:
            List of toolpath passes, each pass is list of ToolpathPoint
        """
        if self._ocl_surface is None:
            raise ValueError("No mesh loaded - call load_mesh() first")
            
        passes = []
        
        if direction in ('x', 'both'):
            passes.extend(self._dropcutter_passes(
                x_min, x_max, y_min, y_max, stepover, parallel_to='x'
            ))
            
        if direction in ('y', 'both'):
            passes.extend(self._dropcutter_passes(
                x_min, x_max, y_min, y_max, stepover, parallel_to='y'
            ))
            
        return passes
        
    def _dropcutter_passes(self, x_min, x_max, y_min, y_max, 
                           stepover, parallel_to='x') -> List[List[ToolpathPoint]]:
        """Generate drop cutter passes in one direction."""
        passes = []
        
        bdc = ocl.BatchDropCutter()
        bdc.setSTL(self._ocl_surface)
        bdc.setCutter(self._ocl_cutter)
        
        # Sample resolution along cut direction
        sample_step = stepover / 2  # finer sampling along cut
        
        if parallel_to == 'x':
            y_values = np.arange(y_min, y_max + stepover, stepover)
            x_values = np.arange(x_min, x_max + sample_step, sample_step)
            
            for i, y in enumerate(y_values):
                # Alternate direction for climb milling efficiency
                xs = x_values if i % 2 == 0 else x_values[::-1]
                for x in xs:
                    bdc.appendPoint(ocl.CLPoint(x, y, 0))
        else:
            x_values = np.arange(x_min, x_max + stepover, stepover)
            y_values = np.arange(y_min, y_max + sample_step, sample_step)
            
            for i, x in enumerate(x_values):
                ys = y_values if i % 2 == 0 else y_values[::-1]
                for y in ys:
                    bdc.appendPoint(ocl.CLPoint(x, y, 0))
                    
        bdc.run()
        
        # Extract results and organize into passes
        cl_points = bdc.getCLPoints()
        
        current_pass = []
        last_y = None if parallel_to == 'x' else None
        last_x = None if parallel_to == 'y' else None
        
        for clp in cl_points:
            pt = ToolpathPoint(x=clp.x, y=clp.y, z=clp.z, feed_type='cut')
            
            # Detect pass breaks
            if parallel_to == 'x':
                if last_y is not None and abs(clp.y - last_y) > stepover * 0.9:
                    if current_pass:
                        passes.append(current_pass)
                    current_pass = []
                last_y = clp.y
            else:
                if last_x is not None and abs(clp.x - last_x) > stepover * 0.9:
                    if current_pass:
                        passes.append(current_pass)
                    current_pass = []
                last_x = clp.x
                
            current_pass.append(pt)
            
        if current_pass:
            passes.append(current_pass)
            
        logger.info(f"Generated {len(passes)} {parallel_to}-direction passes")
        return passes
        
    def waterline(self, z_min: float, z_max: float, z_step: float,
                  x_min: float, x_max: float,
                  y_min: float, y_max: float,
                  sampling: float = 0.5) -> List[List[ToolpathPoint]]:
        """
        Generate waterline (contour) toolpath.
        
        Creates horizontal slices at each Z level.
        
        Args:
            z_min, z_max: Z range
            z_step: Vertical step between levels
            x_min, x_max, y_min, y_max: XY bounds for sampling
            sampling: Sample resolution
        """
        if self._ocl_surface is None:
            raise ValueError("No mesh loaded - call load_mesh() first")
            
        all_passes = []
        z_levels = np.arange(z_min, z_max + z_step, z_step)
        
        for z in z_levels:
            wl = ocl.Waterline()
            wl.setSTL(self._ocl_surface)
            wl.setCutter(self._ocl_cutter)
            wl.setZ(z)
            wl.setSampling(sampling)
            wl.run()
            
            loops = wl.getLoops()
            for loop in loops:
                pass_points = []
                for pt in loop:
                    pass_points.append(ToolpathPoint(
                        x=pt.x, y=pt.y, z=z, feed_type='cut'
                    ))
                if pass_points:
                    all_passes.append(pass_points)
                    
        logger.info(f"Generated waterline toolpath: {len(all_passes)} loops "
                   f"across {len(z_levels)} Z levels")
        return all_passes
        
    def add_lead_in_out(self, passes: List[List[ToolpathPoint]], 
                        clearance_z: float) -> List[List[ToolpathPoint]]:
        """Add rapid moves and plunges between passes."""
        result = []
        
        for pass_points in passes:
            if not pass_points:
                continue
                
            modified = []
            first = pass_points[0]
            last = pass_points[-1]
            
            # Rapid to start position at clearance height
            modified.append(ToolpathPoint(
                x=first.x, y=first.y, z=clearance_z, feed_type='rapid'
            ))
            # Plunge to cut depth
            modified.append(ToolpathPoint(
                x=first.x, y=first.y, z=first.z, feed_type='plunge'
            ))
            # Cutting moves
            modified.extend(pass_points)
            # Retract at end
            modified.append(ToolpathPoint(
                x=last.x, y=last.y, z=clearance_z, feed_type='rapid'
            ))
            
            result.append(modified)
            
        return result