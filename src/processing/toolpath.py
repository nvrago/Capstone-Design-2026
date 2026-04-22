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
    diameter: float = 6.0       # mm
    length: float = 25.0        # mm
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
        self._bounds_mm = None
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
        """Load mesh into OpenCAMLib STL surface.

        mesh is in meters (scanner coordinate system); OCL and the g-code
        writer both expect mm. scale m -> mm here, then shift so the min
        corner sits at (0, 0) and the top surface sits at Z=0. that matches
        how you'd zero the Genmitsu on the top-front-left corner of the stock.

        returns the shifted+scaled bounds in mm so the caller doesn't have
        to guess what coordinate space the toolpath lives in.
        """
        verts_mm = mesh.vertices * 1000.0

        # shift so min corner -> (0,0) and top -> 0 (cuts go negative Z)
        min_b = verts_mm.min(axis=0)
        max_b = verts_mm.max(axis=0)
        shift = np.array([-min_b[0], -min_b[1], -max_b[2]])
        verts_mm = verts_mm + shift

        self._ocl_surface = ocl.STLSurf()
        triangles = mesh.triangles
        for tri in triangles:
            p1 = ocl.Point(*verts_mm[tri[0]])
            p2 = ocl.Point(*verts_mm[tri[1]])
            p3 = ocl.Point(*verts_mm[tri[2]])
            t = ocl.Triangle(p1, p2, p3)
            self._ocl_surface.addTriangle(t)

        self._bounds_mm = {
            'x': (0.0, float(max_b[0] - min_b[0])),
            'y': (0.0, float(max_b[1] - min_b[1])),
            'z': (float(min_b[2] - max_b[2]), 0.0),
        }

        logger.info(f"Loaded {len(triangles)} triangles into OCL surface "
                    f"(m -> mm, shifted to origin). "
                    f"bounds: x={self._bounds_mm['x']}, "
                    f"y={self._bounds_mm['y']}, z={self._bounds_mm['z']}")

        return self._bounds_mm

    def surface_dropcutter(self, x_min: float, x_max: float,
                           y_min: float, y_max: float,
                           stepover: float,
                           direction: str = 'x') -> List[List[ToolpathPoint]]:
        """
        Generate parallel finishing toolpath using drop cutter.

        Args:
            x_min, x_max, y_min, y_max: Toolpath bounds (mm)
            stepover: Distance between passes (mm)
            direction: 'x' for X-parallel, 'y' for Y-parallel, 'both' for crosshatch
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
        """Generate drop cutter passes in one direction.

        Uses PathDropCutter (one path per pass) instead of BatchDropCutter.
        BatchDropCutter is broken on the ARM64 OCL build (returns start z for
        every drop). PathDropCutter works correctly and is the recommended
        API for line-based dropcutter operations anyway.
        """
        passes = []

        z_start = self._ocl_surface.bb.maxpt.z + 10.0
        sample_step = stepover / 2  # finer sampling along cut direction

        if parallel_to == 'x':
            y_values = np.arange(y_min, y_max + stepover, stepover)
            for i, y in enumerate(y_values):
                # one PathDropCutter per pass (one Y row)
                pdc = ocl.PathDropCutter()
                pdc.setSTL(self._ocl_surface)
                pdc.setCutter(self._ocl_cutter)
                pdc.setSampling(sample_step)
                pdc.setZ(z_start)

                path = ocl.Path()
                # alternate direction each row to minimize rapid moves
                if i % 2 == 0:
                    p_start = ocl.Point(x_min, y, z_start)
                    p_end = ocl.Point(x_max, y, z_start)
                else:
                    p_start = ocl.Point(x_max, y, z_start)
                    p_end = ocl.Point(x_min, y, z_start)
                path.append(ocl.Line(p_start, p_end))
                pdc.setPath(path)
                pdc.run()

                pts = pdc.getCLPoints()
                current_pass = []
                for clp in pts:
                    # missed-mesh points come back at z_start; skip them
                    if abs(clp.z - z_start) < 1e-6:
                        if current_pass:
                            passes.append(current_pass)
                            current_pass = []
                        continue
                    current_pass.append(ToolpathPoint(
                        x=clp.x, y=clp.y, z=clp.z, feed_type='cut'
                    ))
                if current_pass:
                    passes.append(current_pass)
        else:
            x_values = np.arange(x_min, x_max + stepover, stepover)
            for i, x in enumerate(x_values):
                pdc = ocl.PathDropCutter()
                pdc.setSTL(self._ocl_surface)
                pdc.setCutter(self._ocl_cutter)
                pdc.setSampling(sample_step)
                pdc.setZ(z_start)

                path = ocl.Path()
                if i % 2 == 0:
                    p_start = ocl.Point(x, y_min, z_start)
                    p_end = ocl.Point(x, y_max, z_start)
                else:
                    p_start = ocl.Point(x, y_max, z_start)
                    p_end = ocl.Point(x, y_min, z_start)
                path.append(ocl.Line(p_start, p_end))
                pdc.setPath(path)
                pdc.run()

                pts = pdc.getCLPoints()
                current_pass = []
                for clp in pts:
                    if abs(clp.z - z_start) < 1e-6:
                        if current_pass:
                            passes.append(current_pass)
                            current_pass = []
                        continue
                    current_pass.append(ToolpathPoint(
                        x=clp.x, y=clp.y, z=clp.z, feed_type='cut'
                    ))
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

            modified.append(ToolpathPoint(
                x=first.x, y=first.y, z=clearance_z, feed_type='rapid'
            ))
            modified.append(ToolpathPoint(
                x=first.x, y=first.y, z=first.z, feed_type='plunge'
            ))
            modified.extend(pass_points)
            modified.append(ToolpathPoint(
                x=last.x, y=last.y, z=clearance_z, feed_type='rapid'
            ))

            result.append(modified)

        return result