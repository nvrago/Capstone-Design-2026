#!/bin/bash
# deploy/verify_env.sh
source ~/pipeline-env/bin/activate

echo "checking packages..."
python3 -c "import open3d as o3d; print(f'open3d {o3d.__version__} ok')" || echo "FAIL: open3d"
python3 -c "import ocl; print('opencamlib ok')" || echo "FAIL: opencamlib"
python3 -c "import pyrealsense2 as rs; print(f'pyrealsense2 {rs.__version__} ok')" || echo "FAIL: pyrealsense2"
python3 -c "import serial; print('pyserial ok')" || echo "FAIL: pyserial"
python3 -c "import numpy; print(f'numpy {numpy.__version__} ok')" || echo "FAIL: numpy"
python3 -c "import yaml; print('pyyaml ok')" || echo "FAIL: pyyaml"

echo ""
echo "checking pipeline imports..."
cd ~/Capstone-Design-2026
PYTHONPATH=src python3 -c "
from pipeline import ScanPipeline, PipelineConfig
from arc.controller import ArcController, MockArcController
from scanner.capture import RealSenseCapture
from processing.registration import CloudRegistrator
from processing.zero_mesh import apply_zero_subtraction
print('all pipeline imports ok')
" || echo "FAIL: pipeline imports"