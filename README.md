# Scan-to-CNC

Automated 3D scanning and CNC toolpath generation pipeline for Raspberry Pi 5.

## Overview

This project integrates:
- **Structured light 3D scanning** using grid laser + Pi Camera
- **CNC control** via GRBL 1.1 serial communication
- **Point cloud processing** with Open3D
- **Toolpath generation** with OpenCAMLib
- **G-code output** ready for milling

## Hardware

- Raspberry Pi 5 (16GB)
- Genmitsu 3020 Pro CNC (GRBL 1.1f)
- Pi Camera (global shutter recommended)
- Grid laser module (red, 635nm)
- Samsung 1TB SSD (for data storage)

## Installation

```bash
# clone the repo
git clone https://github.com/yourusername/scan-to-cnc.git
cd scan-to-cnc

# create virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# install librealsense sdk (required for pyrealsense2)
# ubuntu/debian:
sudo apt install librealsense2-dev librealsense2-utils

# raspberry pi 5 (build from source):
# see https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_raspbian.md

# install opencamlib (may require build from source on arm64)
pip install opencamlib
# if pip install fails on arm64:
# git clone https://github.com/aewallin/opencamlib.git
# cd opencamlib && mkdir build && cd build
# cmake .. && make -j4 && sudo make install

# verify realsense connection
realsense-viewer  # gui check (optional)
rs-enumerate-devices  # headless check
```

## Project Structure

```
scan-to-cnc/
├── config/
│   ├── machine.yaml        # genmitsu 3020 pro / grbl settings
│   ├── scanner.yaml        # realsense d405 parameters
│   └── processing.yaml     # point cloud, mesh, and toolpath settings
├── src/
│   ├── cnc/
│   │   └── grbl.py         # grbl 1.1 serial communication
│   ├── scanner/
│   │   └── capture.py      # realsense d405 depth capture
│   ├── processing/
│   │   ├── pointcloud.py   # point cloud operations (open3d)
│   │   ├── mesh.py         # surface reconstruction
│   │   └── toolpath.py     # opencamlib toolpath generation
│   ├── gcode/
│   │   └── writer.py       # g-code generation
│   └── pipeline.py         # main orchestration (all 5 stages)
├── scripts/
│   ├── scan.py             # run a scan job
│   └── process.py          # process existing point cloud
├── tests/
├── data/                   # output: point clouds, meshes, gcode, bags
└── requirements.txt
```

## Usage

### Scanning
```bash
# live capture from d405
python scripts/scan.py -o data/scan/pointcloud.ply

# record a .bag file on the pi for offline development
python scripts/scan.py --record my_scan.bag --duration 5

# replay a .bag file (no sensor needed)
python scripts/scan.py --bag data/bags/my_scan.bag -o data/scan/pointcloud.ply
```

### Processing (no hardware needed)
```bash
# process existing point cloud through mesh + gcode
python scripts/process.py --input data/scan/pointcloud.ply --output data/output

# mesh only, skip gcode generation
python scripts/process.py --input data/scan/pointcloud.ply --output data/output --mesh-only
```

### Full Pipeline
```bash
# run all 5 stages end to end (live capture -> cnc execution)
python src/pipeline.py

# full pipeline with .bag playback, skip cnc execution
python src/pipeline.py --bag data/bags/my_scan.bag --skip-execute

# run up to a specific stage (e.g. stop after mesh reconstruction)
python src/pipeline.py --stage 3

# dry run (generates gcode but doesn't send to cnc)
python src/pipeline.py --dry-run
```

## Configuration

Edit `config/machine.yaml` for your CNC setup:
```yaml
serial:
  port: /dev/ttyUSB0
  baud_rate: 115200
work_envelope:
  x: 300
  y: 200
  z: 60
```
Edit `config/scanner.yaml` for D405 capture settings:
```yaml
streams:
  depth:
    width: 640
    height: 480
    fps: 30
filters:
  temporal:
    frames: 15
capture:
  clipping:
    min_distance: 0.04
    max_distance: 0.50
```
## License
MIT
