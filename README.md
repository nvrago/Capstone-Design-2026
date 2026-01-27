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
# Clone the repo
git clone https://github.com/yourusername/scan-to-cnc.git
cd scan-to-cnc

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install OpenCAMLib (may require build from source on ARM64)
./scripts/install_opencamlib.sh
```

## Project Structure

```
scan-to-cnc/
├── config/
│   ├── machine.yaml      # CNC/GRBL settings
│   ├── scanner.yaml      # Camera and laser parameters
│   └── processing.yaml   # Mesh and toolpath settings
├── src/
│   ├── cnc/
│   │   ├── grbl.py       # GRBL serial communication
│   │   └── motion.py     # Scan patterns and path planning
│   ├── scanner/
│   │   ├── camera.py     # Pi Camera capture
│   │   ├── laser.py      # Laser line detection
│   │   └── triangulation.py
│   ├── processing/
│   │   ├── pointcloud.py # Point cloud operations
│   │   ├── mesh.py       # Surface reconstruction
│   │   └── toolpath.py   # OpenCAMLib wrapper
│   ├── gcode/
│   │   └── writer.py     # G-code generation
│   └── pipeline.py       # Main orchestration
├── scripts/
│   ├── calibrate.py      # Camera/laser calibration
│   ├── scan.py           # Run a scan job
│   └── process.py        # Process existing point cloud
├── tests/
├── data/                 # Output: scans, meshes, gcode
└── requirements.txt
```

## Usage

### Calibration
```bash
python scripts/calibrate.py --camera    # Camera intrinsics
python scripts/calibrate.py --laser     # Laser plane fitting
```

### Scanning
```bash
python scripts/scan.py --output data/my_part
```

### Processing
```bash
python scripts/process.py --input data/my_part --generate-gcode
```

## Configuration

Edit `config/machine.yaml` for your CNC setup:
```yaml
serial_port: /dev/ttyUSB0
baud_rate: 115200
work_envelope:
  x: 300  # mm
  y: 200
  z: 60
```

## License

MIT
```