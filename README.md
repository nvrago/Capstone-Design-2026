# Scan-to-CNC

automated 3D scanning and CNC toolpath generation pipeline for Raspberry Pi 5. captures depth data from an Intel RealSense D405 mounted on an arc carriage, processes multi-view point clouds, reconstructs a watertight mesh, generates toolpaths, and streams G-code to a CNC mill. fully automated, no manual intervention.

## hardware

- Raspberry Pi 5 (16GB RAM, 1TB SSD)
- Intel RealSense D405 on arc carriage
- Teknic ClearCore motor controller (arc positioning)
- Genmitsu 3020 Pro CNC (GRBL 1.1f)
- 7" touchscreen for GUI

## installation

```bash
# clone the repo
git clone https://github.com/yourusername/Capstone-Design-2026.git
cd Capstone-Design-2026

# create virtual environment
python3 -m venv ~/pipeline-env
source ~/pipeline-env/bin/activate

# install pip packages
pip install pyrealsense2 numpy pyserial pyyaml

# open3d (no arm64 pip wheel, build from source)
git clone --recursive https://github.com/isl-org/Open3D.git ~/Open3D
cd ~/Open3D && mkdir build && cd build
cmake -DPYTHON_EXECUTABLE=$(which python3) \
  -DBUILD_GUI=OFF -DBUILD_WEBRTC=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_PYTHON_MODULE=ON ..
make -j4
make install-pip-package

# opencamlib (install via apt, copy into venv)
sudo apt install libopencamlib-dev python3-opencamlib
cp /usr/lib/python3/dist-packages/opencamlib/ocl.so \
  ~/pipeline-env/lib/python3.12/site-packages/

# verify everything works
python3 -c "import open3d; import ocl; import pyrealsense2; print('all good')"
```

for full Pi deployment (systemd service, udev rules, auto-start):
```bash
chmod +x deploy/setup_pi.sh
./deploy/setup_pi.sh
```

## project structure

```
Capstone-Design-2026/
├── src/
│   ├── pipeline.py             # main orchestrator, 6 stages
│   ├── server.py               # TCP server for GUI communication
│   ├── arc/
│   │   └── controller.py       # ClearCore serial protocol
│   ├── scanner/
│   │   └── capture.py          # RealSense D405 depth capture
│   ├── processing/
│   │   ├── registration.py     # multi-view ICP stitching
│   │   ├── zero_mesh.py        # background subtraction
│   │   ├── pointcloud.py       # point cloud ops (Open3D)
│   │   ├── mesh.py             # poisson surface reconstruction
│   │   └── toolpath.py         # OpenCAMLib toolpath generation
│   ├── gcode/
│   │   └── writer.py           # G-code output
│   └── cnc/
│       └── grbl.py             # GRBL 1.1 serial communication
├── scripts/
│   ├── run_pipeline.py         # CLI entry point
│   ├── run_zero_capture.py     # capture zero reference
│   └── tests/                  # test scripts
├── deploy/
│   ├── scantocnc.service       # systemd auto-start
│   ├── setup_pi.sh             # one-time Pi setup
│   └── verify_env.sh           # package check
├── config/
│   ├── machine.yaml            # CNC / GRBL settings
│   ├── scanner.yaml            # D405 capture parameters
│   └── processing.yaml         # mesh, toolpath, gcode settings
└── data/
    ├── reference/              # zero point cloud
    ├── runs/                   # timestamped scan outputs
    └── test/                   # test data
```

## pipeline stages

1. **capture**: moves D405 along arc, captures depth frames at each position with temporal averaging
2. **register**: ICP-aligns all position clouds into one unified cloud, subtracts zero reference to remove background
3. **process**: voxel downsample, statistical outlier removal, normal estimation
4. **mesh**: poisson surface reconstruction to watertight STL
5. **toolpath**: OpenCAMLib drop cutter / waterline toolpath generation, G-code output
6. **execute**: streams G-code to Genmitsu 3020 Pro via GRBL line-by-line

every run saves all intermediates to `data/runs/<timestamp>/` so nothing is ever overwritten.

## usage

### full pipeline
```bash
# full automated run (skip CNC execution)
python scripts/run_pipeline.py --skip-execute

# full run including CNC
python scripts/run_pipeline.py

# with mock arc controller (no ClearCore hardware)
python scripts/run_pipeline.py --mock --skip-execute

# replay a .bag recording instead of live camera
python scripts/run_pipeline.py --bag data/recordings/test.bag --mock
```

### run specific stages
```bash
# capture only (debug arc movement)
python scripts/run_pipeline.py --end-stage 1

# skip capture, run processing through gcode
python scripts/run_pipeline.py --start-stage 2 --skip-execute

# mesh and toolpath only
python scripts/run_pipeline.py --start-stage 4 --end-stage 5
```

### zero reference
```bash
# capture empty plate for background subtraction (run once)
python scripts/run_pipeline.py --zero

# or use the standalone script
python scripts/run_zero_capture.py
```

### override parameters
```bash
# finer arc steps, more frames per position
python scripts/run_pipeline.py --arc-step 10 --frames 60 --skip-execute

# different voxel size and poisson depth
python scripts/run_pipeline.py --voxel-size 0.003 --poisson-depth 9 --skip-execute

# load from yaml config files
python scripts/run_pipeline.py --config config/
```

## GUI and ClearCore integration

the pipeline runs as a TCP server on `localhost:5001`. the GUI sends JSON commands, the server streams status and log messages back.

GUI commands:
- `{"cmd": "scan"}` -- start a scan
- `{"cmd": "zero"}` -- capture zero reference
- `{"cmd": "stop"}` -- emergency stop
- `{"cmd": "status"}` -- query state

ClearCore serial protocol (115200 baud, ASCII):
- `ARC:HOME\n` -> `ARC:OK\n`
- `ARC:MOVE:45.0\n` -> `ARC:OK\n`
- `ARC:STEP:15.0\n` -> `ARC:OK\n`
- `ARC:POS?\n` -> `ARC:POS:45.0\n`
- `ARC:STOP\n` -> `ARC:OK\n`

## configuration

edit `config/scanner.yaml` for capture settings:
```yaml
streams:
  depth:
    width: 640
    height: 480
    fps: 30
filters:
  temporal:
    frames: 30
  decimation:
    magnitude: 2
```

edit `config/machine.yaml` for CNC:
```yaml
serial:
  port: /dev/grbl
  baud_rate: 115200
work_envelope:
  x: 300
  y: 200
  z: 60
```

edit `config/processing.yaml` for pipeline tuning:
```yaml
pointcloud:
  voxel_size: 0.005
  outlier_removal:
    nb_neighbors: 20
    std_ratio: 2.0
mesh:
  method: poisson
  poisson:
    depth: 8
toolpath:
  cutter:
    type: cylindrical
    diameter: 6.0
  operation: surface
  surface:
    stepover: 2.0
```

## license

MIT
