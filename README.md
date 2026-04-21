# Scan-to-CNC

automated 3D scanning and CNC toolpath generation pipeline for Raspberry Pi 5. captures depth data from an Intel RealSense D405 mounted on an arc carriage, processes multi-view point clouds, reconstructs a watertight mesh, generates toolpaths, and streams G-code to a CNC mill. fully automated, no manual intervention.

## hardware

- Raspberry Pi 5 (16GB RAM, 1TB SSD)
- Intel RealSense D405 on arc carriage
- Teknic ClearCore motor controller (arc positioning, Modbus TCP)
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
pip install pyrealsense2 numpy pyserial pyyaml pymodbus

# open3d (no arm64 pip wheel, build via docker or from source)
# option 1: docker build (recommended)
git clone --recursive https://github.com/isl-org/Open3D.git ~/Open3D
cd ~/Open3D/docker
./docker_build.sh openblas-arm64-py312
pip install open3d-*.whl

# option 2: build from source
cd ~/Open3D && mkdir build && cd build
cmake -DPYTHON_EXECUTABLE=$(which python3) \
  -DBUILD_GUI=OFF -DBUILD_WEBRTC=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_PYTHON_MODULE=ON ..
make -j4
make install-pip-package

# opencamlib (build from source on Trixie)
sudo apt install -y build-essential cmake git libboost-dev python3-dev
git clone https://github.com/aewallin/opencamlib.git ~/opencamlib
cd ~/opencamlib && mkdir build && cd build
cmake -DPYTHON_EXECUTABLE=$(which python3) ..
make -j4
sudo make install
# find and copy ocl.so into venv
find /usr/local/lib -name "ocl*.so" 2>/dev/null
cp <path_to_ocl.so> ~/pipeline-env/lib/python3.*/site-packages/

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
│   │   └── controller.py       # ClearCore Modbus TCP interface
│   ├── scanner/
│   │   └── capture.py          # RealSense D405 depth capture
│   ├── processing/
│   │   ├── registration.py     # multi-view ICP stitching
│   │   ├── zero_mesh.py        # background subtraction + z-clip
│   │   ├── pointcloud.py       # point cloud ops (Open3D)
│   │   ├── mesh.py             # poisson reconstruction + density trim
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
│   └── verify_env.sh           # package check on boot
├── config/
│   ├── machine.yaml            # CNC, GRBL, ClearCore settings
│   ├── scanner.yaml            # D405 capture parameters
│   └── processing.yaml         # mesh, toolpath, gcode settings
└── data/
    ├── reference/              # zero point cloud
    ├── runs/                   # timestamped scan outputs
    └── test/                   # test data
```

## pipeline stages

1. **capture**: moves D405 along arc, captures depth frames at each position with temporal averaging
2. **register**: ICP-aligns all position clouds into one unified cloud, subtracts zero reference, clips below plate surface
3. **process**: voxel downsample, statistical outlier removal, normal estimation
4. **mesh**: poisson surface reconstruction, density trim, small component removal
5. **toolpath**: OpenCAMLib drop cutter / waterline toolpath generation, G-code output
6. **execute**: streams G-code to Genmitsu 3020 Pro via GRBL line-by-line

every run saves all intermediates to `data/runs/<timestamp>/` so nothing is ever overwritten.

## usage

### full pipeline (autonomous mode)
```bash
# full automated run (skip CNC execution)
python scripts/run_pipeline.py --skip-execute

# with mock arc controller (no ClearCore hardware)
python scripts/run_pipeline.py --mock --skip-execute

# replay a .bag recording instead of live camera
python scripts/run_pipeline.py --bag data/recordings/test.bag --mock

# 1D single-position test scan
python scripts/run_pipeline.py --mock --skip-execute --arc-start 0 --arc-end 0 --arc-step 15 --voxel-size 0.002
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
python scripts/run_pipeline.py --zero --mock

# 1D zero reference (single position)
python scripts/run_pipeline.py --zero --mock --arc-start 0 --arc-end 0 --arc-step 15
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

## system architecture

```
┌─────────────────┐     Modbus TCP      ┌─────────────────┐
│                 │    port 502         │                 │
│   Touchscreen   │◄──────────────────►│   ClearCore     │
│   GUI (UI.py)   │   motor commands    │   (.ino)        │
│                 │   status polling    │   arc motor     │
│                 │                     │   limit switches│
│                 │     TCP JSON        │   e-stop        │
│                 │    port 5001        └─────────────────┘
│                 │◄──────────────────►┌─────────────────┐
│                 │   capture/process   │  Pipeline Server│
└─────────────────┘   commands          │  (server.py)    │
                                        │                 │
                                        │  ┌───────────┐  │
                                        │  │ D405 USB  │  │
                                        │  │ capture   │  │
                                        │  └───────────┘  │
                                        │  ┌───────────┐  │
                                        │  │ pipeline  │  │
                                        │  │ stages 2-5│  │
                                        │  └───────────┘  │
                                        └────────┬────────┘
                                                 │ serial
                                                 ▼
                                        ┌─────────────────┐
                                        │  Genmitsu 3020  │
                                        │  GRBL 1.1f      │
                                        │  (stage 6)      │
                                        └─────────────────┘
```

the GUI owns two independent communication channels. it talks to the ClearCore over Modbus TCP to control the arc motor (home, move, stop, status polling). separately, it talks to the pipeline server over TCP JSON to control the camera and processing (start session, capture frames, trigger processing). the pipeline server never touches the motor. the ClearCore never touches the camera. the GUI coordinates the two: move motor to position, tell server to capture, repeat, then tell server to process everything.

the GRBL CNC is only involved in stage 6 when the pipeline streams the final G-code for milling. it runs over serial from the pipeline server, not from the GUI.

## GUI and ClearCore integration

the system has two communication channels running in parallel:

**pipeline server (TCP on localhost:5001):**
the pipeline runs as a TCP server. the GUI sends JSON commands, the server manages the camera and processing pipeline.

GUI-driven scan flow:
- `{"cmd": "start_scan_session"}` -- prepare fresh pipeline for captures
- `{"cmd": "capture_frame", "request_id": 1, "index": 0, "angle_deg": 45.0}` -- capture at current position
- `{"cmd": "end_scan_session"}` -- process all captured frames (stages 2-5)

other commands:
- `{"cmd": "scan"}` -- run full autonomous pipeline
- `{"cmd": "zero"}` -- capture zero reference
- `{"cmd": "stop"}` -- emergency stop
- `{"cmd": "status"}` -- query state
- `{"cmd": "config", "key": "...", "value": ...}` -- change parameter

**ClearCore (Modbus TCP on 192.168.1.20:502):**
the GUI talks to the ClearCore directly via Modbus TCP to control arc motor positioning. the pipeline server is not involved in motor control during GUI-driven scans.

register map matches CLIENT_INFC struct in ClearCoreModbusTest.ino. command codes:
- CCMD_ENAB_MTRS (1) -- enable motor
- CCMD_MOVE (4) -- absolute position move
- CCMD_STOP (5) -- emergency stop
- CCMD_RUN_1 (6) -- home + scan sequence

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

edit `config/machine.yaml` for CNC and ClearCore:
```yaml
serial:
  port: /dev/grbl
  baud_rate: 115200
work_envelope:
  x: 300
  y: 200
  z: 60
arc:
  host: 192.168.1.20
  modbus_port: 502
  slave_id: 1
  steps_per_degree: 333.33
```

edit `config/processing.yaml` for pipeline tuning:
```yaml
pointcloud:
  voxel_size: 0.005
  outlier_removal:
    nb_neighbors: 30
    std_ratio: 1.5
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
