# Scan-to-Mill — ClearCore ↔ Raspberry Pi Comms Test

A standalone test rig that validates the Modbus TCP communication path between
the Raspberry Pi operator interface and the Teknic ClearCore motion controller
in the OSU Scan-to-Mill capstone project. When this works end-to-end, you have
proof that your network, framing, register layout, command dispatch, and
status readback are all healthy — which is the foundation everything else in
the system depends on.

This README is split into two paths:

- **Quick Start** — for someone who has the hardware in front of them and
  just wants to bring up the test in the next 30 minutes.
- **Deep Dive** — architecture, register map, debugging procedures, and
  rationale for the design choices.

---

## What This Project Is

The Scan-to-Mill capstone integrates a depth camera, a motion-controlled scan
carriage, and a CNC router into one workflow: scan a part, generate a
toolpath, mill a copy. There are roughly four subsystems:

1. **Capture** — Intel RealSense D405 depth camera mounted on a motorized arc
   carriage, scanning the part from multiple angles.
2. **Motion** — Teknic ClearCore PLC driving a NEMA17 stepper through a TB6600
   driver, moving the camera carriage along the arc track.
3. **Operator interface** — PyQt6/PyVista UI on a Raspberry Pi 5 that shows
   the live point cloud, controls the scan, and exports G-code.
4. **Machining** — Genmitsu 3020 Pro Max V2 CNC router that consumes the
   exported G-code.

The motion subsystem (the ClearCore) and the operator interface (the Pi) are
separate computers connected by an Ethernet cable. **They speak Modbus TCP to
each other.** This repository contains the test harness that proves that link
works before you try to run any of the actual scan logic on top of it.

If the comms test passes, you can be confident that:

- The Pi can reach the ClearCore over the wire
- The ClearCore is running its sketch and listening on port 502
- Commands sent from the Pi reach the ClearCore and trigger the right actions
- Status read by the Pi accurately reflects ClearCore state
- The 32-bit numeric encoding agrees on both sides
- The MBAP framing and Modbus function codes are correctly implemented

If it fails, you have a layered diagnostic tool that tells you exactly which
of those things is broken.

---

## Recent Progress (Session Log)

This section captures work completed in each debugging/feature session so
that progress isn't lost between sittings. Most recent at the top.

### Session — April 13, 2026

**Scan sequence with limit switches (ClearCore side).** The previous
`CCMD_RUN_1` back-and-forth placeholder was replaced with a real three-phase
scan motion: **home → scan → return**. The carriage now drives slowly toward
a home limit switch, then sweeps across to a far limit switch, then returns
to home and parks. Two limit switch inputs on `ConnectorIO2` (far) and
`ConnectorIO3` (home) drive the state transitions. A new state machine
(`MS_HOMING`, `MS_SCANNING`, `MS_RETURNING`, `MS_STOPPED`) in
`serviceMotion()` handles the transitions; `motor.PositionRefSet(0)` is
called when home is reached so `cur_posn` becomes meaningful.

Limit-switch polarity was configurable via `LIMIT_ACTIVE_HIGH` — set to
`true` to match the installed switches (active-high when pressed). Without
this, the firmware thought both switches were always pressed and the whole
sequence would complete in microseconds with no motor movement, which
produced the spurious "SCAN COMPLETE" message immediately on Start. Fixed.

**Emergency stop on `ConnectorIO1`.** Added `serviceEstop()`, called first
from `loop()` so it takes priority over everything else. On engage, the
drive is disabled (`EnableRequest(false)`), `motionState` → `MS_DISABLED`,
and `regs.state` → `CST_FAULT`. On release, `motionState` is reset to
`MS_STOPPED` and `regs.state` to `CST_IDLE`, but the drive is NOT
auto-re-enabled — the operator must explicitly re-arm via START. A new
`ESTOP` bit (bit 4) was exposed in the status bitfield so the Pi knows
when e-stop is engaged without having to poll a separate register.

A command-filter was added to the top of `handleCommand()`: while e-stop
is engaged, all commands except `CCMD_STOP` and `CCMD_DISAB_MTRS` are
rejected with `CCMD_NACK`.

**Non-blocking rearm (the important one).** The initial e-stop
implementation had a 2-second `delay()` in `startSequence1()` to let the
ClearPath drive come up after an enable-edge. This blocked the main
`loop()` long enough that `serviceModbus()` didn't service inbound writes
in time, and the Pi's pymodbus client would time out with
`No response received after 3 retries, CLOSING CONNECTION` — causing a
~10-second gap between button press and motion on each Start Scan.

Fix: added a new `MS_REARM_WAIT` state. `startSequence1()` now just
issues an enable-cycle (`EnableRequest(false)` / 50ms / `EnableRequest(true)`),
sets `motionState = MS_REARM_WAIT`, records the timestamp, and returns
immediately. `serviceMotion()` handles the 2-second wait across subsequent
`loop()` iterations, so Modbus polls keep flowing the entire time. Response
time from button press to motor start is now ~2 seconds (the drive settle
time) instead of ~10.

**Operator interface — E-STOP banner.** Added a full-width red overlay
label (`estop_banner`) that appears when the ClearCore's `ESTOP` status
bit is set and hides when it clears. Uses edge detection in
`_on_modbus_status()` so `_on_estop_engaged()` / `_on_estop_cleared()`
fire exactly once per transition. Chose a persistent banner over a modal
pop-up because modal dialogs block the log panel and need dismissal —
a banner is industrial-HMI standard for annunciators.

**Operator interface — removed fake ScanWorker.** The wireframe-era
`ScanWorker` QThread was generating fake "25% front face captured" log
lines and progress bar updates on an 8-second timer. It was also blocking
the Qt event loop enough to cause Modbus write timeouts on Start Scan.
Stripped it out. The progress/stage labels are now driven from real
ClearCore status: the `SCANNING` bit is the authoritative "motion in
progress" signal, and falling-edge on that bit (while not faulted/estopped)
triggers `_on_scan_complete()`, which re-enables the Start button, sets
the stage to COMPLETE, and unlocks the STL export button.

**Fixed: Start button not re-enabling after scan completion.** Before the
scan-complete edge detector was added, the UI had no way to know the motor
had parked, so `btn_start` stayed disabled until the operator pressed
STOP. The falling-edge detection on `SCANNING` fixed this — now Start
re-enables automatically when the carriage returns home.

**Miscellaneous bug fixes:**

- `KeyError: 'fault_code'` crash in `_on_modbus_status()` — the code
  referenced a key that was never populated in the status dict. Removed
  the reference and switched to `.get()` with defaults throughout.
- Boot-time auto-start of the back-and-forth sequence was removed.
  Motion now only happens in response to `CCMD_RUN_1` from the Pi.
- `_make_estop_banner()` and `resizeEvent()` were placed at module scope
  instead of inside `ScanToMillUI` — fixed indentation.
- Duplicate `estop` bit declaration in `CCMTR_STATUS` caused by re-editing
  the struct — removed the duplicate.
- git sync foot-gun on the Pi: edits made via the GitHub web UI must be
  **committed** before `git pull` sees them on the Pi. A stale
  `UI.py` on the Pi was producing the `fault_code` crash even after
  "fixing" the file in the browser.

**Outcome:** Start Scan now runs a clean home-scan-return cycle in ~60 s
end-to-end (at `SCAN_VEL_SPS = 2000`). E-stop halts cleanly, the UI banner
shows/hides on the correct edges, re-arming after e-stop works, and the
Start button resets automatically when a scan completes. The system is in
a usable demo state.

---

## Hardware Bill of Materials

| Component | Notes |
|---|---|
| Raspberry Pi 5 (16GB) | Running Raspberry Pi OS Bookworm or Trixie |
| Official 7" Raspberry Pi Touchscreen | Or any HDMI display ≥ 1280×800 |
| Teknic ClearCore (CLCR-4-13) | The motion controller |
| ClearPath-SD or generic stepper | M-0 connector, step-and-direction mode |
| TB6600 stepper driver | If using a generic stepper instead of ClearPath |
| Ethernet cable, Cat5e or better | Direct Pi → ClearCore, no switch required |
| ClearCore 24V power supply | Required for motion |
| USB-C power for Pi 5 | 5.1V 5A official supply recommended |
| USB-C cable, Pi to ClearCore | For flashing the ClearCore and reading its serial debug log |

You can run the comms test without the motor connected — the ClearCore will
boot, log "M-0 enabled", and report "no motion" status, but the Modbus link
itself will still come up. This is useful for software-only debugging.

---

## Quick Start

These steps assume you have the hardware listed above, fresh OS images on
both devices, and roughly 30 minutes. For the why and how behind each step,
see the Deep Dive sections below.

### 1. Flash the Raspberry Pi (if not already done)

Download Raspberry Pi Imager, flash Raspberry Pi OS (64-bit, Desktop) onto
a microSD card. Boot the Pi, complete first-run setup, connect to WiFi for
package downloads.

### 2. Set up the project's Python environment

The project uses Python 3.11 inside a pyenv-managed venv. Setup is a
one-time multi-step process — see the **Installing Python dependencies**
section under Detailed Setup. Allow about 30 minutes including the Python
3.11 build.

After setup, every time you work on this project:

```bash
cd ~/CapstoneUI
source venv/bin/activate    # pyenv switches Python; venv activates packages
```

You'll know you're in the right environment when your shell prompt has
`(venv)` in front of it AND `python --version` says `Python 3.11.10`.

### 3. Configure the Pi's Ethernet for the comms link

The Pi's `eth0` interface has no IP address by default. Give it a static one:

```bash
sudo nmcli connection modify netplan-eth0 ipv4.method manual ipv4.addresses 192.168.1.10/24 ipv4.gateway "" ipv6.method ignore
sudo nmcli connection down netplan-eth0
sudo nmcli connection up netplan-eth0
```

Verify with `ip addr show eth0` — you should see `inet 192.168.1.10/24`.

If your eth0 connection has a different name, run `nmcli connection show`
first and substitute the actual name in the commands above.

### 4. Flash the ClearCore sketch

Open `ClearCoreModbusTest.ino` in the Arduino IDE with the Teknic ClearCore
board package installed (Tools → Board → Boards Manager → search "ClearCore").
Make sure the static IP at the top of the file matches what the Pi expects:

```cpp
IPAddress STATIC_IP (192, 168, 1, 20);
```

Compile and upload. Open the Arduino Serial Monitor at 9600 baud and verify
you see the boot sequence:

```
[BOOT] Scan-to-Mill ClearCore Modbus TCP test
[BOOT] CLIENT_INFC = 88 bytes = 44 regs, cmd@w6
[MOTOR] M-0 enabled, step-and-dir
[MOTOR] limit switches on IO-2 (far) and IO-3 (home) and e-stop on IO-1
[NET] PHY link up
[NET] IP = 192.168.1.20
[NET] Modbus TCP listening on :502 unit 1
```

The motor stays still at boot now — motion only starts when the Pi sends
`CCMD_RUN_1` (via the Start Scan button). If you want to verify motor
hardware independently of the Pi, use `cc_modbus_test.py` on another
machine with `run` to trigger a sequence manually.

### 5. Connect the cable and run the test

Plug an Ethernet cable directly from the Pi's RJ-45 jack to the ClearCore's
ETHERNET port. No router or switch needed. Then on the Pi:

```bash
cd ~/CapstoneUI
source venv/bin/activate
python cc_modbus_test.py
```

(All `python3 ...` commands in this README are replaced by `python ...`
inside the venv. The venv's `python` is Python 3.11.10. Outside the venv,
`python` may be the wrong version or may not exist at all.)


## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Raspberry Pi 5                               │
│                                                                     │
│   ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐   │
│   │   UI.py      │    │ cc_modbus_test.py│    │  (future)       │   │
│   │  (PyQt6 +    │    │  (CLI tester)    │    │  Camera capture │   │
│   │   PyVista)   │    │                  │    │  pipeline       │   │
│   └──────┬───────┘    └────────┬─────────┘    └─────────────────┘   │
│          │                     │                                    │
│          └─────────┬───────────┘                                    │
│                    │                                                │
│           ┌────────▼─────────┐                                      │
│           │   pymodbus       │                                      │
│           │   ModbusTcpClient│                                      │
│           └────────┬─────────┘                                      │
│                    │ TCP/IP                                         │
│                    │                                                │
│              ┌─────▼──────┐                                         │
│              │   eth0     │ 192.168.1.10                            │
│              │  (RJ-45)   │                                         │
│              └─────┬──────┘                                         │
└────────────────────┼────────────────────────────────────────────────┘
                     │
               Ethernet cable
                     │
┌────────────────────┼─────────────────────────────────────────────────┐
│              ┌─────▼──────┐                                          │
│              │  ETHERNET  │ 192.168.1.20                             │
│              │   port     │                                          │
│              └─────┬──────┘                                          │
│                    │                                                 │
│           ┌────────▼─────────┐                                       │
│           │  EthernetServer  │ port 502                              │
│           │  (Arduino API)   │                                       │
│           └────────┬─────────┘                                       │
│                    │                                                 │
│           ┌────────▼─────────┐                                       │
│           │  MBAP parser     │                                       │
│           │  (hand-rolled)   │                                       │
│           └────────┬─────────┘                                       │
│                    │                                                 │
│           ┌────────▼─────────┐                                       │
│           │  CLIENT_INFC     │  ← single source of truth for         │
│           │  shared struct   │     register layout                   │
│           └────────┬─────────┘                                       │
│                    │                                                 │
│           ┌────────▼─────────┐    ┌────────────────┐                 │
│           │  Motion state    │───▶│  ConnectorM0   │                 |
│           │  machine         │    │  (StepGen)     │                 │
│           └──────────────────┘    └────────────────┘                 │
│                                                                      │
│                       Teknic ClearCore                               │
└──────────────────────────────────────────────────────────────────────┘
```

The Pi acts as the Modbus **master** (client). The ClearCore acts as the
Modbus **slave** (server). The Pi periodically polls the ClearCore for
status, and writes commands when the user clicks a button. There is no
unsolicited data flow from the ClearCore — every byte the Pi reads was
requested by the Pi.

### Why Modbus TCP and not something else

Modbus TCP is well-supported on both sides (pymodbus on Python, multiple
libraries on ClearCore), it's a well-established industrial protocol, and
the hand-rolled server on the ClearCore side is small enough (~150 lines)
to audit completely. Alternatives considered and rejected:

- **Raw TCP with custom protocol** — would require designing and debugging
  a framing layer from scratch, no library support
- **Modbus RTU over RS-485** — would work but requires RS-485 transceivers
  on both ends, an extra USB-RS485 adapter on the Pi side, and adds wires
- **OPC UA** — heavyweight, ClearCore library support is immature
- **MQTT** — fundamentally pub/sub, not request/response, awkward fit for
  command/status semantics

Modbus TCP gives request/response semantics natively, sits directly on
TCP/IP with no extra hardware, and the protocol is dead simple to debug
with Wireshark or even `nc` if you really need to.

### Why TCP and not the original Serial RTU plan

An earlier iteration of this project used Modbus RTU over the ClearCore's
COM-0 serial port via a USB-RS485 adapter on the Pi. That works fine, but
the test rig already had Ethernet ports on both devices and direct Ethernet
is cleaner: no adapter, no driver dependencies on the Pi for the serial
device, lower latency, and no risk of corrupted UART frames at high baud
rates. The protocol-level register map and command dispatch is identical
between RTU and TCP — only the transport layer differs.

---

## Detailed Setup — Raspberry Pi

This section assumes Raspberry Pi OS Bookworm or Trixie on a Pi 5. Other
Pi models and other Linux distros will work, but commands and package names
may differ.

### Installing Python dependencies

**Important:** Raspberry Pi OS Trixie ships Python 3.13 as its system Python,
but several of this project's dependencies (most notably Open3D) only have
ARM64 wheels available for Python 3.11. Trying to install them against the
system Python will fail with `Could not find a version that satisfies the
requirement` errors. The project therefore uses a dedicated Python 3.11
environment built with `pyenv`.

This is a one-time setup. Once it's done, you don't have to think about it
again — your venv is "the right Python" and everything inside it just works.

#### Step 1: Install pyenv build dependencies

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev
```

These are the libraries Python's source build needs. Most are also dragged
in by Open3D's runtime requirements, so installing them now saves a round
trip later.

#### Step 2: Install pyenv

```bash
curl https://pyenv.run | bash
```

Then add pyenv to your shell config:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
exec bash
```

Verify with `pyenv --version` — should print `pyenv 2.x.x`.

#### Step 3: Build Python 3.11

```bash
pyenv install 3.11.10
```

This compiles Python from source and takes 15–25 minutes on a Pi 5. Walk
away, come back when it's done. Verify with `pyenv versions` — should
list `3.11.10`.

#### Step 4: Create the project venv

```bash
cd ~/CapstoneUI
pyenv local 3.11.10        # writes .python-version, locks this dir to 3.11
python -m venv venv
source venv/bin/activate
python --version           # should say Python 3.11.10
```

The `pyenv local` command creates a `.python-version` file in `~/CapstoneUI`
that pyenv reads automatically. Whenever you `cd` into this directory, your
shell will pick up Python 3.11 instead of the system 3.13. **You still need
to `source venv/bin/activate`** to enter the venv itself — pyenv handles the
interpreter, venv handles the package isolation.

#### Step 5: Install Python packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This pulls everything from PyPI, with **piwheels** (`https://www.piwheels.org/simple`)
as a secondary index that provides prebuilt ARM64 wheels for things PyPI
doesn't have. Piwheels is the secret sauce that makes Open3D installable
on the Pi at all — without it, you'd be doing a multi-hour from-source
build that fails in a dozen different ways.

#### Step 6: Sanity check

```bash
python -c "
import sys; print('python:', sys.version.split()[0])
import open3d; print('open3d:', open3d.__version__)
import pyvista, pyvistaqt
print('pyvista:', pyvista.__version__, 'pyvistaqt:', pyvistaqt.__version__)
import vtkmodules.vtkCommonCore as cc
print('vtk:', cc.vtkVersion.GetVTKVersion())
import PyQt6
print('PyQt6: ok')
import pymodbus
print('pymodbus: ok')
"
```

Expected output (versions will drift over time but the structure should match):

```
python: 3.11.10
open3d: 0.18.0
pyvista: 0.46.3 pyvistaqt: 0.11.4
vtk: 9.5.2
PyQt6: ok
pymodbus: ok
```

If you see `OpenBLAS : munmap failed::` after the script finishes, ignore
it. It's a known harmless ARM64 OpenBLAS warning printed during process
shutdown. Your imports succeeded.

#### Notes on `requirements.txt`

The pinned versions in `requirements.txt` matter. They were chosen to
satisfy a tight web of constraints between PyVista, VTK, and Open3D that
took several evenings to figure out. Don't bump them casually.

```
# Python 3.11 ONLY — see README "Installing Python dependencies"
open3d==0.18.0          # piwheels has ARM64 wheel for py311 only
pyvista==0.46.3         # works with VTK 9.5+, embeds in PyQt6
pyvistaqt==0.11.4       # pairs with pyvista 0.46
numpy>=1.24.0
pymodbus>=3.5.0
PyQt6>=6.5.0
pyserial>=3.5
pyyaml>=6.0

# pyrealsense2 — install librealsense from source on the Pi, NOT via pip.
# See: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_raspbian.md
# pyrealsense2>=2.55.0
```

The historical reasons for each pin, in case you ever need to change them:

- **Open3D 0.18.0** — only ARM64 wheel piwheels has for Python 3.11. Newer
  versions (0.19+) require Python 3.12 or compile from source.
- **PyVista 0.46.3** — 0.43 is too old (uses removed VTK class
  `vtkCompositePolyDataMapper2`); 0.45 hard-pins `vtk<9.5` which piwheels
  doesn't have; 0.47 introduced an embedding regression on Pi where the
  viewport pops out as a separate window. 0.46 is the sweet spot.
- **VTK 9.5.2** — what piwheels offers and what PyVista 0.46 accepts. Not
  pinned explicitly in requirements.txt because PyVista's own dependency
  metadata picks it correctly.
- **PyQt6** — replaces apt's PyQt5. The fallback `import PyQt5` block in
  `UI.py` exists for portability but the canonical setup is PyQt6 from pip
  inside the venv.
  
## Detailed Setup — ClearCore

### Installing the ClearCore Arduino board package

1. Open the Arduino IDE (1.8.x or 2.x)
2. File → Preferences → Additional Boards Manager URLs → add:
   `https://raw.githubusercontent.com/Teknic-Inc/ClearCore-Arduino-wrapper/master/package_clearcore_index.json`
3. Tools → Board → Boards Manager → search "ClearCore" → Install
4. Tools → Board → Teknic → ClearCore

### Important: which ClearCore API are you using?

There are two ClearCore programming environments and they use different
APIs even though they target the same hardware:

- **Native C++ library** (Microchip Studio / Atmel Studio) — uses
  `EthernetTcpServer`, `EthernetTcpClient`, `IpAddress`, `EthernetMgr`,
  `ConnectorUsb.SendLine()`, etc. All under the `ClearCore::` namespace.
- **Arduino wrapper** (Arduino IDE) — uses standard Arduino names:
  `EthernetServer`, `EthernetClient`, `IPAddress`, `Ethernet.begin()`,
  `Serial.println()`, etc.

`ClearCoreModbusTest.ino` uses the **Arduino wrapper API**. If you see
compile errors like "EthernetTcpServer does not name a type", you're
either trying to compile this sketch in the wrong environment, or you've
mixed up the two APIs in your edits. Stick to the Arduino names.

### Configuring the static IP

Open `ClearCoreModbusTest.ino` and find the network config section near
the top:

```cpp
IPAddress STATIC_IP (192, 168, 1, 20);
IPAddress NETMASK   (255, 255, 255, 0);
IPAddress GATEWAY   (192, 168, 1, 1);
IPAddress DNS_IP    (192, 168, 1, 1);
```

These must match what the Pi side is configured to expect. If you change
the IP here, also change `MODBUS_DEFAULT_HOST` in `UI.py` and `DEFAULT_HOST`
in `cc_modbus_test.py`.

### Wiring

Direct Ethernet cable from the ClearCore's RJ-45 ETHERNET jack to the Pi's
Ethernet jack. No switch, no router. Modern RJ-45 hardware on both ends has
auto-MDIX so a regular cable works fine; you do not need a crossover cable.

For motion (optional for the comms test):

- Stepper or ClearPath-SD on **M-0** connector
- ClearCore powered with 24V on the COM/PWR terminals
- For TB6600 + generic stepper: tie M-0 STEP and DIR outputs to TB6600 PUL
  and DIR inputs respectively, with a common ground

### Verifying the ClearCore is alive

After flashing, plug the ClearCore's USB into any computer (Pi or laptop),
then open the Arduino Serial Monitor at 9600 baud. You should see the boot
sequence shown earlier in the Quick Start.

On the Pi, you can use `screen` instead:

```bash
ls /dev/ttyACM*               # find which port the ClearCore is on
sudo screen /dev/ttyACM0 9600 # connect at 9600 baud
```

Press the ClearCore's reset button to see a fresh boot. Exit screen with
`Ctrl+A` then `\` then `y`.

The critical line in the boot log is `[NET] IP = 192.168.1.20`. If that
shows a different address, the static IP didn't take effect — usually
because you uploaded an older sketch by mistake. Re-flash and re-check.

---

## Verifying the Comms Link

`cc_modbus_test.py` is the standalone test tool. It walks through three
layers in order and tells you exactly which one fails:

1. **TCP layer** — raw `socket()` connect to 192.168.1.20:502, no
   pymodbus involved. Failure here means a network problem.
2. **Modbus client open** — pymodbus wrapping the socket. Failure here
   is unusual and usually indicates a pymodbus version issue.
3. **Modbus protocol** — one round-trip read of HR0. Failure here means
   the ClearCore is reachable but the protocol-level conversation isn't
   working — wrong unit ID, malformed framing, etc.

After all three layers pass, the script drops into an interactive shell:

```
cc> status        # one-shot read of cur_posn, status flags, state, msg
cc> watch         # 4 Hz live updates until Ctrl+C
cc> stop          # send CCMD_STOP — verify motor halts
cc> run           # send CCMD_RUN_1 — restart sequence 1
cc> enab          # CCMD_ENAB_MTRS — enable motor drive
cc> disab         # CCMD_DISAB_MTRS — disable drive (software estop)
cc> move 5000     # CCMD_MOVE to absolute position 5000 steps
cc> raw 6         # read any single register by word address
cc> q             # quit
```

The `watch` command is the most useful for verifying everything works.
You should see `cur_posn` flipping between +5000 and −5000 every couple
of seconds, the `SCANNING` status flag set, and the `msg` field showing
ClearCore-side events as they happen.

For automation or scripts, run with `--once` to read status once and
exit without entering the shell:

```bash
python3 cc_modbus_test.py --once
```

---

## Running the Full UI

Once the comms test confirms the link works, you can launch the operator
interface:

```bash
QT_QPA_PLATFORM=xcb python3 UI.py
```

The UI brings up a PyQt6 window with three panels: scan controls on the
left, the 3D point cloud viewport in the middle, and status/log on the
right. The Modbus link to the ClearCore is established automatically at
startup — watch the log panel for the `[MODBUS] ClearCore link
ESTABLISHED` line.

**Known issue:** the current UI layout assumes a desktop-class display
(≥1280×800). On the official 7" Pi touchscreen (800×480) it's unusably
small. This is documented in Future Work below.

---

## Register Map Reference

The contract between the Pi and the ClearCore is the `CLIENT_INFC` struct
defined in `ClearCoreModbusTest.ino`. The Modbus register address of any
field is `offsetof(field) / 2` (i.e. byte offset in the packed struct,
divided by two because Modbus addresses are word-based).

| Word | Field | Type | Direction | Description |
|---:|---|---|---|---|
| 0–1 | `acc` | uint32 | Pi → CC | Acceleration, steps/sec² |
| 2–3 | `vel` | int32 | Pi → CC | Max velocity, steps/sec |
| 4–5 | `target_posn` | int32 | Pi → CC | Move target, absolute steps |
| 6 | `cmd` | uint16 | Pi → CC | Command code (see below) |
| 7–8 | `cur_posn` | int32 | CC → Pi | Current position, steps |
| 9 | `status` | uint16 | CC → Pi | Status bitfield (see below) |
| 10 | `msg_cnt` | uint16 | CC → Pi | Increments on each new debug msg |
| 11–42 | `msg` | char[64] | CC → Pi | Debug message text |
| 43 | `state` | uint16 | CC → Pi | State machine state (see below) |

Total: 44 holding registers (88 bytes).

### Command codes (`CCMTR_CMD`, written to word 6)

| Value | Name | Effect |
|---:|---|---|
| 0 | `CCMD_NONE` | No-op |
| 1 | `CCMD_ENAB_MTRS` | Enable motor drive |
| 2 | `CCMD_DISAB_MTRS` | Disable motor drive (software estop) |
| 3 | `CCMD_SET_ZERO` | Zero current position (not implemented in test) |
| 4 | `CCMD_MOVE` | Absolute move using acc/vel/target_posn |
| 5 | `CCMD_STOP` | Abrupt stop, drive stays enabled |
| 6 | `CCMD_RUN_1` | Start/restart sequence 1 (back-and-forth test) |
| 7 | `CCMD_ACK` | Reserved for future use |
| 8 | `CCMD_NACK` | Returned by ClearCore on unknown command |

### Status bits (`CCMTR_STATUS`, word 9)

| Bit | Mask | Name | Meaning |
|---:|---:|---|---|
| 0 | 0x01 | `READY` | Drive enabled |
| 1 | 0x02 | `MOVING` | Motion in progress |
| 2 | 0x04 | `HOMED` | Homing complete (unused in test) |
| 3 | 0x08 | `FAULT` | Fault latched |
| 4 | 0x10 | `ESTOP` | E-stop latched |
| 5 | 0x20 | `SCANNING` | Sequence 1 running |
| 6 | 0x40 | `HLFB` | ClearPath HLFB OK (unused in test) |

### State codes (`CCMTR_STATE`, word 43)

| Value | Name |
|---:|---|
| 0 | `INIT` |
| 1 | `IDLE` |
| 2 | `ENABLED` |
| 3 | `RUNNING` |
| 4 | `STOPPED` |
| 5 | `FAULT` |

### Endianness — important

ClearCore is little-endian ARM Cortex-M4. The `CLIENT_INFC` struct is
accessed via `reinterpret_cast<uint16_t*>` on the ClearCore side, so
32-bit fields appear on the wire **low-word-first**. For example, the
value `0x12345678` in `acc` is transmitted as register pair
`[0x5678, 0x1234]` (low word at the lower address).

The Pi side handles this correctly via `pack_u32()` / `pack_i32()` /
`unpack_i32()` helpers in both `UI.py` and `cc_modbus_test.py`.
**Do not bypass these helpers** for new 32-bit fields, or you'll get
mysterious value corruption that's painful to debug.

---

## Common Errors and Fixes

These are the things that actually went wrong during initial bring-up.
If you hit any of them, the fix is here. They're roughly ordered by
how early in the workflow you'll hit them.

### `error: externally-managed-environment` when running pip

Add `--break-system-packages` to your pip command:

```bash
pip install pymodbus --break-system-packages
```

Raspberry Pi OS marks the system Python as externally-managed under PEP
668. The flag installs to your user site-packages instead of the system
site-packages, which is safe.

### `'EthernetTcpServer' does not name a type` when compiling the ClearCore sketch

You're trying to compile the Arduino-API sketch using native ClearCore
library APIs. Make sure:

- You're using the Arduino IDE, not Microchip Studio or Atmel Studio
- The Tools → Board menu has "Teknic ClearCore" selected
- You haven't pasted in any code from the native library examples

The Arduino API uses `EthernetServer`, `EthernetClient`, `Ethernet.begin()`,
not `EthernetTcpServer`/`EthernetTcpClient`/`EthernetMgr.Setup()`.

### `BadWindow (invalid Window parameter)` when launching UI.py on Pi 5

The current `UI.py` sets `QT_QPA_PLATFORM=xcb` automatically at the top
of the file, before any Qt imports. If you're still seeing this error,
either:

1. You're running an old version of `UI.py` that doesn't have the env
   var set. Pull the latest from the repo.
2. You've modified the file and accidentally moved the env var setting
   to *after* an `import PyQt6` or `import pyvista` line. Qt only reads
   the env var on first import — it must be set first. Check the top
   of the file.

If neither applies, you can still force it from the command line as a
sanity check:

```bash
QT_QPA_PLATFORM=xcb python UI.py
```

If that works but `python UI.py` alone doesn't, the env var line in the
file isn't taking effect. Compare against the version in the repo.

### `[MODBUS] Failed to reach ClearCore at 192.168.1.20:502` even though everything is plugged in

Almost certainly the Pi's `eth0` doesn't have an IPv4 address. Run:

```bash
ip addr show eth0
```

If you only see an `inet6` line and no `inet` line, configure a static IP:

```bash
sudo nmcli connection modify netplan-eth0 ipv4.method manual ipv4.addresses 192.168.1.10/24 ipv4.gateway "" ipv6.method ignore
sudo nmcli connection down netplan-eth0
sudo nmcli connection up netplan-eth0
```

If your eth0 connection has a different name, find it with `nmcli
connection show` and substitute.

### `From <campus-ip> Destination Net Unreachable` when pinging the ClearCore

Same root cause as above. The Pi has no route to 192.168.1.0/24, so it
sent your ping out over WiFi (`wlan0`), the campus router got it, looked
in its routing table, didn't find anything, and bounced an unreachable
back. The fix is the same nmcli command — give `eth0` an IP on the
192.168.1.0/24 subnet.

### Sketch compiles and uploads but the ClearCore boot log shows the wrong IP

The `STATIC_IP` constant near the top of the .ino file didn't get changed,
or you uploaded an older version of the sketch. Verify the constant matches
what UI.py expects, then re-flash.

### Sketch boots fine but the motor doesn't move

This is a motion problem, not a comms problem. Check in this order:

1. ClearCore 24V power LED is on
2. Stepper driver power LED is on
3. M-0 connector wiring is correct (STEP, DIR, GND)
4. Motor isn't physically jammed
5. The serial log shows `[MOTOR] M-0 enabled, step-and-dir`
6. Sequence 1 actually started — look for `[SEQ1] started` in the log

If all of the above are healthy and the motor still doesn't move, check
the TB6600 (or equivalent) DIP switches for current and microstep settings,
and verify the motor is wired correctly (coil A and coil B, not A and B
of the same coil — that's the most common wiring mistake with TB6600s).

### `from pymodbus.client import ModbusTcpClient` raises `SyntaxError`

You have a typo in `UI.py`, almost certainly a stray character on the
`try:` line above the import. The Python parser sees `try:` followed by
something invalid, decides the try block is empty, and complains that
there's no matching `except`. Look at lines 14–16 and check the `try:`
line is exactly `try:` with nothing after the colon.

### `pyvista` install hangs or fails on the Pi

PyVista pulls in VTK as a dependency, and VTK is a chunky download even
when piwheels has a prebuilt wheel. If pip appears to hang, give it 5–10
minutes before assuming it's broken. If it actually fails, check disk
space with `df -h /` — VTK plus dependencies want about 500 MB free.

### `python3 UI.py` works but `python UI.py` says `ModuleNotFoundError`

`python` and `python3` are different interpreters on Raspberry Pi OS,
and pymodbus was installed for `python3`. Always use `python3` explicitly.

### `Could not find a version that satisfies the requirement open3d>=0.18.0`

You're trying to `pip install open3d` against Python 3.13 (the system Python
on Raspberry Pi OS Trixie). Open3D's piwheels wheels are built for Python
3.11 only. Solution: use the project's pyenv-managed Python 3.11 venv. See
the "Installing Python dependencies" section for setup.

If you've already done the venv setup but still see this error, run
`python --version` — if it says 3.13, your venv isn't activated. Run
`source venv/bin/activate` from `~/CapstoneUI` and try again.

### `Could not find a version that satisfies the requirement pyrealsense2>=2.55.0`

`pyrealsense2` doesn't ship ARM64 wheels at all. The Python bindings have
to be installed by building librealsense from source on the Pi. See:
https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_raspbian.md

This is a known multi-hour build. Until you actually need camera capture,
leave the line commented out in `requirements.txt` and the rest of the
project will work fine. The scan worker currently emits placeholder data
while pyrealsense2 isn't installed.

### `ImportError: cannot import name 'vtkCompositePolyDataMapper2' from 'vtkmodules.vtkRenderingOpenGL2'`

You have an old PyVista (0.43.x) trying to use a class that was removed
in VTK 9.4. Either the venv was created with the wrong PyVista pin or
something downgraded it. Fix:

```bash
pip install "pyvista==0.46.3" "pyvistaqt==0.11.4"
```

### VTK viewport pops out as a separate window instead of embedding in the UI

Two possible causes, both addressed in the current `UI.py`:

1. **Viewport initialized too early.** PyVista's QtInteractor must be
   created *after* the Qt window is shown, not during `__init__`. The
   current code uses `showEvent` to defer init via `QTimer.singleShot`.
   If you've reverted that, the embedding will break.

2. **Qt running on Wayland instead of X11.** PyVista's VTK rendering only
   works correctly with the xcb (X11) backend on Pi 5. The current `UI.py`
   forces this with `os.environ["QT_QPA_PLATFORM"] = "xcb"` at the very
   top of the file, before any Qt or PyVista imports. **Order matters** —
   Qt reads the env var exactly once when first imported.

### `[VIZ] _init_viewport FAILED: Install IPython to display with pyvista in a notebook`

You have a stray `pv.set_jupyter_backend(None)` call somewhere in
`_init_viewport`. PyVista 0.46 tries to validate the backend by importing
IPython even when passing `None`. Just delete the line — you don't need it,
and it doesn't do what its name suggests.

### `ValueError: Empty meshes cannot be plotted` when clicking Point Cloud or Mesh

PyVista 0.46 refuses to plot meshes with zero points. The placeholder
cloud has zero points until camera data starts flowing. Fix at app
startup, in the `__main__` block:

```python
pv.global_theme.allow_empty_mesh = True
```

The Mesh button additionally needs a guard against running surface
reconstruction on too-few points. See `_set_view_mode` in `UI.py`.

### `OpenBLAS : munmap failed:: Invalid argument` after a script exits

Harmless. OpenBLAS prints this on ARM64 during process teardown when its
memory release call hits a kernel alignment edge case. Your script's
output before this line is the real result and is correct. Ignore it.
---

## Project Structure

```
CapstoneUI/
├── UI.py                      # Main PyQt6 operator interface
├── cc_modbus_test.py          # Standalone CLI Modbus tester
├── ClearCoreModbusTest.ino    # ClearCore Arduino sketch (the slave)
└── README.md                  # This file
```

That's it. Three source files plus this README. The simplicity is
intentional — the comms test is meant to be auditable end-to-end by one
person in one sitting.

### What's in each file

**`UI.py`** (~1130 lines) — The PyQt6 operator interface. Contains the
main window, the 3D viewport (PyVista/VTK), the scan worker thread, and
the `ClearCoreModbus` QThread that owns the Modbus client. Most of the
file is UI layout code; the comms layer is roughly 250 lines starting
near line 245.

**`cc_modbus_test.py`** (~410 lines) — Standalone CLI tester for the
Modbus link. Has zero dependencies on PyQt or PyVista, so it works even
if the UI is broken. Drops into an interactive shell after verifying
the link.

**`ClearCoreModbusTest.ino`** (~575 lines) — The ClearCore-side sketch.
Implements a Modbus TCP server with a hand-rolled MBAP parser, a
back-and-forth motion test sequence, the `CLIENT_INFC` shared register
struct, and command dispatch. Uses the ClearCore Arduino wrapper API.

---

## Future Work / Known Limitations

These are the things this comms test does NOT do, in rough priority order
of "what to tackle next."


### No real point cloud capture yet

`UI.py`'s viewport currently shows an empty point cloud placeholder. The
Intel RealSense D405 camera capture pipeline is not yet wired in.
`_update_pointcloud_live()` is the hook point — it accepts a `pv.PolyData`
and renders it. Plug your camera capture thread in there.

### No real scan motion profile (partially complete)

The ClearCore sketch's `CCMD_RUN_1` now runs a real home→scan→return cycle
driven by limit switches on IO-2/IO-3 instead of the old back-and-forth
placeholder. What's still missing is **coordinated motion+capture**: the
camera needs to grab frames at known carriage positions during the scan
sweep. The `CCMD_MOVE` path already accepts acc/vel/target if you want to
drive discrete stop-and-capture moves from the Pi side instead of a
continuous sweep.

### Homing (complete)

Limit switches on IO-2 (far) and IO-3 (home) are now wired in. The
`MS_HOMING` state drives toward the home switch and zeroes
`PositionRefCommanded` when it trips, establishing a canonical origin
for subsequent scans. `CCMD_SET_ZERO` is still a no-op but is no longer
needed for the basic scan workflow.

### G-code export not yet implemented

The Scan-to-Mill workflow ends with exporting a toolpath to the
Genmitsu CNC. None of that exists in this codebase yet.

### E-stop — software path done, hardware cutoff still needed

An e-stop button wired to `ConnectorIO1` is now polled every loop iteration
by `serviceEstop()`. On press it halts the motor (`MoveStopAbrupt` +
`EnableRequest(false)`), latches the `ESTOP` status bit, and drives a
red banner on the Pi UI via the Modbus status channel. Release clears the
banner and unlocks START for a manual re-arm. Command filter in
`handleCommand()` rejects anything except STOP/DISABLE while engaged.

What's still missing for real safety compliance: a **hardware contactor**
in series with the 24V drive supply, wired to the same physical button.
Software e-stop is valid functional behavior but should never be the only
thing between the operator and the motor. The `panicButtonISR()` hook on
DI-6 is also still available as a second software path if you want
interrupt-driven halting for cases where the main loop might be blocked.

### No authentication or encryption

Modbus TCP has no built-in authentication or encryption. Anyone on the
same network segment can read or write the registers. Fine for a direct
cable between two devices on a benchtop, NOT fine for anything connected
to a shared network. If you ever put this on a real network, put it
behind a firewall that only allows the Pi's IP to reach port 502 on the
ClearCore.

### The MBAP parser is single-client

`serviceModbus()` only handles one TCP client at a time. If a second
client tries to connect, it'll be ignored until the first one disconnects.
For this test rig that's fine — there's only ever one Pi talking to one
ClearCore — but it would need to be reworked to use the Arduino
`server.accept()` API and a client array if you ever want multiple
masters. See the Arduino `EthernetServerAccept` reference for how.

### Adding SD Card Slot
An Adafruit SD Card holder/reader will be added so that the SD can be taken to the Genmitsu 3020-pro max v2 CNC in Endeavor 110 for operation and testing

## Credits and Contact

**Project:** Scan-to-Mill capstone, Oklahoma State University
School of Electrical and Computer Engineering, 2025–2026

**Faculty advisor:** Dr. Piao, ENDEAVOR Room 220

**Hardware:**
- Teknic Inc. — ClearCore PLC and ClearPath-SD servos
- Intel — RealSense D405 depth camera
- Genmitsu — 3020 Pro Max V2 CNC router

**References:**
- Teknic ClearCore Arduino wrapper:
  https://github.com/Teknic-Inc/ClearCore-Arduino-wrapper
- Teknic Modbus reference (RTU example):
  https://teknic-inc.github.io/ClearCore-library/
- pymodbus documentation:
  https://pymodbus.readthedocs.io/
- Modbus TCP specification:
  https://modbus.org/docs/Modbus_Messaging_Implementation_Guide_V1_0b.pdf

---

## Quick Reference Card

For when you've forgotten everything and just need to get back to a
working state. Tape this to the wall.

```bash
# Pi network setup (one-time)
sudo nmcli connection modify netplan-eth0 ipv4.method manual \
  ipv4.addresses 192.168.1.10/24 ipv4.gateway "" ipv6.method ignore
sudo nmcli connection down netplan-eth0
sudo nmcli connection up netplan-eth0

# ClearCore IP (in ClearCoreModbusTest.ino)
IPAddress STATIC_IP (192, 168, 1, 20);

# Ping test
ping -c 4 192.168.1.20

# Read ClearCore boot log
sudo screen /dev/ttyACM0 9600
# (exit with Ctrl+A then \ then y)

# Activate the Python 3.11 venv (every session)
cd ~/CapstoneUI
source venv/bin/activate
python --version    # should say 3.11.10

# Run the comms test
python cc_modbus_test.py

# Run the full UI (xcb is set automatically inside UI.py)
python UI.py
```
