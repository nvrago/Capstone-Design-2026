#!/bin/bash
# one-time setup script for raspberry pi 5 deployment
#
# run this after cloning the repo and building the venv:
#   cd ~/Capstone-Design-2026
#   chmod +x deploy/setup_pi.sh
#   ./deploy/setup_pi.sh

set -e

REPO_DIR="$HOME/Capstone-Design-2026"
VENV_DIR="$HOME/pipeline-env"
SERVICE_FILE="deploy/scantocnc.service"

echo "=== scan-to-cnc pi deployment ==="

# check venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: venv not found at $VENV_DIR"
    echo "create it first: python3 -m venv $VENV_DIR"
    exit 1
fi

# check we're in the repo
if [ ! -f "$REPO_DIR/src/pipeline.py" ]; then
    echo "ERROR: repo not found at $REPO_DIR"
    exit 1
fi

# ensure user is in dialout group (serial port access)
if ! groups | grep -q dialout; then
    echo "adding $USER to dialout group for serial access..."
    sudo usermod -a -G dialout $USER
    echo "NOTE: log out and back in for group change to take effect"
fi

# create data directories
echo "creating data directories..."
mkdir -p "$REPO_DIR/data/reference"
mkdir -p "$REPO_DIR/data/runs"

# update service file with correct username if not 'pi'
CURRENT_USER=$(whoami)
if [ "$CURRENT_USER" != "pi" ]; then
    echo "adjusting service file for user: $CURRENT_USER"
    sed -i "s|User=pi|User=$CURRENT_USER|g" "$REPO_DIR/$SERVICE_FILE"
    sed -i "s|Group=pi|Group=$CURRENT_USER|g" "$REPO_DIR/$SERVICE_FILE"
    sed -i "s|/home/pi|/home/$CURRENT_USER|g" "$REPO_DIR/$SERVICE_FILE"
fi

# install systemd service
echo "installing systemd service..."
sudo cp "$REPO_DIR/$SERVICE_FILE" /etc/systemd/system/scantocnc.service
sudo systemctl daemon-reload
sudo systemctl enable scantocnc

# udev rule for consistent serial device names
echo "installing udev rules for USB serial devices..."
cat << 'EOF' | sudo tee /etc/udev/rules.d/99-scantocnc.rules
# ClearCore, assigns /dev/clearcore when plugged in
# adjust idVendor/idProduct after running: udevadm info -a /dev/ttyUSB0
SUBSYSTEM=="tty", ATTRS{idVendor}=="2890", ATTRS{idProduct}=="0001", SYMLINK+="clearcore"

# GRBL CNC controller, assigns /dev/grbl
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="grbl"
EOF
sudo udevadm control --reload-rules

echo ""
echo "=== setup complete ==="
echo ""
echo "what happens on boot now:"
echo "  1. systemd starts scantocnc.service"
echo "  2. pipeline server starts, listens on localhost:5001"
echo "  3. GUI connects to localhost:5001"
echo "  4. GUI sends JSON commands, pipeline runs"
echo ""
echo "commands:"
echo "  sudo systemctl start scantocnc     # start now"
echo "  sudo systemctl status scantocnc    # check status"
echo "  journalctl -u scantocnc -f         # live logs"
echo "  sudo systemctl restart scantocnc   # restart after code changes"
echo ""
echo "IMPORTANT: update the udev vendor/product IDs in"
echo "  /etc/udev/rules.d/99-scantocnc.rules"
echo "after plugging in your actual ClearCore and GRBL devices."
echo "run 'udevadm info -a /dev/ttyUSB0' to find the right IDs."
echo ""
echo "if you changed the udev rules, the serial port paths"
echo "in config/machine.yaml should use /dev/clearcore and /dev/grbl"
echo "instead of /dev/ttyUSB0 and /dev/ttyUSB1"