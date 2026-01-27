#!/bin/bash
# Install OpenCAMLib on Raspberry Pi 5 (ARM64)
# May need to build from source if no prebuilt wheel exists

set -e

echo "=== OpenCAMLib Installation ==="

# Try pip install first (may have ARM64 wheels now)
echo "Attempting pip install..."
if pip install opencamlib 2>/dev/null; then
    echo "✓ OpenCAMLib installed via pip"
    python -c "import ocl; print(f'Version: {ocl.__version__}')" 2>/dev/null || echo "Installed successfully"
    exit 0
fi

echo "pip install failed, building from source..."

# Install build dependencies
echo "Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libboost-python-dev \
    python3-dev \
    git

# Clone OpenCAMLib
WORK_DIR=$(mktemp -d)
cd "$WORK_DIR"

echo "Cloning OpenCAMLib..."
git clone https://github.com/aewallin/opencamlib.git
cd opencamlib

# Create build directory
mkdir build && cd build

# Configure with Python bindings
echo "Configuring build..."
cmake .. \
    -DBUILD_PY_LIB=ON \
    -DUSE_OPENMP=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python3)

# Build (use multiple cores)
echo "Building (this may take a while)..."
make -j$(nproc)

# Install
echo "Installing..."
sudo make install

# Install Python package
cd ../src/pythonlib
pip install .

# Verify installation
echo "Verifying installation..."
if python -c "import ocl; print('OpenCAMLib imported successfully')"; then
    echo "✓ OpenCAMLib installed successfully"
else
    echo "✗ Installation may have failed - check errors above"
    exit 1
fi

# Cleanup
cd /
rm -rf "$WORK_DIR"

echo "=== Installation Complete ==="