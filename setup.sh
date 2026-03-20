#!/bin/bash
set -euo pipefail

echo "=== pix-sense-rs system dependency setup ==="

# Install librealsense2
echo ""
echo "--- Installing librealsense2 ---"
if ! dpkg -s librealsense2-dev &>/dev/null; then
    # Add Intel RealSense repository
    sudo mkdir -p /etc/apt/keyrings
    curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
        | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
    echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
        https://librealsense.intel.com/Debian/apt-repo \
        $(lsb_release -cs) main" \
        | sudo tee /etc/apt/sources.list.d/librealsense.list
    sudo apt-get update
    sudo apt-get install -y librealsense2-dkms librealsense2-dev librealsense2-utils
    echo "librealsense2 installed successfully"
else
    echo "librealsense2 already installed"
fi

# Install ONNX Runtime with CUDA (ort crate can download this automatically,
# but having it system-wide is more reliable)
echo ""
echo "--- ONNX Runtime ---"
echo "The 'ort' crate will download ONNX Runtime automatically during build."
echo "If you prefer a system install, download from:"
echo "  https://github.com/microsoft/onnxruntime/releases"

# Download SCRFD face detection ONNX model
echo ""
echo "--- Downloading SCRFD face detection model ---"
MODEL_DIR="models"
SCRFD_PATH="${MODEL_DIR}/scrfd_10g_bnkps.onnx"
if [ ! -f "$SCRFD_PATH" ]; then
    mkdir -p "$MODEL_DIR"
    echo "Downloading SCRFD 10G (with keypoints) ONNX model..."
    curl -L -o "$SCRFD_PATH" "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx"
    echo "Model saved to $SCRFD_PATH"
else
    echo "Model already exists at $SCRFD_PATH"
fi

echo ""
echo "=== Setup complete ==="
echo "Build with: cargo build --release"
echo "Run with:   cargo run --release"
