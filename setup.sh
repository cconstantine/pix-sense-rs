#!/bin/bash
set -euo pipefail

echo "=== pix-sense-rs system dependency setup ==="

# Install librealsense2
# echo ""
# echo "--- Installing librealsense2 ---"
# if ! dpkg -s librealsense2-dev &>/dev/null; then
#     # Add Intel RealSense repository
#     sudo mkdir -p /etc/apt/keyrings
#     curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
#         | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
#     echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
#         https://librealsense.intel.com/Debian/apt-repo \
#         $(lsb_release -cs) main" \
#         | sudo tee /etc/apt/sources.list.d/librealsense.list
#     sudo apt-get update
#     sudo apt-get install -y librealsense2-dkms librealsense2-dev librealsense2-utils
#     echo "librealsense2 installed successfully"
# else
#     echo "librealsense2 already installed"
# fi

MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

# Download SCRFD face detection model (used for landmark extraction in stage 2)
echo ""
echo "--- Downloading SCRFD face detection model ---"
SCRFD_PATH="${MODEL_DIR}/scrfd_10g_bnkps.onnx"
if [ ! -f "$SCRFD_PATH" ]; then
    echo "Downloading SCRFD 10G (with keypoints) ONNX model..."
    curl -L -o "$SCRFD_PATH" "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx"
    echo "Model saved to $SCRFD_PATH"
else
    echo "Model already exists at $SCRFD_PATH"
fi

# Download YOLOv8n head detection model (stage 1 — finds heads at longer range and in IR)
echo ""
echo "--- Downloading YOLOv8n head detection model ---"
YOLO_PATH="${MODEL_DIR}/yolov8n_head.onnx"
if [ ! -f "$YOLO_PATH" ]; then
    echo "Downloading YOLOv8n head detection ONNX model..."
    curl -L -o "$YOLO_PATH" "https://huggingface.co/trysem/YOLOv8_head_detector/resolve/main/yolo-head-nano.onnx"
    echo "Model saved to $YOLO_PATH"
else
    echo "Model already exists at $YOLO_PATH"
fi

echo ""
echo "=== Setup complete ==="
echo "Build with: cargo build -p pix-sense-server --release"
echo "Run with:   cargo run   -p pix-sense-server --release"
