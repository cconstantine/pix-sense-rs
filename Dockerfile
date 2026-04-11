FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    pkg-config \
    libssl-dev \
    libusb-1.0-0-dev \
    libclang-dev \
    nasm \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Build libjpeg-turbo from source (JetPack image lacks -dev headers)
RUN git clone --depth 1 --branch 3.0.1 https://github.com/libjpeg-turbo/libjpeg-turbo.git /tmp/libjpeg-turbo \
    && cd /tmp/libjpeg-turbo && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build build -j$(nproc) && cmake --install build \
    && ldconfig && rm -rf /tmp/libjpeg-turbo

# Build librealsense2 v2.56.5 from source (no arm64 packages available)
RUN git clone --depth 1 --branch v2.56.5 https://github.com/IntelRealSense/librealsense.git /tmp/librealsense \
    && mkdir /tmp/librealsense/build && cd /tmp/librealsense/build \
    && cmake .. \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_GRAPHICAL_EXAMPLES=OFF \
        -DBUILD_TOOLS=OFF \
        -DFORCE_RSUSB_BACKEND=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && rm -rf /tmp/librealsense

# Download the ONNX model if not present (runs at container start)
# The model can't be baked into the image since /app is volume-mounted
COPY <<'EOF' /usr/local/bin/ensure-model.sh
#!/bin/bash
MODEL_PATH="/app/models/scrfd_10g_bnkps.onnx"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading SCRFD model..."
    mkdir -p /app/models
    curl -L -o "$MODEL_PATH" \
        "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx"
    echo "Model downloaded to $MODEL_PATH"
else
    echo "Model already present at $MODEL_PATH"
fi
EOF
RUN chmod +x /usr/local/bin/ensure-model.sh

# Entrypoint fixes USB permissions via sudo, then execs as dev
COPY <<'ENTRY' /usr/local/bin/entrypoint.sh
#!/bin/bash
sudo chmod a+rw /dev/bus/usb/*/* 2>/dev/null || true
exec "$@"
ENTRY
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create non-root user matching host UID/GID
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID dev \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash dev \
    && usermod -aG video,plugdev dev \
    && echo "dev ALL=(root) NOPASSWD: /bin/chmod" > /etc/sudoers.d/dev-usb \
    && chmod 0440 /etc/sudoers.d/dev-usb

# Install Rust as the dev user
USER dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/dev/.cargo/bin:${PATH}"

# Add WASM target and install trunk for client builds
RUN rustup target add wasm32-unknown-unknown \
    && cargo install cargo-binstall \
    && cargo binstall trunk

WORKDIR /app
EXPOSE 3000


ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
