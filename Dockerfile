FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS deps

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

# Download ONNX models if not present (runs at container start)
# Models are volume-mounted at /app/models so they persist across rebuilds
COPY <<'EOF' /usr/local/bin/ensure-model.sh
#!/bin/bash
mkdir -p /app/models

SCRFD_PATH="/app/models/scrfd_10g_bnkps.onnx"
if [ ! -f "$SCRFD_PATH" ]; then
    echo "Downloading SCRFD model..."
    curl -L -o "$SCRFD_PATH" \
        "https://huggingface.co/DIAMONIK7777/antelopev2/resolve/main/scrfd_10g_bnkps.onnx"
    echo "SCRFD model downloaded to $SCRFD_PATH"
else
    echo "SCRFD model already present at $SCRFD_PATH"
fi

YOLO_PATH="/app/models/yolov8n_head.onnx"
if [ ! -f "$YOLO_PATH" ]; then
    echo "Downloading YOLOv8n head model..."
    curl -L -o "$YOLO_PATH" \
        "https://huggingface.co/trysem/YOLOv8_head_detector/resolve/main/yolo-head-nano.onnx"
    echo "YOLOv8n model downloaded to $YOLO_PATH"
else
    echo "YOLOv8n model already present at $YOLO_PATH"
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

WORKDIR /app
RUN chown dev:dev /app
EXPOSE 3000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


FROM deps AS rust

# Install Rust as the dev user
USER dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/dev/.cargo/bin:${PATH}"

# Add WASM target and install trunk for client builds
RUN rustup target add wasm32-unknown-unknown \
    && curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash \
    && cargo binstall trunk


FROM rust AS builder

COPY --chown=dev:dev . /app/
RUN --mount=type=cache,uid=1000,gid=1000,target=/home/dev/.cargo/registry \
    --mount=type=cache,uid=1000,gid=1000,target=/home/dev/.cargo/git \
    --mount=type=cache,uid=1000,gid=1000,target=/app/target \
    cd /app/client && trunk build --release
RUN --mount=type=cache,uid=1000,gid=1000,target=/home/dev/.cargo/registry \
    --mount=type=cache,uid=1000,gid=1000,target=/home/dev/.cargo/git \
    --mount=type=cache,uid=1000,gid=1000,target=/app/target \
    RUSTFLAGS="-C link-arg=-Wl,--allow-shlib-undefined" cargo build -p pix-sense-server --release \
    && cp target/release/pix-sense-server /app/pix-sense-server


FROM deps AS prod

COPY --from=builder /app/pix-sense-server /app/pix-sense-server
COPY --from=builder /app/client/dist /app/client/dist
COPY --from=builder /app/migrations /app/migrations

CMD ["bash", "-c", "ensure-model.sh && /app/pix-sense-server"]


FROM rust AS dev

CMD ["bash"]
