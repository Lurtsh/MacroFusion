FROM nvidia/cuda:13.3.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libglvnd-dev \
    libglx-dev \
    libopengl-dev \
    libtbb-dev \
    libx11-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    libxext-dev \
    libxrender-dev \
    libxfixes-dev \
    libxxf86vm-dev \
    ninja-build \
    pkg-config \
    python3 \
    python3-jinja2 \
    && rm -rf /var/lib/apt/lists/*

RUN git config --system --add safe.directory /workspace/MacroFusion \
    && git config --system --add safe.directory /workspace/MacroFusion/.git

WORKDIR /workspace/MacroFusion

CMD ["/bin/bash"]
