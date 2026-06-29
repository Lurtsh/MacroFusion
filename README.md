# MacroFusion

MacroFusion is a CUDA-focused student implementation of a
[KinectFusion](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf)-style
3D reconstruction pipeline using a dense TSDF volume.

## Dataset

The viewer currently expects the TUM RGB-D `freiburg1_xyz` sequence at
`data/tum/rgbd_dataset_freiburg1_xyz`. Download it from the repository root with:

```bash
./scripts/download_tum_dataset.sh
```

The downloader requires `curl` and `tar` on the host.

The script also accepts a TUM sequence name and dataset group, for example:

```bash
./scripts/download_tum_dataset.sh rgbd_dataset_freiburg1_desk freiburg1
```

## Build

MacroFusion is developed on Ubuntu 24.04 with CUDA 13.3. The Docker
development environment provides that baseline. Other host distributions and
CUDA Toolkit versions are not blocked by CMake but are user-supported. A Turing,
Ampere, Ada, or Blackwell desktop GPU and a working NVIDIA driver are required.

### NVIDIA container

Install Docker Engine, the Compose plugin, and NVIDIA Container Toolkit on the
Ubuntu 24.04 host. The single Compose service requests all GPUs and enables the
NVIDIA `compute`, `utility`, `graphics`, and `display` driver capabilities:

```bash
docker compose build dev
docker compose run --rm dev cmake -S . -B build -G Ninja
docker compose run --rm dev cmake --build build --parallel
```

Compose stores `build/`, including CMake `FetchContent` dependencies, in the
`macrofusion-build` named volume. It persists across temporary containers and is
isolated from the source checkout. X11/XWayland is forwarded through `DISPLAY`,
the X11 socket, and the active `XAUTHORITY` file. Set `XAUTHORITY` explicitly if
it is not the default `~/.Xauthority`.

Run the viewer only from a native Linux X11/XWayland host:

```bash
docker compose run --rm dev ./build/macrofusion
```

The viewer rejects any OpenGL context whose `GL_VENDOR` is not NVIDIA. WSL2 and
Docker Desktop are limited to configuring and building when they can provide the
required NVIDIA CUDA environment; the interactive viewer requires a native Linux
X11/XWayland host.

### CMake Options

Pass custom values during the configure step with `-D<name>=<value>`.

| Option | Default | Values | Purpose |
| --- | --- | --- | --- |
| `CMAKE_CUDA_ARCHITECTURES` | `75;86;89;120` | CMake CUDA architecture list | Compile for Turing, Ampere, Ada, and Blackwell desktop GPUs. |

## Changelog

### Latest Snapshot: `week-1` - Host Tracking Viewer

- Added synchronized TUM RGB-D acquisition and host preprocessing for filtered
  depth, vertex maps, normal maps, and the base tracking pyramid.
- Added a GLFW/OpenGL/Dear ImGui viewer for RGB, filtered depth, vertex XYZ, and
  normal-map panes.
- Updated Linux Docker launch to use X11/XWayland authentication and NVIDIA
  graphics passthrough.

## License

Copyright (c) 2026 MacroFusion contributors. Released under the MIT License.
