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

### Linux Container Quick Start

For convenience on Linux desktops, these commands create the development image,
build the MacroFusion viewer, verify that Docker sees hardware OpenGL, and run
the viewer through Docker:

```bash
docker compose build dev
docker compose run --rm dev bash -lc \
  "cmake -S . -B build -G Ninja -DMACROFUSION_GLFW_LINUX_BACKEND=X11 -DMACROFUSION_BUILD_TESTS=OFF && cmake --build build --parallel"
docker compose run --rm dev glxinfo -B
./scripts/download_tum_dataset.sh
docker compose run --rm dev ./build/macrofusion
```

`glxinfo -B` should report `direct rendering: Yes`, the host GPU renderer, and
an OpenGL 4.6 core profile before launching the viewer.

Optionally, build and run the tests in the container to verify the project runtime:

```bash
docker compose run --rm dev bash -lc \
  "cmake -S . -B build -G Ninja -DMACROFUSION_GLFW_LINUX_BACKEND=X11 -DMACROFUSION_BUILD_TESTS=ON && cmake --build build --parallel && ctest --test-dir build --output-on-failure"
```

### Windows Launch

Docker Desktop on Windows can build and test the Linux container, but it does not
provide the same Linux `/dev/dri` Intel/Mesa OpenGL path used by the viewer on a
Linux host. Use one of these paths:

- For build and tests only, run the same Docker commands from PowerShell,
  Windows Terminal, or a WSL shell.
- For the interactive viewer, run the project inside a WSL2 Ubuntu environment
  with WSLg and the required native build dependencies, or run on a Linux host.

When running natively in WSL2, configure the viewer with the Linux backend exposed
by WSLg:

```bash
cmake -S . -B build -G Ninja -DMACROFUSION_GLFW_LINUX_BACKEND=WAYLAND -DMACROFUSION_BUILD_TESTS=OFF
cmake --build build --parallel
./build/macrofusion
```

### Docker Environment

Install Docker Engine or Docker Desktop with the Docker Compose plugin, then build
the development image:

```bash
docker compose build dev
```

Open a shell in a temporary development container. The repository is mounted at
`/workspace/MacroFusion`:

```bash
docker compose run --rm dev
```

On Linux desktops, the Docker services forward `DISPLAY`, the X11 socket, the
current `XAUTHORITY` cookie, and `/dev/dri`, and set `XDG_SESSION_TYPE=x11` so
GLFW uses X11/XWayland and Mesa can use the host render device. Run
`docker compose run --rm dev glxinfo -B` to verify that the container reports
hardware rendering and OpenGL 4.6 before launching the viewer. If `XAUTHORITY`
is not set on the host, set it to the active Xauthority file before running
Compose.

GPU execution requires a compatible NVIDIA driver and NVIDIA Container Toolkit:

```bash
docker compose --profile gpu run --rm dev-gpu
```

### Native Linux Environment

Native builds require a C++20 compiler, the CUDA 13.3 toolkit, CMake 3.28 or newer,
Ninja, Git, Python 3 with Jinja2, and OpenGL development packages. GLFW additionally
requires the X11 or Wayland development packages for the selected backend. Third-party
C++ libraries are downloaded by CMake during configuration.

After installing those prerequisites, run the project commands below directly from
the repository root. The same commands also work inside either Docker shell above.

### Build and Run

Configure and build the MacroFusion executable:

```bash
cmake -S . -B build -G Ninja -DMACROFUSION_BUILD_TESTS=OFF
cmake --build build --parallel
```

Run the reconstruction application with:

```bash
./build/macrofusion
```

If this checkout was previously built as `kinectfusion-dev`, remove the stale
`build/` directory before configuring again. The old CMake cache records
`/workspace/kinectfusion-dev` and only contains the retired `kinectfusion_dev`
executable target.

Optionally, enable, build, and run the tests to verify the build and runtime behavior:

```bash
cmake -S . -B build -G Ninja -DMACROFUSION_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### CMake Options

Pass custom values during the configure step with `-D<name>=<value>`.

| Option | Default | Values | Purpose |
| --- | --- | --- | --- |
| `MACROFUSION_BUILD_TESTS` | `ON` | `ON`, `OFF` | Build and register the project tests. |
| `MACROFUSION_GLFW_LINUX_BACKEND` | `AUTO` | `AUTO`, `X11`, `WAYLAND` | Select the GLFW Linux display backend. `AUTO` uses the current session and falls back to X11. |
| `CMAKE_CUDA_ARCHITECTURES` | `75;86;89;90` | CMake CUDA architecture list | Select the NVIDIA architectures compiled into CUDA targets. |

## Changelog

### Latest Snapshot: `week-1` - Host Tracking Viewer

- Added synchronized TUM RGB-D acquisition and host preprocessing for filtered
  depth, vertex maps, normal maps, and the base tracking pyramid.
- Added a GLFW/OpenGL/Dear ImGui viewer for RGB, filtered depth, vertex XYZ, and
  normal-map panes.
- Updated Linux Docker launch to use X11/XWayland authentication, `/dev/dri`
  hardware OpenGL passthrough, and container-side `glxinfo` verification.

## License

Copyright (c) 2026 MacroFusion contributors. Released under the MIT License.
