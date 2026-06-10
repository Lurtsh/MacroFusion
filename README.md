# MacroFusion

MacroFusion is a CUDA-focused student implementation of a
[KinectFusion](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf)-style
3D reconstruction pipeline using a dense TSDF volume.

## Build

### Container Quick Start

For convenience, these commands create the development image, build the MacroFusion
application, and run it entirely through Docker:

```bash
docker compose build dev
docker compose run --rm dev bash -lc \
  "cmake -S . -B build -G Ninja -DMACROFUSION_BUILD_TESTS=OFF && cmake --build build --parallel"
docker compose run --rm dev ./build/macrofusion
```

Optionally, build and run the tests in the container to verify the project runtime:

```bash
docker compose run --rm dev bash -lc \
  "cmake -S . -B build -G Ninja -DMACROFUSION_BUILD_TESTS=ON && cmake --build build --parallel && ctest --test-dir build --output-on-failure"
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

### Latest Snapshot: `week-0` - Project Scaffold

- Added the CUDA 13.3, C++20, CMake, and Docker development environment.
- Added pinned Eigen, GoogleTest, glog, GLFW, OpenGL, Dear ImGui, GLAD, stb, and JSON dependencies.
- Added the initial application and CPU smoke-test targets.

## License

Copyright (c) 2026 MacroFusion contributors. Released under the MIT License.
