/// @file
/// @brief Provides CUDA display conversion operation definitions.

#include "modules/render_cuda.cuh"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace macro_fusion {
namespace {

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

constexpr dim3 kRenderBlockDim{16, 16, 1};
constexpr float kDisplayDepthMinMeters = 0.3f;
constexpr float kDisplayDepthMaxMeters = 5.0f;

__device__ std::uint8_t UnitToByte(float value) {
  const float clamped = fminf(fmaxf(value, 0.0f), 1.0f);
  return static_cast<std::uint8_t>(clamped * 255.0f + 0.5f);
}

__device__ uchar4 Rgba(std::uint8_t red, std::uint8_t green,
                       std::uint8_t blue) {
  return uchar4{red, green, blue, std::uint8_t{255}};
}

__device__ uchar4 Black() { return Rgba(0, 0, 0); }

__device__ float DepthUnit(float depth) {
  return (depth - kDisplayDepthMinMeters) /
         (kDisplayDepthMaxMeters - kDisplayDepthMinMeters);
}

/// @brief Converts a row-major RGB image to opaque RGBA.
///
/// @par Parallelization
/// One thread converts one pixel in a 16x16 2-D block. The grid is rounded up
/// to cover the whole image; out-of-range threads return without writing.
/// @verbatim
///   col = blockIdx.x * blockDim.x + threadIdx.x
///   row = blockIdx.y * blockDim.y + threadIdx.y
/// @endverbatim
///
/// @par Memory layout
/// `d_rgb` and `d_rgba` are tightly packed row-major images with flat index
/// `row * image_extent.x + col`.
///
/// @par Access pattern
/// Consecutive `threadIdx.x` lanes read adjacent `uchar3` inputs and write
/// adjacent `uchar4` outputs. Each pixel is read once and written once.
///
/// @param[in] image_extent Width and height of the image in pixels.
/// @param[in] d_rgb Device RGB input with `width * height` elements.
/// @param[out] d_rgba Device RGBA output with `width * height` elements.
__global__ void RgbToRgbaKernel(uint2 image_extent,
                                const uchar3* __restrict__ d_rgb,
                                uchar4* __restrict__ d_rgba) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= image_extent.x || row >= image_extent.y) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(row) * image_extent.x + col;
  const uchar3 rgb = d_rgb[index];
  d_rgba[index] = Rgba(rgb.x, rgb.y, rgb.z);
}

/// @brief Converts a row-major metric-depth image to grayscale RGBA.
///
/// @par Parallelization
/// One thread converts one pixel in a 16x16 2-D block. The grid is rounded up
/// to cover the whole image; out-of-range threads return without writing.
///
/// @par Memory layout
/// `d_depth` and `d_rgba` are tightly packed row-major images with flat index
/// `row * image_extent.x + col`.
///
/// @par Access pattern
/// Adjacent `threadIdx.x` lanes read adjacent depth samples and write adjacent
/// RGBA pixels. Invalid or non-positive depth is written as black.
///
/// @param[in] image_extent Width and height of the image in pixels.
/// @param[in] d_depth Device metric depth with zero denoting invalid samples.
/// @param[out] d_rgba Device RGBA output with `width * height` elements.
__global__ void DepthToRgbaKernel(uint2 image_extent,
                                  const float* __restrict__ d_depth,
                                  uchar4* __restrict__ d_rgba) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= image_extent.x || row >= image_extent.y) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(row) * image_extent.x + col;
  const float depth = d_depth[index];
  if (!isfinite(depth) || depth <= 0.0f) {
    d_rgba[index] = Black();
    return;
  }
  const std::uint8_t intensity = UnitToByte(DepthUnit(depth));
  d_rgba[index] = Rgba(intensity, intensity, intensity);
}

/// @brief Converts row-major packed normals to RGB-normal RGBA.
///
/// @par Parallelization
/// One thread converts one pixel in a 16x16 2-D block. The grid is rounded up
/// to cover the whole image; out-of-range threads return without writing.
///
/// @par Memory layout
/// `d_normals` and `d_rgba` are tightly packed row-major images with flat index
/// `row * image_extent.x + col`; normal validity is stored in `.w`.
///
/// @par Access pattern
/// Adjacent `threadIdx.x` lanes read adjacent `float4` normals and write
/// adjacent `uchar4` outputs. Invalid or non-finite normals are black.
///
/// @param[in] image_extent Width and height of the image in pixels.
/// @param[in] d_normals Device normals in `[-1, 1]`, with validity in `.w`.
/// @param[out] d_rgba Device RGBA output with `width * height` elements.
__global__ void NormalsToRgbaKernel(uint2 image_extent,
                                    const float4* __restrict__ d_normals,
                                    uchar4* __restrict__ d_rgba) {
  const std::uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= image_extent.x || row >= image_extent.y) {
    return;
  }
  const std::size_t index = static_cast<std::size_t>(row) * image_extent.x + col;
  const float4 normal = d_normals[index];
  if (normal.w == 0.0f || !isfinite(normal.x) || !isfinite(normal.y) ||
      !isfinite(normal.z)) {
    d_rgba[index] = Black();
    return;
  }
  d_rgba[index] = Rgba(UnitToByte(normal.x * 0.5f + 0.5f),
                       UnitToByte(normal.y * 0.5f + 0.5f),
                       UnitToByte(normal.z * 0.5f + 0.5f));
}

dim3 RenderGrid(uint2 image_extent) {
  return dim3((image_extent.x + kRenderBlockDim.x - 1) / kRenderBlockDim.x,
              (image_extent.y + kRenderBlockDim.y - 1) / kRenderBlockDim.y, 1);
}

}  // namespace

/// @brief Copies solid debug-color device buffers into mapped display PBOs.
///
/// @par Parallelization
/// No project kernel is launched in this smoke test. Each pane allocates a
/// `thrust::device_vector<uchar4>` initialized by Thrust with one element per
/// pixel, then CUDA Runtime copies the contiguous pane into the matching mapped
/// PBO with `cudaMemcpyDeviceToDevice`.
///
/// @par Memory layout
/// Every pane is tightly packed row-major RGBA8:
/// `index(row, col) = row * image_extent.x + col`.
///
/// @par Access pattern
/// Thrust writes a contiguous temporary buffer per pane. The following
/// device-to-device copy writes the mapped PBO contiguously once; pane PBOs do
/// not alias each other.
///
/// @param[in] image_extent Pane width (`x`) and height (`y`) in pixels.
/// @param[in,out] mapped_panes Device pointers returned by the viewer's
///     CUDA/OpenGL map step, in shared pane-index order.
void RenderDummyPanesDevice(
    uint2 image_extent, const std::array<uchar4*, kPaneCount>& mapped_panes) {
  const std::size_t count =
      static_cast<std::size_t>(image_extent.x) * image_extent.y;

  std::array<uchar4, kPaneCount> colors{};
  colors[kRgbPane] = uchar4{255, 0, 0, 255};
  colors[kDepthPane] = uchar4{0, 255, 0, 255};
  colors[kVertexPane] = uchar4{0, 0, 255, 255};
  colors[kNormalPane] = uchar4{255, 255, 255, 255};

  for (std::size_t pane = 0; pane < kPaneCount; ++pane) {
    thrust::device_vector<uchar4> buffer(count, colors[pane]);
    CheckCuda(cudaMemcpy(mapped_panes[pane],
                         thrust::raw_pointer_cast(buffer.data()),
                         count * sizeof(uchar4), cudaMemcpyDeviceToDevice),
              "cudaMemcpy dummy pane");
  }

  CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void RenderReconstructionPanesDevice(
    uint2 image_extent, const DeviceVector<uchar3>& rgb,
    const DeviceVector<float>& filtered_depth,
    const DeviceSurfacePrediction& prediction,
    const std::array<uchar4*, kPaneCount>& mapped_panes) {
  const dim3 grid = RenderGrid(image_extent);

  RgbToRgbaKernel<<<grid, kRenderBlockDim>>>(
      image_extent, thrust::raw_pointer_cast(rgb.data()),
      mapped_panes[kRgbPane]);
  CheckCuda(cudaGetLastError(), "RgbToRgbaKernel launch");

  DepthToRgbaKernel<<<grid, kRenderBlockDim>>>(
      image_extent, thrust::raw_pointer_cast(filtered_depth.data()),
      mapped_panes[kDepthPane]);
  CheckCuda(cudaGetLastError(), "DepthToRgbaKernel filtered launch");

  DepthToRgbaKernel<<<grid, kRenderBlockDim>>>(
      image_extent, thrust::raw_pointer_cast(prediction.depth.data()),
      mapped_panes[kVertexPane]);
  CheckCuda(cudaGetLastError(), "DepthToRgbaKernel raycast launch");

  NormalsToRgbaKernel<<<grid, kRenderBlockDim>>>(
      image_extent, thrust::raw_pointer_cast(prediction.normals.data()),
      mapped_panes[kNormalPane]);
  CheckCuda(cudaGetLastError(), "NormalsToRgbaKernel launch");

  CheckCuda(cudaDeviceSynchronize(), "RenderReconstructionPanesDevice sync");
}

}  // namespace macro_fusion
