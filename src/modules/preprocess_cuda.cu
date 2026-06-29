/// @file
/// @brief Provides CUDA preprocessing operation definitions.

#include "modules/preprocess_cuda.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace macro_fusion {
namespace {

constexpr dim3 kBackProjectBlockDim{16, 16, 1};
constexpr float kMaxNormalDepthDiscontinuityMeters = 0.1f;
constexpr float kMinPositiveFloat = std::numeric_limits<float>::min();

std::size_t PixelCount(uint2 image_extent) {
  if (image_extent.x >
      std::numeric_limits<std::size_t>::max() / image_extent.y) {
    throw std::invalid_argument("image extent exceeds addressable storage");
  }
  return static_cast<std::size_t>(image_extent.x) * image_extent.y;
}

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

__device__ bool IsValidDepth(float depth) {
  return isfinite(depth) && depth > 0.0f;
}

__device__ bool IsFinite(float3 value) {
  return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

/// @brief Back-projects depth into camera-frame vertices for one tracking level.
///
/// @par Parallelization
/// One thread per pixel in a 16x16-thread 2-D grid; out-of-range threads return
/// without writing.
/// @verbatim
///   col = blockIdx.x * blockDim.x + threadIdx.x
///   row = blockIdx.y * blockDim.y + threadIdx.y
/// @endverbatim
///
/// @par Memory layout
/// `d_depth` and `d_vertices` are row-major over `image_extent.y` rows and
/// `image_extent.x` columns: `index(row, col) = row * image_extent.x + col`.
///
/// @par Access pattern
/// Coalesced along `threadIdx.x` (column). Each pixel is read once from
/// `d_depth` and written once to `d_vertices`; invalid depth leaves the output
/// zeroed with `w = 0`.
///
/// @param[in] image_extent Image width (`x`) and height (`y`) in pixels.
/// @param[in] level_intrinsics Scaled `fx/fy/cx/cy` in `x/y/z/w`.
/// @param[in] d_depth Row-major metric depth buffer (`width * height` floats).
/// @param[out] d_vertices Row-major vertex buffer (`width * height` float4s).
__global__ void BackProjectKernel(uint2 image_extent, float4 level_intrinsics,
                                  const float* __restrict__ d_depth,
                                  float4* __restrict__ d_vertices) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= image_extent.x || y >= image_extent.y) return;

  const std::size_t index = static_cast<std::size_t>(y) * image_extent.x + x;
  const float depth = d_depth[index];
  if (!IsValidDepth(depth)) {
    d_vertices[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    return;
  }

  d_vertices[index] = make_float4(
      (static_cast<float>(x) - level_intrinsics.z) * depth / level_intrinsics.x,
      (static_cast<float>(y) - level_intrinsics.w) * depth / level_intrinsics.y,
      depth, 1.0f);
}

/// @brief Computes unit normals from central vertex differences.
///
/// @par Parallelization
/// One thread per pixel in a 16x16-thread 2-D grid. Border pixels and threads
/// outside the image return without writing.
/// @verbatim
///   col = blockIdx.x * blockDim.x + threadIdx.x
///   row = blockIdx.y * blockDim.y + threadIdx.y
/// @endverbatim
///
/// @par Memory layout
/// `d_depth`, `d_vertices`, and `d_normals` share row-major indexing:
/// `index(row, col) = row * image_extent.x + col`.
///
/// @par Access pattern
/// Each interior thread reads five `d_vertices` samples and two `d_depth` pairs;
/// accesses along `threadIdx.x` are coalesced. Valid normals are written once
/// to `d_normals` with `w = 1`; rejected pixels leave the buffer unchanged.
///
/// @param[in] image_extent Image width (`x`) and height (`y`) in pixels.
/// @param[in] d_depth Row-major depth used for the 0.1 m discontinuity check.
/// @param[in] d_vertices Row-major back-projected vertices with validity in `w`.
/// @param[out] d_normals Row-major normal buffer preset to zero by the host.
__global__ void ComputeNormalsKernel(uint2 image_extent,
                                     const float* __restrict__ d_depth,
                                     const float4* __restrict__ d_vertices,
                                     float4* __restrict__ d_normals) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  // check if the thread is within the image extent
  if (x >= image_extent.x || y >= image_extent.y) return;
  // check if the thread is on the border of the image
  if (x == 0 || y == 0 || x + 1 >= image_extent.x || y + 1 >= image_extent.y) return;

  const std::size_t index = static_cast<std::size_t>(y) * image_extent.x + x;
  const std::size_t left = index - 1;
  const std::size_t right = index + 1;
  const std::size_t up = index - image_extent.x;
  const std::size_t down = index + image_extent.x;
  // check if the vertices are valid
  if (d_vertices[index].w == 0.0f || d_vertices[left].w == 0.0f ||
      d_vertices[right].w == 0.0f || d_vertices[up].w == 0.0f || d_vertices[down].w == 0.0f) return;
  // check if the depth discontinuity is within the threshold
  if (fabsf(d_depth[right] - d_depth[left]) > kMaxNormalDepthDiscontinuityMeters ||
      fabsf(d_depth[down] - d_depth[up]) > kMaxNormalDepthDiscontinuityMeters) return;

  const float3 horizontal{
      d_vertices[right].x - d_vertices[left].x,
      d_vertices[right].y - d_vertices[left].y,
      d_vertices[right].z - d_vertices[left].z};
  const float3 vertical{d_vertices[down].x - d_vertices[up].x,
                        d_vertices[down].y - d_vertices[up].y,
                        d_vertices[down].z - d_vertices[up].z};
  const float3 cross{horizontal.y * vertical.z - horizontal.z * vertical.y,
                     horizontal.z * vertical.x - horizontal.x * vertical.z,
                     horizontal.x * vertical.y - horizontal.y * vertical.x};
  const float squared_norm = cross.x * cross.x + cross.y * cross.y + cross.z * cross.z;
  // check if the cross product is finite and the squared norm is positive
  if (!IsFinite(cross) || !isfinite(squared_norm) || squared_norm <= kMinPositiveFloat) return;

  const float inverse_norm = rsqrtf(squared_norm);
  d_normals[index] = make_float4(cross.x * inverse_norm, cross.y * inverse_norm, cross.z * inverse_norm, 1.0f);
}

/// @brief Computes unit normals from neighboring vertices (KinectFusion Eq. 4).
///
/// @par Parallelization
/// One thread per pixel in a 16x16-thread 2-D grid. The rightmost column and
/// bottom row return without writing.
/// @verbatim
///   col = blockIdx.x * blockDim.x + threadIdx.x
///   row = blockIdx.y * blockDim.y + threadIdx.y
/// @endverbatim
///
/// @par Memory layout
/// `d_vertices` and `d_normals` are row-major:
/// `index(row, col) = row * image_extent.x + col`; right/down neighbors use
/// offsets `+1` and `+image_extent.x`.
///
/// @par Access pattern
/// Each thread reads center, right, and down vertices; stores are coalesced
/// along `threadIdx.x`. Depth discontinuities above 0.1 m along the
/// center-to-right and center-to-down edges are rejected.
///
/// @param[in] image_extent Image width (`x`) and height (`y`) in pixels.
/// @param[in] d_depth Row-major depth used for the 0.1 m discontinuity check.
/// @param[in] d_vertices Row-major back-projected vertices with validity in `w`.
/// @param[out] d_normals Row-major normal buffer preset to zero by the host.
__global__ void ComputeNormalsNeighborKernel(
    uint2 image_extent, const float* __restrict__ d_depth,
    const float4* __restrict__ d_vertices,
    float4* __restrict__ d_normals) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  // check if the thread is within the image extent
  if (x >= image_extent.x || y >= image_extent.y) return;
  if (x + 1 >= image_extent.x || y + 1 >= image_extent.y) return;

  const std::size_t index = static_cast<std::size_t>(y) * image_extent.x + x;
  const std::size_t right = index + 1;
  const std::size_t down = index + image_extent.x;
  if (fabsf(d_depth[right] - d_depth[index]) > kMaxNormalDepthDiscontinuityMeters ||
      fabsf(d_depth[down] - d_depth[index]) > kMaxNormalDepthDiscontinuityMeters) {
    return;
  }

  const float4 center = d_vertices[index];
  const float4 right_vertex = d_vertices[right];
  const float4 down_vertex = d_vertices[down];
  // check if the vertices are valid
  if (center.w == 0.0f || right_vertex.w == 0.0f || down_vertex.w == 0.0f) return;

  const float3 horizontal = make_float3(right_vertex.x - center.x, right_vertex.y - center.y,
                                        right_vertex.z - center.z);
  const float3 vertical = make_float3(down_vertex.x - center.x, down_vertex.y - center.y,
                                        down_vertex.z - center.z);
  const float3 cross =
      make_float3(horizontal.y * vertical.z - horizontal.z * vertical.y,
                  horizontal.z * vertical.x - horizontal.x * vertical.z,
                  horizontal.x * vertical.y - horizontal.y * vertical.x);
  const float squared_norm = cross.x * cross.x + cross.y * cross.y + cross.z * cross.z;
  // check if the cross product is finite and the squared norm is positive
  if (!IsFinite(cross) || !isfinite(squared_norm) || squared_norm <= kMinPositiveFloat) return;

  const float inverse_norm = rsqrtf(squared_norm);
  d_normals[index] = make_float4(cross.x * inverse_norm, cross.y * inverse_norm, cross.z * inverse_norm, 1.0f);
}

}  // namespace

void BackProject(uint2 image_extent, float4 level_intrinsics,
                 DeviceTrackingLevel* level) {
  const std::size_t pixel_count = PixelCount(image_extent);
  level->vertices.assign(pixel_count, make_float4(0.0f, 0.0f, 0.0f, 0.0f));

  const dim3 grid_dim(
      (image_extent.x + kBackProjectBlockDim.x - 1) / kBackProjectBlockDim.x,
      (image_extent.y + kBackProjectBlockDim.y - 1) / kBackProjectBlockDim.y,
      1);

  BackProjectKernel<<<grid_dim, kBackProjectBlockDim>>>(
      image_extent, level_intrinsics, level->depth.data().get(),
      level->vertices.data().get());
  CheckCuda(cudaGetLastError(), "BackProjectKernel launch");
  CheckCuda(cudaDeviceSynchronize(), "BackProjectKernel synchronize");
}

void ComputeNormals(uint2 image_extent, DeviceTrackingLevel* level,
                    NormalMethod method) {
  const std::size_t pixel_count = PixelCount(image_extent);
  level->normals.assign(pixel_count, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
  if (image_extent.x < 3 || image_extent.y < 3) {
    return;
  }

  const dim3 grid_dim(
      (image_extent.x + kBackProjectBlockDim.x - 1) / kBackProjectBlockDim.x,
      (image_extent.y + kBackProjectBlockDim.y - 1) / kBackProjectBlockDim.y,
      1);

  switch (method) {
    case NormalMethod::kCentralDifferences:
      ComputeNormalsKernel<<<grid_dim, kBackProjectBlockDim>>>(
          image_extent, level->depth.data().get(), level->vertices.data().get(),
          level->normals.data().get());
      CheckCuda(cudaGetLastError(), "ComputeNormalsKernel launch");
      break;
    case NormalMethod::kNeighborVertices:
      ComputeNormalsNeighborKernel<<<grid_dim, kBackProjectBlockDim>>>(
          image_extent, level->depth.data().get(), level->vertices.data().get(),
          level->normals.data().get());
      CheckCuda(cudaGetLastError(), "ComputeNormalsNeighborKernel launch");
      break;
  }
  CheckCuda(cudaDeviceSynchronize(), "ComputeNormals synchronize");
}

}  // namespace macro_fusion
