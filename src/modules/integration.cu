/// @file
/// @brief Implements the host and CUDA TSDF integration backends.
///
/// Both backends are compiled in this single NVCC translation unit because the
/// per-voxel update reuses the camera-pose helpers in `core/math_types.cuh`,
/// which are device-context math and do not compile under a bare host compiler.
/// Co-locating the backends lets the host loop and the kernel share one
/// `FuseVoxel` definition instead of duplicating the projective update.

#include "modules/integration.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace macro_fusion {
namespace {

/// @brief Applies the projective TSDF update for one voxel center.
///
/// Mirrors the canonical KinectFusion update: project the voxel into the camera,
/// read the nearest raw-depth sample, form the truncation-normalized signed
/// distance (positive in front of the surface), and running-average blend it
/// while the weight saturates. Free space clamps to `+1`; samples further than
/// the truncation behind the surface are skipped so thin surfaces are not carved
/// from the back.
///
/// @return `true` when the voxel was updated.
__host__ __device__ inline bool FuseVoxel(float3 world_center,
                                          const CameraPose& world_from_camera,
                                          float4 intrinsics, uint2 image_extent,
                                          const float* raw_depth,
                                          float truncation_distance_meters,
                                          std::uint8_t max_weight,
                                          TsdfDistance* distance,
                                          TsdfWeight* weight) {
  const float4 camera = WorldToCameraSpace(
      world_from_camera,
      float4{world_center.x, world_center.y, world_center.z, 1.0f});
  if (camera.z <= 0.0f) {
    return false;  // behind the camera
  }
  const float2 pixel = CameraToImageSpace(intrinsics, camera);
  const int u = static_cast<int>(floorf(pixel.x + 0.5f));
  const int v = static_cast<int>(floorf(pixel.y + 0.5f));
  if (u < 0 || v < 0 || u >= static_cast<int>(image_extent.x) ||
      v >= static_cast<int>(image_extent.y)) {
    return false;  // projects outside the image
  }
  const float depth =
      raw_depth[static_cast<std::size_t>(v) * image_extent.x + u];
  if (!(depth > 0.0f)) {
    return false;  // invalid / zero depth
  }
  const float signed_distance = depth - camera.z;  // positive is free space
  const float normalized = signed_distance / truncation_distance_meters;
  if (normalized < -1.0f) {
    return false;  // beyond truncation behind the surface
  }
  const float clamped = normalized > 1.0f ? 1.0f : normalized;
  const float old_weight = static_cast<float>(*weight);
  const float old_distance = DecodeTsdf(*distance);
  const float new_weight = old_weight + 1.0f;
  *distance = EncodeTsdf((old_distance * old_weight + clamped) / new_weight);
  *weight = new_weight > static_cast<float>(max_weight)
                ? max_weight
                : static_cast<TsdfWeight>(new_weight);
  return true;
}

/// @brief Validates the shared integration contract for both backends.
void ValidateIntegrateContract(uint2 image_extent,
                               std::size_t raw_depth_size,
                               float truncation_distance_meters,
                               std::uint8_t max_weight,
                               const BlockMajorLayout& layout,
                               std::size_t distance_size,
                               std::size_t weight_size) {
  if (image_extent.x == 0 || image_extent.y == 0) {
    throw std::invalid_argument("Integrate: image_extent must be nonzero");
  }
  if (raw_depth_size !=
      static_cast<std::size_t>(image_extent.x) * image_extent.y) {
    throw std::invalid_argument("Integrate: raw_depth size must equal extent");
  }
  if (!(truncation_distance_meters > 0.0f)) {
    throw std::invalid_argument("Integrate: truncation must be positive");
  }
  if (max_weight == 0) {
    throw std::invalid_argument("Integrate: max_weight must be in [1, 255]");
  }
  if (distance_size != layout.storage_elements ||
      weight_size != layout.storage_elements) {
    throw std::invalid_argument("Integrate: volume arrays must match layout");
  }
}

/// @brief One thread per padded storage-block voxel; updates real voxels only.
__global__ void IntegrateKernel(uint2 image_extent, float4 intrinsics,
                                CameraPose world_from_camera,
                                const float* raw_depth, float3 origin,
                                float3 voxel_size,
                                float truncation_distance_meters,
                                std::uint8_t max_weight, BlockMajorLayout layout,
                                TsdfDistance* distance, TsdfWeight* weight) {
  const std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const std::uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
  if (!InBounds(layout, x, y, z)) {
    return;  // padding thread
  }
  const float3 world{origin.x + (static_cast<float>(x) + 0.5f) * voxel_size.x,
                     origin.y + (static_cast<float>(y) + 0.5f) * voxel_size.y,
                     origin.z + (static_cast<float>(z) + 0.5f) * voxel_size.z};
  const std::uint32_t flat = Flatten(layout, x, y, z);
  FuseVoxel(world, world_from_camera, intrinsics, image_extent, raw_depth,
            truncation_distance_meters, max_weight, &distance[flat],
            &weight[flat]);
}

}  // namespace

void Integrate(uint2 image_extent, float4 intrinsics,
               CameraPose world_from_camera,
               const std::vector<float>& raw_depth, float3 volume_origin_meters,
               float3 voxel_size_meters, float truncation_distance_meters,
               std::uint8_t max_weight, HostTsdfVolume* volume) {
  if (volume == nullptr) {
    throw std::invalid_argument("Integrate: volume must be non-null");
  }
  const BlockMajorLayout& layout = volume->layout;
  ValidateIntegrateContract(image_extent, raw_depth.size(),
                            truncation_distance_meters, max_weight, layout,
                            volume->distance.size(), volume->weight.size());

  for (std::uint32_t z = 0; z < layout.logical_resolution.z; ++z) {
    for (std::uint32_t y = 0; y < layout.logical_resolution.y; ++y) {
      for (std::uint32_t x = 0; x < layout.logical_resolution.x; ++x) {
        const float3 world{
            volume_origin_meters.x + (static_cast<float>(x) + 0.5f) * voxel_size_meters.x,
            volume_origin_meters.y + (static_cast<float>(y) + 0.5f) * voxel_size_meters.y,
            volume_origin_meters.z + (static_cast<float>(z) + 0.5f) * voxel_size_meters.z};
        const std::uint32_t flat = Flatten(layout, x, y, z);
        FuseVoxel(world, world_from_camera, intrinsics, image_extent,
                  raw_depth.data(), truncation_distance_meters, max_weight,
                  &volume->distance[flat], &volume->weight[flat]);
      }
    }
  }
}

void IntegrateCuda(uint2 image_extent, float4 intrinsics,
                   CameraPose world_from_camera,
                   const DeviceVector<float>& raw_depth,
                   float3 volume_origin_meters, float3 voxel_size_meters,
                   float truncation_distance_meters, std::uint8_t max_weight,
                   DeviceTsdfVolume* volume) {
  if (volume == nullptr) {
    throw std::invalid_argument("IntegrateCuda: volume must be non-null");
  }
  const BlockMajorLayout layout = volume->layout;
  ValidateIntegrateContract(image_extent, raw_depth.size(),
                            truncation_distance_meters, max_weight, layout,
                            volume->distance.size(), volume->weight.size());
  if (layout.voxels_per_block > 1024u) {
    throw std::invalid_argument(
        "IntegrateCuda: block_shape exceeds the 1024-thread launch limit");
  }

  const dim3 block(layout.block_shape.x, layout.block_shape.y,
                   layout.block_shape.z);
  const dim3 grid(layout.block_grid.x, layout.block_grid.y, layout.block_grid.z);
  IntegrateKernel<<<grid, block>>>(
      image_extent, intrinsics, world_from_camera,
      thrust::raw_pointer_cast(raw_depth.data()), volume_origin_meters,
      voxel_size_meters, truncation_distance_meters, max_weight, layout,
      thrust::raw_pointer_cast(volume->distance.data()),
      thrust::raw_pointer_cast(volume->weight.data()));
}

}  // namespace macro_fusion
