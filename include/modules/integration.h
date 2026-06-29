#pragma once

/// @file
/// @brief Declares the TSDF integration stage (host and CUDA backends).
///
/// Integration fuses one accepted frame's raw metric depth into the quantized
/// block-major TSDF (pipeline.md 2.4) with a projective signed-distance update.
/// Both backends share the same per-voxel math; the device launch gives one
/// CUDA block to each storage block so contiguous `threadIdx.x` voxels map to
/// contiguous storage.
///
/// The pipeline.md 2.4 signature omits the volume's world placement, which the
/// voxel->camera projection needs. The storage struct stays pure (layout +
/// arrays), so `volume_origin_meters` and `voxel_size_meters` are passed
/// explicitly, matching the explicit-parameter style of the other stages
/// (e.g. `EstimatePose` takes `base_intrinsics` directly).

#include <cstdint>
#include <vector>

#include <vector_types.h>

#include "core/math_types.cuh"
#include "core/volume.h"
#include "core/device_storage.cuh"

namespace macro_fusion {

/// @brief Fuses one accepted frame's raw depth into a host TSDF volume.
///
/// For every logical voxel the center is projected into the camera, the nearest
/// raw-depth sample is read, and the truncation-normalized signed distance is
/// running-average blended into the stored value while the weight saturates at
/// `max_weight`. Voxels behind the camera, projecting outside the image, reading
/// zero depth, or further than the truncation behind the surface are left
/// unchanged.
///
/// @param[in] image_extent Raw-depth width and height in pixels.
/// @param[in] intrinsics Full-resolution `fx/fy/cx/cy` in `x/y/z/w`.
/// @param[in] world_from_camera Accepted camera-to-world pose for the frame.
/// @param[in] raw_depth Row-major raw metric depth; zero denotes invalid.
/// @param[in] volume_origin_meters World minimum corner of voxel `(0,0,0)`.
/// @param[in] voxel_size_meters Per-axis world voxel edge length in meters.
/// @param[in] truncation_distance_meters Positive TSDF truncation in meters.
/// @param[in] max_weight Saturating observation weight in `[1, 255]`.
/// @param[in,out] volume Non-null initialized volume mutated in place.
/// @throws std::invalid_argument on a null/mismatched volume, zero extent,
///     depth size mismatch, non-positive truncation, or zero `max_weight`.
void Integrate(uint2 image_extent, float4 intrinsics,
               CameraPose world_from_camera,
               const std::vector<float>& raw_depth, float3 volume_origin_meters,
               float3 voxel_size_meters, float truncation_distance_meters,
               std::uint8_t max_weight, HostTsdfVolume* volume);

/// @brief CUDA backend of @ref Integrate over a device TSDF volume.
///
/// Launches one CUDA block per storage block with block dimensions equal to the
/// layout `block_shape`; padding threads exit before any read. The launch is
/// asynchronous on the default stream, matching the no-implicit-sync profiling
/// convention; callers synchronize (a download does).
///
/// @param[in] image_extent Raw-depth width and height in pixels.
/// @param[in] intrinsics Full-resolution `fx/fy/cx/cy` in `x/y/z/w`.
/// @param[in] world_from_camera Accepted camera-to-world pose for the frame.
/// @param[in] raw_depth Device row-major raw metric depth; zero denotes invalid.
/// @param[in] volume_origin_meters World minimum corner of voxel `(0,0,0)`.
/// @param[in] voxel_size_meters Per-axis world voxel edge length in meters.
/// @param[in] truncation_distance_meters Positive TSDF truncation in meters.
/// @param[in] max_weight Saturating observation weight in `[1, 255]`.
/// @param[in,out] volume Non-null initialized device volume mutated in place.
/// @throws std::invalid_argument on the same contract violations as @ref
///     Integrate, plus a per-block thread count above the device limit.
void IntegrateCuda(uint2 image_extent, float4 intrinsics,
                   CameraPose world_from_camera,
                   const DeviceVector<float>& raw_depth,
                   float3 volume_origin_meters, float3 voxel_size_meters,
                   float truncation_distance_meters, std::uint8_t max_weight,
                   DeviceTsdfVolume* volume);

}  // namespace macro_fusion
