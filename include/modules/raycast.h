#pragma once

/// @file
/// @brief Declares the TSDF raycasting stage (host and CUDA backends).
///
/// Raycast renders a world-frame surface prediction from the TSDF: per pixel it
/// builds a world ray, traverses with the selected mode, refines the first
/// zero crossing, and writes vertex/normal/depth/validity maps that feed the
/// next frame's tracking (pipeline.md 2.6). The primary benchmark pair is
/// `kMuSkipBaseline` vs `kMacroCellDistance`; `kFixedStepDebug` is a correctness
/// oracle. As with Integrate, the volume's world placement
/// (`volume_origin_meters`, `voxel_size_meters`) and the truncation distance are
/// passed explicitly because the storage struct carries only the layout/arrays.

#include <cstdint>
#include <vector>

#include <vector_types.h>

#include "core/math_types.cuh"
#include "core/volume.h"
#include "core/device_storage.cuh"
#include "modules/macro_cells.h"

namespace macro_fusion {

/// @brief Ray traversal strategy.
enum class RaycastMode {
  kFixedStepDebug,    ///< Constant `min_step` march; correctness oracle.
  kMuSkipBaseline,    ///< Step by the TSDF-implied safe distance (baseline).
  kMacroCellDistance  ///< Leap empty macro cells, then fine-step (accelerated).
};

/// @brief Read-only per-launch raycast configuration (flat POD).
struct RaycastConfig {
  RaycastMode mode;
  float near_meters;             ///< Ray start distance from the camera.
  float far_meters;              ///< Ray stop distance; no hit past this.
  float min_step_voxels;         ///< Fixed-step floor, in voxels.
  float truncation_step_scale;   ///< mu-skip safe-distance scale in `(0, 1]`.
  std::uint32_t max_iterations;  ///< Hard cap on traversal steps per ray.
  std::uint32_t zero_crossing_iterations;  ///< Bisection refinement steps.
};

/// @brief Host-resident world-frame surface prediction (Raycast output).
struct HostSurfacePrediction {
  CameraPose world_from_camera;   ///< Render camera pose.
  std::vector<float3> vertices;   ///< World-frame surface points.
  std::vector<float3> normals;    ///< World-frame unit normals.
  std::vector<float> depth;       ///< Camera-frame surface depth; zero invalid.
  std::vector<std::uint8_t> validity;  ///< 1 valid, 0 invalid, per pixel.
};

/// @brief Device-resident world-frame surface prediction.
struct DeviceSurfacePrediction {
  CameraPose world_from_camera;
  thrust::device_vector<float4> vertices;  ///< xyz position, w = validity.
  thrust::device_vector<float4> normals;   ///< xyz normal, w = validity.
  thrust::device_vector<float> depth;      ///< zero = invalid.
};

/// @brief Packs a host prediction into device storage (float3 + validity ->
///     float4 with validity in `w`).
inline void UploadSurfacePrediction(const HostSurfacePrediction& host,
                                    DeviceSurfacePrediction* device) {
  device->world_from_camera = host.world_from_camera;
  const std::size_t count = host.vertices.size();
  std::vector<float4> vertices(count);
  std::vector<float4> normals(count);
  for (std::size_t i = 0; i < count; ++i) {
    const float valid =
        i < host.validity.size() ? static_cast<float>(host.validity[i]) : 0.0f;
    vertices[i] =
        float4{host.vertices[i].x, host.vertices[i].y, host.vertices[i].z, valid};
    normals[i] =
        float4{host.normals[i].x, host.normals[i].y, host.normals[i].z, valid};
  }
  UploadImage(vertices, &device->vertices);
  UploadImage(normals, &device->normals);
  UploadImage(host.depth, &device->depth);
}

/// @brief Unpacks a device prediction into host storage (float4 with validity
///     in `w` -> float3 + validity byte).
inline void DownloadSurfacePrediction(const DeviceSurfacePrediction& device,
                                      HostSurfacePrediction* host) {
  host->world_from_camera = device.world_from_camera;
  std::vector<float4> vertices;
  std::vector<float4> normals;
  DownloadImage(device.vertices, &vertices);
  DownloadImage(device.normals, &normals);
  DownloadImage(device.depth, &host->depth);
  const std::size_t count = vertices.size();
  host->vertices.resize(count);
  host->normals.resize(count);
  host->validity.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    host->vertices[i] = float3{vertices[i].x, vertices[i].y, vertices[i].z};
    host->normals[i] = float3{normals[i].x, normals[i].y, normals[i].z};
    host->validity[i] = vertices[i].w > 0.5f ? 1u : 0u;
  }
}

/// @brief Renders a world-frame surface prediction from a host TSDF volume.
///
/// @param[in] volume Current TSDF volume.
/// @param[in] macro_cells Optional clearance grid; required for
///     `kMacroCellDistance`, ignored (may be null) otherwise.
/// @param[in] image_extent Output prediction width and height in pixels.
/// @param[in] intrinsics Render-camera `fx/fy/cx/cy` in `x/y/z/w`.
/// @param[in] world_from_camera Render-camera pose.
/// @param[in] volume_origin_meters World minimum corner of voxel `(0,0,0)`.
/// @param[in] voxel_size_meters Per-axis world voxel edge length in meters.
/// @param[in] truncation_distance_meters Positive TSDF truncation in meters.
/// @param[in] config Traversal mode and step parameters.
/// @param[out] prediction Non-null destination resized and overwritten.
/// @param[out] fetch_field Optional diagnostic memory-traffic map; when non-null
///     it is resized and filled with the per-pixel trilinear TSDF sample count
///     (pipeline.md 2.6 traversal counter). Null in production.
/// @throws std::invalid_argument on a null prediction, zero extent, non-positive
///     truncation, or `kMacroCellDistance` with a null `macro_cells`.
void Raycast(const HostTsdfVolume& volume, const HostMacroCellGrid* macro_cells,
             uint2 image_extent, float4 intrinsics, CameraPose world_from_camera,
             float3 volume_origin_meters, float3 voxel_size_meters,
             float truncation_distance_meters, const RaycastConfig& config,
             HostSurfacePrediction* prediction,
             std::vector<unsigned>* fetch_field = nullptr);

/// @brief CUDA backend of @ref Raycast over device storage.
///
/// Uses a 2D image launch (independent of the integration block shape). The
/// launch is asynchronous; callers synchronize (a download does). `fetch_field`
/// is the same optional per-pixel diagnostic memory-traffic map as the host
/// overload; null in production.
void RaycastCuda(const DeviceTsdfVolume& volume,
                 const DeviceMacroCellGrid* macro_cells, uint2 image_extent,
                 float4 intrinsics, CameraPose world_from_camera,
                 float3 volume_origin_meters, float3 voxel_size_meters,
                 float truncation_distance_meters, const RaycastConfig& config,
                 DeviceSurfacePrediction* prediction,
                 thrust::device_vector<unsigned>* fetch_field = nullptr);

}  // namespace macro_fusion
