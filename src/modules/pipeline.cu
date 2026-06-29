/// @file
/// @brief Implements GL-free host application pipeline stages.

#include "modules/pipeline.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "modules/render_cuda.cuh"
#include "modules/tracking.h"
#include "core/volume.h"
#include "core/device_storage.cuh"
#include "modules/integration.h"
#include "modules/preprocess.h"
#include "modules/raycast.h"
#include "modules/render.h"

namespace macro_fusion {
namespace {

constexpr float kBilateralSpatialSigmaPixels = 1.0f;
constexpr float kBilateralRangeSigmaMeters = 0.1f;
constexpr float kBilateralRadiusPixels = 1.0f;
constexpr float kDisplayDepthMinMeters = 0.3f;
constexpr float kDisplayDepthMaxMeters = 5.0f;

const VolumeConfig kVolumeConfig{
    uint3{512, 512, 512},
    float3{3.0f, 3.0f, 3.0f},
    float3{-1.0f, -1.0f, -1.0f},
    0.04f,
    std::uint8_t{64},
    uint3{8, 8, 8}};

const RaycastConfig kRaycastConfig{
    RaycastMode::kMuSkipBaseline,
    0.3f,
    5.0f,
    0.5f,
    0.8f,
    512,
    8};

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

CameraPose ToCameraPose(const RigidTransform& transform) {
  return CameraPose{float4{transform.data[0], transform.data[1],
                           transform.data[2], transform.data[3]},
                    float4{transform.data[4], transform.data[5],
                           transform.data[6], transform.data[7]},
                    float4{transform.data[8], transform.data[9],
                           transform.data[10], transform.data[11]}};
}

}  // namespace

std::vector<HostTrackingLevel> PreprocessFrame(
    const FrameSourceConfig& config, const std::vector<float>& raw_depth) {
  return BuildTrackingPyramid(
      config.image_extent, config.intrinsics,
      float3{kBilateralSpatialSigmaPixels, kBilateralRangeSigmaMeters,
             kBilateralRadiusPixels},
      raw_depth, 1);
}

void RenderTrackingViews(
    const FrameSourceConfig& config, const std::vector<uchar3>& rgb,
    const HostTrackingLevel& level,
    std::array<std::vector<uchar4>, kPaneCount>* panes) {
  RenderRgbToRgba(config.image_extent, rgb, &(*panes)[kRgbPane]);
  RenderDepthToRgba(config.image_extent, kDisplayDepthMinMeters,
                    kDisplayDepthMaxMeters, level.depth,
                    &(*panes)[kDepthPane]);
  RenderVerticesToRgba(config.image_extent, config.intrinsics,
                       kDisplayDepthMinMeters, kDisplayDepthMaxMeters,
                       level.vertices, &(*panes)[kVertexPane]);
  RenderNormalsToRgba(config.image_extent, level.normals,
                      &(*panes)[kNormalPane]);
}

struct CudaReconstructionPipeline::Impl {
  explicit Impl(const FrameSourceConfig& source_config)
      : config(source_config), voxel_size(VoxelSize(kVolumeConfig)) {
    HostTsdfVolume host_volume;
    InitializeTsdfVolume(
        MakeBlockMajorLayout(kVolumeConfig.resolution,
                             kVolumeConfig.block_shape),
        &host_volume);
    UploadTsdfVolume(host_volume, &device_volume);
  }

  FrameSourceConfig config;
  float3 voxel_size;
  DeviceTsdfVolume device_volume;
  DeviceVector<uchar3> device_rgb;
  DeviceVector<float> device_raw_depth;
  DeviceTrackingLevel device_level;
  DeviceSurfacePrediction device_prediction;
};

CudaReconstructionPipeline::CudaReconstructionPipeline(
    const FrameSourceConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

CudaReconstructionPipeline::~CudaReconstructionPipeline() = default;

void CudaReconstructionPipeline::ProcessFrame(
    const FrameMetadata& metadata, const std::vector<float>& raw_depth,
    const std::vector<uchar3>& rgb, const HostTrackingLevel& live_level,
    const std::array<uchar4*, kPaneCount>& mapped_panes) {
  UploadImage(rgb, &impl_->device_rgb);
  UploadImage(raw_depth, &impl_->device_raw_depth);
  UploadTrackingLevel(live_level, &impl_->device_level);

  const CameraPose world_from_camera =
      ToCameraPose(metadata.dataset_world_from_camera);
  IntegrateCuda(impl_->config.image_extent, impl_->config.intrinsics,
                world_from_camera, impl_->device_raw_depth,
                kVolumeConfig.origin_m, impl_->voxel_size,
                kVolumeConfig.truncation_distance_m,
                kVolumeConfig.max_weight, &impl_->device_volume);
  CheckCuda(cudaGetLastError(), "IntegrateCuda launch");

  RaycastCuda(impl_->device_volume, nullptr, impl_->config.image_extent,
              impl_->config.intrinsics, world_from_camera,
              kVolumeConfig.origin_m, impl_->voxel_size,
              kVolumeConfig.truncation_distance_m, kRaycastConfig,
              &impl_->device_prediction);
  CheckCuda(cudaGetLastError(), "RaycastCuda launch");

  RenderReconstructionPanesDevice(impl_->config.image_extent, impl_->device_rgb,
                                  impl_->device_level.depth,
                                  impl_->device_prediction, mapped_panes);
}

}  // namespace macro_fusion
