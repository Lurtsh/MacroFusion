#pragma once

/// @file
/// @brief Declares GL-free host application pipeline stages.

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "core/config.h"
#include "modules/tracking.h"
#include "modules/frame_source.h"

namespace macro_fusion {

constexpr std::size_t kPaneCount = 4;
constexpr std::size_t kRgbPane = 0;
constexpr std::size_t kDepthPane = 1;
constexpr std::size_t kVertexPane = 2;
constexpr std::size_t kNormalPane = 3;

/// @brief Builds the live tracking pyramid from raw metric depth.
/// @param[in] config Image geometry and intrinsics used by preprocessing and
///     surface measurement.
/// @param[in] raw_depth Row-major unfiltered metric depth with zero denoting
///     invalid samples.
/// @return The single-level live tracking pyramid.
std::vector<HostTrackingLevel> PreprocessFrame(
    const FrameSourceConfig& config, const std::vector<float>& raw_depth);

/// @brief Converts the current frame and tracking level into four debug panes.
/// @param[in] config Image geometry, intrinsics, and display conversion inputs.
/// @param[in] rgb Row-major RGB pixels for the current frame.
/// @param[in] level Live tracking level providing depth and surface maps.
/// @param[out] panes Pane storage resized and overwritten in shared pane-index
///     order.
void RenderTrackingViews(
    const FrameSourceConfig& config, const std::vector<uchar3>& rgb,
    const HostTrackingLevel& level,
    std::array<std::vector<uchar4>, kPaneCount>* panes);

/// @brief Fills the four mapped display PBOs with solid debug colors on device.
/// @param[in] image_extent Pane size in pixels (width, height).
/// @param[in,out] mapped_panes CUDA-mapped pointers to the four pane PBOs, in
///     shared pane-index order; each is overwritten with width*height pixels.
void RenderDummyPanesDevice(
    uint2 image_extent, const std::array<uchar4*, kPaneCount>& mapped_panes);

/// @brief Owns CUDA state for TSDF integration, raycast, and display conversion.
class CudaReconstructionPipeline {
 public:
  /// @brief Initializes the persistent TSDF and device work buffers.
  /// @param[in] config Frame geometry used by every reconstruction stage.
  explicit CudaReconstructionPipeline(const FrameSourceConfig& config);

  ~CudaReconstructionPipeline();

  CudaReconstructionPipeline(const CudaReconstructionPipeline&) = delete;
  CudaReconstructionPipeline& operator=(const CudaReconstructionPipeline&) =
      delete;

  /// @brief Runs one RGB-D frame through upload, integration, raycast, and render.
  /// @param[in] metadata Dataset metadata carrying the ground-truth pose.
  /// @param[in] raw_depth Row-major raw metric depth used for TSDF integration.
  /// @param[in] rgb Row-major RGB pixels rendered beside the raycasted model.
  /// @param[in] live_level CPU-preprocessed base level uploaded for display.
  /// @param[in,out] mapped_panes CUDA-mapped pane PBOs overwritten with RGBA8.
  void ProcessFrame(const FrameMetadata& metadata,
                    const std::vector<float>& raw_depth,
                    const std::vector<uchar3>& rgb,
                    const HostTrackingLevel& live_level,
                    const std::array<uchar4*, kPaneCount>& mapped_panes);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace macro_fusion
