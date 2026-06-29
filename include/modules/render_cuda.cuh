#pragma once

/// @file
/// @brief Declares CUDA display conversion helpers for app-owned panes.

#include <array>

#include "modules/pipeline.h"
#include "core/device_storage.cuh"
#include "modules/raycast.h"

namespace macro_fusion {

/// @brief Converts device-resident input and model views into mapped PBO panes.
/// @param[in] image_extent Width and height shared by every display pane.
/// @param[in] rgb Current input RGB frame in row-major device storage.
/// @param[in] filtered_depth Current CPU-preprocessed depth uploaded to device.
/// @param[in] prediction Raycast output whose depth and normals are displayed.
/// @param[in,out] mapped_panes CUDA-mapped pane PBOs overwritten with RGBA8
///     pixels in shared pane-index order.
/// @throws std::runtime_error If a CUDA launch or synchronization fails.
void RenderReconstructionPanesDevice(
    uint2 image_extent, const DeviceVector<uchar3>& rgb,
    const DeviceVector<float>& filtered_depth,
    const DeviceSurfacePrediction& prediction,
    const std::array<uchar4*, kPaneCount>& mapped_panes);

}  // namespace macro_fusion
