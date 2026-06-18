#pragma once

/// @file
/// @brief Declares the minimal frame-source configuration.

#include <filesystem>

#include "macro_fusion/core/math_types.h"

namespace macro_fusion {

/// @brief Configures a file-based frame source.
struct FrameSourceConfig {
  std::filesystem::path path;
  float depth_scale_to_meters;  ///< Positive multiplier for encoded depth.
  uint2 image_extent;            ///< Width and height of every source image.
  float4 intrinsics;  ///< `x/y/z/w` store `fx/fy/cx/cy`, respectively.
};

/// @brief Returns the config for the TUM RGB-D freiburg1_xyz sequence.
inline FrameSourceConfig TumFreiburgConfig() {
  return FrameSourceConfig{
      "data/tum/rgbd_dataset_freiburg1_xyz",
      1.0f / 5000.0f,
      uint2{640, 480},
      float4{525.0f, 525.0f, 319.5f, 239.5f}};
}

}  // namespace macro_fusion
