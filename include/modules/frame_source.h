#pragma once

/// @file
/// @brief Declares the TUM host RGB-D frame source.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <tuple>
#include <vector>

#include "core/config.h"
#include "core/math_types.cuh"

namespace macro_fusion {

/// @brief Describes one loaded RGB-D frame.
struct FrameMetadata {
  uint64_t frame_index;
  double depth_timestamp_seconds;
  double rgb_timestamp_seconds;
  RigidTransform dataset_world_from_camera;
};

/// @brief Supplies one forward-only pass over a TUM RGB-D sequence.
class TumFrameSource {
 public:
  /// @brief Loads the configured TUM depth, RGB, and ground-truth indexes.
  /// @throws std::runtime_error if an index is missing, malformed, or empty.
  explicit TumFrameSource(const FrameSourceConfig& config);

  bool HasNext() const;

  /// @brief Loads and returns the next frame, then advances the source.
  /// @return Metadata, row-major metric depth, and row-major RGB pixels.
  /// @throws std::out_of_range if the source is exhausted.
  /// @throws std::runtime_error if image decoding or extent validation fails.
  std::tuple<FrameMetadata, std::vector<float>, std::vector<uchar3>> Next();

 private:
  std::filesystem::path path_;
  float depth_scale_to_meters_;
  uint2 image_extent_;
  std::vector<std::tuple<double, std::filesystem::path, double,
                         std::filesystem::path, RigidTransform>> entries_;
  std::size_t cursor_ = 0;
};

}  // namespace macro_fusion
