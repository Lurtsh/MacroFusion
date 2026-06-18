/// @file
/// @brief Implements the TUM host RGB-D frame source.

#include "macro_fusion/modules/frame_source.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "macro_fusion/core/math_types.h"

namespace macro_fusion {
namespace {

using ImageEntry = std::pair<double, std::filesystem::path>;
using PoseEntry = std::pair<double, RigidTransform>;

RigidTransform TransformFromQuaternion(float tx, float ty, float tz, float qx,
                                       float qy, float qz, float qw) {
  const float norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (norm == 0.0f) {
    throw std::runtime_error("ground-truth quaternion has zero length");
  }
  qx /= norm;
  qy /= norm;
  qz /= norm;
  qw /= norm;

  const float xx = qx * qx;
  const float yy = qy * qy;
  const float zz = qz * qz;
  const float xy = qx * qy;
  const float xz = qx * qz;
  const float yz = qy * qz;
  const float wx = qw * qx;
  const float wy = qw * qy;
  const float wz = qw * qz;
  return RigidTransform{{1.0f - 2.0f * (yy + zz),
                         2.0f * (xy - wz),
                         2.0f * (xz + wy),
                         tx,
                         2.0f * (xy + wz),
                         1.0f - 2.0f * (xx + zz),
                         2.0f * (yz - wx),
                         ty,
                         2.0f * (xz - wy),
                         2.0f * (yz + wx),
                         1.0f - 2.0f * (xx + yy),
                         tz}};
}

std::vector<ImageEntry> LoadImageEntries(const std::filesystem::path& path,
                                         const char* feed_name) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open " + path.string());
  }

  std::vector<ImageEntry> entries;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    double timestamp = 0.0;
    std::filesystem::path relative_path;
    std::istringstream stream(line);
    if (!(stream >> timestamp >> relative_path)) {
      throw std::runtime_error("malformed " + std::string(feed_name) +
                               " entry in " + path.string());
    }
    entries.emplace_back(timestamp, std::move(relative_path));
  }
  if (entries.empty()) {
    throw std::runtime_error("no " + std::string(feed_name) + " entries in " +
                             path.string());
  }
  return entries;
}

std::vector<PoseEntry> LoadPoseEntries(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open " + path.string());
  }

  std::vector<PoseEntry> entries;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    double timestamp = 0.0;
    float tx = 0.0f;
    float ty = 0.0f;
    float tz = 0.0f;
    float qx = 0.0f;
    float qy = 0.0f;
    float qz = 0.0f;
    float qw = 0.0f;
    std::istringstream stream(line);
    if (!(stream >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
      throw std::runtime_error("malformed ground-truth entry in " +
                               path.string());
    }
    entries.emplace_back(
        timestamp, TransformFromQuaternion(tx, ty, tz, qx, qy, qz, qw));
  }
  if (entries.empty()) {
    throw std::runtime_error("no ground-truth entries in " + path.string());
  }
  return entries;
}

const RigidTransform& FindNearestPose(const std::vector<PoseEntry>& poses,
                                      double timestamp) {
  const auto after = std::lower_bound(
      poses.begin(), poses.end(), timestamp,
      [](const PoseEntry& pose, double value) { return pose.first < value; });
  if (after == poses.begin()) {
    return after->second;
  }
  if (after == poses.end()) {
    return poses.back().second;
  }
  const PoseEntry& before = *(after - 1);
  return timestamp - before.first <= after->first - timestamp
             ? before.second
             : after->second;
}

const ImageEntry& FindNearestImage(const std::vector<ImageEntry>& images,
                                   double timestamp) {
  const auto after = std::lower_bound(
      images.begin(), images.end(), timestamp,
      [](const ImageEntry& image, double value) {
        return image.first < value;
      });
  if (after == images.begin()) {
    return *after;
  }
  if (after == images.end()) {
    return images.back();
  }
  const ImageEntry& before = *(after - 1);
  return timestamp - before.first <= after->first - timestamp ? before
                                                              : *after;
}

std::vector<float> LoadDepthMeters(const std::filesystem::path& path,
                                   uint2 extent, float scale) {
  const std::size_t pixel_count =
      static_cast<std::size_t>(extent.x) * extent.y;
  std::vector<float> depth_meters(pixel_count);

  int width = 0;
  int height = 0;
  int channels = 0;
  uint16_t* pixels =
      stbi_load_16(path.string().c_str(), &width, &height, &channels, 1);
  if (pixels == nullptr) {
    throw std::runtime_error("cannot decode " + path.string());
  }
  if (width != static_cast<int>(extent.x) ||
      height != static_cast<int>(extent.y)) {
    stbi_image_free(pixels);
    throw std::runtime_error("unexpected image extent in " + path.string());
  }

  for (std::size_t i = 0; i < pixel_count; ++i) {
    depth_meters[i] = static_cast<float>(pixels[i]) * scale;
  }
  stbi_image_free(pixels);
  return depth_meters;
}

std::vector<uchar3> LoadRgb(const std::filesystem::path& path, uint2 extent) {
  const std::size_t pixel_count =
      static_cast<std::size_t>(extent.x) * extent.y;

  int width = 0;
  int height = 0;
  int channels = 0;
  uint8_t* pixels =
      stbi_load(path.string().c_str(), &width, &height, &channels, 3);
  if (pixels == nullptr) {
    throw std::runtime_error("cannot decode " + path.string());
  }
  if (width != static_cast<int>(extent.x) ||
      height != static_cast<int>(extent.y)) {
    stbi_image_free(pixels);
    throw std::runtime_error("unexpected image extent in " + path.string());
  }

  std::vector<uchar3> rgb(pixel_count);
  for (std::size_t i = 0; i < pixel_count; ++i) {
    rgb[i] = uchar3{pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2]};
  }
  stbi_image_free(pixels);
  return rgb;
}

}  // namespace

TumFrameSource::TumFrameSource(const FrameSourceConfig& config)
    : path_(config.path),
      depth_scale_to_meters_(config.depth_scale_to_meters),
      image_extent_(config.image_extent) {
  const std::vector<PoseEntry> poses =
      LoadPoseEntries(path_ / "groundtruth.txt");
  const std::vector<ImageEntry> rgb_entries =
      LoadImageEntries(path_ / "rgb.txt", "RGB");
  for (auto& [timestamp, relative_path] :
       LoadImageEntries(path_ / "depth.txt", "depth")) {
    const auto& [rgb_timestamp, rgb_relative_path] =
        FindNearestImage(rgb_entries, timestamp);
    entries_.emplace_back(timestamp, std::move(relative_path), rgb_timestamp,
                          rgb_relative_path,
                          FindNearestPose(poses, timestamp));
  }
}

bool TumFrameSource::HasNext() const { return cursor_ < entries_.size(); }

std::tuple<FrameMetadata, std::vector<float>, std::vector<uchar3>>
TumFrameSource::Next() {
  const auto& [depth_timestamp, depth_relative_path, rgb_timestamp,
               rgb_relative_path, pose] = entries_.at(cursor_);
  std::vector<float> depth_meters = LoadDepthMeters(
      path_ / depth_relative_path, image_extent_, depth_scale_to_meters_);
  std::vector<uchar3> rgb = LoadRgb(path_ / rgb_relative_path, image_extent_);
  FrameMetadata metadata{cursor_, depth_timestamp, rgb_timestamp, pose};
  ++cursor_;
  return {std::move(metadata), std::move(depth_meters), std::move(rgb)};
}

}  // namespace macro_fusion
