/// @file
/// @brief Provides host display conversion operation definitions.

#include "macro_fusion/modules/render.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace macro_fusion {
namespace {

std::size_t PixelCount(uint2 image_extent) {
  if (image_extent.x == 0 || image_extent.y == 0) {
    throw std::invalid_argument("image extent must be nonzero");
  }
  if (image_extent.x >
      std::numeric_limits<std::size_t>::max() / image_extent.y) {
    throw std::invalid_argument("image extent exceeds addressable storage");
  }
  return static_cast<std::size_t>(image_extent.x) * image_extent.y;
}

void ValidateDepthRange(float depth_min_meters, float depth_max_meters) {
  if (!std::isfinite(depth_min_meters) ||
      !std::isfinite(depth_max_meters) || depth_min_meters < 0.0f ||
      depth_min_meters >= depth_max_meters) {
    throw std::invalid_argument("display depth range must be finite and ordered");
  }
}

void ValidateIntrinsics(float4 intrinsics) {
  if (!std::isfinite(intrinsics.x) || intrinsics.x <= 0.0f ||
      !std::isfinite(intrinsics.y) || intrinsics.y <= 0.0f ||
      !std::isfinite(intrinsics.z) || !std::isfinite(intrinsics.w)) {
    throw std::invalid_argument(
        "camera intrinsics must be finite with positive focal lengths");
  }
}

bool IsFinite(float4 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

uint8_t UnitToByte(float value) {
  const float clamped_value = std::clamp(value, 0.0f, 1.0f);
  return static_cast<uint8_t>(std::lround(clamped_value * 255.0f));
}

uchar4 Rgba(uint8_t red, uint8_t green, uint8_t blue) {
  return uchar4{red, green, blue, uint8_t{255}};
}

uchar4 Black() { return Rgba(0, 0, 0); }

float DepthUnit(float depth, float depth_min_meters,
                float depth_max_meters) {
  return (depth - depth_min_meters) /
         (depth_max_meters - depth_min_meters);
}

}  // namespace

void RenderRgbToRgba(uint2 image_extent, const std::vector<uchar3>& rgb,
                     std::vector<uchar4>* rgba) {
  if (rgba == nullptr) {
    throw std::invalid_argument("rgba storage must not be null");
  }

  const std::size_t pixel_count = PixelCount(image_extent);
  if (rgb.size() != pixel_count) {
    throw std::invalid_argument("rgb storage does not match extent");
  }

  rgba->resize(pixel_count);
  for (std::size_t index = 0; index < pixel_count; ++index) {
    (*rgba)[index] = Rgba(rgb[index].x, rgb[index].y, rgb[index].z);
  }
}

void RenderDepthToRgba(uint2 image_extent, float depth_min_meters,
                       float depth_max_meters,
                       const std::vector<float>& depth,
                       std::vector<uchar4>* rgba) {
  if (rgba == nullptr) {
    throw std::invalid_argument("rgba storage must not be null");
  }
  ValidateDepthRange(depth_min_meters, depth_max_meters);

  const std::size_t pixel_count = PixelCount(image_extent);
  if (depth.size() != pixel_count) {
    throw std::invalid_argument("depth storage does not match extent");
  }

  rgba->resize(pixel_count);
  for (std::size_t index = 0; index < pixel_count; ++index) {
    const float depth_value = depth[index];
    if (!std::isfinite(depth_value) || depth_value <= 0.0f) {
      (*rgba)[index] = Black();
      continue;
    }

    const uint8_t intensity =
        UnitToByte(DepthUnit(depth_value, depth_min_meters, depth_max_meters));
    (*rgba)[index] = Rgba(intensity, intensity, intensity);
  }
}

void RenderVerticesToRgba(uint2 image_extent, float4 intrinsics,
                          float depth_min_meters, float depth_max_meters,
                          const std::vector<float4>& vertices,
                          std::vector<uchar4>* rgba) {
  if (rgba == nullptr) {
    throw std::invalid_argument("rgba storage must not be null");
  }
  ValidateIntrinsics(intrinsics);
  ValidateDepthRange(depth_min_meters, depth_max_meters);

  const std::size_t pixel_count = PixelCount(image_extent);
  if (vertices.size() != pixel_count) {
    throw std::invalid_argument("vertex storage does not match extent");
  }

  const float max_x_pixels =
      std::max(std::abs(intrinsics.z),
               std::abs(static_cast<float>(image_extent.x - 1) -
                        intrinsics.z));
  const float max_y_pixels =
      std::max(std::abs(intrinsics.w),
               std::abs(static_cast<float>(image_extent.y - 1) -
                        intrinsics.w));
  const float x_range =
      std::max(max_x_pixels * depth_max_meters / intrinsics.x,
               std::numeric_limits<float>::epsilon());
  const float y_range =
      std::max(max_y_pixels * depth_max_meters / intrinsics.y,
               std::numeric_limits<float>::epsilon());

  rgba->resize(pixel_count);
  for (std::size_t index = 0; index < pixel_count; ++index) {
    const float4 vertex = vertices[index];
    if (vertex.w == 0.0f || !IsFinite(vertex) || vertex.z <= 0.0f) {
      (*rgba)[index] = Black();
      continue;
    }

    (*rgba)[index] =
        Rgba(UnitToByte(0.5f + 0.5f * vertex.x / x_range),
             UnitToByte(0.5f + 0.5f * vertex.y / y_range),
             UnitToByte(DepthUnit(vertex.z, depth_min_meters,
                                  depth_max_meters)));
  }
}

void RenderNormalsToRgba(uint2 image_extent,
                         const std::vector<float4>& normals,
                         std::vector<uchar4>* rgba) {
  if (rgba == nullptr) {
    throw std::invalid_argument("rgba storage must not be null");
  }

  const std::size_t pixel_count = PixelCount(image_extent);
  if (normals.size() != pixel_count) {
    throw std::invalid_argument("normal storage does not match extent");
  }

  rgba->resize(pixel_count);
  for (std::size_t index = 0; index < pixel_count; ++index) {
    const float4 normal = normals[index];
    if (normal.w == 0.0f || !IsFinite(normal)) {
      (*rgba)[index] = Black();
      continue;
    }

    (*rgba)[index] = Rgba(UnitToByte(normal.x * 0.5f + 0.5f),
                          UnitToByte(normal.y * 0.5f + 0.5f),
                          UnitToByte(normal.z * 0.5f + 0.5f));
  }
}

}  // namespace macro_fusion
