/// @file
/// @brief Provides host preprocessing operation definitions.

#include "macro_fusion/modules/preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <limits>
#include <stdexcept>

namespace macro_fusion {
namespace {

constexpr float kMaxNormalDepthDiscontinuityMeters = 0.1f;

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

bool IsFinite(float3 value) {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z);
}

bool HasValidIntrinsics(float4 intrinsics) {
  return std::isfinite(intrinsics.x) && intrinsics.x > 0.0f &&
         std::isfinite(intrinsics.y) && intrinsics.y > 0.0f &&
         std::isfinite(intrinsics.z) && std::isfinite(intrinsics.w);
}

uint32_t BilateralKernelRadius(float3 bilateral_params,
                               uint2 image_extent) {
  if (!std::isfinite(bilateral_params.z) || bilateral_params.z <= 0.0f) {
    return 0;
  }

  const float maximum_radius = static_cast<float>(
      std::max(image_extent.x - 1, image_extent.y - 1));
  return static_cast<uint32_t>(
      std::min(std::floor(bilateral_params.z), maximum_radius));
}

uint32_t WindowStart(uint32_t origin, uint32_t radius) {
  return (origin > radius) ? origin - radius : 0;
}

uint32_t WindowEnd(uint32_t origin, uint32_t radius, uint32_t extent) {
  const std::uint64_t unclamped_end =
      static_cast<std::uint64_t>(origin) + radius + 1;
  return static_cast<uint32_t>(
      std::min<std::uint64_t>(extent, unclamped_end));
}

uint2 PyramidLevelExtent(uint2 image_extent, unsigned level) {
  if (level >= std::numeric_limits<decltype(image_extent.x)>::digits) {
    return uint2{0, 0};
  }
  return uint2{image_extent.x >> level, image_extent.y >> level};
}

float4 ScaleIntrinsics(float4 intrinsics, unsigned level) {
  const float scale = std::ldexp(1.0f, static_cast<int>(level));
  return float4{intrinsics.x / scale, intrinsics.y / scale,
                (intrinsics.z + 0.5f) / scale - 0.5f,
                (intrinsics.w + 0.5f) / scale - 0.5f};
}

}  // namespace

void BilateralFilter(uint2 image_extent, float3 bilateral_params,
                     const std::vector<float>& input_depth,
                     std::vector<float>* output_depth) {
  if (output_depth == nullptr) {
    throw std::invalid_argument("output depth storage must not be null");
  }
  if (!std::isfinite(bilateral_params.x) || bilateral_params.x <= 0.0f ||
      !std::isfinite(bilateral_params.y) || bilateral_params.y <= 0.0f) {
    throw std::invalid_argument(
        "bilateral sigmas must be finite and positive");
  }

  const std::size_t pixel_count = PixelCount(image_extent);
  if (input_depth.size() != pixel_count) {
    throw std::invalid_argument("input depth storage does not match extent");
  }

  std::vector<float> input_copy;
  const std::vector<float>* source_depth = &input_depth;
  if (output_depth == &input_depth) {
    input_copy = input_depth;
    source_depth = &input_copy;
  }

  output_depth->resize(pixel_count);

  const uint32_t kernel_radius =
      BilateralKernelRadius(bilateral_params, image_extent);
  const float sigma_factor_s =
      1.0f / (2.0f * bilateral_params.x * bilateral_params.x);
  const float sigma_factor_r =
      1.0f / (2.0f * bilateral_params.y * bilateral_params.y);

  std::for_each(std::execution::par_unseq, output_depth->begin(),
                output_depth->end(), [&](float& filtered_depth) {
                  const std::size_t index = static_cast<std::size_t>(
                      &filtered_depth - output_depth->data());
                  const uint32_t x_origin =
                      static_cast<uint32_t>(index % image_extent.x);
                  const uint32_t y_origin =
                      static_cast<uint32_t>(index / image_extent.x);
                  const float depth_value = (*source_depth)[index];
                  if (depth_value == 0.0f) {
                    filtered_depth = 0.0f;
                    return;
                  }

                  const uint32_t y_start =
                      WindowStart(y_origin, kernel_radius);
                  const uint32_t y_end =
                      WindowEnd(y_origin, kernel_radius, image_extent.y);
                  const uint32_t x_start =
                      WindowStart(x_origin, kernel_radius);
                  const uint32_t x_end =
                      WindowEnd(x_origin, kernel_radius, image_extent.x);

                  float max_exponent =
                      -std::numeric_limits<float>::infinity();
                  for (uint32_t y = y_start; y < y_end; ++y) {
                    for (uint32_t x = x_start; x < x_end; ++x) {
                      const float dx =
                          static_cast<float>(x_origin) - static_cast<float>(x);
                      const float dy =
                          static_cast<float>(y_origin) - static_cast<float>(y);
                      const float neighbor_depth =
                          (*source_depth)[static_cast<std::size_t>(y) *
                                              image_extent.x +
                                          x];
                      const float depth_delta = depth_value - neighbor_depth;
                      const float spatial =
                          (dx * dx + dy * dy) * sigma_factor_s;
                      const float range =
                          depth_delta * depth_delta * sigma_factor_r;
                      max_exponent =
                          std::max(max_exponent, -(spatial + range));
                    }
                  }

                  float factor = 0.0f;
                  float sum = 0.0f;
                  for (uint32_t y = y_start; y < y_end; ++y) {
                    for (uint32_t x = x_start; x < x_end; ++x) {
                      const float dx =
                          static_cast<float>(x_origin) - static_cast<float>(x);
                      const float dy =
                          static_cast<float>(y_origin) - static_cast<float>(y);
                      const float neighbor_depth =
                          (*source_depth)[static_cast<std::size_t>(y) *
                                              image_extent.x +
                                          x];
                      const float depth_delta = depth_value - neighbor_depth;
                      const float spatial =
                          (dx * dx + dy * dy) * sigma_factor_s;
                      const float range =
                          depth_delta * depth_delta * sigma_factor_r;
                      const float weight =
                          std::exp(-(spatial + range) - max_exponent);
                      factor += weight;
                      sum += weight * neighbor_depth;
                    }
                  }

                  filtered_depth = sum / factor;
                });
}

void HalfSample(uint2 input_extent, float depth_discontinuity,
                const std::vector<float>& input_depth,
                std::vector<float>* output_depth) {
  if (output_depth == nullptr) {
    throw std::invalid_argument("output depth storage must not be null");
  }

  const std::size_t input_pixel_count = PixelCount(input_extent);
  if (input_depth.size() != input_pixel_count) {
    throw std::invalid_argument("input depth storage does not match extent");
  }
  if (input_extent.x < 2 || input_extent.y < 2) {
    throw std::invalid_argument("input extent is too small to half-sample");
  }

  std::vector<float> input_copy;
  const std::vector<float>* source_depth = &input_depth;
  if (output_depth == &input_depth) {
    input_copy = input_depth;
    source_depth = &input_copy;
  }

  const uint2 output_extent{input_extent.x / 2, input_extent.y / 2};
  const std::size_t output_pixel_count = PixelCount(output_extent);
  output_depth->resize(output_pixel_count);

  std::for_each(std::execution::par_unseq, output_depth->begin(),
                output_depth->end(), [&](float& sampled_depth) {
                  const std::size_t index = static_cast<std::size_t>(
                      &sampled_depth - output_depth->data());
                  const uint32_t output_x =
                      static_cast<uint32_t>(index % output_extent.x);
                  const uint32_t output_y =
                      static_cast<uint32_t>(index / output_extent.x);
                  const uint32_t input_x = output_x * 2;
                  const uint32_t input_y = output_y * 2;
                  const float reference =
                      (*source_depth)[static_cast<std::size_t>(input_y) *
                                          input_extent.x +
                                      input_x];

                  float sum = 0.0f;
                  float count = 0.0f;
                  for (uint32_t dy = 0; dy < 2; ++dy) {
                    for (uint32_t dx = 0; dx < 2; ++dx) {
                      const std::size_t source_index =
                          static_cast<std::size_t>(input_y + dy) *
                              input_extent.x +
                          input_x + dx;
                      const float depth = (*source_depth)[source_index];
                      if (depth != 0.0f &&
                          std::abs(depth - reference) < depth_discontinuity) {
                        sum += depth;
                        count += 1.0f;
                      }
                    }
                  }

                  sampled_depth = (count > 0.0f) ? sum / count : 0.0f;
                });
}

std::vector<HostTrackingLevel> BuildTrackingPyramid(
    uint2 image_extent, float4 intrinsics, float3 bilateral_params,
    const std::vector<float>& raw_depth, unsigned level_count) {
  if (level_count == 0) {
    throw std::invalid_argument("tracking pyramid must contain a level");
  }
  if (!HasValidIntrinsics(intrinsics)) {
    throw std::invalid_argument(
        "camera intrinsics must be finite with positive focal lengths");
  }

  const std::size_t pixel_count = PixelCount(image_extent);
  if (raw_depth.size() != pixel_count) {
    throw std::invalid_argument(
        "raw depth storage does not match image extent");
  }
  for (unsigned level = 0; level < level_count; ++level) {
    const uint2 level_extent = PyramidLevelExtent(image_extent, level);
    if (level_extent.x < 3 || level_extent.y < 3) {
      throw std::invalid_argument(
          "tracking pyramid level extent is too small");
    }
  }

  std::vector<HostTrackingLevel> pyramid(level_count);
  BilateralFilter(image_extent, bilateral_params, raw_depth, &pyramid[0].depth);
  BackProject(image_extent, intrinsics, &pyramid[0]);
  ComputeNormals(image_extent, &pyramid[0]);

  const float depth_discontinuity = 3.0f * bilateral_params.y;
  for (unsigned level = 1; level < level_count; ++level) {
    const uint2 previous_extent = PyramidLevelExtent(image_extent, level - 1);
    const uint2 level_extent = PyramidLevelExtent(image_extent, level);
    HalfSample(previous_extent, depth_discontinuity, pyramid[level - 1].depth,
               &pyramid[level].depth);
    BackProject(level_extent, ScaleIntrinsics(intrinsics, level),
                &pyramid[level]);
    ComputeNormals(level_extent, &pyramid[level]);
  }

  return pyramid;
}

void BackProject(uint2 image_extent, float4 level_intrinsics,
                 HostTrackingLevel* level) {
  if (level == nullptr) {
    throw std::invalid_argument("tracking level must not be null");
  }
  if (!std::isfinite(level_intrinsics.x) || level_intrinsics.x <= 0.0f ||
      !std::isfinite(level_intrinsics.y) || level_intrinsics.y <= 0.0f ||
      !std::isfinite(level_intrinsics.z) ||
      !std::isfinite(level_intrinsics.w)) {
    throw std::invalid_argument(
        "camera intrinsics must be finite with positive focal lengths");
  }

  const std::size_t pixel_count = PixelCount(image_extent);
  if (level->depth.size() != pixel_count) {
    throw std::invalid_argument("depth storage does not match image extent");
  }

  level->vertices.assign(pixel_count, float4{0.0f, 0.0f, 0.0f, 0.0f});

  for (uint32_t y = 0; y < image_extent.y; ++y) {
    for (uint32_t x = 0; x < image_extent.x; ++x) {
      const std::size_t index =
          static_cast<std::size_t>(y) * image_extent.x + x;
      const float depth = level->depth[index];
      if (!std::isfinite(depth) || depth <= 0.0f) {
        continue;
      }

      level->vertices[index] =
          float4{(static_cast<float>(x) - level_intrinsics.z) * depth /
                     level_intrinsics.x,
                 (static_cast<float>(y) - level_intrinsics.w) * depth /
                     level_intrinsics.y,
                 depth, 1.0f};
    }
  }
}

void ComputeNormals(uint2 image_extent, HostTrackingLevel* level) {
  if (level == nullptr) {
    throw std::invalid_argument("tracking level must not be null");
  }

  const std::size_t pixel_count = PixelCount(image_extent);
  if (level->depth.size() != pixel_count ||
      level->vertices.size() != pixel_count) {
    throw std::invalid_argument(
        "tracking level storage does not match image extent");
  }

  level->normals.assign(pixel_count, float4{0.0f, 0.0f, 0.0f, 0.0f});
  if (image_extent.x < 3 || image_extent.y < 3) {
    return;
  }

  for (uint32_t y = 1; y + 1 < image_extent.y; ++y) {
    for (uint32_t x = 1; x + 1 < image_extent.x; ++x) {
      const std::size_t center =
          static_cast<std::size_t>(y) * image_extent.x + x;
      const std::size_t left = center - 1;
      const std::size_t right = center + 1;
      const std::size_t up = center - image_extent.x;
      const std::size_t down = center + image_extent.x;
      if (level->vertices[center].w == 0.0f ||
          level->vertices[left].w == 0.0f ||
          level->vertices[right].w == 0.0f ||
          level->vertices[up].w == 0.0f ||
          level->vertices[down].w == 0.0f) {
        continue;
      }
      if (std::abs(level->depth[right] - level->depth[left]) >
              kMaxNormalDepthDiscontinuityMeters ||
          std::abs(level->depth[down] - level->depth[up]) >
              kMaxNormalDepthDiscontinuityMeters) {
        continue;
      }

      const float3 horizontal{
          level->vertices[right].x - level->vertices[left].x,
          level->vertices[right].y - level->vertices[left].y,
          level->vertices[right].z - level->vertices[left].z};
      const float3 vertical{
          level->vertices[down].x - level->vertices[up].x,
          level->vertices[down].y - level->vertices[up].y,
          level->vertices[down].z - level->vertices[up].z};
      const float3 cross{horizontal.y * vertical.z -
                             horizontal.z * vertical.y,
                         horizontal.z * vertical.x -
                             horizontal.x * vertical.z,
                         horizontal.x * vertical.y -
                             horizontal.y * vertical.x};
      const float squared_norm =
          cross.x * cross.x + cross.y * cross.y + cross.z * cross.z;
      if (!IsFinite(cross) || !std::isfinite(squared_norm) ||
          squared_norm <= std::numeric_limits<float>::min()) {
        continue;
      }

      const float inverse_norm = 1.0f / std::sqrt(squared_norm);
      level->normals[center] = float4{cross.x * inverse_norm,
                                      cross.y * inverse_norm,
                                      cross.z * inverse_norm, 1.0f};
    }
  }
}

}  // namespace macro_fusion
