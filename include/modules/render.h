#pragma once

/// @file
/// @brief Declares host display conversion operations.

#include <cstdint>
#include <vector>

#include "core/math_types.cuh"

namespace macro_fusion {

/// @brief Converts RGB pixels to an opaque RGBA display image.
/// @param[in] image_extent Width and height represented by `rgb`.
/// @param[in] rgb Row-major RGB pixels.
/// @param[out] rgba Non-null storage resized and overwritten with RGBA pixels.
void RenderRgbToRgba(uint2 image_extent, const std::vector<uchar3>& rgb,
                     std::vector<uchar4>* rgba);

/// @brief Converts metric depth to a grayscale RGBA display image.
/// @param[in] image_extent Width and height represented by `depth`.
/// @param[in] depth_min_meters Depth mapped to black; must be less than
///     `depth_max_meters`.
/// @param[in] depth_max_meters Depth mapped to white.
/// @param[in] depth Row-major metric depth with zero denoting invalid samples.
/// @param[out] rgba Non-null storage resized and overwritten with RGBA pixels.
/// @note Invalid, non-finite, and non-positive depths are rendered as black.
void RenderDepthToRgba(uint2 image_extent, float depth_min_meters,
                       float depth_max_meters,
                       const std::vector<float>& depth,
                       std::vector<uchar4>* rgba);

/// @brief Converts camera-frame vertices to an XYZ-color RGBA display image.
/// @param[in] image_extent Width and height represented by `vertices`.
/// @param[in] intrinsics Full-resolution `fx/fy/cx/cy` in `x/y/z/w`; used to
///     derive the horizontal and vertical display range at `depth_max_meters`.
/// @param[in] depth_min_meters Depth mapped to zero blue; must be less than
///     `depth_max_meters`.
/// @param[in] depth_max_meters Depth mapped to full blue.
/// @param[in] vertices Row-major camera-frame vertices with validity in `w`.
/// @param[out] rgba Non-null storage resized and overwritten with RGBA pixels.
/// @note Invalid, non-finite, and non-positive-z vertices are rendered as
///     black.
void RenderVerticesToRgba(uint2 image_extent, float4 intrinsics,
                          float depth_min_meters, float depth_max_meters,
                          const std::vector<float4>& vertices,
                          std::vector<uchar4>* rgba);

/// @brief Converts normal vectors to an RGB display image.
/// @param[in] image_extent Width and height represented by `normals`.
/// @param[in] normals Row-major normal vectors in the `[-1, 1]` component
///     range with validity in `w`.
/// @param[out] rgba Non-null storage resized and overwritten with RGBA pixels.
void RenderNormalsToRgba(uint2 image_extent,
                         const std::vector<float4>& normals,
                         std::vector<uchar4>* rgba);

}  // namespace macro_fusion
