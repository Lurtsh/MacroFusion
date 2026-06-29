#pragma once

/// @file
/// @brief Declares host preprocessing operations.

#include "core/math_types.cuh"
#include "modules/tracking.h"

namespace macro_fusion {

/// @brief Filters metric depth while preserving sharp depth changes.
/// @param[in] image_extent Width and height represented by `input_depth`.
/// @param[in] bilateral_params `x/y/z` hold positive spatial sigma, positive
///     range sigma in meters, and a nonnegative pixel radius rounded down.
/// @param[in] input_depth Row-major metric depth with zero denoting invalid
///     samples.
/// @param[out] output_depth Non-null storage resized and overwritten with the
///     filtered depth image.
/// @throws std::invalid_argument If the output pointer, extent, sigma values,
///     or input storage do not satisfy the preprocessing contract.
/// @note Invalid center samples pass through as zero.
void BilateralFilter(uint2 image_extent, float3 bilateral_params,
                     const std::vector<float>& input_depth,
                     std::vector<float>* output_depth);

/// @brief Downsamples metric depth by two while rejecting discontinuities.
/// @param[in] input_extent Width and height represented by `input_depth`.
/// @param[in] depth_discontinuity Maximum accepted metric-depth difference from
///     each 2x2 block's top-left reference sample.
/// @param[in] input_depth Row-major metric depth with zero denoting invalid
///     samples.
/// @param[out] output_depth Non-null storage resized and overwritten with
///     `input_extent.x / 2 * input_extent.y / 2` samples.
/// @throws std::invalid_argument If the output pointer, extent, or input
///     storage do not satisfy the preprocessing contract.
/// @note Zero samples never contribute to block averages.
void HalfSample(uint2 input_extent, float depth_discontinuity,
                const std::vector<float>& input_depth,
                std::vector<float>* output_depth);

/// @brief Builds a filtered host tracking pyramid for projective ICP.
/// @param[in] image_extent Full-resolution width and height represented by
///     `raw_depth`.
/// @param[in] intrinsics Full-resolution `fx/fy/cx/cy` in `x/y/z/w`; focal
///     lengths must be positive.
/// @param[in] bilateral_params `x/y/z` hold positive spatial sigma, positive
///     range sigma in meters, and a nonnegative pixel radius rounded down.
/// @param[in] raw_depth Full-resolution row-major metric depth with zero
///     denoting invalid samples.
/// @param[in] level_count Number of levels to produce; each level `l` has
///     extent `{image_extent.x >> l, image_extent.y >> l}`.
/// @return Host levels whose depth, vertices, and normals are populated for
///     tracking; surface-map validity is packed in each `w` lane.
/// @throws std::invalid_argument If the level count, extent, intrinsics,
///     bilateral parameters, or raw depth storage do not satisfy the
///     preprocessing contract.
/// @note Intrinsics are scaled per level using the half-pixel center
///     convention.
std::vector<HostTrackingLevel> BuildTrackingPyramid(
    uint2 image_extent, float4 intrinsics, float3 bilateral_params,
    const std::vector<float>& raw_depth, unsigned level_count);

/// @brief Back-projects one tracking level's depth into camera-frame vertices.
/// @param[in] image_extent Width and height represented by `level`.
/// @param[in] level_intrinsics Scaled `fx/fy/cx/cy` in `x/y/z/w`.
/// @param[in,out] level Non-null level whose metric depth is read and whose
///     vertices are resized and overwritten.
/// @throws std::invalid_argument If the level, extent, intrinsics, or depth
///     storage do not satisfy the preprocessing contract.
/// @note Non-finite or non-positive depth produces a zero vertex with `w` set
///     to zero.
void BackProject(uint2 image_extent, float4 level_intrinsics,
                 HostTrackingLevel* level);

/// @brief Computes camera-frame normals for one back-projected tracking level.
/// @param[in] image_extent Width and height represented by `level`.
/// @param[in,out] level Non-null level whose depth and vertex `w` validity are
///     read; normals are resized and overwritten with validity in `w`.
/// @throws std::invalid_argument If the level, extent, or input storage do not
///     satisfy the preprocessing contract.
/// @note Borders, unsupported neighborhoods, degenerate cross products, and
///     opposing-neighbor depth changes above 0.1 meters are invalidated.
void ComputeNormals(uint2 image_extent, HostTrackingLevel* level);

}  // namespace macro_fusion
