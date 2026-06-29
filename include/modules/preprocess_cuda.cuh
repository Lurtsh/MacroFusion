#pragma once

/// @file
/// @brief Declares CUDA preprocessing operations.

#include "core/math_types.cuh"
#include "core/device_storage.cuh"

namespace macro_fusion {

/// @brief Back-projects one tracking level's depth into camera-frame vertices.
/// @param[in] image_extent Width and height represented by `level`.
/// @param[in] level_intrinsics Scaled `fx/fy/cx/cy` in `x/y/z/w`.
/// @param[in,out] level Level whose metric depth is read and whose vertices are
///     resized and overwritten.
/// @note `level` must be non-null, `image_extent` must be nonzero, intrinsics
///     must be finite with positive focal lengths, and `level->depth` must
///     already match `image_extent`.
/// @throws std::runtime_error If the CUDA launch or runtime check fails.
/// @note Non-finite or non-positive depth produces a zero vertex with `w` set
///     to zero.
void BackProject(uint2 image_extent, float4 level_intrinsics,
                 DeviceTrackingLevel* level);

/// Selects which CUDA normal-estimation kernel `ComputeNormals` launches.
enum class NormalMethod {
  kCentralDifferences,  ///< Matches the host `ComputeNormals` reference.
  kNeighborVertices,    ///< KinectFusion paper Eq. 4; comparison use only.
};

/// @brief Computes camera-frame normals for one back-projected tracking level.
/// @param[in] image_extent Width and height represented by `level`.
/// @param[in,out] level Level whose depth and vertex `w` validity are read;
///     normals are resized and overwritten with validity in `w`.
/// @param[in] method Estimation kernel to launch; central differences match the
///     host reference and are selected by default.
/// @note `level` must be non-null, `image_extent` must be nonzero, and
///     `level->depth` and `level->vertices` must already match `image_extent`.
/// @throws std::runtime_error If the CUDA launch or runtime check fails.
/// @note Borders, unsupported neighborhoods, degenerate cross products, and
///     depth changes above 0.1 meters along the estimation stencil are invalidated.
void ComputeNormals(uint2 image_extent, DeviceTrackingLevel* level,
                    NormalMethod method = NormalMethod::kCentralDifferences);

}  // namespace macro_fusion
