#pragma once

/// @file
/// @brief Declares the mathematical types shared by host and CUDA code.

#include <type_traits>

#include <vector_types.h>

namespace macro_fusion {

static_assert(std::is_trivially_copyable_v<uint2>);
static_assert(std::is_trivially_copyable_v<uchar3>);
static_assert(std::is_trivially_copyable_v<uchar4>);
static_assert(std::is_trivially_copyable_v<float3>);
static_assert(std::is_trivially_copyable_v<float4>);

/// @brief Stores a camera-to-world transform as a row-major 3x4 matrix.
struct RigidTransform {
  float data[12];
};

/// @brief Creates the identity camera-to-world transform.
inline RigidTransform IdentityRigidTransform() {
  return RigidTransform{{1.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 1.0f, 0.0f}};
}

}  // namespace macro_fusion
