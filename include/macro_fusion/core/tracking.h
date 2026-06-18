#pragma once

/// @file
/// @brief Declares container-templated tracking-pyramid storage.

#include <vector>

#include "macro_fusion/core/math_types.h"

namespace macro_fusion {

/// @brief Owns the surface measurements for one tracking-pyramid level.
/// @tparam Vector Container template used for each tightly packed image.
template <template <typename> class Vector>
struct TrackingLevel {
  Vector<float> depth;      ///< Metric depth; zero denotes invalid samples.
  Vector<float4> vertices;  ///< Camera-frame xyz; w is 1 valid, 0 invalid.
  Vector<float4> normals;   ///< Unit normal xyz; w is 1 valid, 0 invalid.
};

template <typename T>
using HostVector = std::vector<T>;

using HostTrackingLevel = TrackingLevel<HostVector>;

}  // namespace macro_fusion
