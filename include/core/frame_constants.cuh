#pragma once

/// @file
/// @brief Declares frame geometry stored in CUDA constant memory.

#include <driver_types.h>

#include "core/math_types.cuh"

namespace macro_fusion {

/// Base source-image extent used by CUDA image kernels.
extern __constant__ uint2 kImageExtent;

/// Base-resolution pinhole intrinsics used by CUDA image kernels.
extern __constant__ float4 kCameraIntrinsics;

/// @brief Uploads the base image extent used by CUDA image kernels.
/// @param[in] image_extent Source-image width and height in pixels.
/// @return `cudaSuccess` or the CUDA runtime copy error.
cudaError_t UploadImageExtent(uint2 image_extent);

/// @brief Uploads the bundled camera intrinsics used by CUDA image kernels.
/// @param[in] intrinsics `fx/fy/cx/cy` stored in `x/y/z/w`.
/// @return `cudaSuccess` or the CUDA runtime copy error.
cudaError_t UploadCameraIntrinsics(float4 intrinsics);

}  // namespace macro_fusion
