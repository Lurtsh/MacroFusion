/// @file
/// @brief Defines frame geometry stored in CUDA constant memory.

#include "macro_fusion/cuda/frame_constants.cuh"

#include <cuda_runtime.h>

namespace macro_fusion {

__constant__ uint2 kImageExtent;
__constant__ float4 kCameraIntrinsics;

cudaError_t UploadImageExtent(uint2 image_extent) {
  return cudaMemcpyToSymbol(kImageExtent, &image_extent, sizeof(image_extent),
                            0, cudaMemcpyHostToDevice);
}

cudaError_t UploadCameraIntrinsics(float4 intrinsics) {
  return cudaMemcpyToSymbol(kCameraIntrinsics, &intrinsics,
                            sizeof(intrinsics), 0, cudaMemcpyHostToDevice);
}

}  // namespace macro_fusion
