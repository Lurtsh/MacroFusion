#pragma once

/// @file
/// @brief Declares the quantized block-major TSDF volume and host-safe helpers.
///
/// The volume is a structure-of-arrays of normalized binary16 distance and
/// saturating `uint8` weight (pipeline.md 2.4). Both backends share this
/// representation so CPU/CUDA comparisons include identical storage effects.
/// This header is host-safe: it includes `<cuda_fp16.h>` to convert `__half`
/// but never calls the CUDA runtime, so CPU tests compile without a device.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_fp16.h>
#include <vector_types.h>

namespace macro_fusion {

/// @brief Decorates inline math reused verbatim in CUDA kernels.
///
/// Expands to `__host__ __device__` under NVCC and to nothing for a host
/// compiler, so `Flatten`, `Unflatten`, and the voxel-geometry helpers below
/// are usable in both device kernels and host-reference code.
#if defined(__CUDACC__)
#define MF_HOST_DEVICE __host__ __device__
#else
#define MF_HOST_DEVICE
#endif

/// @brief Normalized truncated signed distance stored as IEEE binary16.
///
/// Values are normalized by the truncation distance into `[-1, 1]`; positive is
/// observed free space, negative is behind the surface.
using TsdfDistance = __half;

/// @brief Saturating per-voxel observation weight; zero means unknown.
using TsdfWeight = std::uint8_t;

/// @brief Encodes a normalized `[-1, 1]` distance into stored binary16 form.
/// @param[in] normalized_distance Truncation-normalized signed distance.
/// @return The binary16 quantization used in both CPU and CUDA storage.
MF_HOST_DEVICE inline TsdfDistance EncodeTsdf(float normalized_distance) {
  return __float2half(normalized_distance);
}

/// @brief Decodes a stored binary16 distance back to float for arithmetic.
/// @param[in] stored_distance Quantized distance read from the volume.
/// @return The float value used for sampling and fusion math.
MF_HOST_DEVICE inline float DecodeTsdf(TsdfDistance stored_distance) {
  return __half2float(stored_distance);
}

/// @brief Describes the dense TSDF geometry and storage block shape.
///
/// Voxel size is `physical_size_m / resolution`; both are configurable so voxel
/// density and workspace extent are independent experiments.
struct VolumeConfig {
  uint3 resolution;             ///< Logical voxel counts `(Nx, Ny, Nz)`.
  float3 physical_size_m;       ///< World extents `(Lx, Ly, Lz)` in meters.
  float3 origin_m;              ///< World minimum corner of voxel `(0, 0, 0)`.
  float truncation_distance_m;  ///< Positive TSDF truncation in meters.
  std::uint8_t max_weight;      ///< Saturating integration weight in `[1, 255]`.
  uint3 block_shape;            ///< Storage block voxel counts `(Bx, By, Bz)`.
};

/// @brief Padded block-major storage layout; both block and local axes are
///     `x`-major so adjacent `threadIdx.x` reads adjacent storage elements.
struct BlockMajorLayout {
  uint3 logical_resolution;      ///< Addressable voxel extent `(Nx, Ny, Nz)`.
  uint3 block_shape;             ///< Voxels per storage block `(Bx, By, Bz)`.
  uint3 block_grid;              ///< `ceil(resolution / block_shape)`.
  std::uint32_t voxels_per_block;  ///< `Bx * By * Bz`.
  std::uint32_t storage_elements;  ///< Padded element count of the SoA arrays.
};

/// @brief Builds the padded block-major layout from a configured volume.
/// @param[in] resolution Logical voxel counts; every component must be nonzero.
/// @param[in] block_shape Storage block voxel counts; every component nonzero.
/// @return A populated layout whose padded `storage_elements` fits in 32 bits.
/// @throws std::invalid_argument on a zero dimension or padded overflow above
///     `UINT32_MAX` (size math is done in 64 bits to detect the overflow).
inline BlockMajorLayout MakeBlockMajorLayout(uint3 resolution,
                                             uint3 block_shape) {
  if (resolution.x == 0 || resolution.y == 0 || resolution.z == 0 ||
      block_shape.x == 0 || block_shape.y == 0 || block_shape.z == 0) {
    throw std::invalid_argument(
        "MakeBlockMajorLayout: resolution and block_shape must be nonzero");
  }
  const auto ceil_div = [](std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
  };
  BlockMajorLayout layout;
  layout.logical_resolution = resolution;
  layout.block_shape = block_shape;
  layout.block_grid = uint3{ceil_div(resolution.x, block_shape.x),
                            ceil_div(resolution.y, block_shape.y),
                            ceil_div(resolution.z, block_shape.z)};
  layout.voxels_per_block = block_shape.x * block_shape.y * block_shape.z;
  const std::uint64_t blocks =
      static_cast<std::uint64_t>(layout.block_grid.x) * layout.block_grid.y *
      layout.block_grid.z;
  const std::uint64_t elements = blocks * layout.voxels_per_block;
  if (elements > 0xFFFFFFFFull) {
    throw std::invalid_argument(
        "MakeBlockMajorLayout: padded storage exceeds UINT32_MAX elements");
  }
  layout.storage_elements = static_cast<std::uint32_t>(elements);
  return layout;
}

/// @brief Tests whether logical voxel coordinates address a real (unpadded)
///     voxel inside the volume.
/// @param[in] layout Volume storage layout.
/// @param[in] x Logical voxel `x` coordinate.
/// @param[in] y Logical voxel `y` coordinate.
/// @param[in] z Logical voxel `z` coordinate.
/// @return `true` when `(x, y, z)` is inside `logical_resolution`.
MF_HOST_DEVICE inline bool InBounds(const BlockMajorLayout& layout,
                                    std::uint32_t x, std::uint32_t y,
                                    std::uint32_t z) {
  return x < layout.logical_resolution.x && y < layout.logical_resolution.y &&
         z < layout.logical_resolution.z;
}

/// @brief Maps logical voxel coordinates to a flat storage index.
///
/// Both the block index and the in-block local index are `x`-major, so this is
/// branch-free. Callers must pass coordinates inside the logical resolution;
/// use @ref InBounds first when the source may be padded.
///
/// @param[in] layout Volume storage layout.
/// @param[in] x Logical voxel `x` coordinate.
/// @param[in] y Logical voxel `y` coordinate.
/// @param[in] z Logical voxel `z` coordinate.
/// @return The flat index into the SoA distance/weight arrays.
MF_HOST_DEVICE inline std::uint32_t Flatten(const BlockMajorLayout& layout,
                                            std::uint32_t x, std::uint32_t y,
                                            std::uint32_t z) {
  const std::uint32_t bx = x / layout.block_shape.x;
  const std::uint32_t by = y / layout.block_shape.y;
  const std::uint32_t bz = z / layout.block_shape.z;
  const std::uint32_t lx = x - bx * layout.block_shape.x;
  const std::uint32_t ly = y - by * layout.block_shape.y;
  const std::uint32_t lz = z - bz * layout.block_shape.z;
  const std::uint32_t block_linear =
      (bz * layout.block_grid.y + by) * layout.block_grid.x + bx;
  const std::uint32_t local_linear =
      (lz * layout.block_shape.y + ly) * layout.block_shape.x + lx;
  return block_linear * layout.voxels_per_block + local_linear;
}

/// @brief Recovers logical voxel coordinates from a flat storage index.
///
/// Intended for tests, initialization, debugging, and flat-index kernels;
/// production sampling uses @ref Flatten directly. The result may address a
/// padded voxel outside the logical resolution.
///
/// @param[in] layout Volume storage layout.
/// @param[in] flat Flat index in `[0, storage_elements)`.
/// @return The logical voxel coordinate `(x, y, z)` for `flat`.
MF_HOST_DEVICE inline uint3 Unflatten(const BlockMajorLayout& layout,
                                      std::uint32_t flat) {
  const std::uint32_t block_linear = flat / layout.voxels_per_block;
  const std::uint32_t local_linear = flat - block_linear * layout.voxels_per_block;
  const std::uint32_t lx = local_linear % layout.block_shape.x;
  const std::uint32_t ly =
      (local_linear / layout.block_shape.x) % layout.block_shape.y;
  const std::uint32_t lz =
      local_linear / (layout.block_shape.x * layout.block_shape.y);
  const std::uint32_t bx = block_linear % layout.block_grid.x;
  const std::uint32_t by = (block_linear / layout.block_grid.x) % layout.block_grid.y;
  const std::uint32_t bz =
      block_linear / (layout.block_grid.x * layout.block_grid.y);
  return uint3{bx * layout.block_shape.x + lx, by * layout.block_shape.y + ly,
               bz * layout.block_shape.z + lz};
}

/// @brief Returns the world-space voxel edge length per axis.
/// @param[in] config Volume geometry.
/// @return `physical_size_m / resolution` in meters.
MF_HOST_DEVICE inline float3 VoxelSize(const VolumeConfig& config) {
  return float3{config.physical_size_m.x / config.resolution.x,
                config.physical_size_m.y / config.resolution.y,
                config.physical_size_m.z / config.resolution.z};
}

/// @brief Returns the world-space center of a logical voxel.
/// @param[in] config Volume geometry.
/// @param[in] x Logical voxel `x` coordinate.
/// @param[in] y Logical voxel `y` coordinate.
/// @param[in] z Logical voxel `z` coordinate.
/// @return `origin + (voxel + 0.5) * voxel_size` in meters.
MF_HOST_DEVICE inline float3 VoxelCenterToWorld(const VolumeConfig& config,
                                                std::uint32_t x,
                                                std::uint32_t y,
                                                std::uint32_t z) {
  const float3 voxel_size = VoxelSize(config);
  return float3{config.origin_m.x + (static_cast<float>(x) + 0.5f) * voxel_size.x,
                config.origin_m.y + (static_cast<float>(y) + 0.5f) * voxel_size.y,
                config.origin_m.z + (static_cast<float>(z) + 0.5f) * voxel_size.z};
}

/// @brief Host-resident quantized TSDF volume in block-major SoA storage.
struct HostTsdfVolume {
  BlockMajorLayout layout;             ///< Storage layout for both arrays.
  std::vector<TsdfDistance> distance;  ///< Normalized binary16 distance, x-major.
  std::vector<TsdfWeight> weight;      ///< Saturating `uint8` weight, x-major.
};

/// @brief Allocates and initializes a host volume to empty space.
///
/// Every storage element, padding included, is set to distance `+1` (normalized
/// free space) and weight `0` (unknown).
///
/// @param[in] layout Padded block-major layout from @ref MakeBlockMajorLayout.
/// @param[out] volume Non-null destination resized and overwritten.
/// @throws std::invalid_argument when `volume` is null.
inline void InitializeTsdfVolume(const BlockMajorLayout& layout,
                                 HostTsdfVolume* volume) {
  if (volume == nullptr) {
    throw std::invalid_argument("InitializeTsdfVolume: volume must be non-null");
  }
  volume->layout = layout;
  volume->distance.assign(layout.storage_elements, EncodeTsdf(1.0f));
  volume->weight.assign(layout.storage_elements, TsdfWeight{0});
}

}  // namespace macro_fusion
