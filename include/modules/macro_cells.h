#pragma once

/// @file
/// @brief Declares the conservative macro-cell clearance grid (pipeline.md 2.5).
///
/// The macro-cell grid is a coarse, conservative distance transform over the
/// TSDF truncation band. Each cell stores the guaranteed empty-space clearance
/// in whole cells (`uint8`), so the accelerated raycaster can leap that many
/// cells without missing a surface. Zero means the cell touches the observed
/// band and must fall back to fine TSDF sampling.

#include <cstdint>
#include <vector>

#include <vector_types.h>

#include "core/volume.h"

namespace macro_fusion {

/// @brief Conservative clearance measured in whole macro cells.
using MacroCellDistance = std::uint8_t;

/// @brief Configures macro-cell construction relative to the fine TSDF grid.
struct MacroCellConfig {
  std::uint32_t cells_per_axis;     ///< Macro-grid resolution; cell spans
                                    ///< `ceil(resolution / cells_per_axis)` voxels.
  std::uint32_t safety_apron_cells;  ///< Initial dilation of the marked band.
  std::uint8_t max_clearance_cells;  ///< Capped clearance stored in one byte.
};

/// @brief Host-resident coarse clearance grid; `0` means use fine sampling.
struct HostMacroCellGrid {
  uint3 resolution;            ///< `ceil(volume resolution / cells_per_axis)`.
  std::uint32_t cell_size_voxels;  ///< Fine voxels spanned per macro cell.
  float3 cell_size_meters;     ///< `cell_size_voxels * voxel_size`.
  std::vector<MacroCellDistance> clearance_cells;  ///< x-major clearance.
};

/// @brief Flattens a macro-cell coordinate with simple x-major ordering.
/// @param[in] resolution Macro-grid cell counts per axis.
/// @param[in] x Macro-cell `x` coordinate.
/// @param[in] y Macro-cell `y` coordinate.
/// @param[in] z Macro-cell `z` coordinate.
/// @return `(z * Ry + y) * Rx + x`.
MF_HOST_DEVICE inline std::uint32_t MacroCellFlatten(uint3 resolution,
                                                     std::uint32_t x,
                                                     std::uint32_t y,
                                                     std::uint32_t z) {
  return (z * resolution.y + y) * resolution.x + x;
}

/// @brief Rebuilds the conservative clearance grid from the current TSDF.
///
/// Marks cells overlapping the truncation band (`|normalized| < 1`, weight > 0),
/// dilates by the safety apron, runs a capped Chebyshev distance transform, and
/// stores the guaranteed clearance after subtracting the safety margin. Must be
/// called after every integration that changes the TSDF.
///
/// @param[in] volume Current host TSDF volume.
/// @param[in] config Macro-cell sizing and apron configuration.
/// @param[in] voxel_size_meters Per-axis fine voxel edge length in meters.
/// @param[out] grid Non-null destination resized and overwritten.
/// @throws std::invalid_argument on a null grid or `cells_per_axis` outside
///     `[1, min(resolution)]`.
void RebuildMacroCells(const HostTsdfVolume& volume,
                       const MacroCellConfig& config, float3 voxel_size_meters,
                       HostMacroCellGrid* grid);

}  // namespace macro_fusion
