/// @file
/// @brief Implements the host conservative macro-cell clearance build.
///
/// This is the correctness baseline for pipeline.md 2.5: a full capped Chebyshev
/// distance transform over the cells that overlap the TSDF truncation band. It
/// is pure host math (no CUDA runtime), so it compiles in a plain `.cpp` and the
/// accelerated raycaster consumes the grid after an `UploadMacroCellGrid`.

#include "modules/macro_cells.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace macro_fusion {
namespace {

std::uint32_t CeilDiv(std::uint32_t value, std::uint32_t divisor) {
  return (value + divisor - 1u) / divisor;
}

}  // namespace

void RebuildMacroCells(const HostTsdfVolume& volume,
                       const MacroCellConfig& config, float3 voxel_size_meters,
                       HostMacroCellGrid* grid) {
  if (grid == nullptr) {
    throw std::invalid_argument("RebuildMacroCells: grid must be non-null");
  }
  const uint3 resolution = volume.layout.logical_resolution;
  const std::uint32_t min_resolution =
      std::min({resolution.x, resolution.y, resolution.z});
  if (config.cells_per_axis < 1u || config.cells_per_axis > min_resolution) {
    throw std::invalid_argument(
        "RebuildMacroCells: cells_per_axis must be in [1, min(resolution)]");
  }

  // Sizing: one cubic cell spans cell_size_voxels fine voxels; the macro grid is
  // ceil(resolution / cell_size_voxels) so the coarse granularity is invariant
  // under resolution sweeps (pipeline.md 2.5).
  const std::uint32_t max_axis =
      std::max({resolution.x, resolution.y, resolution.z});
  const std::uint32_t cell_size = CeilDiv(max_axis, config.cells_per_axis);
  const uint3 macro{CeilDiv(resolution.x, cell_size),
                    CeilDiv(resolution.y, cell_size),
                    CeilDiv(resolution.z, cell_size)};
  grid->resolution = macro;
  grid->cell_size_voxels = cell_size;
  grid->cell_size_meters = float3{cell_size * voxel_size_meters.x,
                                  cell_size * voxel_size_meters.y,
                                  cell_size * voxel_size_meters.z};
  const std::size_t cell_count =
      static_cast<std::size_t>(macro.x) * macro.y * macro.z;
  grid->clearance_cells.assign(cell_count, 0);

  // 1) Mark cells overlapping the observed truncation band.
  const BlockMajorLayout& layout = volume.layout;
  std::vector<std::uint8_t> marked(cell_count, 0);
  bool any_marked = false;
  for (std::uint32_t cz = 0; cz < macro.z; ++cz) {
    for (std::uint32_t cy = 0; cy < macro.y; ++cy) {
      for (std::uint32_t cx = 0; cx < macro.x; ++cx) {
        const std::uint32_t x_end = std::min((cx + 1u) * cell_size, resolution.x);
        const std::uint32_t y_end = std::min((cy + 1u) * cell_size, resolution.y);
        const std::uint32_t z_end = std::min((cz + 1u) * cell_size, resolution.z);
        bool band = false;
        for (std::uint32_t z = cz * cell_size; z < z_end && !band; ++z) {
          for (std::uint32_t y = cy * cell_size; y < y_end && !band; ++y) {
            for (std::uint32_t x = cx * cell_size; x < x_end && !band; ++x) {
              const std::uint32_t flat = Flatten(layout, x, y, z);
              if (volume.weight[flat] > 0) {
                const float distance = DecodeTsdf(volume.distance[flat]);
                if (distance > -1.0f && distance < 1.0f) {
                  band = true;
                }
              }
            }
          }
        }
        if (band) {
          marked[MacroCellFlatten(macro, cx, cy, cz)] = 1u;
          any_marked = true;
        }
      }
    }
  }

  // No surface yet: the whole volume is guaranteed-free, so cap every cell.
  if (!any_marked) {
    std::fill(grid->clearance_cells.begin(), grid->clearance_cells.end(),
              config.max_clearance_cells);
    return;
  }

  // 2) Capped 26-connected BFS = Chebyshev distance (in cells) to the band.
  const std::uint32_t cap = static_cast<std::uint32_t>(config.max_clearance_cells) +
                            config.safety_apron_cells + 1u;
  std::vector<std::uint32_t> distance(cell_count, cap);
  std::vector<std::uint32_t> frontier;
  std::vector<std::uint32_t> next;
  for (std::size_t i = 0; i < cell_count; ++i) {
    if (marked[i]) {
      distance[i] = 0;
      frontier.push_back(static_cast<std::uint32_t>(i));
    }
  }
  for (std::uint32_t layer = 1; layer <= cap && !frontier.empty(); ++layer) {
    next.clear();
    for (std::uint32_t index : frontier) {
      const std::uint32_t cx = index % macro.x;
      const std::uint32_t cy = (index / macro.x) % macro.y;
      const std::uint32_t cz = index / (macro.x * macro.y);
      for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0 && dz == 0) {
              continue;
            }
            const int nx = static_cast<int>(cx) + dx;
            const int ny = static_cast<int>(cy) + dy;
            const int nz = static_cast<int>(cz) + dz;
            if (nx < 0 || ny < 0 || nz < 0 ||
                nx >= static_cast<int>(macro.x) ||
                ny >= static_cast<int>(macro.y) ||
                nz >= static_cast<int>(macro.z)) {
              continue;
            }
            const std::uint32_t neighbor = MacroCellFlatten(
                macro, static_cast<std::uint32_t>(nx),
                static_cast<std::uint32_t>(ny), static_cast<std::uint32_t>(nz));
            if (distance[neighbor] > layer) {
              distance[neighbor] = layer;
              next.push_back(neighbor);
            }
          }
        }
      }
    }
    frontier.swap(next);
  }

  // 3) Clearance = Chebyshev distance minus the safety apron, capped to a byte.
  for (std::size_t i = 0; i < cell_count; ++i) {
    std::uint32_t clearance = distance[i] > config.safety_apron_cells
                                  ? distance[i] - config.safety_apron_cells
                                  : 0u;
    if (clearance > config.max_clearance_cells) {
      clearance = config.max_clearance_cells;
    }
    grid->clearance_cells[i] = static_cast<MacroCellDistance>(clearance);
  }
}

}  // namespace macro_fusion
