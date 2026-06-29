#pragma once

/// @file
/// @brief Declares CUDA tracking storage and explicit host/device transfers.

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <vector>

#include "modules/tracking.h"
#include "core/volume.h"
#include "modules/macro_cells.h"

namespace macro_fusion {

template <typename T>
using DeviceVector = thrust::device_vector<T>;

using DeviceTrackingLevel = TrackingLevel<DeviceVector>;

/// @brief Device-resident quantized TSDF volume in block-major SoA storage.
///
/// Mirrors @ref HostTsdfVolume field for field; only the container family
/// differs, so transfers are plain per-array @ref UploadImage / @ref
/// DownloadImage calls plus a direct copy of the POD layout.
struct DeviceTsdfVolume {
  BlockMajorLayout layout;  ///< Storage layout shared with the host volume.
  thrust::device_vector<TsdfDistance> distance;  ///< Normalized binary16 distance.
  thrust::device_vector<TsdfWeight> weight;      ///< Saturating `uint8` weight.
};

/// @brief Copies a host vector into reusable device storage.
/// @param[in] host Source storage for one tightly packed image.
/// @param[out] device Non-null destination resized and overwritten with the
///     source data.
/// @note The transfer completes before this function returns.
template <typename T>
void UploadImage(const std::vector<T>& host,
                 thrust::device_vector<T>* device) {
  device->assign(host.begin(), host.end());
}

/// @brief Copies a device vector into reusable host storage.
/// @param[in] device Source storage for one tightly packed image.
/// @param[out] host Non-null destination resized and overwritten with the
///     source data.
/// @note The transfer completes before this function returns.
template <typename T>
void DownloadImage(const thrust::device_vector<T>& device,
                   std::vector<T>* host) {
  host->resize(device.size());
  thrust::copy(device.begin(), device.end(), host->begin());
}

/// @brief Copies a host tracking level into reusable device storage.
/// @param[in] host Source level with byte-identical surface-map layout.
/// @param[out] device Non-null destination resized and overwritten field by
///     field.
/// @note The transfer completes before this function returns.
inline void UploadTrackingLevel(const HostTrackingLevel& host,
                                DeviceTrackingLevel* device) {
  UploadImage(host.depth, &device->depth);
  UploadImage(host.vertices, &device->vertices);
  UploadImage(host.normals, &device->normals);
}

/// @brief Copies a device tracking level into reusable host storage.
/// @param[in] device Source level with byte-identical surface-map layout.
/// @param[out] host Non-null destination resized and overwritten field by
///     field.
/// @note The transfer completes before this function returns.
inline void DownloadTrackingLevel(const DeviceTrackingLevel& device,
                                  HostTrackingLevel* host) {
  DownloadImage(device.depth, &host->depth);
  DownloadImage(device.vertices, &host->vertices);
  DownloadImage(device.normals, &host->normals);
}

/// @brief Copies a host TSDF volume into reusable device storage.
/// @param[in] host Source volume with block-major SoA distance/weight arrays.
/// @param[out] device Non-null destination resized and overwritten field by
///     field; the POD layout is copied directly.
/// @note The transfer completes before this function returns.
inline void UploadTsdfVolume(const HostTsdfVolume& host,
                             DeviceTsdfVolume* device) {
  device->layout = host.layout;
  UploadImage(host.distance, &device->distance);
  UploadImage(host.weight, &device->weight);
}

/// @brief Copies a device TSDF volume into reusable host storage.
/// @param[in] device Source volume with block-major SoA distance/weight arrays.
/// @param[out] host Non-null destination resized and overwritten field by
///     field; the POD layout is copied directly.
/// @note The transfer completes before this function returns.
inline void DownloadTsdfVolume(const DeviceTsdfVolume& device,
                               HostTsdfVolume* host) {
  host->layout = device.layout;
  DownloadImage(device.distance, &host->distance);
  DownloadImage(device.weight, &host->weight);
}

/// @brief Device-resident conservative macro-cell clearance grid.
///
/// Mirrors @ref HostMacroCellGrid; only the clearance container family differs.
struct DeviceMacroCellGrid {
  uint3 resolution;                ///< Macro-grid cell counts per axis.
  std::uint32_t cell_size_voxels;  ///< Fine voxels spanned per macro cell.
  float3 cell_size_meters;         ///< `cell_size_voxels * voxel_size`.
  thrust::device_vector<MacroCellDistance> clearance_cells;  ///< x-major.
};

/// @brief Copies a host macro-cell grid into reusable device storage.
/// @param[in] host Source grid with x-major clearance cells.
/// @param[out] device Non-null destination resized and overwritten.
/// @note The transfer completes before this function returns.
inline void UploadMacroCellGrid(const HostMacroCellGrid& host,
                                DeviceMacroCellGrid* device) {
  device->resolution = host.resolution;
  device->cell_size_voxels = host.cell_size_voxels;
  device->cell_size_meters = host.cell_size_meters;
  UploadImage(host.clearance_cells, &device->clearance_cells);
}

/// @brief Copies a device macro-cell grid into reusable host storage.
/// @param[in] device Source grid with x-major clearance cells.
/// @param[out] host Non-null destination resized and overwritten.
/// @note The transfer completes before this function returns.
inline void DownloadMacroCellGrid(const DeviceMacroCellGrid& device,
                                  HostMacroCellGrid* host) {
  host->resolution = device.resolution;
  host->cell_size_voxels = device.cell_size_voxels;
  host->cell_size_meters = device.cell_size_meters;
  DownloadImage(device.clearance_cells, &host->clearance_cells);
}

}  // namespace macro_fusion
