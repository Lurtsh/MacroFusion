#pragma once

/// @file
/// @brief Declares CUDA tracking storage and explicit host/device transfers.

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <vector>

#include "macro_fusion/core/tracking.h"

namespace macro_fusion {

template <typename T>
using DeviceVector = thrust::device_vector<T>;

using DeviceTrackingLevel = TrackingLevel<DeviceVector>;

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

}  // namespace macro_fusion
