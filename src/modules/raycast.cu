/// @file
/// @brief Implements the host and CUDA TSDF raycasting backends.
///
/// Both backends compile in this single NVCC translation unit and share one
/// `RaycastPixel` definition, for the same reason as `integration.cu`: the per
/// pixel traversal reuses the `core/math_types.cuh` camera helpers, which are
/// device-context math. The raycaster reads only the TSDF distance array;
/// unobserved voxels keep their `+1` initialization and read as free space, so
/// the first positive-to-negative crossing is the observed surface.

#include "modules/raycast.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace macro_fusion {
namespace {

/// @brief Decodes the TSDF distance at one logical voxel.
__host__ __device__ inline float SampleVoxel(const TsdfDistance* distance,
                                             const BlockMajorLayout& layout,
                                             std::uint32_t x, std::uint32_t y,
                                             std::uint32_t z) {
  return DecodeTsdf(distance[Flatten(layout, x, y, z)]);
}

/// @brief Trilinearly samples the normalized TSDF at a world point.
///
/// @param[out] out_value Interpolated distance; `+1` (free space) when the point
///     is outside the volume.
/// @return `true` when all eight corners are in bounds.
__host__ __device__ inline bool SampleTsdf(const TsdfDistance* distance,
                                           const BlockMajorLayout& layout,
                                           float3 origin, float3 voxel_size,
                                           float3 world, float* out_value) {
  const float gx = (world.x - origin.x) / voxel_size.x - 0.5f;
  const float gy = (world.y - origin.y) / voxel_size.y - 0.5f;
  const float gz = (world.z - origin.z) / voxel_size.z - 0.5f;
  const float fx = floorf(gx);
  const float fy = floorf(gy);
  const float fz = floorf(gz);
  const int x0 = static_cast<int>(fx);
  const int y0 = static_cast<int>(fy);
  const int z0 = static_cast<int>(fz);
  if (x0 < 0 || y0 < 0 || z0 < 0 ||
      x0 + 1 >= static_cast<int>(layout.logical_resolution.x) ||
      y0 + 1 >= static_cast<int>(layout.logical_resolution.y) ||
      z0 + 1 >= static_cast<int>(layout.logical_resolution.z)) {
    *out_value = 1.0f;
    return false;
  }
  const float tx = gx - fx;
  const float ty = gy - fy;
  const float tz = gz - fz;
  const std::uint32_t ux = static_cast<std::uint32_t>(x0);
  const std::uint32_t uy = static_cast<std::uint32_t>(y0);
  const std::uint32_t uz = static_cast<std::uint32_t>(z0);
  const float c00 = SampleVoxel(distance, layout, ux, uy, uz) * (1.0f - tx) +
                    SampleVoxel(distance, layout, ux + 1, uy, uz) * tx;
  const float c10 = SampleVoxel(distance, layout, ux, uy + 1, uz) * (1.0f - tx) +
                    SampleVoxel(distance, layout, ux + 1, uy + 1, uz) * tx;
  const float c01 = SampleVoxel(distance, layout, ux, uy, uz + 1) * (1.0f - tx) +
                    SampleVoxel(distance, layout, ux + 1, uy, uz + 1) * tx;
  const float c11 =
      SampleVoxel(distance, layout, ux, uy + 1, uz + 1) * (1.0f - tx) +
      SampleVoxel(distance, layout, ux + 1, uy + 1, uz + 1) * tx;
  const float c0 = c00 * (1.0f - ty) + c10 * ty;
  const float c1 = c01 * (1.0f - ty) + c11 * ty;
  *out_value = c0 * (1.0f - tz) + c1 * tz;
  return true;
}

/// @brief Bundles the per-launch raycast inputs other than the pixel.
struct RaycastContext {
  const TsdfDistance* distance;
  BlockMajorLayout layout;
  const MacroCellDistance* clearance;  ///< null unless kMacroCellDistance.
  uint3 macro_resolution;
  float3 macro_cell_size;
  uint2 image_extent;
  float4 intrinsics;
  CameraPose world_from_camera;
  float3 origin;
  float3 voxel_size;
  float truncation;
  RaycastConfig config;
};

/// @brief Traverses one pixel's ray and writes its surface hit (or a miss).
///
/// @param[out] out_vertex World position with `w` = validity.
/// @param[out] out_normal World unit normal with `w` = validity.
/// @param[out] out_depth Camera-frame surface depth (valid only on a hit).
__host__ __device__ inline void RaycastPixel(int u, int v,
                                             const RaycastContext& ctx,
                                             float4* out_vertex,
                                             float4* out_normal,
                                             float* out_depth,
                                             unsigned* out_fetches) {
  *out_vertex = float4{0.0f, 0.0f, 0.0f, 0.0f};
  *out_normal = float4{0.0f, 0.0f, 0.0f, 0.0f};
  *out_depth = 0.0f;
  unsigned fetches = 0;  // trilinear TSDF samples = the memory-traffic metric

  const float2 pixel{static_cast<float>(u) + 0.5f, static_cast<float>(v) + 0.5f};
  const float4 dir_camera = ImageToCameraSpace(ctx.intrinsics, pixel);
  const float dcl = sqrtf(dir_camera.x * dir_camera.x +
                          dir_camera.y * dir_camera.y +
                          dir_camera.z * dir_camera.z);
  float4 dir = RotateNormal(
      ctx.world_from_camera,
      float4{dir_camera.x / dcl, dir_camera.y / dcl, dir_camera.z / dcl, 1.0f});
  const float dwl = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
  dir = float4{dir.x / dwl, dir.y / dwl, dir.z / dwl, 1.0f};
  const float3 ray_o{ctx.world_from_camera.r0.w, ctx.world_from_camera.r1.w,
                     ctx.world_from_camera.r2.w};

  const float min_voxel =
      fminf(ctx.voxel_size.x, fminf(ctx.voxel_size.y, ctx.voxel_size.z));
  const float min_step = ctx.config.min_step_voxels * min_voxel;
  const float mu = ctx.truncation * ctx.config.truncation_step_scale;
  const float macro_min = fminf(ctx.macro_cell_size.x,
                                fminf(ctx.macro_cell_size.y,
                                      ctx.macro_cell_size.z));
  const bool use_macro =
      ctx.config.mode == RaycastMode::kMacroCellDistance && ctx.clearance;

  float t = ctx.config.near_meters;
  float prev_val = 1.0f;
  bool prev_in = false;
  float prev_t = t;
  for (std::uint32_t step_index = 0;
       step_index < ctx.config.max_iterations && t <= ctx.config.far_meters;
       ++step_index) {
    const float3 p{ray_o.x + t * dir.x, ray_o.y + t * dir.y,
                   ray_o.z + t * dir.z};
    float value;
    const bool in = SampleTsdf(ctx.distance, ctx.layout, ctx.origin,
                               ctx.voxel_size, p, &value);
    ++fetches;

    if (in && prev_in && prev_val > 0.0f && value < 0.0f) {
      // Refine the bracketed crossing, then linearly interpolate the hit.
      float a = prev_t;
      float b = t;
      float va = prev_val;
      float vb = value;
      for (std::uint32_t k = 0; k < ctx.config.zero_crossing_iterations; ++k) {
        const float mid = 0.5f * (a + b);
        const float3 pm{ray_o.x + mid * dir.x, ray_o.y + mid * dir.y,
                        ray_o.z + mid * dir.z};
        float vm;
        SampleTsdf(ctx.distance, ctx.layout, ctx.origin, ctx.voxel_size, pm, &vm);
        ++fetches;
        if (vm > 0.0f) {
          a = mid;
          va = vm;
        } else {
          b = mid;
          vb = vm;
        }
      }
      const float denom = va - vb;
      const float t_hit =
          fabsf(denom) > 1e-12f ? a + (b - a) * va / denom : 0.5f * (a + b);
      const float3 s{ray_o.x + t_hit * dir.x, ray_o.y + t_hit * dir.y,
                     ray_o.z + t_hit * dir.z};

      // Central-difference TSDF gradient -> outward surface normal.
      const float h = min_voxel;
      float xp, xm, yp, ym, zp, zm;
      bool ok = true;
      ok &= SampleTsdf(ctx.distance, ctx.layout, ctx.origin, ctx.voxel_size,
                       float3{s.x + h, s.y, s.z}, &xp);
      ok &= SampleTsdf(ctx.distance, ctx.layout, ctx.origin, ctx.voxel_size,
                       float3{s.x - h, s.y, s.z}, &xm);
      ok &= SampleTsdf(ctx.distance, ctx.layout, ctx.origin, ctx.voxel_size,
                       float3{s.x, s.y + h, s.z}, &yp);
      ok &= SampleTsdf(ctx.distance, ctx.layout, ctx.origin, ctx.voxel_size,
                       float3{s.x, s.y - h, s.z}, &ym);
      ok &= SampleTsdf(ctx.distance, ctx.layout, ctx.origin, ctx.voxel_size,
                       float3{s.x, s.y, s.z + h}, &zp);
      ok &= SampleTsdf(ctx.distance, ctx.layout, ctx.origin, ctx.voxel_size,
                       float3{s.x, s.y, s.z - h}, &zm);
      fetches += 6;
      float4 normal{0.0f, 0.0f, 0.0f, 0.0f};
      if (ok) {
        normal = float4{xp - xm, yp - ym, zp - zm, 1.0f};
        const float nl =
            sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (nl > 1e-12f) {
          normal.x /= nl;
          normal.y /= nl;
          normal.z /= nl;
        }
        if (normal.x * dir.x + normal.y * dir.y + normal.z * dir.z > 0.0f) {
          normal.x = -normal.x;
          normal.y = -normal.y;
          normal.z = -normal.z;
        }
      }
      const float4 camera =
          WorldToCameraSpace(ctx.world_from_camera, float4{s.x, s.y, s.z, 1.0f});
      *out_vertex = float4{s.x, s.y, s.z, 1.0f};
      *out_normal = float4{normal.x, normal.y, normal.z, ok ? 1.0f : 0.0f};
      *out_depth = camera.z;
      *out_fetches = fetches;
      return;
    }

    float step;
    if (ctx.config.mode == RaycastMode::kFixedStepDebug) {
      step = min_step;
    } else if (use_macro) {
      const int cx = static_cast<int>(floorf((p.x - ctx.origin.x) /
                                             ctx.macro_cell_size.x));
      const int cy = static_cast<int>(floorf((p.y - ctx.origin.y) /
                                             ctx.macro_cell_size.y));
      const int cz = static_cast<int>(floorf((p.z - ctx.origin.z) /
                                             ctx.macro_cell_size.z));
      if (cx >= 0 && cy >= 0 && cz >= 0 &&
          cx < static_cast<int>(ctx.macro_resolution.x) &&
          cy < static_cast<int>(ctx.macro_resolution.y) &&
          cz < static_cast<int>(ctx.macro_resolution.z)) {
        const MacroCellDistance clearance =
            ctx.clearance[MacroCellFlatten(ctx.macro_resolution,
                                           static_cast<std::uint32_t>(cx),
                                           static_cast<std::uint32_t>(cy),
                                           static_cast<std::uint32_t>(cz))];
        step = clearance > 0
                   ? fmaxf(min_step, static_cast<float>(clearance) * macro_min)
                   : (in ? fmaxf(min_step, fabsf(value) * mu)
                         : fmaxf(min_step, mu));
      } else {
        step = fmaxf(min_step, mu);
      }
    } else {  // kMuSkipBaseline
      step = in ? fmaxf(min_step, fabsf(value) * mu) : fmaxf(min_step, mu);
    }

    prev_t = t;
    prev_val = value;
    prev_in = in;
    t += step;
  }
  *out_fetches = fetches;  // ray exited without a hit
}

/// @brief Validates the shared raycast contract for both backends.
void ValidateRaycastContract(uint2 image_extent, float truncation,
                             RaycastMode mode, bool has_macro_cells,
                             bool prediction_is_null) {
  if (prediction_is_null) {
    throw std::invalid_argument("Raycast: prediction must be non-null");
  }
  if (image_extent.x == 0 || image_extent.y == 0) {
    throw std::invalid_argument("Raycast: image_extent must be nonzero");
  }
  if (!(truncation > 0.0f)) {
    throw std::invalid_argument("Raycast: truncation must be positive");
  }
  if (mode == RaycastMode::kMacroCellDistance && !has_macro_cells) {
    throw std::invalid_argument(
        "Raycast: kMacroCellDistance requires a macro-cell grid");
  }
}

/// @brief One thread per output pixel. `fetch_field` is an optional diagnostic
///     memory-traffic map (per-pixel trilinear sample count); null in production.
__global__ void RaycastKernel(RaycastContext ctx, float4* vertices,
                              float4* normals, float* depth,
                              unsigned* fetch_field) {
  const std::uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
  const std::uint32_t v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= ctx.image_extent.x || v >= ctx.image_extent.y) {
    return;
  }
  float4 vertex;
  float4 normal;
  float pixel_depth;
  unsigned pixel_fetches;
  RaycastPixel(static_cast<int>(u), static_cast<int>(v), ctx, &vertex, &normal,
               &pixel_depth, &pixel_fetches);
  const std::size_t i = static_cast<std::size_t>(v) * ctx.image_extent.x + u;
  vertices[i] = vertex;
  normals[i] = normal;
  depth[i] = vertex.w > 0.5f ? pixel_depth : 0.0f;
  if (fetch_field != nullptr) {
    fetch_field[i] = pixel_fetches;
  }
}

}  // namespace

void Raycast(const HostTsdfVolume& volume, const HostMacroCellGrid* macro_cells,
             uint2 image_extent, float4 intrinsics, CameraPose world_from_camera,
             float3 volume_origin_meters, float3 voxel_size_meters,
             float truncation_distance_meters, const RaycastConfig& config,
             HostSurfacePrediction* prediction,
             std::vector<unsigned>* fetch_field) {
  ValidateRaycastContract(image_extent, truncation_distance_meters, config.mode,
                          macro_cells != nullptr, prediction == nullptr);

  RaycastContext ctx;
  ctx.distance = volume.distance.data();
  ctx.layout = volume.layout;
  ctx.clearance = macro_cells ? macro_cells->clearance_cells.data() : nullptr;
  ctx.macro_resolution = macro_cells ? macro_cells->resolution : uint3{0, 0, 0};
  ctx.macro_cell_size =
      macro_cells ? macro_cells->cell_size_meters : float3{0.0f, 0.0f, 0.0f};
  ctx.image_extent = image_extent;
  ctx.intrinsics = intrinsics;
  ctx.world_from_camera = world_from_camera;
  ctx.origin = volume_origin_meters;
  ctx.voxel_size = voxel_size_meters;
  ctx.truncation = truncation_distance_meters;
  ctx.config = config;

  prediction->world_from_camera = world_from_camera;
  const std::size_t count =
      static_cast<std::size_t>(image_extent.x) * image_extent.y;
  prediction->vertices.resize(count);
  prediction->normals.resize(count);
  prediction->depth.resize(count);
  prediction->validity.resize(count);
  if (fetch_field != nullptr) {
    fetch_field->resize(count);
  }

  for (std::uint32_t v = 0; v < image_extent.y; ++v) {
    for (std::uint32_t u = 0; u < image_extent.x; ++u) {
      float4 vertex;
      float4 normal;
      float pixel_depth;
      unsigned pixel_fetches;
      RaycastPixel(static_cast<int>(u), static_cast<int>(v), ctx, &vertex,
                   &normal, &pixel_depth, &pixel_fetches);
      const std::size_t i = static_cast<std::size_t>(v) * image_extent.x + u;
      const bool hit = vertex.w > 0.5f;
      prediction->vertices[i] = float3{vertex.x, vertex.y, vertex.z};
      prediction->normals[i] = float3{normal.x, normal.y, normal.z};
      prediction->depth[i] = hit ? pixel_depth : 0.0f;
      prediction->validity[i] = hit ? 1u : 0u;
      if (fetch_field != nullptr) {
        (*fetch_field)[i] = pixel_fetches;
      }
    }
  }
}

void RaycastCuda(const DeviceTsdfVolume& volume,
                 const DeviceMacroCellGrid* macro_cells, uint2 image_extent,
                 float4 intrinsics, CameraPose world_from_camera,
                 float3 volume_origin_meters, float3 voxel_size_meters,
                 float truncation_distance_meters, const RaycastConfig& config,
                 DeviceSurfacePrediction* prediction,
                 thrust::device_vector<unsigned>* fetch_field) {
  ValidateRaycastContract(image_extent, truncation_distance_meters, config.mode,
                          macro_cells != nullptr, prediction == nullptr);

  RaycastContext ctx;
  ctx.distance = thrust::raw_pointer_cast(volume.distance.data());
  ctx.layout = volume.layout;
  ctx.clearance = macro_cells
                      ? thrust::raw_pointer_cast(macro_cells->clearance_cells.data())
                      : nullptr;
  ctx.macro_resolution = macro_cells ? macro_cells->resolution : uint3{0, 0, 0};
  ctx.macro_cell_size =
      macro_cells ? macro_cells->cell_size_meters : float3{0.0f, 0.0f, 0.0f};
  ctx.image_extent = image_extent;
  ctx.intrinsics = intrinsics;
  ctx.world_from_camera = world_from_camera;
  ctx.origin = volume_origin_meters;
  ctx.voxel_size = voxel_size_meters;
  ctx.truncation = truncation_distance_meters;
  ctx.config = config;

  prediction->world_from_camera = world_from_camera;
  const std::size_t count =
      static_cast<std::size_t>(image_extent.x) * image_extent.y;
  prediction->vertices.resize(count);
  prediction->normals.resize(count);
  prediction->depth.resize(count);
  unsigned* fetch_ptr = nullptr;
  if (fetch_field != nullptr) {
    fetch_field->resize(count);
    fetch_ptr = thrust::raw_pointer_cast(fetch_field->data());
  }

  const dim3 block(16, 16, 1);
  const dim3 grid((image_extent.x + block.x - 1) / block.x,
                  (image_extent.y + block.y - 1) / block.y, 1);
  RaycastKernel<<<grid, block>>>(
      ctx, thrust::raw_pointer_cast(prediction->vertices.data()),
      thrust::raw_pointer_cast(prediction->normals.data()),
      thrust::raw_pointer_cast(prediction->depth.data()), fetch_ptr);
}

}  // namespace macro_fusion
