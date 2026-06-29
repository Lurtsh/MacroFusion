#pragma once

/// @file
/// @brief Defines CUDA-friendly small-vector, camera, and pose math helpers.

#include <cuda_runtime.h>
#include <cmath>
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

/// @brief Computes a single-precision fused multiply-add on host or device.
/// @param[in] a Multiplicand for the product term.
/// @param[in] b Multiplier for the product term.
/// @param[in] c Addend accumulated with one final rounding.
/// @return `a * b + c` using the platform fused operation.
__host__ __device__ __forceinline__ float Fma(float a, float b, float c) {
#ifdef __CUDA_ARCH__
  return fmaf(a, b, c);
#else
  return std::fma(a, b, c);
#endif
}

/// @brief Computes the reciprocal square root on host or device.
/// @param[in] value Positive finite input value.
/// @return `1 / sqrt(value)` using the CUDA intrinsic on device.
__host__ __device__ __forceinline__ float Rsqrt(float value) {
#ifdef __CUDA_ARCH__
  return rsqrtf(value);
#else
  return 1.0f / std::sqrt(value);
#endif
}

/// @brief Stores `world_from_camera` as three row-major 3x4 `float4` rows.
///
/// Translation is packed in each row's `.w` lane: `r0.w = tx`, `r1.w = ty`,
/// and `r2.w = tz`. The layout is 48 bytes and 16-byte aligned so CUDA kernels
/// can load each row with vectorized 128-bit memory instructions.
struct alignas(16) CameraPose {
  float4 r0;  ///< `{R00, R01, R02, tx}`.
  float4 r1;  ///< `{R10, R11, R12, ty}`.
  float4 r2;  ///< `{R20, R21, R22, tz}`.
};

static_assert(sizeof(CameraPose) == 3 * sizeof(float4));
static_assert(alignof(CameraPose) == alignof(float4));

/// @brief Creates the identity `world_from_camera` pose.
/// @return A row-packed camera pose with identity rotation and zero translation.
__host__ __device__ __forceinline__ CameraPose IdentityCameraPose() {
  return CameraPose{float4{1.0f, 0.0f, 0.0f, 0.0f},
                    float4{0.0f, 1.0f, 0.0f, 0.0f},
                    float4{0.0f, 0.0f, 1.0f, 0.0f}};
}

/// @brief Adds the xyz lanes of two valid surface vectors.
/// @param[in] a First vector in meters or unit-normal coordinates.
/// @param[in] b Second vector in the same units and coordinate frame as `a`.
/// @return `a + b` in xyz with `.w = 1.0f`; input validity is assumed.
__host__ __device__ __forceinline__ float4 Add(const float4& a,
                                               const float4& b) {
  return float4{a.x + b.x, a.y + b.y, a.z + b.z, 1.0f};
}

/// @brief Subtracts the xyz lanes of two valid surface vectors.
/// @param[in] a Vector providing the minuend coordinates.
/// @param[in] b Vector providing the subtrahend coordinates.
/// @return `a - b` in xyz with `.w = 1.0f`; input validity is assumed.
__host__ __device__ __forceinline__ float4 Subtract(const float4& a,
                                                    const float4& b) {
  return float4{a.x - b.x, a.y - b.y, a.z - b.z, 1.0f};
}

/// @brief Scales the xyz lanes of a valid surface vector.
/// @param[in] a Vector in meters or unit-normal coordinates.
/// @param[in] s Scalar multiplier applied to xyz only.
/// @return `a * s` in xyz with `.w = 1.0f`; input validity is assumed.
__host__ __device__ __forceinline__ float4 Multiply(const float4& a,
                                                    const float& s) {
  return float4{s * a.x, s * a.y, s * a.z, 1.0f};
}

/// @brief Scales the xyz lanes of a valid surface vector.
/// @param[in] s Scalar multiplier applied to xyz only.
/// @param[in] a Vector in meters or unit-normal coordinates.
/// @return `s * a` in xyz with `.w = 1.0f`; input validity is assumed.
__host__ __device__ __forceinline__ float4 Multiply(const float& s,
                                                    const float4& a) {
  return Multiply(a, s);
}

/// @brief Divides the xyz lanes of a valid surface vector by a scalar.
/// @param[in] a Vector in meters or unit-normal coordinates.
/// @param[in] s Nonzero divisor; ordinary `/` is used for numerical fidelity.
/// @return `a / s` in xyz with `.w = 1.0f`; input validity is assumed.
__host__ __device__ __forceinline__ float4 Divide(const float4& a,
                                                  const float& s) {
  return float4{a.x / s, a.y / s, a.z / s, 1.0f};
}

/// @brief Computes the xyz dot product of two valid surface vectors.
/// @param[in] a First vector in meters or unit-normal coordinates.
/// @param[in] b Second vector in the same coordinate frame as `a`.
/// @return Scalar xyz dot product; `.w` validity lanes are ignored.
__host__ __device__ __forceinline__ float Dot(const float4& a,
                                             const float4& b) {
  return Fma(a.z, b.z, Fma(a.y, b.y, a.x * b.x));
}

/// @brief Computes the xyz cross product of two valid surface vectors.
/// @param[in] a First vector in meters or unit-normal coordinates.
/// @param[in] b Second vector in the same coordinate frame as `a`.
/// @return `a x b` in xyz with `.w = 1.0f`; input validity is assumed.
__host__ __device__ __forceinline__ float4 Cross(const float4& a,
                                                const float4& b) {
  return float4{Fma(a.y, b.z, -a.z * b.y),
                Fma(a.z, b.x, -a.x * b.z),
                Fma(a.x, b.y, -a.y * b.x), 1.0f};
}

/// @brief Computes the squared xyz Euclidean norm of a valid surface vector.
/// @param[in] a Vector in meters or unit-normal coordinates.
/// @return `dot(a, a)` over xyz only; `.w` validity is ignored.
__host__ __device__ __forceinline__ float SquaredNorm(const float4& a) {
  return Dot(a, a);
}

/// @brief Normalizes the xyz lanes of a valid nonzero surface vector.
/// @param[in] a Vector whose xyz squared norm is positive and finite.
/// @return Unit-length xyz vector with `.w = 1.0f`; input validity is assumed.
__host__ __device__ __forceinline__ float4 Normalize(const float4& a) {
  const float inv_norm = Rsqrt(SquaredNorm(a));
  return Multiply(a, inv_norm);
}

/// @brief Projects a valid camera-space point into image coordinates.
/// @param[in] intrinsics Camera intrinsics `{fx, fy, cx, cy}` in pixels.
/// @param[in] camera_point Camera-space point in meters with positive `z`.
/// @return Pixel coordinates `(fx*x/z + cx, fy*y/z + cy)`.
__host__ __device__ __forceinline__ float2 CameraToImageSpace(
    const float4& intrinsics, const float4& camera_point) {
  const float inv_z = 1.0f / camera_point.z;
  return float2{Fma(intrinsics.x * camera_point.x, inv_z, intrinsics.z),
                Fma(intrinsics.y * camera_point.y, inv_z, intrinsics.w)};
}

/// @brief Lifts image coordinates to a unit-depth camera-space ray point.
///
/// The returned `.w = 1.0f` is the surface-vector validity lane used throughout
/// the pipeline, not a homogeneous direction flag. Use rotation-only pose math
/// for ray directions so translation is not applied.
///
/// @param[in] intrinsics Camera intrinsics `{fx, fy, cx, cy}` in pixels.
/// @param[in] pixel Pixel coordinate `(u, v)` in image units.
/// @return `((u - cx) / fx, (v - cy) / fy, 1, 1)` in camera space.
__host__ __device__ __forceinline__ float4 ImageToCameraSpace(
    const float4& intrinsics, const float2& pixel) {
  return float4{(pixel.x - intrinsics.z) / intrinsics.x,
                (pixel.y - intrinsics.w) / intrinsics.y, 1.0f, 1.0f};
}

/// @brief Transforms a camera-frame point to world space.
///
/// This computes `world_from_camera * [point.xyz, 1]` with the pose rows packed
/// for FMA-friendly dot products. The input `.w` validity lane is assumed valid
/// and is not used as a homogeneous coordinate.
///
/// @param[in] pose Camera pose mapping camera coordinates to world coordinates.
/// @param[in] point Camera-frame point in meters.
/// @return World-frame point in meters with `.w = 1.0f`.
__host__ __device__ __forceinline__ float4 TransformVector(
    const CameraPose& pose, const float4& point) {
  return float4{
      Fma(pose.r0.z, point.z,
           Fma(pose.r0.y, point.y, Fma(pose.r0.x, point.x, pose.r0.w))),
      Fma(pose.r1.z, point.z,
           Fma(pose.r1.y, point.y, Fma(pose.r1.x, point.x, pose.r1.w))),
      Fma(pose.r2.z, point.z,
           Fma(pose.r2.y, point.y, Fma(pose.r2.x, point.x, pose.r2.w))),
      1.0f};
}

/// @brief Rotates a camera-frame normal or direction to world space.
/// @param[in] pose Camera pose whose rotation maps camera axes to world axes.
/// @param[in] normal Unit normal or ray direction in camera coordinates.
/// @return Rotated xyz vector with `.w = 1.0f`; translation is ignored.
__host__ __device__ __forceinline__ float4 RotateNormal(
    const CameraPose& pose, const float4& normal) {
  return float4{
      Fma(pose.r0.z, normal.z,
           Fma(pose.r0.y, normal.y, pose.r0.x * normal.x)),
      Fma(pose.r1.z, normal.z,
           Fma(pose.r1.y, normal.y, pose.r1.x * normal.x)),
      Fma(pose.r2.z, normal.z,
           Fma(pose.r2.y, normal.y, pose.r2.x * normal.x)),
      1.0f};
}

/// @brief Computes the inverse camera pose `camera_from_world`.
/// @param[in] pose Rigid `world_from_camera` transform.
/// @return Inverse pose `[R^T | -R^T t]` in row-packed `CameraPose` layout.
__host__ __device__ __forceinline__ CameraPose ViewMatrix(
    const CameraPose& pose) {
  const float tx = pose.r0.w;
  const float ty = pose.r1.w;
  const float tz = pose.r2.w;
  return CameraPose{
      float4{pose.r0.x, pose.r1.x, pose.r2.x,
             -Fma(pose.r2.x, tz, Fma(pose.r1.x, ty, pose.r0.x * tx))},
      float4{pose.r0.y, pose.r1.y, pose.r2.y,
             -Fma(pose.r2.y, tz, Fma(pose.r1.y, ty, pose.r0.y * tx))},
      float4{pose.r0.z, pose.r1.z, pose.r2.z,
             -Fma(pose.r2.z, tz, Fma(pose.r1.z, ty, pose.r0.z * tx))}};
}

/// @brief Transforms a world-frame point into the camera frame.
///
/// This uses the fused `R^T * (point - t)` form instead of materializing the
/// inverse matrix, saving the inverse translation rows for per-point use.
///
/// @param[in] pose Rigid `world_from_camera` transform.
/// @param[in] world_point World-frame point in meters.
/// @return Camera-frame point in meters with `.w = 1.0f`.
__host__ __device__ __forceinline__ float4 WorldToCameraSpace(
    const CameraPose& pose, const float4& world_point) {
  const float dx = world_point.x - pose.r0.w;
  const float dy = world_point.y - pose.r1.w;
  const float dz = world_point.z - pose.r2.w;
  return float4{
      Fma(pose.r2.x, dz, Fma(pose.r1.x, dy, pose.r0.x * dx)),
      Fma(pose.r2.y, dz, Fma(pose.r1.y, dy, pose.r0.y * dx)),
      Fma(pose.r2.z, dz, Fma(pose.r1.z, dy, pose.r0.z * dx)),
      1.0f};
}

/// @brief Composes two camera poses as homogeneous 3x4 transforms.
///
/// The result is `t1 * t2`, so it maps points by applying `t2` first and then
/// `t1`. Both inputs must be rigid `world_from_camera`-style transforms.
///
/// @param[in] t1 Left transform in the composition.
/// @param[in] t2 Right transform in the composition.
/// @return Row-packed composed transform with `R = R1*R2` and `t = R1*t2 + t1`.
__host__ __device__ __forceinline__ CameraPose Compose(const CameraPose& t1,
                                                       const CameraPose& t2) {
  return CameraPose{
      float4{Fma(t1.r0.z, t2.r2.x,
                  Fma(t1.r0.y, t2.r1.x, t1.r0.x * t2.r0.x)),
             Fma(t1.r0.z, t2.r2.y,
                  Fma(t1.r0.y, t2.r1.y, t1.r0.x * t2.r0.y)),
             Fma(t1.r0.z, t2.r2.z,
                  Fma(t1.r0.y, t2.r1.z, t1.r0.x * t2.r0.z)),
             Fma(t1.r0.z, t2.r2.w,
                  Fma(t1.r0.y, t2.r1.w,
                       Fma(t1.r0.x, t2.r0.w, t1.r0.w)))},
      float4{Fma(t1.r1.z, t2.r2.x,
                  Fma(t1.r1.y, t2.r1.x, t1.r1.x * t2.r0.x)),
             Fma(t1.r1.z, t2.r2.y,
                  Fma(t1.r1.y, t2.r1.y, t1.r1.x * t2.r0.y)),
             Fma(t1.r1.z, t2.r2.z,
                  Fma(t1.r1.y, t2.r1.z, t1.r1.x * t2.r0.z)),
             Fma(t1.r1.z, t2.r2.w,
                  Fma(t1.r1.y, t2.r1.w,
                       Fma(t1.r1.x, t2.r0.w, t1.r1.w)))},
      float4{Fma(t1.r2.z, t2.r2.x,
                  Fma(t1.r2.y, t2.r1.x, t1.r2.x * t2.r0.x)),
             Fma(t1.r2.z, t2.r2.y,
                  Fma(t1.r2.y, t2.r1.y, t1.r2.x * t2.r0.y)),
             Fma(t1.r2.z, t2.r2.z,
                  Fma(t1.r2.y, t2.r1.z, t1.r2.x * t2.r0.z)),
             Fma(t1.r2.z, t2.r2.w,
                  Fma(t1.r2.y, t2.r1.w,
                       Fma(t1.r2.x, t2.r0.w, t1.r2.w)))}};
}

}  // namespace macro_fusion
