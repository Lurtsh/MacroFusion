/// @file
/// @brief Tests host preprocessing operations.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "macro_fusion/core/tracking.h"
#include "macro_fusion/modules/preprocess.h"

namespace {

void ExpectFloat4Near(float4 actual, float4 expected, float tolerance = 1e-6f) {
  EXPECT_NEAR(actual.x, expected.x, tolerance);
  EXPECT_NEAR(actual.y, expected.y, tolerance);
  EXPECT_NEAR(actual.z, expected.z, tolerance);
}

void ExpectFloatVectorNear(const std::vector<float>& actual,
                           const std::vector<float>& expected,
                           float tolerance = 1e-6f) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], tolerance);
  }
}

macro_fusion::HostTrackingLevel MakePlanarLevel(uint2 image_extent,
                                                float depth = 1.0f) {
  macro_fusion::HostTrackingLevel level;
  level.depth.assign(static_cast<std::size_t>(image_extent.x) * image_extent.y,
                     depth);
  macro_fusion::BackProject(image_extent,
                            float4{1.0f, 1.0f, 1.0f, 1.0f}, &level);
  return level;
}

TEST(BackProjectTest, ProducesCameraFrameVerticesInRowMajorOrder) {
  macro_fusion::HostTrackingLevel level;
  level.depth = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  level.vertices.assign(1, float4{9.0f, 9.0f, 9.0f, 1.0f});

  macro_fusion::BackProject(uint2{3, 2}, float4{2.0f, 4.0f, 1.0f, 0.0f},
                            &level);

  ASSERT_EQ(level.vertices.size(), 6);
  ExpectFloat4Near(level.vertices[0], float4{-1.0f, 0.0f, 2.0f, 1.0f});
  ExpectFloat4Near(level.vertices[2], float4{1.0f, 0.0f, 2.0f, 1.0f});
  ExpectFloat4Near(level.vertices[4], float4{0.0f, 0.5f, 2.0f, 1.0f});
  for (const float4& vertex : level.vertices) {
    EXPECT_FLOAT_EQ(vertex.w, 1.0f);
  }
}

TEST(BackProjectTest, ZeroesInvalidDepthSamplesAndMarksThemInvalid) {
  macro_fusion::HostTrackingLevel level;
  level.depth = {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::quiet_NaN(), 1.0f};

  macro_fusion::BackProject(uint2{5, 1}, float4{1.0f, 1.0f, 0.0f, 0.0f},
                            &level);

  for (std::size_t index = 0; index < 4; ++index) {
    ExpectFloat4Near(level.vertices[index],
                     float4{0.0f, 0.0f, 0.0f, 0.0f});
    EXPECT_FLOAT_EQ(level.vertices[index].w, 0.0f);
  }
  ExpectFloat4Near(level.vertices[4], float4{4.0f, 0.0f, 1.0f, 1.0f});
  EXPECT_FLOAT_EQ(level.vertices[4].w, 1.0f);
}

TEST(BackProjectTest, RejectsInvalidContractInputs) {
  macro_fusion::HostTrackingLevel level;
  level.depth.assign(4, 1.0f);

  EXPECT_THROW(macro_fusion::BackProject(
                   uint2{2, 2}, float4{1.0f, 1.0f, 0.0f, 0.0f}, nullptr),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::BackProject(
                   uint2{0, 2}, float4{1.0f, 1.0f, 0.0f, 0.0f}, &level),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::BackProject(
                   uint2{2, 2}, float4{0.0f, 1.0f, 0.0f, 0.0f}, &level),
               std::invalid_argument);
  level.depth.pop_back();
  EXPECT_THROW(macro_fusion::BackProject(
                   uint2{2, 2}, float4{1.0f, 1.0f, 0.0f, 0.0f}, &level),
               std::invalid_argument);
}

TEST(ComputeNormalsTest, ProducesPositiveUnitNormalForPlanarDepth) {
  const uint2 image_extent{3, 3};
  macro_fusion::HostTrackingLevel level = MakePlanarLevel(image_extent);
  const std::vector<float> original_depth = level.depth;
  const std::vector<float4> original_vertices = level.vertices;

  macro_fusion::ComputeNormals(image_extent, &level);

  ASSERT_EQ(level.normals.size(), 9);
  ExpectFloat4Near(level.normals[4], float4{0.0f, 0.0f, 1.0f, 1.0f});
  EXPECT_FLOAT_EQ(level.normals[4].w, 1.0f);
  for (std::size_t index = 0; index < level.normals.size(); ++index) {
    if (index != 4) {
      EXPECT_FLOAT_EQ(level.normals[index].w, 0.0f);
      ExpectFloat4Near(level.normals[index],
                       float4{0.0f, 0.0f, 0.0f, 0.0f});
    }
  }
  EXPECT_EQ(level.depth, original_depth);
  ASSERT_EQ(level.vertices.size(), original_vertices.size());
  for (std::size_t index = 0; index < level.vertices.size(); ++index) {
    ExpectFloat4Near(level.vertices[index], original_vertices[index]);
    EXPECT_FLOAT_EQ(level.vertices[index].w, original_vertices[index].w);
  }
}

TEST(ComputeNormalsTest, PreservesNormalsAtRealisticFocalLengths) {
  const uint2 image_extent{3, 3};
  macro_fusion::HostTrackingLevel level;
  level.depth.assign(9, 1.0f);
  macro_fusion::BackProject(
      image_extent, float4{525.0f, 525.0f, 1.0f, 1.0f}, &level);

  macro_fusion::ComputeNormals(image_extent, &level);

  EXPECT_FLOAT_EQ(level.normals[4].w, 1.0f);
  ExpectFloat4Near(level.normals[4], float4{0.0f, 0.0f, 1.0f, 1.0f});
}

TEST(ComputeNormalsTest, RejectsUnsupportedAndDiscontinuousPixels) {
  const uint2 image_extent{3, 3};
  macro_fusion::HostTrackingLevel missing_neighbor =
      MakePlanarLevel(image_extent);
  missing_neighbor.vertices[3].w = 0.0f;
  macro_fusion::ComputeNormals(image_extent, &missing_neighbor);
  EXPECT_FLOAT_EQ(missing_neighbor.normals[4].w, 0.0f);

  macro_fusion::HostTrackingLevel discontinuous = MakePlanarLevel(image_extent);
  discontinuous.depth[3] = 0.9f;
  discontinuous.depth[5] = 1.1f;
  macro_fusion::ComputeNormals(image_extent, &discontinuous);
  EXPECT_FLOAT_EQ(discontinuous.normals[4].w, 0.0f);
}

TEST(ComputeNormalsTest, RejectsDegenerateGeometry) {
  macro_fusion::HostTrackingLevel level;
  level.depth.assign(9, 1.0f);
  level.vertices.assign(9, float4{1.0f, 1.0f, 1.0f, 1.0f});

  macro_fusion::ComputeNormals(uint2{3, 3}, &level);

  for (const float4& normal : level.normals) {
    EXPECT_FLOAT_EQ(normal.w, 0.0f);
  }
  ExpectFloat4Near(level.normals[4], float4{0.0f, 0.0f, 0.0f, 0.0f});
}

TEST(ComputeNormalsTest, InvalidatesImagesWithoutInteriorPixels) {
  macro_fusion::HostTrackingLevel level = MakePlanarLevel(uint2{2, 2});

  macro_fusion::ComputeNormals(uint2{2, 2}, &level);

  ASSERT_EQ(level.normals.size(), 4);
  for (float4 normal : level.normals) {
    ExpectFloat4Near(normal, float4{0.0f, 0.0f, 0.0f, 0.0f});
    EXPECT_FLOAT_EQ(normal.w, 0.0f);
  }
}

TEST(ComputeNormalsTest, RejectsMalformedStorage) {
  macro_fusion::HostTrackingLevel level = MakePlanarLevel(uint2{3, 3});
  level.vertices.pop_back();

  EXPECT_THROW(macro_fusion::ComputeNormals(uint2{3, 3}, &level),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::ComputeNormals(uint2{3, 3}, nullptr),
               std::invalid_argument);
}

TEST(BilateralFilterTest, PreservesConstantDepthAndPassesThroughZeros) {
  std::vector<float> depth(20, 2.0f);
  std::vector<float> filtered;

  macro_fusion::BilateralFilter(uint2{5, 4}, float3{2.0f, 0.2f, 2.0f},
                                depth, &filtered);

  ExpectFloatVectorNear(filtered, depth);

  depth.assign(9, 1.0f);
  depth[4] = 0.0f;
  macro_fusion::BilateralFilter(uint2{3, 3}, float3{1.0f, 0.1f, 1.0f},
                                depth, &filtered);

  ASSERT_EQ(filtered.size(), 9);
  EXPECT_FLOAT_EQ(filtered[4], 0.0f);
}

TEST(BilateralFilterTest, RejectsInvalidContractInputs) {
  const std::vector<float> depth(4, 1.0f);
  std::vector<float> filtered;

  EXPECT_THROW(macro_fusion::BilateralFilter(
                   uint2{2, 2}, float3{1.0f, 0.1f, 1.0f}, depth, nullptr),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::BilateralFilter(
                   uint2{2, 2}, float3{1.0f, 0.1f, 1.0f},
                   std::vector<float>(3, 1.0f), &filtered),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::BilateralFilter(
                   uint2{2, 2}, float3{0.0f, 0.1f, 1.0f}, depth, &filtered),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::BilateralFilter(
                   uint2{2, 2}, float3{1.0f, -0.1f, 1.0f}, depth, &filtered),
               std::invalid_argument);
}

TEST(HalfSampleTest, AveragesValidSamplesWithinEachDiscontinuityWindow) {
  const std::vector<float> depth = {
      1.0f, 1.0f, 2.0f, 2.0f,
      1.0f, 0.0f, 2.0f, 10.0f,
      3.0f, 3.0f, 4.0f, 4.0f,
      0.0f, 3.2f, 4.0f, 4.2f};
  std::vector<float> sampled;

  macro_fusion::HalfSample(uint2{4, 4}, 0.25f, depth, &sampled);

  ExpectFloatVectorNear(sampled,
                        std::vector<float>{1.0f, 2.0f, 3.0666666f, 4.05f});
}

TEST(HalfSampleTest, ResizesToHalvedExtent) {
  const std::vector<float> depth(24, 1.0f);
  std::vector<float> sampled(1, 9.0f);

  macro_fusion::HalfSample(uint2{6, 4}, 0.25f, depth, &sampled);

  ASSERT_EQ(sampled.size(), 6);
  for (float depth_sample : sampled) {
    EXPECT_FLOAT_EQ(depth_sample, 1.0f);
  }
}

TEST(HalfSampleTest, RejectsInvalidContractInputs) {
  const std::vector<float> depth(4, 1.0f);
  std::vector<float> sampled;

  EXPECT_THROW(macro_fusion::HalfSample(uint2{2, 2}, 0.1f, depth, nullptr),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::HalfSample(uint2{2, 2}, 0.1f,
                                        std::vector<float>(3, 1.0f),
                                        &sampled),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::HalfSample(uint2{1, 2}, 0.1f,
                                        std::vector<float>(2, 1.0f),
                                        &sampled),
               std::invalid_argument);
}

TEST(BuildTrackingPyramidTest, BuildsFilteredLevelsAndScalesSurfaceMaps) {
  const uint2 image_extent{16, 16};
  const float4 intrinsics{8.0f, 8.0f, 7.5f, 7.5f};
  const float3 bilateral_params{1.0f, 0.1f, 1.0f};
  const std::vector<float> raw_depth(256, 2.0f);

  std::vector<macro_fusion::HostTrackingLevel> pyramid =
      macro_fusion::BuildTrackingPyramid(image_extent, intrinsics,
                                         bilateral_params, raw_depth, 3);

  ASSERT_EQ(pyramid.size(), 3);
  const std::vector<uint2> expected_extents = {uint2{16, 16}, uint2{8, 8},
                                              uint2{4, 4}};
  for (std::size_t level = 0; level < pyramid.size(); ++level) {
    const std::size_t expected_pixels =
        static_cast<std::size_t>(expected_extents[level].x) *
        expected_extents[level].y;
    EXPECT_EQ(pyramid[level].depth.size(), expected_pixels);
    EXPECT_EQ(pyramid[level].vertices.size(), expected_pixels);
    EXPECT_EQ(pyramid[level].normals.size(), expected_pixels);

    const std::size_t interior =
        static_cast<std::size_t>(expected_extents[level].y / 2) *
            expected_extents[level].x +
        expected_extents[level].x / 2;
    EXPECT_FLOAT_EQ(pyramid[level].normals[interior].w, 1.0f);
    ExpectFloat4Near(pyramid[level].normals[interior],
                     float4{0.0f, 0.0f, 1.0f, 1.0f});
  }

  macro_fusion::HostTrackingLevel direct_level;
  macro_fusion::BilateralFilter(image_extent, bilateral_params, raw_depth,
                                &direct_level.depth);
  macro_fusion::BackProject(image_extent, intrinsics, &direct_level);
  macro_fusion::ComputeNormals(image_extent, &direct_level);

  ExpectFloatVectorNear(pyramid[0].depth, direct_level.depth);
  ASSERT_EQ(pyramid[0].vertices.size(), direct_level.vertices.size());
  ASSERT_EQ(pyramid[0].normals.size(), direct_level.normals.size());
  for (std::size_t index = 0; index < pyramid[0].vertices.size(); ++index) {
    ExpectFloat4Near(pyramid[0].vertices[index], direct_level.vertices[index]);
    EXPECT_FLOAT_EQ(pyramid[0].vertices[index].w,
                    direct_level.vertices[index].w);
    ExpectFloat4Near(pyramid[0].normals[index], direct_level.normals[index]);
    EXPECT_FLOAT_EQ(pyramid[0].normals[index].w,
                    direct_level.normals[index].w);
  }
}

TEST(BuildTrackingPyramidTest, RejectsInvalidContractInputs) {
  const std::vector<float> raw_depth(16, 1.0f);

  EXPECT_THROW(macro_fusion::BuildTrackingPyramid(
                   uint2{4, 4}, float4{2.0f, 2.0f, 1.5f, 1.5f},
                   float3{1.0f, 0.1f, 1.0f}, raw_depth, 0),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::BuildTrackingPyramid(
                   uint2{4, 4}, float4{2.0f, 2.0f, 1.5f, 1.5f},
                   float3{1.0f, 0.1f, 1.0f},
                   std::vector<float>(15, 1.0f), 1),
               std::invalid_argument);
}

}  // namespace
