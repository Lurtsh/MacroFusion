/// @file
/// @brief Tests host display conversion operations.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "macro_fusion/modules/render.h"

namespace {

void ExpectPixel(uchar4 actual, uint8_t red, uint8_t green, uint8_t blue,
                 uint8_t alpha = 255) {
  EXPECT_EQ(actual.x, red);
  EXPECT_EQ(actual.y, green);
  EXPECT_EQ(actual.z, blue);
  EXPECT_EQ(actual.w, alpha);
}

TEST(RenderRgbToRgbaTest, PreservesChannelsAndSetsOpaqueAlpha) {
  const std::vector<uchar3> rgb = {uchar3{1, 2, 3}, uchar3{4, 5, 6}};
  std::vector<uchar4> rgba;

  macro_fusion::RenderRgbToRgba(uint2{2, 1}, rgb, &rgba);

  ASSERT_EQ(rgba.size(), 2);
  ExpectPixel(rgba[0], 1, 2, 3);
  ExpectPixel(rgba[1], 4, 5, 6);
}

TEST(RenderDepthToRgbaTest, MapsMetricRangeAndInvalidSamples) {
  const std::vector<float> depth = {
      0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
      std::numeric_limits<float>::quiet_NaN()};
  std::vector<uchar4> rgba;

  macro_fusion::RenderDepthToRgba(uint2{6, 1}, 1.0f, 3.0f, depth, &rgba);

  ASSERT_EQ(rgba.size(), 6);
  ExpectPixel(rgba[0], 0, 0, 0);
  ExpectPixel(rgba[1], 0, 0, 0);
  ExpectPixel(rgba[2], 128, 128, 128);
  ExpectPixel(rgba[3], 255, 255, 255);
  ExpectPixel(rgba[4], 255, 255, 255);
  ExpectPixel(rgba[5], 0, 0, 0);
}

TEST(RenderVerticesToRgbaTest, MapsCameraFrameXyzToRgb) {
  std::vector<float4> vertices(9, float4{0.0f, 0.0f, 0.0f, 0.0f});
  vertices[4] = float4{0.0f, 0.0f, 2.0f, 1.0f};
  vertices[8] = float4{3.0f, 3.0f, 3.0f, 1.0f};
  std::vector<uchar4> rgba;

  macro_fusion::RenderVerticesToRgba(
      uint2{3, 3}, float4{1.0f, 1.0f, 1.0f, 1.0f}, 1.0f, 3.0f,
      vertices, &rgba);

  ASSERT_EQ(rgba.size(), 9);
  ExpectPixel(rgba[0], 0, 0, 0);
  ExpectPixel(rgba[4], 128, 128, 128);
  ExpectPixel(rgba[8], 255, 255, 255);
}

TEST(RenderNormalsToRgbaTest, MapsNormalsAndUsesValidity) {
  const std::vector<float4> normals = {
      float4{-1.0f, 0.0f, 1.0f, 1.0f},
      float4{1.0f, 1.0f, 1.0f, 0.0f},
      float4{std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 1.0f}};
  std::vector<uchar4> rgba;

  macro_fusion::RenderNormalsToRgba(uint2{3, 1}, normals, &rgba);

  ASSERT_EQ(rgba.size(), 3);
  ExpectPixel(rgba[0], 0, 128, 255);
  ExpectPixel(rgba[1], 0, 0, 0);
  ExpectPixel(rgba[2], 0, 0, 0);
}

TEST(RenderConversionTest, RejectsInvalidContracts) {
  const std::vector<uchar3> rgb(4, uchar3{0, 0, 0});
  const std::vector<float> depth(4, 1.0f);
  const std::vector<float4> vertices(4, float4{0.0f, 0.0f, 1.0f, 1.0f});
  std::vector<uchar4> rgba;

  EXPECT_THROW(macro_fusion::RenderRgbToRgba(uint2{2, 2}, rgb, nullptr),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::RenderRgbToRgba(
                   uint2{0, 2}, rgb, &rgba),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::RenderRgbToRgba(
                   uint2{2, 2}, std::vector<uchar3>(3), &rgba),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::RenderDepthToRgba(uint2{2, 2}, 2.0f, 1.0f,
                                               depth, &rgba),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::RenderVerticesToRgba(
                   uint2{2, 2}, float4{0.0f, 1.0f, 0.0f, 0.0f}, 1.0f,
                   3.0f, vertices, &rgba),
               std::invalid_argument);
  EXPECT_THROW(macro_fusion::RenderNormalsToRgba(
                   uint2{2, 2}, std::vector<float4>(3), &rgba),
               std::invalid_argument);
  EXPECT_NO_THROW(macro_fusion::RenderNormalsToRgba(
      uint2{2, 2}, vertices, &rgba));
}

}  // namespace
