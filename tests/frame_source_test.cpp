/// @file
/// @brief Tests the TUM frame source with runtime RGB-D fixtures.

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <vector>

#include "macro_fusion/core/config.h"
#include "macro_fusion/core/math_types.h"
#include "macro_fusion/modules/frame_source.h"

namespace {

/// @brief Owns a unique temporary directory for one test fixture.
class TempDirectory {
 public:
  /// @brief Creates an empty directory below the system temporary directory.
  TempDirectory() {
    static uint64_t sequence = 0;
    const auto timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("macrofusion_frame_source_" + std::to_string(timestamp) + "_" +
             std::to_string(sequence++));
    std::filesystem::create_directories(path_);
  }

  /// @brief Removes the fixture directory and all generated files.
  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  /// @brief Returns the directory used by the fixture.
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

/// @brief Writes a small text fixture exactly as supplied.
void WriteTextFile(const std::filesystem::path& path,
                   std::string_view contents) {
  std::ofstream output(path, std::ios::binary);
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
}

/// @brief Appends a 32-bit integer in PNG network byte order.
void AppendBigEndian32(uint32_t value, std::vector<uint8_t>* output) {
  output->push_back(static_cast<uint8_t>(value >> 24));
  output->push_back(static_cast<uint8_t>(value >> 16));
  output->push_back(static_cast<uint8_t>(value >> 8));
  output->push_back(static_cast<uint8_t>(value));
}

/// @brief Appends a 16-bit integer in DEFLATE stored-block byte order.
void AppendLittleEndian16(uint16_t value, std::vector<uint8_t>* output) {
  output->push_back(static_cast<uint8_t>(value));
  output->push_back(static_cast<uint8_t>(value >> 8));
}

/// @brief Computes the PNG CRC over one chunk type and payload.
uint32_t Crc32(std::string_view type, const std::vector<uint8_t>& data) {
  uint32_t crc = 0xFFFFFFFFu;
  const auto update = [&crc](uint8_t byte) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) {
      const uint32_t mask = 0u - (crc & 1u);
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  };
  for (char character : type) {
    update(static_cast<uint8_t>(character));
  }
  for (uint8_t byte : data) {
    update(byte);
  }
  return crc ^ 0xFFFFFFFFu;
}

/// @brief Computes the zlib checksum for uncompressed image bytes.
uint32_t Adler32(const std::vector<uint8_t>& data) {
  constexpr uint32_t kModulus = 65521u;
  uint32_t a = 1;
  uint32_t b = 0;
  for (uint8_t byte : data) {
    a = (a + byte) % kModulus;
    b = (b + a) % kModulus;
  }
  return (b << 16) | a;
}

/// @brief Appends one complete PNG chunk including length and CRC fields.
void AppendChunk(std::string_view type, const std::vector<uint8_t>& data,
                 std::vector<uint8_t>* png) {
  AppendBigEndian32(static_cast<uint32_t>(data.size()), png);
  png->insert(png->end(), type.begin(), type.end());
  png->insert(png->end(), data.begin(), data.end());
  AppendBigEndian32(Crc32(type, data), png);
}

/// @brief Writes a PNG that preserves tightly packed row-major 16-bit samples.
/// @param[in] pixels Exactly `width * height` samples in row-major order.
void WriteGray16Png(const std::filesystem::path& path, uint32_t width,
                    uint32_t height, const std::vector<uint16_t>& pixels) {
  const std::array<uint8_t, 8> signature{0x89, 0x50, 0x4E, 0x47,
                                         0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<uint8_t> png(signature.begin(), signature.end());

  std::vector<uint8_t> ihdr;
  AppendBigEndian32(width, &ihdr);
  AppendBigEndian32(height, &ihdr);
  ihdr.insert(ihdr.end(), {16, 0, 0, 0, 0});
  AppendChunk("IHDR", ihdr, &png);

  // Filter type zero leaves each big-endian 16-bit sample unchanged.
  std::vector<uint8_t> raw;
  raw.reserve(static_cast<std::size_t>(height) * (1 + 2 * width));
  for (uint32_t y = 0; y < height; ++y) {
    raw.push_back(0);
    for (uint32_t x = 0; x < width; ++x) {
      const uint16_t value = pixels[static_cast<std::size_t>(y) * width + x];
      raw.push_back(static_cast<uint8_t>(value >> 8));
      raw.push_back(static_cast<uint8_t>(value));
    }
  }

  // The tiny fixtures fit in one final uncompressed DEFLATE block.
  std::vector<uint8_t> idat{0x78, 0x01, 0x01};
  const uint16_t length = static_cast<uint16_t>(raw.size());
  AppendLittleEndian16(length, &idat);
  AppendLittleEndian16(static_cast<uint16_t>(~length), &idat);
  idat.insert(idat.end(), raw.begin(), raw.end());
  AppendBigEndian32(Adler32(raw), &idat);
  AppendChunk("IDAT", idat, &png);
  AppendChunk("IEND", {}, &png);

  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(png.data()),
               static_cast<std::streamsize>(png.size()));
}

/// @brief Writes a PNG that preserves tightly packed row-major RGB pixels.
/// @param[in] pixels Exactly `width * height` pixels in row-major order.
void WriteRgb8Png(const std::filesystem::path& path, uint32_t width,
                  uint32_t height, const std::vector<uchar3>& pixels) {
  const std::array<uint8_t, 8> signature{0x89, 0x50, 0x4E, 0x47,
                                         0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<uint8_t> png(signature.begin(), signature.end());

  std::vector<uint8_t> ihdr;
  AppendBigEndian32(width, &ihdr);
  AppendBigEndian32(height, &ihdr);
  ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
  AppendChunk("IHDR", ihdr, &png);

  std::vector<uint8_t> raw;
  raw.reserve(static_cast<std::size_t>(height) * (1 + 3 * width));
  for (uint32_t y = 0; y < height; ++y) {
    raw.push_back(0);
    for (uint32_t x = 0; x < width; ++x) {
      const uchar3 pixel = pixels[static_cast<std::size_t>(y) * width + x];
      raw.insert(raw.end(), {pixel.x, pixel.y, pixel.z});
    }
  }

  std::vector<uint8_t> idat{0x78, 0x01, 0x01};
  const uint16_t length = static_cast<uint16_t>(raw.size());
  AppendLittleEndian16(length, &idat);
  AppendLittleEndian16(static_cast<uint16_t>(~length), &idat);
  idat.insert(idat.end(), raw.begin(), raw.end());
  AppendBigEndian32(Adler32(raw), &idat);
  AppendChunk("IDAT", idat, &png);
  AppendChunk("IEND", {}, &png);

  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(png.data()),
               static_cast<std::streamsize>(png.size()));
}

/// @brief Creates a small frame-source configuration for one fixture.
macro_fusion::FrameSourceConfig MakeConfig(const std::filesystem::path& path,
                                           uint32_t width = 2,
                                           uint32_t height = 2) {
  macro_fusion::FrameSourceConfig config = macro_fusion::TumFreiburgConfig();
  config.path = path;
  config.image_extent = uint2{width, height};
  return config;
}

/// @brief Writes one depth image below the fixture's `depth` directory.
void WriteDepthImage(const std::filesystem::path& root,
                     std::string_view filename, uint32_t width,
                     uint32_t height,
                     const std::vector<uint16_t>& pixels) {
  std::filesystem::create_directories(root / "depth");
  WriteGray16Png(root / "depth" / std::string(filename), width, height,
                 pixels);
}

/// @brief Writes one RGB image below the fixture's `rgb` directory.
void WriteRgbImage(const std::filesystem::path& root, std::string_view filename,
                   uint32_t width, uint32_t height,
                   const std::vector<uchar3>& pixels) {
  std::filesystem::create_directories(root / "rgb");
  WriteRgb8Png(root / "rgb" / std::string(filename), width, height, pixels);
}

/// @brief Writes one ordinary RGB feed usable by depth-focused tests.
void WriteDefaultRgbFeed(const std::filesystem::path& root,
                         uint32_t width = 2, uint32_t height = 2) {
  std::vector<uchar3> pixels(static_cast<std::size_t>(width) * height,
                            uchar3{10, 20, 30});
  WriteRgbImage(root, "rgb.png", width, height, pixels);
  WriteTextFile(root / "rgb.txt", "0.0 rgb/rgb.png\n");
}

/// @brief Writes one identity ground-truth pose for ordinary fixtures.
void WriteIdentityPose(const std::filesystem::path& root) {
  WriteTextFile(root / "groundtruth.txt", "0.0 0 0 0 0 0 0 1\n");
}

/// @brief Compares an associated pose with one expected row-major transform.
void ExpectTransformNear(
    const macro_fusion::RigidTransform& actual,
    const std::array<float, 12>& expected) {
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual.data[i], expected[i], 1e-5f) << "element " << i;
  }
}

TEST(TumFrameSourceTest, IteratesAllFramesInOrderForwardOnly) {
  TempDirectory temp;
  const std::vector<uint16_t> pixels{1, 2, 3, 4};
  WriteDepthImage(temp.path(), "frame0.png", 2, 2, pixels);
  WriteDepthImage(temp.path(), "frame1.png", 2, 2, pixels);
  WriteDepthImage(temp.path(), "frame2.png", 2, 2, pixels);
  WriteTextFile(temp.path() / "depth.txt",
                "0.0 depth/frame0.png\n"
                "1.0 depth/frame1.png\n"
                "2.0 depth/frame2.png\n");
  WriteDefaultRgbFeed(temp.path());
  WriteIdentityPose(temp.path());

  macro_fusion::TumFrameSource source(MakeConfig(temp.path()));
  for (uint64_t expected_index = 0; expected_index < 3; ++expected_index) {
    ASSERT_TRUE(source.HasNext());
    const auto [metadata, depth_meters, rgb] = source.Next();
    EXPECT_EQ(metadata.frame_index, expected_index);
    EXPECT_EQ(depth_meters.size(), pixels.size());
    EXPECT_EQ(rgb.size(), pixels.size());
  }
  EXPECT_FALSE(source.HasNext());
  EXPECT_THROW(source.Next(), std::out_of_range);
}

TEST(TumFrameSourceTest, ConvertsDepthToMetersAndPreservesInvalidZero) {
  TempDirectory temp;
  const std::vector<uint16_t> pixels{0, 1, 5000, 65535};
  WriteDepthImage(temp.path(), "depth.png", 2, 2, pixels);
  WriteTextFile(temp.path() / "depth.txt", "1.25 depth/depth.png\n");
  WriteDefaultRgbFeed(temp.path());
  WriteIdentityPose(temp.path());

  macro_fusion::TumFrameSource source(MakeConfig(temp.path()));
  const auto frame = source.Next();
  const auto& depth_meters = std::get<1>(frame);

  ASSERT_EQ(depth_meters.size(), pixels.size());
  EXPECT_FLOAT_EQ(depth_meters[0], 0.0f);
  EXPECT_FLOAT_EQ(depth_meters[1], 0.0002f);
  EXPECT_FLOAT_EQ(depth_meters[2], 1.0f);
  EXPECT_FLOAT_EQ(depth_meters[3], 13.107f);
}

TEST(TumFrameSourceTest, UsesConfiguredDepthScale) {
  TempDirectory temp;
  const std::vector<uint16_t> pixels{0, 1, 2, 4};
  WriteDepthImage(temp.path(), "depth.png", 2, 2, pixels);
  WriteTextFile(temp.path() / "depth.txt", "1.25 depth/depth.png\n");
  WriteDefaultRgbFeed(temp.path());
  WriteIdentityPose(temp.path());

  macro_fusion::FrameSourceConfig config = MakeConfig(temp.path());
  config.depth_scale_to_meters = 0.25f;
  macro_fusion::TumFrameSource source(config);
  const auto frame = source.Next();
  const auto& depth_meters = std::get<1>(frame);

  EXPECT_EQ(depth_meters,
            (std::vector<float>{0.0f, 0.25f, 0.5f, 1.0f}));
}

TEST(TumFrameSourceTest, AssociatesNearestGroundTruthPose) {
  TempDirectory temp;
  const std::vector<uint16_t> pixels{1, 2, 3, 4};
  WriteDepthImage(temp.path(), "near_identity.png", 2, 2, pixels);
  WriteDepthImage(temp.path(), "near_rotation.png", 2, 2, pixels);
  WriteTextFile(temp.path() / "depth.txt",
                "0.1 depth/near_identity.png\n"
                "9.9 depth/near_rotation.png\n");
  WriteDefaultRgbFeed(temp.path());
  WriteTextFile(temp.path() / "groundtruth.txt",
                "0.0 1 2 3 0 0 0 1\n"
                "10.0 4 5 6 0 0 1 0\n");

  macro_fusion::TumFrameSource source(MakeConfig(temp.path()));
  const macro_fusion::FrameMetadata first_metadata =
      std::get<0>(source.Next());
  ExpectTransformNear(first_metadata.dataset_world_from_camera,
                      {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                       0.0f, 2.0f, 0.0f, 0.0f, 1.0f, 3.0f});

  const macro_fusion::FrameMetadata second_metadata =
      std::get<0>(source.Next());
  ExpectTransformNear(second_metadata.dataset_world_from_camera,
                      {-1.0f, 0.0f, 0.0f, 4.0f, 0.0f, -1.0f,
                       0.0f, 5.0f, 0.0f, 0.0f, 1.0f, 6.0f});
}

TEST(TumFrameSourceTest, SkipsCommentAndBlankLines) {
  TempDirectory temp;
  const std::vector<uint16_t> pixels{10, 20, 30, 40};
  WriteDepthImage(temp.path(), "depth.png", 2, 2, pixels);
  WriteTextFile(temp.path() / "depth.txt",
                "# depth association\n\n"
                "0.5 depth/depth.png\n\n"
                "# end\n");
  WriteDefaultRgbFeed(temp.path());
  WriteTextFile(temp.path() / "groundtruth.txt",
                "# trajectory\n\n"
                "0.4 7 8 9 0 0 0 1\n");

  macro_fusion::TumFrameSource source(MakeConfig(temp.path()));
  const auto [metadata, depth_meters, rgb] = source.Next();
  ASSERT_EQ(depth_meters.size(), pixels.size());
  EXPECT_EQ(rgb.size(), pixels.size());
  for (std::size_t index = 0; index < pixels.size(); ++index) {
    EXPECT_FLOAT_EQ(depth_meters[index],
                    static_cast<float>(pixels[index]) / 5000.0f);
  }
  ExpectTransformNear(metadata.dataset_world_from_camera,
                      {1.0f, 0.0f, 0.0f, 7.0f, 0.0f, 1.0f,
                       0.0f, 8.0f, 0.0f, 0.0f, 1.0f, 9.0f});
  EXPECT_FALSE(source.HasNext());
}

TEST(TumFrameSourceTest, MissingGroundTruthFailsConstruction) {
  TempDirectory temp;
  const std::vector<uint16_t> pixels{1, 2, 3, 4};
  WriteDepthImage(temp.path(), "depth.png", 2, 2, pixels);
  WriteTextFile(temp.path() / "depth.txt", "0.0 depth/depth.png\n");

  EXPECT_THROW(macro_fusion::TumFrameSource(MakeConfig(temp.path())),
               std::runtime_error);
}

TEST(TumFrameSourceTest, RejectsUnexpectedImageExtent) {
  TempDirectory temp;
  WriteDepthImage(temp.path(), "depth.png", 3, 2,
                  std::vector<uint16_t>{1, 2, 3, 4, 5, 6});
  WriteTextFile(temp.path() / "depth.txt", "0.0 depth/depth.png\n");
  WriteDefaultRgbFeed(temp.path());
  WriteIdentityPose(temp.path());

  macro_fusion::TumFrameSource source(MakeConfig(temp.path(), 2, 2));

  EXPECT_THROW(source.Next(), std::runtime_error);
}

TEST(TumFrameSourceTest, ReturnsNearestRgbFeedInRgbOrder) {
  TempDirectory temp;
  WriteDepthImage(temp.path(), "depth.png", 2, 2, {1, 2, 3, 4});
  WriteTextFile(temp.path() / "depth.txt", "1.0 depth/depth.png\n");
  const std::vector<uchar3> earlier{{1, 2, 3}, {4, 5, 6},
                                    {7, 8, 9}, {10, 11, 12}};
  const std::vector<uchar3> later(4, uchar3{20, 21, 22});
  WriteRgbImage(temp.path(), "earlier.png", 2, 2, earlier);
  WriteRgbImage(temp.path(), "later.png", 2, 2, later);
  WriteTextFile(temp.path() / "rgb.txt",
                "0.75 rgb/earlier.png\n"
                "1.25 rgb/later.png\n");
  WriteIdentityPose(temp.path());

  macro_fusion::TumFrameSource source(MakeConfig(temp.path()));
  const auto [metadata, depth_meters, rgb] = source.Next();

  EXPECT_DOUBLE_EQ(metadata.depth_timestamp_seconds, 1.0);
  EXPECT_DOUBLE_EQ(metadata.rgb_timestamp_seconds, 0.75);
  ASSERT_EQ(rgb.size(), earlier.size());
  for (std::size_t i = 0; i < rgb.size(); ++i) {
    EXPECT_EQ(rgb[i].x, earlier[i].x);
    EXPECT_EQ(rgb[i].y, earlier[i].y);
    EXPECT_EQ(rgb[i].z, earlier[i].z);
  }
}

TEST(TumFrameSourceTest, MissingRgbIndexFailsConstruction) {
  TempDirectory temp;
  WriteDepthImage(temp.path(), "depth.png", 2, 2, {1, 2, 3, 4});
  WriteTextFile(temp.path() / "depth.txt", "0.0 depth/depth.png\n");
  WriteIdentityPose(temp.path());

  EXPECT_THROW(macro_fusion::TumFrameSource(MakeConfig(temp.path())),
               std::runtime_error);
}

TEST(TumFrameSourceTest, MalformedRgbIndexFailsConstruction) {
  TempDirectory temp;
  WriteDepthImage(temp.path(), "depth.png", 2, 2, {1, 2, 3, 4});
  WriteTextFile(temp.path() / "depth.txt", "0.0 depth/depth.png\n");
  WriteTextFile(temp.path() / "rgb.txt", "not-a-timestamp rgb/image.png\n");
  WriteIdentityPose(temp.path());

  EXPECT_THROW(macro_fusion::TumFrameSource(MakeConfig(temp.path())),
               std::runtime_error);
}

TEST(TumFrameSourceTest, EmptyRgbIndexFailsConstruction) {
  TempDirectory temp;
  WriteDepthImage(temp.path(), "depth.png", 2, 2, {1, 2, 3, 4});
  WriteTextFile(temp.path() / "depth.txt", "0.0 depth/depth.png\n");
  WriteTextFile(temp.path() / "rgb.txt", "# no RGB frames\n\n");
  WriteIdentityPose(temp.path());

  EXPECT_THROW(macro_fusion::TumFrameSource(MakeConfig(temp.path())),
               std::runtime_error);
}

TEST(TumFrameSourceTest, RejectsUndecodableRgbImage) {
  TempDirectory temp;
  WriteDepthImage(temp.path(), "depth.png", 2, 2, {1, 2, 3, 4});
  WriteTextFile(temp.path() / "depth.txt", "0.0 depth/depth.png\n");
  std::filesystem::create_directories(temp.path() / "rgb");
  WriteTextFile(temp.path() / "rgb" / "bad.png", "not a PNG");
  WriteTextFile(temp.path() / "rgb.txt", "0.0 rgb/bad.png\n");
  WriteIdentityPose(temp.path());

  macro_fusion::TumFrameSource source(MakeConfig(temp.path()));
  EXPECT_THROW(source.Next(), std::runtime_error);
  EXPECT_TRUE(source.HasNext());
}

TEST(TumFrameSourceTest, RejectsUnexpectedRgbExtent) {
  TempDirectory temp;
  WriteDepthImage(temp.path(), "depth.png", 2, 2, {1, 2, 3, 4});
  WriteTextFile(temp.path() / "depth.txt", "0.0 depth/depth.png\n");
  WriteRgbImage(temp.path(), "rgb.png", 3, 2,
                std::vector<uchar3>(6, uchar3{1, 2, 3}));
  WriteTextFile(temp.path() / "rgb.txt", "0.0 rgb/rgb.png\n");
  WriteIdentityPose(temp.path());

  macro_fusion::TumFrameSource source(MakeConfig(temp.path()));
  EXPECT_THROW(source.Next(), std::runtime_error);
}

TEST(TumFrameSourceTest, MissingDirectoryFailsConstruction) {
  TempDirectory temp;
  const std::filesystem::path missing = temp.path() / "missing";

  EXPECT_THROW(macro_fusion::TumFrameSource(MakeConfig(missing)),
               std::runtime_error);
}

}  // namespace
