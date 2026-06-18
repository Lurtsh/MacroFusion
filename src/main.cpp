/// @file
/// @brief Runs the host OpenGL tracking-pyramid viewer.

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glog/logging.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "macro_fusion/core/config.h"
#include "macro_fusion/core/tracking.h"
#include "macro_fusion/modules/frame_source.h"
#include "macro_fusion/modules/preprocess.h"
#include "macro_fusion/modules/render.h"

namespace {

constexpr std::size_t kPaneCount = 4;
constexpr std::size_t kRgbPane = 0;
constexpr std::size_t kDepthPane = 1;
constexpr std::size_t kVertexPane = 2;
constexpr std::size_t kNormalPane = 3;

constexpr float kBilateralSpatialSigmaPixels = 1.0f;
constexpr float kBilateralRangeSigmaMeters = 0.1f;
constexpr float kBilateralRadiusPixels = 1.0f;
constexpr float kDisplayDepthMinMeters = 0.3f;
constexpr float kDisplayDepthMaxMeters = 5.0f;
constexpr const char* kGlslVersion = "#version 460";

void GlfwErrorCallback(int error, const char* description) {
  LOG(ERROR) << "GLFW error " << error << ": " << description;
}

ImTextureID TextureId(GLuint texture) {
  return (ImTextureID)(static_cast<intptr_t>(texture));
}

ImVec2 FitImageSize(uint2 image_extent, ImVec2 available_size) {
  const float width = std::max(1.0f, available_size.x);
  const float height = std::max(1.0f, available_size.y);
  const float aspect =
      static_cast<float>(image_extent.x) / static_cast<float>(image_extent.y);

  if (width / height > aspect) {
    return ImVec2(height * aspect, height);
  }
  return ImVec2(width, width / aspect);
}

GLuint CreateDisplayTexture(uint2 image_extent) {
  GLuint texture = 0;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
               static_cast<GLsizei>(image_extent.x),
               static_cast<GLsizei>(image_extent.y), 0, GL_RGBA,
               GL_UNSIGNED_BYTE, nullptr);
  return texture;
}

void UploadDisplayTexture(GLuint texture, uint2 image_extent,
                          const std::vector<uchar4>& rgba) {
  glBindTexture(GL_TEXTURE_2D, texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                  static_cast<GLsizei>(image_extent.x),
                  static_cast<GLsizei>(image_extent.y), GL_RGBA,
                  GL_UNSIGNED_BYTE, rgba.data());
}

void DrawPane(const char* title, GLuint texture, uint2 image_extent,
              float pane_height) {
  ImGui::BeginChild(title, ImVec2(0.0f, pane_height), true);
  ImGui::TextUnformatted(title);
  const ImVec2 image_size =
      FitImageSize(image_extent, ImGui::GetContentRegionAvail());
  ImGui::Image(TextureId(texture), image_size);
  ImGui::EndChild();
}

bool LoadAndUploadFrame(
    macro_fusion::TumFrameSource* source,
    const macro_fusion::FrameSourceConfig& config,
    const std::array<GLuint, kPaneCount>& textures,
    std::array<std::vector<uchar4>, kPaneCount>* rgba_images,
    uint64_t* frame_index, double* depth_timestamp_seconds,
    double* rgb_timestamp_seconds) {
  if (source == nullptr || rgba_images == nullptr || frame_index == nullptr ||
      depth_timestamp_seconds == nullptr || rgb_timestamp_seconds == nullptr) {
    throw std::invalid_argument("viewer frame state must not be null");
  }
  if (!source->HasNext()) {
    return false;
  }

  auto [metadata, raw_depth, rgb] = source->Next();
  const std::vector<macro_fusion::HostTrackingLevel> pyramid =
      macro_fusion::BuildTrackingPyramid(
          config.image_extent, config.intrinsics,
          float3{kBilateralSpatialSigmaPixels, kBilateralRangeSigmaMeters,
                 kBilateralRadiusPixels},
          raw_depth, 1);
  const macro_fusion::HostTrackingLevel& level = pyramid.front();

  macro_fusion::RenderRgbToRgba(config.image_extent, rgb,
                                &(*rgba_images)[kRgbPane]);
  macro_fusion::RenderDepthToRgba(config.image_extent, kDisplayDepthMinMeters,
                                  kDisplayDepthMaxMeters, level.depth,
                                  &(*rgba_images)[kDepthPane]);
  macro_fusion::RenderVerticesToRgba(
      config.image_extent, config.intrinsics, kDisplayDepthMinMeters,
      kDisplayDepthMaxMeters, level.vertices, &(*rgba_images)[kVertexPane]);
  macro_fusion::RenderNormalsToRgba(config.image_extent, level.normals,
                                    &(*rgba_images)[kNormalPane]);

  for (std::size_t pane = 0; pane < kPaneCount; ++pane) {
    UploadDisplayTexture(textures[pane], config.image_extent,
                         (*rgba_images)[pane]);
  }

  *frame_index = metadata.frame_index;
  *depth_timestamp_seconds = metadata.depth_timestamp_seconds;
  *rgb_timestamp_seconds = metadata.rgb_timestamp_seconds;
  return true;
}

int RunViewer() {
  const macro_fusion::FrameSourceConfig config =
      macro_fusion::TumFreiburgConfig();
  macro_fusion::TumFrameSource source(config);

  glfwSetErrorCallback(GlfwErrorCallback);
  if (glfwInit() != GLFW_TRUE) {
    throw std::runtime_error("failed to initialize GLFW");
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window =
      glfwCreateWindow(1280, 900, "MacroFusion Tracking Pyramid", nullptr,
                       nullptr);
  if (window == nullptr) {
    glfwTerminate();
    throw std::runtime_error("failed to create GLFW window");
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  if (gladLoadGL(glfwGetProcAddress) == 0) {
    glfwDestroyWindow(window);
    glfwTerminate();
    throw std::runtime_error("failed to initialize GLAD");
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(kGlslVersion);

  std::array<GLuint, kPaneCount> textures{};
  for (GLuint& texture : textures) {
    texture = CreateDisplayTexture(config.image_extent);
  }

  std::array<std::vector<uchar4>, kPaneCount> rgba_images;
  uint64_t frame_index = 0;
  double depth_timestamp_seconds = 0.0;
  double rgb_timestamp_seconds = 0.0;
  bool has_frame =
      LoadAndUploadFrame(&source, config, textures, &rgba_images, &frame_index,
                         &depth_timestamp_seconds, &rgb_timestamp_seconds);
  bool playing = false;

  while (glfwWindowShouldClose(window) == GLFW_FALSE) {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
    ImGui::Begin("Tracking Pyramid", nullptr,
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

    bool request_next = false;
    const bool can_load_next = source.HasNext();
    ImGui::BeginDisabled(!can_load_next);
    if (ImGui::Button("Next")) {
      request_next = true;
    }
    ImGui::SameLine();
    if (ImGui::Button(playing ? "Pause" : "Play")) {
      playing = !playing;
    }
    ImGui::EndDisabled();

    if (!can_load_next) {
      playing = false;
    }

    if ((request_next || playing) && source.HasNext()) {
      has_frame =
          LoadAndUploadFrame(&source, config, textures, &rgba_images,
                             &frame_index, &depth_timestamp_seconds,
                             &rgb_timestamp_seconds);
    }

    if (has_frame) {
      ImGui::SameLine();
      ImGui::Text("Frame %llu  Depth %.6f  RGB %.6f",
                  static_cast<unsigned long long>(frame_index),
                  depth_timestamp_seconds, rgb_timestamp_seconds);
    }

    const ImVec2 grid_available = ImGui::GetContentRegionAvail();
    const float pane_height = std::max(
        120.0f,
        (grid_available.y - ImGui::GetStyle().ItemSpacing.y) * 0.5f);

    if (ImGui::BeginTable("tracking_level_views", 2,
                          ImGuiTableFlags_SizingStretchSame |
                              ImGuiTableFlags_BordersInner)) {
      const std::array<const char*, kPaneCount> titles = {
          "Input RGB", "Filtered Depth", "Vertex XYZ", "Normal Map"};
      for (std::size_t row = 0; row < 2; ++row) {
        ImGui::TableNextRow();
        for (std::size_t column = 0; column < 2; ++column) {
          const std::size_t pane = row * 2 + column;
          ImGui::TableSetColumnIndex(static_cast<int>(column));
          DrawPane(titles[pane], textures[pane], config.image_extent,
                   pane_height);
        }
      }
      ImGui::EndTable();
    }

    ImGui::End();

    ImGui::Render();
    int framebuffer_width = 0;
    int framebuffer_height = 0;
    glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
    glViewport(0, 0, framebuffer_width, framebuffer_height);
    glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  glDeleteTextures(static_cast<GLsizei>(textures.size()), textures.data());
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}

}  // namespace

int main(int, char** argv) {
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  int result = 1;
  try {
    result = RunViewer();
  } catch (const std::exception& error) {
    LOG(ERROR) << error.what();
  }

  google::ShutdownGoogleLogging();
  return result;
}
