/// @file
/// @brief Implements the host tracking-pyramid renderer.

#include "core/viewer.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <glog/logging.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "modules/pipeline.h"

namespace macro_fusion {
namespace {

constexpr const char* kGlslVersion = "#version 460";
constexpr std::array<const char*, kPaneCount> kPaneTitles = {
    "Input RGB", "Filtered Depth", "Raycast Depth", "Raycast Normals"};

void CheckCuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

/// @brief Logs GLFW errors through glog.
/// @param[in] error GLFW error code.
/// @param[in] description Human-readable message owned by GLFW.
void GlfwErrorCallback(int error, const char* description) {
  LOG(ERROR) << "GLFW error " << error << ": " << description;
}

/// @brief Scales an image to fit inside a box while preserving aspect ratio.
/// @param[in] image_extent Source image size in pixels (width, height).
/// @param[in] available_size Target box; clamped to at least 1x1.
/// @return The largest size with the source aspect ratio that fits the box;
///         one axis matches the box and the other is letterboxed.
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

/// @brief Splits the framebuffer area below the control strip into a 2x2 grid.
///
/// Shared by the GL blit pass and the ImGui title overlay so pane images and
/// their labels stay in lockstep. Cells are ordered row-major: top-left,
/// top-right, bottom-left, bottom-right.
///
/// @param[in] fb_width Framebuffer width in pixels.
/// @param[in] fb_height Framebuffer height in pixels.
/// @param[in] top_bar_height Height reserved at the top for the control strip.
/// @return Four cell rects packed as `float4{x, y, width, height}` in
///         framebuffer pixels, with the origin at the top-left.
std::array<float4, kPaneCount> PaneCellRects(int fb_width, int fb_height,
                                             float top_bar_height) {
  const float width = static_cast<float>(fb_width);
  const float height =
      std::max(0.0f, static_cast<float>(fb_height) - top_bar_height);
  const float cell_width = width * 0.5f;
  const float cell_height = height * 0.5f;

  return {float4{0.0f, top_bar_height, cell_width, cell_height},
          float4{cell_width, top_bar_height, cell_width, cell_height},
          float4{0.0f, top_bar_height + cell_height, cell_width, cell_height},
          float4{cell_width, top_bar_height + cell_height, cell_width,
                 cell_height}};
}

/// @brief Centers an aspect-fit image inside a cell rect.
/// @param[in] image_extent Source image size in pixels (width, height).
/// @param[in] cell_rect Target cell as `float4{x, y, width, height}`.
/// @return The image rect `float4{x, y, width, height}` centered in the cell,
///         leaving equal letterbox margins on the unfilled axis.
float4 FitPaneRect(uint2 image_extent, float4 cell_rect) {
  const ImVec2 image_size =
      FitImageSize(image_extent, ImVec2(cell_rect.z, cell_rect.w));
  return float4{cell_rect.x + (cell_rect.z - image_size.x) * 0.5f,
                cell_rect.y + (cell_rect.w - image_size.y) * 0.5f,
                image_size.x, image_size.y};
}

/// @brief Reserves a title strip and margins inside a pane cell.
///
/// Keeps the pane image clear of the title drawn at the cell top and leaves a
/// gutter on the other sides so neighboring panes do not touch.
///
/// @param[in] cell_rect Cell as `float4{x, y, width, height}`.
/// @param[in] title_strip Height reserved at the cell top for the pane title.
/// @param[in] margin Gutter `{x, y}` inset on the remaining sides.
/// @return The image content rect `float4{x, y, width, height}`, at least 1x1.
float4 PaneContentRect(float4 cell_rect, float title_strip, float2 margin) {
  return float4{cell_rect.x + margin.x, cell_rect.y + title_strip,
                std::max(1.0f, cell_rect.z - 2.0f * margin.x),
                std::max(1.0f, cell_rect.w - title_strip - margin.y)};
}

/// @brief Allocates an RGBA8 texture sized for one debug pane.
///
/// Uses nearest filtering and clamp-to-edge wrapping; storage is left
/// uninitialized for later @ref UploadDisplayTexture calls.
///
/// @param[in] image_extent Texture size in pixels (width, height).
/// @return The new GL texture name; the caller owns it and must delete it.
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

/// @brief Uploads a host RGBA buffer into an existing pane texture.
/// @param[in] texture Destination texture name, sized to `image_extent`.
/// @param[in] image_extent Image size in pixels (width, height).
/// @param[in] rgba Source pixels, row-major, at least `width * height` entries.
void UploadDisplayTexture(GLuint texture, uint2 image_extent,
                          const std::vector<uchar4>& rgba) {
  glBindTexture(GL_TEXTURE_2D, texture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                  static_cast<GLsizei>(image_extent.x),
                  static_cast<GLsizei>(image_extent.y), GL_RGBA,
                  GL_UNSIGNED_BYTE, rgba.data());
}

}  // namespace

Viewer::Viewer(uint2 image_extent, bool cuda_display)
    : image_extent_(image_extent),
      window_(nullptr),
      textures_{},
      pbos_{},
      cuda_resources_{},
      pane_fbo_(0),
      cuda_display_(cuda_display),
      playing_(false),
      first_frame_(true),
      advance_(false) {
  glfwSetErrorCallback(GlfwErrorCallback);
  if (glfwInit() != GLFW_TRUE) {
    throw std::runtime_error("failed to initialize GLFW");
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window_ = glfwCreateWindow(1280, 900, "MacroFusion Reconstruction",
                             nullptr, nullptr);
  if (window_ == nullptr) {
    glfwTerminate();
    throw std::runtime_error("failed to create GLFW window");
  }

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);

  if (gladLoadGL(glfwGetProcAddress) == 0) {
    glfwDestroyWindow(window_);
    glfwTerminate();
    throw std::runtime_error("failed to initialize GLAD");
  }

  const GLubyte* vendor_value = glGetString(GL_VENDOR);
  const std::string vendor =
      vendor_value == nullptr
          ? std::string{}
          : reinterpret_cast<const char*>(vendor_value);
  if (vendor.find("NVIDIA") == std::string::npos) {
    glfwDestroyWindow(window_);
    glfwTerminate();
    throw std::runtime_error("OpenGL vendor is not NVIDIA: " + vendor);
  }
  LOG(INFO) << "OpenGL vendor: " << vendor;

  glGenFramebuffers(1, &pane_fbo_);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init(kGlslVersion);

  for (unsigned int& texture : textures_) {
    texture = CreateDisplayTexture(image_extent_);
  }

  if (cuda_display_) {
    for (std::size_t pane = 0; pane < kPaneCount; ++pane) {
      glGenBuffers(1, &pbos_[pane]);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos_[pane]);
      glBufferData(GL_PIXEL_UNPACK_BUFFER,
                   static_cast<GLsizeiptr>(image_extent_.x) *
                       image_extent_.y * sizeof(uchar4),
                   nullptr, GL_STREAM_DRAW);
      CheckCuda(cudaGraphicsGLRegisterBuffer(
                    &cuda_resources_[pane], pbos_[pane],
                    cudaGraphicsRegisterFlagsWriteDiscard),
                "cudaGraphicsGLRegisterBuffer");
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  }
}

Viewer::~Viewer() {
  glfwMakeContextCurrent(window_);
  if (cuda_display_) {
    for (cudaGraphicsResource* resource : cuda_resources_) {
      if (resource != nullptr) {
        CheckCuda(cudaGraphicsUnregisterResource(resource),
                  "cudaGraphicsUnregisterResource");
      }
    }
    glDeleteBuffers(static_cast<GLsizei>(pbos_.size()), pbos_.data());
  }
  glDeleteTextures(static_cast<GLsizei>(textures_.size()), textures_.data());
  glDeleteFramebuffers(1, &pane_fbo_);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window_);
  glfwTerminate();
}

bool Viewer::BeginFrame(bool can_load_next) {
  glfwPollEvents();
  if (glfwWindowShouldClose(window_) != GLFW_FALSE) {
    return false;
  }

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
  // Full width, auto-fit height: the control strip only covers the chrome, so
  // its background no longer dims the GL-blitted panes below it. A 0.0f extent
  // tells ImGui to auto-fit that axis to the window contents.
  ImGui::SetNextWindowSize(ImVec2(ImGui::GetIO().DisplaySize.x, 0.0f),
                           ImGuiCond_Always);
  ImGui::Begin("Reconstruction", nullptr,
               ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

  bool next_clicked = false;
  ImGui::BeginDisabled(!can_load_next);
  if (ImGui::Button("Next")) {
    next_clicked = true;
  }
  ImGui::SameLine();
  if (ImGui::Button(playing_ ? "Pause" : "Play")) {
    playing_ = !playing_;
  }
  ImGui::EndDisabled();
  if (!can_load_next) {
    playing_ = false;
  }

  advance_ = (first_frame_ || next_clicked || playing_) && can_load_next;
  first_frame_ = false;
  return true;
}

bool Viewer::AdvanceRequested() const { return advance_; }

void Viewer::UploadPanes(
    const std::array<std::vector<uchar4>, kPaneCount>& panes) {
  for (std::size_t pane = 0; pane < kPaneCount; ++pane) {
    UploadDisplayTexture(textures_[pane], image_extent_, panes[pane]);
  }
}

std::array<uchar4*, kPaneCount> Viewer::MapDisplayBuffers() {
  CheckCuda(cudaGraphicsMapResources(static_cast<int>(kPaneCount),
                                     cuda_resources_.data(), 0),
            "cudaGraphicsMapResources");

  std::array<uchar4*, kPaneCount> ptrs{};
  for (std::size_t pane = 0; pane < kPaneCount; ++pane) {
    std::size_t bytes = 0;
    CheckCuda(cudaGraphicsResourceGetMappedPointer(
                  reinterpret_cast<void**>(&ptrs[pane]), &bytes,
                  cuda_resources_[pane]),
              "cudaGraphicsResourceGetMappedPointer");
  }
  return ptrs;
}

void Viewer::UnmapAndRefreshTextures() {
  CheckCuda(cudaGraphicsUnmapResources(static_cast<int>(kPaneCount),
                                       cuda_resources_.data(), 0),
            "cudaGraphicsUnmapResources");

  for (std::size_t pane = 0; pane < kPaneCount; ++pane) {
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos_[pane]);
    glBindTexture(GL_TEXTURE_2D, textures_[pane]);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    static_cast<GLsizei>(image_extent_.x),
                    static_cast<GLsizei>(image_extent_.y), GL_RGBA,
                    GL_UNSIGNED_BYTE, nullptr);
  }
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void Viewer::EndFrame(const FrameMetadata& metadata) {
  ImGui::SameLine();
  ImGui::Text("Frame %llu  Depth %.6f  RGB %.6f",
              static_cast<unsigned long long>(metadata.frame_index),
              metadata.depth_timestamp_seconds,
              metadata.rgb_timestamp_seconds);
  const float top_bar_height =
      ImGui::GetCursorScreenPos().y + ImGui::GetStyle().WindowPadding.y;
  ImGui::End();

  const ImVec2 display_size = ImGui::GetIO().DisplaySize;
  const std::array<float4, kPaneCount> title_rects = PaneCellRects(
      static_cast<int>(display_size.x), static_cast<int>(display_size.y),
      top_bar_height);
  ImDrawList* foreground_draw_list = ImGui::GetForegroundDrawList();
  const ImVec2 title_padding(8.0f, 6.0f);
  for (std::size_t pane = 0; pane < kPaneCount; ++pane) {
    foreground_draw_list->AddText(
        ImVec2(title_rects[pane].x + title_padding.x,
               title_rects[pane].y + title_padding.y),
        IM_COL32(255, 255, 255, 255), kPaneTitles[pane]);
  }

  ImGui::Render();

  int framebuffer_width = 0;
  int framebuffer_height = 0;
  glfwGetFramebufferSize(window_, &framebuffer_width, &framebuffer_height);
  glViewport(0, 0, framebuffer_width, framebuffer_height);
  glClearColor(0.08f, 0.08f, 0.08f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  const float scale_x =
      static_cast<float>(framebuffer_width) / std::max(1.0f, display_size.x);
  const float scale_y =
      static_cast<float>(framebuffer_height) / std::max(1.0f, display_size.y);
  const float framebuffer_top_bar_height = top_bar_height * scale_y;
  // Reserve a label strip at the top of each cell and a gutter on the other
  // sides so titles sit above the image and neighboring panes stay separated.
  const float title_strip = ImGui::GetTextLineHeightWithSpacing() * scale_y;
  const float2 pane_margin{8.0f * scale_x, 8.0f * scale_y};
  const std::array<float4, kPaneCount> pane_rects = PaneCellRects(
      framebuffer_width, framebuffer_height, framebuffer_top_bar_height);
  for (std::size_t pane = 0; pane < kPaneCount; ++pane) {
    const float4 content_rect =
        PaneContentRect(pane_rects[pane], title_strip, pane_margin);
    const float4 blit_rect = FitPaneRect(image_extent_, content_rect);
    const GLint dst_x0 = static_cast<GLint>(blit_rect.x);
    const GLint dst_x1 = static_cast<GLint>(blit_rect.x + blit_rect.z);
    const GLint dst_y0 = static_cast<GLint>(
        static_cast<float>(framebuffer_height) - blit_rect.y - blit_rect.w);
    const GLint dst_y1 = static_cast<GLint>(
        static_cast<float>(framebuffer_height) - blit_rect.y);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, pane_fbo_);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, textures_[pane], 0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, static_cast<GLint>(image_extent_.x),
                      static_cast<GLint>(image_extent_.y), dst_x0, dst_y1,
                      dst_x1, dst_y0, GL_COLOR_BUFFER_BIT, GL_LINEAR);
  }
  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  glfwSwapBuffers(window_);
}

}  // namespace macro_fusion
