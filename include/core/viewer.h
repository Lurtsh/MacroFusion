#pragma once

/// @file
/// @brief Declares the host tracking-pyramid viewer.

#include <array>
#include <vector>

#include "modules/pipeline.h"
#include "core/math_types.cuh"
#include "modules/frame_source.h"

struct GLFWwindow;
struct cudaGraphicsResource;

namespace macro_fusion {

/// @brief Owns the OpenGL window, Dear ImGui context, and display transport.
///
/// The viewer brackets each iteration of the application loop: `BeginFrame`
/// polls input and draws the transport controls, and `EndFrame` draws status
/// and presents the display textures. The viewer owns the GL window, Dear ImGui
/// context, transport state, display textures, and CUDA-registered PBOs for the
/// interop transport. It does not own algorithm buffers or display content and
/// invokes no compute kernels.
class Viewer {
 public:
  /// @brief Initializes the renderer and its four display textures.
  ///
  /// Setup installs the GLFW error callback, initializes GLFW, creates and
  /// activates the OpenGL 4.6 core window, enables swap synchronization, loads
  /// GLAD, verifies the NVIDIA vendor, creates the pane framebuffer, initializes
  /// Dear ImGui, creates the textures, and optionally creates and registers one
  /// CUDA-mapped OpenGL PBO per pane, in that order.
  /// @param[in] image_extent Width and height of every display pane.
  /// @param[in] cuda_display Whether to allocate CUDA/OpenGL interop PBOs.
  /// @throws std::runtime_error If GLFW, the window, GLAD, or NVIDIA vendor
  ///     verification fails, or if CUDA interop registration fails.
  explicit Viewer(uint2 image_extent, bool cuda_display = true);

  /// @brief Releases textures, the pane framebuffer, Dear ImGui, the window,
  ///     and GLFW in reverse ownership order.
  ~Viewer();

  Viewer(const Viewer&) = delete;
  Viewer& operator=(const Viewer&) = delete;

  /// @brief Opens the per-frame UI: polls input, draws the transport controls,
  ///     and resolves whether the pipeline should advance this iteration.
  ///
  /// Begins the fullscreen borderless Dear ImGui frame and draws the Next / Play
  /// controls, updating the internal continuous-play and first-frame state. The
  /// advance verdict is cached for `AdvanceRequested`.
  /// @param[in] can_load_next Whether another frame is available to load.
  /// @return `false` once the user requests that the window close, ending the
  ///     loop without a matching `EndFrame`; `true` otherwise.
  bool BeginFrame(bool can_load_next);

  /// @brief Reports whether the pipeline should advance this iteration.
  /// @return The advance verdict cached by the most recent `BeginFrame`.
  bool AdvanceRequested() const;

  /// @brief Uploads host-owned RGBA panes into the display textures.
  /// @param[in] panes Row-major RGBA panes in shared pane-index order.
  void UploadPanes(const std::array<std::vector<uchar4>, kPaneCount>& panes);

  /// @brief Maps the CUDA-registered display PBOs for device writes.
  /// @return Device pointers to row-major RGBA pane buffers in shared
  ///     pane-index order. Each pointer remains valid until
  ///     `UnmapAndRefreshTextures` is called.
  std::array<uchar4*, kPaneCount> MapDisplayBuffers();

  /// @brief Unmaps the display PBOs and refreshes the OpenGL display textures.
  void UnmapAndRefreshTextures();

  /// @brief Draws status, presents the render target, and swaps the back buffer.
  ///
  /// Appends the processed-frame index and timestamps to the transport-control
  /// line, blits the current display textures below the controls, renders the
  /// Dear ImGui chrome, and presents the OpenGL back buffer. Called after the
  /// pipeline advance so the status reflects the panes drawn this iteration.
  /// @param[in] metadata Metadata for the frame currently shown in the panes.
  void EndFrame(const FrameMetadata& metadata);

 private:
  uint2 image_extent_;
  GLFWwindow* window_;
  std::array<unsigned int, kPaneCount> textures_;
  std::array<unsigned int, kPaneCount> pbos_;
  std::array<cudaGraphicsResource*, kPaneCount> cuda_resources_;
  unsigned int pane_fbo_;
  bool cuda_display_;
  bool playing_;
  bool first_frame_;
  bool advance_;
};

}  // namespace macro_fusion
