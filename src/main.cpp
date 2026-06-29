/// @file
/// @brief Configures the process environment and runs the application pipeline.

#include <glog/logging.h>

#include <array>
#include <cstdlib>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "modules/pipeline.h"
#include "core/viewer.h"
#include "core/config.h"
#include "modules/tracking.h"
#include "modules/frame_source.h"

namespace {

void ConfigureEnvironment() {
  if (setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1) != 0) {
    throw std::runtime_error("failed to select the NVIDIA GLX vendor");
  }
}

}  // namespace

int main(int, char** argv) {
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  int result = 1;
  try {
    ConfigureEnvironment();

    const macro_fusion::FrameSourceConfig config =
        macro_fusion::TumFreiburgConfig();
    macro_fusion::TumFrameSource source(config);
    macro_fusion::Viewer viewer(config.image_extent, true);
    macro_fusion::CudaReconstructionPipeline reconstruction(config);

    macro_fusion::FrameMetadata metadata{};

    while (viewer.BeginFrame(source.HasNext())) {
      if (viewer.AdvanceRequested()) {
        // TODO(profile "acquire")
        auto frame = source.Next();
        metadata = std::get<0>(frame);
        const std::vector<float>& raw_depth = std::get<1>(frame);
        const std::vector<uchar3>& rgb = std::get<2>(frame);
        // TODO(profile "preprocess")
        const std::vector<macro_fusion::HostTrackingLevel> pyramid =
            macro_fusion::PreprocessFrame(config, raw_depth);

        // TODO(profile "render")
        const std::array<uchar4*, macro_fusion::kPaneCount> mapped_panes =
            viewer.MapDisplayBuffers();
        reconstruction.ProcessFrame(metadata, raw_depth, rgb, pyramid.front(),
                                    mapped_panes);
        viewer.UnmapAndRefreshTextures();
      }
      // TODO(profile "present")
      viewer.EndFrame(metadata);
      // TODO(profile "pipeline_total" / "frame_wall_time")
    }
    result = 0;
  } catch (const std::exception& error) {
    LOG(ERROR) << error.what();
  }

  google::ShutdownGoogleLogging();
  return result;
}
