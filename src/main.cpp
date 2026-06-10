#include <glog/logging.h>

#include <iostream>

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    LOG(INFO) << "MacroFusion development scaffold started";
    std::cout << "Hello from MacroFusion CUDA project" << std::endl;

    google::ShutdownGoogleLogging();
    return 0;
}
