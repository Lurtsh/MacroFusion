cmake_minimum_required(VERSION 3.28)

set(CMAKE_CUDA_ARCHITECTURES "75;86;89;120" CACHE STRING
    "CUDA architectures targeted by MacroFusion")

project(MacroFusion
    VERSION 0.1.0
    LANGUAGES C CXX CUDA
)

include(FetchContent)

set(OpenGL_GL_PREFERENCE GLVND)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL GLX)
find_package(TBB REQUIRED)

if(NOT TARGET OpenGL::OpenGL OR NOT TARGET OpenGL::GLX)
    message(FATAL_ERROR "MacroFusion requires GLVND libOpenGL and libGLX")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MACROFUSION_BUILD_TESTS "Build MacroFusion tests" ON)

set(BUILD_TESTING OFF CACHE BOOL "Disable dependency test targets" FORCE)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.0
)

set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
set(WITH_GTEST OFF CACHE BOOL "" FORCE)
set(WITH_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    glog
    GIT_REPOSITORY https://github.com/google/glog.git
    GIT_TAG v0.7.1
)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG f0569113c93ad095470c54bf34a17b36646bbbb5
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.8
)

FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v2.0.8
)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(eigen glog glfw nlohmann_json stb imgui glad)

add_library(stb_headers INTERFACE)
add_library(stb::headers ALIAS stb_headers)
target_include_directories(stb_headers SYSTEM INTERFACE "${stb_SOURCE_DIR}")

add_subdirectory("${glad_SOURCE_DIR}/cmake" "${glad_BINARY_DIR}/cmake")
glad_add_library(glad_gl_core_46 STATIC REPRODUCIBLE LOADER API gl:core=4.6)
add_library(glad::gl ALIAS glad_gl_core_46)

add_library(imgui STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_demo.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
)
add_library(imgui::imgui ALIAS imgui)
target_include_directories(imgui SYSTEM PUBLIC
    "${imgui_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}/backends"
)
target_compile_definitions(imgui PUBLIC
    GLFW_INCLUDE_NONE
    IMGUI_IMPL_OPENGL_LOADER_CUSTOM
)
target_compile_options(imgui PRIVATE "SHELL:-include glad/gl.h")
target_link_libraries(imgui PUBLIC glfw glad::gl OpenGL::OpenGL OpenGL::GLX)

add_library(macro_fusion STATIC
    src/modules/pipeline.cu
    src/core/frame_constants.cu
    src/modules/integration.cu
    src/modules/macro_cells.cpp
    src/modules/preprocess.cpp
    src/modules/raycast.cu
    src/modules/preprocess_cuda.cu
    src/modules/render.cpp
    src/modules/render_cuda.cu
    src/modules/tum_frame_source.cpp
)

target_include_directories(macro_fusion
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(macro_fusion SYSTEM PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(macro_fusion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(macro_fusion
    PUBLIC
        Eigen3::Eigen
        TBB::tbb
    PRIVATE
        stb::headers
)

add_executable(macrofusion
    src/core/viewer.cpp
    src/main.cpp
)

target_link_libraries(macrofusion
    PRIVATE
        macro_fusion
        glad::gl
        CUDA::cudart
        glog::glog
        imgui::imgui
)

if(MACROFUSION_BUILD_TESTS
        AND EXISTS "${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()
