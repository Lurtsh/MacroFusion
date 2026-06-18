cmake_minimum_required(VERSION 3.28)

set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90" CACHE STRING
    "CUDA architectures to build without requiring a GPU during configure")

project(MacroFusion
    VERSION 0.1.0
    LANGUAGES C CXX CUDA
)

include(FetchContent)

find_package(OpenGL REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MACROFUSION_BUILD_TESTS "Build MacroFusion tests" ON)
set(MACROFUSION_GLFW_LINUX_BACKEND "AUTO" CACHE STRING
    "GLFW Linux backend: AUTO, X11, or WAYLAND")
set_property(CACHE MACROFUSION_GLFW_LINUX_BACKEND PROPERTY STRINGS AUTO X11 WAYLAND)

set(BUILD_TESTING OFF CACHE BOOL "Disable dependency test targets" FORCE)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.0
)

FetchContent_Declare(
    gflags
    GIT_REPOSITORY https://github.com/gflags/gflags.git
    GIT_TAG v2.2.2
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

if(UNIX AND NOT APPLE)
    string(TOUPPER "${MACROFUSION_GLFW_LINUX_BACKEND}" _glfw_linux_backend)
    if(_glfw_linux_backend STREQUAL "AUTO")
        if(DEFINED ENV{XDG_SESSION_TYPE})
            string(TOUPPER "$ENV{XDG_SESSION_TYPE}" _session_type)
        else()
            set(_session_type "")
        endif()

        if(_session_type STREQUAL "WAYLAND")
            set(_glfw_linux_backend "WAYLAND")
        else()
            set(_glfw_linux_backend "X11")
        endif()
    endif()

    if(_glfw_linux_backend STREQUAL "WAYLAND")
        find_program(WAYLAND_SCANNER_EXECUTABLE wayland-scanner)
        if(NOT WAYLAND_SCANNER_EXECUTABLE)
            message(WARNING "Wayland session requested, but wayland-scanner was not found. Falling back to GLFW X11.")
            set(_glfw_linux_backend "X11")
        endif()
    endif()

    if(_glfw_linux_backend STREQUAL "WAYLAND")
        set(GLFW_BUILD_WAYLAND ON CACHE BOOL "" FORCE)
        set(GLFW_BUILD_X11 OFF CACHE BOOL "" FORCE)
    elseif(_glfw_linux_backend STREQUAL "X11")
        set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
    else()
        message(FATAL_ERROR "MACROFUSION_GLFW_LINUX_BACKEND must be AUTO, X11, or WAYLAND")
    endif()

    message(STATUS "GLFW Linux backend: ${_glfw_linux_backend}")
endif()

FetchContent_MakeAvailable(eigen gflags glog glfw nlohmann_json stb imgui glad)

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
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

add_library(macro_fusion STATIC
    src/cuda/frame_constants.cu
    src/modules/preprocess.cpp
    src/modules/render.cpp
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
    PRIVATE
        stb::headers
)

add_executable(macrofusion
    src/main.cpp
)

target_link_libraries(macrofusion
    PRIVATE
        macro_fusion
        glad::gl
        glog::glog
        imgui::imgui
)

if(MACROFUSION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
