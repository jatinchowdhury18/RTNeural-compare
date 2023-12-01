cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(RTNeural-compare)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Using RTNeural with backend
set(RTNEURAL_STL ON CACHE BOOL "Use RTNeural with this backend" FORCE)
add_subdirectory(modules/RTNeural)
include_directories(modules/RTNeural/RTNeural)
target_compile_features(RTNeural PRIVATE cxx_std_17)

add_executable(rtneural_compare_bench src/benchmark.cpp)
target_link_libraries(rtneural_compare_bench PRIVATE "${TORCH_LIBRARIES}" RTNeural)
target_compile_features(rtneural_compare_bench PRIVATE cxx_std_17)
target_compile_definitions(rtneural_compare_bench PRIVATE BENCH_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

set(ONNXRUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/modules/onnxruntime-osx-x86_64-1.15.1")
target_include_directories(rtneural_compare_bench PRIVATE
    "${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session") # Windows local install
target_link_directories(rtneural_compare_bench PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib")
target_link_libraries(rtneural_compare_bench PRIVATE onnxruntime)

set(TFLITERUNTIME_ROOTDIR "${CMAKE_CURRENT_SOURCE_DIR}/modules/tensorflowlite_c-2.14.0")
target_include_directories(rtneural_compare_bench PRIVATE "${TFLITERUNTIME_ROOTDIR}/include")
target_link_directories(rtneural_compare_bench PRIVATE "${TFLITERUNTIME_ROOTDIR}/lib")
target_link_libraries(rtneural_compare_bench PRIVATE tensorflowlite_c)

option(RTNEURAL_ONLY "Run RTNeural benchmarks only" OFF)
if(RTNEURAL_ONLY)
    message(STATUS "Only running RTNeural benchmarks")
    target_compile_definitions(rtneural_compare_bench PRIVATE RTNEURAL_ONLY=1)
endif()

target_compile_definitions(rtneural_compare_bench PUBLIC EIGEN_STACK_ALLOCATION_LIMIT=300000)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET rtneural_compare_bench
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
