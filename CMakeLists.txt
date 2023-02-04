cmake_minimum_required(VERSION 3.22)
project(spanet_onnx)

set(CMAKE_CXX_STANDARD 17)

option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")

if(onnxruntime_USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_DNNL)
    add_definitions(-DUSE_DNNL)
endif()
if(onnxruntime_USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
endif()

set(HIGHFIVE_USE_XTENSOR ON)

add_subdirectory(ext/xtl)
add_subdirectory(ext/xtensor)

set(xtl_DIR ${CMAKE_BINARY_DIR}/ext/xtl)
set(xtensor_DIR ${CMAKE_BINARY_DIR}/ext/xtensor)

add_subdirectory(ext/HighFive)
add_subdirectory(ext/rapidyaml)


file(GLOB spanet_headers "include/*.hpp")
file(GLOB spanet_src "src/*.cpp")

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS "${ONNXRUNTIME_ROOTDIR}/include")
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS "${ONNXRUNTIME_ROOTDIR}/lib")

add_executable(spanet_onnx main.cpp ${spanet_src} ${spanet_headers})
target_link_libraries(spanet_onnx xtensor HighFive ryml ${ONNX_RUNTIME_LIB})
target_include_directories(spanet_onnx PRIVATE include/ ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS})



