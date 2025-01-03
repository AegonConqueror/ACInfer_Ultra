cmake_minimum_required(VERSION 3.15)

project(ACInfer_Ultra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH_DIR x86)
else()
    set(ARCH_DIR arm)
endif()
message(STATUS ">>\tPLATFORM : ${ARCH_DIR}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options : Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS ">>\tCMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")

list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# 可执行文件
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin/")
# 动态库
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/")
# 静态库
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/")

option(USE_TENSORRT     "option for open tensorrt inference api"    ON)
option(USE_ONNXRUNTIME  "option for open onnxruntime inference api" ON)
option(USE_ATALS        "option for open atals inference api"       OFF)

if(USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
endif(USE_TENSORRT)

if(USE_ONNXRUNTIME)
    add_definitions(-DUSE_ONNXRUNTIME)
endif(USE_ONNXRUNTIME)

if(USE_ATALS)
    add_definitions(-DUSE_ATALS)
endif(USE_ATALS)

if(USE_TENSORRT)
    message(STATUS ">>\tuse tensorrt to model inference")
    enable_language(CUDA)

    find_package(CUDA QUIET)
    if (CUDA_FOUND)
        message(STATUS ">>\t\tCUDA version: 11.8")
    else()
        message(FATAL_ERROR "Cannot find CUDA")
    endif()

    find_package(TensorRT QUIET)
    if (TensorRT_FOUND)
        message(STATUS ">>\t\tTensorRT version: ${TensorRT_VERSION_STRING}")
    else()
        message(FATAL_ERROR "Cannot find TensorRT")
    endif()

    find_package(OpenCV QUIET)
    if (OpenCV_FOUND)
        message(STATUS ">>\t\tOpenCV version: ${OpenCV_VERSION}")
    else()
        message(FATAL_ERROR "Cannot find OpenCV")
    endif()
endif(USE_TENSORRT)

if(USE_ONNXRUNTIME)
    message(STATUS ">>\tuse onnxruntime to model inference")
    find_package(ONNXRuntime QUIET)
    if (ONNXRuntime_FOUND)
        message(STATUS ">>\t\tONNXRuntime version: 1.19.2")
    else()
        message(FATAL_ERROR "Cannot find ONNXRuntime")
    endif()

    find_package(OpenCV QUIET)
    if (OpenCV_FOUND)
        message(STATUS ">>\t\tOpenCV version: ${OpenCV_VERSION}")
    else()
        message(FATAL_ERROR "Cannot find OpenCV")
    endif()
endif(USE_ONNXRUNTIME)

if(USE_ATALS)
    message(STATUS ">>\tuse atlas to model inference")
    
    set(CROSS_COMPILE "aarch64-mix210-linux")

    message(STATUS ">>\t\tAtlas cross complie platform : ${CROSS_COMPILE}")
    set(CMAKE_C_COMPILER "${CROSS_COMPILE}-gcc")
    set(CMAKE_CXX_COMPILER "${CROSS_COMPILE}-g++")
    set(CMAKE_AR "${CROSS_COMPILE}-ar")
    set(CMAKE_RANLIB "${CROSS_COMPILE}-ranlib")
    set(STRIP_TOOL "${CROSS_COMPILE}-strip")

    set(DDK_PATH "/home/aegon/Ascend/ascend-toolkit/5.20.t6.2.b060/arm64-lmixlinux200")
    set(OpenCV_PATH "${PROJECT_SOURCE_DIR}/3rdparty/opencv")

endif(USE_ATALS)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/app
    ${PROJECT_SOURCE_DIR}/src/engine
)

add_subdirectory(src)