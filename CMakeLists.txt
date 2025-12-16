cmake_minimum_required(VERSION 3.15)

project(ACInfer_Ultra LANGUAGES CXX C)

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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib/")

message(STATUS ">>\tUsing TensorRT for Model Inference... ")
enable_language(CUDA)

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/app
    ${PROJECT_SOURCE_DIR}/src/engine
    ${PROJECT_SOURCE_DIR}/src/plugin
)

add_subdirectory(src)