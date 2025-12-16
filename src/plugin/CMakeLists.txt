cmake_minimum_required(VERSION 3.15)

project(ACInferPlugin)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

add_library(
    yolov8DetLayerPlugin
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8DetLayerPlugin/yolov8DetLayerPlugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8DetLayerPlugin/yolov8DetLayerInference.cu
)
target_include_directories(yolov8DetLayerPlugin PUBLIC ${TensorRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
target_link_libraries(yolov8DetLayerPlugin PUBLIC ${TensorRT_LIBS})
set_target_properties(yolov8DetLayerPlugin PROPERTIES CUDA_ARCHITECTURES "61;70;75")

set_target_properties(
    yolov8DetLayerPlugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/plugins"
    ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_SOURCE_DIR}/plugins"
)
