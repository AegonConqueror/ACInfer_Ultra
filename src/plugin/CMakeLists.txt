cmake_minimum_required(VERSION 3.15)

project(yolov8PosePlugin)

# add_library(
#     customLayerPlugin
#     SHARED
#     ${CMAKE_CURRENT_SOURCE_DIR}/customLayerPlugin/customLayerPlugin.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/customLayerPlugin/customLayerInference.cu
# )
# target_include_directories(customLayerPlugin PUBLIC ${TensorRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
# target_link_libraries(customLayerPlugin PUBLIC ${TensorRT_LIBS})
# set_target_properties(customLayerPlugin PROPERTIES CUDA_ARCHITECTURES "61;70;75")

add_library(
    yolov8PoseLayerPlugin
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8PoseLayerPlugin/yolov8PoseLayerPlugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8PoseLayerPlugin/yolov8PoseLayerInference.cu
)
target_include_directories(yolov8PoseLayerPlugin PUBLIC ${TensorRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
target_link_libraries(yolov8PoseLayerPlugin PUBLIC ${TensorRT_LIBS})
set_target_properties(yolov8PoseLayerPlugin PROPERTIES CUDA_ARCHITECTURES "61;70;75")