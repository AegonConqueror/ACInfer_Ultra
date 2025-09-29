cmake_minimum_required(VERSION 3.15)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})

add_subdirectory(plugin)

# ============== trt & tools lib ==============
file(GLOB_RECURSE TRT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/trt/*.cpp)
file(GLOB_RECURSE TOOLS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cpp)

add_library(
    trt_utils
    SHARED
    ${TRT_SOURCES}
    ${TOOLS_SOURCES}
)
target_link_libraries(trt_utils PUBLIC ${TensorRT_LIBS} ${OpenCV_LIBS} ${CUDA_LIBS})
set_target_properties(trt_utils PROPERTIES CUDA_ARCHITECTURES "61;70;75")

# ============== trt engine builder lib ==============
add_library(
    engine_builder
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/builder/trt_builder.cpp
)
target_link_libraries(engine_builder PUBLIC trt_utils)
target_link_libraries(engine_builder PRIVATE -Wl,--no-as-needed yolov8DetLayerPlugin)

# ============== trt engine infer lib ==============
add_library(
    infer_engine
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/trt_engine.cpp
)
target_link_libraries(infer_engine PUBLIC trt_utils)
target_link_libraries(infer_engine PRIVATE -Wl,--no-as-needed yolov8DetLayerPlugin)

# ============== yolo engine builder exe ==============
add_executable(
    build_yolo_engine
    ${CMAKE_CURRENT_SOURCE_DIR}/build_yolo_engine.cpp
)
target_link_libraries(build_yolo_engine PRIVATE engine_builder)